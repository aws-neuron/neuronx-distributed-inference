import logging
import os
import warnings
from functools import partial

import torch
import torch.nn.functional as F
from neuronx_distributed.quantization.quantization_config import (
    ActivationQuantizationType,
    QuantizationType,
    QuantizedDtype,
    get_default_custom_qconfig_dict,
    get_default_per_channel_custom_qconfig_dict,
    get_default_expert_wise_per_channel_custom_qconfig_dict,
)
from neuronx_distributed.quantization.quantize import convert
from neuronx_distributed.trace import parallel_model_load, parallel_model_trace
from neuronx_distributed.trace.model_builder import BaseModelInstance

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.modules.async_execution import (
    AsyncTensorWrapper,
    get_async_output,
    is_ranked_io,
)
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"
SPECULATION_MODEL_TAG = "speculation_model"
MEDUSA_MODEL_TAG = "medusa_speculation_model"
FUSED_SPECULATION_MODEL_TAG = "fused_speculation_model"
VISION_ENCODER_MODEL_TAG = "vision_encoder_model"


# Get the modules_to_not_convert from the neuron configs
def get_modules_to_not_convert(neuron_config: NeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = False,
        return_ranked_to_cpu: bool = False,
        model_init_kwargs={},
    ) -> None:
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config

        if not self.neuron_config.torch_dtype:
            self.neuron_config.torch_dtype = torch.float32

        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = 0

        self.model_cls = model_cls
        self.model = None
        self.is_compiled = False
        self.serialize_base_path = None
        self.tag = tag
        self.is_block_kv_layout = config.neuron_config.is_block_kv_layout
        self.is_prefix_caching = config.neuron_config.is_prefix_caching
        self.is_chunked_prefill = config.neuron_config.is_chunked_prefill
        self.is_medusa = config.neuron_config.is_medusa

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
        self.compiler_workdir = os.path.join(base_compile_work_dir, self.tag)

        hlo2tensorizer = ""
        if compiler_args is None:
            if self.tag in [TOKEN_GENERATION_MODEL_TAG, FUSED_SPECULATION_MODEL_TAG]:
                self.neuron_config.cc_pipeline_tiling_factor = 1
            tensorizer_options = (
                "--enable-ccop-compute-overlap "
                f"--cc-pipeline-tiling-factor={self.neuron_config.cc_pipeline_tiling_factor} "
                "--vectorize-strided-dma "
            )

            long_ctx_reqs = ""

            if self.neuron_config.enable_long_context_mode:
                long_ctx_reqs = " --internal-disable-fma-on-ios "  # reduce dma rings io by disabling FMA that doesn't use DGE

            self.compiler_args = (
                f"--auto-cast=none --model-type=transformer {long_ctx_reqs} "
                f"--tensorizer-options='{tensorizer_options}'"
                f" --lnc={self.neuron_config.logical_nc_config}"
            )

            if self.neuron_config.scratchpad_page_size:
                self.compiler_args += f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size} "

            if self.is_block_kv_layout and (
                self.neuron_config.attn_block_tkg_nki_kernel_enabled
                or self.neuron_config.attn_tkg_nki_kernel_enabled
            ):
                # Remove once NCC-6661 is resolved.
                self.compiler_args += " --internal-backend-options='--enable-verifier=false' "

            if (tag == CONTEXT_ENCODING_MODEL_TAG or tag == VISION_ENCODER_MODEL_TAG):
                # force Modular flow for CTE model to save compile time
                self.compiler_args += " -O1 "
                # Set threshold low so that modular flow always kicks in for context encoding graph.
                # This is needed because lots of mac-ops happen in kernels which are not visible to the graph partitioner at the moment.
                hlo2tensorizer += " --modular-flow-mac-threshold=10 "

                if self.is_chunked_prefill:
                    # To reduce layer boundary that blocks overlapping of DMA
                    # and compute
                    self.compiler_args += " --layer-unroll-factor=4 "

            else:
                # NxD will add -O1 if existing compiler args do not have -Ox. We want to
                # use -O2 for TKG model to avoid function call overheads with Modular flow.
                self.compiler_args += " -O2 "

            if self.neuron_config.enable_spill_reload_dge:
                self.compiler_args += " --internal-enable-dge-levels spill_reload "

            if self.neuron_config.target:
                self.compiler_args += f" --target {self.neuron_config.target}"

            # disable further partitioning when using markers to avoid compiler errors"
            if self.neuron_config.layer_boundary_markers:
                hlo2tensorizer += " --recursive-layer-det=false "

        else:
            self.compiler_args = compiler_args

        if (
            (
                self.neuron_config.quantized is True
                and self.neuron_config.quantization_dtype == "f8e4m3"
            )
            or self.neuron_config.kv_cache_quant
            or self.neuron_config.quantized_mlp_kernel_enabled
            or self.neuron_config.activation_quantization_type
        ):
            hlo2tensorizer += " --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 "

        if hlo2tensorizer:
            self.compiler_args += f" --internal-hlo2tensorizer-options='{hlo2tensorizer} --verify-hlo=true' "
        else:
            self.compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true' "

        if self.neuron_config.enable_output_completion_notifications:
            self.compiler_args += " --internal-backend-options=' --enable-output-completion-notifications ' "

        logging.info(f"neuronx-cc compiler_args are: {self.compiler_args}")

        self.bucket_config = None
        self.priority_model_idx = priority_model_idx
        self.pipeline_execution = pipeline_execution
        self.return_ranked_to_cpu = return_ranked_to_cpu
        self.model_init_kwargs = model_init_kwargs
        self.async_mode = self.neuron_config.async_mode

    def is_neuron(self):
        return self.model is not None and isinstance(self.model, torch.jit.ScriptModule)

    def compile(self, checkpoint_loader, serialize_base_path):
        inputs = self.input_generator()

        # cannot pass partial func with multiprocess using model directly
        parallel_model_trace(
            partial(get_trace_callable, self.model_cls, self.neuron_config),
            inputs,
            tp_degree=self.neuron_config.tp_degree,
            compiler_workdir=self.compiler_workdir,
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
            spmd_mode=True,
            checkpoint_loader_callable=checkpoint_loader,
            bucket_config=self.bucket_config,
            force_custom_init_on_device=True,
            serialization_path=os.path.join(serialize_base_path, self.tag),
        )
        print(f"Successfully traced the {self.tag}!")

    def load(self, serialize_base_path):
        self.model = parallel_model_load(os.path.join(serialize_base_path, self.tag))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.model = self.model_cls(self.config)
        self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def input_generator(
        self,
    ):
        """Generate a list of valid sample inputs containing one input list for each bucket."""
        inputs = []
        for bucket in self.neuron_config.buckets:
            batch_size = self.neuron_config.batch_size
            if isinstance(bucket, list) and self.neuron_config.is_prefix_caching:
                n_active_tokens = (
                    bucket[0]
                    if self.neuron_config.bucket_n_active_tokens
                    else self.neuron_config.n_active_tokens
                )
                prefix_size = bucket[1]
                if prefix_size == 0:
                    attention_mask = torch.zeros(1, dtype=torch.int32)
                else:
                    attention_mask = torch.ones((batch_size, prefix_size), dtype=torch.int32)
            elif self.is_chunked_prefill and self.tag == CONTEXT_ENCODING_MODEL_TAG:
                assert isinstance(bucket, list)
                n_active_tokens = (
                    bucket[0]
                    if self.neuron_config.bucket_n_active_tokens
                    else self.neuron_config.n_active_tokens
                )  # chunk size
                attention_mask = torch.ones((batch_size, n_active_tokens), dtype=torch.int32)
            else:
                n_active_tokens = (
                    bucket
                    if self.neuron_config.bucket_n_active_tokens
                    else self.neuron_config.n_active_tokens
                )
                attention_mask = torch.ones((batch_size, bucket), dtype=torch.int32)

            # TODO: Find a better way to ensure warmup invocations work. Some models
            # like 405B with eagle speculation seem to have problems with torch.zeros
            # as warmup input that causes some buckets to not warmup up.
            input_ids = torch.ones((batch_size, n_active_tokens), dtype=torch.int32)
            if self.tag != CONTEXT_ENCODING_MODEL_TAG and self.tag != VISION_ENCODER_MODEL_TAG and self.tag != MEDUSA_MODEL_TAG:
                position_ids = torch.ones((batch_size, n_active_tokens), dtype=torch.int32)
            else:
                position_ids = torch.arange(0, n_active_tokens, dtype=torch.int32).unsqueeze(0)
                position_ids = position_ids.repeat(batch_size, 1)
            seq_ids = torch.arange(0, batch_size, dtype=torch.int32)
            adapter_ids = torch.zeros((batch_size), dtype=torch.int32)

            # Get the count of sampling params currently supported.
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.ones((batch_size, sampling_params_len), dtype=torch.float32)
            if self.neuron_config.on_device_sampling_config:
                if self.neuron_config.on_device_sampling_config.do_sample:
                    sampling_params[:, 0] = self.neuron_config.on_device_sampling_config.top_k
                    sampling_params[:, 1] = self.neuron_config.on_device_sampling_config.top_p
                    sampling_params[:, 2] = self.neuron_config.on_device_sampling_config.temperature

            hidden_states = (
                torch.zeros(
                    (batch_size, n_active_tokens, self.config.hidden_size),
                    dtype=self.config.neuron_config.torch_dtype,
                )
                if self.neuron_config.is_eagle_draft
                else torch.zeros((batch_size), dtype=torch.int32)
            )

            if self.is_medusa:
                assert (
                    self.neuron_config.on_device_sampling_config
                ), "Medusa speculation must use on-device sampling"
                # Set top_k to signal to the sampler that we're not doing greedy sampling.
                # This affects the output shape for Medusa speculation
                sampling_params[:, 0] = self.neuron_config.on_device_sampling_config.top_k
                accepted_indices = torch.zeros(
                    (batch_size, self.neuron_config.num_medusa_heads + 1),
                    dtype=torch.int32,
                )
                current_length = torch.zeros(
                    (batch_size, self.neuron_config.num_medusa_heads + 1),
                    dtype=torch.int32,
                )
                medusa_mask = torch.zeros(
                    (
                        batch_size,
                        self.neuron_config.medusa_speculation_length,
                        self.neuron_config.medusa_speculation_length,
                    ),
                    dtype=torch.int32,
                )
                scatter_index = torch.zeros(
                    (batch_size, self.neuron_config.medusa_speculation_length),
                    dtype=torch.int32,
                )

                inputs.append(
                    (
                        input_ids,
                        attention_mask,
                        position_ids,
                        seq_ids,
                        sampling_params,
                        hidden_states,
                        adapter_ids,
                        accepted_indices,
                        current_length,
                        medusa_mask,
                        scatter_index,
                    )
                )
            elif self.is_chunked_prefill:
                input_shape = self._get_input_shape_for_chunked_prefill(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    n_active_tokens,
                    bucket,
                )
                inputs.append(input_shape)
            elif self.is_prefix_caching:
                input_shape = self._get_input_shape_for_prefix_caching(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    batch_size,
                    n_active_tokens,
                    prefix_size,
                )
                inputs.append(input_shape)
            else:
                inputs.append(
                    (
                        input_ids,
                        attention_mask,
                        position_ids,
                        seq_ids,
                        sampling_params,
                        hidden_states,
                        adapter_ids,
                    )
                )

        return inputs

    def _get_input_shape_for_chunked_prefill(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        n_active_tokens,
        bucket,
    ):
        cp_config = self.neuron_config.chunked_prefill_config

        if self.neuron_config.is_prefill_stage:
            assert len(bucket) == 2
            num_tiles = bucket[1]

            # batch_size is 1 for CTE, so (batch_size, -1) -> (max_num_seqs, -1)
            sampling_params = sampling_params.expand(cp_config.max_num_seqs, -1)

            slot_mapping = torch.zeros(n_active_tokens, dtype=torch.int32)
            # CTE reads the whole KV cache blocks, and it uses tile_block_tables
            # instead of active_block_table
            active_block_table = torch.empty(0)

            # length of query for each request
            query_lens = torch.zeros(cp_config.max_num_seqs, dtype=torch.int32)
            # length of computed context
            computed_context_lens = torch.zeros(cp_config.max_num_seqs, dtype=torch.int32)

            # params for chunked prefill attn kernel
            q_tile_size = cp_config.kernel_q_tile_size
            kv_tile_size = cp_config.kernel_kv_tile_size
            assert kv_tile_size % self.neuron_config.pa_block_size == 0
            num_blocks_per_kv_tile = kv_tile_size // self.neuron_config.pa_block_size

            tile_q_indices = torch.zeros(num_tiles, dtype=torch.int32)
            tile_block_tables = torch.zeros(
                (num_tiles, num_blocks_per_kv_tile), dtype=torch.int32
            )
            tile_masks = torch.zeros(
                (num_tiles, q_tile_size, kv_tile_size), dtype=torch.bool
            )

        else:
            # TKG for chunked prefill will be in batching format (batch_size,
            # seq_len), in order to reuse existing code path.
            kv_cache_len = bucket
            assert n_active_tokens == 1

            slot_mapping = torch.zeros(
                (cp_config.max_num_seqs, n_active_tokens),
                dtype=torch.int32,
            )

            assert kv_cache_len % self.neuron_config.pa_block_size == 0
            num_blocks_per_seq = kv_cache_len // self.neuron_config.pa_block_size
            active_block_table = torch.zeros(
                (cp_config.max_num_seqs, num_blocks_per_seq), dtype=torch.int32
            )

            # length of query for each request
            query_lens = torch.zeros(
                (cp_config.max_num_seqs, n_active_tokens), dtype=torch.int32
            )
            # length of computed context length
            computed_context_lens = torch.zeros(
                (cp_config.max_num_seqs, n_active_tokens), dtype=torch.int32
            )

            # Need to ensure CTE and TKG has same number of inputs due to
            # tracing limitation.
            tile_q_indices = torch.empty(0)
            tile_block_tables = torch.empty(0)
            tile_masks = torch.empty(0)

        input_shape = (
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            torch.empty(0),  # prev_hidden
            torch.empty(0),  # adapter_ids
            torch.empty(0),  # accepted_indices
            torch.empty(0),  # current_length
            torch.empty(0),  # medusa_mask
            torch.empty(0),  # scatter_index
            slot_mapping,
            active_block_table,
            query_lens,
            computed_context_lens,
            tile_q_indices,
            tile_block_tables,
            tile_masks,
        )
        return input_shape

    def _get_input_shape_for_prefix_caching(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        batch_size,
        n_active_tokens,
        prefix_size,
    ):
        if self.neuron_config.enable_fused_speculation and self.tag == FUSED_SPECULATION_MODEL_TAG:
            slot_mapping = torch.zeros((batch_size, self.neuron_config.speculation_length), dtype=torch.int32)
        else:
            slot_mapping = torch.zeros((batch_size, n_active_tokens), dtype=torch.int32)

        num_blocks = prefix_size // self.neuron_config.pa_block_size
        active_block_table = torch.zeros(1, dtype=torch.int32) if num_blocks == 0 else torch.zeros(
            (batch_size, num_blocks), dtype=torch.int32
        )

        num_queries = torch.full((batch_size, 1), n_active_tokens, dtype=torch.int32)
        computed_context_lens = torch.full((batch_size, 1), prefix_size, dtype=torch.int32)
        if self.neuron_config.enable_eagle_speculation:
            if self.tag == FUSED_SPECULATION_MODEL_TAG:
                return (
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    torch.empty(0),  # prev_hidden
                    torch.empty(0),  # adapter_ids
                    slot_mapping,
                    active_block_table,
                    num_queries,
                    computed_context_lens,
                    torch.empty(0),  # target_input_ids
                    torch.empty(0),  # target_attention_mask
                    torch.empty(0),  # target_position_ids
                    torch.empty(0),  # target_slot_mapping
                    torch.empty(0),  # target_active_block_table
                )
            else:
                prefill = input_ids.shape[-1]
                target_input_ids = torch.ones((batch_size, prefill), dtype=torch.int32)
                target_attention_mask = torch.ones((batch_size, prefix_size), dtype=torch.int32)
                if prefix_size == 0:
                    target_attention_mask = torch.zeros(1, dtype=torch.int32)
                else:
                    target_attention_mask = torch.ones((batch_size, prefix_size), dtype=torch.int32)
                target_position_ids = torch.arange(0, prefill, dtype=torch.int32).unsqueeze(0)
                target_position_ids = target_position_ids.repeat(batch_size, 1)
                target_slot_mapping = torch.zeros((batch_size, prefill), dtype=torch.int32)
                target_num_blocks = prefix_size // self.neuron_config.pa_block_size
                target_active_block_table = torch.zeros(1, dtype=torch.int32) if target_num_blocks == 0 else torch.zeros(
                    (batch_size, target_num_blocks), dtype=torch.int32
                )
                return (
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    torch.empty(0),  # prev_hidden
                    torch.empty(0),  # adapter_ids
                    slot_mapping,
                    active_block_table,
                    num_queries,
                    computed_context_lens,
                    target_input_ids,
                    target_attention_mask,
                    target_position_ids,
                    target_slot_mapping,
                    target_active_block_table
                )
        elif self.neuron_config.enable_fused_speculation:
            return (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),  # prev_hidden
                torch.empty(0),  # adapter_ids
                slot_mapping,
                active_block_table,
                num_queries,
                computed_context_lens,
            )
        else:
            return (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),  # prev_hidden
                torch.empty(0),  # adapter_ids
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                slot_mapping,
                active_block_table,
                num_queries,
                computed_context_lens,
            )

    def get_model_instance(self):
        return DecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )

    def _forward_with_pad(self, *args):
        # Note: NxD's tracing flow (Model Builder) does not yet support kwargs, because of which we cannot support
        # optional parameters. Kwargs support is being added as a part of the new Model Builder API. Until then we
        # maintain a specific set of inputs that the ModelWrapper can support.
        # This is not the best way to maintain code. But soon kwargs suport will render this irrelevant.
        seq_ids = args[3]
        sampling_params = args[4]
        if self.is_block_kv_layout:
            medusa_args = None
        elif len(args) > 5:
            medusa_args = args[5:8]
        else:
            medusa_args = None

        if self.is_prefix_caching:
            if self.neuron_config.enable_fused_speculation:
                block_kv_empty_args = args[5:7]
                block_kv_slot_mapping = args[7]
                block_kv_args = args[8:11]
            else:
                block_kv_empty_args = args[5:11]
                block_kv_slot_mapping = args[11]
                block_kv_args = args[12:15]

        # pad the inputs up to the compiled batch size in the end
        def pad_helper(tensor, pad_type="fill_0", batch_sort_indices=None):
            """
            As part of continuous batching:
            * If users provide us input batch size less than compiled batch size, NxDI
              need to pad the inputs to the compiled batch size.
            * seq_ids are used to indicate which kv cache line is used for each input batch line.
              NxDI expects the seq_ids to always be [0, 1, 2, ..., compiled_batch_size) by default.
            * To fulfill these requirements, NxDI pads the seq_ids with the missing slots and sorts
              it in ascending order. Every other input args are reordered accordingly and
              missing slots are padded with `repeat_first_batchline`. While returning back response,
              we use index selct to pick the outputs corresponding to user provided seq_ids.
            Eg:
            Input [[10],[20]] and seq_ids [[3], [2]] with compiled batch size as 4.
            seq_ids [[3], [2]] -> [[3], [2], [0], [1]] (filled missing slots)    -> [[0], [1], [2], [3]] (sort)
            Input  [[10],[20]] -> [[10],[20],[10],[10]] (repeat_first_batchline) -> [[10],[10],[20],[10]](reorder)

            As part of continuous batching with prefix caching, the second restriction no longer holds true,
            so sorting of seq_ids and reordering of input args is no longer needed. Padding is required which is added
            towards the end using `repeat_first_batchline` with the exception of slot_mapping (set to -1 instead)
            as this is used to update the block kv cache. While returning back response, we just drop off the
            padded outputs lines at the end of the batch.
            Eg:
            Input [[10],[20]] ; seq_ids [[3], [2]] and slot mapping [[50],[100]] with compiled batch size as 4.
            seq_ids [[3], [2]]        -> [[3], [2], [0], [1]]  (filled missing slots)
            Input [[10],[20]]         -> [[10],[20],[10],[10]] (repeat_first_batchline)
            slot mapping [[50],[100]] -> [[50],[100],[-1], [-1]] (padded with -1)
            """
            if tensor is None or tensor.shape[0] == self.neuron_config.batch_size:
                return tensor

            padded_shape = list(tensor.shape)
            padded_shape[0] = self.neuron_config.batch_size

            def repeat_first_batchline(tensor, padded_shape):
                return tensor[0].unsqueeze(0).repeat(padded_shape[0], 1).to(tensor.dtype)

            def fill_value_tensor(value):
                return lambda tensor, padded_shape: torch.full(padded_shape, fill_value=value, dtype=tensor.dtype)

            PAD_TYPES = {
                "repeat_first_batchline": repeat_first_batchline,
                "fill_0": fill_value_tensor(0),
                "fill_1": fill_value_tensor(1),
                "fill_-1": fill_value_tensor(-1),
            }

            if pad_type not in PAD_TYPES:
                raise ValueError(f"Unknown pad_type '{pad_type}'. Available: {list(PAD_TYPES.keys())}")

            padded_tensor = PAD_TYPES[pad_type](tensor, padded_shape)
            padded_tensor[: tensor.shape[0]] = tensor

            if batch_sort_indices is not None:
                padded_tensor = torch.index_select(padded_tensor, 0, batch_sort_indices)

            return padded_tensor

        reorder_seq_ids = not self.is_prefix_caching
        seq_ids_list = seq_ids.tolist()
        if self.config.neuron_config.apply_seq_ids_mask:
            seq_ids_mask = torch.tensor([1 if x in seq_ids_list else 0 for x in range(self.neuron_config.max_batch_size)])
            logging.debug(f"NxDI: running with seq_ids_mask: {seq_ids_mask}")
        padded_seq_ids = torch.tensor(
            seq_ids_list
            + [x for x in range(self.neuron_config.max_batch_size) if x not in seq_ids_list],
            dtype=seq_ids.dtype,
        )
        padded_seq_ids, indices = torch.sort(padded_seq_ids) if reorder_seq_ids else (padded_seq_ids, None)
        if self.config.neuron_config.apply_seq_ids_mask:
            padded_seq_ids = torch.where(seq_ids_mask == 1, padded_seq_ids, torch.full_like(padded_seq_ids, -1))
            logging.debug(f"NxDI: running with padded_seq_ids: {padded_seq_ids}")
        padded_args = []
        # pad input_ids, attn_mask and position_ids
        for arg in args[0:3]:
            if is_ranked_io(arg):  # async output
                # ===========READ THIS=============
                # args[0] can be either input_ids
                # or an async_output. If the output
                # is async, it means that the sorting
                # and padding has already been done
                # properly, so we simply append the
                # result. This is true because the
                # results from async are fed directly
                # to the next iteration without data
                # modification, and the model was
                # executed with padded & sorted inputs.
                # =================================
                padded_args.append(arg)
            else:
                padded_arg = pad_helper(
                    arg,
                    pad_type="repeat_first_batchline",
                    batch_sort_indices=indices,
                )
                padded_args.append(padded_arg)

        # for block kv layout the seq_ids may lies outside of range(self.neuron_config.max_batch_size)
        # therefore, we need to remove potential extra paddings to seq_ids
        padded_seq_ids = padded_seq_ids[: self.neuron_config.max_batch_size]
        padded_args.append(padded_seq_ids)

        # pad sampling params by repeating first batchline
        padded_sampling_params = pad_helper(
            sampling_params,
            pad_type="repeat_first_batchline",
            batch_sort_indices=indices,
        )
        padded_args.append(padded_sampling_params)

        if medusa_args is not None:
            for arg in medusa_args:
                padded_args.append(pad_helper(arg, batch_sort_indices=indices))

        if self.is_prefix_caching:
            for arg in block_kv_empty_args:
                padded_args.append(arg)
            padded_args.append(pad_helper(block_kv_slot_mapping, pad_type="fill_-1"))
            for arg in block_kv_args:
                padded_args.append(pad_helper(arg, pad_type="repeat_first_batchline"))
            if self.neuron_config.enable_eagle_speculation:
                eagle_empty_args = args[11:16]
                for arg in eagle_empty_args:
                    padded_args.append(arg)

        outputs = self._forward(*padded_args)

        # note that we don't do index select here as it should already be handled, simply sliced out padding here
        if self.is_neuron():
            logits = outputs
            if self.async_mode:
                return logits
            elif self.is_prefix_caching:
                if self.neuron_config.enable_fused_speculation:
                    returned_logits = [logit[: seq_ids.shape[0]] for logit in logits]
                    return returned_logits
                else:
                    return logits[: seq_ids.shape[0]]
            elif self.neuron_config.enable_fused_speculation or \
                    (self.neuron_config.on_device_sampling_config and self.neuron_config.output_logits and self.neuron_config.is_continuous_batching):
                returned_logits = [torch.index_select(logit, 0, seq_ids) for logit in logits]
                return returned_logits
            return torch.index_select(logits, 0, seq_ids)
        else:
            logits, *kv_cache = outputs
            return [torch.index_select(logits, 0, seq_ids), *kv_cache]

    def _forward(self, *args):
        if self.async_mode:
            return self._process_async_inputs(*args)

        if self.pipeline_execution:
            ranked_output = self._process_ranked_inputs(*args)
            if self.return_ranked_to_cpu:
                return ranked_output[0][0].to('cpu')
            else:
                return ranked_output

        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug(f"Processed inputs to the model. tag={self.tag}, args={args}")
        return self.model(*args)

    def convert_int64_to_int32(self, *args):
        """
        Convert int64 args to int32 to match compiled input types.
        Neuron compiler handles int32 better than int64. Context: P165494809
        """
        return [
            t.to(torch.int32) if not isinstance(t, list) and t.dtype == torch.int64 else t
            for t in args
        ]

    def pad_inputs(self, *args, pad_type="first_fit"):
        """
        The following padding strategies are supported:

        1) "max": Pad to the max bucket length
            ex. If we have input_length = 8 and buckets [5, 15, 25, 35] we choose 35
            because 35 is the last (highest) bucket.
        2) "first_fit" (default): Pad to the nearest highest bucket.
            ex. If we have input_length = 8 and buckets [5, 15, 25, 35] we choose 15
            because while 8 is closer to bucket 5, we pad up, so the nearest highest is 15
        3) "second_fit": Pad to the second nearest highest bucket.
            ex. If we have input_length = 8 and buckets [5, 15, 25, 35] we choose 25
            because 15 is the next bucket, and 25 follows 15 as the "next next"
            or "second fit" bucket.
        """

        if self.is_chunked_prefill:
            return self._pad_inputs_for_chunked_prefill(args)

        VALID_PAD_TYPES = {"max", "first_fit", "second_fit"}
        assert (
            pad_type in VALID_PAD_TYPES
        ), f"Found {pad_type=}, when it should be one of: {VALID_PAD_TYPES=}"
        pad_length = (
            self.neuron_config.max_context_length
            if (self.tag == CONTEXT_ENCODING_MODEL_TAG or self.tag == VISION_ENCODER_MODEL_TAG)
            else self.neuron_config.max_length
        )
        if isinstance(self.neuron_config.buckets[-1], list) and self.neuron_config.is_prefix_caching:
            return self._pad_prefix_caching_inputs(*args, pad_type=pad_type)

        if pad_type == "first_fit" or pad_type == "second_fit":
            pad_length = self.get_target_bucket(*args, strategy=pad_type)

        if (self.tag == CONTEXT_ENCODING_MODEL_TAG or self.tag == VISION_ENCODER_MODEL_TAG):
            to_pad = args[:3]
            pad_lengths = [pad_length - arg.shape[1] for arg in to_pad]
            tensor_pad_vals = [self.config.pad_token_id, 0, 1]
            if any([pad_len < 0 for pad_len in pad_lengths]):
                if self.neuron_config.allow_input_truncation:
                    warnings.warn(
                        f"Truncating input ({to_pad[1].shape[1]} tokens) to max_context_length ({self.neuron_config.max_context_length} tokens). This may cause unexpected outputs."
                    )
                else:
                    raise ValueError(
                        f"Inputs supplied ({to_pad[1].shape[1]} tokens) are longer than max_context_length ({self.neuron_config.max_context_length} tokens). To truncate inputs, set allow_input_truncation=True."
                    )

            padded_args = [
                F.pad(arg, (0, pad_len), "constant", pad_val)
                for arg, pad_val, pad_len in zip(to_pad, tensor_pad_vals, pad_lengths)
            ]
            args = (*padded_args, *args[3:])
        else:
            input_ids, attention_mask, *rest_of_args = args
            pad_len = pad_length - attention_mask.shape[1]
            if pad_len < 0:
                if self.neuron_config.allow_input_truncation:
                    warnings.warn(
                        f"Truncating attention mask (length={attention_mask.shape[1]}) to max_length ({self.neuron_config.max_length} tokens). This may cause unexpected outputs."
                    )
                else:
                    raise ValueError(
                        f"Attention mask supplied (length={attention_mask.shape[1]}) is longer than max_length ({self.neuron_config.max_length} tokens). To truncate attention mask, set allow_input_truncation=True."
                    )

            padded_attention_mask = F.pad(attention_mask, (0, pad_len), "constant", 0)
            args = (input_ids, padded_attention_mask, *rest_of_args)

        return args

    def _pad_inputs_for_chunked_prefill(self, args):
        args = list(args)  # convert tuple to list so it can be updated

        cp_config = self.neuron_config.chunked_prefill_config
        max_num_seqs = cp_config.max_num_seqs

        if self.tag == CONTEXT_ENCODING_MODEL_TAG:

            chunk_size, num_tiles = self.get_target_2d_bucket_for_chunked_prefill(*args)

            kv_tile_size = cp_config.kernel_kv_tile_size
            assert kv_tile_size % self.neuron_config.pa_block_size == 0
            num_blocks_per_kv_tile = kv_tile_size // self.neuron_config.pa_block_size

            input_ids = args[0]  # shape (batch_size, seq_len)
            input_ids = F.pad(
                input_ids,
                [0, chunk_size - input_ids.shape[1]],
                value=self.config.pad_token_id,
            )
            args[0] = input_ids

            attention_mask = args[1]  # shape (batch_size, seq_len)
            attention_mask = F.pad(attention_mask, [0, chunk_size - attention_mask.shape[1]])
            args[1] = attention_mask

            position_ids = args[2]  # shape (batch_size, seq_len)
            position_ids = F.pad(position_ids, [0, chunk_size - position_ids.shape[1]])
            args[2] = position_ids

            sampling_params = args[4]  # shape (max_num_seqs, num_params)
            sampling_params = F.pad(
                sampling_params,
                [0, 0, 0, max_num_seqs - sampling_params.shape[0]],
            )
            args[4] = sampling_params

            slot_mapping = args[11]
            # need to be padded by -1 to avoid overriding existing KV cache.
            slot_mapping = F.pad(
                slot_mapping, [0, chunk_size - slot_mapping.shape[0]], value=-1
            )
            args[11] = slot_mapping

            # args[12] is active_block_table but it is not needed by CTE

            query_lens = args[13]
            query_lens = F.pad(query_lens, [0, max_num_seqs - query_lens.shape[0]])
            args[13] = query_lens

            context_lens = args[14]
            context_lens = F.pad(context_lens, [0, max_num_seqs - context_lens.shape[0]])
            args[14] = context_lens

            tile_q_indices = args[15]  # (num_tiles,)
            tile_q_indices = F.pad(tile_q_indices, [0, num_tiles - tile_q_indices.shape[0]])
            args[15] = tile_q_indices

            tile_block_tables = args[16]  # (num_tiles, num_blocks_per_tile)
            tile_block_tables = F.pad(
                tile_block_tables,
                [
                    0,
                    num_blocks_per_kv_tile - tile_block_tables.shape[1],
                    0,
                    num_tiles - tile_block_tables.shape[0],
                ],
            )
            args[16] = tile_block_tables

            tile_masks = args[17]  # (num_tiles, q_tile_size, k_tile_size)
            tile_masks = F.pad(
                tile_masks,
                [0, 0, 0, 0, 0, num_tiles - tile_masks.shape[0]],
            )
            args[17] = tile_masks
        else:
            kv_cache_len = self.get_target_bucket(*args)

            input_ids = args[0]  # shape (batch_size, 1)
            input_ids = F.pad(
                input_ids,
                [0, 0, 0, max_num_seqs - input_ids.shape[0]],
                value=self.config.pad_token_id,
            )
            args[0] = input_ids

            attention_mask = args[1]  # shape (batch_size, seq_len)
            attention_mask = F.pad(
                attention_mask,
                [
                    0,
                    kv_cache_len - attention_mask.shape[1],
                    0,
                    max_num_seqs - attention_mask.shape[0],
                ],
            )
            args[1] = attention_mask

            position_ids = args[2]  # shape (batch_size, 1)
            position_ids = F.pad(
                position_ids,
                [0, 0, 0, max_num_seqs - position_ids.shape[0]],
            )
            args[2] = position_ids

            sampling_params = args[4]  # shape (max_num_seqs, num_params)
            sampling_params = F.pad(
                sampling_params,
                [0, 0, 0, max_num_seqs - sampling_params.shape[0]],
            )
            args[4] = sampling_params

            slot_mapping = args[11]
            # need to be padded by -1 to avoid overriding existing KV cache.
            slot_mapping = F.pad(
                slot_mapping,
                [0, 0, 0, max_num_seqs - slot_mapping.shape[0]],
                value=-1,
            )
            args[11] = slot_mapping

            assert kv_cache_len % self.neuron_config.pa_block_size == 0
            num_blocks_per_seq = kv_cache_len // self.neuron_config.pa_block_size
            block_table = args[12]
            block_table = F.pad(
                block_table,
                [
                    0,
                    num_blocks_per_seq - block_table.shape[1],
                    0,
                    max_num_seqs - block_table.shape[0]
                ],
            )
            args[12] = block_table

            query_lens = args[13]
            query_lens = F.pad(query_lens, [0, 0, 0, max_num_seqs - query_lens.shape[0]])
            args[13] = query_lens

            context_lens = args[14]
            context_lens = F.pad(context_lens, [0, 0, 0, max_num_seqs - context_lens.shape[0]])
            args[14] = context_lens

        args = tuple(args)
        return args

    def get_target_bucket(self, *args, strategy="first_fit"):
        # NOTE: strategy must be a subset of pad_type for consistency
        input_len = args[1].shape[1]
        buckets = self.neuron_config.buckets
        speculation_length = (
            self.neuron_config.speculation_length
            if self.tag == FUSED_SPECULATION_MODEL_TAG or self.tag == SPECULATION_MODEL_TAG
            else 0
        )
        largest_bucket_idx = len(buckets) - 1
        for i, bucket in enumerate(buckets):
            if strategy == "first_fit":
                if input_len + speculation_length < bucket:
                    return bucket
            elif strategy == "second_fit":
                if input_len < bucket:
                    return buckets[min(i + 1, largest_bucket_idx)]

        largest_bucket = buckets[largest_bucket_idx]
        if input_len + speculation_length == largest_bucket:
            return largest_bucket
        elif self.neuron_config.allow_input_truncation:
            return largest_bucket

        raise ValueError(
            f"Input len {input_len} exceeds largest bucket ({largest_bucket}) for {self.tag}"
        )

    def get_target_2d_bucket_for_prefix_caching(self, *args, strategy="first_fit"):
        """
        * We have active_len and prefix_len which are runtime variables.
        * We have S_prefix, S_active, block_length, spec_len which are compile time constants.

        CTE (Always BS=1 with continuous batching):
          - Non Eagle flow:
            * Find the smallest S_active bucket that is bigger than active_len
            * Find the smallest S_prefix bucket which fits prefix_len - (S_active - active_len)
          - Eagle flow (We need to recompute 1 additional block for target):
            * Find the smallest S_active bucket that is bigger than active_len + block_length
            * Find the smallest S_prefix bucket which fits prefix_len - (S_active - active_len - block_length)
        TKG:
            * Find the smallest S_active bucket that is bigger than active_len
        SPEC:
            * Find the smallest S_active bucket that is bigger than active_len + spec_len

        Corner Case:
            * Based on perf collection results, for 256 < (prefix_len + active_len) <= 512,
              use 512 prefill, 0 prefix bucket for serving the CTE request.
        """
        if strategy not in ("first_fit", "second_fit"):
            raise ValueError("Strategy must be either \"first_fit\" or \"second_fit\"")

        buckets = torch.tensor(self.neuron_config.buckets, dtype=torch.int32)
        speculation_length = (
            self.neuron_config.speculation_length
            if self.tag == FUSED_SPECULATION_MODEL_TAG
            else 0
        )
        # Speculation length already accounted for and handled with bucketing strategy in async.
        if self.async_mode:
            speculation_length = 0

        if self.neuron_config.enable_fused_speculation:
            vertical_dim = args[9]
            horizontal_dim = args[10]
        else:
            vertical_dim = args[13]
            horizontal_dim = args[14]

        if not self.tag == CONTEXT_ENCODING_MODEL_TAG:
            # Determine all buckets that meet horizontal condition
            horizontal_max = torch.max(horizontal_dim)
            horizontal_mask = buckets[:, 1] > horizontal_max + speculation_length

            # Determine all buckets that meet vertical condition
            vertical_max = torch.max(vertical_dim)
            vertical_mask = buckets[:, 0] >= vertical_max

            mask = horizontal_mask & vertical_mask
            final_buckets = buckets[mask]
            final_indices = torch.arange(len(buckets))[mask]

            # Take the remaining bucket smallest along the horizontal dimension
            if final_buckets.shape[0] > 0:
                selected_index = torch.argmin(final_buckets[:, 1])
                bucket_idx = final_indices[selected_index]
                if strategy == "second_fit":
                    bucket_idx = min(bucket_idx + 1, len(buckets) - 1)
            # If no buckets remain, take the largest bucket if input truncation enabled
            else:
                if not self.neuron_config.allow_input_truncation:
                    raise ValueError(
                        f"Input len {vertical_dim} exceeds largest bucket ({buckets[-1][1]}) for {self.tag}"
                    )
                else:
                    bucket_idx = -1
            return buckets[bucket_idx]
        # recover the bucket for special handling
        else:
            horizontal_dim = horizontal_dim[0][0]
            vertical_dim = vertical_dim[0][0]
            prefix_buckets = []
            prefill_buckets = []
            for b in buckets:
                if b[0] not in prefill_buckets:
                    prefill_buckets.append(b[0])
                if b[1] not in prefix_buckets:
                    prefix_buckets.append(b[1])
            # Corner case
            total_context = vertical_dim + horizontal_dim
            if total_context <= 512 and total_context > 256:
                for b in buckets:
                    if b[0] == 512 and b[1] == 0:
                        return b

            # Select prefill bucket
            prefill_index = 0
            if self.neuron_config.enable_eagle_speculation:
                vertical_dim = vertical_dim + self.neuron_config.pa_block_size
            for b in prefill_buckets:
                if vertical_dim > b:
                    prefill_index += 1
                else:
                    break
            # check prefill overflow
            if prefill_index == len(prefill_buckets):
                if not self.neuron_config.allow_input_truncation:
                    raise ValueError(
                        f"Prefill len {vertical_dim} exceeds largest bucket ({prefill_buckets[-1]}) for {self.tag}"
                    )
                else:
                    prefill_index = len(prefill_buckets) - 1
            # Select prefix bucket
            prefill_len = prefill_buckets[prefill_index]
            empty_prefill_slots = max(0, prefill_len - vertical_dim)
            if self.neuron_config.enable_eagle_speculation:
                # Calculate how many blocks can be moved from prefix to prefill.
                empty_prefill_block_slots = empty_prefill_slots // self.neuron_config.pa_block_size
                horizontal_dim = max(0, horizontal_dim - empty_prefill_block_slots * self.neuron_config.pa_block_size)
            else:
                horizontal_dim = max(0, horizontal_dim - empty_prefill_slots)
            prefix_index = 0
            for b in prefix_buckets:
                if horizontal_dim > b:
                    prefix_index += 1
                else:
                    break
            # TODO: Handle this corner scenario by using the largest prefix bucket and up the prefill bucket
            assert prefix_index != len(prefix_buckets), f"Prefix len {horizontal_dim} exceeds largest bucket {prefix_buckets[-1]} for {self.tag}"
            bucket_idx = prefill_index * len(prefix_buckets) + prefix_index
            return buckets[bucket_idx]

    def get_target_2d_bucket_for_chunked_prefill(self, *args):
        """
        Get the expected bucket for chunked prefill, for context encoding model
        only

        For now, this func only supports the "first_fit" strategy
        """
        buckets = self.neuron_config.buckets
        assert buckets == sorted(buckets), f"buckets should be in ascending order, but it is {buckets}"
        assert self.tag == CONTEXT_ENCODING_MODEL_TAG

        input_ids = args[0]  # shape (batch_size, seq_len)
        actual_chunk_size = input_ids.shape[1]

        tile_q_indices = args[15]  # shape (num_tiles)
        actual_num_tiles = tile_q_indices.shape[0]

        for bucket in buckets:
            bucket_chunk_size, bucket_num_tiles = bucket
            # If it is equal to the bucket size, the bucket can be used because
            # it relies on slot mapping to update KV cache.
            if actual_chunk_size <= bucket_chunk_size and actual_num_tiles <= bucket_num_tiles:
                return bucket

        raise ValueError(
            f"Can't find a bucket from {self.neuron_config.buckets} for "
            f"chunked prefill {self.tag}"
        )

    def _pad_prefix_caching_inputs(self, *args, pad_type="first_fit"):
        if self.tag == CONTEXT_ENCODING_MODEL_TAG and args[0].shape[0] > 1:
            # We delay all paddings for CTE until we really need them
            return args
        # Calculate the buckets
        prefill_bucket, prefix_bucket = self.get_target_2d_bucket_for_prefix_caching(*args, strategy=pad_type)

        if self.tag == CONTEXT_ENCODING_MODEL_TAG:
            if self.neuron_config.enable_fused_speculation:
                slot_mapping = args[7]
                block_table = args[8]
                prefill_len = args[9][0]
                prefix_len = args[10][0]
            else:
                slot_mapping = args[11]
                block_table = args[12]
                prefill_len = args[13][0]
                prefix_len = args[14][0]
            if self.neuron_config.enable_eagle_speculation:
                target_recomputation = 0 if prefix_bucket == 0 else self.neuron_config.pa_block_size
                extra_prefill_slots = max(0, prefill_bucket - prefill_len - target_recomputation)
                extra_prefill_blocks = extra_prefill_slots // self.neuron_config.pa_block_size
                adjusted_prefix_len = max(0, prefix_len - extra_prefill_blocks * self.neuron_config.pa_block_size)
                target_adjusted_prefix_len = max(0, adjusted_prefix_len - target_recomputation)

                sliced_inputs = args[0][:, adjusted_prefix_len:]
                padded_inputs = F.pad(sliced_inputs, (0, prefill_bucket - sliced_inputs.shape[1]), "constant", self.config.pad_token_id)
                target_sliced_inputs = args[0][:, target_adjusted_prefix_len:]
                target_padded_inputs = F.pad(target_sliced_inputs, (0, prefill_bucket - target_sliced_inputs.shape[1]), "constant", self.config.pad_token_id)

                sliced_attn_mask = args[1][:, :adjusted_prefix_len].clone()
                target_sliced_attn_mask = args[1][:, :target_adjusted_prefix_len]
                if prefix_bucket == 0:
                    padded_attn_mask = torch.zeros(1, dtype=torch.int)
                    target_padded_attn_mask = torch.zeros(1, dtype=torch.int)
                else:
                    sliced_attn_mask[:, 0] = 0  # Mask out the first draft position.
                    padded_attn_mask = F.pad(sliced_attn_mask, (0, prefix_bucket - sliced_attn_mask.shape[1]), "constant", 0)
                    # Can be all 0s when target_prefix_len is 0.
                    target_padded_attn_mask = F.pad(target_sliced_attn_mask, (0, prefix_bucket - target_sliced_attn_mask.shape[1]), "constant", 0)

                sliced_position_id = args[2][:, adjusted_prefix_len:]
                padded_position_id = F.pad(sliced_position_id, (0, prefill_bucket - sliced_position_id.shape[1]), "constant", 1)
                target_sliced_position_id = args[2][:, target_adjusted_prefix_len:]
                target_padded_position_id = F.pad(target_sliced_position_id, (0, prefill_bucket - target_sliced_position_id.shape[1]), "constant", 1)

                padded_slot_mapping = F.pad(slot_mapping, (prefix_len - adjusted_prefix_len, 0), "constant", -1)
                padded_slot_mapping = F.pad(padded_slot_mapping, (0, prefill_bucket - padded_slot_mapping.shape[1]), "constant", -1)
                target_padded_slot_mapping = F.pad(slot_mapping, (prefix_len - target_adjusted_prefix_len, 0), "constant", -1)
                target_padded_slot_mapping = F.pad(target_padded_slot_mapping, (0, prefill_bucket - target_padded_slot_mapping.shape[1]), "constant", -1)

                num_blocks = prefix_bucket // self.neuron_config.pa_block_size
                if num_blocks == 0:
                    padded_block_table = torch.zeros(1, dtype=torch.int)
                    target_padded_block_table = torch.zeros(1, dtype=torch.int)
                else:
                    padded_block_table = F.pad(block_table, (0, num_blocks - block_table.shape[1]), "constant", 0)
                    target_padded_block_table = F.pad(block_table, (0, num_blocks - block_table.shape[1]), "constant", 0)
                args = (padded_inputs, padded_attn_mask, padded_position_id, *args[3:7], padded_slot_mapping, padded_block_table, *args[9:11], target_padded_inputs, target_padded_attn_mask, target_padded_position_id, target_padded_slot_mapping, target_padded_block_table)
                return tuple(args)
            else:
                extra_prefill_slots = max(0, prefill_bucket - prefill_len)
                adjusted_prefix_len = max(0, prefix_len - extra_prefill_slots)
                sliced_inputs = args[0][:, adjusted_prefix_len:]
                sliced_attn_mask = args[1][:, :adjusted_prefix_len]
                sliced_position_id = args[2][:, adjusted_prefix_len:]

                padded_inputs = F.pad(sliced_inputs, (0, prefill_bucket - sliced_inputs.shape[1]), "constant", self.config.pad_token_id)
                if prefix_bucket == 0:
                    padded_attn_mask = torch.zeros(1, dtype=torch.int)
                else:
                    padded_attn_mask = F.pad(sliced_attn_mask, (0, prefix_bucket - sliced_attn_mask.shape[1]), "constant", 0)
                padded_position_id = F.pad(sliced_position_id, (0, prefill_bucket - sliced_position_id.shape[1]), "constant", 1)
                padded_slot_mapping = F.pad(slot_mapping, (prefix_len - adjusted_prefix_len, 0), "constant", -1)
                padded_slot_mapping = F.pad(padded_slot_mapping, (0, prefill_bucket - padded_slot_mapping.shape[1]), "constant", -1)

                num_blocks = prefix_bucket // self.neuron_config.pa_block_size
                if num_blocks == 0:
                    padded_block_table = torch.zeros(1, dtype=torch.int)
                else:
                    padded_block_table = F.pad(block_table, (0, num_blocks - block_table.shape[1]), "constant", 0)
                if self.neuron_config.enable_fused_speculation:
                    args = (padded_inputs, padded_attn_mask, padded_position_id, *args[3:7], padded_slot_mapping, padded_block_table, *args[9:])
                else:
                    args = (padded_inputs, padded_attn_mask, padded_position_id, *args[3:11], padded_slot_mapping, padded_block_table, *args[13:])
                return tuple(args)
        else:
            padded_attn_mask = F.pad(args[1], (0, prefix_bucket - args[1].shape[1]), "constant", 0)
            attn_tkg_nki_kernel_enabled = (
                self.neuron_config.attn_block_tkg_nki_kernel_enabled
                or self.neuron_config.attn_tkg_nki_kernel_enabled
            )
            block_table_arg_idx = 8 if self.neuron_config.enable_fused_speculation else 12
            block_table = args[block_table_arg_idx]
            pad_right = (prefix_bucket // self.neuron_config.pa_block_size) - block_table.shape[1]
            block_table_padding = -1 if attn_tkg_nki_kernel_enabled else 0
            padded_block_table = F.pad(block_table, (0, pad_right), "constant", block_table_padding)
            new_args = list(args)
            new_args[1] = padded_attn_mask
            new_args[block_table_arg_idx] = padded_block_table
            return tuple(new_args)

    def _process_async_inputs(self, *args):
        """
        Process Async outputs as follows:

        Example inputs:
        (
            [
                [ranked_input_ids0, ranked_pos_ids0],
                ...,
                []
            ],
            tensor,
            position_ids_to_be_replaced,
            tensor,
            ...
        )

        Ranked inputs can be identified as lists.
        Another factor to consider is that ranked inputs are only passed
        to token gen models. Given that, we know that for normal tkg
        the only ranked input is the input_ids, but for fused speculation
        we know the ranked inputs include the input_ids and the position_ids.

        Given that info above, we reshape the inputs to the expected input shape.

        When we have multiple ranked inputs, this will result in
        replacing existing args with the ranked version. We do this with position_ids,
        as implied by the example input above.

        The return value of this function will be a tuple of the following form:
        (
            [ranked_input_ids0, ranked_input_ids1, ...],
            [ranked_attn_mask0, ranked_attn_mask1, ...],
            [ranked_pos_ids0, ranked_pos_ids1, ...],
            ...
        )
        """
        batch_size = self.neuron_config.batch_size
        if self.neuron_config.ctx_batch_size != self.neuron_config.tkg_batch_size:
            if (self.tag == CONTEXT_ENCODING_MODEL_TAG or self.tag == VISION_ENCODER_MODEL_TAG):
                batch_size = self.neuron_config.ctx_batch_size
            else:
                batch_size = self.neuron_config.tkg_batch_size

        n_active_tokens = self.neuron_config.n_active_tokens
        is_ranked_input = is_ranked_io(args[0])
        ranked_out_replacing = set()
        ranked_args = [[] for _ in range(self.neuron_config.local_ranks_size)]

        bucket_size = args[2].shape[1]  # position_ids shape
        if is_ranked_input:
            if self.tag == FUSED_SPECULATION_MODEL_TAG:  # fused spec case
                ranked_args = [
                    [args[0][i][1].reshape(batch_size, n_active_tokens)]
                    for i in range(self.neuron_config.local_ranks_size)
                ]
                ranked_out_replacing.add(0)  # input_ids
                ranked_out_replacing.add(1)  # attention_mask
                ranked_out_replacing.add(2)  # position_ids
                for i in range(self.neuron_config.local_ranks_size):
                    ranked_args[i].append(
                        args[0][i][2].reshape(batch_size, -1)  # attention_mask  # B x bucket_dim
                    )
                    ranked_args[i].append(
                        args[0][i][3].reshape(batch_size, 1)  # position_ids  # B x 1
                    )
            else:  # cte + tkg flow
                n_active_tokens = (
                    bucket_size if (self.tag == CONTEXT_ENCODING_MODEL_TAG or self.tag == VISION_ENCODER_MODEL_TAG) else n_active_tokens
                )
                ranked_args = [
                    [args[0][i][0].reshape(batch_size, n_active_tokens)]
                    for i in range(self.neuron_config.local_ranks_size)
                ]
                ranked_out_replacing.add(0)  # input_ids

        for argnum, arg in enumerate(args):
            if argnum in ranked_out_replacing:
                continue

            for i in range(self.neuron_config.local_ranks_size):
                if argnum == 0:
                    n_active_tokens = (
                        bucket_size if (self.tag == CONTEXT_ENCODING_MODEL_TAG or self.tag == VISION_ENCODER_MODEL_TAG) else n_active_tokens
                    )
                    arg = arg.reshape(batch_size, n_active_tokens)

                ranked_args[i].insert(argnum, arg)

        return self.model.nxd_model.forward_async(ranked_args)

    def _process_ranked_inputs(self, *args):
        """
        Process inputs to handle both CPU and ranked Neuron inputs.
        Any argument can be either ranked or non-ranked.

        Args:
            *args: Variable number of input arguments where each arg can be either:
                - Ranked format: [[rank0_tensor], [rank1_tensor], ...]
                Example for tp_degree=2:
                [
                    [tensor([1,2,3])],  # rank0's data
                    [tensor([4,5,6])]   # rank1's data
                ]

                - Non-ranked format: regular tensor on CPU
                Example: tensor([1,2,3,4,5,6])
                Will be copied to all ranks

        Returns:
            List[List]: Processed ranked inputs in format [[rank0_inputs], [rank1_inputs], ...]
            Example for 2 arguments, tp_degree=2:
            [
                [rank0_arg1, rank0_arg2],  # rank0's inputs
                [rank1_arg1, rank1_arg2]   # rank1's inputs
            ]
        """
        # Initialize ranked args for each rank
        ranked_args = [[] for _ in range(self.neuron_config.local_ranks_size)]

        # Process all arguments
        for arg in args:
            if is_ranked_io(arg):
                # Already distributed across ranks, just extract
                for i in range(self.neuron_config.local_ranks_size):
                    ranked_args[i].append(arg[i][0])
            else:
                # Single CPU tensor, copy to all ranks
                for i in range(self.neuron_config.local_ranks_size):
                    ranked_args[i].append(arg)

        return self.model.nxd_model.forward_ranked(ranked_args)

    def _process_args(self, *args):
        """
        Process None args in `inputs` to ensure a unified set of args for difference features, such as default, medusa, lora, and eagle_draft.
        Refer to `inputs` in `input_generator()` for the meaning of each arg.
        """
        seq_ids = args[3]
        input_batch_size = seq_ids.shape[0]

        # set hidden_states if None
        if args[5] is None:
            dummy_hidden_states = torch.zeros((input_batch_size), dtype=torch.int32)
            args = (*args[:5], dummy_hidden_states, *args[6:])

        # set adapter_ids if None
        if args[6] is None:
            dummy_adapter_ids = torch.zeros((input_batch_size), dtype=torch.int32)
            args = (*args[:6], dummy_adapter_ids, *args[7:])
        return args

    def vllm_cte_repadding(self, batch_args):
        # This function is used to undo the padding from vllm serving for bs > 1 and
        # repad to nearest bucket size

        # retrieve the actual_len by using max of position ids (batch_args[2])
        # assumption here: the padding_id is 0
        actual_len = torch.max(batch_args[2]) + 1

        # Undo padding from vllm
        batch_args[0] = batch_args[0][:, :actual_len]
        batch_args[1] = batch_args[1][:, :actual_len]
        batch_args[2] = batch_args[2][:, :actual_len]

        # repad to nearest bucket
        batch_args = self.pad_inputs(*batch_args, pad_type="first_fit")

        return batch_args

    def forward(self, *args, pad_type="first_fit"):
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        args = self._process_args(*args)
        input_ids = args[0]
        args = tuple([arg for arg in args if isinstance(arg, torch.Tensor)])
        if is_ranked_io(input_ids):
            args = list([input_ids] + list(args))

        # convert int64 to int32 to improve compatibility with compiler; need to apply to cpu case
        args = self.convert_int64_to_int32(*args)

        args = self.pad_inputs(*args, pad_type=pad_type)

        seq_ids = args[3]

        input_batch_size = seq_ids.shape[0]

        if input_batch_size == self.neuron_config.batch_size:
            outputs = self._forward(*args)
            if self.async_mode:
                return AsyncTensorWrapper(async_result=outputs, batch_padded=False, on_cpu=False)

            return outputs

        cur_batch = 0
        output_logits = []

        logging.debug(
            f"get input_batch_size as {input_batch_size} but compiled batch_size as {self.neuron_config.batch_size}"
        )
        was_padded = False
        on_cpu = False
        while cur_batch < input_batch_size:
            if cur_batch + self.neuron_config.batch_size <= input_batch_size:
                # we only process part of the input to run
                logging.debug(
                    f"running foward on batch {cur_batch}:{cur_batch + self.neuron_config.batch_size}"
                )

                # pad to next bucket for context encoding with bs > 1
                # batch_arg represent single prompt in batch of prompts
                batch_args = [
                    arg[cur_batch : cur_batch + self.neuron_config.batch_size] for arg in args
                ]
                batch_args = self.vllm_cte_repadding(batch_args)

                outputs = self._forward(*batch_args)

                # sequential execution must be done in sync mode to avoid buffer write race conditions
                if self.async_mode:
                    on_cpu = True
                    outputs = get_async_output(outputs)
            else:
                # we need to pad the input to run
                logging.debug(
                    f"running forward on batch {cur_batch}:{input_batch_size}, padded up to {self.neuron_config.batch_size}"
                )
                was_padded = True
                outputs = self._forward_with_pad(
                    *[
                        arg[cur_batch:input_batch_size] if not is_ranked_io(arg) else arg
                        for arg in args
                    ]
                )

                # indicates uneven division of batch for sequential execution scenario, which must be run in sync mode
                if len(output_logits) > 0 and self.async_mode:
                    on_cpu = True
                    outputs = get_async_output(outputs)

            if self.is_neuron():
                logits = outputs
            else:
                logits, *kv_caches = outputs
                for i, kv_cache in enumerate(kv_caches):
                    self.model.kv_mgr.past_key_values[i].data = kv_cache

            output_logits.append(logits)
            cur_batch += self.neuron_config.batch_size

        if self.is_neuron():
            if self.async_mode:
                if not on_cpu:
                    # length of the concat list will be 1
                    output_logits = output_logits[0]
                return AsyncTensorWrapper(
                    async_result=output_logits, batch_padded=was_padded, on_cpu=on_cpu
                )
            elif self.neuron_config.enable_fused_speculation or \
                    (self.neuron_config.on_device_sampling_config and self.neuron_config.output_logits and self.neuron_config.is_continuous_batching):
                output_logits = [torch.cat(x, dim=0) for x in zip(*output_logits)]
                return output_logits

            return torch.cat(output_logits, dim=0)
        else:
            return [torch.cat(output_logits, dim=0), *kv_caches]


class DecoderModelInstance(BaseModelInstance):
    def __init__(self, model_cls, config: InferenceConfig, **kwargs):
        self.model_cls = model_cls
        self.module = None
        self.input_output_aliases = None
        self.config = config
        self.neuron_config = config.neuron_config
        self.kwargs = kwargs if kwargs is not None else {}

    def initialize_process_group(self, world_size):
        self.model_cls.initialize_process_group(world_size)

    def load_module(self):
        float_model = self.model_cls(self.config, **self.kwargs)
        float_model.eval()

        if self.neuron_config.cast_type == "config":
            if self.neuron_config.torch_dtype != torch.float32:
                float_model._apply(
                    lambda t: (
                        t.to(self.neuron_config.torch_dtype)
                        if t.is_floating_point()
                        and t.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]
                        else t
                    )
                )

                # TODO: In the current case we initialize the float_model which has Quantization layers as well
                # the above code will convert fp32 scales to bfloat16. This should be fixed when we remove
                # Quantization layers from NeuronLLamaMLP
                for name, param in float_model.named_parameters():
                    if name.endswith("scale"):
                        param.data = param.data.to(torch.float32)

        if self.neuron_config.quantized or self.neuron_config.is_mlp_quantized():
            quantization_type = QuantizationType(self.neuron_config.quantization_type)
            if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
                q_config = get_default_per_channel_custom_qconfig_dict()
            elif quantization_type == QuantizationType.EXPERT_WISE_PER_CHANNEL_SYMMETRIC:
                q_config = get_default_expert_wise_per_channel_custom_qconfig_dict()
            elif quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
                q_config = get_default_custom_qconfig_dict()
            else:
                raise RuntimeError(f"{self.neuron_config.quantization_type} is not supported")
            if self.neuron_config.quantization_dtype == "f8e4m3":
                q_config["quantized_dtype"] = QuantizedDtype.F8E4M3

            q_config["activation_quantization_type"] = ActivationQuantizationType(
                self.neuron_config.activation_quantization_type
            )
            q_config["clamp_bound"] = self.neuron_config.quantize_clamp_bound

            """
            The below code handles the conversion of modules for quantization:

            1. If fused speculation is enabled:
                - Iterate named children of the float_model in models_to_convert: draft_model and target_model
            2. If not fused speculation:
                - Stores the entire float_model in models_to_convert
            Note: The conversions are done in place
            """

            models_to_convert = []
            if self.neuron_config.enable_fused_speculation:
                models_to_convert = [float_model.draft_model, float_model.target_model]
            else:
                models_to_convert.append(float_model)

            for model in models_to_convert:
                convert(
                    model,
                    q_config=q_config,
                    inplace=True,
                    mapping=None,
                    modules_to_not_convert=get_modules_to_not_convert(model.config.neuron_config),
                )
            self.module = float_model

        else:
            self.module = float_model

    def get(self, bucket_rank, **kwargs):
        if bucket_rank is not None:
            if self.neuron_config.is_prefix_caching:
                self.module.prefix_size = self.neuron_config.buckets[bucket_rank][1]
                self.module.n_active_tokens = self.neuron_config.buckets[bucket_rank][0]
                if self.config.neuron_config.n_active_tokens == 1:
                    self.module.n_positions = self.neuron_config.buckets[bucket_rank][1]
                    if self.neuron_config.enable_fused_speculation:
                        self.module.draft_model.prefix_size = self.module.prefix_size
                        self.module.target_model.prefix_size = self.module.prefix_size
                        self.module.draft_model.n_active_tokens = self.module.n_active_tokens
                        self.module.target_model.n_active_tokens = self.neuron_config.speculation_length
                else:
                    self.module.n_positions = min(self.module.prefix_size + self.module.n_active_tokens, self.neuron_config.max_context_length)
                    if self.neuron_config.enable_fused_speculation:
                        self.module.draft_model.prefix_size = self.module.prefix_size
                        self.module.target_model.prefix_size = self.module.prefix_size
                        self.module.draft_model.n_active_tokens = self.module.n_active_tokens
                        self.module.target_model.n_active_tokens = self.module.n_active_tokens
            else:
                self.module.n_positions = self.neuron_config.buckets[bucket_rank]
            if self.neuron_config.enable_fused_speculation:
                self.module.draft_model.n_positions = self.module.n_positions
                self.module.target_model.n_positions = self.module.n_positions

        # Currently we have to init an input_output_aliases map for
        # each buckets, otherwise it will fail the aliasing setup when
        # generating HLO
        self.input_output_aliases = {}
        num_output_from_trace = 1 if not self.neuron_config.output_logits else 2
        if self.neuron_config.enable_fused_speculation:
            num_output_from_trace += 1
            if self.module.draft_model.kv_mgr is not None:
                draft_past_key_values = self.module.draft_model.kv_mgr.past_key_values
            else:
                draft_past_key_values = self.module.draft_model.past_key_values

            if self.module.target_model.kv_mgr is not None:
                target_past_key_values = self.module.target_model.kv_mgr.past_key_values
            else:
                target_past_key_values = self.module.target_model.past_key_values

            for i in range(len(draft_past_key_values)):
                self.input_output_aliases[draft_past_key_values[i]] = num_output_from_trace * 2 + i
            for j in range(len(target_past_key_values)):
                self.input_output_aliases[target_past_key_values[j]] = (
                    num_output_from_trace * 2 + len(draft_past_key_values)
                ) + j

            if self.neuron_config.enable_eagle_speculation:
                self.input_output_aliases[self.module.hidden_state_rolling_buffer.hidden_states] = (
                    num_output_from_trace * 2 + len(draft_past_key_values)
                ) + len(target_past_key_values)

        else:
            # TODO: This else block is a short-term fix for Llava/ViT models to use DecoderModelInstance.
            #       Long-term, these models should use a different implementation of BaseModelInstance.
            if self.module.kv_mgr is not None:
                past_key_values = self.module.kv_mgr.past_key_values
            else:
                past_key_values = self.module.past_key_values
            for i in range(len(past_key_values)):
                self.input_output_aliases[past_key_values[i]] = num_output_from_trace + i
        return self.module, self.input_output_aliases


class EncoderModelInstance(BaseModelInstance):
    def __init__(self, model_cls, config: InferenceConfig, **kwargs):
        """Copied from DecoderModelInstance.__init__()"""
        self.model_cls = model_cls
        self.module = None
        self.input_output_aliases = None
        self.config = config
        self.neuron_config = config.neuron_config
        self.kwargs = kwargs if kwargs is not None else {}

    def load_module(self):
        """Copied from DecoderModelInstance.load_module()"""
        # TODO: we should consider move this to BaseModelInstance
        float_model = self.model_cls(self.config, **self.kwargs)
        float_model.eval()

        if self.neuron_config.cast_type == "config":
            if self.neuron_config.torch_dtype != torch.float32:
                float_model._apply(
                    lambda t: (
                        t.to(self.neuron_config.torch_dtype)
                        if t.is_floating_point()
                        and t.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]
                        else t
                    )
                )

                # TODO: In the current case we initialize the float_model which has Quantization layers as well
                # the above code will convert fp32 scales to bfloat16. This should be fixed when we remove
                # Quantization layers from NeuronLLamaMLP
                for name, param in float_model.named_parameters():
                    if name.endswith("scale"):
                        param.data = param.data.to(torch.float32)

        if self.neuron_config.quantized is True and not (self.neuron_config.is_mlp_quantized()):
            quantization_type = QuantizationType(self.neuron_config.quantization_type)
            if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
                q_config = get_default_per_channel_custom_qconfig_dict()
            elif quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
                q_config = get_default_custom_qconfig_dict()
            else:
                raise RuntimeError(f"{self.neuron_config.quantization_type} is not supported")
            if self.neuron_config.quantization_dtype == "f8e4m3":
                q_config["quantized_dtype"] = QuantizedDtype.F8E4M3
            self.module = convert(float_model, q_config=q_config, inplace=True, mapping=None)
        else:
            self.module = float_model

    def get(self, bucket_rank, **kwargs):
        # TODO: Add aliasing and caching. Check out how DecoderModelInstance uses KVCacheManager
        self.input_output_aliases = {}
        return self.module, self.input_output_aliases


def get_trace_callable(model_cls, config: InferenceConfig, bucket_rank=None):
    if bucket_rank is not None:
        config.neuron_config.n_positions = config.neuron_config.buckets[bucket_rank]
    float_model = model_cls(config)
    float_model.eval()
    if config.neuron_config.torch_dtype != torch.float32:
        float_model.to(config.neuron_config.torch_dtype)

    if config.neuron_config.quantized:
        quantization_type = QuantizationType(config.neuron_config.quantization_type)
        if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            q_config = get_default_per_channel_custom_qconfig_dict()
        elif quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            q_config = get_default_custom_qconfig_dict()
        else:
            raise RuntimeError(f"{config.neuron_config.quantization_type} is not supported")
        model = convert(
            float_model,
            q_config=q_config,
            inplace=True,
            mapping=None,
            modules_to_not_convert=get_modules_to_not_convert(config.neuron_config),
        )
    else:
        model = float_model

    aliases = {}
    num_output_from_trace = 1
    for i in range(len(model.kv_mgr.past_key_values)):
        aliases[model.kv_mgr.past_key_values[i]] = num_output_from_trace + i
    return model, aliases
