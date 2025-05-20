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
            tensorizer_options = (
                "--enable-ccop-compute-overlap "
                f"--cc-pipeline-tiling-factor={self.neuron_config.cc_pipeline_tiling_factor} "
                "--vectorize-strided-dma "
            )

            long_ctx_reqs = ""
            max_len_gt_32k = self.neuron_config.seq_len >= 32 * 1024
            if max_len_gt_32k and self.neuron_config.flash_decoding_enabled:
                long_ctx_reqs = "--hbm-scratchpad-page-size=1024 "
                os.environ["NEURON_SCRATCHPAD_PAGE_SIZE"] = "1024"
                os.environ["NEURON_RT_EXEC_TIMEOUT"] = "600"

            self.compiler_args = (
                f"--auto-cast=none --model-type=transformer {long_ctx_reqs}"
                f"--tensorizer-options='{tensorizer_options}'"
                f" --lnc={self.neuron_config.logical_nc_config}"
            )

            if tag == CONTEXT_ENCODING_MODEL_TAG:
                # force Modular flow for CTE model to save compile time
                # TKG model can use default (-O2) to avoid function call overheads with Modular flow
                self.compiler_args += " -O1 "
                # Set threshold low so that modular flow always kicks in for context encoding graph.
                # This is needed because lots of mac-ops happen in kernels which are not visible to the graph partitioner at the moment.
                hlo2tensorizer += " --modular-flow-mac-threshold=10 "

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
            self.compiler_args += f" --internal-hlo2tensorizer-options='{hlo2tensorizer}' "

        logging.info(f"neuronx-cc compiler_args are: {self.compiler_args}")

        self.bucket_config = None
        self.priority_model_idx = priority_model_idx
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
                attention_mask = torch.ones((batch_size, bucket[1]), dtype=torch.int32)
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
            if self.tag != CONTEXT_ENCODING_MODEL_TAG and self.tag != MEDUSA_MODEL_TAG:
                position_ids = torch.ones((batch_size, n_active_tokens), dtype=torch.int32)
            else:
                position_ids = torch.arange(0, n_active_tokens, dtype=torch.int32).unsqueeze(0)
                position_ids = position_ids.repeat(batch_size, 1)
            seq_ids = torch.arange(0, batch_size, dtype=torch.int32)
            adapter_ids = torch.zeros((batch_size), dtype=torch.int32)

            # Get the count of sampling params currently supported.
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros((batch_size, sampling_params_len), dtype=torch.float32)
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
                slot_mapping = torch.zeros(n_active_tokens, dtype=torch.int32)

                active_block_table = torch.zeros(
                    (self.neuron_config.cp_num_active_blocks), dtype=torch.int32
                )

                # number of queries in current request only
                # set to max for now, but needs to be bucketed
                num_queries = torch.zeros(self.neuron_config.cp_max_num_seqs, dtype=torch.int32)

                # length of context excluding the current request
                context_lens = torch.zeros(self.neuron_config.cp_max_num_seqs, dtype=torch.int32)

                max_total_len = (
                    self.neuron_config.cp_num_active_blocks * self.neuron_config.pa_block_size
                    + self.neuron_config.max_context_length
                )

                cache_mask = torch.zeros(max_total_len, dtype=torch.bool)
                cache_to_ctx = torch.zeros(max_total_len, dtype=torch.int32)
                active_to_ctx = torch.zeros(max_total_len, dtype=torch.int32)

                inputs.append(
                    (
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
                        context_lens,
                        cache_mask,
                        active_to_ctx,
                        cache_to_ctx,
                    )
                )
            elif self.is_prefix_caching:
                slot_mapping = torch.zeros((batch_size, n_active_tokens), dtype=torch.int32)

                active_block_table = torch.zeros(
                    (batch_size, prefix_size // self.neuron_config.pa_block_size), dtype=torch.int32
                )

                # number of queries in current request only
                # set to max for now, but needs to be bucketed
                num_queries = torch.zeros((batch_size, 1), dtype=torch.int32)

                # length of context excluding the current request
                computed_context_lens = torch.full((batch_size, 1), prefix_size, dtype=torch.int32)

                inputs.append(
                    (
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
                )
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

    def get_model_instance(self):
        return DecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )

    def _forward_with_pad(self, *args):
        seq_ids = args[3]
        sampling_params = args[4]
        if self.is_block_kv_layout:
            medusa_args = None
        elif len(args) > 5:
            medusa_args = args[5:8]
        else:
            medusa_args = None

        if self.is_prefix_caching:
            block_kv_empty_args = args[5:11]
            block_kv_args = args[11:15]

        # pad the inputs up to the compiled batch size in the end
        def pad_helper(tensor, pad_type="zeros", batch_sort_indices=None):
            VALID_PAD_TYPES = set(["zeros", "ones", "repeat_first_batchline"])
            assert (
                pad_type in VALID_PAD_TYPES
            ), f"Found {pad_type=}, but valid pad types are {VALID_PAD_TYPES}"
            if tensor is None or tensor.shape[0] == self.neuron_config.batch_size:
                return tensor

            padded_shape = list(tensor.shape)
            padded_shape[0] = self.neuron_config.batch_size
            if pad_type == "repeat_first_batchline":
                # pad with first batch line values instead of zeros, to reduce chances of NaN
                padded_tensor = tensor[0].unsqueeze(0).repeat(padded_shape[0], 1).to(tensor.dtype)
            else:
                fill_value = 0 if pad_type == "zeros" else 1
                padded_tensor = torch.full(padded_shape, fill_value=fill_value, dtype=tensor.dtype)
            padded_tensor[: tensor.shape[0]] = tensor

            if batch_sort_indices is not None:
                padded_tensor = torch.index_select(padded_tensor, 0, batch_sort_indices)

            return padded_tensor

        # need to handle seq_ids separately, when compiled batch is 4, if we pad seq_ids from [0,2,1] to [0,2,1,
        # 0]. then the kv cache of padded input could be written into the first cache line, so we need to pad as [0,
        # 2, 1, 3] instead

        seq_ids_list = seq_ids.tolist()
        padded_seq_ids = torch.tensor(
            seq_ids_list
            + [x for x in range(self.neuron_config.max_batch_size) if x not in seq_ids_list],
            dtype=seq_ids.dtype,
        )
        padded_seq_ids, indices = torch.sort(padded_seq_ids)

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
                    batch_sort_indices=indices if not self.is_prefix_caching else None,
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
            batch_sort_indices=indices if not self.is_prefix_caching else None,
        )
        padded_args.append(padded_sampling_params)

        if medusa_args is not None:
            for arg in medusa_args:
                padded_args.append(pad_helper(arg, batch_sort_indices=indices))

        if self.is_prefix_caching:
            for arg in block_kv_empty_args:
                padded_args.append(arg)
            for arg in block_kv_args:
                padded_args.append(pad_helper(arg, pad_type="repeat_first_batchline"))

        outputs = self._forward(*padded_args)

        # note that we don't do index select here as it should already be handled, simply sliced out padding here
        if self.is_neuron():
            logits = outputs
            if self.async_mode:
                return logits
            elif self.is_prefix_caching:
                return logits[: seq_ids.shape[0]]
            elif self.neuron_config.enable_fused_speculation:
                returned_logits = [torch.index_select(logit, 0, seq_ids) for logit in logits]
                return returned_logits
            return torch.index_select(logits, 0, seq_ids)
        else:
            logits, *kv_cache = outputs
            return [torch.index_select(logits, 0, seq_ids), *kv_cache]

    def _forward(self, *args):
        if self.async_mode:
            return self._process_async_inputs(*args)

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
        VALID_PAD_TYPES = {"max", "first_fit", "second_fit"}
        assert (
            pad_type in VALID_PAD_TYPES
        ), f"Found {pad_type=}, when it should be one of: {VALID_PAD_TYPES=}"
        pad_length = (
            self.neuron_config.max_context_length
            if self.tag == CONTEXT_ENCODING_MODEL_TAG
            else self.neuron_config.max_length
        )
        if isinstance(self.neuron_config.buckets[-1], list) and self.neuron_config.is_prefix_caching:
            # 2D bucketing for prefix caching
            pad_lengths = self.get_target_2d_bucket_for_prefix_caching(*args, strategy=pad_type)
            if self.tag == CONTEXT_ENCODING_MODEL_TAG and self.neuron_config.is_prefix_caching:
                padded_inputs = F.pad(args[0], (0, pad_lengths[0] - args[0].shape[1]), "constant", self.config.pad_token_id)
                padded_attn_mask = F.pad(args[1], (0, pad_lengths[1] - args[1].shape[1]), "constant", 0)
                padded_position_id = F.pad(args[2], (0, pad_lengths[0] - args[2].shape[1]), "constant", 1)
                padded_slot_mapping = F.pad(args[11], (0, pad_lengths[0] - args[11].shape[1]), "constant", -1)
                padded_block_table = F.pad(args[12], (0, (pad_lengths[1] // self.neuron_config.pa_block_size) - args[12].shape[1]), "constant", 0)
                args = (padded_inputs, padded_attn_mask, padded_position_id, *args[3:11], padded_slot_mapping, padded_block_table, *args[13:])
                return tuple(args)
            else:
                padded_attn_mask = F.pad(args[1], (0, pad_lengths[1] - args[1].shape[1]), "constant", 0)
                padded_block_table = F.pad(args[12], (0, (pad_lengths[1] // self.neuron_config.pa_block_size) - args[12].shape[1]), "constant", 0)
                args = (*args[0:1], padded_attn_mask, *args[2:12], padded_block_table, *args[13:])
                return tuple(args)

        if pad_type == "first_fit" or pad_type == "second_fit":
            pad_length = self.get_target_bucket(*args, strategy=pad_type)

        if self.tag == CONTEXT_ENCODING_MODEL_TAG:
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

            if self.is_chunked_prefill:
                args = list(args)

                # Pad arguments and stack List args.
                cp_max_num_seqs = self.neuron_config.cp_max_num_seqs
                max_context_length = self.neuron_config.max_context_length
                max_total_len = (
                    self.neuron_config.cp_num_active_blocks * self.neuron_config.pa_block_size
                    + self.neuron_config.max_context_length
                )

                cache_to_ctx = args[-1]
                cache_to_ctx = F.pad(cache_to_ctx, [0, max_total_len - cache_to_ctx.shape[0]])
                args[-1] = cache_to_ctx

                active_to_ctx = args[-2]
                active_to_ctx = F.pad(active_to_ctx, [0, max_total_len - active_to_ctx.shape[0]])
                args[-2] = active_to_ctx

                cache_mask = args[-3]
                cache_mask = F.pad(cache_mask, [0, max_total_len - cache_mask.shape[0]])
                args[-3] = cache_mask

                context_lens = args[-4]
                context_lens = F.pad(context_lens, [0, cp_max_num_seqs - context_lens.shape[0]])
                args[-4] = context_lens

                num_queries = args[-5]
                num_queries = F.pad(num_queries, [0, cp_max_num_seqs - num_queries.shape[0]])
                args[-5] = num_queries

                active_block_table = args[-6]
                cp_num_active_blocks = self.neuron_config.cp_num_active_blocks
                active_block_table = F.pad(
                    active_block_table, [0, cp_num_active_blocks - len(active_block_table)]
                )
                args[-6] = active_block_table

                slot_mapping = args[-7]
                # need to be padded by -1 to avoid overriding existing KV cache.
                slot_mapping = F.pad(
                    slot_mapping, [0, max_context_length - slot_mapping.shape[0]], value=-1
                )
                args[-7] = slot_mapping
                args = tuple(args)
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
        buckets = torch.tensor(self.neuron_config.buckets)
        assert strategy != "second_fit", f"2D bucketing does not support {strategy}"
        vertical_dim = args[13]
        horizontal_dim = args[14]
        mask = (buckets[:, 0] >= vertical_dim) & (buckets[:, 1] >= horizontal_dim)
        index = torch.nonzero(mask)
        bucket_idx = index[0][1]
        return buckets[bucket_idx]

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
            if self.tag == CONTEXT_ENCODING_MODEL_TAG:
                batch_size = self.neuron_config.ctx_batch_size
            else:
                batch_size = self.neuron_config.tkg_batch_size

        n_active_tokens = self.neuron_config.n_active_tokens
        is_ranked_input = is_ranked_io(args[0])
        ranked_out_replacing = set()
        ranked_args = [[] for _ in range(self.neuron_config.local_ranks_size)]

        bucket_size = args[1].shape[1]  # attn mask shape
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
                    bucket_size if self.tag == CONTEXT_ENCODING_MODEL_TAG else n_active_tokens
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
                        bucket_size if self.tag == CONTEXT_ENCODING_MODEL_TAG else n_active_tokens
                    )
                    arg = arg.reshape(batch_size, n_active_tokens)

                ranked_args[i].insert(argnum, arg)

        return self.model.nxd_model.forward_async(ranked_args)

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

        # convert int64 to int32 to improve compatibility with compiler; does not apply to cpu case
        if not self.neuron_config.on_cpu:
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
            elif self.neuron_config.enable_fused_speculation:
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
                self.module.n_positions = min(self.module.prefix_size + self.module.n_active_tokens, self.neuron_config.max_context_length)
            else:
                self.module.n_positions = self.neuron_config.buckets[bucket_rank]
            if self.neuron_config.enable_fused_speculation:
                self.module.draft_model.n_positions = self.neuron_config.buckets[bucket_rank]
                self.module.target_model.n_positions = self.neuron_config.buckets[bucket_rank]

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
