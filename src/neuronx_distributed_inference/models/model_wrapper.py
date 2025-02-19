import logging
import os
from functools import partial

import torch
import torch.nn.functional as F
from neuronx_distributed.quantization.quantization_config import (
    QuantizationType,
    QuantizedDtype,
    get_default_custom_qconfig_dict,
    get_default_per_channel_custom_qconfig_dict,
)
from neuronx_distributed.quantization.quantize import convert
from neuronx_distributed.trace import parallel_model_load, parallel_model_trace
from neuronx_distributed.trace.model_builder import BaseModelInstance
from torch_neuronx import BucketModelConfig

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.autobucketing import (
    get_context_encoder_bk,
    get_generation_model_bk,
)
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"
SPECULATION_MODEL_TAG = "speculation_model"
MEDUSA_MODEL_TAG = "medusa_speculation_model"
FUSED_SPECULATION_MODEL_TAG = "fused_speculation_model"


def get_bucket_model_config_from_tag(tag, config: InferenceConfig):
    bucket_degree = len(config.neuron_config.buckets)
    if bucket_degree == 1:
        return None

    pad_token = config.pad_token_id

    # NOTE: KV Cache preprocessing is done within the model and not the
    # shared buffer preprocessor due to lack of support of non-contiguous
    # slicing of nrt tensors via the NRT API.
    if tag == CONTEXT_ENCODING_MODEL_TAG:
        return BucketModelConfig(
            bucket_kernel=get_context_encoder_bk,
            bucket_kernel_constant_args=(
                torch.tensor(config.neuron_config.buckets),
                config.neuron_config.padding_side,
                pad_token,
            ),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
    elif (
        tag == TOKEN_GENERATION_MODEL_TAG
        or tag == SPECULATION_MODEL_TAG
        or tag == FUSED_SPECULATION_MODEL_TAG
    ):
        return BucketModelConfig(
            bucket_kernel=get_generation_model_bk,
            bucket_kernel_constant_args=(
                torch.tensor(config.neuron_config.buckets),
                config.neuron_config.padding_side,
                config.neuron_config.speculation_length
                if tag == FUSED_SPECULATION_MODEL_TAG
                else 0,
            ),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
    else:
        raise ValueError(
            f"The supplied tag: {tag} is not supported for Bucketing. Only {CONTEXT_ENCODING_MODEL_TAG} and {TOKEN_GENERATION_MODEL_TAG} are supported"
        )


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

        if config.pad_token_id is None:
            config.pad_token_id = 0

        self.model_cls = model_cls
        self.model = None
        self.is_compiled = False
        self.serialize_base_path = None
        self.tag = tag
        self.is_medusa = config.neuron_config.is_medusa

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
        self.compiler_workdir = os.path.join(base_compile_work_dir, self.tag)

        if compiler_args is None:
            tensorizer_options = (
                "--enable-ccop-compute-overlap "
                f"--cc-pipeline-tiling-factor={self.neuron_config.cc_pipeline_tiling_factor} "
                "--vectorize-dge-dma "
                "--vectorize-strided-dma "
            )

            if self.neuron_config.enable_fused_speculation:
                tensorizer_options += "--optimize-alias-chain "

            self.compiler_args = (
                "--auto-cast=none --model-type=transformer "
                f"--tensorizer-options='{tensorizer_options}'"
                " -O1 "
                f" --internal-num-neuroncores-per-sengine={self.neuron_config.logical_neuron_cores}"
            )

            if self.neuron_config.target:
                self.compiler_args += f" --target {self.neuron_config.target}"

        else:
            self.compiler_args = compiler_args

        if (
            (
                self.neuron_config.quantized is True
                and self.neuron_config.quantization_dtype == "f8e4m3"
            )
            or self.neuron_config.kv_cache_quant
            or self.neuron_config.quantized_mlp_kernel_enabled
        ):
            self.compiler_args += (
                " --internal-hlo2tensorizer-options='--experimental-unsafe-fp8e4m3fn-as-fp8e4m3' "
            )

        logging.info(f"neuronx-cc compiler_args are: {self.compiler_args}")

        self.bucket_config = get_bucket_model_config_from_tag(tag, self.config)
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
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket
                if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )

            input_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            attention_mask = torch.zeros((self.neuron_config.batch_size, bucket), dtype=torch.int32)
            position_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)
            adapter_ids = (
                torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)
                if self.neuron_config.lora_config is not None
                else None
            )
            # Get the count of sampling params currently supported.
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros(
                (self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32
            )
            if self.neuron_config.on_device_sampling_config:
                if self.neuron_config.on_device_sampling_config.do_sample:
                    sampling_params[:, 0] = self.neuron_config.on_device_sampling_config.top_k
                    sampling_params[:, 1] = self.neuron_config.on_device_sampling_config.top_p
                    sampling_params[:, 2] = self.neuron_config.on_device_sampling_config.temperature
            hidden_states = (
                torch.zeros(
                    (self.neuron_config.batch_size, n_active_tokens, self.config.hidden_size),
                    dtype=self.config.neuron_config.torch_dtype,
                )
                if self.neuron_config.is_eagle_draft
                else None
            )

            if self.is_medusa:
                assert (
                    self.neuron_config.on_device_sampling_config
                ), "Medusa speculation must use on-device sampling"
                # Set top_k to signal to the sampler that we're not doing greedy sampling.
                # This affects the output shape for Medusa speculation
                sampling_params[:, 0] = self.neuron_config.on_device_sampling_config.top_k
                accepted_indices = torch.zeros(
                    (self.neuron_config.batch_size, self.neuron_config.num_medusa_heads + 1),
                    dtype=torch.int32,
                )
                current_length = torch.zeros(
                    (self.neuron_config.batch_size, self.neuron_config.num_medusa_heads + 1),
                    dtype=torch.int32,
                )
                medusa_mask = torch.zeros(
                    (
                        self.neuron_config.batch_size,
                        self.neuron_config.medusa_speculation_length,
                        self.neuron_config.medusa_speculation_length,
                    ),
                    dtype=torch.int32,
                )
                scatter_index = torch.zeros(
                    (self.neuron_config.batch_size, self.neuron_config.medusa_speculation_length),
                    dtype=torch.int32,
                )

                inputs.append(
                    (
                        input_ids,
                        attention_mask,
                        position_ids,
                        seq_ids,
                        sampling_params,
                        torch.empty((0)),  # prev_hidden
                        torch.empty((0)),  # adapter_ids
                        accepted_indices,
                        current_length,
                        medusa_mask,
                        scatter_index,
                    )
                )
            else:
                if self.neuron_config.lora_config is not None:
                    inputs.append(
                        (
                            input_ids,
                            attention_mask,
                            position_ids,
                            seq_ids,
                            sampling_params,
                            adapter_ids,
                        )
                    )
                elif self.neuron_config.is_eagle_draft:
                    inputs.append(
                        (
                            input_ids,
                            attention_mask,
                            position_ids,
                            seq_ids,
                            sampling_params,
                            hidden_states,
                        )
                    )
                else:
                    inputs.append(
                        (input_ids, attention_mask, position_ids, seq_ids, sampling_params)
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
        if len(args) > 5:
            medusa_args = args[5:8]
        else:
            medusa_args = None

        # pad the inputs up to the compiled batch size in the end
        def pad_helper(tensor, pad_type="zeros"):
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
            return padded_tensor

        padded_args = []
        # pad input_ids, attn_mask and position_ids
        for arg in args[0:3]:
            padded_args.append(pad_helper(arg, pad_type="repeat_first_batchline"))

        # need to handle seq_ids separately, when compiled batch is 4, if we pad seq_ids from [0,2,1] to [0,2,1,
        # 0]. then the kv cache of padded input could be written into the first cache line, so we need to pad as [0,
        # 2, 1, 3] instead

        seq_ids_list = seq_ids.tolist()
        padded_seq_ids = torch.tensor(
            seq_ids_list
            + [x for x in range(self.neuron_config.max_batch_size) if x not in seq_ids_list],
            dtype=seq_ids.dtype,
        )
        padded_args.append(padded_seq_ids)

        # pad sampling params by repeating first batchline
        padded_sampling_params = pad_helper(sampling_params, pad_type="repeat_first_batchline")
        padded_args.append(padded_sampling_params)

        if medusa_args is not None:
            for arg in medusa_args:
                padded_args.append(pad_helper(arg))

        outputs = self._forward(*padded_args)

        # note that we don't do index select here as it should already be handled, simply sliced out padding here
        if self.is_neuron():
            logits = outputs
            return logits[: seq_ids.shape[0]]
        else:
            logits, *kv_cache = outputs
            return [logits[: seq_ids.shape[0]], *kv_cache]

    def _forward(self, *args):
        logging.debug(f"Processed inputs to the model. tag={self.tag}, args={args}")
        return self.model(*args)

    def convert_int64_to_int32(self, *args):
        """
        Convert int64 args to int32 to match compiled input types.
        Neuron compiler handles int32 better than int64. Context: P165494809
        """
        return [t.to(torch.int32) if t.dtype == torch.int64 else t for t in args]

    def pad_to_max_compiled_seq(self, *args):
        if self.tag == CONTEXT_ENCODING_MODEL_TAG:
            to_pad = args[:3]
            pad_lengths = [self.neuron_config.max_context_length - arg.shape[1] for arg in to_pad]
            tensor_pad_vals = [self.config.pad_token_id, 0, 1]
            padded_args = [
                F.pad(arg, (0, pad_len), "constant", pad_val)
                for arg, pad_val, pad_len in zip(to_pad, tensor_pad_vals, pad_lengths)
            ]
            args = (*padded_args, *args[3:])
        else:
            input_ids, attention_mask, *rest_of_args = args
            pad_len = self.neuron_config.max_length - attention_mask.shape[1]
            padded_attention_mask = F.pad(attention_mask, (0, pad_len), "constant", 0)
            args = (input_ids, padded_attention_mask, *rest_of_args)

        return args

    def _get_async_output(self, ranked_async_tensor):
        outputs = [[async_tensor[0].cpu()] for async_tensor in ranked_async_tensor]
        return outputs[0][0]

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        if args[6] is None:
            # if adapter_ids for LoRA is None, pop it out
            args = (*args[:6], *args[7:])

        # remove hidden state if None
        # This is hacky and is being fixed
        if args[5] is None:
            args = (*args[:5], *args[6:])

        args = self.convert_int64_to_int32(*args)
        args = self.pad_to_max_compiled_seq(*args)

        seq_ids = args[3]

        input_batch_size = seq_ids.shape[0]

        if input_batch_size == self.neuron_config.batch_size:
            return self._forward(*args)

        cur_batch = 0
        output_logits = []

        logging.debug(
            f"get input_batch_size as {input_batch_size} but compiled batch_size as {self.neuron_config.batch_size}"
        )
        while cur_batch < input_batch_size:
            if cur_batch + self.neuron_config.batch_size <= input_batch_size:
                # we only process part of the input to run
                logging.debug(
                    f"running foward on batch {cur_batch}:{cur_batch + self.neuron_config.batch_size}"
                )
                outputs = self._forward(
                    *[arg[cur_batch : cur_batch + self.neuron_config.batch_size] for arg in args]
                )
            else:
                # we need to pad the input to run
                logging.debug(
                    f"running forward on batch {cur_batch}:{input_batch_size}, padded up to {self.neuron_config.batch_size}"
                )
                outputs = self._forward_with_pad(*[arg[cur_batch:input_batch_size] for arg in args])

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
                # block on all requests here, since this is output manipulation
                output_logits = [
                    self._get_async_output(ranked_logits) for ranked_logits in output_logits
                ]

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
                lambda t: t.to(self.neuron_config.torch_dtype)
                if t.is_floating_point() and t.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]
                else t
            )

            # TODO: In the current case we initialize the float_model which has Quantization layers as well
            # the above code will convert fp32 scales to bfloat16. This should be fixed when we remove
            # Quantization layers from NeuronLLamaMLP
            for name, param in float_model.named_parameters():
                if name.endswith("scale"):
                    param.data = param.data.to(torch.float32)

        if (
            self.neuron_config.quantized is True
            and not self.neuron_config.quantized_mlp_kernel_enabled
        ):
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
        if bucket_rank is not None:
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
                self.input_output_aliases[self.module.hidden_state] = (
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


def get_trace_callable(model_cls, config: InferenceConfig, bucket_rank=None):
    if bucket_rank is not None:
        config.neuron_config.n_positions = config.neuron_config.buckets[bucket_rank]
    float_model = model_cls(config)
    float_model.eval()
    if config.neuron_config.torch_dtype != torch.float32:
        float_model.to(config.neuron_config.torch_dtype)

    if config.neuron_config.quantized is True:
        quantization_type = QuantizationType(config.neuron_config.quantization_type)
        if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            q_config = get_default_per_channel_custom_qconfig_dict()
        elif quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            q_config = get_default_custom_qconfig_dict()
        else:
            raise RuntimeError(f"{config.neuron_config.quantization_type} is not supported")
        model = convert(float_model, q_config=q_config, inplace=True, mapping=None)
    else:
        model = float_model

    aliases = {}
    num_output_from_trace = 1
    for i in range(len(model.kv_mgr.past_key_values)):
        aliases[model.kv_mgr.past_key_values[i]] = num_output_from_trace + i
    return model, aliases
