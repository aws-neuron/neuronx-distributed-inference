from gemma3_vision.ndxi_patch import apply_patch
apply_patch()

import copy # noqa: E402
import math # noqa: E402
import logging # noqa: E402
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any # noqa: E402

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed.quantization.quantization_utils import convert_qint8_to_int8_state_dict
import neuronx_distributed_inference.modules.autobucketing as autobucketing
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
    ImageToTextModelWrapper,
    IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import pad_vision_embeddings
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    VISION_ENCODER_MODEL_TAG
)
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM

from gemma3_vision.modeling_gemma3_text import NeuronGemma3TextModel
from gemma3_vision.modeling_gemma3_vision import NeuronGemma3VisionModel, Gemma3VisionModelWrapper
from gemma3_vision.utils import convert_state_dict_to_fused_qkv, StateDict

logger = logging.getLogger("Neuron")


class Gemma3InferenceConfig(ImageToTextInferenceConfig):
    def __init__(
        self,
        text_neuron_config,
        vision_neuron_config,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )

        # NeuronLlamaMLP expects the activation type to be at text_config.hidden_act
        # Enable to fully reuse NeuronLlamaMLP
        if not hasattr(self.text_config, "hidden_act"):
            self.text_config.hidden_act = self.text_config.hidden_activation
            del self.text_config.hidden_activation

        if self.text_config.neuron_config.is_block_kv_layout:
            raise ValueError("Gemma3 does not yet support block_kv_layout.")
        if self.text_config.neuron_config.is_prefix_caching:
            raise ValueError("Gemma3 does not yet support prefix_caching.")
        if self.text_config.neuron_config.is_chunked_prefill:
            raise ValueError("Gemma3 does not yet support chunked_prefill.")
        if self.text_config.neuron_config.is_medusa:
            raise ValueError("Gemma3 does not yet support medusa.")
        if self.text_config.neuron_config.enable_fused_speculation:
            raise ValueError("Gemma3 does not yet support fused speculation.")

        if self.neuron_config.flash_decoding_enabled:
            # Following pixtral implementation, we use REPLICATE_TO_TP_DEGREE as the sharding_strategy
            # Hence attn_heads are padded to become divisible by tp_degree
            num_attn_heads, num_kv_heads = self.text_config.num_attention_heads, self.text_config.num_key_value_heads
            num_attn_heads = (num_attn_heads // self.neuron_config.tp_degree + 1) * self.neuron_config.tp_degree
            self.text_config.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "text_config.head_dim", # for gemma3, head_dim != hidden_size // num_attention_heads
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.query_pre_attn_scalar",
            "text_config.rope_scaling",
            "text_config.sliding_window",
            "vision_config.hidden_size",
            "vision_config.image_size",
            "vision_config.num_attention_heads",
            "vision_config.num_hidden_layers",
            "vision_config.patch_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class TextGemma3InferenceConfig(InferenceConfig):

    def __init__(
        self,
        neuron_config: NeuronConfig,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(
            neuron_config=neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )

        # NeuronLlamaMLP expects the activation type to be at text_config.hidden_act
        # Enable to fully reuse NeuronLlamaMLP
        if not hasattr(self, "hidden_act"):
            self.hidden_act = self.hidden_activation
            del self.hidden_activation

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim", # for gemma3, head_dim != hidden_size // num_attention_heads
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "query_pre_attn_scalar",
            "rope_scaling",
            "sliding_window",
        ]


class NeuronGemma3ForConditionalGeneration(NeuronBaseForImageToText):
    # model cls
    text_model_cls = NeuronGemma3TextModel
    vision_model_cls = NeuronGemma3VisionModel

    # model wrappers
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = Gemma3VisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls):
        # Gemma3-specific
        return Gemma3InferenceConfig

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        # Identical to NeuronPixtralForCausalLM.enable_vision_encoder
        # - except use get_compiler_args + VISION_ENCODER_MODEL_TAG (instead of get_vision_compiler_args)
        #   like NeuronLlama4ForCausalLM.enable_vision_encoder
        self.compile_tag = VISION_ENCODER_MODEL_TAG

        new_config = copy.deepcopy(self.config)
        if new_config.vision_config.neuron_config.enable_bucketing:
            # neuron_config.buckets default to neuron_config.seq_len is not given. For vision we want to do auto-bucketing here
            if new_config.vision_config.neuron_config.buckets == [new_config.vision_config.neuron_config.seq_len] or \
                    new_config.vision_config.neuron_config.buckets is None:
                # 1024 vision seq len corresponds to a single 512x512 image. Smaller bucket size does not make sense in real life.
                if new_config.vision_config.neuron_config.seq_len > 1024:
                    new_config.vision_config.neuron_config.buckets = autobucketing.generate_buckets(
                        1024, new_config.vision_config.neuron_config.seq_len
                    )
                else:
                    new_config.vision_config.neuron_config.buckets = [new_config.vision_config.neuron_config.seq_len]
        # This should not be needed as in vision modeling code we should always use vision_config.neuron_config as vision model's neuron config
        # added this line just to add insurance to avoid mix-up
        new_config.neuron_config = copy.deepcopy(new_config.vision_config.neuron_config)

        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            # to turn on weight layout optimization
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=True
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: StateDict) -> None:
    # Gemma3-specific
        try:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
        except KeyError:
            state_dict["embed_tokens.weight"] = state_dict["lm_head.weight"].clone()

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: StateDict, inference_config: InferenceConfig) -> StateDict:
    # Gemma3-specific
        neuron_config = inference_config.neuron_config
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
            ".self_attn.out_proj.": ".self_attn.o_proj.o_proj.", # for siglip
            ".self_attn.q_norm.": ".self_attn.q_layernorm.",
            ".self_attn.k_norm.": ".self_attn.k_layernorm.",
        }

        # At the time of writing, NxDI (Neuron 2.26) attention layer does not provide a simple way to use a custom
        # scaling factor for raw attention scores (QK^T) while ensuring all optimizations (e.g. kernels) remain available
        # To work around this, we fuse the scaling factor into the weights (knowing that the attention layer will use the
        # default math.sqrt(inference_config.head_dim) value)
        default_qk_scaling_factor_inv = math.sqrt(float(inference_config.text_config.query_pre_attn_scalar))
        gemma_qk_scaling_factor = 1.0 / math.sqrt(float(inference_config.text_config.head_dim))
        gamma = math.sqrt(gemma_qk_scaling_factor * default_qk_scaling_factor_inv)

        new_state_dict = {}
        for key, weights in state_dict.items():
            if 'language_model.model.' in key:
                key = key.replace('language_model.model.', "")
                for atten_key in attention_keys:
                    if atten_key in key:
                        replacement_atten_key = attention_keys[atten_key]
                        key = key.replace(atten_key, replacement_atten_key)
                        break
                if key.endswith((".q_proj.weight", ".k_proj.weight")):
                    orig_dtype = weights.dtype
                    weights = (weights.to(dtype=torch.float32) * gamma).to(dtype=orig_dtype)
            if 'language_model.lm_head.' in key:
                key = key.replace('language_model.', "")
            if 'vision_tower.' in key:
                key = key.replace('vision_tower.', 'vision_encoder.')
                for atten_key in attention_keys:
                    if atten_key in key:
                        replacement_atten_key = attention_keys[atten_key]
                        key = key.replace(atten_key, replacement_atten_key)
                        break
            new_state_dict[key] = weights

        # If LNC > 1, model requires lm_head.bias which is equivalent to lm_head_pad
        if "language_model.lm_head.bias" not in state_dict and inference_config.neuron_config.lm_head_pad:
            # Use embed_tokens.weight instead of lm_head.weight as lm_head.weight is tied to embed_tokens.weight in Gemma3
            new_state_dict["lm_head.bias"] = torch.zeros(new_state_dict["embed_tokens.weight"].shape[0], dtype=torch.float32)

        if inference_config.text_config.neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(
                state_dict=new_state_dict,
                num_layers=inference_config.text_config.num_hidden_layers,
                neuron_config=inference_config.text_config.neuron_config,
                prefix="layers.{layer_num}.self_attn"
                )

        if inference_config.vision_config.neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(
                state_dict=new_state_dict,
                num_layers=inference_config.vision_config.num_hidden_layers,
                neuron_config=inference_config.vision_config.neuron_config,
                prefix="vision_encoder.vision_model.encoder.layers.{layer_num}.self_attn"
                )

        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.rank_util.rank"] = torch.arange(0, neuron_config.local_ranks_size)

        tp_degree = neuron_config.tp_degree
        for i in range(inference_config.text_config.num_hidden_layers):
            new_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return new_state_dict

    @staticmethod
    def _convert_input_dict_to_ordered_tuple(input_dict: Dict[str, Any]):
        # Identical NeuronLlama4ForCausalLM._convert_input_dict_to_ordered_tuple, to be removed?
        """
        Utility function to convert input dictionary to ordered tuple
        based on outputs of _get_model_outputs
        """
        args = []

        for key in IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS:
            if key in input_dict and input_dict[key] is not None:
                arg = input_dict[key]
            else:
                arg = torch.empty(0)
            args.append(arg)

        return tuple(args)

    def _select_buckets_for_padding_length(self, position_ids):
        # Identical to NeuronLlama4ForCausalLM._select_buckets_for_padding_length
        neuron_config = self.config.neuron_config
        context_encoding_buckets = neuron_config.context_encoding_buckets if neuron_config.context_encoding_buckets is not None \
            else neuron_config.buckets
        token_generation_buckets = neuron_config.token_generation_buckets if neuron_config.token_generation_buckets is not None \
            else neuron_config.buckets

        selected_buckets = token_generation_buckets
        if self._is_prefill(position_ids):
            selected_buckets = context_encoding_buckets

        return selected_buckets

    @staticmethod
    def get_padding_length(buckets, position_ids):
        # Identical to [NeuronLlama4ForCausalLM|NeuronPixtralForCausalLM]._select_buckets_for_padding_length
        max_position_id = torch.max(position_ids).item()
        for val in buckets:
            if val > max_position_id:
                return val
        raise ValueError("No bucket found for provided input_ids!")

    @staticmethod
    def get_required_kwargs() -> List[str]:
        # Gemma3-specific
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return [
            "pixel_values",
            "vision_mask",
        ]

    @staticmethod
    def generate_positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
        # Gemma3-specific
        """
        Generate position indices from a boolean mask.
        Compared to generate_positions_from_mask() of models/llama4/utils/encoder_utils.py,
        this function can generate 1D or 2D masks to support batch size > 1.

        Args:
        mask (torch.Tensor): A 1D or 2D boolean tensor

        Returns:
        torch.Tensor: A 1D or 2D tensor containing the indices where the mask is True
        """
        if mask.dim() == 1:
            return torch.nonzero(mask).squeeze()
        else:
            rows, cols = torch.nonzero(mask, as_tuple=True)
            row_counts = torch.bincount(rows, minlength=mask.shape[0])
            cols_per_row = torch.split(cols, row_counts.tolist())
            return rnn_utils.pad_sequence(cols_per_row, batch_first=True, padding_value=0)

    @staticmethod
    def pad_positions(positions: torch.LongTensor, target_size: int, fill_value: float) -> torch.LongTensor:
        """
        Pad the positions tensor to a target size.
        Compared to pad_positions() of models/llama4/utils/encoder_utils.py,
        this function can support batch size > 1.

        Args:
        positions (torch.Tensor): A 1D or 2D tensor containing position indices
        target_size (int): The desired size of the padded tensor
        fill_value (int): The value used for padding

        Returns:
        torch.Tensor: A 3D tensor of shape (batch_size, target_size, 1) containing padded position indices
        """
        # positions_2d of shape (batch_sz, seq_len)
        positions_2d = positions.unsqueeze(0) if positions.dim() == 1 else positions
        padding_size = target_size - positions_2d.shape[1]
        assert padding_size >= 0, "Text model sequence length is not enough to handle all vision embeddings"
        positions_padded = F.pad(positions_2d, (0, padding_size), value=fill_value)
        # output tensor of shape (batch_sz, target_sz, 1)
        return positions_padded.unsqueeze(-1)

    @staticmethod
    def _create_position_ids(attention_mask_2d: torch.LongTensor, is_prefill: bool) -> torch.LongTensor:
        position_ids = attention_mask_2d.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask_2d == 0, 1)
        if is_prefill:
            return position_ids
        else:
            return torch.amax(position_ids, dim=1, keepdim=True) + 1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Very close to NeuronLlama4ForCausalLM.forward
        is_prefill = (input_ids.shape[-1] > 1)
        include_images = (pixel_values is not None) and (vision_mask is not None) and (pixel_values.sum() != 0)

        if position_ids is None:
            position_ids = self._create_position_ids(attention_mask_2d=attention_mask, is_prefill=is_prefill)

        buckets = self._select_buckets_for_padding_length(position_ids=position_ids)
        pad_target_size = self.get_padding_length(buckets=buckets, position_ids=position_ids)
        pad_fill_value = (pad_target_size - 1)
        if (is_prefill and include_images):
            assert (
                vision_mask.dtype == torch.bool
            ), f"Parameter `vision_mask` must be of type bool, recieved {vision_mask.dtype}"
            # Call the vision encoder to create a sequence of vision token embeddings for each input image
            #   pixel_values of shape (batch_sz * img_per_sample, 3, height, width)
            vision_embeddings = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype),
            ).to(self.text_config.neuron_config.torch_dtype)

            # Flatten vision embeddings: required if img_per_sample > 1
            #   vision_embeddings of shape (batch_sz * img_per_sample, seq_len_per_image, embedding_dim)
            #   vision_mask of shape (batch_sz, total_seq_len)
            batch_sz = 1 if (vision_mask.dim() == 1) else vision_mask.shape[0]
            num_images, seq_len, embedding_dim = vision_embeddings.shape
            img_per_sample = num_images // batch_sz
            vision_embeddings = vision_embeddings.view(batch_sz, img_per_sample * seq_len, embedding_dim)

            # Sequences of vision token embeddings are padded to the bucket size the text model has been compiled with
            vision_embeddings = pad_vision_embeddings(vision_embeddings=vision_embeddings, pad_limit=pad_target_size)

            # Positions used to scatter vision embeddings at specific positions into the sequence passed to the text model
            # are created from the vision mask
            vision_mask = self.generate_positions_from_mask(mask=vision_mask.squeeze())
            vision_mask = self.pad_positions(
                positions=vision_mask,
                target_size=pad_target_size,
                fill_value=pad_fill_value
            )
        else:
            # Either token generation or text-only prefill -> still need dummy inputs for the compiled text model
            vision_embeddings, vision_mask = self.context_encoding_model.get_dummy_vision_inputs(
                config=self.text_config,
                input_ids=input_ids,
                n_active_tokens=pad_target_size,
                fill_value=pad_fill_value
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

    def enable_token_generation(self):
        # Identical to NeuronLlama4ForCausalLM.enable_token_generation -> Required for get_compiler_args to succeed
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def enable_context_encoding(self):
        # Identical to NeuronLlama4ForCausalLM.enable_context_encoding -> Required for get_compiler_args to succeed
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def get_compiler_args(self) -> str:
        # Identical to NeuronLlama4ForCausalLM.get_compiler_args
        logical_nc_config = self.text_config.neuron_config.logical_nc_config

        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O2"
        elif self.compile_tag == VISION_ENCODER_MODEL_TAG:
            return f"-O1 --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap' " \
                   f"--auto-cast=none --lnc={logical_nc_config}"
        else:
            raise ValueError(f"get_compiler_args() Invalid compile tag encountered: {self.compile_tag}")

        args = f"--auto-cast=none --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap " \
               f"--cc-pipeline-tiling-factor=1 --vectorize-strided-dma --enable-scalar-dge-vectorization' " \
               f"--lnc={logical_nc_config} {optimization_level} "
        return args

    @classmethod
    def prepare_quantized_state_dict(cls, hf_model_quant):
        # Gemma3-specific
        model_quant_sd = hf_model_quant.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        return model_quant_sd

    def _get_constructed_outputs(self, outputs, is_run_on_neuron):
        if self.on_device_sampling and self.text_config.neuron_config.output_logits and not \
                (self.text_config.neuron_config.enable_fused_speculation or self.text_config.neuron_config.is_medusa):
            logits_or_next_tokens = outputs[:2]
            constructed_outputs = self._construct_output_with_tokens_and_logits(next_tokens=logits_or_next_tokens[0], logits=logits_or_next_tokens[1])
        else:
            if is_run_on_neuron:
                # FIX: Remove updated KV cache tensor (outputs[1])
                logits_or_next_tokens = logits_or_next_tokens = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            else:
                # When run on cpu, KV cache is returned which has to be ignored
                logits_or_next_tokens, *_ = outputs
            constructed_outputs = self._construct_output(logits_or_next_tokens)

        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug("---output---")
            logging.debug(
                f"{'tokens' if self.on_device_sampling else 'logits'} = %s, ",
                logits_or_next_tokens,
            )

        return constructed_outputs

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Gemma3ForConditionalGeneration, Gemma3Config
        config = Gemma3Config.from_pretrained(model_path)
        model = Gemma3ForConditionalGeneration.from_pretrained(model_path, config=config).eval()
        return model


class NeuronTextGemma3ForCausalLM(NeuronBaseForCausalLM):

    _model_cls = NeuronGemma3TextModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Gemma3ForCausalLM
        return Gemma3ForCausalLM.from_pretrained(model_path, **kwargs) # nosec B615

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: StateDict) -> None:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: StateDict, inference_config: InferenceConfig) -> StateDict:
        neuron_config = inference_config.neuron_config
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
            ".self_attn.q_norm.": ".self_attn.q_layernorm.",
            ".self_attn.k_norm.": ".self_attn.k_layernorm.",
        }

        # At the time of writing, NxDI (Neuron 2.26) attention layer does not provide a simple way to use a custom
        # scaling factor for raw attention scores (QK^T) while ensuring all optimizations (e.g. kernels) remain available
        # To work around this, we fuse the scaling factor into the weights (knowing that  the attention layer will use the
        # default math.sqrt(inference_config.head_dim) value)
        default_qk_scaling_factor_inv = math.sqrt(float(inference_config.query_pre_attn_scalar))
        gemma_qk_scaling_factor = 1.0 / math.sqrt(float(inference_config.head_dim))
        gamma = math.sqrt(gemma_qk_scaling_factor * default_qk_scaling_factor_inv)

        new_state_dict = {}
        for key, weights in state_dict.items():
            if 'vision_tower.' in key:
                continue
            if 'language_model.model.' in key:
                key = key.replace('language_model.model.', "")
            for atten_key in attention_keys:
                if atten_key in key:
                    replacement_atten_key = attention_keys[atten_key]
                    key = key.replace(atten_key, replacement_atten_key)
                    break
            if key.endswith((".q_proj.weight", ".k_proj.weight")):
                orig_dtype = weights.dtype
                weights = (weights.to(dtype=torch.float32) * gamma).to(dtype=orig_dtype)
            new_state_dict[key] = weights

        if neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(
                state_dict=new_state_dict,
                num_layers=inference_config.num_hidden_layers,
                neuron_config=inference_config.neuron_config,
                prefix="layers.{layer_num}.self_attn"
                )

        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.rank_util.rank"] = torch.arange(0, neuron_config.local_ranks_size)

        tp_degree = neuron_config.tp_degree
        for i in range(inference_config.num_hidden_layers):
            new_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return new_state_dict

    @classmethod
    def get_config_cls(cls):
        return TextGemma3InferenceConfig
