# coding=utf-8
# Copyright 2025 The Qwen Team. All rights reserved.
# Adapted for Qwen2.5-VL NxDI implementation.
#
# Top-level VL orchestrator: wires text decoder + vision encoder through
# NeuronBaseForImageToText and ImageToTextInferenceConfig.
# Pattern follows qwen2_vl/modeling_qwen2_vl.py.

import copy
import logging
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.model_wrapper import VISION_ENCODER_MODEL_TAG
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
)

from src.modeling_qwen2_5_vl_text import (
    NeuronQwen2_5_VLTextModel,
    NeuronQwen2_5_VLTextForCausalLM,
    Qwen2_5_VLTextModelWrapper,
)
from src.modeling_qwen2_5_vl_vision import (
    NeuronQwen2_5_VLVisionModel,
    NeuronQwen2_5_VLForImageEncoding,
    Qwen2_5_VLVisionModelWrapper,
)

logger = logging.getLogger("Neuron")

# Keys to propagate from top-level config to text_config
QWEN2_5_VL_TEXT_CONFIG_KEYS = [
    "hidden_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "vocab_size",
    "intermediate_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "hidden_act",
    "bos_token_id",
    "eos_token_id",
    "qkv_bias",
    "o_bias",
    "vision_token_id",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
]

# Default pixels per image for bucket validation
DEFAULT_PIXELS_PER_IMAGE = 1024


class Qwen2_5_VLInferenceConfig(ImageToTextInferenceConfig):
    """Configuration for Qwen2.5-VL multimodal model."""

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
        self.add_special_config()
        self.validate_model_supported_configs()

    def add_special_config(self):
        """Set Qwen2.5-VL-specific config values."""
        self.num_cores_per_group = 1
        self.qkv_bias = True
        self.o_bias = False

        # Vision config: compute head_dim from hidden_size and num_heads
        if hasattr(self, "vision_config") and hasattr(
            self.vision_config, "hidden_size"
        ):
            self.vision_config.head_dim = (
                self.vision_config.hidden_size // self.vision_config.num_heads
            )
            self.vision_config.num_cores_per_group = 1

        # Vision config: set default image dimensions and bucket
        # vllm's HF processor may resize images larger than 640x640, producing
        # more patches than bucket=[1] with 640x640 pixels_per_image can hold.
        # Use bucket=[2] for headroom (supports images up to ~2x pixels_per_image patches).
        if hasattr(self, "vision_config") and hasattr(
            self.vision_config, "neuron_config"
        ):
            vnc = self.vision_config.neuron_config
            if (
                not hasattr(vnc, "default_image_width")
                or vnc.default_image_width is None
            ):
                vnc.default_image_width = 672
            if (
                not hasattr(vnc, "default_image_height")
                or vnc.default_image_height is None
            ):
                vnc.default_image_height = 672
            # Default vision buckets to [2] for vllm compatibility
            if not hasattr(vnc, "buckets") or vnc.buckets is None or vnc.buckets == []:
                vnc.buckets = [2]
                logger.info("Qwen2.5-VL vision: set default buckets=[2]")

        # Copy text config keys
        if hasattr(self, "text_config"):
            for key in QWEN2_5_VL_TEXT_CONFIG_KEYS:
                if hasattr(self, key):
                    setattr(self.text_config, key, getattr(self, key))
            self.pad_token_id = getattr(self.text_config, "pad_token_id", None)

    def validate_model_supported_configs(self):
        # Ensure text_config matches top-level config (set if missing)
        for key in QWEN2_5_VL_TEXT_CONFIG_KEYS:
            if hasattr(self, key) and hasattr(self.text_config, key):
                top_val = getattr(self, key)
                text_val = getattr(self.text_config, key)
                if top_val != text_val:
                    logger.warning(
                        f"Config mismatch: {key}: top={top_val} vs text={text_val}. "
                        f"Setting text_config.{key} = {top_val}"
                    )
                    setattr(self.text_config, key, top_val)

        # Disable unsupported text features
        for unsupported in [
            "is_block_kv_layout",
            "is_prefix_caching",
            "is_chunked_prefill",
            "is_medusa",
            "enable_fused_speculation",
        ]:
            if getattr(self.text_config.neuron_config, unsupported, False) is not False:
                setattr(self.text_config.neuron_config, unsupported, False)
                logger.warning(
                    f"Qwen2.5-VL text model does not support '{unsupported}'. Disabled."
                )

        # Disable unsupported vision features
        for unsupported in [
            "sequence_parallel_enabled",
            "flash_decoding_enabled",
            "qkv_kernel_enabled",  # Fused RMSNorm+QKV fails: eps type mismatch with vision RMSNorm
            "attn_block_tkg_nki_kernel_cache_update",
            "attn_block_tkg_nki_kernel_enabled",
        ]:
            if (
                getattr(self.vision_config.neuron_config, unsupported, False)
                is not False
            ):
                setattr(self.vision_config.neuron_config, unsupported, False)
                logger.warning(
                    f"Qwen2.5-VL vision: '{unsupported}' unsupported, disabled."
                )

        # Vision encoder requires fused_qkv -- enforce it
        if not getattr(self.vision_config.neuron_config, "fused_qkv", False):
            self.vision_config.neuron_config.fused_qkv = True
            logger.warning("Qwen2.5-VL vision: fused_qkv was not set, forcing to True.")

        # Text model also requires fused_qkv
        if not getattr(self.text_config.neuron_config, "fused_qkv", False):
            self.text_config.neuron_config.fused_qkv = True
            logger.warning("Qwen2.5-VL text: fused_qkv was not set, forcing to True.")

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.pad_token_id",
            "text_config.vocab_size",
            "text_config.max_position_embeddings",
            "text_config.rope_theta",
            "text_config.rms_norm_eps",
            "text_config.hidden_act",
            "vision_config.depth",
            "vision_config.hidden_size",
            "vision_config.num_heads",
            "vision_config.in_chans",
            "vision_config.patch_size",
            "vision_config.spatial_merge_size",
            "vision_config.temporal_patch_size",
            "vision_config.intermediate_size",
            "vision_config.out_hidden_size",
            "vision_config.window_size",
            "vision_config.fullatt_block_indexes",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronQwen2_5_VLForCausalLM(NeuronBaseForImageToText):
    """Top-level Qwen2.5-VL model for NxDI inference."""

    text_model_cls = NeuronQwen2_5_VLTextModel
    vision_model_cls = NeuronQwen2_5_VLVisionModel
    text_model_wrapper = Qwen2_5_VLTextModelWrapper
    vision_model_wrapper = Qwen2_5_VLVisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

    def get_vision_compiler_args(self) -> str:
        cc_factor = self.vision_config.neuron_config.cc_pipeline_tiling_factor
        return (
            f"--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc_factor}' -O1 "
            f"--hbm-scratchpad-page-size=1024 "
            f"--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def get_compiler_args(self) -> str:
        cc_factor = self.text_config.neuron_config.cc_pipeline_tiling_factor
        return (
            f"--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc_factor}' -O1 "
            f"--hbm-scratchpad-page-size=1024 "
            f"--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def get_required_kwargs(self) -> List[str]:
        return ["pixel_values", "vision_mask", "image_grid_thw"]

    def enable_vision_encoder(self, enable_wlt_optimization=True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=False,
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights (embed_tokens == lm_head) for models like 3B.
        Delegates to text model class which clones embed_tokens -> lm_head."""
        NeuronQwen2_5_VLTextForCausalLM.update_state_dict_for_tied_weights(state_dict)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: "Qwen2_5_VLInferenceConfig"
    ) -> dict:
        """Convert full HF state dict: split into vision + text conversion."""
        state_dict = NeuronQwen2_5_VLForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, inference_config
        )
        state_dict = NeuronQwen2_5_VLTextForCausalLM.convert_hf_to_neuron_state_dict(
            state_dict, inference_config.text_config
        )
        return state_dict

    def get_padding_length(self, input_ids):
        buckets = self.context_encoding_model.config.neuron_config.buckets
        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise Exception("No bucket found for provided input_ids!")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        pad_limit = self.get_padding_length(input_ids)

        if (
            pixel_values is not None
            and input_ids.shape[-1] > 1
            and pixel_values.sum() != 0
        ):
            # Compute vision mask from image_token_id positions
            vision_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            vision_mask = vision_mask.to(torch.bool)
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())
            vision_mask = pad_positions(vision_mask, pad_limit, (pad_limit - 1))

            # Run vision encoder
            vision_embeddings = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype),
                image_grid_thw,
            )
        else:
            # Text-only or decode step -- use dummy vision inputs
            vision_embeddings, vision_mask = (
                self.text_model_wrapper.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_limit,
                    fill_value=(pad_limit - 1),
                )
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            input_capture_hook=input_capture_hook,
            tensor_capture_hook=tensor_capture_hook,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

    @classmethod
    def get_config_cls(cls):
        return Qwen2_5_VLInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        return NeuronQwen2_5_VLForImageEncoding.prepare_input_args(
            prompts, images, processor, role, config
        )
