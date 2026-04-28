# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NxDI contrib model for InternVL3-8B-Instruct (OpenGVLab).

Architecture: InternViT-300M vision encoder + pixel shuffle MLP projector + Qwen2.5-7B text backbone.

This is the top-level VLM class that inherits from NeuronBaseForImageToText,
orchestrating both the vision encoder (separate NEFF) and text decoder (CTE + TKG NEFFs).

Vision pipeline:
  pixel_values [B,3,448,448] -> InternViT -> strip CLS -> pixel_shuffle -> MLP projector
  -> [1, seq_len, 3584] padded vision embeddings

Text pipeline:
  input_ids -> embed_tokens -> scatter vision embeddings at <IMG_CONTEXT> positions
  -> 28 Qwen2.5 decoder layers -> lm_head -> logits

Special tokens:
  <IMG_CONTEXT> = 151667 (image placeholder token in input_ids)
  <img> = 151665 (image start)
  </img> = 151666 (image end)
"""

import copy
import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
)
from neuronx_distributed_inference.models.model_wrapper import VISION_ENCODER_MODEL_TAG

from modeling_internvl3_text import (
    InternVL3TextModelWrapper,
    NeuronInternVL3TextForCausalLM,
    NeuronInternVL3TextModel,
)
from modeling_internvl3_vision import (
    InternVL3VisionModelWrapper,
    NeuronInternVL3VisionModel,
    convert_vision_hf_to_neuron_state_dict,
)

logger = logging.getLogger("Neuron")

# InternVL3 special token ID for image context placeholder
IMG_CONTEXT_TOKEN_ID = 151667

# Keys from top-level config that must be copied to text_config
INTERNVL3_TEXT_CONFIG_KEYS = [
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
    "hidden_act",
    "bos_token_id",
    "eos_token_id",
    "tie_word_embeddings",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class InternVL3InferenceConfig(ImageToTextInferenceConfig):
    """
    Inference configuration for InternVL3 on Neuron.

    Requires two NeuronConfig objects:
    - text_neuron_config: for the Qwen2.5-7B text decoder (CTE + TKG)
    - vision_neuron_config: for the InternViT-300M vision encoder

    The HF config.json has text params under "llm_config" and vision params
    under "vision_config". This class handles the mapping.
    """

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
        """Set InternVL3-specific config defaults."""
        self.num_cores_per_group = 1

        # Qwen2.5 text backbone: QKV bias, no O bias
        self.qkv_bias = True
        self.o_bias = False

        # Image token ID for vision mask generation
        self.image_token_id = IMG_CONTEXT_TOKEN_ID

        # Copy text keys from top-level config to text_config
        for key in INTERNVL3_TEXT_CONFIG_KEYS:
            if hasattr(self, key):
                setattr(self.text_config, key, getattr(self, key))

        self.pad_token_id = getattr(self.text_config, "pad_token_id", 0)

    def validate_model_supported_configs(self):
        """Validate and disable unsupported NeuronConfig options."""
        # Validate text config keys match
        for key in INTERNVL3_TEXT_CONFIG_KEYS:
            if hasattr(self, key) and hasattr(self.text_config, key):
                top_val = getattr(self, key)
                text_val = getattr(self.text_config, key)
                if top_val != text_val:
                    logger.warning(
                        f"Config mismatch: {key} top={top_val} vs text={text_val}, using top"
                    )
                    setattr(self.text_config, key, top_val)

        # Disable unsupported text model features
        TEXT_UNSUPPORTED = [
            "is_block_kv_layout",
            "is_prefix_caching",
            "is_chunked_prefill",
            "is_medusa",
            "enable_fused_speculation",
        ]
        for cfg_name in TEXT_UNSUPPORTED:
            if getattr(self.text_config.neuron_config, cfg_name, False) is not False:
                setattr(self.text_config.neuron_config, cfg_name, False)
                logger.warning(
                    f"InternVL3 text model: '{cfg_name}' unsupported, disabled."
                )

        # Disable unsupported vision model features
        VISION_UNSUPPORTED = [
            "sequence_parallel_enabled",
            "flash_decoding_enabled",
            "qkv_kernel_enabled",
            "attn_block_tkg_nki_kernel_cache_update",
            "attn_block_tkg_nki_kernel_enabled",
        ]
        for cfg_name in VISION_UNSUPPORTED:
            if getattr(self.vision_config.neuron_config, cfg_name, False) is not False:
                setattr(self.vision_config.neuron_config, cfg_name, False)
                logger.warning(
                    f"InternVL3 vision model: '{cfg_name}' unsupported, disabled."
                )

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
            "vision_config.hidden_size",
            "vision_config.num_attention_heads",
            "vision_config.num_hidden_layers",
            "vision_config.image_size",
            "vision_config.patch_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        text_neuron_config=None,
        vision_neuron_config=None,
        **kwargs,
    ) -> "InternVL3InferenceConfig":
        """
        Load configuration from a pretrained InternVL3 model directory.

        InternVL3 config.json structure:
          - Top-level: model_type, downsample_ratio, force_image_size, etc.
          - llm_config: Qwen2.5-7B text params (model_type=qwen2)
          - vision_config: InternViT params (model_type=intern_vit_6b)

        The ImageToTextInferenceConfig parent expects "text_config" and
        "vision_config" keys in kwargs. We map llm_config -> text_config.
        """
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Extract text config (InternVL uses "llm_config", not "text_config")
        llm_config = config_dict.get("llm_config", {})
        vision_config = config_dict.get("vision_config", {})

        # Build the config dict that ImageToTextInferenceConfig expects
        # Must have "text_config" and "vision_config" at top level
        inference_kwargs = {}

        # Copy top-level InternVL params
        for key in [
            "downsample_ratio",
            "force_image_size",
            "select_layer",
            "model_type",
            "architectures",
            "tie_word_embeddings",
        ]:
            if key in config_dict:
                inference_kwargs[key] = config_dict[key]

        # Copy text params from llm_config -> text_config
        text_config_dict = {}
        for key in [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "tie_word_embeddings",
        ]:
            if key in llm_config:
                text_config_dict[key] = llm_config[key]

        # HF PretrainedConfig defaults required by NxDI model_base._setup_func_config()
        # These are normally set by HF's from_pretrained() but we build config manually
        text_config_dict.setdefault("output_attentions", False)
        text_config_dict.setdefault("output_hidden_states", False)
        # Note: do NOT set use_return_dict here — it's a read-only computed attribute
        # in PretrainedConfig and will raise AttributeError when HuggingFaceGenerationAdapter
        # converts our config via to_pretrained_config() → PretrainedConfig(**text_config_dict)

        # Also set at top level (required by ImageToTextInferenceConfig)
        for key, value in text_config_dict.items():
            inference_kwargs[key] = value

        inference_kwargs["text_config"] = text_config_dict

        # Copy vision config as-is
        inference_kwargs["vision_config"] = vision_config

        # Set image_token_id
        inference_kwargs["image_token_id"] = IMG_CONTEXT_TOKEN_ID

        # Set _name_or_path
        inference_kwargs["_name_or_path"] = model_path

        # Merge user kwargs
        inference_kwargs.update(kwargs)

        return cls(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            **inference_kwargs,
        )


# ---------------------------------------------------------------------------
# Top-level VLM class
# ---------------------------------------------------------------------------


class NeuronInternVL3ForCausalLM(NeuronBaseForImageToText):
    """
    InternVL3 vision-language model for Neuron inference.

    Orchestrates:
    - Vision encoder NEFF (InternViT-300M + pixel shuffle + projector)
    - Text decoder CTE NEFF (Qwen2.5-7B context encoding with vision embedding injection)
    - Text decoder TKG NEFF (Qwen2.5-7B token generation)

    Usage:
        text_nc = NeuronConfig(tp_degree=4, max_batch_size=1, seq_len=4096, ...)
        vision_nc = NeuronConfig(tp_degree=1, max_batch_size=1, buckets=[1], ...)
        config = InternVL3InferenceConfig.from_pretrained(
            model_path, text_neuron_config=text_nc, vision_neuron_config=vision_nc
        )
        model = NeuronInternVL3ForCausalLM(config)
        model.compile(compiled_path)
        model.load(compiled_path)
        output = model(input_ids, attention_mask, position_ids, seq_ids,
                       sampling_params, pixel_values=pixel_values)
    """

    text_model_cls = NeuronInternVL3TextModel
    vision_model_cls = NeuronInternVL3VisionModel
    text_model_wrapper = InternVL3TextModelWrapper
    vision_model_wrapper = InternVL3VisionModelWrapper

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
        """Compiler args for vision encoder NEFF."""
        return "--auto-cast=matmult --model-type=transformer -O1"

    def get_compiler_args(self) -> str:
        """Compiler args for text model NEFFs (CTE + TKG)."""
        return "--auto-cast=matmult --model-type=transformer -O1"

    def get_required_kwargs(self) -> List[str]:
        """Additional input args for HuggingFaceGenerationAdapter."""
        return ["pixel_values", "vision_mask"]

    def enable_vision_encoder(
        self, enable_wlt_optimization: bool = True, **model_init_kwargs
    ):
        """Create the vision encoder model wrapper."""
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

    def get_padding_length(self, input_ids):
        """Get the CTE bucket size for the given input length."""
        buckets = self.context_encoding_model.config.neuron_config.buckets
        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise RuntimeError(
            f"No bucket found for input_ids length {input_ids.shape[1]}. "
            f"Available buckets: {buckets}"
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass orchestrating vision encoder and text decoder.

        For context encoding with images:
          1. Identify <IMG_CONTEXT> token positions in input_ids
          2. Run vision encoder NEFF -> padded vision embeddings
          3. Pass vision_embeddings + vision_mask to text decoder CTE
          4. Inside CTE NEFF: embed_tokens -> scatter vision -> decoder layers -> logits

        For token generation or text-only:
          - Pass dummy (zero) vision tensors to text decoder TKG
        """
        # Work around NxDI issue: NeuronBaseForImageToText.forward() doesn't
        # capture preprocess_inputs() return values, so sampling_params=None
        # flows through to the compiled NEFF which expects [batch, 3].
        # Provide default sampling_params here if not supplied by caller.
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        pad_limit = self.get_padding_length(input_ids)

        if (
            pixel_values is not None
            and input_ids.shape[-1] > 1
            and pixel_values.sum() != 0
        ):
            # Context encoding with images
            # Build vision_mask: find <IMG_CONTEXT> positions in input_ids
            vision_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            vision_mask = vision_mask.to(torch.bool)
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())
            vision_mask = pad_positions(vision_mask, pad_limit, (pad_limit - 1))

            # Run vision encoder NEFF
            vision_embeddings = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype)
            )
        else:
            # Token generation or text-only: use dummy zeros
            vision_embeddings, vision_mask = (
                self.text_model_wrapper.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_limit,
                    fill_value=(pad_limit - 1),
                )
            )

        output_token = super().forward(
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
        return output_token

    @classmethod
    def get_config_cls(cls):
        return InternVL3InferenceConfig

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HF model for weight extraction. Returns None (load from safetensors)."""
        return None

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InternVL3InferenceConfig
    ) -> dict:
        """
        Convert full InternVL3 HF state dict to Neuron format.

        Delegates to:
        - convert_vision_hf_to_neuron_state_dict() for vision + projector weights
        - NeuronInternVL3TextForCausalLM.convert_hf_to_neuron_state_dict() for text weights
        """
        # Vision weights (encoder + projector)
        vision_state_dict = convert_vision_hf_to_neuron_state_dict(state_dict)

        # Text weights
        text_state_dict = (
            NeuronInternVL3TextForCausalLM.convert_hf_to_neuron_state_dict(
                state_dict, inference_config.text_config
            )
        )

        # Merge (vision and text keys should not overlap)
        merged = {}
        merged.update(vision_state_dict)
        merged.update(text_state_dict)
        return merged

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied embeddings."""
        NeuronInternVL3TextForCausalLM.update_state_dict_for_tied_weights(state_dict)
