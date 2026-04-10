# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni text-only (Thinker) support for NXD inference.
# The Thinker's text backbone is architecturally identical to Qwen2.5,
# so we reuse the Qwen2 NxDI implementation with state-dict remapping.
#
# Reference: https://huggingface.co/Qwen/Qwen2.5-Omni-7B

"""Qwen2.5-Omni model (Thinker text component) for NXD inference."""

import gc
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Type

import torch

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import (
    NeuronQwen2ForCausalLM,
    NeuronQwen2Model,
    convert_state_dict_to_fused_qkv,
)

logger = logging.getLogger("Neuron")


# Attributes to extract from the thinker's text_config to the top-level config.
_TEXT_CONFIG_ATTRS = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "intermediate_size",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "hidden_act",
    "tie_word_embeddings",
    "max_window_layers",
    "use_sliding_window",
    "sliding_window",
]


class Qwen25OmniInferenceConfig(InferenceConfig):
    """Inference config for Qwen2.5-Omni (Thinker text component).

    Handles the nested config structure: the HF config has attributes under
    thinker_config.text_config that we need at the top level for NxDI.
    """

    def add_derived_config(self):
        self.num_cores_per_group = 1
        # Qwen2.5-Omni text model has QKV bias but no output projection bias
        self.qkv_bias = True
        self.o_bias = False

        # Extract text config attributes from nested thinker_config
        if hasattr(self, "thinker_config"):
            text_cfg = self.thinker_config.text_config
            if isinstance(text_cfg, dict):
                text_cfg = SimpleNamespace(**text_cfg)
                self.thinker_config.text_config = text_cfg

            for attr in _TEXT_CONFIG_ATTRS:
                if hasattr(text_cfg, attr) and not hasattr(self, attr):
                    setattr(self, attr, getattr(text_cfg, attr))

            # Set pad_token_id from thinker_config
            if hasattr(self.thinker_config, "pad_token_id"):
                if not hasattr(self, "pad_token_id") or self.pad_token_id is None:
                    self.pad_token_id = self.thinker_config.pad_token_id

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronQwen25OmniForCausalLM(NeuronQwen2ForCausalLM):
    """Qwen2.5-Omni Thinker text model for Causal LM on Neuron.

    Reuses the Qwen2 model architecture since the Thinker's text backbone
    is architecturally identical to Qwen2.5. The main differences are:
      - Weight keys are prefixed with 'thinker.model.' / 'thinker.lm_head.'
      - Non-text weights (talker, token2wav, audio_tower, visual) are discarded
    """

    _model_cls = NeuronQwen2Model
    _STATE_DICT_MODEL_PREFIX = "thinker.model."

    @staticmethod
    def load_hf_model(model_path: str, **kwargs):
        """Load the full Qwen2.5-Omni model from HuggingFace.

        Note: We load the full model and filter to thinker text weights
        in convert_hf_to_neuron_state_dict.
        """
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls) -> Type[Qwen25OmniInferenceConfig]:
        return Qwen25OmniInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, Any],
        config: Qwen25OmniInferenceConfig,
    ) -> Dict[str, Any]:
        """Convert Qwen2.5-Omni state dict to NxDI format.

        1. Keep only thinker text model weights (discard talker, token2wav, audio, visual)
        2. Map 'thinker.lm_head.' -> 'lm_head.'
        3. Apply standard Qwen2 conversions (fused QKV, rank utils, etc.)
        """
        neuron_config = config.neuron_config

        # Filter: keep only thinker text weights and lm_head
        # After base-class prefix stripping, 'thinker.model.*' becomes '*'
        # but 'thinker.lm_head.*' is NOT stripped, so handle it here.
        keys_to_remove = []
        keys_to_rename = {}
        for key in state_dict:
            if key.startswith("thinker.lm_head."):
                # Map thinker.lm_head.weight -> lm_head.weight
                new_key = key.replace("thinker.lm_head.", "lm_head.", 1)
                keys_to_rename[key] = new_key
            elif not key.startswith(("layers.", "embed_tokens.", "norm.", "lm_head.")):
                # After base-class prefix stripping, valid text keys start with
                # layers.*, embed_tokens.*, norm.*, or lm_head.*
                # Everything else (talker, token2wav, audio_tower, visual) should be removed
                keys_to_remove.append(key)

        # Apply renames
        for old_key, new_key in keys_to_rename.items():
            state_dict[new_key] = state_dict.pop(old_key)

        # Remove non-text weights
        for key in keys_to_remove:
            del state_dict[key]

        gc.collect()
        logger.info(
            "Filtered state dict to %d thinker text model weights", len(state_dict)
        )

        # Add rank utilities (same as Qwen2)
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    def get_compiler_args(self):
        compiler_args = (
            "--enable-saturate-infinity "
            "--enable-mixed-precision-accumulation "
            "--auto-cast=none "
            "--model-type transformer "
            "-O1"
        )
        compiler_args += (
            " --tensorizer-options='"
            "--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 "
            "--vectorize-strided-dma'"
        )
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args
