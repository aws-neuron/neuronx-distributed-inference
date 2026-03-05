# coding=utf-8
# Copyright 2023 Baichuan Inc. All rights reserved.
# Adapted for NeuronX Distributed Inference.
"""PyTorch Baichuan2-7B model for NeuronX Distributed Inference.

Architecture: Identical to Llama-2-7b with these differences:
- Fused QKV projection (W_pack) instead of separate q/k/v_proj
- NormHead LM head (weight-normalized linear layer)
- Larger vocabulary (125696 vs 32000)
- rms_norm_eps = 1e-06 (vs 1e-05)

This port extends NXDI's Llama infrastructure and only overrides:
- Config: from_pretrained for Baichuan2 HF config
- Weight conversion: W_pack split + NormHead normalization
- load_hf_model: Direct state dict loading (custom HF code)
"""

import json
import logging
import os
from typing import Type

import torch
import torch.nn.functional as F

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    NeuronLlamaForCausalLM,
    NeuronLlamaModel,
)
from neuronx_distributed_inference.modules.checkpoint import load_state_dict

logger = logging.getLogger("Neuron")


def _load_baichuan2_config(model_path: str):
    """Return a load_config hook that loads Baichuan2 config.json directly.

    Bypasses AutoConfig.from_pretrained which requires trust_remote_code=True
    for Baichuan2's custom code. Adds Llama-required keys missing from Baichuan2 config:
    - num_key_value_heads: Baichuan2 uses MHA (= num_attention_heads)
    - rope_theta: Default 10000.0
    """
    def load_config(self):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if not key.startswith("_"):
                setattr(self, key, value)
        # Baichuan2 uses MHA (not GQA) — set num_key_value_heads = num_attention_heads
        if not hasattr(self, 'num_key_value_heads'):
            self.num_key_value_heads = self.num_attention_heads
        # Default rope_theta
        if not hasattr(self, 'rope_theta'):
            self.rope_theta = 10000.0
        # HF PretrainedConfig defaults not in config.json
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_cache'):
            self.use_cache = True
    return load_config


class Baichuan2InferenceConfig(LlamaInferenceConfig):
    """Configuration for Baichuan2-7B inference on NeuronX.

    Inherits from LlamaInferenceConfig since the architecture is identical.
    The HF config.json uses standard Llama-compatible key names.
    """

    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        if neuron_config is None:
            neuron_config = NeuronConfig(tp_degree=1, batch_size=1, seq_len=128)

        config = cls(
            neuron_config=neuron_config,
            load_config=_load_baichuan2_config(model_path),
            **kwargs,
        )
        return config


class NeuronBaichuan2ForCausalLM(NeuronLlamaForCausalLM):
    """Baichuan2-7B for NeuronX, extending Llama infrastructure.

    Only overrides weight loading/conversion to handle:
    1. W_pack (fused QKV) -> separate q_proj, k_proj, v_proj
    2. NormHead lm_head -> pre-normalized weights
    """

    _model_cls = NeuronLlamaModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load Baichuan2 weights directly from files.

        Bypasses AutoModelForCausalLM which requires trust_remote_code=True.
        Uses NXDI's checkpoint utility for efficient safetensors/bin loading.
        Note: Only called when model_path is a HF hub ID; for local dirs the
        framework uses load_state_dict directly via get_state_dict.
        """
        state_dict = load_state_dict(os.path.expanduser(model_path))

        class _DummyModel:
            def __init__(self, sd):
                self._state_dict = sd
            def state_dict(self):
                return self._state_dict

        return _DummyModel(state_dict)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Convert Baichuan2 weights to Llama-compatible format, then delegate.

        1. Split W_pack [3*H, H] into separate q_proj, k_proj, v_proj [H, H]
        2. Pre-normalize lm_head.weight (NormHead behavior)
        3. Delegate to Llama's weight conversion for rank_util etc.
        """
        # Pre-process: convert Baichuan2-specific keys to Llama-compatible format
        keys_to_delete = []
        keys_to_add = {}

        for key in list(state_dict.keys()):
            if "W_pack" in key:
                # W_pack.weight: [3*hidden_size, hidden_size] -> split into q/k/v_proj
                layer_prefix = key.rsplit("W_pack", 1)[0]  # e.g., "layers.0.self_attn."
                w = state_dict[key]
                q, k, v = w.chunk(3, dim=0)
                keys_to_add[f"{layer_prefix}q_proj.weight"] = q
                keys_to_add[f"{layer_prefix}k_proj.weight"] = k
                keys_to_add[f"{layer_prefix}v_proj.weight"] = v
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del state_dict[key]
        state_dict.update(keys_to_add)

        # Pre-normalize lm_head weights (NormHead: F.normalize along last dim)
        if "lm_head.weight" in state_dict:
            state_dict["lm_head.weight"] = F.normalize(state_dict["lm_head.weight"], dim=-1)

        # Delegate to Llama's conversion (adds rank_util, handles fused_qkv, etc.)
        return NeuronLlamaForCausalLM.convert_hf_to_neuron_state_dict(state_dict, config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Baichuan2 does NOT tie weights (tie_word_embeddings=false)."""
        pass

    @classmethod
    def get_config_cls(cls) -> Type:
        return Baichuan2InferenceConfig
