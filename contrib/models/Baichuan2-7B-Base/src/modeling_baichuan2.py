# coding=utf-8
# Copyright 2023 Baichuan Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
NeuronX implementation of Baichuan2-7B-Base for AWS Trainium.

This implementation leverages the existing NeuronLlama infrastructure
from NeuronxDistributedInference. Baichuan2-7B is architecturally identical
to Llama-2-7b with these differences:

Architecture:
    - Model: Baichuan2-7B-Base (32 layers, 4096 hidden size)
    - Attention: Multi-Head Attention (32 heads, head_dim=128)
    - MLP: SwiGLU activation (gate_proj, up_proj, down_proj)
    - Normalization: RMSNorm (eps=1e-06)
    - Position Encoding: RoPE (theta=10000.0)
    - Vocabulary: 125696 tokens
    - Max Position Embeddings: 4096

Key Differences from Llama-2:
    - Fused QKV projection (W_pack) instead of separate q/k/v_proj
    - NormHead LM head (weight-normalized linear layer)
    - Larger vocabulary (125696 vs 32000)
    - rms_norm_eps = 1e-06 (vs 1e-05)
    - Custom HF code requires trust_remote_code (bypassed by direct loading)
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
    """
    Configuration class for Baichuan2-7B-Base inference on NeuronX.

    Inherits from LlamaInferenceConfig since the architecture is identical.
    Uses a custom config loader to bypass trust_remote_code requirement.
    """

    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        if neuron_config is None:
            neuron_config = NeuronConfig(tp_degree=1, batch_size=1, seq_len=128)
            logger.debug("Created default neuron_config for config loading")

        config = cls(
            neuron_config=neuron_config,
            load_config=_load_baichuan2_config(model_path),
            **kwargs,
        )
        return config


class NeuronBaichuan2ForCausalLM(NeuronLlamaForCausalLM):
    """
    NeuronX implementation of Baichuan2-7B-Base for causal language modeling.

    This class wraps the existing NeuronLlamaForCausalLM implementation
    and overrides weight loading/conversion to handle:
    1. W_pack (fused QKV) -> separate q_proj, k_proj, v_proj
    2. NormHead lm_head -> pre-normalized weights
    3. Direct state dict loading (bypasses trust_remote_code)
    """

    _model_cls = NeuronLlamaModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load Baichuan2 weights directly from files.

        Bypasses AutoModelForCausalLM which requires trust_remote_code=True.
        Uses NXDI's checkpoint utility for efficient safetensors/bin loading.
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
                layer_prefix = key.rsplit("W_pack", 1)[0]
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

        # Delegate to Llama's conversion
        return NeuronLlamaForCausalLM.convert_hf_to_neuron_state_dict(state_dict, config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Baichuan2 does NOT tie weights (tie_word_embeddings=false)."""
        pass

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for Baichuan2"""
        return Baichuan2InferenceConfig


# Export classes
__all__ = [
    "Baichuan2InferenceConfig",
    "NeuronBaichuan2ForCausalLM",
]
