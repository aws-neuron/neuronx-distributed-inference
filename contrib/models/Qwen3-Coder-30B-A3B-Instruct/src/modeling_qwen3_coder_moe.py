# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3-Coder-30B-A3B-Instruct model for NXD inference - Custom Port

Wraps the framework's Qwen3MoE implementation with a config class that
supports from_pretrained loading from HuggingFace model directories.

Architecture: Qwen3MoeForCausalLM (standard qwen3_moe)
  - 48 decoder layers, GQA with QK-norm (RMSNorm on head_dim)
  - MoE FFN: softmax top-k routing, norm_topk_prob, no shared experts
  - RoPE positional embeddings, SwiGLU activation

Reference: neuronx_distributed_inference/models/qwen3_moe/modeling_qwen3_moe.py
"""
import json
import os
from typing import List, Type

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import (
    NeuronQwen3MoeForCausalLM as BaseNeuronQwen3MoeForCausalLM,
    convert_qwen3_moe_hf_to_neuron_state_dict,
)


class Qwen3CoderMoeInferenceConfig(InferenceConfig):
    """
    Configuration for Qwen3-Coder-30B-A3B-Instruct inference on NeuronX.

    Extends InferenceConfig with Qwen3MoE-specific parameters and
    from_pretrained support for loading from HuggingFace model directories.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Map HF num_experts -> num_local_experts expected by initialize_moe_module
        self.num_local_experts = self.num_experts
        # No shared experts in Qwen3MoE
        self.n_shared_experts = 0
        # ExpertMLPsV2 reads intermediate_size from config
        self.intermediate_size = self.moe_intermediate_size
        # Router: softmax in float32 for accuracy
        self.neuron_config.router_config.dtype = __import__("torch").float32
        self.neuron_config.router_config.act_fn = "softmax"

    def add_derived_config(self):
        self.num_cores_per_group = 1
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "norm_topk_prob",
            "num_attention_heads",
            "num_experts",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MoENeuronConfig]:
        return MoENeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load configuration from a pretrained model directory."""
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found at {config_path}")

        with open(config_path) as f:
            hf = json.load(f)

        config_dict = {
            "head_dim": hf.get("head_dim", hf.get("hidden_size", 2048) // hf.get("num_attention_heads", 32)),
            "hidden_act": hf.get("hidden_act", "silu"),
            "hidden_size": hf.get("hidden_size", 2048),
            "intermediate_size": hf.get("intermediate_size", 6144),
            "max_position_embeddings": hf.get("max_position_embeddings", 32768),
            "moe_intermediate_size": hf.get("moe_intermediate_size", 768),
            "norm_topk_prob": hf.get("norm_topk_prob", False),
            "num_attention_heads": hf.get("num_attention_heads", 32),
            "num_experts": hf.get("num_experts", 128),
            "num_experts_per_tok": hf.get("num_experts_per_tok", 8),
            "num_hidden_layers": hf.get("num_hidden_layers", 48),
            "num_key_value_heads": hf.get("num_key_value_heads", 4),
            "rms_norm_eps": hf.get("rms_norm_eps", 1e-6),
            "rope_theta": hf.get("rope_theta", 10000.0),
            "rope_scaling": hf.get("rope_scaling"),
            "vocab_size": hf.get("vocab_size", 151936),
            "tie_word_embeddings": hf.get("tie_word_embeddings", False),
            "pad_token_id": hf.get("pad_token_id") or hf.get("eos_token_id", 0),
            "bos_token_id": hf.get("bos_token_id"),
            "eos_token_id": hf.get("eos_token_id"),
            "decoder_sparse_step": hf.get("decoder_sparse_step", 1),
            "mlp_only_layers": hf.get("mlp_only_layers", []),
            "attention_bias": hf.get("attention_bias", False),
            "attention_dropout": hf.get("attention_dropout", 0.0),
            "sliding_window": hf.get("sliding_window"),
            "output_attentions": False,
            "output_hidden_states": False,
        }

        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronQwen3CoderMoeForCausalLM(BaseNeuronQwen3MoeForCausalLM):
    """
    Qwen3-Coder-30B-A3B-Instruct for NeuronX inference.

    Extends the framework's NeuronQwen3MoeForCausalLM with our custom config
    that includes from_pretrained support.
    """

    @classmethod
    def get_config_cls(cls):
        return Qwen3CoderMoeInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        return convert_qwen3_moe_hf_to_neuron_state_dict(state_dict, config)
