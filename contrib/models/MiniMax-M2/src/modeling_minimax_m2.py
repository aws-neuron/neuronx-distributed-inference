# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax M2 model for NXD inference - Contrib wrapper."""

from typing import List

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import (
    NeuronMiniMaxM2ForCausalLMV3 as BaseNeuronMiniMaxM2ForCausalLM,
    MiniMaxM2InferenceConfigV3 as BaseMiniMaxM2InferenceConfig,
    convert_minimax_m2_hf_to_neuron_state_dict,
)


class MiniMaxM2InferenceConfig(BaseMiniMaxM2InferenceConfig):
    """Configuration class for MiniMax M2 inference on NeuronX."""
    pass


class NeuronMiniMaxM2ForCausalLM(BaseNeuronMiniMaxM2ForCausalLM):
    """MiniMax M2 Causal Language Model for NeuronX inference.

    Architecture:
    - 62 decoder layers with Mixture of 256 Experts (top-8)
    - GQA: 48 Q heads, 8 KV heads, head_dim=128
    - Partial RoPE (50% of head_dim)
    - QK norm, sigmoid router with e_score_correction_bias
    - fused_qkv support
    """

    @classmethod
    def get_config_cls(cls):
        return MiniMaxM2InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        return convert_minimax_m2_hf_to_neuron_state_dict(state_dict, config)
