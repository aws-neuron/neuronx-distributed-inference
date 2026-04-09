# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2-Flash model for NXD inference - Contrib wrapper."""

from typing import List

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.mimo_v2.modeling_mimo_v2 import (
    NeuronMiMoV2ForCausalLM as BaseNeuronMiMoV2ForCausalLM,
    MiMoV2InferenceConfig as BaseMiMoV2InferenceConfig,
    convert_mimo_v2_hf_to_neuron_state_dict,
)


class MiMoV2InferenceConfig(BaseMiMoV2InferenceConfig):
    """Configuration class for MiMo-V2-Flash inference on NeuronX."""
    pass


class NeuronMiMoV2ForCausalLM(BaseNeuronMiMoV2ForCausalLM):
    """MiMo-V2-Flash Causal Language Model for NeuronX inference.

    Architecture:
    - 48 decoder layers with Mixture of 256 Experts (top-8)
    - Hybrid attention: full (4 KV heads) + sliding window (8 KV heads)
    - Asymmetric head dims: Q/K=192, V=128
    - Partial RoPE (34%), attention sink bias
    - Sigmoid router
    """

    @classmethod
    def get_config_cls(cls):
        return MiMoV2InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        return convert_mimo_v2_hf_to_neuron_state_dict(state_dict, config)
