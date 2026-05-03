# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax M2 model for NXD inference - Contrib wrapper."""

from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import (
    NeuronMiniMaxM2ForCausalLM,
    MiniMaxM2InferenceConfig,
    convert_minimax_m2_hf_to_neuron_state_dict,
)

__all__ = [
    "MiniMaxM2InferenceConfig",
    "NeuronMiniMaxM2ForCausalLM",
]
