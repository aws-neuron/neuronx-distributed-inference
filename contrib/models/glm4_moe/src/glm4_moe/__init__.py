# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""GLM-4.5 MoE contrib model package."""

from glm4_moe.modeling_glm4_moe import (
    NeuronGlm4MoeForCausalLM,
    Glm4MoeInferenceConfig,
    convert_glm4_moe_hf_to_neuron_state_dict,
)

__all__ = [
    "NeuronGlm4MoeForCausalLM",
    "Glm4MoeInferenceConfig",
    "convert_glm4_moe_hf_to_neuron_state_dict",
]
