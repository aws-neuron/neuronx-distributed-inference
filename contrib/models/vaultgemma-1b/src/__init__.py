# coding=utf-8
# Copyright 2025 AWS. All rights reserved.
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
VaultGemma model implementation for NeuronX Distributed Inference.

This module provides the VaultGemma model ported to run on AWS Trainium hardware
using the NeuronX Distributed framework.

Classes:
    VaultGemmaInferenceConfig: Configuration class for VaultGemma inference
    NeuronVaultGemmaForCausalLM: Main model class with language modeling head
    NeuronVaultGemmaModel: Base transformer model
    NeuronVaultGemmaAttention: Attention layer implementation
    NeuronVaultGemmaMLP: MLP layer implementation
    NeuronVaultGemmaDecoderLayer: Decoder layer combining attention and MLP
"""

from .modeling_vaultgemma import (
    VaultGemmaInferenceConfig,
    VaultGemmaNeuronConfig,
    NeuronVaultGemmaForCausalLM,
    NeuronVaultGemmaModel,
    NeuronVaultGemmaAttention,
    NeuronVaultGemmaMLP,
    NeuronVaultGemmaDecoderLayer,
    get_rmsnorm_cls,
)

__all__ = [
    "VaultGemmaInferenceConfig",
    "VaultGemmaNeuronConfig",
    "NeuronVaultGemmaForCausalLM",
    "NeuronVaultGemmaModel",
    "NeuronVaultGemmaAttention",
    "NeuronVaultGemmaMLP",
    "NeuronVaultGemmaDecoderLayer",
    "get_rmsnorm_cls",
]
