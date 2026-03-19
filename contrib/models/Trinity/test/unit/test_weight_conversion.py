#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
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
CPU-only unit tests for Trinity HF→Neuron weight conversion.

Tests verify key weight name mappings, transformations (muP scaling,
route_scale baking, expert stacking, gate padding), and edge cases
without requiring Neuron hardware or model weights.

Usage:
    pytest test/unit/test_weight_conversion.py -v
"""

import math
import pytest
import torch

from neuronx_distributed_inference.models.config import MoENeuronConfig

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_trinity import NeuronTrinityForCausalLM, TrinityInferenceConfig


# Minimal Nano config for weight conversion tests
NANO_CONFIG = {
    "vocab_size": 200192,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "moe_intermediate_size": 256,
    "num_hidden_layers": 56,
    "num_dense_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_shared_experts": 1,
    "score_func": "sigmoid",
    "route_norm": True,
    "route_scale": 2.826,
    "sliding_window": 2048,
    "mup_enabled": True,
    "global_attn_every_n_layers": 4,
    "tie_word_embeddings": False,
    "n_group": 1,
    "topk_group": 1,
}


def make_config(tp_degree=2):
    """Create a TrinityInferenceConfig for testing."""
    neuron_config = MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        seq_len=2048,
        torch_dtype=torch.bfloat16,
    )
    return TrinityInferenceConfig(neuron_config=neuron_config, **NANO_CONFIG.copy())


def make_minimal_state_dict(config, num_layers=2, num_experts=4):
    """Create a minimal HF state dict for testing weight conversion.

    Only includes enough layers and experts to validate transformations.
    Uses small tensor sizes for fast CPU testing.
    """
    H = config.hidden_size  # 1024
    I_dense = config.dense_intermediate_size  # 3072
    I_moe = config.moe_intermediate_size  # 256
    V = config.vocab_size  # 200192
    n_heads = config.num_attention_heads  # 8
    n_kv_heads = config.num_key_value_heads  # 2
    head_dim = config.head_dim  # 128

    sd = {}

    # Embedding
    sd["model.embed_tokens.weight"] = torch.randn(V, H)

    # LM head
    sd["model.norm.weight"] = torch.ones(H)
    sd["lm_head.weight"] = torch.randn(V, H)

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"

        # Attention weights
        sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, H)
        sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, H)
        sd[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, H)
        sd[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(H, n_heads * head_dim)

        # QK norms (Trinity-specific)
        sd[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(head_dim)
        sd[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(head_dim)

        # Attention gate (gated attention)
        sd[f"{prefix}.self_attn.gate_proj.weight"] = torch.randn(n_heads, H)

        # Layer norms
        sd[f"{prefix}.input_layernorm.weight"] = torch.ones(H)
        sd[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(H)
        sd[f"{prefix}.pre_feedforward_layernorm.weight"] = torch.ones(H)
        sd[f"{prefix}.post_feedforward_layernorm.weight"] = torch.ones(H)

        if layer_idx < config.num_dense_layers:
            # Dense MLP layers
            sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(I_dense, H)
            sd[f"{prefix}.mlp.up_proj.weight"] = torch.randn(I_dense, H)
            sd[f"{prefix}.mlp.down_proj.weight"] = torch.randn(H, I_dense)
        else:
            # MoE layers
            sd[f"{prefix}.mlp.router.gate.weight"] = torch.randn(num_experts, H)
            sd[f"{prefix}.mlp.expert_bias"] = torch.randn(num_experts)

            for e in range(num_experts):
                sd[f"{prefix}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(I_moe, H)
                sd[f"{prefix}.mlp.experts.{e}.up_proj.weight"] = torch.randn(I_moe, H)
                sd[f"{prefix}.mlp.experts.{e}.down_proj.weight"] = torch.randn(H, I_moe)

            # Shared expert (no index -- Trinity uses shared_experts.{proj} directly)
            sd[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = torch.randn(I_moe, H)
            sd[f"{prefix}.mlp.shared_experts.up_proj.weight"] = torch.randn(I_moe, H)
            sd[f"{prefix}.mlp.shared_experts.down_proj.weight"] = torch.randn(H, I_moe)

    return sd


class TestModelPrefixRemoval:
    """Test that 'model.' prefix is correctly removed from HF keys."""

    def test_model_prefix_removed(self):
        config = make_config()
        sd = {"model.embed_tokens.weight": torch.randn(200192, 1024)}
        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)
        assert "embed_tokens.weight" in result
        assert "model.embed_tokens.weight" not in result


class TestMuPScaling:
    """Test muP embedding scaling: embed_weight *= sqrt(hidden_size)."""

    def test_mup_scaling_applied(self):
        config = make_config()
        original_embed = torch.randn(200192, 1024)
        sd = {"model.embed_tokens.weight": original_embed.clone()}
        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)

        # Conversion outputs bf16, so cast expected to match
        expected = (original_embed * math.sqrt(1024)).to(torch.bfloat16)
        assert torch.allclose(result["embed_tokens.weight"], expected, atol=1e-2), (
            "muP scaling should multiply embedding by sqrt(hidden_size)"
        )


class TestQKNormRename:
    """Test q_norm → q_layernorm, k_norm → k_layernorm rename."""

    def test_qk_norm_renamed(self):
        config = make_config()
        sd = {
            "model.layers.0.self_attn.q_norm.weight": torch.ones(128),
            "model.layers.0.self_attn.k_norm.weight": torch.ones(128),
        }
        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)

        assert "layers.0.self_attn.q_layernorm.weight" in result
        assert "layers.0.self_attn.k_layernorm.weight" in result
        assert "layers.0.self_attn.q_norm.weight" not in result
        assert "layers.0.self_attn.k_norm.weight" not in result


class TestAttentionGateRename:
    """Test self_attn.gate_proj → self_attn.attn_gate_proj rename."""

    def test_gate_renamed(self):
        config = make_config()
        sd = {
            "model.layers.0.self_attn.gate_proj.weight": torch.randn(8, 1024),
        }
        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)

        assert "layers.0.self_attn.attn_gate_proj.weight" in result
        assert "layers.0.self_attn.gate_proj.weight" not in result


class TestRouterRename:
    """Test router.gate.weight → router.linear_router.weight rename."""

    def test_router_weight_renamed(self):
        config = make_config()
        sd = {
            "model.layers.2.mlp.router.gate.weight": torch.randn(128, 1024),
        }
        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)

        assert "layers.2.mlp.router.linear_router.weight" in result
        assert "layers.2.mlp.router.gate.weight" not in result


class TestExpertBiasMapping:
    """Test expert_bias mapping: mlp.expert_bias → mlp.router.expert_bias."""

    def test_expert_bias_mapped(self):
        config = make_config()
        bias = torch.randn(128)
        sd = {"model.layers.2.mlp.expert_bias": bias.clone()}
        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)

        assert "layers.2.mlp.router.expert_bias" in result
        assert torch.equal(result["layers.2.mlp.router.expert_bias"], bias)

    def test_expert_bias_kept_float32(self):
        """Expert bias should remain float32 (not converted to bf16)."""
        config = make_config()
        bias = torch.randn(128, dtype=torch.float32)
        sd = {"model.layers.2.mlp.expert_bias": bias}
        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)

        assert result["layers.2.mlp.router.expert_bias"].dtype == torch.float32


class TestExpertWeightStacking:
    """Test per-expert weights are stacked into [E, H, 2*I] format."""

    def test_expert_stacking(self):
        config = make_config()
        H = 1024
        I = 256
        # Must provide all num_experts from config (128) since the
        # conversion iterates over all experts and aborts if any are missing.
        num_experts = config.num_local_experts

        sd = {}
        for e in range(num_experts):
            sd[f"model.layers.2.mlp.experts.{e}.gate_proj.weight"] = torch.randn(I, H)
            sd[f"model.layers.2.mlp.experts.{e}.up_proj.weight"] = torch.randn(I, H)
            sd[f"model.layers.2.mlp.experts.{e}.down_proj.weight"] = torch.randn(H, I)

        # Need router key too for the conversion to proceed
        sd["model.layers.2.mlp.router.gate.weight"] = torch.randn(num_experts, H)

        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)

        # gate_up_proj should be [E, H, 2*I] (key includes .mlp_op. and .weight)
        gate_up_key = "layers.2.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
        assert gate_up_key in result, (
            f"Expected key '{gate_up_key}' in result. Keys: "
            f"{[k for k in result if 'expert' in k]}"
        )
        assert result[gate_up_key].shape == (num_experts, H, 2 * I)

        # down_proj should be [E, I, H]
        down_key = "layers.2.mlp.expert_mlps.mlp_op.down_proj.weight"
        assert down_key in result
        assert result[down_key].shape == (num_experts, I, H)


class TestRouteScaleBaking:
    """Test that route_scale is baked into routed expert down_proj weights."""

    def test_route_scale_applied_to_down_proj(self):
        config = make_config()
        H = 1024
        I = 256
        # Must provide all num_experts from config
        num_experts = config.num_local_experts
        route_scale = config.route_scale  # 2.826

        sd = {}
        down_projs = []
        for e in range(num_experts):
            sd[f"model.layers.2.mlp.experts.{e}.gate_proj.weight"] = torch.randn(I, H)
            sd[f"model.layers.2.mlp.experts.{e}.up_proj.weight"] = torch.randn(I, H)
            down = torch.randn(H, I)
            sd[f"model.layers.2.mlp.experts.{e}.down_proj.weight"] = down.clone()
            down_projs.append(down)

        sd["model.layers.2.mlp.router.gate.weight"] = torch.randn(num_experts, H)

        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)

        # The down_proj weights should be scaled by route_scale
        result_down = result["layers.2.mlp.expert_mlps.mlp_op.down_proj.weight"]
        for e in range(min(4, num_experts)):
            # Original down_proj is transposed to [I, H], cast to bf16, then scaled
            expected = (down_projs[e].T.to(torch.bfloat16)) * route_scale
            assert torch.allclose(result_down[e], expected, atol=1e-2), (
                f"Expert {e} down_proj should be scaled by route_scale={route_scale}"
            )


class TestSharedExpertMapping:
    """Test shared expert weight key mapping."""

    def test_shared_expert_keys(self):
        config = make_config()
        H = 1024
        I = 256

        # HF Trinity uses shared_experts.{proj} (no index) for single shared expert
        sd = {
            "model.layers.2.mlp.shared_experts.gate_proj.weight": torch.randn(I, H),
            "model.layers.2.mlp.shared_experts.up_proj.weight": torch.randn(I, H),
            "model.layers.2.mlp.shared_experts.down_proj.weight": torch.randn(H, I),
        }

        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)

        # Should be mapped to standalone shared_expert module
        assert (
            "layers.2.shared_expert.gate_proj.weight" in result
            or "layers.2.mlp.shared_expert.gate_proj.weight" in result
        ), (
            f"Shared expert keys not found. Keys with 'shared': "
            f"{[k for k in result if 'shared' in k]}"
        )


class TestGatePadding:
    """Test gate weight padding when num_heads % TP != 0."""

    def test_no_padding_when_divisible(self):
        """Nano: 8 heads / TP=2 = 4 per rank, no padding needed."""
        config = make_config(tp_degree=2)
        n_heads = 8
        H = 1024
        gate = torch.randn(n_heads, H)
        sd = {"model.layers.0.self_attn.gate_proj.weight": gate.clone()}
        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)
        result_gate = result["layers.0.self_attn.attn_gate_proj.weight"]
        # No padding: shape should be (8, H)
        assert result_gate.shape[0] == n_heads

    def test_padding_when_not_divisible(self):
        """Large: 48 heads / TP=64 requires padding to 64."""
        # Create a Large-like config
        large_config_dict = NANO_CONFIG.copy()
        large_config_dict.update(
            {
                "hidden_size": 3072,
                "intermediate_size": 12288,
                "moe_intermediate_size": 3072,
                "num_attention_heads": 48,
                "num_key_value_heads": 8,
                "num_hidden_layers": 60,
                "num_dense_layers": 6,
                "num_experts": 256,
                "num_experts_per_tok": 4,
                "sliding_window": 4096,
            }
        )
        neuron_config = MoENeuronConfig(
            tp_degree=64, batch_size=1, seq_len=4096, torch_dtype=torch.bfloat16
        )
        config = TrinityInferenceConfig(
            neuron_config=neuron_config, **large_config_dict
        )

        n_heads = 48
        H = 3072
        gate = torch.randn(n_heads, H)
        sd = {"model.layers.0.self_attn.gate_proj.weight": gate.clone()}
        result = NeuronTrinityForCausalLM.convert_hf_to_neuron_state_dict(sd, config)
        result_gate = result["layers.0.self_attn.attn_gate_proj.weight"]
        # Gate weight is (num_heads, hidden_size). After interleaved padding,
        # the output should be (padded_total_heads, hidden_size).
        padded_heads = 64  # next multiple of TP=64 >= 48
        assert result_gate.shape == (padded_heads, H), (
            f"Expected shape ({padded_heads}, {H}), got {result_gate.shape}"
        )
