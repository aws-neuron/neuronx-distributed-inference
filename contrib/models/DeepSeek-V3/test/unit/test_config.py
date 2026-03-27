# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DeepSeek V3 inference configuration.

CPU-only tests that validate config parsing, MLA parameter setup,
MoE routing config, dense layer detection, and FP8 dequantization logic.
"""

import unittest
from unittest.mock import MagicMock

import torch

from src.modeling_deepseek import (
    DeepseekV3InferenceConfig,
    DeepseekV3NeuronConfig,
    _dequantize_fp8_state_dict,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig


def _make_config(**overrides):
    """Create a DeepseekV3InferenceConfig with reasonable defaults."""
    neuron_config = MoENeuronConfig(
        tp_degree=overrides.pop("tp_degree", 2),
        batch_size=1,
        seq_len=128,
        torch_dtype=torch.bfloat16,
    )
    defaults = dict(
        hidden_size=7168,
        num_hidden_layers=61,
        num_attention_heads=128,
        num_key_value_heads=1,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        first_k_dense_replace=3,
        vocab_size=129280,
        rms_norm_eps=1e-6,
        max_position_embeddings=163840,
        rope_theta=10000,
    )
    defaults.update(overrides)
    config = DeepseekV3InferenceConfig(neuron_config=neuron_config, **defaults)
    return config


class TestConfigParsing(unittest.TestCase):
    """Test basic config attribute initialization."""

    def test_mla_parameters(self):
        config = _make_config()
        self.assertEqual(config.kv_lora_rank, 512)
        self.assertEqual(config.q_lora_rank, 1536)
        self.assertEqual(config.qk_nope_head_dim, 128)
        self.assertEqual(config.qk_rope_head_dim, 64)
        self.assertEqual(config.v_head_dim, 128)

    def test_head_dim_override_for_kv_cache(self):
        """MLA overrides head_dim to rope_dim + kv_lora_rank for KV cache allocation."""
        config = _make_config()
        self.assertEqual(config.head_dim, 64 + 512)  # rope_dim + kv_lora_rank = 576

    def test_num_kv_heads_override(self):
        """MLA sets num_key_value_heads=1 (MLA uses a single compressed KV, not GQA)."""
        config = _make_config()
        self.assertEqual(config.num_key_value_heads, 1)

    def test_moe_expert_params(self):
        config = _make_config()
        self.assertEqual(config.num_local_experts, 256)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.num_experts_per_tok, 8)

    def test_intermediate_size_swap(self):
        """intermediate_size should be swapped to moe_intermediate_size for MoE experts."""
        config = _make_config(intermediate_size=18432, moe_intermediate_size=2048)
        self.assertEqual(config.intermediate_size, 2048)
        self.assertEqual(config.dense_intermediate_size, 18432)

    def test_dense_layer_count(self):
        config = _make_config(first_k_dense_replace=3)
        self.assertEqual(config.first_k_dense_replace, 3)

    def test_hidden_act_default(self):
        config = _make_config()
        self.assertEqual(config.hidden_act, "silu")


class TestRoPEConfig(unittest.TestCase):
    """Test RoPE configuration handling."""

    def test_noop_yarn_injected_when_no_rope_scaling(self):
        """When HF config has no rope_scaling, a no-op YaRN config is injected."""
        config = _make_config()
        self.assertIsNotNone(config.rope_scaling)
        self.assertEqual(config.rope_scaling["type"], "yarn")
        self.assertEqual(config.rope_scaling["factor"], 1.0)

    def test_explicit_rope_scaling_preserved(self):
        """When HF config specifies rope_scaling, it should be preserved."""
        custom_scaling = {
            "type": "yarn",
            "factor": 40.0,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 4096,
        }
        config = _make_config(rope_scaling=custom_scaling)
        self.assertEqual(config.rope_scaling["factor"], 40.0)
        self.assertEqual(config.rope_scaling["mscale"], 0.707)


class TestNeuronConfig(unittest.TestCase):
    """Test Neuron-specific configuration settings."""

    def test_disable_numeric_cc_token(self):
        config = _make_config()
        self.assertTrue(config.neuron_config.disable_numeric_cc_token)

    def test_neuron_config_cls(self):
        self.assertEqual(
            DeepseekV3InferenceConfig.get_neuron_config_cls(),
            DeepseekV3NeuronConfig,
        )

    def test_required_attributes(self):
        config = _make_config()
        required = config.get_required_attributes()
        self.assertIn("kv_lora_rank", required)
        self.assertIn("n_routed_experts", required)
        self.assertIn("moe_intermediate_size", required)


class TestFP8Dequantization(unittest.TestCase):
    """Test FP8 weight dequantization logic."""

    def test_noop_on_bf16_weights(self):
        """No-op when weights are already BF16 (no scale_inv keys)."""
        state_dict = {
            "model.layers.0.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        }
        result = _dequantize_fp8_state_dict(state_dict, block_size=128)
        self.assertIn("model.layers.0.weight", result)
        self.assertEqual(result["model.layers.0.weight"].dtype, torch.bfloat16)

    def test_scale_inv_keys_removed(self):
        """All weight_scale_inv keys should be removed after dequantization."""
        state_dict = {
            "model.layers.0.weight": torch.randn(128, 128).to(torch.float8_e4m3fn),
            "model.layers.0.weight_scale_inv": torch.ones(1, 1),
        }
        result = _dequantize_fp8_state_dict(state_dict, block_size=128)
        self.assertNotIn("model.layers.0.weight_scale_inv", result)
        self.assertEqual(result["model.layers.0.weight"].dtype, torch.bfloat16)

    def test_dequantized_values_correct(self):
        """Dequantized weight = FP8 weight * scale_inv, converted to BF16."""
        block_size = 4
        M, N = 4, 4
        weight_f32 = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                    [5.0, 6.0, 7.0, 8.0],
                                    [1.0, 1.0, 1.0, 1.0],
                                    [2.0, 2.0, 2.0, 2.0]], dtype=torch.float32)
        weight_fp8 = weight_f32.to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([[2.0]], dtype=torch.float32)

        state_dict = {
            "w.weight": weight_fp8,
            "w.weight_scale_inv": scale_inv,
        }
        result = _dequantize_fp8_state_dict(state_dict, block_size=block_size)

        expected = (weight_fp8.to(torch.float32) * 2.0).to(torch.bfloat16)
        torch.testing.assert_close(result["w.weight"], expected)


if __name__ == "__main__":
    unittest.main()
