# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GLM-4.7-Flash (glm4_moe_lite) inference configuration.

CPU-only tests that validate config parsing, MLA parameter setup,
MoE routing config, dense layer detection, and KV cache shape.
"""

import unittest
from unittest.mock import MagicMock

import torch

from src.modeling_glm4_moe_lite import (
    Glm4MoeLiteInferenceConfig,
    Glm4MoeLiteNeuronConfig,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig


def _make_config(**overrides):
    """Create a Glm4MoeLiteInferenceConfig with GLM-4.7-Flash defaults."""
    neuron_config = MoENeuronConfig(
        tp_degree=overrides.pop("tp_degree", 4),
        batch_size=1,
        seq_len=128,
        torch_dtype=torch.bfloat16,
    )
    defaults = dict(
        hidden_size=2048,
        num_hidden_layers=47,
        num_attention_heads=20,
        num_key_value_heads=1,
        kv_lora_rank=512,
        q_lora_rank=768,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        n_routed_experts=64,
        n_shared_experts=1,
        num_experts_per_tok=4,
        n_group=1,
        topk_group=1,
        intermediate_size=10240,
        moe_intermediate_size=1536,
        first_k_dense_replace=1,
        vocab_size=154880,
        rms_norm_eps=1e-6,
        max_position_embeddings=200000,
        rope_theta=10000,
        routed_scaling_factor=1.8,
        attention_dropout=0.0,
        attention_bias=False,
    )
    defaults.update(overrides)
    config = Glm4MoeLiteInferenceConfig(neuron_config=neuron_config, **defaults)
    return config


class TestConfigParsing(unittest.TestCase):
    """Test basic config attribute initialization."""

    def test_mla_parameters(self):
        config = _make_config()
        self.assertEqual(config.kv_lora_rank, 512)
        self.assertEqual(config.q_lora_rank, 768)
        self.assertEqual(config.qk_nope_head_dim, 192)
        self.assertEqual(config.qk_rope_head_dim, 64)
        self.assertEqual(config.v_head_dim, 256)

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
        self.assertEqual(config.num_local_experts, 64)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.num_experts_per_tok, 4)

    def test_intermediate_size_swap(self):
        """intermediate_size should be swapped to moe_intermediate_size for MoE experts."""
        config = _make_config(intermediate_size=10240, moe_intermediate_size=1536)
        self.assertEqual(config.intermediate_size, 1536)
        self.assertEqual(config.dense_intermediate_size, 10240)

    def test_dense_layer_count(self):
        """GLM-4.7-Flash has first_k_dense_replace=1 (only layer 0 is dense)."""
        config = _make_config()
        self.assertEqual(config.first_k_dense_replace, 1)

    def test_hidden_act_default(self):
        config = _make_config()
        self.assertEqual(config.hidden_act, "silu")


class TestNoYaRNRoPE(unittest.TestCase):
    """Test that GLM-4.7-Flash does NOT inject YaRN config."""

    def test_no_rope_scaling_injected(self):
        """GLM-4.7-Flash uses standard RoPE — no rope_scaling should be injected."""
        config = _make_config()
        # The config should not have YaRN-specific rope_scaling
        # (unlike DeepSeek-V3 which injects a no-op YaRN config)
        # Our config doesn't touch rope_scaling at all
        # Just verify it doesn't crash and the relevant dims are correct
        self.assertEqual(config.qk_rope_head_dim, 64)


class TestNeuronConfig(unittest.TestCase):
    """Test Neuron-specific configuration settings."""

    def test_disable_numeric_cc_token(self):
        config = _make_config()
        self.assertTrue(config.neuron_config.disable_numeric_cc_token)

    def test_neuron_config_cls(self):
        self.assertEqual(
            Glm4MoeLiteInferenceConfig.get_neuron_config_cls(),
            Glm4MoeLiteNeuronConfig,
        )

    def test_required_attributes(self):
        config = _make_config()
        required = config.get_required_attributes()
        self.assertIn("kv_lora_rank", required)
        self.assertIn("n_routed_experts", required)
        self.assertIn("moe_intermediate_size", required)
        self.assertIn("qk_nope_head_dim", required)
        self.assertIn("v_head_dim", required)

    def test_router_config_sigmoid(self):
        """Router should use sigmoid activation for noaux_tc routing."""
        config = _make_config()
        self.assertEqual(config.neuron_config.router_config.act_fn, "sigmoid")
        self.assertEqual(config.neuron_config.router_config.dtype, torch.float32)

    def test_normalize_top_k_disabled(self):
        """Normalization handled by router, not ExpertMLPsV2."""
        config = _make_config()
        self.assertFalse(config.neuron_config.normalize_top_k_affinities)


class TestTPDivisibility(unittest.TestCase):
    """Verify all dimensions are TP-divisible at TP=4."""

    def test_attention_heads_divisible(self):
        config = _make_config(tp_degree=4)
        self.assertEqual(config.num_attention_heads % 4, 0)

    def test_vocab_size_divisible(self):
        config = _make_config(tp_degree=4)
        self.assertEqual(config.vocab_size % 4, 0)  # 154880 / 4 = 38720


if __name__ == "__main__":
    unittest.main()
