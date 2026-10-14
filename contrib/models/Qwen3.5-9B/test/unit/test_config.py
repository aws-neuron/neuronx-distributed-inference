# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Qwen3.5-9B inference configuration.

CPU-only tests that validate config parsing, layer type setup,
DeltaNet parameter defaults, RoPE configuration, and weight conversion logic.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

# Ensure the contrib root (Qwen3.5-9B/) is on sys.path
_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from src.modeling_qwen35 import (
    Qwen35InferenceConfig,
    convert_qwen35_hf_to_neuron_state_dict,
)
from neuronx_distributed_inference.models.config import NeuronConfig


def _make_config(**overrides):
    """Create a Qwen35InferenceConfig with reasonable defaults."""
    neuron_config = NeuronConfig(
        tp_degree=overrides.pop("tp_degree", 4),
        batch_size=1,
        seq_len=128,
        torch_dtype=torch.bfloat16,
    )
    defaults = dict(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        intermediate_size=12288,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        max_position_embeddings=262144,
        rope_theta=10000000,
        hidden_act="silu",
        tie_word_embeddings=False,
        # DeltaNet-specific
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    defaults.update(overrides)
    config = Qwen35InferenceConfig(neuron_config=neuron_config, **defaults)
    return config


class TestConfigParsing(unittest.TestCase):
    """Test basic config attribute initialization."""

    def test_hidden_size(self):
        config = _make_config()
        self.assertEqual(config.hidden_size, 4096)

    def test_num_hidden_layers(self):
        config = _make_config()
        self.assertEqual(config.num_hidden_layers, 32)

    def test_num_attention_heads(self):
        config = _make_config()
        self.assertEqual(config.num_attention_heads, 16)

    def test_num_key_value_heads(self):
        config = _make_config()
        self.assertEqual(config.num_key_value_heads, 4)

    def test_head_dim(self):
        config = _make_config()
        self.assertEqual(config.head_dim, 256)

    def test_intermediate_size(self):
        config = _make_config()
        self.assertEqual(config.intermediate_size, 12288)

    def test_vocab_size(self):
        config = _make_config()
        self.assertEqual(config.vocab_size, 248320)

    def test_hidden_act(self):
        config = _make_config()
        self.assertEqual(config.hidden_act, "silu")


class TestLayerTypes(unittest.TestCase):
    """Test hybrid layer type assignment (3 DeltaNet + 1 GQA) x 8."""

    def test_layer_types_length(self):
        config = _make_config()
        self.assertEqual(len(config.layer_types), 32)

    def test_layer_types_pattern(self):
        """Every 4th layer (3, 7, 11, ...) should be full_attention."""
        config = _make_config()
        for i in range(32):
            expected = "full_attention" if i % 4 == 3 else "linear_attention"
            self.assertEqual(config.layer_types[i], expected, f"Layer {i} mismatch")

    def test_deltanet_layer_count(self):
        config = _make_config()
        dn_count = sum(1 for t in config.layer_types if t == "linear_attention")
        self.assertEqual(dn_count, 24)

    def test_gqa_layer_count(self):
        config = _make_config()
        gqa_count = sum(1 for t in config.layer_types if t == "full_attention")
        self.assertEqual(gqa_count, 8)


class TestDeltaNetConfig(unittest.TestCase):
    """Test DeltaNet-specific configuration defaults."""

    def test_linear_num_value_heads(self):
        config = _make_config()
        self.assertEqual(config.linear_num_value_heads, 32)

    def test_linear_num_key_heads(self):
        config = _make_config()
        self.assertEqual(config.linear_num_key_heads, 16)

    def test_linear_key_head_dim(self):
        config = _make_config()
        self.assertEqual(config.linear_key_head_dim, 128)

    def test_linear_value_head_dim(self):
        config = _make_config()
        self.assertEqual(config.linear_value_head_dim, 128)

    def test_linear_conv_kernel_dim(self):
        config = _make_config()
        self.assertEqual(config.linear_conv_kernel_dim, 4)


class TestRoPEConfig(unittest.TestCase):
    """Test partial RoPE configuration."""

    def test_partial_rotary_factor(self):
        config = _make_config()
        self.assertAlmostEqual(config.partial_rotary_factor, 0.25)

    def test_rope_dim(self):
        """rope_dim = head_dim * partial_rotary_factor = 256 * 0.25 = 64."""
        config = _make_config()
        self.assertEqual(config.rope_dim, 64)

    def test_attn_output_gate(self):
        config = _make_config()
        self.assertTrue(config.attn_output_gate)

    def test_mrope_section(self):
        config = _make_config()
        self.assertEqual(config.mrope_section, [11, 11, 10])

    def test_mrope_interleaved(self):
        config = _make_config()
        self.assertTrue(config.mrope_interleaved)


class TestNeuronConfig(unittest.TestCase):
    """Test Neuron-specific configuration settings."""

    def test_neuron_config_cls(self):
        """Qwen3.5-9B is dense -- uses NeuronConfig, NOT MoENeuronConfig."""
        self.assertEqual(
            Qwen35InferenceConfig.get_neuron_config_cls(),
            NeuronConfig,
        )

    def test_required_attributes(self):
        config = _make_config()
        required = config.get_required_attributes()
        self.assertIn("hidden_size", required)
        self.assertIn("num_hidden_layers", required)
        self.assertIn("linear_num_value_heads", required)
        self.assertIn("linear_key_head_dim", required)
        self.assertIn("layer_types", required)

    def test_output_attentions_default(self):
        config = _make_config()
        self.assertFalse(config.output_attentions)

    def test_output_hidden_states_default(self):
        config = _make_config()
        self.assertFalse(config.output_hidden_states)


if __name__ == "__main__":
    unittest.main()
