# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Qwen3.5/3.6-27B inference configuration.

CPU-only tests that validate config parsing, layer type setup,
DeltaNet parameter defaults, RoPE configuration, and weight conversion logic.
These tests are architecture-level and apply to both Qwen3.5-27B and Qwen3.6-27B.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

# Ensure the contrib root (Qwen3.6-27B/) is on sys.path
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
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        intermediate_size=17408,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        max_position_embeddings=131072,
        rope_theta=10000,
        hidden_act="silu",
        # DeltaNet-specific
        linear_num_value_heads=48,
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
        self.assertEqual(config.hidden_size, 5120)

    def test_num_hidden_layers(self):
        config = _make_config()
        self.assertEqual(config.num_hidden_layers, 64)

    def test_num_attention_heads(self):
        config = _make_config()
        self.assertEqual(config.num_attention_heads, 24)

    def test_num_key_value_heads(self):
        config = _make_config()
        self.assertEqual(config.num_key_value_heads, 4)

    def test_head_dim(self):
        config = _make_config()
        self.assertEqual(config.head_dim, 256)

    def test_intermediate_size(self):
        config = _make_config()
        self.assertEqual(config.intermediate_size, 17408)

    def test_vocab_size(self):
        config = _make_config()
        self.assertEqual(config.vocab_size, 248320)

    def test_hidden_act(self):
        config = _make_config()
        self.assertEqual(config.hidden_act, "silu")


class TestLayerTypes(unittest.TestCase):
    """Test hybrid layer type assignment (3 DeltaNet + 1 GQA) x 16."""

    def test_layer_types_length(self):
        config = _make_config()
        self.assertEqual(len(config.layer_types), 64)

    def test_layer_types_pattern(self):
        """Every 4th layer (3, 7, 11, ...) should be full_attention."""
        config = _make_config()
        for i in range(64):
            expected = "full_attention" if i % 4 == 3 else "linear_attention"
            self.assertEqual(config.layer_types[i], expected, f"Layer {i} mismatch")

    def test_deltanet_layer_count(self):
        config = _make_config()
        dn_count = sum(1 for t in config.layer_types if t == "linear_attention")
        self.assertEqual(dn_count, 48)

    def test_gqa_layer_count(self):
        config = _make_config()
        gqa_count = sum(1 for t in config.layer_types if t == "full_attention")
        self.assertEqual(gqa_count, 16)


class TestDeltaNetConfig(unittest.TestCase):
    """Test DeltaNet-specific configuration defaults."""

    def test_linear_num_value_heads(self):
        config = _make_config()
        self.assertEqual(config.linear_num_value_heads, 48)

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

    def test_gdn_apc_checkpoint_defaults(self):
        config = _make_config()
        self.assertFalse(config.use_hybrid_cache_manager)
        self.assertFalse(config.use_hybrid_apc_manager)
        self.assertEqual(config.gdn_checkpoint_interval, 256)
        self.assertEqual(config.max_gdn_checkpoint_slots, 8)
        self.assertEqual(config.hybrid_apc_layout_version, 1)
        self.assertFalse(config.hybrid_apc_allow_residual_replay)
        self.assertEqual(config.gdn_recurrent_cache_dtype, "float32")
        self.assertEqual(config.gdn_conv_cache_dtype, "bfloat16")
        self.assertEqual(config.hybrid_recurrent_cache_dtype, "float32")
        self.assertEqual(config.hybrid_conv_cache_dtype, "bfloat16")
        self.assertEqual(config.hybrid_cache_mode, "all")
        self.assertTrue(config.hybrid_cache_prefix_boundary_only)
        self.assertTrue(config.hybrid_cache_block_boundary_only)
        self.assertFalse(config.hybrid_apc_require_vllm_metadata)
        self.assertTrue(config.hybrid_apc_allow_local_hash_fallback)
        self.assertFalse(config.hybrid_apc_require_attention_block_refs)
        self.assertTrue(config.hybrid_apc_reject_unbacked_attention_hits)
        self.assertFalse(config.hybrid_apc_disable_unbacked_prefix_reads)
        self.assertTrue(config.use_text_only_cte_inputs)
        self.assertTrue(config.use_compact_cte_attention_mask)
        self.assertFalse(config.use_cold_zero_conv_fast_path)

    def test_hybrid_apc_manager_defaults_fail_closed(self):
        config = _make_config(
            use_hybrid_apc_manager=True,
            gdn_checkpoint_interval=128,
        )

        self.assertTrue(config.hybrid_apc_require_vllm_metadata)
        self.assertFalse(config.hybrid_apc_allow_local_hash_fallback)
        self.assertTrue(config.hybrid_apc_require_attention_block_refs)
        self.assertTrue(config.hybrid_apc_reject_unbacked_attention_hits)

    def test_hybrid_apc_validation_can_opt_into_local_fallback(self):
        config = _make_config(
            use_hybrid_apc_manager=True,
            hybrid_apc_require_vllm_metadata=False,
            hybrid_apc_allow_local_hash_fallback=True,
            hybrid_apc_require_attention_block_refs=False,
            gdn_checkpoint_interval=128,
        )

        self.assertFalse(config.hybrid_apc_require_vllm_metadata)
        self.assertTrue(config.hybrid_apc_allow_local_hash_fallback)
        self.assertFalse(config.hybrid_apc_require_attention_block_refs)

    def test_hybrid_apc_require_vllm_metadata_enables_strict_metadata(self):
        config = _make_config(
            use_hybrid_apc_manager=True,
            hybrid_apc_require_vllm_metadata=True,
            gdn_checkpoint_interval=128,
        )

        self.assertFalse(config.hybrid_apc_allow_local_hash_fallback)
        self.assertTrue(config.hybrid_apc_require_attention_block_refs)
        self.assertTrue(config.hybrid_apc_reject_unbacked_attention_hits)

    def test_gdn_checkpoint_interval_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "gdn_checkpoint_interval"):
            _make_config(gdn_checkpoint_interval=0)

    def test_hybrid_cache_dtype_aliases_are_normalized(self):
        config = _make_config(
            hybrid_recurrent_cache_dtype="fp32",
            hybrid_conv_cache_dtype="bf16",
        )
        self.assertEqual(config.hybrid_recurrent_cache_dtype, "float32")
        self.assertEqual(config.hybrid_conv_cache_dtype, "bfloat16")

    def test_hybrid_cache_dtype_rejects_fp8(self):
        with self.assertRaisesRegex(ValueError, "hybrid_recurrent_cache_dtype"):
            _make_config(hybrid_recurrent_cache_dtype="fp8")

    def test_hybrid_apc_requires_float32_recurrent_checkpoint_cache(self):
        with self.assertRaisesRegex(ValueError, "requires float32 recurrent GDN"):
            _make_config(
                use_hybrid_apc_manager=True,
                gdn_checkpoint_interval=128,
                hybrid_recurrent_cache_dtype="bf16",
            )

    def test_hybrid_apc_rejects_non_all_mode(self):
        with self.assertRaisesRegex(ValueError, "hybrid_cache_mode='all'"):
            _make_config(use_hybrid_apc_manager=True, hybrid_cache_mode="align")

    def test_hybrid_apc_rejects_residual_replay_in_v0(self):
        with self.assertRaisesRegex(ValueError, "reserved for v1"):
            _make_config(
                use_hybrid_apc_manager=True,
                hybrid_apc_allow_residual_replay=True,
                gdn_checkpoint_interval=128,
            )

    def test_static_and_apc_managers_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _make_config(
                use_hybrid_cache_manager=True,
                use_hybrid_apc_manager=True,
            )


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
        """Qwen3.5/3.6-27B is dense -- uses NeuronConfig, NOT MoENeuronConfig."""
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
