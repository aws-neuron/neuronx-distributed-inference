# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Qwen3.5-2B HF-to-NxDI weight conversion.

CPU-only tests that validate:
- RMSNorm (+1 convention) weight conversion
- GQA q_proj interleaved split (query + gate)
- QK norm key renaming (q_norm -> q_layernorm, k_norm -> k_layernorm)
- Fused QKV concatenation
- DeltaNet layer weights pass through unchanged
- VL wrapper prefix stripping
- rank_util injection
"""

import os
import sys
import unittest

import torch

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from src.modeling_qwen35 import (
    Qwen35InferenceConfig,
    NeuronQwen35ForCausalLM,
    convert_qwen35_hf_to_neuron_state_dict,
)
from neuronx_distributed_inference.models.config import NeuronConfig


def _make_mini_config(num_layers=4, tp_degree=2, fused_qkv=True):
    """Create a small Qwen35InferenceConfig for testing."""
    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        seq_len=128,
        torch_dtype=torch.bfloat16,
        fused_qkv=fused_qkv,
    )
    config = Qwen35InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10000,
        hidden_act="silu",
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )
    return config


def _make_mini_state_dict(config):
    """Create a minimal HF-style state dict for conversion testing."""
    sd = {}
    H = config.hidden_size  # 256
    I = config.intermediate_size  # 512
    V = config.vocab_size  # 1000
    num_heads = config.num_attention_heads  # 4
    num_kv = config.num_key_value_heads  # 2
    head_dim = config.head_dim  # 64

    sd["embed_tokens.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    sd["norm.weight"] = torch.zeros(H, dtype=torch.bfloat16)  # +1 convention: zeros

    for l in range(config.num_hidden_layers):
        sd[f"layers.{l}.input_layernorm.weight"] = torch.zeros(H, dtype=torch.bfloat16)
        sd[f"layers.{l}.post_attention_layernorm.weight"] = torch.zeros(
            H, dtype=torch.bfloat16
        )

        # Dense MLP (all layers)
        sd[f"layers.{l}.mlp.gate_proj.weight"] = (
            torch.randn(I, H, dtype=torch.bfloat16) * 0.02
        )
        sd[f"layers.{l}.mlp.up_proj.weight"] = (
            torch.randn(I, H, dtype=torch.bfloat16) * 0.02
        )
        sd[f"layers.{l}.mlp.down_proj.weight"] = (
            torch.randn(H, I, dtype=torch.bfloat16) * 0.02
        )

        if config.layer_types[l] == "full_attention":
            # GQA layer: q_proj is interleaved [head0_q | head0_gate | head1_q | ...]
            q_proj = (
                torch.randn(num_heads * head_dim * 2, H, dtype=torch.bfloat16) * 0.02
            )
            sd[f"layers.{l}.self_attn.q_proj.weight"] = q_proj
            sd[f"layers.{l}.self_attn.k_proj.weight"] = (
                torch.randn(num_kv * head_dim, H, dtype=torch.bfloat16) * 0.02
            )
            sd[f"layers.{l}.self_attn.v_proj.weight"] = (
                torch.randn(num_kv * head_dim, H, dtype=torch.bfloat16) * 0.02
            )
            sd[f"layers.{l}.self_attn.o_proj.weight"] = (
                torch.randn(H, num_heads * head_dim, dtype=torch.bfloat16) * 0.02
            )
            sd[f"layers.{l}.self_attn.q_norm.weight"] = torch.zeros(
                head_dim, dtype=torch.bfloat16
            )
            sd[f"layers.{l}.self_attn.k_norm.weight"] = torch.zeros(
                head_dim, dtype=torch.bfloat16
            )
        else:
            # DeltaNet layer: minimal required weights
            key_dim = config.linear_num_key_heads * config.linear_key_head_dim  # 128
            value_dim = (
                config.linear_num_value_heads * config.linear_value_head_dim
            )  # 256
            conv_dim = key_dim * 2 + value_dim  # 512
            sd[f"layers.{l}.linear_attn.in_proj_qkv.weight"] = (
                torch.randn(conv_dim, H, dtype=torch.bfloat16) * 0.02
            )
            sd[f"layers.{l}.linear_attn.in_proj_z.weight"] = (
                torch.randn(value_dim, H, dtype=torch.bfloat16) * 0.02
            )
            sd[f"layers.{l}.linear_attn.in_proj_a.weight"] = (
                torch.randn(config.linear_num_value_heads, H, dtype=torch.bfloat16)
                * 0.02
            )
            sd[f"layers.{l}.linear_attn.in_proj_b.weight"] = (
                torch.randn(config.linear_num_value_heads, H, dtype=torch.bfloat16)
                * 0.02
            )
            sd[f"layers.{l}.linear_attn.conv1d.weight"] = (
                torch.randn(
                    conv_dim, 1, config.linear_conv_kernel_dim, dtype=torch.bfloat16
                )
                * 0.02
            )
            sd[f"layers.{l}.linear_attn.A_log"] = torch.randn(
                config.linear_num_value_heads, dtype=torch.bfloat16
            )
            sd[f"layers.{l}.linear_attn.dt_bias"] = torch.randn(
                config.linear_num_value_heads, dtype=torch.bfloat16
            )
            sd[f"layers.{l}.linear_attn.norm.weight"] = (
                torch.randn(value_dim, dtype=torch.bfloat16) * 0.5
            )
            sd[f"layers.{l}.linear_attn.out_proj.weight"] = (
                torch.randn(H, value_dim, dtype=torch.bfloat16) * 0.02
            )

    return sd


class TestNormConversion(unittest.TestCase):
    """Test (+1 convention) RMSNorm weight conversion."""

    def test_norm_weight_adds_one(self):
        """Weights initialized to zero should become 1.0 after conversion."""
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)
        # norm.weight was zeros -> should now be ones
        torch.testing.assert_close(
            result["norm.weight"],
            torch.ones_like(result["norm.weight"]),
        )

    def test_input_layernorm_adds_one(self):
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)
        for l in range(config.num_hidden_layers):
            w = result[f"layers.{l}.input_layernorm.weight"]
            self.assertTrue(
                torch.allclose(w, torch.ones_like(w)),
                f"Layer {l} input_layernorm not converted",
            )

    def test_post_attn_layernorm_adds_one(self):
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)
        for l in range(config.num_hidden_layers):
            w = result[f"layers.{l}.post_attention_layernorm.weight"]
            self.assertTrue(
                torch.allclose(w, torch.ones_like(w)),
                f"Layer {l} post_attention_layernorm not converted",
            )

    def test_qk_norm_adds_one(self):
        """Q/K norms on GQA layers should also get +1 applied."""
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)
        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "full_attention":
                q_w = result[f"layers.{l}.self_attn.q_layernorm.weight"]
                k_w = result[f"layers.{l}.self_attn.k_layernorm.weight"]
                self.assertTrue(
                    torch.allclose(q_w, torch.ones_like(q_w)),
                    f"Layer {l} q_layernorm not converted",
                )
                self.assertTrue(
                    torch.allclose(k_w, torch.ones_like(k_w)),
                    f"Layer {l} k_layernorm not converted",
                )


class TestQProjSplit(unittest.TestCase):
    """Test q_proj interleaved split into query + gate."""

    def test_q_proj_split_shapes(self):
        """q_proj (num_heads * head_dim * 2, H) -> separate query and gate."""
        config = _make_mini_config(fused_qkv=False)
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)

        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "full_attention":
                # After split: q_proj should be (num_heads * head_dim, H) = (256, 256)
                q_w = result[f"layers.{l}.self_attn.q_proj.weight"]
                gate_w = result[f"layers.{l}.self_attn.output_gate_proj.weight"]
                expected_shape = (
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                )
                self.assertEqual(
                    q_w.shape, expected_shape, f"Layer {l} q_proj shape wrong"
                )
                self.assertEqual(
                    gate_w.shape, expected_shape, f"Layer {l} gate shape wrong"
                )

    def test_q_proj_deinterleave_correct(self):
        """Verify the interleaved split correctly separates query and gate."""
        config = _make_mini_config(fused_qkv=False)
        sd = _make_mini_state_dict(config)

        # Create a known pattern: head0 query is 1s, head0 gate is 2s, etc.
        l = 3  # First full_attention layer (layer 3)
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        H = config.hidden_size

        interleaved = torch.zeros(num_heads * head_dim * 2, H, dtype=torch.bfloat16)
        for h in range(num_heads):
            interleaved[h * head_dim * 2 : h * head_dim * 2 + head_dim, :] = float(
                h + 1
            )  # query
            interleaved[h * head_dim * 2 + head_dim : (h + 1) * head_dim * 2, :] = (
                float(h + 100)
            )  # gate

        sd[f"layers.{l}.self_attn.q_proj.weight"] = interleaved
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)

        q_w = result[f"layers.{l}.self_attn.q_proj.weight"]
        gate_w = result[f"layers.{l}.self_attn.output_gate_proj.weight"]

        for h in range(num_heads):
            q_head = q_w[h * head_dim : (h + 1) * head_dim, :]
            gate_head = gate_w[h * head_dim : (h + 1) * head_dim, :]
            self.assertTrue(
                torch.all(q_head == float(h + 1)), f"Head {h} query values wrong"
            )
            self.assertTrue(
                torch.all(gate_head == float(h + 100)), f"Head {h} gate values wrong"
            )


class TestQKNormRename(unittest.TestCase):
    """Test q_norm -> q_layernorm and k_norm -> k_layernorm renaming."""

    def test_old_keys_removed(self):
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)
        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "full_attention":
                self.assertNotIn(f"layers.{l}.self_attn.q_norm.weight", result)
                self.assertNotIn(f"layers.{l}.self_attn.k_norm.weight", result)

    def test_new_keys_present(self):
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)
        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "full_attention":
                self.assertIn(f"layers.{l}.self_attn.q_layernorm.weight", result)
                self.assertIn(f"layers.{l}.self_attn.k_layernorm.weight", result)


class TestFusedQKV(unittest.TestCase):
    """Test fused QKV concatenation for attention layers."""

    def test_fused_qkv_shape(self):
        config = _make_mini_config(fused_qkv=True)
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)

        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "full_attention":
                fused_key = f"layers.{l}.self_attn.Wqkv.weight"
                self.assertIn(fused_key, result, f"Layer {l} missing Wqkv")

                q_dim = config.num_attention_heads * config.head_dim
                k_dim = config.num_key_value_heads * config.head_dim
                v_dim = config.num_key_value_heads * config.head_dim
                expected_rows = q_dim + k_dim + v_dim
                self.assertEqual(result[fused_key].shape[0], expected_rows)

    def test_fused_qkv_removes_individual_keys(self):
        config = _make_mini_config(fused_qkv=True)
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)

        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "full_attention":
                self.assertNotIn(f"layers.{l}.self_attn.q_proj.weight", result)
                self.assertNotIn(f"layers.{l}.self_attn.k_proj.weight", result)
                self.assertNotIn(f"layers.{l}.self_attn.v_proj.weight", result)


class TestDeltaNetPassthrough(unittest.TestCase):
    """Test that DeltaNet layer weights pass through conversion unchanged."""

    def test_deltanet_weights_unchanged(self):
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)

        # Record original DeltaNet weights
        originals = {}
        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "linear_attention":
                key = f"layers.{l}.linear_attn.in_proj_qkv.weight"
                originals[key] = sd[key].clone()

        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)

        for key, orig in originals.items():
            self.assertIn(key, result, f"Missing: {key}")
            torch.testing.assert_close(
                result[key], orig, msg=f"DeltaNet weight changed: {key}"
            )

    def test_deltanet_norm_not_converted(self):
        """DeltaNet layers use standard RMSNorm (NOT +1 convention).
        The norm weight should NOT be changed."""
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)

        # Set DeltaNet norm to a known non-zero value
        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "linear_attention":
                sd[f"layers.{l}.linear_attn.norm.weight"] = torch.full(
                    (config.linear_num_value_heads * config.linear_value_head_dim,),
                    0.87,
                    dtype=torch.bfloat16,
                )

        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)

        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "linear_attention":
                w = result[f"layers.{l}.linear_attn.norm.weight"]
                # Should still be ~0.87, NOT 1.87
                self.assertTrue(
                    torch.allclose(w, torch.full_like(w, 0.87), atol=0.01),
                    f"Layer {l} DeltaNet norm was incorrectly modified",
                )


class TestRankUtil(unittest.TestCase):
    """Test rank_util tensor injection."""

    def test_rank_util_present(self):
        config = _make_mini_config(tp_degree=4)
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)
        self.assertIn("rank_util.rank", result)
        expected = torch.arange(0, 4, dtype=torch.int32)
        torch.testing.assert_close(result["rank_util.rank"], expected)

    def test_gqa_layer_rank_util(self):
        config = _make_mini_config(tp_degree=4)
        sd = _make_mini_state_dict(config)
        result = convert_qwen35_hf_to_neuron_state_dict(sd, config)
        for l in range(config.num_hidden_layers):
            if config.layer_types[l] == "full_attention":
                key = f"layers.{l}.self_attn.rank_util.rank"
                self.assertIn(key, result)
                expected = torch.arange(0, 4, dtype=torch.int32)
                torch.testing.assert_close(result[key], expected)


class TestVLPrefixStripping(unittest.TestCase):
    """Test VL wrapper prefix stripping in convert_hf_to_neuron_state_dict."""

    def test_language_model_prefix_stripped(self):
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)

        # Wrap with VL prefix
        vl_sd = {}
        for k, v in sd.items():
            vl_sd[f"language_model.{k}"] = v
        vl_sd["visual.encoder.weight"] = torch.zeros(10)  # should be skipped
        vl_sd["mtp.something"] = torch.zeros(5)  # should be skipped

        result = NeuronQwen35ForCausalLM.convert_hf_to_neuron_state_dict(vl_sd, config)
        self.assertNotIn("visual.encoder.weight", result)
        self.assertNotIn("mtp.something", result)
        self.assertIn("norm.weight", result)

    def test_model_language_model_prefix_stripped(self):
        config = _make_mini_config()
        sd = _make_mini_state_dict(config)

        vl_sd = {}
        for k, v in sd.items():
            vl_sd[f"model.language_model.{k}"] = v

        result = NeuronQwen35ForCausalLM.convert_hf_to_neuron_state_dict(vl_sd, config)
        self.assertIn("norm.weight", result)


if __name__ == "__main__":
    unittest.main()
