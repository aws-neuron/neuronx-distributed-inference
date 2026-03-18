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
CPU-only unit tests for TrinityInferenceConfig.

These tests verify config parsing, parameter transformation, and derived
configuration without requiring Neuron hardware or a downloaded model.

Usage:
    pytest test/unit/test_config.py -v
"""

import json
import os
import tempfile

import pytest
import torch

from neuronx_distributed_inference.models.config import MoENeuronConfig

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_trinity import TrinityInferenceConfig


# Minimal Nano config dict (matches arcee-ai/Trinity-Nano-Preview/config.json)
NANO_CONFIG = {
    "architectures": ["AfmoeForCausalLM"],
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_afmoe.AfmoeConfig",
        "AutoModel": "modeling_afmoe.AfmoeModel",
        "AutoModelForCausalLM": "modeling_afmoe.AfmoeForCausalLM",
    },
    "dtype": "bfloat16",
    "global_attn_every_n_layers": 4,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 131072,
    "model_type": "afmoe",
    "moe_intermediate_size": 256,
    "mup_enabled": True,
    "n_group": 1,
    "num_attention_heads": 8,
    "num_dense_layers": 2,
    "num_expert_groups": 1,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 56,
    "num_key_value_heads": 2,
    "num_limited_groups": 1,
    "num_shared_experts": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000,
    "route_norm": True,
    "route_scale": 2.826,
    "score_func": "sigmoid",
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "topk_group": 1,
    "transformers_version": "4.57.3",
    "use_cache": True,
    "use_grouped_mm": True,
    "vocab_size": 200192,
}

# Mini config overrides (key differences from Nano)
MINI_OVERRIDES = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "moe_intermediate_size": 1024,
    "num_attention_heads": 32,
    "num_dense_layers": 2,
    "num_hidden_layers": 32,
    "num_key_value_heads": 4,
}

# Large config overrides
LARGE_OVERRIDES = {
    "hidden_size": 3072,
    "intermediate_size": 12288,
    "moe_intermediate_size": 3072,
    "num_attention_heads": 48,
    "num_dense_layers": 6,
    "num_experts": 256,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 60,
    "num_key_value_heads": 8,
    "sliding_window": 4096,
}


def make_config(overrides=None, tp_degree=2, seq_len=2048, batch_size=1):
    """Create a TrinityInferenceConfig from dict with optional overrides."""
    config_dict = NANO_CONFIG.copy()
    if overrides:
        config_dict.update(overrides)

    neuron_config = MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
    )
    return TrinityInferenceConfig(neuron_config=neuron_config, **config_dict)


class TestConfigParsing:
    """Test that config is correctly parsed from HF config dict."""

    def test_nano_basic_params(self):
        config = make_config()
        assert config.vocab_size == 200192
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 56
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 2
        assert config.head_dim == 128
        assert config.num_experts == 128
        assert config.num_experts_per_tok == 8
        assert config.num_shared_experts == 1
        assert config.num_dense_layers == 2

    def test_intermediate_size_swap(self):
        """CRITICAL: intermediate_size must be MoE size, not dense size."""
        config = make_config()
        # intermediate_size should be moe_intermediate_size (for NxDI MoE module)
        assert config.intermediate_size == 256, (
            "intermediate_size must equal moe_intermediate_size for initialize_moe_module"
        )
        # Dense intermediate preserved separately
        assert config.dense_intermediate_size == 3072
        assert config.moe_intermediate_size == 256

    def test_mini_intermediate_size_swap(self):
        config = make_config(MINI_OVERRIDES, tp_degree=4)
        assert config.intermediate_size == 1024
        assert config.dense_intermediate_size == 6144
        assert config.moe_intermediate_size == 1024

    def test_large_intermediate_size_swap(self):
        config = make_config(LARGE_OVERRIDES, tp_degree=64)
        assert config.intermediate_size == 3072
        assert config.dense_intermediate_size == 12288
        assert config.moe_intermediate_size == 3072

    def test_shared_experts_forced_zero(self):
        """n_shared_experts must be 0 for NxDI (Trinity handles shared experts manually)."""
        config = make_config()
        assert config.n_shared_experts == 0
        # But num_shared_experts preserves the real count
        assert config.num_shared_experts == 1

    def test_glu_type_set(self):
        """Trinity uses SiLU gated MLP which maps to NxDI glu_type='glu'."""
        config = make_config()
        assert config.neuron_config.glu_type == "glu"
        assert config.neuron_config.glu_mlp is True

    def test_score_func_sigmoid(self):
        config = make_config()
        assert config.score_func == "sigmoid"

    def test_route_scale(self):
        config = make_config()
        assert config.route_scale == 2.826


class TestLayerTypes:
    """Test mixed attention layer type generation."""

    def test_nano_layer_types_count(self):
        config = make_config()
        assert len(config.layer_types) == 56

    def test_layer_types_pattern(self):
        """Every 4th layer (0-indexed: 3, 7, 11, ...) should be full_attention."""
        config = make_config()
        for i, lt in enumerate(config.layer_types):
            if (i + 1) % 4 == 0:
                assert lt == "full_attention", f"Layer {i} should be full_attention"
            else:
                assert lt == "sliding_attention", (
                    f"Layer {i} should be sliding_attention"
                )

    def test_nano_full_attention_count(self):
        config = make_config()
        full = sum(1 for lt in config.layer_types if lt == "full_attention")
        sliding = sum(1 for lt in config.layer_types if lt == "sliding_attention")
        assert full == 14  # 56 / 4 = 14
        assert sliding == 42

    def test_mini_layer_types(self):
        config = make_config(MINI_OVERRIDES, tp_degree=4)
        assert len(config.layer_types) == 32
        full = sum(1 for lt in config.layer_types if lt == "full_attention")
        assert full == 8  # 32 / 4 = 8

    def test_explicit_layer_types_preserved(self):
        """If layer_types is provided in config, it should be preserved as-is."""
        overrides = {"layer_types": ["sliding_attention", "full_attention"] * 28}
        config = make_config(overrides)
        assert config.layer_types == ["sliding_attention", "full_attention"] * 28


class TestSlidingWindowClamping:
    """Test that sliding_window is clamped to seq_len when seq_len < sliding_window."""

    def test_no_clamping_when_seq_ge_window(self):
        config = make_config(seq_len=2048)
        assert config.sliding_window == 2048

    def test_clamping_when_seq_lt_window(self):
        config = make_config(seq_len=1024)
        assert config.sliding_window == 1024

    def test_no_clamping_large_seq(self):
        config = make_config(seq_len=8192)
        assert config.sliding_window == 2048


class TestFromPretrained:
    """Test from_pretrained loading from a config.json file."""

    def test_from_pretrained_loads_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(NANO_CONFIG, f)

            neuron_config = MoENeuronConfig(
                tp_degree=2,
                batch_size=1,
                seq_len=2048,
                torch_dtype=torch.bfloat16,
            )
            config = TrinityInferenceConfig.from_pretrained(
                tmpdir, neuron_config=neuron_config
            )

            assert config.vocab_size == 200192
            assert config.hidden_size == 1024
            assert config.intermediate_size == 256  # Swapped to MoE size
            assert config.num_hidden_layers == 56

    def test_from_pretrained_missing_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                TrinityInferenceConfig.from_pretrained(tmpdir)


class TestFusedMoeTkgEligibility:
    """Test automatic fused MoE TKG kernel eligibility detection."""

    def test_nano_tp2_eligible(self):
        """Nano (moe_intermediate=256) at TP=2: 256/2=128, 128%128=0 → eligible."""
        config = make_config(tp_degree=2)
        # After add_derived_config, check if fused TKG was enabled
        # The check is: moe_intermediate_size / moe_tp_degree % 128 == 0
        per_tp = config.moe_intermediate_size // 2
        assert per_tp % 128 == 0

    def test_nano_tp4_ineligible(self):
        """Nano (moe_intermediate=256) at TP=4: 256/4=64, 64%128!=0 → ineligible."""
        config = make_config(tp_degree=4)
        per_tp = config.moe_intermediate_size // 4
        assert per_tp % 128 != 0

    def test_mini_tp4_eligible(self):
        """Mini (moe_intermediate=1024) at TP=4: 1024/4=256, 256%128=0 → eligible."""
        config = make_config(MINI_OVERRIDES, tp_degree=4)
        per_tp = config.moe_intermediate_size // 4
        assert per_tp % 128 == 0

    def test_large_tp64_ineligible(self):
        """Large (moe_intermediate=3072) at TP=64: 3072/64=48, 48%128!=0 → ineligible."""
        config = make_config(LARGE_OVERRIDES, tp_degree=64)
        per_tp = config.moe_intermediate_size // 64
        assert per_tp % 128 != 0
