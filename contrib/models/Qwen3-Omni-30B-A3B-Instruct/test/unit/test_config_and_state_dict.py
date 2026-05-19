#!/usr/bin/env python3
"""
Unit tests for Qwen3-Omni-MoE config loading and state dict conversion.
These tests run on CPU without Neuron devices.
"""
import json
import os
import tempfile

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modeling_qwen3_omni_moe import (
    Qwen3OmniMoeInferenceConfig,
    load_qwen3_omni_thinker_text_config,
    _strip_thinker_prefix,
    convert_qwen3_omni_moe_hf_to_neuron_state_dict,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig


SAMPLE_CONFIG = {
    "model_type": "qwen3_omni_moe",
    "thinker_config": {
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 152064,
            "max_position_embeddings": 65536,
            "moe_intermediate_size": 768,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "hidden_act": "silu",
            "tie_word_embeddings": False,
            "shared_expert_intermediate_size": 0,
            "rope_scaling": {
                "interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
        },
        "pad_token_id": None,
    },
}


@pytest.fixture
def config_dir(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(SAMPLE_CONFIG))
    return str(tmp_path)


@pytest.fixture
def neuron_config():
    return MoENeuronConfig(
        tp_degree=4,
        batch_size=1,
        seq_len=128,
        torch_dtype=torch.bfloat16,
    )


def test_config_loads_from_nested_structure(config_dir, neuron_config):
    config = Qwen3OmniMoeInferenceConfig(
        neuron_config,
        load_config=load_qwen3_omni_thinker_text_config(config_dir),
    )
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 4
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 4
    assert config.head_dim == 128
    assert config.num_experts == 8
    assert config.num_experts_per_tok == 2
    assert config.moe_intermediate_size == 768
    assert config.vocab_size == 152064
    assert config.rope_theta == 1000000


def test_config_moe_settings(config_dir, neuron_config):
    config = Qwen3OmniMoeInferenceConfig(
        neuron_config,
        load_config=load_qwen3_omni_thinker_text_config(config_dir),
    )
    assert config.num_local_experts == config.num_experts
    assert config.n_shared_experts == 0
    assert config.intermediate_size == config.moe_intermediate_size
    assert config.neuron_config.router_config.dtype == torch.float32
    assert config.neuron_config.router_config.act_fn == "softmax"
    assert config.neuron_config.normalize_top_k_affinities is True
    assert config.neuron_config.disable_numeric_cc_token is True


def test_config_missing_thinker_raises(tmp_path, neuron_config):
    bad_config = {"model_type": "something_else"}
    (tmp_path / "config.json").write_text(json.dumps(bad_config))
    with pytest.raises(ValueError, match="thinker_config.text_config"):
        Qwen3OmniMoeInferenceConfig(
            neuron_config,
            load_config=load_qwen3_omni_thinker_text_config(str(tmp_path)),
        )


def test_strip_thinker_prefix():
    sd = {
        "thinker.model.embed_tokens.weight": torch.randn(10, 10),
        "thinker.model.layers.0.self_attn.q_proj.weight": torch.randn(10, 10),
        "thinker.model.layers.0.mlp.gate.weight": torch.randn(10, 10),
        "thinker.model.norm.weight": torch.randn(10),
        "thinker.lm_head.weight": torch.randn(10, 10),
        "talker.model.layers.0.weight": torch.randn(5, 5),
        "code2wav.decoder.weight": torch.randn(3, 3),
    }
    stripped = _strip_thinker_prefix(sd)
    assert "embed_tokens.weight" in stripped
    assert "layers.0.self_attn.q_proj.weight" in stripped
    assert "layers.0.mlp.gate.weight" in stripped
    assert "norm.weight" in stripped
    assert "lm_head.weight" in stripped
    assert "talker.model.layers.0.weight" not in stripped
    assert "code2wav.decoder.weight" not in stripped
    assert len(stripped) == 5


def test_strip_with_model_thinker_prefix():
    sd = {
        "model.thinker.model.embed_tokens.weight": torch.randn(10, 10),
        "model.thinker.lm_head.weight": torch.randn(10, 10),
    }
    stripped = _strip_thinker_prefix(sd)
    assert "embed_tokens.weight" in stripped
    assert "lm_head.weight" in stripped


def test_strip_no_prefix():
    sd = {
        "embed_tokens.weight": torch.randn(10, 10),
        "layers.0.self_attn.q_proj.weight": torch.randn(10, 10),
        "lm_head.weight": torch.randn(10, 10),
    }
    stripped = _strip_thinker_prefix(sd)
    assert "embed_tokens.weight" in stripped
    assert len(stripped) == 3


def _make_fake_thinker_state_dict(num_layers=2, num_experts=4, hidden=64, intermediate=32):
    """Build a fake HF-format state dict with thinker prefix."""
    sd = {}
    sd["thinker.model.embed_tokens.weight"] = torch.randn(100, hidden)
    sd["thinker.model.norm.weight"] = torch.randn(hidden)
    sd["thinker.lm_head.weight"] = torch.randn(100, hidden)

    for l in range(num_layers):
        pfx = f"thinker.model.layers.{l}"
        sd[f"{pfx}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"{pfx}.self_attn.k_proj.weight"] = torch.randn(hidden // 8, hidden)
        sd[f"{pfx}.self_attn.v_proj.weight"] = torch.randn(hidden // 8, hidden)
        sd[f"{pfx}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"{pfx}.self_attn.q_norm.weight"] = torch.randn(hidden // (hidden // 8))
        sd[f"{pfx}.self_attn.k_norm.weight"] = torch.randn(hidden // (hidden // 8))
        sd[f"{pfx}.input_layernorm.weight"] = torch.randn(hidden)
        sd[f"{pfx}.post_attention_layernorm.weight"] = torch.randn(hidden)
        sd[f"{pfx}.mlp.gate.weight"] = torch.randn(num_experts, hidden)
        for e in range(num_experts):
            sd[f"{pfx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(intermediate, hidden)
            sd[f"{pfx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(intermediate, hidden)
            sd[f"{pfx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(hidden, intermediate)

    return sd


def test_full_state_dict_conversion(config_dir, neuron_config):
    config = Qwen3OmniMoeInferenceConfig(
        neuron_config,
        load_config=load_qwen3_omni_thinker_text_config(config_dir),
    )
    # override for small test
    config.num_hidden_layers = 2
    config.num_experts = 4
    config.num_local_experts = 4
    config.hidden_size = 64
    config.moe_intermediate_size = 32
    config.intermediate_size = 32

    sd = _make_fake_thinker_state_dict(num_layers=2, num_experts=4, hidden=64, intermediate=32)
    neuron_sd = convert_qwen3_omni_moe_hf_to_neuron_state_dict(sd, config)

    # Check prefix stripped
    assert "embed_tokens.weight" in neuron_sd
    assert "lm_head.weight" in neuron_sd
    assert "norm.weight" in neuron_sd

    # Check rank utils added
    assert "rank_util.rank" in neuron_sd
    assert "layers.0.self_attn.rank_util.rank" in neuron_sd

    # Check qk norm renamed
    assert "layers.0.self_attn.q_layernorm.weight" in neuron_sd
    assert "layers.0.self_attn.k_layernorm.weight" in neuron_sd
    assert "layers.0.self_attn.q_norm.weight" not in neuron_sd

    # Check router renamed
    assert "layers.0.mlp.router.linear_router.weight" in neuron_sd
    assert "layers.0.mlp.gate.weight" not in neuron_sd

    # Check expert weights reorganized
    gate_up = neuron_sd["layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"]
    assert gate_up.shape == (4, 64, 64)  # (num_experts, hidden, 2*intermediate)
    down = neuron_sd["layers.0.mlp.expert_mlps.mlp_op.down_proj.weight"]
    assert down.shape == (4, 32, 64)  # (num_experts, intermediate, hidden)

    # Check individual expert keys removed
    assert "layers.0.mlp.experts.0.gate_proj.weight" not in neuron_sd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
