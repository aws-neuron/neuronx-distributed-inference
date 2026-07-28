# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for GLM-5.2 config parsing (no Neuron device required)."""

import os
import sys

import pytest
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from neuronx_distributed_inference.models.config import MoENeuronConfig  # noqa: E402
from modeling_glm5 import GLM5InferenceConfig  # noqa: E402


# Minimal subset of the real zai-org/GLM-5.2-FP8 config.json.
GLM52_HF_CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "attention_bias": False,
    "first_k_dense_replace": 3,
    "head_dim": 192,
    "hidden_act": "silu",
    "hidden_size": 6144,
    "intermediate_size": 12288,
    "kv_lora_rank": 512,
    "max_position_embeddings": 1048576,
    "moe_intermediate_size": 2048,
    "n_group": 1,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 78,
    "num_key_value_heads": 64,
    "q_lora_rank": 2048,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_group": 1,
    "v_head_dim": 256,
    "vocab_size": 154880,
    "index_topk": 2048,
    "index_head_dim": 128,
    "pad_token_id": 154820,
    "eos_token_id": [154820, 154827, 154829],
}


def _make_config(dsa_enabled=None):
    nc = MoENeuronConfig(tp_degree=64, batch_size=1, seq_len=2048, torch_dtype=torch.bfloat16)

    def load_config(c):
        for k, v in GLM52_HF_CONFIG.items():
            setattr(c, k, v)
        if dsa_enabled is not None:
            c.dsa_enabled = dsa_enabled

    return GLM5InferenceConfig(neuron_config=nc, load_config=load_config)


def test_core_dims():
    cfg = _make_config()
    assert cfg.hidden_size == 6144
    assert cfg.num_attention_heads == 64
    assert cfg.num_hidden_layers == 78
    assert cfg.vocab_size == 154880
    assert cfg.q_lora_rank == 2048
    assert cfg.kv_lora_rank == 512
    assert cfg.qk_nope_head_dim == 192
    assert cfg.qk_rope_head_dim == 64
    assert cfg.v_head_dim == 256


def test_rope_theta_extracted_from_nested():
    # GLM-5.2 nests rope_theta under rope_parameters (8e6, not GLM-5's 1e6).
    cfg = _make_config()
    assert cfg.rope_theta == 8000000


def test_mla_cache_dim_with_dsa_disabled():
    # DSA off => KV cache stores [k_pe | compressed_kv] = 64 + 512 = 576.
    cfg = _make_config(dsa_enabled=False)
    assert cfg.dsa_enabled is False
    assert cfg.head_dim == cfg.qk_rope_head_dim + cfg.kv_lora_rank == 576


def test_mla_cache_dim_with_dsa_enabled():
    # DSA on => cache also holds indexer key (index_head_dim=128): 576 + 128 = 704.
    cfg = _make_config(dsa_enabled=True)
    assert cfg.dsa_enabled is True
    assert cfg.head_dim == 576 + cfg.index_head_dim == 704


def test_dsa_defaults_on_when_index_topk_positive():
    # The GLM-5.2 gotcha: index_topk=2048 > 0 auto-enables DSA, which breaks load
    # due to heterogeneous per-layer indexers. We disable it explicitly in usage.
    cfg = _make_config()  # no explicit dsa_enabled
    assert cfg.index_topk == 2048
    assert cfg.dsa_enabled is True


def test_moe_params():
    cfg = _make_config()
    assert cfg.n_routed_experts == 256
    assert cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 2048
    assert cfg.first_k_dense_replace == 3
    assert cfg.n_group == 1
    assert cfg.topk_group == 1
    assert cfg.routed_scaling_factor == 2.5


def test_shared_expert_handled_outside_fused_moe():
    # GLM-5.2 has 1 shared expert, but the modeling builds it as a separate
    # GLM5SharedExpert module (not via NXDI's fused MoE). So n_shared_experts is
    # forced to 0 for the fused path, and the real count is preserved separately.
    cfg = _make_config()
    assert cfg.n_shared_experts == 0
    assert cfg.num_shared_experts_actual == 1
