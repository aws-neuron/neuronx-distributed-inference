# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for DeepSeek V3 on Neuron.

Tests compilation, loading, inference accuracy, and performance using either:
- A mini DeepSeek V3 model (1 dense + 1 MoE layer, random weights, tp=2)
- The full 671B model with pre-sharded weights (tp=64, trn2.48xlarge)

The mini model validates the full NXDI pipeline (compile -> load -> generate)
without requiring the 671B weights or a trn2.48xlarge instance.

Environment variables:
    DEEPSEEK_MODEL_PATH     Path to HF model weights (default: creates mini model)
    DEEPSEEK_COMPILED_PATH  Path to compiled artifacts (default: /tmp/deepseek_v3_test_traced)
    DEEPSEEK_TP_DEGREE      Tensor parallelism degree (default: 2)
    DEEPSEEK_SEQ_LEN        Max sequence length (default: 128)
    TTFT_THRESHOLD_MS       Max TTFT in ms (default: 60000 for mini model)
    THROUGHPUT_THRESHOLD     Min throughput in tok/s (default: 1.0 for mini model)

Prerequisites:
    - Neuron device with >= DEEPSEEK_TP_DEGREE NeuronCores available
    - NXDI installed (neuronx_distributed_inference)
    - transformers >= 4.56.2

Usage:
    # Mini model (default, needs 2 NeuronCores):
    pytest test/integration/test_model.py --capture=tee-sys

    # Full 671B model (needs trn2.48xlarge):
    DEEPSEEK_MODEL_PATH=/path/to/DeepSeek-V3-0324-FP8 \
    DEEPSEEK_COMPILED_PATH=/scratch/deepseek_v3_traced \
    DEEPSEEK_TP_DEGREE=64 \
    pytest test/integration/test_model.py --capture=tee-sys -k "not mini"

Known Issues:
    - Mini model compilation fails with NCC_IBIR297 internal compiler error on
      SDK 2.28 (neuronx-cc 2.23.6484). This is a BIR verifier regression in the
      Neuron compiler that affects the DeepSeek V3 MLA attention graph at small
      tensor dimensions. The full 671B model at tp=64 compiles successfully.
      Mini model tests are skipped until this compiler issue is resolved.
"""

import gc
import json
import os
import shutil
import sys
import time

import pytest
import torch
from safetensors.torch import save_file

# Ensure the contrib root (DeepSeek-V3/) is on sys.path so that `src.*` imports
# resolve to the local contrib code regardless of how the test is invoked.
_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

# ── Configuration from environment ──────────────────────────────────────

MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "")
COMPILED_PATH = os.environ.get("DEEPSEEK_COMPILED_PATH", "/tmp/deepseek_v3_test_traced")
TP_DEGREE = int(os.environ.get("DEEPSEEK_TP_DEGREE", "2"))
SEQ_LEN = int(os.environ.get("DEEPSEEK_SEQ_LEN", "128"))
TTFT_THRESHOLD_MS = float(os.environ.get("TTFT_THRESHOLD_MS", "60000"))
THROUGHPUT_THRESHOLD = float(os.environ.get("THROUGHPUT_THRESHOLD", "1.0"))

USE_MINI_MODEL = not MODEL_PATH

# Mini model compilation hits NCC_IBIR297 on SDK 2.28 (neuronx-cc 2.23.6484).
# The bug is in the BIR verifier stage and is invariant to model dimensions,
# MoE configuration, compiler flags, and optimization levels. The full 671B
# model compiles fine at tp=64.
MINI_MODEL_SKIP_REASON = (
    "Mini model compilation blocked by NCC_IBIR297 internal compiler error "
    "in neuronx-cc 2.23.6484 (SDK 2.28). The DeepSeek V3 MLA attention graph "
    "triggers a BIR verifier bug at small TP degrees. Full 671B model at tp=64 "
    "is unaffected. See README.md Caveats section."
)

requires_compiled_model = pytest.mark.skipif(
    USE_MINI_MODEL, reason=MINI_MODEL_SKIP_REASON
)

# ── Mini model config ───────────────────────────────────────────────────

MINI_CONFIG = {
    "_name_or_path": "deepseek-ai/DeepSeek-V3",
    "architectures": ["DeepseekV3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "deepseek-ai/DeepSeek-V3--configuration_deepseek.DeepseekV3Config",
        "AutoModel": "deepseek-ai/DeepSeek-V3--modeling_deepseek.DeepseekV3Model",
        "AutoModelForCausalLM": "deepseek-ai/DeepSeek-V3--modeling_deepseek.DeepseekV3ForCausalLM",
    },
    "aux_loss_alpha": 0.001,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "ep_size": 1,
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "kv_lora_rank": 512,
    "max_position_embeddings": 4096,
    "model_type": "deepseek_v3",
    "moe_intermediate_size": 1408,
    "moe_layer_freq": 1,
    "n_group": 8,
    "n_routed_experts": 64,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 16,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 2,
    "num_key_value_heads": 16,
    "num_nextn_predict_layers": 0,
    "pretraining_tp": 1,
    "q_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32, "beta_slow": 1, "factor": 40,
        "mscale": 1.0, "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096, "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "seq_aux": True,
    "tie_word_embeddings": False,
    "topk_group": 4,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.38.2",
    "use_cache": True,
    "v_head_dim": 128,
    "vocab_size": 32000,
}


def _create_mini_model(model_dir):
    """Create a mini DeepSeek V3 model with random weights and a tokenizer."""
    os.makedirs(model_dir, exist_ok=True)
    cfg = MINI_CONFIG

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    torch.manual_seed(42)
    hidden = cfg["hidden_size"]
    intermediate = cfg["intermediate_size"]
    moe_intermediate = cfg["moe_intermediate_size"]
    vocab = cfg["vocab_size"]
    n_heads = cfg["num_attention_heads"]
    kv_lora_rank = cfg["kv_lora_rank"]
    q_lora_rank = cfg["q_lora_rank"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head = cfg["v_head_dim"]
    n_experts = cfg["n_routed_experts"]
    n_layers = cfg["num_hidden_layers"]
    first_k_dense = cfg["first_k_dense_replace"]

    sd = {}
    sd["model.embed_tokens.weight"] = torch.randn(vocab, hidden, dtype=torch.bfloat16) * 0.02
    for i in range(n_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.q_a_proj.weight"] = torch.randn(q_lora_rank, hidden, dtype=torch.bfloat16) * 0.02
        sd[f"{p}.self_attn.q_a_layernorm.weight"] = torch.ones(q_lora_rank, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.q_b_proj.weight"] = torch.randn(n_heads * (qk_nope + qk_rope), q_lora_rank, dtype=torch.bfloat16) * 0.02
        sd[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(kv_lora_rank + qk_rope, hidden, dtype=torch.bfloat16) * 0.02
        sd[f"{p}.self_attn.kv_a_layernorm.weight"] = torch.ones(kv_lora_rank, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.kv_b_proj.weight"] = torch.randn(n_heads * (qk_nope + v_head), kv_lora_rank, dtype=torch.bfloat16) * 0.02
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, n_heads * v_head, dtype=torch.bfloat16) * 0.02
        if i < first_k_dense:
            sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(intermediate, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.up_proj.weight"] = torch.randn(intermediate, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02
        else:
            sd[f"{p}.mlp.gate.weight"] = torch.randn(n_experts, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.gate.e_score_correction_bias"] = torch.randn(n_experts, dtype=torch.bfloat16) * 0.01
            for e in range(n_experts):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(moe_intermediate, hidden, dtype=torch.bfloat16) * 0.02
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(moe_intermediate, hidden, dtype=torch.bfloat16) * 0.02
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(hidden, moe_intermediate, dtype=torch.bfloat16) * 0.02
            shared_int = moe_intermediate * cfg["n_shared_experts"]
            sd[f"{p}.mlp.shared_experts.gate_proj.weight"] = torch.randn(shared_int, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.shared_experts.up_proj.weight"] = torch.randn(shared_int, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.shared_experts.down_proj.weight"] = torch.randn(hidden, shared_int, dtype=torch.bfloat16) * 0.02

    sd["model.norm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(vocab, hidden, dtype=torch.bfloat16) * 0.02
    save_file(sd, os.path.join(model_dir, "model.safetensors"))

    # Download a small tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tok.save_pretrained(model_dir)
    return model_dir


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_path():
    """Return path to model weights (creates mini model if needed)."""
    if USE_MINI_MODEL:
        path = "/tmp/deepseek_v3_mini_model"
        if not os.path.exists(os.path.join(path, "model.safetensors")):
            _create_mini_model(path)
        return path
    return MODEL_PATH


@pytest.fixture(scope="module")
def compiled_model(model_path):
    """Compile and load the model on Neuron."""
    from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
    from src.modeling_deepseek import (
        DeepseekV3InferenceConfig,
        NeuronDeepseekV3ForCausalLM,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
    )

    inf_config = DeepseekV3InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    # Compile if no existing artifacts
    compiled_path = COMPILED_PATH
    neff_path = os.path.join(compiled_path, "model.pt")
    if not os.path.exists(neff_path):
        print(f"Compiling to {compiled_path}...")
        model = NeuronDeepseekV3ForCausalLM(model_path, inf_config)
        model.compile(compiled_path)
        del model
        gc.collect()

    # Load
    print(f"Loading from {compiled_path}...")
    model = NeuronDeepseekV3ForCausalLM(compiled_path)
    model.load(compiled_path)
    return model


@pytest.fixture(scope="module")
def tokenizer(model_path):
    """Load tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def generation_config(tokenizer):
    """Create generation config."""
    from transformers import GenerationConfig
    return GenerationConfig(
        do_sample=True,
        top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def _generate(model, tokenizer, generation_config, prompt, max_new_tokens=20):
    """Generate text using the NXDI model."""
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

    inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    gen_model = HuggingFaceGenerationAdapter(model)
    outputs = gen_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
    )
    return outputs[0].tolist(), tokenizer.decode(outputs[0], skip_special_tokens=True)


def _is_repetitive(text, max_repeat=5):
    """Check for excessive word repetition."""
    words = text.split()
    if len(words) < max_repeat:
        return False
    for i in range(len(words) - max_repeat + 1):
        if len(set(words[i : i + max_repeat])) == 1:
            return True
    return False


# ── Smoke Tests ─────────────────────────────────────────────────────────

@requires_compiled_model
def test_model_loads(compiled_model):
    """Model compiles and loads successfully."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "neuron_config")
    print("  Model loaded successfully")


@requires_compiled_model
def test_model_generates(compiled_model, tokenizer, generation_config):
    """Model generates at least 5 tokens."""
    tokens, text = _generate(compiled_model, tokenizer, generation_config,
                             "Hello, I am a language model", max_new_tokens=20)
    input_len = len(tokenizer.encode("Hello, I am a language model"))
    new_tokens = len(tokens) - input_len
    assert new_tokens >= 5, f"Expected >= 5 new tokens, got {new_tokens}"
    print(f"  Generated {new_tokens} tokens: {text[:100]}...")


# ── Accuracy Tests ──────────────────────────────────────────────────────

@requires_compiled_model
def test_output_coherence(compiled_model, tokenizer, generation_config):
    """Output should contain multiple words and not be excessively repetitive."""
    _, text = _generate(compiled_model, tokenizer, generation_config,
                        "The capital of France is", max_new_tokens=30)
    generated = text[len("The capital of France is"):].strip()
    words = generated.split()
    assert len(words) >= 3, f"Expected >= 3 words, got {len(words)}: '{generated}'"
    assert not _is_repetitive(generated), f"Output is excessively repetitive: '{generated}'"
    print(f"  Output coherent ({len(words)} words): {generated[:80]}...")


@requires_compiled_model
def test_top_token_valid(compiled_model, tokenizer, generation_config):
    """First generated token should be a valid decodable token."""
    tokens, _ = _generate(compiled_model, tokenizer, generation_config,
                          "Hello!", max_new_tokens=1)
    input_len = len(tokenizer.encode("Hello!"))
    first_new = tokens[input_len]
    assert 0 <= first_new < tokenizer.vocab_size, f"Token {first_new} out of vocab range"
    decoded = tokenizer.decode([first_new])
    assert len(decoded) > 0, f"Token {first_new} decoded to empty string"
    print(f"  First token: {first_new} -> '{decoded}'")


@requires_compiled_model
def test_first_token_matches_hf(compiled_model, tokenizer, generation_config, model_path):
    """First predicted token should match HF reference (CPU, FP32) for mini model."""
    if not USE_MINI_MODEL:
        pytest.skip("First-token HF comparison only for mini model (671B too large for CPU)")

    from transformers import AutoModelForCausalLM

    prompt = "The capital of France is"
    # HF reference
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        hf_out = hf_model(inputs.input_ids, use_cache=False)
    hf_top1 = hf_out.logits[0, -1, :].argmax().item()
    hf_top5 = hf_out.logits[0, -1, :].topk(5).indices.tolist()
    del hf_model
    gc.collect()

    # NXDI
    tokens, _ = _generate(compiled_model, tokenizer, generation_config, prompt, max_new_tokens=1)
    input_len = inputs.input_ids.shape[1]
    nxdi_first = tokens[input_len]

    print(f"  HF top-1: {hf_top1} ({tokenizer.decode([hf_top1])})")
    print(f"  NXDI first: {nxdi_first} ({tokenizer.decode([nxdi_first])})")

    if nxdi_first == hf_top1:
        print("  EXACT MATCH")
    elif nxdi_first in hf_top5:
        print(f"  In HF top-5 (position {hf_top5.index(nxdi_first) + 1})")
    else:
        pytest.fail(f"NXDI token {nxdi_first} not in HF top-5 {hf_top5}")


# ── Performance Tests ───────────────────────────────────────────────────

@requires_compiled_model
def test_performance_ttft(compiled_model, tokenizer, generation_config):
    """Time to first token should be within threshold."""
    prompt = "Hello, I am a language model"

    # Warmup
    _generate(compiled_model, tokenizer, generation_config, prompt, max_new_tokens=1)

    # Measure
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _generate(compiled_model, tokenizer, generation_config, prompt, max_new_tokens=1)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    print(f"  TTFT: {avg_ms:.1f} ms (threshold: {TTFT_THRESHOLD_MS} ms)")
    assert avg_ms < TTFT_THRESHOLD_MS, f"TTFT {avg_ms:.1f}ms > threshold {TTFT_THRESHOLD_MS}ms"


@requires_compiled_model
def test_performance_throughput(compiled_model, tokenizer, generation_config):
    """Throughput should meet minimum threshold."""
    prompt = "Once upon a time"
    num_new_tokens = 20

    # Warmup
    _generate(compiled_model, tokenizer, generation_config, prompt, max_new_tokens=5)

    # Measure
    t0 = time.perf_counter()
    tokens, _ = _generate(compiled_model, tokenizer, generation_config, prompt, max_new_tokens=num_new_tokens)
    elapsed = time.perf_counter() - t0

    input_len = len(tokenizer.encode(prompt))
    actual_new = len(tokens) - input_len
    throughput = actual_new / elapsed if elapsed > 0 else 0

    print(f"  Throughput: {throughput:.1f} tok/s ({actual_new} tokens in {elapsed:.2f}s)")
    print(f"  Threshold: {THROUGHPUT_THRESHOLD} tok/s")
    assert throughput > THROUGHPUT_THRESHOLD, \
        f"Throughput {throughput:.1f} tok/s < threshold {THROUGHPUT_THRESHOLD}"


# ── Standalone runner ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek V3 Integration Tests")
    print("=" * 60)

    if USE_MINI_MODEL:
        print(f"\nSKIPPED: {MINI_MODEL_SKIP_REASON}")
        print("\nTo run integration tests, provide the full 671B model path:")
        print("  DEEPSEEK_MODEL_PATH=/path/to/DeepSeek-V3-0324-FP8 \\")
        print("  DEEPSEEK_COMPILED_PATH=/scratch/deepseek_v3_traced \\")
        print("  DEEPSEEK_TP_DEGREE=64 \\")
        print("  python -m pytest test/integration/test_model.py --capture=tee-sys")
        sys.exit(0)

    # Setup
    from transformers import AutoTokenizer, GenerationConfig as GenConfig
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gen_cfg = GenConfig(do_sample=True, top_k=1,
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

    # Build model
    from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
    from src.modeling_deepseek import (
        DeepseekV3InferenceConfig, NeuronDeepseekV3ForCausalLM,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    nc = MoENeuronConfig(
        tp_degree=TP_DEGREE, batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
        seq_len=SEQ_LEN, torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False, flash_decoding_enabled=False, logical_nc_config=2,
    )
    ic = DeepseekV3InferenceConfig(nc, load_config=load_pretrained_config(MODEL_PATH))

    cp = COMPILED_PATH
    if not os.path.exists(os.path.join(cp, "model.pt")):
        print(f"Compiling to {cp}...")
        m = NeuronDeepseekV3ForCausalLM(MODEL_PATH, ic)
        m.compile(cp)
        del m; gc.collect()

    print(f"Loading from {cp}...")
    model = NeuronDeepseekV3ForCausalLM(cp)
    model.load(cp)

    tests = [
        ("model_loads", lambda: test_model_loads(model)),
        ("model_generates", lambda: test_model_generates(model, tok, gen_cfg)),
        ("output_coherence", lambda: test_output_coherence(model, tok, gen_cfg)),
        ("top_token_valid", lambda: test_top_token_valid(model, tok, gen_cfg)),
        ("first_token_matches_hf", lambda: test_first_token_matches_hf(model, tok, gen_cfg, MODEL_PATH)),
        ("performance_ttft", lambda: test_performance_ttft(model, tok, gen_cfg)),
        ("performance_throughput", lambda: test_performance_throughput(model, tok, gen_cfg)),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed")
    print(f"{'=' * 60}")
