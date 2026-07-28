#!/usr/bin/env python3
"""
Integration test for MiniMax-M3 (text-only backbone) on Neuron.

This test compiles a small subset of the model (via NeuronConfig
`num_hidden_layers` override) so that the test can run within reasonable
time/memory bounds without downloading all ~854GB of weights. To run the test
against the full model, point `MODEL_PATH` at the downloaded checkpoint and
remove the `num_hidden_layers` / `num_local_experts` overrides below.

Performance metrics measured:
  * TTFT (Time To First Token, ms): prefill latency on the input prompt.
  * ITL  (Inter-Token Latency, ms/token): average decode time once the cache
    is warmed up.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Make `src/` importable when running this file directly.
_TEST_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _TEST_DIR.parent.parent
sys.path.insert(0, str(_MODEL_DIR / "src"))

from modeling_minimax_m3 import (  # noqa: E402
    NeuronMiniMaxM3ForCausalLM,
    MiniMaxM3InferenceConfig,
)


# -----------------------------------------------------------------------------
# Paths / config — override via env when needed.
# -----------------------------------------------------------------------------
MODEL_PATH = os.environ.get("M3_MODEL_PATH", "/home/ubuntu/models/MiniMax-M3/")
COMPILED_MODEL_PATH = os.environ.get(
    "M3_COMPILED_PATH", "/home/ubuntu/neuron_models/MiniMax-M3/"
)

# Compile-time knobs — set conservatively so the test can run end-to-end on a
# single trn2.48xlarge without downloading the full 854GB checkpoint. For an
# accuracy-validated run against the released weights, set TP_DEGREE >= 16 and
# remove the layer/expert overrides.
TP_DEGREE = int(os.environ.get("M3_TP_DEGREE", "32"))
BATCH_SIZE = int(os.environ.get("M3_BATCH_SIZE", "1"))
SEQ_LEN = int(os.environ.get("M3_SEQ_LEN", "512"))
NUM_LAYERS_OVERRIDE = int(os.environ.get("M3_NUM_LAYERS", "0"))  # 0 = use full
NUM_EXPERTS_OVERRIDE = int(os.environ.get("M3_NUM_EXPERTS", "0"))  # 0 = full

# Performance thresholds (informational only; the test reports values but does
# not fail on perf — too dependent on the truncation knobs above).
TTFT_THRESHOLD_MS = float(os.environ.get("M3_TTFT_THRESHOLD_MS", "2000"))
ITL_THRESHOLD_MS = float(os.environ.get("M3_ITL_THRESHOLD_MS", "500"))


def _build_inference_config():
    """Build the M3 inference config with optional layer/expert overrides."""
    neuron_config_kwargs = {
        "tp_degree": TP_DEGREE,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "max_context_length": SEQ_LEN,
        "torch_dtype": torch.bfloat16,
    }
    neuron_config = NeuronConfig(**neuron_config_kwargs)

    config = MiniMaxM3InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    # Optional smoke-test overrides for compile time / memory.
    if NUM_LAYERS_OVERRIDE > 0:
        config.num_hidden_layers = NUM_LAYERS_OVERRIDE
        # Trim the per-layer dense/MoE schedule to the new layer count.
        config.moe_layer_freq = list(config.moe_layer_freq)[:NUM_LAYERS_OVERRIDE]
    if NUM_EXPERTS_OVERRIDE > 0:
        config.num_local_experts = NUM_EXPERTS_OVERRIDE

    return config, neuron_config


def _load_compiled_neuron_config(compiled_path: str):
    p = Path(compiled_path) / "neuron_config.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return data.get("neuron_config", data)


@pytest.fixture(scope="module")
def compiled_model():
    """Compile (or skip-if-already-compiled) and load the M3 model."""
    compiled_path = Path(COMPILED_MODEL_PATH)

    if not (compiled_path / "model.pt").exists():
        if not Path(MODEL_PATH).exists():
            pytest.skip(f"Model checkpoint not found at {MODEL_PATH}; "
                        "download MiniMaxAI/MiniMax-M3 to that path first.")
        print(f"\nCompiling MiniMax-M3 to {COMPILED_MODEL_PATH}...")
        config, _ = _build_inference_config()
        model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete")

    # Rebuild config (the on-disk neuron_config.json drives any post-compile
    # overrides), then load weights.
    saved = _load_compiled_neuron_config(COMPILED_MODEL_PATH)
    tp_degree = saved.get("tp_degree", TP_DEGREE) if saved else TP_DEGREE
    seq_len = saved.get("seq_len", SEQ_LEN) if saved else SEQ_LEN

    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=saved.get("batch_size", BATCH_SIZE) if saved else BATCH_SIZE,
        seq_len=seq_len,
        max_context_length=saved.get("max_context_length", seq_len) if saved else seq_len,
        torch_dtype=torch.bfloat16,
    )
    config = MiniMaxM3InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )
    if NUM_LAYERS_OVERRIDE > 0:
        config.num_hidden_layers = NUM_LAYERS_OVERRIDE
        config.moe_layer_freq = list(config.moe_layer_freq)[:NUM_LAYERS_OVERRIDE]
    if NUM_EXPERTS_OVERRIDE > 0:
        config.num_local_experts = NUM_EXPERTS_OVERRIDE

    model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _forward(model, input_ids, position_ids=None):
    if position_ids is None:
        position_ids = (
            torch.arange(input_ids.shape[1])
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )
    with torch.no_grad():
        return model(input_ids, position_ids=position_ids)


def _generate(model, input_ids, max_new_tokens: int):
    """Manual greedy decode loop (the contrib test pattern)."""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        outputs = _forward(model, generated)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    return generated


def test_smoke(compiled_model):
    """Verify the model object came back with a valid neuron_config."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    assert hasattr(compiled_model.config, "neuron_config")
    print("Smoke test passed: model loaded successfully")


def test_generate(compiled_model, tokenizer):
    """Generate a short continuation; just check we get tokens back."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    generated = _generate(compiled_model, inputs.input_ids, max_new_tokens=10)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    assert len(text) > len(prompt), "Output should extend the prompt"
    print(f"Generated text: {text!r}")


def test_ttft(compiled_model, tokenizer):
    """Measure prefill latency (TTFT)."""
    prompt = "Hello, how are you today?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids

    # Warmup
    for _ in range(2):
        _forward(compiled_model, input_ids)

    # Measure
    latencies_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        _forward(compiled_model, input_ids)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    avg_ttft_ms = sum(latencies_ms) / len(latencies_ms)
    print(f"\n[TTFT] mean={avg_ttft_ms:.2f} ms over {len(latencies_ms)} runs "
          f"(prompt_tokens={input_ids.shape[1]})")
    print(f"[TTFT] individual: {[f'{x:.2f}' for x in latencies_ms]}")
    # Informational threshold only
    if avg_ttft_ms > TTFT_THRESHOLD_MS:
        print(f"[TTFT] WARNING: above threshold {TTFT_THRESHOLD_MS}ms")


def test_itl(compiled_model, tokenizer):
    """Measure decode latency per token (ITL)."""
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    num_tokens = 20

    # Warmup (generate a couple of tokens to warm decode caches)
    _ = _generate(compiled_model, input_ids, max_new_tokens=3)

    # Measure: prefill once, then time `num_tokens` decode steps.
    generated = input_ids.clone()
    _forward(compiled_model, generated)  # prefill

    per_token_ms = []
    for _ in range(num_tokens):
        t0 = time.perf_counter()
        outputs = _forward(compiled_model, generated)
        per_token_ms.append((time.perf_counter() - t0) * 1000)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

    avg_itl_ms = sum(per_token_ms) / len(per_token_ms)
    throughput = 1000.0 / avg_itl_ms if avg_itl_ms > 0 else 0.0
    print(f"\n[ITL]  mean={avg_itl_ms:.2f} ms/token  throughput={throughput:.2f} tok/s "
          f"over {num_tokens} decode steps")
    print(f"[ITL]  first_5={[f'{x:.2f}' for x in per_token_ms[:5]]} "
          f"last_5={[f'{x:.2f}' for x in per_token_ms[-5:]]}")
    if avg_itl_ms > ITL_THRESHOLD_MS:
        print(f"[ITL]  WARNING: above threshold {ITL_THRESHOLD_MS}ms")


if __name__ == "__main__":
    print("=" * 80)
    print("MiniMax-M3 Integration Test")
    print("=" * 80)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        if not Path(MODEL_PATH).exists():
            print(f"ERROR: MODEL_PATH={MODEL_PATH} does not exist.")
            print("Download MiniMaxAI/MiniMax-M3 first, e.g.")
            print("  hf download MiniMaxAI/MiniMax-M3 --local-dir ${MODEL_PATH}")
            sys.exit(1)
        print(f"\n[compile] {MODEL_PATH} -> {COMPILED_MODEL_PATH}")
        cfg, _ = _build_inference_config()
        model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, cfg)
        model.compile(COMPILED_MODEL_PATH)

    print(f"\n[load] {COMPILED_MODEL_PATH}")
    saved = _load_compiled_neuron_config(COMPILED_MODEL_PATH)
    tp_degree = saved.get("tp_degree", TP_DEGREE) if saved else TP_DEGREE
    seq_len = saved.get("seq_len", SEQ_LEN) if saved else SEQ_LEN
    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=saved.get("batch_size", BATCH_SIZE) if saved else BATCH_SIZE,
        seq_len=seq_len,
        max_context_length=saved.get("max_context_length", seq_len) if saved else seq_len,
        torch_dtype=torch.bfloat16,
    )
    cfg = MiniMaxM3InferenceConfig(
        neuron_config, load_config=load_pretrained_config(MODEL_PATH),
    )
    if NUM_LAYERS_OVERRIDE > 0:
        cfg.num_hidden_layers = NUM_LAYERS_OVERRIDE
        cfg.moe_layer_freq = list(cfg.moe_layer_freq)[:NUM_LAYERS_OVERRIDE]
    if NUM_EXPERTS_OVERRIDE > 0:
        cfg.num_local_experts = NUM_EXPERTS_OVERRIDE
    model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, cfg)
    model.load(COMPILED_MODEL_PATH)

    tok = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\n[1/4] smoke test")
    test_smoke(model)

    print("\n[2/4] generate test")
    test_generate(model, tok)

    print("\n[3/4] TTFT test")
    test_ttft(model, tok)

    print("\n[4/4] ITL test")
    test_itl(model, tok)

    print("\n" + "=" * 80)
    print("All tests done.")
    print("=" * 80)
