# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Qwen3.5-35B-A3B on Neuron (Trn2).

Runs full compile + inference against the HuggingFace Qwen/Qwen3.5-35B-A3B weights.
The 2B model shares the hybrid DeltaNet + full-attention architecture of
Qwen3.5-27B (model_type = qwen3_5) at a smaller scale (24 layers, 18 linear
+ 6 full). This test reuses the modeling code (originally contributed for
Qwen3.6-27B in PR #173) at 2B scale.

Env vars:
    QWEN35_MODEL_PATH        HF weights path (required)
    QWEN35_COMPILED_PATH     Where to write NEFF (default /tmp/qwen35_35b_a3b_traced)
    QWEN35_TP_DEGREE         TP degree (default 8 for 2B on Trn2)
    QWEN35_SEQ_LEN           Max seq len (default 512)
    QWEN35_MAX_NEW_TOKENS    Decode budget for latency measurement (default 32)
    TTFT_THRESHOLD_MS        Guard rail for TTFT (default 8000)
    THROUGHPUT_THRESHOLD     Min decode throughput tok/s (default 4.0)
"""

import gc
import json
import os
import sys
import time

import pytest
import torch


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "/mnt/nvme/models/Qwen3.5-35B-A3B")
COMPILED_PATH = os.environ.get("QWEN35_COMPILED_PATH", "/tmp/qwen35_35b_a3b_traced")
TP_DEGREE = int(os.environ.get("QWEN35_TP_DEGREE", "8"))
SEQ_LEN = int(os.environ.get("QWEN35_SEQ_LEN", "512"))
MAX_NEW_TOKENS = int(os.environ.get("QWEN35_MAX_NEW_TOKENS", "32"))
TTFT_THRESHOLD_MS = float(os.environ.get("TTFT_THRESHOLD_MS", "8000"))
THROUGHPUT_THRESHOLD = float(os.environ.get("THROUGHPUT_THRESHOLD", "4.0"))

requires_model_path = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH),
    reason=f"Qwen3.5-35B-A3B weights not found at {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def compiled_model():
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

    neuron_config = NeuronConfig(
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
        save_sharded_checkpoint=True,
    )

    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)
    cfg_dict = dict(text_config)
    cfg_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        cfg_dict["rope_theta"] = text_config["rope_parameters"].get("rope_theta", 10000000)
        cfg_dict["partial_rotary_factor"] = text_config["rope_parameters"].get(
            "partial_rotary_factor", 0.25
        )
        cfg_dict["mrope_section"] = text_config["rope_parameters"].get(
            "mrope_section", [11, 11, 10]
        )
    cfg_dict.setdefault("tie_word_embeddings", text_config.get("tie_word_embeddings", True))

    inf_config = Qwen35InferenceConfig(
        neuron_config=neuron_config,
        use_hybrid_cache_manager=False,
        **cfg_dict,
    )

    neff = os.path.join(COMPILED_PATH, "model.pt")
    if not os.path.exists(neff):
        print(f"[qwen35_35b_a3b] compiling → {COMPILED_PATH}")
        m = NeuronQwen35ForCausalLM(MODEL_PATH, inf_config)
        m.compile(COMPILED_PATH)
        del m
        gc.collect()

    print(f"[qwen35_35b_a3b] loading ← {COMPILED_PATH}")
    m = NeuronQwen35ForCausalLM(COMPILED_PATH)
    m.load(COMPILED_PATH)
    return m


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _generate_and_time(model, tokenizer, prompt, max_new_tokens):
    """Return (token_ids, decoded_text, ttft_ms, tpot_ms)."""
    from transformers import GenerationConfig
    import transformers
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    gen_cfg = GenerationConfig(
        do_sample=True,
        top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_cfg.transformers_version = transformers.__version__

    enc = tokenizer(prompt, padding=True, return_tensors="pt")
    gen_model = HuggingFaceGenerationAdapter(model)
    gen_model.generation_config.transformers_version = transformers.__version__

    t0 = time.perf_counter()
    out_first = gen_model.generate(
        enc.input_ids,
        generation_config=gen_cfg,
        attention_mask=enc.attention_mask,
        max_new_tokens=1,
    )
    ttft_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    out_all = gen_model.generate(
        enc.input_ids,
        generation_config=gen_cfg,
        attention_mask=enc.attention_mask,
        max_new_tokens=max_new_tokens,
    )
    elapsed_all = time.perf_counter() - t1

    generated = out_all[0].tolist()[enc.input_ids.shape[-1]:]
    # After TTFT, the remaining (n-1) tokens run in decode. TPOT is the
    # per-decode-step latency.
    n_new = max(1, len(generated))
    n_decode = max(1, n_new - 1)
    ttft_end2end_ms = (t1 - t0)  # approximate; use one-token pass above
    tpot_ms = (elapsed_all * 1000 - ttft_ms) / n_decode

    text = tokenizer.decode(out_all[0], skip_special_tokens=True)
    return out_all[0].tolist(), text, ttft_ms, tpot_ms


@requires_model_path
def test_generation_and_latency(compiled_model, tokenizer):
    """Compile, load, and generate. Report TTFT + TPOT + text."""
    prompt = "The capital of France is"
    tokens, text, ttft, tpot = _generate_and_time(
        compiled_model, tokenizer, prompt, MAX_NEW_TOKENS
    )
    print(f"\n[qwen35_35b_a3b] prompt   : {prompt!r}")
    print(f"[qwen35_35b_a3b] output   : {text!r}")
    print(f"[qwen35_35b_a3b] TTFT     : {ttft:.1f} ms")
    print(f"[qwen35_35b_a3b] TPOT     : {tpot:.1f} ms  ({1000.0/max(tpot,1e-6):.2f} tok/s)")

    assert text and len(text) > len(prompt), "Model produced no continuation"
    assert ttft < TTFT_THRESHOLD_MS, (
        f"TTFT {ttft:.1f} ms > threshold {TTFT_THRESHOLD_MS} ms"
    )
    tokens_per_sec = 1000.0 / max(tpot, 1e-6)
    assert tokens_per_sec > THROUGHPUT_THRESHOLD, (
        f"decode throughput {tokens_per_sec:.2f} tok/s < {THROUGHPUT_THRESHOLD}"
    )


@requires_model_path
def test_accuracy_vs_hf(compiled_model, tokenizer):
    """Greedy top-1 match against HuggingFace on CPU for short prompts.

    HF may be prohibitively slow at 2B on CPU, so we bail out if it takes
    too long — this is a coarse coherence check, not a strict harness.
    """
    prompts = [
        "The capital of France is",
        "In one word, the color of the sky is",
    ]

    hf_ok = False
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        hf_model.eval()
        hf_ok = True
    except Exception as e:
        print(f"[qwen35_35b_a3b] skipping HF cross-check: {e}")
        pytest.skip("HuggingFace model load failed on CPU")

    from transformers import GenerationConfig
    import transformers
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    n_match = 0
    n_total = 0
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            hf_out = hf_model.generate(
                enc.input_ids, do_sample=False, max_new_tokens=16
            )[0].tolist()

        gen_cfg = GenerationConfig(
            do_sample=True, top_k=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_cfg.transformers_version = transformers.__version__
        gen_model = HuggingFaceGenerationAdapter(compiled_model)
        gen_model.generation_config.transformers_version = transformers.__version__
        nxd_out = gen_model.generate(
            enc.input_ids,
            generation_config=gen_cfg,
            max_new_tokens=16,
        )[0].tolist()

        L = min(len(hf_out), len(nxd_out))
        for a, b in zip(hf_out[:L], nxd_out[:L]):
            n_total += 1
            if a == b:
                n_match += 1

    print(f"[qwen35_35b_a3b] HF↔Neuron token match: {n_match}/{n_total}")
    assert n_match / max(n_total, 1) >= 0.75, (
        f"HF↔Neuron accuracy too low: {n_match}/{n_total}"
    )
