#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone text-only smoke test for Qwen3.5-2B on Neuron.

Compiles the text backbone, runs a short generation, and prints TTFT + TPOT.
Use this before running the pytest suite to catch compile errors interactively.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python contrib/models/Qwen3.5-2B/test/integration/run_text_smoke.py \\
        --model-path /mnt/nvme/models/Qwen3.5-2B \\
        --tp 8 --seq-len 512 --max-new-tokens 32
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


def build_config(model_path: str, tp: int, seq_len: int):
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig

    neuron_config = NeuronConfig(
        tp_degree=tp,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )

    with open(os.path.join(model_path, "config.json")) as f:
        full = json.load(f)
    text_cfg = full.get("text_config", full)

    cfg = dict(text_cfg)
    cfg["pad_token_id"] = text_cfg.get("eos_token_id", 248044)
    if "rope_parameters" in text_cfg:
        rp = text_cfg["rope_parameters"]
        cfg["rope_theta"] = rp.get("rope_theta", 10000000)
        cfg["partial_rotary_factor"] = rp.get("partial_rotary_factor", 0.25)
        cfg["mrope_section"] = rp.get("mrope_section", [11, 11, 10])
    cfg.setdefault("tie_word_embeddings", text_cfg.get("tie_word_embeddings", True))

    return Qwen35InferenceConfig(
        neuron_config=neuron_config,
        use_hybrid_cache_manager=False,
        **cfg,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/nvme/models/Qwen3.5-2B")
    ap.add_argument("--compiled-path", default="/tmp/qwen35_2b_traced")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--skip-compile", action="store_true",
                    help="Assume artifacts already exist at compiled-path")
    args = ap.parse_args()

    from src.modeling_qwen35 import NeuronQwen35ForCausalLM
    from transformers import AutoTokenizer, GenerationConfig
    import transformers
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    inf_config = build_config(args.model_path, args.tp, args.seq_len)

    neff = os.path.join(args.compiled_path, "model.pt")
    if not args.skip_compile and not os.path.exists(neff):
        print(f"[compile] → {args.compiled_path}")
        m = NeuronQwen35ForCausalLM(args.model_path, inf_config)
        t0 = time.perf_counter()
        m.compile(args.compiled_path)
        print(f"[compile] done in {(time.perf_counter()-t0):.1f} s")
        del m
        gc.collect()

    print(f"[load] ← {args.compiled_path}")
    m = NeuronQwen35ForCausalLM(args.compiled_path)
    m.load(args.compiled_path)

    tok = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gen_cfg = GenerationConfig(
        do_sample=True, top_k=1,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    gen_cfg.transformers_version = transformers.__version__
    gen_model = HuggingFaceGenerationAdapter(m)
    gen_model.generation_config.transformers_version = transformers.__version__

    enc = tok(args.prompt, return_tensors="pt")

    # Warmup (first call may include tracing overhead)
    _ = gen_model.generate(enc.input_ids, generation_config=gen_cfg, max_new_tokens=1)

    # TTFT
    t0 = time.perf_counter()
    _ = gen_model.generate(enc.input_ids, generation_config=gen_cfg, max_new_tokens=1)
    ttft = (time.perf_counter() - t0) * 1000

    # Full run
    t1 = time.perf_counter()
    out = gen_model.generate(
        enc.input_ids,
        generation_config=gen_cfg,
        max_new_tokens=args.max_new_tokens,
    )
    total_ms = (time.perf_counter() - t1) * 1000

    n_new = out.shape[-1] - enc.input_ids.shape[-1]
    n_decode = max(1, n_new - 1)
    tpot = (total_ms - ttft) / n_decode

    text = tok.decode(out[0], skip_special_tokens=True)

    print("=" * 72)
    print(f"prompt : {args.prompt!r}")
    print(f"output : {text!r}")
    print(f"n_new  : {n_new}")
    print(f"TTFT   : {ttft:.1f} ms")
    print(f"TPOT   : {tpot:.1f} ms  ({1000.0/max(tpot,1e-6):.2f} tok/s)")
    print("=" * 72)


if __name__ == "__main__":
    main()
