#!/usr/bin/env python3
"""Per-module Neuron micro-benchmark for Qwen2.5-Omni-7B.

Loads Thinker, Talker, DiT and BigVGAN once via the same paths used by
``generate_qwen25_omni_speech.py`` and times each module in isolation
with fixed input sizes and fixed ``max_new_tokens``. Sampling is left
on; randomness is absorbed by reporting the median over N runs and
locking generation length per module.

Output JSON schema is identical to ``bench_modules_gpu.py`` so the two
files can be compared by ``compare_bench.py``.

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd contrib/models/Qwen2.5-Omni-7B
  NEURON_RT_VISIBLE_CORES=0-7 \\
      python examples/bench_modules_neuron.py \\
      --num-runs 5 --json bench_neuron.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

# Reuse the speech-pipeline entrypoint for loaders + bootstrap. Importing it
# also primes NEURON_RT_VISIBLE_CORES (default 0-7), the contrib `src/`
# path, and the hf_adapter shim.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import generate_qwen25_omni_speech as gen  # noqa: E402

# After the gen import, sys.path contains contrib/.../src; pick up modeling
# helpers we need for direct DiT / BigVGAN access.
from modeling_qwen25_omni_token2wav import (  # noqa: E402
    NeuronQwen25OmniToken2WavWithNeuronDiT,  # type: ignore  # noqa: F401
)


# Default fixed-size workloads. Match bench_modules_gpu.py.
DEFAULTS = {
    "thinker_max_new_tokens": 32,
    "talker_max_new_tokens": 200,
    "dit_mel_len": 1024,
    "dit_num_steps": 10,
    "bigvgan_mel_len": 1024,
}


def _stats(name, vals_s, extra=None):
    out = {
        "runs": len(vals_s),
        "wall_per_run_s": [round(v, 4) for v in vals_s],
        "median_s": round(statistics.median(vals_s), 4),
        "mean_s": round(statistics.mean(vals_s), 4),
        "min_s": round(min(vals_s), 4),
        "max_s": round(max(vals_s), 4),
    }
    if extra:
        out.update(extra)
    return out


def bench_thinker(thinker_adapter, tokenizer, runs, max_new_tokens):
    prompt = "Say hello and briefly introduce yourself in two sentences."
    sys_p = gen.DEFAULT_SYSTEM
    chat = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt")

    # Warmup
    _ = thinker_adapter.generate(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        eos_token_id=[tokenizer.eos_token_id, 151645],
    )

    wall = []
    for _ in range(runs):
        t0 = time.time()
        out = thinker_adapter.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            eos_token_id=[tokenizer.eos_token_id, 151645],
        )
        wall.append(time.time() - t0)
    n_gen = out.shape[1] - enc["input_ids"].shape[1]
    return _stats(
        "thinker", wall,
        extra={
            "max_new_tokens": max_new_tokens,
            "generated_tokens_last_run": int(n_gen),
            "tpot_ms": round(1000.0 * statistics.median(wall) / max(n_gen, 1), 3),
        },
    )


def bench_talker(
    prep_cache, thinker_adapter, tokenizer,
    talker_model, talker_adapter, talker_cfg,
    runs, max_new_tokens, speaker,
):
    """Bench Talker with real Thinker hidden state (option A).

    Reuses the speech pipeline's helpers verbatim:
      run_thinker -> {all_ids, prompt_len, ...}
      extract_hidden_states(thinker_neuron_model, prep_cache, thinker_result)
      prepare_talker_input(prep_cache, outputs, full_ids, prompt_len, speaker)
    """
    prompt = "Say hello and briefly introduce yourself in two sentences."
    thinker_result = gen.run_thinker(
        thinker_adapter, tokenizer, prompt, gen.DEFAULT_SYSTEM,
    )
    outputs, full_ids, prompt_len, _ = gen.extract_hidden_states(
        thinker_adapter.neuron_model, prep_cache, thinker_result,
    )
    talker_input = gen.prepare_talker_input(
        prep_cache, outputs, full_ids, prompt_len, speaker,
    )

    codec_bos = talker_cfg.tts_codec_start_token_id
    codec_eos = talker_cfg.tts_codec_end_token_id
    codec_pad = talker_cfg.tts_codec_pad_token_id
    codec_mask = talker_cfg.tts_codec_mask_token_id
    context_len = talker_input["context_len"]

    talker_input_ids = torch.cat([
        torch.full((1, context_len - 2), codec_mask, dtype=torch.long),
        torch.tensor([[codec_pad]], dtype=torch.long),
        torch.tensor([[codec_bos]], dtype=torch.long),
    ], dim=1)
    attn = torch.ones_like(talker_input_ids, dtype=torch.long)

    def _set_ve():
        ve = talker_input["projected_context"].to(torch.bfloat16)
        vm = torch.ones(1, context_len, 1, dtype=torch.int32)
        reply = talker_input["projected_reply"].to(torch.bfloat16)
        talker_model.set_vision_embeddings(ve, vm, thinker_reply_embeds=reply)

    # Warmup
    _set_ve()
    _ = talker_adapter.generate(
        input_ids=talker_input_ids, attention_mask=attn,
        max_new_tokens=max_new_tokens,
        eos_token_id=[codec_eos, codec_pad], suppress_tokens=[codec_bos],
        do_sample=True, temperature=0.9, top_k=40, top_p=0.8,
        repetition_penalty=1.05,
    )

    wall = []
    for _ in range(runs):
        _set_ve()
        t0 = time.time()
        out = talker_adapter.generate(
            input_ids=talker_input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens,
            eos_token_id=[codec_eos, codec_pad], suppress_tokens=[codec_bos],
            do_sample=True, temperature=0.9, top_k=40, top_p=0.8,
            repetition_penalty=1.05,
        )
        wall.append(time.time() - t0)
    n_gen = out.shape[1] - context_len
    return _stats(
        "talker", wall,
        extra={
            "max_new_tokens": max_new_tokens,
            "generated_tokens_last_run": int(n_gen),
            "context_len": int(context_len),
            "tpot_ms": round(1000.0 * statistics.median(wall) / max(n_gen, 1), 3),
        },
    )


def bench_dit(t2w, runs, mel_len, num_steps):
    """Bench the DiT NEFF in isolation: random tensors of the right shape,
    fired ``num_steps`` times to model the ODE solver loop."""
    cores = getattr(t2w, "_neuron_dit_cores", None) or {}
    if not cores:
        raise RuntimeError("DiT not loaded; run --compile first.")
    # Pick smallest bucket >= mel_len, falling back to the largest.
    bucket = None
    for b in sorted(cores):
        if b >= mel_len:
            bucket = b
            break
    if bucket is None:
        bucket = max(cores)
    neff = cores[bucket]

    meta_path = os.path.join(t2w._dit_compiled_path, "dit_core_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    bs = int(meta.get("batch_size", 2))
    dim = int(meta.get("dim", 1024))
    head_dim = int(meta.get("head_dim", dim // int(meta.get("num_heads", 16))))

    h = torch.randn(bs, bucket, dim, dtype=torch.float32)
    te = torch.randn(1, dim, dtype=torch.float32)
    cos = torch.randn(bs, bucket, head_dim, dtype=torch.float32)
    sin = torch.randn(bs, bucket, head_dim, dtype=torch.float32)
    m_local = torch.zeros(bs, 1, bucket, bucket, dtype=torch.float32)
    m_back = torch.zeros(bs, 1, bucket, bucket, dtype=torch.float32)
    m_ahead = torch.zeros(bs, 1, bucket, bucket, dtype=torch.float32)

    # Warmup: one call for NEFF dispatch + cache.
    _ = neff(h, te, cos, sin, m_local, m_back, m_ahead)

    wall = []
    for _ in range(runs):
        t0 = time.time()
        for _ in range(num_steps):
            _ = neff(h, te, cos, sin, m_local, m_back, m_ahead)
        wall.append(time.time() - t0)
    return _stats(
        "dit", wall,
        extra={
            "mel_len_requested": int(mel_len),
            "bucket_used": int(bucket),
            "num_steps": int(num_steps),
            "batch_size": bs,
            "dim": dim,
            "per_step_ms": round(
                1000.0 * statistics.median(wall) / num_steps, 3
            ),
        },
    )


def bench_bigvgan(t2w, runs, mel_len):
    cores = getattr(t2w, "_neuron_bigvgan_cores", None) or {}
    if not cores:
        return {
            "skipped": "BigVGAN NEFF not loaded (mel_len > compiled buckets "
                       "→ runtime falls back to CPU). Skipping module bench."
        }
    bucket = None
    for b in sorted(cores):
        if b >= mel_len:
            bucket = b
            break
    if bucket is None:
        bucket = max(cores)
    neff = cores[bucket]

    meta_path = os.path.join(t2w._bigvgan_compiled_path, "bigvgan_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    mel_dim = int(meta.get("mel_dim", 80))

    mel = torch.randn(1, mel_dim, bucket, dtype=torch.float32)
    _ = neff(mel)  # warmup

    wall = []
    for _ in range(runs):
        t0 = time.time()
        _ = neff(mel)
        wall.append(time.time() - t0)
    return _stats(
        "bigvgan", wall,
        extra={
            "mel_len_requested": int(mel_len),
            "bucket_used": int(bucket),
            "mel_dim": mel_dim,
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-runs", type=int, default=5)
    p.add_argument("--json", default="bench_neuron.json")
    p.add_argument("--model-path", default=gen.MODEL_PATH)
    p.add_argument("--compiled-path", default=gen.COMPILED_PATH)
    p.add_argument("--speaker", default=gen.DEFAULT_SPEAKER,
                   choices=["Ethan", "Chelsie"])
    p.add_argument("--thinker-max-new-tokens", type=int,
                   default=DEFAULTS["thinker_max_new_tokens"])
    p.add_argument("--talker-max-new-tokens", type=int,
                   default=DEFAULTS["talker_max_new_tokens"])
    p.add_argument("--dit-mel-len", type=int,
                   default=DEFAULTS["dit_mel_len"])
    p.add_argument("--dit-num-steps", type=int,
                   default=DEFAULTS["dit_num_steps"])
    p.add_argument("--bigvgan-mel-len", type=int,
                   default=DEFAULTS["bigvgan_mel_len"])
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["thinker", "talker", "dit", "bigvgan"])
    args = p.parse_args()

    if not gen._check_compiled(args.compiled_path):
        sys.exit(1)

    print("=" * 60)
    print("Qwen2.5-Omni Per-Module Bench (Neuron)")
    print("=" * 60)
    print(f"  Model:    {args.model_path}")
    print(f"  Compiled: {args.compiled_path}")
    print(f"  Runs:     {args.num_runs}")
    print(f"  Visible:  {os.environ.get('NEURON_RT_VISIBLE_CORES', '?')}")

    print("\n--- Loading models ---")
    thinker_adapter, tokenizer, _ = gen.load_thinker(
        args.model_path, args.compiled_path,
    )
    hf_model, _ = gen.load_hf_cpu(args.model_path)
    talker_model, talker_adapter, talker_cfg, _ = gen.load_talker(
        args.model_path, args.compiled_path,
    )
    t2w, t2w_cfg, _ = gen.load_token2wav(args.model_path, args.compiled_path)

    # Build the talker prep cache once and free the 17GB HF CPU model.
    prep_cache = gen.TalkerPrepCache.build(args.model_path, hf_model)
    del hf_model
    import gc as _gc
    _gc.collect()

    results = {
        "platform": "neuron",
        "device": f"trn2 / TP={gen.TP_DEGREE}",
        "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES", "?"),
        "dtype": "bfloat16",
        "modules": {},
    }

    if "thinker" not in args.skip:
        print("\n--- Bench Thinker ---")
        results["modules"]["thinker"] = bench_thinker(
            thinker_adapter, tokenizer,
            args.num_runs, args.thinker_max_new_tokens,
        )
        print(f"  median={results['modules']['thinker']['median_s']:.3f}s "
              f"tpot={results['modules']['thinker']['tpot_ms']:.2f}ms")

    if "talker" not in args.skip:
        print("\n--- Bench Talker ---")
        results["modules"]["talker"] = bench_talker(
            prep_cache, thinker_adapter, tokenizer,
            talker_model, talker_adapter, talker_cfg,
            args.num_runs, args.talker_max_new_tokens, args.speaker,
        )
        print(f"  median={results['modules']['talker']['median_s']:.3f}s "
              f"tpot={results['modules']['talker']['tpot_ms']:.2f}ms")

    if "dit" not in args.skip:
        print("\n--- Bench DiT ---")
        results["modules"]["dit"] = bench_dit(
            t2w, args.num_runs, args.dit_mel_len, args.dit_num_steps,
        )
        print(f"  median={results['modules']['dit']['median_s']:.3f}s "
              f"per_step={results['modules']['dit']['per_step_ms']:.2f}ms "
              f"bucket={results['modules']['dit']['bucket_used']}")

    if "bigvgan" not in args.skip:
        print("\n--- Bench BigVGAN ---")
        results["modules"]["bigvgan"] = bench_bigvgan(
            t2w, args.num_runs, args.bigvgan_mel_len,
        )
        bv = results["modules"]["bigvgan"]
        if "skipped" in bv:
            print(f"  SKIP: {bv['skipped']}")
        else:
            print(f"  median={bv['median_s']:.3f}s bucket={bv['bucket_used']}")

    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
