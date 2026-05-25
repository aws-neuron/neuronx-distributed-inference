#!/usr/bin/env python3
"""Per-module GPU micro-benchmark for Qwen2.5-Omni-7B.

Mirrors ``bench_modules_neuron.py`` so the two outputs can be compared.

Loads the HF Qwen2.5-Omni model on a single GPU and times each
sub-module (thinker / talker / DiT / BigVGAN) independently with the
same fixed input sizes the Neuron bench uses. Sampling stays on; we
absorb run-to-run variance by reporting the median over N runs and
locking generation length per module.

Usage (on a P5/A100/H100 host):
  python -m venv ~/qwen-omni-gpu && source ~/qwen-omni-gpu/bin/activate
  pip install torch>=2.4 transformers>=4.46 qwen-omni-utils accelerate \\
              soundfile pynvml
  cd contrib/models/Qwen2.5-Omni-7B
  python examples/bench_modules_gpu.py \\
      --num-runs 5 --json bench_gpu.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import test_gpu_baseline_bench as gpu  # noqa: E402

DEFAULT_PROMPT = (
    "Say hello and briefly introduce yourself in two sentences."
)
DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
DEFAULT_SPEAKER = "Ethan"

DEFAULTS = {
    "thinker_max_new_tokens": 32,
    "talker_max_new_tokens": 200,
    "dit_mel_len": 1024,
    "dit_num_steps": 10,
    "bigvgan_mel_len": 128,
}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(vals_s, extra=None):
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


def _build_inputs(processor, prompt, system_prompt, device):
    return gpu.build_text_inputs(processor, prompt, system_prompt, device)


def bench_thinker(model, processor, runs, max_new_tokens):
    device = next(model.parameters()).device
    inputs = _build_inputs(processor, DEFAULT_PROMPT, DEFAULT_SYSTEM, str(device))
    thinker = model.thinker

    # Warmup
    with torch.inference_mode():
        _ = thinker.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    _sync()

    wall = []
    for _ in range(runs):
        _sync()
        t0 = time.time()
        with torch.inference_mode():
            out = thinker.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        _sync()
        wall.append(time.time() - t0)
    n_gen = int(out.shape[1] - inputs["input_ids"].shape[1])
    return _stats(
        wall,
        extra={
            "max_new_tokens": max_new_tokens,
            "generated_tokens_last_run": n_gen,
            "tpot_ms": round(1000.0 * statistics.median(wall) / max(n_gen, 1), 3),
        },
    )


def _prepare_talker_inputs(model, processor, speaker):
    """Mirror HF's Qwen2_5OmniForConditionalGeneration.generate talker prep.

    Returns the kwargs needed by ``model.talker.generate(...)`` plus the
    starting ``context_len`` so we can subtract it to count generated codec
    tokens.
    """
    device = next(model.parameters()).device
    inputs = _build_inputs(processor, DEFAULT_PROMPT, DEFAULT_SYSTEM, str(device))
    thinker = model.thinker
    talker = model.talker

    with torch.inference_mode():
        thinker_out = thinker.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            return_dict_in_generate=True, output_hidden_states=True,
        )
    full_ids = thinker_out.sequences
    prompt_len = inputs["input_ids"].shape[1]

    speaker_params = model.speaker_map[speaker]
    embeds_to_talker = thinker_out.hidden_states[0][0].clone().to(device)
    input_ids_t = inputs["input_ids"]
    if "input_features" in inputs:
        audio_ids_mask = (
            input_ids_t == model.config.thinker_config.audio_token_index
        )
        m = audio_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
        z = torch.zeros(
            [int(audio_ids_mask.sum()), embeds_to_talker.shape[-1]],
            dtype=embeds_to_talker.dtype, device=device,
        )
        embeds_to_talker.masked_scatter_(m, z)

    processed_thinker_hidden = (
        ((embeds_to_talker,) + thinker_out.hidden_states[0][1:],)
        + thinker_out.hidden_states[1:]
    )
    thinker_token_embeds = [h[0].to(device) for h in processed_thinker_hidden]
    thinker_hidden_states = [h[-1].to(device) for h in processed_thinker_hidden]

    talker_text_bos_token = int(speaker_params["bos_token"])
    talker_input_ids = torch.cat(
        [
            torch.full_like(input_ids_t, fill_value=int(talker.codec_mask_token)),
            torch.tensor([[int(talker.codec_pad_token)]],
                         dtype=torch.long, device=device),
            torch.tensor([[int(talker.codec_bos_token)]],
                         dtype=torch.long, device=device),
        ],
        dim=1,
    )

    thinker_embed_tokens = thinker.get_input_embeddings()
    thinker_reply_part = (
        torch.cat(thinker_hidden_states[1:], dim=1)
        + torch.cat(thinker_token_embeds[1:], dim=1)
    )
    talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
    bos_embed = thinker_embed_tokens(
        torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=device)
    )
    talker_inputs_embeds = torch.cat(
        [talker_inputs_embeds, bos_embed, thinker_reply_part[:, :1, :]], dim=1
    )

    eos_embed = thinker_embed_tokens(
        torch.tensor([[int(talker.text_eos_token)]],
                     dtype=torch.long, device=device)
    )
    pad_embed = thinker_embed_tokens(
        torch.tensor([[int(talker.text_pad_token)]],
                     dtype=torch.long, device=device)
    )
    thinker_reply_part = torch.cat(
        [thinker_reply_part[:, 1:, :], eos_embed, pad_embed], dim=1
    )

    talker_input_text_ids = torch.cat(
        [
            input_ids_t,
            torch.tensor([[talker_text_bos_token]],
                         dtype=torch.long, device=device),
            full_ids[:, prompt_len:prompt_len + 1],
        ],
        dim=-1,
    )

    talker_attention_mask = None
    if "attention_mask" in inputs:
        talker_attention_mask = torch.cat(
            [inputs["attention_mask"],
             inputs["attention_mask"].new_ones((1, 2))],
            dim=1,
        ).to(device)

    return {
        "input_ids": talker_input_ids,
        "input_text_ids": talker_input_text_ids,
        "thinker_reply_part": thinker_reply_part,
        "inputs_embeds": talker_inputs_embeds,
        "attention_mask": talker_attention_mask,
        "context_len": int(talker_input_ids.shape[1]),
    }


def bench_talker(model, processor, runs, max_new_tokens, speaker):
    talker = model.talker
    talker_kwargs = _prepare_talker_inputs(model, processor, speaker)
    context_len = talker_kwargs.pop("context_len")

    codec_eos = int(talker.config.tts_codec_end_token_id)
    codec_pad = int(talker.config.tts_codec_pad_token_id)
    codec_bos = int(talker.config.tts_codec_start_token_id)

    common = dict(
        eos_token_id=[codec_eos, codec_pad],
        suppress_tokens=[codec_bos],
        do_sample=True, temperature=0.9, top_k=40, top_p=0.8,
        repetition_penalty=1.05,
    )

    # Warmup
    with torch.inference_mode():
        _ = talker.generate(
            **talker_kwargs, max_new_tokens=max_new_tokens, **common,
        )
    _sync()

    wall = []
    for _ in range(runs):
        _sync()
        t0 = time.time()
        with torch.inference_mode():
            out = talker.generate(
                **talker_kwargs, max_new_tokens=max_new_tokens, **common,
            )
        _sync()
        wall.append(time.time() - t0)
    n_gen = int(out.shape[1] - context_len)
    return _stats(
        wall,
        extra={
            "max_new_tokens": max_new_tokens,
            "generated_tokens_last_run": n_gen,
            "context_len": context_len,
            "tpot_ms": round(1000.0 * statistics.median(wall) / max(n_gen, 1), 3),
        },
    )


def bench_dit(model, runs, mel_len, num_steps):
    """Bench the DiT transformer core only.

    HF Qwen2_5Omni puts the DiT under
    ``model.token2wav.code2wav_dit_model``. We feed random tensors of the
    right shape into ``transformer_blocks + norm_out + proj_out`` (matching
    what ``_NeuronDiTCore`` exports) ``num_steps`` times, mirroring the
    Neuron bench.
    """
    device = next(model.parameters()).device
    code2wav = model.token2wav
    if code2wav.dtype != torch.float:
        code2wav.float()
    dit = code2wav.code2wav_dit_model
    dit_cfg = getattr(dit, "config", None)
    dim = int(getattr(dit_cfg, "dim", 1024))
    num_heads = int(getattr(dit_cfg, "num_attention_heads", 16))
    head_dim = dim // num_heads
    bs = 2  # CFG doubles batch, matches Neuron compile config

    h = torch.randn(bs, mel_len, dim, dtype=torch.float32, device=device)
    te = torch.randn(1, dim, dtype=torch.float32, device=device)
    cos = torch.randn(bs, mel_len, head_dim, dtype=torch.float32, device=device)
    sin = torch.randn(bs, mel_len, head_dim, dtype=torch.float32, device=device)
    mask_local = torch.zeros(bs, 1, mel_len, mel_len, dtype=torch.float32,
                             device=device)
    mask_back = torch.zeros_like(mask_local)
    mask_ahead = torch.zeros_like(mask_local)

    @torch.inference_mode()
    def _step():
        position_embeddings = (cos, sin)
        masks = [mask_local, mask_back, mask_ahead]
        hidden = h
        for i, block in enumerate(dit.transformer_blocks):
            lb = block.look_backward_block
            la = block.look_ahead_block
            if lb == 0 and la == 0:
                am = masks[0]
            elif lb >= 1 and la == 0:
                am = masks[1]
            else:
                am = masks[2]
            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(
                hidden, emb=te,
            )
            attn_out = block.attn(
                hidden_states=norm,
                position_embeddings=position_embeddings,
                attention_mask=am,
            )
            hidden = hidden + gate_msa.unsqueeze(1) * attn_out
            n = block.ff_norm(hidden) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff = block.ff(n)
            hidden = hidden + gate_mlp.unsqueeze(1) * ff
        hidden = dit.norm_out(hidden, te)
        return dit.proj_out(hidden)

    # Warmup
    _ = _step()
    _sync()

    wall = []
    for _ in range(runs):
        _sync()
        t0 = time.time()
        for _ in range(num_steps):
            _ = _step()
        _sync()
        wall.append(time.time() - t0)
    return _stats(
        wall,
        extra={
            "mel_len_requested": mel_len,
            "bucket_used": mel_len,
            "num_steps": num_steps,
            "batch_size": bs,
            "dim": dim,
            "per_step_ms": round(
                1000.0 * statistics.median(wall) / num_steps, 3
            ),
        },
    )


def bench_bigvgan(model, runs, mel_len):
    device = next(model.parameters()).device
    code2wav = model.token2wav
    if code2wav.dtype != torch.float:
        code2wav.float()
    bigvgan = code2wav.code2wav_bigvgan_model
    mel_dim = int(bigvgan.config.mel_dim)

    mel = torch.randn(1, mel_dim, mel_len, dtype=torch.float32, device=device)

    @torch.inference_mode()
    def _fwd():
        # HF BigVGAN forward returns (1, T_wav). We don't care about value
        # correctness, only timing.
        return bigvgan(mel)

    _ = _fwd()
    _sync()

    wall = []
    for _ in range(runs):
        _sync()
        t0 = time.time()
        _ = _fwd()
        _sync()
        wall.append(time.time() - t0)
    return _stats(
        wall,
        extra={
            "mel_len_requested": mel_len,
            "bucket_used": mel_len,
            "mel_dim": mel_dim,
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get(
        "QWEN25_OMNI_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B"))
    p.add_argument("--num-runs", type=int, default=5)
    p.add_argument("--json", default="bench_gpu.json")
    p.add_argument("--speaker", default=DEFAULT_SPEAKER,
                   choices=["Ethan", "Chelsie"])
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
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

    if not torch.cuda.is_available():
        sys.exit("CUDA not available -- this script requires a GPU.")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print("=" * 60)
    print("Qwen2.5-Omni Per-Module Bench (GPU)")
    print("=" * 60)
    print(f"  Device:    {torch.cuda.get_device_name(0)}")
    print(f"  Model:     {args.model_path}")
    print(f"  dtype:     {args.dtype}")
    print(f"  attn_impl: {args.attn_impl}")
    print(f"  Runs:      {args.num_runs}")

    print("\n--- Loading model ---")
    model, processor, _ = gpu.load_model(args.model_path, dtype, args.attn_impl)

    results = {
        "platform": "gpu",
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "modules": {},
    }

    if "thinker" not in args.skip:
        print("\n--- Bench Thinker ---")
        results["modules"]["thinker"] = bench_thinker(
            model, processor, args.num_runs, args.thinker_max_new_tokens,
        )
        m = results["modules"]["thinker"]
        print(f"  median={m['median_s']:.3f}s tpot={m['tpot_ms']:.2f}ms")

    if "talker" not in args.skip:
        print("\n--- Bench Talker ---")
        results["modules"]["talker"] = bench_talker(
            model, processor,
            args.num_runs, args.talker_max_new_tokens, args.speaker,
        )
        m = results["modules"]["talker"]
        print(f"  median={m['median_s']:.3f}s tpot={m['tpot_ms']:.2f}ms")

    if "dit" not in args.skip:
        print("\n--- Bench DiT ---")
        results["modules"]["dit"] = bench_dit(
            model, args.num_runs, args.dit_mel_len, args.dit_num_steps,
        )
        m = results["modules"]["dit"]
        print(f"  median={m['median_s']:.3f}s "
              f"per_step={m['per_step_ms']:.2f}ms")

    if "bigvgan" not in args.skip:
        print("\n--- Bench BigVGAN ---")
        results["modules"]["bigvgan"] = bench_bigvgan(
            model, args.num_runs, args.bigvgan_mel_len,
        )
        m = results["modules"]["bigvgan"]
        print(f"  median={m['median_s']:.3f}s")

    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
