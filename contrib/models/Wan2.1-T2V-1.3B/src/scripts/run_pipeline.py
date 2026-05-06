#!/usr/bin/env python3
"""
WAN 2.1 T2V 1.3B — Full E2E pipeline on Neuron.

Assumes traced NEFFs already exist in this directory:
  - traced_t5_lnc1.pt
  - traced_wan_480x832.pt
  - vae_blocks_cached/*.pt

Usage (two-step due to LNC mismatch):

  # Step 1: T5 encoding (LNC=1)
  NEURON_RT_VISIBLE_CORES=0-3 NEURON_RT_VIRTUAL_CORE_SIZE=1 \
    python3 run_pipeline.py --step=t5 --prompt="a cat on a beach at sunset"

  # Step 2: Backbone + VAE (LNC=2)
  NEURON_RT_VISIBLE_CORES=0-3 \
    python3 run_pipeline.py --step=generate --seed=42 --steps=20 --cfg=5.0

Or run both steps:
  python3 run_pipeline.py --prompt="a cat on a beach at sunset" --seed=42
"""

import argparse
import os
import subprocess
import sys
import time

import torch
import torch_neuronx  # must import before torch.jit.load for Neuron models
import numpy as np
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("WAN_MODEL_DIR", "/mnt/work/wan/agent_artifacts/data"))
EMBEDS_PATH = Path("/tmp/t5_embeds.pt")


def step_t5(prompt: str, negative_prompt: str = ""):
    """Encode prompt with T5 on Neuron (LNC=1). Saves embeddings to /tmp."""
    print(f"[T5] Encoding: {prompt!r}")
    t0 = time.time()

    traced_t5 = torch.jit.load(str(SCRIPT_DIR / "traced_t5_lnc1.pt"))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))

    def encode(text):
        tokens = tokenizer(text, max_length=512, padding="max_length",
                           truncation=True, return_tensors="pt")
        ids = tokens.input_ids.to(torch.int64)
        mask = tokens.attention_mask.to(torch.int64)
        embeds = traced_t5(ids, mask)
        # Zero out padding positions (critical — HF pipeline does this)
        seq_len = mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        positions = torch.arange(512).unsqueeze(0).unsqueeze(-1)
        embeds = embeds * (positions < seq_len).to(embeds.dtype)
        return embeds

    with torch.no_grad():
        pe = encode(prompt)
        ne = encode(negative_prompt)

    torch.save({"pe": pe, "ne": ne}, str(EMBEDS_PATH))
    print(f"[T5] Done in {time.time()-t0:.1f}s -> {EMBEDS_PATH}")


def step_generate(seed: int = 42, steps: int = 20, cfg: float = 5.0):
    """Denoise with backbone on Neuron + decode with VAE hybrid CPU/Neuron."""
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    from modeling import VAE_BLOCK_ORDER

    # Load embeddings
    embeds = torch.load(str(EMBEDS_PATH))
    pe, ne = embeds["pe"], embeds["ne"]

    # Load backbone
    print("[Backbone] Loading traced model...")
    traced_bb = torch.jit.load(str(SCRIPT_DIR / "traced_wan_480x832.pt"))

    # Pre-compute RoPE on CPU
    m = WanTransformer3DModel.from_pretrained(str(MODEL_DIR / "transformer"), torch_dtype=torch.bfloat16)
    rc, rs = m.rope(torch.randn(1, 16, 4, 60, 104, dtype=torch.bfloat16))
    del m

    # Scheduler
    scheduler = UniPCMultistepScheduler.from_pretrained(str(MODEL_DIR / "scheduler"))

    # Denoise
    print(f"[Backbone] Denoising ({steps} steps, seed={seed})...")
    gen = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn(1, 16, 4, 60, 104, dtype=torch.float32, generator=gen)
    scheduler.set_timesteps(steps, device="cpu")

    t0 = time.time()
    for i, tt in enumerate(scheduler.timesteps):
        li = latents.to(torch.bfloat16)
        c = traced_bb(li, tt.expand(1), pe, rc, rs)
        u = traced_bb(li, tt.expand(1), ne, rc, rs)
        latents = scheduler.step(u + cfg * (c - u), tt, latents, return_dict=False)[0]
        if i % 5 == 0:
            print(f"  step {i}/{steps}")
    bb_time = time.time() - t0
    print(f"[Backbone] {bb_time:.1f}s")

    # VAE decode (hybrid)
    print("[VAE] Loading...")
    vae = AutoencoderKLWan.from_pretrained(str(MODEL_DIR / "vae"), torch_dtype=torch.bfloat16)
    vae.eval()

    # Make decoder XLA-compatible (monkey-patches forward methods, no source changes)
    from modeling import make_decoder_xla_compatible
    make_decoder_xla_compatible(vae.decoder)

    blocks_dir = SCRIPT_DIR / "vae_blocks_cached"
    B = {f.stem: torch.jit.load(str(f)) for f in sorted(blocks_dir.glob("*.pt"))}

    lm = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(torch.bfloat16)
    ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(torch.bfloat16)
    x = vae.post_quant_conv(latents.to(torch.bfloat16) / ls + lm)

    print("[VAE] CPU frames 0+1...")
    t0 = time.time()
    vae.clear_cache()
    frames = []
    for i in range(2):
        vae._conv_idx = [0]
        kw = dict(first_chunk=True) if i == 0 else {}
        with torch.no_grad():
            frames.append(vae.decoder(x[:, :, i:i+1, :, :],
                                      feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **kw))

    print("[VAE] Neuron frames 2+3...")
    cache = [v.clone() if isinstance(v, torch.Tensor) else v for v in vae._feat_map]
    for fi in range(2, 4):
        h = x[:, :, fi:fi+1, :, :]
        for name, cs, cc in VAE_BLOCK_ORDER:
            c_in = tuple(cache[cs + i] for i in range(cc))
            result = B[name](h, *c_in)
            if cc > 0:
                h = result[0]
                for i in range(cc):
                    cache[cs + i] = result[1 + i]
            else:
                h = result
        frames.append(h)

    vae_time = time.time() - t0
    print(f"[VAE] {vae_time:.1f}s")

    # Save frames
    video = torch.cat(frames, dim=2)
    video = ((video.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)[0].permute(1, 2, 3, 0).cpu().numpy()

    out_dir = SCRIPT_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    for i in range(video.shape[0]):
        Image.fromarray(video[i]).save(out_dir / f"frame_{i:03d}.png")

    print(f"\nDone! {video.shape[0]} frames at {video.shape[2]}x{video.shape[1]}")
    print(f"  Backbone: {bb_time:.1f}s")
    print(f"  VAE:      {vae_time:.1f}s")
    print(f"  Total:    {bb_time + vae_time:.1f}s")
    print(f"  Output:   {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="WAN 2.1 T2V on Neuron")
    parser.add_argument("--step", choices=["t5", "generate", "both"], default="both")
    parser.add_argument("--prompt", default="a cat sitting on a beach at sunset")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=5.0)
    args = parser.parse_args()

    if args.step in ("t5", "both"):
        if args.step == "both":
            # Run T5 in subprocess with LNC=1
            env = os.environ.copy()
            env["NEURON_RT_VISIBLE_CORES"] = env.get("NEURON_RT_VISIBLE_CORES", "0-3")
            env["NEURON_RT_VIRTUAL_CORE_SIZE"] = "1"
            subprocess.run([
                sys.executable, __file__,
                "--step=t5", f"--prompt={args.prompt}", f"--negative-prompt={args.negative_prompt}",
            ], env=env, check=True)
        else:
            step_t5(args.prompt, args.negative_prompt)

    if args.step in ("generate", "both"):
        step_generate(args.seed, args.steps, args.cfg)


if __name__ == "__main__":
    main()
