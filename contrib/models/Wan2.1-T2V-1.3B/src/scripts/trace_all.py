#!/usr/bin/env python3
"""
Trace all Neuron NEFFs from scratch (backbone, T5, VAE blocks).

Prerequisites:
  - Model weights at WAN_MODEL_DIR (default: /mnt/work/wan/agent_artifacts/data)
  - VAE patches applied (run: python3 -c "from modeling import apply_vae_patches; apply_vae_patches('<path>')")

Usage:
  # Trace backbone + VAE (LNC=2)
  NEURON_RT_VISIBLE_CORES=0-3 python3 trace_all.py --component=backbone
  NEURON_RT_VISIBLE_CORES=0-3 python3 trace_all.py --component=vae

  # Trace T5 (LNC=1)
  NEURON_RT_VISIBLE_CORES=0-3 NEURON_RT_VIRTUAL_CORE_SIZE=1 python3 trace_all.py --component=t5

  # Or all at once (handles LNC switching via subprocess):
  python3 trace_all.py --component=all
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch_neuronx

from modeling import (
    WanBackboneWrapper, T5Wrapper,
    ConvInCached, BlockCached, NormConvOutCached, NoCacheWrapper,
    VAE_BLOCK_ORDER, make_decoder_xla_compatible,
)

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("WAN_MODEL_DIR", "/mnt/work/wan/agent_artifacts/data"))
CC_TRANSFORMER = "--model-type=transformer -O1 --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=false'"
CC_VAE = "--model-type=unet-inference -O1 --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=false'"


def trace_backbone():
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    print("[Backbone] Loading weights...")
    model = WanTransformer3DModel.from_pretrained(str(MODEL_DIR / "transformer"), torch_dtype=torch.bfloat16)
    model.eval()

    wrapper = WanBackboneWrapper(model)
    h = torch.randn(1, 16, 4, 60, 104, dtype=torch.bfloat16)
    t = torch.tensor([999], dtype=torch.int64)
    e = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
    rc, rs = model.rope(h)

    print("[Backbone] Tracing (this takes ~3 min)...")
    t0 = time.time()
    traced = torch_neuronx.trace(wrapper, (h, t, e, rc, rs), compiler_args=CC_TRANSFORMER)
    print(f"[Backbone] Compiled in {time.time()-t0:.0f}s")

    out = SCRIPT_DIR / "traced_wan_480x832.pt"
    torch.jit.save(traced, str(out))
    print(f"[Backbone] Saved: {out}")


def trace_t5():
    from transformers import T5EncoderModel

    print("[T5] Loading weights...")
    t5 = T5EncoderModel.from_pretrained(str(MODEL_DIR / "text_encoder"), torch_dtype=torch.bfloat16)
    t5.eval()

    wrapper = T5Wrapper(t5)
    ids = torch.ones(1, 512, dtype=torch.int64)
    mask = torch.cat([torch.ones(1, 9, dtype=torch.int64), torch.zeros(1, 503, dtype=torch.int64)], dim=1)

    print("[T5] Tracing (LNC=1, ~2 min)...")
    t0 = time.time()
    traced = torch_neuronx.trace(wrapper, (ids, mask),
        compiler_args=CC_TRANSFORMER + " --logical-nc-config=1")
    print(f"[T5] Compiled in {time.time()-t0:.0f}s")

    out = SCRIPT_DIR / "traced_t5_lnc1.pt"
    torch.jit.save(traced, str(out))
    print(f"[T5] Saved: {out}")


def trace_vae():
    from diffusers import AutoencoderKLWan

    print("[VAE] Loading weights...")
    vae = AutoencoderKLWan.from_pretrained(str(MODEL_DIR / "vae"), torch_dtype=torch.bfloat16)
    vae.eval()
    d = vae.decoder

    # Make decoder XLA-compatible (monkey-patches forward methods, no source changes)
    make_decoder_xla_compatible(d)

    # CPU warmup to populate cache
    z = torch.randn(1, 16, 4, 60, 104, dtype=torch.bfloat16)
    lm = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(torch.bfloat16)
    ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(torch.bfloat16)
    x = vae.post_quant_conv(z / ls + lm)

    vae.clear_cache()
    for i in range(2):
        vae._conv_idx = [0]
        kw = dict(first_chunk=True) if i == 0 else {}
        vae.decoder(x[:, :, i:i+1, :, :], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **kw)

    cache = [v.clone() if isinstance(v, torch.Tensor) else v for v in vae._feat_map]

    # Get per-block input shapes by running frame 2 on CPU
    h = x[:, :, 2:3, :, :]
    block_inputs = {}
    idx = [0]

    with torch.no_grad():
        # conv_in
        block_inputs["conv_in"] = h.clone()
        h = d.conv_in(h, cache_x=cache[0])
        cache[0] = torch.cat([cache[0], block_inputs["conv_in"]], dim=2)[:, :, -2:, :, :]
        idx[0] = 1

        # mid_block
        block_inputs["mid_block"] = h.clone()
        h = d.mid_block(h, feat_cache=cache, feat_idx=idx)

        # up_blocks
        for i in range(4):
            ub = d.up_blocks[i]
            for j in range(3):
                block_inputs[f"up{i}_resnet{j}"] = h.clone()
                h = ub.resnets[j](h, feat_cache=cache, feat_idx=idx)
            if hasattr(ub, "upsamplers") and ub.upsamplers and len(ub.upsamplers) > 0:
                block_inputs[f"up{i}_upsample"] = h.clone()
                h = ub.upsamplers[0](h, feat_cache=cache, feat_idx=idx)

        # norm_conv_out
        block_inputs["norm_conv_out"] = h.clone()

    # Re-populate cache for tracing
    vae.clear_cache()
    for i in range(2):
        vae._conv_idx = [0]
        kw = dict(first_chunk=True) if i == 0 else {}
        vae.decoder(x[:, :, i:i+1, :, :], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **kw)
    cache = [v.clone() if isinstance(v, torch.Tensor) else v for v in vae._feat_map]

    # Trace each block
    out_dir = SCRIPT_DIR / "vae_blocks_cached"
    out_dir.mkdir(exist_ok=True)

    def get_module(name):
        if name == "conv_in": return d.conv_in
        if name == "mid_block": return d.mid_block
        if name == "norm_conv_out": return None
        parts = name.split("_")
        i = int(parts[0][2:])
        if "resnet" in name:
            return d.up_blocks[i].resnets[int(parts[1][-1])]
        return d.up_blocks[i].upsamplers[0]

    total = len(VAE_BLOCK_ORDER)
    for bi, (name, cs, cc) in enumerate(VAE_BLOCK_ORDER):
        pct = int((bi / total) * 100)
        h_in = block_inputs[name]

        if name == "conv_in":
            wrapper = ConvInCached(d.conv_in)
            c_in = (cache[cs],)
        elif name == "norm_conv_out":
            wrapper = NormConvOutCached(d)
            c_in = (cache[cs],)
        elif cc == 0:
            wrapper = NoCacheWrapper(get_module(name))
            c_in = ()
        else:
            wrapper = BlockCached(get_module(name), cs, cc)
            c_in = tuple(cache[cs + i] for i in range(cc))

        with torch.no_grad():
            cpu_result = wrapper(h_in, *c_in)

        cpu_out = cpu_result[0] if isinstance(cpu_result, tuple) and cc > 0 else cpu_result
        print(f"[{pct:3d}%] {name}: {list(h_in.shape)} -> {list(cpu_out.shape)}...", end=" ", flush=True)

        traced = torch_neuronx.trace(wrapper, (h_in, *c_in), compiler_args=CC_VAE)

        import torch.nn.functional as F
        n_result = traced(h_in, *c_in)
        n_out = n_result[0] if isinstance(n_result, tuple) and cc > 0 else n_result
        cos = F.cosine_similarity(cpu_out.flatten().float(), n_out.flatten().float(), dim=0)
        print(f"cos={cos.item():.4f}")

        torch.jit.save(traced, str(out_dir / f"{name}.pt"))

        # Update cache for next block
        if cc > 0 and isinstance(cpu_result, tuple):
            for i in range(cc):
                cache[cs + i] = cpu_result[1 + i]

    print(f"[100%] All {total} VAE blocks traced -> {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["backbone", "t5", "vae", "all"], required=True)
    args = parser.parse_args()

    if args.component == "all":
        env = os.environ.copy()
        cores = env.get("NEURON_RT_VISIBLE_CORES", "0-3")

        # T5 in subprocess (LNC=1)
        env["NEURON_RT_VISIBLE_CORES"] = cores
        env["NEURON_RT_VIRTUAL_CORE_SIZE"] = "1"
        subprocess.run([sys.executable, __file__, "--component=t5"], env=env, check=True)

        # Backbone + VAE (LNC=2)
        env["NEURON_RT_VISIBLE_CORES"] = cores
        env.pop("NEURON_RT_VIRTUAL_CORE_SIZE", None)
        subprocess.run([sys.executable, __file__, "--component=backbone"], env=env, check=True)
        subprocess.run([sys.executable, __file__, "--component=vae"], env=env, check=True)
    elif args.component == "backbone":
        trace_backbone()
    elif args.component == "t5":
        trace_t5()
    elif args.component == "vae":
        trace_vae()


if __name__ == "__main__":
    main()
