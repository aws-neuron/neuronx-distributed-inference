"""
VAE Decoder Neuron Compilation — v2 with torch_neuronx.trace()
==============================================================
Uses torch_neuronx.trace() instead of ModelBuilder.
Previous ModelBuilder attempt hit BrokenPipeError in hlo-neff-wrapper.

Patches applied:
  1. CausalConv3d: F.pad(mode='replicate') -> torch.cat + F.pad(mode='constant')
  2. swish: inplace=True -> inplace=False
  3. prepare_causal_attention_mask: vectorized (no Python loop)

Launch:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    python ./compile_vae_neuron.py
"""

import sys
import os
import time
import json
import argparse

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "0"  # Disable to avoid potential issues

sys.path.insert(0, os.environ.get("HUNYUAN_REPO_DIR", "./HunyuanVideo-1.5"))
os.environ["HUNYUAN_ATTN_MODE"] = "torch"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

MODELS_DIR = os.environ.get("HUNYUAN_MODELS_DIR", "./models")
COMPILED_DIR = os.environ.get("HUNYUAN_COMPILED_DIR", "./compiled")


# ============================================================
# Monkey-patches for Neuron compatibility
# ============================================================


def patched_causal_conv3d_forward(self, x):
    """CausalConv3d.forward() without F.pad(mode='replicate')."""
    w_pad_l, w_pad_r, h_pad_l, h_pad_r, t_pad_l, t_pad_r = self.time_causal_padding

    # Temporal padding via torch.cat (replicate edge frames)
    if t_pad_l > 0:
        first_frame = x[:, :, :1]
        x = torch.cat([first_frame.expand(-1, -1, t_pad_l, -1, -1), x], dim=2)
    if t_pad_r > 0:
        last_frame = x[:, :, -1:]
        x = torch.cat([x, last_frame.expand(-1, -1, t_pad_r, -1, -1)], dim=2)

    # Spatial padding via constant (zero) padding
    if h_pad_l > 0 or h_pad_r > 0 or w_pad_l > 0 or w_pad_r > 0:
        x = F.pad(
            x, (w_pad_l, w_pad_r, h_pad_l, h_pad_r, 0, 0), mode="constant", value=0
        )

    return self.conv(x)


def patched_swish(x, inplace=False):
    """SiLU/swish without inplace."""
    return F.silu(x, inplace=False)


def patched_prepare_causal_attention_mask(
    n_frame, n_hw, dtype, device, batch_size=None
):
    """Vectorized causal attention mask (no Python loop)."""
    seq_len = n_frame * n_hw
    frame_idx = torch.arange(seq_len, device=device) // n_hw
    mask = frame_idx.unsqueeze(1) >= frame_idx.unsqueeze(0)
    float_mask = torch.where(
        mask,
        torch.zeros(1, dtype=dtype, device=device),
        torch.full((1,), float("-inf"), dtype=dtype, device=device),
    )
    if batch_size is not None:
        float_mask = float_mask.unsqueeze(0).expand(batch_size, -1, -1)
    return float_mask


def apply_all_patches():
    """Apply all Neuron compatibility patches."""
    import hyvideo.models.autoencoders.hunyuanvideo_15_vae as vae_module

    vae_module.CausalConv3d.forward = patched_causal_conv3d_forward
    vae_module.swish = patched_swish
    vae_module.prepare_causal_attention_mask = patched_prepare_causal_attention_mask
    print("[PATCH] CausalConv3d: replicate pad -> torch.cat + constant pad")
    print("[PATCH] swish: inplace=False")
    print("[PATCH] prepare_causal_attention_mask: vectorized")


# ============================================================
# VAE Decoder Wrapper
# ============================================================


class VAEDecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        return self.decoder(z)


# ============================================================
# Compilation
# ============================================================


def compile_vae_decoder(args):
    dtype = torch.bfloat16
    tile_h = args.tile_h
    tile_w = args.tile_w
    T_lat = args.t_lat

    print("=" * 60)
    print("HunyuanVideo VAE Decoder — Neuron Compilation (trace)")
    print("=" * 60)
    print(f"  Tile latent:  {tile_h}x{tile_w} (pixels: {tile_h * 16}x{tile_w * 16})")
    print(f"  T_lat:        {T_lat} ({(T_lat - 1) * 4 + 1} output frames)")
    print(f"  Dtype:        {dtype}")
    print(f"  Output:       {args.output_dir}")
    print()

    # Apply patches
    apply_all_patches()

    # Load VAE
    print("\nLoading VAE...")
    t0 = time.time()
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae import AutoencoderKLConv3D

    vae = AutoencoderKLConv3D.from_pretrained(
        f"{MODELS_DIR}/HunyuanVideo-1.5/vae",
        torch_dtype=dtype,
    )
    vae.eval()
    decoder_params = sum(p.numel() for p in vae.decoder.parameters())
    print(
        f"Loaded in {time.time() - t0:.1f}s, decoder: {decoder_params / 1e6:.1f}M params"
    )

    # CPU reference
    print("\nCPU reference (patched)...")
    z_test = torch.randn(1, 32, T_lat, tile_h, tile_w, dtype=dtype)
    with torch.no_grad():
        cpu_out = vae.decoder(z_test)
    print(f"  Input:  {list(z_test.shape)}")
    print(f"  Output: {list(cpu_out.shape)}")
    print(f"  Range:  [{cpu_out.min():.3f}, {cpu_out.max():.3f}]")

    # Wrap
    wrapper = VAEDecoderWrapper(vae.decoder)
    wrapper.eval()

    # Trace with torch_neuronx
    z_input = torch.randn(1, 32, T_lat, tile_h, tile_w, dtype=dtype)
    compiler_args = [
        "--model-type=unet-inference",
        "-O1",
        "--auto-cast=none",
    ]

    print(f"\nTracing with torch_neuronx.trace()...")
    print(f"  Input shape: {list(z_input.shape)}")
    print(f"  Compiler args: {compiler_args}")
    t_compile = time.time()

    try:
        traced = torch_neuronx.trace(
            wrapper,
            (z_input,),
            compiler_args=compiler_args,
            compiler_workdir=args.compiler_workdir,
        )
        compile_time = time.time() - t_compile
        print(f"Traced + compiled in {compile_time:.1f}s")
    except Exception as e:
        compile_time = time.time() - t_compile
        print(f"\nTRACE FAILED after {compile_time:.1f}s: {type(e).__name__}: {e}")

        # Try with smaller tile
        if tile_h > 4 or tile_w > 4:
            print("\nRetrying with 4x4 tile...")
            tile_h_small = 4
            tile_w_small = 4
            z_small = torch.randn(1, 32, T_lat, tile_h_small, tile_w_small, dtype=dtype)

            with torch.no_grad():
                cpu_out_small = vae.decoder(z_small)
            print(f"  CPU ref: {list(z_small.shape)} -> {list(cpu_out_small.shape)}")

            t2 = time.time()
            try:
                traced = torch_neuronx.trace(
                    wrapper,
                    (z_small,),
                    compiler_args=compiler_args,
                    compiler_workdir=args.compiler_workdir + "_4x4",
                )
                compile_time = time.time() - t2
                print(f"4x4 tile traced in {compile_time:.1f}s")
                tile_h = tile_h_small
                tile_w = tile_w_small
                z_test = z_small
                cpu_out = cpu_out_small
            except Exception as e2:
                print(f"4x4 ALSO FAILED: {type(e2).__name__}: {e2}")
                return None
        else:
            return None

    # Save
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "vae_decoder.pt")
    traced.save(save_path)
    print(f"Saved to {save_path}")

    # Validate
    print("\nValidating...")
    neuron_out = traced(z_test)
    cos_sim = F.cosine_similarity(
        cpu_out.float().flatten().unsqueeze(0),
        neuron_out.float().flatten().unsqueeze(0),
    ).item()
    mae = (cpu_out.float() - neuron_out.float()).abs().mean().item()
    print(f"  Accuracy: cos_sim={cos_sim:.6f}, mae={mae:.6f}")

    # Benchmark
    print("\nBenchmarking (warmup 3, bench 10)...")
    for _ in range(3):
        _ = traced(z_test)

    times = []
    for _ in range(10):
        t0 = time.time()
        _ = traced(z_test)
        times.append(time.time() - t0)

    avg_ms = sum(times) * 1000 / len(times)
    min_ms = min(times) * 1000
    print(f"  Tile decode: avg={avg_ms:.1f}ms, min={min_ms:.1f}ms")

    # Save config
    config = {
        "tile_latent_h": tile_h,
        "tile_latent_w": tile_w,
        "tile_pixel_h": tile_h * 16,
        "tile_pixel_w": tile_w * 16,
        "T_lat": T_lat,
        "T_out": (T_lat - 1) * 4 + 1,
        "latent_channels": 32,
        "ffactor_spatial": 16,
        "ffactor_temporal": 4,
        "dtype": "bfloat16",
        "decoder_params_M": round(decoder_params / 1e6, 1),
        "compile_time_s": round(compile_time, 1),
        "tile_latency_avg_ms": round(avg_ms, 1),
        "tile_latency_min_ms": round(min_ms, 1),
        "accuracy_cos_sim": round(cos_sim, 6),
        "accuracy_mae": round(mae, 6),
        "method": "torch_neuronx.trace",
        "patches": [
            "CausalConv3d: replicate_pad -> torch.cat + constant_pad",
            "swish: inplace=False",
            "prepare_causal_attention_mask: vectorized",
        ],
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {output_dir}/config.json")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Decoder params:   {decoder_params / 1e6:.1f}M")
    print(f"  Tile latent:      {tile_h}x{tile_w}")
    print(f"  Compile time:     {compile_time:.1f}s")
    print(f"  Tile decode:      {avg_ms:.1f}ms avg, {min_ms:.1f}ms min")
    print(f"  Accuracy:         cos_sim={cos_sim:.6f}")
    print(f"  Saved to:         {output_dir}")

    return config


def main():
    parser = argparse.ArgumentParser(description="Compile HunyuanVideo VAE for Neuron")
    parser.add_argument("--tile-h", type=int, default=8, help="Tile latent height")
    parser.add_argument("--tile-w", type=int, default=8, help="Tile latent width")
    parser.add_argument("--t-lat", type=int, default=2, help="Temporal latent frames")
    parser.add_argument(
        "--output-dir", type=str, default=f"{COMPILED_DIR}/vae_decoder_neuron"
    )
    parser.add_argument(
        "--compiler-workdir", type=str, default="./compiler_workdir_vae"
    )
    args = parser.parse_args()

    compile_vae_decoder(args)


if __name__ == "__main__":
    main()
