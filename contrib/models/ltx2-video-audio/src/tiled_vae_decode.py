"""
NxDI LTX-2 Tiled VAE Decode — Spatial Tiling with Overlap + Blending
=====================================================================
Tiles the latent space into overlapping patches, decodes each on Neuron,
then blends in output pixel space.

Optimal tile shape: 4x16 latent (128x512 pixels) with overlap_h=1, overlap_w=0.

Tile count examples (1024x1536, latent 32x48):
  8x8  tiles, overlap=4:     77 tiles, ~111s (CPU wins)
  4x16 tiles, oh=1, ow=4:   44 tiles, ~56s  (1.78x vs CPU)
  4x16 tiles, oh=1, ow=0:   33 tiles, ~42s  (2.32x vs CPU)
  4x16 tiles, oh=0, ow=0:   24 tiles, ~31s  (3.16x vs CPU)

Usage (standalone):
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  NEURON_RT_VISIBLE_CORES=0-3 python tiled_vae_decode.py \\
    --compiled-dir /home/ubuntu/ltx2_vae_tp4_128x512 \\
    --tp-degree 4 --height 1024 --width 1536 \\
    --tile-latent-h 4 --tile-latent-w 16 \\
    --overlap-h 1 --overlap-w 0

Usage (as library):
  from tiled_vae_decode import tiled_decode
  output = tiled_decode(latent, compiled_model, tile_latent_h=4, tile_latent_w=16,
                        overlap_latent_h=1, overlap_latent_w=0)
"""

import os
import time

import torch

os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")

COMPILER_FLAGS = (
    "--model-type=unet-inference -O1 --auto-cast none "
    "--enable-fast-loading-neuron-binaries"
)


def create_blend_mask_1d(length, blend_size, device="cpu"):
    """Create a 1D linear blending mask.

    Returns a tensor of shape [length] where:
    - First blend_size pixels ramp from 0 to 1
    - Middle pixels are 1
    - Last blend_size pixels ramp from 1 to 0
    """
    mask = torch.ones(length, device=device)
    if blend_size > 0:
        ramp = torch.linspace(0, 1, blend_size + 2, device=device)[1:-1]
        mask[:blend_size] = ramp
        mask[-blend_size:] = ramp.flip(0)
    return mask


def tiled_decode(
    latent,
    compiled_model,
    tile_latent_h=4,
    tile_latent_w=16,
    overlap_latent_h=1,
    overlap_latent_w=0,
    spatial_scale=32,
    verbose=True,
):
    """Decode latent tensor using spatial tiling with overlap blending.

    Args:
        latent: [1, 128, T, H_lat, W_lat] latent tensor
        compiled_model: Neuron-compiled VAE decoder
        tile_latent_h: tile height in latent space (default 4)
        tile_latent_w: tile width in latent space (default 16)
        overlap_latent_h: overlap in latent H pixels (default 1)
        overlap_latent_w: overlap in latent W pixels (default 0)
        spatial_scale: latent-to-pixel spatial scale (32 for LTX-2)
        verbose: print progress information

    Returns:
        [1, 3, T_out, H_out, W_out] decoded video tensor
    """
    B, C, T, H_lat, W_lat = latent.shape
    assert B == 1, "Batch size must be 1"

    H_out = H_lat * spatial_scale
    W_out = W_lat * spatial_scale

    stride_h = tile_latent_h - overlap_latent_h
    stride_w = tile_latent_w - overlap_latent_w
    assert stride_h > 0, (
        f"stride_h={stride_h} must be > 0 "
        f"(tile_h={tile_latent_h}, overlap_h={overlap_latent_h})"
    )
    assert stride_w > 0, (
        f"stride_w={stride_w} must be > 0 "
        f"(tile_w={tile_latent_w}, overlap_w={overlap_latent_w})"
    )

    overlap_h_pixels = overlap_latent_h * spatial_scale
    overlap_w_pixels = overlap_latent_w * spatial_scale
    tile_h_pixels = tile_latent_h * spatial_scale
    tile_w_pixels = tile_latent_w * spatial_scale

    # Determine tile start positions
    n_tiles_h = max(1, (H_lat - tile_latent_h + stride_h - 1) // stride_h + 1)
    n_tiles_w = max(1, (W_lat - tile_latent_w + stride_w - 1) // stride_w + 1)

    tile_starts_h = []
    for i in range(n_tiles_h):
        start = min(i * stride_h, H_lat - tile_latent_h)
        tile_starts_h.append(start)
    tile_starts_h = sorted(set(tile_starts_h))

    tile_starts_w = []
    for i in range(n_tiles_w):
        start = min(i * stride_w, W_lat - tile_latent_w)
        tile_starts_w.append(start)
    tile_starts_w = sorted(set(tile_starts_w))

    n_tiles_h = len(tile_starts_h)
    n_tiles_w = len(tile_starts_w)
    total_tiles = n_tiles_h * n_tiles_w

    if verbose:
        print(f"  Tiling: {n_tiles_h}x{n_tiles_w} = {total_tiles} tiles")
        print(
            f"  Tile latent: {tile_latent_h}x{tile_latent_w}, "
            f"overlap: h={overlap_latent_h}, w={overlap_latent_w}"
        )

    # Output temporal dimension: T_out = (T-1)*8 + 1
    T_out = (T - 1) * 8 + 1

    output_accum = torch.zeros(1, 3, T_out, H_out, W_out, dtype=torch.float32)
    weight_accum = torch.zeros(1, 1, 1, H_out, W_out, dtype=torch.float32)

    decode_times = []

    for ti, h_start_lat in enumerate(tile_starts_h):
        for tj, w_start_lat in enumerate(tile_starts_w):
            tile_idx = ti * n_tiles_w + tj + 1

            h_end_lat = h_start_lat + tile_latent_h
            w_end_lat = w_start_lat + tile_latent_w
            tile_latent = latent[:, :, :, h_start_lat:h_end_lat, w_start_lat:w_end_lat]

            t0 = time.time()
            with torch.no_grad():
                tile_output = compiled_model(tile_latent)
            dt = time.time() - t0
            decode_times.append(dt)

            h_start_px = h_start_lat * spatial_scale
            w_start_px = w_start_lat * spatial_scale
            h_end_px = h_start_px + tile_h_pixels
            w_end_px = w_start_px + tile_w_pixels

            # Create spatial blend mask
            blend_h = create_blend_mask_1d(
                tile_h_pixels, overlap_h_pixels if h_start_lat > 0 else 0
            )
            blend_w = create_blend_mask_1d(
                tile_w_pixels, overlap_w_pixels if w_start_lat > 0 else 0
            )

            # Handle end tiles
            if h_end_lat < H_lat and overlap_h_pixels > 0:
                blend_h[-overlap_h_pixels:] = torch.linspace(
                    1, 0, overlap_h_pixels + 2
                )[1:-1]
            if w_end_lat < W_lat and overlap_w_pixels > 0:
                blend_w[-overlap_w_pixels:] = torch.linspace(
                    1, 0, overlap_w_pixels + 2
                )[1:-1]

            # 2D blend mask: [1, 1, 1, H, W]
            blend_mask = blend_h.unsqueeze(1) * blend_w.unsqueeze(0)
            blend_mask = blend_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            output_accum[:, :, :, h_start_px:h_end_px, w_start_px:w_end_px] += (
                tile_output.float() * blend_mask
            )
            weight_accum[:, :, :, h_start_px:h_end_px, w_start_px:w_end_px] += (
                blend_mask
            )

            if verbose:
                print(
                    f"    Tile {tile_idx}/{total_tiles}: "
                    f"lat[{h_start_lat}:{h_end_lat}, {w_start_lat}:{w_end_lat}] "
                    f"-> px[{h_start_px}:{h_end_px}, {w_start_px}:{w_end_px}], "
                    f"{dt * 1000:.0f}ms"
                )

    output = output_accum / weight_accum.clamp(min=1e-6)

    total_decode = sum(decode_times)
    if verbose:
        print(f"\n  Total decode time: {total_decode:.2f}s ({total_tiles} tiles)")
        print(f"  Avg per tile: {total_decode / total_tiles * 1000:.0f}ms")

    return output


def load_compiled_vae(compiled_dir):
    """Load a compiled TP VAE decoder from disk.

    Args:
        compiled_dir: Directory containing tp_0.pt, tp_1.pt, etc.

    Returns:
        compiled: The loaded TensorParallelNeuronModel
    """
    import neuronx_distributed

    return neuronx_distributed.trace.parallel_model_load(compiled_dir)


def main():
    """Standalone CLI for tiled VAE decode."""
    import argparse

    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description="LTX-2 Tiled VAE Decode")
    parser.add_argument(
        "--compiled-dir",
        type=str,
        required=True,
        help="Directory with compiled TP model (tp_0.pt, etc.)",
    )
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--height", type=int, default=512, help="Target pixel height")
    parser.add_argument("--width", type=int, default=768, help="Target pixel width")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        help="Default overlap in latent pixels (used for both H and W unless overridden)",
    )
    parser.add_argument(
        "--overlap-h", type=int, default=None, help="Override overlap for H dimension"
    )
    parser.add_argument(
        "--overlap-w", type=int, default=None, help="Override overlap for W dimension"
    )
    parser.add_argument(
        "--tile-latent-h", type=int, default=4, help="Tile height in latent space"
    )
    parser.add_argument(
        "--tile-latent-w", type=int, default=16, help="Tile width in latent space"
    )
    parser.add_argument(
        "--validate-cpu", action="store_true", help="Also run CPU decode and compare"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["NEURON_CC_FLAGS"] = COMPILER_FLAGS

    latent_t = (args.num_frames - 1) // 8 + 1
    latent_h = args.height // 32
    latent_w = args.width // 32

    print("=" * 60)
    print("LTX-2 VAE Tiled Decode")
    print("=" * 60)
    print(f"  Target: {args.height}x{args.width}, {args.num_frames} frames")
    print(f"  Latent: [1, 128, {latent_t}, {latent_h}, {latent_w}]")
    print(f"  Tile latent: {args.tile_latent_h}x{args.tile_latent_w}")

    torch.manual_seed(args.seed)
    latent = torch.randn(1, 128, latent_t, latent_h, latent_w, dtype=torch.float32)

    print("\n  Loading compiled model...")
    compiled = load_compiled_vae(args.compiled_dir)
    print("  Model loaded.")

    # Warmup
    print("  Warmup (2 iterations)...")
    dummy = torch.randn(
        1, 128, latent_t, args.tile_latent_h, args.tile_latent_w, dtype=torch.float32
    )
    for _ in range(2):
        with torch.no_grad():
            compiled(dummy)
    print("  Warmup done.")

    # Compute effective overlap
    overlap_h = args.overlap_h if args.overlap_h is not None else args.overlap
    overlap_w = args.overlap_w if args.overlap_w is not None else args.overlap
    overlap_h = min(overlap_h, args.tile_latent_h - 1)
    overlap_w = min(overlap_w, args.tile_latent_w - 1)

    print("\n  Running tiled decode...")
    t0 = time.time()
    output_neuron = tiled_decode(
        latent,
        compiled,
        tile_latent_h=args.tile_latent_h,
        tile_latent_w=args.tile_latent_w,
        overlap_latent_h=overlap_h,
        overlap_latent_w=overlap_w,
        spatial_scale=32,
    )
    total_time = time.time() - t0
    print(f"\n  RESULT: Tiled decode {args.height}x{args.width} in {total_time:.2f}s")
    print(f"  Output shape: {list(output_neuron.shape)}")
    print(f"  Output range: [{output_neuron.min():.3f}, {output_neuron.max():.3f}]")

    if args.validate_cpu:
        print("\n  Running CPU decode for validation...")
        from diffusers import AutoencoderKLLTX2Video

        vae = AutoencoderKLLTX2Video.from_pretrained(
            "Lightricks/LTX-2", subfolder="vae", torch_dtype=torch.float32
        )
        vae.eval()

        t0_cpu = time.time()
        with torch.no_grad():
            cpu_output = vae.decoder(latent, temb=None, causal=False)
        cpu_time = time.time() - t0_cpu

        print(f"  CPU decode time: {cpu_time:.2f}s")

        if cpu_output.shape == output_neuron.shape:
            cos_sims = []
            for f in range(output_neuron.shape[2]):
                a = output_neuron[0, :, f].flatten()
                b = cpu_output[0, :, f].flatten()
                cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
                cos_sims.append(cos.item())
            avg_cos = sum(cos_sims) / len(cos_sims)
            min_cos = min(cos_sims)
            print(f"  Cosine similarity: avg={avg_cos:.6f}, min={min_cos:.6f}")
            print(f"  MSE: {F.mse_loss(output_neuron, cpu_output).item():.6f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Resolution: {args.height}x{args.width}, {args.num_frames} frames")
    print(f"  Tiled Neuron decode: {total_time:.2f}s")


if __name__ == "__main__":
    main()
