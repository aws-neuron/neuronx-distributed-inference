"""
NxDI LTX-2.3 Tiled VAE Decode — Spatial Tiling with Overlap + Blending
=======================================================================
Tiles the latent space into overlapping patches, decodes each on Neuron,
then blends in output pixel space.

Key difference from LTX-2: LTX-2.3 has timestep_conditioning, so we must
also pass the precomputed scaled_timestep to the compiled model.

Pre-processing (done on CPU before calling compiled model):
  1. Noise injection: sample = noise * 0.025 + sample * 0.975
  2. Denormalization: per_channel_statistics.un_normalize(sample)
  Both are applied to the FULL latent before tiling (unlike per-tile).

Optimal tile: 4x16 latent (128x512 pixels) with overlap_h=1, overlap_w=0.

Usage (as library):
  from tiled_vae_decode_23 import tiled_decode, load_compiled_vae, preprocess_latent
  latent = preprocess_latent(raw_latent, video_decoder, seed=42)
  scaled_timestep = torch.tensor([0.05 * video_decoder.timestep_scale_multiplier.item()])
  output = tiled_decode(latent, compiled_model, scaled_timestep=scaled_timestep,
                        tile_latent_h=4, tile_latent_w=16, overlap_latent_h=1, overlap_latent_w=0)
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


def preprocess_latent(raw_latent, video_decoder, seed=None):
    """Apply noise injection + denormalization on CPU.

    This must be done BEFORE tiling, on the full latent.

    Args:
        raw_latent: [1, 128, T, H, W] raw latent from denoising
        video_decoder: CPU VideoDecoder (has per_channel_statistics, decode_noise_scale)
        seed: Random seed for noise injection (None = random)

    Returns:
        [1, 128, T, H, W] preprocessed latent ready for tiled Neuron decode
    """
    sample = raw_latent.float()

    if video_decoder.timestep_conditioning:
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        noise = (
            torch.randn(
                sample.size(), generator=gen, dtype=sample.dtype, device=sample.device
            )
            * video_decoder.decode_noise_scale
        )
        sample = noise + (1.0 - video_decoder.decode_noise_scale) * sample

    # Denormalize
    sample = video_decoder.per_channel_statistics.un_normalize(sample)

    return sample


def get_scaled_timestep(video_decoder, batch_size=1):
    """Compute scaled_timestep on CPU.

    Returns:
        [batch_size] tensor of scaled timestep values, or None if no conditioning
    """
    if not video_decoder.timestep_conditioning:
        return None

    timestep = torch.full(
        (batch_size,),
        video_decoder.decode_timestep,
        dtype=torch.float32,
    )
    scaled = timestep * video_decoder.timestep_scale_multiplier.item()
    return scaled


def create_blend_mask_1d(length, blend_size, device="cpu"):
    """Create a 1D linear blending mask."""
    mask = torch.ones(length, device=device)
    if blend_size > 0:
        ramp = torch.linspace(0, 1, blend_size + 2, device=device)[1:-1]
        mask[:blend_size] = ramp
        mask[-blend_size:] = ramp.flip(0)
    return mask


def tiled_decode(
    latent,
    compiled_model,
    scaled_timestep=None,
    tile_latent_h=4,
    tile_latent_w=16,
    overlap_latent_h=1,
    overlap_latent_w=0,
    spatial_scale=32,
    verbose=True,
):
    """Decode preprocessed latent tensor using spatial tiling with overlap blending.

    IMPORTANT: latent must be preprocessed (noise injected + denormalized)
    using preprocess_latent() before calling this function.

    Args:
        latent: [1, 128, T, H_lat, W_lat] preprocessed latent tensor
        compiled_model: Neuron-compiled VAE decoder
        scaled_timestep: [B] scaled timestep (or None if no conditioning)
        tile_latent_h: tile height in latent space (default 4)
        tile_latent_w: tile width in latent space (default 16)
        overlap_latent_h: overlap in latent H pixels (default 1)
        overlap_latent_w: overlap in latent W pixels (default 0)
        spatial_scale: latent-to-pixel spatial scale (32 for LTX-2.3)
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
    assert stride_h > 0, f"stride_h={stride_h} must be > 0"
    assert stride_w > 0, f"stride_w={stride_w} must be > 0"

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
                if scaled_timestep is not None:
                    tile_output = compiled_model(tile_latent, scaled_timestep)
                else:
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

            if h_end_lat < H_lat and overlap_h_pixels > 0:
                blend_h[-overlap_h_pixels:] = torch.linspace(
                    1, 0, overlap_h_pixels + 2
                )[1:-1]
            if w_end_lat < W_lat and overlap_w_pixels > 0:
                blend_w[-overlap_w_pixels:] = torch.linspace(
                    1, 0, overlap_w_pixels + 2
                )[1:-1]

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
        compiled_dir: Directory containing the compiled model files

    Returns:
        compiled: The loaded TensorParallelNeuronModel
    """
    import neuronx_distributed

    return neuronx_distributed.trace.parallel_model_load(compiled_dir)
