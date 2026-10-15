"""
HunyuanVideo VAE — Tiled Neuron Decode Runtime
===============================================
Loads the compiled VAE tile decoder and decodes full-resolution latents
using spatial tiling with overlap blending.

HunyuanVideo VAE specifics:
  - Spatial scale: 16x (not 32x like LTX-2)
  - Temporal scale: 4x (T_out = (T_lat - 1) * 4 + 1)
  - Latent channels: 32
  - Output channels: 3

480p (848x480, 5 frames):
  - Latent: [1, 32, 2, 30, 53]
  - Tile 8x8 latent, overlap 2: stride=6 -> 5x9 = 45 tiles
  - Tile 8x8 latent, overlap 0: stride=8 -> 4x7 = 28 tiles

Usage (standalone):
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    python ./tiled_vae_decode.py

Usage (as library):
    from tiled_vae_decode import TiledVAEDecoderNeuron
    decoder = TiledVAEDecoderNeuron("./compiled/vae_decoder_neuron")
    output = decoder.decode(latent)  # [1, 32, 2, 30, 53] -> [1, 3, 5, 480, 848]
"""

import os
import sys
import time
import json
import argparse

import torch
import torch.nn.functional as F

try:
    import torch_neuronx  # Required for loading traced models
except ImportError:
    pass

MODELS_DIR = os.environ.get("HUNYUAN_MODELS_DIR", "./models")
COMPILED_DIR = os.environ.get("HUNYUAN_COMPILED_DIR", "./compiled")

# Spatial scale for HunyuanVideo VAE
FFACTOR_SPATIAL = 16
FFACTOR_TEMPORAL = 4


def create_blend_mask_1d(length, blend_left, blend_right):
    """1D linear blend mask: ramp up at left, 1 in middle, ramp down at right."""
    mask = torch.ones(length)
    if blend_left > 0:
        ramp = torch.linspace(0, 1, blend_left + 2)[1:-1]
        mask[:blend_left] = ramp
    if blend_right > 0:
        ramp = torch.linspace(1, 0, blend_right + 2)[1:-1]
        mask[-blend_right:] = ramp
    return mask


class TiledVAEDecoderNeuron:
    """Tiled VAE decode using a Neuron-compiled tile decoder.

    Splits latent into overlapping spatial tiles, decodes each on Neuron,
    and blends in pixel space.
    """

    def __init__(
        self, compiled_dir, tile_latent_h=8, tile_latent_w=8, overlap_h=2, overlap_w=2
    ):
        """
        Args:
            compiled_dir: Directory with vae_decoder.pt (torch_neuronx traced)
                          or nxd_model.pt + weights/ (ModelBuilder)
            tile_latent_h: Tile height in latent space
            tile_latent_w: Tile width in latent space
            overlap_h: Overlap in latent H dimension
            overlap_w: Overlap in latent W dimension
        """
        self.tile_h = tile_latent_h
        self.tile_w = tile_latent_w
        self.overlap_h = overlap_h
        self.overlap_w = overlap_w
        self.stride_h = tile_latent_h - overlap_h
        self.stride_w = tile_latent_w - overlap_w
        assert self.stride_h > 0, f"stride_h must be > 0, got {self.stride_h}"
        assert self.stride_w > 0, f"stride_w must be > 0, got {self.stride_w}"

        self.model, self._is_traced_model = self._load(compiled_dir)
        self._warmed_up = False

    def _load(self, compiled_dir):
        """Load compiled VAE model.

        Supports two formats:
          1. torch_neuronx.trace -> vae_decoder.pt (direct call with tensor args)
          2. ModelBuilder -> nxd_model.pt + weights/ (NxD model, call with list+dict)

        Returns:
            (model, is_traced) tuple
        """
        traced_path = os.path.join(compiled_dir, "vae_decoder.pt")
        nxd_path = os.path.join(compiled_dir, "nxd_model.pt")

        if os.path.exists(traced_path):
            # torch_neuronx.trace format
            print(f"Loading traced VAE from {traced_path}...")
            model = torch.jit.load(traced_path)
            print(f"VAE decoder loaded (torch_neuronx.trace format).")
            return model, True
        elif os.path.exists(nxd_path):
            # ModelBuilder format
            from safetensors.torch import load_file

            weights_dir = os.path.join(compiled_dir, "weights")

            print(f"Loading NxD VAE from {compiled_dir}...")
            nxd_model = torch.jit.load(nxd_path)
            weights_files = sorted(
                [f for f in os.listdir(weights_dir) if f.endswith(".safetensors")]
            )
            sharded_weights = [
                load_file(os.path.join(weights_dir, wf)) for wf in weights_files
            ]
            nxd_model.set_weights(sharded_weights)
            nxd_model.to_neuron()
            print(f"VAE decoder loaded (ModelBuilder format).")
            return nxd_model, False
        else:
            raise FileNotFoundError(
                f"No VAE model found in {compiled_dir}. "
                f"Expected vae_decoder.pt or nxd_model.pt"
            )

    def warmup(self, T_lat=2, n=3):
        """Run warmup iterations."""
        dummy = torch.randn(
            1, 32, T_lat, self.tile_h, self.tile_w, dtype=torch.bfloat16
        )
        for i in range(n):
            self._decode_tile(dummy)
        self._warmed_up = True
        print(f"VAE warmup done ({n} iterations).")

    def _decode_tile(self, tile_z):
        """Decode a single tile on Neuron."""
        with torch.no_grad():
            if self._is_traced_model:
                # torch_neuronx.trace format: direct call
                out = self.model(tile_z)
            else:
                # ModelBuilder format: list + dict
                out = self.model([tile_z], {})
        if isinstance(out, tuple):
            return out[0]
        return out

    def decode(self, z, verbose=True):
        """Decode full latent using spatial tiling with overlap blending.

        Args:
            z: [1, 32, T_lat, H_lat, W_lat] latent tensor (bfloat16)
            verbose: Print progress

        Returns:
            [1, 3, T_out, H_out, W_out] decoded video tensor (float32)
        """
        if not self._warmed_up:
            self.warmup(T_lat=z.shape[2])

        B, C, T, H_lat, W_lat = z.shape
        assert B == 1, "Batch size must be 1"

        H_out = H_lat * FFACTOR_SPATIAL
        W_out = W_lat * FFACTOR_SPATIAL
        T_out = (T - 1) * FFACTOR_TEMPORAL + 1

        tile_h_px = self.tile_h * FFACTOR_SPATIAL
        tile_w_px = self.tile_w * FFACTOR_SPATIAL
        overlap_h_px = self.overlap_h * FFACTOR_SPATIAL
        overlap_w_px = self.overlap_w * FFACTOR_SPATIAL

        # Compute tile start positions
        h_starts = []
        pos = 0
        while pos < H_lat:
            h_starts.append(min(pos, max(0, H_lat - self.tile_h)))
            if pos + self.tile_h >= H_lat:
                break
            pos += self.stride_h
        h_starts = sorted(set(h_starts))

        w_starts = []
        pos = 0
        while pos < W_lat:
            w_starts.append(min(pos, max(0, W_lat - self.tile_w)))
            if pos + self.tile_w >= W_lat:
                break
            pos += self.stride_w
        w_starts = sorted(set(w_starts))

        n_h = len(h_starts)
        n_w = len(w_starts)
        total_tiles = n_h * n_w

        if verbose:
            print(
                f"  Tiling: {n_h}x{n_w} = {total_tiles} tiles "
                f"(latent {H_lat}x{W_lat}, tile {self.tile_h}x{self.tile_w}, "
                f"overlap h={self.overlap_h} w={self.overlap_w})"
            )

        # Output accumulator
        output_accum = torch.zeros(1, 3, T_out, H_out, W_out, dtype=torch.float32)
        weight_accum = torch.zeros(1, 1, 1, H_out, W_out, dtype=torch.float32)

        tile_times = []

        for ti, h_start in enumerate(h_starts):
            for wi, w_start in enumerate(w_starts):
                tile_idx = ti * n_w + wi + 1
                h_end = h_start + self.tile_h
                w_end = w_start + self.tile_w

                # Extract tile
                tile_z = z[:, :, :, h_start:h_end, w_start:w_end].contiguous()

                # Decode on Neuron
                t0 = time.time()
                tile_out = self._decode_tile(tile_z)
                dt = time.time() - t0
                tile_times.append(dt)

                # Output pixel coordinates
                h_px = h_start * FFACTOR_SPATIAL
                w_px = w_start * FFACTOR_SPATIAL

                # Actual output size (may be smaller than tile_h_px for edge tiles)
                out_h = tile_out.shape[3]
                out_w = tile_out.shape[4]

                # Create blend mask
                blend_left_h = overlap_h_px if h_start > 0 else 0
                blend_right_h = overlap_h_px if h_end < H_lat else 0
                blend_left_w = overlap_w_px if w_start > 0 else 0
                blend_right_w = overlap_w_px if w_end < W_lat else 0

                bh = create_blend_mask_1d(out_h, blend_left_h, blend_right_h)
                bw = create_blend_mask_1d(out_w, blend_left_w, blend_right_w)
                blend_2d = bh.unsqueeze(1) * bw.unsqueeze(0)  # [H, W]
                blend_2d = (
                    blend_2d.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                )  # [1,1,1,H,W]

                output_accum[:, :, :, h_px : h_px + out_h, w_px : w_px + out_w] += (
                    tile_out.float() * blend_2d
                )
                weight_accum[:, :, :, h_px : h_px + out_h, w_px : w_px + out_w] += (
                    blend_2d
                )

                if verbose:
                    print(
                        f"    Tile {tile_idx:2d}/{total_tiles}: "
                        f"lat[{h_start}:{h_end}, {w_start}:{w_end}] "
                        f"-> px[{h_px}:{h_px + out_h}, {w_px}:{w_px + out_w}], "
                        f"{dt * 1000:.0f}ms"
                    )

        output = output_accum / weight_accum.clamp(min=1e-6)

        total_decode = sum(tile_times)
        if verbose:
            avg_tile = total_decode / total_tiles * 1000
            print(
                f"\n  Total decode: {total_decode:.2f}s "
                f"({total_tiles} tiles, avg {avg_tile:.0f}ms/tile)"
            )

        return output, {
            "total_s": round(total_decode, 2),
            "n_tiles": total_tiles,
            "avg_tile_ms": round(total_decode / total_tiles * 1000, 1),
            "tile_times_ms": [round(t * 1000, 1) for t in tile_times],
        }


def main():
    parser = argparse.ArgumentParser(description="HunyuanVideo Tiled VAE Decode")
    parser.add_argument(
        "--compiled-dir", type=str, default=f"{COMPILED_DIR}/vae_decoder_neuron"
    )
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--overlap-h", type=int, default=2)
    parser.add_argument("--overlap-w", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validate-cpu",
        action="store_true",
        help="Run CPU decode and compare accuracy",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16
    T_lat, H_lat, W_lat = 2, 30, 53  # 480p_5f

    print("=" * 60)
    print("HunyuanVideo VAE — Tiled Neuron Decode")
    print("=" * 60)
    print(f"  Latent: [1, 32, {T_lat}, {H_lat}, {W_lat}]")
    print(f"  Output: [1, 3, {(T_lat - 1) * 4 + 1}, {H_lat * 16}, {W_lat * 16}]")
    print(
        f"  Tile:   {args.tile_h}x{args.tile_w} latent, overlap h={args.overlap_h} w={args.overlap_w}"
    )
    print()

    # Create test latent
    torch.manual_seed(args.seed)
    z = torch.randn(1, 32, T_lat, H_lat, W_lat, dtype=dtype)

    # Load and decode
    decoder = TiledVAEDecoderNeuron(
        args.compiled_dir,
        tile_latent_h=args.tile_h,
        tile_latent_w=args.tile_w,
        overlap_h=args.overlap_h,
        overlap_w=args.overlap_w,
    )

    # Warmup
    decoder.warmup(T_lat=T_lat)

    # Tiled decode
    print("\nTiled decode (run 1 — may include additional warmup)...")
    output1, stats1 = decoder.decode(z)
    print(
        f"Output: {list(output1.shape)}, range [{output1.min():.3f}, {output1.max():.3f}]"
    )

    # Benchmark (3 runs)
    print("\nBenchmark (3 runs)...")
    run_times = []
    for i in range(3):
        t0 = time.time()
        output, stats = decoder.decode(z, verbose=False)
        dt = time.time() - t0
        run_times.append(dt)
        print(
            f"  Run {i + 1}: {dt:.2f}s ({stats['n_tiles']} tiles, {stats['avg_tile_ms']:.0f}ms/tile)"
        )

    avg_total = sum(run_times) / len(run_times)
    min_total = min(run_times)

    # CPU comparison
    cpu_time = None
    cos_sim = None
    if args.validate_cpu:
        print("\nCPU reference decode...")

        # Apply patch for CPU too
        sys.path.insert(0, os.environ.get("HUNYUAN_REPO_DIR", "./HunyuanVideo-1.5"))
        os.environ["HUNYUAN_ATTN_MODE"] = "torch"

        # Import and patch
        from compile_vae_neuron import apply_all_patches

        apply_all_patches()

        from hyvideo.models.autoencoders.hunyuanvideo_15_vae import AutoencoderKLConv3D

        vae = AutoencoderKLConv3D.from_pretrained(
            f"{MODELS_DIR}/HunyuanVideo-1.5/vae",
            torch_dtype=dtype,
        ).eval()

        t0_cpu = time.time()
        with torch.no_grad():
            cpu_out = vae.decoder(z)
        cpu_time = time.time() - t0_cpu
        print(f"  CPU decode: {cpu_time:.1f}s")

        # Compare (note: tiled vs direct may differ slightly due to blending)
        cos_sim = F.cosine_similarity(
            cpu_out.float().flatten().unsqueeze(0),
            output.float().flatten().unsqueeze(0),
        ).item()
        mae = (cpu_out.float() - output.float()).abs().mean().item()
        print(f"  cos_sim={cos_sim:.6f}, mae={mae:.6f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Resolution:    {H_lat * 16}x{W_lat * 16}, {(T_lat - 1) * 4 + 1} frames")
    print(
        f"  Tiles:         {stats['n_tiles']} ({args.tile_h}x{args.tile_w} latent, "
        f"overlap h={args.overlap_h} w={args.overlap_w})"
    )
    print(f"  Avg tile:      {stats['avg_tile_ms']:.0f}ms")
    print(f"  Total decode:  {avg_total:.2f}s avg, {min_total:.2f}s min")
    if cpu_time:
        print(f"  CPU decode:    {cpu_time:.1f}s")
        print(f"  Speedup:       {cpu_time / min_total:.2f}x")
    if cos_sim:
        print(f"  Accuracy:      cos_sim={cos_sim:.6f}")



    # Save results
    results = {
        "tile_latent": f"{args.tile_h}x{args.tile_w}",
        "overlap": f"h={args.overlap_h}, w={args.overlap_w}",
        "n_tiles": stats["n_tiles"],
        "avg_tile_ms": stats["avg_tile_ms"],
        "total_decode_avg_s": round(avg_total, 2),
        "total_decode_min_s": round(min_total, 2),
        "cpu_decode_s": round(cpu_time, 1) if cpu_time else None,
        "speedup_vs_cpu": round(cpu_time / min_total, 2) if cpu_time else None,
        "cos_sim": round(cos_sim, 6) if cos_sim else None,
    }
    results_path = os.path.join(args.compiled_dir, "tiled_decode_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
