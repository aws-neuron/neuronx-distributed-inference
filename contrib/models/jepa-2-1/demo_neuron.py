"""
V-JEPA 2.1 Neuron smoke test.

Runs a video through the pretrained ViT-B encoder on both CPU (FP32)
and Neuron (BF16), then compares the feature embeddings.

Requires: trn2/inf2 instance with torch-neuronx.

Usage (on Neuron instance):
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python demo_neuron.py                        # synthetic video (no deps)
    python demo_neuron.py path/to/video.mp4      # your own video (needs decord, pillow)
"""

import sys
import os
import time

import numpy as np
import torch
import torch_neuronx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from modeling_jepa21 import build_vjepa21_encoder


def make_synthetic_video(num_frames=16, size=384):
    """Generate synthetic video: moving circle on gradient background. Returns (1, 3, T, H, W)."""
    frames = []
    for i in range(num_frames):
        frame = np.zeros((size, size, 3), dtype=np.float32)
        frame[:, :, 2] = np.linspace(0, 0.3, size).reshape(1, -1)
        cx = int(size * (0.2 + 0.6 * i / num_frames))
        y, x = np.ogrid[:size, :size]
        mask = ((x - cx)**2 + (y - size // 2)**2) < (size // 10)**2
        frame[mask] = 1.0
        frames.append(frame)
    # (T, H, W, 3) -> (3, T, H, W)
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)
    return video.unsqueeze(0)  # (1, 3, T, H, W)


def load_video_tensor(path, num_frames=16, size=384):
    """Load video file as (1, 3, T, H, W) tensor. Needs decord and pillow."""
    from decord import VideoReader
    from PIL import Image
    vr = VideoReader(path)
    indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    processed = []
    for f in frames:
        img = Image.fromarray(f).resize((size, size))
        processed.append(np.array(img, dtype=np.float32) / 255.0)
    video = torch.from_numpy(np.stack(processed)).permute(3, 0, 1, 2)
    return video.unsqueeze(0)


def main():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print(f"Video: {sys.argv[1]}")
        video = load_video_tensor(sys.argv[1])
    else:
        print("Using synthetic video (moving circle, no dependencies needed)")
        video = make_synthetic_video()
    print(f"Input shape: {video.shape}")

    # --- CPU reference (FP32) ---
    print("\nLoading pretrained ViT-B (CPU, FP32)...")
    encoder = build_vjepa21_encoder(
        arch="vit_base", img_size=384, num_frames=16,
        pretrained=True, use_sdpa=False,
    )
    encoder.eval()

    with torch.no_grad():
        cpu_out = encoder(video)
    print(f"CPU output: shape={cpu_out.shape}, norm={cpu_out.float().norm():.1f}")

    # --- Neuron (BF16) ---
    print("\nTracing for Neuron (BF16)...")
    encoder.bfloat16()
    video_bf16 = video.bfloat16()

    t0 = time.time()
    traced = torch_neuronx.trace(encoder, video_bf16, compiler_args=["--auto-cast", "none"])
    compile_time = time.time() - t0
    print(f"Compilation: {compile_time:.1f}s")

    # Warmup
    for _ in range(3):
        traced(video_bf16)

    # Timed run
    t0 = time.time()
    neuron_out = traced(video_bf16)
    latency = (time.time() - t0) * 1000
    print(f"Neuron output: shape={neuron_out.shape}, norm={neuron_out.float().norm():.1f}")
    print(f"Latency: {latency:.1f}ms")

    # --- Compare ---
    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_out.float().flatten().unsqueeze(0),
        neuron_out.float().flatten().unsqueeze(0),
    ).item()

    status = "PASS" if cos_sim > 0.999 else "FAIL"
    print(f"\nCosine similarity (CPU FP32 vs Neuron BF16): {cos_sim:.6f}  [{status}]")


if __name__ == "__main__":
    main()
