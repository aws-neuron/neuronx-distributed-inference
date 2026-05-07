#!/usr/bin/env python3
"""
Generate reference video on GPU using same settings as Neuron E2E test.

Settings must match the Neuron pipeline (uses pipeline defaults):
  - Prompt: "A golden retriever puppy runs across a sunny green meadow, ..."
  - Resolution: 512x384
  - Frames: 25
  - Steps: 8
  - Guidance scale: 4.0 (pipeline default)
  - Seed: 42
  - Max sequence length: 1024 (pipeline default)
  - Model: Lightricks/LTX-2

Usage:
  pip install diffusers transformers accelerate imageio-ffmpeg
  python gpu_generate.py
"""

import os
import time
import torch

print("=" * 60)
print("LTX-2 GPU Reference Generation")
print("=" * 60)

# 1. Load pipeline
print("\n[1/3] Loading LTX2Pipeline on GPU...")
t0 = time.time()
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2",
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload()
print(f"  Loaded in {time.time() - t0:.1f}s (with sequential CPU offload)")

# Print GPU info
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_mem / 1e9:.1f} GB")

# 2. Generate — EXACT same settings as Neuron E2E test
PROMPT = (
    "A golden retriever puppy runs across a sunny green meadow, "
    "its ears flapping in the wind. The camera follows from a low angle. "
    "Birds chirp in the background."
)
HEIGHT = 384
WIDTH = 512
NUM_FRAMES = 25
NUM_STEPS = 8

print(f"\n[2/3] Generating video...")
print(f"  Prompt: {PROMPT[:80]}...")
print(f"  {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames, {NUM_STEPS} steps, seed=42")

generator = torch.Generator(device="cpu").manual_seed(42)

t0 = time.time()
output = pipe(
    prompt=PROMPT,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    num_inference_steps=NUM_STEPS,
    generator=generator,
    output_type="pil",
)
gen_time = time.time() - t0
print(f"  Generated in {gen_time:.1f}s")

# 3. Save output
print("\n[3/3] Saving output...")
output_dir = "/tmp/ltx2_gpu_output/"
os.makedirs(output_dir, exist_ok=True)

frames = output.frames[0]
frames_dir = os.path.join(output_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)
for i, frame in enumerate(frames):
    frame.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
print(f"  Saved {len(frames)} frames to {frames_dir}/")

try:
    from diffusers.utils import export_to_video

    video_path = os.path.join(output_dir, "output.mp4")
    export_to_video(frames, video_path, fps=24)
    size_kb = os.path.getsize(video_path) / 1024
    print(f"  Video: {video_path} ({size_kb:.1f} KB)")
except Exception as e:
    print(f"  Video export failed: {e}")

print(f"\n{'=' * 60}")
print(f"Summary:")
print(f"  Generation time: {gen_time:.1f}s")
print(f"  Output frames: {len(frames)}")
print(f"  Output dir: {output_dir}")
print(f"{'=' * 60}")
