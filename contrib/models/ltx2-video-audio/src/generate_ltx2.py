#!/usr/bin/env python3
"""
LTX-2 Video+Audio Generation on AWS Neuron
============================================
Example script demonstrating end-to-end LTX-2 inference on Neuron.

Compiles (or loads) the DiT transformer backbone with TP=4 on trn2.3xlarge,
then generates video+audio from a text prompt.

Hardware: trn2.3xlarge (1 NeuronDevice, 4 logical cores with LNC=2)
Memory:   ~9.4 GB HBM per TP rank (fits in 24 GB per core)

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  python generate_ltx2.py --prompt "A golden retriever runs across a sunny meadow"
  python generate_ltx2.py --compile-only  # Compile without generating
  python generate_ltx2.py --load-only     # Skip compilation, load existing

Requirements:
  - neuronx-distributed-inference (NxDI)
  - diffusers >= 0.37.0.dev0 (from git main, with LTX-2 support)
  - Neuron SDK 2.27+
"""

import argparse
import json
import os
import time

import torch

from nxdi.application import NeuronLTX2Application
from nxdi.modeling_ltx2 import LTX2BackboneInferenceConfig, replace_sdpa_with_bmm

try:
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
except ImportError:
    raise ImportError(
        "neuronx_distributed_inference is required. "
        "Activate the Neuron venv: source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate"
    )


# Default configuration for trn2.3xlarge
DEFAULT_CONFIG = {
    "model_path": "Lightricks/LTX-2",
    "height": 384,
    "width": 512,
    "num_frames": 25,
    "num_inference_steps": 8,
    "tp_degree": 4,
    "world_size": 4,
    "compile_dir": "/tmp/ltx2_nxdi/compiler_workdir/",
    "output_dir": "/tmp/ltx2_nxdi/output/",
    "prompt": "A golden retriever puppy runs across a sunny green meadow, "
    "its ears flapping in the wind. The camera follows from a low angle. "
    "Birds chirp in the background.",
    "seed": 42,
    "max_sequence_length": 1024,
    "frame_rate": 24.0,
}


def create_ltx2_config(args):
    """Create NxDI configuration for the LTX-2 transformer backbone."""

    # Latent dimensions (derived from video dims and VAE compression)
    latent_num_frames = (args.num_frames - 1) // 8 + 1  # temporal compression
    latent_height = args.height // 32  # spatial compression
    latent_width = args.width // 32

    video_seq = latent_num_frames * latent_height * latent_width
    audio_num_frames = round(
        (args.num_frames / args.frame_rate) * 24.97  # audio tokens
    )

    # Load transformer config from HuggingFace
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(args.model_path, "transformer/config.json")
    with open(config_path) as f:
        hf_config = json.load(f)

    num_heads = hf_config["num_attention_heads"]
    head_dim = hf_config["attention_head_dim"]
    inner_dim = num_heads * head_dim
    audio_num_heads = hf_config["audio_num_attention_heads"]
    audio_head_dim = hf_config["audio_attention_head_dim"]
    audio_inner_dim = audio_num_heads * audio_head_dim
    audio_ca_dim = hf_config.get("audio_cross_attention_dim", audio_inner_dim)
    caption_channels = hf_config.get("caption_channels", 3840)

    # NeuronConfig for the backbone
    backbone_neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        world_size=args.world_size,
        torch_dtype=torch.bfloat16,
    )

    # LTX2 backbone inference config
    backbone_config = LTX2BackboneInferenceConfig(
        neuron_config=backbone_neuron_config,
        num_layers=hf_config["num_layers"],
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        inner_dim=inner_dim,
        audio_num_attention_heads=audio_num_heads,
        audio_attention_head_dim=audio_head_dim,
        audio_inner_dim=audio_inner_dim,
        audio_cross_attention_dim=audio_ca_dim,
        caption_channels=caption_channels,
        video_seq=video_seq,
        audio_seq=audio_num_frames,
        text_seq=args.max_sequence_length,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    # Store HF config dict so the backbone model can auto-build from diffusers
    backbone_config.hf_config_dict = hf_config

    print(
        f"  Config: {hf_config['num_layers']} layers, {num_heads} heads, "
        f"head_dim={head_dim}, inner_dim={inner_dim}"
    )
    print(
        f"  Audio: {audio_num_heads} heads, audio_head_dim={audio_head_dim}, "
        f"audio_inner_dim={audio_inner_dim}"
    )
    print(
        f"  Video: {args.height}x{args.width}, {args.num_frames} frames → "
        f"latent {latent_height}x{latent_width}x{latent_num_frames} = {video_seq} tokens"
    )
    print(f"  Audio: {audio_num_frames} tokens")
    print(f"  TP={args.tp_degree}, world_size={args.world_size}")

    return backbone_config, hf_config


def run_generate(args):
    """Run the full LTX-2 generation pipeline."""
    print("=" * 60)
    print("LTX-2 Video+Audio Generation on Neuron")
    print("=" * 60)

    t_total = time.time()

    # Replace SDPA before any model loading
    replace_sdpa_with_bmm()

    # Create configuration
    print("\n[1/4] Creating configuration...")
    backbone_config, hf_config = create_ltx2_config(args)

    # Create application
    print("\n[2/4] Creating NeuronLTX2Application...")
    app = NeuronLTX2Application(
        model_path=args.model_path,
        backbone_config=backbone_config,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
    )

    # Compile or load
    if args.compile_only or not os.path.exists(
        os.path.join(args.compile_dir, "transformer/model.pt")
    ):
        print("\n[3/4] Compiling transformer backbone...")
        t0 = time.time()
        app.compile(args.compile_dir)
        print(f"  Compiled in {time.time() - t0:.1f}s")
        if args.compile_only:
            print(
                "\nCompilation complete. Use --load-only to skip compilation next time."
            )
            return

    print("\n[3/4] Loading compiled transformer...")
    t0 = time.time()
    app.load(args.compile_dir)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Generate
    print("\n[4/4] Generating video+audio...")
    print(f"  Prompt: {args.prompt[:80]}...")
    print(
        f"  Resolution: {args.width}x{args.height}, {args.num_frames} frames, "
        f"{args.num_inference_steps} steps"
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    t0 = time.time()
    output = app(
        prompt=args.prompt,
        generator=generator,
        output_type="pil",
        max_sequence_length=args.max_sequence_length,
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    frames = output.frames[0]
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
    print(f"  Saved {len(frames)} frames to {frames_dir}/")

    try:
        from diffusers.utils import export_to_video

        video_path = os.path.join(args.output_dir, "output.mp4")
        export_to_video(frames, video_path, fps=int(args.frame_rate))
        print(f"  Video: {video_path}")
    except Exception as e:
        print(f"  Video export failed: {e}")

    # Summary
    total_time = time.time() - t_total
    print(f"\nSummary:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Output frames: {len(frames)}")
    print(f"  Output dir: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2 Video Generation on Neuron")
    parser.add_argument("-p", "--prompt", type=str, default=DEFAULT_CONFIG["prompt"])
    parser.add_argument("--height", type=int, default=DEFAULT_CONFIG["height"])
    parser.add_argument("--width", type=int, default=DEFAULT_CONFIG["width"])
    parser.add_argument("--num-frames", type=int, default=DEFAULT_CONFIG["num_frames"])
    parser.add_argument(
        "--num-inference-steps", type=int, default=DEFAULT_CONFIG["num_inference_steps"]
    )
    parser.add_argument("--tp-degree", type=int, default=DEFAULT_CONFIG["tp_degree"])
    parser.add_argument("--world-size", type=int, default=DEFAULT_CONFIG["world_size"])
    parser.add_argument("--model-path", type=str, default=DEFAULT_CONFIG["model_path"])
    parser.add_argument(
        "--compile-dir", type=str, default=DEFAULT_CONFIG["compile_dir"]
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_CONFIG["max_sequence_length"]
    )
    parser.add_argument(
        "--frame-rate", type=float, default=DEFAULT_CONFIG["frame_rate"]
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Only compile, don't generate"
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Skip compilation, load existing compiled model",
    )

    args = parser.parse_args()
    run_generate(args)
