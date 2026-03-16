#!/usr/bin/env python3
"""
LTX-2 E2E: Neuron Text Encoder + Neuron DiT Backbone + Neuron VAE Decoder
=========================================================================
All three components can run on Neuron (TP=4). They coexist on the same
4 NeuronCores and execute sequentially:
  text encoding -> denoising -> VAE decode (tiled)

The VAE decoder is optional — if LTX2_VAE_COMPILE_DIR is not set or the
directory doesn't exist, falls back to CPU VAE decode (the Diffusers default).

Uses correct pipeline defaults:
  - guidance_scale=4.0 (CFG with batch-splitting for Neuron)
  - max_sequence_length=1024

Prerequisites:
  1. DiT compiled:    python ../src/compile_gemma3.py  (or use notebook)
  2. Gemma3 compiled:  (produces tp_0.pt ... tp_3.pt)
  3. Gemma3 sharded:   python ../src/shard_gemma3_weights.py
  4. VAE compiled (optional): python ../src/compile_vae.py

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
    python neuron_e2e.py
"""

import gc
import json
import os
import sys
import time

import torch

os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modeling_ltx2 import (
    LTX2BackboneInferenceConfig,
    NeuronLTX2BackboneApplication,
    replace_sdpa_with_bmm,
)
from pipeline import NeuronTransformerWrapper
from neuronx_distributed_inference.models.config import NeuronConfig

# Directories — adjust these to your instance layout
DIT_COMPILE_DIR = os.environ.get(
    "LTX2_DIT_COMPILE_DIR", "/home/ubuntu/ltx2_nxdi_compiled_1024/"
)
GEMMA3_COMPILE_DIR = os.environ.get(
    "LTX2_GEMMA3_COMPILE_DIR", "/home/ubuntu/gemma3_encoder_compiled_1024/"
)
GEMMA3_SHARDED_DIR = os.environ.get(
    "LTX2_GEMMA3_SHARDED_DIR", "/home/ubuntu/gemma3_encoder_sharded/"
)
VAE_COMPILE_DIR = os.environ.get(
    "LTX2_VAE_COMPILE_DIR", "/home/ubuntu/ltx2_vae_tp4_128x512/"
)
VAE_TILE_LATENT_H = int(os.environ.get("LTX2_VAE_TILE_H", "8"))
VAE_TILE_LATENT_W = int(os.environ.get("LTX2_VAE_TILE_W", "8"))
VAE_OVERLAP_H = int(os.environ.get("LTX2_VAE_OVERLAP_H", "2"))
VAE_OVERLAP_W = int(os.environ.get("LTX2_VAE_OVERLAP_W", "2"))
OUTPUT_DIR = os.environ.get("LTX2_OUTPUT_DIR", "/home/ubuntu/ltx2_output/")
TP_DEGREE = 4
HEIGHT, WIDTH, NUM_FRAMES = 512, 768, 121
NUM_STEPS = 40
PROMPT = (
    "A close-up shot of a young waitress in a retro 1950s diner, her warm brown eyes "
    "meeting the camera with a gentle smile. She wears a black polka-dot dress with an "
    "elegant cream lace collar, her reddish-brown hair styled in an elaborate updo with "
    "delicate curls framing her freckled face. Soft, warm light from overhead fixtures "
    "illuminates her features as she stands behind a yellow counter. The camera begins "
    "slightly to her side, then slowly pushes in toward her face, revealing the subtle "
    "rosy blush on her cheeks. In the blurred background, the soft teal walls and a "
    'glowing red "Diner" sign create a nostalgic atmosphere. The ambient sounds of '
    "clinking dishes, distant conversations, and the gentle hum of a jukebox fill the "
    'air. She tilts her head slightly and says in a friendly, warm voice: "Welcome to '
    "Rosie's. What can I get for you today?\" The mood is inviting, timeless, and full "
    "of classic American diner charm."
)
SEED = 10


# -- Neuron Gemma3 Text Encoder -----------------------------------------------


def load_neuron_gemma3(sharded_dir, compile_dir, tp_degree):
    """Load TP=4 compiled Gemma3 encoder with pre-sharded weights."""
    import torch_neuronx
    from neuronx_distributed.trace.trace import (
        replace_weights,
        TensorParallelNeuronModel,
    )

    models = []
    for rank in range(tp_degree):
        t0 = time.time()
        rank_ckpt_path = os.path.join(sharded_dir, f"rank_{rank}.pt")
        ckpt = torch.load(rank_ckpt_path, weights_only=True)

        neff_path = os.path.join(compile_dir, f"tp_{rank}.pt")
        with torch_neuronx.contexts.disable_nrt_load():
            traced_model = torch.jit.load(neff_path)

        replace_weights(traced_model, ckpt)
        print(f"    [Gemma3 rank {rank}] {time.time() - t0:.1f}s")
        models.append(traced_model)
        del ckpt
        gc.collect()

    compiled = TensorParallelNeuronModel(models)
    print(f"    Gemma3: all {tp_degree} ranks loaded")
    return compiled


class NeuronTextEncoderOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class NeuronTextEncoderWrapper:
    """Drop-in replacement for Gemma3ForConditionalGeneration."""

    def __init__(self, compiled_gemma3, dtype=torch.bfloat16):
        self.compiled_model = compiled_gemma3
        self.dtype = dtype
        self._device = torch.device("cpu")
        self.config = type("Config", (), {"output_hidden_states": True})()

    def __call__(
        self, input_ids=None, attention_mask=None, output_hidden_states=True, **kwargs
    ):
        with torch.no_grad():
            stacked = self.compiled_model(input_ids, attention_mask)
            num_states = stacked.shape[-1]
            hidden_states = tuple(stacked[:, :, :, i] for i in range(num_states))
        return NeuronTextEncoderOutput(hidden_states=hidden_states)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    @property
    def device(self):
        return self._device


# -- Main Pipeline -------------------------------------------------------------


def main():
    replace_sdpa_with_bmm()

    print("=" * 60)
    print(f"LTX-2 E2E: Neuron Text Encoder + DiT + VAE (TP={TP_DEGREE})")
    print("=" * 60)
    t_total = time.time()

    # Check if VAE compiled dir exists
    use_neuron_vae = os.path.isdir(VAE_COMPILE_DIR) and os.path.exists(
        os.path.join(VAE_COMPILE_DIR, "tp_0.pt")
    )
    if use_neuron_vae:
        print(f"  Neuron VAE: {VAE_COMPILE_DIR}")
    else:
        print(f"  Neuron VAE: NOT FOUND at {VAE_COMPILE_DIR} (using CPU fallback)")

    # 1. Create DiT config
    print("\n[1/7] Creating DiT config...")
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download("Lightricks/LTX-2", "transformer/config.json")
    with open(config_path) as f:
        hf_config = json.load(f)

    num_heads = hf_config["num_attention_heads"]
    head_dim = hf_config["attention_head_dim"]
    inner_dim = num_heads * head_dim
    audio_num_heads = hf_config["audio_num_attention_heads"]
    audio_head_dim = hf_config["audio_attention_head_dim"]
    audio_inner_dim = audio_num_heads * audio_head_dim
    audio_ca_dim = hf_config.get("audio_cross_attention_dim", audio_inner_dim)

    latent_num_frames = (NUM_FRAMES - 1) // 8 + 1
    latent_height = HEIGHT // 32
    latent_width = WIDTH // 32
    video_seq = latent_num_frames * latent_height * latent_width
    audio_num_frames = round((NUM_FRAMES / 24.0) * 24.97)

    backbone_neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        world_size=TP_DEGREE,
        torch_dtype=torch.bfloat16,
    )

    config = LTX2BackboneInferenceConfig(
        neuron_config=backbone_neuron_config,
        num_layers=hf_config["num_layers"],
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        inner_dim=inner_dim,
        audio_num_attention_heads=audio_num_heads,
        audio_attention_head_dim=audio_head_dim,
        audio_inner_dim=audio_inner_dim,
        audio_cross_attention_dim=audio_ca_dim,
        caption_channels=hf_config.get("caption_channels", 3840),
        video_seq=video_seq,
        audio_seq=audio_num_frames,
        text_seq=1024,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
    )
    config.hf_config_dict = hf_config
    print(f"  48 blocks, TP={TP_DEGREE}, {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames")

    # 2. Load diffusers pipeline (CPU)
    print("\n[2/7] Loading Diffusers LTX2Pipeline (CPU)...")
    t0 = time.time()
    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # 3. Load Neuron DiT backbone
    print(f"\n[3/7] Loading Neuron DiT backbone from {DIT_COMPILE_DIR}...")
    from huggingface_hub import snapshot_download

    local_transformer_path = snapshot_download(
        "Lightricks/LTX-2", allow_patterns=["transformer/*"]
    )
    local_transformer_path = os.path.join(local_transformer_path, "transformer")

    cpu_transformer = pipe.transformer

    t0 = time.time()
    backbone_app = NeuronLTX2BackboneApplication(
        model_path=local_transformer_path, config=config
    )
    backbone_app.load(DIT_COMPILE_DIR)
    print(f"  DiT loaded in {time.time() - t0:.1f}s")

    # Swap DiT transformer
    wrapper = NeuronTransformerWrapper(
        compiled_backbone=backbone_app, cpu_transformer=cpu_transformer, text_seq=1024
    )
    del cpu_transformer.transformer_blocks
    del cpu_transformer.norm_out, cpu_transformer.proj_out
    del cpu_transformer.audio_norm_out, cpu_transformer.audio_proj_out
    gc.collect()
    pipe.transformer = wrapper
    print("  DiT swapped")

    # 4. Swap text encoder: CPU -> Neuron
    print("\n[4/7] Swapping text encoder to Neuron...")
    t0 = time.time()
    del pipe.text_encoder
    gc.collect()
    print("  Freed CPU text encoder")

    compiled_gemma3 = load_neuron_gemma3(
        GEMMA3_SHARDED_DIR, GEMMA3_COMPILE_DIR, TP_DEGREE
    )
    pipe.text_encoder = NeuronTextEncoderWrapper(compiled_gemma3)
    print(f"  Neuron text encoder loaded in {time.time() - t0:.1f}s")

    # 5. Swap VAE decoder: CPU -> Neuron (if compiled)
    if use_neuron_vae:
        print(f"\n[5/7] Swapping VAE decoder to Neuron...")
        t0 = time.time()

        from pipeline import NeuronTiledVAEDecoder

        original_decoder = pipe.vae.decoder
        neuron_decoder = NeuronTiledVAEDecoder(
            compiled_dir=VAE_COMPILE_DIR,
            tile_latent_h=VAE_TILE_LATENT_H,
            tile_latent_w=VAE_TILE_LATENT_W,
            overlap_latent_h=VAE_OVERLAP_H,
            overlap_latent_w=VAE_OVERLAP_W,
            original_decoder=original_decoder,
        )
        del original_decoder
        gc.collect()

        pipe.vae.decoder = neuron_decoder
        print(f"  Neuron VAE loaded in {time.time() - t0:.1f}s")

        # Warmup VAE
        print("  Warming up VAE...")
        neuron_decoder.warmup(num_frames=NUM_FRAMES)
        print("  VAE warmup done")
    else:
        print(f"\n[5/7] Skipping Neuron VAE (using CPU fallback)")

    # 6. Generate
    print("\n[6/7] Generating video+audio...")
    print(f"  Prompt: {PROMPT[:80]}...")
    print(f"  {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames, {NUM_STEPS} steps")

    generator = torch.Generator(device="cpu").manual_seed(SEED)

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

    # 7. Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = output.frames[0]
    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
    print(f"  Saved {len(frames)} frames to {frames_dir}/")

    try:
        from diffusers.utils import export_to_video

        video_path = os.path.join(OUTPUT_DIR, "output.mp4")
        export_to_video(frames, video_path, fps=24)
        print(f"  Video: {video_path}")
    except Exception as e:
        print(f"  Video export failed: {e}")

    metadata = {
        "model": "Lightricks/LTX-2",
        "prompt": PROMPT,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "num_frames": NUM_FRAMES,
        "num_steps": NUM_STEPS,
        "guidance_scale": 4.0,
        "max_sequence_length": 1024,
        "seed": SEED,
        "generation_time_s": gen_time,
        "total_time_s": time.time() - t_total,
        "text_encoder": f"Neuron Gemma3-12B (TP={TP_DEGREE})",
        "dit": f"Neuron LTX2 DiT 48 blocks (TP={TP_DEGREE})",
        "vae_decoder": f"Neuron tiled 4x16 (TP={TP_DEGREE})"
        if use_neuron_vae
        else "CPU (Diffusers default)",
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Output frames: {len(frames)}")
    print(f"  Text encoder: Neuron Gemma3-12B")
    print(f"  DiT: Neuron LTX2 (48 blocks)")
    print(
        f"  VAE decoder: {'Neuron tiled 4x16' if use_neuron_vae else 'CPU (fallback)'}"
    )
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
