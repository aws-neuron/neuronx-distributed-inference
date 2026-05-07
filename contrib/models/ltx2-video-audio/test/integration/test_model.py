#!/usr/bin/env python3
"""
Integration tests for LTX-2 video+audio diffusion model on Neuron.

Validates that the Neuron-compiled pipeline (DiT backbone + Gemma3 text encoder)
produces output frames that are visually consistent with GPU reference frames.

Since this is a diffusion model (not a CausalLM), accuracy is measured via
structural similarity (SSIM) between Neuron and GPU output frames generated
with identical settings (same seed, guidance_scale, etc.), rather than logit
or token matching.

Prerequisites:
  - Compiled DiT backbone at $LTX2_DIT_COMPILE_DIR (default: /home/ubuntu/ltx2_nxdi_compiled_1024/)
  - Compiled Gemma3 encoder at $LTX2_GEMMA3_COMPILE_DIR (default: /home/ubuntu/gemma3_encoder_compiled_1024/)
  - Pre-sharded Gemma3 weights at $LTX2_GEMMA3_SHARDED_DIR (default: /home/ubuntu/gemma3_encoder_sharded/)

  To compile these, run from the src/ directory:
    NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
      python compile_gemma3.py
    python shard_gemma3_weights.py
  The DiT backbone is compiled automatically on first use.

Usage:
  # With pytest:
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
    pytest test/integration/test_model.py -v --capture=tee-sys

  # Standalone:
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
    python test/integration/test_model.py
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# Environment variables for Neuron
os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src directory to path
CONTRIB_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CONTRIB_ROOT / "src"))

# Model directories (override with environment variables)
DIT_COMPILE_DIR = os.environ.get(
    "LTX2_DIT_COMPILE_DIR", "/home/ubuntu/ltx2_nxdi_compiled_1024/"
)
GEMMA3_COMPILE_DIR = os.environ.get(
    "LTX2_GEMMA3_COMPILE_DIR", "/home/ubuntu/gemma3_encoder_compiled_1024/"
)
GEMMA3_SHARDED_DIR = os.environ.get(
    "LTX2_GEMMA3_SHARDED_DIR", "/home/ubuntu/gemma3_encoder_sharded/"
)
OUTPUT_DIR = os.environ.get("LTX2_TEST_OUTPUT_DIR", "/tmp/ltx2_test_output/")
GPU_SAMPLES_DIR = CONTRIB_ROOT / "samples" / "gpu"

# Generation settings (must match GPU reference)
TP_DEGREE = 4
HEIGHT, WIDTH, NUM_FRAMES = 384, 512, 25
NUM_STEPS = 8
SEED = 42
PROMPT = (
    "A golden retriever puppy runs across a sunny green meadow, "
    "its ears flapping in the wind. The camera follows from a low angle. "
    "Birds chirp in the background."
)


def compute_ssim(img1, img2):
    """Compute structural similarity index between two PIL images.

    Uses a simplified SSIM that does not require scikit-image.
    Returns a value between 0 and 1 (1 = identical).
    """
    a = np.array(img1, dtype=np.float64)
    b = np.array(img2, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a_sq = a.var()
    sigma_b_sq = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()

    ssim = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (sigma_a_sq + sigma_b_sq + c2)
    )
    return float(ssim)


# ---------------------------------------------------------------------------
# Pipeline fixture (shared across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def neuron_pipeline():
    """Load the full Neuron LTX-2 pipeline (DiT + Gemma3 on Neuron, VAE on CPU)."""
    from modeling_ltx2 import (
        LTX2BackboneInferenceConfig,
        NeuronLTX2BackboneApplication,
        replace_sdpa_with_bmm,
    )
    from pipeline import NeuronTransformerWrapper
    from neuronx_distributed_inference.models.config import NeuronConfig

    replace_sdpa_with_bmm()

    # Load transformer config
    from huggingface_hub import hf_hub_download, snapshot_download

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

    # Load Diffusers pipeline
    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)

    # Load DiT backbone onto Neuron
    local_transformer_path = snapshot_download(
        "Lightricks/LTX-2", allow_patterns=["transformer/*"]
    )
    local_transformer_path = os.path.join(local_transformer_path, "transformer")

    cpu_transformer = pipe.transformer
    backbone_app = NeuronLTX2BackboneApplication(
        model_path=local_transformer_path, config=config
    )

    compiled_model_file = os.path.join(DIT_COMPILE_DIR, "model.pt")
    if not os.path.exists(compiled_model_file):
        os.makedirs(DIT_COMPILE_DIR, exist_ok=True)
        backbone_app.compile(DIT_COMPILE_DIR)

    backbone_app.load(DIT_COMPILE_DIR)

    wrapper = NeuronTransformerWrapper(
        compiled_backbone=backbone_app, cpu_transformer=cpu_transformer, text_seq=1024
    )
    del cpu_transformer.transformer_blocks
    del cpu_transformer.norm_out, cpu_transformer.proj_out
    del cpu_transformer.audio_norm_out, cpu_transformer.audio_proj_out
    gc.collect()
    pipe.transformer = wrapper

    # Load Gemma3 text encoder onto Neuron
    import torch_neuronx
    from neuronx_distributed.trace.trace import (
        replace_weights,
        TensorParallelNeuronModel,
    )

    del pipe.text_encoder
    gc.collect()

    models = []
    for rank in range(TP_DEGREE):
        rank_ckpt_path = os.path.join(GEMMA3_SHARDED_DIR, f"rank_{rank}.pt")
        ckpt = torch.load(rank_ckpt_path, weights_only=True)
        neff_path = os.path.join(GEMMA3_COMPILE_DIR, f"tp_{rank}.pt")
        with torch_neuronx.contexts.disable_nrt_load():
            traced_model = torch.jit.load(neff_path)
        replace_weights(traced_model, ckpt)
        models.append(traced_model)
        del ckpt
        gc.collect()

    compiled_gemma3 = TensorParallelNeuronModel(models)

    class NeuronTextEncoderOutput:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    class NeuronTextEncoderWrapper:
        def __init__(self, compiled_gemma3, dtype=torch.bfloat16):
            self.compiled_model = compiled_gemma3
            self.dtype = dtype
            self._device = torch.device("cpu")
            self.config = type("Config", (), {"output_hidden_states": True})()

        def __call__(
            self,
            input_ids=None,
            attention_mask=None,
            output_hidden_states=True,
            **kwargs,
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

    pipe.text_encoder = NeuronTextEncoderWrapper(compiled_gemma3)
    return pipe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_smoke_pipeline_loads(neuron_pipeline):
    """Test that the Neuron pipeline loads without errors."""
    assert neuron_pipeline is not None
    assert hasattr(neuron_pipeline, "transformer")
    assert hasattr(neuron_pipeline, "text_encoder")
    print("PASS: Pipeline loaded successfully")


def test_generation_produces_frames(neuron_pipeline):
    """Test that generation produces the expected number of output frames."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    output = neuron_pipeline(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        generator=generator,
        output_type="pil",
    )

    frames = output.frames[0]
    assert len(frames) == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {len(frames)}"
    assert frames[0].size == (WIDTH, HEIGHT), (
        f"Expected ({WIDTH}, {HEIGHT}), got {frames[0].size}"
    )

    # Save frames for inspection
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png"))

    print(f"PASS: Generated {len(frames)} frames at {frames[0].size}")


def test_accuracy_vs_gpu_reference(neuron_pipeline):
    """Test that Neuron output is visually similar to GPU reference frames.

    Compares SSIM between Neuron-generated frames and GPU reference frames
    (both generated with identical seed=42, guidance_scale=4.0, etc.).
    SSIM > 0.7 indicates strong structural similarity.
    """
    from PIL import Image

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    output = neuron_pipeline(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        generator=generator,
        output_type="pil",
    )

    frames = output.frames[0]

    # Compare against GPU reference samples (frame_0000, frame_0012, frame_0024)
    reference_indices = [0, 12, 24]
    ssim_scores = []

    for idx in reference_indices:
        gpu_path = GPU_SAMPLES_DIR / f"frame_{idx:04d}.png"
        if not gpu_path.exists():
            pytest.skip(f"GPU reference frame not found: {gpu_path}")

        gpu_frame = Image.open(gpu_path).convert("RGB")
        neuron_frame = frames[idx]

        ssim = compute_ssim(neuron_frame, gpu_frame)
        ssim_scores.append(ssim)
        print(f"  Frame {idx:04d}: SSIM = {ssim:.4f}")

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    min_ssim = min(ssim_scores)

    print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Min SSIM: {min_ssim:.4f}")

    # SSIM > 0.7 indicates strong structural similarity.
    # Neuron BF16 vs GPU FP16 produces minor numerical differences, but
    # the visual output should remain structurally very similar.
    assert min_ssim > 0.7, (
        f"SSIM too low: min={min_ssim:.4f}. Neuron output may not match GPU reference."
    )
    print(f"PASS: Accuracy validated (avg SSIM={avg_ssim:.4f}, min={min_ssim:.4f})")


def test_warm_generation_time(neuron_pipeline):
    """Test that warm generation completes within a reasonable time.

    The first generation includes warmup overhead. This test measures
    a second generation to capture steady-state performance.
    """
    # First generation (warmup)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    _ = neuron_pipeline(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        generator=generator,
        output_type="pil",
    )

    # Second generation (warm)
    generator = torch.Generator(device="cpu").manual_seed(123)
    t0 = time.time()
    _ = neuron_pipeline(
        prompt="A cat sitting on a windowsill watches rain falling outside.",
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        generator=generator,
        output_type="pil",
    )
    warm_time = time.time() - t0

    # Warm generation should complete in under 120s (generous threshold)
    assert warm_time < 120, f"Warm generation took {warm_time:.1f}s, expected < 120s"
    print(f"PASS: Warm generation time = {warm_time:.1f}s")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("LTX-2 Integration Tests (Neuron)")
    print("=" * 70)

    print("\nLoading pipeline...")
    t0 = time.time()

    # Inline the fixture for standalone mode
    from modeling_ltx2 import (
        LTX2BackboneInferenceConfig,
        NeuronLTX2BackboneApplication,
        replace_sdpa_with_bmm,
    )
    from pipeline import NeuronTransformerWrapper
    from neuronx_distributed_inference.models.config import NeuronConfig

    replace_sdpa_with_bmm()

    from huggingface_hub import hf_hub_download, snapshot_download

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
    video_seq = latent_num_frames * (HEIGHT // 32) * (WIDTH // 32)
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

    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)

    local_transformer_path = snapshot_download(
        "Lightricks/LTX-2", allow_patterns=["transformer/*"]
    )
    local_transformer_path = os.path.join(local_transformer_path, "transformer")
    cpu_transformer = pipe.transformer
    backbone_app = NeuronLTX2BackboneApplication(
        model_path=local_transformer_path, config=config
    )

    compiled_model_file = os.path.join(DIT_COMPILE_DIR, "model.pt")
    if not os.path.exists(compiled_model_file):
        os.makedirs(DIT_COMPILE_DIR, exist_ok=True)
        backbone_app.compile(DIT_COMPILE_DIR)
    backbone_app.load(DIT_COMPILE_DIR)

    wrapper = NeuronTransformerWrapper(
        compiled_backbone=backbone_app, cpu_transformer=cpu_transformer, text_seq=1024
    )
    del cpu_transformer.transformer_blocks
    del cpu_transformer.norm_out, cpu_transformer.proj_out
    del cpu_transformer.audio_norm_out, cpu_transformer.audio_proj_out
    gc.collect()
    pipe.transformer = wrapper

    import torch_neuronx
    from neuronx_distributed.trace.trace import (
        replace_weights,
        TensorParallelNeuronModel,
    )

    del pipe.text_encoder
    gc.collect()

    models = []
    for rank in range(TP_DEGREE):
        ckpt = torch.load(
            os.path.join(GEMMA3_SHARDED_DIR, f"rank_{rank}.pt"), weights_only=True
        )
        with torch_neuronx.contexts.disable_nrt_load():
            traced = torch.jit.load(os.path.join(GEMMA3_COMPILE_DIR, f"tp_{rank}.pt"))
        replace_weights(traced, ckpt)
        models.append(traced)
        del ckpt
        gc.collect()

    compiled_gemma3 = TensorParallelNeuronModel(models)

    class _Output:
        def __init__(self, hs):
            self.hidden_states = hs

    class _Wrapper:
        def __init__(self, m):
            self.compiled_model = m
            self.dtype = torch.bfloat16
            self._device = torch.device("cpu")
            self.config = type("C", (), {"output_hidden_states": True})()

        def __call__(self, input_ids=None, attention_mask=None, **kw):
            with torch.no_grad():
                s = self.compiled_model(input_ids, attention_mask)
                return _Output(tuple(s[:, :, :, i] for i in range(s.shape[-1])))

        def eval(self):
            return self

        def to(self, *a, **k):
            return self

        @property
        def device(self):
            return self._device

    pipe.text_encoder = _Wrapper(compiled_gemma3)
    print(f"Pipeline loaded in {time.time() - t0:.1f}s")

    print("\n1. Smoke Test...")
    test_smoke_pipeline_loads(pipe)

    print("\n2. Generation Test...")
    test_generation_produces_frames(pipe)

    print("\n3. Accuracy vs GPU Reference...")
    test_accuracy_vs_gpu_reference(pipe)

    print("\n4. Warm Generation Time...")
    test_warm_generation_time(pipe)

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
