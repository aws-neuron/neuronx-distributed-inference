# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test: DiT accuracy validation using neuron_allclose.

Validates that the compiled Neuron DiT produces outputs within numerical
tolerance of the CPU reference model (same weights, same inputs).

This test:
1. Loads FlashVSR DiT weights into a CPU reference model
2. Compiles the same model for Neuron via NxDI ModelBuilder
3. Runs both models with identical inputs
4. Compares outputs using neuron_allclose with rtol=0.01, atol=1e-5

Requires:
- trn2.3xlarge instance with SDK 2.29
- FlashVSR-v1.1 weights at WEIGHTS_DIR
- Pre-compiled NEFF at COMPILED_DIR (or will compile on first run)
"""

import os
import sys
import pytest
import torch

# Test configuration -- override via environment variables
WEIGHTS_DIR = os.environ.get(
    "FLASHVSR_WEIGHTS_DIR", "/home/ubuntu/flash_vsr/FlashVSR-v1.1"
)
COMPILED_DIR = os.environ.get(
    "FLASHVSR_COMPILED_DIR", "/home/ubuntu/flash_vsr/compiled/flashvsr_first_tp4"
)
TP_DEGREE = int(os.environ.get("FLASHVSR_TP_DEGREE", "4"))
HEIGHT = 768
WIDTH = 1280


@pytest.fixture(scope="module")
def cpu_model():
    """Load FlashVSR DiT with real weights on CPU."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.modeling_flashvsr import NeuronFlashVSRDiT, NUM_LAYERS
    from src.weights import detect_format_and_convert

    model = NeuronFlashVSRDiT(num_layers=NUM_LAYERS).to(torch.bfloat16).eval()

    weights_path = os.path.join(
        WEIGHTS_DIR, "diffusion_pytorch_model_streaming_dmd.safetensors"
    )
    if not os.path.exists(weights_path):
        pytest.skip(f"Weights not found at {weights_path}")

    from safetensors import safe_open

    raw_sd = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            raw_sd[key] = f.get_tensor(key)

    neuron_sd = detect_format_and_convert(raw_sd, tp_degree=1)
    model.load_state_dict(neuron_sd, strict=False)
    return model


@pytest.fixture(scope="module")
def neuron_app():
    """Load compiled Neuron DiT application."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    if not os.path.exists(COMPILED_DIR):
        pytest.skip(f"Compiled model not found at {COMPILED_DIR}")

    import concurrent.futures
    from src.modeling_flashvsr import FlashVSRApplication, FlashVSRInferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig

    original_init = concurrent.futures.ThreadPoolExecutor.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["max_workers"] = 1
        original_init(self, *args, **kwargs)

    concurrent.futures.ThreadPoolExecutor.__init__ = patched_init

    try:
        neuron_config = NeuronConfig(
            tp_degree=TP_DEGREE,
            torch_dtype=torch.bfloat16,
            batch_size=1,
            save_sharded_checkpoint=True,
        )
        config = FlashVSRInferenceConfig(
            neuron_config=neuron_config,
            attn_mode="first",
            height=HEIGHT,
            width=WIDTH,
        )
        app = FlashVSRApplication(model_path=WEIGHTS_DIR, config=config)
        app.load(COMPILED_DIR)
        return app
    finally:
        concurrent.futures.ThreadPoolExecutor.__init__ = original_init


@pytest.fixture(scope="module")
def test_inputs():
    """Generate deterministic test inputs."""
    from src.modeling_flashvsr import (
        precompute_freqs_cis_3d,
        build_rope_for_grid,
        HEAD_DIM,
        DIM,
        NUM_HEADS,
        PATCH_T,
        PATCH_H,
        PATCH_W,
        LCSA_WIN,
        IN_CHANNELS,
    )

    torch.manual_seed(42)
    lat_h = HEIGHT // 8
    lat_w = WIDTH // 8
    num_frames = 6  # First chunk

    post_f = num_frames // PATCH_T
    post_h = lat_h // PATCH_H
    post_w = lat_w // PATCH_W
    seq_len = post_f * post_h * post_w

    hidden_states = torch.randn(
        1, IN_CHANNELS, num_frames, lat_h, lat_w, dtype=torch.bfloat16
    )
    timestep = torch.tensor([1000.0], dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    base_freqs = precompute_freqs_cis_3d(HEAD_DIM)
    rope_cos, rope_sin = build_rope_for_grid(*base_freqs, post_f, post_h, post_w)

    num_q_blocks = (
        (post_f // LCSA_WIN[0]) * (post_h // LCSA_WIN[1]) * (post_w // LCSA_WIN[2])
    )
    attn_mask = torch.zeros(
        1, NUM_HEADS, num_q_blocks, num_q_blocks, dtype=torch.bfloat16
    )
    lq_residual_0 = torch.zeros(1, seq_len, DIM, dtype=torch.bfloat16)

    return (
        hidden_states,
        timestep,
        encoder_hidden_states,
        rope_cos,
        rope_sin,
        attn_mask,
        lq_residual_0,
    )


def test_dit_neuron_allclose(cpu_model, neuron_app, test_inputs):
    """Validate Neuron DiT output matches CPU reference within tolerance.

    Uses neuron_allclose with rtol=0.01 (1% relative tolerance) which is
    the standard for BF16 encoder models on Neuron hardware.
    """
    from torch_neuronx.testing.validation import neuron_allclose

    # CPU reference
    with torch.no_grad():
        cpu_outputs = cpu_model(*test_inputs)
    cpu_output = cpu_outputs[0]  # First element is the noise prediction

    # Neuron inference
    with torch.no_grad():
        neuron_outputs = neuron_app(*test_inputs)
    neuron_output = neuron_outputs[0]

    # Compare
    result = neuron_allclose(
        neuron_output.cpu(),
        cpu_output,
        rtol=0.01,
        atol=1e-5,
    )
    assert result.allclose, (
        f"DiT output mismatch: max_rel_error={result.max_rel_error:.6f}, "
        f"max_abs_error={result.max_abs_error:.6f}"
    )


def test_dit_output_shape(neuron_app, test_inputs):
    """Validate Neuron DiT produces correct output shapes."""
    with torch.no_grad():
        outputs = neuron_app(*test_inputs)

    # Output should be (1, 16, 6, H_lat, W_lat)
    lat_h = HEIGHT // 8
    lat_w = WIDTH // 8
    expected_shape = (1, 16, 6, lat_h, lat_w)
    assert outputs[0].shape == expected_shape, (
        f"Expected shape {expected_shape}, got {outputs[0].shape}"
    )

    # Should also return 60 KV cache tensors (30 layers x 2)
    assert len(outputs) == 61, (
        f"Expected 61 outputs (1 + 60 caches), got {len(outputs)}"
    )
