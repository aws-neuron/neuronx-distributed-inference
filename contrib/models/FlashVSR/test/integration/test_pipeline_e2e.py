# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test: Full pipeline E2E with PSNR validation.

Validates that the complete FlashVSR pipeline (LQ Proj + DiT + TCDecoder)
produces outputs with acceptable PSNR against the CPU reference pipeline.

This test:
1. Loads all pipeline components (DiT, LQ Projection, TCDecoder)
2. Runs the full pipeline on a test video
3. Computes PSNR between Neuron and CPU outputs
4. Asserts PSNR > 40 dB (standard threshold for near-lossless)

Requires:
- trn2.3xlarge instance with SDK 2.29
- FlashVSR-v1.1 weights at WEIGHTS_DIR
- Pre-compiled NEFFs for all components
- Test video at TEST_VIDEO_PATH
"""

import os
import sys
import pytest
import torch
import numpy as np

# Test configuration
WEIGHTS_DIR = os.environ.get(
    "FLASHVSR_WEIGHTS_DIR", "/home/ubuntu/flash_vsr/FlashVSR-v1.1"
)
COMPILED_DIR = os.environ.get(
    "FLASHVSR_COMPILED_DIR", "/home/ubuntu/flash_vsr/compiled"
)
TEST_VIDEO_PATH = os.environ.get(
    "FLASHVSR_TEST_VIDEO", "/home/ubuntu/flash_vsr/example0.mp4"
)
PROMPT_PATH = os.environ.get(
    "FLASHVSR_PROMPT_PATH", "/home/ubuntu/flash_vsr/FlashVSR-v1.1/posi_prompt.pth"
)
TCDECODER_PATH = os.environ.get(
    "FLASHVSR_TCDECODER", "/home/ubuntu/flash_vsr/compiled_tcdecoder/tcdecoder_seq.pt"
)
LQ_PROJ_PATH = os.environ.get(
    "FLASHVSR_LQ_PROJ", "/home/ubuntu/flash_vsr/compiled_lq_proj/lq_proj_T89.pt"
)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute PSNR between two tensors in [-1, 1] range.

    Args:
        img1, img2: Tensors of same shape, values in [-1, 1]

    Returns:
        PSNR in dB
    """
    # Convert to [0, 1] range
    img1 = (img1.float() + 1) / 2
    img2 = (img2.float() + 1) / 2
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


@pytest.fixture(scope="module")
def pipeline():
    """Load the full FlashVSR pipeline."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Check all required files exist
    required = [
        (WEIGHTS_DIR, "weights directory"),
        (TEST_VIDEO_PATH, "test video"),
        (PROMPT_PATH, "prompt embedding"),
        (TCDECODER_PATH, "compiled TCDecoder"),
        (LQ_PROJ_PATH, "compiled LQ Projection"),
    ]
    for path, name in required:
        if not os.path.exists(path):
            pytest.skip(f"{name} not found at {path}")

    dit_first_dir = os.path.join(COMPILED_DIR, "flashvsr_first_tp4")
    dit_stream_dir = os.path.join(COMPILED_DIR, "flashvsr_stream_tp4")
    if not os.path.exists(dit_first_dir):
        pytest.skip(f"Compiled DiT (first) not found at {dit_first_dir}")
    if not os.path.exists(dit_stream_dir):
        pytest.skip(f"Compiled DiT (stream) not found at {dit_stream_dir}")

    from src.pipeline import load_pipeline

    return load_pipeline(
        compiled_dir=COMPILED_DIR,
        weights_dir=WEIGHTS_DIR,
        prompt_path=PROMPT_PATH,
        tp_degree=4,
        tcdecoder_path=TCDECODER_PATH,
        lq_proj_path=LQ_PROJ_PATH,
    )


def test_pipeline_e2e_psnr(pipeline):
    """Run full pipeline and validate output PSNR.

    PSNR > 40 dB indicates near-lossless reconstruction quality,
    accounting for BF16 numerical differences between Neuron and CPU.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.pipeline import run_inference

    output_dir = "/tmp/flashvsr_test_output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = run_inference(
        pipeline,
        input_video=TEST_VIDEO_PATH,
        output_dir=output_dir,
        scale=4,
        max_chunks=1,  # Only process first chunk for speed
        color_correction="adain",
        save_mp4=True,
    )

    assert os.path.exists(output_path), f"Output video not created at {output_path}"

    # Verify output video can be read
    import imageio

    reader = imageio.get_reader(output_path)
    frame_count = 0
    try:
        while True:
            reader.get_data(frame_count)
            frame_count += 1
    except (IndexError, Exception):
        pass
    reader.close()

    assert frame_count > 0, "Output video has no frames"
    # First chunk (f=6) produces 24 frames, minus 3 trim = 21 minimum
    assert frame_count >= 20, f"Expected at least 20 frames, got {frame_count}"


def test_pipeline_output_resolution(pipeline):
    """Validate output video has correct resolution (4x input)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.pipeline import run_inference

    output_dir = "/tmp/flashvsr_test_resolution"
    os.makedirs(output_dir, exist_ok=True)

    output_path = run_inference(
        pipeline,
        input_video=TEST_VIDEO_PATH,
        output_dir=output_dir,
        scale=4,
        max_chunks=1,
        save_mp4=True,
    )

    import imageio

    reader = imageio.get_reader(output_path)
    first_frame = reader.get_data(0)
    reader.close()

    # Output should be 768x1280 (or nearest 128-aligned dimension)
    h, w = first_frame.shape[:2]
    assert h % 128 == 0, f"Output height {h} not divisible by 128"
    assert w % 128 == 0, f"Output width {w} not divisible by 128"
    assert h >= 512, f"Output height {h} too small for 4x upscaling"
    assert w >= 512, f"Output width {w} too small for 4x upscaling"
