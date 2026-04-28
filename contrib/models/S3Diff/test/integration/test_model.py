"""
Integration tests for S3Diff one-step super-resolution on Neuron.

Tests:
1. test_smoke_pipeline_loads: Pipeline loads without errors
2. test_sr_produces_correct_size: 128x128 -> 512x512
3. test_warm_generation_time: < 2s per image

Requirements:
    - trn2.3xlarge
    - Neuron SDK 2.29+
    - diffusers, transformers, peft, torchvision
    - Weights downloaded (see generate_s3diff.py --download)

Run:
    pytest test_model.py -v
    # Or standalone:
    python test_model.py
"""

import gc
import os
import sys
import time

import numpy as np
import pytest
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "src"))
sys.path.insert(0, SRC_DIR)

from modeling_s3diff import S3DiffNeuronPipeline

SD_TURBO_PATH = os.environ.get("SD_TURBO_PATH", "/shared/sd-turbo/")
S3DIFF_WEIGHTS = os.environ.get("S3DIFF_WEIGHTS", "/shared/s3diff/s3diff.pkl")
DE_NET_WEIGHTS = os.environ.get("DE_NET_WEIGHTS", "/shared/s3diff/de_net.pth")
COMPILE_DIR = os.environ.get("S3DIFF_COMPILE_DIR", "/tmp/s3diff_test/compiled/")
LR_SIZE = 128
HR_SIZE = 512


def make_test_image(size=128):
    """Create a deterministic test image."""
    np.random.seed(42)
    data = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)


@pytest.fixture(scope="module")
def pipeline():
    """Create, load, compile, and warm up the S3Diff pipeline."""
    pipe = S3DiffNeuronPipeline(
        sd_turbo_path=SD_TURBO_PATH,
        s3diff_weights_path=S3DIFF_WEIGHTS,
        de_net_path=DE_NET_WEIGHTS,
        compile_dir=COMPILE_DIR,
        lr_size=LR_SIZE,
    )
    pipe.load()
    pipe.compile()

    # Warmup
    test_img = make_test_image()
    pipe(test_img)

    yield pipe

    del pipe
    gc.collect()


def test_smoke_pipeline_loads(pipeline):
    """Pipeline loads without errors and has required components."""
    assert pipeline is not None
    assert pipeline.de_net_neuron is not None
    assert pipeline.text_enc_neuron is not None
    assert pipeline.vae_enc_neuron is not None
    assert pipeline.unet_neuron is not None
    assert pipeline.vae_dec_neuron is not None
    assert pipeline.tokenizer is not None
    assert pipeline.sched is not None
    assert pipeline.compute_modulation is not None


def test_sr_produces_correct_size(pipeline):
    """128x128 input produces 512x512 output."""
    test_img = make_test_image(LR_SIZE)
    sr_img = pipeline(test_img)

    assert isinstance(sr_img, Image.Image)
    assert sr_img.size == (HR_SIZE, HR_SIZE), (
        f"Expected ({HR_SIZE}, {HR_SIZE}), got {sr_img.size}"
    )

    # Verify reasonable pixel distribution (not blank)
    sr_array = np.array(sr_img)
    assert sr_array.shape == (HR_SIZE, HR_SIZE, 3)
    assert sr_array.std() > 10, "Output appears blank or uniform"

    # Save for inspection
    os.makedirs(os.path.join(COMPILE_DIR, "test_outputs"), exist_ok=True)
    sr_img.save(os.path.join(COMPILE_DIR, "test_outputs", "test_sr.png"))
    print(f"SR output saved, pixel std={sr_array.std():.1f}")


def test_warm_generation_time(pipeline):
    """Warm generation should complete in < 2s."""
    test_img = make_test_image(LR_SIZE)

    t0 = time.time()
    pipeline(test_img)
    elapsed = time.time() - t0
    print(f"Warm generation time: {elapsed:.3f}s")

    assert elapsed < 2.0, f"Generation took {elapsed:.3f}s, expected < 2s"


# Standalone runner
if __name__ == "__main__":
    print("=" * 60)
    print("S3Diff Integration Tests")
    print("=" * 60)

    pipe = S3DiffNeuronPipeline(
        sd_turbo_path=SD_TURBO_PATH,
        s3diff_weights_path=S3DIFF_WEIGHTS,
        de_net_path=DE_NET_WEIGHTS,
        compile_dir=COMPILE_DIR,
        lr_size=LR_SIZE,
    )

    print("\n[1/5] Loading...")
    pipe.load()

    print("\n[2/5] Compiling...")
    t0 = time.time()
    pipe.compile()
    print(f"  Compilation: {time.time() - t0:.1f}s")

    print("\n[3/5] Warmup...")
    test_img = make_test_image()
    pipe(test_img)

    print("\n[4/5] test_smoke_pipeline_loads")
    test_smoke_pipeline_loads(pipe)
    print("  PASSED")

    print("\n[5/5] test_sr_produces_correct_size")
    test_sr_produces_correct_size(pipe)
    print("  PASSED")

    print("\n[6/6] test_warm_generation_time")
    test_warm_generation_time(pipe)
    print("  PASSED")

    print("\nAll tests passed!")
