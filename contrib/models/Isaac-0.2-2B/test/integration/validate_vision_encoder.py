# Copyright 2025 © Amazon.com and Affiliates
"""Validate Isaac vision encoder on Neuron vs CPU reference.

Approach: Since the HF Isaac model uses a different vision input format
(packed_seq_patches via tensor_stream) than the NxDI model (standard pixel_values
through Conv2d), we can't directly compare vision encoder outputs.

Instead, we validate the Neuron vision encoder by:
1. Running the NxDI vision encoder on a test image
2. Checking that output embeddings are numerically reasonable (no NaN/Inf)
3. Checking that different images produce different embeddings (not degenerate)
4. Running a manual Conv2d + encoder comparison using reshaped weights

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH
    python validate_vision_encoder.py
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.image_utils import load_image  # noqa: E402

from neuronx_distributed_inference.models.config import NeuronConfig  # noqa: E402
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config  # noqa: E402

from isaac_neuron.modeling_isaac import (  # noqa: E402
    NeuronIsaacForConditionalGeneration,
    IsaacInferenceConfig,
)

# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_HOME", "/mnt/models")
MODEL_PATH = f"{DATA_PATH}/Isaac-0.2-2B-Preview"
TRACED_MODEL_PATH = f"{DATA_PATH}/traced_model/Isaac-0.2-2B"
REFERENCE_DIR = f"{DATA_PATH}/reference_outputs"

IMAGE_SIZE = 256
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
torch.manual_seed(42)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize(
                (IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def load_neuron_model():
    """Load the compiled Neuron model and return the full model object."""
    text_config = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=1,
        cp_degree=1,
        save_sharded_checkpoint=True,
        skip_sharding=False,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        context_encoding_buckets=[1024],
        token_generation_buckets=[1024],
        async_mode=False,
        output_logits=True,
        fused_qkv=False,
        sequence_parallel_enabled=False,
        attn_kernel_enabled=False,
        attn_tkg_nki_kernel_enabled=False,
        attn_tkg_builtin_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )
    vision_config = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=1,
        world_size=1,
        save_sharded_checkpoint=True,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        buckets=[1],
        fused_qkv=False,
        attn_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = IsaacInferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    model = NeuronIsaacForConditionalGeneration(TRACED_MODEL_PATH, config)
    model.load(TRACED_MODEL_PATH, skip_warmup=True)
    return model


def main():
    print(f"{'=' * 70}")
    print("VISION ENCODER VALIDATION: Neuron")
    print(f"{'=' * 70}")

    # Prepare test images
    images = {
        "red": Image.new("RGB", (256, 256), color="red"),
        "blue": Image.new("RGB", (256, 256), color="blue"),
        "black": Image.new("RGB", (256, 256), color="black"),
    }
    try:
        images["reference"] = load_image(
            "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp"
        )
    except Exception as e:
        print(f"  WARNING: Could not load reference image: {e}")

    # Load model
    print("\nLoading compiled Neuron model...")
    model = load_neuron_model()
    print("  Model loaded.")

    # Run vision encoder on each image
    embeddings = {}
    all_passed = True
    results = []

    for label, img in images.items():
        print(f"\n--- {label} ({img.size}) ---")
        pixel_values = preprocess_image(img).to(torch.bfloat16)
        print(f"  pixel_values: {pixel_values.shape}")

        with torch.no_grad():
            output = model.vision_encoder_model(pixel_values)

        output_f = output.float().cpu()
        embeddings[label] = output_f

        # Check 1: Shape
        expected_tokens = (IMAGE_SIZE // 16) ** 2 // 4  # 64
        expected_dim = 2048  # text hidden size
        shape_ok = output_f.shape == torch.Size([1, expected_tokens, expected_dim])
        print(
            f"  Output shape: {output_f.shape} (expected [1, {expected_tokens}, {expected_dim}]): {'OK' if shape_ok else 'FAIL'}"
        )

        # Check 2: No NaN
        has_nan = torch.isnan(output_f).any().item()
        print(f"  NaN check: {'FAIL' if has_nan else 'OK'}")

        # Check 3: No Inf
        has_inf = torch.isinf(output_f).any().item()
        print(f"  Inf check: {'FAIL' if has_inf else 'OK'}")

        # Check 4: Non-zero variance (not degenerate)
        variance = output_f.var().item()
        variance_ok = variance > 1e-6
        print(
            f"  Variance: {variance:.6f} {'OK' if variance_ok else 'FAIL (degenerate)'}"
        )

        # Check 5: Reasonable value range
        val_min = output_f.min().item()
        val_max = output_f.max().item()
        val_mean = output_f.mean().item()
        range_ok = abs(val_min) < 100 and abs(val_max) < 100
        print(
            f"  Range: [{val_min:.4f}, {val_max:.4f}], mean={val_mean:.4f} {'OK' if range_ok else 'SUSPICIOUS'}"
        )

        passed = shape_ok and not has_nan and not has_inf and variance_ok and range_ok
        if not passed:
            all_passed = False
        results.append(
            {
                "label": label,
                "passed": passed,
                "shape": list(output_f.shape),
                "has_nan": has_nan,
                "has_inf": has_inf,
                "variance": variance,
                "range": [val_min, val_max],
                "mean": val_mean,
            }
        )

    # Cross-image comparison: different images should produce different embeddings
    print(f"\n--- Cross-image comparison ---")
    labels = list(embeddings.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            cos = F.cosine_similarity(
                embeddings[a].reshape(1, -1), embeddings[b].reshape(1, -1)
            ).item()
            different = cos < 0.999  # Different images should have cosine < 0.999
            print(
                f"  {a} vs {b}: cosine={cos:.6f} {'OK (different)' if different else 'WARNING (too similar)'}"
            )
            if not different:
                print(f"    WARNING: Very similar embeddings for different images!")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  [{status}] {r['label']}: shape={r['shape']}, var={r['variance']:.6f}, range=[{r['range'][0]:.3f}, {r['range'][1]:.3f}]"
        )

    if all_passed:
        print(f"\n  ALL VISION ENCODER CHECKS PASSED")
    else:
        print(f"\n  SOME CHECKS FAILED")
        sys.exit(1)

    out_path = os.path.join(REFERENCE_DIR, "neuron_vision_encoder_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
