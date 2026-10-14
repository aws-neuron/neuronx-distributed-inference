#!/usr/bin/env python3
"""
Multimodal Text Encoder Test: Verify text + image processing works correctly.

This test is critical because it tests the ACTUAL inference scenario:
- Images are processed through vision encoder
- Image embeddings are merged with text embeddings
- Proper multimodal position_ids (M-RoPE) are calculated
- Language model processes the combined embeddings

Key issues this test catches:
1. Processor pixel count mismatch (image_size must match compiled vision encoder)
2. Wrong position_ids for multimodal input (need M-RoPE, not simple sequential)
3. Vision encoder shape mismatch (compiled vs runtime)
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Neuron environment BEFORE imports
# Now using TP=8 for language model with KV head replication
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from diffusers import QwenImageEditPlusPipeline
from neuron_qwen_image_edit.neuron_commons import NeuronTextEncoderWrapper

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"


def test_multimodal_text_encoder(args):
    """Test text encoder with images (multimodal mode)."""
    print("=" * 60)
    print("Testing Multimodal Text Encoder (Text + Image)")
    print("=" * 60)

    dtype = torch.bfloat16
    image_size = args.image_size

    # Load pipeline
    print("\nLoading pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # CRITICAL FIX #1: Configure processor for compiled vision encoder size
    # Without this, the processor outputs variable-sized pixel_values that
    # don't match the compiled vision encoder's expected input shape.
    target_pixels = image_size * image_size
    print(f"\n[FIX #1] Configuring processor for {image_size}x{image_size}")
    print(f"  Setting min_pixels = max_pixels = {target_pixels}")
    pipe.processor.image_processor.min_pixels = target_pixels
    pipe.processor.image_processor.max_pixels = target_pixels

    # Load compiled vision encoder
    vision_path = f"{args.compiled_models_dir}/vision_encoder/model.pt"
    if not os.path.exists(vision_path):
        print(f"\nERROR: Vision encoder not found at {vision_path}")
        return None

    print(f"\nLoading compiled vision encoder from {vision_path}...")
    compiled_vision_encoder = torch.jit.load(vision_path)

    # Get CPU language model
    cpu_language_model = pipe.text_encoder.model.language_model
    cpu_language_model.eval()

    # Create wrapper with FIX #2: Proper M-RoPE position_ids calculation
    print("\n[FIX #2] Creating NeuronTextEncoderWrapper with M-RoPE support")
    wrapper = NeuronTextEncoderWrapper(
        original_text_encoder=pipe.text_encoder,
        compiled_vision_encoder=compiled_vision_encoder,
        compiled_language_model=None,
        cpu_language_model=cpu_language_model,
        image_size=image_size,
        max_seq_len=args.max_sequence_length
    )

    # Create test image (any size - processor will resize to image_size)
    print(f"\nCreating test image (will be resized to {image_size}x{image_size})...")
    test_image = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )

    # Process with images
    prompt = "change the color to blue"
    base_img_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    template = pipe.prompt_template_encode
    txt = [template.format(base_img_prompt + prompt)]

    print(f"\nProcessing prompt: \"{prompt}\"")
    model_inputs = pipe.processor(
        text=txt,
        images=[test_image],
        padding=True,
        return_tensors="pt",
    )

    # Verify processor output matches compiled vision encoder
    expected_patches = (image_size // 14) ** 2
    actual_patches = model_inputs.pixel_values.shape[0]
    print(f"\n  Processor output verification:")
    print(f"    Expected patches: {expected_patches}")
    print(f"    Actual patches: {actual_patches}")
    print(f"    input_ids shape: {model_inputs.input_ids.shape}")
    print(f"    pixel_values shape: {model_inputs.pixel_values.shape}")
    print(f"    image_grid_thw: {model_inputs.image_grid_thw.tolist()}")

    if actual_patches != expected_patches:
        print(f"  [ERROR] Patch count mismatch! Vision encoder expects {expected_patches}")
        return None

    # Run original text encoder
    print("\nRunning original text encoder...")
    with torch.no_grad():
        orig_output = pipe.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )
    orig_hidden = orig_output.hidden_states[-1]
    print(f"  Output shape: {orig_hidden.shape}")

    # Run wrapper
    print("\nRunning NeuronTextEncoderWrapper...")
    with torch.no_grad():
        wrapper_output = wrapper(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )
    wrapper_hidden = wrapper_output.hidden_states[-1]
    print(f"  Output shape: {wrapper_hidden.shape}")

    # Compare
    cosine_sim = F.cosine_similarity(
        orig_hidden.flatten().unsqueeze(0).float(),
        wrapper_hidden.flatten().unsqueeze(0).float()
    ).item()

    max_ae = (orig_hidden.float() - wrapper_hidden.float()).abs().max().item()
    mean_ae = (orig_hidden.float() - wrapper_hidden.float()).abs().mean().item()

    print(f"\n{'='*60}")
    print("RESULTS (Multimodal Text + Image)")
    print(f"{'='*60}")
    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    print(f"  Max Absolute Error: {max_ae:.6e}")
    print(f"  Mean Absolute Error: {mean_ae:.6e}")

    passed = cosine_sim > 0.99
    if passed:
        print("  [PASS] Multimodal text encoder works correctly!")
    else:
        print("  [FAIL] Output mismatch - check vision encoder and position_ids!")

    return {
        "cosine_sim": cosine_sim,
        "max_ae": max_ae,
        "mean_ae": mean_ae,
        "passed": passed
    }


def main():
    parser = argparse.ArgumentParser(description="Multimodal Text Encoder Test")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Vision encoder image size (must match compiled model)")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max text sequence length")
    parser.add_argument("--compiled_models_dir", type=str,
                        default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    args = parser.parse_args()

    print(f"Image size: {args.image_size}")
    print(f"Max sequence length: {args.max_sequence_length}")
    print(f"Compiled models: {args.compiled_models_dir}")

    result = test_multimodal_text_encoder(args)

    if result is None:
        print("\n[ERROR] Test failed to run")
        sys.exit(1)
    elif result["passed"]:
        print("\n[SUCCESS] All multimodal tests passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Multimodal test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
