#!/usr/bin/env python3
"""
逐组件对比测试: CPU vs Neuron

按照推理流程逐步对比每个组件的输出:
1. Processor 输出 (input_ids, pixel_values, image_grid_thw)
2. Vision Encoder 输出 (image_embeds)
3. Embedding 合并后的结果 (inputs_embeds)
4. Position IDs 计算
5. Language Model 输出 (hidden_states)
6. 完整 Text Encoder 输出

这个脚本帮助定位数值差异的来源。
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"


def cosine_sim(a, b):
    """Calculate cosine similarity."""
    return F.cosine_similarity(
        a.flatten().unsqueeze(0).float(),
        b.flatten().unsqueeze(0).float()
    ).item()


def print_stats(name, tensor):
    """Print tensor statistics."""
    t = tensor.float()
    print(f"  {name}:")
    print(f"    shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"    mean: {t.mean().item():.6f}, std: {t.std().item():.6f}")
    print(f"    min: {t.min().item():.6f}, max: {t.max().item():.6f}")


def compare_tensors(name, cpu_tensor, neuron_tensor):
    """Compare two tensors and print metrics."""
    print(f"\n{'='*60}")
    print(f"Comparing: {name}")
    print(f"{'='*60}")

    print_stats("CPU", cpu_tensor)
    print_stats("Neuron", neuron_tensor)

    if cpu_tensor.shape != neuron_tensor.shape:
        print(f"\n  [ERROR] Shape mismatch!")
        return False

    diff = (cpu_tensor.float() - neuron_tensor.float()).abs()
    cos_sim = cosine_sim(cpu_tensor, neuron_tensor)

    print(f"\n  Difference:")
    print(f"    Max AE: {diff.max().item():.6e}")
    print(f"    Mean AE: {diff.mean().item():.6e}")
    print(f"    Cosine Sim: {cos_sim:.6f}")

    passed = cos_sim > 0.99
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n  {status} Cosine Similarity: {cos_sim:.6f}")

    return passed


def test_step_by_step(args):
    """逐步对比每个组件."""
    from diffusers import QwenImageEditPlusPipeline

    print("\n" + "="*60)
    print("Step-by-Step Component Comparison")
    print("="*60)

    dtype = torch.bfloat16
    image_size = args.image_size

    # Load pipeline
    print("\n[0] Loading pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # Configure processor for fixed image size
    target_pixels = image_size * image_size
    pipe.processor.image_processor.min_pixels = target_pixels
    pipe.processor.image_processor.max_pixels = target_pixels
    print(f"  Processor configured for {image_size}x{image_size}")

    # Create test image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )

    # Process input
    prompt = "change the color to blue"
    base_img_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    template = pipe.prompt_template_encode
    txt = [template.format(base_img_prompt + prompt)]

    print(f"\n[1] Processing input...")
    model_inputs = pipe.processor(
        text=txt,
        images=[test_image],
        padding=True,
        return_tensors="pt",
    )

    print(f"  input_ids: {model_inputs.input_ids.shape}")
    print(f"  pixel_values: {model_inputs.pixel_values.shape}")
    print(f"  image_grid_thw: {model_inputs.image_grid_thw.tolist()}")

    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    pixel_values = model_inputs.pixel_values.to(dtype)
    image_grid_thw = model_inputs.image_grid_thw

    results = {}

    # ========================================
    # Step 2: Vision Encoder
    # ========================================
    print(f"\n[2] Testing Vision Encoder...")

    # CPU Vision Encoder
    original_visual = pipe.text_encoder.model.visual
    original_visual.eval()

    with torch.no_grad():
        cpu_image_embeds = original_visual(pixel_values, image_grid_thw)
    print(f"  CPU image_embeds: {cpu_image_embeds.shape}")

    # Neuron Vision Encoder
    vision_path = f"{args.compiled_models_dir}/vision_encoder/model.pt"
    if os.path.exists(vision_path):
        compiled_vision = torch.jit.load(vision_path)
        with torch.no_grad():
            neuron_image_embeds = compiled_vision(pixel_values, image_grid_thw)
        results["vision_encoder"] = compare_tensors(
            "Vision Encoder", cpu_image_embeds, neuron_image_embeds
        )
    else:
        print(f"  [SKIP] Vision encoder not found at {vision_path}")
        neuron_image_embeds = cpu_image_embeds
        results["vision_encoder"] = None

    # ========================================
    # Step 3: Embed Tokens
    # ========================================
    print(f"\n[3] Testing Embed Tokens...")

    embed_tokens = pipe.text_encoder.model.language_model.embed_tokens

    with torch.no_grad():
        cpu_text_embeds = embed_tokens(input_ids)
    print(f"  CPU text_embeds: {cpu_text_embeds.shape}")
    print_stats("text_embeds", cpu_text_embeds)

    # ========================================
    # Step 4: Merge Embeddings
    # ========================================
    print(f"\n[4] Testing Embedding Merge...")

    # Find image token positions
    image_token_id = pipe.text_encoder.config.image_token_id
    batch_size, seq_len, hidden_dim = cpu_text_embeds.shape

    # Merge on CPU
    cpu_merged = cpu_text_embeds.clone()
    image_mask = (input_ids == image_token_id)
    num_image_tokens = image_mask.sum().item()
    print(f"  Number of image tokens: {num_image_tokens}")
    print(f"  Image embeds to merge: {cpu_image_embeds.shape}")

    if num_image_tokens > 0 and cpu_image_embeds.shape[0] == num_image_tokens:
        cpu_merged[image_mask] = cpu_image_embeds.to(cpu_merged.dtype)
        print(f"  Merged embeddings: {cpu_merged.shape}")
    else:
        print(f"  [WARNING] Token count mismatch: {num_image_tokens} vs {cpu_image_embeds.shape[0]}")

    print_stats("merged_embeds", cpu_merged)

    # ========================================
    # Step 5: Position IDs (M-RoPE)
    # ========================================
    print(f"\n[5] Testing Position IDs...")

    # Calculate position IDs using original model's method
    original_model = pipe.text_encoder.model

    with torch.no_grad():
        cpu_position_ids, _ = original_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask
        )
    print(f"  CPU position_ids: {cpu_position_ids.shape}")
    print(f"  position_ids range: [{cpu_position_ids.min().item()}, {cpu_position_ids.max().item()}]")

    # Compare with our implementation
    from neuron_qwen_image_edit.neuron_commons import NeuronTextEncoderWrapper

    # Create a minimal wrapper to test _get_rope_index
    wrapper = NeuronTextEncoderWrapper(
        original_text_encoder=pipe.text_encoder,
        compiled_vision_encoder=None,
        compiled_language_model=None,
        cpu_language_model=None,
        image_size=image_size,
        max_seq_len=args.max_sequence_length
    )

    neuron_position_ids = wrapper._get_rope_index(input_ids, image_grid_thw, attention_mask)
    print(f"  Neuron position_ids: {neuron_position_ids.shape}")

    # Compare position IDs
    if cpu_position_ids.shape == neuron_position_ids.shape:
        pos_match = (cpu_position_ids == neuron_position_ids).all().item()
        print(f"  Position IDs match: {pos_match}")
        if not pos_match:
            diff_count = (cpu_position_ids != neuron_position_ids).sum().item()
            print(f"  Mismatched positions: {diff_count} / {cpu_position_ids.numel()}")
            # Show first few differences
            diff_mask = cpu_position_ids != neuron_position_ids
            diff_indices = diff_mask.nonzero()[:10]
            for idx in diff_indices:
                d, b, s = idx.tolist()
                print(f"    [{d},{b},{s}]: CPU={cpu_position_ids[d,b,s].item()}, Neuron={neuron_position_ids[d,b,s].item()}")
        results["position_ids"] = pos_match
    else:
        print(f"  [ERROR] Shape mismatch!")
        results["position_ids"] = False

    # ========================================
    # Step 6: Language Model
    # ========================================
    print(f"\n[6] Testing Language Model...")

    language_model = pipe.text_encoder.model.language_model
    language_model.eval()

    with torch.no_grad():
        cpu_lm_output = language_model(
            inputs_embeds=cpu_merged.to(dtype),
            attention_mask=attention_mask,
            position_ids=cpu_position_ids,
            output_hidden_states=True,
            return_dict=True
        )
    cpu_hidden = cpu_lm_output.last_hidden_state
    print(f"  CPU hidden_states: {cpu_hidden.shape}")

    # Test with neuron position_ids
    with torch.no_grad():
        neuron_pos_lm_output = language_model(
            inputs_embeds=cpu_merged.to(dtype),
            attention_mask=attention_mask,
            position_ids=neuron_position_ids,
            output_hidden_states=True,
            return_dict=True
        )
    neuron_pos_hidden = neuron_pos_lm_output.last_hidden_state

    results["lm_with_neuron_pos"] = compare_tensors(
        "LM Output (Neuron position_ids)", cpu_hidden, neuron_pos_hidden
    )

    # ========================================
    # Step 7: Full Text Encoder
    # ========================================
    print(f"\n[7] Testing Full Text Encoder...")

    # CPU full text encoder
    with torch.no_grad():
        cpu_full_output = pipe.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
    cpu_full_hidden = cpu_full_output.hidden_states[-1]
    print(f"  CPU full output: {cpu_full_hidden.shape}")

    # Neuron wrapper
    cpu_language_model = pipe.text_encoder.model.language_model
    cpu_language_model.eval()

    if os.path.exists(vision_path):
        compiled_vision = torch.jit.load(vision_path)
    else:
        compiled_vision = None

    neuron_wrapper = NeuronTextEncoderWrapper(
        original_text_encoder=pipe.text_encoder,
        compiled_vision_encoder=compiled_vision,
        compiled_language_model=None,
        cpu_language_model=cpu_language_model,
        image_size=image_size,
        max_seq_len=args.max_sequence_length
    )

    with torch.no_grad():
        neuron_full_output = neuron_wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
    neuron_full_hidden = neuron_full_output.hidden_states[-1]

    results["full_text_encoder"] = compare_tensors(
        "Full Text Encoder", cpu_full_hidden, neuron_full_hidden
    )

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Component Comparison Test")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Vision encoder image size")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--compiled_models_dir", type=str,
                        default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    args = parser.parse_args()

    test_step_by_step(args)


if __name__ == "__main__":
    main()
