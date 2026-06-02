#!/usr/bin/env python3
"""
Simple Language Model Test without Tensor Parallelism

This test compiles the Language Model on a SINGLE device (no TP)
to verify that the model itself works correctly before adding TP complexity.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch
import torch.nn.functional as F

from diffusers import QwenImageEditPlusPipeline


CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


class SimpleLanguageModelWrapper(torch.nn.Module):
    """Simple wrapper for Language Model without TP."""
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.last_hidden_state


def test_language_model_cpu_only():
    """Test Language Model on CPU without any Neuron compilation."""
    print("=" * 60)
    print("Test 1: Language Model CPU Only (No Neuron)")
    print("=" * 60)

    dtype = torch.bfloat16
    batch_size = 1
    seq_len = 64  # Use smaller seq for quick test
    hidden_size = 3584

    print("\nLoading pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    lang_model = pipe.text_encoder.model.language_model
    lang_model.eval()

    print(f"\nLanguage Model config:")
    print(f"  num_hidden_layers: {lang_model.config.num_hidden_layers}")
    print(f"  num_attention_heads: {lang_model.config.num_attention_heads}")
    print(f"  num_key_value_heads: {lang_model.config.num_key_value_heads}")
    print(f"  hidden_size: {lang_model.config.hidden_size}")

    # Create test input
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)

    # Run CPU inference
    print("\nRunning CPU inference...")
    with torch.no_grad():
        output = lang_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        ).last_hidden_state

    print(f"\nOutput shape: {output.shape}")
    print(f"Output stats:")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    print(f"  Min: {output.min().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")
    print(f"  Has NaN: {torch.isnan(output).any()}")
    print(f"  Has Inf: {torch.isinf(output).any()}")

    return output


def test_language_model_single_device():
    """Test Language Model compiled on single device (no TP)."""
    print("\n" + "=" * 60)
    print("Test 2: Language Model Single Device Compilation")
    print("=" * 60)

    import torch_neuronx

    dtype = torch.bfloat16
    batch_size = 1
    seq_len = 64
    hidden_size = 3584

    print("\nLoading pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    lang_model = pipe.text_encoder.model.language_model
    lang_model.eval()

    # Create wrapper
    wrapper = SimpleLanguageModelWrapper(lang_model)

    # Create test inputs
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)

    # CPU inference first
    print("\nRunning CPU inference...")
    with torch.no_grad():
        cpu_output = wrapper(inputs_embeds, attention_mask)

    print(f"CPU output shape: {cpu_output.shape}")

    # Try Neuron compilation (single device)
    print("\nCompiling for Neuron (single device, this will take time)...")
    print("NOTE: This is just to test if single-device works. For production, use TP.")

    compiler_flags = "--target=trn2 --lnc=2 --model-type=transformer"

    try:
        with torch.no_grad():
            compiled = torch_neuronx.trace(
                wrapper,
                (inputs_embeds, attention_mask),
                compiler_args=compiler_flags,
                inline_weights_to_neff=False
            )

        print("Compilation successful!")

        # Run Neuron inference
        print("Running Neuron inference...")
        with torch.no_grad():
            neuron_output = compiled(inputs_embeds, attention_mask)

        print(f"Neuron output shape: {neuron_output.shape}")

        # Compare
        abs_error = torch.abs(cpu_output.float() - neuron_output.float())
        cosine_sim = F.cosine_similarity(
            cpu_output.flatten().unsqueeze(0).float(),
            neuron_output.flatten().unsqueeze(0).float()
        ).item()

        print(f"\nComparison:")
        print(f"  Max Absolute Error: {abs_error.max().item():.6e}")
        print(f"  Mean Absolute Error: {abs_error.mean().item():.6e}")
        print(f"  Cosine Similarity: {cosine_sim:.6f}")

        if cosine_sim > 0.99:
            print("\n[PASS] Single device compilation works correctly!")
            print("Problem is likely in Tensor Parallelism implementation.")
        else:
            print("\n[FAIL] Even single device compilation has issues!")

    except Exception as e:
        print(f"Compilation failed: {e}")
        print("\nThis is expected if the model is too large for single device.")


def test_attention_gqa():
    """Test GQA attention specifically."""
    print("\n" + "=" * 60)
    print("Test 3: GQA Attention Test")
    print("=" * 60)

    dtype = torch.bfloat16

    print("\nLoading pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    lang_model = pipe.text_encoder.model.language_model
    first_layer = lang_model.layers[0]
    attn = first_layer.self_attn

    print(f"\nAttention config:")
    print(f"  num_heads: {attn.num_heads}")
    print(f"  num_key_value_heads: {attn.num_key_value_heads}")
    print(f"  head_dim: {attn.head_dim}")
    print(f"  hidden_size: {attn.hidden_size}")

    print(f"\nProjection shapes:")
    print(f"  q_proj: {attn.q_proj.weight.shape}")  # (3584, 3584) = 28 heads * 128
    print(f"  k_proj: {attn.k_proj.weight.shape}")  # (512, 3584) = 4 heads * 128
    print(f"  v_proj: {attn.v_proj.weight.shape}")  # (512, 3584) = 4 heads * 128
    print(f"  o_proj: {attn.o_proj.weight.shape}")  # (3584, 3584)

    # Check GQA ratio
    gqa_ratio = attn.num_heads // attn.num_key_value_heads
    print(f"\nGQA ratio (num_heads / num_kv_heads): {gqa_ratio}")
    print(f"  Each KV head is shared by {gqa_ratio} Q heads")


def main():
    print("=" * 60)
    print("Language Model Debug Tests")
    print("=" * 60)

    # Test 1: CPU only
    cpu_output = test_language_model_cpu_only()

    # Test 2: GQA analysis
    test_attention_gqa()

    # Test 3: Single device (optional, takes time)
    # Uncomment to test single device compilation
    # test_language_model_single_device()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The Language Model uses Grouped Query Attention (GQA):
- 28 Q heads, 4 KV heads
- Each KV head is shared by 7 Q heads

With TP=8:
- Q: 28 -> padded to 32 -> 4 per rank
- KV: 4 heads replicated to 8 -> 1 per rank

Potential issues:
1. The attention forward() may not handle the modified head counts correctly
2. The KV replication logic may be broken
3. parallel_state may not be properly initialized during compilation
""")


if __name__ == "__main__":
    main()
