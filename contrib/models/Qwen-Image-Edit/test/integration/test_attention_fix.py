#!/usr/bin/env python3
"""
Test to verify the attention fix works correctly.

This test:
1. Tests our custom SDPA implementation matches PyTorch's SDPA
2. Tests causal masking works correctly
3. Tests with the actual Qwen2 model dimensions
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math


def reference_sdpa(query, key, value, attn_mask=None, is_causal=False, scale=None):
    """Reference SDPA implementation using PyTorch."""
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale
    )


def custom_sdpa(query, key, value, attn_mask=None, is_causal=False, scale=None):
    """Our custom SDPA implementation."""
    from neuron_qwen_image_edit.neuron_commons import neuron_scaled_dot_product_attention
    return neuron_scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale
    )


def test_basic_attention():
    """Test basic attention without masking."""
    print("=" * 60)
    print("Test 1: Basic Attention (no masking)")
    print("=" * 60)

    batch = 2
    heads = 4
    seq_len = 16
    head_dim = 64

    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    ref_out = reference_sdpa(query, key, value)
    custom_out = custom_sdpa(query, key, value)

    cosine_sim = F.cosine_similarity(
        ref_out.flatten().unsqueeze(0),
        custom_out.flatten().unsqueeze(0)
    ).item()

    max_diff = (ref_out - custom_out).abs().max().item()

    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    print(f"  Max Absolute Diff: {max_diff:.6e}")

    if cosine_sim > 0.999:
        print("  [PASS] Basic attention works correctly")
        return True
    else:
        print("  [FAIL] Basic attention mismatch!")
        return False


def test_causal_attention():
    """Test causal attention (critical for language models)."""
    print("\n" + "=" * 60)
    print("Test 2: Causal Attention (is_causal=True)")
    print("=" * 60)

    batch = 2
    heads = 4
    seq_len = 16
    head_dim = 64

    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    ref_out = reference_sdpa(query, key, value, is_causal=True)
    custom_out = custom_sdpa(query, key, value, is_causal=True)

    cosine_sim = F.cosine_similarity(
        ref_out.flatten().unsqueeze(0),
        custom_out.flatten().unsqueeze(0)
    ).item()

    max_diff = (ref_out - custom_out).abs().max().item()

    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    print(f"  Max Absolute Diff: {max_diff:.6e}")

    # Verify that positions can't attend to future positions
    # by checking attention to the first position differs from last
    print("\n  Verifying causal property:")
    print(f"    First position output std: {custom_out[0, 0, 0].std().item():.6f}")
    print(f"    Last position output std: {custom_out[0, 0, -1].std().item():.6f}")

    if cosine_sim > 0.999:
        print("  [PASS] Causal attention works correctly")
        return True
    else:
        print("  [FAIL] Causal attention mismatch!")
        return False


def test_attention_with_mask():
    """Test attention with explicit mask."""
    print("\n" + "=" * 60)
    print("Test 3: Attention with Explicit Mask")
    print("=" * 60)

    batch = 2
    heads = 4
    seq_len = 16
    head_dim = 64

    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    # Create a simple mask (mask out last 4 positions)
    attn_mask = torch.zeros(seq_len, seq_len)
    attn_mask[:, -4:] = float('-inf')

    ref_out = reference_sdpa(query, key, value, attn_mask=attn_mask)
    custom_out = custom_sdpa(query, key, value, attn_mask=attn_mask)

    cosine_sim = F.cosine_similarity(
        ref_out.flatten().unsqueeze(0),
        custom_out.flatten().unsqueeze(0)
    ).item()

    max_diff = (ref_out - custom_out).abs().max().item()

    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    print(f"  Max Absolute Diff: {max_diff:.6e}")

    if cosine_sim > 0.999:
        print("  [PASS] Attention with mask works correctly")
        return True
    else:
        print("  [FAIL] Attention with mask mismatch!")
        return False


def test_gqa_attention():
    """Test Grouped Query Attention (GQA) where num_kv_heads < num_q_heads."""
    print("\n" + "=" * 60)
    print("Test 4: Grouped Query Attention (GQA)")
    print("=" * 60)

    batch = 2
    num_q_heads = 28  # Qwen2.5-VL has 28 Q heads
    num_kv_heads = 4  # Qwen2.5-VL has 4 KV heads
    seq_len = 16
    head_dim = 128

    query = torch.randn(batch, num_q_heads, seq_len, head_dim)
    key = torch.randn(batch, num_kv_heads, seq_len, head_dim)
    value = torch.randn(batch, num_kv_heads, seq_len, head_dim)

    # PyTorch SDPA with enable_gqa
    ref_out = F.scaled_dot_product_attention(
        query, key, value, enable_gqa=True
    )

    custom_out = custom_sdpa(query, key, value)

    cosine_sim = F.cosine_similarity(
        ref_out.flatten().unsqueeze(0),
        custom_out.flatten().unsqueeze(0)
    ).item()

    max_diff = (ref_out - custom_out).abs().max().item()

    print(f"  Q shape: {query.shape}")
    print(f"  K shape: {key.shape}")
    print(f"  V shape: {value.shape}")
    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    print(f"  Max Absolute Diff: {max_diff:.6e}")

    if cosine_sim > 0.999:
        print("  [PASS] GQA attention works correctly")
        return True
    else:
        print("  [FAIL] GQA attention mismatch!")
        return False


def test_qwen2_dimensions():
    """Test with actual Qwen2.5-VL dimensions (sharded for TP=8)."""
    print("\n" + "=" * 60)
    print("Test 5: Qwen2.5-VL Sharded Dimensions (TP=8)")
    print("=" * 60)

    # After sharding with TP=8:
    # Q: 28 -> padded to 32 -> 4 per rank
    # KV: 4 -> replicated to 8 -> 1 per rank

    batch = 1
    num_q_heads = 4  # After sharding
    num_kv_heads = 1  # After sharding (replicated)
    seq_len = 512
    head_dim = 128

    query = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Test with causal masking (language model uses causal)
    ref_out = F.scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=True
    )

    custom_out = custom_sdpa(query, key, value, is_causal=True)

    cosine_sim = F.cosine_similarity(
        ref_out.flatten().unsqueeze(0).float(),
        custom_out.flatten().unsqueeze(0).float()
    ).item()

    max_diff = (ref_out.float() - custom_out.float()).abs().max().item()

    print(f"  Configuration (simulating TP=8 sharding):")
    print(f"    Q heads per rank: {num_q_heads}")
    print(f"    KV heads per rank: {num_kv_heads}")
    print(f"    Sequence length: {seq_len}")
    print(f"    Head dim: {head_dim}")
    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    print(f"  Max Absolute Diff: {max_diff:.6e}")

    if cosine_sim > 0.99:
        print("  [PASS] Qwen2 sharded dimensions work correctly")
        return True
    else:
        print("  [FAIL] Qwen2 sharded dimensions mismatch!")
        return False


def test_cpu_vs_neuron_attention():
    """Test that our custom attention produces same results on CPU."""
    print("\n" + "=" * 60)
    print("Test 6: Verify Custom Attention Implementation")
    print("=" * 60)

    batch = 1
    heads = 4
    seq_len = 64
    head_dim = 128

    torch.manual_seed(42)
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    # Reference: PyTorch's SDPA with is_causal
    ref_out = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    # Custom implementation
    custom_out = custom_sdpa(query, key, value, is_causal=True)

    # Manual implementation for verification
    scale = 1 / math.sqrt(head_dim)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Create causal mask (avoiding 0 * -inf = NaN)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = torch.where(mask == 1, float('-inf'), 0.0)
    scores = scores + mask

    probs = F.softmax(scores, dim=-1)
    manual_out = torch.matmul(probs, value)

    # Compare all three
    ref_vs_manual = F.cosine_similarity(
        ref_out.flatten().unsqueeze(0),
        manual_out.flatten().unsqueeze(0)
    ).item()

    custom_vs_manual = F.cosine_similarity(
        custom_out.flatten().unsqueeze(0),
        manual_out.flatten().unsqueeze(0)
    ).item()

    custom_vs_ref = F.cosine_similarity(
        custom_out.flatten().unsqueeze(0),
        ref_out.flatten().unsqueeze(0)
    ).item()

    print(f"  Reference vs Manual: {ref_vs_manual:.6f}")
    print(f"  Custom vs Manual: {custom_vs_manual:.6f}")
    print(f"  Custom vs Reference: {custom_vs_ref:.6f}")

    if custom_vs_ref > 0.999 and custom_vs_manual > 0.999:
        print("  [PASS] Custom attention matches reference implementations")
        return True
    else:
        print("  [FAIL] Custom attention has issues!")
        return False


def main():
    print("=" * 60)
    print("Attention Implementation Verification Tests")
    print("=" * 60)

    results = []

    results.append(("Basic Attention", test_basic_attention()))
    results.append(("Causal Attention", test_causal_attention()))
    results.append(("Attention with Mask", test_attention_with_mask()))
    results.append(("GQA Attention", test_gqa_attention()))
    results.append(("Qwen2 Sharded Dims", test_qwen2_dimensions()))
    results.append(("CPU Verification", test_cpu_vs_neuron_attention()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s}: [{status}]")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! The attention implementation is correct.")
        print("\nNext steps:")
        print("1. Recompile the language model:")
        print("   python neuron_qwen_image_edit/compile_text_encoder.py --language_only")
        print("2. Run the text encoder test:")
        print("   python tests/test_text_encoder.py --test language")
    else:
        print("\nSome tests failed. Please review the attention implementation.")

    return all_passed


if __name__ == "__main__":
    main()
