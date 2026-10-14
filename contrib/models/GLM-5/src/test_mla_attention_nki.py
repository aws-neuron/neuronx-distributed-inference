#!/usr/bin/env python3
"""
Correctness test for the MLA attention NKI kernel.

Tests the NKI kernel against the PyTorch reference implementation using
random inputs matching GLM-5 dimensions.

Usage:
  # On a trn2 instance with Neuron SDK 2.29+:
  python3 test_mla_attention_nki.py

  # CPU-only validation (reference only, no NKI execution):
  python3 test_mla_attention_nki.py --cpu-only
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

# Import the reference implementation (always available)
sys.path.insert(0, ".")
from mla_attention_nki import mla_attention_tkg_reference, NKI_AVAILABLE


def create_test_inputs(
    batch_size: int = 1,
    n_heads: int = 2,
    seq_len: int = 512,
    d_rope: int = 64,
    d_c: int = 512,
    d_v: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
):
    """Create random test inputs matching GLM-5 MLA dimensions."""
    BH = batch_size * n_heads
    d_cache = d_rope + d_c  # 576

    q_pe = torch.randn(BH, d_rope, dtype=dtype, device=device) * 0.1
    q_nope = torch.randn(BH, d_c, dtype=dtype, device=device) * 0.1
    kv_cache = (
        torch.randn(batch_size, seq_len, d_cache, dtype=dtype, device=device) * 0.1
    )
    v_absorb = torch.randn(n_heads, d_v, d_c, dtype=dtype, device=device) * 0.1

    # Causal mask: all True (attend to all prior positions in decode)
    attn_mask = torch.ones(BH, seq_len, dtype=torch.bool, device=device)

    return q_pe, q_nope, kv_cache, v_absorb, attn_mask


def test_reference_basic():
    """Test the PyTorch reference produces reasonable output."""
    print("=" * 60)
    print("TEST: Reference implementation basic sanity check")
    print("=" * 60)

    B, H, S = 1, 2, 128
    d_rope, d_c, d_v = 64, 512, 256
    softmax_scale = 256 ** (-0.5)

    q_pe, q_nope, kv_cache, v_absorb, attn_mask = create_test_inputs(
        batch_size=B,
        n_heads=H,
        seq_len=S,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
    )

    output = mla_attention_tkg_reference(
        q_pe=q_pe,
        q_nope=q_nope,
        kv_cache=kv_cache,
        v_absorb=v_absorb,
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        n_heads=H,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S,
    )

    assert output.shape == (B * H, d_v), (
        f"Expected ({B * H}, {d_v}), got {output.shape}"
    )
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Output norm: {output.norm():.4f}")
    print("  PASSED\n")


def test_reference_masked():
    """Test that masking works correctly."""
    print("=" * 60)
    print("TEST: Reference masked attention")
    print("=" * 60)

    B, H, S = 1, 2, 64
    d_rope, d_c, d_v = 64, 512, 256
    softmax_scale = 256 ** (-0.5)

    q_pe, q_nope, kv_cache, v_absorb, attn_mask = create_test_inputs(
        batch_size=B,
        n_heads=H,
        seq_len=S,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
    )

    # Mask out last half of cache positions
    attn_mask[:, S // 2 :] = False

    output_masked = mla_attention_tkg_reference(
        q_pe=q_pe,
        q_nope=q_nope,
        kv_cache=kv_cache,
        v_absorb=v_absorb,
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        n_heads=H,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S,
    )

    # Compare with output using only first half of cache
    kv_cache_half = kv_cache[:, : S // 2, :]
    attn_mask_half = torch.ones(B * H, S // 2, dtype=torch.bool)
    output_half = mla_attention_tkg_reference(
        q_pe=q_pe,
        q_nope=q_nope,
        kv_cache=kv_cache_half,
        v_absorb=v_absorb,
        attn_mask=attn_mask_half,
        softmax_scale=softmax_scale,
        n_heads=H,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S // 2,
    )

    diff = (output_masked - output_half).abs().max()
    print(f"  Max diff between masked and truncated: {diff:.6e}")
    assert diff < 1e-3, f"Masked output should match truncated cache, diff={diff}"
    print("  PASSED\n")


def test_reference_vs_manual():
    """Test reference against manual computation for a small example."""
    print("=" * 60)
    print("TEST: Reference vs manual computation")
    print("=" * 60)

    B, H, S = 1, 1, 4
    d_rope, d_c, d_v = 4, 8, 4  # Small dims for manual verification
    softmax_scale = (d_rope + d_c) ** (-0.5)  # Approximate

    torch.manual_seed(42)
    q_pe = torch.randn(1, d_rope)
    q_nope = torch.randn(1, d_c)
    kv_cache = torch.randn(1, S, d_rope + d_c)
    v_absorb = torch.randn(1, d_v, d_c)
    attn_mask = torch.ones(1, S, dtype=torch.bool)

    # Manual computation
    k_pe = kv_cache[0, :, :d_rope]  # [S, d_rope]
    c_kv = kv_cache[0, :, d_rope:]  # [S, d_c]

    scores = (q_pe @ k_pe.T + q_nope @ c_kv.T) * softmax_scale  # [1, S]
    weights = F.softmax(scores, dim=-1)  # [1, S]
    v_accum = weights @ c_kv  # [1, d_c]
    manual_out = v_accum @ v_absorb[0].T  # [1, d_v]

    # Reference
    ref_out = mla_attention_tkg_reference(
        q_pe=q_pe,
        q_nope=q_nope,
        kv_cache=kv_cache,
        v_absorb=v_absorb,
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        n_heads=1,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S,
    )

    diff = (ref_out - manual_out).abs().max()
    print(f"  Manual output: {manual_out[0, :4]}")
    print(f"  Reference output: {ref_out[0, :4]}")
    print(f"  Max diff: {diff:.6e}")
    assert diff < 1e-5, f"Reference should match manual computation, diff={diff}"
    print("  PASSED\n")


def test_nki_kernel():
    """Test NKI kernel against PyTorch reference on Neuron hardware."""
    print("=" * 60)
    print("TEST: NKI kernel vs reference (Neuron hardware)")
    print("=" * 60)

    try:
        import torch_neuronx
        from mla_attention_nki import mla_attention_tkg
    except ImportError as e:
        print(f"  SKIPPED: {e}")
        return

    B, H, S = 1, 2, 512
    d_rope, d_c, d_v = 64, 512, 256
    softmax_scale = 256 ** (-0.5)

    # Create inputs on CPU first (for reference)
    q_pe, q_nope, kv_cache, v_absorb, attn_mask = create_test_inputs(
        batch_size=B,
        n_heads=H,
        seq_len=S,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        dtype=torch.bfloat16,
    )

    # Reference on CPU
    ref_output = mla_attention_tkg_reference(
        q_pe=q_pe.float(),
        q_nope=q_nope.float(),
        kv_cache=kv_cache.float(),
        v_absorb=v_absorb.float(),
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        n_heads=H,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S,
    ).to(torch.bfloat16)

    # Run NKI kernel
    print("  Compiling NKI kernel...")
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    # Move tensors to XLA device
    q_pe_xla = q_pe.to(device)
    q_nope_xla = q_nope.to(device)
    kv_cache_xla = kv_cache.to(device)
    v_absorb_xla = v_absorb.to(device)
    attn_mask_xla = attn_mask.to(device)

    t0 = time.time()
    nki_output = mla_attention_tkg(
        q_pe=q_pe_xla,
        q_nope=q_nope_xla,
        kv_cache=kv_cache_xla,
        v_absorb=v_absorb_xla,
        attn_mask=attn_mask_xla,
        softmax_scale=softmax_scale,
        n_heads=H,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S,
    )
    t1 = time.time()
    print(f"  Kernel execution time: {(t1 - t0) * 1000:.1f} ms")

    # Move output back to CPU for comparison
    nki_output_cpu = nki_output.cpu()

    # Compare
    max_diff = (nki_output_cpu.float() - ref_output.float()).abs().max()
    cos_sim = F.cosine_similarity(
        nki_output_cpu.float().flatten().unsqueeze(0),
        ref_output.float().flatten().unsqueeze(0),
    ).item()

    print(f"  Output shape: {nki_output_cpu.shape}")
    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Ref norm: {ref_output.float().norm():.4f}")
    print(f"  NKI norm: {nki_output_cpu.float().norm():.4f}")

    # BF16 tolerance: allow up to ~1% relative error for attention
    if cos_sim > 0.99 and max_diff < 0.1:
        print("  PASSED\n")
    else:
        print("  FAILED: accuracy below threshold\n")
        sys.exit(1)


def test_nki_kernel_longer_seq():
    """Test NKI kernel with longer sequence (2048 tokens)."""
    print("=" * 60)
    print("TEST: NKI kernel with S=2048 (full GLM-5 cache)")
    print("=" * 60)

    try:
        import torch_neuronx
        from mla_attention_nki import mla_attention_tkg
    except ImportError as e:
        print(f"  SKIPPED: {e}")
        return

    B, H, S = 1, 2, 2048
    d_rope, d_c, d_v = 64, 512, 256
    softmax_scale = 256 ** (-0.5)

    q_pe, q_nope, kv_cache, v_absorb, attn_mask = create_test_inputs(
        batch_size=B,
        n_heads=H,
        seq_len=S,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        dtype=torch.bfloat16,
    )

    # Reference
    ref_output = mla_attention_tkg_reference(
        q_pe=q_pe.float(),
        q_nope=q_nope.float(),
        kv_cache=kv_cache.float(),
        v_absorb=v_absorb.float(),
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        n_heads=H,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S,
    ).to(torch.bfloat16)

    # NKI - move to XLA device
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    q_pe_xla = q_pe.to(device)
    q_nope_xla = q_nope.to(device)
    kv_cache_xla = kv_cache.to(device)
    v_absorb_xla = v_absorb.to(device)
    attn_mask_xla = attn_mask.to(device)

    t0 = time.time()
    nki_output = mla_attention_tkg(
        q_pe=q_pe_xla,
        q_nope=q_nope_xla,
        kv_cache=kv_cache_xla,
        v_absorb=v_absorb_xla,
        attn_mask=attn_mask_xla,
        softmax_scale=softmax_scale,
        n_heads=H,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S,
    )
    t1 = time.time()
    print(f"  Kernel execution time: {(t1 - t0) * 1000:.1f} ms")

    # Move output back to CPU for comparison
    nki_output_cpu = nki_output.cpu()

    max_diff = (nki_output_cpu.float() - ref_output.float()).abs().max()
    cos_sim = F.cosine_similarity(
        nki_output_cpu.float().flatten().unsqueeze(0),
        ref_output.float().flatten().unsqueeze(0),
    ).item()

    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    if cos_sim > 0.99 and max_diff < 0.1:
        print("  PASSED\n")
    else:
        print("  FAILED: accuracy below threshold\n")
        sys.exit(1)


def test_nki_kernel_batched():
    """Test NKI kernel with batch size > 1."""
    print("=" * 60)
    print("TEST: NKI kernel with B=4, H=2 (8 queries)")
    print("=" * 60)

    try:
        import torch_neuronx
        from mla_attention_nki import mla_attention_tkg
    except ImportError as e:
        print(f"  SKIPPED: {e}")
        return

    B, H, S = 4, 2, 256
    d_rope, d_c, d_v = 64, 512, 256
    softmax_scale = 256 ** (-0.5)

    q_pe, q_nope, kv_cache, v_absorb, attn_mask = create_test_inputs(
        batch_size=B,
        n_heads=H,
        seq_len=S,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        dtype=torch.bfloat16,
    )

    # Reference
    ref_output = mla_attention_tkg_reference(
        q_pe=q_pe.float(),
        q_nope=q_nope.float(),
        kv_cache=kv_cache.float(),
        v_absorb=v_absorb.float(),
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        n_heads=H,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S,
    ).to(torch.bfloat16)

    # NKI - move to XLA device
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    q_pe_xla = q_pe.to(device)
    q_nope_xla = q_nope.to(device)
    kv_cache_xla = kv_cache.to(device)
    v_absorb_xla = v_absorb.to(device)
    attn_mask_xla = attn_mask.to(device)

    nki_output = mla_attention_tkg(
        q_pe=q_pe_xla,
        q_nope=q_nope_xla,
        kv_cache=kv_cache_xla,
        v_absorb=v_absorb_xla,
        attn_mask=attn_mask_xla,
        softmax_scale=softmax_scale,
        n_heads=H,
        d_rope=d_rope,
        d_c=d_c,
        d_v=d_v,
        seq_len=S,
    )

    # Move output back to CPU for comparison
    nki_output_cpu = nki_output.cpu()

    max_diff = (nki_output_cpu.float() - ref_output.float()).abs().max()
    cos_sim = F.cosine_similarity(
        nki_output_cpu.float().flatten().unsqueeze(0),
        ref_output.float().flatten().unsqueeze(0),
    ).item()

    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    if cos_sim > 0.99 and max_diff < 0.1:
        print("  PASSED\n")
    else:
        print("  FAILED: accuracy below threshold\n")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run only CPU reference tests (no Neuron hardware)",
    )
    args = parser.parse_args()

    print("\nMLA Attention NKI Kernel - Correctness Tests")
    print("=" * 60)
    print(f"Config: d_rope=64, d_c=512, d_v=256 (GLM-5 dimensions)")
    print()

    # CPU reference tests (always run)
    test_reference_basic()
    test_reference_masked()
    test_reference_vs_manual()

    if not args.cpu_only:
        # NKI kernel tests (require Neuron hardware)
        test_nki_kernel()
        test_nki_kernel_longer_seq()
        test_nki_kernel_batched()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
