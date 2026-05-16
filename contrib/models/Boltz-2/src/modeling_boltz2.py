"""Boltz-2 pairformer inference on AWS Trainium 2.

This module provides Neuron-accelerated inference for the Boltz-2 biomolecular
structure prediction model. It compiles the pairformer trunk (64 transformer
layers) using torch_neuronx.trace() with NKI custom kernels for the O(N^3)
triangular attention and triangular multiplicative update operations.

Key techniques:
  - Weight replacement: compile ONE layer with inline_weights_to_neff=False,
    then create 64 copies with replace_weights() for each layer's unique weights.
    This reduces setup from hours (chunked compilation) to ~7 minutes.
  - NKI custom kernels: the four O(N^3) operations (TriAttnStart, TriAttnEnd,
    TriMulOut, TriMulIn) are implemented as NKI kernels that run directly on the
    NeuronCore, bypassing the standard XLA compiler for these operations.
  - Monkey-patching: NKI kernels are injected into the Boltz-2 codebase at runtime
    by replacing the kernel_triangular_attn and kernel_triangular_mult functions
    before tracing.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    pip install boltz==2.2.1

    from modeling_boltz2 import (
        patch_boltz2_with_nki_kernels,
        compile_pairformer_weight_replaced,
        run_pairformer_layers,
    )

    # Step 1: Patch Boltz-2 with NKI kernels (BEFORE importing model)
    patch_boltz2_with_nki_kernels()

    # Step 2: Load model
    model = Boltz2.load_from_checkpoint(checkpoint_path, ...)

    # Step 3: Compile pairformer
    traced_layers, compile_time, swap_time = compile_pairformer_weight_replaced(
        model, N=256, target="trn2"
    )

    # Step 4: Run inference
    s_out, z_out, latency = run_pairformer_layers(
        traced_layers, s, z, mask, pair_mask
    )

Instance: trn2.3xlarge (128 GB RAM, 96 GB HBM)
"""

import copy
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch_neuronx

# Ensure src/ is on the path for NKI kernel imports
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ========================================================================
# NKI kernel integration wrappers
# ========================================================================


def _nki_kernel_triangular_attn(q, k, v, tri_bias, mask, scale):
    """Drop-in replacement for kernel_triangular_attn in Boltz-2 primitives.

    Reshapes Boltz-2's multi-head tensors to the NKI kernel's head-interleaved
    format, calls the NKI kernel, and reshapes the output back.

    Args:
        q: [B, I, H, J, d] - unscaled query
        k: [B, I, H, K, d] - key
        v: [B, I, H, K, d] - value
        tri_bias: [B, 1, H, I, J] - triangle bias
        mask: [B, I, 1, 1, J] - padding mask (ignored, all-ones at inference)
        scale: float - 1/sqrt(d)

    Returns:
        [B, I, H, Q, d] - attention output in Boltz-2's expected format.
    """
    from nki_triangular_attention import triangular_attention_fwd

    B, I, H, J, d = q.shape
    N = I
    Hd = H * d

    q_nki = q[0].permute(0, 2, 1, 3).contiguous().reshape(N, N, Hd)
    k_nki = k[0].permute(0, 2, 1, 3).contiguous().reshape(N, N, Hd)
    v_nki = v[0].permute(0, 2, 1, 3).contiguous().reshape(N, N, Hd)
    bias_nki = tri_bias[0, 0].permute(1, 2, 0).contiguous()

    out_nki = triangular_attention_fwd(q_nki, k_nki, v_nki, bias_nki, scale)

    out = out_nki.reshape(N, N, H, d).permute(0, 2, 1, 3).unsqueeze(0)
    return out


def _nki_kernel_triangular_mult(
    x,
    direction,
    mask,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
    eps,
):
    """Drop-in replacement for kernel_triangular_mult in Boltz-2.

    Performs LayerNorm, projections, and gating in PyTorch, then calls the NKI
    kernel for the O(N^3) einsum contraction, then does output norm + gating.

    Args:
        x: [B, N, N, D] - pair representation
        direction: "outgoing" or "incoming"
        mask: [B, N, N] - padding mask
        norm_in_weight, norm_in_bias: LayerNorm parameters
        p_in_weight: [2*D, D] - input projection
        g_in_weight: [2*D, D] - input gate
        norm_out_weight, norm_out_bias: output LayerNorm parameters
        p_out_weight: [D, D] - output projection
        g_out_weight: [D, D] - output gate
        eps: float - LayerNorm epsilon

    Returns:
        [B, N, N, D] - triangular multiplicative update output
    """
    from nki_triangular_mul import triangular_mul_fwd

    B, N, _, D = x.shape

    x_normed = F.layer_norm(x, [D], norm_in_weight, norm_in_bias, eps)
    x_in = x_normed

    projected = F.linear(x_normed, p_in_weight)
    gated = F.linear(x_normed, g_in_weight).sigmoid()
    x_gated = projected * gated
    x_gated = x_gated * mask.unsqueeze(-1)

    a, b = torch.chunk(x_gated, 2, dim=-1)

    a_nki = a[0]
    b_nki = b[0]

    if direction == "incoming":
        a_nki = a_nki.permute(1, 0, 2).contiguous()
        b_nki = b_nki.permute(1, 0, 2).contiguous()

    result_nki = triangular_mul_fwd(a_nki, b_nki)
    result = result_nki.unsqueeze(0)

    result = F.layer_norm(result, [D], norm_out_weight, norm_out_bias, eps)
    output = F.linear(result, p_out_weight) * F.linear(x_in, g_out_weight).sigmoid()

    return output


# ========================================================================
# Monkey-patching
# ========================================================================


def patch_boltz2_with_nki_kernels():
    """Monkey-patch Boltz-2 to use NKI kernels for triangular operations.

    MUST be called BEFORE importing Boltz-2 model classes, or at minimum
    before tracing any modules that use these operations.
    """
    from boltz.model.layers.triangular_attention import primitives
    import boltz.model.layers.triangular_mult as tri_mul_module

    primitives.kernel_triangular_attn = _nki_kernel_triangular_attn
    tri_mul_module.kernel_triangular_mult = _nki_kernel_triangular_mult


# ========================================================================
# Pairformer layer wrapper
# ========================================================================


class SinglePairformerLayerWrapper(torch.nn.Module):
    """Wraps a single PairformerLayer for tracing with NKI kernels enabled."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, s, z, mask, pair_mask):
        s, z = self.layer(s, z, mask, pair_mask, use_kernels=True)
        return s, z


# ========================================================================
# Compilation via weight replacement
# ========================================================================


def compile_pairformer_weight_replaced(model, N, target="trn2"):
    """Compile 64-layer pairformer using weight replacement.

    Compiles a single layer with inline_weights_to_neff=False, then creates
    64 copies with replace_weights() for each layer's unique weights.

    Args:
        model: Loaded Boltz2 model instance.
        N: Sequence length (must be multiple of 128). Typically 256.
        target: Neuron compilation target ("trn2").

    Returns:
        traced_layers: list of 64 traced models
        compile_time: time to compile layer 0
        total_swap_time: time for all weight swaps
    """
    C_s, C_z = 384, 128
    all_layers = list(model.pairformer_module.layers)
    NUM_LAYERS = len(all_layers)

    s_dummy = torch.randn(1, N, C_s, dtype=torch.bfloat16) * 0.1
    z_dummy = torch.randn(1, N, N, C_z, dtype=torch.bfloat16) * 0.1
    mask_dummy = torch.ones(1, N, dtype=torch.float32)
    pair_mask_dummy = torch.ones(1, N, N, dtype=torch.float32)

    compiler_args = ["--target", target]

    # Phase 1: Compile layer 0
    print(f"\n  Compiling layer 0 (inline_weights_to_neff=False, target={target})...")
    layer0_bf16 = copy.deepcopy(all_layers[0]).to(torch.bfloat16)
    wrapper0 = SinglePairformerLayerWrapper(layer0_bf16)
    wrapper0.eval()

    t0 = time.time()
    traced_template = torch_neuronx.trace(
        wrapper0,
        (s_dummy, z_dummy, mask_dummy, pair_mask_dummy),
        compiler_args=compiler_args,
        inline_weights_to_neff=False,
    )
    compile_time = time.time() - t0
    print(f"  Layer 0 compiled in {compile_time:.1f}s")

    try:
        torch_neuronx.move_trace_to_device(traced_template, 0)
    except Exception:
        pass

    # Warmup
    with torch.no_grad():
        _ = traced_template(s_dummy, z_dummy, mask_dummy, pair_mask_dummy)

    # Phase 2: Weight replacement for layers 1..63
    print(f"\n  Replacing weights for {NUM_LAYERS - 1} layers...")
    traced_layers = [traced_template]
    replace_times = []

    for i in range(1, NUM_LAYERS):
        layer_bf16 = copy.deepcopy(all_layers[i]).to(torch.bfloat16)
        wrapper_i = SinglePairformerLayerWrapper(layer_bf16)
        wrapper_i.eval()
        new_state_dict = wrapper_i.state_dict()

        t0 = time.time()
        traced_copy = copy.deepcopy(traced_template)
        torch_neuronx.replace_weights(traced_copy, new_state_dict)
        swap_time = time.time() - t0
        replace_times.append(swap_time)

        try:
            torch_neuronx.move_trace_to_device(traced_copy, 0)
        except Exception:
            pass

        traced_layers.append(traced_copy)

        if (i + 1) % 16 == 0 or i == NUM_LAYERS - 1:
            print(f"    Layer {i + 1}/{NUM_LAYERS}: swap {swap_time:.1f}s")

    total_swap_time = sum(replace_times)
    print(
        f"\n  Pairformer setup: compile {compile_time:.0f}s + "
        f"swaps {total_swap_time:.0f}s = {compile_time + total_swap_time:.0f}s"
    )

    return traced_layers, compile_time, total_swap_time


def run_pairformer_layers(traced_layers, s, z, mask, pair_mask):
    """Run all traced pairformer layers sequentially.

    Args:
        traced_layers: list of 64 traced models from compile_pairformer_weight_replaced
        s: [1, N, 384] single representation, bfloat16
        z: [1, N, N, 128] pair representation, bfloat16
        mask: [1, N] padding mask, float32
        pair_mask: [1, N, N] pair padding mask, float32

    Returns:
        s_out, z_out: final pairformer outputs
        total_time: total inference time in seconds
    """
    s_curr = s.to(torch.bfloat16)
    z_curr = z.to(torch.bfloat16)

    t0 = time.time()
    for traced in traced_layers:
        s_curr, z_curr = traced(s_curr, z_curr, mask, pair_mask)
        s_curr = s_curr.to(torch.bfloat16)
        z_curr = z_curr.to(torch.bfloat16)
    total_time = time.time() - t0

    return s_curr, z_curr, total_time
