#!/usr/bin/env python3
"""Isolation probe: does the nkilib `qkv` kernel emit BSD as a contiguous
[Q | gate | K | V] block in weight-row order when num_q_heads is doubled?

WHY THIS EXISTS
---------------
The qkvgate decode path folds the output-gate projection into the fused Wqkv
weight and projects Q/gate/K/V in ONE nkilib `qkv` kernel call with
`num_q_heads = 2 * num_heads` (gate masquerades as a second set of Q heads).
It then splits the result assuming the layout is contiguous:

    q_width  = num_heads * head_dim
    Q    = packed[..., 0          : q_width]
    gate = packed[..., q_width    : 2*q_width]
    K    = packed[..., 2*q_width  : 2*q_width + kv_width]
    V    = packed[..., 2*q_width + kv_width : ]

Everything UPSTREAM (weight packing) and DOWNSTREAM (sigmoid gate, attention,
o_proj) is shared with the known-good `qknormrope` baseline and is proven.
The ONE never-isolated assumption is the kernel's output head ordering under a
doubled num_q_heads. This probe tests exactly that and PRINTS the kernel's real
permutation so the fix is unambiguous.

HOW IT WORKS
------------
We build an Wqkv whose every output column carries an identifiable "code":
    Q    head h -> all head_dim columns == 100 + h
    gate head h -> 200 + h
    K    head h -> 300 + h
    V    head h -> 400 + h
With a ones() input and no bias, output[col] == code(col) (the column's
row-sum). We then run the SAME kernel call the model uses and decode the codes
straight off the output. The expected (contiguous) order is
    [Q0..Q(n-1), gate0..gate(n-1), K0..K(kv-1), V0..V(kv-1)]
If the kernel reorders (e.g. GQA-interleaves Q with K/V, or splits the doubled
q-region differently), the decoded order reveals precisely how -> that IS the
corrected split.

RUN ON A HOST WITH nkilib + neuronxcc (e.g. the compile host 16.51.94.87):
    python probe_qkvgate_kernel_layout.py --num-heads 2 --num-kv-heads 1 --head-dim 256
Two host-specific knobs are flagged with `HOST:` below — adjust if the local
nki API differs.
"""
import argparse

import torch

import nki as _nkilib_nki
from nkilib.core.qkv.qkv import qkv as _nkilib_qkv
from nkilib.core.utils.common_types import (
    NormType as NormType,
    QKVOutputLayout as QKVOutputLayout,
    QuantizationType as QuantizationType,
)

KERNEL = _nkilib_nki.jit(_nkilib_qkv)

Q_BASE, GATE_BASE, K_BASE, V_BASE = 100, 200, 300, 400


def region_of(code: int) -> str:
    base = (code // 100) * 100
    return {100: "Q", 200: "gate", 300: "K", 400: "V"}.get(base, "?")


def build_identifiable_weight(hidden, num_q, num_kv, head_dim, dtype):
    """Wqkv with column codes. Layout matches the packed weight rows:
    [Q(num_q) | gate(num_q) | K(num_kv) | V(num_kv)]  (num_q == real num_heads).
    Returns a Linear-style weight [out_features, hidden]; each row set so that
    a ones() input yields output[col] == code(col)."""
    regions = (
        [(Q_BASE, h) for h in range(num_q)]
        + [(GATE_BASE, h) for h in range(num_q)]
        + [(K_BASE, h) for h in range(num_kv)]
        + [(V_BASE, h) for h in range(num_kv)]
    )
    out_features = len(regions) * head_dim
    w = torch.zeros(out_features, hidden, dtype=torch.float32)
    col = 0
    for base, h in regions:
        code = base + h
        w[col : col + head_dim, :] = code / hidden  # row-sum == code
        col += head_dim
    return w.to(dtype)


def decode_layout(out_row, head_dim):
    """out_row: 1D tensor of length out_features. Returns list of decoded codes,
    one per head block (head_dim columns), using the block's median value."""
    codes = []
    n_blocks = out_row.shape[0] // head_dim
    for b in range(n_blocks):
        block = out_row[b * head_dim : (b + 1) * head_dim].float()
        codes.append(int(round(block.median().item())))
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-heads", type=int, default=2, help="real num_heads (per rank)")
    ap.add_argument("--num-kv-heads", type=int, default=1)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lnc", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    n, kv, hd, hidden = args.num_heads, args.num_kv_heads, args.head_dim, args.hidden

    weight = build_identifiable_weight(hidden, n, kv, hd, dtype)
    x = torch.ones(1, 1, hidden, dtype=dtype)

    # ----- reference: plain matmul, then the SAME split the model uses -----
    ref = (x.float() @ weight.float().t()).squeeze(0).squeeze(0)  # [out_features]
    ref_codes = decode_layout(ref, hd)
    print("REFERENCE (contiguous weight-row order):")
    print("  ", [f"{region_of(c)}{c % 100}" for c in ref_codes])

    # ----- kernel under test: exact call from _qkv_gate_packed_projection_nki -----
    # HOST: weight orientation. The model applies transpose_parallel_linear_layer
    # before handing the weight to the kernel. If the call below errors on shape,
    # pass weight.t().contiguous() instead.
    kernel_weight = weight
    # HOST: device vs simulator. On a Trainium core the jitted call below runs
    # directly. For CPU simulation use: out = _nkilib_nki.simulate_kernel(
    #     _nkilib_qkv, input=x, fused_qkv_weights=kernel_weight, ...same kwargs...)
    packed = KERNEL[args.lnc](
        input=x,
        fused_qkv_weights=kernel_weight,
        output_layout=QKVOutputLayout.BSD,
        bias=None,
        fused_residual_add=False,
        mlp_prev=None,
        attention_prev=None,
        fused_norm_type=NormType.NO_NORM,
        gamma_norm_weights=None,
        norm_eps=1e-6,
        fused_rope=False,
        cos_cache=None,
        sin_cache=None,
        quantization_type=QuantizationType.NONE,
        qkv_w_scale=None,
        qkv_in_scale=None,
        d_head=hd,
        num_q_heads=n * 2,   # gate folded in as extra Q heads
        num_kv_heads=kv,
    )
    packed = torch.as_tensor(packed).reshape(-1)
    out_codes = decode_layout(packed, hd)
    print("KERNEL OUTPUT (actual order):")
    print("  ", [f"{region_of(c)}{c % 100}" for c in out_codes])

    if out_codes == ref_codes:
        print("\nRESULT: layout MATCHES -> split is correct; bug is NOT head order.")
        print("        Re-run with FP8 weights/scales to test the ROW-quant path.")
    else:
        print("\nRESULT: layout MISMATCH -> the tensor_split offsets are wrong.")
        print("        Map each output block to its code above to derive the fix:")
        for i, c in enumerate(out_codes):
            print(f"          out block {i:>2} -> {region_of(c)} head {c % 100}")


if __name__ == "__main__":
    main()
