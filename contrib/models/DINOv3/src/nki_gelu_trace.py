#!/usr/bin/env python3
"""Exact-erf GELU NKI kernel for the torch_neuronx.trace() path on inf2.

STATUS: REFERENCE IMPLEMENTATION ONLY -- MEASURED SLOWER, DO NOT ENABLE.
    Numerically exact (cos_sim 1.000000 on ViT, 0.99999 on ConvNeXt) but slower
    in 13 of 14 model/batch configurations on inf2.xlarge: ViT-S -5.8..-11.8%,
    ViT-B -7.9..-10.7%, ConvNeXt-T -8.4..-12.7%, ConvNeXt-B -7.3..-14.5%,
    ViT-L -5.5% at BS=8 and +2.8% (noise) at BS=1.
    Cause: neuronx-cc already fuses GELU into the surrounding MLP GEMMs, so a
    standalone kernel adds an HBM round-trip (load -> activate -> store) per call
    that costs more than the cheaper transcendental saves.
    Kept because it is a working, non-obvious example of the NKI `nki_jit`
    calling convention for torch_neuronx.trace() -- see the five API differences
    documented below, which are not covered in the public NKI docs.

Written against the OLDER NKI calling convention, which is what
`torch_neuronx.trace()` supports.

Why that matters: the same kernel written for the standalone `nki` wheel does NOT
work here. That form looks like:
    import nki                    # standalone wheel
    @nki.jit
    def gelu_fwd(input_hbm):
        R, C = input_hbm.shape             # <- unresolvable under trace()
        out = nl.ndarray(..., nl.shared_hbm)
        nisa.dma_copy(dst=x, src=input_hbm[a:b, 0:C])   # <- slice indexing
        return out                          # <- returns a value
Every one of those four things fails inside torch_neuronx.trace(), in order:
    --target gen2 rejected / failed to resolve 'nl.ndarray'
    failed to resolve name 'input_hbm.shape'
    invalid kwarg 'dtype': expecting 'enum', got 'NoneType'
    cannot perform multi-dimensional access into value of type 'NoneType'

The convention that DOES work (proven in Hunyuan sdk_inference/nki_rope.py,
which ships TP=4 NKI flash attention + RoPE through torch_neuronx.trace(), and
in the Boltz-2 nkilib submission) is:
    import neuronxcc.nki as nki           # compiler-BUNDLED nki, not the wheel
    from torch_neuronx.xla_impl.ops import nki_jit
    @nki.jit
    def kernel(x_in, x_out):              # output is a PRE-ALLOCATED PARAMETER
        d = x_in.shape[0]                 # .shape IS allowed on the bundled API
        i_p = nl.arange(P)[:, None]       # index tensors, not Python slices
        t = nl.load(x_in[i_p, i_f])       # nl.load / nl.store, not nisa.dma_copy
        y = nisa.activation(nl.gelu, t)   # RETURNS a tile; no dst= kwarg here
        nl.store(x_out[i_p, i_f], value=y)
    _k = nki_jit()(kernel)                # wrap for the XLA/trace dispatcher

Math is unchanged from the training kernel: one `nisa.activation(op=nl.gelu)`
per <=128-partition tile, i.e. exact erf GELU on the Scalar/activation engine.
This is the point of the kernel -- neuronx-cc's native erf lowering emits
1664 matmuls + 256 reciprocals + 2816 vector ops, where this emits ~26
activation-table ops, and the DINOv3 inference forward is Vector-bound (86%
active) with the Tensor engine at only 57%.
"""

import torch
import torch.nn.functional as F

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from torch_neuronx.xla_impl.ops import nki_jit

PMAX = 128


@nki.jit
def gelu_fwd_kernel(x_in, x_out):
    """Exact (erf) GELU. x_in/x_out: [R, C] in HBM, x_out pre-allocated.

    Tiles the partition (row) axis by PMAX=128. GELU is elementwise, so the
    partition split is arbitrary and no cross-partition communication is needed.
    """
    R = x_in.shape[0]
    C = x_in.shape[1]

    i_f = nl.arange(C)[None, :]
    n_tiles = (R + PMAX - 1) // PMAX

    for t in nl.affine_range(n_tiles):
        # Rows [t*PMAX, t*PMAX + PMAX). The trailing tile is masked rather than
        # shortened, because the tile shape must stay a compile-time constant.
        i_p = t * PMAX + nl.arange(PMAX)[:, None]
        m = i_p < R

        x = nl.load(x_in[i_p, i_f], mask=m)
        # Single scalar/activation-engine op: gelu(data * scale + bias).
        # NOTE: in the compiler-bundled neuronxcc.nki, nisa.activation RETURNS
        # the result tile -- it has no `dst=` kwarg (that is the newer standalone
        # `nki` wheel's signature). Signature here is:
        #   activation(op, data, *, bias=None, scale=1.0, ...) -> local_tile
        y = nisa.activation(nl.gelu, x, bias=None, scale=1.0)
        nl.store(x_out[i_p, i_f], value=y, mask=m)


_gelu_kernel = nki_jit()(gelu_fwd_kernel)


def nki_gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact-erf GELU over the last dim of x, via the NKI kernel.

    Flattens leading dims to 2D for the kernel and restores the shape. The
    output buffer is pre-allocated because nki_jit kernels cannot return values.
    """
    # The NKI kernel is only dispatchable on XLA tensors. The same nn.Module is
    # also executed on CPU -- torch_neuronx.trace() runs a CPU forward to build
    # the JIT graph, and validate_accuracy() runs the CPU reference -- and there
    # the kernel raises:
    #   "Expected all tensors in the given list to be XLA tensors.
    #    Element at index 0 is not an XLA tensor. Got: torch.FloatTensor"
    # So fall back to native exact-erf GELU off-device. F.gelu's default
    # approximate='none' IS the erf form, i.e. mathematically the same op the
    # kernel computes, so the CPU reference stays valid.
    if x.device.type != "xla":
        return F.gelu(x)

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    # Output buffer must live on the same device as the input. empty_like
    # inherits device/dtype/layout; empty rather than zeros because the kernel
    # overwrites every masked-in element.
    out = torch.empty_like(x2)
    _gelu_kernel(x2, out)
    return out.reshape(shape)


class NkiGELU(torch.nn.Module):
    """Drop-in replacement for nn.GELU() (exact/erf), backed by the NKI kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return nki_gelu(x)
