"""Fused residual-add + affine-free LayerNorm, ported to the inf2 torch_neuronx.trace() path.

SOURCE: the residual+LayerNorm kernel from a whole-block fused DINOv3 NKI path that measured
1.5-3.4x over the compiler on Trainium2 (a different compilation path).

This is the FEASIBILITY GATE for porting that fused-block path to Inferentia2 under
`torch_neuronx.trace()`. It is the simplest of the fused kernels -- pure elementwise + one
reduction, no matmul, no dma_transpose, no nkilib dependency -- so if THIS cannot be made to
run correctly and fast on gen2 under trace, the attention/MLP kernels certainly cannot.

Port deltas vs the source kernel, per the five calling-convention differences
documented in nki_gelu_trace.py:

  1. dispatch:  `@nki_op` + `kernel[LNC](...)`  ->  `nki_jit()(kernel)` from
                torch_neuronx.xla_impl.ops. `torch_neuronx.nki_op` does not exist on SDK 2.31.
  2. outputs:   returned `nl.shared_hbm` tensors  ->  pre-allocated output PARAMETERS.
  3. LNC:       inf2 has no LNC; `nl.program_id` sharding removed (single program, all rows).
  4. shapes:    R and H passed as Python ints -- trace() cannot resolve `x.shape` inside the
                kernel ("failed to resolve name 'input_hbm.shape'").
  5. GEN2 ISA:  the source uses the newer `nisa.activation_reduce` / fused two-op
                `tensor_scalar`. Those are kept only if they exist on this NKI; the gate
                script checks which instruction forms compile on gen2.

The math is unchanged: x_new = a + b (fp32), z = (x_new - mean) * rstd (bf16), eps on var.
"""

import torch
import torch.nn.functional as F

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from torch_neuronx.xla_impl.ops import nki_jit

PMAX = 128


@nki.jit
def res_ln_kernel(a_hbm, b_hbm, x_out, z_out, R, H, eps):
    """a_hbm [R,H] fp32, b_hbm [R,H] fp32 -> x_out = a+b fp32, z_out = LN(a+b) fp32.

    One 128-row tile at a time. R must be a multiple of 128 (caller pads).
    """
    inv_h = 1.0 / float(H)
    i_f = nl.arange(H)[None, :]
    n_t = R // PMAX
    for t in nl.affine_range(n_t):
        i_p = t * PMAX + nl.arange(PMAX)[:, None]
        a = nl.load(a_hbm[i_p, i_f])
        b = nl.load(b_hbm[i_p, i_f])
        s = nl.add(a, b)
        nl.store(x_out[i_p, i_f], value=s)

        mean = nl.multiply(nl.sum(s, axis=1, keepdims=True), inv_h)
        d = nl.subtract(s, mean)
        var = nl.multiply(nl.sum(nl.multiply(d, d), axis=1, keepdims=True), inv_h)
        rstd = nl.rsqrt(nl.add(var, eps))
        z = nl.multiply(d, rstd)
        nl.store(z_out[i_p, i_f], value=z)


_k = nki_jit()(res_ln_kernel)


def nki_res_ln(a, b, eps=1e-6):
    """Residual add + affine-free LN over the last dim. a, b: [..., H].

    CPU fallback: torch_neuronx.trace() runs a CPU forward to build the graph, and the
    kernel only dispatches on XLA tensors.
    """
    if a.device.type != "xla":
        s = a + b
        return s, F.layer_norm(s, (s.shape[-1],), eps=eps)
    shp = a.shape
    H = shp[-1]
    a2 = a.reshape(-1, H)
    b2 = b.reshape(-1, H)
    R = a2.shape[0]
    pad = (-R) % PMAX
    if pad:
        a2 = torch.cat([a2, a2.new_zeros(pad, H)], 0)
        b2 = torch.cat([b2, b2.new_zeros(pad, H)], 0)
    Rp = a2.shape[0]
    x_out = torch.empty_like(a2)
    z_out = torch.empty_like(a2)
    _k(a2, b2, x_out, z_out, Rp, H, eps)
    return x_out[:R].reshape(shp), z_out[:R].reshape(shp)
