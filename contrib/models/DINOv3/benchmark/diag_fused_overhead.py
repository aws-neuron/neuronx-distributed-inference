"""Why did the fused residual+LN kernel LOSE inside a block (-4 to -8%)?

Before deciding whether the larger fused kernels (attention+LN, MLP+LN) are worth porting,
separate two explanations:

  (A) INTEGRATION OVERHEAD -- the wrapper, not the kernel. nki_res_ln pads R from 201 to
      256 rows with torch.cat, slices back, and the caller applies norm2's affine as two
      extra elementwise ops. That is ~27% extra rows plus 3+ extra ops the compiler-fused
      path never pays.
  (B) BOUNDARY COST -- the kernel boundary itself. Even a perfect kernel forces the
      attention output to round-trip HBM and blocks the compiler from fusing LN2 into the
      following fc1 GEMM. This is what killed the single-op GELU kernel.

Test: three arms on the same block.
  compiler        -- stock
  fused_padded    -- the G3 arm (pads 201->256, affine applied outside)
  fused_lean      -- kernel takes N=201 directly via a masked final tile (no cat/slice),
                     and folds norm2's gamma/beta INTO the kernel (no extra ops).

If fused_lean recovers to >= compiler, the loss was (A) and the bigger kernels are worth
porting. If fused_lean still loses, it is (B): on trace, inserting ANY kernel boundary
inside the block costs more than the fusion saves, and the multi-op kernels would face
the same cost at every boundary they introduce.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3"))

import torch
import torch.nn.functional as F
import torch_neuronx

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from torch_neuronx.xla_impl.ops import nki_jit

PMAX = 128


@nki.jit
def res_ln_affine_masked(a_hbm, b_hbm, g_hbm, beta_hbm, x_out, z_out, R, H, eps):
    """x_out = a + b ; z_out = LN(x_out) * g + beta. R need NOT be a multiple of 128 --
    the last tile is masked, so no host-side pad/slice is needed."""
    inv_h = 1.0 / float(H)
    i_f = nl.arange(H)[None, :]
    g = nl.load(g_hbm[nl.arange(1)[:, None], i_f])
    bt = nl.load(beta_hbm[nl.arange(1)[:, None], i_f])
    n_t = (R + PMAX - 1) // PMAX
    for t in nl.affine_range(n_t):
        i_p = t * PMAX + nl.arange(PMAX)[:, None]
        msk = i_p < R
        a = nl.load(a_hbm[i_p, i_f], mask=msk)
        b = nl.load(b_hbm[i_p, i_f], mask=msk)
        s = nl.add(a, b)
        nl.store(x_out[i_p, i_f], value=s, mask=msk)
        mean = nl.multiply(nl.sum(s, axis=1, keepdims=True), inv_h)
        d = nl.subtract(s, mean)
        var = nl.multiply(nl.sum(nl.multiply(d, d), axis=1, keepdims=True), inv_h)
        rstd = nl.rsqrt(nl.add(var, eps))
        z = nl.add(nl.multiply(nl.multiply(d, rstd), g), bt)
        nl.store(z_out[i_p, i_f], value=z, mask=msk)


_lean = nki_jit()(res_ln_affine_masked)


class Block(torch.nn.Module):
    def __init__(self, blk, rope, mode):
        super().__init__()
        self.blk, self.rope, self.mode = blk, rope, mode

    def forward(self, x):
        b = self.blk
        r = b.ls1(b.attn(b.norm1(x), rope=self.rope))
        if self.mode == "compiler" or x.device.type != "xla":
            x_attn = x + r
            n2 = b.norm2(x_attn)
        elif self.mode == "fused_padded":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from nki_res_ln_trace import nki_res_ln
            x_attn, z = nki_res_ln(x, r, eps=b.norm2.eps)
            n2 = z * b.norm2.weight + b.norm2.bias
        else:  # fused_lean
            H = x.shape[-1]
            a2, r2 = x.reshape(-1, H), r.reshape(-1, H)
            R = a2.shape[0]
            xo, zo = torch.empty_like(a2), torch.empty_like(a2)
            _lean(a2, r2, b.norm2.weight.reshape(1, H), b.norm2.bias.reshape(1, H),
                  xo, zo, R, H, b.norm2.eps)
            x_attn, n2 = xo.reshape(x.shape), zo.reshape(x.shape)
        return x_attn + b.ls2(b.mlp(n2))


def timeit(m, x, iters=60, warmup=80, repeats=6):
    for _ in range(warmup):
        m(x)
    runs = []
    for _ in range(repeats):
        t0 = time.time()
        for _ in range(iters):
            m(x)
        runs.append(iters / (time.time() - t0))
    return [round(r, 1) for r in runs], round(max(runs), 1), abs(runs[-1] - runs[-2]) / runs[-2] < 0.03


def main():
    import dinov3.hub.backbones as B
    fac = {"vit_s": B.dinov3_vits16, "vit_b": B.dinov3_vitb16, "vit_l": B.dinov3_vitl16}
    out = []
    for key in ("vit_s", "vit_b", "vit_l"):
        torch.manual_seed(0)
        m = fac[key](pretrained=False).eval()
        with torch.no_grad():
            rope = m.rope_embed(H=14, W=14)
        blk = m.blocks[0]
        x = torch.randn(1, 201, m.cls_token.shape[-1])
        with torch.no_grad():
            ref = Block(blk, rope, "compiler")(x)
        rec = {"model": key}
        for mode in ("compiler", "fused_padded", "fused_lean"):
            try:
                tm = torch_neuronx.trace(Block(blk, rope, mode).eval(), x,
                                         compiler_args=["--auto-cast", "matmult",
                                                        "--auto-cast-type", "bf16",
                                                        "--model-type", "transformer"])
                c = F.cosine_similarity(ref.flatten().unsqueeze(0),
                                        tm(x).flatten().unsqueeze(0)).item()
                runs, best, conv = timeit(tm, x)
                rec[mode] = {"cos": c, "runs": runs, "best": best, "conv": conv}
            except Exception as e:
                rec[mode] = {"error": f"{type(e).__name__}: {str(e)[:220]}"}
            print(f"  {key} {mode:13s} {rec[mode]}", flush=True)
        base = rec["compiler"].get("best")
        for mode in ("fused_padded", "fused_lean"):
            if base and rec[mode].get("best"):
                rec[mode]["vs_compiler"] = round(rec[mode]["best"] / base, 3)
        print(f"  -> {key}: padded {rec['fused_padded'].get('vs_compiler')}x  "
              f"lean {rec['fused_lean'].get('vs_compiler')}x", flush=True)
        out.append(rec)
    with open("results_fused_diag.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results_fused_diag.json")


if __name__ == "__main__":
    main()
