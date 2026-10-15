"""Confirm the HF-DINOv3 miscompile root cause: torch.split() on dim -2 in apply_rotary_pos_emb.

transformers' DINOv3 attention does:
    q_prefix, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
on a 4D [B, heads, tokens, head_dim] tensor. DINOv3 project Task 017 (SDK 2.30) found
neuronx-cc miscompiles torch.split() along dim>=2 of 4D tensors on Inferentia2, collapsing
ViT-L cosine similarity to ~0.13. The upstream facebookresearch/dinov3 repo uses index
slicing in Attention.apply_rope -- which is why the reference-repo architecture compiles
correctly and the HF port does not.

Test: monkey-patch ONLY apply_rotary_pos_emb to use index slicing (numerically identical),
change nothing else, re-trace with real weights. If cos goes 0.50 -> ~1.0, root cause
confirmed. Also a pure-op micro-repro with no model at all, for an upstream report.
"""

import json
import os

import torch
import torch.nn.functional as F
import torch_neuronx

CARGS = ["--auto-cast", "none"]
REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"


def cos(a, b):
    return F.cosine_similarity(a.flatten().float().unsqueeze(0),
                               b.flatten().float().unsqueeze(0)).item()


def apply_rope_slicing(q, k, cos_, sin_, **kw):
    """Identical math to transformers' apply_rotary_pos_emb, but index slicing not split()."""
    import transformers.models.dinov3_vit.modeling_dinov3_vit as M
    n = q.shape[-2] - sin_.shape[-2]
    qp, qq = q[:, :, :n, :], q[:, :, n:, :]
    kp, kk = k[:, :, :n, :], k[:, :, n:, :]
    qq = (qq * cos_) + (M.rotate_half(qq) * sin_)
    kk = (kk * cos_) + (M.rotate_half(kk) * sin_)
    return torch.cat((qp, qq), dim=-2), torch.cat((kp, kk), dim=-2)


class SplitOp(torch.nn.Module):
    """Micro-repro: just the split + rotate + cat, no weights, no model."""
    def __init__(self, n_prefix, use_split):
        super().__init__(); self.n, self.s = n_prefix, use_split
    def forward(self, q, c, s):
        n, p = self.n, q.shape[-2] - self.n
        if self.s:
            a, b = q.split((n, p), dim=-2)
        else:
            a, b = q[:, :, :n, :], q[:, :, n:, :]
        h = b.shape[-1] // 2
        rot = torch.cat((-b[..., h:], b[..., :h]), dim=-1)
        return torch.cat((a, b * c + rot * s), dim=-2)


def main():
    out = {}
    print("== micro-repro: split vs index-slice, no model ==", flush=True)
    torch.manual_seed(0)
    q = torch.randn(1, 6, 201, 64) * 20      # real DINOv3 q has large magnitudes
    c, s = torch.randn(196, 64), torch.randn(196, 64)
    for use_split in (True, False):
        mod = SplitOp(5, use_split).eval()
        ref = mod(q, c, s)
        got = torch_neuronx.trace(mod, (q, c, s), compiler_args=CARGS)(q, c, s)
        k = "split" if use_split else "index_slice"
        out[f"micro_{k}"] = {"cos": round(cos(ref, got), 6),
                             "max_abs": float((ref - got).abs().max())}
        print(f"  {k:12s} cos={out[f'micro_{k}']['cos']}  "
              f"max_abs={out[f'micro_{k}']['max_abs']:.4g}", flush=True)

    print("\n== real weights, full model: stock vs one-function patch ==", flush=True)
    import transformers.models.dinov3_vit.modeling_dinov3_vit as M
    from transformers import DINOv3ViTModel
    tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    orig = M.apply_rotary_pos_emb
    for label, fn in (("stock (split)", orig), ("patched (index slice)", apply_rope_slicing)):
        M.apply_rotary_pos_emb = fn
        m = DINOv3ViTModel.from_pretrained(REPO, token=tok, dtype=torch.float32).eval()
        torch.manual_seed(0)
        x = torch.randn(1, 3, 224, 224)

        class W(torch.nn.Module):
            def __init__(s): super().__init__(); s.m = m
            def forward(s, xx): return s.m(pixel_values=xx).last_hidden_state
        with torch.no_grad():
            ref = W()(x)
        got = torch_neuronx.trace(W().eval(), x, compiler_args=CARGS)(x)
        out[label] = {"cos": round(cos(ref, got), 6),
                      "cls_cos": round(cos(ref[:, 0], got[:, 0]), 6)}
        print(f"  {label:24s} cos={out[label]['cos']}  CLS cos={out[label]['cls_cos']}",
              flush=True)
        del m
    M.apply_rotary_pos_emb = orig

    with open("results_hf_rootcause.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results_hf_rootcause.json")


if __name__ == "__main__":
    main()
