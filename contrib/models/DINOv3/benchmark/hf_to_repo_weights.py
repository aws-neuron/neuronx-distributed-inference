"""Load REAL pretrained DINOv3 weights into the dinov3-repo architecture.

Why this exists. Two separate implementations of DINOv3 are in play:

  A) `dinov3.hub.backbones` -- the reference repo. This is what the contrib
     traces, and it compiles CORRECTLY (cos 0.9999999 with random weights).
  B) `transformers.AutoModel` -- the HuggingFace port. This is the only format
     the pretrained checkpoints are published in.

Implementation (B) does NOT compile correctly under torch_neuronx.trace() on SDK
2.31: cos_sim ~0.47-0.62 even with --auto-cast=none, while plain torch.jit.trace
of the same module on CPU gives exactly 1.0. So the failure is in the Neuron
lowering of the HF module, not in the weights and not in the harness.

To validate real weights on the architecture the contrib actually ships, the HF
state_dict has to be remapped into the dinov3-repo module. The two use different
parameter names for the same tensors.

This module does that remap and verifies it by comparing the two implementations
on CPU -- if the remap is wrong, CPU-vs-CPU cos will not be ~1.0 and we stop
rather than reporting a bogus accuracy number.
"""

import re

import torch


def hf_to_repo_state_dict(hf_sd, ref_sd):
    """Remap a transformers DINOv3 state_dict to dinov3-repo parameter names.

    Verified name/shape correspondence (ViT-S/16):

      HF                                     repo
      embeddings.cls_token        (1,1,C)  -> cls_token         (1,1,C)
      embeddings.mask_token       (1,1,C)  -> mask_token        (1,C)    [squeeze]
      embeddings.register_tokens  (1,4,C)  -> storage_tokens    (1,4,C)
      embeddings.patch_embeddings.{weight,bias} -> patch_embed.proj.{weight,bias}
      layer.{i}.norm{1,2}.*                -> blocks.{i}.norm{1,2}.*
      layer.{i}.attention.{q,k,v}_proj.*   -> blocks.{i}.attn.qkv.*  [concat]
      layer.{i}.attention.o_proj.*         -> blocks.{i}.attn.proj.*
      layer.{i}.layer_scale{1,2}.lambda1   -> blocks.{i}.ls{1,2}.gamma
      layer.{i}.mlp.up_proj.*              -> blocks.{i}.mlp.fc1.*
      layer.{i}.mlp.down_proj.*            -> blocks.{i}.mlp.fc2.*
      norm.*                               -> norm.*

    Notes:
      - HF publishes NO k_proj.bias. The repo's fused qkv.bias expects one, so the
        k slice is zero-filled (which is what the HF attention does implicitly).
      - repo-only buffers (rope_embed.periods, attn.qkv.bias_mask) are not in the
        HF checkpoint and are left at their constructed values.
    """
    out = {}
    qkv = {}

    for k, v in hf_sd.items():
        k = k.replace("backbone.", "")

        if k.endswith("embeddings.cls_token"):
            out["cls_token"] = v; continue
        if k.endswith("embeddings.mask_token"):
            # (1,1,C) -> (1,C)
            out["mask_token"] = v.reshape(ref_sd["mask_token"].shape); continue
        if k.endswith("embeddings.register_tokens"):
            out["storage_tokens"] = v; continue
        if "embeddings.patch_embeddings." in k:
            out["patch_embed.proj." + k.rsplit(".", 1)[1]] = v; continue

        m = re.search(r"layer\.(\d+)\.(.*)", k)
        if not m:
            if k.startswith("norm."):
                out[k] = v
            continue
        i, rest = int(m.group(1)), m.group(2)

        if rest.startswith(("norm1.", "norm2.")):
            out[f"blocks.{i}.{rest}"] = v
        elif rest.startswith("layer_scale1."):
            out[f"blocks.{i}.ls1.gamma"] = v
        elif rest.startswith("layer_scale2."):
            out[f"blocks.{i}.ls2.gamma"] = v
        elif rest.startswith("attention.o_proj."):
            out[f"blocks.{i}.attn.proj." + rest.rsplit(".", 1)[1]] = v
        elif re.match(r"attention\.[qkv]_proj\.", rest):
            which = rest.split(".")[1][0]
            suffix = rest.rsplit(".", 1)[1]
            qkv.setdefault((i, suffix), {})[which] = v
        elif rest.startswith(("mlp.up_proj.", "mlp.gate_proj.")):
            out[f"blocks.{i}.mlp.fc1." + rest.rsplit(".", 1)[1]] = v
        elif rest.startswith("mlp.down_proj."):
            out[f"blocks.{i}.mlp.fc2." + rest.rsplit(".", 1)[1]] = v

    # Fuse q/k/v. HF omits k_proj.bias, so zero-fill that slice.
    for (i, suffix), parts in qkv.items():
        if "q" not in parts or "v" not in parts:
            continue
        kk = parts.get("k")
        if kk is None and suffix == "bias":
            kk = torch.zeros_like(parts["q"])
        if kk is None:
            continue
        out[f"blocks.{i}.attn.qkv.{suffix}"] = torch.cat(
            [parts["q"], kk, parts["v"]], dim=0)

    return out


def load_real_into_repo(key, repo_dir, hf_repo, token=None, verify=True):
    """Build the dinov3-repo model and load the real HF checkpoint into it.

    Returns (model, report). `report["remap_cos"]` is the CPU-vs-CPU agreement
    between the HF implementation and the remapped repo implementation -- if that
    is not ~1.0 the remap is wrong and any downstream accuracy number is
    meaningless, so callers must check it.
    """
    import sys
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import dinov3.hub.backbones as B
    from transformers import AutoModel

    factory = {
        "vit_s": B.dinov3_vits16, "vit_b": B.dinov3_vitb16,
        "vit_l": B.dinov3_vitl16, "vit_h_plus": B.dinov3_vith16plus,
    }[key]
    model = factory(pretrained=False)
    model.eval()

    hf = AutoModel.from_pretrained(hf_repo, token=token, dtype=torch.float32)
    hf.eval()

    mapped = hf_to_repo_state_dict(hf.state_dict(), model.state_dict())
    missing, unexpected = model.load_state_dict(mapped, strict=False)

    report = {
        "n_mapped": len(mapped),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "missing_sample": list(missing)[:8],
        "unexpected_sample": list(unexpected)[:8],
    }

    if verify:
        import torch.nn.functional as F
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            a = hf(pixel_values=x).pooler_output.float()
            b = model(x).float()
        report["remap_cos"] = float(
            F.cosine_similarity(a, b, dim=1).mean().item())

    return model, report
