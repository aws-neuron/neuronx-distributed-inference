#!/usr/bin/env python3
"""In-place patch: replace q/k/v FP8+scale with BF16 in a MiMo-V2.5-Pro
preprocessed Neuron checkpoint, for every decoder layer. Leaves MoE experts,
norms, embed, lm_head, o_proj untouched.

Rationale: Pro's attention q/k/v weights have abs_mean ~0.001-0.005, roughly
4x smaller than V2.5. The NKI blockwise FP8 accumulator on the attention
path loses enough precision at this magnitude to drift the logits across
70 layers; dequantizing q/k/v to BF16 before the matmul restores coherent
output. MoE experts (also small-scale) can stay FP8.

Note: simply adding q_proj/k_proj/v_proj to NxDI's `modules_to_not_convert`
at compile time is NOT equivalent — NxDI casts the raw fp8_e4m3fn bytes to
bfloat16 without applying the blockwise scale, which produces nonsense
weights. This script reads the HF fused qkv + scale, dequants per-group,
and writes BF16 weights back into the preprocessed Neuron checkpoint in
place.

Run under /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16 (or any venv
with torch and safetensors). Takes ~22 min on a trn2.48xlarge for 70
layers.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(
        description="In-place dequant q/k/v from FP8+scale to BF16 in the "
        "preprocessed Neuron checkpoint.",
    )
    parser.add_argument(
        "--hf_model_path",
        required=True,
        help="Path to the original HuggingFace MiMo-V2.5-Pro checkpoint "
        "(fused qkv_proj + qkv_proj.weight_scale_inv).",
    )
    parser.add_argument(
        "--neuron_model_path",
        required=True,
        help="Path to the preprocessed Neuron-FP8 checkpoint. q/k/v entries "
        "in its model_layer{N}.safetensors shards will be overwritten in "
        "place with BF16 values; the scale entries are dropped from the "
        "index.",
    )
    args = parser.parse_args()

    # Import split_qkv_fused from the neighbouring preprocess script so the
    # group layout math (hpg, qg_rows, kg_rows, vg_rows) stays in one place.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess_mimo_v2_fp8 import LazyWeightMap  # noqa: F401

    hf_src = args.hf_model_path
    neuron = args.neuron_model_path
    cfg = json.load(open(os.path.join(hf_src, "config.json")))
    hp = cfg["hybrid_layer_pattern"]
    num_hidden_layers = cfg["num_hidden_layers"]

    with open(os.path.join(hf_src, "model.safetensors.index.json")) as f:
        hf_wm = json.load(f)["weight_map"]
    lazy = LazyWeightMap(hf_src, hf_wm)

    print(
        f"Patching q/k/v -> BF16 in {neuron}/model_layer{{0..{num_hidden_layers - 1}}}.safetensors",
        flush=True,
    )
    t0 = time.time()

    for li in range(num_hidden_layers):
        layer_file = os.path.join(neuron, f"model_layer{li}.safetensors")
        if not os.path.exists(layer_file):
            print(f"  layer {li}: FILE MISSING, skip", flush=True)
            continue
        with safe_open(layer_file, framework="pt") as fp:
            layer_sd = {k: fp.get_tensor(k) for k in fp.keys()}

        is_swa = hp[li] == 1
        num_q = cfg["swa_num_attention_heads" if is_swa else "num_attention_heads"]
        num_kv = cfg["swa_num_key_value_heads" if is_swa else "num_key_value_heads"]
        hd = cfg["swa_head_dim" if is_swa else "head_dim"]
        vhd = cfg["swa_v_head_dim" if is_swa else "v_head_dim"]
        prefix = f"model.layers.{li}.self_attn"
        qkv_w = lazy.get(f"{prefix}.qkv_proj.weight")
        qkv_s = lazy.get(f"{prefix}.qkv_proj.weight_scale_inv")

        BLOCK = 128
        hpg = num_q // num_kv
        qg_rows = hpg * hd
        kg_rows = 1 * hd
        vg_rows = 1 * vhd
        R = qg_rows + kg_rows + vg_rows
        in_features = qkv_w.shape[1]
        q_blk = qg_rows // BLOCK
        k_blk = (kg_rows + BLOCK - 1) // BLOCK
        v_blk = (vg_rows + BLOCK - 1) // BLOCK
        per = q_blk + k_blk + v_blk
        padded = per * BLOCK

        w = qkv_w.to(torch.float32).view(num_kv, R, in_features)
        w_padded = torch.zeros(num_kv, padded, in_features, dtype=torch.float32)
        w_padded[:, :R, :] = w
        s = qkv_s.to(torch.float32).view(num_kv, per, (in_features + BLOCK - 1) // BLOCK)
        s_exp = s.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
        s_exp = s_exp[:, :padded, :in_features]
        deq_padded = w_padded * s_exp
        deq = deq_padded[:, :R, :]

        q_bf16 = (
            deq[:, :qg_rows, :]
            .reshape(num_kv * qg_rows, in_features)
            .contiguous()
            .to(torch.bfloat16)
        )
        k_bf16 = (
            deq[:, qg_rows : qg_rows + kg_rows, :]
            .reshape(num_kv * kg_rows, in_features)
            .contiguous()
            .to(torch.bfloat16)
        )
        v_bf16 = (
            deq[:, qg_rows + kg_rows :, :]
            .reshape(num_kv * vg_rows, in_features)
            .contiguous()
            .to(torch.bfloat16)
        )

        for key in list(layer_sd):
            if any(
                key.endswith(f".{p}.weight") or key.endswith(f".{p}.scale")
                for p in ("q_proj", "k_proj", "v_proj")
            ):
                del layer_sd[key]

        out_prefix = f"layers.{li}.self_attn"
        layer_sd[f"{out_prefix}.q_proj.weight"] = q_bf16
        layer_sd[f"{out_prefix}.k_proj.weight"] = k_bf16
        layer_sd[f"{out_prefix}.v_proj.weight"] = v_bf16

        save_file(layer_sd, layer_file)
        dt = time.time() - t0
        if li % 5 == 0 or li == num_hidden_layers - 1:
            print(
                f"  layer {li:2d} [{'swa' if is_swa else 'full'}]: "
                f"q{list(q_bf16.shape)} k{list(k_bf16.shape)} v{list(v_bf16.shape)}  "
                f"elapsed={dt:.1f}s",
                flush=True,
            )

    print("\nRewrite weight_map to reflect dtype change.", flush=True)
    idx_path = os.path.join(neuron, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    keys_to_drop = [
        k
        for k in idx["weight_map"]
        if any(k.endswith(f".{p}.scale") for p in ("q_proj", "k_proj", "v_proj"))
    ]
    for k in keys_to_drop:
        idx["weight_map"].pop(k, None)

    total = 0
    for f_path in sorted(glob.glob(os.path.join(neuron, "*.safetensors"))):
        with safe_open(f_path, framework="pt") as fp:
            for k in fp.keys():
                t = fp.get_slice(k)
                shape = list(t.get_shape())
                dt_bytes = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1}.get(
                    t.get_dtype(), 2
                )
                total += dt_bytes * max(1, int(math.prod(shape)))
    idx["metadata"] = idx.get("metadata", {})
    idx["metadata"]["total_size"] = total
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    print(f"  dropped {len(keys_to_drop)} scale entries from index", flush=True)
    print(f"  total_size now {total / 1e9:.2f} GB", flush=True)
    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
