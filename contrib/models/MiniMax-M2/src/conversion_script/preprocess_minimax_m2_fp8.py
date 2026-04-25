"""
Preprocess MiniMax-M2 / M2.7 FP8 checkpoint for Neuron inference.

Streaming (per-layer) rewrite that mirrors the MiMo-V2-Flash preprocess
flow. The HF checkpoint ships 130 sharded safetensors files with weights
of one layer scattered across several shards; this script keeps one
`safe_open` handle live at a time (via `LazyWeightMap`) and writes one
output file per decoder layer (`model_layer{N}.safetensors`), plus
`model_extras.safetensors` for embed / norm / lm_head. Peak RAM is
~15 GB and total runtime is ~20 minutes on trn2.48xlarge.

MiniMax-M2 checkpoint layout:
  - q/k/v/o_proj and expert w1/w2/w3 are stored FP8 blockwise (128x128)
    with separate `.weight_scale_inv` fp32 tensors. The NxDI-side layers
    expect:
        Quantized{Column,Row}Parallel  for attention q/k/v/o and
        QuantizedExpertFused{Column,Row}Parallel for MoE experts
    so we rescale OCP FP8 (±448) to Neuron FP8 (±240) and emit `.scale`.
  - q/k/v and o_proj are 2D with out-dim (= num_heads * head_dim) that at
    TP=64 goes below the 128-row scale block. The NxDI modeling code
    runs a `_apply_2d_per_channel_fix` monkey-patch at compile time to
    swap these layers' q_config to PER_CHANNEL_SYMMETRIC, which expects
    per-row scales of shape [out, 1]. So for these 2D tensors we emit
    per-row Neuron-FP8 scales (one scalar per output row).
  - Expert w1/w2/w3 stay block-quantized; we fuse gate+up along the last
    dim and stack experts to match ExpertFusedRowParallelLinear's packed
    layout:
        gate_up_proj.weight [num_experts, hidden, 2*IM]
        gate_up_proj.scale  [num_experts, H_blocks, 2*IM_blocks]
        down_proj.weight    [num_experts, IM, hidden]
        down_proj.scale     [num_experts, IM_blocks, H_blocks]
  - `block_sparse_moe.gate.weight` and `e_score_correction_bias` are
    renamed into the NxDI router namespace:
        block_sparse_moe.router.linear_router.weight
        block_sparse_moe.router.e_score_correction_bias
  - embed_tokens / norm / lm_head / per-layer {input,post_attention}_
    layernorm / q_norm / k_norm are passed through BF16 unchanged.

Output layout:
  save_path/
    config.json, tokenizer.*, chat_template.jinja if present
    configuration_minimax_m2.py, modeling_minimax_m2.py (trust_remote_code)
    model.safetensors.index.json  (regenerated)
    model_extras.safetensors       (embed_tokens, norm, lm_head)
    model_layer{N}.safetensors     (one per decoder layer, N=0..61)

Usage:
    python preprocess_minimax_m2_fp8.py \\
        --hf_model_path /opt/dlami/nvme/models/MiniMax-M2.7 \\
        --save_path /opt/dlami/nvme/models/MiniMax-M2.7-Neuron-FP8 \\
        --tp_degree 64
"""

import argparse
import gc
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP8_SCALING_FACTOR = 448.0 / 240.0
NEURON_FP8_MAX = 240.0


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

def convert_bf16_to_fp8_per_row(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 [out, in] -> Neuron FP8 per-row (scales shape [out, 1])."""
    weight_float = weight.float()
    row_max_abs = weight_float.abs().max(dim=1, keepdim=True)[0]
    scales = torch.clamp(row_max_abs / NEURON_FP8_MAX, min=1e-10)
    quantized = (weight_float / scales).to(torch.float8_e4m3fn)
    return quantized, scales.to(torch.float32)


def rescale_fp8_to_per_row(
    weight: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 + blockwise scale -> Neuron per-row FP8.

    Dequantize to float32 using block broadcast, then per-row requantize.
    """
    out_features, in_features = weight.shape
    scale_h, scale_w = scale.shape

    block_h = (out_features + scale_h - 1) // scale_h
    block_w = (in_features + scale_w - 1) // scale_w

    weight_float = weight.float()
    dequantized = torch.zeros(out_features, in_features, dtype=torch.float32)
    for i in range(scale_h):
        for j in range(scale_w):
            h0, h1 = i * block_h, min((i + 1) * block_h, out_features)
            w0, w1 = j * block_w, min((j + 1) * block_w, in_features)
            dequantized[h0:h1, w0:w1] = (
                weight_float[h0:h1, w0:w1] * scale[i, j].item()
            )

    row_max_abs = dequantized.abs().max(dim=1, keepdim=True)[0]
    scales = torch.clamp(row_max_abs / NEURON_FP8_MAX, min=1e-10)
    quantized = (dequantized / scales).to(torch.float8_e4m3fn)
    return quantized, scales.to(torch.float32)


def rescale_fp8_weight_blockwise(
    weight: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep blockwise scales, just rescale into Neuron FP8 range.

    MoE expert weights stay block-quantized; only the dtype range changes.
    """
    weight_bf16 = weight.bfloat16()
    rescaled = (weight_bf16 / FP8_SCALING_FACTOR).to(torch.float8_e4m3fn)
    neuron_scale = scale.float() * FP8_SCALING_FACTOR
    return rescaled, neuron_scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Streaming weight access (one open safetensors handle at a time)
# ---------------------------------------------------------------------------

class LazyWeightMap:
    """Lazily fetch tensors from sharded safetensors, keeping one handle live."""

    def __init__(self, model_dir: str, weight_map: Dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._cur_filename: Optional[str] = None
        self._cur_handle = None

    def _open(self, filename: str):
        if self._cur_filename == filename:
            return self._cur_handle
        if self._cur_handle is not None:
            self._cur_handle.__exit__(None, None, None)
            self._cur_handle = None
        path = os.path.join(self.model_dir, filename)
        self._cur_handle = safe_open(path, framework="pt", device="cpu")
        self._cur_handle.__enter__()
        self._cur_filename = filename
        return self._cur_handle

    def get(self, key: str) -> Optional[torch.Tensor]:
        filename = self.weight_map.get(key)
        if filename is None:
            return None
        return self._open(filename).get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def close(self):
        if self._cur_handle is not None:
            self._cur_handle.__exit__(None, None, None)
            self._cur_handle = None
            self._cur_filename = None


# ---------------------------------------------------------------------------
# Per-tensor helper
# ---------------------------------------------------------------------------

def _maybe_fp8_to_neuron_per_row(
    weight: torch.Tensor, scale: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """FP8 blockwise -> per-row, or BF16 -> FP8 per-row. Pass-through otherwise."""
    if weight.dtype == torch.float8_e4m3fn and scale is not None:
        return rescale_fp8_to_per_row(weight, scale)
    if weight.dtype == torch.bfloat16:
        return convert_bf16_to_fp8_per_row(weight)
    return weight, scale


# ---------------------------------------------------------------------------
# Per-layer processing
# ---------------------------------------------------------------------------

def process_layer(
    layer_idx: int,
    lazy: LazyWeightMap,
    config: dict,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    prefix = f"model.layers.{layer_idx}."
    out_prefix = f"layers.{layer_idx}."

    # --- Layer norms (BF16, untouched) ---
    for name in ("input_layernorm", "post_attention_layernorm"):
        t = lazy.get(f"{prefix}{name}.weight")
        if t is not None:
            out[f"{out_prefix}{name}.weight"] = t.detach().clone()

    # --- Attention: q/k/v/o, all FP8 -> Neuron FP8 per-row ---
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w = lazy.get(f"{prefix}self_attn.{proj}.weight")
        if w is None:
            continue
        s = lazy.get(f"{prefix}self_attn.{proj}.weight_scale_inv")
        w2, s2 = _maybe_fp8_to_neuron_per_row(w, s)
        out[f"{out_prefix}self_attn.{proj}.weight"] = w2
        if s2 is not None:
            out[f"{out_prefix}self_attn.{proj}.scale"] = s2

    # --- QK norm (BF16) ---
    for name in ("q_norm", "k_norm"):
        t = lazy.get(f"{prefix}self_attn.{name}.weight")
        if t is not None:
            out[f"{out_prefix}self_attn.{name}.weight"] = t.detach().clone()

    # --- MoE router ---
    # HF: block_sparse_moe.gate.weight
    # NxDI: block_sparse_moe.router.linear_router.weight
    router_w = lazy.get(f"{prefix}block_sparse_moe.gate.weight")
    if router_w is not None:
        out[f"{out_prefix}block_sparse_moe.router.linear_router.weight"] = (
            router_w.detach().clone()
        )
    router_bias = lazy.get(f"{prefix}block_sparse_moe.e_score_correction_bias")
    if router_bias is not None:
        out[f"{out_prefix}block_sparse_moe.router.e_score_correction_bias"] = (
            router_bias.detach().clone()
        )

    # --- MoE experts: fuse gate+up, stack across experts ---
    num_experts = config["num_local_experts"]

    # Peek expert 0 to know shapes/dtypes.
    e0_w1 = lazy.get(f"{prefix}block_sparse_moe.experts.0.w1.weight")
    if e0_w1 is None:
        return out
    e0_w1_s = lazy.get(f"{prefix}block_sparse_moe.experts.0.w1.weight_scale_inv")

    if e0_w1.dtype == torch.float8_e4m3fn and e0_w1_s is not None:
        sample_w, sample_s = rescale_fp8_weight_blockwise(e0_w1, e0_w1_s)
    elif e0_w1.dtype == torch.bfloat16:
        raise NotImplementedError(
            f"Layer {layer_idx} expert 0 w1 is BF16; MiniMax-M2 expects FP8."
        )
    else:
        sample_w, sample_s = e0_w1, e0_w1_s

    intermediate_size, hidden_size = sample_w.shape  # [IM, H]
    # Packed transpose layout: [num_experts, H, 2*IM] for gate_up.
    gate_up_proj = torch.empty(
        num_experts, hidden_size, 2 * intermediate_size, dtype=sample_w.dtype
    )
    i_blocks, h_blocks = sample_s.shape  # [IM_blocks, H_blocks]
    gate_up_scale = torch.empty(
        num_experts, h_blocks, 2 * i_blocks, dtype=sample_s.dtype
    )

    e0_w2 = lazy.get(f"{prefix}block_sparse_moe.experts.0.w2.weight")
    e0_w2_s = lazy.get(f"{prefix}block_sparse_moe.experts.0.w2.weight_scale_inv")
    if e0_w2.dtype == torch.float8_e4m3fn and e0_w2_s is not None:
        sample_dw, sample_ds = rescale_fp8_weight_blockwise(e0_w2, e0_w2_s)
    else:
        raise NotImplementedError(
            f"Layer {layer_idx} expert 0 w2 dtype {e0_w2.dtype} not handled."
        )
    d_h_blocks, d_i_blocks = sample_ds.shape  # [H_blocks, IM_blocks]
    down_proj = torch.empty(
        num_experts, intermediate_size, hidden_size, dtype=sample_dw.dtype
    )
    down_scale = torch.empty(
        num_experts, d_i_blocks, d_h_blocks, dtype=sample_ds.dtype
    )

    # Slot expert 0 (already rescaled above).
    gate_up_proj[0, :, :intermediate_size] = sample_w.T
    gate_up_scale[0, :, :i_blocks] = sample_s.T
    e0_w3 = lazy.get(f"{prefix}block_sparse_moe.experts.0.w3.weight")
    e0_w3_s = lazy.get(f"{prefix}block_sparse_moe.experts.0.w3.weight_scale_inv")
    up_w0, up_s0 = rescale_fp8_weight_blockwise(e0_w3, e0_w3_s)
    gate_up_proj[0, :, intermediate_size:] = up_w0.T
    gate_up_scale[0, :, i_blocks:] = up_s0.T
    down_proj[0] = sample_dw.T
    down_scale[0] = sample_ds.T
    del e0_w1, e0_w1_s, e0_w3, e0_w3_s, e0_w2, e0_w2_s
    del sample_w, sample_s, sample_dw, sample_ds, up_w0, up_s0

    for e in range(1, num_experts):
        w1 = lazy.get(f"{prefix}block_sparse_moe.experts.{e}.w1.weight")
        w1_s = lazy.get(f"{prefix}block_sparse_moe.experts.{e}.w1.weight_scale_inv")
        w3 = lazy.get(f"{prefix}block_sparse_moe.experts.{e}.w3.weight")
        w3_s = lazy.get(f"{prefix}block_sparse_moe.experts.{e}.w3.weight_scale_inv")
        w2 = lazy.get(f"{prefix}block_sparse_moe.experts.{e}.w2.weight")
        w2_s = lazy.get(f"{prefix}block_sparse_moe.experts.{e}.w2.weight_scale_inv")
        g_w, g_s = rescale_fp8_weight_blockwise(w1, w1_s)
        u_w, u_s = rescale_fp8_weight_blockwise(w3, w3_s)
        d_w, d_s = rescale_fp8_weight_blockwise(w2, w2_s)
        gate_up_proj[e, :, :intermediate_size] = g_w.T
        gate_up_proj[e, :, intermediate_size:] = u_w.T
        gate_up_scale[e, :, :i_blocks] = g_s.T
        gate_up_scale[e, :, i_blocks:] = u_s.T
        down_proj[e] = d_w.T
        down_scale[e] = d_s.T
        del w1, w1_s, w3, w3_s, w2, w2_s, g_w, g_s, u_w, u_s, d_w, d_s

    out[f"{out_prefix}block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
    out[f"{out_prefix}block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.scale"] = gate_up_scale
    out[f"{out_prefix}block_sparse_moe.expert_mlps.mlp_op.down_proj.weight"] = down_proj
    out[f"{out_prefix}block_sparse_moe.expert_mlps.mlp_op.down_proj.scale"] = down_scale
    return out


# ---------------------------------------------------------------------------
# Shard saving / index
# ---------------------------------------------------------------------------

def save_shard(
    tensors: Dict[str, torch.Tensor],
    save_path: str,
    filename: str,
    weight_map: Dict[str, str],
) -> int:
    """Save a sub-state-dict; clone tensors so safetensors doesn't complain
    about views of mmapped storage. Returns bytes written."""
    path = os.path.join(save_path, filename)
    materialized: Dict[str, torch.Tensor] = {}
    total_bytes = 0
    for k, v in tensors.items():
        if not v.is_contiguous():
            v = v.contiguous()
        v = v.detach().clone()
        materialized[k] = v
        total_bytes += v.numel() * v.element_size()
    save_file(materialized, path)
    for k in materialized.keys():
        weight_map[k] = filename
    del materialized
    return total_bytes


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def process_minimax_m2_checkpoint(hf_model_path: str, save_path: str, tp_degree: int):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(hf_model_path, "model.safetensors.index.json")) as f:
        weight_map_in = json.load(f)["weight_map"]

    with open(os.path.join(hf_model_path, "config.json")) as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]

    print(
        f"Processing {num_layers} decoder layers "
        f"(hidden={config['hidden_size']}, moe_IM={config['intermediate_size']}, "
        f"experts={config['num_local_experts']})",
        flush=True,
    )

    lazy = LazyWeightMap(hf_model_path, weight_map_in)
    weight_map_out: Dict[str, str] = {}

    try:
        for li in range(num_layers):
            t0 = time.time()
            layer_sd = process_layer(li, lazy, config)
            filename = f"model_layer{li}.safetensors"
            size = save_shard(layer_sd, save_path, filename, weight_map_out)
            del layer_sd
            gc.collect()
            print(
                f"  layer {li:2d}  {size/1e9:6.2f} GB in {time.time()-t0:5.1f}s",
                flush=True,
            )

        print("Processing embed_tokens, norm, lm_head ...", flush=True)
        extras: Dict[str, torch.Tensor] = {}
        for src, dst in (
            ("model.embed_tokens.weight", "embed_tokens.weight"),
            ("model.norm.weight", "norm.weight"),
            ("lm_head.weight", "lm_head.weight"),
        ):
            t = lazy.get(src)
            if t is not None:
                extras[dst] = t.detach().clone()
            else:
                print(f"  WARNING: missing {src}", flush=True)
        if "lm_head.weight" not in extras and "embed_tokens.weight" in extras:
            extras["lm_head.weight"] = extras["embed_tokens.weight"].detach().clone()
        save_shard(extras, save_path, "model_extras.safetensors", weight_map_out)
        del extras
    finally:
        lazy.close()

    # --- Index file ---
    total_size = 0
    for f in set(weight_map_out.values()):
        total_size += os.path.getsize(os.path.join(save_path, f))
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_out,
    }
    with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # --- Copy auxiliary files (config.json, tokenizer, chat template, and the
    # trust_remote_code modules the HF config references).
    for name in sorted(os.listdir(hf_model_path)):
        if name.endswith(".safetensors"):
            continue
        if name == "model.safetensors.index.json":
            continue
        src = os.path.join(hf_model_path, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(save_path, name))

    print(f"\nPreprocess complete. total_size={total_size/1e9:.2f} GB", flush=True)
    print(f"  tensors written: {len(weight_map_out)}", flush=True)
    print(f"  output dir: {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MiniMax-M2 FP8 checkpoint for Neuron inference"
    )
    parser.add_argument("--hf_model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument(
        "--tp_degree", type=int, default=64,
        help="Tensor parallelism (currently informational only; "
             "the framework does the TP sharding at load time).",
    )
    args = parser.parse_args()
    process_minimax_m2_checkpoint(args.hf_model_path, args.save_path, args.tp_degree)


if __name__ == "__main__":
    main()
