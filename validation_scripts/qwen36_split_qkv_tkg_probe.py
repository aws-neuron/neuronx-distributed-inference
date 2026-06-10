#!/usr/bin/env python3
"""Minimal Qwen3.6 split-QKV TKG kernel probe.

Runs one preprod qkv_tkg projection shape at a time so runtime OOBs can be
localized to Q, K, or V without compiling the full model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_HANDLE_SPECIAL_SCALAR", "1")
os.environ.setdefault("UNSAFE_FP8FNCAST", "1")

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", choices=("q", "k", "v"), required=True)
    parser.add_argument("--weight-dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--lnc", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=24)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inspect-dir", default=None)
    parser.add_argument("--cpu-backend-hlo", action="store_true")
    return parser.parse_args()


def _local_heads(args: argparse.Namespace) -> int:
    if args.projection == "q":
        return args.num_attention_heads // args.tp_degree
    return args.num_key_value_heads // args.tp_degree


def _make_weight(
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = torch.randn(hidden_size, output_size, dtype=torch.bfloat16) * 0.01
    if dtype is torch.float8_e4m3fn:
        return weight.to(torch.float8_e4m3fn)
    return weight.to(dtype)


def main() -> int:
    args = _parse_args()
    if args.num_attention_heads % args.tp_degree != 0:
        raise ValueError("num_attention_heads must be divisible by tp_degree")
    if args.num_key_value_heads % args.tp_degree != 0:
        raise ValueError("num_key_value_heads must be divisible by tp_degree")

    neuron_cc_flags = f"--target trn2 --lnc {args.lnc}"
    if args.weight_dtype == "fp8":
        neuron_cc_flags += (
            " --internal-hlo2tensorizer-options=' "
            "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3 --verify-hlo=true'"
        )
    os.environ.setdefault("NEURON_CC_FLAGS", neuron_cc_flags)
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0-1" if args.lnc > 1 else "0")
    os.environ.setdefault("NEURON_RT_ENABLE_DGE_NOTIFICATIONS", "1")
    os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
    os.environ.setdefault("XLA_IR_DEBUG", "1")
    os.environ.setdefault("XLA_HLO_DEBUG", "1")
    if args.inspect_dir:
        inspect_dir = Path(args.inspect_dir).expanduser().resolve()
        inspect_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
        os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
        os.environ["NEURON_RT_INSPECT_SYSTEM_PROFILE"] = "0"
        os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = str(inspect_dir)

    from neuronxcc.nki._pre_prod_kernels import (  # noqa: PLC0415
        NormType,
        QKVOutputLayout,
        QuantizationType,
    )
    from neuronxcc.nki._pre_prod_kernels.qkv_tkg_impl import (  # noqa: PLC0415
        nki_qkv_projection_tkg_impl,
    )

    local_heads = _local_heads(args)
    output_size = local_heads * args.head_dim
    weight_dtype = (
        torch.float8_e4m3fn if args.weight_dtype == "fp8" else torch.bfloat16
    )
    quantization_type = (
        QuantizationType.ROW if args.weight_dtype == "fp8" else QuantizationType.NONE
    )
    kernel = nki_qkv_projection_tkg_impl[args.lnc]

    def run_kernel(hidden: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor):
        return kernel(
            hidden=hidden,
            qkv_w=weight,
            norm_w=None,
            fused_add=False,
            mlp_prev=None,
            attn_prev=None,
            d_head=args.head_dim,
            output_layout=QKVOutputLayout.BSD,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            qkvInSB=False,
            qkv_bias=None,
            norm_bias=None,
            hidden_actual=args.hidden_size,
            B=1,
            S=1,
            H=args.hidden_size,
            num_q_heads=local_heads,
            num_kv_heads=local_heads,
            quantization_type=quantization_type,
            qkv_w_scales=scales if args.weight_dtype == "fp8" else None,
            qkv_in_scales=None,
        )

    torch.manual_seed(args.seed)
    hidden_cpu = torch.randn(1, 1, args.hidden_size, dtype=torch.bfloat16)
    weight_cpu = _make_weight(args.hidden_size, output_size, weight_dtype)
    scale_cpu = torch.ones((128, output_size), dtype=torch.float32)

    metadata = {
        "projection": args.projection,
        "hidden_shape": list(hidden_cpu.shape),
        "weight_shape": list(weight_cpu.shape),
        "scale_shape": list(scale_cpu.shape),
        "weight_dtype": str(weight_cpu.dtype),
        "local_heads": local_heads,
        "head_dim": args.head_dim,
        "lnc": args.lnc,
        "quantization_type": str(quantization_type),
        "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS"),
        "neuron_compile_cache_url": os.environ.get("NEURON_COMPILE_CACHE_URL"),
        "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
    }
    print("PROBE_CONFIG", json.dumps(metadata, sort_keys=True), flush=True)

    if args.cpu_backend_hlo:
        import torch_neuronx.xla_impl.trace as trace  # noqa: PLC0415

        artifacts = trace.generate_hlo(
            run_kernel,
            (hidden_cpu, weight_cpu, scale_cpu),
            inline_weights_to_neff=False,
            return_weights=False,
            cpu_backend=True,
            preserve_parameters=False,
        )
        print("HLO_OK", type(artifacts), flush=True)
        return 0

    from torch_xla.core import xla_model as xm  # noqa: PLC0415

    device = xm.xla_device()
    hidden = hidden_cpu.to(device)
    weight = weight_cpu.to(device)
    scales = scale_cpu.to(device)
    output = run_kernel(hidden, weight, scales)
    xm.mark_step()
    output_cpu = output.detach().cpu()
    print(
        "OUTPUT",
        tuple(output_cpu.shape),
        output_cpu.dtype,
        "finite",
        bool(torch.isfinite(output_cpu.float()).all()),
        "sum",
        float(output_cpu.float().sum()),
        flush=True,
    )

    if args.weight_dtype == "bf16":
        ref = hidden_cpu.float().reshape(1, args.hidden_size) @ weight_cpu.float()
        diff = (output_cpu.float().reshape_as(ref) - ref).abs()
        print(
            "BF16_REF",
            "max_abs",
            float(diff.max()),
            "mean_abs",
            float(diff.mean()),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
