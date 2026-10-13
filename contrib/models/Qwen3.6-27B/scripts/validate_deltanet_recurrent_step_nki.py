#!/usr/bin/env python3
"""Validate/profile the Qwen DeltaNet one-token decode NKI kernel.

The reference path is CPU-only by design. Keeping reference math off the XLA
device avoids compiling extra NEFFs that obscure the NKI kernel profile.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


P_MAX = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate/profile deltanet_recurrent_step against CPU math."
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--batch-heads", type=int, default=4)
    parser.add_argument("--target", default="trn2")
    parser.add_argument("--lnc", type=int, default=1)
    parser.add_argument("--visible-cores", default="0")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--dge", action="store_true")
    parser.add_argument(
        "--inspect-dir",
        default="/mnt/trainium_artifacts/profiles/deltanet_recurrent_step_isolated",
    )
    parser.add_argument("--atol", type=float, default=2.0e-2)
    parser.add_argument("--rtol", type=float, default=2.0e-2)
    parser.add_argument("--value-scale", type=float, default=0.05)
    parser.add_argument("--state-scale", type=float, default=0.01)
    parser.add_argument("--gate-scale", type=float, default=0.01)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> Path:
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", args.target)
    os.environ.setdefault("NEURON_CC_FLAGS", f"--target {args.target} --lnc {args.lnc}")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", args.visible_cores)

    inspect_dir = Path(args.inspect_dir).expanduser().resolve()
    if args.inspect:
        inspect_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
        os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
        os.environ["NEURON_RT_INSPECT_SYSTEM_PROFILE"] = "0"
        os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = str(inspect_dir)
        os.environ["XLA_IR_DEBUG"] = "1"
        os.environ["XLA_HLO_DEBUG"] = "1"
        os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    if args.dge:
        os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"
    return inspect_dir


def add_qwen_to_path() -> None:
    script_path = Path(__file__).resolve()
    qwen_root = script_path.parents[1]
    sys.path.insert(0, str(qwen_root))


def load_step_kernel():
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "nki_kernels"
        / "nki_deltanet.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen36_nki_deltanet_step_under_test",
        kernel_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.deltanet_recurrent_step_batched


def make_inputs(torch: Any, args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    if args.batch_heads <= 0:
        raise ValueError("--batch-heads must be positive")

    def randn(shape: tuple[int, ...], scale: float) -> Any:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    query = randn((args.batch_heads, P_MAX), args.value_scale)
    key = randn((args.batch_heads, P_MAX), args.value_scale)
    value = randn((args.batch_heads, P_MAX), args.value_scale)
    state_in = randn((args.batch_heads * P_MAX, P_MAX), args.state_scale)

    query = torch.nn.functional.normalize(query, p=2, dim=-1) / math.sqrt(P_MAX)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)

    beta = torch.sigmoid(randn((args.batch_heads, 1), 1.0)).contiguous()

    g = (
        -torch.nn.functional.softplus(randn((args.batch_heads, 1), 1.0))
        * args.gate_scale
    ).contiguous()

    return {
        "query": query.contiguous(),
        "key": key.contiguous(),
        "value": value.contiguous(),
        "g": g,
        "beta": beta,
        "state_in": state_in.contiguous(),
    }


def reference_math(torch: Any, inputs: dict[str, Any]) -> tuple[Any, Any]:
    outputs = []
    states = []
    batch_heads = inputs["query"].shape[0]

    for bh in range(batch_heads):
        q = inputs["query"][bh]
        k = inputs["key"][bh]
        v = inputs["value"][bh]
        g = inputs["g"][bh].reshape(1, 1)
        beta = inputs["beta"][bh]
        state = inputs["state_in"][bh * P_MAX : (bh + 1) * P_MAX]

        state_decayed = state * torch.exp(g)
        kv_mem = (state_decayed * k.unsqueeze(-1)).sum(dim=0)
        delta = (v - kv_mem) * beta
        state_out = state_decayed + k.unsqueeze(-1) * delta.unsqueeze(0)
        output = (state_out * q.unsqueeze(-1)).sum(dim=0)
        outputs.append(output)
        states.append(state_out)

    return torch.stack(outputs, dim=0).contiguous(), torch.cat(states, dim=0).contiguous()


def tensor_metrics(torch: Any, actual: Any, expected: Any) -> dict[str, float | bool]:
    diff = actual - expected
    expected_norm = torch.linalg.vector_norm(expected).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    actual_flat = actual.reshape(-1).to(torch.float64)
    expected_flat = expected.reshape(-1).to(torch.float64)
    denom = torch.linalg.vector_norm(actual_flat) * torch.linalg.vector_norm(
        expected_flat
    )
    cosine = (
        float(torch.dot(actual_flat, expected_flat) / denom)
        if denom.item() != 0.0
        else float("nan")
    )
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(torch.max(torch.abs(diff)).item()),
        "mean_abs": float(torch.mean(torch.abs(diff)).item()),
        "diff_norm": float(diff_norm),
        "expected_norm": float(expected_norm),
        "relative_norm": float(diff_norm / max(expected_norm, 1.0e-12)),
        "cosine": cosine,
    }


def main() -> int:
    args = parse_args()
    inspect_dir = configure_environment(args)
    add_qwen_to_path()

    import torch
    import torch_xla.core.xla_model as xm

    deltanet_recurrent_step_batched = load_step_kernel()

    inputs = make_inputs(torch, args)
    ref_out, ref_state = reference_math(torch, inputs)

    device = xm.xla_device()
    xla_inputs = {name: tensor.to(device=device) for name, tensor in inputs.items()}

    out_cpu = state_cpu = None
    for _ in range(args.runs):
        out_dev, state_dev = deltanet_recurrent_step_batched(
            xla_inputs["query"],
            xla_inputs["key"],
            xla_inputs["value"],
            xla_inputs["g"],
            xla_inputs["beta"],
            xla_inputs["state_in"],
        )
        xm.mark_step()
        out_cpu = out_dev.detach().cpu().float()
        state_cpu = state_dev.detach().cpu().float()

    assert out_cpu is not None
    assert state_cpu is not None

    output_close = torch.allclose(out_cpu, ref_out, atol=args.atol, rtol=args.rtol)
    state_close = torch.allclose(state_cpu, ref_state, atol=args.atol, rtol=args.rtol)
    output_finite = bool(torch.isfinite(out_cpu).all().item())
    state_finite = bool(torch.isfinite(state_cpu).all().item())
    passed = bool(output_close and state_close and output_finite and state_finite)

    result = {
        "passed": passed,
        "seed": args.seed,
        "runs": args.runs,
        "batch_heads": args.batch_heads,
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "output_finite": output_finite,
        "state_finite": state_finite,
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_OUTPUT_DIR",
            )
        },
        "metrics": {
            "output": tensor_metrics(torch, out_cpu, ref_out),
            "state": tensor_metrics(torch, state_cpu, ref_state),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
