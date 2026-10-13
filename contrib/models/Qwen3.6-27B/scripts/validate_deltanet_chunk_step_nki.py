#!/usr/bin/env python3
"""Validate and optionally profile the Qwen DeltaNet per-chunk NKI kernel.

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
        description="Validate/profile deltanet_chunk_step against a CPU reference."
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--target", default="trn2")
    parser.add_argument("--lnc", type=int, default=1)
    parser.add_argument("--visible-cores", default="0")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--dge", action="store_true")
    parser.add_argument(
        "--inspect-dir",
        default="/mnt/trainium_artifacts/profiles/deltanet_chunk_step_isolated",
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


def load_chunked_kernel():
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "nki_kernels"
        / "nki_deltanet_chunked.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen36_nki_deltanet_chunked_under_test",
        kernel_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.deltanet_chunk_step


def make_inputs(torch: Any, args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    def randn(scale: float) -> Any:
        return torch.randn((P_MAX, P_MAX), generator=generator, dtype=torch.float32) * scale

    query = randn(args.value_scale)
    key = randn(args.value_scale)
    value = randn(args.value_scale)
    state_in = randn(args.state_scale)

    query = torch.nn.functional.normalize(query, p=2, dim=-1) / math.sqrt(P_MAX)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)

    beta_col = torch.sigmoid(
        torch.randn((P_MAX, 1), generator=generator, dtype=torch.float32)
    )
    beta_broadcast = beta_col.expand(P_MAX, P_MAX).contiguous()

    # Qwen GDN decay gates are negative log-decays. Keep them small enough that
    # reference comparisons isolate algorithmic errors rather than overflow.
    g_raw = -torch.nn.functional.softplus(
        torch.randn((P_MAX, 1), generator=generator, dtype=torch.float32)
    )
    g_raw = g_raw * args.gate_scale
    g_cumsum_col = torch.cumsum(g_raw, dim=0)
    g_cumsum = g_cumsum_col.expand(P_MAX, P_MAX).contiguous()
    g_last = g_cumsum_col[-1:].expand(P_MAX, P_MAX).contiguous()

    lower_mask = torch.tril(torch.ones((P_MAX, P_MAX), dtype=torch.float32), diagonal=-1)
    lower_mask_diag = torch.tril(torch.ones((P_MAX, P_MAX), dtype=torch.float32))
    identity = torch.eye(P_MAX, dtype=torch.float32)

    return {
        "query": query.contiguous(),
        "key": key.contiguous(),
        "value": value.contiguous(),
        "beta_broadcast": beta_broadcast,
        "g_cumsum": g_cumsum,
        "g_last": g_last,
        "state_in": state_in.contiguous(),
        "lower_mask": lower_mask.contiguous(),
        "identity": identity.contiguous(),
        "lower_mask_diag": lower_mask_diag.contiguous(),
    }


def reference_math(torch: Any, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
    q = inputs["query"]
    k = inputs["key"]
    v = inputs["value"]
    beta = inputs["beta_broadcast"]
    gc = inputs["g_cumsum"][:, 0:1]
    gl = inputs["g_last"][:, 0:1]
    state = inputs["state_in"]
    lower = inputs["lower_mask"]
    lower_diag = inputs["lower_mask_diag"]
    eye = inputs["identity"]

    k_beta = k * beta
    v_beta = v * beta

    decay = torch.exp(gc - gc.T)
    decay_strict = decay * lower
    decay_diag = decay * lower_diag

    qk_beta = k_beta @ k.T
    a_mat = -(qk_beta * decay_strict)
    a_mat = a_mat * lower

    # Intended kernel math: A is strictly lower triangular, N = inv(I - A).
    lhs = eye - a_mat
    n_mat = torch.linalg.solve_triangular(lhs, eye, upper=False)

    exp_gc = torch.exp(gc)
    value_corr = n_mat @ v_beta
    k_cumdecay = n_mat @ (k_beta * exp_gc)

    qk_raw = q @ k.T
    attn_intra = qk_raw * decay_diag

    v_prime = k_cumdecay @ state
    v_new = value_corr - v_prime

    attn_inter = (q * exp_gc) @ state
    intra_out = attn_intra @ v_new
    output = attn_inter + intra_out

    exp_gl_minus_gc = torch.exp(gl - gc)
    k_raw_decay = k * exp_gl_minus_gc
    kv_outer = k_raw_decay.T @ v_new
    state_out = state * torch.exp(gl) + kv_outer

    return output.contiguous(), state_out.contiguous(), n_mat.contiguous()


def reference_kernel_mirror(torch: Any, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
    q = inputs["query"]
    k = inputs["key"]
    v = inputs["value"]
    beta = inputs["beta_broadcast"]
    gc = inputs["g_cumsum"][:, 0:1]
    gl = inputs["g_last"][:, 0:1]
    state = inputs["state_in"]
    lower = inputs["lower_mask"]
    lower_diag = inputs["lower_mask_diag"]
    eye = inputs["identity"]

    k_beta = k * beta
    v_beta = v * beta

    decay = torch.exp(gc - gc.T)
    qk = k_beta @ k.T
    a_mat = -(qk * decay * lower) * lower

    p_acc = eye + a_mat
    a_pow = a_mat.clone()
    for _ in range(6):
        a_pow = (a_pow @ a_pow) * lower
        p_acc = ((eye + a_pow) @ p_acc) * lower_diag

    exp_gc = torch.exp(gc)
    value_corr = p_acc @ v_beta
    k_cumdecay = p_acc @ (k_beta * exp_gc)
    attn_intra = (q @ k.T) * (decay * lower_diag)
    v_new = value_corr - (k_cumdecay @ state)
    output = ((q * exp_gc) @ state) + (attn_intra @ v_new)

    k_raw_decay = k * torch.exp(gl - gc)
    state_out = state * torch.exp(gl) + (k_raw_decay.T @ v_new)

    return output.contiguous(), state_out.contiguous(), p_acc.contiguous()


def tensor_metrics(torch: Any, actual: Any, expected: Any) -> dict[str, float | bool]:
    diff = actual - expected
    expected_norm = torch.linalg.vector_norm(expected).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    actual_flat = actual.reshape(-1).to(torch.float64)
    expected_flat = expected.reshape(-1).to(torch.float64)
    denom = torch.linalg.vector_norm(actual_flat) * torch.linalg.vector_norm(expected_flat)
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

    deltanet_chunk_step = load_chunked_kernel()

    inputs = make_inputs(torch, args)
    math_out, math_state, math_n = reference_math(torch, inputs)
    mirror_out, mirror_state, mirror_n = reference_kernel_mirror(torch, inputs)

    mirror_vs_math = {
        "output": tensor_metrics(torch, mirror_out, math_out),
        "state": tensor_metrics(torch, mirror_state, math_state),
        "n_matrix": tensor_metrics(torch, mirror_n, math_n),
    }

    device = xm.xla_device()
    xla_inputs = {name: tensor.to(device=device) for name, tensor in inputs.items()}

    out_cpu = state_cpu = None
    for _ in range(args.runs):
        out_dev, state_dev = deltanet_chunk_step(
            xla_inputs["query"],
            xla_inputs["key"],
            xla_inputs["value"],
            xla_inputs["beta_broadcast"],
            xla_inputs["g_cumsum"],
            xla_inputs["g_last"],
            xla_inputs["state_in"],
            xla_inputs["lower_mask"],
            xla_inputs["identity"],
            xla_inputs["lower_mask_diag"],
        )
        xm.mark_step()
        out_cpu = out_dev.detach().cpu().float()
        state_cpu = state_dev.detach().cpu().float()

    assert out_cpu is not None
    assert state_cpu is not None

    nki_vs_math = {
        "output": tensor_metrics(torch, out_cpu, math_out),
        "state": tensor_metrics(torch, state_cpu, math_state),
    }
    nki_vs_mirror = {
        "output": tensor_metrics(torch, out_cpu, mirror_out),
        "state": tensor_metrics(torch, state_cpu, mirror_state),
    }

    output_close = torch.allclose(out_cpu, math_out, atol=args.atol, rtol=args.rtol)
    state_close = torch.allclose(state_cpu, math_state, atol=args.atol, rtol=args.rtol)
    output_finite = bool(torch.isfinite(out_cpu).all().item())
    state_finite = bool(torch.isfinite(state_cpu).all().item())
    passed = bool(output_close and state_close and output_finite and state_finite)

    result = {
        "passed": passed,
        "seed": args.seed,
        "runs": args.runs,
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
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
            )
        },
        "mirror_vs_math": mirror_vs_math,
        "nki_vs_math": nki_vs_math,
        "nki_vs_mirror": nki_vs_mirror,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
