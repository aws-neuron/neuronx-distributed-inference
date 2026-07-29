#!/usr/bin/env python3
"""Validate Qwen head_dim=256 segmented CTE attention against CPU math.

The CPU reference stays off XLA so the generated NEFF is the NKI kernel under
test, matching the NKI debugging workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    prior_len: int
    active_real_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate qwen_segcte256 segmented CTE attention."
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--q-len", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--prior-seg-size", type=int, default=512)
    parser.add_argument(
        "--cases",
        default="0:512,512:512,1024:512,1024:201",
        help="Comma-separated prior:real-active cases. q_len stays padded.",
    )
    parser.add_argument("--target", default="trn2")
    parser.add_argument("--lnc", type=int, default=2)
    parser.add_argument("--visible-cores", default="0")
    parser.add_argument("--scale", type=float, default=0.12)
    parser.add_argument(
        "--reference-score-scale",
        type=float,
        default=1.0,
        help="Extra multiplier applied only to CPU-reference attention scores.",
    )
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument(
        "--value-pattern",
        choices=("random", "ones", "token"),
        default="random",
        help="V pattern for diagnostics. ones should return ones for any valid softmax.",
    )
    parser.add_argument(
        "--identity-block-table",
        action="store_true",
        help="Use logical block i -> physical block i instead of random physical IDs.",
    )
    parser.add_argument(
        "--pad-block-table-to-q-len",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Size the block table for prior_len + padded q_len, matching the "
            "segmented CTE serving wrapper. The CPU comparison still uses only "
            "active_real_len tokens."
        ),
    )
    parser.add_argument(
        "--normalize-qk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate model-faithful Q/K: l2-normalized K and l2-normalized "
            "Q divided by sqrt(head_dim), matching Qwen qk-norm attention."
        ),
    )
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--rtol", type=float, default=5.0e-2)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", args.target)
    os.environ.setdefault("NEURON_CC_FLAGS", f"--target {args.target} --lnc {args.lnc}")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", args.visible_cores)


def parse_cases(raw: str, q_len: int) -> list[Case]:
    cases: list[Case] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        prior_raw, active_raw = item.split(":", 1)
        case = Case(prior_len=int(prior_raw), active_real_len=int(active_raw))
        if case.prior_len < 0:
            raise ValueError(f"prior_len must be non-negative: {case}")
        if case.active_real_len <= 0 or case.active_real_len > q_len:
            raise ValueError(f"active_real_len must be in 1..q_len: {case}")
        cases.append(case)
    if not cases:
        raise ValueError("--cases produced no cases")
    return cases


def make_case_tensors(torch: Any, args: argparse.Namespace, case: Case) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + case.prior_len * 1009 + case.active_real_len)

    batch = args.batch_size
    q_heads = args.q_heads
    kv_heads = args.kv_heads
    q_len = args.q_len
    block = args.block_size
    head_dim = args.head_dim
    total_padded_len = case.prior_len + q_len
    real_total_len = case.prior_len + case.active_real_len
    needed_blocks = (real_total_len + block - 1) // block
    padded_blocks = (total_padded_len + block - 1) // block
    block_table_blocks = padded_blocks if args.pad_block_table_to_q_len else needed_blocks
    physical_blocks = max(block_table_blocks + 7, padded_blocks + 3, 8)

    q_raw = torch.randn(
        (batch * q_heads, q_len, head_dim),
        generator=generator,
        dtype=torch.float32,
    )
    k_raw = torch.randn(
        (batch, kv_heads, total_padded_len, head_dim),
        generator=generator,
        dtype=torch.float32,
    )
    if args.normalize_qk:
        q = torch.nn.functional.normalize(q_raw, p=2, dim=-1, eps=1.0e-6)
        q = q * (args.scale / math.sqrt(head_dim))
        logical_k = torch.nn.functional.normalize(
            k_raw,
            p=2,
            dim=-1,
            eps=1.0e-6,
        )
        logical_k = logical_k * args.scale
    else:
        q = q_raw * (args.scale / math.sqrt(head_dim))
        logical_k = k_raw * args.scale
    if args.value_pattern == "ones":
        logical_v = torch.ones(
            (batch, kv_heads, total_padded_len, head_dim),
            dtype=torch.float32,
        ) * args.value_scale
    elif args.value_pattern == "token":
        token_values = torch.arange(total_padded_len, dtype=torch.float32)
        token_values = token_values.view(1, 1, total_padded_len, 1)
        logical_v = token_values.expand(batch, kv_heads, -1, head_dim).contiguous()
        logical_v = logical_v / max(1.0, float(total_padded_len)) * args.value_scale
    else:
        logical_v = torch.randn(
            (batch, kv_heads, total_padded_len, head_dim),
            generator=generator,
            dtype=torch.float32,
        ) * args.value_scale

    block_table = torch.empty((batch, block_table_blocks), dtype=torch.int32)
    k_cache = torch.zeros(
        (physical_blocks, kv_heads, block, head_dim),
        dtype=torch.float32,
    )
    v_cache = torch.zeros_like(k_cache)

    for b in range(batch):
        if args.identity_block_table:
            perm = torch.arange(block_table_blocks, dtype=torch.int64)
        else:
            # Exercise indirect block-table reads instead of the identity layout.
            perm = torch.randperm(physical_blocks, generator=generator)[:block_table_blocks]
        block_table[b] = perm.to(torch.int32)
        for logical_block, physical_block_t in enumerate(perm.tolist()):
            start = logical_block * block
            end = min(start + block, total_padded_len)
            width = end - start
            if width <= 0:
                continue
            k_cache[physical_block_t, :, :width, :] = logical_k[b, :, start:end, :]
            v_cache[physical_block_t, :, :width, :] = logical_v[b, :, start:end, :]

    return {
        "q": q.contiguous(),
        "logical_k": logical_k.contiguous(),
        "logical_v": logical_v.contiguous(),
        "k_cache": k_cache.contiguous(),
        "v_cache": v_cache.contiguous(),
        "block_table": block_table.contiguous(),
        "prior_tokens": torch.full((batch, 1), case.prior_len, dtype=torch.int32),
    }


def cpu_reference(torch: Any, tensors: dict[str, Any], args: argparse.Namespace, case: Case) -> Any:
    batch = args.batch_size
    q_heads = args.q_heads
    kv_heads = args.kv_heads
    q_len = args.q_len
    active_real = case.active_real_len
    compare_len = active_real
    output = torch.empty(
        (batch * q_heads, compare_len, args.head_dim),
        dtype=torch.float32,
    )

    key_positions = torch.arange(
        case.prior_len + active_real,
        dtype=torch.int64,
    ).view(1, -1)
    query_positions = (
        case.prior_len + torch.arange(active_real, dtype=torch.int64)
    ).view(-1, 1)
    causal = key_positions <= query_positions

    for b in range(batch):
        for qh in range(q_heads):
            flat_head = b * q_heads + qh
            kvh = qh * kv_heads // q_heads
            q = tensors["q"][flat_head, :active_real, :].to(torch.bfloat16).float()
            k = tensors["logical_k"][
                b,
                kvh,
                : case.prior_len + active_real,
                :,
            ].to(torch.bfloat16).float()
            v = tensors["logical_v"][
                b,
                kvh,
                : case.prior_len + active_real,
                :,
            ].to(torch.bfloat16).float()
            scores = q @ k.T
            scores = scores * args.reference_score_scale
            scores = scores.masked_fill(~causal, -float("inf"))
            probs = torch.softmax(scores, dim=-1)
            output[flat_head] = probs @ v
    return output


def metrics(torch: Any, actual: Any, expected: Any, args: argparse.Namespace) -> dict[str, Any]:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    diff = actual_f - expected_f
    diff_norm = torch.linalg.vector_norm(diff)
    expected_norm = torch.linalg.vector_norm(expected_f)
    rel_norm = diff_norm / expected_norm.clamp_min(1.0e-12)
    actual_flat = actual_f.reshape(-1)
    expected_flat = expected_f.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        actual_flat,
        expected_flat,
        dim=0,
        eps=1.0e-12,
    )
    allclose = torch.allclose(actual_f, expected_f, atol=args.atol, rtol=args.rtol)
    return {
        "allclose": bool(allclose),
        "max_abs": float(diff.abs().max().item()),
        "rel_norm": float(rel_norm.item()),
        "cosine": float(cosine.item()),
        "actual_min": float(actual_f.min().item()),
        "actual_max": float(actual_f.max().item()),
        "actual_mean": float(actual_f.mean().item()),
        "expected_min": float(expected_f.min().item()),
        "expected_max": float(expected_f.max().item()),
        "expected_mean": float(expected_f.mean().item()),
    }


def main() -> int:
    args = parse_args()
    configure_environment(args)

    import nki
    import torch
    from torch_xla.core import xla_model as xm

    from neuronx_distributed_inference.modules.attention.nki_kernels.qwen_segcte256.attention_segmented_cte_256 import (
        attention_segmented_cte,
    )

    if args.head_dim != 256:
        raise ValueError("qwen_segcte256 validation must use --head-dim 256")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads")
    if args.q_len % args.block_size != 0 or args.q_len % 128 != 0:
        raise ValueError("--q-len must be divisible by block size and 128")
    if args.prior_seg_size % args.block_size != 0:
        raise ValueError("--prior-seg-size must be divisible by --block-size")

    cases = parse_cases(args.cases, args.q_len)
    device = xm.xla_device()
    kernel = nki.jit(attention_segmented_cte)
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for case in cases:
        tensors = make_case_tensors(torch, args, case)
        expected = cpu_reference(torch, tensors, args, case)
        q = tensors["q"].to(torch.bfloat16).to(device=device)
        k_cache = tensors["k_cache"].to(torch.bfloat16).to(device=device)
        v_cache = tensors["v_cache"].to(torch.bfloat16).to(device=device)
        block_table = tensors["block_table"].to(torch.int32).to(device=device)
        prior_tokens = tensors["prior_tokens"].to(torch.int32).to(device=device)

        launch = kernel[args.lnc] if args.lnc > 1 else kernel
        actual_full = launch(
            q,
            k_cache,
            v_cache,
            block_table,
            prior_tokens,
            args.block_size,
            args.prior_seg_size,
            1.0,
            tp_q=True,
            tp_out=False,
            sliding_window=None,
            sink=None,
            num_q_heads=args.q_heads,
            k_pre_transposed=False,
        ).cpu()
        actual = actual_full[:, : case.active_real_len, :]
        row = {
            "case": {
                "prior_len": case.prior_len,
                "active_real_len": case.active_real_len,
                "q_len": args.q_len,
            },
            "block_table_shape": list(tensors["block_table"].shape),
            "k_cache_shape": list(tensors["k_cache"].shape),
            "metrics": metrics(torch, actual, expected, args),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if not row["metrics"]["allclose"]:
            failures.append(row)

    summary = {
        "ok": not failures,
        "num_cases": len(rows),
        "num_failures": len(failures),
        "args": vars(args),
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if failures and args.fail_on_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
