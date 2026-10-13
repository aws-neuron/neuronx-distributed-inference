#!/usr/bin/env python3
"""Validate and optionally inspect/profile the fused Qwen DeltaNet NKI kernel.

The CPU reference stays off the XLA device so the generated NEFFs are from the
NKI kernel under test, not from reference PyTorch ops.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


P_MAX = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate/profile deltanet_fused_chunked_fwd against CPU math."
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("QWEN36_DELTANET_CHUNK_SIZE", "128")),
        choices=(64, 128),
        help=(
            "Active fused GDN CTE token chunk size. 128 is the existing path; "
            "64 is the FlashQLA-inspired probe path."
        ),
    )
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument(
        "--multihead",
        action="store_true",
        help="Validate deltanet_fused_chunked_fwd_multihead with one grid program per head.",
    )
    parser.add_argument(
        "--validate-cpu-chunk-invariance",
        action="store_true",
        help=(
            "Run a CPU-only check that exact 64-token and 128-token chunked "
            "DeltaNet references produce equivalent outputs/states."
        ),
    )
    parser.add_argument(
        "--validate-restored-suffix-carry",
        action="store_true",
        help=(
            "Validate the serving-style GDN carry boundary: run one full padded "
            "sequence and compare it with restored calls over split CTE buckets."
        ),
    )
    parser.add_argument(
        "--restore-split-lens",
        default="512,512,201",
        help=(
            "Comma-separated real token counts for "
            "--validate-restored-suffix-carry. Each segment is padded to "
            "--restore-bucket-size before the next restored call."
        ),
    )
    parser.add_argument(
        "--restore-bucket-size",
        type=int,
        default=512,
        help="Per-call padded CTE bucket size for --validate-restored-suffix-carry.",
    )
    parser.add_argument(
        "--validate-autocp-affine",
        action="store_true",
        help=(
            "Validate the isolated one-chunk AutoCP affine-piece NKI probe "
            "instead of the fused forward kernel."
        ),
    )
    parser.add_argument(
        "--validate-autocp-prefix",
        action="store_true",
        help=(
            "Validate the isolated AutoCP state-prefix NKI probe over all "
            "128-token chunks in --seq-len."
        ),
    )
    parser.add_argument(
        "--validate-autocp-chain",
        action="store_true",
        help=(
            "Validate the isolated AutoCP prefix plus output-apply NKI chain "
            "against the CPU AutoCP reference."
        ),
    )
    parser.add_argument(
        "--validate-autocp-prefix-apply",
        action="store_true",
        help=(
            "Validate the fused AutoCP state-prefix/output-apply NKI pass "
            "against CPU affine stacks."
        ),
    )
    parser.add_argument(
        "--validate-autocp-full",
        action="store_true",
        help=(
            "Validate NKI chunk-parallel affine generation plus prefix/apply "
            "against the CPU AutoCP reference."
        ),
    )
    parser.add_argument(
        "--validate-autocp-state-summary",
        action="store_true",
        help=(
            "Validate compact AutoCP NKI state-summary generation against "
            "CPU segment state transforms."
        ),
    )
    parser.add_argument(
        "--validate-autocp-compact-chain",
        action="store_true",
        help=(
            "Validate compact AutoCP summary + state-prefix + recurrent segment "
            "replay against the CPU sequential reference."
        ),
    )
    parser.add_argument(
        "--validate-compact-autocp-reference",
        action="store_true",
        help=(
            "Run a CPU-only check for compact AutoCP state summaries: compose "
            "chunk transforms into per-segment state transforms, prefix segment "
            "states, then replay each segment recurrently."
        ),
    )
    parser.add_argument(
        "--autocp-cp-chunks",
        type=int,
        default=4,
        help="Number of 128-token chunks per compact AutoCP segment.",
    )
    parser.add_argument(
        "--head-group-size",
        type=int,
        default=1,
        help=(
            "Number of flattened (batch, head) rows per multihead NKI launch. "
            "Use larger values to test launch-grid batching."
        ),
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--target", default="trn2")
    parser.add_argument("--lnc", type=int, default=1)
    parser.add_argument("--visible-cores", default="0")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--dge", action="store_true")
    parser.add_argument(
        "--inspect-dir",
        default="/mnt/trainium_artifacts/profiles/deltanet_fused_isolated",
    )
    parser.add_argument("--atol", type=float, default=3.0e-2)
    parser.add_argument("--rtol", type=float, default=3.0e-2)
    parser.add_argument("--value-scale", type=float, default=0.05)
    parser.add_argument("--state-scale", type=float, default=0.01)
    parser.add_argument("--gate-scale", type=float, default=0.01)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> Path:
    chunk_size = int(getattr(args, "chunk_size", P_MAX))
    args.chunk_size = chunk_size
    if args.seq_len <= 0 or args.seq_len % chunk_size != 0:
        raise ValueError(
            "--seq-len must be a positive multiple of --chunk-size; "
            f"got seq_len={args.seq_len}, chunk_size={chunk_size}"
        )
    autocp_modes = (
        getattr(args, "validate_autocp_affine", False)
        or getattr(args, "validate_autocp_prefix", False)
        or getattr(args, "validate_autocp_chain", False)
        or getattr(args, "validate_autocp_prefix_apply", False)
        or getattr(args, "validate_autocp_full", False)
        or getattr(args, "validate_autocp_state_summary", False)
        or getattr(args, "validate_autocp_compact_chain", False)
        or getattr(args, "validate_compact_autocp_reference", False)
    )
    if autocp_modes and chunk_size != P_MAX:
        raise ValueError(
            "AutoCP validators are still 128-token chunk probes; "
            f"got --chunk-size={chunk_size}"
        )
    if args.multihead and args.heads <= 0:
        raise ValueError("--heads must be positive when --multihead is set")
    if args.head_group_size <= 0:
        raise ValueError("--head-group-size must be positive")
    if getattr(args, "validate_autocp_full", False) and (args.seq_len // P_MAX) % 2 != 0:
        raise ValueError(
            "--validate-autocp-full uses LNC2-striped affine generation and "
            "requires an even 128-token chunk count; "
            f"got seq_len={args.seq_len}, chunks={args.seq_len // P_MAX}"
        )
    cp_chunks = int(getattr(args, "autocp_cp_chunks", 4))
    args.autocp_cp_chunks = cp_chunks
    if cp_chunks <= 0:
        raise ValueError("--autocp-cp-chunks must be positive")
    compact_nki_modes = (
        getattr(args, "validate_autocp_state_summary", False)
        or getattr(args, "validate_autocp_compact_chain", False)
    )
    if compact_nki_modes:
        num_chunks = args.seq_len // P_MAX
        if num_chunks % cp_chunks != 0:
            raise ValueError(
                "Compact AutoCP NKI validators require the 128-token chunk "
                "count to be divisible by --autocp-cp-chunks; "
                f"chunks={num_chunks}, cp_chunks={cp_chunks}"
            )
        num_segments = num_chunks // cp_chunks
        if num_segments % args.lnc != 0:
            raise ValueError(
                "--validate-autocp-state-summary requires the segment count "
                "to be divisible by --lnc; "
                f"segments={num_segments}, lnc={args.lnc}"
            )
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", args.target)
    os.environ.setdefault("NEURON_CC_FLAGS", f"--target {args.target} --lnc {args.lnc}")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", args.visible_cores)
    os.environ["QWEN36_DELTANET_CHUNK_SIZE"] = str(chunk_size)
    os.environ["QWEN36_DELTANET_AUTOCP_CP_CHUNKS"] = str(cp_chunks)

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


def load_kernel_module():
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "nki_kernels"
        / "nki_deltanet_fused.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen36_nki_deltanet_fused_under_test",
        kernel_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_fused_kernel(multihead: bool):
    module = load_kernel_module()
    if multihead:
        return module.deltanet_fused_chunked_fwd_multihead
    return module.deltanet_fused_chunked_fwd


def load_autocp_affine_kernel():
    return load_kernel_module().deltanet_autocp_affine_chunk


def load_autocp_affine_sequence_kernel():
    return load_kernel_module().deltanet_autocp_affine_sequence


def load_autocp_state_summary_kernel():
    return load_kernel_module().deltanet_autocp_state_summary_sequence


def load_autocp_prefix_kernel():
    return load_kernel_module().deltanet_autocp_state_prefix


def load_autocp_apply_kernel():
    return load_kernel_module().deltanet_autocp_apply_output


def load_autocp_prefix_apply_kernel():
    return load_kernel_module().deltanet_autocp_prefix_apply_output


def multihead_launch_spec(num_heads: int, lnc: int):
    if num_heads <= lnc:
        return num_heads
    if os.environ.get("QWEN36_DELTANET_MULTIHEAD_SPMD", "1") == "0":
        raise ValueError(
            "--head-group-size exceeds --lnc but "
            "QWEN36_DELTANET_MULTIHEAD_SPMD=0; "
            f"group_size={num_heads}, lnc={lnc}"
        )

    import nki.language as nl

    if not hasattr(nl, "spmd_dim") or not hasattr(nl, "nc"):
        if os.environ.get("QWEN36_DELTANET_MULTIHEAD_GRID_FALLBACK", "0") == "1":
            return (num_heads, 1)
        raise ValueError(
            "--head-group-size exceeds --lnc, but this NKI runtime does not "
            f"expose spmd_dim/nc; group_size={num_heads}, lnc={lnc}"
        )
    return (nl.spmd_dim(num_heads, nl.nc(lnc)),)


def launch_spec_label(spec: Any) -> str:
    if isinstance(spec, int):
        return str(spec)
    return repr(spec)


def make_inputs(torch: Any, args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    def randn(shape: tuple[int, ...], scale: float) -> Any:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    multihead = bool(getattr(args, "multihead", False))
    heads = int(getattr(args, "heads", 1))
    chunk_size = int(getattr(args, "chunk_size", P_MAX))
    prefix_shape = (heads,) if multihead else ()
    query = randn((*prefix_shape, args.seq_len, P_MAX), args.value_scale)
    key = randn((*prefix_shape, args.seq_len, P_MAX), args.value_scale)
    value = randn((*prefix_shape, args.seq_len, P_MAX), args.value_scale)
    state_in = randn((*prefix_shape, P_MAX, P_MAX), args.state_scale)

    beta = torch.sigmoid(randn((*prefix_shape, args.seq_len, 1), 1.0))
    g_raw = -torch.nn.functional.softplus(randn((*prefix_shape, args.seq_len, 1), 1.0))
    g_raw = g_raw * args.gate_scale

    lower_mask = torch.zeros((P_MAX, P_MAX), dtype=torch.float32)
    lower_mask_diag = torch.zeros((P_MAX, P_MAX), dtype=torch.float32)
    identity = torch.zeros((P_MAX, P_MAX), dtype=torch.float32)
    lower_mask[:chunk_size, :chunk_size] = torch.tril(
        torch.ones((chunk_size, chunk_size), dtype=torch.float32),
        diagonal=-1,
    )
    lower_mask_diag[:chunk_size, :chunk_size] = torch.tril(
        torch.ones((chunk_size, chunk_size), dtype=torch.float32)
    )
    identity[:chunk_size, :chunk_size] = torch.eye(
        chunk_size,
        dtype=torch.float32,
    )

    return {
        "chunk_size": chunk_size,
        "query": query.contiguous(),
        "key": key.contiguous(),
        "value": value.contiguous(),
        "g_raw": g_raw.contiguous(),
        "beta": beta.contiguous(),
        "state_in": state_in.contiguous(),
        "lower_mask": lower_mask.contiguous(),
        "identity": identity.contiguous(),
        "lower_mask_diag": lower_mask_diag.contiguous(),
    }


def stable_causal_decay(torch: Any, gc: Any, mask: Any) -> Any:
    """Compute exp(gc[i] - gc[j]) only where the causal mask is active."""
    diff = gc - gc.T
    masked_diff = torch.where(mask.bool(), diff, torch.zeros_like(diff))
    return torch.exp(masked_diff) * mask


def move_tensor_inputs_to_device(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        name: tensor.to(device=device)
        for name, tensor in inputs.items()
        if hasattr(tensor, "to")
    }


def normalize_reference_qk(torch: Any, query: Any, key: Any) -> tuple[Any, Any]:
    """Match the fused kernel's in-kernel Q/K l2norm and Q scale."""
    query_norm = torch.nn.functional.normalize(
        query,
        p=2,
        dim=-1,
        eps=1.0e-6,
    ) / math.sqrt(P_MAX)
    key_norm = torch.nn.functional.normalize(
        key,
        p=2,
        dim=-1,
        eps=1.0e-6,
    )
    return query_norm, key_norm


def blocked_lower_triangular_solve(
    torch: Any,
    lhs: Any,
    rhs: Any,
    block_size: int,
) -> Any:
    """Solve lower-triangular lhs @ x = rhs by block forward substitution."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rows, cols = lhs.shape
    if rows != cols:
        raise ValueError(f"lhs must be square, got {lhs.shape}")
    if rhs.shape[0] != rows:
        raise ValueError(
            f"rhs row count must match lhs, got lhs={lhs.shape} rhs={rhs.shape}"
        )

    solved_blocks = []
    for row_start in range(0, rows, block_size):
        row_end = min(row_start + block_size, rows)
        rhs_block = rhs[row_start:row_end].clone()
        for col_start, solved in zip(
            range(0, row_start, block_size),
            solved_blocks,
        ):
            col_end = min(col_start + block_size, rows)
            lhs_block = lhs[row_start:row_end, col_start:col_end]
            rhs_block = rhs_block - lhs_block @ solved

        diag_block = lhs[row_start:row_end, row_start:row_end]
        solved_block = torch.linalg.solve_triangular(
            diag_block,
            rhs_block,
            upper=False,
        )
        solved_blocks.append(solved_block)

    return torch.cat(solved_blocks, dim=0)


def block_prefix_lower_triangular_solve(
    torch: Any,
    lhs: Any,
    rhs: Any,
    block_size: int,
) -> Any:
    """Solve lhs @ x = rhs with FLA-style block affine segment combines.

    Each diagonal block first produces an affine map from all preceding rows
    to the block output:

        x_i = solve(L_ii, rhs_i) - solve(L_ii, L_i,<i) @ x_<i

    Adjacent segment maps are then combined in a tree. This is a CPU reference
    for the dense block-prefix algorithm; the NKI kernel can mirror this only
    through explicit matmul-based segment combines, not tensor_tensor_scan.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rows, cols = lhs.shape
    if rows != cols:
        raise ValueError(f"lhs must be square, got {lhs.shape}")
    if rhs.shape[0] != rows:
        raise ValueError(
            f"rhs row count must match lhs, got lhs={lhs.shape} rhs={rhs.shape}"
        )

    segments = []
    for row_start in range(0, rows, block_size):
        row_end = min(row_start + block_size, rows)
        diag_block = lhs[row_start:row_end, row_start:row_end]
        solved_rhs = torch.linalg.solve_triangular(
            diag_block,
            rhs[row_start:row_end],
            upper=False,
        )
        if row_start == 0:
            transfer = lhs.new_empty((row_end - row_start, 0))
        else:
            transfer = -torch.linalg.solve_triangular(
                diag_block,
                lhs[row_start:row_end, 0:row_start],
                upper=False,
            )
        segments.append(
            {
                "start": row_start,
                "end": row_end,
                "rhs": solved_rhs,
                "transfer": transfer,
            }
        )

    while len(segments) > 1:
        merged = []
        for idx in range(0, len(segments), 2):
            if idx + 1 == len(segments):
                merged.append(segments[idx])
                continue

            left = segments[idx]
            right = segments[idx + 1]
            if left["end"] != right["start"]:
                raise ValueError(
                    "segments must be contiguous, got "
                    f"{left['start']}:{left['end']} and "
                    f"{right['start']}:{right['end']}"
                )

            ext_width = left["start"]
            left_width = left["end"] - left["start"]
            right_transfer = right["transfer"]
            right_ext = right_transfer[:, 0:ext_width]
            right_left = right_transfer[:, ext_width : ext_width + left_width]

            merged_right_rhs = right["rhs"] + right_left @ left["rhs"]
            if ext_width == 0:
                merged_transfer = lhs.new_empty(
                    (left["end"] - left["start"] + right["end"] - right["start"], 0)
                )
            else:
                merged_right_transfer = right_ext + right_left @ left["transfer"]
                merged_transfer = torch.cat(
                    [left["transfer"], merged_right_transfer],
                    dim=0,
                )

            merged.append(
                {
                    "start": left["start"],
                    "end": right["end"],
                    "rhs": torch.cat([left["rhs"], merged_right_rhs], dim=0),
                    "transfer": merged_transfer,
                }
            )
        segments = merged

    return segments[0]["rhs"]


def _hierarchical_kkt_inverse_transpose(
    torch: Any,
    a_t: Any,
    leaf_size: int,
) -> Any:
    """Build ``inv(I - A).T`` using FlashQLA-style block combines."""
    rows, cols = a_t.shape
    if rows != cols:
        raise ValueError(f"a_t must be square, got {a_t.shape}")
    if rows <= leaf_size:
        steps = math.ceil(math.log2(rows))
        inv_t = torch.eye(rows, dtype=a_t.dtype, device=a_t.device)
        power_t = a_t.clone()
        power = power_t.T.contiguous()
        for step_idx in range(steps):
            inv_t = inv_t + inv_t @ power_t
            if step_idx != steps - 1:
                power = power @ power
                power_t = power.T.contiguous()
        return inv_t

    if rows % 2 != 0:
        raise ValueError(f"rows must split evenly, got {rows}")
    mid = rows // 2
    left_t = _hierarchical_kkt_inverse_transpose(
        torch,
        a_t[0:mid, 0:mid],
        leaf_size,
    )
    right_t = _hierarchical_kkt_inverse_transpose(
        torch,
        a_t[mid:rows, mid:rows],
        leaf_size,
    )
    a_cross_t = a_t[0:mid, mid:rows]
    cross_t = left_t @ a_cross_t @ right_t

    inv_t = torch.zeros_like(a_t)
    inv_t[0:mid, 0:mid] = left_t
    inv_t[0:mid, mid:rows] = cross_t
    inv_t[mid:rows, mid:rows] = right_t
    return inv_t


def hierarchical_kkt_lower_triangular_solve(
    torch: Any,
    lhs: Any,
    rhs: Any,
    leaf_size: int = 16,
) -> Any:
    """Solve ``lhs @ x = rhs`` with FlashQLA-style KKT hierarchy.

    FlashQLA's Hopper KKT kernel builds the intra-chunk triangular inverse
    through small diagonal inversions and block combines. This CPU reference
    mirrors that algebra: for ``lhs = I - A`` with strictly lower ``A``, build
    ``N.T = inv(I - A).T`` recursively, then compute ``x = N @ rhs``.
    """
    if leaf_size <= 0:
        raise ValueError("leaf_size must be positive")
    rows, cols = lhs.shape
    if rows != cols:
        raise ValueError(f"lhs must be square, got {lhs.shape}")
    if rhs.shape[0] != rows:
        raise ValueError(
            f"rhs row count must match lhs, got lhs={lhs.shape} rhs={rhs.shape}"
        )
    if rows % leaf_size != 0 or leaf_size & (leaf_size - 1) != 0:
        raise ValueError(
            "leaf_size must be a power of two that divides the row count; "
            f"got rows={rows}, leaf_size={leaf_size}"
        )

    eye = torch.eye(rows, dtype=lhs.dtype, device=lhs.device)
    a_t = (eye - lhs).T.contiguous()
    inv_t = _hierarchical_kkt_inverse_transpose(torch, a_t, leaf_size)
    return inv_t.T @ rhs


def scan_doubling_lower_triangular_solve(
    torch: Any,
    lhs: Any,
    rhs: Any,
    steps: int,
) -> Any:
    """Solve/approximate lower-triangular lhs @ x = rhs by Neumann doubling."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    rows, cols = lhs.shape
    if rows != cols:
        raise ValueError(f"lhs must be square, got {lhs.shape}")
    if rhs.shape[0] != rows:
        raise ValueError(
            f"rhs row count must match lhs, got lhs={lhs.shape} rhs={rhs.shape}"
        )

    eye = torch.eye(rows, dtype=lhs.dtype, device=lhs.device)
    power = eye - lhs
    solved = rhs.clone()
    for scan_idx in range(steps):
        solved = solved + power @ solved
        if scan_idx != steps - 1:
            power = power @ power
    return solved


def reference_math_one_head(torch: Any, inputs: dict[str, Any]) -> tuple[Any, Any]:
    chunk_size = int(inputs.get("chunk_size", P_MAX))
    lower = inputs["lower_mask"][0:chunk_size, 0:chunk_size]
    lower_diag = inputs["lower_mask_diag"][0:chunk_size, 0:chunk_size]
    eye = inputs["identity"][0:chunk_size, 0:chunk_size]
    state = inputs["state_in"].clone()
    outputs = []

    for start in range(0, inputs["query"].shape[0], chunk_size):
        end = start + chunk_size
        q, k = normalize_reference_qk(
            torch,
            inputs["query"][start:end],
            inputs["key"][start:end],
        )
        v = inputs["value"][start:end]
        g = inputs["g_raw"][start:end]
        beta = inputs["beta"][start:end]

        gc = torch.cumsum(g, dim=0)
        gl = gc[-1:]
        k_beta = k * beta
        v_beta = v * beta

        decay_strict = stable_causal_decay(torch, gc, lower)
        decay_diag = stable_causal_decay(torch, gc, lower_diag)

        qk_beta = k_beta @ k.T
        a_mat = -(qk_beta * decay_strict) * lower

        lhs = eye - a_mat

        exp_gc = torch.exp(gc)
        solve_rhs = v_beta - ((k_beta * exp_gc) @ state)
        v_new = torch.linalg.solve_triangular(lhs, solve_rhs, upper=False)
        attn_intra = (q @ k.T) * decay_diag

        chunk_out = ((q * exp_gc) @ state) + (attn_intra @ v_new)
        outputs.append(chunk_out)

        k_raw_decay = k * torch.exp(gl - gc)
        state = (state * torch.exp(gl)) + (k_raw_decay.T @ v_new)

    return torch.cat(outputs, dim=0).contiguous(), state.contiguous()


def deltanet_chunk_affine_parts(
    torch: Any,
    inputs: dict[str, Any],
    start: int,
) -> dict[str, Any]:
    """Return FlashQLA-style per-chunk affine pieces independent of state.

    For each chunk, DeltaNet can be represented as:

        output_i = output_base_i + output_state_i @ state_i
        state_{i+1} = state_matrix_i @ state_i + state_bias_i

    The current fused NKI path computes this implicitly while carrying
    ``state`` through chunks. AutoCP needs these pieces first, then performs a
    prefix over ``(state_matrix, state_bias)`` to recover chunk initial states.
    """
    end = start + P_MAX
    lower = inputs["lower_mask"]
    lower_diag = inputs["lower_mask_diag"]
    eye = inputs["identity"]

    q, k = normalize_reference_qk(
        torch,
        inputs["query"][start:end],
        inputs["key"][start:end],
    )
    v = inputs["value"][start:end]
    g = inputs["g_raw"][start:end]
    beta = inputs["beta"][start:end]

    gc = torch.cumsum(g, dim=0)
    gl = gc[-1:]
    exp_gc = torch.exp(gc)
    exp_gl = torch.exp(gl).reshape(())
    k_beta = k * beta
    v_beta = v * beta

    decay_strict = stable_causal_decay(torch, gc, lower)
    decay_diag = stable_causal_decay(torch, gc, lower_diag)

    qk_beta = k_beta @ k.T
    a_mat = -(qk_beta * decay_strict) * lower
    lhs = eye - a_mat

    value_u = torch.linalg.solve_triangular(lhs, v_beta, upper=False)
    state_w = torch.linalg.solve_triangular(lhs, k_beta * exp_gc, upper=False)
    attn_intra = (q @ k.T) * decay_diag
    k_raw_decay = k * torch.exp(gl - gc)

    output_base = attn_intra @ value_u
    output_state = (q * exp_gc) - (attn_intra @ state_w)
    state_matrix = (exp_gl * eye) - (k_raw_decay.T @ state_w)
    state_bias = k_raw_decay.T @ value_u

    return {
        "output_base": output_base,
        "output_state": output_state,
        "state_matrix": state_matrix,
        "state_bias": state_bias,
    }


def apply_deltanet_chunk_affine(
    torch: Any,
    parts: dict[str, Any],
    state: Any,
) -> tuple[Any, Any]:
    output = parts["output_base"] + (parts["output_state"] @ state)
    next_state = (parts["state_matrix"] @ state) + parts["state_bias"]
    return output, next_state


def compose_deltanet_state_affine(
    torch: Any,
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    """Compose two state transforms, applying ``first`` then ``second``."""
    matrix = second["state_matrix"] @ first["state_matrix"]
    bias = (second["state_matrix"] @ first["state_bias"]) + second["state_bias"]
    return {"state_matrix": matrix, "state_bias": bias}


def autocp_reference_math_one_head(
    torch: Any,
    inputs: dict[str, Any],
    cp_chunks: int = 4,
) -> tuple[Any, Any]:
    """Reference FlashQLA-style grouped context-parallel state prepass.

    ``cp_chunks`` is the number of 128-token chunks in each local CP segment.
    This CPU reference still executes serially, but its dataflow is the one we
    need for a NKI port: build state-independent chunk transforms, combine them
    per segment, then run each segment from a corrected initial state.
    """
    if cp_chunks <= 0:
        raise ValueError("cp_chunks must be positive")
    seq_len = inputs["query"].shape[0]
    if seq_len % P_MAX != 0:
        raise ValueError(f"seq_len must be divisible by {P_MAX}, got {seq_len}")

    parts = [
        deltanet_chunk_affine_parts(torch, inputs, start)
        for start in range(0, seq_len, P_MAX)
    ]

    eye = inputs["identity"]
    zero_state = torch.zeros_like(inputs["state_in"])
    group_transforms = []
    for group_start in range(0, len(parts), cp_chunks):
        group_transform = {"state_matrix": eye, "state_bias": zero_state}
        for chunk_parts in parts[group_start : group_start + cp_chunks]:
            group_transform = compose_deltanet_state_affine(
                torch,
                group_transform,
                chunk_parts,
            )
        group_transforms.append(group_transform)

    group_initial_states = []
    state = inputs["state_in"].clone()
    for transform in group_transforms:
        group_initial_states.append(state)
        state = (transform["state_matrix"] @ state) + transform["state_bias"]

    outputs = []
    for group_idx, group_start in enumerate(range(0, len(parts), cp_chunks)):
        state = group_initial_states[group_idx]
        for chunk_parts in parts[group_start : group_start + cp_chunks]:
            output, state = apply_deltanet_chunk_affine(torch, chunk_parts, state)
            outputs.append(output)

    final_state = state
    return torch.cat(outputs, dim=0).contiguous(), final_state.contiguous()


def blocked_reference_math_one_head(
    torch: Any,
    inputs: dict[str, Any],
    block_size: int = 16,
) -> tuple[Any, Any]:
    lower = inputs["lower_mask"]
    lower_diag = inputs["lower_mask_diag"]
    eye = inputs["identity"]
    state = inputs["state_in"].clone()
    outputs = []

    for start in range(0, inputs["query"].shape[0], P_MAX):
        end = start + P_MAX
        q, k = normalize_reference_qk(
            torch,
            inputs["query"][start:end],
            inputs["key"][start:end],
        )
        v = inputs["value"][start:end]
        g = inputs["g_raw"][start:end]
        beta = inputs["beta"][start:end]

        gc = torch.cumsum(g, dim=0)
        gl = gc[-1:]
        k_beta = k * beta
        v_beta = v * beta

        decay_strict = stable_causal_decay(torch, gc, lower)
        decay_diag = stable_causal_decay(torch, gc, lower_diag)

        qk_beta = k_beta @ k.T
        a_mat = -(qk_beta * decay_strict) * lower
        lhs = eye - a_mat

        exp_gc = torch.exp(gc)
        solve_rhs = v_beta - ((k_beta * exp_gc) @ state)
        v_new = blocked_lower_triangular_solve(
            torch,
            lhs,
            solve_rhs,
            block_size,
        )
        attn_intra = (q @ k.T) * decay_diag

        chunk_out = ((q * exp_gc) @ state) + (attn_intra @ v_new)
        outputs.append(chunk_out)

        k_raw_decay = k * torch.exp(gl - gc)
        state = (state * torch.exp(gl)) + (k_raw_decay.T @ v_new)

    return torch.cat(outputs, dim=0).contiguous(), state.contiguous()


def reference_math(torch: Any, inputs: dict[str, Any]) -> tuple[Any, Any]:
    if inputs["query"].dim() == 2:
        return reference_math_one_head(torch, inputs)

    outputs = []
    states = []
    for head_idx in range(inputs["query"].shape[0]):
        head_inputs = {
            "chunk_size": inputs.get("chunk_size", P_MAX),
            "query": inputs["query"][head_idx],
            "key": inputs["key"][head_idx],
            "value": inputs["value"][head_idx],
            "g_raw": inputs["g_raw"][head_idx],
            "beta": inputs["beta"][head_idx],
            "state_in": inputs["state_in"][head_idx],
            "lower_mask": inputs["lower_mask"],
            "identity": inputs["identity"],
            "lower_mask_diag": inputs["lower_mask_diag"],
        }
        out, state = reference_math_one_head(torch, head_inputs)
        outputs.append(out)
        states.append(state)

    return torch.stack(outputs, dim=0).contiguous(), torch.stack(states, dim=0).contiguous()


def blocked_reference_math(
    torch: Any,
    inputs: dict[str, Any],
    block_size: int = 16,
) -> tuple[Any, Any]:
    if inputs["query"].dim() == 2:
        return blocked_reference_math_one_head(torch, inputs, block_size)

    outputs = []
    states = []
    for head_idx in range(inputs["query"].shape[0]):
        head_inputs = {
            "chunk_size": inputs.get("chunk_size", P_MAX),
            "query": inputs["query"][head_idx],
            "key": inputs["key"][head_idx],
            "value": inputs["value"][head_idx],
            "g_raw": inputs["g_raw"][head_idx],
            "beta": inputs["beta"][head_idx],
            "state_in": inputs["state_in"][head_idx],
            "lower_mask": inputs["lower_mask"],
            "identity": inputs["identity"],
            "lower_mask_diag": inputs["lower_mask_diag"],
        }
        out, state = blocked_reference_math_one_head(
            torch,
            head_inputs,
            block_size,
        )
        outputs.append(out)
        states.append(state)

    return torch.stack(outputs, dim=0).contiguous(), torch.stack(states, dim=0).contiguous()


def autocp_reference_math(
    torch: Any,
    inputs: dict[str, Any],
    cp_chunks: int = 4,
) -> tuple[Any, Any]:
    if inputs["query"].dim() == 2:
        return autocp_reference_math_one_head(torch, inputs, cp_chunks)

    outputs = []
    states = []
    for head_idx in range(inputs["query"].shape[0]):
        head_inputs = {
            "chunk_size": inputs.get("chunk_size", P_MAX),
            "query": inputs["query"][head_idx],
            "key": inputs["key"][head_idx],
            "value": inputs["value"][head_idx],
            "g_raw": inputs["g_raw"][head_idx],
            "beta": inputs["beta"][head_idx],
            "state_in": inputs["state_in"][head_idx],
            "lower_mask": inputs["lower_mask"],
            "identity": inputs["identity"],
            "lower_mask_diag": inputs["lower_mask_diag"],
        }
        out, state = autocp_reference_math_one_head(torch, head_inputs, cp_chunks)
        outputs.append(out)
        states.append(state)

    return torch.stack(outputs, dim=0).contiguous(), torch.stack(states, dim=0).contiguous()


def build_compact_autocp_segment_transforms(
    torch: Any,
    inputs: dict[str, Any],
    cp_chunks: int = 4,
) -> dict[str, Any]:
    """Compose chunk state transforms into compact per-segment transforms.

    Unlike ``build_autocp_affine_stacks``, this deliberately does not retain
    output-affine pieces. The intended NKI path prefixes only segment-level
    state transforms, then replays each segment recurrently from its corrected
    initial state.
    """
    if cp_chunks <= 0:
        raise ValueError("cp_chunks must be positive")
    seq_len = inputs["query"].shape[0]
    if seq_len % P_MAX != 0:
        raise ValueError(f"seq_len must be divisible by {P_MAX}, got {seq_len}")

    num_chunks = seq_len // P_MAX
    identity = inputs["identity"]
    zero_state = torch.zeros_like(inputs["state_in"])
    segment_matrices = []
    segment_biases = []
    segment_chunk_counts = []

    for chunk_start in range(0, num_chunks, cp_chunks):
        segment_transform = {"state_matrix": identity, "state_bias": zero_state}
        chunk_end = min(chunk_start + cp_chunks, num_chunks)
        for chunk_idx in range(chunk_start, chunk_end):
            chunk_parts = deltanet_chunk_affine_parts(
                torch,
                inputs,
                chunk_idx * P_MAX,
            )
            segment_transform = compose_deltanet_state_affine(
                torch,
                segment_transform,
                chunk_parts,
            )
        segment_matrices.append(segment_transform["state_matrix"])
        segment_biases.append(segment_transform["state_bias"])
        segment_chunk_counts.append(chunk_end - chunk_start)

    return {
        "state_matrix": torch.stack(segment_matrices, dim=0).contiguous(),
        "state_bias": torch.stack(segment_biases, dim=0).contiguous(),
        "chunk_counts": segment_chunk_counts,
    }


def slice_single_head_inputs(inputs: dict[str, Any], start: int, end: int, state: Any) -> dict[str, Any]:
    return {
        "chunk_size": inputs.get("chunk_size", P_MAX),
        "query": inputs["query"][start:end],
        "key": inputs["key"][start:end],
        "value": inputs["value"][start:end],
        "g_raw": inputs["g_raw"][start:end],
        "beta": inputs["beta"][start:end],
        "state_in": state,
        "lower_mask": inputs["lower_mask"],
        "identity": inputs["identity"],
        "lower_mask_diag": inputs["lower_mask_diag"],
    }


def compact_autocp_reference_math_one_head(
    torch: Any,
    inputs: dict[str, Any],
    cp_chunks: int = 4,
) -> tuple[Any, Any]:
    """Reference compact AutoCP: segment state prefix plus recurrent replay."""
    segment_transforms = build_compact_autocp_segment_transforms(
        torch,
        inputs,
        cp_chunks,
    )
    segment_states, final_state = autocp_state_prefix_reference(
        torch,
        segment_transforms["state_matrix"],
        segment_transforms["state_bias"],
        inputs["state_in"],
    )

    outputs = []
    token_start = 0
    for segment_idx, chunk_count in enumerate(segment_transforms["chunk_counts"]):
        token_end = token_start + (chunk_count * P_MAX)
        segment_inputs = slice_single_head_inputs(
            inputs,
            token_start,
            token_end,
            segment_states[segment_idx],
        )
        segment_output, _ = reference_math_one_head(
            torch,
            segment_inputs,
        )
        outputs.append(segment_output)
        token_start = token_end

    return torch.cat(outputs, dim=0).contiguous(), final_state.contiguous()


def compact_autocp_reference_math(
    torch: Any,
    inputs: dict[str, Any],
    cp_chunks: int = 4,
) -> tuple[Any, Any]:
    if inputs["query"].dim() == 2:
        return compact_autocp_reference_math_one_head(torch, inputs, cp_chunks)

    outputs = []
    states = []
    for head_idx in range(inputs["query"].shape[0]):
        head_inputs = {
            "chunk_size": inputs.get("chunk_size", P_MAX),
            "query": inputs["query"][head_idx],
            "key": inputs["key"][head_idx],
            "value": inputs["value"][head_idx],
            "g_raw": inputs["g_raw"][head_idx],
            "beta": inputs["beta"][head_idx],
            "state_in": inputs["state_in"][head_idx],
            "lower_mask": inputs["lower_mask"],
            "identity": inputs["identity"],
            "lower_mask_diag": inputs["lower_mask_diag"],
        }
        out, state = compact_autocp_reference_math_one_head(
            torch,
            head_inputs,
            cp_chunks,
        )
        outputs.append(out)
        states.append(state)

    return torch.stack(outputs, dim=0).contiguous(), torch.stack(states, dim=0).contiguous()


def compact_autocp_materialization_counts(
    seq_len: int,
    cp_chunks: int,
) -> dict[str, int]:
    if cp_chunks <= 0:
        raise ValueError("cp_chunks must be positive")
    if seq_len % P_MAX != 0:
        raise ValueError(f"seq_len must be divisible by {P_MAX}, got {seq_len}")
    num_chunks = seq_len // P_MAX
    num_segments = math.ceil(num_chunks / cp_chunks)
    return {
        "num_chunks": num_chunks,
        "num_segments": num_segments,
        "existing_autocp_dense_128x128_tensors": 4 * num_chunks,
        "compact_prefix_dense_128x128_tensors": 2 * num_segments,
        "dense_tensor_reduction": (4 * num_chunks) - (2 * num_segments),
    }


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


def multihead_tensor_metrics(torch: Any, actual: Any, expected: Any) -> list[dict[str, Any]]:
    """Return per-head metrics plus pairwise relative norms for head-mix debugging."""
    metrics = []
    for actual_head in range(actual.shape[0]):
        pairwise_relative_norm = []
        for expected_head in range(expected.shape[0]):
            pairwise_relative_norm.append(
                tensor_metrics(
                    torch,
                    actual[actual_head],
                    expected[expected_head],
                )["relative_norm"]
            )
        head_metrics = tensor_metrics(
            torch,
            actual[actual_head],
            expected[actual_head],
        )
        head_metrics["actual_head"] = actual_head
        head_metrics["pairwise_relative_norm"] = pairwise_relative_norm
        metrics.append(head_metrics)
    return metrics


def build_autocp_affine_stacks(
    torch: Any,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    parts = [
        deltanet_chunk_affine_parts(torch, inputs, start)
        for start in range(0, inputs["query"].shape[0], P_MAX)
    ]
    return {
        "state_matrix": torch.stack(
            [chunk_parts["state_matrix"] for chunk_parts in parts],
            dim=0,
        ).contiguous(),
        "state_bias": torch.stack(
            [chunk_parts["state_bias"] for chunk_parts in parts],
            dim=0,
        ).contiguous(),
        "output_base": torch.stack(
            [chunk_parts["output_base"] for chunk_parts in parts],
            dim=0,
        ).contiguous(),
        "output_state": torch.stack(
            [chunk_parts["output_state"] for chunk_parts in parts],
            dim=0,
        ).contiguous(),
    }


def autocp_state_prefix_reference(
    torch: Any,
    state_matrix: Any,
    state_bias: Any,
    initial_state: Any,
) -> tuple[Any, Any]:
    chunk_states = []
    state = initial_state.clone()
    for chunk_idx in range(state_matrix.shape[0]):
        chunk_states.append(state.clone())
        state = (state_matrix[chunk_idx] @ state) + state_bias[chunk_idx]
    return torch.stack(chunk_states, dim=0).contiguous(), state.contiguous()


def validate_cpu_chunk_invariance(torch: Any, args: argparse.Namespace) -> int:
    if args.seq_len % P_MAX != 0:
        raise ValueError(
            "--validate-cpu-chunk-invariance requires --seq-len to be a "
            f"positive multiple of {P_MAX}; got {args.seq_len}"
        )

    args128 = argparse.Namespace(**{**vars(args), "chunk_size": P_MAX})
    args64 = argparse.Namespace(**{**vars(args), "chunk_size": 64})
    inputs128 = make_inputs(torch, args128)
    inputs64 = make_inputs(torch, args64)

    ref128_out, ref128_state = reference_math(torch, inputs128)
    ref64_out, ref64_state = reference_math(torch, inputs64)
    output_close = bool(
        torch.allclose(ref64_out, ref128_out, atol=args.atol, rtol=args.rtol)
    )
    state_close = bool(
        torch.allclose(ref64_state, ref128_state, atol=args.atol, rtol=args.rtol)
    )
    output_finite = bool(torch.isfinite(ref64_out).all().item())
    state_finite = bool(torch.isfinite(ref64_state).all().item())
    passed = bool(output_close and state_close and output_finite and state_finite)

    result = {
        "passed": passed,
        "validate_cpu_chunk_invariance": True,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "heads": args.heads if args.multihead else 1,
        "multihead": args.multihead,
        "atol": args.atol,
        "rtol": args.rtol,
        "output_finite": output_finite,
        "state_finite": state_finite,
        "chunk64_vs_chunk128": {
            "output": tensor_metrics(torch, ref64_out, ref128_out),
            "state": tensor_metrics(torch, ref64_state, ref128_state),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def parse_restore_split_lens(spec: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --restore-split-lens {spec!r}") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(
            "--restore-split-lens must contain positive integer lengths; "
            f"got {spec!r}"
        )
    return values


def zero_sequence_tail(torch: Any, inputs: dict[str, Any], real_seq_len: int) -> None:
    sequence_keys = ("query", "key", "value", "g_raw", "beta")
    for key in sequence_keys:
        tensor = inputs[key]
        if tensor.dim() == 2:
            tensor[real_seq_len:] = 0
        else:
            tensor[:, real_seq_len:] = 0


def slice_sequence_tensor(tensor: Any, start: int, end: int) -> Any:
    if tensor.dim() == 2:
        return tensor[start:end].contiguous()
    return tensor[:, start:end].contiguous()


def slice_sequence_for_compare(tensor: Any, end: int) -> Any:
    if tensor.dim() == 2:
        return tensor[:end]
    return tensor[:, :end]


def cat_sequence_outputs(torch: Any, tensors: list[Any]) -> Any:
    if not tensors:
        raise ValueError("No output tensors to concatenate")
    if tensors[0].dim() == 2:
        return torch.cat(tensors, dim=0)
    return torch.cat(tensors, dim=1)


def copy_inputs_with_state_and_slice(
    inputs: dict[str, Any],
    *,
    start: int,
    end: int,
    state: Any,
) -> dict[str, Any]:
    return {
        "chunk_size": inputs.get("chunk_size", P_MAX),
        "query": slice_sequence_tensor(inputs["query"], start, end),
        "key": slice_sequence_tensor(inputs["key"], start, end),
        "value": slice_sequence_tensor(inputs["value"], start, end),
        "g_raw": slice_sequence_tensor(inputs["g_raw"], start, end),
        "beta": slice_sequence_tensor(inputs["beta"], start, end),
        "state_in": state,
        "lower_mask": inputs["lower_mask"],
        "identity": inputs["identity"],
        "lower_mask_diag": inputs["lower_mask_diag"],
    }


def run_fused_kernel_once(
    torch: Any,
    deltanet_fused_chunked_fwd: Any,
    inputs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, Any, list[str]]:
    if args.multihead:
        pair_outputs = []
        pair_states = []
        launch_spec_labels = []
        head_group_size = min(args.head_group_size, args.heads)
        for head_start in range(0, args.heads, head_group_size):
            head_end = min(head_start + head_group_size, args.heads)
            launch_heads = head_end - head_start
            launch_spec = multihead_launch_spec(launch_heads, args.lnc)
            launch_spec_labels.append(launch_spec_label(launch_spec))
            out_pair, state_pair = deltanet_fused_chunked_fwd[launch_spec](
                inputs["query"][head_start:head_end],
                inputs["key"][head_start:head_end],
                inputs["value"][head_start:head_end],
                inputs["g_raw"][head_start:head_end],
                inputs["beta"][head_start:head_end],
                inputs["state_in"][head_start:head_end],
                inputs["lower_mask"],
                inputs["identity"],
                inputs["lower_mask_diag"],
            )
            pair_outputs.append(out_pair)
            pair_states.append(state_pair)
        return torch.cat(pair_outputs, dim=0), torch.cat(pair_states, dim=0), launch_spec_labels

    out_dev, state_dev = deltanet_fused_chunked_fwd(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["g_raw"],
        inputs["beta"],
        inputs["state_in"],
        inputs["lower_mask"],
        inputs["identity"],
        inputs["lower_mask_diag"],
    )
    return out_dev, state_dev, []


def validate_restored_suffix_carry(
    torch: Any,
    xm: Any,
    args: argparse.Namespace,
    inspect_dir: Path,
) -> int:
    split_lens = parse_restore_split_lens(args.restore_split_lens)
    bucket_size = int(args.restore_bucket_size)
    chunk_size = int(args.chunk_size)
    if bucket_size <= 0:
        raise ValueError("--restore-bucket-size must be positive")
    if bucket_size % chunk_size != 0:
        raise ValueError(
            "--restore-bucket-size must be a multiple of --chunk-size; "
            f"got bucket_size={bucket_size}, chunk_size={chunk_size}"
        )
    if any(length > bucket_size for length in split_lens):
        raise ValueError(
            "Each --restore-split-lens value must fit in --restore-bucket-size; "
            f"split_lens={split_lens}, bucket_size={bucket_size}"
        )

    real_seq_len = sum(split_lens)
    padded_seq_len = len(split_lens) * bucket_size
    input_args = argparse.Namespace(**{**vars(args), "seq_len": padded_seq_len})
    inputs = make_inputs(torch, input_args)
    zero_sequence_tail(torch, inputs, real_seq_len)
    ref_out, ref_state = reference_math(torch, inputs)

    deltanet_fused_chunked_fwd = load_fused_kernel(args.multihead)
    device = xm.xla_device()
    xla_inputs = move_tensor_inputs_to_device(inputs, device)

    full_out_cpu = full_state_cpu = None
    split_out_cpu = split_state_cpu = None
    run_elapsed_seconds = []
    launch_spec_labels = []
    for _ in range(args.runs):
        run_start = time.perf_counter()
        full_out_dev, full_state_dev, full_launch_specs = run_fused_kernel_once(
            torch,
            deltanet_fused_chunked_fwd,
            xla_inputs,
            args,
        )
        if full_launch_specs and not launch_spec_labels:
            launch_spec_labels = full_launch_specs

        state_dev = xla_inputs["state_in"]
        split_outputs = []
        offset = 0
        for real_len in split_lens:
            segment_inputs = copy_inputs_with_state_and_slice(
                xla_inputs,
                start=offset,
                end=offset + bucket_size,
                state=state_dev,
            )
            out_dev, state_dev, segment_launch_specs = run_fused_kernel_once(
                torch,
                deltanet_fused_chunked_fwd,
                segment_inputs,
                args,
            )
            if segment_launch_specs and not launch_spec_labels:
                launch_spec_labels = segment_launch_specs
            split_outputs.append(out_dev)
            offset += bucket_size

        split_out_dev = cat_sequence_outputs(torch, split_outputs)
        split_state_dev = state_dev
        xm.mark_step()
        full_out_cpu = full_out_dev.detach().cpu().float()
        full_state_cpu = full_state_dev.detach().cpu().float()
        split_out_cpu = split_out_dev.detach().cpu().float()
        split_state_cpu = split_state_dev.detach().cpu().float()
        run_elapsed_seconds.append(time.perf_counter() - run_start)

    assert full_out_cpu is not None
    assert full_state_cpu is not None
    assert split_out_cpu is not None
    assert split_state_cpu is not None

    ref_real = slice_sequence_for_compare(ref_out, real_seq_len)
    full_real = slice_sequence_for_compare(full_out_cpu, real_seq_len)
    split_real = slice_sequence_for_compare(split_out_cpu, real_seq_len)

    full_output_close = bool(
        torch.allclose(full_real, ref_real, atol=args.atol, rtol=args.rtol)
    )
    full_state_close = bool(
        torch.allclose(full_state_cpu, ref_state, atol=args.atol, rtol=args.rtol)
    )
    split_output_close = bool(
        torch.allclose(split_real, ref_real, atol=args.atol, rtol=args.rtol)
    )
    split_state_close = bool(
        torch.allclose(split_state_cpu, ref_state, atol=args.atol, rtol=args.rtol)
    )
    split_vs_full_output_close = bool(
        torch.allclose(split_real, full_real, atol=args.atol, rtol=args.rtol)
    )
    split_vs_full_state_close = bool(
        torch.allclose(split_state_cpu, full_state_cpu, atol=args.atol, rtol=args.rtol)
    )
    finite = {
        "full_output": bool(torch.isfinite(full_real).all().item()),
        "full_state": bool(torch.isfinite(full_state_cpu).all().item()),
        "split_output": bool(torch.isfinite(split_real).all().item()),
        "split_state": bool(torch.isfinite(split_state_cpu).all().item()),
    }
    passed = bool(
        full_output_close
        and full_state_close
        and split_output_close
        and split_state_close
        and split_vs_full_output_close
        and split_vs_full_state_close
        and all(finite.values())
    )

    result = {
        "passed": passed,
        "validate_restored_suffix_carry": True,
        "seed": args.seed,
        "split_lens": split_lens,
        "restore_bucket_size": bucket_size,
        "real_seq_len": real_seq_len,
        "padded_seq_len": padded_seq_len,
        "chunk_size": chunk_size,
        "heads": args.heads if args.multihead else 1,
        "head_group_size": args.head_group_size if args.multihead else 1,
        "launch_specs": launch_spec_labels if args.multihead else [],
        "multihead": args.multihead,
        "runs": args.runs,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cached_run_elapsed_seconds": run_elapsed_seconds[1:],
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "finite": finite,
        "close": {
            "full_output": full_output_close,
            "full_state": full_state_close,
            "split_output": split_output_close,
            "split_state": split_state_close,
            "split_vs_full_output": split_vs_full_output_close,
            "split_vs_full_state": split_vs_full_state_close,
        },
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
                "QWEN36_DELTANET_CHUNK_SIZE",
                "QWEN36_DELTANET_SOLVE_BLOCK_SIZE",
                "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
                "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
                "QWEN36_DELTANET_SOLVE_MODE",
            )
        },
        "nki_vs_reference": {
            "full_output_real": tensor_metrics(torch, full_real, ref_real),
            "full_state": tensor_metrics(torch, full_state_cpu, ref_state),
            "split_output_real": tensor_metrics(torch, split_real, ref_real),
            "split_state": tensor_metrics(torch, split_state_cpu, ref_state),
            "split_vs_full_output_real": tensor_metrics(torch, split_real, full_real),
            "split_vs_full_state": tensor_metrics(
                torch,
                split_state_cpu,
                full_state_cpu,
            ),
        },
    }
    if args.multihead:
        result["nki_vs_reference"]["split_output_per_head"] = multihead_tensor_metrics(
            torch,
            split_real,
            ref_real,
        )
        result["nki_vs_reference"]["split_state_per_head"] = multihead_tensor_metrics(
            torch,
            split_state_cpu,
            ref_state,
        )
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def validate_autocp_affine_chunk(torch: Any, xm: Any, args: argparse.Namespace, inspect_dir: Path) -> int:
    if args.multihead:
        raise ValueError("--validate-autocp-affine expects single-head inputs")

    deltanet_autocp_affine_chunk = load_autocp_affine_kernel()
    inputs = make_inputs(torch, args)
    ref_parts = deltanet_chunk_affine_parts(torch, inputs, 0)

    device = xm.xla_device()
    xla_inputs = move_tensor_inputs_to_device(inputs, device)

    part_names = ("output_base", "output_state", "state_matrix", "state_bias")
    actual_parts = None
    run_elapsed_seconds = []
    for _ in range(args.runs):
        run_start = time.perf_counter()
        parts_dev = deltanet_autocp_affine_chunk(
            xla_inputs["query"][0:P_MAX],
            xla_inputs["key"][0:P_MAX],
            xla_inputs["value"][0:P_MAX],
            xla_inputs["g_raw"][0:P_MAX],
            xla_inputs["beta"][0:P_MAX],
            xla_inputs["lower_mask"],
            xla_inputs["identity"],
            xla_inputs["lower_mask_diag"],
        )
        xm.mark_step()
        actual_parts = {
            name: part.detach().cpu().float()
            for name, part in zip(part_names, parts_dev, strict=True)
        }
        run_elapsed_seconds.append(time.perf_counter() - run_start)

    assert actual_parts is not None

    close = {
        name: bool(
            torch.allclose(
                actual_parts[name],
                ref_parts[name],
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        for name in part_names
    }
    finite = {
        name: bool(torch.isfinite(actual_parts[name]).all().item())
        for name in part_names
    }
    passed = all(close.values()) and all(finite.values())

    result = {
        "passed": bool(passed),
        "validate_autocp_affine": True,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "runs": args.runs,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cached_run_elapsed_seconds": run_elapsed_seconds[1:],
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "finite": finite,
        "close": close,
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
                "QWEN36_DELTANET_SOLVE_BLOCK_SIZE",
                "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
                "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
                "QWEN36_DELTANET_SOLVE_MODE",
            )
        },
        "nki_vs_reference": {
            name: tensor_metrics(torch, actual_parts[name], ref_parts[name])
            for name in part_names
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def validate_autocp_state_prefix(torch: Any, xm: Any, args: argparse.Namespace, inspect_dir: Path) -> int:
    if args.multihead:
        raise ValueError("--validate-autocp-prefix expects single-head inputs")

    deltanet_autocp_state_prefix = load_autocp_prefix_kernel()
    inputs = make_inputs(torch, args)
    affine = build_autocp_affine_stacks(torch, inputs)
    ref_chunk_states, ref_final_state = autocp_state_prefix_reference(
        torch,
        affine["state_matrix"],
        affine["state_bias"],
        inputs["state_in"],
    )

    device = xm.xla_device()
    state_matrix_dev = affine["state_matrix"].to(device=device)
    state_bias_dev = affine["state_bias"].to(device=device)
    initial_state_dev = inputs["state_in"].to(device=device)

    chunk_states_cpu = final_state_cpu = None
    run_elapsed_seconds = []
    for _ in range(args.runs):
        run_start = time.perf_counter()
        chunk_states_dev, final_state_dev = deltanet_autocp_state_prefix(
            state_matrix_dev,
            state_bias_dev,
            initial_state_dev,
        )
        xm.mark_step()
        chunk_states_cpu = chunk_states_dev.detach().cpu().float()
        final_state_cpu = final_state_dev.detach().cpu().float()
        run_elapsed_seconds.append(time.perf_counter() - run_start)

    assert chunk_states_cpu is not None
    assert final_state_cpu is not None

    chunk_states_close = bool(
        torch.allclose(
            chunk_states_cpu,
            ref_chunk_states,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    final_state_close = bool(
        torch.allclose(
            final_state_cpu,
            ref_final_state,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    chunk_states_finite = bool(torch.isfinite(chunk_states_cpu).all().item())
    final_state_finite = bool(torch.isfinite(final_state_cpu).all().item())
    passed = bool(
        chunk_states_close
        and final_state_close
        and chunk_states_finite
        and final_state_finite
    )

    result = {
        "passed": passed,
        "validate_autocp_prefix": True,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "num_chunks": args.seq_len // P_MAX,
        "runs": args.runs,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cached_run_elapsed_seconds": run_elapsed_seconds[1:],
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "finite": {
            "chunk_states": chunk_states_finite,
            "final_state": final_state_finite,
        },
        "close": {
            "chunk_states": chunk_states_close,
            "final_state": final_state_close,
        },
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
                "QWEN36_DELTANET_SOLVE_BLOCK_SIZE",
                "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
                "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
                "QWEN36_DELTANET_SOLVE_MODE",
            )
        },
        "nki_vs_reference": {
            "chunk_states": tensor_metrics(
                torch,
                chunk_states_cpu,
                ref_chunk_states,
            ),
            "final_state": tensor_metrics(
                torch,
                final_state_cpu,
                ref_final_state,
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def validate_autocp_chain(torch: Any, xm: Any, args: argparse.Namespace, inspect_dir: Path) -> int:
    if args.multihead:
        raise ValueError("--validate-autocp-chain expects single-head inputs")

    deltanet_autocp_state_prefix = load_autocp_prefix_kernel()
    deltanet_autocp_apply_output = load_autocp_apply_kernel()

    inputs = make_inputs(torch, args)
    affine = build_autocp_affine_stacks(torch, inputs)
    ref_out, ref_final_state = autocp_reference_math(torch, inputs)

    device = xm.xla_device()
    state_matrix_dev = affine["state_matrix"].to(device=device)
    state_bias_dev = affine["state_bias"].to(device=device)
    output_base_dev = affine["output_base"].to(device=device)
    output_state_dev = affine["output_state"].to(device=device)
    initial_state_dev = inputs["state_in"].to(device=device)

    out_cpu = final_state_cpu = None
    run_elapsed_seconds = []
    for _ in range(args.runs):
        run_start = time.perf_counter()
        chunk_states_dev, final_state_dev = deltanet_autocp_state_prefix(
            state_matrix_dev,
            state_bias_dev,
            initial_state_dev,
        )
        out_dev = deltanet_autocp_apply_output(
            output_base_dev,
            output_state_dev,
            chunk_states_dev,
        )
        xm.mark_step()
        out_cpu = out_dev.detach().cpu().float()
        final_state_cpu = final_state_dev.detach().cpu().float()
        run_elapsed_seconds.append(time.perf_counter() - run_start)

    assert out_cpu is not None
    assert final_state_cpu is not None

    output_close = bool(torch.allclose(out_cpu, ref_out, atol=args.atol, rtol=args.rtol))
    final_state_close = bool(
        torch.allclose(
            final_state_cpu,
            ref_final_state,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    output_finite = bool(torch.isfinite(out_cpu).all().item())
    final_state_finite = bool(torch.isfinite(final_state_cpu).all().item())
    passed = bool(output_close and final_state_close and output_finite and final_state_finite)

    result = {
        "passed": passed,
        "validate_autocp_chain": True,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "num_chunks": args.seq_len // P_MAX,
        "runs": args.runs,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cached_run_elapsed_seconds": run_elapsed_seconds[1:],
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "finite": {
            "output": output_finite,
            "final_state": final_state_finite,
        },
        "close": {
            "output": output_close,
            "final_state": final_state_close,
        },
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
                "QWEN36_DELTANET_SOLVE_BLOCK_SIZE",
                "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
                "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
                "QWEN36_DELTANET_SOLVE_MODE",
            )
        },
        "nki_vs_reference": {
            "output": tensor_metrics(torch, out_cpu, ref_out),
            "final_state": tensor_metrics(torch, final_state_cpu, ref_final_state),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def validate_autocp_prefix_apply(torch: Any, xm: Any, args: argparse.Namespace, inspect_dir: Path) -> int:
    if args.multihead:
        raise ValueError("--validate-autocp-prefix-apply expects single-head inputs")

    deltanet_autocp_prefix_apply = load_autocp_prefix_apply_kernel()

    inputs = make_inputs(torch, args)
    affine = build_autocp_affine_stacks(torch, inputs)
    ref_out, ref_final_state = autocp_reference_math(torch, inputs)

    device = xm.xla_device()
    output_base_dev = affine["output_base"].to(device=device)
    output_state_dev = affine["output_state"].to(device=device)
    state_matrix_dev = affine["state_matrix"].to(device=device)
    state_bias_dev = affine["state_bias"].to(device=device)
    initial_state_dev = inputs["state_in"].to(device=device)

    out_cpu = final_state_cpu = None
    run_elapsed_seconds = []
    for _ in range(args.runs):
        run_start = time.perf_counter()
        out_dev, final_state_dev = deltanet_autocp_prefix_apply(
            output_base_dev,
            output_state_dev,
            state_matrix_dev,
            state_bias_dev,
            initial_state_dev,
        )
        xm.mark_step()
        out_cpu = out_dev.detach().cpu().float()
        final_state_cpu = final_state_dev.detach().cpu().float()
        run_elapsed_seconds.append(time.perf_counter() - run_start)

    assert out_cpu is not None
    assert final_state_cpu is not None

    output_close = bool(torch.allclose(out_cpu, ref_out, atol=args.atol, rtol=args.rtol))
    final_state_close = bool(
        torch.allclose(
            final_state_cpu,
            ref_final_state,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    output_finite = bool(torch.isfinite(out_cpu).all().item())
    final_state_finite = bool(torch.isfinite(final_state_cpu).all().item())
    passed = bool(output_close and final_state_close and output_finite and final_state_finite)

    result = {
        "passed": passed,
        "validate_autocp_prefix_apply": True,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "num_chunks": args.seq_len // P_MAX,
        "runs": args.runs,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cached_run_elapsed_seconds": run_elapsed_seconds[1:],
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "finite": {
            "output": output_finite,
            "final_state": final_state_finite,
        },
        "close": {
            "output": output_close,
            "final_state": final_state_close,
        },
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
                "QWEN36_DELTANET_SOLVE_BLOCK_SIZE",
                "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
                "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
                "QWEN36_DELTANET_SOLVE_MODE",
            )
        },
        "nki_vs_reference": {
            "output": tensor_metrics(torch, out_cpu, ref_out),
            "final_state": tensor_metrics(torch, final_state_cpu, ref_final_state),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def validate_autocp_full(torch: Any, xm: Any, args: argparse.Namespace, inspect_dir: Path) -> int:
    if args.multihead:
        raise ValueError("--validate-autocp-full expects single-head inputs")

    import nki.language as nl

    deltanet_autocp_affine_sequence = load_autocp_affine_sequence_kernel()
    deltanet_autocp_prefix_apply = load_autocp_prefix_apply_kernel()

    inputs = make_inputs(torch, args)
    ref_out, ref_final_state = autocp_reference_math(torch, inputs)
    ref_affine = build_autocp_affine_stacks(torch, inputs)

    device = xm.xla_device()
    xla_inputs = move_tensor_inputs_to_device(inputs, device)

    out_cpu = final_state_cpu = None
    affine_cpu = None
    run_elapsed_seconds = []
    num_chunks = args.seq_len // P_MAX
    if hasattr(nl, "spmd_dim") and hasattr(nl, "nc"):
        affine_launch_spec = (
            nl.spmd_dim(num_chunks, nl.nc(args.lnc)),
            1,
        )
    else:
        affine_launch_spec = args.lnc
    for _ in range(args.runs):
        run_start = time.perf_counter()
        output_base_dev, output_state_dev, state_matrix_dev, state_bias_dev = (
            deltanet_autocp_affine_sequence[affine_launch_spec](
                xla_inputs["query"],
                xla_inputs["key"],
                xla_inputs["value"],
                xla_inputs["g_raw"],
                xla_inputs["beta"],
                xla_inputs["lower_mask"],
                xla_inputs["identity"],
                xla_inputs["lower_mask_diag"],
            )
        )
        out_dev, final_state_dev = deltanet_autocp_prefix_apply(
            output_base_dev,
            output_state_dev,
            state_matrix_dev,
            state_bias_dev,
            xla_inputs["state_in"],
        )
        xm.mark_step()
        out_cpu = out_dev.detach().cpu().float()
        final_state_cpu = final_state_dev.detach().cpu().float()
        affine_cpu = {
            "output_base": output_base_dev.detach().cpu().float(),
            "output_state": output_state_dev.detach().cpu().float(),
            "state_matrix": state_matrix_dev.detach().cpu().float(),
            "state_bias": state_bias_dev.detach().cpu().float(),
        }
        run_elapsed_seconds.append(time.perf_counter() - run_start)

    assert out_cpu is not None
    assert final_state_cpu is not None
    assert affine_cpu is not None

    output_close = bool(torch.allclose(out_cpu, ref_out, atol=args.atol, rtol=args.rtol))
    final_state_close = bool(
        torch.allclose(
            final_state_cpu,
            ref_final_state,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    affine_close = {
        name: bool(
            torch.allclose(
                affine_cpu[name],
                ref_affine[name],
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        for name in affine_cpu
    }
    output_finite = bool(torch.isfinite(out_cpu).all().item())
    final_state_finite = bool(torch.isfinite(final_state_cpu).all().item())
    affine_finite = {
        name: bool(torch.isfinite(affine_cpu[name]).all().item())
        for name in affine_cpu
    }
    passed = bool(
        output_close
        and final_state_close
        and output_finite
        and final_state_finite
        and all(affine_close.values())
        and all(affine_finite.values())
    )

    result = {
        "passed": passed,
        "validate_autocp_full": True,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "num_chunks": num_chunks,
        "runs": args.runs,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cached_run_elapsed_seconds": run_elapsed_seconds[1:],
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "finite": {
            "output": output_finite,
            "final_state": final_state_finite,
            **{f"affine_{name}": value for name, value in affine_finite.items()},
        },
        "close": {
            "output": output_close,
            "final_state": final_state_close,
            **{f"affine_{name}": value for name, value in affine_close.items()},
        },
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
                "QWEN36_DELTANET_SOLVE_BLOCK_SIZE",
                "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
                "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
                "QWEN36_DELTANET_SOLVE_MODE",
            )
        },
        "nki_vs_reference": {
            "output": tensor_metrics(torch, out_cpu, ref_out),
            "final_state": tensor_metrics(torch, final_state_cpu, ref_final_state),
            **{
                f"affine_{name}": tensor_metrics(
                    torch,
                    affine_cpu[name],
                    ref_affine[name],
                )
                for name in affine_cpu
            },
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def validate_autocp_state_summary(
    torch: Any,
    xm: Any,
    args: argparse.Namespace,
    inspect_dir: Path,
) -> int:
    if args.multihead:
        raise ValueError("--validate-autocp-state-summary expects single-head inputs")

    import nki.language as nl

    deltanet_autocp_state_summary = load_autocp_state_summary_kernel()

    inputs = make_inputs(torch, args)
    ref_segments = build_compact_autocp_segment_transforms(
        torch,
        inputs,
        cp_chunks=args.autocp_cp_chunks,
    )

    device = xm.xla_device()
    xla_inputs = move_tensor_inputs_to_device(inputs, device)
    num_segments = ref_segments["state_matrix"].shape[0]
    if hasattr(nl, "spmd_dim") and hasattr(nl, "nc"):
        launch_spec = (
            nl.spmd_dim(num_segments, nl.nc(args.lnc)),
            1,
        )
    else:
        launch_spec = args.lnc

    state_matrix_cpu = state_bias_cpu = None
    run_elapsed_seconds = []
    for _ in range(args.runs):
        run_start = time.perf_counter()
        state_matrix_dev, state_bias_dev = deltanet_autocp_state_summary[launch_spec](
            xla_inputs["key"],
            xla_inputs["value"],
            xla_inputs["g_raw"],
            xla_inputs["beta"],
            xla_inputs["lower_mask"],
            xla_inputs["identity"],
        )
        xm.mark_step()
        state_matrix_cpu = state_matrix_dev.detach().cpu().float()
        state_bias_cpu = state_bias_dev.detach().cpu().float()
        run_elapsed_seconds.append(time.perf_counter() - run_start)

    assert state_matrix_cpu is not None
    assert state_bias_cpu is not None

    matrix_close = bool(
        torch.allclose(
            state_matrix_cpu,
            ref_segments["state_matrix"],
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    bias_close = bool(
        torch.allclose(
            state_bias_cpu,
            ref_segments["state_bias"],
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    matrix_finite = bool(torch.isfinite(state_matrix_cpu).all().item())
    bias_finite = bool(torch.isfinite(state_bias_cpu).all().item())
    passed = bool(matrix_close and bias_close and matrix_finite and bias_finite)

    result = {
        "passed": passed,
        "validate_autocp_state_summary": True,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "num_chunks": args.seq_len // P_MAX,
        "num_segments": num_segments,
        "cp_chunks": args.autocp_cp_chunks,
        "runs": args.runs,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cached_run_elapsed_seconds": run_elapsed_seconds[1:],
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "finite": {
            "state_matrix": matrix_finite,
            "state_bias": bias_finite,
        },
        "close": {
            "state_matrix": matrix_close,
            "state_bias": bias_close,
        },
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
                "QWEN36_DELTANET_CHUNK_SIZE",
                "QWEN36_DELTANET_AUTOCP_CP_CHUNKS",
                "QWEN36_DELTANET_SOLVE_BLOCK_SIZE",
                "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
                "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
                "QWEN36_DELTANET_SOLVE_MODE",
            )
        },
        "nki_vs_reference": {
            "state_matrix": tensor_metrics(
                torch,
                state_matrix_cpu,
                ref_segments["state_matrix"],
            ),
            "state_bias": tensor_metrics(
                torch,
                state_bias_cpu,
                ref_segments["state_bias"],
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def validate_autocp_compact_chain(
    torch: Any,
    xm: Any,
    args: argparse.Namespace,
    inspect_dir: Path,
) -> int:
    if args.multihead:
        raise ValueError("--validate-autocp-compact-chain expects single-head inputs")

    import nki.language as nl

    deltanet_autocp_state_summary = load_autocp_state_summary_kernel()
    deltanet_autocp_state_prefix = load_autocp_prefix_kernel()
    deltanet_fused_multihead = load_fused_kernel(True)

    inputs = make_inputs(torch, args)
    ref_out, ref_final_state = reference_math(torch, inputs)

    device = xm.xla_device()
    xla_inputs = move_tensor_inputs_to_device(inputs, device)
    num_chunks = args.seq_len // P_MAX
    num_segments = num_chunks // args.autocp_cp_chunks
    if hasattr(nl, "spmd_dim") and hasattr(nl, "nc"):
        summary_launch_spec = (
            nl.spmd_dim(num_segments, nl.nc(args.lnc)),
            1,
        )
    else:
        summary_launch_spec = args.lnc

    out_cpu = final_state_cpu = None
    run_elapsed_seconds = []
    segment_len = args.autocp_cp_chunks * P_MAX
    replay_group_size = min(num_segments, args.lnc)
    for _ in range(args.runs):
        run_start = time.perf_counter()
        state_matrix_dev, state_bias_dev = deltanet_autocp_state_summary[
            summary_launch_spec
        ](
            xla_inputs["key"],
            xla_inputs["value"],
            xla_inputs["g_raw"],
            xla_inputs["beta"],
            xla_inputs["lower_mask"],
            xla_inputs["identity"],
        )
        segment_states_dev, final_state_dev = deltanet_autocp_state_prefix(
            state_matrix_dev,
            state_bias_dev,
            xla_inputs["state_in"],
        )

        q_segments = xla_inputs["query"].reshape(num_segments, segment_len, P_MAX).contiguous()
        k_segments = xla_inputs["key"].reshape(num_segments, segment_len, P_MAX).contiguous()
        v_segments = xla_inputs["value"].reshape(num_segments, segment_len, P_MAX).contiguous()
        g_segments = xla_inputs["g_raw"].reshape(num_segments, segment_len, 1).contiguous()
        beta_segments = xla_inputs["beta"].reshape(num_segments, segment_len, 1).contiguous()
        replay_outputs = []
        for segment_start in range(0, num_segments, replay_group_size):
            segment_end = min(segment_start + replay_group_size, num_segments)
            launch_segments = segment_end - segment_start
            replay_launch_spec = multihead_launch_spec(launch_segments, args.lnc)
            out_group, _ = deltanet_fused_multihead[replay_launch_spec](
                q_segments[segment_start:segment_end],
                k_segments[segment_start:segment_end],
                v_segments[segment_start:segment_end],
                g_segments[segment_start:segment_end],
                beta_segments[segment_start:segment_end],
                segment_states_dev[segment_start:segment_end],
                xla_inputs["lower_mask"],
                xla_inputs["identity"],
                xla_inputs["lower_mask_diag"],
            )
            replay_outputs.append(out_group)
        out_segments_dev = torch.cat(replay_outputs, dim=0)
        out_dev = out_segments_dev.reshape(args.seq_len, P_MAX)
        xm.mark_step()
        out_cpu = out_dev.detach().cpu().float()
        final_state_cpu = final_state_dev.detach().cpu().float()
        run_elapsed_seconds.append(time.perf_counter() - run_start)

    assert out_cpu is not None
    assert final_state_cpu is not None

    output_close = bool(torch.allclose(out_cpu, ref_out, atol=args.atol, rtol=args.rtol))
    final_state_close = bool(
        torch.allclose(
            final_state_cpu,
            ref_final_state,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    output_finite = bool(torch.isfinite(out_cpu).all().item())
    final_state_finite = bool(torch.isfinite(final_state_cpu).all().item())
    passed = bool(
        output_close
        and final_state_close
        and output_finite
        and final_state_finite
    )

    result = {
        "passed": passed,
        "validate_autocp_compact_chain": True,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "num_chunks": num_chunks,
        "num_segments": num_segments,
        "cp_chunks": args.autocp_cp_chunks,
        "replay_group_size": replay_group_size,
        "runs": args.runs,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cached_run_elapsed_seconds": run_elapsed_seconds[1:],
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "finite": {
            "output": output_finite,
            "final_state": final_state_finite,
        },
        "close": {
            "output": output_close,
            "final_state": final_state_close,
        },
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
                "QWEN36_DELTANET_CHUNK_SIZE",
                "QWEN36_DELTANET_AUTOCP_CP_CHUNKS",
                "QWEN36_DELTANET_SOLVE_BLOCK_SIZE",
                "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
                "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
                "QWEN36_DELTANET_SOLVE_MODE",
            )
        },
        "nki_vs_reference": {
            "output": tensor_metrics(torch, out_cpu, ref_out),
            "final_state": tensor_metrics(
                torch,
                final_state_cpu,
                ref_final_state,
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def validate_compact_autocp_reference(torch: Any, args: argparse.Namespace) -> int:
    inputs = make_inputs(torch, args)
    expected_out, expected_state = reference_math(torch, inputs)
    actual_out, actual_state = compact_autocp_reference_math(
        torch,
        inputs,
        cp_chunks=args.autocp_cp_chunks,
    )

    output_close = bool(
        torch.allclose(actual_out, expected_out, atol=args.atol, rtol=args.rtol)
    )
    state_close = bool(
        torch.allclose(actual_state, expected_state, atol=args.atol, rtol=args.rtol)
    )
    output_finite = bool(torch.isfinite(actual_out).all().item())
    state_finite = bool(torch.isfinite(actual_state).all().item())
    passed = bool(output_close and state_close and output_finite and state_finite)

    result = {
        "passed": passed,
        "validate_compact_autocp_reference": True,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "heads": args.heads if args.multihead else 1,
        "multihead": args.multihead,
        "cp_chunks": args.autocp_cp_chunks,
        "atol": args.atol,
        "rtol": args.rtol,
        "output_finite": output_finite,
        "state_finite": state_finite,
        "materialization": compact_autocp_materialization_counts(
            args.seq_len,
            args.autocp_cp_chunks,
        ),
        "compact_vs_sequential": {
            "output": tensor_metrics(torch, actual_out, expected_out),
            "state": tensor_metrics(torch, actual_state, expected_state),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


def main() -> int:
    args = parse_args()
    inspect_dir = configure_environment(args)
    add_qwen_to_path()

    import torch

    if args.validate_cpu_chunk_invariance:
        return validate_cpu_chunk_invariance(torch, args)
    if args.validate_compact_autocp_reference:
        return validate_compact_autocp_reference(torch, args)

    import torch_xla.core.xla_model as xm

    if args.validate_autocp_affine:
        return validate_autocp_affine_chunk(torch, xm, args, inspect_dir)
    if args.validate_autocp_prefix:
        return validate_autocp_state_prefix(torch, xm, args, inspect_dir)
    if args.validate_autocp_chain:
        return validate_autocp_chain(torch, xm, args, inspect_dir)
    if args.validate_autocp_prefix_apply:
        return validate_autocp_prefix_apply(torch, xm, args, inspect_dir)
    if args.validate_autocp_full:
        return validate_autocp_full(torch, xm, args, inspect_dir)
    if args.validate_autocp_state_summary:
        return validate_autocp_state_summary(torch, xm, args, inspect_dir)
    if args.validate_autocp_compact_chain:
        return validate_autocp_compact_chain(torch, xm, args, inspect_dir)
    if args.validate_restored_suffix_carry:
        return validate_restored_suffix_carry(torch, xm, args, inspect_dir)

    deltanet_fused_chunked_fwd = load_fused_kernel(args.multihead)

    inputs = make_inputs(torch, args)
    ref_out, ref_state = reference_math(torch, inputs)

    device = xm.xla_device()
    xla_inputs = move_tensor_inputs_to_device(inputs, device)

    out_cpu = state_cpu = None
    run_elapsed_seconds = []
    launch_spec_labels = []
    for _ in range(args.runs):
        run_start = time.perf_counter()
        if args.multihead:
            pair_outputs = []
            pair_states = []
            head_group_size = min(args.head_group_size, args.heads)
            for head_start in range(0, args.heads, head_group_size):
                head_end = min(head_start + head_group_size, args.heads)
                launch_heads = head_end - head_start
                launch_spec = multihead_launch_spec(launch_heads, args.lnc)
                if len(launch_spec_labels) < math.ceil(args.heads / head_group_size):
                    launch_spec_labels.append(launch_spec_label(launch_spec))
                out_pair, state_pair = deltanet_fused_chunked_fwd[launch_spec](
                    xla_inputs["query"][head_start:head_end],
                    xla_inputs["key"][head_start:head_end],
                    xla_inputs["value"][head_start:head_end],
                    xla_inputs["g_raw"][head_start:head_end],
                    xla_inputs["beta"][head_start:head_end],
                    xla_inputs["state_in"][head_start:head_end],
                    xla_inputs["lower_mask"],
                    xla_inputs["identity"],
                    xla_inputs["lower_mask_diag"],
                )
                pair_outputs.append(out_pair)
                pair_states.append(state_pair)
            out_dev = torch.cat(pair_outputs, dim=0)
            state_dev = torch.cat(pair_states, dim=0)
        else:
            out_dev, state_dev = deltanet_fused_chunked_fwd(
                xla_inputs["query"],
                xla_inputs["key"],
                xla_inputs["value"],
                xla_inputs["g_raw"],
                xla_inputs["beta"],
                xla_inputs["state_in"],
                xla_inputs["lower_mask"],
                xla_inputs["identity"],
                xla_inputs["lower_mask_diag"],
            )
        xm.mark_step()
        out_cpu = out_dev.detach().cpu().float()
        state_cpu = state_dev.detach().cpu().float()
        run_elapsed_seconds.append(time.perf_counter() - run_start)

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
        "seq_len": args.seq_len,
        "chunk_size": args.chunk_size,
        "heads": args.heads if args.multihead else 1,
        "head_group_size": args.head_group_size if args.multihead else 1,
        "launch_specs": launch_spec_labels if args.multihead else [],
        "multihead": args.multihead,
        "runs": args.runs,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cached_run_elapsed_seconds": run_elapsed_seconds[1:],
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
                "QWEN36_DELTANET_CHUNK_SIZE",
                "QWEN36_DELTANET_SOLVE_BLOCK_SIZE",
                "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
                "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
                "QWEN36_DELTANET_SOLVE_MODE",
            )
        },
        "nki_vs_reference": {
            "output": tensor_metrics(torch, out_cpu, ref_out),
            "state": tensor_metrics(torch, state_cpu, ref_state),
        },
    }
    if args.multihead:
        result["nki_vs_reference"]["output_per_head"] = multihead_tensor_metrics(
            torch,
            out_cpu,
            ref_out,
        )
        result["nki_vs_reference"]["state_per_head"] = multihead_tensor_metrics(
            torch,
            state_cpu,
            ref_state,
        )
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
