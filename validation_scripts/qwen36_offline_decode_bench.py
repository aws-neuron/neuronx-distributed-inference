#!/usr/bin/env python3
"""Offline vLLM decode benchmark for Qwen3.6 Neuron artifacts.

This intentionally bypasses the OpenAI HTTP server while keeping the same
vLLM/NxDI model runner and compiled artifact path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


FP8_ENV_DEFAULTS = {
    "XLA_HANDLE_SPECIAL_SCALAR": "1",
    "UNSAFE_FP8FNCAST": "1",
}


def _parse_buckets(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(",", " ").split() if item]


def _parse_bucket_pairs(raw: str | None) -> list[list[int]] | None:
    if not raw:
        return None
    pairs: set[tuple[int, int]] = set()
    for token in raw.replace(",", " ").split():
        if ":" in token:
            active, prefix = token.split(":", 1)
        elif "x" in token:
            active, prefix = token.split("x", 1)
        else:
            raise ValueError(
                "context-encoding bucket pairs must use ACTIVE:PREFIX syntax, "
                f"got {token!r}"
            )
        active_tokens, prefix_tokens = int(active), int(prefix)
        if active_tokens <= 0 or prefix_tokens < 0:
            raise ValueError(
                "context-encoding bucket pairs must use positive active tokens "
                f"and non-negative prefix tokens, got {token!r}"
            )
        pairs.add((active_tokens, prefix_tokens))
    return [[active, prefix] for active, prefix in sorted(pairs)]


def _validated_int_list(
    values: list[int],
    *,
    name: str,
    maximum: int | None = None,
) -> list[int]:
    values = sorted(set(int(item) for item in values))
    if not values:
        raise ValueError(f"{name} cannot be empty")
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} value {value} exceeds {maximum}")
    return values


def _artifact_neuron_config(compiled_artifacts: Path) -> dict[str, Any]:
    config_path = compiled_artifacts / "neuron_config.json"
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    nested = config.get("neuron_config")
    return nested if isinstance(nested, dict) else config


def _runtime_pa_override(
    args: argparse.Namespace,
    artifact_config: dict[str, Any],
    *,
    max_model_len: int,
) -> int:
    """Return vLLM's user-intended block count, excluding its null block."""

    block_size = int(args.block_size)
    max_num_seqs = int(args.max_num_seqs or 1)
    min_usable_blocks = ((max_model_len + block_size - 1) // block_size) * max_num_seqs
    if args.pa_num_blocks is not None:
        return max(1, int(args.pa_num_blocks))

    artifact_blocks = int(artifact_config.get("pa_num_blocks") or 0)
    if artifact_blocks <= 0:
        return max(1, min_usable_blocks)

    uses_block_kv = bool(
        artifact_config.get("is_block_kv_layout")
        or artifact_config.get("is_prefix_caching")
    )
    if uses_block_kv and artifact_blocks > min_usable_blocks:
        return artifact_blocks - 1
    return artifact_blocks


def _resolve_config_defaults(args: argparse.Namespace) -> dict[str, Any]:
    artifact_config = _artifact_neuron_config(args.compiled_artifacts)
    seq_len = int(
        args.seq_len
        or artifact_config.get("seq_len")
        or artifact_config.get("max_context_length")
        or artifact_config.get("max_length")
        or 131072
    )
    max_model_len = int(args.max_model_len or seq_len)
    cte_buckets = (
        _parse_buckets(args.cte_buckets)
        if args.cte_buckets
        else [int(item) for item in artifact_config.get("context_encoding_buckets", [])]
    )
    if not cte_buckets:
        cte_buckets = [256, 512]
    cte_buckets = _validated_int_list(
        cte_buckets,
        name="context encoding buckets",
        maximum=seq_len,
    )
    token_generation_buckets = (
        _parse_buckets(args.token_generation_buckets)
        if args.token_generation_buckets
        else [
            int(item)
            for item in artifact_config.get("token_generation_buckets", [])
        ]
    )
    if not token_generation_buckets:
        token_generation_buckets = [seq_len]
    token_generation_buckets = _validated_int_list(
        token_generation_buckets,
        name="token generation buckets",
        maximum=seq_len,
    )
    token_generation_batches = (
        _parse_buckets(args.token_generation_batches)
        if args.token_generation_batches
        else None
    )
    if token_generation_batches is not None:
        token_generation_batches = _validated_int_list(
            token_generation_batches,
            name="token generation batches",
            maximum=args.max_num_seqs,
        )
    pa_num_blocks = _runtime_pa_override(
        args,
        artifact_config,
        max_model_len=max_model_len,
    )
    context_encoding_bucket_pairs = _parse_bucket_pairs(
        args.context_encoding_bucket_pairs
    )
    if context_encoding_bucket_pairs is None:
        artifact_pairs = artifact_config.get("context_encoding_bucket_pairs") or []
        if artifact_pairs:
            context_encoding_bucket_pairs = [
                [int(active), int(prefix)]
                for active, prefix in artifact_pairs
            ]
    return {
        "artifact_config": artifact_config,
        "seq_len": seq_len,
        "max_model_len": max_model_len,
        "cte_buckets": cte_buckets,
        "context_encoding_bucket_pairs": context_encoding_bucket_pairs,
        "token_generation_buckets": token_generation_buckets,
        "token_generation_batches": token_generation_batches,
        "pa_num_blocks": pa_num_blocks,
    }


def _ensure_paths(repo_root: Path) -> Path:
    qwen_root = repo_root / "contrib" / "models" / "Qwen3.6-27B"
    for path in (repo_root / "src", qwen_root / "vllm", qwen_root):
        sys.path.insert(0, str(path))
    os.environ["PYTHONPATH"] = (
        f"{repo_root / 'src'}:{qwen_root / 'vllm'}:{qwen_root}:"
        f"{os.environ.get('PYTHONPATH', '')}"
    )
    return qwen_root


def _ensure_runtime_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    os.environ.setdefault("VLLM_PLUGINS", "neuron")
    os.environ.setdefault("QWEN36_HYBRID_APC_INSTALL_PATCH", "1")
    os.environ.setdefault("DISABLE_NEURON_CUSTOM_SCHEDULER", "1")
    for name, value in FP8_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)
    os.environ["NEURON_COMPILED_ARTIFACTS"] = str(args.compiled_artifacts)


def _additional_config(args: argparse.Namespace) -> dict[str, Any]:
    override_neuron_config = {
        "tp_degree": args.tensor_parallel_size,
        "batch_size": args.max_num_seqs,
        "ctx_batch_size": args.ctx_batch_size,
        "tkg_batch_size": args.max_num_seqs,
        "seq_len": args.seq_len,
        "max_length": args.seq_len,
        "max_context_length": args.seq_len,
        "context_encoding_buckets": args.resolved_cte_buckets,
        "token_generation_buckets": args.resolved_token_generation_buckets,
        "enable_bucketing": len(args.resolved_cte_buckets) > 1
        or len(args.resolved_token_generation_buckets) > 1,
        "logical_nc_config": args.logical_nc_config,
        "torch_dtype": "bfloat16",
        "save_sharded_checkpoint": True,
        "pa_block_size": args.block_size,
        "pa_num_blocks": args.pa_num_blocks,
        "gdn_checkpoint_interval": args.gdn_checkpoint_interval,
        "max_gdn_checkpoint_slots": args.max_gdn_checkpoint_slots,
        "gdn_recurrent_cache_dtype": args.gdn_recurrent_cache_dtype,
        "gdn_conv_cache_dtype": args.gdn_conv_cache_dtype,
        "hybrid_recurrent_cache_dtype": args.gdn_recurrent_cache_dtype,
        "hybrid_conv_cache_dtype": args.gdn_conv_cache_dtype,
        "hybrid_cache_mode": "all",
        "is_block_kv_layout": True,
        "is_prefix_caching": True,
        "chunked_prefill_config": {
            "max_num_seqs": args.max_num_seqs,
            "tkg_model_enabled": True,
            "kernel_q_tile_size": args.kernel_q_tile_size,
            "kernel_kv_tile_size": args.kernel_kv_tile_size,
        },
    }
    if args.async_mode:
        override_neuron_config["async_mode"] = True
    if args.resolved_context_encoding_bucket_pairs is not None:
        override_neuron_config["context_encoding_bucket_pairs"] = (
            args.resolved_context_encoding_bucket_pairs
        )
    if args.resolved_token_generation_batches is not None:
        override_neuron_config["token_generation_batches"] = (
            args.resolved_token_generation_batches
        )

    return {
        "max_prompt_length": args.seq_len,
        "use_hybrid_apc_manager": True,
        "use_text_only_cte_inputs": True,
        "use_compact_cte_attention_mask": True,
        "use_cold_zero_conv_fast_path": False,
        "gdn_checkpoint_interval": args.gdn_checkpoint_interval,
        "max_gdn_checkpoint_slots": args.max_gdn_checkpoint_slots,
        "gdn_recurrent_cache_dtype": args.gdn_recurrent_cache_dtype,
        "gdn_conv_cache_dtype": args.gdn_conv_cache_dtype,
        "hybrid_recurrent_cache_dtype": args.gdn_recurrent_cache_dtype,
        "hybrid_conv_cache_dtype": args.gdn_conv_cache_dtype,
        "hybrid_cache_mode": "all",
        "hybrid_cache_prefix_boundary_only": True,
        "hybrid_cache_block_boundary_only": True,
        "hybrid_cache_validate_exact": False,
        "hybrid_apc_require_vllm_metadata": True,
        "hybrid_apc_allow_local_hash_fallback": False,
        "hybrid_apc_require_attention_block_refs": True,
        "hybrid_apc_disable_unbacked_prefix_reads": False,
        "hybrid_apc_enable_backed_prefix_reads": True,
        "use_qwen_hybrid_chunked_prefill": True,
        "use_qwen_hybrid_chunked_prefill_nki": True,
        "override_neuron_config": override_neuron_config,
    }


def _build_llm(args: argparse.Namespace):
    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433
    from qwen36_hybrid_apc_scheduler_patch import (  # noqa: WPS433
        install_import_hook as install_hybrid_apc_scheduler_patch,
    )

    register_qwen35_config()
    install_hybrid_apc_scheduler_patch()

    from vllm import LLM, SamplingParams  # noqa: WPS433

    recurrent_cache_dtype = str(args.gdn_recurrent_cache_dtype).lower()
    if recurrent_cache_dtype in {"bfloat16", "bf16"}:
        recurrent_cache_dtype = "auto"

    llm = LLM(
        model=str(args.model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        additional_config=_additional_config(args),
        block_size=args.block_size,
        num_gpu_blocks_override=args.pa_num_blocks,
        mamba_cache_mode="all",
        mamba_ssm_cache_dtype=recurrent_cache_dtype,
        max_num_batched_tokens=max(args.resolved_cte_buckets),
        max_num_seqs=args.max_num_seqs,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_k=1,
        max_tokens=args.max_tokens,
    )
    warmup_sampling = SamplingParams(
        temperature=0.0,
        top_k=1,
        max_tokens=args.warmup_tokens,
    )
    return llm, sampling, warmup_sampling


def _run_generate(llm: Any, prompt: str, sampling: Any) -> dict[str, Any]:
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - start
    output = outputs[0].outputs[0]
    token_ids = list(output.token_ids)
    return {
        "elapsed_s": elapsed,
        "completion_tokens": len(token_ids),
        "tok_s": (len(token_ids) / elapsed) if elapsed > 0 else None,
        "text": output.text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compiled-artifacts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--prompt", default="Explain software benchmarking in two concise paragraphs.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--cte-buckets")
    parser.add_argument("--context-encoding-bucket-pairs")
    parser.add_argument("--token-generation-buckets")
    parser.add_argument("--token-generation-batches")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--ctx-batch-size", type=int, default=1)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--pa-num-blocks", type=int)
    parser.add_argument("--gdn-checkpoint-interval", type=int, default=256)
    parser.add_argument("--max-gdn-checkpoint-slots", type=int, default=64)
    parser.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    parser.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    parser.add_argument("--kernel-q-tile-size", type=int, default=128)
    parser.add_argument("--kernel-kv-tile-size", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.compiled_artifacts = args.compiled_artifacts.expanduser().resolve()
    resolved = _resolve_config_defaults(args)
    args.artifact_config = resolved["artifact_config"]
    args.seq_len = resolved["seq_len"]
    args.max_model_len = resolved["max_model_len"]
    args.resolved_cte_buckets = resolved["cte_buckets"]
    args.resolved_context_encoding_bucket_pairs = resolved[
        "context_encoding_bucket_pairs"
    ]
    args.resolved_token_generation_buckets = resolved["token_generation_buckets"]
    args.resolved_token_generation_batches = resolved["token_generation_batches"]
    args.pa_num_blocks = resolved["pa_num_blocks"]
    _ensure_paths(args.repo_root)
    _ensure_runtime_env(args)

    llm = None
    report: dict[str, Any] = {
        "artifact": str(args.compiled_artifacts),
        "model_path": str(args.model_path),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "max_num_seqs": args.max_num_seqs,
        "pa_num_blocks": args.pa_num_blocks,
        "cte_buckets": args.resolved_cte_buckets,
        "token_generation_buckets": args.resolved_token_generation_buckets,
        "context_encoding_bucket_pairs": args.resolved_context_encoding_bucket_pairs,
        "token_generation_batches": args.resolved_token_generation_batches,
        "async_mode": args.async_mode,
        "max_model_len": args.max_model_len,
        "seq_len": args.seq_len,
        "artifact_neuron_config": {
            key: args.artifact_config.get(key)
            for key in (
                "seq_len",
                "max_length",
                "max_context_length",
                "context_encoding_buckets",
                "prefix_buckets",
                "token_generation_buckets",
                "tkg_batch_size",
                "ctx_batch_size",
                "pa_block_size",
                "pa_num_blocks",
                "output_logits",
                "on_device_sampling_config",
            )
        },
    }
    try:
        llm, sampling, warmup_sampling = _build_llm(args)
        report["warmup"] = _run_generate(llm, args.prompt, warmup_sampling)
        rows = []
        for index in range(args.repeats):
            row = _run_generate(llm, args.prompt, sampling)
            row["run"] = index + 1
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        report["runs"] = rows
        report["avg_tok_s"] = sum(float(row["tok_s"]) for row in rows) / len(rows)
        report["avg_elapsed_s"] = (
            sum(float(row["elapsed_s"]) for row in rows) / len(rows)
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if args.output_json:
            args.output_json.expanduser().write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        if llm is not None:
            shutdown = getattr(llm, "shutdown", None)
            if shutdown is not None:
                shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
