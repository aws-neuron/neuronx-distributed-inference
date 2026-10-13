#!/usr/bin/env python3
"""Offline context-length sweep for Qwen3.6 Hybrid APC artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import qwen36_hybrid_apc_validation as hybrid_validation


def _parse_lengths(raw: str) -> list[int]:
    lengths = [int(item) for item in raw.replace(",", " ").split() if item]
    if not lengths:
        raise ValueError("at least one length is required")
    return lengths


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
    seq_len: int,
    max_num_seqs: int,
) -> int:
    """Return vLLM's user-intended block count, excluding its null block."""

    block_size = int(args.block_size)
    min_usable_blocks = ((seq_len + block_size - 1) // block_size) * max_num_seqs
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


def _single_token_pool(tokenizer) -> list[int]:
    return hybrid_validation._compact_single_token_ids(tokenizer)


def _role_token_ids(tokenizer, *, role_index: int, token_count: int) -> list[int]:
    if token_count <= 0:
        return []
    pool = _single_token_pool(tokenizer)
    return [pool[(role_index + (position * 7)) % len(pool)] for position in range(token_count)]


def _prompt_parts_for_length(
    tokenizer,
    *,
    target_tokens: int,
    suffix_tokens: int,
    prefix_role_index: int,
    suffix_role_index: int,
) -> tuple[list[int], list[int]]:
    if target_tokens <= suffix_tokens:
        raise ValueError(
            f"target length {target_tokens} must be larger than suffix length {suffix_tokens}"
        )
    prefix = _role_token_ids(
        tokenizer,
        role_index=prefix_role_index,
        token_count=target_tokens - suffix_tokens,
    )
    suffix = _role_token_ids(
        tokenizer,
        role_index=suffix_role_index,
        token_count=suffix_tokens,
    )
    return prefix, suffix


def _prompt_for_length(
    tokenizer,
    *,
    target_tokens: int,
    suffix_tokens: int,
    role_index: int,
) -> dict[str, list[int]]:
    prefix, suffix = _prompt_parts_for_length(
        tokenizer,
        target_tokens=target_tokens,
        suffix_tokens=suffix_tokens,
        prefix_role_index=role_index,
        suffix_role_index=role_index + 997,
    )
    return {"prompt_token_ids": prefix + suffix}


def _partial_refill_prompts(
    tokenizer,
    *,
    target_tokens: int,
    suffix_tokens: int,
    role_index: int,
) -> tuple[
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, list[int]],
    int,
]:
    shared_prefix, warmup_suffix = _prompt_parts_for_length(
        tokenizer,
        target_tokens=target_tokens,
        suffix_tokens=suffix_tokens,
        prefix_role_index=role_index,
        suffix_role_index=role_index + 997,
    )
    cold_prefix, measured_suffix = _prompt_parts_for_length(
        tokenizer,
        target_tokens=target_tokens,
        suffix_tokens=suffix_tokens,
        prefix_role_index=role_index + 2003,
        suffix_role_index=role_index + 3001,
    )
    warm_prompt = {"prompt_token_ids": shared_prefix + measured_suffix}
    warmup_prompt = {"prompt_token_ids": shared_prefix + warmup_suffix}
    cold_prompt = {"prompt_token_ids": cold_prefix + measured_suffix}
    return cold_prompt, warmup_prompt, warm_prompt, len(shared_prefix)


def _generate(llm: Any, sampling: Any, prompt: dict[str, list[int]]) -> dict[str, Any]:
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - start
    output = outputs[0].outputs[0]
    tokens = [int(token_id) for token_id in output.token_ids]
    return {
        "elapsed_seconds": elapsed,
        "generated_token_count": len(tokens),
        "generated_tokens": tokens,
        "generated_text": output.text,
    }


def _effective_vocab_size(model_path: Path, tokenizer: Any) -> int:
    sizes = [
        int(size)
        for size in (
            getattr(tokenizer, "vocab_size", None),
            len(tokenizer),
        )
        if size
    ]
    try:
        from transformers import AutoConfig  # noqa: WPS433

        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        config_vocab_size = getattr(config, "vocab_size", None)
        if config_vocab_size:
            sizes.append(int(config_vocab_size))
    except Exception:
        pass
    if not sizes:
        raise ValueError("could not determine model/tokenizer vocabulary size")
    return max(sizes)


def _build_args(args: argparse.Namespace, artifact_config: dict[str, Any]) -> SimpleNamespace:
    seq_len = int(
        args.seq_len
        or artifact_config.get("seq_len")
        or artifact_config.get("max_context_length")
        or artifact_config.get("max_length")
        or 131072
    )
    cte_buckets = args.cte_buckets or ",".join(
        str(item) for item in artifact_config.get("context_encoding_buckets", [])
    )
    if not cte_buckets:
        cte_buckets = "256,512"
    token_generation_buckets = args.token_generation_buckets
    if token_generation_buckets is None:
        artifact_tkg_buckets = artifact_config.get("token_generation_buckets") or []
        if artifact_tkg_buckets:
            token_generation_buckets = [
                ",".join(str(item) for item in artifact_tkg_buckets)
            ]
    token_generation_batches = args.token_generation_batches
    if token_generation_batches is None:
        artifact_tkg_batches = artifact_config.get("token_generation_batches") or []
        if artifact_tkg_batches:
            token_generation_batches = [
                ",".join(str(item) for item in artifact_tkg_batches)
            ]
    context_encoding_bucket_pairs = args.context_encoding_bucket_pairs
    if context_encoding_bucket_pairs is None:
        artifact_pairs = artifact_config.get("context_encoding_bucket_pairs") or []
        if artifact_pairs:
            context_encoding_bucket_pairs = [
                f"{int(active)}:{int(prefix)}"
                for active, prefix in artifact_pairs
            ]
    async_mode = (
        bool(args.async_mode)
        if args.async_mode is not None
        else bool(artifact_config.get("async_mode", False))
    )
    ctx_batch_size = int(
        args.ctx_batch_size
        if args.ctx_batch_size is not None
        else artifact_config.get("ctx_batch_size") or 1
    )
    max_num_seqs = int(args.max_num_seqs or 1)
    pa_num_blocks = _runtime_pa_override(
        args,
        artifact_config,
        seq_len=seq_len,
        max_num_seqs=max_num_seqs,
    )
    return SimpleNamespace(
        model_path=str(args.model_path),
        compiled_artifacts=str(args.compiled_artifacts),
        skip_fp8_env=args.skip_fp8_env,
        max_model_len=int(args.max_model_len or seq_len),
        seq_len=seq_len,
        cte_bucket=max(hybrid_validation._parse_bucket_values([cte_buckets])),
        cte_buckets=[cte_buckets],
        context_encoding_bucket_pairs=context_encoding_bucket_pairs,
        cte_bucket_profile="single",
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        logical_nc_config=args.logical_nc_config,
        ctx_batch_size=ctx_batch_size,
        token_generation_buckets=token_generation_buckets,
        token_generation_batches=token_generation_batches,
        async_mode=async_mode,
        block_size=args.block_size,
        gdn_checkpoint_interval=args.gdn_checkpoint_interval,
        max_gdn_checkpoint_slots=args.max_gdn_checkpoint_slots,
        gdn_recurrent_cache_dtype=args.gdn_recurrent_cache_dtype,
        gdn_conv_cache_dtype=args.gdn_conv_cache_dtype,
        hybrid_apc_require_vllm_metadata=True,
        hybrid_apc_reject_unbacked_attention_hits=True,
        hybrid_apc_disable_unbacked_prefix_reads=False,
        hybrid_apc_enable_backed_prefix_reads=True,
        hybrid_apc_prefill_chunk_tokens=args.hybrid_apc_prefill_chunk_tokens,
        hybrid_apc_max_backed_prefix_read_len=0,
        enable_vllm_chunked_prefill=True,
        kernel_q_tile_size=args.kernel_q_tile_size,
        kernel_kv_tile_size=args.kernel_kv_tile_size,
        num_gpu_blocks_override=pa_num_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens,
        dummy_token_ids=args.dummy_token_ids,
        require_real_tokens=args.require_real_tokens,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compiled-artifacts", type=Path, required=True)
    parser.add_argument("--lengths", default="16384,32768,65536,131000")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument(
        "--warm-mode",
        choices=("partial", "exact"),
        default="partial",
        help=(
            "partial warms a shared prefix with one suffix, then measures the "
            "same prefix with a different suffix; exact repeats the full prompt."
        ),
    )
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--cte-buckets")
    parser.add_argument("--context-encoding-bucket-pairs", nargs="+", default=None)
    parser.add_argument("--pa-num-blocks", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--ctx-batch-size", type=int)
    parser.add_argument("--token-generation-buckets", nargs="+", default=None)
    parser.add_argument("--token-generation-batches", nargs="+", default=None)
    parser.add_argument("--async-mode", action="store_true", default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gdn-checkpoint-interval", type=int, default=256)
    parser.add_argument("--max-gdn-checkpoint-slots", type=int, default=64)
    parser.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    parser.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    parser.add_argument("--hybrid-apc-prefill-chunk-tokens", type=int, default=0)
    parser.add_argument("--kernel-q-tile-size", type=int, default=128)
    parser.add_argument("--kernel-kv-tile-size", type=int, default=1024)
    parser.add_argument("--skip-fp8-env", action="store_true")
    parser.add_argument("--require-real-tokens", action="store_true")
    parser.add_argument("--dummy-token-ids", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    args.model_path = args.model_path.expanduser().resolve()
    args.compiled_artifacts = args.compiled_artifacts.expanduser().resolve()
    artifact_config = _artifact_neuron_config(args.compiled_artifacts)
    runtime_args = _build_args(args, artifact_config)

    from transformers import AutoTokenizer  # noqa: WPS433
    from vllm import SamplingParams  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_tokens)
    configured_dummy_ids = {int(token_id) for token_id in args.dummy_token_ids}
    dummy_ids = configured_dummy_ids | hybrid_validation._effective_dummy_token_ids(
        runtime_args,
        tokenizer,
    )
    vocab_size = _effective_vocab_size(args.model_path, tokenizer)
    llm = None
    rows: list[dict[str, Any]] = []
    try:
        llm, _unused_sampling = hybrid_validation._build_llm(
            runtime_args,
            enable_hybrid_apc=True,
        )
        for index, target_tokens in enumerate(_parse_lengths(args.lengths)):
            if target_tokens + args.max_tokens > runtime_args.seq_len:
                raise ValueError(
                    f"target_tokens + max_tokens exceeds seq_len: "
                    f"{target_tokens} + {args.max_tokens} > {runtime_args.seq_len}"
                )
            role_index = index * 1009
            if args.warm_mode == "exact":
                prompt = _prompt_for_length(
                    tokenizer,
                    target_tokens=target_tokens,
                    suffix_tokens=args.suffix_tokens,
                    role_index=role_index,
                )
                cold = _generate(llm, sampling, prompt)
                prefix_warmup = None
                warm = _generate(llm, sampling, prompt)
                actual_prompt_tokens = len(prompt["prompt_token_ids"])
                shared_prefix_tokens = actual_prompt_tokens
            else:
                (
                    cold_prompt,
                    warmup_prompt,
                    warm_prompt,
                    shared_prefix_tokens,
                ) = _partial_refill_prompts(
                    tokenizer,
                    target_tokens=target_tokens,
                    suffix_tokens=args.suffix_tokens,
                    role_index=role_index,
                )
                cold = _generate(llm, sampling, cold_prompt)
                prefix_warmup = _generate(llm, sampling, warmup_prompt)
                warm = _generate(llm, sampling, warm_prompt)
                actual_prompt_tokens = len(warm_prompt["prompt_token_ids"])
            generated_tokens = [
                token
                for result in (cold, prefix_warmup, warm)
                if result is not None
                for token in result["generated_tokens"]
            ]
            non_dummy = [
                token
                for token in generated_tokens
                if token not in dummy_ids
            ]
            invalid_token_ids = [
                token
                for token in generated_tokens
                if token < 0 or token >= vocab_size
            ]
            unique_generated_tokens = sorted(set(generated_tokens))
            row = {
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": actual_prompt_tokens,
                "warm_mode": args.warm_mode,
                "shared_prefix_tokens": shared_prefix_tokens,
                "suffix_tokens": args.suffix_tokens,
                "max_tokens": args.max_tokens,
                "cold": cold,
                "prefix_warmup": prefix_warmup,
                "warm": warm,
                "repeat_exact": cold["generated_tokens"] == warm["generated_tokens"],
                "real_tokens_passed": bool(non_dummy),
                "non_dummy_generated_token_count": len(non_dummy),
                "all_generated_tokens_dummy": bool(generated_tokens)
                and all(token in dummy_ids for token in generated_tokens),
                "unique_generated_token_count": len(unique_generated_tokens),
                "unique_generated_tokens": unique_generated_tokens,
                "configured_dummy_token_ids": sorted(configured_dummy_ids),
                "effective_dummy_token_ids": sorted(dummy_ids),
                "token_range_passed": not invalid_token_ids,
                "invalid_token_ids": sorted(set(invalid_token_ids)),
                "vocab_size": vocab_size,
                "cold_effective_prompt_tokens_per_second": target_tokens
                / cold["elapsed_seconds"]
                if cold["elapsed_seconds"] > 0
                else None,
                "warm_effective_prompt_tokens_per_second": target_tokens
                / warm["elapsed_seconds"]
                if warm["elapsed_seconds"] > 0
                else None,
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
    finally:
        if llm is not None:
            hybrid_validation._shutdown_llm(llm)

    report = {
        "artifact": str(args.compiled_artifacts),
        "artifact_neuron_config": {
            key: artifact_config.get(key)
            for key in (
                "seq_len",
                "max_context_length",
                "context_encoding_buckets",
                "prefix_buckets",
                "token_generation_buckets",
                "ctx_batch_size",
                "tkg_batch_size",
                "pa_block_size",
                "pa_num_blocks",
                "output_logits",
                "on_device_sampling_config",
            )
        },
        "lengths": _parse_lengths(args.lengths),
        "warm_mode": args.warm_mode,
        "rows": rows,
        "passed": all(
            (args.warm_mode != "exact" or row["repeat_exact"])
            and row["real_tokens_passed"]
            and row["token_range_passed"]
            for row in rows
        ),
    }
    args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
