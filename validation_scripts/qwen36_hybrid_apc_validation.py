#!/usr/bin/env python3
"""Trainium validation harness for Qwen3.6 hybrid APC.

This script is intentionally separate from unit tests because it expects a
Neuron/vLLM runtime and compiled artifacts. It covers two gates:

* token exactness for cold vs warm full-prefix and partial-prefix reuse;
* HBM planning for GDN checkpoint slot budgets.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import multiprocessing
import os
import queue
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_ROOT = REPO_ROOT / "contrib" / "models" / "Qwen3.6-27B"
RUNNER_PATH = QWEN_ROOT / "vllm" / "run_offline_inference.py"
HYBRID_APC_PATH = QWEN_ROOT / "src" / "hybrid_apc.py"
FP8_ENV_DEFAULTS = {
    "XLA_HANDLE_SPECIAL_SCALAR": "1",
    "UNSAFE_FP8FNCAST": "1",
}
COMPACT_SINGLE_TOKEN_PIECES = [
    " one",
    " two",
    " three",
    " four",
    " five",
    " six",
    " seven",
    " eight",
    " nine",
    " ten",
    " alpha",
    " beta",
    " gamma",
    " delta",
    " token",
]


def _ensure_fp8_environment() -> None:
    for name, value in FP8_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _compiled_neuron_config(args) -> dict:
    if not args.compiled_artifacts:
        return {}
    config_path = Path(args.compiled_artifacts).expanduser() / "neuron_config.json"
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    nested = config.get("neuron_config")
    return nested if isinstance(nested, dict) else config


def _align_additional_config_to_compiled_artifact(
    args,
    additional_config: dict,
) -> dict:
    """Keep runtime additional_config compatible with precompiled artifacts.

    The Qwen chunked-prefill runner uses CTE buckets for active prefill chunk
    shapes, but vLLM-Neuron validates the top-level max_prompt_length against
    the artifact's compiled max_context_length when loading precompiled NEFFs.
    """

    compiled_config = _compiled_neuron_config(args)
    if not compiled_config:
        return additional_config
    compiled_max_prompt = int(
        compiled_config.get("max_context_length")
        or compiled_config.get("max_length")
        or compiled_config.get("seq_len")
        or 0
    )
    if compiled_max_prompt <= 0:
        return additional_config

    aligned = dict(additional_config)
    aligned["max_prompt_length"] = compiled_max_prompt
    override = dict(aligned.get("override_neuron_config") or {})
    override["max_context_length"] = compiled_max_prompt
    if (
        "context_encoding_bucket_pairs" not in override
        and compiled_config.get("context_encoding_bucket_pairs") is not None
    ):
        override["context_encoding_bucket_pairs"] = compiled_config[
            "context_encoding_bucket_pairs"
        ]
    aligned["override_neuron_config"] = override
    return aligned


def _validate_generation_batch_support(args) -> None:
    if args.max_tokens <= 0 or args.max_num_seqs <= 1:
        return
    neuron_config = _compiled_neuron_config(args)
    if not neuron_config:
        return
    tkg_batch_size = int(
        neuron_config.get("tkg_batch_size")
        or neuron_config.get("batch_size")
        or neuron_config.get("max_batch_size")
        or 1
    )
    if args.max_num_seqs > tkg_batch_size:
        raise ValueError(
            "batched generation requires a compiled artifact with "
            f"tkg_batch_size >= --max-num-seqs; got tkg_batch_size={tkg_batch_size} "
            f"and max_num_seqs={args.max_num_seqs}"
        )
    ctx_batch_size = int(
        neuron_config.get("ctx_batch_size")
        or neuron_config.get("batch_size")
        or neuron_config.get("max_batch_size")
        or 1
    )
    if args.max_num_seqs > ctx_batch_size:
        raise ValueError(
            "batched generation requires a compiled artifact with "
            f"ctx_batch_size >= --max-num-seqs for grouped prefill host logits; "
            f"got ctx_batch_size={ctx_batch_size} and max_num_seqs={args.max_num_seqs}"
        )


def _parse_bucket_values(values) -> list[int]:
    if isinstance(values, str):
        values = [values]
    buckets = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                buckets.append(int(part))
    return sorted(set(buckets))


def _next_bucket(token_count: int, buckets: list[int]) -> int:
    for bucket in buckets:
        if token_count <= bucket:
            return bucket
    raise ValueError(
        f"prompt token length {token_count} exceeds compiled CTE buckets {buckets}"
    )


def _padding_token_id(tokenizer) -> int:
    for token_id in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if token_id is not None:
            return int(token_id)
    raise ValueError("tokenizer must define a pad_token_id or eos_token_id")


def _maybe_bucket_align_labeled_prompts(args, labeled_prompts):
    if not getattr(args, "align_prompts_to_cte_buckets", False):
        return labeled_prompts

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        trust_remote_code=True,
    )
    buckets = _parse_bucket_values(args.cte_buckets)
    pad_token_id = _padding_token_id(tokenizer)
    aligned = []
    for label, prompt in labeled_prompts:
        if isinstance(prompt, dict):
            prompt_token_ids = list(prompt.get("prompt_token_ids", []))
        else:
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        bucket = _next_bucket(len(prompt_token_ids), buckets)
        aligned.append(
            (
                label,
                {
                    "prompt_token_ids": prompt_token_ids
                    + [pad_token_id] * (bucket - len(prompt_token_ids)),
                },
            )
        )
    return aligned


def _runner_args(args, *, enable_hybrid_apc: bool):
    return SimpleNamespace(
        cte_bucket=args.cte_bucket,
        cte_buckets=args.cte_buckets,
        context_encoding_bucket_pairs=getattr(
            args, "context_encoding_bucket_pairs", None
        ),
        cte_bucket_profile=args.cte_bucket_profile,
        seq_len=args.seq_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        ctx_batch_size=args.ctx_batch_size,
        logical_nc_config=args.logical_nc_config,
        block_size=args.block_size,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        enable_prefix_caching=enable_hybrid_apc,
        enable_hybrid_apc=enable_hybrid_apc,
        enable_vllm_chunked_prefill=args.enable_vllm_chunked_prefill,
        token_generation_buckets=getattr(args, "token_generation_buckets", None),
        token_generation_batches=getattr(args, "token_generation_batches", None),
        async_mode=getattr(args, "async_mode", False),
        kernel_q_tile_size=args.kernel_q_tile_size,
        kernel_kv_tile_size=args.kernel_kv_tile_size,
        hybrid_gdn_recurrent_cache_dtype=None,
        gdn_recurrent_cache_dtype=args.gdn_recurrent_cache_dtype,
        hybrid_gdn_conv_cache_dtype=None,
        gdn_conv_cache_dtype=args.gdn_conv_cache_dtype,
        gdn_checkpoint_interval=args.gdn_checkpoint_interval,
        max_gdn_checkpoint_slots=args.max_gdn_checkpoint_slots,
        hybrid_cache_mode="all",
        hybrid_cache_prefix_boundary_only=True,
        hybrid_cache_validate_exact=True,
        hybrid_apc_require_vllm_metadata=getattr(
            args, "hybrid_apc_require_vllm_metadata", False
        ),
        hybrid_apc_reject_unbacked_attention_hits=getattr(
            args,
            "hybrid_apc_reject_unbacked_attention_hits",
            True,
        ),
        hybrid_apc_disable_unbacked_prefix_reads=getattr(
            args,
            "hybrid_apc_disable_unbacked_prefix_reads",
            False,
        ),
        hybrid_apc_enable_backed_prefix_reads=getattr(
            args,
            "hybrid_apc_enable_backed_prefix_reads",
            False,
        ),
        hybrid_apc_prefill_chunk_tokens=getattr(
            args,
            "hybrid_apc_prefill_chunk_tokens",
            0,
        ),
        hybrid_apc_max_backed_prefix_read_len=getattr(
            args,
            "hybrid_apc_max_backed_prefix_read_len",
            0,
        ),
        text_only_cte=True,
        compact_cte_attention_mask=True,
        cold_zero_conv_fast_path=False,
    )


def _build_llm(args, *, enable_hybrid_apc: bool):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(QWEN_ROOT / "vllm"))
    sys.path.insert(0, str(QWEN_ROOT))
    os.environ["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}:{QWEN_ROOT / 'vllm'}:{QWEN_ROOT}:"
        f"{os.environ.get('PYTHONPATH', '')}"
    )
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    os.environ.setdefault("VLLM_PLUGINS", "neuron")
    if enable_hybrid_apc:
        os.environ.setdefault("QWEN36_HYBRID_APC_INSTALL_PATCH", "1")
    if args.enable_vllm_chunked_prefill:
        os.environ["DISABLE_NEURON_CUSTOM_SCHEDULER"] = "1"
    if args.hybrid_apc_disable_unbacked_prefix_reads:
        os.environ["QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS"] = "1"
    if args.compiled_artifacts:
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(
            Path(args.compiled_artifacts).expanduser().resolve()
        )
        if not args.skip_fp8_env:
            _ensure_fp8_environment()

    runner = _load_module("qwen36_run_offline_inference_validation", RUNNER_PATH)
    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433
    from qwen36_hybrid_apc_scheduler_patch import (  # noqa: WPS433
        install_import_hook as install_hybrid_apc_scheduler_patch,
    )

    register_qwen35_config()
    install_hybrid_apc_scheduler_patch()
    from vllm import LLM, SamplingParams  # noqa: WPS433

    runner_args = _runner_args(args, enable_hybrid_apc=enable_hybrid_apc)
    additional_config = _align_additional_config_to_compiled_artifact(
        args,
        runner._override_config(runner_args),
    )
    llm_kwargs = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": enable_hybrid_apc,
        "enable_chunked_prefill": args.enable_vllm_chunked_prefill,
        "additional_config": additional_config,
        # vLLM multiplies its default CPU swap space by tensor_parallel_size.
        # Neuron validation runs with large TP counts and no CUDA swap path, so
        # the default can exceed host RAM before the Neuron model is loaded.
        "swap_space": 0,
    }
    gpu_memory_utilization = getattr(args, "gpu_memory_utilization", None)
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
    if enable_hybrid_apc or args.enable_vllm_chunked_prefill:
        llm_kwargs["block_size"] = args.block_size
    if enable_hybrid_apc:
        llm_kwargs["mamba_cache_mode"] = "all"
        recurrent_cache_dtype = str(args.gdn_recurrent_cache_dtype).lower()
        if recurrent_cache_dtype in {"bfloat16", "bf16"}:
            recurrent_cache_dtype = "auto"
        llm_kwargs["mamba_ssm_cache_dtype"] = recurrent_cache_dtype
    if args.enable_vllm_chunked_prefill:
        llm_kwargs["max_num_batched_tokens"] = runner._max_num_batched_tokens(
            runner_args,
            runner._cte_buckets(runner_args),
        )
    if (
        runner_args.enable_prefix_caching
        or runner_args.enable_hybrid_apc
        or runner_args.enable_vllm_chunked_prefill
    ):
        llm_kwargs["num_gpu_blocks_override"] = runner._pa_num_blocks(runner_args)
    sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_tokens)
    return LLM(**llm_kwargs), sampling


def _generate(llm, sampling, prompt: str):
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - start
    token_ids = list(outputs[0].outputs[0].token_ids)
    return {"tokens": token_ids, "elapsed_seconds": elapsed}


def _generate_many(llm, sampling, labeled_prompts):
    start = time.perf_counter()
    outputs = llm.generate([prompt for _label, prompt in labeled_prompts], sampling)
    elapsed = time.perf_counter() - start
    return {
        label: {
            "tokens": list(output.outputs[0].token_ids),
            "elapsed_seconds": elapsed,
        }
        for (label, _prompt), output in zip(labeled_prompts, outputs)
    }


def _shutdown_llm(llm) -> None:
    if llm is None:
        return
    for target in (
        llm,
        getattr(llm, "llm_engine", None),
        getattr(getattr(llm, "llm_engine", None), "engine_core", None),
        getattr(getattr(llm, "llm_engine", None), "engine_core_client", None),
    ):
        shutdown = getattr(target, "shutdown", None)
        if shutdown is None:
            continue
        try:
            shutdown()
        except Exception:
            pass
    del llm
    gc.collect()


def _generate_batch_worker(args_dict, enable_hybrid_apc: bool, labeled_prompts, result_queue):
    llm = None
    try:
        args = argparse.Namespace(**args_dict)
        llm, sampling = _build_llm(args, enable_hybrid_apc=enable_hybrid_apc)
        labeled_prompts = _maybe_bucket_align_labeled_prompts(args, labeled_prompts)
        results = {}
        for label, prompt in labeled_prompts:
            if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
                prompt_len = (
                    len(prompt.get("prompt_token_ids", []))
                    if isinstance(prompt, dict)
                    else len(prompt)
                )
                print(
                    "[hybrid_apc_debug] generate "
                    f"label={label} enable_hybrid_apc={enable_hybrid_apc} "
                    f"prompt_len={prompt_len}",
                    flush=True,
                )
            results[label] = _generate(llm, sampling, prompt)
        result_queue.put({"ok": True, "results": results})
    except BaseException:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})
    finally:
        _shutdown_llm(llm)


def _generate_batch(args, *, enable_hybrid_apc: bool, labeled_prompts):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_generate_batch_worker,
        args=(vars(args), enable_hybrid_apc, labeled_prompts, result_queue),
    )
    proc.start()
    proc.join()

    try:
        message = result_queue.get(timeout=1.0)
    except queue.Empty as exc:
        raise RuntimeError(
            f"generation worker exited with code {proc.exitcode} without a report"
        ) from exc
    if not message["ok"]:
        raise RuntimeError(message["traceback"])
    if proc.exitcode not in (0, None):
        raise RuntimeError(f"generation worker exited with code {proc.exitcode}")
    return message["results"]


def _generate_grouped_batch_worker(
    args_dict,
    enable_hybrid_apc: bool,
    labeled_prompt_groups,
    result_queue,
):
    llm = None
    try:
        args = argparse.Namespace(**args_dict)
        llm, sampling = _build_llm(args, enable_hybrid_apc=enable_hybrid_apc)
        results = {}
        for group in labeled_prompt_groups:
            group = _maybe_bucket_align_labeled_prompts(args, group)
            if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
                print(
                    "[hybrid_apc_debug] generate-group "
                    f"labels={[label for label, _prompt in group]} "
                    f"enable_hybrid_apc={enable_hybrid_apc}",
                    flush=True,
                )
            if len(group) == 1:
                label, prompt = group[0]
                results[label] = _generate(llm, sampling, prompt)
            else:
                results.update(_generate_many(llm, sampling, group))
        result_queue.put({"ok": True, "results": results})
    except BaseException:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})
    finally:
        _shutdown_llm(llm)


def _generate_grouped_batch(args, *, enable_hybrid_apc: bool, labeled_prompt_groups):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_generate_grouped_batch_worker,
        args=(vars(args), enable_hybrid_apc, labeled_prompt_groups, result_queue),
    )
    proc.start()
    proc.join()

    try:
        message = result_queue.get(timeout=1.0)
    except queue.Empty as exc:
        raise RuntimeError(
            f"generation worker exited with code {proc.exitcode} without a report"
        ) from exc
    if not message["ok"]:
        raise RuntimeError(message["traceback"])
    if proc.exitcode not in (0, None):
        raise RuntimeError(f"generation worker exited with code {proc.exitcode}")
    return message["results"]


def _token_check(label: str, result: dict, dummy_token_ids: set[int]) -> dict:
    tokens = [int(token) for token in result.get("tokens", [])]
    unique_token_ids = sorted(set(tokens))
    non_dummy_tokens = [token for token in tokens if token not in dummy_token_ids]
    passed = bool(non_dummy_tokens)
    check = {
        "label": label,
        "generated_token_count": len(tokens),
        "unique_token_ids": unique_token_ids,
        "dummy_token_ids": sorted(dummy_token_ids),
        "non_dummy_token_count": len(non_dummy_tokens),
        "passed": passed,
    }
    if not passed:
        check["failure"] = "generated tokens are empty or all configured dummy tokens"
    return check


def _real_token_checks(results_by_label: dict[str, dict], dummy_token_ids: set[int]) -> dict:
    checks = {
        label: _token_check(label, result, dummy_token_ids)
        for label, result in sorted(results_by_label.items())
    }
    return {
        "passed": bool(checks) and all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def _effective_dummy_token_ids(args, tokenizer=None) -> set[int]:
    configured = {int(token_id) for token_id in args.dummy_token_ids}
    if tokenizer is None or configured != {0}:
        return configured
    special_ids = {
        int(token_id)
        for token_id in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
        )
        if token_id is not None
    }
    return special_ids or configured


def _prompt_token_count(prompt: Any) -> int:
    if isinstance(prompt, dict):
        return len(prompt.get("prompt_token_ids", []))
    return len(str(prompt))


def _token_prompt(prefix_ids: list[int], suffix_ids: list[int] | None = None) -> dict:
    return {"prompt_token_ids": list(prefix_ids) + list(suffix_ids or [])}


def _compact_boundary_lengths(args) -> list[int]:
    if getattr(args, "compact_boundary_lens", None):
        candidates = _parse_bucket_values(args.compact_boundary_lens)
    else:
        block_size = int(args.block_size)
        candidates = [
            block_size - 1,
            block_size,
            block_size + 1,
            (2 * block_size) - 1,
            2 * block_size,
            (2 * block_size) + 1,
        ]
    max_suffix = max(1, int(getattr(args, "compact_suffix_tokens", 16)))
    max_prompt_len = max(1, int(args.seq_len) - max_suffix - max(1, int(args.max_tokens)))
    return [
        prefix_len
        for prefix_len in sorted(set(candidates))
        if 0 < prefix_len <= max_prompt_len
    ]


def _make_prefix_ids(tokenizer, *, label: str, target_len: int) -> list[int]:
    seed = f"System {label}: answer deterministically.\n"
    text = seed
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_len:
        text += seed
        ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids[:target_len])


def _make_suffix_ids(tokenizer, *, label: str, target_len: int) -> list[int]:
    seed = f"\nUser: compact gate suffix {label}. Answer with one token.\nAssistant:"
    ids = tokenizer.encode(seed, add_special_tokens=False)
    if len(ids) >= target_len:
        return list(ids[:target_len])
    pad_piece = tokenizer.encode(" detail", add_special_tokens=False)
    if not pad_piece:
        raise ValueError("tokenizer returned no tokens for compact suffix padding")
    while len(ids) < target_len:
        ids.extend(pad_piece)
    return list(ids[:target_len])


def _single_token_piece(tokenizer, start_index: int) -> str:
    for offset in range(len(COMPACT_SINGLE_TOKEN_PIECES)):
        piece = COMPACT_SINGLE_TOKEN_PIECES[
            (int(start_index) + offset) % len(COMPACT_SINGLE_TOKEN_PIECES)
        ]
        if len(tokenizer.encode(piece, add_special_tokens=False)) == 1:
            return piece
    raise ValueError("could not find a compact-gate single-token text piece")


def _compact_single_token_ids(tokenizer) -> list[int]:
    special_ids = {
        int(token_id)
        for token_id in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
        )
        if token_id is not None
    }
    ids = []
    for piece in COMPACT_SINGLE_TOKEN_PIECES:
        piece_ids = tokenizer.encode(piece, add_special_tokens=False)
        if len(piece_ids) == 1 and int(piece_ids[0]) not in special_ids:
            token_id = int(piece_ids[0])
            if token_id not in ids:
                ids.append(token_id)
    if len(ids) < 4:
        next_id = max(ids or [0]) + 1
        while len(ids) < 4:
            if next_id not in special_ids and next_id not in ids:
                ids.append(next_id)
            next_id += 1
    return ids


def _compact_role_prefix_ids(
    tokenizer,
    *,
    role_index: int,
    token_count: int,
) -> list[int]:
    """Build a globally unique, tokenizer-stable prefix for one compact-gate role."""

    token_count = int(token_count)
    if token_count <= 0:
        return []
    pool = _compact_single_token_ids(tokenizer)
    base = len(pool)
    header_len = min(token_count, 6)
    header = [
        pool[(int(role_index) // (base**offset)) % base]
        for offset in range(header_len)
    ]
    if token_count <= header_len:
        return header[:token_count]
    body = [
        pool[(int(role_index) + 3 + (position * 7)) % base]
        for position in range(token_count - header_len)
    ]
    return header + body


def _repeat_single_token_piece(tokenizer, *, start_index: int, token_count: int) -> str:
    piece = _single_token_piece(tokenizer, start_index)
    text = piece * int(token_count)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    if actual != int(token_count):
        raise ValueError(
            "compact-gate tokenizer-stable text construction failed: "
            f"wanted {token_count} tokens but built {actual}"
        )
    return text


def _compact_instruction_suffix_text(
    tokenizer,
    *,
    label: str,
    start_index: int,
    token_count: int,
) -> str:
    tails = (
        " one two three four five",
        " one two three",
        " one",
    )
    tail = None
    tail_len = 0
    for candidate in tails:
        candidate_len = len(tokenizer.encode(candidate, add_special_tokens=False))
        if candidate_len <= int(token_count):
            tail = candidate
            tail_len = candidate_len
            break
    if tail is None:
        raise ValueError(f"compact-gate suffix {label!r} cannot fit in {token_count} tokens")
    filler_len = int(token_count) - tail_len
    filler = (
        _repeat_single_token_piece(
            tokenizer,
            start_index=start_index,
            token_count=filler_len,
        )
        if filler_len > 0
        else ""
    )
    text = filler + tail
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    if actual != int(token_count):
        raise ValueError(
            "compact-gate instruction suffix construction failed: "
            f"wanted {token_count} tokens but built {actual} for {label}"
        )
    return text


def _stable_text_prompt(tokenizer, prefix: str, suffix: str = "") -> str:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full = prefix + suffix
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "compact-gate text prompt is not tokenizer-stable at the prefix/suffix "
            "boundary"
        )
    return full


def _compact_case_plan(args, tokenizer) -> dict:
    suffix_tokens = int(getattr(args, "compact_suffix_tokens", 16))
    prefill_batch_budget = max(_parse_bucket_values(args.cte_buckets))
    largest_cte_bucket = prefill_batch_budget
    cases = []
    for index, prefix_len in enumerate(_compact_boundary_lengths(args)):
        prefix_a_ids = _compact_role_prefix_ids(
            tokenizer,
            role_index=(index * 3),
            token_count=prefix_len,
        )
        prefix_b_ids = _compact_role_prefix_ids(
            tokenizer,
            role_index=(index * 3) + 1,
            token_count=prefix_len,
        )
        cold_prefix_ids = _compact_role_prefix_ids(
            tokenizer,
            role_index=(index * 3) + 2,
            token_count=prefix_len,
        )
        suffix_a = _compact_instruction_suffix_text(
            tokenizer,
            label=f"{prefix_len}-a",
            start_index=(index * 5) + 2,
            token_count=suffix_tokens,
        )
        suffix_b = _compact_instruction_suffix_text(
            tokenizer,
            label=f"{prefix_len}-b",
            start_index=(index * 5) + 3,
            token_count=suffix_tokens,
        )
        cold_suffix = _compact_instruction_suffix_text(
            tokenizer,
            label=f"{prefix_len}-cold",
            start_index=(index * 5) + 4,
            token_count=suffix_tokens,
        )
        suffix_a_ids = tokenizer.encode(suffix_a, add_special_tokens=False)
        suffix_b_ids = tokenizer.encode(suffix_b, add_special_tokens=False)
        cold_suffix_ids = tokenizer.encode(cold_suffix, add_special_tokens=False)
        backed_checkpoint_hit = (
            prefix_len % int(args.gdn_checkpoint_interval) == 0
            and (
                int(getattr(args, "hybrid_apc_max_backed_prefix_read_len", 0) or 0)
                <= 0
                or prefix_len <= int(args.hybrid_apc_max_backed_prefix_read_len)
            )
        )
        speedup_required = backed_checkpoint_hit and prefix_len < largest_cte_bucket
        speedup_skip_reason = None
        if backed_checkpoint_hit and not speedup_required:
            speedup_skip_reason = (
                "restore prefix reaches largest CTE bucket; current artifacts "
                "cannot prove grouped warm speedup for this boundary"
            )
        warm_partial_active_len = suffix_tokens if backed_checkpoint_hit else (
            prefix_len + suffix_tokens
        )
        cold_mixed_active_len = prefix_len + suffix_tokens
        cases.append(
            {
                "case": f"boundary_{prefix_len}",
                "prefix_len": prefix_len,
                "full_token_len": prefix_len,
                "partial_token_len": prefix_len + suffix_tokens,
                "full_a": _token_prompt(prefix_a_ids),
                "full_b": _token_prompt(prefix_b_ids),
                "partial_a": _token_prompt(prefix_a_ids, suffix_a_ids),
                "partial_b": _token_prompt(prefix_b_ids, suffix_b_ids),
                "mixed_cold": _token_prompt(cold_prefix_ids, cold_suffix_ids),
                "full_grouped": (2 * prefix_len) <= prefill_batch_budget,
                "partial_grouped": (
                    2 * warm_partial_active_len
                )
                <= prefill_batch_budget,
                "mixed_grouped": (
                    warm_partial_active_len + cold_mixed_active_len
                )
                <= prefill_batch_budget,
                "backed_checkpoint_hit": backed_checkpoint_hit,
                "speedup_required": speedup_required,
                "speedup_skip_reason": speedup_skip_reason,
            }
        )
    return {
        "boundary_lengths": [case["prefix_len"] for case in cases],
        "cases": cases,
    }


def _compact_exactness_check(
    *,
    name: str,
    cold_label: str,
    warm_label: str,
    cold_results: dict[str, dict],
    warm_results: dict[str, dict],
) -> dict:
    cold_tokens = list(cold_results[cold_label]["tokens"])
    warm_tokens = list(warm_results[warm_label]["tokens"])
    return {
        "name": name,
        "cold_label": cold_label,
        "warm_label": warm_label,
        "passed": cold_tokens == warm_tokens,
        "cold_tokens": cold_tokens,
        "warm_tokens": warm_tokens,
    }


def _compact_speedup_check(
    *,
    name: str,
    cold_labels: list[str],
    warm_label: str,
    cold_results: dict[str, dict],
    warm_results: dict[str, dict],
    min_speedup: float,
) -> dict:
    cold_serial = sum(float(cold_results[label]["elapsed_seconds"]) for label in cold_labels)
    warm_elapsed = float(warm_results[warm_label]["elapsed_seconds"])
    speedup = cold_serial / warm_elapsed if warm_elapsed > 0 else float("inf")
    return {
        "name": name,
        "cold_labels": list(cold_labels),
        "warm_label": warm_label,
        "cold_serial_seconds": cold_serial,
        "warm_group_seconds": warm_elapsed,
        "speedup": speedup,
        "min_speedup": min_speedup,
        "passed": speedup >= min_speedup,
    }


def _write_report(args, report: dict) -> None:
    if args.output_json:
        args.output_json.expanduser().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


def run_exactness(args) -> int:
    shared = args.shared_prefix
    prompt_a = shared + args.suffix_a
    prompt_b = shared + args.suffix_b

    # This validation uses the v3 vLLM APC artifact, which is compiled for
    # prefix/block KV layout. Keep prefix metadata enabled even for cold
    # references, and isolate each cold prompt in a fresh process so it cannot
    # observe cache state from the other reference prompt.
    cold_full = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("cold_full", prompt_a),
        ],
    )["cold_full"]
    cold_partial = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("cold_partial", prompt_b),
        ],
    )["cold_partial"]
    warm_results = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("warmup_full", prompt_a),
            ("warm_full", prompt_a),
            ("warmup_partial", prompt_a),
            ("warm_partial", prompt_b),
        ],
    )

    warmup_full = warm_results["warmup_full"]
    warm_full = warm_results["warm_full"]
    warm_partial = warm_results["warm_partial"]
    all_results = {
        "cold_full": cold_full,
        "warmup_full": warmup_full,
        "warm_full": warm_full,
        "cold_partial": cold_partial,
        "warm_partial": warm_partial,
    }
    real_token_checks = _real_token_checks(
        all_results,
        _effective_dummy_token_ids(args),
    )

    report = {
        "full_prefix_exact": cold_full["tokens"] == warm_full["tokens"],
        "partial_prefix_exact": cold_partial["tokens"] == warm_partial["tokens"],
        "cold_full": cold_full,
        "warmup_full": warmup_full,
        "warm_full": warm_full,
        "cold_partial": cold_partial,
        "warm_partial": warm_partial,
        "real_generated_tokens_required": args.require_real_tokens,
        "real_generated_tokens_passed": real_token_checks["passed"],
        "real_generated_token_checks": real_token_checks["checks"],
        "negative_tests": {
            "missing_gdn_state_fallback": "requires scheduler fault injection",
            "zeroed_conv_state": "requires model debug hook",
        },
    }
    if args.output_json:
        args.output_json.expanduser().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    passed = report["full_prefix_exact"] and report["partial_prefix_exact"]
    if args.require_real_tokens:
        passed = passed and real_token_checks["passed"]
    return 0 if passed else 1


def run_batched_exactness(args) -> int:
    if not args.shared_prefix_2:
        raise ValueError("--shared-prefix-2 is required for batched-exactness")
    _validate_generation_batch_support(args)

    prompt_full_a = args.shared_prefix + args.suffix_a
    prompt_partial_a = args.shared_prefix + args.suffix_b
    prompt_full_b = args.shared_prefix_2 + args.suffix_c
    prompt_partial_b = args.shared_prefix_2 + args.suffix_d

    cold_partial_a = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("cold_partial_a", prompt_partial_a),
        ],
    )["cold_partial_a"]
    cold_partial_b = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("cold_partial_b", prompt_partial_b),
        ],
    )["cold_partial_b"]
    warm_results = _generate_grouped_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompt_groups=[
            [("warmup_full_a", prompt_full_a)],
            [("warmup_full_b", prompt_full_b)],
            [
                ("warm_partial_a", prompt_partial_a),
                ("warm_partial_b", prompt_partial_b),
            ],
        ],
    )

    all_results = {
        "cold_partial_a": cold_partial_a,
        "cold_partial_b": cold_partial_b,
        **warm_results,
    }
    real_token_checks = _real_token_checks(
        all_results,
        _effective_dummy_token_ids(args),
    )
    report = {
        "batched_partial_a_exact": (
            cold_partial_a["tokens"] == warm_results["warm_partial_a"]["tokens"]
        ),
        "batched_partial_b_exact": (
            cold_partial_b["tokens"] == warm_results["warm_partial_b"]["tokens"]
        ),
        "max_num_seqs": args.max_num_seqs,
        "cold_partial_a": cold_partial_a,
        "cold_partial_b": cold_partial_b,
        "warmup_full_a": warm_results["warmup_full_a"],
        "warmup_full_b": warm_results["warmup_full_b"],
        "warm_partial_a": warm_results["warm_partial_a"],
        "warm_partial_b": warm_results["warm_partial_b"],
        "real_generated_tokens_required": args.require_real_tokens,
        "real_generated_tokens_passed": real_token_checks["passed"],
        "real_generated_token_checks": real_token_checks["checks"],
    }
    if args.output_json:
        args.output_json.expanduser().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    passed = report["batched_partial_a_exact"] and report["batched_partial_b_exact"]
    if args.require_real_tokens:
        passed = passed and real_token_checks["passed"]
    return 0 if passed else 1


def run_compact_gate(args) -> int:
    if not args.hybrid_apc_require_vllm_metadata:
        raise ValueError("compact-gate requires --hybrid-apc-require-vllm-metadata")
    if not args.hybrid_apc_disable_unbacked_prefix_reads:
        raise ValueError(
            "compact-gate requires --hybrid-apc-disable-unbacked-prefix-reads"
        )
    if not args.hybrid_apc_enable_backed_prefix_reads:
        raise ValueError(
            "compact-gate requires --hybrid-apc-enable-backed-prefix-reads"
        )
    if args.max_num_seqs < 2:
        raise ValueError("compact-gate requires --max-num-seqs >= 2")
    _validate_generation_batch_support(args)

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        trust_remote_code=True,
    )
    plan = _compact_case_plan(args, tokenizer)
    cases = plan["cases"]
    if not cases:
        raise ValueError("compact-gate produced no cases")

    cold_full_prompts = []
    cold_partial_prompts = []
    for case in cases:
        name = case["case"]
        cold_full_prompts.extend(
            [
                (f"cold_full_a__{name}", case["full_a"]),
                (f"cold_full_b__{name}", case["full_b"]),
            ]
        )
        cold_partial_prompts.extend(
            [
                (f"cold_partial_a__{name}", case["partial_a"]),
                (f"cold_partial_b__{name}", case["partial_b"]),
                (f"cold_mixed__{name}", case["mixed_cold"]),
            ]
        )

    cold_results = {}
    cold_results.update(
        _generate_batch(
            args,
            enable_hybrid_apc=True,
            labeled_prompts=cold_full_prompts,
        )
    )
    cold_results.update(
        _generate_batch(
            args,
            enable_hybrid_apc=True,
            labeled_prompts=cold_partial_prompts,
        )
    )

    warm_groups = []
    for case in cases:
        name = case["case"]
        warm_groups.extend(
            [
                [(f"warmup_full_a__{name}", case["full_a"])],
                [(f"warmup_full_b__{name}", case["full_b"])],
            ]
        )
        if case["full_grouped"]:
            warm_groups.append(
                [
                    (f"warm_full_a__{name}", case["full_a"]),
                    (f"warm_full_b__{name}", case["full_b"]),
                ]
            )
        else:
            warm_groups.extend(
                [
                    [(f"warm_full_a__{name}", case["full_a"])],
                    [(f"warm_full_b__{name}", case["full_b"])],
                ]
            )
        if case["partial_grouped"]:
            warm_groups.append(
                [
                    (f"warm_partial_a__{name}", case["partial_a"]),
                    (f"warm_partial_b__{name}", case["partial_b"]),
                ]
            )
        else:
            warm_groups.extend(
                [
                    [(f"warm_partial_a__{name}", case["partial_a"])],
                    [(f"warm_partial_b__{name}", case["partial_b"])],
                ]
            )
        if case["mixed_grouped"]:
            warm_groups.append(
                [
                    (f"mixed_warm_a__{name}", case["partial_a"]),
                    (f"mixed_cold__{name}", case["mixed_cold"]),
                ]
            )
        else:
            warm_groups.extend(
                [
                    [(f"mixed_warm_a__{name}", case["partial_a"])],
                    [(f"mixed_cold__{name}", case["mixed_cold"])],
                ]
            )
    first_case = cases[0]
    warm_groups.append(
        [
            (
                f"eviction_probe_partial_a__{first_case['case']}",
                first_case["partial_a"],
            )
        ]
    )
    warm_results = _generate_grouped_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompt_groups=warm_groups,
    )

    exactness_checks = []
    speedup_checks = []
    speedup_skipped = []
    for case in cases:
        name = case["case"]
        exactness_checks.extend(
            [
                _compact_exactness_check(
                    name=f"same_full_a__{name}",
                    cold_label=f"cold_full_a__{name}",
                    warm_label=f"warm_full_a__{name}",
                    cold_results=cold_results,
                    warm_results=warm_results,
                ),
                _compact_exactness_check(
                    name=f"same_full_b__{name}",
                    cold_label=f"cold_full_b__{name}",
                    warm_label=f"warm_full_b__{name}",
                    cold_results=cold_results,
                    warm_results=warm_results,
                ),
                _compact_exactness_check(
                    name=f"partial_a__{name}",
                    cold_label=f"cold_partial_a__{name}",
                    warm_label=f"warm_partial_a__{name}",
                    cold_results=cold_results,
                    warm_results=warm_results,
                ),
                _compact_exactness_check(
                    name=f"partial_b__{name}",
                    cold_label=f"cold_partial_b__{name}",
                    warm_label=f"warm_partial_b__{name}",
                    cold_results=cold_results,
                    warm_results=warm_results,
                ),
                _compact_exactness_check(
                    name=f"mixed_warm_a__{name}",
                    cold_label=f"cold_partial_a__{name}",
                    warm_label=f"mixed_warm_a__{name}",
                    cold_results=cold_results,
                    warm_results=warm_results,
                ),
                _compact_exactness_check(
                    name=f"mixed_cold__{name}",
                    cold_label=f"cold_mixed__{name}",
                    warm_label=f"mixed_cold__{name}",
                    cold_results=cold_results,
                    warm_results=warm_results,
                ),
            ]
        )
        if case["speedup_required"] and case["partial_grouped"]:
            speedup_checks.append(
                _compact_speedup_check(
                    name=f"grouped_warm_partials__{name}",
                    cold_labels=[
                        f"cold_partial_a__{name}",
                        f"cold_partial_b__{name}",
                    ],
                    warm_label=f"warm_partial_a__{name}",
                    cold_results=cold_results,
                    warm_results=warm_results,
                    min_speedup=float(args.compact_min_grouped_speedup),
                )
            )
        elif case["backed_checkpoint_hit"] and case["partial_grouped"]:
            speedup_skipped.append(
                {
                    "name": f"grouped_warm_partials__{name}",
                    "prefix_len": case["prefix_len"],
                    "reason": case["speedup_skip_reason"]
                    or "speedup is not required for this boundary",
                }
            )

    exactness_checks.append(
        _compact_exactness_check(
            name=f"eviction_probe_partial_a__{first_case['case']}",
            cold_label=f"cold_partial_a__{first_case['case']}",
            warm_label=f"eviction_probe_partial_a__{first_case['case']}",
            cold_results=cold_results,
            warm_results=warm_results,
        )
    )

    all_results = {
        **{f"cold::{label}": result for label, result in cold_results.items()},
        **{f"warm::{label}": result for label, result in warm_results.items()},
    }
    real_token_checks = _real_token_checks(
        all_results,
        _effective_dummy_token_ids(args, tokenizer),
    )
    total_requests = len(cold_results) + len(warm_results)
    grouped_partial_case_count = sum(1 for case in cases if case["partial_grouped"])
    grouped_mixed_case_count = sum(1 for case in cases if case["mixed_grouped"])
    acceptance = {
        "request_count": total_requests,
        "min_request_count": args.compact_min_requests,
        "request_count_passed": total_requests >= args.compact_min_requests,
        "exactness_passed": all(check["passed"] for check in exactness_checks),
        "real_generated_tokens_passed": real_token_checks["passed"],
        "grouped_partial_case_count": grouped_partial_case_count,
        "grouped_partial_coverage_passed": grouped_partial_case_count > 0,
        "grouped_mixed_case_count": grouped_mixed_case_count,
        "grouped_mixed_coverage_passed": grouped_mixed_case_count > 0,
        "speedup_checks_required": len(speedup_checks),
        "speedup_passed": bool(speedup_checks)
        and all(check["passed"] for check in speedup_checks),
        "runtime_exception_free": True,
        "eviction_probe_passed": exactness_checks[-1]["passed"],
    }
    acceptance["passed"] = all(
        bool(acceptance[name])
        for name in (
            "request_count_passed",
            "exactness_passed",
            "real_generated_tokens_passed",
            "grouped_partial_coverage_passed",
            "grouped_mixed_coverage_passed",
            "speedup_passed",
            "runtime_exception_free",
            "eviction_probe_passed",
        )
    )
    report = {
        "compact_gate_passed": acceptance["passed"],
        "acceptance": acceptance,
        "boundary_lengths": plan["boundary_lengths"],
        "block_size": args.block_size,
        "gdn_checkpoint_interval": args.gdn_checkpoint_interval,
        "max_gdn_checkpoint_slots": args.max_gdn_checkpoint_slots,
        "max_num_seqs": args.max_num_seqs,
        "max_tokens": args.max_tokens,
        "compact_suffix_tokens": args.compact_suffix_tokens,
        "hybrid_apc_require_vllm_metadata": args.hybrid_apc_require_vllm_metadata,
        "hybrid_apc_disable_unbacked_prefix_reads": (
            args.hybrid_apc_disable_unbacked_prefix_reads
        ),
        "hybrid_apc_enable_backed_prefix_reads": (
            args.hybrid_apc_enable_backed_prefix_reads
        ),
        "hybrid_apc_max_backed_prefix_read_len": (
            args.hybrid_apc_max_backed_prefix_read_len
        ),
        "exactness_checks": exactness_checks,
        "speedup_checks": speedup_checks,
        "speedup_skipped": speedup_skipped,
        "real_generated_token_checks": real_token_checks["checks"],
    }
    _write_report(args, report)
    return 0 if acceptance["passed"] else 1


def run_hbm(args) -> int:
    hybrid_apc = _load_module("qwen36_hybrid_apc_validation", HYBRID_APC_PATH)
    rows = []
    for context_len in args.context_lens:
        for interval in args.checkpoint_intervals:
            estimate = hybrid_apc.estimate_qwen_hybrid_cache_bytes_per_rank(
                max_context_len=context_len,
                checkpoint_interval=interval,
                recurrent_dtype=args.gdn_recurrent_cache_dtype,
                conv_dtype=args.gdn_conv_cache_dtype,
            )
            rows.append(
                {
                    "context_len": context_len,
                    "checkpoint_interval": interval,
                    "num_gdn_checkpoints": estimate["num_gdn_checkpoints"],
                    "attention_kv_gib": estimate["attention_kv_bytes"] / 2**30,
                    "gdn_checkpoint_gib": estimate["gdn_checkpoint_bytes"] / 2**30,
                    "total_gib": estimate["total_bytes"] / 2**30,
                    "bytes_per_gdn_checkpoint": estimate["gdn_bytes_per_checkpoint"],
                }
            )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_exact_args(exact):
        exact.add_argument("--model-path", required=True)
        exact.add_argument("--compiled-artifacts")
        exact.add_argument(
            "--skip-fp8-env",
            action="store_true",
            help="Do not set FP8 runtime environment defaults for BF16 control artifacts.",
        )
        exact.add_argument("--max-model-len", type=int, default=2048)
        exact.add_argument("--seq-len", type=int, default=2048)
        exact.add_argument("--cte-bucket", type=int, default=512)
        exact.add_argument("--cte-buckets", nargs="+", default=["256,512"])
        exact.add_argument("--context-encoding-bucket-pairs", nargs="+", default=None)
        exact.add_argument(
            "--align-prompts-to-cte-buckets",
            action="store_true",
            help=(
                "Tokenize prompts and pad token ids to the next compiled CTE bucket "
                "before calling vLLM. This is useful for static Neuron artifacts "
                "that reject non-bucket prompt shapes."
            ),
        )
        exact.add_argument("--cte-bucket-profile", default="single")
        exact.add_argument("--tensor-parallel-size", type=int, default=4)
        exact.add_argument("--max-num-seqs", type=int, default=1)
        exact.add_argument("--logical-nc-config", type=int, default=2)
        exact.add_argument("--ctx-batch-size", type=int, default=1)
        exact.add_argument("--token-generation-buckets", nargs="+", default=None)
        exact.add_argument("--token-generation-batches", nargs="+", default=None)
        exact.add_argument("--async-mode", action="store_true")
        exact.add_argument("--block-size", type=int, default=256)
        exact.add_argument("--gdn-checkpoint-interval", type=int, default=256)
        exact.add_argument("--max-gdn-checkpoint-slots", type=int, default=8)
        exact.add_argument("--gdn-recurrent-cache-dtype", default="float32")
        exact.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
        exact.add_argument("--hybrid-apc-require-vllm-metadata", action="store_true")
        exact.add_argument(
            "--hybrid-apc-reject-unbacked-attention-hits",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        exact.add_argument(
            "--hybrid-apc-disable-unbacked-prefix-reads",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        exact.add_argument(
            "--hybrid-apc-enable-backed-prefix-reads",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        exact.add_argument("--hybrid-apc-max-backed-prefix-read-len", type=int, default=0)
        exact.add_argument("--enable-vllm-chunked-prefill", action="store_true")
        exact.add_argument("--kernel-q-tile-size", type=int, default=128)
        exact.add_argument("--kernel-kv-tile-size", type=int, default=1024)
        exact.add_argument("--num-gpu-blocks-override", type=int)
        exact.add_argument("--max-tokens", type=int, default=32)
        exact.add_argument(
            "--shared-prefix",
            default="System: answer deterministically.\n" * 64,
        )
        exact.add_argument("--suffix-a", default="\nUser: What is 17 * 23?\nAssistant:")
        exact.add_argument("--suffix-b", default="\nUser: What is 19 * 29?\nAssistant:")
        exact.add_argument(
            "--require-real-tokens",
            action="store_true",
            help=(
                "Fail exactness if every generated token for any checked request is a "
                "configured dummy token."
            ),
        )
        exact.add_argument(
            "--dummy-token-ids",
            nargs="+",
            type=int,
            default=[0],
            help="Token ids treated as dummy generated output when --require-real-tokens is set.",
        )
        exact.add_argument("--output-json", type=Path)

    exact = subparsers.add_parser("exactness")
    add_common_exact_args(exact)
    exact.set_defaults(func=run_exactness)

    batched = subparsers.add_parser("batched-exactness")
    add_common_exact_args(batched)
    batched.add_argument("--shared-prefix-2", required=True)
    batched.add_argument("--suffix-c", default="")
    batched.add_argument("--suffix-d", default="\nUser: What is 23 * 31?\nAssistant:")
    batched.set_defaults(func=run_batched_exactness)

    compact = subparsers.add_parser("compact-gate")
    add_common_exact_args(compact)
    compact.add_argument(
        "--compact-boundary-lens",
        nargs="+",
        help=(
            "Prefix token lengths to test. Defaults to block_size +/- 1 and "
            "2*block_size +/- 1."
        ),
    )
    compact.add_argument(
        "--compact-suffix-tokens",
        type=int,
        default=16,
        help="Tokenized suffix length appended to partial-prefix prompts.",
    )
    compact.add_argument(
        "--compact-min-requests",
        type=int,
        default=50,
        help="Minimum generated request count required for the compact gate.",
    )
    compact.add_argument(
        "--compact-min-grouped-speedup",
        type=float,
        default=1.5,
        help=(
            "Minimum warm grouped throughput speedup required for checkpoint "
            "boundary cases."
        ),
    )
    compact.set_defaults(func=run_compact_gate)

    hbm = subparsers.add_parser("hbm")
    hbm.add_argument("--context-lens", nargs="+", type=int, default=[131072, 262144])
    hbm.add_argument("--checkpoint-intervals", nargs="+", type=int, default=[128, 256, 512])
    hbm.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    hbm.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    hbm.set_defaults(func=run_hbm)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
