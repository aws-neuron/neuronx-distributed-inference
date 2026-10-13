#!/usr/bin/env python3
"""Offline vLLM smoke runner for Qwen3.6-27B on Neuron."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


_FP8_ENV_DEFAULTS = {
    "XLA_HANDLE_SPECIAL_SCALAR": "1",
    "UNSAFE_FP8FNCAST": "1",
}


def _ensure_fp8_environment() -> None:
    for name, value in _FP8_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


def _contrib_root(repo_root: str | None) -> Path:
    if repo_root:
        return Path(repo_root).expanduser().resolve() / "contrib" / "models" / "Qwen3.6-27B"
    return Path(__file__).resolve().parents[1]


def _parse_int_list(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    tokens: list[str] = []
    for value in values:
        tokens.extend(value.replace(",", " ").split())
    return [int(token) for token in tokens]


def _parse_bucket_pairs(values: list[str] | None) -> list[list[int]] | None:
    if values is None:
        return None
    pairs: set[tuple[int, int]] = set()
    for value in values:
        for token in value.replace(",", " ").split():
            if ":" in token:
                active, prefix = token.split(":", 1)
            elif "x" in token:
                active, prefix = token.split("x", 1)
            else:
                raise ValueError(
                    "--context-encoding-bucket-pairs entries must use "
                    f"ACTIVE:PREFIX syntax, got {token!r}"
                )
            active_tokens, prefix_tokens = int(active), int(prefix)
            if active_tokens <= 0 or prefix_tokens < 0:
                raise ValueError(
                    "Context-encoding bucket pairs must be positive active "
                    f"tokens and non-negative prefix tokens, got {token!r}"
                )
            pairs.add((active_tokens, prefix_tokens))
    return [[active, prefix] for active, prefix in sorted(pairs)]


def _cte_buckets(args: argparse.Namespace) -> list[int]:
    profile_buckets = {
        "short": [128, 256, 512, 1024],
        "general": [256, 512, 1024, 2048],
        "long": [4096, 8192, 16384, 32768],
        "262k": [256],
    }
    if args.cte_bucket_profile != "single":
        buckets = list(profile_buckets[args.cte_bucket_profile])
    else:
        buckets = _parse_int_list(args.cte_buckets) or [args.cte_bucket]
    buckets = sorted(set(buckets))
    if not buckets:
        raise ValueError("At least one CTE bucket is required")
    for bucket in buckets:
        if bucket <= 0:
            raise ValueError(f"CTE buckets must be positive, got {bucket}")
        if bucket % 128 != 0:
            raise ValueError(
                f"CTE bucket {bucket} is not 128-aligned; DeltaNet CTE uses 128-token chunks"
            )
    if buckets[-1] > args.seq_len:
        raise ValueError(
            f"Largest CTE bucket {buckets[-1]} exceeds --seq-len {args.seq_len}"
        )
    return buckets


def _token_generation_buckets(args: argparse.Namespace) -> list[int]:
    buckets = _parse_int_list(args.token_generation_buckets) or [args.seq_len]
    buckets = sorted(set(buckets))
    if not buckets:
        raise ValueError("At least one token-generation bucket is required")
    for bucket in buckets:
        if bucket <= 0:
            raise ValueError(
                f"Token-generation buckets must be positive, got {bucket}"
            )
        if bucket > args.seq_len:
            raise ValueError(
                f"Token-generation bucket {bucket} exceeds --seq-len {args.seq_len}"
            )
    return buckets


def _token_generation_batches(args: argparse.Namespace) -> list[int] | None:
    batches = _parse_int_list(args.token_generation_batches)
    if batches is None:
        return None
    batches = sorted(set(batches))
    if not batches:
        raise ValueError("Token-generation batches cannot be empty")
    for batch in batches:
        if batch <= 0:
            raise ValueError(
                f"Token-generation batches must be positive, got {batch}"
            )
        if batch > args.max_num_seqs:
            raise ValueError(
                f"Token-generation batch {batch} exceeds --max-num-seqs "
                f"{args.max_num_seqs}"
            )
    return batches


def _validate_hybrid_apc_args(args: argparse.Namespace):
    if not args.enable_hybrid_apc:
        return
    if args.hybrid_cache_mode != "all":
        raise ValueError("--enable-hybrid-apc requires --hybrid-cache-mode all")
    if args.gdn_checkpoint_interval != args.block_size:
        raise ValueError(
            "--enable-hybrid-apc v0 requires --gdn-checkpoint-interval "
            "to equal --block-size"
        )
    args.enable_prefix_caching = True


def _max_num_batched_tokens(args: argparse.Namespace, cte_buckets: list[int]) -> int:
    max_cte_bucket = cte_buckets[-1]
    if not args.enable_vllm_chunked_prefill:
        return max_cte_bucket
    if not args.enable_hybrid_apc:
        return max_cte_bucket

    checkpoint_interval = int(args.gdn_checkpoint_interval)
    checkpoint_aligned_buckets = [
        bucket for bucket in cte_buckets if bucket % checkpoint_interval == 0
    ]
    if not checkpoint_aligned_buckets:
        raise ValueError(
            "--enable-hybrid-apc with vLLM chunked prefill requires at least "
            "one compiled CTE bucket that is a multiple of "
            f"--gdn-checkpoint-interval ({checkpoint_interval}); got "
            f"{cte_buckets}"
        )
    requested_chunk = int(getattr(args, "hybrid_apc_prefill_chunk_tokens", 0) or 0)
    if requested_chunk <= 0:
        return min(max_cte_bucket, checkpoint_aligned_buckets[-1])
    if requested_chunk % checkpoint_interval != 0:
        raise ValueError(
            "--hybrid-apc-prefill-chunk-tokens must be a multiple of "
            f"--gdn-checkpoint-interval ({checkpoint_interval}), got {requested_chunk}"
        )
    if requested_chunk not in cte_buckets:
        raise ValueError(
            "--hybrid-apc-prefill-chunk-tokens must match a compiled CTE bucket, "
            f"got {requested_chunk} with buckets {cte_buckets}"
        )
    return min(max_cte_bucket, requested_chunk)


def _effective_prefill_group_size(
    args: argparse.Namespace,
    cte_buckets: list[int],
) -> int:
    if args.enable_vllm_chunked_prefill:
        return _max_num_batched_tokens(args, cte_buckets)
    return cte_buckets[-1]


def _pa_num_blocks(args: argparse.Namespace) -> int:
    num_gpu_blocks_override = getattr(args, "num_gpu_blocks_override", None)
    if num_gpu_blocks_override is not None:
        return max(1, num_gpu_blocks_override)
    return max(
        1,
        ((args.seq_len + args.block_size - 1) // args.block_size)
        * args.max_num_seqs,
    )


def _normalize_cache_dtype(value: str | None, *, default: str = "float32") -> str:
    if value is None:
        value = default
    normalized = str(value).lower()
    aliases = {
        "fp32": "float32",
        "float32": "float32",
        "torch.float32": "float32",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "torch.bfloat16": "bfloat16",
    }
    if normalized not in aliases:
        raise ValueError(
            "GDN recurrent cache dtype must be float32 or bfloat16, "
            f"got {value}"
        )
    return aliases[normalized]


def _recurrent_cache_dtype(args: argparse.Namespace) -> str:
    dtype = _normalize_cache_dtype(
        args.hybrid_gdn_recurrent_cache_dtype or args.gdn_recurrent_cache_dtype,
        default="float32",
    )
    if args.enable_hybrid_apc and args.hybrid_cache_mode == "all" and dtype != "float32":
        raise ValueError(
            "Hybrid APC all-mode requires float32 recurrent GDN checkpoint "
            "cache state; use --gdn-recurrent-cache-dtype float32"
        )
    return dtype


def _override_config(args: argparse.Namespace) -> dict:
    _validate_hybrid_apc_args(args)
    cte_buckets = _cte_buckets(args)
    max_cte_bucket = cte_buckets[-1]
    prefill_group_size = _effective_prefill_group_size(args, cte_buckets)
    context_encoding_bucket_pairs = _parse_bucket_pairs(
        args.context_encoding_bucket_pairs
    )
    token_generation_buckets = _token_generation_buckets(args)
    token_generation_batches = _token_generation_batches(args)
    recurrent_cache_dtype = _recurrent_cache_dtype(args)
    conv_cache_dtype = args.hybrid_gdn_conv_cache_dtype or args.gdn_conv_cache_dtype
    neuron_config = {
        "tp_degree": args.tensor_parallel_size,
        "batch_size": args.max_num_seqs,
        "ctx_batch_size": args.ctx_batch_size,
        "tkg_batch_size": args.max_num_seqs,
        "seq_len": args.seq_len,
        "max_length": args.seq_len,
        "max_context_length": max_cte_bucket,
        "context_encoding_buckets": cte_buckets,
        "token_generation_buckets": token_generation_buckets,
        "enable_bucketing": len(cte_buckets) > 1
        or len(token_generation_buckets) > 1,
        "logical_nc_config": args.logical_nc_config,
        "torch_dtype": "bfloat16",
        "save_sharded_checkpoint": True,
        "gdn_checkpoint_interval": args.gdn_checkpoint_interval,
        "max_gdn_checkpoint_slots": args.max_gdn_checkpoint_slots,
        "gdn_recurrent_cache_dtype": recurrent_cache_dtype,
        "gdn_conv_cache_dtype": conv_cache_dtype,
        "hybrid_recurrent_cache_dtype": recurrent_cache_dtype,
        "hybrid_conv_cache_dtype": conv_cache_dtype,
        "hybrid_cache_mode": args.hybrid_cache_mode,
    }
    if args.async_mode:
        neuron_config["async_mode"] = True
    if token_generation_batches is not None:
        neuron_config["token_generation_batches"] = token_generation_batches
    if (
        args.enable_prefix_caching
        or args.enable_hybrid_apc
        or args.enable_vllm_chunked_prefill
    ):
        neuron_config["is_block_kv_layout"] = True
        neuron_config["pa_block_size"] = args.block_size
        neuron_config["pa_num_blocks"] = _pa_num_blocks(args)
    uses_prefix_cte_contract = context_encoding_bucket_pairs is not None
    if args.enable_prefix_caching or args.enable_hybrid_apc or uses_prefix_cte_contract:
        neuron_config["is_prefix_caching"] = True
        if context_encoding_bucket_pairs is not None:
            neuron_config["context_encoding_bucket_pairs"] = (
                context_encoding_bucket_pairs
            )
    if args.enable_vllm_chunked_prefill:
        neuron_config.update(
            {
                "chunked_prefill_config": {
                    "max_num_seqs": args.max_num_seqs,
                    "tkg_model_enabled": True,
                    "kernel_q_tile_size": args.kernel_q_tile_size,
                    "kernel_kv_tile_size": args.kernel_kv_tile_size,
                },
            }
        )
    return {
        "max_prompt_length": max_cte_bucket,
        "use_hybrid_apc_manager": args.enable_hybrid_apc,
        "use_text_only_cte_inputs": args.text_only_cte,
        "use_compact_cte_attention_mask": args.compact_cte_attention_mask,
        "use_cold_zero_conv_fast_path": args.cold_zero_conv_fast_path,
        "gdn_checkpoint_interval": args.gdn_checkpoint_interval,
        "max_gdn_checkpoint_slots": args.max_gdn_checkpoint_slots,
        "gdn_recurrent_cache_dtype": recurrent_cache_dtype,
        "gdn_conv_cache_dtype": conv_cache_dtype,
        "hybrid_recurrent_cache_dtype": recurrent_cache_dtype,
        "hybrid_conv_cache_dtype": conv_cache_dtype,
        "hybrid_cache_mode": args.hybrid_cache_mode,
        "hybrid_cache_prefix_boundary_only": args.hybrid_cache_prefix_boundary_only,
        "hybrid_cache_block_boundary_only": args.hybrid_cache_prefix_boundary_only,
        "hybrid_cache_validate_exact": args.hybrid_cache_validate_exact,
        "hybrid_apc_require_vllm_metadata": args.hybrid_apc_require_vllm_metadata,
        "hybrid_apc_allow_local_hash_fallback": not args.hybrid_apc_require_vllm_metadata,
        "hybrid_apc_require_attention_block_refs": args.hybrid_apc_require_vllm_metadata,
        "hybrid_apc_reject_unbacked_attention_hits": getattr(
            args,
            "hybrid_apc_reject_unbacked_attention_hits",
            True,
        ),
        "hybrid_apc_disable_unbacked_prefix_reads": getattr(
            args,
            "hybrid_apc_disable_unbacked_prefix_reads",
            False,
        ),
        "hybrid_apc_enable_backed_prefix_reads": getattr(
            args,
            "hybrid_apc_enable_backed_prefix_reads",
            False,
        ),
        "hybrid_apc_max_backed_prefix_read_len": getattr(
            args,
            "hybrid_apc_max_backed_prefix_read_len",
            0,
        ),
        "hybrid_apc_prefill_chunk_tokens": (
            prefill_group_size
            if args.enable_hybrid_apc and args.enable_vllm_chunked_prefill
            else 0
        ),
        "qwen_prefill_group_size": prefill_group_size,
        "use_qwen_hybrid_chunked_prefill": args.enable_vllm_chunked_prefill,
        "use_qwen_hybrid_chunked_prefill_nki": args.enable_vllm_chunked_prefill,
        "override_neuron_config": neuron_config,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-artifacts", default=None)
    parser.add_argument("--prompt", default="What is 17 * 23? Answer with the number only.")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--enable-vllm-chunked-prefill", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-hybrid-apc", action="store_true")
    parser.add_argument("--mamba-cache-mode", default=None)
    parser.add_argument("--mamba-cache-dtype", default=None)
    parser.add_argument("--mamba-ssm-cache-dtype", default=None)
    parser.add_argument("--gdn-checkpoint-interval", type=int, default=256)
    parser.add_argument("--max-gdn-checkpoint-slots", type=int, default=8)
    parser.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    parser.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    parser.add_argument("--hybrid-gdn-recurrent-cache-dtype", default=None)
    parser.add_argument("--hybrid-gdn-conv-cache-dtype", default=None)
    parser.add_argument("--hybrid-cache-mode", default="all")
    parser.add_argument(
        "--hybrid-cache-prefix-boundary-only",
        "--hybrid-cache-block-boundary-only",
        dest="hybrid_cache_prefix_boundary_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hybrid-cache-validate-exact", action="store_true")
    parser.add_argument(
        "--hybrid-apc-require-vllm-metadata",
        action="store_true",
        help=(
            "Require serving-provided vLLM cumulative prefix hashes and attention "
            "block refs instead of the local token-hash validation fallback."
        ),
    )
    parser.add_argument(
        "--hybrid-apc-reject-unbacked-attention-hits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reject attention prefix-cache hits that do not have a matching GDN "
            "checkpoint. Disable only for controlled plumbing/debug isolation."
        ),
    )
    parser.add_argument(
        "--hybrid-apc-disable-unbacked-prefix-reads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Make vLLM skip prefix-cache reads for Qwen Hybrid APC until scheduler "
            "GDN checkpoint metadata is available."
        ),
    )
    parser.add_argument(
        "--hybrid-apc-enable-backed-prefix-reads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow vLLM prefix-cache reads when both attention KV and GDN "
            "checkpoint state are backed by a CTE artifact compiled for that "
            "contract."
        ),
    )
    parser.add_argument(
        "--hybrid-apc-max-backed-prefix-read-len",
        type=int,
        default=0,
        help=(
            "Optional safety cap for backed prefix reads. Prefix reads above this "
            "token length are disabled even when a GDN checkpoint is registered."
        ),
    )
    parser.add_argument(
        "--hybrid-apc-prefill-chunk-tokens",
        type=int,
        default=0,
        help=(
            "Opt into larger vLLM chunked-prefill chunks for Hybrid APC. The "
            "value must be a compiled CTE bucket and a multiple of "
            "--gdn-checkpoint-interval. Default 0 keeps conservative "
            "checkpoint-sized chunks."
        ),
    )
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--ctx-batch-size", type=int, default=1)
    parser.add_argument("--token-generation-buckets", nargs="+", default=None)
    parser.add_argument("--token-generation-batches", nargs="+", default=None)
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--cte-buckets", nargs="+", default=None)
    parser.add_argument("--context-encoding-bucket-pairs", nargs="+", default=None)
    parser.add_argument(
        "--cte-bucket-profile",
        choices=("single", "short", "general", "long", "262k"),
        default="single",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--kernel-q-tile-size", type=int, default=128)
    parser.add_argument("--kernel-kv-tile-size", type=int, default=1024)
    parser.add_argument(
        "--text-only-cte",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compact-cte-attention-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cold-zero-conv-fast-path",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    contrib_root = _contrib_root(args.repo_root)
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(contrib_root))
    os.environ["PYTHONPATH"] = (
        f"{script_dir}:{contrib_root}:{os.environ.get('PYTHONPATH', '')}"
    )
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    os.environ.setdefault("VLLM_PLUGINS", "neuron")
    if args.enable_hybrid_apc:
        os.environ.setdefault("QWEN36_HYBRID_APC_INSTALL_PATCH", "1")
    if args.enable_vllm_chunked_prefill:
        os.environ["DISABLE_NEURON_CUSTOM_SCHEDULER"] = "1"
    if args.hybrid_apc_disable_unbacked_prefix_reads:
        os.environ["QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS"] = "1"
    if args.compiled_artifacts:
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(
            Path(args.compiled_artifacts).expanduser().resolve()
        )
        _ensure_fp8_environment()

    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433
    from qwen36_hybrid_apc_scheduler_patch import (  # noqa: WPS433
        install_import_hook as install_hybrid_apc_scheduler_patch,
    )

    register_qwen35_config()
    install_hybrid_apc_scheduler_patch()

    from vllm import LLM, SamplingParams  # noqa: WPS433

    prompt = args.prompt
    if args.chat:
        from transformers import AutoTokenizer  # noqa: WPS433

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    additional_config = _override_config(args)
    print("VLLM_QWEN36_CONFIG", json.dumps(additional_config, sort_keys=True), flush=True)
    cte_buckets = _cte_buckets(args)
    max_cte_bucket = max(cte_buckets)

    llm_kwargs = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_vllm_chunked_prefill,
        "additional_config": additional_config,
    }
    recurrent_cache_dtype = _recurrent_cache_dtype(args)
    if args.enable_prefix_caching or args.enable_hybrid_apc:
        llm_kwargs["mamba_cache_mode"] = args.mamba_cache_mode or "all"
        llm_kwargs["mamba_ssm_cache_dtype"] = (
            args.mamba_ssm_cache_dtype or recurrent_cache_dtype
        )
    elif args.mamba_cache_mode is not None:
        llm_kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    if args.mamba_cache_dtype is not None:
        llm_kwargs["mamba_cache_dtype"] = args.mamba_cache_dtype
    if (
        args.mamba_ssm_cache_dtype is not None
        and "mamba_ssm_cache_dtype" not in llm_kwargs
    ):
        llm_kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype
    if (
        args.num_gpu_blocks_override is not None
        or args.enable_prefix_caching
        or args.enable_hybrid_apc
        or args.enable_vllm_chunked_prefill
    ):
        llm_kwargs["num_gpu_blocks_override"] = _pa_num_blocks(args)
    if (
        args.enable_prefix_caching
        or args.enable_hybrid_apc
        or args.enable_vllm_chunked_prefill
    ):
        llm_kwargs["block_size"] = args.block_size
    if args.enable_vllm_chunked_prefill:
        llm_kwargs["max_num_batched_tokens"] = _effective_prefill_group_size(
            args, cte_buckets
        )
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - start
    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids

    print("PROMPT", prompt)
    print("OUTPUT", text)
    print("TOKENS", list(token_ids))
    print("ELAPSED_SECONDS", f"{elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
