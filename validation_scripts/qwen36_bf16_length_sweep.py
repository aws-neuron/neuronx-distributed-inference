#!/usr/bin/env python3
"""Sweep no-prefix Qwen3.6 BF16 prompt lengths and print raw debug markers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        out.extend(int(token) for token in value.replace(",", " ").split())
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-artifacts", required=True)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", nargs="+", default=["0,1,2,4,8,16,24,32,40,48,56,64"])
    parser.add_argument("--line", default="System: answer deterministically.\n")
    parser.add_argument("--suffix", default="\nUser: What is 17 * 23?\nAssistant:")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--cte-buckets", nargs="+", default=["512"])
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--ctx-batch-size", type=int, default=1)
    parser.add_argument("--skip-fp8-env", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    qwen_root = repo_root / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(qwen_root / "vllm"))
    sys.path.insert(0, str(qwen_root))

    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    os.environ.setdefault("VLLM_PLUGINS", "neuron")
    os.environ["NEURON_COMPILED_ARTIFACTS"] = str(
        Path(args.compiled_artifacts).expanduser().resolve()
    )
    if not args.skip_fp8_env:
        os.environ.setdefault("XLA_HANDLE_SPECIAL_SCALAR", "1")
        os.environ.setdefault("UNSAFE_FP8FNCAST", "1")

    from transformers import AutoTokenizer  # noqa: WPS433
    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433
    import run_offline_inference as runner  # noqa: WPS433
    from vllm import LLM, SamplingParams  # noqa: WPS433

    register_qwen35_config()
    cte_buckets = [str(bucket) for bucket in _parse_int_list(args.cte_buckets)]
    runner_args = SimpleNamespace(
        enable_hybrid_apc=False,
        hybrid_cache_mode="all",
        gdn_checkpoint_interval=256,
        max_gdn_checkpoint_slots=8,
        block_size=256,
        enable_prefix_caching=False,
        enable_vllm_chunked_prefill=False,
        cte_bucket_profile="single",
        cte_buckets=cte_buckets,
        cte_bucket=int(cte_buckets[-1]),
        seq_len=args.seq_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=1,
        ctx_batch_size=args.ctx_batch_size,
        logical_nc_config=args.logical_nc_config,
        hybrid_gdn_recurrent_cache_dtype=None,
        gdn_recurrent_cache_dtype="float32",
        hybrid_gdn_conv_cache_dtype=None,
        gdn_conv_cache_dtype="bfloat16",
        kernel_q_tile_size=128,
        kernel_kv_tile_size=1024,
        text_only_cte=True,
        compact_cte_attention_mask=True,
        cold_zero_conv_fast_path=False,
        hybrid_cache_prefix_boundary_only=True,
        hybrid_cache_validate_exact=False,
        hybrid_apc_require_vllm_metadata=False,
        num_gpu_blocks_override=None,
    )
    llm = LLM(
        model=str(Path(args.model_path).expanduser().resolve()),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=1,
        max_model_len=args.seq_len,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        additional_config=runner._override_config(runner_args),
    )
    sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    for repeats in _parse_int_list(args.repeats):
        prompt = args.line * repeats + args.suffix
        token_count = len(tokenizer(prompt).input_ids)
        print(f"SWEEP_CASE repeats={repeats} tokens={token_count}", flush=True)
        outputs = llm.generate([prompt], sampling)
        token_ids = list(outputs[0].outputs[0].token_ids)
        print(
            f"SWEEP_RESULT repeats={repeats} tokens={token_count} "
            f"output_tokens={token_ids}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
