# MiMo-V2.5 single-node H100 — prefill throughput (16K input / 10 output)

Isolates **prefill** performance: 16000 input, only 10 output tokens, so the run
is dominated by context encoding and total throughput ≈ prefill token rate.
Concurrency 1 / 16 / 32. Same servers as the 15K/1K bench (see `../ctx15k/`).

Bench: `vllm bench serve --dataset-name random --random-input-len 16000
--random-output-len 10 --random-range-ratio 0.03 --num-prompts 2*C
--max-concurrency C`. Single-node 8xH100, FP8, no spec decode. Ran the two
backends in parallel on P5-1 (SGLang) and P5-2 (vLLM). Raw logs:
`sglang_prefill.log`, `vllm_prefill.log`.

## Results (16K in / 10 out ≈ pure prefill)

| Concurrency | Backend | Prefill throughput (tok/s) | Batch wall (s) | TTFT median (ms) | TTFT P99 (ms) |
|---|---|---|---|---|---|
| 1  | **SGLang** | **20539** | 1.6  | 709  | 913   |
| 1  | vLLM       | 5875      | 5.5  | 2716 | 4980  |
| 16 | **SGLang** | **51958** | 9.9  | 2806 | 4810  |
| 16 | vLLM       | 29913     | 17.2 | 3917 | 8520  |
| 32 | **SGLang** | **52984** | 19.4 | 5221 | 9444  |
| 32 | vLLM       | 39605     | 25.9 | 8954 | 11796 |

All runs: 0 failures.

## Observations

- **SGLang prefill is markedly faster than vLLM here** — 3.5x at c=1 (20.5K vs
  5.9K tok/s), 1.7x at c=16, 1.3x at c=32. The gap is widest single-stream and
  narrows as concurrency fills the pipeline.
- **SGLang prefill throughput saturates by c=16** (~52K tok/s; c=32 barely higher
  at 53K) — the H100 prefill compute ceiling for this model. vLLM is still
  climbing at c=32 (30K → 40K), i.e. it needs more concurrency to hit peak.

## Why vLLM is slower — it's the DiffKV kernel, not a config gap

Investigated the 3.5x c=1 gap. It is **not** an unaligned setting:

- **Both backends chunk prefill at the same 8192** (`chunked_prefill_size=8192`
  on SGLang, forced by DP-attention; `max_num_batched_tokens=8192` vLLM default).
  So the chunk size is identical.
- **vLLM must use the `FLASH_ATTN_DIFFKV` attention backend** for MiMo-V2.5 (log:
  `Using FLASH_ATTN_DIFFKV for attention` / `Diff-KV with sinks: upgrading
  FlashAttention 3 -> 4`). This is the kernel that handles V2.5's asymmetric head
  dims (Q/K=192, V=128) + attention-sink bias; it's the new path PR #42270 just
  made loadable.
- **That DiffKV kernel is pinned to the 8192 chunk and cannot be enlarged.**
  Setting `--max-num-batched-tokens 16384` (to encode the 16K prompt in one pass)
  makes vLLM fall back to the plain FlashAttention path, which then crashes at
  init: `AssertionError: TP size must evenly split the number of KV heads` (V2.5
  has 4 full-attn KV heads, TP=8 can't split them — the exact reason plain vLLM
  needs DiffKV in the first place). So the prefill chunk can't be tuned up.

**Conclusion:** vLLM's lower prefill throughput is the efficiency of its DiffKV
kernel (still a young code path — only just enabled by PR #42270) versus SGLang's
mature `fa3` backend, at the same 8192 chunk. It is a kernel-maturity gap for
asymmetric-head-dim / hybrid-attention models, not a misconfiguration. SGLang is
the better choice for prefill-heavy long-context serving of MiMo-V2.5 today.

- The 16K prefill compute ceiling (~53K tok/s, SGLang) is why the 15K/1K total
  throughput in `../ctx15k/` tops out around 20K tok/s once the 1K decode mixes in.

> H100 reference for the prefill-bound regime at Trn2's 16K compiled context.
