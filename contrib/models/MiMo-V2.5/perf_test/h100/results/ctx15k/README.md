# MiMo-V2.5 single-node H100 @ 15K input / 1K output

Matches the Trn2 port's **maximum compiled context (16K)** — 15000 input + 1000
output = 16000 — so these numbers are the H100 reference for the long-context
point Trn2 can actually serve. Concurrency 1 / 16 / 32.

Bench (per user template): `vllm bench serve --dataset-name random
--random-input-len 15000 --random-output-len 1000 --random-range-ratio 0.03
--num-prompts 2*C --max-concurrency C`. Both servers single-node 8xH100, FP8,
no speculative decode. Raw logs: `sglang_15k.log`, `vllm_15k.log`.

- **SGLang**: `run_sglang_h100.sh` shape (TP=8, DP=2 + DP-attention), but
  `--mem-fraction-static 0.9 --context-length 17408` for the 16K KV pool
  (max_total_num_tokens=678837).
- **vLLM**: `run_vllm_h100.sh` (TP=8 + EP + PR #42270, chunked prefill ON),
  `--max-model-len 17408`.

## Results (15K in / 1K out)

| Concurrency | Backend | Output tput (tok/s) | Total tput (tok/s) | TTFT median (ms) | TTFT P99 (ms) | TPOT median (ms) |
|---|---|---|---|---|---|---|
| 1  | SGLang | **94.6**  | **1544**  | 699  | 898   | 9.9  |
| 1  | vLLM   | 82.7      | 1350      | 6587 | 7487  | **5.5** |
| 16 | SGLang | **867.8** | **13912** | 1336 | 4501  | 16.7 |
| 16 | vLLM   | 712.5     | 11423     | 1132 | 10021 | 16.3 |
| 32 | SGLang | **1255.2**| **20080** | 1473 | 8828  | 22.1 |
| 32 | vLLM   | 1181.7    | 18904     | 2054 | 9957  | 24.1 |

All runs: 0 failures (2 / 32 / 64 prompts at c=1 / 16 / 32).

## Observations

- **SGLang wins output throughput at every concurrency** (1.1–1.2x vLLM), and its
  c=1 TTFT is ~10x lower (699 ms vs 6.6 s). SGLang chunks the 15K prefill more
  aggressively into the decode loop, so first token lands fast even single-stream.
- **vLLM has the lowest TPOT** (5.5 ms at c=1) — once decoding, it's the fastest
  per-token — but its c=1 TTFT is dominated by the 15K prefill (chunked-prefill
  interleaves it over many steps before the first output token).
- Throughput scales cleanly 1→32 for both (SGLang 94→1255, vLLM 83→1182 out
  tok/s); the crossover where they converge is c=32.
- At 15K/1K the workload is prefill-heavy (15:1 in:out); total throughput
  (~20K tok/s SGLang @ c=32) is dominated by prefill, which is why it's ~13x the
  output-token rate.

> These are the H100 reference numbers for a direct comparison against the Trn2
> 16K-compiled port at 15K/1K. Language-only, no prefix caching, no spec decode.

## Cross-platform: H100 vs Trn2 (15K in / 1K out)

Trn2 numbers from the main README's "Long Context" section (trn2.48xlarge,
seq_len=16384, `tp=64, attention_dp=16, cp=16, moe_ep=64`, BS=32).

**Output throughput (tok/s):**

| Concurrency | Trn2 | H100 SGLang | H100 vLLM | SGLang / Trn2 |
|---|---:|---:|---:|---:|
| 1  | 4.49  | 94.6   | 82.7   | **21×** |
| 16 | 48.49 | 867.8  | 712.5  | **18×** |
| 32 | 74.01 | 1255.2 | 1181.7 | **17×** |

**TTFT median (ms, lower = faster first token):**

| Concurrency | Trn2 | H100 SGLang | H100 vLLM | Trn2 / SGLang |
|---|---:|---:|---:|---:|
| 1  | 6051  | 699  | 6587 | 8.7× slower |
| 16 | 17985 | 1336 | 1132 | 13× slower |
| 32 | 23941 | 1473 | 2054 | 16× slower |

At 16K context the H100/Trn2 gap widens to **~17–21× throughput** (vs ~7–11× at
the short 900/90 shape) — long context is where Trn2 is hit hardest. Both
platforms are prefill-bound here, but Trn2 more severely: it must serve 16K via
`cp=dp=16` with `chunked_prefill=false`, so 16K prefills run one-at-a-time and
queue (main-README Trn2 P99 TTFT reaches ~187 s at c=32). A single 16K prefill is
~6.06 s on Trn2 vs ~0.7 s on H100 SGLang. Trn2 also cannot currently fit 32K
(HBM overflow), while single-node H100 can (see `../../` 32K exploration).
