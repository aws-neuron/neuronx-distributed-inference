# MiMo-V2.5-Pro 2-node H100 @ 4K input / 128 output — c=1/16/48

4K-input point to align with the Trn2 Pro 4K recompile. Concurrency 1 / 16 / 48
(48 = Pro's minimum working batch: 384 experts / top-8 = 48). 2× p5.48xlarge
(16× H100), TP=16, DP=2 + DP-attention, EP=16. Raw log: `pro_4k_sweep.log`.

Bench: `vllm bench serve --dataset-name random --random-input-len 4096
--random-output-len 128 --random-range-ratio 0.03 --num-prompts 2*C
--max-concurrency C`.

## Results (4K in / 128 out)

| Concurrency | Succeeded | Output tput (tok/s) | Total tput (tok/s) | TTFT median (ms) | TTFT P99 (ms) | TPOT median (ms) |
|---|---|---|---|---|---|---|
| 1  | 2/2 ✅   | 29.4  | 992   | 629  | 703   | 29.3  |
| 16 | 32/32 ✅ | 235.2 | 7779  | 1723 | 4262  | 51.3  |
| 48 | 96/96 ✅ | 368.5 | 12133 | 3055 | 10918 | 100.1 |

All 0 failures.

## ⚠️ Memory tuning required (mem-fraction 0.90 + small chunked-prefill)

Same prefill-activation-OOM constraint as the c=48 long-context sweep: Pro's FP8
weights are ~60 GB/GPU, leaving little headroom. The gating knob is the
**chunked-prefill size**:

- `mem-fraction 0.90` with the default `chunked-prefill 16384` (DP-attention
  auto-reduces to 16384) **OOMs at c=16** (`torch.OutOfMemoryError`, tried 2.97 GB,
  2.48 GB free) — 16 requests prefilling 4K each at once overflows.
- **`CHUNK=4096` (DP-attention halves it to 2048)** keeps the prefill activation
  peak low enough to run c=1/16/48 stably. This is the stable operating point.

So launch with `MEM_FRAC=0.90 CHUNK=4096` on `run_pro_d2.sh` (or set
`--chunked-prefill-size 4096` on the multinode script).

## Observations

- Throughput scales 29 → 235 → 368 out tok/s (992 → 7779 → 12133 total) across
  c=1/16/48. Total throughput is prefill-dominated (4K:128 ≈ 32:1 in:out).
- TTFT stays modest at 4K (629 ms → 3.1 s median) — far better than the long-context
  sweep (16K was 37 s @ c=48), since 4K prefill is cheap and mostly fits without
  heavy queueing.
- TPOT grows 29 → 100 ms with concurrency as the decode batch fills (Pro's 148K
  KV pool is shared across the 48 slots).

> Cross-platform reference for the Trn2 Pro 4K-compiled port at c=1/16/48.
