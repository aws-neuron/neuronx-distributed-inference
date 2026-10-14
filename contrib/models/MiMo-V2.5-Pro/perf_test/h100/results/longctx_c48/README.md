# MiMo-V2.5-Pro 2-node H100 long-context @ concurrency 48

How long an input can 2× p5.48xlarge (16× H100-80GB) sustain at **concurrency 48**
(Pro's minimum working batch: 384 experts / top-8 routing = 48)? The default
Pro bench used ISL=360/OSL=120 (seq_len 512) — this sweep pushes the input length.

Server: `run_sglang_h100_multinode.sh` on both nodes (TP=16, DP=2 + DP-attention,
EP=16), **`MEM_FRAC=0.90`** and **chunked-prefill auto-reduced to 4096** (DP
attention lowers it). Bench: `bench_ctx_c48.sh 2048 4096 8192 12288 16384 24576
32768` (96 prompts, OSL=128). Raw logs: `pro_2node_c48_sweep.txt`, `isl*.txt`.

## Result: c=48 sustains up to 32K input (0 failures)

| ISL (input) | Succeeded | Total tput (tok/s) | TTFT median (s) | TTFT P99 (s) | TPOT median (ms) |
|---|---|---|---|---|---|
| 2,048  | 96/96 ✅ | 9359  | 1.8  | 5.6  | 67  |
| 4,096  | 96/96 ✅ | 9738  | 5.6  | 16.3 | 78  |
| 8,192  | 96/96 ✅ | 13140 | 11.7 | 25.5 | 113 |
| 12,288 | 96/96 ✅ | 13745 | 24.1 | 34.2 | 132 |
| 16,384 | 96/96 ✅ | 13872 | 37.1 | 49.9 | 140 |
| 24,576 | 96/96 ✅ | 14485 | 62.2 | 72.8 | 125 |
| 32,768 | 96/96 ✅ | 14791 | 91.3 | 99.7 | 112 |

**Every level to 32K completed with zero failures** — at ISL=32K that's 96 ×
32768 = 3.15M total input tokens pushed through a 148K-token KV pool.

## Why it works despite a small KV pool

- **KV pool is small because weights dominate.** Pro's FP8 weights are ~963 GB /
  16 GPUs = **60 GB/GPU**, leaving only ~11 GB/GPU for KV at mem-fraction 0.90 →
  `max_total_num_tokens = 148672` (~3K tokens/req if split evenly across 48).
- **Hybrid attention makes long inputs cheap.** 60 of 70 layers are sliding-window
  (window 128) with window-capped KV; only 10 are full attention. A long prompt's
  KV is dominated by those 10 layers, far below the naive per-token estimate, so
  requests fit well beyond the even-split figure.
- **SGLang queues the rest.** At high ISL the resident set is capped by the KV
  pool; excess requests queue, so throughput stays flat (~14K tok/s, prefill-bound)
  while **TTFT grows** (1.8 s at 2K → 91 s at 32K). No request is dropped.

## ⚠️ Memory tuning is load-bearing (mem-fraction + chunked-prefill)

The failure that gates this is **prefill activation OOM, not KV capacity**:

| mem-fraction | KV pool | idle free/GPU | c=48 prefill result |
|---|---|---|---|
| 0.95 | 378,240 | 2.43 GB | ✅ loads, ❌ **OOM on first c=48 prefill batch** |
| 0.92 | 240,512 | 4.66 GB | ✅ loads, ❌ **OOM at c=48 ISL=2048** |
| **0.90** | 148,672 | 6.14 GB | ✅ **stable to 32K** (chunked-prefill 4096) |

Raising mem-fraction grows the *paper* KV pool but shrinks the headroom that 48
concurrent prefills need for activations — 0.95/0.92 crash with
`torch.OutOfMemoryError` the moment a real c=48 batch runs. **0.90 + the
DP-attention-reduced 4096 chunked-prefill size is the stable operating point.**

> **Prerequisite:** both nodes needed an NVIDIA Fabric Manager fix first — the
> DLAMI's background update had bumped fabricmanager to 610 while the driver
> stayed at 595, breaking multi-GPU CUDA init (`Error 802: system not yet
> initialized`). Reinstall the matching `nvidia-fabricmanager-595=595.71.05-*`
> and reboot on **both** nodes before launching.
