# H100 GPU baseline for MiMo-V2.5

Scripts to serve the **official HuggingFace OCP-FP8 checkpoint**
(`XiaomiMiMo/MiMo-V2.5`) on a **single** 8xH100-80GB node, for a cross-platform
throughput comparison against the Trn2 Neuron port.

## Single node (unlike MiMo-V2.5-Pro)

V2.5's FP8 weights are ~295 GB, which fits on one 8xH100 node (640 GB), so no
cross-node NCCL / EFA is needed (the Pro variant is ~963 GB and requires 2
nodes). We still reuse the `sglang-efa` / `vllm-efa` images built for the Pro
work (strict supersets of the stock images); EFA just goes unused here.

## TL;DR — use SGLang; vLLM is broken for this model

| Backend | Status |
|---|---|
| **SGLang** (`run_sglang_h100.sh`) | ✅ **Works.** Cookbook `--tp 8 --dp 2 --enable-dp-attention` (no DeepEP, no spec decode). |
| **vLLM** (`run_vllm_h100.sh`) | ❌ **Broken** (upstream bug, 0.25.1 **and** nightly). Kept only to document the reference command + reproduce the bug. |

### Why vLLM fails

vLLM crashes loading the fused `qkv_proj` on the **sliding-window** layers:

```
RuntimeError: The size of tensor a (1856) must match the size of tensor b (1792)
              at non-singleton dimension 0
              (vllm/model_executor/models/mimo_v2.py::_shard_fp8_qkv_proj)
```

A SWA-layer fused-qkv group is `(64/8)*192 + 192 + 128 = 1856` rows = **14.5**
blocks of the 128-row FP8 scale, so it is **not 128-aligned**. vLLM's loader
slices the scale per KV-head group (`scale_rows_per_group = 116 // 8 = 14` ->
1792 rows) and multiplies it against the 1856-row weight group. Because
MiMo-V2.5 has only **4 full-attention KV heads** (vs 8 for the Pro), no 8-GPU TP
config avoids the buggy path:

- `TP=8` — can't start: "TP size must evenly split the number of KV heads" (4).
- `TP=4` / `TP=2` / `DP=2 x TP=4` — each rank owns 2 SWA KV heads -> the `g>1`
  de-interleave path -> the 1856-vs-1792 crash.

(The Pro variant never hits this: with 8 KV heads and TP=8 each rank gets 1 KV
head and takes the trivial `kv_heads_per_rank == 1` fast path.) A shape-only
monkeypatch makes the server start but produces gibberish, so it is not a real
fix. This should be reported/fixed upstream in vLLM.

### SGLang parallelism: DP=2 (cookbook, default) — no DeepEP needed

SGLang requires the *effective* attention TP size to be exactly 4 (the qkv is
TP=4-interleaved), so plain `--tp 8` is rejected. Two 8-GPU shapes give
effective attn TP = 4, and **both work single-node with no DeepEP**:

- **DP=2 + DP-attention (default)** — the SGLang cookbook / model-card command:
  `--tp 8 --dp 2 --enable-dp-attention --enable-dp-lm-head --mm-enable-dp-encoder
  --mem-fraction-static 0.65`. Effective attn TP = tp/dp = 4. DeepEP is **not**
  required — the three DP flags together (dp-attention + dp-lm-head +
  dp-encoder) at mem-fraction 0.65 make the shared-MoE / lm-head collective work.
  (An earlier attempt with only `--enable-dp-attention` and mem-fraction 0.9 hung
  the collective; adding the other two DP flags + 0.65 fixes it — the full
  cookbook command is what to use.)
- **`--attention-context-parallel-size 2`** (`DP=1 ATTN_CP=2`): effective attn
  TP = tp/cp = 4, MoE shards over TP=8, no DP flags. Splits attention along the
  sequence so a single request uses all 8 GPUs, measured **slightly faster at
  low concurrency** (see table). Kept as a documented alternative.

> **On DeepEP**: the cookbook *also* shows a DeepEP variant
> (`--moe-a2a-backend deepep`). DeepEP is not needed here, and on this plain P5
> box it doesn't init anyway (its nvshmem RDMA/IBGDA transport wants `mlx5` NICs;
> this box has `rdmapXXs0` — set `DEEPEP=1` only on hosts where it works, e.g.
> `p5en.48xlarge` with mlx5 EFA, cf. `xiaomi_datalab/mimo_v25`).

## Usage

Download the checkpoint (~295 GB; uses the `/opt/pytorch` env on the P5 box):

```bash
/opt/pytorch/bin/hf download XiaomiMiMo/MiMo-V2.5 \
    --local-dir /opt/dlami/nvme/models/MiMo-V2.5 --max-workers 16
```

Serve + bench (SGLang, the working path):

```bash
# Terminal 1: launch (first start ~4 min: weight load + DeepGEMM warmup + CUDA
# graph capture). Wait for "The server is fired up and ready to roll!".
bash run_sglang_h100.sh

# Terminal 2: bench at c=1/16/32 (MiMo-V2.5 bs=32, so cap concurrency at 32).
# The bench client runs inside the vllm-efa container (has `vllm bench serve`).
bash bench_all.sh 30000 sglang_dp2
```

`bench_all.sh` is a thin loop over `run_bench_single.sh` (c=1 x16 prompts,
c=16/c=32 x96 prompts), the same client and 900/90 token random dataset as the
Trn2 table. Per-run logs land in `results/`.

For a single ad-hoc run:

```bash
docker run --rm --network host -v /opt/dlami/nvme/models:/wk -v $PWD:/sc \
    --entrypoint bash sglang-efa:latest -c '
      export SERVED_MODEL_NAME=MiMo-V2.5 TOKENIZER_PATH=/wk/MiMo-V2.5 PORT=30000 \
             CONFIG_NAME=sglang_dp2 RESULTS_DIR=/wk/bench_results/mimo_v2_5_h100
      CONCURRENCY=32 NUM_PROMPTS=96 bash /sc/run_bench_single.sh'
```

## Performance (SGLang, single-node 8xH100, FP8)

Input/output: 900 / 90 tokens (random dataset), no speculative decoding — same
recipe and shape as the Trn2 table in the main README, for direct comparison.

**DP=2 (cookbook default)** — raw logs `results/sglang_dp2_c{1,16,32}.txt`:

| Concurrency | Output throughput (tok/s) | Total throughput (tok/s) | TPOT median (ms) | TTFT median (ms) | TTFT P99 (ms) |
|---|---|---|---|---|---|
| 1  | 91.10   | 999.23   | 9.78  | 112 | 199 |
| 16 | 702.18  | 7706.69  | 20.33 | 210 | 735 |
| 32 | 1420.97 | 15595.60 | 18.63 | 299 | 440 |

**attn-CP=2 alternative** (`DP=1 ATTN_CP=2`) — raw logs
`results/sglang_cp2_c{1,16,32}.txt`. Slightly faster (a single request uses all
8 GPUs instead of 4 per DP group):

| Concurrency | Output throughput (tok/s) | Total throughput (tok/s) | TPOT median (ms) | TTFT median (ms) | TTFT P99 (ms) |
|---|---|---|---|---|---|
| 1  | 125.69  | 1378.59  | 6.93  | 95   | 157 |
| 16 | 792.68  | 8699.94  | 17.66 | 186  | 726 |
| 32 | 1570.83 | 17240.36 | 17.78 | 237  | 293 |

### H100 (SGLang) vs Trn2 (vLLM-Neuron) — same 900/90 shape

| Concurrency | H100 DP=2 out tok/s | H100 CP=2 out tok/s | Trn2 out tok/s | DP=2 / Trn2 |
|---|---:|---:|---:|---:|
| 1  | 91.10   | 125.69  | 15.88  | **5.7×** |
| 16 | 702.18  | 792.68  | 113.92 | **6.2×** |
| 32 | 1420.97 | 1570.83 | 147.39 | **9.6×** |

(Trn2 numbers from the main README's Performance section: BS=32, TP=64 /
moe_ep=64, CB + bucketing.) The H100 gap is largest at c=32 because SGLang's
median ITL stays ~15–20 ms across concurrency while the Trn2 BS=32 TKG NEFF runs
at a fixed ~58 ms/token regardless of occupancy.

> **Note:** these are the *language-only, no-spec-decode* baselines for
> apples-to-apples comparison with Trn2. Turning on EAGLE (`SPEC=1`) would raise
> H100 decode throughput further; the model card's full recipe also enables it.
