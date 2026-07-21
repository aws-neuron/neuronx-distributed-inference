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
| **SGLang** (`run_sglang_h100.sh`) | ✅ **Works.** `--tp 8 --attention-context-parallel-size 2`, no DeepEP, no spec decode. |
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

### Why SGLang uses `--attention-context-parallel-size 2`

SGLang also requires the *effective* attention TP size to be exactly 4 (the
qkv is TP=4-interleaved), so plain `--tp 8` is rejected. Two 8-GPU shapes give
effective attn TP = 4:

- **DP=2 + `--enable-dp-attention`** (the model-card reference command): but
  single-node it **deadlocks** the shared-MoE / lm-head collective when a
  request occupies only one DP group (idle group never launches its matching
  forward -> 300 s scheduler watchdog -> crash). The reference relies on
  `--moe-a2a-backend deepep` to change that collective, but **DeepEP needs a
  working nvshmem RDMA transport, which this plain P5 box's NICs (`rdmapXXs0`,
  not `mlx5`) do not provide** — IBGDA init fails and the forward hangs (tested
  `low_latency` + `normal` + `NVSHMEM_REMOTE_TRANSPORT=none` + `IBGDA=0`). DeepEP
  *does* work on `p5en.48xlarge` (mlx5 EFA + a custom Mooncake/nvshmem image);
  see `xiaomi_datalab/mimo_v25`.
- **`--attention-context-parallel-size 2`**: effective attn TP = tp/cp = 4, MoE
  shards over TP=8, no DP idle-group problem, no DeepEP. **Works out of the box.
  This is the default.**

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
bash bench_all.sh 30000 sglang_cp2
```

`bench_all.sh` is a thin loop over `run_bench_single.sh` (c=1 x16 prompts,
c=16/c=32 x96 prompts), the same client and 900/90 token random dataset as the
Trn2 table. Per-run logs land in `results/`.

For a single ad-hoc run:

```bash
docker run --rm --network host -v /opt/dlami/nvme/models:/wk -v $PWD:/sc \
    --entrypoint bash sglang-efa:latest -c '
      export SERVED_MODEL_NAME=MiMo-V2.5 TOKENIZER_PATH=/wk/MiMo-V2.5 PORT=30000 \
             CONFIG_NAME=sglang_cp2 RESULTS_DIR=/wk/bench_results/mimo_v2_5_h100
      CONCURRENCY=32 NUM_PROMPTS=96 bash /sc/run_bench_single.sh'
```

## Performance (SGLang, single-node 8xH100, FP8, TP=8 + attn-CP=2)

Input/output: 900 / 90 tokens (random dataset), no speculative decoding — same
recipe and shape as the Trn2 table in the main README, for direct comparison.
Raw logs in `results/sglang_cp2_c{1,16,32}.txt`.

| Concurrency | Output throughput (tok/s) | Total throughput (tok/s) | TPOT median (ms) | TTFT median (ms) | TTFT P99 (ms) |
|---|---|---|---|---|---|
| 1  | 125.69  | 1378.59  | 6.93  | 95   | 157 |
| 16 | 792.68  | 8699.94  | 17.66 | 186  | 726 |
| 32 | 1570.83 | 17240.36 | 17.78 | 237  | 293 |

### H100 (SGLang) vs Trn2 (vLLM-Neuron) — same 900/90 shape

| Concurrency | H100 out tok/s | Trn2 out tok/s | H100 / Trn2 |
|---|---:|---:|---:|
| 1  | 125.69  | 15.88  | **7.9×** |
| 16 | 792.68  | 113.92 | **7.0×** |
| 32 | 1570.83 | 147.39 | **10.7×** |

(Trn2 numbers from the main README's Performance section: BS=32, TP=64 /
moe_ep=64, CB + bucketing.) The H100 gap is largest at c=32 because SGLang's
median ITL stays ~13–18 ms across concurrency while the Trn2 BS=32 TKG NEFF runs
at a fixed ~58 ms/token regardless of occupancy.

> **Note:** these are the *language-only, no-spec-decode* baselines for
> apples-to-apples comparison with Trn2. Turning on EAGLE (`SPEC=1`) would raise
> H100 decode throughput further; the model card's full recipe also enables it.
