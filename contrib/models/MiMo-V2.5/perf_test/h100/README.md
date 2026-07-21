# H100 GPU baseline for MiMo-V2.5

Scripts to serve the **official HuggingFace OCP-FP8 checkpoint**
(`XiaomiMiMo/MiMo-V2.5`) on a **single** 8xH100-80GB node, for a cross-platform
throughput comparison against the Trn2 Neuron port.

## Single node (unlike MiMo-V2.5-Pro)

V2.5's FP8 weights are ~295 GB, which fits on one 8xH100 node (640 GB), so no
cross-node NCCL / EFA is needed (the Pro variant is ~963 GB and requires 2
nodes). We still reuse the `sglang-efa` / `vllm-efa` images built for the Pro
work (strict supersets of the stock images); EFA just goes unused here.

## TL;DR — both backends work; vLLM needs PR #42270

| Backend | Status |
|---|---|
| **SGLang** (`run_sglang_h100.sh`) | ✅ **Works out of the box.** Cookbook `--tp 8 --dp 2 --enable-dp-attention` (no DeepEP, no spec decode). |
| **vLLM** (`run_vllm_h100.sh`) | ✅ **Works with vLLM PR #42270** (not yet in any released image; the script bind-mounts the PR's two model files into the nightly image). Reference command `--tp 8 --enable-expert-parallel`. |

### vLLM needs PR #42270 (fused-qkv FP8 loader fix)

Stock vLLM (0.25.1 **and** nightly) crashes loading the fused `qkv_proj` on the
**sliding-window** layers:

```
RuntimeError: The size of tensor a (1856) must match the size of tensor b (1792)
              at non-singleton dimension 0
              (vllm/model_executor/models/mimo_v2.py::_shard_fp8_qkv_proj)
```

A SWA-layer fused-qkv group is `(64/8)*192 + 192 + 128 = 1856` rows = **14.5**
blocks of the 128-row FP8 scale, so it is **not 128-aligned**. The stock loader
slices the scale per KV-head group (`scale_rows_per_group = 116 // 8 = 14` ->
1792 rows) and multiplies it against the 1856-row weight group. Because
MiMo-V2.5 has only **4 full-attention KV heads** (vs 8 for the Pro), no 8-GPU TP
config avoids the buggy path (`TP=8` can't split 4 KV heads; `TP=4`/`TP=2`/
`DP=2 x TP=4` put 2 SWA KV heads per rank -> the `g>1` crash). The Pro variant
never hits this (8 KV heads + TP=8 -> 1 head/rank -> trivial fast path).

**Fix**: vLLM [PR #42270](https://github.com/vllm-project/vllm/pull/42270)
("MiMo V2: Pro fused-QKV FP8 loader + fix SWA wrong-data on V2.5 base") replaces
the loader with one that "dequantizes and requantizes local shards for TP
configurations that cut through 128-row FP8 scale blocks". As of 2026-07 it is
still **open** (merge conflicts / awaiting review), so not in any release. This
directory vendors the PR's two model files under `pr42270/`, and
`run_vllm_h100.sh` `cp`s them into the nightly image at container start.
**Verified 2026-07-21**: with the PR applied, `--tp 8 --enable-expert-parallel`
loads and produces correct output (coherent short answers + a full 500-token
B-tree explanation, no gibberish). Related upstream: issue
[#42803](https://github.com/vllm-project/vllm/issues/42803) (root bug report),
which the PR closes.

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

## Performance (single-node 8xH100, FP8)

Input/output: 900 / 90 tokens (random dataset), no speculative decoding — same
recipe and shape as the Trn2 table in the main README, for direct comparison.
Raw logs in `results/`.

**SGLang, DP=2 (cookbook default)** — `results/sglang_dp2_c{1,16,32}.txt`:

| Concurrency | Output throughput (tok/s) | Total throughput (tok/s) | TPOT median (ms) | TTFT median (ms) | TTFT P99 (ms) |
|---|---|---|---|---|---|
| 1  | 91.10   | 999.23   | 9.78  | 112 | 199 |
| 16 | 702.18  | 7706.69  | 20.33 | 210 | 735 |
| 32 | 1420.97 | 15595.60 | 18.63 | 299 | 440 |

**SGLang, attn-CP=2** (`DP=1 ATTN_CP=2`) — `results/sglang_cp2_c{1,16,32}.txt`.
Slightly faster (a single request uses all 8 GPUs instead of 4 per DP group):

| Concurrency | Output throughput (tok/s) | Total throughput (tok/s) | TPOT median (ms) | TTFT median (ms) | TTFT P99 (ms) |
|---|---|---|---|---|---|
| 1  | 125.69  | 1378.59  | 6.93  | 95   | 157 |
| 16 | 792.68  | 8699.94  | 17.66 | 186  | 726 |
| 32 | 1570.83 | 17240.36 | 17.78 | 237  | 293 |

**vLLM + PR #42270, TP=8 + EP** — `results/vllm_pr42270_c{1,16,32}.txt`:

| Concurrency | Output throughput (tok/s) | Total throughput (tok/s) | TPOT median (ms) | TTFT median (ms) | TTFT P99 (ms) |
|---|---|---|---|---|---|
| 1  | 48.26   | 529.33   | 5.42  | 84  | 12469 |
| 16 | 214.49  | 2354.09  | 16.74 | 224 | 16225 |
| 32 | 1459.66 | 16020.21 | 18.31 | 273 | 715 |

vLLM's median TPOT is actually the lowest of the three (5.4 ms at c=1), but with
`--no-enable-chunked-prefill` (matching Trn2) its c=1/c=16 output throughput and
tail TTFT suffer badly — each new request's 900-token prefill preempts decode
(P99 TTFT 12–16 s at c≤16). By c=32 the pipeline stays saturated and vLLM catches
SGLang (1460 vs 1421 out tok/s). SGLang's DP-attention absorbs the prefill
interleaving far better at low/mid concurrency.

### H100 vs Trn2 (vLLM-Neuron) — same 900/90 shape, out tok/s

| Concurrency | SGLang DP=2 | SGLang CP=2 | vLLM+PR#42270 | Trn2 | best / Trn2 |
|---|---:|---:|---:|---:|---:|
| 1  | 91.10   | 125.69  | 48.26   | 15.88  | **7.9×** |
| 16 | 702.18  | 792.68  | 214.49  | 113.92 | **7.0×** |
| 32 | 1420.97 | 1570.83 | 1459.66 | 147.39 | **10.7×** |

(Trn2 numbers from the main README's Performance section: BS=32, TP=64 /
moe_ep=64, CB + bucketing.) The H100 gap is largest at c=32 because the GPU
median ITL stays ~13–20 ms across concurrency while the Trn2 BS=32 TKG NEFF runs
at a fixed ~58 ms/token regardless of occupancy.

> **Note:** these are the *language-only, no-spec-decode* baselines for
> apples-to-apples comparison with Trn2. Turning on EAGLE (`SPEC=1` on SGLang)
> would raise decode throughput further; the model card's full recipe enables it.
