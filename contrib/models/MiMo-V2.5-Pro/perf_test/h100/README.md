# H100 GPU baseline for MiMo-V2.5-Pro

Scripts to serve the **official HuggingFace OCP-FP8 checkpoint**
(`XiaomiMiMo/MiMo-V2.5-Pro`) on H100 across two nodes, for a cross-platform
throughput comparison against the Trn2 Neuron port. Covers both vLLM and SGLang.

## Why 2 nodes (not 1)

The FP8 weights are ~963 GB, which does not fit on a single 8×H100-80GB node
(640 GB). MiMo-V2.5-Pro also has only **8 KV heads**, so vLLM tensor parallel
must divide 8 (TP≤8) — `--tensor-parallel-size 16` fails with *"TP size must
evenly split the number of KV heads"*. So:

- **vLLM**: DP=2 × TP=8 + `--enable-expert-parallel` + chunked prefill = world
  size 16. Each DP rank runs TP=8 attention (dividing the 8 KV heads); MoE shards
  across TP×DP=16. (Do NOT use PP=2 as the multi-node fallback — it leaves a
  pipeline bubble and measured ~5× slower.)
- **SGLang**: TP=16 × DP=2 with `--enable-dp-attention` (DP-attention shards the
  KV heads differently, so TP=16 is fine) + EP=16.

## ⚠️ EFA is required — stock images fall back to TCP sockets

The stock `vllm/vllm-openai` and `lmsysorg/sglang` images **do not ship
aws-ofi-nccl**, so cross-node NCCL silently falls back to TCP sockets
(~14 GB/s) instead of EFA RDMA (~400 GB/s). On the stock vLLM image we measured
NCCL logging `Using network Socket` and 13 s median TTFT. That is a
**~5× throughput / ~12× TTFT penalty** and makes any multi-node comparison
meaningless.

`Dockerfile.vllm-efa` and `Dockerfile.sglang-efa` add GDRCopy + the AWS EFA
installer (libfabric + aws-ofi-nccl) on top of the stock images, and set
`NCCL_NET_PLUGIN=ofi`. With these, NCCL logs:

```
NET/OFI Initializing aws-ofi-nccl 1.20.0 ... Using transport protocol RDMA
NET/OFI Selected provider is efa, fabric is efa-direct (found 32 nics)
```

Build once per node (no GPU needed at build time):

```bash
docker build -t vllm-efa:latest   -f Dockerfile.vllm-efa   .
docker build -t sglang-efa:latest -f Dockerfile.sglang-efa .
```

## Usage

Download the checkpoint on **both** nodes first:

```bash
hf download XiaomiMiMo/MiMo-V2.5-Pro --local-dir /opt/dlami/nvme/models/MiMo-V2.5-Pro
```

Run the **same** launch script on both nodes, changing only `NODE_RANK`
(`MASTER_ADDR` / `DIST_INIT_ADDR` = node-0's private IP, reachable from both):

```bash
# vLLM (node 0 = API server on :8000, node 1 = --headless follower)
DP_RANK=0 DP_ADDR=<node0-ip> bash run_vllm_h100_dp.sh
DP_RANK=1 DP_ADDR=<node0-ip> bash run_vllm_h100_dp.sh

# SGLang (both nodes run the same command; node 0 serves on :30000)
NODE_RANK=0 DIST_INIT_ADDR=<node0-ip>:20000 bash run_sglang_h100_multinode.sh
NODE_RANK=1 DIST_INIT_ADDR=<node0-ip>:20000 bash run_sglang_h100_multinode.sh
```

Notes baked into the scripts:
- `--network host` (cross-node NCCL/torch.distributed can't use port mapping),
  EFA via `/dev/infiniband`, `NCCL_NET_PLUGIN=ofi`.
- JIT caches (DeepGEMM warmup over 32768 shapes ≈ several min, flashinfer, etc.)
  are bind-mounted to the host at `/opt/dlami/nvme/sglang_cache` so a restart
  doesn't recompile.
- vLLM: `--no-enable-prefix-caching` + `--no-enable-chunked-prefill` to match
  Trn2 (prefix caching on gave a 66% hit rate and inflated throughput).
- SGLang: `--mem-fraction-static 0.9` (the reference 0.7 left no room for KV
  cache — weights nearly fill the GPUs).

## Benchmark

Reuse the **same** `perf_test/run_bench_single.sh` as Trn2 (skips the Neuron
venv when absent; takes `SERVED_MODEL_NAME` / `TOKENIZER_PATH` / `PORT`). Run
the bench client inside a container:

```bash
docker run --rm --network host -v /opt/dlami/nvme/models:/wk --entrypoint bash \
    vllm-efa:latest -c '
      export SERVED_MODEL_NAME=MiMo-V2.5-Pro TOKENIZER_PATH=/wk/MiMo-V2.5-Pro \
             PORT=8000 RESULTS_DIR=/wk/bench_results CONFIG_NAME=vllm_efa
      CONCURRENCY=48 NUM_PROMPTS=96 bash /wk/run_bench_single.sh'   # PORT=30000 for SGLang
```

See the main README's Performance section for the measured numbers.
