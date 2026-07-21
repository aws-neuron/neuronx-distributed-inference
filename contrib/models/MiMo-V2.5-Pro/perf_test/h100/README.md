# H100 GPU baseline for MiMo-V2.5-Pro

Scripts to serve the **official HuggingFace OCP-FP8 checkpoint**
(`XiaomiMiMo/MiMo-V2.5-Pro`) on H100 via the stock `vllm/vllm-openai` Docker
image, for a cross-platform throughput comparison against the Trn2 Neuron port.

## Why 2 nodes (not 1)

The FP8 weights are ~963 GB, which does not fit on a single 8×H100-80GB node
(640 GB). MiMo-V2.5-Pro also has only **8 KV heads**, so TP must divide 8
(TP≤8) — `--tensor-parallel-size 16` fails with *"TP size must evenly split the
number of KV heads"*. We therefore use **TP=8 (within a node) × PP=2 (across two
nodes)** = world size 16.

`run_vllm_h100_multinode.sh` is the only server script; there is no single-node
variant because the model cannot fit on one node.

## Usage

Download the checkpoint on **both** nodes first:

```bash
hf download XiaomiMiMo/MiMo-V2.5-Pro --local-dir /opt/dlami/nvme/models/MiMo-V2.5-Pro
docker pull vllm/vllm-openai:latest
```

Run the **same** script on both nodes, changing only `NODE_RANK`
(`MASTER_ADDR` = node-0's private IP, reachable from both):

```bash
# node 0 (API server on :8000)
NODE_RANK=0 MASTER_ADDR=<node0-ip> bash run_vllm_h100_multinode.sh
# node 1 (headless follower)
NODE_RANK=1 MASTER_ADDR=<node0-ip> bash run_vllm_h100_multinode.sh
```

Node 0 runs the OpenAI API server; node 1 runs `--headless` (no API server,
else it crashes with *"collective_rpc should not be called on follower node"*).
Both use `--network host` (cross-node NCCL/torch.distributed can't use port
mapping) and expose EFA via `/dev/infiniband`.

`--no-enable-prefix-caching` and `--no-enable-chunked-prefill` match the Trn2
`start_vllm_server.sh` so the comparison is fair (with prefix caching on we saw
a 66% cache-hit rate that inflated throughput).

## Benchmark

Reuse the **same** `perf_test/run_bench_single.sh` as Trn2 (it skips the Neuron
venv when absent and takes `SERVED_MODEL_NAME` / `TOKENIZER_PATH` for the GPU
case). Run the bench client inside the vllm container:

```bash
docker run --rm --network host -v /opt/dlami/nvme/models:/wk --entrypoint bash \
    vllm/vllm-openai:latest -c '
      export SERVED_MODEL_NAME=MiMo-V2.5-Pro TOKENIZER_PATH=/wk/MiMo-V2.5-Pro \
             RESULTS_DIR=/wk/bench_results CONFIG_NAME=h100_tp8pp2_nocache
      CONCURRENCY=48 NUM_PROMPTS=96 bash /wk/run_bench_single.sh'
```

See the main README's Performance section for the measured H100-vs-Trn2 numbers.
