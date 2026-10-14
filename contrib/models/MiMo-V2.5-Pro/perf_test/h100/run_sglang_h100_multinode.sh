#!/bin/bash
# 2-node x 8 H100 SGLang server for MiMo-V2.5-Pro (OCP-FP8), via Docker.
# Cross-check against the vLLM H100 numbers (user suspected vLLM was slow).
#
# Run the SAME command on both nodes, changing only NODE_RANK:
#   node 0 (head):   NODE_RANK=0 bash run_sglang_h100_multinode.sh
#   node 1 (worker): NODE_RANK=1 bash run_sglang_h100_multinode.sh
# DIST_INIT_ADDR must be node 0's private IP:port, reachable from both.
#
# Based on the reference command: TP=16, DP=2 + DP-attention, EP=16, EAGLE
# speculative decoding, mimo reasoning/tool parsers.
set -e

MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/MiMo-V2.5-Pro}"
PORT="${PORT:-30000}"
TP="${TP:-16}"
DP="${DP:-2}"
EP="${EP:-16}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on head, 1 on worker}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-172.31.45.21:20000}"   # P5-1 private IP
IFACE="${IFACE:-enp71s0}"
# Weights nearly fill the GPUs; KV cache needs the remainder. 0.7 (reference
# value) failed with "no GPU memory for the KV cache"; min viable ~0.851.
MEM_FRAC="${MEM_FRAC:-0.9}"
# Custom image with GDRCopy + AWS EFA (aws-ofi-nccl) baked in, so NCCL uses EFA
# instead of falling back to TCP sockets. Built from Dockerfile.sglang-efa-minimal.
IMAGE="${IMAGE:-sglang-efa:latest}"
CTR_MODEL="/models/MiMo-V2.5-Pro"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: model not found at $MODEL_DIR" >&2
    exit 1
fi

echo "=========================================="
echo "MiMo-V2.5-Pro SGLang (2-node, Docker)"
echo "  Node rank:   $NODE_RANK / $NNODES"
echo "  Dist init:   $DIST_INIT_ADDR   iface: $IFACE"
echo "  TP=$TP DP=$DP EP=$EP   Port: $PORT"
echo "=========================================="

# --network host for cross-node NCCL; EFA via /dev/infiniband; --ipc=host +
# shm for NCCL buffers.
# Persist JIT caches (DeepGEMM warmup over 32768 shapes takes ~20 min,
# flashinfer, sglang, torch) to the host so a restart doesn't recompile.
CACHE_DIR="${CACHE_DIR:-/opt/dlami/nvme/sglang_cache}"
mkdir -p "$CACHE_DIR"

exec docker run --rm --gpus all \
    --network host --privileged --ipc=host --shm-size=32g \
    --device /dev/infiniband \
    -v "${MODEL_DIR}:${CTR_MODEL}:ro" \
    -v "${CACHE_DIR}:/root/.cache" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e GLOO_SOCKET_IFNAME="$IFACE" \
    -e NCCL_SOCKET_IFNAME="$IFACE" \
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    -e NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}" \
    -e FI_PROVIDER="${FI_PROVIDER:-efa}" \
    -e NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-ofi}" \
    "$IMAGE" \
    python3 -m sglang.launch_server \
    --model-path "$CTR_MODEL" \
    --served-model-name MiMo-V2.5-Pro \
    --trust-remote-code \
    --tp "$TP" \
    --dp "$DP" \
    --enable-dp-attention \
    --ep "$EP" \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "$DIST_INIT_ADDR" \
    --mem-fraction-static "$MEM_FRAC" \
    --max-running-requests 128 \
    --chunked-prefill-size 32768 \
    --cuda-graph-max-bs-decode 64 \
    --page-size 64 \
    --swa-full-tokens-ratio 0.3 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --host 0.0.0.0 \
    --port "$PORT"
