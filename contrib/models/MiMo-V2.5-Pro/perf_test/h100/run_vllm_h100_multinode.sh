#!/bin/bash
# 2-node x 8 H100 = TP=16 vLLM server for MiMo-V2.5-Pro (OCP-FP8), via Docker.
# The model weights (~963 GB FP8) do not fit on a single 8x80GB node, so we
# shard across two nodes with vLLM's native multi-node serving (no Ray).
#
# Run the SAME command on both nodes, changing only NODE_RANK:
#   node 0 (master): NODE_RANK=0 bash run_vllm_h100_multinode.sh
#   node 1 (worker): NODE_RANK=1 bash run_vllm_h100_multinode.sh
#
# MASTER_ADDR must be node 0's private IP, reachable from both.
set -e

MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/MiMo-V2.5-Pro}"
PORT="${PORT:-8000}"
# MiMo-V2.5-Pro has only 8 KV heads, so TP must divide 8 -> TP<=8. The ~963 GB
# FP8 weights don't fit on one 8x80GB node, so we use TP=8 (within a node) x
# PP=2 (across the two nodes) = world size 16, instead of TP=16 (which fails:
# "TP size must evenly split the number of KV heads").
TP="${TP:-8}"
PP="${PP:-2}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on master, 1 on worker}"
MASTER_ADDR="${MASTER_ADDR:-172.31.45.21}"     # P5-1 private IP
MASTER_PORT="${MASTER_PORT:-29500}"
IFACE="${IFACE:-enp71s0}"                       # NCCL/GLOO socket iface
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
CTR_MODEL="/models/MiMo-V2.5-Pro"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: model not found at $MODEL_DIR (config.json missing)." >&2
    exit 1
fi

echo "=========================================="
echo "MiMo-V2.5-Pro vLLM (2-node TP=16, Docker)"
echo "  Model:        $MODEL_DIR"
echo "  Node rank:    $NODE_RANK / $NNODES"
echo "  Master:       $MASTER_ADDR:$MASTER_PORT   iface: $IFACE"
echo "  TP:           $TP   PP: $PP   Port: $PORT   max-model-len: $MAX_MODEL_LEN"
echo "=========================================="

# Only node 0 runs the API server; follower nodes (rank != 0) run --headless
# (no API server / engine-core client), otherwise they crash with
# "AssertionError: collective_rpc should not be called on follower node".
COMMON_ARGS=(
    --served-model-name MiMo-V2.5-Pro
    --trust-remote-code
    --generation-config vllm
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --max-model-len "$MAX_MODEL_LEN"
)
if [ "$NODE_RANK" = "0" ]; then
    ROLE_ARGS=(
        --host 0.0.0.0
        --port "$PORT"
        # Match Trn2 start_vllm_server.sh for a fair comparison: no prefix cache
        # (else prefill is skipped on repeated prompts -- we saw 66% hit rate and
        # inflated throughput) and no chunked prefill.
        --no-enable-prefix-caching
        --no-enable-chunked-prefill
        --tool-call-parser mimo
        --enable-auto-tool-choice
        --reasoning-parser mimo
    )
else
    ROLE_ARGS=( --headless )
fi

# --network host is required for cross-node NCCL/torch.distributed rendezvous
# (port mapping does not work for the collective transport). --ipc=host and
# --shm-size for large NCCL buffers. EFA devices are exposed via /dev/infiniband.
exec docker run --rm --gpus all \
    --network host --privileged --ipc=host --shm-size=32g \
    --device /dev/infiniband \
    -v "${MODEL_DIR}:${CTR_MODEL}:ro" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e GLOO_SOCKET_IFNAME="$IFACE" \
    -e NCCL_SOCKET_IFNAME="$IFACE" \
    -e VLLM_HOST_IP="$(hostname -I | awk '{print $1}')" \
    "$IMAGE" "$CTR_MODEL" \
    "${COMMON_ARGS[@]}" "${ROLE_ARGS[@]}"
