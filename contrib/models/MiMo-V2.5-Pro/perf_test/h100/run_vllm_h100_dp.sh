#!/bin/bash
# 2-node x 8 H100 vLLM server for MiMo-V2.5-Pro (OCP-FP8) via DATA PARALLEL +
# EXPERT PARALLEL, instead of pipeline parallel.
#
# Why DP+EP over TP8xPP2: MiMo has only 8 KV heads (TP<=8), and the ~963 GB FP8
# weights don't fit on one node. The PP=2 approach works but leaves a pipeline
# bubble (measured 158 out-tok/s @ c=48). DP=2 x TP=8 instead replicates the
# attention path per DP rank (each is TP=8, dividing 8 KV heads) and shards MoE
# experts across TP*DP=16 (equivalent to EP=16) -- the same shape SGLang uses
# with --enable-dp-attention, and with NO pipeline bubble. This is vLLM's
# closest analogue to SGLang's DP-attention.
#
# Run the SAME command on both nodes, changing only DP_RANK:
#   node 0 (head, API server on :8000): DP_RANK=0 bash run_vllm_h100_dp.sh
#   node 1 (headless):                  DP_RANK=1 bash run_vllm_h100_dp.sh
# DP_ADDR must be node 0's private IP, reachable from both.
set -e

MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/MiMo-V2.5-Pro}"
PORT="${PORT:-8000}"
TP="${TP:-8}"                                   # per-node tensor parallel (divides 8 KV heads)
DP="${DP:-2}"                                   # data parallel across the 2 nodes
DP_RANK="${DP_RANK:?set DP_RANK=0 on head, 1 on worker}"
DP_ADDR="${DP_ADDR:-172.31.45.21}"              # P5-1 private IP
DP_RPC_PORT="${DP_RPC_PORT:-29550}"
IFACE="${IFACE:-enp71s0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
IMAGE="${IMAGE:-vllm-efa:latest}"
CACHE_DIR="${CACHE_DIR:-/opt/dlami/nvme/vllm_cache}"
CTR_MODEL="/models/MiMo-V2.5-Pro"
mkdir -p "$CACHE_DIR"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: model not found at $MODEL_DIR" >&2
    exit 1
fi

echo "=========================================="
echo "MiMo-V2.5-Pro vLLM (2-node DP=$DP x TP=$TP + EP, EFA, Docker)"
echo "  DP rank:  $DP_RANK   DP addr: $DP_ADDR:$DP_RPC_PORT   iface: $IFACE"
echo "  Port: $PORT   max-model-len: $MAX_MODEL_LEN"
echo "=========================================="

COMMON_ARGS=(
    --served-model-name MiMo-V2.5-Pro
    --trust-remote-code
    --generation-config vllm
    --tensor-parallel-size "$TP"
    --data-parallel-size "$DP"
    --data-parallel-size-local 1
    --data-parallel-address "$DP_ADDR"
    --data-parallel-rpc-port "$DP_RPC_PORT"
    --enable-expert-parallel
    --max-model-len "$MAX_MODEL_LEN"
)
if [ "$DP_RANK" = "0" ]; then
    ROLE_ARGS=(
        --host 0.0.0.0
        --port "$PORT"
        --data-parallel-start-rank 0
        --no-enable-prefix-caching
        --no-enable-chunked-prefill
        --tool-call-parser mimo
        --enable-auto-tool-choice
        --reasoning-parser mimo
    )
else
    ROLE_ARGS=( --headless --data-parallel-start-rank 1 )
fi

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
    -e FI_PROVIDER=efa \
    -e NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-ofi}" \
    -e VLLM_HOST_IP="$(hostname -I | awk '{print $1}')" \
    "$IMAGE" "$CTR_MODEL" \
    "${COMMON_ARGS[@]}" "${ROLE_ARGS[@]}"
