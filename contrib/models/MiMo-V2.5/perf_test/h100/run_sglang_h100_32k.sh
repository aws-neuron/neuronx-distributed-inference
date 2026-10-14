#!/bin/bash
# Long-context SGLang server for MiMo-V2.5 on 8xH100: 32K input capacity.
# Same cookbook DP=2 shape as run_sglang_h100.sh, but with a raised context
# length and mem-fraction to maximize the KV pool for long prompts.
#
# Why this fits so well: MiMo-V2.5 is HYBRID attention -- only 9 of 48 layers are
# full attention (4 KV heads); the other 39 are sliding-window (window=128), whose
# KV is capped by the window. So a 32K prompt's KV is dominated by just the 9 full
# layers. At 32K the server logs "full token usage ~0.39, swa token usage ~0.08".
#
# Observed KV pool (mem-fraction 0.9): max_total_num_tokens = 678149
#   -> ~20 fully-resident 32K requests (678149 / 32768). Higher client concurrency
#      still completes via SGLang's queue/preemption, just with growing TTFT.
#
# Usage:
#   bash run_sglang_h100_32k.sh
#   MEM=0.92 MML=40960 bash run_sglang_h100_32k.sh   # push the KV pool higher
set -e

MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/MiMo-V2.5}"
PORT="${PORT:-30000}"
MEM="${MEM:-0.9}"                                 # weights ~37GB/GPU; rest -> KV
MML="${MML:-34816}"                               # 32768 in + 2048 out headroom
MAX_RUN="${MAX_RUN:-256}"                          # scheduler cap (not KV-resident cap)
IMAGE="${IMAGE:-sglang-efa:latest}"
CACHE_DIR="${CACHE_DIR:-/opt/dlami/nvme/sglang_cache}"
CTR_MODEL="/models/MiMo-V2.5"
mkdir -p "$CACHE_DIR"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: model not found at $MODEL_DIR" >&2
    exit 1
fi

echo "=========================================="
echo "MiMo-V2.5 SGLang 32K-context (single-node 8xH100, Docker)"
echo "  context-length: $MML   mem-frac: $MEM   max-running-requests: $MAX_RUN"
echo "=========================================="

exec docker run --rm --gpus all \
    --network host --privileged --ipc=host --shm-size=32g \
    -v "${MODEL_DIR}:${CTR_MODEL}:ro" \
    -v "${CACHE_DIR}:/root/.cache" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "$IMAGE" \
    python3 -m sglang.launch_server \
    --model-path "$CTR_MODEL" \
    --served-model-name MiMo-V2.5 \
    --trust-remote-code \
    --tp 8 --dp 2 --enable-dp-attention --enable-dp-lm-head --mm-enable-dp-encoder \
    --moe-dense-tp-size 1 \
    --mem-fraction-static "$MEM" \
    --context-length "$MML" \
    --chunked-prefill-size 16384 \
    --max-running-requests "$MAX_RUN" \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --host 0.0.0.0 \
    --port "$PORT"
