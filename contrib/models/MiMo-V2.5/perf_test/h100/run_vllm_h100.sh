#!/bin/bash
# Single-node 8xH100 vLLM server for MiMo-V2.5 (official HF OCP-FP8 checkpoint).
#
# ---- Requires vLLM PR #42270 (not yet merged into any released image) ----
# Stock vLLM (0.25.1 AND nightly) crashes loading V2.5 FP8 in
# _shard_fp8_qkv_proj: "RuntimeError: The size of tensor a (1856) must match
# tensor b (1792)". A sliding-window layer's fused-qkv group is 1856 rows =
# 14.5 blocks of the 128-row FP8 scale, so per-KV-head scale slicing misaligns.
# vLLM PR #42270 ("MiMo V2: Pro fused-QKV FP8 loader + fix SWA wrong-data on
# V2.5 base") replaces the loader with one that dequant/requantizes shards that
# cut through 128-row scale blocks. As of 2026-07 the PR is still OPEN, so we
# bind-mount its two model files (pr42270/) and cp them into the nightly image
# at container start. Once the PR merges, drop the cp step + pr42270/ and use a
# release image.
#
# With the PR applied, the model-card reference command works as-is:
# TP=8 + expert parallel (the paired-qkv loader handles the 4-KV-head split
# that plain vLLM rejects).
#
# Optimizations: chunked prefill ON by default (matches the Pro serving recipe;
# it roughly 4x's mid-concurrency throughput vs off by interleaving the 900-token
# prefills with decode instead of preempting). Prefix caching OFF (would inflate
# throughput via cache hits on the random dataset) and speculative decode
# (MTP/EAGLE) OFF, both for parity with the Trn2 baseline. Set CHUNKED=0 to get
# the no-chunked config that exactly matches Trn2 (Neuron has no chunked prefill).
#
# Usage:
#   bash run_vllm_h100.sh              # chunked prefill ON (recommended)
#   CHUNKED=0 bash run_vllm_h100.sh    # no chunked prefill (exact Trn2 parity)
set -e

MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/MiMo-V2.5}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"              # MiMo-V2.5 bs=32
CHUNKED="${CHUNKED:-1}"                          # 1 = chunked prefill on (default)
IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"      # PR #42270 not in a release yet
CACHE_DIR="${CACHE_DIR:-/opt/dlami/nvme/vllm_cache}"
PATCH_DIR="${PATCH_DIR:-$(cd "$(dirname "$0")/pr42270" && pwd)}"
CTR_MODEL="/models/MiMo-V2.5"
mkdir -p "$CACHE_DIR"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: model not found at $MODEL_DIR" >&2
    exit 1
fi
if [ ! -f "$PATCH_DIR/mimo_v2.py" ]; then
    echo "ERROR: PR #42270 files not found at $PATCH_DIR" >&2
    exit 1
fi

if [ "$CHUNKED" = "1" ]; then
    CHUNKED_FLAG="--enable-chunked-prefill"
else
    CHUNKED_FLAG="--no-enable-chunked-prefill"
fi

echo "=========================================="
echo "MiMo-V2.5 vLLM (single-node TP=$TP + EP, PR #42270, Docker)"
echo "  Port: $PORT   max-model-len: $MAX_MODEL_LEN   max-num-seqs: $MAX_NUM_SEQS"
echo "  chunked-prefill: $CHUNKED_FLAG"
echo "=========================================="

M=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models
exec docker run --rm --gpus all \
    --network host --privileged --ipc=host --shm-size=32g \
    -v "${MODEL_DIR}:${CTR_MODEL}:ro" \
    -v "${CACHE_DIR}:/root/.cache" \
    -v "${PATCH_DIR}:/pr:ro" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --entrypoint bash \
    "$IMAGE" -c "
      cp /pr/mimo_v2.py $M/mimo_v2.py &&
      cp /pr/mimo_v2_mtp.py $M/mimo_v2_mtp.py &&
      echo '[run_vllm_h100] applied PR #42270 model files' &&
      exec vllm serve '$CTR_MODEL' \
        --served-model-name MiMo-V2.5 \
        --trust-remote-code \
        --generation-config vllm \
        --enable-expert-parallel \
        --tensor-parallel-size $TP \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        --no-enable-prefix-caching \
        $CHUNKED_FLAG \
        --host 0.0.0.0 \
        --port $PORT \
        --tool-call-parser mimo \
        --enable-auto-tool-choice \
        --reasoning-parser mimo"
