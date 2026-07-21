#!/bin/bash
# Single-node 8xH100 vLLM server for MiMo-V2.5 (official HF OCP-FP8 checkpoint).
#
# ============================ KNOWN BROKEN ============================
# vLLM CANNOT currently serve MiMo-V2.5 FP8. Both vllm 0.25.1 and nightly
# crash while loading the fused qkv_proj on the sliding-window layers:
#
#     RuntimeError: The size of tensor a (1856) must match the size of
#     tensor b (1792) at non-singleton dimension 0
#     (vllm/model_executor/models/mimo_v2.py::_shard_fp8_qkv_proj)
#
# Root cause (upstream bug): a SWA layer's fused-qkv group is 1856 rows =
# 14.5 blocks of the 128-row FP8 scale, so it is NOT 128-aligned. The loader
# slices the scale per KV-head group (scale_rows_per_group = 116 // 8 = 14 ->
# 1792 rows) and multiplies it against the 1856-row weight group -> shape
# mismatch. This fires at EVERY 8-GPU shape:
#   - TP=8: can't even start, "TP size must evenly split the 4 KV heads".
#   - TP=4 / TP=2 (incl. DP=2 x TP=4): each rank gets 2 SWA KV heads -> the
#     g>1 de-interleave path -> the 1856-vs-1792 crash.
# There is no vLLM TP config that both (a) splits the 4 full-attn KV heads and
# (b) gives 1 SWA KV head per rank, so the buggy path is unavoidable.
#
# apply_mimo_fp8_patch.py makes the server START (whole-tensor dequant), but the
# output is gibberish -- the re-quantization path is not numerically correct --
# so it is NOT a usable fix, only a diagnostic. Use SGLang (run_sglang_h100.sh)
# for the H100 baseline. This script is kept to document the reference command
# and reproduce the bug.
# =====================================================================
#
# The command below is the reference vLLM command from the model card
# (TP=8 + expert parallel, mimo parsers), which is what you WOULD run once the
# upstream loader is fixed.
set -e

MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/MiMo-V2.5}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"              # MiMo-V2.5 bs=32
IMAGE="${IMAGE:-vllm-efa:latest}"
CACHE_DIR="${CACHE_DIR:-/opt/dlami/nvme/vllm_cache}"
CTR_MODEL="/models/MiMo-V2.5"
mkdir -p "$CACHE_DIR"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: model not found at $MODEL_DIR" >&2
    exit 1
fi

echo "=========================================="
echo "MiMo-V2.5 vLLM (single-node TP=$TP + EP, Docker) -- EXPECTED TO FAIL"
echo "  Port: $PORT   max-model-len: $MAX_MODEL_LEN   max-num-seqs: $MAX_NUM_SEQS"
echo "=========================================="

exec docker run --rm --gpus all \
    --network host --privileged --ipc=host --shm-size=32g \
    -v "${MODEL_DIR}:${CTR_MODEL}:ro" \
    -v "${CACHE_DIR}:/root/.cache" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "$IMAGE" "$CTR_MODEL" \
    --served-model-name MiMo-V2.5 \
    --trust-remote-code \
    --generation-config vllm \
    --enable-expert-parallel \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tool-call-parser mimo \
    --enable-auto-tool-choice \
    --reasoning-parser mimo
