#!/bin/bash
# Single-node 8xH100 SGLang server for MiMo-V2.5 (official HF OCP-FP8 checkpoint).
# Works out of the box (vLLM also works but needs PR #42270 -- see
# run_vllm_h100.sh). V2.5 FP8 (~295 GB) fits on one 8xH100 node (640 GB).
#
# Reuses sglang-efa:latest (superset of stock lmsysorg/sglang).
#
# ---- Default = the model-card / SGLang cookbook command (DP-attention) ----
# https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.5
#   --tp 8 --dp 2 --enable-dp-attention --enable-dp-lm-head --mm-enable-dp-encoder
#   --mem-fraction-static 0.65 --chunked-prefill-size 16384
# DP=2 gives effective attention TP = tp/dp = 4, which MiMoV2ForCausalLM requires
# (its fused qkv_proj is TP=4-interleaved; plain --tp 8 is rejected). This runs
# out of the box single-node -- NO DeepEP needed. The three DP flags
# (--enable-dp-attention + --enable-dp-lm-head + --mm-enable-dp-encoder) together
# with mem-fraction 0.65 are what make the shared-MoE / lm-head collective work;
# an earlier attempt with only --enable-dp-attention and mem-fraction 0.9 hung.
#
# ---- Speculative decoding (MTP/EAGLE) ----
# The checkpoint ships model_mtp.safetensors and the cookbook enables EAGLE. We
# LEAVE IT OFF by default (the Trn2 port has no spec decoding, so this is the
# apples-to-apples baseline). SPEC=1 turns on the cookbook EAGLE flags.
#
# ---- Alternative: attention context parallel (ATTN_CP) ----
# Set DP=1 ATTN_CP=2 for --attention-context-parallel-size 2 instead of DP. Also
# gives effective attn TP=4, splits attention along the sequence so a single
# request uses all 8 GPUs (measured slightly faster at low concurrency), and
# needs no DP flags. Kept as a documented alternative.
#
# Usage:
#   bash run_sglang_h100.sh                 # cookbook DP=2 baseline (no spec)
#   SPEC=1 bash run_sglang_h100.sh          # + EAGLE speculative decoding
#   DP=1 ATTN_CP=2 bash run_sglang_h100.sh  # attention-context-parallel variant
set -e

MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/MiMo-V2.5}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
DP="${DP:-2}"                                    # cookbook default; DP=1 => use ATTN_CP
ATTN_CP="${ATTN_CP:-2}"                           # only used when DP=1
MEM_FRAC="${MEM_FRAC:-0.65}"                       # cookbook value
IMAGE="${IMAGE:-sglang-efa:latest}"
CACHE_DIR="${CACHE_DIR:-/opt/dlami/nvme/sglang_cache}"
CTR_MODEL="/models/MiMo-V2.5"
mkdir -p "$CACHE_DIR"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: model not found at $MODEL_DIR" >&2
    exit 1
fi

# Parallelism shape: cookbook DP-attention if DP>1, else attention-CP.
PAR_ARGS=()
if [ "$DP" -gt 1 ]; then
    PAR_ARGS=( --dp "$DP" --enable-dp-attention --enable-dp-lm-head --mm-enable-dp-encoder )
    echo "  DP-attention: dp=$DP (effective attn TP = $((TP/DP)))"
else
    PAR_ARGS=( --attention-context-parallel-size "$ATTN_CP" )
    echo "  Attention context parallel: $ATTN_CP (effective attn TP = $((TP/ATTN_CP)))"
fi

# DeepEP MoE all-to-all: NOT needed single-node (and its nvshmem RDMA transport
# fails on this box's non-mlx5 NICs). Off by default; DEEPEP=1 to force it where
# it works (e.g. p5en.48xlarge with mlx5 EFA).
A2A_ARGS=()
DEEPEP_ENV=()
if [ "${DEEPEP:-0}" = "1" ]; then
    A2A_ARGS=( --moe-a2a-backend deepep --deepep-mode "${DEEPEP_MODE:-auto}" )
    DEEPEP_ENV=( -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 )
    echo "  MoE a2a backend: DeepEP ON (mode=${DEEPEP_MODE:-auto})"
fi

# EAGLE speculative decoding (off by default for parity with Trn2).
SPEC_ARGS=()
if [ "${SPEC:-0}" = "1" ]; then
    SPEC_ARGS=(
        --speculative-algorithm EAGLE
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
        --enable-multi-layer-eagle
    )
    echo "  Speculative decoding: EAGLE ON"
fi

echo "=========================================="
echo "MiMo-V2.5 SGLang (single-node 8xH100, Docker)"
echo "  TP=$TP DP=$DP   Port: $PORT   mem-frac: $MEM_FRAC"
echo "=========================================="

exec docker run --rm --gpus all \
    --network host --privileged --ipc=host --shm-size=32g \
    -v "${MODEL_DIR}:${CTR_MODEL}:ro" \
    -v "${CACHE_DIR}:/root/.cache" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "${DEEPEP_ENV[@]}" \
    "$IMAGE" \
    python3 -m sglang.launch_server \
    --model-path "$CTR_MODEL" \
    --served-model-name MiMo-V2.5 \
    --trust-remote-code \
    --tp "$TP" \
    "${PAR_ARGS[@]}" \
    --moe-dense-tp-size 1 \
    --mem-fraction-static "$MEM_FRAC" \
    --max-running-requests 128 \
    --chunked-prefill-size 16384 \
    "${A2A_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --host 0.0.0.0 \
    --port "$PORT"
