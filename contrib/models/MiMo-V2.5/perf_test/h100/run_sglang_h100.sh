#!/bin/bash
# Single-node 8xH100 SGLang server for MiMo-V2.5 (official HF OCP-FP8 checkpoint).
# This is the WORKING H100 baseline (vLLM cannot serve V2.5 FP8 -- see
# run_vllm_h100.sh). V2.5 FP8 (~295 GB) fits on one 8xH100 node (640 GB).
#
# Reuses sglang-efa:latest (superset of stock lmsysorg/sglang).
#
# ---- Why --attention-context-parallel-size 2 (not plain TP=8 or DP=2) ----
# MiMoV2ForCausalLM's fused qkv_proj is TP=4-interleaved, so SGLang requires the
# *effective* attention TP size to be exactly 4 (plain --tp 8 is rejected with
# "requires effective attention TP size 4"). Two 8-GPU shapes give effective
# attn TP = 4:
#   - DP=2 + --enable-dp-attention (the model-card reference): but single-node it
#     deadlocks the shared-MoE / lm-head collective when a request occupies only
#     one DP group -- the idle group never launches its matching forward, so the
#     collective hangs (300s scheduler watchdog -> crash). The reference relies on
#     --moe-a2a-backend deepep to change that collective, but DeepEP needs a
#     working nvshmem RDMA transport which this box's NICs (rdmapXXs0, not mlx5)
#     don't provide -- IBGDA init fails and the forward pass hangs (tested
#     low_latency + normal + REMOTE_TRANSPORT=none + IBGDA=0). DeepEP does work on
#     p5en.48xlarge (mlx5 EFA + a custom Mooncake/nvshmem image); see
#     xiaomi_datalab/mimo_v25.
#   - ATTN_CP=2 (attention context parallel): effective attn TP = tp/cp = 4, MoE
#     shards over TP=8, no DP idle-group problem, no DeepEP. Works out of the box
#     on plain P5. THIS IS THE DEFAULT.
#
# ---- Speculative decoding (MTP/EAGLE) ----
# The checkpoint ships model_mtp.safetensors, and the reference command enables
# EAGLE. We LEAVE IT OFF (the Trn2 port has no spec decoding, so this is the
# apples-to-apples baseline). SPEC=1 turns on the reference EAGLE flags.
#
# Usage:
#   bash run_sglang_h100.sh                 # working default (attn-cp=2, no spec)
#   SPEC=1 bash run_sglang_h100.sh          # + EAGLE speculative decoding
#   DP=2 DEEPEP=1 bash run_sglang_h100.sh   # model-card DP-attention + DeepEP
#                                           #   (only works where nvshmem/DeepEP does)
set -e

MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/MiMo-V2.5}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
DP="${DP:-1}"                                    # 1 => use ATTN_CP; >1 => DP-attention
ATTN_CP="${ATTN_CP:-2}"                          # effective attn TP = TP/ATTN_CP = 4
MEM_FRAC="${MEM_FRAC:-0.9}"                       # weights ~295GB; rest for KV cache
IMAGE="${IMAGE:-sglang-efa:latest}"
CACHE_DIR="${CACHE_DIR:-/opt/dlami/nvme/sglang_cache}"
CTR_MODEL="/models/MiMo-V2.5"
mkdir -p "$CACHE_DIR"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: model not found at $MODEL_DIR" >&2
    exit 1
fi

# Parallelism shape: DP-attention (reference) if DP>1, else attention-CP.
DP_ARGS=()
if [ "$DP" -gt 1 ]; then
    DP_ARGS=( --dp "$DP" --enable-dp-attention --enable-dp-lm-head )
    echo "  DP-attention: ON (dp=$DP, effective attn TP = $((TP/DP)))"
else
    DP_ARGS=( --attention-context-parallel-size "$ATTN_CP" )
    echo "  Attention context parallel: $ATTN_CP (effective attn TP = $((TP/ATTN_CP)))"
fi

# DeepEP MoE all-to-all (needs a working nvshmem RDMA transport; see header).
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
    "${DP_ARGS[@]}" \
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
