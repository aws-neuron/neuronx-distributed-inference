#!/bin/bash
# Run a single vllm-bench-serve pass against an already-running server (vLLM or
# SGLang -- both expose an OpenAI /v1/completions endpoint). Bring your own
# server; this script does not launch or kill it.
#
# Host-agnostic: on Trn2 the vllm CLI lives in the Neuron DLAMI venv (sourced if
# present); on this H100 box we run the bench client inside the vllm-efa
# container instead, so `vllm` is already on PATH and the venv is skipped.
#
# Usage (inside the vllm-efa container, see h100/README.md):
#   SERVED_MODEL_NAME=MiMo-V2.5 TOKENIZER_PATH=/models/MiMo-V2.5 \
#       CONCURRENCY=32 NUM_PROMPTS=96 CONFIG_NAME=vllm bash run_bench_single.sh
#
# Environment knobs:
#   PORT               server port (8000 vLLM / 30000 SGLang)
#   SERVED_MODEL_NAME  model id the server registered (--served-model-name)
#   TOKENIZER_PATH     local checkpoint dir the bench client loads the tokenizer from
#   CONCURRENCY        --max-concurrency (default 1; MiMo-V2.5 bs=32 -> cap at 32)
#   NUM_PROMPTS        --num-prompts (default 16)
#   INPUT_LEN          --random-input-len (default 900, matches Trn2 table)
#   OUTPUT_LEN         --random-output-len (default 90, matches Trn2 table)
#   RANGE_RATIO        --random-range-ratio (default 0.03)
#   CONFIG_NAME        used in the output filename (default vllm)
#   RESULTS_DIR        where to dump per-run log (default /opt/dlami/nvme/logs/bench_results/mimo_v2_5_h100)

set -e

NEURON_VENV="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate"
[ -f "$NEURON_VENV" ] && source "$NEURON_VENV"

MODEL_PATH="${MIMO_V2_5_PATH:-/models/MiMo-V2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.5}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
INPUT_LEN="${INPUT_LEN:-900}"
OUTPUT_LEN="${OUTPUT_LEN:-90}"
RANGE_RATIO="${RANGE_RATIO:-0.03}"
CONFIG_NAME="${CONFIG_NAME:-vllm}"
RESULTS_DIR="${RESULTS_DIR:-/opt/dlami/nvme/logs/bench_results/mimo_v2_5_h100}"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "MiMo-V2.5 H100 single-run benchmark"
echo "=========================================="
echo "  Served model: $SERVED_MODEL_NAME"
echo "  Tokenizer:    $TOKENIZER_PATH"
echo "  Port:         $PORT"
echo "  Config:       $CONFIG_NAME"
echo "  Concurrency:  $CONCURRENCY   Prompts: $NUM_PROMPTS"
echo "  Input len:    $INPUT_LEN   Output len: $OUTPUT_LEN"
echo "  Results:      $RESULTS_DIR/${CONFIG_NAME}_c${CONCURRENCY}.txt"
echo ""

if ! curl -sf "http://localhost:$PORT/health" > /dev/null; then
    echo "ERROR: server is not responding on http://localhost:$PORT"
    exit 1
fi

vllm bench serve \
    --backend vllm \
    --host localhost \
    --port "$PORT" \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$TOKENIZER_PATH" \
    --endpoint /v1/completions \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio "$RANGE_RATIO" \
    --max-concurrency "$CONCURRENCY" \
    2>&1 | tee "$RESULTS_DIR/${CONFIG_NAME}_c${CONCURRENCY}.txt"

echo ""
echo "Saved to: $RESULTS_DIR/${CONFIG_NAME}_c${CONCURRENCY}.txt"
