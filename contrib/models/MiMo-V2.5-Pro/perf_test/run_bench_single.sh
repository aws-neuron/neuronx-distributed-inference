#!/bin/bash
# Run a single vllm-bench-serve pass against an already-running vLLM server.
#
# Unlike bench_mimo_v2.sh this script does NOT launch or kill the vLLM
# server — you bring your own. That makes it convenient when the bench driver
# in bench_mimo_v2.sh times out during first-time compilation: the server
# keeps running, and once it's ready you can collect numbers with this.
#
# Usage:
#   bash run_bench_single.sh                       # defaults: c=1, 16 prompts
#   CONCURRENCY=16 NUM_PROMPTS=128 bash run_bench_single.sh
#   CONFIG_NAME=bs32_tp1_ep64_opt CONCURRENCY=16 NUM_PROMPTS=128 bash run_bench_single.sh
#
# Environment knobs:
#   PORT             vLLM server port (default 8000)
#   MIMO_V2_FLASH_PATH  Path to the Neuron-FP8 checkpoint (default
#                    /opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8)
#   CONCURRENCY      --max-concurrency (default 1)
#   NUM_PROMPTS      --num-prompts (default 16)
#   INPUT_LEN        --random-input-len (default 360; matches seq_len=512)
#   OUTPUT_LEN       --random-output-len (default 120; matches seq_len=512)
#   RANGE_RATIO      --random-range-ratio (default 0.03)
#   CONFIG_NAME      Used in the output filename (default bs48_tp64_moetp1_ep64)
#   RESULTS_DIR      Where to dump per-run log
#                    (default /opt/dlami/nvme/logs/bench_results/mimo_v2_5_pro)

set -e

# On Trn2 the vllm CLI lives in the Neuron DLAMI venv; source it if present.
# On other hosts (e.g. an H100 GPU box for cross-platform comparison) `vllm`
# is expected to already be on PATH, so skip the venv activation there.
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate"
[ -f "$NEURON_VENV" ] && source "$NEURON_VENV"

MODEL_PATH="${MIMO_V2_FLASH_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8}"
# --model is the name the server registered (request routing); --tokenizer is a
# local path the bench client loads. On Trn2 both equal MODEL_PATH. On a host
# where the server was started with --served-model-name (e.g. the H100 Docker
# run), set SERVED_MODEL_NAME to that name and TOKENIZER_PATH to the local
# checkpoint dir, else the client sends an unknown model id and gets 404.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
INPUT_LEN="${INPUT_LEN:-360}"
OUTPUT_LEN="${OUTPUT_LEN:-120}"
RANGE_RATIO="${RANGE_RATIO:-0.03}"
# seq_len=512 on the compiled server, so input+output must stay under 512.
# Default 360+120=480 leaves a small margin for random-range-ratio expansion.
CONFIG_NAME="${CONFIG_NAME:-bs48_tp64_moetp1_ep64}"
RESULTS_DIR="${RESULTS_DIR:-/opt/dlami/nvme/logs/bench_results/mimo_v2_5_pro}"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "MiMo-V2.5-Pro single-run benchmark"
echo "=========================================="
echo "  Model:        $MODEL_PATH"
echo "  Port:         $PORT"
echo "  Config:       $CONFIG_NAME"
echo "  Concurrency:  $CONCURRENCY"
echo "  Prompts:      $NUM_PROMPTS"
echo "  Input len:    $INPUT_LEN   Output len: $OUTPUT_LEN"
echo "  Results:      $RESULTS_DIR/${CONFIG_NAME}_c${CONCURRENCY}.txt"
echo ""

# Quick health check
if ! curl -sf "http://localhost:$PORT/health" > /dev/null; then
    echo "ERROR: vLLM server is not responding on http://localhost:$PORT"
    echo "Start it first (e.g., bench_mimo_v2.sh) and wait until"
    echo "'Application startup complete.' is printed."
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
