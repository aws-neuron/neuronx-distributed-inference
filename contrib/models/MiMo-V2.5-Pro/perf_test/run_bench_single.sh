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
#   INPUT_LEN        --random-input-len (default 900)
#   OUTPUT_LEN       --random-output-len (default 90)
#   RANGE_RATIO      --random-range-ratio (default 0.03)
#   CONFIG_NAME      Used in the output filename (default bs48_tp64_moetp1_ep64)
#   RESULTS_DIR      Where to dump per-run log
#                    (default /opt/dlami/nvme/logs/bench_results/mimo_v2_5_pro)

set -e

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

MODEL_PATH="${MIMO_V2_FLASH_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
INPUT_LEN="${INPUT_LEN:-900}"
OUTPUT_LEN="${OUTPUT_LEN:-90}"
RANGE_RATIO="${RANGE_RATIO:-0.03}"
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
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
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
