#!/bin/bash
# Bench a running MiMo-V2.5 H100 server at c=1/16/32 (bs=32 -> cap at 32).
# Runs the bench client inside the vllm-efa container (has `vllm bench serve`),
# reusing run_bench_single.sh. Prints the key metrics; full logs go to
# /opt/dlami/nvme/models/bench_results/mimo_v2_5_h100/<CONFIG>_c<C>.txt.
#
# Usage: bash bench_all.sh <PORT> <CONFIG_NAME>
#   bash bench_all.sh 30000 sglang_cp2     # SGLang
#   bash bench_all.sh 8000  vllm           # vLLM (if/when it works)
set -e
PORT="${1:-30000}"
CONFIG_NAME="${2:-sglang_dp2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for C in 1 16 32; do
    NP=96; [ "$C" = "1" ] && NP=16
    echo "===== $CONFIG_NAME c=$C np=$NP ====="
    docker run --rm --network host \
        -v /opt/dlami/nvme/models:/wk -v "$SCRIPT_DIR":/sc \
        --entrypoint bash vllm-efa:latest -c "
          export SERVED_MODEL_NAME=MiMo-V2.5 TOKENIZER_PATH=/wk/MiMo-V2.5 PORT=$PORT \
                 RESULTS_DIR=/wk/bench_results/mimo_v2_5_h100 CONFIG_NAME=$CONFIG_NAME
          CONCURRENCY=$C NUM_PROMPTS=$NP bash /sc/run_bench_single.sh" 2>&1 | \
        grep -E "Successful requests|Maximum request concurrency:|Benchmark duration|Total input tokens|Total generated tokens|Output token throughput|Total token throughput|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Median ITL"
done
