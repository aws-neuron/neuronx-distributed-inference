#!/bin/bash
# 32K-input concurrency sweep against a running long-context server
# (run_sglang_h100_32k.sh on :30000). Output 128 tok isolates the
# prefill + KV-capacity limit. Sends 2x concurrency prompts per level so the
# client keeps `--max-concurrency` requests in flight throughout.
#
# Usage:
#   bash bench_32k.sh 1 8 16 32 64 128     # full sweep
#   bash bench_32k.sh 32                    # single level (e.g. to match Trn2)
set -e
PORT="${PORT:-30000}"
INPUT_LEN="${INPUT_LEN:-32768}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
RESULTS_DIR="${RESULTS_DIR:-/opt/dlami/nvme/models/bench_results/mimo_v2_5_h100/longctx_32k}"
mkdir -p "$RESULTS_DIR"

for C in "$@"; do
    NP=$((C * 2))
    echo "===== ${INPUT_LEN} in / ${OUTPUT_LEN} out  c=$C  np=$NP ====="
    docker run --rm --network host -v /opt/dlami/nvme/models:/wk --entrypoint bash \
        vllm-efa:latest -c "
          vllm bench serve --backend vllm --host localhost --port $PORT \
            --model MiMo-V2.5 --tokenizer /wk/MiMo-V2.5 --endpoint /v1/completions \
            --dataset-name random --num-prompts $NP \
            --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
            --random-range-ratio 0.02 --max-concurrency $C" 2>&1 | \
        tee "$RESULTS_DIR/c${C}.txt" | \
        grep -E "Successful requests|Failed|Maximum request concurrency:|Benchmark duration|Total input tokens|Output token throughput|Total token throughput|Median TTFT|P99 TTFT|Median TPOT|Median E2E|Mean E2E"
done
