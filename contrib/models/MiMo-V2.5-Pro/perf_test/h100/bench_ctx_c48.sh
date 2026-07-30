#!/bin/bash
# Fixed concurrency 48 (Pro's minimum working BS: 384 experts / top-8 = 48),
# sweep input length to find the max sustainable context on the 2-node cluster.
# OSL=128 isolates prefill + KV capacity. Run on the head node (P5-1) against the
# already-running multinode server on :30000.
#
# Usage (head node):
#   bash bench_ctx_c48.sh 2048 4096 8192 16384 32768
set -e
PORT="${PORT:-30000}"
C="${C:-48}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
RESULTS_DIR="${RESULTS_DIR:-/opt/dlami/nvme/models/bench_results/mimo_v2_5_pro_h100/longctx_c48}"
mkdir -p "$RESULTS_DIR"

for ISL in "$@"; do
    echo "===== c=$C ISL=$ISL OSL=$OUTPUT_LEN np=96 ====="
    docker run --rm --network host -v /opt/dlami/nvme/models:/wk --entrypoint bash \
        vllm-efa:latest -c "
          vllm bench serve --backend vllm --host localhost --port $PORT \
            --model MiMo-V2.5-Pro --tokenizer /wk/MiMo-V2.5-Pro --endpoint /v1/completions \
            --dataset-name random --num-prompts 96 \
            --random-input-len $ISL --random-output-len $OUTPUT_LEN \
            --random-range-ratio 0.02 --max-concurrency $C" 2>&1 | \
        tee "$RESULTS_DIR/isl${ISL}.txt" | \
        grep -E "Successful requests|Failed|Maximum request concurrency:|Benchmark duration|Total input|Output token throughput|Total token throughput|Median TTFT|P99 TTFT|Median TPOT"
    echo "--- ISL=$ISL done ---"
done
echo ALL_DONE
