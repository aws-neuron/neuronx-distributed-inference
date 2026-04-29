#!/bin/bash
set -e

# MiMo-V2.5-Pro FP8 vLLM benchmark on Trn2. One-shot wrapper:
#   launch server -> sanity check -> bench at c=1,16,48 -> stop server.
#
# This script composes three building blocks in perf_test/:
#   start_vllm_server.sh  - server launch + env-var setup (backgrounded here)
#   sanity_check.sh       - one-shot curl against the running server
#   run_bench_single.sh   - one concurrency level of `vllm bench serve`
#
# Use those directly if you want to keep a long-running server and iterate
# on bench parameters from another shell.
#
# Server recipe: TP=64, moe_tp=1/moe_ep=64, BS=48, continuous batching.
# BS=48 is the smallest working batch size on the FP8 path (NxDI's TKG
# path refuses Expert Parallelism with BS < num_experts/top_k = 384/8 = 48).
# BS=1 single-stream latency demos are not currently supported on Pro FP8.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
RESULTS_DIR="${RESULTS_DIR:-/opt/dlami/nvme/logs/bench_results/mimo_v2_5_pro}"
CONFIG_NAME="bs48_tp64_moetp1_ep64"

mkdir -p "$RESULTS_DIR"

# Wait for vLLM server to be ready. First-time compile of the 384-expert
# MoE model takes ~90 min and can stretch past 2 h under contention, so
# poll for up to 2 h.
wait_for_server() {
    echo "  Waiting for vLLM server on port $PORT (up to 2 h for first compile)..."
    local interval=10
    local max_attempts=720
    local start=$SECONDS
    for i in $(seq 1 $max_attempts); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "  Server ready after $((SECONDS - start))s."
            return 0
        fi
        if [ $((i % 6)) -eq 0 ]; then
            echo "    ...still waiting ($((SECONDS - start))s elapsed)"
        fi
        sleep $interval
    done
    echo "  ERROR: Server did not start within $((max_attempts * interval))s"
    return 1
}

stop_server() {
    echo "  Stopping vLLM server..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 5
}

echo "=========================================="
echo "MiMo-V2.5-Pro FP8 Performance Benchmark"
echo "=========================================="
echo "Port:    $PORT"
echo "Results: $RESULTS_DIR"
echo ""

# Start the server in the background. start_vllm_server.sh handles all the
# env vars (MODEL_PATH, NEURON_COMPILED_ARTIFACTS, BASE_COMPILE_WORK_DIR,
# contrib src registration, etc.) and execs `python3 -m vllm...`.
bash "$SCRIPT_DIR/start_vllm_server.sh" &
SERVER_PID=$!
trap stop_server EXIT

wait_for_server

# One-shot sanity check (curl the chat endpoint).
PORT="$PORT" bash "$SCRIPT_DIR/sanity_check.sh" || true

# Three concurrency levels. run_bench_single.sh reads knobs from the
# environment; see its header for all the options.
PORT="$PORT" RESULTS_DIR="$RESULTS_DIR" CONFIG_NAME="$CONFIG_NAME" \
    CONCURRENCY=1  NUM_PROMPTS=16  bash "$SCRIPT_DIR/run_bench_single.sh"
PORT="$PORT" RESULTS_DIR="$RESULTS_DIR" CONFIG_NAME="$CONFIG_NAME" \
    CONCURRENCY=16 NUM_PROMPTS=128 bash "$SCRIPT_DIR/run_bench_single.sh"
PORT="$PORT" RESULTS_DIR="$RESULTS_DIR" CONFIG_NAME="$CONFIG_NAME" \
    CONCURRENCY=48 NUM_PROMPTS=192 bash "$SCRIPT_DIR/run_bench_single.sh"

echo "=========================================="
echo "MiMo-V2.5-Pro FP8 benchmark complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
ls -la "$RESULTS_DIR"
