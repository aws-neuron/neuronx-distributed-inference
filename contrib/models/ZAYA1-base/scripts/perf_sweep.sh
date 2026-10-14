#!/usr/bin/env bash
#
# ZAYA Performance Optimization Sweep via vLLM
# trn2.3xlarge, TP=2, LNC=2
#
# Tests 4 configurations sequentially. Each config:
#   1. Starts vLLM server with the config
#   2. Waits for server to be ready
#   3. Runs benchmark at concurrency=1 and concurrency=4
#   4. Kills the server
#
# Usage:
#   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
#   bash ~/zaya/perf_sweep.sh 2>&1 | tee ~/zaya/perf_sweep.log

set -euo pipefail

MODEL=/home/ubuntu/models/ZAYA1-base
PORT=8000
RESULTS_DIR=/home/ubuntu/zaya/perf_sweep_results
mkdir -p "$RESULTS_DIR"

export NEURON_PLATFORM_TARGET_OVERRIDE=trn2

# Benchmark function: sends N requests, measures throughput
benchmark() {
    local name=$1
    local prompt_len=$2
    local max_tokens=$3
    local concurrency=$4
    local num_requests=$5

    echo "  Benchmark: prompt=$prompt_len, gen=$max_tokens, conc=$concurrency, n=$num_requests"

    # Generate a prompt of the desired length (rough: 1 word ~ 1.3 tokens)
    local prompt
    prompt=$(python3 -c "print('hello ' * $prompt_len)")

    local start end elapsed
    start=$(date +%s%N)

    # Send requests in parallel using background curl processes
    local pids=()
    local tmpdir
    tmpdir=$(mktemp -d)

    for i in $(seq 1 "$num_requests"); do
        # Wait if we have too many concurrent requests
        while [ ${#pids[@]} -ge "$concurrency" ]; do
            local new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            pids=("${new_pids[@]}")
            [ ${#pids[@]} -ge "$concurrency" ] && sleep 0.1
        done

        curl -s -X POST "http://localhost:$PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$MODEL\", \"prompt\": \"$prompt\", \"max_tokens\": $max_tokens, \"temperature\": 0}" \
            -o "$tmpdir/resp_$i.json" &
        pids+=($!)
    done

    # Wait for all requests
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))  # milliseconds

    # Count tokens generated
    local total_tokens=0
    local success=0
    for f in "$tmpdir"/resp_*.json; do
        if [ -f "$f" ]; then
            local tokens
            tokens=$(python3 -c "
import json, sys
try:
    d = json.load(open('$f'))
    u = d.get('usage', {})
    print(u.get('completion_tokens', 0))
except:
    print(0)
" 2>/dev/null)
            total_tokens=$((total_tokens + tokens))
            [ "$tokens" -gt 0 ] && success=$((success + 1))
        fi
    done

    local tps
    tps=$(python3 -c "print(f'{$total_tokens / ($elapsed / 1000):.1f}')")
    local avg_latency
    avg_latency=$(python3 -c "print(f'{$elapsed / $num_requests:.0f}')")

    echo "    Results: ${success}/${num_requests} ok, ${total_tokens} tokens in ${elapsed}ms"
    echo "    Throughput: ${tps} tok/s, Avg latency: ${avg_latency}ms/req"

    # Save result
    python3 -c "
import json
result = {
    'config': '$name',
    'prompt_len': $prompt_len,
    'max_tokens': $max_tokens,
    'concurrency': $concurrency,
    'num_requests': $num_requests,
    'success': $success,
    'total_tokens': $total_tokens,
    'elapsed_ms': $elapsed,
    'throughput_tok_s': $total_tokens / ($elapsed / 1000),
    'avg_latency_ms': $elapsed / $num_requests,
}
with open('$RESULTS_DIR/${name}_c${concurrency}_p${prompt_len}.json', 'w') as f:
    json.dump(result, f, indent=2)
"

    rm -rf "$tmpdir"
}

# Wait for vLLM server to be ready
wait_for_server() {
    echo "  Waiting for vLLM server..."
    local max_wait=1800  # 30 min (includes compilation)
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "  Server ready after ${waited}s"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        if [ $((waited % 60)) -eq 0 ]; then
            echo "  Still waiting... (${waited}s)"
        fi
    done
    echo "  ERROR: Server did not start within ${max_wait}s"
    return 1
}

# Kill any existing vLLM server
kill_server() {
    pkill -f "vllm.*serve.*$MODEL" 2>/dev/null || true
    sleep 5
    # Force kill if still running
    pkill -9 -f "vllm.*serve.*$MODEL" 2>/dev/null || true
    sleep 2
}

# Run a single configuration
run_config() {
    local name=$1
    local max_model_len=$2
    local max_num_seqs=$3
    local extra_env=$4

    echo ""
    echo "=================================================================="
    echo "Config: $name"
    echo "  max_model_len=$max_model_len, max_num_seqs=$max_num_seqs"
    echo "  extra_env: $extra_env"
    echo "=================================================================="

    kill_server

    # Clear compile cache
    rm -rf /var/tmp/neuron-compile-cache/ 2>/dev/null

    # Start vLLM server
    local override='{"override_neuron_config": {"is_continuous_batching": true}}'
    local compile_start
    compile_start=$(date +%s)

    echo "  Starting vLLM server..."
    env $extra_env python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size 2 \
        --max-model-len "$max_model_len" \
        --max-num-seqs "$max_num_seqs" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --trust-remote-code \
        --block-size 128 \
        --no-enable-prefix-caching \
        --additional-config "$override" \
        > "$RESULTS_DIR/${name}_server.log" 2>&1 &

    local server_pid=$!
    echo "  Server PID: $server_pid"

    if ! wait_for_server; then
        echo "  FAILED to start server for config $name"
        kill_server
        return 1
    fi

    local compile_end
    compile_end=$(date +%s)
    local compile_time=$((compile_end - compile_start))
    echo "  Compile + startup: ${compile_time}s ($(( compile_time / 60 ))m)"

    # Run benchmarks
    echo ""
    echo "  --- Benchmarks for $name ---"

    # Short prompt (4 tokens), 30 generated tokens, concurrency=1
    benchmark "$name" 4 30 1 10

    # Short prompt, concurrency=4
    benchmark "$name" 4 30 4 20

    # Medium prompt (64 tokens), 30 generated tokens, concurrency=1
    benchmark "$name" 64 30 1 10

    # Medium prompt, concurrency=4
    benchmark "$name" 64 30 4 20

    # Long prompt (256 tokens), 30 generated tokens, concurrency=1
    if [ "$max_model_len" -ge 512 ]; then
        benchmark "$name" 256 30 1 10
    fi

    # Long prompt (512 tokens), concurrency=1
    if [ "$max_model_len" -ge 768 ]; then
        benchmark "$name" 512 30 1 10
    fi

    # Save compile time
    echo "{\"compile_startup_s\": $compile_time}" > "$RESULTS_DIR/${name}_compile.json"

    kill_server
    echo "  Config $name complete."
}

# =========================================================================
# Configuration sweep
# =========================================================================

echo "ZAYA Performance Optimization Sweep"
echo "===================================="
echo "Instance: trn2.3xlarge, TP=2, LNC=2"
echo "Model: $MODEL"
echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

# Config A: Baseline (max_model_len=256, like previous benchmark)
run_config "A_baseline_256" 256 4 ""

# Config B: Multi-bucket with max_model_len=1024
# vLLM + NxDI will auto-generate CTE buckets based on max_model_len
run_config "B_multibucket_1024" 1024 4 ""

# Config C: MLP ISA kernel (max_model_len=256)
# Pass mlp_kernel_enabled via override_neuron_config is not directly supported,
# so we need to set it via the model's config. For now, skip this -- requires
# code change to ZayaNeuronConfig default or a custom override mechanism.
# run_config "C_mlpisa_256" 256 4 ""

# Config D: Multi-bucket with max_model_len=1024 + NKI disabled for safety
run_config "D_multibucket_1024_nonki" 1024 4 "ZAYA_DISABLE_NKI=1"

echo ""
echo "===================================="
echo "Sweep complete: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Results in: $RESULTS_DIR"
echo ""

# Summarize
echo "=== SUMMARY ==="
python3 -c "
import json, glob, os
results_dir = '$RESULTS_DIR'
files = sorted(glob.glob(os.path.join(results_dir, '*_c*_p*.json')))
print(f\"{'Config':<30} {'Prompt':>6} {'Conc':>5} {'Tok/s':>8} {'Lat ms':>8}\")
print('-' * 60)
for f in files:
    with open(f) as fh:
        r = json.load(fh)
    print(f\"{r['config']:<30} {r['prompt_len']:>6} {r['concurrency']:>5} {r['throughput_tok_s']:>7.1f} {r['avg_latency_ms']:>7.0f}\")
"
