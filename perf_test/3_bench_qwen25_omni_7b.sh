#!/bin/bash
set -e

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

MODEL_PATH="/opt/dlami/nvme/models/Qwen2.5-Omni-7B"
PORT=8000
RESULTS_DIR="/var/tmp/bench_results/qwen25_omni_7b"
mkdir -p "$RESULTS_DIR"

# Helper: wait for vLLM server to be ready
wait_for_server() {
    echo "  Waiting for vLLM server to be ready..."
    for i in $(seq 1 360); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "  Server ready! (${i}s * 5 = $((i*5))s)"
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: Server did not start within 1800s"
    return 1
}

# Helper: run benchmark
run_bench() {
    local config_name=$1
    local concurrency=$2
    local num_prompts=$3

    echo "    Benchmark: concurrency=$concurrency, prompts=$num_prompts"
    vllm bench serve \
        --backend vllm \
        --model "$MODEL_PATH" \
        --tokenizer "$MODEL_PATH" \
        --endpoint /v1/completions \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --random-input-len 900 \
        --random-output-len 90 \
        --random-range-ratio 0.03 \
        --max-concurrency "$concurrency" \
        2>&1 | tee "$RESULTS_DIR/${config_name}_c${concurrency}.txt"
    echo ""
}

# Helper: stop server
stop_server() {
    echo "  Stopping vLLM server..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 5
}

# Helper: quick sanity check
sanity_check() {
    echo "  Running sanity check..."
    curl -s http://localhost:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
            "messages": [{"role": "user", "content": "What is 1+1? Answer briefly."}],
            "model": "'"$MODEL_PATH"'",
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": false
        }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('  Sanity:', r['choices'][0]['message']['content'][:100])" 2>/dev/null || echo "  Sanity check: could not parse response"
}

echo "=========================================="
echo "Qwen2.5-Omni-7B Performance Benchmark"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo ""

###############################################################################
# Config 1: BS=1, TP=32, non-CB (baseline latency)
# Qwen2.5-Omni-7B is a dense 7B model, TP=32 is sufficient
###############################################################################
CONFIG_NAME="bs1_tp32"
echo "--- Config 1: BS=1, TP=32, non-CB (baseline) ---"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --tensor-parallel-size 32 \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port $PORT \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 32,
            "fused_qkv": false,
            "flash_decoding_enabled": false,
            "sequence_parallel_enabled": false,
            "qkv_kernel_enabled": false,
            "qkv_nki_kernel_enabled": false,
            "attn_kernel_enabled": false,
            "batch_size": 1,
            "ctx_batch_size": 1,
            "tkg_batch_size": 1,
            "max_context_length": 4096,
            "seq_len": 4096,
            "is_continuous_batching": false,
            "enable_bucketing": false,
            "async_mode": true,
            "on_device_sampling_config": {
                "do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95
            }
        }
    }' &

wait_for_server
sanity_check
run_bench "$CONFIG_NAME" 1 16
stop_server

###############################################################################
# Config 2: BS=32, TP=32, CB (throughput)
###############################################################################
CONFIG_NAME="bs32_tp32_cb"
echo "--- Config 2: BS=32, TP=32, CB ---"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --tensor-parallel-size 32 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port $PORT \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 32,
            "fused_qkv": false,
            "flash_decoding_enabled": false,
            "sequence_parallel_enabled": false,
            "qkv_kernel_enabled": false,
            "qkv_nki_kernel_enabled": false,
            "attn_kernel_enabled": false,
            "batch_size": 32,
            "ctx_batch_size": 1,
            "tkg_batch_size": 32,
            "max_context_length": 4096,
            "seq_len": 4096,
            "is_continuous_batching": true,
            "enable_bucketing": true,
            "context_encoding_buckets": [4096],
            "token_generation_buckets": [4096],
            "async_mode": true,
            "on_device_sampling_config": {
                "do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95
            }
        }
    }' &

wait_for_server
sanity_check
run_bench "$CONFIG_NAME" 1 16
run_bench "$CONFIG_NAME" 16 128
run_bench "$CONFIG_NAME" 32 128
stop_server

echo "=========================================="
echo "Qwen2.5-Omni-7B benchmarks complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
ls -la "$RESULTS_DIR"
