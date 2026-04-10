#!/bin/bash
set -e

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

MODEL_PATH="/opt/dlami/nvme/models/MiniMax-M2-BF16"
PORT=8000
RESULTS_DIR="/tmp/bench_results/minimax_m2"
mkdir -p "$RESULTS_DIR"

# Common neuron config shared across all MiniMax configs
COMMON_MINIMAX_CONFIG='"tp_degree": 64,
            "logical_nc_config": 2,
            "flash_decoding_enabled": false,
            "sequence_parallel_enabled": true,
            "qkv_kernel_enabled": false,
            "qkv_nki_kernel_enabled": false,
            "attn_kernel_enabled": false,
            "glu_mlp": true,
            "moe_mask_padded_tokens": true,
            "disable_numeric_cc_token": true,
            "router_config": {"act_fn": "sigmoid", "dtype": "float32"}'

# Helper: wait for vLLM server to be ready
wait_for_server() {
    echo "  Waiting for vLLM server to be ready..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "  Server ready! (${i}s)"
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: Server did not start within 600s"
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
echo "MiniMax-M2 Performance Benchmark"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo ""

###############################################################################
# Config 1: BS=1, TP=64/EP=1, non-CB (baseline latency)
# NOTE: fused_qkv=true, use_shard_on_intermediate=false (avoids 10.7x padding)
###############################################################################
CONFIG_NAME="bs1_tp64_ep1"
echo "--- Config 1: BS=1, TP=64/EP=1, non-CB (baseline) ---"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 1 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port $PORT \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            '"$COMMON_MINIMAX_CONFIG"',
            "moe_tp_degree": 64,
            "moe_ep_degree": 1,
            "batch_size": 1,
            "ctx_batch_size": 1,
            "tkg_batch_size": 1,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": false,
            "fused_qkv": true,
            "enable_bucketing": false,
            "async_mode": false,
            "use_index_calc_kernel": false,
            "normalize_top_k_affinities": true,
            "on_device_sampling_config": {
                "do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95
            },
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": false,
                "skip_dma_token": true
            }
        }
    }' &

wait_for_server
sanity_check
run_bench "$CONFIG_NAME" 1 16
stop_server

###############################################################################
# Config 2: BS=256, TP=1/EP=64, CB + optimizations
# NOTE: With EP=64, I_TP=1536/1=1536, 1536%256=0, so shard_on_intermediate is safe
###############################################################################
CONFIG_NAME="bs256_tp1_ep64_opt"
echo "--- Config 2: BS=256, TP=1/EP=64, CB + optimizations ---"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 256 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port $PORT \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            '"$COMMON_MINIMAX_CONFIG"',
            "moe_tp_degree": 1,
            "moe_ep_degree": 64,
            "batch_size": 256,
            "ctx_batch_size": 1,
            "tkg_batch_size": 256,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": true,
            "fused_qkv": false,
            "enable_bucketing": true,
            "context_encoding_buckets": [1024],
            "token_generation_buckets": [1024],
            "async_mode": false,
            "normalize_top_k_affinities": true,
            "strided_context_parallel_kernel_enabled": false,
            "qkv_cte_nki_kernel_fuse_rope": false,
            "on_device_sampling_config": null,
            "use_index_calc_kernel": true,
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": true,
                "skip_dma_token": true
            },
            "scratchpad_page_size": 1024
        }
    }' &

wait_for_server
sanity_check
run_bench "$CONFIG_NAME" 1 16
run_bench "$CONFIG_NAME" 16 128
run_bench "$CONFIG_NAME" 32 128
run_bench "$CONFIG_NAME" 128 512
run_bench "$CONFIG_NAME" 256 512
stop_server

echo "=========================================="
echo "MiniMax-M2 benchmarks complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
ls -la "$RESULTS_DIR"
