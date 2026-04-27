#!/bin/bash
set -e

# MiMo-V2.5 FP8 vLLM benchmark on Trn2.
#
# Requires a Neuron-FP8 preprocessed checkpoint (see
# `src/conversion_script/preprocess_mimo_v2_5_fp8.py`). The configs below
# all use moe_tp_degree=1 / moe_ep_degree=64 (experts sharded by expert
# parallelism only, no intra-expert TP split) because moe_tp_degree=64 collapses
# the per-rank FP8 blockwise scale to a singleton — per-rank expert
# intermediate is 32 rows, below the 128-row blockwise block, so
# NxDI's `_setup_for_scale` drops per-channel scale granularity. The resulting
# drift compounds across 47 MoE layers and gives repetition / output collapse.
# Using moe_ep_degree=64 keeps all of each expert's weight + scale on one rank
# (4 experts per rank), which preserves the blockwise scale intact.
#
# NxDI's TKG path refuses Expert Parallelism with BS < num_experts/top_k
# (256 / 8 = 32 for MiMo-V2.5), so the smallest working batch size here is 32.
# If you want BS=1 behaviour, the FP8 path is not currently supported on
# this model on Trn2 — use the BF16 checkpoint with the old bench recipe
# (`moe_tp_degree=64, moe_ep_degree=1, batch_size=1`).

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

MODEL_PATH="${MIMO_V2_5_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Neuron-FP8}"
# The NxDI contrib MiMo-V2.5 modeling code is registered into vLLM /
# NxDI lookup tables by vllm-neuron's register() hook using this env var.
# Default to this contrib package's own src/ relative to the script.
: "${NXDI_CONTRIB_MIMO_V2_5_SRC:=$(cd "$(dirname "$0")/.." && pwd)/src}"
export NXDI_CONTRIB_MIMO_V2_5_SRC

# First-time MiMo-V2.5 FP8 compile takes 30-60 minutes; extend vLLM's ready
# timeout and the compiler's environment variables for FP8 numerics.
export VLLM_ENGINE_READY_TIMEOUT_S=7200

PORT=8000
RESULTS_DIR="/tmp/bench_results/mimo_v2_5"
mkdir -p "$RESULTS_DIR"

# Common neuron config shared across all MiMo-V2.5 FP8 configs.
# save_sharded_checkpoint=true persists per-rank sharded weights to
# <compiled-path>/weights/tp{N}_sharded_checkpoint.safetensors during compile;
# load() then reads those directly (~30s) instead of re-sharding the entire
# checkpoint on every vllm-neuron startup (~10+ min).
COMMON_MIMO_CONFIG='"tp_degree": 64,
            "logical_nc_config": 2,
            "fused_qkv": false,
            "sequence_parallel_enabled": false,
            "glu_mlp": true,
            "normalize_top_k_affinities": true,
            "save_sharded_checkpoint": true,
            "router_config": {"act_fn": "sigmoid", "dtype": "float32"},
            "quantized": true,
            "quantized_checkpoints_path": "'"$MODEL_PATH"'",
            "quantization_dtype": "f8e4m3",
            "quantization_type": "blockwise_symmetric",
            "quantization_block_axis": [1, 2],
            "quantization_block_size": [128, 128],
            "modules_to_not_convert": ["embed_tokens", "lm_head", "norm", "router", "o_proj"],
            "blockwise_matmul_config": {"use_shard_on_block_dynamic_while": true, "block_sharding_strategy": "PING_PONG"}'

# Helper: wait for vLLM server to be ready. First-time compilation of a
# 256-expert MoE model takes 30-90 minutes, so we poll for up to 2 hours.
wait_for_server() {
    echo "  Waiting for vLLM server to be ready (up to 2h for first compile)..."
    local interval=10
    local max_attempts=720  # 720 * 10s = 7200s = 2h
    local start=$SECONDS
    for i in $(seq 1 $max_attempts); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "  Server ready! (waited $((SECONDS - start))s)"
            return 0
        fi
        # Show a progress blip every minute so the user knows we're alive
        if [ $((i % 6)) -eq 0 ]; then
            echo "    ...still waiting ($((SECONDS - start))s elapsed)"
        fi
        sleep $interval
    done
    echo "  ERROR: Server did not start within $((max_attempts * interval))s"
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
echo "MiMo-V2.5 FP8 Performance Benchmark"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo ""

###############################################################################
# Config 1: BS=32, TP=64 + moe_tp=1/moe_ep=64, CB + bucketing (smallest BS
# that satisfies NxDI's Expert-Parallel BS >= num_experts/top_k requirement).
###############################################################################
CONFIG_NAME="bs32_tp64_moetp1_ep64"
echo "--- Config 1: BS=32, moe_tp=1/moe_ep=64, CB + bucketing ---"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port $PORT \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            '"$COMMON_MIMO_CONFIG"',
            "moe_tp_degree": 1,
            "moe_ep_degree": 64,
            "batch_size": 32,
            "ctx_batch_size": 1,
            "tkg_batch_size": 32,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": true,
            "enable_bucketing": true,
            "context_encoding_buckets": [1024],
            "token_generation_buckets": [1024],
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

###############################################################################
# Config 2: BS=128, TP=64 + moe_tp=1/moe_ep=64, CB + bucketing (throughput).
###############################################################################
CONFIG_NAME="bs128_tp64_moetp1_ep64"
echo "--- Config 2: BS=128, moe_tp=1/moe_ep=64, CB + bucketing ---"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 128 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port $PORT \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            '"$COMMON_MIMO_CONFIG"',
            "moe_tp_degree": 1,
            "moe_ep_degree": 64,
            "batch_size": 128,
            "ctx_batch_size": 1,
            "tkg_batch_size": 128,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": true,
            "enable_bucketing": true,
            "context_encoding_buckets": [1024],
            "token_generation_buckets": [1024],
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
run_bench "$CONFIG_NAME" 128 512
stop_server

echo "=========================================="
echo "MiMo-V2.5 FP8 benchmarks complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
ls -la "$RESULTS_DIR"
