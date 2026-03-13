#!/bin/bash
# Qwen3-Coder-480B-A35B-Instruct: vLLM serving on trn2.48xlarge
#
# Throughput-optimized configuration:
#   - seq_len=8192, batch_size=16, pure TP=64
#   - Auto-bucketing (128, 256, 512, 1024, 2048, 4096, 8192)
#   - All NKI QKV kernels + fused RoPE enabled
#   - No on_device_sampling (vLLM samples on CPU for better throughput)
#
# First launch: ~22 min compile + ~10 min weight load
# Subsequent launches: ~10 min (load from cache)
#
# Usage:
#   ./qwen3_coder_vllm.sh                          # Use default model path
#   MODEL_PATH=/path/to/model ./qwen3_coder_vllm.sh  # Custom model path
#   CONFIG=long_context ./qwen3_coder_vllm.sh       # Use 16K context config

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/nvme/Qwen3-Coder-480B-A35B-Instruct/}"
CACHE_DIR="${CACHE_DIR:-/mnt/nvme/neuron_cache}"
PORT="${PORT:-8000}"
CONFIG="${CONFIG:-throughput}"

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

export NEURON_COMPILED_ARTIFACTS="$CACHE_DIR"
export NEURON_CC_FLAGS="--retry_failed_compilation"
export VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference'
export MALLOC_ARENA_MAX=64

mkdir -p "$CACHE_DIR"

if [ "$CONFIG" = "long_context" ]; then
    echo "=== Qwen3-Coder-480B: Long Context Config (16384, BS=8) ==="
    ADDITIONAL_CONFIG='{"override_neuron_config": {"async_mode": true, "attn_kernel_enabled": true, "batch_size": 8, "cc_pipeline_tiling_factor": 2, "context_encoding_buckets": [16384], "cp_degree": 1, "ctx_batch_size": 1, "enable_bucketing": true, "flash_decoding_enabled": false, "fused_qkv": true, "is_continuous_batching": true, "logical_nc_config": 2, "max_context_length": 16384, "mode_mask_padded_tokens": true, "moe_ep_degree": 1, "moe_tp_degree": 64, "on_device_sampling_config": {"do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95}, "qkv_cte_nki_kernel_fuse_rope": true, "qkv_kernel_enabled": true, "qkv_nki_kernel_enabled": true, "scratch_pad_size": 1024, "seq_len": 16384, "sequence_parallel_enabled": true, "token_generation_buckets": [16384], "torch_dtype": "bfloat16", "tp_degree": 64}}'
    MAX_NUM_SEQS=8
    MAX_MODEL_LEN=16384
else
    echo "=== Qwen3-Coder-480B: Throughput Optimized Config (8192, BS=16) ==="
    ADDITIONAL_CONFIG='{"override_neuron_config": {"async_mode": true, "attn_kernel_enabled": true, "batch_size": 16, "cc_pipeline_tiling_factor": 2, "cp_degree": 1, "ctx_batch_size": 1, "enable_bucketing": true, "flash_decoding_enabled": false, "fused_qkv": true, "is_continuous_batching": true, "logical_nc_config": 2, "max_context_length": 8192, "mode_mask_padded_tokens": true, "moe_ep_degree": 1, "moe_tp_degree": 64, "qkv_cte_nki_kernel_fuse_rope": true, "qkv_kernel_enabled": true, "qkv_nki_kernel_enabled": true, "scratch_pad_size": 1024, "seq_len": 8192, "sequence_parallel_enabled": true, "torch_dtype": "bfloat16", "tp_degree": 64}}'
    MAX_NUM_SEQS=16
    MAX_MODEL_LEN=8192
fi

echo "Model: $MODEL_PATH"
echo "Cache: $CACHE_DIR"
echo "Port: $PORT"
echo "Time: $(date -u)"

python -u -m vllm.entrypoints.openai.api_server \
    --model="$MODEL_PATH" \
    --tensor-parallel-size=64 \
    --max-num-seqs="$MAX_NUM_SEQS" \
    --max-model-len="$MAX_MODEL_LEN" \
    --additional-config="$ADDITIONAL_CONFIG" \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port="$PORT"
