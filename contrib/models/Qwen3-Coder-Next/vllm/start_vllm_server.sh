#!/bin/bash
# Launch vLLM server for Qwen3-Coder-Next on trn2.48xlarge (TP=8)
#
# Prerequisites:
#   - Model downloaded to /mnt/Qwen3-Coder-Next
#   - vLLM venv: /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
#   - Contrib code at /home/ubuntu/Qwen3-Coder-Next-contrib/src/
#   - vllm-neuron patched with qwen3next model type registration
#
# Compilation:
#   First run will compile ~15 min. NEFFs cached in neuron compile cache.
#   If NEURON_COMPILED_ARTIFACTS is set, will attempt to load from there first.
#
# Usage:
#   ./start_vllm_server.sh [PORT]
#
# Known limitations:
#   - max_context_length=128 (larger buckets fail to compile with on-device sampling)
#   - max_batch_size=1 (single request serving)
#   - Total max_model_len=256 (128 context + 128 generation)

set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="/mnt/Qwen3-Coder-Next"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTRIB_SRC="${SCRIPT_DIR}/../src"

# Activate vLLM environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

# Set environment
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export PYTHONPATH="${CONTRIB_SRC}:${PYTHONPATH:-}"

echo "================================================"
echo " Qwen3-Coder-Next vLLM Server"
echo " Model: ${MODEL_PATH}"
echo " Port:  ${PORT}"
echo " TP:    8, Context: 128, Gen: 128"
echo "================================================"

# Launch vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --tensor-parallel-size 8 \
    --max-model-len 256 \
    --max-num-seqs 1 \
    --block-size 128 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 8,
            "max_batch_size": 1,
            "max_context_length": 128,
            "max_new_tokens": 128,
            "max_length": 256,
            "torch_dtype": "bfloat16",
            "fused_qkv": true,
            "on_device_sampling_config": {"dynamic": true, "deterministic": false},
            "moe_tp_degree": 8,
            "moe_ep_degree": 1,
            "blockwise_matmul_config": {
                "block_size": 128,
                "use_shard_on_block_dynamic_while": true,
                "block_sharding_strategy": "PING_PONG"
            }
        }
    }'
