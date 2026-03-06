#!/bin/bash
# Start GLM-4.5 MoE OpenAI-compatible API server via vLLM + NxDI backend.
#
# Usage:
#   MODEL_PATH=/path/to/GLM-4.5-Air bash vllm/start-vllm-server.sh
#
# Environment variables (with defaults):
#   MODEL_PATH       - Required: path to HuggingFace checkpoint
#   TP_DEGREE        - Tensor parallelism degree (default: 32)
#   SEQ_LEN          - Max sequence length (default: 4096)
#   MAX_NUM_SEQS     - Max concurrent requests (default: 1)
#   PORT             - Server port (default: 8000)

set -euo pipefail

: "${MODEL_PATH:?ERROR: MODEL_PATH environment variable must be set}"
: "${TP_DEGREE:=32}"
: "${SEQ_LEN:=4096}"
: "${MAX_NUM_SEQS:=1}"
: "${PORT:=8000}"

echo "Starting GLM-4.5 MoE vLLM server..."
echo "  Model:      ${MODEL_PATH}"
echo "  TP degree:  ${TP_DEGREE}"
echo "  Seq len:    ${SEQ_LEN}"
echo "  Port:       ${PORT}"

VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference" \
python -m vllm.entrypoints.openai.api_server \
  --model="${MODEL_PATH}" \
  --max-model-len="${SEQ_LEN}" \
  --tensor-parallel-size="${TP_DEGREE}" \
  --port="${PORT}" \
  --max-num-seqs="${MAX_NUM_SEQS}" \
  --trust-remote-code \
  --override-neuron-config "{
    \"tp_degree\": ${TP_DEGREE},
    \"moe_tp_degree\": ${TP_DEGREE},
    \"moe_ep_degree\": 1,
    \"batch_size\": ${MAX_NUM_SEQS},
    \"seq_len\": ${SEQ_LEN},
    \"max_context_length\": ${SEQ_LEN},
    \"fused_qkv\": true,
    \"flash_decoding_enabled\": true,
    \"on_device_sampling_config\": {
      \"dynamic\": true,
      \"global_topk\": 64,
      \"top_p\": 1.0,
      \"temperature\": 1.0
    }
  }"
