#!/bin/bash
# Start vLLM server for InternVL3-8B-Instruct
#
# Prerequisites:
#   - Apply patches from vllm/README.md first
#   - source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
#
# Usage:
#   PYTHONPATH="$PWD/contrib/models/InternVL3-8B-Instruct/src:$PYTHONPATH" bash start-vllm-server.sh

MODEL_PATH="${MODEL_PATH:-/mnt/models/InternVL3-8B-Instruct/}"
PORT="${PORT:-8000}"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --device neuron \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --dtype bfloat16 \
    --trust-remote-code \
    --port "$PORT" \
    --override-neuron-config '{"vision_neuron_config": {"fused_qkv": true, "buckets": [1]}}' \
    "$@"
