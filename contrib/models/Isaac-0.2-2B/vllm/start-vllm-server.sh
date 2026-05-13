#!/bin/bash
# Copyright 2025 (c) Amazon.com and Affiliates
# Start vLLM server for Isaac-0.2-2B on Neuron
#
# Prerequisites:
#   1. Apply vLLM patches: python patch_vllm_isaac.py
#   2. Model compiled at NEURON_COMPILED_ARTIFACTS path
#
# Usage:
#   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
#   bash start-vllm-server.sh

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_COMPILED_ARTIFACTS="/mnt/models/traced_model/Isaac-0.2-2B"
export VLLM_RPC_TIMEOUT=100000

NXDI_ROOT="/mnt/models/neuronx-distributed-inference"
ISAAC_SRC="${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/src"
export PYTHONPATH="${ISAAC_SRC}:${NXDI_ROOT}/src:${PYTHONPATH}"

python -m vllm.entrypoints.openai.api_server \
    --port=8080 \
    --model="/mnt/models/Isaac-0.2-2B-Preview" \
    --max-num-seqs=1 \
    --max-model-len=1024 \
    --limit-mm-per-prompt='{"image": 1}' \
    --allowed-local-media-path="/mnt/models" \
    --tensor-parallel-size=1 \
    --trust-remote-code \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --additional-config='{"override_neuron_config":{"text_neuron_config":{"attn_kernel_enabled":true,"enable_bucketing":true,"context_encoding_buckets":[1024],"token_generation_buckets":[1024],"is_continuous_batching":true,"async_mode":false},"vision_neuron_config":{"enable_bucketing":true,"buckets":[1],"is_continuous_batching":true}}}'
