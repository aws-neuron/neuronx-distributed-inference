#!/bin/bash
# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.


# Arguments: MODEL_PATH MAX_NUM_SEQS MAX_MODEL_LEN TP_SIZE ON_DEVICE_SAMPLING
MODEL_PATH=${1:-"/home/ubuntu/model_hf/gemma-3-27b-it/"}
MAX_NUM_SEQS=${2:-1}
MAX_MODEL_LEN=${3:-4096}
TP_SIZE=${4:-16}
ON_DEVICE_SAMPLING=${5:-"1"}

# Read quantized from neuron_config.json
QUANTIZED=$(jq -r '.text_config.neuron_config.quantized' "${MODEL_PATH}neuron_config.json")

# Set environment variables
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_ON_DEVICE_SAMPLING_DISABLED="$ON_DEVICE_SAMPLING"
export VLLM_RPC_TIMEOUT=1800000
export NEURON_COMPILED_ARTIFACTS="$MODEL_PATH"

# Set XLA_HANDLE_SPECIAL_SCALAR based on quantized
if [[ "$QUANTIZED" == "true" ]]; then
    export XLA_HANDLE_SPECIAL_SCALAR="1"
else
    export XLA_HANDLE_SPECIAL_SCALAR="0"
fi

echo "XLA_HANDLE_SPECIAL_SCALAR=$XLA_HANDLE_SPECIAL_SCALAR"

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model="/home/ubuntu/model_hf/gemma-3-27b-it/" \
    --max-num-seqs=$MAX_NUM_SEQS \
    --max-model-len=$MAX_MODEL_LEN \
    --tensor-parallel-size=$TP_SIZE \
    --no-enable-prefix-caching \
    --port=8080 \
    --allowed-local-media-path="/home/ubuntu"

