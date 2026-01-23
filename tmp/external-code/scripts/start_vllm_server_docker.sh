#!/bin/bash

# Set environment variables
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_ON_DEVICE_SAMPLING_DISABLED="0"
export VLLM_RPC_TIMEOUT=1800000 

# Optional Environment Variables
export NEURON_COMPILED_ARTIFACTS="/data/traced_model/gemma-3-27b-it"
# export XLA_HANDLE_SPECIAL_SCALAR="1" # For FP8 (E4M3) quantized models

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model="/data/model_hf/gemma-3-27b-it/" \
    --max-num-seqs=1 \
    --max-model-len=2048 \
    --tensor-parallel-size=8 \
    --port=8080 \
    --device "neuron" \
    --allowed-local-media-path="/opt/" \
    --override-neuron-config="{}"