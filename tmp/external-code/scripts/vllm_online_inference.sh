#!/bin/bash
# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_COMPILED_ARTIFACTS="/home/ubuntu/traced_model/gemma-3-27b-it-4096-tp16-bs1"  # pragma: allowlist secret
export NEURON_ON_DEVICE_SAMPLING_DISABLED="1"
export VLLM_RPC_TIMEOUT=100000
# Uncomment if compiling a quantized model with FP8 (E4M3) data type
#export XLA_HANDLE_SPECIAL_SCALAR="1"

python -m vllm.entrypoints.openai.api_server \
    --model="/home/ubuntu/model_hf/gemma-3-27b-it" \
    --max-num-seqs=1 \
    --max-model-len=4096 \
    --tensor-parallel-size=16 \
    --port=8080 \
    --device="neuron" \
    --allowed-local-media-path="/home/ubuntu" \
    --override-neuron-config="{}" # Required or crashes, provide at least "{}" 
