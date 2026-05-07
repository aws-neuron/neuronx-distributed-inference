#!/bin/bash

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_COMPILED_ARTIFACTS="/home/ubuntu/traced_model/gemma-3-27b-it"  # pragma: allowlist secret
export VLLM_RPC_TIMEOUT=100000

python -m vllm.entrypoints.openai.api_server \
    --port=8080 \
    --model="/home/ubuntu/models/gemma-3-27b-it" \
    --max-num-seqs=1 \
    --max-model-len=1024 \
    --limit-mm-per-prompt='{"image": 1}' \
    --allowed-local-media-path="/home/ubuntu" \
    --tensor-parallel-size=8 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --additional-config='{"override_neuron_config":{"text_neuron_config":{"attn_kernel_enabled":true,"enable_bucketing":true,"context_encoding_buckets":[1024],"token_generation_buckets":[1024],"is_continuous_batching":true,"async_mode":true},"vision_neuron_config":{"enable_bucketing":true,"buckets":[1],"is_continuous_batching":true}}}'
