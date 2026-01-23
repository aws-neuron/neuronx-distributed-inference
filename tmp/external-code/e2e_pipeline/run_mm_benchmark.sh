#!/bin/bash
# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.


# Activate virtual environment
source /home/ubuntu/vllm_orig_venv/bin/activate

# Arguments: MAX_CONCURRENCY RESULT_FILENAME
MAX_CONCURRENCY=${1:-1}
NUM_PROMPTS=$((100 * MAX_CONCURRENCY))
# Cap at 500
if [ $NUM_PROMPTS -gt 500 ]; then
    NUM_PROMPTS=500
fi
RESULT_FILENAME=${2:-"benchmark_result"}

vllm bench serve \
  --backend openai-chat \
  --model /home/ubuntu/model_hf/gemma-3-27b-it/ \
  --dataset-name sharegpt \
  --dataset-path /home/ubuntu/daanggn-neuron-inference-migration/latency_benchmark/sharegpt4v_instruct_gpt4-vision_coco_only.json \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-dir results \
  --result-filename "$RESULT_FILENAME" \
  --save-detailed \
  --base_url http://localhost:8080 \
  --endpoint /v1/chat/completions 