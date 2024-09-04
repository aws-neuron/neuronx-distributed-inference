 ```bash

inference_demo \
  --model-type llama \
  --task-type causal-lm \
  run \
    --model-path /dev/shm/tiny_llama \
    --compiled-model-path /dev/shm/traced_model/tiny_llama/ \
    --torch-dtype bfloat16 \
    --tp-degree 32 \
    --batch-size 2 \
    --max-context-length 16 \
    --seq-len 16 \
    --on-device-sampling \
    --enable-bucketing \
    --top-k 1 \
    --do-sample \
    --pad-token-id 2 \
    --prompt "I believe the meaning of life is" \
    --prompt "The color of the sky is" \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank 16 \
    --target-modules tokens_embed q_proj k_proj v_proj o_proj up_proj down_proj \
```
