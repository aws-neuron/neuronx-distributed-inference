## My Project

This package provides a model hub for running inference on Neuronx Distributed (NxD).

## Examples
This package includes examples that you can reference when you implement code that uses NxD Inference.
* `generation_demo.py` - A basic generation example for Llama.

## Run inference with the inference demo
This package includes an inference demo console script that you can use to run inference. This script includes benchmarking and accuracy checking features that are useful for developers to verify that their models and modules work correctly.

After you install this package, you can run the inference demo with `inference-demo`. See examples below for how to run the inference demo. You can also run `inference_demo --help` to view all available arguments.

### Example 1: Llama inference with token matching accuracy check
```
inference_demo \
  --model-type llama \
  --task-type causal-lm \
  run \
    --model-path /home/ubuntu/model_hf/Llama-3.1-8B-Instruct/ \
    --compiled-model-path /home/ubuntu/traced_model/Llama-3.1-8B-Instruct/ \
    --torch-dtype bfloat16 \
    --tp-degree 32 \
    --batch-size 2 \
    --max-context-length 32 \
    --seq-len 64 \
    --on-device-sampling \
    --enable-bucketing \
    --top-k 1 \
    --do-sample \
    --pad-token-id 2 \
    --prompt "I believe the meaning of life is" \
    --prompt "The color of the sky is" \
    --check-accuracy-mode token-matching \
    --benchmark
```

### Example 2. DBRX inference with logit matching accuracy check

```
inference_demo \
  --model-type dbrx \
  --task-type causal-lm \
  run \
    --model-path /home/ubuntu/model_hf/dbrx-1layer/ \
    --compiled-model-path /home/ubuntu/traced_model/dbrx-1layer-demo/ \
    --torch-dtype bfloat16 \
    --tp-degree 32 \
    --batch-size 2 \
    --max-context-length 1024 \
    --seq-len 1152 \
    --enable-bucketing \
    --top-k 1 \
    --do-sample \
    --pad-token-id 0 \
    --prompt "I believe the meaning of life is" \
    --prompt "The color of the sky is" \
    --check-accuracy-mode logit-matching
```

### Example 3. Llama with speculation

```
inference_demo \
  --model-type llama \
  --task-type causal-lm \
  run \
    --model-path /home/ubuntu/model_hf/open_llama_7b/ \
    --compiled-model-path /home/ubuntu/traced_model/open_llama_7b/ \
    --draft-model-path /home/ubuntu/model_hf/open_llama_3b/ \
    --compiled-draft-model-path /home/ubuntu/traced_model/open_llama_3b/ \
    --torch-dtype bfloat16 \
    --tp-degree 32 \
    --batch-size 1 \
    --max-context-length 32 \
    --seq-len 64 \
    --enable-bucketing \
    --speculation-length 5 \
    --no-trace-tokengen-model \
    --top-k 1 \
    --do-sample \
    --pad-token-id 2 \
    --prompt "I believe the meaning of life is" \
    --check-accuracy-mode token-matching \
    --benchmark
```

### Example 4. Llama with quantization

```
inference_demo \
  --model-type llama \
  --task-type causal-lm \
  run \
    --model-path /home/ubuntu/model_hf/Llama-2-7b/ \
    --compiled-model-path /home/ubuntu/traced_model/Llama-2-7b-demo/ \
    --torch-dtype bfloat16 \
    --tp-degree 32 \
    --batch-size 2 \
    --max-context-length 32 \
    --seq-len 64 \
    --on-device-sampling \
    --enable-bucketing \
    --quantized \
    --quantized-checkpoints-path /home/ubuntu/model_hf/Llama-2-7b/model_quant.pt \
    --quantization-type per_channel_symmetric \
    --top-k 1 \
    --do-sample \
    --pad-token-id 2 \
    --prompt "I believe the meaning of life is" \
    --prompt "The color of the sky is"
```

### Example 5. Llama inference with logit matching accuracy check using custom error tolerances

```
inference_demo \
  --model-type llama \
  --task-type causal-lm \
  run \
    --model-path /home/ubuntu/model_hf/Llama-2-7b/ \
    --compiled-model-path /home/ubuntu/traced_model/Llama-2-7b-demo/ \
    --torch-dtype bfloat16 \
    --tp-degree 32 \
    --batch-size 2 \
    --max-context-length 32 \
    --seq-len 64 \
    --check-accuracy-mode logit-matching \
    --divergence-difference-tol 0.005 \
    --tol-map "{5: (1e-5, 0.02)}" \
    --enable-bucketing \
    --top-k 1 \
    --do-sample \
    --pad-token-id 2 \
    --prompt "I believe the meaning of life is" \
    --prompt "The color of the sky is"
```

## Contributing

### Formatting code

To format the code, use the following command:
```
pre-commit run --all-files
```