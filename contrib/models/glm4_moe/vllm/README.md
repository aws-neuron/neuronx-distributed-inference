# GLM-4.5 MoE — vLLM Integration

This directory contains scripts for serving GLM-4.5 MoE via vLLM with the
`neuronx-distributed-inference` backend on Trn1/Trn2 instances.

## Prerequisites

- AWS Neuron SDK 2.21+
- `vllm-neuron` fork (supports `VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference`)
- `transformers>=4.56.0` (required for `Glm4MoeForCausalLM`)
- Trn1 (`trn1.32xlarge`) or Trn2 (`trn2.48xlarge`) instance

## Registration

GLM-4.5 MoE must be registered with vLLM before serving. Add the following
to your vLLM registration file (typically `vllm/model_executor/models/registry.py`):

```python
# In the MoE models section:
"Glm4MoeForCausalLM": ("glm4_moe", "NeuronGlm4MoeForCausalLM"),
```

Alternatively, set the model class via the NxDI model registry:

```python
from neuronx_distributed_inference.models.registry import register_model
from glm4_moe.modeling_glm4_moe import NeuronGlm4MoeForCausalLM

register_model("Glm4MoeForCausalLM", NeuronGlm4MoeForCausalLM)
```

## Offline Inference

```bash
python vllm/run_offline_inference.py \
    --model /path/to/GLM-4.5-Air \
    --tp-degree 32 \
    --seq-len 4096
```

## Online Serving (OpenAI-compatible API)

```bash
bash vllm/start-vllm-server.sh
```

Then query:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/GLM-4.5-Air",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Configuration

Key `override-neuron-config` parameters for GLM-4.5 MoE:

| Parameter | Recommended Value | Description |
|---|---|---|
| `tp_degree` | 32 (Trn1) / 64+ (Trn2) | Tensor parallelism |
| `moe_tp_degree` | Same as `tp_degree` | MoE tensor parallelism |
| `moe_ep_degree` | 1 | Expert parallelism (increase for EP) |
| `batch_size` | 1 | Static batch size |
| `seq_len` | 4096–32768 | Max sequence length |
| `fused_qkv` | `true` | Fused QKV for performance |
| `flash_decoding_enabled` | `true` (Trn2) | Flash attention for decoding |
