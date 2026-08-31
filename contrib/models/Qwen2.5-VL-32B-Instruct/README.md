# Qwen2.5-VL-32B-Instruct

> **Use the unified implementation at [`Qwen2.5-VL-7B-Instruct`](../Qwen2.5-VL-7B-Instruct/).** That implementation is config-driven and supports all Qwen2.5-VL sizes (3B, 7B, 32B, 72B) including both text-only and vision-language inference.

The code previously in this directory's `src/` was a text-only stub that lacked M-RoPE and the vision encoder. It achieved 0% token match in validation and has been removed to avoid confusion.

## 32B-Specific Guidance

### Model Dimensions

| Parameter | Value |
|-----------|-------|
| Layers | 64 |
| Hidden Size | 5120 |
| Attention Heads (Q / KV) | 40 / 8 (GQA) |
| Intermediate Size | 27648 |
| Vocab | 152064 |
| `tie_word_embeddings` | False |
| M-RoPE sections | [16, 24, 24] (same as all Qwen2.5-VL sizes) |

### Recommended Configuration

| Instance | TP | Notes |
|----------|----|-------|
| trn2.3xlarge (LNC=1) | 8 | ~64 GB BF16 weights. Requires LNC=1 for TP=8 on trn2.3xlarge. |
| trn2.48xlarge | 8-16 | More headroom for KV cache and batch size > 1. |

- **TP=8** is the minimum for 32B (~64 GB BF16 weights). On trn2.3xlarge this requires **LNC=1** (8 logical cores, 12 GB HBM each).
- TP=4 will not fit -- the model is too large for 4 cores at LNC=2 (24 GB/core, ~16 GB weights/core).
- The MLP NKI kernel status is untested for 32B (`intermediate_size/TP = 3456` at TP=8, which is within SBUF limits).
- Multi-bucket CTE should work the same as 7B -- use `context_encoding_buckets=[512, 1024, 2048, 4096]`.

### Quick Start

```python
# Point model_path at the 32B checkpoint -- same code as VL-7B
model_path = "/path/to/Qwen2.5-VL-32B-Instruct"

# Use the unified implementation from the VL-7B contrib
from src.modeling_qwen2_5_vl import (
    NeuronQwen2_5_VLForCausalLM,
    Qwen2_5_VLInferenceConfig,
)
```

See the [Qwen2.5-VL-7B-Instruct README](../Qwen2.5-VL-7B-Instruct/README.md) for full usage examples, vllm-neuron serving instructions, and known limitations.
