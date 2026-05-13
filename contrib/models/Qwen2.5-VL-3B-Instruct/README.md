# Qwen2.5-VL-3B-Instruct

> **Use the unified implementation at [`Qwen2.5-VL-7B-Instruct`](../Qwen2.5-VL-7B-Instruct/).** That implementation is config-driven and supports all Qwen2.5-VL sizes (3B, 7B, 32B, 72B) including both text-only and vision-language inference.

The code previously in this directory's `src/` was a text-only stub that lacked M-RoPE and the vision encoder. It has been removed to avoid confusion.

## 3B-Specific Guidance

### Model Dimensions

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden Size | 2048 |
| Attention Heads (Q / KV) | 16 / 2 (GQA) |
| Intermediate Size | 11008 |
| Vocab | 152064 |
| `tie_word_embeddings` | **True** (lm_head shares embed_tokens weights) |
| M-RoPE sections | [16, 24, 24] (same as all Qwen2.5-VL sizes) |

### Recommended Configuration

| Instance | TP | TKG tok/s | Compile Time | Weights/Core |
|----------|----|-----------|--------------|-------------|
| trn2.3xlarge (LNC=2) | 4 | 104.3 | 56.4s | 2.1 GB |
| inf2.xlarge | 2 | ~29 | ~148s | ~3 GB |

- **TP=4 on trn2.3xlarge** is the best option for throughput.
- **TP=2 on inf2.xlarge** works and is the cheapest option. The model is small enough (~6 GB BF16) to fit in 2 NeuronCores.
- The MLP NKI kernel compiles for 3B (`intermediate_size/TP = 2752`) but is 13% slower than baseline -- not recommended.
- Tied weights are handled automatically by `update_state_dict_for_tied_weights` in the unified implementation.

### Quick Start

```python
# Point model_path at the 3B checkpoint -- same code as VL-7B
model_path = "/path/to/Qwen2.5-VL-3B-Instruct"

# Use the unified implementation from the VL-7B contrib
from src.modeling_qwen2_5_vl import (
    NeuronQwen2_5_VLForCausalLM,
    Qwen2_5_VLInferenceConfig,
)
```

See the [Qwen2.5-VL-7B-Instruct README](../Qwen2.5-VL-7B-Instruct/README.md) for full usage examples, vllm-neuron serving instructions, and known limitations.
