# Contrib Model: GPT-J

NeuronX Distributed Inference implementation of GPT-J.

## Model Information

- **HuggingFace ID:** `EleutherAI/gpt-j-6b`
- **Model Type:** Decoder-only transformer
- **Parameters:** ~6B
- **License:** Apache 2.0

## Architecture Details

- **Layers:** 28 decoder layers
- **Hidden Size:** 4096
- **Attention Heads:** 16
- **Intermediate Size:** 16384 (4x hidden)
- **Vocabulary:** 50,400
- **Max Position Embeddings:** 2048

### GPT-J-Specific Features

| Feature | Value | Description |
|---------|-------|-------------|
| Residual Connection | Parallel | attn(ln(x)) + mlp(ln(x)) + x (not sequential) |
| Position Embeddings | Partial RoPE | Rotary on 64/256 dims with GPT-J interleaved rotation |
| Normalization | LayerNorm | Single LayerNorm per block (not RMSNorm) |
| Activation | GELU (new) | GELU new activation in MLP |
| QKV Bias | False | No bias in Q/K/V projections |
| MLP Bias | True | Bias in fc_in and fc_out |
| Key-Value Heads | 16 (MHA) | Multi-head attention, not GQA |

## Validation Results

**Validated:** 2026-03-04
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **72.81% greedy match** (466/640 tokens) |
| Teacher-Forced Matching | PASS | **98.91% match** |

### Per-Prompt Results

| Prompt | Greedy | Teacher-Forced |
|--------|--------|----------------|
| The theory of general relativity explains | 100.0% | 100.0% |
| The French Revolution began in | 29.7% | 98.4% |
| To solve a quadratic equation, you can | 100.0% | 100.0% |
| Once upon a time in a distant galaxy, | 71.9% | 98.4% |
| def fibonacci(n): """Return the nth | 100.0% | 100.0% |
| The Amazon River flows through | 96.9% | 98.4% |
| The concept of free will has been debated | 28.1% | 98.4% |
| To make a cup of coffee, first | 56.2% | 96.9% |
| List three benefits of regular exercise: | 100.0% | 100.0% |
| If all roses are flowers... | 45.3% | 98.4% |

Teacher-forced accuracy of 98.91% confirms per-token predictions are nearly identical to HF. Greedy divergences are from small floating-point differences snowballing during autoregressive generation.

**Status:** PASS

## Implementation Notes

### Parallel Residual Connections

GPT-J uses parallel residual connections where attention and MLP are computed on the same normalized input:

```python
normed = self.input_layernorm(hidden_states)
attn_output = self.self_attn(normed, ...)
mlp_output = self.mlp(normed)
hidden_states = attn_output + mlp_output + hidden_states  # parallel residual
```

### Partial Rotary Embeddings

GPT-J applies rotary embeddings to only the first 64 of 256 head dimensions, using an interleaved rotation pattern (not Llama's rotate_half):

```python
# GPT-J rotation: interleaved pairs (-x_odd, x_even)
def rotate_every_two(x):
    x = x.reshape(*leading, d // 2, 2)
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1).reshape(*leading, d)

# Apply to first rotary_dim dimensions only
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
q_rot = (q_rot * cos) + (rotate_every_two(q_rot) * sin)
q = torch.cat([q_rot, q_pass], dim=-1)
```

### Weight Mapping

GPT-J HF weights map to NXDI as follows:

| HuggingFace Key | NXDI Key |
|-----------------|----------|
| `transformer.wte.weight` | `embed_tokens.weight` |
| `transformer.ln_f.{weight,bias}` | `norm.{weight,bias}` |
| `transformer.h.{i}.ln_1.*` | `layers.{i}.input_layernorm.*` |
| `transformer.h.{i}.attn.{q,k,v}_proj.weight` | `layers.{i}.self_attn.qkv_proj.{q,k,v}_proj.weight` |
| `transformer.h.{i}.attn.out_proj.weight` | `layers.{i}.self_attn.o_proj.weight` |
| `transformer.h.{i}.mlp.fc_in.*` | `layers.{i}.mlp.fc_in.*` |
| `transformer.h.{i}.mlp.fc_out.*` | `layers.{i}.mlp.fc_out.*` |

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from src.modeling_gptj import NeuronGPTJForCausalLM, GPTJInferenceConfig

model_path = "/path/to/gpt-j-6b/"
compiled_model_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = GPTJInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronGPTJForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("Hello, I am a language model", return_tensors="pt")
# Use manual generation loop (see test file for example)
```

## Performance

Measured on trn1.32xlarge, batch_size=1, seq_len=128, bfloat16. Utilization is per-NeuronCore (TP=1).

| Metric | Value |
|--------|-------|
| Throughput | 20.2 tok/s |
| Context Encoding MBU | 18.9% |
| Context Encoding MFU | 10.3% |
| Token Generation MBU | 17.0% |
| Token Generation MFU | 0.1% |

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Functional | Not tested |
| Inf2             | Not tested | Not tested |

## Maintainer

Annapurna Labs

**Last Updated:** 2026-03-04
