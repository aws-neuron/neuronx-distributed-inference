# Contrib Model: GPT-2

NeuronX Distributed Inference implementation of GPT-2.

## Model Information

- **HuggingFace ID:** `openai-community/gpt2`
- **Model Type:** Decoder-only transformer
- **Parameters:** ~124M
- **License:** MIT

## Architecture Details

- **Layers:** 12 decoder layers
- **Hidden Size:** 768
- **Attention Heads:** 12
- **Intermediate Size:** 3072
- **Vocabulary:** 50,257
- **Max Position Embeddings:** 1024

### GPT-2-Specific Features

| Feature | Value | Description |
|---------|-------|-------------|
| Position Embeddings | Absolute | Learned position embeddings (not RoPE) |
| Normalization | LayerNorm | Standard LayerNorm (not RMSNorm) |
| Activation | GELU | GELU activation in MLP |
| QKV Bias | True | Bias in attention projections |
| Tied Embeddings | True | lm_head shares weights with embed_tokens |
| QKV Layout | Fused | Combined QKV projection (c_attn) |

## Validation Results

**Validated:** 2026-02-07  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** |

### Benchmark Results (LightEval)

Full evaluation on all samples, compared against HF reference (CPU, float32).

| Task | Neuron (BF16) | HF (FP32) | Delta | Status |
|------|---------------|-----------|-------|--------|
| arc:challenge | 0.1937 | 0.1903 | +0.003 | ✅ PASS |
| arc:easy | 0.4398 | 0.4381 | +0.002 | ✅ PASS |
| hellaswag (em) | 0.0066 | 0.0050 | +0.002 | ✅ PASS |
| truthfulqa_mc1 | 0.2375 | 0.2277 | +0.010 | ✅ PASS |
| truthfulqa_mc2 | 0.4252 | 0.4069 | +0.018 | ✅ PASS |
| winogrande | 0.4862 | 0.4838 | +0.002 | ✅ PASS |

All benchmarks pass within ±2% of the HF reference. Largest delta is truthfulqa_mc2 at +1.8%.

**Status:** ✅ PASS

## Implementation Notes

### Absolute Position Embeddings

GPT-2 uses learned absolute position embeddings (not RoPE):

```python
# Token embeddings + Position embeddings
inputs_embeds = self.embed_tokens(input_ids)
position_embeds = self.wpe(position_ids)
hidden_states = inputs_embeds + position_embeds
```

### Conv1D Weight Transposition

GPT-2 uses Conv1D layers which store weights transposed:

```python
# HuggingFace Conv1D: weight shape [in_features, out_features]
# Standard Linear: weight shape [out_features, in_features]
# Must transpose during state dict conversion
weight = state_dict[f"{layer_prefix}.attn.c_attn.weight"].t().contiguous()
```

### Fused QKV Projection

GPT-2 uses a single combined QKV projection:

```python
# c_attn.weight shape: [hidden_size, 3 * hidden_size]
# Split into Q, K, V
qkv_weight = qkv_weight.t().contiguous()  # Transpose Conv1D
q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
```

### Vocab Size Padding

GPT-2's vocab size (50257) is not divisible by common TP degrees. Use `pad=True`:

```python
self.lm_head = ColumnParallelLinear(
    config.hidden_size,
    config.vocab_size,
    bias=False,
    gather_output=True,
    pad=True,  # Enable padding for non-divisible vocab sizes
)
```

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from src.modeling_gpt2 import NeuronGPT2ForCausalLM, GPT2InferenceConfig

model_path = "/path/to/gpt2/"
compiled_model_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = GPT2InferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronGPT2ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("Hello, I am a language model", return_tensors="pt")
# Use manual generation loop (see test file for example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Functional | Not tested |
| Inf2             | Not tested | Not tested |

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-07
