# Contrib Model: Gemma 2 9B

NeuronX Distributed Inference implementation of Google's Gemma 2 9B.

## Model Information

- **HuggingFace ID:** `google/gemma-2-9b`
- **Model Type:** Decoder-only transformer
- **Parameters:** 9B
- **License:** Check HuggingFace model card

## Architecture Details

- GQA: 16 attention heads, 8 KV heads
- head_dim: 256
- 42 decoder layers with 4 norms per layer (input, post_attn, pre_ff, post_ff)
- Scaled embeddings (embed * sqrt(hidden_size))
- RMSNorm with (1 + weight) scaling
- GELU tanh activation
- Tied embeddings (embed_tokens = lm_head)
- Final logit softcapping (30.0)
- Attention logit softcapping: disabled (NKI kernel limitation)
- Sliding window: disabled (head_dim=256 exceeds NKI limit of 128)

## Validation Results

**Validated:** 2026-03-20
**Configuration:** TP=8, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Compilation | PASS | Both context encoding and token generation NEFFs |
| Token Matching (Teacher-Forced) | PASS | **99.38% match** (10 prompts, 32 tokens each) |
| Token Matching (Greedy) | PASS | 84.69% match (8/10 prompts at 100%) |

**Status:** VALIDATED

## Usage

```python
import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from src.modeling_gemma2 import NeuronGemma2ForCausalLM, Gemma2InferenceConfig

model_path = "/path/to/gemma-2-9b/"
compiled_model_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = Gemma2InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

model = NeuronGemma2ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | PASS (99.38% TF) | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

```bash
cd contrib/models/gemma-2-9b
python3 test/integration/test_model.py
```

## Maintainer

Annapurna Labs

**Last Updated:** 2026-03-20
