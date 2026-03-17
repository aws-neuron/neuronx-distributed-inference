# Contrib Model: GLM-4-9B-0414

NeuronX Distributed Inference implementation of GLM-4-9B-0414 (model_type="glm4").

## Model Information

- **HuggingFace ID:** `zai-org/GLM-4-9B-0414`
- **Architecture:** Glm4ForCausalLM (decoder-only, 4 RMSNorm layers per block)
- **Parameters:** 9B
- **License:** Check HuggingFace model card

## Key Differences from glm-4-9b-chat-hf

| Feature | GLM-4-9B-0414 | glm-4-9b-chat-hf |
|---------|---------------|-------------------|
| model_type | glm4 | glm |
| Layer norms per block | 4 | 2 |
| rms_norm_eps | 1e-05 | 1.5625e-07 |
| max_position_embeddings | 32768 | 131072 |

## Validation Results

**Validated:** 2026-03-17
**Configuration:** batch_size=1, seq_len=2048, tp_degree=2, bf16

| Metric | Value |
|--------|-------|
| Teacher-forced accuracy | 98.44% |
| Greedy accuracy | 67.81% |
| Test status | PASSED |
| Throughput (tp=2, bs=1) | 2.4 tokens/sec |

## Compilation

- Compiler: neuronx-cc with flags `--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1`
- NEFFs: context_encoding (2048 tokens) + token_generation (1 token)

## Usage

```python
from modeling_glm4 import NeuronGlm4ForCausalLM, Glm4InferenceConfig
```

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-17
