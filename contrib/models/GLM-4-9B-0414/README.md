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

**Validated:** 2026-03-18
**Configuration:** batch_size=1, seq_len=128, tp_degree=2, bf16

| Metric | Value |
|--------|-------|
| Teacher-forced accuracy | **98.12%** (PASSED) |
| Greedy accuracy | 69.38% |
| Throughput (tp=2, bs=1) | 2.4 tokens/sec |
| Instance | trn1.32xlarge |

### Token Match Details (10 prompts, 32 tokens each)

| Prompt | Greedy | Teacher-Forced |
|--------|--------|----------------|
| "The theory of general relativity..." | 100.0% | 100.0% |
| "The French Revolution began in..." | 100.0% | 100.0% |
| "To solve a quadratic equation..." | 100.0% | 100.0% |
| "Once upon a time in a distant galaxy..." | 96.9% | 96.9% |
| "def fibonacci(n):..." | 18.8% | 96.9% |
| "The Amazon River flows through..." | 46.9% | 96.9% |
| "The concept of free will..." | 31.2% | 96.9% |
| "To make a cup of coffee, first..." | 100.0% | 100.0% |
| "List three benefits of regular exercise..." | 9.4% | 96.9% |
| "If all roses are flowers..." | 90.6% | 96.9% |

## Important: Use Glm4NeuronConfig

The compile script must use `Glm4NeuronConfig` (not base `NeuronConfig`) to set
`fused_qkv=True` and `attn_cls="NeuronGlm4Attention"`. Using base `NeuronConfig`
will cause a `KeyError` during weight sharding because the preshard hook expects
separate QKV keys while `convert_hf_to_neuron_state_dict` produces fused `Wqkv`.

## Compilation

- Compiler: neuronx-cc with flags `--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1`
- NEFFs: context_encoding + token_generation

## Usage

```python
from modeling_glm4 import NeuronGlm4ForCausalLM, Glm4InferenceConfig, Glm4NeuronConfig

neuron_config = Glm4NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
    save_sharded_checkpoint=True,
)
config = Glm4InferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronGlm4ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)
```

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-18
