# Contrib Model: minicpm-moe-8x2b

NeuronX Distributed Inference implementation of MiniCPM-MoE-8x2B.

## Model Information

- **HuggingFace ID:** `openbmb/MiniCPM-MoE-8x2b`
- **Architecture:** MiniCPM Mixture of Experts (8 experts, top-2 routing)
- **Parameters:** ~14B total, ~2B active per token
- **Hidden size:** 2304, 40 layers, 36 attention heads (MHA)
- **Vocab size:** 122,753
- **Context length:** 4,096 tokens
- **License:** Check HuggingFace model card

## Key Architecture Features

- Embedding scaling (`scale_emb=12`)
- Residual depth scaling (`scale_depth=1.4`)
- GLU MLP with softmax routing
- RoPE with default theta (10000.0)

## Compilation Config

- **TP degree:** 2
- **Batch size:** 1
- **Sequence length:** 2048
- **Dtype:** bfloat16
- **Compile time:** ~6 min (build) + ~7 min (sharding)

## Validation Results

**Validated:** 2026-03-17

| Metric | Result |
|--------|--------|
| Compilation | PASS |
| Inference (greedy) | PASS |

**Sample outputs:**
- "The capital of France is" -> "a city with a rich history and cultural heritage..."
- "def fibonacci(n):" -> "if n <= 1: return n else: return..."
- "In machine learning, neural networks" -> "are a type of model that is inspired by the structure..."

**Note:** Full HF token-match comparison skipped due to MoE model size (26GB) making CPU-based HF inference impractically slow.

## Usage

```python
from modeling_minicpm_moe_neuronx import NeuronMiniCPMMoEForCausalLM, MiniCPMMoEInferenceConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig

neuron_config = MoENeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
    save_sharded_checkpoint=True,
)
config = MiniCPMMoEInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronMiniCPMMoEForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)
```

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-17
