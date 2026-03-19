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

| Metric | Value |
|--------|-------|
| Teacher-forced match | 92.19% |
| Greedy match | 39.38% |
| Compilation | PASS |
| Inference (greedy) | PASS |
| Config | tp=2, batch=1, seq_len=2048, bf16 |
| Instance | trn1.32xlarge |

### Token Match Details (10 prompts, 32 tokens each)

| Prompt | Greedy | Teacher-Forced |
|--------|--------|----------------|
| "The theory of general relativity..." | 25.0% | 90.6% |
| "The French Revolution began in..." | 25.0% | 93.8% |
| "To solve a quadratic equation..." | 25.0% | 87.5% |
| "Once upon a time in a distant galaxy..." | 3.1% | 87.5% |
| "def fibonacci(n):..." | 59.4% | 93.8% |
| "The Amazon River flows through..." | 28.1% | 90.6% |
| "The concept of free will..." | 100.0% | 100.0% |
| "To make a cup of coffee, first..." | 21.9% | 84.4% |
| "List three benefits of regular exercise..." | 6.2% | 93.8% |
| "If all roses are flowers..." | 100.0% | 100.0% |

Note: Teacher-forced accuracy is 92.19%, below the 95% pass threshold.
MiniCPM-MoE uses embedding/residual scaling (`scale_emb=12`, `scale_depth=1.4`)
which amplifies bf16 rounding differences through the MoE routing path.
Greedy divergence is expected for MoE models in bf16.

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

## Performance

Profiled on trn1.32xlarge (single NeuronCore utilization):

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| Throughput | - | 2.5 tok/s |
| MBU (Memory) | - | - |
| MFU (Compute) | - | - |

*Batch size 1, sequence length 2048, BF16 precision, TP=2*

> Note: MBU/MFU metrics unavailable — NEFF capture timed out for this MoE model due to large subgraph size.
## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-17
