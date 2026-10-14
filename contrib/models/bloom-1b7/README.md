# Contrib Model: Bloom-1b7

NeuronX Distributed Inference implementation of bigscience/bloom-1b7.

## Model Information

- **HuggingFace ID:** `bigscience/bloom-1b7`
- **Model Type:** Decoder-only transformer
- **Parameters:** 1.7B
- **License:** BigScience RAIL License v1.0

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 2048 |
| Num Attention Heads | 16 (MHA) |
| Num Hidden Layers | 24 |
| Vocab Size | 250,880 |
| Position Encoding | ALiBi (Attention with Linear Biases) |
| Activation | GELU (tanh approximation) |
| Normalization | LayerNorm (with embedding LayerNorm) |
| QKV Format | Fused interleaved [num_heads, 3, head_dim, hidden_size] |
| Weight Tying | LM head tied to token embeddings |

### Key Implementation Notes

- **ALiBi positional encoding:** No position embeddings — bias is computed and added in custom `perform_prefill` and `compute_for_token_gen` overrides
- **Fused QKV weights:** Split into separate Q, K, V during weight conversion
- **Embedding LayerNorm:** Applied after token embeddings via `get_model_output` override
- **Standard LayerNorm** (not RMSNorm) with `layer_norm_epsilon`

## Validation Results

**Validated:** 2026-03-04
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **76.9% match** (vs HF bfloat16 reference) |

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

from src.modeling_bloom import NeuronBloomForCausalLM, BloomInferenceConfig

model_path = "/path/to/bloom-1b7/"
compiled_model_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    max_context_length=128,
    torch_dtype=torch.bfloat16,
)

config = BloomInferenceConfig.from_pretrained(
    model_path, neuron_config=neuron_config,
)

model = NeuronBloomForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Performance

Profiled on trn1.32xlarge (single NeuronCore utilization):

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| Throughput | - | 63.3 tok/s |
| MBU (Memory) | 18.6% | 16.1% |
| MFU (Compute) | 7.4% | 0.1% |

*Batch size 1, sequence length 128, BF16 precision, TP=1*

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/bloom-1b7/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* bigscience/bloom-1b7

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-04
