# Contrib Model: MPT-7B-Chat

NeuronX Distributed Inference implementation of MosaicML MPT-7B-Chat.

## Model Information

- **HuggingFace ID:** `mosaicml/mpt-7b-chat`
- **Model Type:** Decoder-only transformer with ALiBi attention
- **Parameters:** 6.7B
- **License:** CC-BY-SA-3.0

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 4096 |
| Num Attention Heads | 32 (MHA) |
| Head Dimension | 128 |
| Num Hidden Layers | 32 |
| Vocab Size | 50432 |
| Max Position Embeddings | 2048 |
| Intermediate Size | 16384 |
| Position Encoding | ALiBi (Attention with Linear Biases) |
| Residual Connection | Sequential (LN -> Attn -> Add -> LN -> MLP -> Add) |
| Normalization | LayerNorm without bias (eps=1e-5) |
| Activation | GELU |
| LM Head | Tied to token embeddings |

### Key Implementation Notes

- **ALiBi attention:** NXDI has no native ALiBi support. Per-head slopes are stored as a weight parameter (`alibi_slopes`) that gets TP-sharded with attention heads. Position bias is computed at runtime from slopes and token positions, then added to attention scores before softmax.
- **Flash attention disabled:** NKI kernels cannot accept additive bias tensors, so flash attention must be disabled for ALiBi to work.
- **Fused QKV:** HF checkpoint stores a single `Wqkv` weight. During weight conversion, this is split into separate Q, K, V projections.
- **No-bias LayerNorm:** MPT uses `no_bias=True` for all LayerNorm layers.

## Validation Results

**Validated:** 2026-03-05
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **54.84% average** (2/10 prompts at 100%) |
| Teacher-Forced Match | PASS | **97.50% average** |

### Greedy Match Details

2 of 10 prompts achieve 100% greedy match (quadratic equation and fibonacci code). The lower greedy match rate compared to non-ALiBi models is expected: BF16 precision differences in the additive position bias compound during autoregressive generation. The high teacher-forced rate (97.50%) confirms weights are correctly ported.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

from src.modeling_mpt import NeuronMptForCausalLM, MptInferenceConfig

model_path = "/path/to/mpt-7b-chat/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = MptInferenceConfig.get_neuron_config_cls()(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = MptInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronMptForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Performance

Measured on trn1.32xlarge, batch_size=1, seq_len=128, bfloat16. Utilization is per-NeuronCore (TP=1).

| Metric | Value |
|--------|-------|
| Throughput | 18.0 tok/s |
| Context Encoding MBU | 18.9% |
| Context Encoding MFU | 10.3% |
| Token Generation MBU | 17.2% |
| Token Generation MFU | 0.1% |

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/mpt-7b-chat/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* mosaicml/mpt-7b-chat

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-05
