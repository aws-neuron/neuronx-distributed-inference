# Contrib Model: CodeGen-350M-mono

NeuronX Distributed Inference implementation of Salesforce CodeGen-350M-mono.

## Model Information

- **HuggingFace ID:** `Salesforce/codegen-350M-mono`
- **Model Type:** Decoder-only transformer (code generation)
- **Parameters:** 350M
- **License:** Apache-2.0

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 1024 |
| Num Attention Heads | 16 (MHA) |
| Num Hidden Layers | 20 |
| Vocab Size | 51200 |
| Max Position Embeddings | 2048 |
| Head Dimension | 64 |
| Rotary Dim | 32 (partial RoPE, 32/64 dims) |
| Rotation Convention | GPT-J rotate_every_two |
| Residual Connection | Parallel (attn + mlp + residual) |
| Normalization | LayerNorm |
| Activation | GELU-new |

### Key Implementation Notes

- **Partial RoPE:** Only 32 of 64 head dimensions use rotary embeddings, following GPT-J's interleaved rotation convention (not the standard LLaMA half-split)
- **Fused QKV:** HuggingFace checkpoint uses fused `qkv_proj` with `mp_num=4` and (Q, V, K) interleaved order — the weight converter handles this decomposition
- **Parallel residual:** Attention and MLP both operate on the layer-normed input, and their outputs are summed with the residual

## Validation Results

**Validated:** 2026-03-04
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **100% match on 14/30 prompts** (64 tokens each) |
| Teacher-Forced Match | PASS | **97.03% average** |

### Greedy Match Details

Average greedy match: 54.48% across all prompts (including high-entropy natural language).
100% greedy match achieved on structured/low-entropy prompts (code, sequences, JSON, math).

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from src.modeling_codegen import NeuronCodeGenForCausalLM, CodeGenInferenceConfig

model_path = "/path/to/codegen-350M-mono/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = CodeGenInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronCodeGenForCausalLM(model_path, config)
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
| Throughput | 187.4 tok/s |
| Context Encoding MBU | 10.3% |
| Context Encoding MFU | 4.5% |
| Token Generation MBU | 11.5% |
| Token Generation MFU | 0.1% |

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/codegen-350M-mono/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* Salesforce/codegen-350M-mono

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-04
