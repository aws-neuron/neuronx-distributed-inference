# Contrib Model: CTRL

NeuronX Distributed Inference implementation of Salesforce CTRL.

## Model Information

- **HuggingFace ID:** `Salesforce/ctrl`
- **Model Type:** Decoder-only transformer (conditional generation with control codes)
- **Parameters:** 1.6B
- **License:** BSD-3-Clause

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 1280 |
| Num Attention Heads | 16 (MHA) |
| Head Dimension | 80 |
| Num Hidden Layers | 48 |
| Vocab Size | 246534 |
| Max Position Embeddings | 50000 |
| Intermediate Size (dff) | 8192 |
| Position Encoding | Sinusoidal (fixed, concatenated [sines\|cosines]) |
| Residual Connection | Sequential (LN -> Attn -> Add -> LN -> MLP -> Add) |
| Normalization | LayerNorm (epsilon=1e-6) |
| Activation | ReLU |
| LM Head | Separate with bias |

### Key Implementation Notes

- **Sinusoidal positions:** CTRL uses fixed sinusoidal encodings (not learned) with HF's concatenated layout `[sin(f0),...,sin(fN), cos(f0),...,cos(fN)]` — generated during weight conversion, not stored in checkpoint.
- **Embedding scaling:** Token embeddings are scaled by sqrt(d_model=1280) before adding position embeddings.
- **Control codes:** CTRL uses special control code tokens (e.g., "Wikipedia", "Books") to condition generation style. The large vocab (246K) includes these codes.
- **Forward override:** Overrides the full forward() signature (not get_model_output()) to inject position + scaled token embeddings before the base class processes them.
- **Custom NeuronConfig:** Uses CTRLNeuronConfig subclass with attn_cls set to NeuronCTRLAttention (required for token generation tracing).

## Validation Results

**Validated:** 2026-03-04
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **99.06% average** (8/10 prompts at 100%) |
| Teacher-Forced Match | PASS | **99.84% average** |

### Greedy Match Details

8 of 10 prompts achieve 100% greedy match. The 2 prompts with slight divergence (95%, 97%) occur on high-entropy natural language where BF16 precision causes late-stage cascading differences. HF reference requires float32 dtype override (CTRL LayerNorm doesn't support bf16 on CPU).

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

from src.modeling_ctrl import NeuronCTRLForCausalLM, CTRLInferenceConfig

model_path = "/path/to/ctrl/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = CTRLInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronCTRLForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate (CTRL uses control codes as prompt prefix)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/ctrl/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* Salesforce/ctrl

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-04
