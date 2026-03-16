# Contrib Model: CTRL

NeuronX Distributed Inference implementation of Salesforce CTRL.

## Model Information

- **HuggingFace ID:** `Salesforce/ctrl`
- **Model Type:** Decoder-only transformer (conditional text generation)
- **Parameters:** 1.63B
- **Architecture:** 48 layers, 16 heads, hidden_size=1280, dff=8192
- **Position Encoding:** Sinusoidal (computed, not learned)
- **Activation:** ReLU
- **License:** BSD-3-Clause

## Architecture Details

CTRL uses control codes (e.g., `Links`, `Wikipedia`, `Science`) prepended to prompts to condition generation on a particular domain/style. Key differences from standard GPT-2:

- Sinusoidal position embeddings (not learned)
- Input embeddings scaled by sqrt(d_model)
- Separate lm_head with bias
- ReLU activation (not GELU)
- Pre-LayerNorm architecture

## Validation Results

**Validated:** 2026-03-16
**Configuration:** TP=1, batch_size=1, seq_len=128, bf16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Token Matching (greedy) | PASS | **99.06% match** (634/640 tokens) |
| Token Matching (teacher-forced) | PASS | **99.84% match** |
| Profiling | PASS | **59.2 tokens/sec** |

**Status:** EXCELLENT

## Usage

```python
import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from modeling_ctrl import NeuronCTRLForCausalLM, CTRLInferenceConfig

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
    model_path, neuron_config=neuron_config,
)

# Compile and load
model = NeuronCTRLForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate with control code
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("Wikipedia The theory of relativity", return_tensors="pt")
# ... (see integration test for full example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | PASS (99.84% TF) | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/ctrl/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/ctrl
python3 test/integration/test_model.py
```

## Example Checkpoints

* Salesforce/ctrl

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-16
**Validation Job:** SLURM 9589
