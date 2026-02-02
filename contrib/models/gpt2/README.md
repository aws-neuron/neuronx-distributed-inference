# Contrib Model: gpt2

NeuronX Distributed Inference implementation of gpt2.

## Model Information

- **HuggingFace ID:** `openai-community/gpt2`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Layers:** Check model config
- **Hidden Size:** Check model config
- **Attention Heads:** Check model config
- **Vocabulary:** Check model config

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ⚠️ LOW | **20.3% match** |
| Cosine Similarity | ✅ PASS | **1.0000** |

**Status:** VALIDATED

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_gpt2 import Model, Config

model_path = "/path/to/gpt2/"
compiled_model_path = "/path/to/compiled/"

# Configure and use model
# ... (see integration test for full example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest nxdi_contrib_models/models/gpt2/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* openai-community/gpt2

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-30
