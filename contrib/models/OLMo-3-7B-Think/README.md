# Contrib Model: OLMo 3 7B Think

NeuronX Distributed Inference implementation of OLMo 3 7B Think.

## Model Information

- **HuggingFace ID:** `allenai/OLMo-3-7B-Think`
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
| Cosine Similarity | ✅ PASS | **0.9975** |
| Top-1 Accuracy | ✅ PASS | **100%** |

**Status:** EXCELLENT

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_olmo_3_7b_think import Model, Config

model_path = "/path/to/OLMo-3-7B-Think/"
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
pytest nxdi_contrib_models/models/OLMo-3-7B-Think/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* allenai/OLMo-3-7B-Think

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-30
