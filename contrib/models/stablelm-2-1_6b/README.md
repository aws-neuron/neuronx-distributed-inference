# Contrib Model: stablelm 2 1 6b

NeuronX Distributed Inference implementation of stablelm 2 1 6b.

## Model Information

- **HuggingFace ID:** `stablelm-2-1_6b`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ⚠️ LOW | **40.6% match** |


**Status:** ⚠️ VALIDATED

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_stablelm_2_1_6b import Neuronstablelm216bForCausalLM, stablelm216bInferenceConfig

model_path = "/path/to/stablelm-2-1_6b/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = stablelm216bInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuronstablelm216bForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
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
pytest nxdi_contrib_models/models/stablelm-2-1_6b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/stablelm-2-1_6b
python3 test/integration/test_model.py
```

## Example Checkpoints

* stablelm-2-1_6b

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-29
