# Contrib Model: AFM 4.5B Base

NeuronX Distributed Inference implementation of AFM 4.5B Base.

## Model Information

- **HuggingFace ID:** `AFM-4.5B-Base`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=32, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ⚠️ LOW | **41.0% match** |
| Throughput | ⚠️ SLOW | 8.10 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 8.10 tokens/s |


**Status:** ⚠️ VALIDATED

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_afm_4_5b_base import NeuronAFM45BBaseForCausalLM, AFM45BBaseInferenceConfig

model_path = "/path/to/AFM-4.5B-Base/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=32,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = AFM45BBaseInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronAFM45BBaseForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/AFM-4.5B-Base/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/AFM-4.5B-Base
python3 test/integration/test_model.py
```

## Example Checkpoints

* AFM-4.5B-Base

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-29
