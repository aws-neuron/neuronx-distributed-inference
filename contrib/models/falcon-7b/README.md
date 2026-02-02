# Contrib Model: falcon 7b

NeuronX Distributed Inference implementation of falcon 7b.

## Model Information

- **HuggingFace ID:** `tiiuae/falcon-7b`
- **Model Type:** Decoder-only transformer
- **License:** Apache-2.0

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **98.8% match** |
| TTFT (P50) | ✅ PASS | 50.00ms (threshold: 100ms) |
| Throughput | ✅ PASS | 18.72 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 50.00ms |
| Throughput | 18.72 tokens/s |


**Status:** ✅ EXCELLENT

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_falcon_7b import Neuronfalcon7bForCausalLM, falcon7bInferenceConfig

model_path = "/path/to/falcon-7b/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = falcon7bInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuronfalcon7bForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/falcon-7b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/falcon-7b
python3 test/integration/test_model.py
```

## Example Checkpoints

* tiiuae/falcon-7b

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-29
