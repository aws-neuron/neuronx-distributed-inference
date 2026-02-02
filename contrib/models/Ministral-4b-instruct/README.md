# Contrib Model: Ministral 4b instruct

NeuronX Distributed Inference implementation of Ministral 4b instruct.

## Model Information

- **HuggingFace ID:** `mistralai/Ministral-4b-instruct`
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
| Token Matching | ✅ PASS | **100.0% match** |
| TTFT (P50) | ✅ PASS | 5.00ms (threshold: 100ms) |
| Throughput | ✅ PASS | 45.35 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 5.00ms |
| Throughput | 45.35 tokens/s |


**Status:** ✅ EXCELLENT

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_ministral_4b_instruct import NeuronMinistral4binstructForCausalLM, Ministral4binstructInferenceConfig

model_path = "/path/to/Ministral-4b-instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = Ministral4binstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronMinistral4binstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Ministral-4b-instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Ministral-4b-instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* mistralai/Ministral-4b-instruct

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-29
