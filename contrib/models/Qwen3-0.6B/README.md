# Contrib Model: Qwen3 0.6B

NeuronX Distributed Inference implementation of Qwen3 0.6B.

## Model Information

- **HuggingFace ID:** `Qwen3-0.6B`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=8, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100.0% match** |
| Throughput | ✅ PASS | 196.00 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 196.00 tokens/s |


**Status:** ✅ EXCELLENT

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_qwen3_0_6b import NeuronQwen306BForCausalLM, Qwen306BInferenceConfig

model_path = "/path/to/Qwen3-0.6B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=8,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = Qwen306BInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronQwen306BForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Qwen3-0.6B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Qwen3-0.6B
python3 test/integration/test_model.py
```

## Example Checkpoints

* Qwen3-0.6B

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-29
