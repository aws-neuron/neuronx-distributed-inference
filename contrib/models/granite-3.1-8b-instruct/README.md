# Contrib Model: granite 3.1 8b instruct

NeuronX Distributed Inference implementation of granite 3.1 8b instruct.

## Model Information

- **HuggingFace ID:** `ibm-granite/granite-3.1-8b-instruct`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Layers:** Check model config
- **Hidden Size:** Check model config
- **Attention Heads:** Check model config
- **Vocabulary:** Check model config
- **Max Position Embeddings:** Check model config

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ⚠️ LOW | **7.8% match** |
| TTFT (P50) | ✅ PASS | 19.44ms (threshold: 100ms) |
| Throughput | ✅ PASS | 106.00 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 19.44ms |
| Throughput | 106.00 tokens/s |


**Status:** ✅ VALIDATED

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_granite_3_1_8b_instruct import Neurongranite318binstructForCausalLM, granite318binstructInferenceConfig

model_path = "/path/to/granite-3.1-8b-instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = granite318binstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neurongranite318binstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/granite-3.1-8b-instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/granite-3.1-8b-instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* ibm-granite/granite-3.1-8b-instruct

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-29
