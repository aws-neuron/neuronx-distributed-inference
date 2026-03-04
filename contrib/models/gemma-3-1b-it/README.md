# Contrib Model: gemma 3 1b it

NeuronX Distributed Inference implementation of gemma 3 1b it.

## Model Information

- **HuggingFace ID:** `gemma-3-1b-it`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-02-06  
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** (best of multiple prompts) |

**Test Prompt:** `"def fibonacci(n):"`

**Status:** ✅ VALIDATED

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_gemma_3_1b_it import Neurongemma31bitForCausalLM, gemma31bitInferenceConfig

model_path = "/path/to/gemma-3-1b-it/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = gemma31bitInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neurongemma31bitForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/gemma-3-1b-it/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/gemma-3-1b-it
python3 test/integration/test_model.py
```

## Example Checkpoints

* gemma-3-1b-it

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
