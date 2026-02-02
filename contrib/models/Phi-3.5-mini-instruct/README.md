# Contrib Model: Phi 3.5 mini instruct

NeuronX Distributed Inference implementation of Phi 3.5 mini instruct.

## Model Information

- **HuggingFace ID:** `Phi-3.5-mini-instruct`
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
| Token Matching | ⚠️ LOW | **28.1% match** |


**Status:** ⚠️ VALIDATED

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_phi_3_5_mini_instruct import NeuronPhi35miniinstructForCausalLM, Phi35miniinstructInferenceConfig

model_path = "/path/to/Phi-3.5-mini-instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = Phi35miniinstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronPhi35miniinstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Phi-3.5-mini-instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Phi-3.5-mini-instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* Phi-3.5-mini-instruct

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-29
