# Contrib Model: helium-1-2b

NeuronX Distributed Inference implementation of helium-1-2b.

## Model Information

- **HuggingFace ID:** `kyutai/helium-1-2b`
- **Model Type:** helium
- **License:** See HuggingFace model card

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_helium_1_2b import Neuronhelium12bForCausalLM, helium12bInferenceConfig

model_path = "/path/to/helium-1-2b/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = helium12bInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuronhelium12bForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/helium-1-2b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/helium-1-2b
python3 test/integration/test_model.py
```

## Example Checkpoints

* kyutai/helium-1-2b

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-27
