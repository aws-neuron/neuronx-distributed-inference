# Contrib Model: SmolLM3-3B

NeuronX Distributed Inference implementation of SmolLM3-3B.

## Model Information

- **HuggingFace ID:** `HuggingFaceTB/SmolLM3-3B`
- **Model Type:** smollm3
- **License:** See HuggingFace model card

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_smollm3_3b import NeuronSmolLM33BForCausalLM, SmolLM33BInferenceConfig

model_path = "/path/to/SmolLM3-3B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = SmolLM33BInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronSmolLM33BForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/smollm3-3b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/smollm3-3b
python3 test/integration/test_model.py
```

## Example Checkpoints

* HuggingFaceTB/SmolLM3-3B

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-27
