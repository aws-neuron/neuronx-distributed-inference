# Contrib Model: gpt_bigcode-santacoder

NeuronX Distributed Inference implementation of gpt_bigcode-santacoder.

## Model Information

- **HuggingFace ID:** ``
- **Model Type:** causal_lm
- **License:** {'model_license': 'BigCode OpenRAIL-M', 'port_license': 'Apache-2.0'}

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_santacoder import Neurongpt_bigcodesantacoderForCausalLM, gpt_bigcodesantacoderInferenceConfig

model_path = "/path/to/gpt_bigcode-santacoder/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = gpt_bigcodesantacoderInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neurongpt_bigcodesantacoderForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/santacoder/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/santacoder
python3 test/integration/test_model.py
```

## Example Checkpoints

* 

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-27
