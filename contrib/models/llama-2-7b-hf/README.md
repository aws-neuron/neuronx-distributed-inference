# Contrib Model: Llama-2-7b-hf

NeuronX Distributed Inference implementation of Llama-2-7b-hf.

## Model Information

- **HuggingFace ID:** `meta-llama/Llama-2-7b-hf`
- **Model Type:** llama
- **License:** Llama 2 Community License Agreement

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_llama_2_7b_hf import NeuronLlama27bhfForCausalLM, Llama27bhfInferenceConfig

model_path = "/path/to/Llama-2-7b-hf/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = Llama27bhfInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronLlama27bhfForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/llama-2-7b-hf/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/llama-2-7b-hf
python3 test/integration/test_model.py
```

## Example Checkpoints

* meta-llama/Llama-2-7b-hf

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-27
