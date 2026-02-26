# Contrib Model: Ovis2.5 9B

NeuronX Distributed Inference implementation of Ovis2.5 9B.

> **Note:** This implementation has been validated using the **text backbone only**. Vision/image modalities are implemented but not yet verified.

## Model Information

- **HuggingFace ID:** `AIDC-AI/Ovis2.5-9B`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Type:** Multimodal (vision-language) model — text backbone validated only
- **Text Backbone:** Decoder-only transformer
- **Layers:** See model config
- **Hidden Size:** See model config
- **Attention Heads:** See model config
- **Vocabulary:** See model config
- **Max Position Embeddings:** See model config

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** (text backbone) |
| TTFT (P50) | ✅ PASS | 32.92ms (threshold: 100ms) |
| Throughput | ✅ PASS | 30.03 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 32.92ms |
| Throughput | 30.03 tokens/s |

**Status:** ✅ VALIDATED

### Multimodal Validation Notes

Ovis2.5 is a vision-language model. The NeuronX port validates the text backbone only. `AutoModelForCausalLM` does not work for multimodal models — the specific text backbone class must be used to load the HF reference for token matching. With the correct text backbone extraction, the model achieves 100% token match.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_ovis2_5_9b import NeuronOvis259BForCausalLM, Ovis259BInferenceConfig

model_path = "/path/to/Ovis2.5-9B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = Ovis259BInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronOvis259BForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Ovis2.5-9B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Ovis2.5-9B
python3 test/integration/test_model.py
```

## Example Checkpoints

* AIDC-AI/Ovis2.5-9B

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
