# Contrib Model: Qwen2.5 Omni 7B

NeuronX Distributed Inference implementation of Qwen2.5 Omni 7B.

> **Note:** This implementation has been validated using the **text backbone only**. Vision/audio modalities are implemented but not yet verified.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen2.5-Omni-7B`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Type:** Multimodal (omni — vision, audio, text) model — text backbone validated only
- **Text Backbone:** Decoder-only transformer (Qwen2-based)
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
| Token Matching | ✅ PASS | **100% match** (text backbone) |
| TTFT (P50) | ✅ PASS | 50.15ms (threshold: 100ms) |
| Throughput | ✅ PASS | 19.82 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 50.15ms |
| Throughput | 19.82 tokens/s |

**Status:** ✅ VALIDATED

### Multimodal Validation Notes

Qwen2.5-Omni is a multimodal model supporting vision, audio, and text. The NeuronX port validates the text backbone only. `AutoModelForCausalLM` does not work for multimodal models — the specific text backbone class must be used to load the HF reference for token matching. Some multimodal configs may be missing attributes expected by the text backbone (e.g., `output_attentions`) and require config patching. With the correct text backbone extraction, the model achieves 100% token match.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_qwen2_5_omni_7b import NeuronQwen25Omni7BForCausalLM, Qwen25Omni7BInferenceConfig

model_path = "/path/to/Qwen2.5-Omni-7B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = Qwen25Omni7BInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronQwen25Omni7BForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Qwen2.5-Omni-7B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Qwen2.5-Omni-7B
python3 test/integration/test_model.py
```

## Example Checkpoints

* Qwen/Qwen2.5-Omni-7B

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
