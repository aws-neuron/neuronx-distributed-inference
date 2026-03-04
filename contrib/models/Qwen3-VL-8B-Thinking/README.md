# Contrib Model: Qwen3 VL 8B Thinking

NeuronX Distributed Inference implementation of Qwen3 VL 8B Thinking.

> **Note:** This implementation has been validated using the **text backbone only**. Vision/image modalities are implemented but not yet verified.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen3-VL-8B-Thinking`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Type:** Multimodal (vision-language) model with thinking/reasoning — text backbone validated only
- **Text Backbone:** Decoder-only transformer (Qwen3-based)
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
| TTFT (P50) | ✅ PASS | 93.57ms (threshold: 100ms) |
| Throughput | ✅ PASS | 10.66 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 93.57ms |
| Throughput | 10.66 tokens/s |

**Status:** ✅ VALIDATED

### Multimodal Validation Notes

Qwen3-VL is a vision-language model with thinking/reasoning capabilities. The NeuronX port validates the text backbone only. `AutoModelForCausalLM` does not work for VLMs — the specific text backbone class must be used to load the HF reference for token matching.

**Note:** Qwen3-VL requires dev transformers (5.0.0.dev0). The validation uses a subprocess approach to run the HF reference in a separate venv with the dev version, allowing version isolation without affecting the main environment. With the correct text backbone extraction, the model achieves 100% token match.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_qwen3_vl_8b_thinking import NeuronQwen3VL8BThinkingForCausalLM, Qwen3VL8BThinkingInferenceConfig

model_path = "/path/to/Qwen3-VL-8B-Thinking/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = Qwen3VL8BThinkingInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronQwen3VL8BThinkingForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Qwen3-VL-8B-Thinking/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Qwen3-VL-8B-Thinking
python3 test/integration/test_model.py
```

## Example Checkpoints

* Qwen/Qwen3-VL-8B-Thinking

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
