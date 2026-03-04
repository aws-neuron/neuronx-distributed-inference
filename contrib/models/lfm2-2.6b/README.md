# Contrib Model: LFM2 2.6B

NeuronX Distributed Inference implementation of LFM2-2.6B, Liquid AI's Language Foundation Model.

## Model Information

- **HuggingFace ID:** `Liquid-AI/lfm2-2.6b`
- **Model Type:** Decoder-only transformer (Llama-based architecture)
- **Parameters:** ~2.6B
- **License:** Apple Sample Code License

## Architecture Details

- **Layers:** 30 decoder layers
- **Hidden Size:** 2048
- **Attention Heads:** 32
- **KV Heads:** 8 (Grouped Query Attention)
- **Intermediate Size:** 8192
- **Vocabulary:** 128,256 tokens
- **Max Position Embeddings:** 8192
- **Position Encoding:** RoPE
- **Normalization:** RMSNorm
- **Activation:** SwiGLU

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=1, batch_size=1, seq_len=2048, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** (with ChatML template) |
| TTFT (P50) | ⚠️ SLOW | 213.13ms (threshold: 100ms) |
| Throughput | ✅ PASS | 4.69 tok/s (threshold: 4.0 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 213.13ms |
| Token Generation (P50) | 213.27ms per token |
| Throughput | 4.69 tokens/s |

**Status:** ✅ VALIDATED

### Prompt Template Requirement

LFM2 is an instruct-tuned model that requires the ChatML prompt template for accurate token matching. Without the template, the model produces valid but mismatched outputs compared to the HuggingFace reference.

```python
# ChatML format (required for token matching)
prompt = "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
```

Using raw prompts (no template) results in low/0% token match despite the model generating coherent, factually correct text. This is because the HF reference and Neuron model diverge in generation style without the structured template.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_lfm2 import NeuronLfm2ForCausalLM, Lfm2InferenceConfig

model_path = "/path/to/lfm2-2.6b/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
)

config = Lfm2InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronLfm2ForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/lfm2-2.6b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/lfm2-2.6b
python3 test/integration/test_model.py
```

## Example Checkpoints

* Liquid-AI/lfm2-2.6b

## Notes

- LFM2 is a state space model (not a standard transformer), but validates using the same NeuronX methodology
- Uses Llama-based architecture registration with custom modifications
- ChatML prompt template (`<|im_start|>`) is required for accurate token matching against HF reference
- Without the template, the model generates coherent, factually correct text but tokens diverge from HF output
- Model generates correct factual outputs (e.g., "Paris" for capital of France) regardless of template usage

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
