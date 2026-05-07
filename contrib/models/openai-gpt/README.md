# Contrib Model: openai-gpt

NeuronX Distributed Inference implementation of OpenAI GPT-1.

## Model Information

- **HuggingFace ID:** `openai-community/openai-gpt`
- **Model Type:** Decoder-only transformer (117M parameters)
- **License:** Check HuggingFace model card

## Architecture Details

OpenAI GPT-1 has several architectural differences from later GPT variants:

- **Post-norm:** LayerNorm is applied AFTER residual addition (not before, as in GPT-2/Neo)
- **No final LayerNorm:** The last transformer block's post-norm handles normalization
- **Conv1D weights:** HuggingFace checkpoint uses Conv1D format (transposed from nn.Linear)
- **Fused QKV:** Single `c_attn` weight matrix for Q, K, V projections
- **Standard GELU:** Uses erf-based GELU activation (not tanh-approximate `gelu_new`)
- **Absolute position embeddings:** Learned position embeddings up to 512 tokens

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

# Import model classes from src
from src.modeling_openai_gpt import NeuronOpenAIGPTForCausalLM, OpenAIGPTInferenceConfig

model_path = "/path/to/openai-gpt/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = OpenAIGPTInferenceConfig.from_pretrained(
    model_path, neuron_config=neuron_config,
)

# Compile and load
model = NeuronOpenAIGPTForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Validation Results

**Validated:** 2026-03-16
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Compilation | PASS | Context encoding + token generation compiled |
| Token Matching | PASS | **100.0% match** (200/200 tokens, 10 prompts) |
| Profiling | PASS | **437.4 tokens/sec** |

**Status:** EXCELLENT

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Performance

Profiled on trn1.32xlarge (single NeuronCore utilization):

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| Throughput | - | 449.3 tok/s |
| MBU (Memory) | 11.0% | 10.8% |
| MFU (Compute) | 4.5% | 0.0% |

*Batch size 1, sequence length 128, BF16 precision, TP=1*

## Testing

Run integration tests:

```bash
pytest contrib/models/openai-gpt/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/openai-gpt
python3 test/integration/test_model.py
```

## Example Checkpoints

* openai-community/openai-gpt

## Maintainer

Neuroboros Team - Annapurna Labs
