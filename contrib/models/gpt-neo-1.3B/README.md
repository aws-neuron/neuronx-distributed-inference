# Contrib Model: GPT-Neo-1.3B

NeuronX Distributed Inference implementation of GPT-Neo-1.3B.

## Model Information

- **HuggingFace ID:** `EleutherAI/gpt-neo-1.3B`
- **Model Type:** Decoder-only transformer
- **Parameters:** 1.3B
- **License:** MIT

## Architecture Details

- GPT-Neo uses **unscaled dot-product attention** (no 1/sqrt(d_k) division). This is handled by overriding `prep_qkv_tensors` to pre-multiply Q by sqrt(head_dim), canceling the base class's mandatory scaling.
- GPT-Neo alternates between **local** and **global** attention layers. For NXDI, all layers use full (global) KV cache for correctness.
- Uses **absolute learned position embeddings** (not rotary).
- Uses **tied weights** between token embeddings and LM head.

## Validation Results

**Validated:** 2026-03-16
**Configuration:** TP=1, batch_size=1, seq_len=128, bf16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Compilation | PASS | Compiler status PASS |
| Token Matching (greedy) | PASS | **52.97% match** (640 tokens, 10 prompts) |
| Token Matching (teacher-forced) | PASS | **98.59% match** |
| Throughput | PASS | **81.5 tokens/sec** |

**Status:** PASS - Teacher-forced accuracy confirms correct weight loading and computation.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from src.modeling_gpt_neo import NeuronGPTNeoForCausalLM, GPTNeoInferenceConfig

model_path = "/path/to/gpt-neo-1.3B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = GPTNeoInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronGPTNeoForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | PASS | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/gpt-neo-1.3B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/gpt-neo-1.3B
python3 test/integration/test_model.py
```

## Example Checkpoints

* EleutherAI/gpt-neo-1.3B

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-16
