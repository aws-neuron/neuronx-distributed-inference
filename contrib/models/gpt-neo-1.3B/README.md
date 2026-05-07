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
| Greedy Token Matching | PASS | **100% match on 10/10 prompts** (1185 tokens) |
| Teacher-Forced Match | PASS | **100% average** |
| Throughput | PASS | **81.5 tokens/sec** |

**Status:** EXCELLENT - Perfect greedy and teacher-forced match across all prompts.

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

## Performance

Profiled on trn1.32xlarge (single NeuronCore utilization):

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| Throughput | - | 81.9 tok/s |
| MBU (Memory) | 18.1% | 15.2% |
| MFU (Compute) | 9.4% | 0.1% |

*Batch size 1, sequence length 128, BF16 precision, TP=1*

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
