# Contrib Model: OLMoE-1B-7B-0924

NeuronX Distributed Inference implementation of AllenAI OLMoE-1B-7B-0924.

## Model Information

- **HuggingFace ID:** `allenai/OLMoE-1B-7B-0924`
- **Model Type:** Decoder-only Mixture of Experts transformer
- **Parameters:** 7B total (1B active per token)
- **License:** Apache-2.0

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 2048 |
| Num Attention Heads | 16 (MHA) |
| Num Hidden Layers | 16 |
| Vocab Size | 50304 |
| Max Position Embeddings | 4096 |
| Head Dimension | 128 |
| Intermediate Size | 1024 (per expert) |
| Num Experts | 64 |
| Num Experts Per Token | 8 (top-8 routing) |
| Position Encoding | RoPE (theta=10000) |
| Residual Connection | Sequential |
| Normalization | RMSNorm |
| Activation | SiLU (SwiGLU) |

### Key Implementation Notes

- **Mixture of Experts:** 64 experts with top-8 routing per token. Uses MoENeuronConfig for compilation. Expert routing is deterministic (softmax + top-k selection).
- **Q/K normalization:** Applies RMSNorm to Q and K projections after splitting heads, similar to OLMo and Qwen3 architectures.
- **Standard RoPE:** Full rotary embeddings on all head dimensions (128/128), standard LLaMA-style half-split convention.
- **SwiGLU MLP experts:** Each expert uses gated linear unit with SiLU activation (gate_proj * up_proj pattern).

## Validation Results

**Validated:** 2026-03-09
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **81.0% average** (7/10 prompts at 100%) |
| Teacher-Forced Match | PASS | **98.0% average** |

### Greedy Match Details

7 of 10 prompts achieve 100% greedy match. The 3 prompts with divergence (10%, 25%, 80%) occur on high-entropy continuations where BF16 precision causes cascading differences. All prompts achieve >= 95% teacher-forced match.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

from src.modeling_olmoe import NeuronOlmoeForCausalLM, OlmoeInferenceConfig

model_path = "/path/to/OLMoE-1B-7B-0924/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = MoENeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = OlmoeInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronOlmoeForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Performance

Profiled on trn1.32xlarge (single NeuronCore utilization):

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| Throughput | - | 2.9 tok/s |
| MBU (Memory) | - | 31.0% |
| MFU (Compute) | - | 0.1% |

*Batch size 1, sequence length 128, BF16 precision, TP=1*
## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/OLMoE-1B-7B-0924/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* allenai/OLMoE-1B-7B-0924

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-09
