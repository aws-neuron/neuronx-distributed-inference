# Contrib Model: GPT-Neo-1.3B

NeuronX Distributed Inference implementation of EleutherAI GPT-Neo-1.3B.

## Model Information

- **HuggingFace ID:** `EleutherAI/gpt-neo-1.3B`
- **Model Type:** Decoder-only transformer
- **Parameters:** 1.3B
- **License:** MIT

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 2048 |
| Num Attention Heads | 16 (MHA) |
| Num Hidden Layers | 24 |
| Vocab Size | 50257 |
| Max Position Embeddings | 2048 |
| Head Dimension | 128 |
| Intermediate Size | 8192 |
| Position Encoding | Absolute (learned embeddings) |
| Attention Pattern | Alternating global/local (window_size=256) |
| Residual Connection | Sequential (LN -> Attn -> Add -> LN -> MLP -> Add) |
| Normalization | LayerNorm |
| Activation | GELU-new |
| LM Head | Tied with token embeddings |

### Key Implementation Notes

- **Unscaled attention:** GPT-Neo does NOT divide attention scores by 1/sqrt(head_dim). The base class always scales, so we pre-multiply Q by sqrt(head_dim) to cancel the scaling (net effect = 1.0).
- **Local/global attention:** GPT-Neo alternates between local (windowed) and global attention layers. All layers are treated as global in this NXDI port for simplicity and correctness.
- **Absolute position embeddings:** Uses learned position embeddings added to token embeddings (ParallelEmbedding), injected via get_model_output() override.
- **Tied weights:** LM head reuses token embedding weights.

## Validation Results

**Validated:** 2026-03-04
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **100% match on 10/10 prompts** |
| Teacher-Forced Match | PASS | **100% average** |

### Greedy Match Details

100% greedy token match across all 10 validation prompts (avg ~119 tokens each). This model achieves perfect bit-exact reproduction of the HF reference on Neuron hardware.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

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

config = GPTNeoInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
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
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/gpt-neo-1.3B/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* EleutherAI/gpt-neo-1.3B

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-04
