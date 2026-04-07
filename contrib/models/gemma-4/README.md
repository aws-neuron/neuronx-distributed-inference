# Contrib Model: Gemma-4

NeuronX Distributed Inference implementation of the Gemma-4 model family (text-only path).

Supports all 4 variants in a single `modeling_gemma4.py`:

| Variant | HuggingFace ID | Params | Architecture |
|---------|---------------|--------|-------------|
| E2B | `google/gemma-4-2b-it` | ~2B | Dense, PLE, KV sharing, double-wide MLP |
| E4B | `google/gemma-4-4b-it` | ~4B | Dense, PLE, KV sharing, double-wide MLP |
| 31B | `google/gemma-4-31B-it` | ~31B | Dense, attention_k_eq_v, no PLE |
| 26B-A4B | `google/gemma-4-26B-A4B-it` | ~4B active / ~26B total | MoE (128 experts, top-8), attention_k_eq_v |

## Architecture Details

Key features handled by this implementation:

- **Variable head_dim**: Sliding attention uses head_dim=256, full attention uses head_dim=512. Unified to 512 for KV cache via zero-padding.
- **Per-Layer Embeddings (PLE)**: E2B/E4B have per-layer token embeddings and projections (hidden_size_per_layer_input=256).
- **Hybrid attention**: Alternating sliding/full attention patterns with different RoPE configurations.
  - Sliding: standard RoPE (theta=10000)
  - Full: proportional RoPE (theta=1000000, partial_rotary_factor=0.25)
- **Q-K-V normalization**: All layers apply RMSNorm to Q, K, and V (V norm has no learnable weight).
- **attention_k_eq_v**: 31B and 26B-A4B share V=K weights (no separate v_proj).
- **KV sharing**: E2B/E4B reuse KV cache from earlier layers for the last N layers (disabled in v1).
- **Double-wide MLP**: KV-shared layers use 2x intermediate_size (E2B/E4B).
- **MoE (Mixture of Experts)**: 26B-A4B has 128 experts with top-8 routing per token, plus a dense MLP in each layer.
- **Final logit softcapping**: `30 * tanh(logits / 30)`.
- **Gemma4 RMSNorm**: Direct weight multiplication (`norm(x) * weight`), not the `(1 + weight)` style.

## Validation Results

**Hardware:** trn2.3xlarge (4 Neuron cores, LNC=2)

| Variant | TP | Batch | Seq Len | Throughput | Status |
|---------|-----|-------|---------|-----------|--------|
| E2B | 2 | 1 | 512 | ~80-120 tok/s | Validated |
| E4B | 2 | 1 | 512 | ~70-80 tok/s | Validated |
| 31B | 4 | 1 | 2048 | ~13-23 tok/s | Validated |
| 26B-A4B | 4 | 1 | 2048 | ~43-60 tok/s | Validated |

All variants produce coherent, correct text output verified against reference prompts.

## Usage

### Compile

```python
import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from src.modeling_gemma4 import NeuronGemma4ForCausalLM, Gemma4InferenceConfig

model_path = "/path/to/gemma-4-2b-it"       # or any variant
compiled_path = "/path/to/compiled/output"

neuron_config = NeuronConfig(
    tp_degree=2,              # 2 for E2B/E4B, 4 for 31B/26B
    batch_size=1,
    seq_len=512,              # 512 for E2B/E4B, 2048 for 31B/26B
    max_context_length=512,
    torch_dtype=torch.bfloat16,
    save_sharded_checkpoint=True,
    attn_kernel_enabled=False,  # Required: head_dim > 128
)

config = Gemma4InferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronGemma4ForCausalLM(model_path, config)
model.compile(compiled_path)
```

### Inference

```python
from transformers import AutoTokenizer

# Load compiled model
model = NeuronGemma4ForCausalLM(model_path, config)
model.load(compiled_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("What is machine learning?", return_tensors="pt")
input_ids = inputs.input_ids

# Manual generation loop
for _ in range(100):
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids, position_ids=position_ids)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
    input_ids = torch.cat([input_ids, next_token], dim=-1)

print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
```

## Compatibility Matrix

| Instance | NxDI 2.20+ | Earlier |
|----------|-----------|---------|
| trn2.3xlarge (4 cores) | All 4 variants validated | Not tested |
| trn1.32xlarge | Not tested | Not tested |
| Inf2 | Not tested | Not tested |

## Testing

Run integration tests for a specific variant:

```bash
# Default (E2B)
python test/integration/test_model.py

# Specific variant
GEMMA4_VARIANT=26b python test/integration/test_model.py

# With pytest
GEMMA4_VARIANT=31b pytest test/integration/test_model.py --capture=tee-sys
```

Available variants: `e2b`, `e4b`, `31b`, `26b`

## Important Notes

- `attn_kernel_enabled=False` is required — the NKI attention kernel does not support head_dim > 128.
- Sliding window is disabled at the Neuron level (all layers use full context).
- KV sharing is disabled in v1 (all layers compute their own KV cache).
- The model loads weights directly from safetensors (HF `AutoModel` does not support Gemma4 in transformers <5.5).

## Example Checkpoints

- google/gemma-4-2b-it
- google/gemma-4-4b-it
- google/gemma-4-31B-it
- google/gemma-4-26B-A4B-it

## Maintainer

Annapurna Labs

**Last Updated:** 2026-04-05
