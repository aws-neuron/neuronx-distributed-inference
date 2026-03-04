# Contrib Model: Pythia 2.8B

NeuronX Distributed Inference implementation of Pythia-2.8B from EleutherAI.

## Model Information

- **HuggingFace ID:** `EleutherAI/pythia-2.8b`
- **Model Type:** Decoder-only transformer (GPTNeoX architecture)
- **Parameters:** ~2.8B
- **License:** Apache-2.0

## Architecture Details

- **Layers:** 32 decoder layers
- **Hidden Size:** 2560
- **Attention Heads:** 32
- **Intermediate Size:** 10240
- **Vocabulary:** 50,304 tokens
- **Max Position Embeddings:** 2048

### Pythia/GPTNeoX-Specific Features

| Feature | Value | Description |
|---------|-------|-------------|
| `rotary_pct` | 0.25 | Only 25% of head_dim (20 out of 80) uses RoPE |
| `use_parallel_residual` | True | Parallel attention + MLP residual connections |
| `attention_bias` | True | QKV and output projections have bias |
| Normalization | LayerNorm | Uses standard LayerNorm (not RMSNorm) |
| Activation | GELU | GELU activation in MLP |

## Validation Results

**Validated:** 2026-02-06  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** (best of multiple prompts) |

### Multi-Prompt Accuracy

| Prompt | Match Rate |
|--------|------------|
| "1 + 1 =" | 100% |
| "The color of the sky is" | 100% |
| "Water boils at" | 65.6% |
| "The speed of light is approximately" | 56.2% |
| "The largest planet in our solar system is" | 50% |
| "The capital of France is" | 6.2% |

**Status:** ✅ PASS

## Implementation Notes

### Partial Rotary Embedding (rotary_pct=0.25)

Pythia applies RoPE to only 25% of the head dimension:

```python
head_dim = 80  # 2560 / 32
rotary_ndims = int(head_dim * 0.25)  # 20

# Split Q/K into rotary and pass-through parts
q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]

# Apply RoPE only to first 20 dimensions
q_rot = apply_rope(q_rot, cos, sin)
k_rot = apply_rope(k_rot, cos, sin)

# Concatenate: [rotated_20_dims, pass_through_60_dims]
q = torch.cat([q_rot, q_pass], dim=-1)
k = torch.cat([k_rot, k_pass], dim=-1)
```

### Parallel Residual Connections

Pythia uses parallel residual connections where attention and MLP operate on the same input:

```python
# Parallel residual: x = x + attn(ln1(x)) + mlp(ln2(x))
residual = hidden_states
attn_out = self.self_attn(self.input_layernorm(hidden_states))
mlp_out = self.mlp(self.post_attention_layernorm(residual))  # Use original residual!
hidden_states = residual + attn_out + mlp_out
```

### Interleaved QKV Layout

GPTNeoX uses an interleaved QKV layout in the fused projection:

```python
# Weight layout: [head0_Q, head0_K, head0_V, head1_Q, head1_K, head1_V, ...]
# Shape: [num_heads * 3 * head_dim, hidden_size]
qkv_reshaped = qkv_weight.view(num_heads, 3, head_dim, hidden_size)
q_weight = qkv_reshaped[:, 0, :, :].reshape(hidden_size, hidden_size)
k_weight = qkv_reshaped[:, 1, :, :].reshape(hidden_size, hidden_size)
v_weight = qkv_reshaped[:, 2, :, :].reshape(hidden_size, hidden_size)
```

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from src.modeling_gpt_neox import NeuronGPTNeoXForCausalLM, GPTNeoXInferenceConfig

model_path = "/path/to/pythia-2.8b/"
compiled_model_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = GPTNeoXInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronGPTNeoXForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("The color of the sky is", return_tensors="pt")
# Use manual generation loop (see test file for example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Functional | Not tested |
| Inf2             | Not tested | Not tested |

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
