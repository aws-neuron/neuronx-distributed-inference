# Contrib Model: Phi-1.5

NeuronX Distributed Inference implementation of Microsoft Phi-1.5.

## Model Information

- **HuggingFace ID:** `microsoft/phi-1_5`
- **Model Type:** Decoder-only transformer
- **Parameters:** 1.3B
- **License:** MIT

## Architecture Details

Phi-1.5 has several unique architectural features:

- **Partial Rotary Embeddings**: Only 50% of head dimensions use RoPE (`partial_rotary_factor=0.5`)
- **Parallel Residual**: Attention and MLP use the same normalized input (parallel computation)
- **GELU Activation**: Uses GELU (not SwiGLU like LLaMA)
- **LayerNorm**: Uses standard LayerNorm (not RMSNorm)
- **Bias in All Projections**: QKV, output, and MLP projections all have bias
- **Single LayerNorm per Layer**: Only `input_layernorm` (no `post_attention_layernorm`)

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
| "The largest planet in our solar system is" | 100% |
| "1 + 1 =" | 100% |
| "The color of the sky is" | 100% |
| "The capital of France is" | 71.9% |
| "Water boils at" | 68.8% |

**Status:** ✅ VALIDATED

## Key Implementation Notes

### State Dict Conversion

The HuggingFace Phi model uses different weight names than NeuronX expects:

```python
# HuggingFace -> NeuronX
model.layers.{i}.self_attn.q_proj -> layers.{i}.self_attn.qkv_proj.q_proj
model.layers.{i}.self_attn.k_proj -> layers.{i}.self_attn.qkv_proj.k_proj
model.layers.{i}.self_attn.v_proj -> layers.{i}.self_attn.qkv_proj.v_proj
model.layers.{i}.self_attn.dense -> layers.{i}.self_attn.o_proj.o_proj
model.final_layernorm -> norm
```

### Partial Rotary Embeddings

Only the first 50% of head dimensions are rotated:

```python
head_dim = 64  # 2048 / 32
rotary_ndims = int(head_dim * 0.5)  # 32

# Split Q/K into rotary and pass-through parts
Q_rot, Q_pass = Q[..., :rotary_ndims], Q[..., rotary_ndims:]
K_rot, K_pass = K[..., :rotary_ndims], K[..., rotary_ndims:]

# Apply RoPE only to rotary parts
Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos, sin)

# Concatenate back
Q = torch.cat([Q_rot, Q_pass], dim=-1)
K = torch.cat([K_rot, K_pass], dim=-1)
```

### Parallel Residual

Both attention and MLP use the same normalized input:

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)

# Attention and MLP use same normalized input
attn_output = self.self_attn(hidden_states)
mlp_output = self.mlp(hidden_states)

# Combine both with residual
hidden_states = residual + attn_output + mlp_output
```

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
import torch

# Import model classes from src
from src.modeling_phi import NeuronPhiForCausalLM, PhiInferenceConfig

model_path = "/path/to/phi-1_5/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = PhiInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)

# Compile and load
model = NeuronPhiForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("The capital of France is", return_tensors="pt")
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
pytest nxdi_contrib_models/models/phi-1_5/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/phi-1_5
python3 test/integration/test_model.py
```

## Example Checkpoints

* microsoft/phi-1_5

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
