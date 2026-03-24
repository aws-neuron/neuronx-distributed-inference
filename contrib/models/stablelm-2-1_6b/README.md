# Contrib Model: StableLM 2 1.6B

NeuronX Distributed Inference implementation of StableLM 2 1.6B.

## Model Information

- **HuggingFace ID:** `stabilityai/stablelm-2-1_6b`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Layers:** 24
- **Hidden Size:** 2048
- **Attention Heads:** 32
- **Key-Value Heads:** 32 (MHA)
- **Vocabulary:** 100352
- **Max Position Embeddings:** 4096

### StableLM-Specific Features

| Feature | Value | Description |
|---------|-------|-------------|
| `partial_rotary_factor` | 0.25 | Only 25% of head_dim (16 out of 64) uses RoPE |
| `use_qkv_bias` | True | QKV projections have bias |
| `qk_layernorm` | False | No Q-K layer normalization |
| `use_parallel_residual` | False | Standard residual connections |
| `layer_norm_eps` | 1e-5 | Uses standard LayerNorm (not RMSNorm) |

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
| "Water boils at" | 100% |
| "The capital of France is" | 0% (different but correct output) |

**Status:** ✅ PASS

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.15 | 0.00 |
| MBU (%) | 0.34 | 0.55 |
| HFU (%) | 0.16 | 0.00 |
| Execution Time (us) | 0.01 | 0.01 |
| HBM Read | 1.50 GB | 1.45 GB |
| HBM Write | 54.79 MB | 999.5 KB |

**Throughput:** 79.95 tok/s | **Compile Time:** 173.75s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Implementation Notes

### Partial Rotary Embedding

StableLM uses `partial_rotary_factor=0.25`, meaning only 16 out of 64 head dimensions get RoPE:

```python
rotary_ndims = int(head_dim * 0.25)  # 16
Q_rot, Q_pass = Q[..., :rotary_ndims], Q[..., rotary_ndims:]
K_rot, K_pass = K[..., :rotary_ndims], K[..., rotary_ndims:]
# Apply RoPE only to Q_rot, K_rot
# Concatenate: [rotated_part, pass_through_part]
```

### LayerNorm (not RMSNorm)

StableLM uses standard `nn.LayerNorm` with bias, unlike most modern LLMs that use RMSNorm:

```python
self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
```

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from src.modeling_stablelm import NeuronStableLmForCausalLM, StableLmInferenceConfig

model_path = "/path/to/stablelm-2-1_6b/"
compiled_model_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = StableLmInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronStableLmForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=64)
print(tokenizer.decode(outputs[0]))
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Functional | Not tested |
| Inf2             | Not tested | Not tested |

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
