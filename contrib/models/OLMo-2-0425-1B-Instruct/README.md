# Contrib Model: OLMo 2 0425 1B Instruct

NeuronX Distributed Inference implementation of OLMo 2 0425 1B Instruct.

## Model Information

- **HuggingFace ID:** `allenai/OLMo-2-0425-1B-Instruct`
- **Model Type:** Decoder-only transformer (OLMo2 architecture)
- **Parameters:** ~1.2B
- **License:** Apache-2.0

## Architecture Details

- **Layers:** 16 decoder layers
- **Hidden Size:** 2048
- **Attention Heads:** 16
- **Key-Value Heads:** 16 (MHA)
- **Vocabulary:** 100,352
- **Max Position Embeddings:** 4096

### OLMo2-Specific Features

| Feature | Value | Description |
|---------|-------|-------------|
| Post-layer normalization | Yes | RMSNorm AFTER attention and MLP (not before) |
| Q-K normalization | Yes | RMSNorm on Q/K projections before RoPE |
| `attention_bias` | False | No bias in attention projections |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `rope_theta` | 500000.0 | RoPE base frequency |

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
| "The capital of France is" | 100% |
| "The largest planet in our solar system is" | 100% |
| "The speed of light is approximately" | 100% |
| "1 + 1 =" | 100% |
| "The color of the sky is" | 100% |
| "Hello, how are you" | 100% |
| "Water boils at" | 12.5% |

**Status:** ✅ PASS

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.21 | 0.00 |
| MBU (%) | 0.45 | 0.58 |
| HFU (%) | 0.22 | 0.00 |
| Execution Time (us) | 0.01 | 0.01 |
| HBM Read | 1.31 GB | 1.29 GB |
| HBM Write | 28.64 MB | 700.7 KB |

**Throughput:** 123.11 tok/s | **Compile Time:** 96.43s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Implementation Notes

### Post-Layer Normalization

OLMo2 uses post-layer normalization (different from LLaMA's pre-norm):

```python
# OLMo2 POST-norm architecture
residual = hidden_states
hidden_states = self_attn(hidden_states)  # No pre-norm!
hidden_states = post_attention_layernorm(hidden_states)  # Norm AFTER
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = mlp(hidden_states)  # No pre-norm!
hidden_states = post_feedforward_layernorm(hidden_states)  # Norm AFTER
hidden_states = residual + hidden_states
```

### Q-K Normalization with Tensor Parallelism

OLMo2 applies RMSNorm to Q and K projections BEFORE reshaping to heads. With TP > 1, this requires special handling:

```python
# The variance must be computed over the FULL dimension, not sharded
# Use ShardedRMSNorm which does all-reduce for correct variance
class ShardedRMSNorm:
    def forward(self, x):
        local_sum_sq = x.pow(2).sum(-1, keepdim=True)
        global_sum_sq = reduce_from_tensor_model_parallel_region(local_sum_sq)
        variance = global_sum_sq / self.full_hidden_size  # Use FULL size!
        return self.weight * x * torch.rsqrt(variance + self.eps)
```

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from src.modeling_olmo import NeuronOlmo2ForCausalLM, Olmo2InferenceConfig

model_path = "/path/to/OLMo-2-0425-1B-Instruct/"
compiled_model_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = Olmo2InferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronOlmo2ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("The capital of France is", return_tensors="pt")
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
