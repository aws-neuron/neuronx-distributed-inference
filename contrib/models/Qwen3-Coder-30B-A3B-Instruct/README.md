# Contrib Model: Qwen3-Coder-30B-A3B-Instruct on AWS Trainium2

NxD Inference configuration for serving **Qwen3-Coder-30B-A3B-Instruct** on `trn2.48xlarge`.

## Model Information

- **HuggingFace ID:** [`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- **Model Type:** Mixture-of-Experts (MoE) decoder-only transformer
- **Total Parameters:** 30B (3B active per token)
- **License:** Check HuggingFace model card

## Architecture Details

Qwen3-Coder-30B-A3B-Instruct uses the `qwen3_moe` architecture already supported in NxDI. The contribution provides a `from_pretrained`-compatible config wrapper and validated compilation/inference scripts.

| Parameter | Qwen3-Coder-30B | Qwen3-MoE-15B-A2B |
|-----------|-----------------|---------------------|
| num_hidden_layers | **48** | 24 |
| hidden_size | **2048** | 2048 |
| num_attention_heads | **32** | 32 |
| num_key_value_heads | **4** | 4 |
| head_dim | 64 | 64 |
| num_experts | **128** | 128 |
| num_experts_per_tok | **8** | 8 |
| moe_intermediate_size | **768** | 768 |
| intermediate_size | **6144** | 6144 |
| vocab_size | 151936 | 151936 |
| max_position_embeddings | **32768** | 32768 |
| norm_topk_prob | False | False |
| n_shared_experts | 0 | 0 |

Key features:
- **GQA with QK-norm:** RMSNorm applied per-head on `head_dim` (both Q and K)
- **RoPE:** Standard rotary position embeddings (`rope_theta=10000.0`)
- **SwiGLU:** Activation in expert MLPs
- **Softmax routing:** Router dtype forced to float32 for accuracy
- **Fused QKV:** Q/K/V projections fused into single `Wqkv` weight

## Hardware Requirements

- **Instance:** `trn2.48xlarge` (64 NeuronCores, 32 Neuron devices)
- **TP Degree:** 32
- **Disk:** ~60 GB for model weights
- **Neuron SDK:** 2.22+ (neuronx-cc 2.22)
- **DLAMI:** Deep Learning AMI Neuron (Ubuntu 24.04)

## Quick Start (NxDI Direct)

### 1. Download Model

```bash
pip install huggingface_hub[cli]
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --local-dir /home/ubuntu/models/Qwen3-Coder-30B-A3B-Instruct/
```

### 2. Compile

```python
import torch
from neuronx_distributed_inference.models.config import MoENeuronConfig

from src.modeling_qwen3_coder_moe import (
    NeuronQwen3CoderMoeForCausalLM,
    Qwen3CoderMoeInferenceConfig,
)

model_path = "/home/ubuntu/models/Qwen3-Coder-30B-A3B-Instruct/"
compiled_path = "/home/ubuntu/neuron_models/Qwen3-Coder-30B-A3B-Instruct/"

neuron_config = MoENeuronConfig(
    tp_degree=32,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
    save_sharded_checkpoint=True,
)

config = Qwen3CoderMoeInferenceConfig.from_pretrained(
    model_path, neuron_config=neuron_config
)

model = NeuronQwen3CoderMoeForCausalLM(model_path, config)
model.compile(compiled_path)
```

First compilation takes ~8.6 minutes (CTE + TKG). Subsequent loads use cached NEFFs.

### 3. Load and Generate

```python
model.load(compiled_path)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

inputs = tokenizer(["fibonacci(n)"], return_tensors="pt", padding=True)
input_ids = inputs.input_ids
position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_ids, position_ids=position_ids)
    logits = outputs.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    print(tokenizer.decode(next_token))
```

## Benchmark Results

Validated on `trn2.48xlarge`, SDK 2.22 (neuronx-cc 2.22.12471), Python 3.12, PyTorch 2.9.1.

### Compilation Metrics

| Metric | Value |
|--------|-------|
| Compilation time | 8.6 minutes |
| CTE HLO generation | 54.4s |
| TKG HLO generation | 0.8s |
| TKG NEFF compilation | 118s (priority) |
| CTE NEFF compilation | 209s |
| Weight sharding (32 ranks) | 77.7s |
| Weight loading (pre-sharded) | 14.0s |
| Warmup | 1.2s |

### Inference Results

| Metric | Value |
|--------|-------|
| Prompt | `fibonacci(n)` |
| Tokens generated | 100 |
| Decode throughput | 2.0 tok/s |
| Decoding mode | Greedy (argmax) |

### Generation Quality

Prompt: `fibonacci(n)`

```
Response: = fibonacci(n-1) + fibonacci(n-2) for n > 1
fibonacci(0) = 0
fibonacci(1) = 1

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Test the function
print(fibonacci(10))  # Output: 55

# Alternative implementation using memoization for better performance
def fibonacci
```

- Correct recursive formula definition
- Correct base cases (0, 1)
- Working Python implementation
- Correct test output (`fibonacci(10) = 55`)
- Was generating memoized variant before hitting 100-token limit

## Weight Conversion Details

The framework's `convert_qwen3_moe_hf_to_neuron_state_dict` handles:

1. **QK-norm renaming:** `q_norm.weight` → `q_layernorm.weight`, `k_norm.weight` → `k_layernorm.weight`
2. **QKV fusion:** Separate `q_proj`, `k_proj`, `v_proj` → fused `Wqkv.weight`
3. **Router renaming:** `mlp.gate.weight` → `mlp.router.linear_router.weight`
4. **Expert weight fusion:** Per-expert `gate_proj`/`up_proj` → fused `gate_up_proj` tensor (shape: `[num_experts, hidden_size, 2*intermediate_size]`); per-expert `down_proj` → stacked `down_proj` tensor (shape: `[num_experts, intermediate_size, hidden_size]`)
5. **Rank utility:** Adds `rank_util.rank` tensors for TP-aware operations

Note: During sharding, unfused QKV keys (`q_proj`, `k_proj`, `v_proj`, `o_proj`) for layers that were already fused are logged as "Removing redundant keys" — this is expected behavior.

## Known Issues and Limitations

### 1. `MoENeuronConfig` Required (Not `NeuronConfig`)

The model's `get_compiler_args()` accesses `moe_ep_degree` which only exists on `MoENeuronConfig`. Using plain `NeuronConfig` causes `AttributeError`. Always use `MoENeuronConfig` when constructing the neuron config.

### 2. `output_attentions` Must Be Set

The framework's `_setup_func_config` reads `config.output_attentions` and `config.output_hidden_states`. The `Qwen3CoderMoeInferenceConfig.add_derived_config()` sets these to `False` automatically, but if constructing config manually, ensure these attributes exist.

### 3. Router Dtype

Router must use `torch.float32` for accuracy. The config class sets this automatically via `neuron_config.router_config.dtype = torch.float32` and `router_config.act_fn = "softmax"`.

## Compatibility Matrix

| Instance/SDK | 2.22+ | 2.21 and earlier |
|-------------|-------|------------------|
| trn2.48xlarge | ✅ Validated (TP=32) | Not tested |
| trn1.32xlarge | Should work (TP=32) | Not tested |
| Inf2 | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/Qwen3-Coder-30B-A3B-Instruct/test/integration/test_model.py -v --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/Qwen3-Coder-30B-A3B-Instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

- [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-11
