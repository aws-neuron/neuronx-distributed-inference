# Contrib Model: Qwen3-Coder-Next

Optimized NxD Inference implementation for Qwen3-Coder-Next, a hybrid Gated DeltaNet + GQA + Sparse MoE model with 80B total parameters and ~3B active per token, running on AWS Trainium2.

## Model Information

- **HuggingFace ID:** [`Qwen/Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- **Model Type:** Hybrid DeltaNet (linear recurrent) + GQA + Sparse MoE decoder
- **Parameters:** 80B total, ~3B active per token (BF16)
- **Architecture:** 48 layers (36 DeltaNet + 12 GQA), 512 experts top-10, head_dim=256, partial RoPE (25%)
- **License:** Apache 2.0
- **Maintainer:** Jim Burtoft

## Validation Results

**Validated:** 2026-05-29
**Instance:** trn2.48xlarge (TP=8, LNC=2)
**SDK:** Neuron SDK 2.30 (neuronx-cc 2.25.3371, neuronx-distributed-inference 0.10.17970)

### Benchmark Results

| Metric | Value |
|--------|-------|
| **Throughput** | **77 tok/s** |
| TPOT (median) | 13.0 ms |
| TPOT (p99) | 13.3 ms |
| TTFT @ 32 tokens | 245 ms |
| TTFT @ 128 tokens | 1,235 ms |
| TTFT @ 256 tokens | 1,939 ms |
| TTFT @ 512 tokens | 3,471 ms |
| TTFT @ 1024 tokens | 7,091 ms |

Configuration: batch_size=1, greedy decoding, single CTE bucket.

### Accuracy Validation

| Metric | Value |
|--------|-------|
| Top-1 token match rate | 100% (14/14 prompts) |
| Cosine similarity (logit vectors) | 0.9998 |
| Max logit difference | 0.38 |

Validated against HuggingFace BF16 CPU reference using greedy decoding with teacher forcing.

## Usage

### Prerequisites

```bash
# Activate NxDI environment on trn2.48xlarge
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Download model weights (~149 GB)
pip install huggingface_hub[cli]
huggingface-cli download Qwen/Qwen3-Coder-Next --local-dir /mnt/models/Qwen3-Coder-Next/
```

### Compile and Run

```python
import os, sys, torch
from transformers import AutoTokenizer, AutoConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig

sys.path.insert(0, '/path/to/contrib/models/Qwen3-Coder-Next/src')
os.environ['NEURON_CC_FLAGS'] = '--auto-cast matmult --auto-cast-type bf16'

from modeling_qwen35_moe import NeuronQwen35MoeForCausalLM, Qwen35MoeInferenceConfig

model_path = '/mnt/models/Qwen3-Coder-Next'

def make_load_config(model_path):
    def _load_config(config_self):
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        for key, value in hf_config.to_dict().items():
            if not key.startswith('_') and key != 'transformers_version':
                setattr(config_self, key, value)
    return _load_config

neuron_config = MoENeuronConfig(
    tp_degree=8,
    max_batch_size=1,
    max_context_length=1024,
    max_new_tokens=128,
    max_length=1152,
    torch_dtype=torch.bfloat16,
    fused_qkv=True,
    moe_tp_degree=8,
    moe_ep_degree=1,
    enable_bucketing=True,
    context_encoding_buckets=[32],
    blockwise_matmul_config={
        'block_size': 128,
        'use_shard_on_block_dynamic_while': True,
        'block_sharding_strategy': 'PING_PONG',
    },
)

inference_config = Qwen35MoeInferenceConfig(
    neuron_config=neuron_config,
    load_config=make_load_config(model_path),
)

# Compile (first time only)
model = NeuronQwen35MoeForCausalLM(model_path, inference_config)
model.compile(compiled_model_path='/mnt/compiled_qwen3')

# Load and generate
model.load('/mnt/compiled_qwen3')
model.reset()

tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer("def quicksort(arr):\n", return_tensors='pt').input_ids
n = input_ids.shape[1]

with torch.no_grad():
    out = model.forward(
        input_ids=input_ids,
        attention_mask=torch.ones(1, n, dtype=torch.int32),
        position_ids=torch.arange(n, dtype=torch.long).unsqueeze(0),
        seq_ids=torch.zeros(1, dtype=torch.long),
    )
    logits = out[0][0][-1] if out[0][0].dim() == 2 else out[0][0]
    print(tokenizer.decode(logits.argmax().item()))
```

## Expert Parallelism (EP)

This model supports Expert Parallelism for distributing the 512 experts across multiple EP ranks. EP reduces per-rank HBM usage for expert weights and enables full-chip utilization on trn2.48xlarge.

### Validated Configurations

| EP | TP | World Size | Cores | Status |
|----|----|-----------:|------:|--------|
| 1 | 8 | 8 | 8 | Baseline |
| 2 | 8 | 16 | 16 | Validated |
| 4 | 8 | 32 | 32 | Validated |
| 8 | 8 | 64 | 64 | Validated (full chip) |

### EP Configuration

```python
neuron_config = MoENeuronConfig(
    tp_degree=8,
    ep_degree=4,           # Expert Parallelism degree
    moe_ep_degree=4,       # Must match ep_degree
    moe_tp_degree=8,       # Must match tp_degree
    # ... other config
)
```

### EP Implementation Notes

- **Shared expert scaling**: With EP > 1, the framework's world_group all-reduce sums the shared expert output `ep_degree` times (since it's identical across EP ranks). The model compensates by dividing the shared expert output by `ep_degree` in the CTE path.
- **CTE dispatch**: The `ExpertMLPsV2.forward` must be patched to use `forward_blockwise` for CTE (the default `forward_selective_loading` does not support EP). A monkeypatch is provided in the test scripts.
- **ctx=32 with EP=8**: Compilation fails due to NKI DeltaNet kernel assertion. Use ctx >= 128 with EP=8.

## Compatibility Matrix

| Instance | SDK 2.30 |
|----------|----------|
| trn2.48xlarge (TP=8, EP=1-8) | VALIDATED |
| trn2.3xlarge (TP=4) | NOT SUPPORTED (HBM OOM) |

## Example Checkpoints

* [`Qwen/Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next)

## Testing Instructions

```bash
# On trn2.48xlarge with SDK 2.30 DLAMI
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd contrib/models/Qwen3-Coder-Next/

# Run integration tests
MODEL_PATH=/mnt/models/Qwen3-Coder-Next \
COMPILED_PATH=/mnt/compiled_qwen3_test/ \
pytest test/integration/test_model.py -v
```

## Architecture Details

| Parameter | Value |
|-----------|-------|
| Total parameters | 80B |
| Active parameters/token | ~3B |
| Layers | 48 (36 DeltaNet + 12 GQA) |
| Hidden size | 2048 |
| Experts per layer | 512, top-10 |
| Expert intermediate size | 512 |
| Attention heads (Q/KV) | 16/2, head_dim=256 |
| DeltaNet heads (QK/V) | 16/32, head_dim=128 |
| Partial RoPE | 25% (64 of 256 dims) |
| Max context (Neuron) | 1024 tokens |

### Key Properties

1. **Constant TPOT**: Token generation latency is O(1) regardless of context length (DeltaNet recurrent state, no growing KV cache for 36/48 layers)
2. **Linear TTFT**: Prefill scales linearly with input tokens (~6.9 ms/token)
3. **Single-bucket optimization**: Use one CTE bucket per deployment for best prefill latency

### Custom NKI Kernels

| Kernel | Purpose | File |
|--------|---------|------|
| DeltaNet Recurrent | Token generation for linear attention layers | `nki_deltanet.py` |
| Flash Attention d=256 | Context encoding for GQA layers (seq >= 512) | `nki_flash_attn_d256_pipe.py` |

## vLLM Integration

This model supports vLLM serving via the `vllm/` directory. See `vllm/start_vllm_server.sh` for usage.

## Known Issues

1. **Max context: 1024 tokens** — Model weights consume 20.8 GB per NeuronCore pair (LNC=2), leaving ~2.6 GB for scratchpad. Context lengths > 1024 exceed available HBM. INT8 quantization would unlock longer contexts.
2. **TP=4 not supported** — Per-rank expert weights (~37 GB) exceed 24 GB HBM per core at TP=4.
3. **TP=16 not supported** — NKI DeltaNet kernel requires `linear_value_head_dim >= 16` per rank (128/16=8 is too small).
4. **DeltaNet state reset** — Must call `model.reset()` between independent prompts to clear recurrent state.
5. **NKI deprecation warnings** on import (cosmetic, from blockwise_mm internals in neuronx-distributed).
6. **EP=8 requires ctx >= 128** — NKI DeltaNet kernel fails at ctx=32 with EP=8 ("Out-of-bound access... index range [0, 127] exceed dimension size of 33"). Use context_encoding_buckets with minimum size 128 for EP=8.
7. **position_ids for padded CTE inputs** — When manually padding inputs for CTE, padding positions must have `position_id = 0` (not incrementing values). The framework uses `torch.max(position_ids)` to find the last real token. Incorrect position_ids will cause the model to output `<|endoftext|>`.
