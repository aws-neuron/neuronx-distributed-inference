# Contrib Model: Gemma 3 1B IT

NeuronX Distributed Inference support for **google/gemma-3-1b-it** (1B parameter variant).

This contrib subclasses the official `models/gemma3/` implementation and adds
the minimal overrides needed for the 1B variant's unusual architecture.

## Model Information

- **HuggingFace ID:** `google/gemma-3-1b-it`
- **Model Type:** Decoder-only transformer (causal LM)
- **Parameters:** 1B
- **License:** Gemma license (see HuggingFace model card)

## Why a Separate Contrib?

The official `models/gemma3/` targets the 4B/12B/27B variants (head_dim=128).
The 1B variant has several unusual architecture parameters that require
additional handling:

| Parameter | 1B | 4B/12B/27B |
|-----------|-----|-----------|
| head_dim | **256** | 128 |
| vocab_size | **262144** | 262208 |
| num_kv_heads | **1** | 4-16 |
| num_attention_heads | **4** | 8-32 |

### Issues Addressed

1. **Chunked attention for head_dim=256** -- The Neuron compiler generates DGE
   scatter/gather instructions that produce out-of-bounds memory accesses when
   head_dim exceeds 128. All Q@K^T and scores@V matmuls are split into
   128-wide chunks along head_dim. Mathematically identical, avoids hardware
   addressing limits.

2. **vocab_size from HF config** -- The upstream `Gemma3InferenceConfig`
   hardcodes `vocab_size=262208`. This contrib reads the actual value from
   the HuggingFace config (262144 for 1B).

3. **Auto-disable NKI attention kernel** -- The NKI flash attention kernel
   asserts `head_dim <= 128`. This contrib auto-disables it when head_dim
   exceeds that limit.

4. **k_cache_transposed + SWA + GQA fix** -- The base class forces
   `k_cache_transposed=False` for sliding window layers, but the KV cache
   manager stores K in BHDS layout for ALL layers when `k_cache_transposed=True`
   in the config. This creates a layout mismatch: `repeat_kv` assumes BHSD but
   receives BHDS, producing incorrect GQA expansion. The fix restores the
   config value and transposes K around `repeat_kv`.

5. **query_pre_attn_scalar weight fusion** -- NxDI uses `QK^T / sqrt(head_dim)`
   for attention scaling, but Gemma 3 specifies `QK^T / sqrt(query_pre_attn_scalar)`.
   Rather than modifying the attention kernel (which risks breaking optimizations),
   we fuse the correction factor into Q/K weight matrices at load time. Zero
   runtime overhead. Pattern from Pierre Lienhart's gemma3-vision contrib.

### Known Compiler Issue

**CTE buckets < 512 crash at runtime** with head_dim=256 + `input_output_aliases`.
This is a Neuron compiler issue (DGE OOB), not a code issue. Workaround:
always use `context_encoding_buckets: [512]` or larger.

| CTE Bucket | Result |
|-----------|--------|
| 128 | OOB crash |
| 256 | OOB crash |
| 384 | OOB crash |
| 512 | **PASS** |

## Usage

### Standalone (NxDI API)

```python
import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

import sys
sys.path.insert(0, "contrib/models/gemma-3-1b-it/src")
from modeling_gemma3 import NeuronGemma3_1B_ForCausalLM, Gemma3_1B_InferenceConfig

model_path = "google/gemma-3-1b-it"

neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=4,
    seq_len=512,
    torch_dtype=torch.bfloat16,
    attn_kernel_enabled=False,
    k_cache_transposed=True,
)

config = Gemma3_1B_InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

model = NeuronGemma3_1B_ForCausalLM(model_path, config)
model.compile("/tmp/gemma3-1b-compiled")
model.load("/tmp/gemma3-1b-compiled")
```

### vLLM Serving

Requires installing the NxDI fork with the `gemma3` model type registered in
`constants.py` (or using the fork's `fix/gemma3-1b-oob` branch).

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-1b-it \
  --tensor-parallel-size 1 \
  --max-model-len 512 \
  --max-num-seqs 4 \
  --dtype bfloat16 \
  --no-enable-prefix-caching \
  --block-size 128 \
  --additional-config '{"override_neuron_config": {
    "tp_degree": 1,
    "batch_size": 4,
    "seq_len": 512,
    "n_active_tokens": 4,
    "context_encoding_buckets": [512],
    "token_generation_buckets": [512],
    "on_device_sampling_config": null,
    "attn_kernel_enabled": false,
    "k_cache_transposed": true
  }}'
```

## Required Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| `attn_kernel_enabled` | `false` | NKI kernel asserts head_dim <= 128 |
| `k_cache_transposed` | `true` | Required for the SWA+GQA fix |
| `context_encoding_buckets` | `[512]` or larger | Compiler OOB for buckets < 512 |
| `on_device_sampling_config` | `null` | Required (not `false`) |

## Compatibility

| Instance | Status | Notes |
|----------|--------|-------|
| trn2.3xlarge | Tested | TP=1, batch_size=4/16, CTE bucket 512 |
| inf2.8xlarge | Not tested with this contrib | OOB confirmed on raw official code |
| trn1.* | Not tested | Should work with same config |

## Architecture

This contrib is structured as thin subclasses of the official implementation:

```
models/gemma3/modeling_gemma3.py  (upstream, unchanged)
  |
  +-- contrib/gemma-3-1b-it/src/modeling_gemma3.py  (this file)
        |-- Gemma3_1B_InferenceConfig    <-- fixes vocab_size, auto-disables NKI
        |-- NeuronGemma3_1B_Attention    <-- chunked attn + k_cache_transposed fix
        +-- NeuronGemma3_1B_ForCausalLM  <-- query_pre_attn_scalar weight fusion
```

No upstream files are modified. The contrib imports from the official
`models/gemma3/` package and overrides only what is necessary.

## Testing

```bash
# On a Neuron instance (trn2 or inf2):
cd neuronx-distributed-inference
PYTHONPATH="contrib/models/gemma-3-1b-it/src:src:$PYTHONPATH" \
  pytest contrib/models/gemma-3-1b-it/test/integration/test_model.py -v --capture=tee-sys
```

## Maintainer

Jim Burtoft (jimburtoft)

**Last Updated:** 2026-03-27
