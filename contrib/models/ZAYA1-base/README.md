# Contrib Model: ZAYA1-base

NeuronX Distributed Inference implementation of ZAYA1-base (Zyphra/ZAYA1-base).

## Model Information

- **HuggingFace ID:** [`Zyphra/ZAYA1-base`](https://huggingface.co/Zyphra/ZAYA1-base)
- **Model Type:** Decoder-only Mixture-of-Experts with CCA attention
- **Parameters:** 8.84B total / 800M active per token
- **License:** Apache-2.0

## Architecture Details

ZAYA1-base is a novel MoE architecture with several unique features:

- **80 layers** alternating between attention (even) and MoE (odd)
- **CCA (Causal Cross-Attention)**: L2-normalized Q/K, learned Conv1d temporal mixing, temperature scaling, and cross-attention between current hidden states and previous layer states
- **Non-linear MLP router**: 3-layer MLP router with exponential depth averaging (EDA) and learned balancing biases
- **MoD (Mixture of Depths)**: 16 real experts + 1 skip expert for conditional computation
- **Partial RoPE**: 50% rotary position embeddings
- **Tied word embeddings**: `lm_head` shares weights with `embed_tokens`

### NxDI-specific adaptations

- **ManualConv1d**: Replaces `nn.Conv1d` with basic tensor ops to avoid Neuron compiler NCC_ITEN404 crash when Conv1d follows all-gather operations
- **SPMDRank**: Uses `SPMDRank` module for per-rank tensor extraction (required for SPMD tracing where `get_tensor_model_parallel_rank()` returns a constant)
- **CCA state caching**: Conv states and previous hidden states are persisted across TKG steps via `input_output_aliases`
- **CTE/TKG branching**: Separate code paths for context encoding and token generation, resolved at trace time
- **Static expert dispatch**: Replaces `torch.bincount` (unsupported in XLA) with mask-based dispatch

## Prerequisites

- Zyphra's custom transformers fork (provides `ZayaForCausalLM`):
  ```bash
  pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya"
  ```

- **KV cache patch** (required for batch > 1): Comment out the assertion at line 489 in `neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py`:
  ```python
  # Before (line 489):
  assert seq_ids.dim() == 1 and seq_ids.shape[0] == 1, "only supports single seq_id"
  # After:
  # Assertion removed — update_cache_const_indices handles batch>1 via torch.index_put
  ```

- **NxD spmd.py patches** (SDK 2.27 and 2.28): Two bugs in `neuronx_distributed/trace/spmd.py`:
  1. `mock_initialization()`: Replace `script_model.original_name` with `getattr(script_model, "original_name", type(script_model).__name__)`
  2. `router` shape comparison: Convert shapes with `str([[int(s) for s in tensor.shape] for tensor in inputs])`

## Validation Results

**Validated:** 2026-03-11
**Configuration:** TP=2, batch_size=4, seq_len=256, max_context_length=128

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles and loads successfully |
| Prefill (" Paris") | PASS | Correct top-1 token for "The capital of France is" |
| Batch Independence | PASS | 4 prompts generate correct, independent outputs |
| Token Matching | PASS | "Paris", "Pacific", "Ulm", "299,792,458" — all factually correct |

### Performance: Neuron vs GPU

| Metric | Neuron trn2 (TP=2) | GPU A10G | Neuron Advantage |
|--------|-------------------|---------|-----------------|
| TKG (batch=1) | **22.3 tok/s** | 6.2 tok/s | **3.6x** |
| TKG (batch=4) | **86.3 tok/s** | 23.9 tok/s | **3.6x** |
| TTFT | **75.6 ms** | 184.7 ms | **2.4x** |
| Per-Token Latency (BS=1) | **44.8 ms** | 161.6 ms | **3.6x** |

GPU benchmark: g5.2xlarge (NVIDIA A10G, 24 GB), BF16, PyTorch 2.9.1. torch.compile provides zero improvement (model is memory-bandwidth bound).

### Neuron Performance Detail (trn2.3xlarge, TP=2)

| Metric | batch=1 | batch=4 |
|--------|---------|---------|
| TKG Throughput | 22.3 tok/s | **86.3 tok/s** |
| Per-Token Latency | 44.8 ms | **11.59 ms** |
| Step Latency | 44.8 ms | 46.35 ms |
| TTFT (P50) | 75.6 ms | — |
| Compilation Time | 16.8 min | 15.4 min |
| NEFF Load Time | 33.4 s | 51.0 s |
| CPU Baseline Speedup | 3.1x | **12.0x** |

**Status:** PASS

## Usage

```python
import torch
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Monkey-patch torch.jit.script to avoid @jit_fuser crash in Zyphra's HF model
_real = torch.jit.script
torch.jit.script = lambda fn=None, *a, **kw: fn if fn is not None else (lambda f: f)

from src.modeling_zaya import NeuronZayaForCausalLM, ZayaInferenceConfig, ZayaNeuronConfig

# CRITICAL: Restore before compilation
torch.jit.script = _real

model_path = "/path/to/ZAYA1-base/"
compiled_path = "/path/to/compiled/"

# Configure
neuron_config = ZayaNeuronConfig(
    tp_degree=2,
    batch_size=4,
    max_batch_size=4,
    max_context_length=128,
    seq_len=256,
    on_device_generation=None,
    is_continuous_batching=True,
    buckets=[256],
)

config = ZayaInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronZayaForCausalLM(model_path, config=config)
model.compile(compiled_model_path=compiled_path)
model.load(compiled_path)

# Generate
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
seq_ids = torch.arange(1, dtype=torch.int32)
position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

model.reset()
with torch.no_grad():
    output = model.forward(input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids)

logits = output[0]
next_token = torch.argmax(logits[0, -1, :].float()).item()
print(tokenizer.decode([next_token]))  # " Paris"
```

## vLLM Serving

ZAYA1-base can be served via vLLM-neuron with an OpenAI-compatible API. This requires additional patches beyond the base prerequisites:

1. **NxDI `constants.py`** — Register `"zaya"` in the `MODEL_TYPES` dict with `NeuronZayaForCausalLM`
2. **vLLM-Neuron `platform.py`** — Auto-register Neuron models in vLLM's ModelRegistry
3. **HF `modeling_zaya.py`** — Wrap `jit_fuser = torch.jit.script` with try/except fallback

### vLLM Throughput (trn2.3xlarge, TP=2, max-model-len=1024)

| Concurrency | Req/s | Tok/s | Avg Latency |
|-------------|-------|-------|-------------|
| 1 | 0.70 | 21.0 | 1.43s |
| 2 | 1.33 | 39.9 | 1.50s |
| 4 | 2.41 | 72.3 | 1.66s |

Near-linear throughput scaling. ~6% overhead vs standalone at concurrency=1, ~16% at concurrency=4.

### vLLM Limitation

`max-model-len` is capped at 1024 due to Neuron compiler NCC_ITEN404 errors on CTE buckets for seq_len > 1024. The standalone model supports seq_len up to 4096.

## Compatibility Matrix

| Instance/Version | SDK 2.28 | SDK 2.27 |
|------------------|----------|----------|
| trn2.3xlarge     | Tested   | Tested   |
| trn2.48xlarge    | Expected | Expected |
| trn1             | Not tested | Not tested |
| inf2             | Not tested | Not tested |

**Note:** Requires `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` environment variable for NKI kernel compilation on trn2.

## Testing

Run integration tests:

```bash
# Set environment
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2

# Run with pytest
pytest contrib/models/ZAYA1-base/test/integration/test_model.py --capture=tee-sys

# Or run manually (with --compile to compile first)
cd contrib/models/ZAYA1-base
python3 test/integration/test_model.py --compile --tp-degree 2
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tp_degree` | 2 | Tensor parallelism degree |
| `batch_size` | 1 | Batch size (1 or 4 tested) |
| `max_context_length` | 128 | Maximum prompt length |
| `seq_len` | 256 | Maximum total sequence length |
| `is_continuous_batching` | True | Required for Neuron KV cache compatibility |
| `mlp_kernel_enabled` | False | MLP ISA kernel (experimental, -4.2% at batch=1) |

## Example Checkpoints

* [Zyphra/ZAYA1-base](https://huggingface.co/Zyphra/ZAYA1-base) (8.84B params, 4 safetensor shards)

## Known Limitations

- **KV cache patch required** for batch > 1 (see Prerequisites)
- **NxD spmd.py patches required** for SDK 2.27 and 2.28
- **Zyphra transformers fork required** — standard HuggingFace transformers does not include ZAYA model support
- **CCA attention** uses NxDI's NKI flash attention for CTE (~1% benefit) but native attention for TKG
- **MLP ISA kernel** disabled by default due to -4.2% regression at batch=1 (weight-read dominated workload)
- **max_position_embeddings = 4096** (architectural limit, tested up to 4096)

## Maintainer

Jim Burtoft
AWS

**Last Updated:** 2026-03-11
