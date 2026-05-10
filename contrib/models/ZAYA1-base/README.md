# Contrib Model: ZAYA1

NeuronX Distributed Inference implementation of the ZAYA1 model family (Zyphra).

## Supported Variants

| Variant | HuggingFace ID | Parameters | Use Case |
|---------|---------------|------------|----------|
| **ZAYA1-base** | [`Zyphra/ZAYA1-base`](https://huggingface.co/Zyphra/ZAYA1-base) | 8.84B total / 800M active | General-purpose generation |
| **ZAYA1-8B** | [`Zyphra/ZAYA1-8B`](https://huggingface.co/Zyphra/ZAYA1-8B) | 8.4B total / 760M active | Reasoning with chain-of-thought |

Both variants share identical architecture and use the same NxDI contrib code.
ZAYA1-8B is a post-trained reasoning variant that generates `<think>...</think>` traces.

## Model Information

- **Model Type:** Decoder-only Mixture-of-Experts with CCA attention
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

- **NxD spmd.py patches** (SDK 2.27 and 2.28 only; fixed in SDK 2.29): Two bugs in `neuronx_distributed/trace/spmd.py`:
  1. `mock_initialization()`: Replace `script_model.original_name` with `getattr(script_model, "original_name", type(script_model).__name__)`
  2. `router` shape comparison: Convert shapes with `str([[int(s) for s in tensor.shape] for tensor in inputs])`

## Validation Results

**Validated:** 2026-03-10
**Configuration:** TP=2, batch_size=4, seq_len=256, max_context_length=128

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles and loads successfully |
| Prefill (" Paris") | PASS | Correct top-1 token for "The capital of France is" |
| Batch Independence | PASS | 4 prompts generate correct, independent outputs |
| Token Matching | PASS | "Paris", "Pacific", "Ulm", "299,792,458" — all factually correct |

### Performance Metrics (trn2.3xlarge, TP=2)

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

### ZAYA1-8B Validation Results

**Validated:** 2026-05-09
**Configuration:** TP=2, batch_size=1, seq_len=2048, max_context_length=512

#### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Compiles and loads with scalar-to-list config normalization |
| Reasoning Trace | PASS | 5/5 prompts produce coherent `<think>...</think>` traces |
| Extended Generation | PASS | 1500 tokens, no KV corruption, correct math solutions |
| vLLM Serving | PASS | OpenAI-compatible API with `deepseek_r1` reasoning parser |

#### Performance Metrics (trn2.3xlarge, TP=2)

| Metric | BS=1 | BS=4 Concurrent |
|--------|------|-----------------|
| TKG Throughput | 31.7 tok/s | **112.6 tok/s** |
| Per-Token Latency | 31.6 ms | 35.5 ms |
| TTFT (CTE bucket=128) | 58.8 ms | 58.8 ms |
| TTFT (CTE bucket=512) | 129.3 ms | 129.3 ms |
| Compilation Time | ~16 min | ~14 min |

#### GPU Comparison (NVIDIA L40S, bf16)

| Metric | GPU (L40S) | Neuron (TP=2) | Neuron Speedup |
|--------|-----------|---------------|----------------|
| TTFT | 193.7 ms | 58.8 ms | **3.3x** |
| TKG BS=1 | 6.7 tok/s | 31.7 tok/s | **4.7x** |
| TKG BS=4 | N/A | 112.6 tok/s | **16.8x** |

> **Note:** GPU uses eager attention (CCA is not compatible with Flash Attention/SDPA).

#### Key Config Differences from ZAYA1-base

| Parameter | ZAYA1-base | ZAYA1-8B |
|-----------|-----------|----------|
| `eos_token_id` | 1 (`<eos>`) | 106 (`<|im_end|>`) |
| `rope_theta` | 1,000,000 | 5,000,000 |
| `max_position_embeddings` | 32,768 | 131,072 |
| `residual_in_fp32` | false | true |
| Config format | Per-layer lists | Scalar values (auto-normalized) |

All differences are handled automatically via `load_pretrained_config()`.

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

### ZAYA1-8B Usage (Reasoning Model)

```python
import torch
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Permanently disable torch.jit.script (Zyphra's HF code needs this throughout)
torch.jit.script = lambda fn=None, *a, **kw: fn if fn is not None else (lambda f: f)

from src.modeling_zaya import NeuronZayaForCausalLM, ZayaInferenceConfig, ZayaNeuronConfig
from transformers import AutoTokenizer

model_path = "/path/to/ZAYA1-8B/"
compiled_path = "/path/to/compiled/"

# Configure for reasoning (longer sequences, BS=1 only)
neuron_config = ZayaNeuronConfig(
    tp_degree=2,
    batch_size=1,
    max_batch_size=1,
    max_context_length=512,      # Max prefill length
    seq_len=2048,                # Max total sequence (reasoning can be long)
    on_device_generation=None,
    is_continuous_batching=True,  # Required
    buckets=[2048],
    context_encoding_buckets=[128, 256, 512],  # Capped at 512 (NCC_ITEN404)
)

config = ZayaInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

model = NeuronZayaForCausalLM(model_path, config=config)
model.compile(compiled_model_path=compiled_path)
model.load(compiled_path)

# Generate with chat template (injects <think>\n for reasoning)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
messages = [{"role": "user", "content": "What is the sum of all primes less than 20?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
seq_ids = torch.arange(1, dtype=torch.int32)
position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

model.reset()

# Prefill
with torch.no_grad():
    output = model.forward(input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids)

# Autoregressive generation
generated_ids = []
next_token_id = torch.argmax(output[0][0, -1, :].float()).item()
generated_ids.append(next_token_id)

for step in range(500):
    token_input = torch.tensor([[next_token_id]], dtype=torch.long)
    pos = torch.tensor([[input_ids.shape[1] + step]], dtype=torch.long)
    with torch.no_grad():
        output = model.forward(input_ids=token_input, seq_ids=seq_ids, position_ids=pos)
    next_token_id = torch.argmax(output[0][0, -1, :].float()).item()
    generated_ids.append(next_token_id)
    if next_token_id == 106:  # <|im_end|>
        break

response = tokenizer.decode(generated_ids, skip_special_tokens=False)
# response contains: "<thinking content></think>\n\n<final answer>"
if "</think>" in response:
    thinking, answer = response.split("</think>", 1)
    print(f"Thinking: {thinking}")
    print(f"Answer: {answer}")
```

## vLLM Serving

Both ZAYA1-base and ZAYA1-8B can be served via vLLM-neuron (0.16.0 + vllm-neuron 0.5.0)
with an OpenAI-compatible API.

### Additional Prerequisites for vLLM

Beyond the base prerequisites above, vLLM integration requires:

1. **Zyphra transformers fork** installed in the vLLM venv:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
   pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya"
   ```

2. **NxDI `constants.py`** — Register ZAYA in the `MODEL_TYPES` dict (both NxDI and vLLM venvs).
   Add before `END_TO_END_MODEL`:
   ```python
   # Keep torch.jit.script patched permanently
   import sys as _sys, torch as _torch
   _zaya_src = "/home/ubuntu/zaya1_8b/src"
   if _zaya_src not in _sys.path:
       _sys.path.insert(0, _zaya_src)
   _torch.jit.script = lambda fn=None, *a, **kw: fn if fn is not None else (lambda f: f)
   from modeling_zaya import NeuronZayaForCausalLM
   ```
   And in `MODEL_TYPES`:
   ```python
   "zaya": {"causal-lm": NeuronZayaForCausalLM},
   ```

3. **vLLM `registry.py`** — Add `ZayaForCausalLM` to the model list:
   ```python
   "ZayaForCausalLM": ("dbrx", "DbrxForCausalLM"),  # placeholder for Neuron loader
   ```

### Launch Server (ZAYA1-8B with reasoning)

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2

python scripts/serve_vllm.py \
    --model /path/to/ZAYA1-8B \
    --tp 2 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --compiled-model-path /path/to/compiled \
    --context-buckets '128,256,512' \
    --reasoning-parser deepseek_r1 \
    --port 8000
```

### Launch Server (ZAYA1-base without reasoning)

```bash
python scripts/serve_vllm.py \
    --model /path/to/ZAYA1-base \
    --tp 2 \
    --max-model-len 1024 \
    --max-num-seqs 4 \
    --reasoning-parser none \
    --port 8000
```

Compilation takes ~33 minutes on first startup (6 CTE buckets + TKG).

### Test

```bash
# Health check
curl http://localhost:8000/health

# Chat completion (ZAYA1-8B with reasoning)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/ZAYA1-8B",
    "messages": [{"role": "user", "content": "What is 15 * 7?"}],
    "max_tokens": 200,
    "temperature": 0
  }'
# Response includes:
#   "reasoning": "<thinking content>",
#   "content": "\n\n15 * 7 = 105."

# Completion (ZAYA1-base)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/ZAYA1-base", "prompt": "The capital of France is", "max_tokens": 20, "temperature": 0}'
```

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

| Instance/Version | SDK 2.29 | SDK 2.28 | SDK 2.27 |
|------------------|----------|----------|----------|
| trn2.3xlarge     | Validated (21.1 tok/s) | Tested   | Tested   |
| trn2.48xlarge    | Expected | Expected | Expected |
| trn1             | Not supported | Not tested | Not tested |
| inf2             | Not supported | Not tested | Not tested |

> **NxDI 2.29+ requires Trn2 or newer hardware.** For Trn1/Inf2 support, pin to SDK 2.28.

**Note:** Requires `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` environment variable for NKI kernel compilation on trn2. SDK 2.29 requires the Zyphra custom transformers fork (4.57.1) with `trust_remote_code=True`.

## Testing

Run integration tests:

```bash
# Set environment
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2

# Run with pytest
pytest contrib/models/ZAYA1-base/test/integration/test_model.py --capture=tee-sys

# Or run manually
cd contrib/models/ZAYA1-base
python3 test/integration/test_model.py
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

* [Zyphra/ZAYA1-base](https://huggingface.co/Zyphra/ZAYA1-base) (8.84B params, 4 safetensor shards) — general-purpose
* [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B) (8.4B params, 4 safetensor shards) — reasoning variant

## Known Limitations

- **ZAYA1-8B: BS=4 concurrent serving** — CTE (context encoding) is limited to BS=1 (one prefill at a time), but TKG (token generation) supports BS>1 in continuous batching mode. Compile with `batch_size=4, ctx_batch_size=1, tkg_batch_size=4` for 112.6 tok/s aggregate throughput (3.78x speedup over sequential). ZAYA1-base supports full BS>1 for both CTE and TKG with the KV cache patch.
- **KV cache patch required** for ZAYA1-base batch > 1 (see Prerequisites)
- **NxD spmd.py patches required** for SDK 2.27 and 2.28 (fixed in SDK 2.29)
- **Zyphra transformers fork required** — standard HuggingFace transformers does not include ZAYA model support
- **CCA attention** uses NxDI's NKI flash attention for CTE (~1% benefit) but native attention for TKG
- **MLP ISA kernel** disabled by default due to -4.2% regression at batch=1 (weight-read dominated workload)
- **CTE buckets capped at 512** — NCC_ITEN404 compiler crash at seq_len >= 1024 for CTE. TKG (seq_len=1) is unaffected.
- **TP=2 only** — ZAYA has 2 KV heads, not divisible by 4. All 16 experts replicated per rank.
- **max_position_embeddings** = 32,768 (ZAYA1-base) / 131,072 (ZAYA1-8B, tested up to 2,048)

## Maintainer

Jim Burtoft
AWS

**Last Updated:** 2026-05-10
