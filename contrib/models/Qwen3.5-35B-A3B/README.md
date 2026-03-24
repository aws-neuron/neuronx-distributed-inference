# Contrib Model: Qwen3.5-35B-A3B

NeuronX Distributed Inference implementation of Qwen3.5-35B-A3B, a hybrid DeltaNet + Standard Attention + Mixture-of-Experts architecture.

## Model Information

- **HuggingFace ID:** [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- **Model Type:** Native multimodal VLM (Vision + DeltaNet + GQA + MoE)
- **Parameters:** 35B total, 3B active per token (sparse MoE)
- **License:** Check HuggingFace model card

## Architecture Details

Qwen3.5-35B-A3B is a novel hybrid architecture combining three key components:

- **30 DeltaNet layers** (linear recurrent attention): Gated delta rule recurrence with causal conv1d, using PyTorch `chunk_forward(64)` for context encoding and a custom NKI v2 kernel for token generation. State is carried between context encoding and token generation via `input_output_aliases` on `nn.Parameter` buffers.
- **10 standard GQA attention layers**: With output gate (sigmoid-gated, head-interleaved in q_proj), partial RoPE (25% of head_dim = 64 of 256), and per-head QK norm.
- **All 40 layers use sparse MoE**: 256 routed experts (top-8) + 1 sigmoid-gated shared expert. Expert weights are pre-fused (gate_up_proj, down_proj).

Key implementation details:

- **DeltaNet CTE (context encoding)**: Uses PyTorch `chunk_forward(chunk_size=64)` for prefill, which splits the sequence into 64-token chunks and processes each with batched matrix operations. This is 6.4x faster than the token-sequential NKI kernel for context encoding. Combined with multi-bucket compilation (`context_encoding_buckets=[512, 1024]`), TTFT is reduced from 10,088ms to ~1,436ms.
- **DeltaNet TKG (token generation)**: Uses the NKI v2 kernel (`nki_deltanet.py`) for per-token recurrence. Processes one (batch, head) pair per kernel call. Returns both output and final recurrent state for CTE-to-TKG state carry-over.
- **Padding-aware recurrence**: NxDI right-pads inputs to bucket size. Padding tokens' decay factor `g` is zeroed so `exp(0) = 1`, preserving recurrent state. Conv state is saved from the last 3 *valid* positions using `torch.gather`.
- **Sigmoid-gated shared expert**: Wraps NxDI's `SharedExperts` with a sigmoid gate (`SigmoidGatedSharedExperts`), since Qwen3.5's shared expert gating differs from NxDI's default additive behavior.
- **RMSNorm weight conversion**: Qwen3.5 uses `(1 + weight)` RMSNorm (weights initialized to 0). Converted to standard `weight` RMSNorm (weights initialized to 1) by adding 1.0 during state dict conversion.

## Validation Results

**Validated:** 2026-03-04
**Configuration:** TP=4, batch_size=1, max_context_length=128, max_new_tokens=32, BF16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Generation ("Paris") | PASS | CTE top-1 = "Paris" (score 17.88) |
| Coherence | PASS | Multi-prompt coherent generation |
| Token Match vs CPU | PASS | 3/3 = 100% token match against CPU reference |

### Generation Examples

| Prompt | CTE Top-1 Token | TKG Output (10 tokens) |
|--------|-----------------|------------------------|
| "The capital of France is" | " Paris" (17.88) | "a city called Paris.\nA. correct" |
| "1 + 1 =" | " " (18.13) | " 2\n1 + 1 = 2" |
| "The color of the sky is" | " blue" (18.75) | "a result of the scattering of light by the atmosphere" |

## Benchmarks

**Date:** 2026-03-24
**Neuron:** trn2.3xlarge (TP=4, BF16, SDK 2.28)
**GPU baseline:** g6.12xlarge (4x NVIDIA L4, BF16, PyTorch 2.9)

### Neuron Performance (BS=1, Recommended Config)

Using multi-bucket CTE with `context_encoding_buckets=[512, 1024]` and `block_size=256`:

| Context Length | TTFT | TKG Latency | Throughput |
|----------------|------|-------------|------------|
| 128 tokens | 1,469 ms | 18.4 ms/tok | 54.3 tok/s |
| 512 tokens | 1,436 ms | 18.4 ms/tok | 54.3 tok/s |
| 1,024 tokens | 1,434 ms | 18.4 ms/tok | 54.3 tok/s |

### TTFT Optimization History

| Configuration | TTFT (1024 ctx) | Speedup | Notes |
|---------------|-----------------|---------|-------|
| NKI recurrent (original) | 14,437 ms | 1.0x | Token-sequential NKI kernel |
| chunk_forward(64), single bucket 2048 | 2,248 ms | 6.4x | PyTorch chunked CTE |
| chunk_forward(64), multi-bucket [512,1024], block_size=256 | **1,434 ms** | **10.1x** | **Recommended** |

### TTFT Component Breakdown

Profiling with 4/8/16/40 decoder layers shows TTFT is dominated by MoE computation:

| Component | Per-Layer Cost | % of Total | 40-Layer Total |
|-----------|---------------|------------|----------------|
| MoE (256 experts, top-8) | ~30 ms | ~85% | ~1,200 ms |
| DeltaNet attention | ~4 ms | ~10% | ~120 ms |
| Norms + residuals | ~2 ms | ~5% | ~80 ms |
| Fixed overhead (embedding + lm_head) | -- | -- | ~7 ms |

**Linear model**: `TTFT = 7.4 ms + 35.7 ms x num_layers` (R^2 > 0.999)

Notes:
- Multi-bucket compilation avoids padding short inputs to 2048 tokens, reducing wasted MoE computation.
- `block_size=256` enables the NxDI blockwise MoE NKI kernel, which is faster than the `forward_all_experts` fallback path used when block_size > seq_len.
- TKG throughput is consistent across sequence lengths (~18 ms/tok).
- TTFT is nearly constant across context lengths within the same bucket because NxDI pads inputs to the bucket boundary.

### GPU Baseline (g6.12xlarge, 4x L4, 128 new tokens)

| Batch Size | TTFT | TKG Latency | Per-Sample Throughput | Batch Throughput |
|------------|------|-------------|----------------------|-----------------|
| 1 | 498 ms | 100.2 ms/tok | 10.0 tok/s | 10.0 tok/s |
| 2 | 386 ms | 103.1 ms/tok | 9.7 tok/s | 19.4 tok/s |
| 4 | 459 ms | 129.1 ms/tok | 7.8 tok/s | 31.0 tok/s |

### Neuron vs GPU Summary (BS=1, 1024-token context, 128 new tokens)

| Metric | Neuron (trn2.3xlarge) | GPU (g6.12xlarge) | Neuron Advantage |
|--------|----------------------|-------------------|------------------|
| TTFT | 1,434 ms | 498 ms | 0.35x (GPU faster) |
| TKG latency | 18.2 ms/tok | 100.2 ms/tok | **5.5x faster** |
| Throughput | **54.9 tok/s** | 10.0 tok/s | **5.5x higher** |

**Key takeaway:** Neuron delivers 5.5x higher token generation throughput than a 4x L4 GPU setup. TTFT gap has been reduced from 20x to 2.9x through multi-bucket CTE optimization. Remaining TTFT is dominated by MoE computation (85% of per-layer cost).

## Usage

### Recommended configuration (multi-bucket CTE)

```python
import json
import os
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

# Import model classes from src
import sys
sys.path.insert(0, "/path/to/Qwen3.5-35B-A3B/src")
from modeling_qwen35_moe import NeuronQwen35MoeForCausalLM, Qwen35MoeInferenceConfig

model_path = "/path/to/Qwen3.5-35B-A3B/"
compiled_model_path = "/path/to/compiled/"

# Load HF config
with open(os.path.join(model_path, "config.json")) as f:
    full_config = json.load(f)
text_config = full_config.get("text_config", full_config)

# Configure with multi-bucket CTE for optimal TTFT
# - context_encoding_buckets=[512, 1024] avoids padding short inputs to a single large bucket
# - block_size=256 enables the blockwise MoE NKI kernel (faster than forward_all_experts)
neuron_config = MoENeuronConfig(
    tp_degree=4,
    max_batch_size=1,
    seq_len=1152,                    # max_context_length + max_new_tokens
    max_context_length=1024,
    max_new_tokens=128,
    context_encoding_buckets=[512, 1024],
    on_device_sampling_config=None,
    torch_dtype=torch.bfloat16,
    fused_qkv=True,
    moe_tp_degree=4,
    moe_ep_degree=1,
    blockwise_matmul_config={"block_size": 256},
)
```

### Short context only (seq_len=128)

```python
# For short-context-only workloads. Uses a single CTE bucket.
neuron_config = MoENeuronConfig(
    tp_degree=4,
    max_batch_size=1,
    max_context_length=128,
    max_new_tokens=32,
    on_device_sampling_config=None,
    torch_dtype=torch.bfloat16,
    fused_qkv=True,
    moe_tp_degree=4,
    moe_ep_degree=1,
    blockwise_matmul_config={"block_size": 256},
)
```

### Common setup (both configurations)

```python
# Merge text_config with NeuronConfig
config_dict = dict(text_config)
config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
if "rope_parameters" in text_config:
    config_dict["rope_theta"] = text_config["rope_parameters"].get("rope_theta", 10000000)
if config_dict.get("tie_word_embeddings") is None:
    config_dict["tie_word_embeddings"] = False

config = Qwen35MoeInferenceConfig(neuron_config=neuron_config, **config_dict)

# Compile, load, and run
model = NeuronQwen35MoeForCausalLM(model_path=model_path, config=config)

# Remove XLA_DISABLE_FUNCTIONALIZATION (set by torch_neuronx import)
os.environ.pop("XLA_DISABLE_FUNCTIONALIZATION", None)

model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("The capital of France is", return_tensors="pt")

# Context encoding (prefill)
with torch.no_grad():
    output = model(
        input_ids=inputs["input_ids"],
        position_ids=torch.arange(inputs["input_ids"].shape[1]).unsqueeze(0),
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    )

logits = output[0] if isinstance(output, tuple) else output.logits
next_token = torch.argmax(logits[:, -1, :], dim=-1)
print(f"Next token: {tokenizer.decode(next_token)}")  # " Paris"
```

## Compatibility Matrix

| Instance/Version | SDK 2.28 | SDK 2.27 and earlier |
|------------------|----------|----------------------|
| trn2 (trn2.3xlarge) | **Validated** | Not tested |
| trn1 | Not tested | Not tested |
| inf2 | Not tested | Not tested |

### Known Issues

- **CTE bucket 256 compiler error (SDK 2.28)**: Compiling a context encoding NEFF with bucket size 256 hits `[NCC_ITEN404] Internal tensorizer error: LegalizePartitionReduce` on the RMSNorm mean operation. Minimum working CTE bucket is 512. This limits multi-bucket configs to `[512, ...]` or larger.
- **NKI kernel OOB in NxDI (SDK 2.28)**: Custom NKI kernels that work standalone and in `torch_neuronx.trace()` fail with "out-of-bound access" errors when compiled as part of an NxDI model graph. This affects the chunked DeltaNet NKI kernel (which would have further reduced TTFT). Suspected conflict between multiple NKI kernels (custom + internal MoE blockwise) in the same NEFF. The workaround is to use the PyTorch `chunk_forward(64)` path instead.
- **NKI flash attention head_dim>128 (SDK 2.28)**: The NKI flash attention kernel (`CausalAttentionMMSoftmaxMMWithoutSwap`) asserts `head_dim <= 128`. Qwen3.5 uses head_dim=256. The model's `perform_prefill()` override automatically falls back to the PyTorch softmax attention path, so no manual `attn_kernel_enabled=False` is required. A custom NKI kernel (`nki_flash_attn_d256.py`) is included that supports head_dim=256 by tiling the QK contraction in 2x128 chunks. However, benchmarks show it is ~2.4x slower than the PyTorch path due to layout conversion overhead (BHSD->BHDS permute). To experiment with it, set `QWEN35_USE_FLASH_ATTN_D256=1`.
- **nkilib kernel integration blocked by NKI compiler bug**: The nki-library flash attention kernel has been extended to support head_dim up to 256 (see [nki-library fork, feature/head-dim-256](https://github.com/jimburtoft/nki-library/tree/feature/head-dim-256)). All standalone unit tests pass. However, integrating it with NxDI is blocked by an NKI compiler bug: `TraceKernel.inline_function` does not propagate the trace context (`builtin` injection) to sub-functions, causing `NameError: name 'builtin' is not defined` for all `nisa.*` ISA operations. The `nkilib_kernel_patch.py` module is included and ready to enable once the compiler bug is fixed (set `QWEN35_PATCH_FLASH_ATTN=1`). Multiple additional torchxla compatibility fixes were made in the nki-library fork (symbolic tile_size, num_programs returning None, address= fallback).
- **NEURON_PLATFORM_TARGET_OVERRIDE**: Must set `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` when running with NKI v2 `@nki.jit` kernels on trn2.
- **Memory**: The full model in BF16 is ~67GB. trn2.3xlarge (124GB system RAM) can load it but requires careful memory management during compilation.
- **Compilation time**: Each (batch_size, context_encoding_bucket) combination requires a separate CTE NEFF compilation. Multi-bucket configs increase total compile time linearly with the number of buckets.

## Testing

### Environment Setup

```bash
# On trn2.3xlarge with Neuron SDK 2.28
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
```

### Run integration tests

```bash
pytest contrib/models/Qwen3.5-35B-A3B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/Qwen3.5-35B-A3B
python3 test/integration/test_model.py
```

## Example Checkpoints

- [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (67 GB, 14 safetensors shards)

## Vision-Language Support

Qwen3.5-35B-A3B is a **native multimodal model** -- all Qwen3.5 models have vision built-in (there is no separate `-VL` variant). The architecture includes a 27-layer ViT vision encoder (1152 hidden, 16 heads, patch_size=16) that projects to the text decoder's embedding space.

### Current Status

| Component | Status |
|-----------|--------|
| Text decoder (DeltaNet + GQA + MoE) | **Validated** on Neuron |
| Vision encoder (ViT) | **Implemented** (CPU execution, Neuron compilation planned) |
| mRoPE (3D multimodal position IDs) | **Implemented** and tested |
| Image+text generation | **Implemented** via `NeuronQwen35MoeVLForCausalLM` |
| Vision encoder on Neuron | Planned (requires separate compilation) |

### Vision-Language Usage

```python
from transformers import AutoProcessor
from modeling_qwen35_moe_vl import NeuronQwen35MoeVLForCausalLM, Qwen35MoeVLInferenceConfig

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Prepare image+text inputs
input_ids, attention_mask, vision_inputs = NeuronQwen35MoeVLForCausalLM.prepare_input_args(
    "What is in this image?",
    "/path/to/image.jpg",
    processor,
)

# Generate
generated_ids = vl_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=vision_inputs.get("pixel_values"),
    image_grid_thw=vision_inputs.get("image_grid_thw"),
    max_new_tokens=32,
)
```

### Running Image Input Tests

```bash
pytest contrib/models/Qwen3.5-35B-A3B/test/integration/test_image_input.py --capture=tee-sys
```

## Files

| File | Description |
|------|-------------|
| `src/modeling_qwen35_moe.py` | Main NxDI model: DeltaNet, attention, MoE, config, state dict converter |
| `src/modeling_qwen35_moe_vision.py` | Vision encoder: ViT blocks, patch merger, rotary embeddings, model wrapper |
| `src/modeling_qwen35_moe_vl.py` | VL orchestrator: mRoPE, vision+text wiring, generate with image inputs |
| `src/nki_deltanet.py` | NKI v2 kernel for DeltaNet gated delta rule (used for TKG token generation) |
| `src/nki_deltanet_chunked.py` | NKI v2 chunked DeltaNet kernel for CTE (blocked by NKI OOB issue, see Known Issues) |
| `src/nki_flash_attn_d256.py` | Custom NKI flash attention kernel for head_dim=256 (opt-in, experimental) |
| `src/nkilib_kernel_patch.py` | nkilib kernel integration for NxDI (blocked by NKI compiler bug, see Known Issues) |
| `src/__init__.py` | Re-exports public classes |
| `test/integration/test_model.py` | Integration tests: text-only accuracy + performance |
| `test/integration/test_image_input.py` | Integration tests: image tokenization, mRoPE, vision encoder, E2E VL pipeline |

## Maintainer

Jim Burtoft - AWS
