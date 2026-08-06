# Contrib Model: Qwen3.5-27B

NeuronX Distributed Inference implementation of Qwen3.5-27B, a 27B parameter dense model from Alibaba Cloud with a hybrid DeltaNet + GQA attention architecture. This is the first NxDI implementation of a model using linear recurrent attention (DeltaNet) with custom NKI kernels.

## Model Family

| Model | HuggingFace ID | Params | Instance |
|-------|----------------|--------|----------|
| **Qwen3.5-27B** | `Qwen/Qwen3.5-27B` | 27B | trn2.3xlarge (TP=4) |
| **Qwen3.5-27B-VL** | `Qwen/Qwen3.5-27B-VL` | 27B + ViT | trn2.3xlarge (TP=4) |

**License:** Apache 2.0

## Architecture Details

| Feature | Value |
|---------|-------|
| Layers | 64 (48 DeltaNet + 16 GQA) |
| Layer Pattern | [3 DeltaNet + 1 GQA] x 16 |
| Hidden Size | 5120 |
| GQA Attention | 24 heads, 4 KV heads, head_dim=256 |
| DeltaNet Attention | 48 value heads, 16 key heads, k_dim=v_dim=128 |
| Dense MLP | SwiGLU (gate_proj + up_proj: 5120 -> 17408, down_proj: 17408 -> 5120) |
| Position Encoding | Partial RoPE (25% of head_dim = 64 dims), mRoPE for VL |
| Vocabulary | 248,320 |
| Normalization | RMSNorm with +1 weight convention |
| Activation | SiLU gated MLP |

### Unique Architecture Features

- **Hybrid DeltaNet + GQA:** 48 of 64 layers use Gated DeltaNet (linear recurrent attention), 16 layers use standard GQA with KV cache. The pattern repeats every 4 layers: 3 DeltaNet + 1 GQA.
- **DeltaNet Linear Attention:** Uses the delta rule for recurrent state updates with gated decay. Per-step: `state *= exp(g); delta = (v - state^T @ k) * beta; state += outer(k, delta); output = state^T @ q`. Runs as a chunked algorithm for context encoding, per-token recurrence for token generation.
- **Custom NKI Kernels:** Three NKI kernels implement the DeltaNet forward pass on Neuron: a per-token recurrent kernel (TKG), a per-chunk kernel (legacy), and a fused single-kernel chunked forward (CTE). The fused kernel uses a Neumann series for intra-chunk correction with state persistence in SBUF across chunks.
- **GQA Output Gate:** Attention layers use a sigmoid output gate. `q_proj` is 2x sized and interleaved: `[head0_query | head0_gate | head1_query | ...]`. The gate is split during weight conversion and applied after attention.
- **Partial RoPE:** Only 25% of head_dim (64 of 256 dimensions) receives rotary embeddings. The remaining 192 dimensions are identity (no rotation).
- **+1 RMSNorm Convention:** HF weights use `output = norm(x) * (1 + weight)` where weight is initialized to zeros. Converted to standard `output = norm(x) * weight` during loading by adding 1.0 to all RMSNorm weights (except DeltaNet internal norms, which use standard convention).
- **Vision-Language Support:** Optional ViT encoder runs on CPU (HBM fully consumed by 27B text decoder). Vision embeddings are injected via a scatter mask at traced input positions.

## Test Results

### Unit Tests (CPU)

| Test Module | Tests | Status |
|-------------|-------|--------|
| test_config.py | 26 | 26/26 PASS |
| test_weight_conversion.py | 16 | 16/16 PASS |
| **Total** | **42** | **42/42 PASS** |

### Integration Test (27B, trn2.3xlarge, TP=4, SDK 2.29)

| Test | Status | Notes |
|------|--------|-------|
| Model loads | PASS | Compiled + loaded with DeltaNet state aliasing |
| Model generates | PASS | Generates coherent multi-sentence text |
| Output coherence | PASS | 3+ words, no excessive repetition |
| Top token valid | PASS | First token decodable and semantically valid |
| Capital of France | PASS | Produces "Paris" as first token |
| TTFT performance | PASS | ~576 ms (128 input tokens, bs=1) |
| Throughput | PASS | ~18.9 tok/s (bs=1) |
| Multi-prompt generation | PASS | 4/4 prompts produce coherent output |

**All 50 tests pass (42 unit + 8 integration) on SDK 2.29.**

### Generation Output (27B, TP=4, seq_len=128, greedy top_k=1)

**Prompt:** "The capital of France is"

**Output:** Paris. It is the largest city in France and serves as the country's political, cultural, and economic center...

**Status:** PASS -- coherent, factually correct, multi-sentence response.

## Performance Benchmarks

**SDK 2.29**, BF16, trn2.3xlarge (4 NeuronCores, LNC=2), seq_len=128, bs=1.

### Text-Only Benchmarks

| Metric | Value |
|--------|-------|
| **TTFT (P50)** | 576 ms |
| **TPOT (P50)** | 53 ms |
| **Throughput** | 18.9 tok/s |
| Compilation time | ~13 min |
| Weight loading | ~31 s |
| HBM usage | 23.57 GB / 24 GB |

### Vision-Language Benchmarks (VL pipeline)

| Metric | Value |
|--------|-------|
| Vision encoder (CPU) | ~918 ms |
| Text generation (Neuron) | ~3.9 s (30 tokens) |
| End-to-end VL | ~4.8 s |

### NKI Kernel Benchmarks (standalone, single NeuronCore)

| Kernel | 128 tokens | 256 tokens | 512 tokens |
|--------|-----------|-----------|-----------|
| Fused chunked (CTE) | 335 us | 339 us | 487 us |
| Recurrent (TKG, S=1) | 183 us | - | - |

### Key Observations

- **HBM-limited at TP=4:** The 27B model consumes 23.57 GB of 24 GB HBM per NeuronCore pair. Context length limited to 128-512 tokens. Use trn2.12xlarge for longer contexts.
- **DeltaNet enables efficient TKG:** Token generation uses O(1) per-token recurrence instead of O(n) KV cache attention for 48/64 layers, keeping TPOT at 53ms.
- **Vision encoder on CPU:** The ViT runs on CPU because HBM is fully consumed by the text decoder. CPU vision adds ~918ms latency per image.
- **Fused NKI kernel 2.8-5.2% faster:** The fused chunked kernel provides modest TTFT improvement over the PyTorch baseline (larger gains at longer contexts).

## Usage

### Text-Only (trn2.3xlarge, TP=4)

```python
import json
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

model_path = "/path/to/Qwen3.5-27B"
compiled_path = "/scratch/qwen35_traced/"

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    save_sharded_checkpoint=True,
)

# Read config.json directly (model_type 'qwen3_5' may not be
# registered in all transformers versions)
import os
with open(os.path.join(model_path, "config.json")) as f:
    hf_config = json.load(f)
text_config = hf_config.get("text_config", hf_config)
config_dict = dict(text_config)
config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
config_dict.setdefault("tie_word_embeddings", False)

config = Qwen35InferenceConfig(
    neuron_config=neuron_config,
    **config_dict,
)

model = NeuronQwen35ForCausalLM(model_path, config)
model.compile(compiled_path)

# Reload from compiled artifacts
model = NeuronQwen35ForCausalLM(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
gen_config = GenerationConfig(
    do_sample=True, top_k=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

inputs = tokenizer("The capital of France is", return_tensors="pt")
gen_model = HuggingFaceGenerationAdapter(model)
outputs = gen_model.generate(
    inputs.input_ids,
    generation_config=gen_config,
    attention_mask=inputs.attention_mask,
    max_new_tokens=50,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Vision-Language (trn2.3xlarge, TP=4)

The VL pipeline uses the text decoder on Neuron and the vision encoder on CPU:

```python
from src.modeling_qwen35_vl import NeuronQwen35VLForCausalLM, Qwen35VLInferenceConfig

vl_model = NeuronQwen35VLForCausalLM(
    model_path="/path/to/Qwen3.5-27B",
    config=vl_config,
)
vl_model.compile(compiled_path)
vl_model.load(compiled_path)

# See test/integration/test_model.py for full VL usage example
```

### DeltaNet Kernel Selection

The DeltaNet forward path can be controlled via environment variables:

| Env Var | Forward Path | Use Case |
|---------|-------------|----------|
| `USE_NKI_FUSED=1` | Fused chunked NKI kernel | Best CTE performance (default for SDK 2.29) |
| `USE_NKI_CHUNKED=1` | Per-chunk NKI kernel | Legacy, superseded by fused |
| `USE_NKI=1` | Per-token NKI kernel | TKG (always used for token generation) |
| `DELTANET_SEQUENTIAL=1` | Sequential PyTorch | Debugging/reference |
| *(none)* | PyTorch chunked | Default fallback for CTE |

## Caveats

1. **HBM-limited at TP=4:** The 27B model consumes 23.57 GB of the 24 GB HBM per NeuronCore pair (LNC=2). Context length is limited to ~512 tokens. Batch size > 1 not possible. Use trn2.12xlarge (TP=16) for production workloads.

2. **SDK 2.29+ required:** The NKI DeltaNet kernels require NKI 0.3.0 (SDK 2.29). No library modifications needed — runs on stock SDK 2.29 DLAMI (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`).

3. **No mini model test:** Unlike DeepSeek-V3, a mini model cannot be provided because DeltaNet layers require NKI kernels that only execute on Neuron devices. Integration tests require a trn2 instance with the full 27B weights.

4. **Vision encoder runs on CPU:** The ViT cannot be placed on Neuron because HBM is fully consumed by the text decoder. This adds ~918ms latency per image. Future optimization: quantize text decoder to free HBM, or use larger instance.

5. **Compilation time:** First compilation takes ~13 minutes. Subsequent compilations with cached NEFFs take ~1 minute.

6. **+1 RMSNorm convention:** Qwen3.5 uses `output = norm(x) * (1 + weight)` for most RMSNorm layers, but DeltaNet internal norms use standard `output = norm(x) * weight`. The weight conversion handles this automatically, but custom weight loading must be aware of both conventions.

7. **Neumann series convergence:** The fused DeltaNet kernel uses a 6-round Neumann series for intra-chunk correction. This requires L2-normalized Q and K inputs. Unnormalized inputs will cause NaN divergence.

## Maximum Sequence Length

| seq_len | Status | Notes |
|---------|--------|-------|
| 128 | **PASS** | Default, all benchmarks |
| 512 | **PASS** | Compiles and runs, 4 DeltaNet chunks |
| 1024 | **FAIL** | Compiler/runtime OOM (HBM full at TP=4) |

For seq_len > 512, use trn2.12xlarge or larger instance with TP > 4.

## Compatibility Matrix

| Instance | TP | LNC | Status | Notes |
|----------|-----|-----|--------|-------|
| trn2.3xlarge | 4 | 2 | **PASS** | Tested, HBM-limited |
| trn2.12xlarge | 16 | 2 | Expected PASS | Untested, recommended for production |

### SDK Configuration

| Component | Version |
|-----------|---------|
| NxDI | 0.9.17334 |
| neuronx-cc | 2.24.5133 |
| torch | 2.9.1 |
| transformers | 4.57.6 |
| NKI | 0.3.0 |
| NXDI venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` |

## Testing

### Unit Tests (CPU only, no device needed)

```bash
cd contrib/models/Qwen3.5-27B/
# On DLAMI: source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pytest test/unit/ -v
```

Tests: config parsing (26), weight conversion (16) = **42 tests**.

### Integration Tests (needs trn2.3xlarge with 4 NeuronCores)

```bash
cd contrib/models/Qwen3.5-27B/

QWEN35_MODEL_PATH=/mnt/models/Qwen3.5-27B \
QWEN35_COMPILED_PATH=/mnt/models/qwen35_traced \
pytest test/integration/test_model.py --capture=tee-sys
```

Tests: model loads, generates, coherence, top-token valid, capital test, TTFT, throughput, multi-prompt = **8 tests**.

## Key Porting Challenges

1. **DeltaNet on Neuron:** No prior NxDI implementation of linear recurrent attention exists. Required writing three custom NKI kernels (recurrent, chunked, fused) with careful SBUF state management and Neumann series approximation for the chunked intra-chunk correction.

2. **Hybrid state management:** DeltaNet layers maintain per-head (128, 128) recurrent state and (conv_dim, kernel_size-1) conv state, while GQA layers use standard KV cache. Both must be aliased as `input_output_aliases` in the XLA trace for HBM persistence across forward calls.

3. **Interleaved q_proj:** The HF checkpoint stores Q and gate weights interleaved as `[head0_q | head0_gate | head1_q | head1_gate | ...]`. Must reshape to (num_heads, 2*head_dim, hidden), then split along dim=1.

4. **Dual RMSNorm conventions:** 48 DeltaNet layers use standard `norm(x) * weight` while all 64 `input_layernorm` / `post_attention_layernorm` and 16 GQA Q/K norms use `norm(x) * (1 + weight)`. Weight conversion must selectively add 1.0 only to the correct subset.

5. **DeltaNet conv1d state:** Each DeltaNet layer has a causal conv1d (kernel_size=4) that requires 3 previous timesteps. Conv state is stored as an nn.Parameter buffer and aliased for HBM persistence, similar to Mamba's conv state.

6. **`aten.scatter.src` unsupported:** Neuron compiler does not support `aten.scatter.src`. DeltaNet state updates use `new_state + buffer * 0` pattern instead.

## Example Checkpoints

- `Qwen/Qwen3.5-27B` (BF16, ~52 GB, 11 shards)
- `Qwen/Qwen3.5-27B-VL` (BF16, VL variant with ViT)

## Maintainer

AWS Neuron

**Last Updated:** 2026-04-12
