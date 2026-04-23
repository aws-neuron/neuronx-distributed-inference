# Contrib Model: Qwen3.6-27B

NeuronX Distributed Inference implementation of Qwen3.6-27B, a 27B parameter dense model from Alibaba Cloud with a hybrid DeltaNet + GQA attention architecture.

## Relationship to Qwen3.5-27B

Qwen3.6-27B is a **post-training update** of Qwen3.5-27B with improved agentic coding and thinking preservation. The models share **identical architecture** (`qwen3_5` model_type, `Qwen3_5ForConditionalGeneration`) -- only weights differ. This contrib reuses the same NxDI implementation as [Qwen3.5-27B](../Qwen3.5-27B/) (PR #128). Any code updates to Qwen3.5-27B should be propagated to this contrib and vice versa.

### Config differences from Qwen3.5-27B

| Field | Value | Impact |
|-------|-------|--------|
| `output_gate_type` | `"swish"` | **Ignored** -- not used by HF transformers or NxDI (gate uses sigmoid) |
| `language_model_only` | `false` | Informational, not used by model code |
| `bos_token_id` | `248044` | New but not architecture-relevant |
| `pad_token_id` | `null` | New at text_config level (already handled) |
| `partial_rotary_factor` | `0.25` | Already in rope_parameters, redundant copy |
| `transformers_version` | `4.57.1` | Updated from `4.57.0.dev0` |

No code changes are required to support Qwen3.6-27B.

## Model Family

| Model | HuggingFace ID | Params | Instance |
|-------|----------------|--------|----------|
| **Qwen3.6-27B** | `Qwen/Qwen3.6-27B` | 27B | trn2.3xlarge (TP=4) |

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

Unit tests are architecture-level and do not depend on weights. Identical results to Qwen3.5-27B.

### Quality Validation (Qwen3.6-27B, trn2.3xlarge, TP=4, SDK 2.29)

7/7 text-only quality tests passed with `enable_thinking=False`:

| Test | Expected | Result |
|------|----------|--------|
| Speed of light | 299,792,458 m/s | PASS |
| 17 * 23 | 391 | PASS |
| 60mph * 2.5h | 150 miles | PASS |
| is_prime function | Correct Python | PASS |
| French translation | Bonjour, comment allez-vous ? | PASS |
| Capital of Japan | Tokyo | PASS |
| sqrt(144) | 12 | PASS |

## Performance Benchmarks

### Qwen3.6-27B on trn2.3xlarge (TP=4, LNC=2, SDK 2.29, BF16)

**TTFT (Time To First Token)**

| Input Length | P50 (ms) | P95 (ms) |
|-------------|----------|----------|
| 16 tokens | 305.3 | 305.6 |
| 64 tokens | 305.4 | 305.9 |
| 128 tokens | 306.6 | 306.8 |
| 256 tokens | 306.2 | 306.3 |

**TPOT / Throughput**

| Output Length | TPOT P50 (ms) | tok/s P50 | E2E P50 (ms) |
|--------------|---------------|-----------|---------------|
| 16 | 54.3 | 18.4 | 1,121 |
| 32 | 54.4 | 18.4 | 1,993 |
| 64 | 54.2 | 18.5 | 3,720 |
| 128 | 54.2 | 18.5 | 4,912 |

### Comparison with Qwen3.5-27B

| Metric | Qwen3.5-27B | Qwen3.6-27B | Delta |
|--------|------------|------------|-------|
| TPOT P50 | 53 ms | 54.2 ms | +2.3% |
| Throughput | 18.9 tok/s | 18.5 tok/s | -2.1% |
| TTFT (128 tok) | 576 ms | 306.6 ms | -47% * |

\* TTFT improvement is due to compilation config differences (256-token bucket vs 128-token bucket), not model differences. Architectural performance is equivalent.

### Key Observations

- **HBM-limited at TP=4:** The 27B model consumes 23.57 GB of 24 GB HBM per NeuronCore pair. Context length limited to 128-512 tokens. Use trn2.12xlarge for longer contexts.
- **DeltaNet enables efficient TKG:** Token generation uses O(1) per-token recurrence instead of O(n) KV cache attention for 48/64 layers, keeping TPOT at ~54ms.
- **Performance equivalent to Qwen3.5-27B:** The +2.3% TPOT difference is within measurement noise. Expected since architectures are identical.

## Usage

### Text-Only (trn2.3xlarge, TP=4)

```python
import json
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

model_path = "/path/to/Qwen3.6-27B"
compiled_path = "/scratch/qwen36_traced/"

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
    model_path="/path/to/Qwen3.6-27B",
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

2. **SDK 2.29+ required:** The NKI DeltaNet kernels require NKI 0.3.0 (SDK 2.29). No library modifications needed -- runs on stock SDK 2.29 DLAMI (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`).

3. **No mini model test:** Unlike DeepSeek-V3, a mini model cannot be provided because DeltaNet layers require NKI kernels that only execute on Neuron devices. Integration tests require a trn2 instance with the full 27B weights.

4. **Vision encoder runs on CPU:** The ViT cannot be placed on Neuron because HBM is fully consumed by the text decoder. This adds ~918ms latency per image. Future optimization: quantize text decoder to free HBM, or use larger instance.

5. **Compilation time:** First compilation takes ~13 minutes. Subsequent compilations with cached NEFFs take ~1 minute.

6. **+1 RMSNorm convention:** Qwen3.5/3.6 uses `output = norm(x) * (1 + weight)` for most RMSNorm layers, but DeltaNet internal norms use standard `output = norm(x) * weight`. The weight conversion handles this automatically, but custom weight loading must be aware of both conventions.

7. **Neumann series convergence:** The fused DeltaNet kernel uses a 6-round Neumann series for intra-chunk correction. This requires L2-normalized Q and K inputs. Unnormalized inputs will cause NaN divergence.

8. **Shared codebase with Qwen3.5-27B:** This contrib uses the same `Qwen35*` class names and `modeling_qwen35*.py` filenames as the [Qwen3.5-27B contrib](../Qwen3.5-27B/). This is intentional -- both models share the `qwen3_5` model_type. The code is identical; only the HuggingFace model ID and weights differ.

## Maximum Sequence Length

| seq_len | Status | Notes |
|---------|--------|-------|
| 128 | **PASS** | Default, all benchmarks |
| 256 | **PASS** | Benchmarked (256-token CTE bucket) |
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
cd contrib/models/Qwen3.6-27B/
# On DLAMI: source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pytest test/unit/ -v
```

Tests: config parsing (26), weight conversion (16) = **42 tests**.

### Integration Tests (needs trn2.3xlarge with 4 NeuronCores)

```bash
cd contrib/models/Qwen3.6-27B/

QWEN35_MODEL_PATH=/mnt/models/Qwen3.6-27B \
QWEN35_COMPILED_PATH=/mnt/models/qwen36_traced \
pytest test/integration/test_model.py --capture=tee-sys
```

Tests: model loads, generates, coherence, top-token valid, capital test, TTFT, throughput, multi-prompt = **8 tests**.

Note: The env var is `QWEN35_MODEL_PATH` (not `QWEN36`) because the code uses the `qwen3_5` model_type internally.

## Example Checkpoints

- `Qwen/Qwen3.6-27B` (BF16, ~52 GB)

## Maintainer

AWS Neuron

**Last Updated:** 2026-04-23
