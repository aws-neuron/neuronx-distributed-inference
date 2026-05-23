# Contrib Model: GLM-4.7-Flash

Neuron inference support for [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) (30B-A3B MoE) using NxD Inference. This model uses DeepSeek-V3-style Multi-head Latent Attention (MLA) with compressed KV cache, 64 routed experts with sigmoid routing, and supports up to 16K context on trn2.3xlarge.

## Model Information

- **HuggingFace ID:** `zai-org/GLM-4.7-Flash`
- **Model Type:** Decoder-only MoE transformer with MLA attention
- **Parameters:** ~31B total, ~3B active per token (64 experts, top-4 routing)
- **Architecture:** MLA (Multi-head Latent Attention), MoE with sigmoid routing + shared expert, SiLU activation, RMSNorm, standard RoPE
- **License:** MIT

## Validation Results

**Validated:** 2026-05-23
**Instance:** trn2.3xlarge (TP=4, LNC=2)
**SDK:** Neuron SDK 2.29.1 (neuronx-cc 2.24.8799, NxDI 0.9.17334)

### Benchmark Results

Configuration: BS=4, SEQ_LEN=16384, FP8 expert quantization, NKI `bwmm_shard_on_block` CTE kernel

| Prompt Length | TTFT | TPOT | Throughput (batch total) |
|--------------|------|------|------------------------|
| 128 tokens | 758 ms | 77 ms | 51.7 tok/s |
| 1,024 tokens | 1,189 ms | 77 ms | 51.7 tok/s |
| 4,096 tokens | 5,012 ms | 78 ms | 51.6 tok/s |
| 8,192 tokens | 11,374 ms | 81 ms | 49.5 tok/s |
| 16,000 tokens | 11,379 ms | 81 ms | 49.6 tok/s |

### Batch Size Scaling

| Batch Size | Seq Len | TPOT | Throughput | SDK Required |
|-----------|---------|------|-----------|--------------|
| 4 | 16384 | 77 ms | 51.7 tok/s | 2.29.1 |
| 8 | 4096 | 105.6 ms | 75.7 tok/s | 2.29.1 |
| 16 | 4096 | 160.3 ms | 99.8 tok/s | 2.29.1 |

### vLLM Serving Performance

Configuration: BS=4, SEQ_LEN=16384, vLLM 0.16.0 + vllm-neuron 0.5.0

| Concurrency | Output Throughput | Mean Latency | TPOT |
|-------------|------------------:|-------------:|-----:|
| 1 | 12.7 tok/s | 10.1s | 78 ms |
| 2 | 24.8 tok/s | 10.3s | 78 ms |
| 4 | 48.1 tok/s | 10.6s | 78 ms |

### Accuracy Validation

First-token accuracy against CPU FP32 reference (greedy/top-k=1):

| Prompt | Expected Token | Got Token | Status |
|--------|---------------|-----------|--------|
| "The capital of France is" | " Paris" (12089) | " Paris" (12089) | EXACT MATCH |
| "In machine learning, a transformer model" | " is" (374) | " is" (374) | EXACT MATCH |
| "def fibonacci(n):" | "\n" (715) | "\n" (715) | EXACT MATCH |

Multi-token generation produces coherent, non-repetitive text at 16K context.

## Usage

```python
import os
import torch
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from src import (
    Glm4MoeLiteGenerationAdapter,
    Glm4MoeLiteInferenceConfig,
    NeuronGlm4MoeLiteForCausalLM,
)

os.environ["NEURON_RT_VISIBLE_CORES"] = "0-3"

# Register glm4_moe_lite model type (not yet in transformers registry)
class Glm4MoeLiteConfig(Glm4MoeConfig):
    model_type = "glm4_moe_lite"

AutoConfig.register("glm4_moe_lite", Glm4MoeLiteConfig)

MODEL_PATH = "/path/to/GLM-4.7-Flash"
COMPILED_PATH = "/path/to/compiled_glm4"

# Load HF config
hf_config = AutoConfig.from_pretrained(MODEL_PATH)

# Configure for Neuron
neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=4,
    ctx_batch_size=1,
    tkg_batch_size=4,
    seq_len=16384,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    enable_bucketing=True,
    flash_decoding_enabled=False,
    logical_nc_config=2,
)

inf_config = Glm4MoeLiteInferenceConfig(
    neuron_config, load_config=load_pretrained_config(hf_config=hf_config)
)

# Compile (first time only)
model = NeuronGlm4MoeLiteForCausalLM(MODEL_PATH, inf_config)
model.compile(COMPILED_PATH)

# Load compiled model
model = NeuronGlm4MoeLiteForCausalLM(COMPILED_PATH, inf_config)
model.load(COMPILED_PATH)

# Generate (MUST use Glm4MoeLiteGenerationAdapter for transformers >= 5.0)
gen_model = Glm4MoeLiteGenerationAdapter(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

gen_config = GenerationConfig(
    do_sample=True, top_k=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

prompt = "The capital of France is"
inputs = tokenizer([prompt] * 4, return_tensors="pt", padding=True)
outputs = gen_model.generate(
    inputs.input_ids,
    generation_config=gen_config,
    attention_mask=inputs.attention_mask,
    max_new_tokens=50,
)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
# Output: "Paris. The capital of Belgium is Brussels..."
```

### FP8 Quantized Inference

For FP8 expert quantization (reduces memory, improves throughput):

```python
# Pre-quantize expert weights (one-time step)
# Use scripts/quantize_experts_fp8.py to generate FP8 checkpoint

neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=4,
    ctx_batch_size=1,
    tkg_batch_size=4,
    seq_len=16384,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    enable_bucketing=True,
    flash_decoding_enabled=False,
    logical_nc_config=2,
    # FP8 configuration
    quantized=True,
    quantization_type="expert_wise_per_channel_symmetric",
    quantization_dtype="f8e4m3",
    quantized_checkpoints_path="/path/to/GLM-4.7-Flash-FP8",
    modules_to_not_convert=[
        "lm_head", "embed_tokens", "self_attn", "norm",
        "layers.0.mlp", "shared_experts", "router",
    ],
    moe_fused_nki_kernel_enabled=True,
)
```

### vLLM Serving

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
export PYTHONPATH=/path/to/GLM-4.7-Flash-contrib:$PYTHONPATH
export NEURON_RT_VISIBLE_CORES=0-3
export UNSAFE_FP8FNCAST=1
export NEURON_COMPILED_ARTIFACTS=/path/to/compiled_model

vllm serve /path/to/GLM-4.7-Flash \
  --tensor-parallel-size 4 --dtype bfloat16 --block-size 128 \
  --max-model-len 16384 --max-num-seqs 4 \
  --additional-config '{"override_neuron_config": {"quantized": true, "quantization_type": "expert_wise_per_channel_symmetric", "quantization_dtype": "f8e4m3", "quantized_checkpoints_path": "/path/to/GLM-4.7-Flash-FP8", "modules_to_not_convert": ["lm_head", "embed_tokens", "self_attn", "norm", "layers.0.mlp", "shared_experts", "router"], "moe_fused_nki_kernel_enabled": true, "logical_nc_config": 2, "flash_decoding_enabled": false}}'
```

**Note:** vLLM integration requires:
1. Changing `model_type` in config.json from `glm4_moe_lite` to `glm4_moe` (for HF AutoConfig compatibility with transformers 4.57.6)
2. Registering `glm4moelite` in NxDI's `constants.py` MODEL_TYPES dict
3. Removing `auto_map` field and custom `.py` files from model directory

## Compatibility Matrix

| Instance | SDK 2.29.1 | SDK 2.29 |
|----------|-----------|----------|
| trn2.3xlarge (TP=4, LNC=2) | **VALIDATED** (BS=4-16) | VALIDATED (BS=4-8 only) |
| trn2.48xlarge | Not tested | Not tested |

## Example Checkpoints

* [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) (BF16, 59 GB)

## Testing Instructions

```bash
# Unit tests (no Neuron device required):
python -m pytest test/unit/ -v

# Integration tests (requires trn2.3xlarge with model weights):
GLM4_MODEL_PATH=/path/to/GLM-4.7-Flash \
GLM4_COMPILED_PATH=/path/to/compiled_glm4 \
pytest test/integration/test_model.py --capture=tee-sys
```

## Known Issues

### transformers 5.x position_ids Compatibility

NxDI's `HuggingFaceGenerationAdapter.prepare_inputs_for_generation` only recomputes `position_ids` when they are `None` in kwargs. In transformers >= 5.0, `_update_model_kwargs_for_generation` passes stale position_ids back, breaking decode.

**Fix**: Use `Glm4MoeLiteGenerationAdapter` (included) which removes stale `position_ids` from kwargs. This issue affects all NxDI contrib models with transformers >= 5.0.

### Minimum batch_size=4 (Compiler Workaround)

The Neuron compiler has an internal issue (NCC_IBIR297) that causes compilation failure with `tkg_batch_size=1` and blockwise MoE at small TP degrees. Workaround: set `batch_size >= 4`.

### BS > 8 requires SDK 2.29.1

On SDK 2.29 (neuronx-cc 2.24.5133), batch sizes > 8 cause a runtime DGE scatter/gather out-of-bounds error. Fixed in SDK 2.29.1 (neuronx-cc 2.24.8799).

### Maximum context length

- **16,384 tokens**: Maximum validated CTE bucket (uses 24 GB HBM budget cleanly)
- **32,768 tokens**: OOM by 0.25 GB per LNC=2 core
- Longer contexts would require LNC=1 (48 GB per core) or FP8 KV cache

### `glm4_moe_lite` model_type registration

The `glm4_moe_lite` model_type is not registered in transformers 4.57.6. Register it manually using the pattern in the Usage section. For vLLM, use `model_type: "glm4_moe"` in config.json instead.

## Architecture Details

### MLA (Multi-head Latent Attention)

- **KV cache compression**: Stores only 576 dims per position (kv_lora_rank=512 + qk_rope_head_dim=64) vs 10,240 for standard MHA — 94% reduction
- **Weight absorption trick**: kv_b_proj weights absorbed into query-side computation to avoid decompression at decode time
- **Critical dimension fix**: `out_absorb = wkv_b[:, qk_nope_head_dim:, :]` (split at 192, not v_head_dim=256)

### MoE Configuration

- 64 routed experts, top-4 selection
- Sigmoid activation with e_score_correction_bias
- L1-normalized affinities, scaled by factor 1.8
- 1 shared expert (always-active dense MLP)
- Layer 0 is fully dense, layers 1-46 are MoE
- FP8 E4M3 quantization for routed expert weights (attention/embeddings remain BF16)

### Compiler Flags

```
--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1
--tensorizer-options='--vectorize-strided-dma'
--auto-cast=none
--lnc=2
```

**Important**: Do NOT add `--enable-ccop-compute-overlap` or `--cc-pipeline-tiling-factor` — these cause an ICE and 4.8x performance degradation.

## Maintainer

Jim Burtoft
