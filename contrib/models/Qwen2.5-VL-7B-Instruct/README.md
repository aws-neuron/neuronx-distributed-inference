# Contrib Model: Qwen2.5-VL-7B-Instruct

Full vision-language implementation of Qwen2.5-VL-7B-Instruct on NeuronX Distributed Inference. Includes both the text decoder and the vision encoder with windowed attention.

> **Note:** Unlike existing Qwen2.5-VL contrib entries (3B, 32B) which only support the text backbone, this implementation provides **complete vision-language inference** including image understanding.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen2.5-VL-7B-Instruct`
- **Model Type:** Vision-Language (encoder-decoder with ViT vision encoder)
- **Architecture:** Qwen2.5-VL (text backbone identical to Qwen2-VL)
- **Parameters:** 7B (text) + 675M (vision) = ~8.3B total
- **License:** Check [HuggingFace model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

## Architecture Details

### Text Decoder
- **Layers:** 28
- **Hidden Size:** 3584
- **Attention Heads:** 28 (Q) / 4 (KV) -- GQA
- **Intermediate Size:** 18944 (SwiGLU MLP)
- **Vocabulary:** 152064
- **Max Position Embeddings:** 128000
- **RoPE:** M-RoPE with sections [16, 24, 24] (temporal, height, width)
- **QKV Bias:** True, O Bias: False

### Vision Encoder
- **Layers:** 32
- **Hidden Size:** 1280
- **Attention Heads:** 16
- **MLP:** Gated SwiGLU with bias (intermediate_size=3420)
- **Normalization:** RMSNorm (not LayerNorm)
- **Attention:** Hybrid windowed (28 layers, 4x4 windows) + global (4 layers: [7,15,23,31])
- **Patch Size:** 14x14, Temporal Patch Size: 2
- **Spatial Merge Size:** 2

## Validation Results

**Validated:** 2026-03-29
**Instance:** trn2.3xlarge (LNC=2, 4 logical cores), trn2.48xlarge (72B)
**SDK:** Neuron SDK 2.28

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads from compiled artifacts |
| Text-only Generation | PASS | "The capital of France is Paris." (exact CPU match) |
| Logit Validation | PASS | Greedy output matches HF CPU reference |
| VL Generation | PASS | Correctly identifies shapes/colors in synthetic images |
| Multi-resolution VL | PASS | 224x224, 448x448, 672x672, 640x480 all working |
| vllm-neuron API | PASS | 6/6 OpenAI-compatible API tests passed |
| Multi-bucket CTE | PASS | 7/7 tests pass with optimized bucketing config |

### Performance Metrics (TP=4, trn2.3xlarge, optimized config)

Configuration: Multi-bucket CTE [512, 1024, 2048, 4096], vision flash attention enabled.

| Metric | Text-only | Vision-Language |
|--------|-----------|-----------------|
| Token Generation | 86.4 tok/s | 86.7 tok/s |
| TPOT | 11.57 ms | 11.57 ms |
| HBM per Core | 4.2 GB | 4.2 GB |
| Compile Time | 81.6 s (5 NEFFs) | 81.6 s (text) + ~30 s (vision) |
| Model Load Time | 12-14 s | 12-14 s |

### TTFT by Input Length (Multi-bucket CTE)

| Input Tokens | CTE Bucket Used | TTFT (P50) |
|-------------|-----------------|------------|
| ~115 | 512 | **38.2 ms** |
| ~484 | 512 | **38.3 ms** |
| ~943 | 1024 | **57.6 ms** |
| ~1861 | 2048 | **95.0 ms** |
| ~3175 | 4096 | **182.8 ms** |

Multi-bucket CTE provides **4.8x TTFT improvement** for short inputs vs single-bucket (38 ms vs 183 ms).

### Comparison with Qwen3-VL-8B (TP=4, trn2.3xlarge)

| Metric | Qwen2.5-VL-7B | Qwen3-VL-8B | Difference |
|--------|---------------|-------------|------------|
| TKG throughput | 86.4 tok/s | 76.8 tok/s | **+12.5%** |
| TTFT (short input) | 38.2 ms | ~200 ms | **~5x faster** |
| HBM per core | 4.2 GB | ~5 GB | 19% smaller |

### NKI Kernel Compatibility

**Text decoder:**

| Kernel | Status | Notes |
|--------|--------|-------|
| `qkv_kernel_enabled` | PASS | Fused RMSNorm+QKV ISA kernel, supports bias |
| `attn_kernel_enabled` | PASS | CTE flash attention NKI kernel |
| `attn_tkg_nki_kernel_enabled` | PASS | TKG NKI attention, 27.6 tok/s, exact match |
| `mlp_kernel_enabled` | FAIL | SBUF OOM: intermediate_size/TP = 4736 > 4096 |
| `attn_tkg_builtin_kernel_enabled` | FAIL | M-RoPE 3D rotary incompatible |
| `out_proj_kernel_enabled` | FAIL | hidden_size=3584 not divisible by 1024 |

**Vision encoder:**

| Kernel | Status | Notes |
|--------|--------|-------|
| `attn_kernel_enabled` | PASS | Flash attention for bidirectional vision |
| `qkv_kernel_enabled` | FAIL | Fused RMSNorm+QKV: eps type mismatch with vision RMSNorm |

## Usage

### Text-only Inference

```python
import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from transformers import AutoProcessor, GenerationConfig

from src.modeling_qwen2_5_vl import NeuronQwen2_5_VLForCausalLM, Qwen2_5_VLInferenceConfig

model_path = "/path/to/Qwen2.5-VL-7B-Instruct"
compiled_path = "/path/to/compiled"

# Configure
text_neuron_config = NeuronConfig(
    batch_size=1, ctx_batch_size=1, seq_len=4096,
    tp_degree=4, world_size=4,
    torch_dtype=torch.bfloat16,
    fused_qkv=True, qkv_kernel_enabled=True,
    attn_kernel_enabled=True, attn_tkg_nki_kernel_enabled=True,
    logical_neuron_cores=2, cc_pipeline_tiling_factor=2,
    cast_type="as-declared", save_sharded_checkpoint=True,
    enable_bucketing=True,  # Multi-bucket CTE for TTFT optimization
    context_encoding_buckets=[512, 1024, 2048, 4096],  # Min 512 for TKG NKI compat
    token_generation_buckets=[4096],  # Single TKG bucket at full seq_len
)
vision_neuron_config = NeuronConfig(
    batch_size=1, seq_len=4096,
    tp_degree=4, world_size=4,
    torch_dtype=torch.bfloat16,
    fused_qkv=True, enable_bucketing=True, buckets=[2],
    attn_kernel_enabled=True,  # Flash attention for bidirectional vision
    logical_neuron_cores=2, cc_pipeline_tiling_factor=2,
    cast_type="as-declared", save_sharded_checkpoint=True,
)
config = Qwen2_5_VLInferenceConfig(
    text_neuron_config=text_neuron_config,
    vision_neuron_config=vision_neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile (first time only)
model = NeuronQwen2_5_VLForCausalLM(model_path=model_path, config=config)
model.compile(compiled_path)

# Load compiled model
model.load(compiled_path)
adapter = HuggingFaceGenerationAdapter(model)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Generate text
messages = [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt")

gen_config = GenerationConfig(do_sample=False, eos_token_id=[151645], pad_token_id=151645)
sampling = prepare_sampling_params(batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0])

with torch.no_grad():
    output = adapter.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask,
        sampling_params=sampling, generation_config=gen_config, max_new_tokens=64,
    )
print(processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
# -> "The capital of France is Paris."
```

### Vision-Language Inference

```python
from PIL import Image

image = Image.open("photo.jpg")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

with torch.no_grad():
    output = adapter.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw,
        sampling_params=sampling, generation_config=gen_config, max_new_tokens=128,
    )
print(processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### vllm-neuron Serving

**Validated on both vllm-neuron 0.4.1 and 0.5.0 (6/6 API tests passed on each).**

#### vllm-neuron 0.5.0 (recommended)

```bash
# Install: git clone --branch release-0.5.0 https://github.com/vllm-project/vllm-neuron.git
#          cd vllm-neuron && pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .

# Apply patches (Qwen2.5-VL is not natively supported in 0.5.0):
python patch_vllm_050_qwen25vl.py --vllm-dir /path/to/vllm-neuron

# Serve:
NEURON_COMPILED_ARTIFACTS=/path/to/compiled \
PYTHONPATH=/path/to/qwen25vl:$PYTHONPATH \
vllm serve /path/to/Qwen2.5-VL-7B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --port 8000 \
  --no-enable-prefix-caching
```

#### vllm-neuron 0.4.1

```bash
NEURON_COMPILED_ARTIFACTS=/path/to/compiled \
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --port 8000 \
  --no-enable-prefix-caching
```

Both versions require 4 file patches (constants.py, model_loader.py, model_runner.py, NxDI constants.py). Patch scripts:
- **0.5.0**: `patch_vllm_050_qwen25vl.py` (supports `--vllm-dir`, `--nxdi-constants`, `--qwen25vl-dir`)
- **0.4.1**: `patch_vllm_qwen25vl.py`

## Compatibility Matrix

| Instance Type | TP=4 | TP=2 |
|--------------|------|------|
| trn2.3xlarge (LNC=2) | **Validated** (86.4 tok/s, 38ms TTFT) | Validated (28.1 tok/s) |
| trn2.48xlarge | Not tested | Not tested |
| trn1.32xlarge | Not tested | Not tested |

### Multi-Size Support

The same code works for all Qwen2.5-VL sizes (config-driven). Tested:

| Model | Instance | TP | TKG tok/s | Compile | Weights/Core | Notes |
|-------|----------|----|-----------|---------|-------------|-------|
| **7B** | trn2.3xlarge | 4 | 86.4 | 81.6s | 4.2 GB | Primary target |
| **3B** | trn2.3xlarge | 4 | 104.3 | 56.4s | 2.1 GB | `tie_word_embeddings=True` |
| **72B** | trn2.48xlarge | 32 | 44.3 | 508.4s | ~4.5 GB | 80 layers, 64 heads, 8 KV heads |

**3B notes**: The 3B model uses tied weights (`lm_head` = `embed_tokens`). The `update_state_dict_for_tied_weights` override handles this automatically. MLP NKI kernel compiles for 3B (`intermediate/TP=2752`) but is 13% slower than baseline -- not recommended for 3B.

**72B notes**: Requires trn2.48xlarge with TP=32 (146 GB BF16 weights). Compilation takes ~8.5 minutes. Vision encoder is identical to 3B/7B except `out_hidden_size=8192` and `intermediate_size=3456`. Steady-state TKG is 44.3 tok/s. VL inference works (33.8 tok/s with 672x672 image).

## Implementation Notes

### Vision Encoder Differences from Qwen2-VL

The Qwen2.5-VL vision encoder differs from Qwen2-VL in several ways:
1. **RMSNorm** instead of LayerNorm
2. **Gated SwiGLU MLP** with bias=True (unique -- neither Qwen2-VL nor Qwen3-VL has this in vision)
3. **Windowed attention** with `get_window_index()` partitioning tokens into 4x4 windows
4. **Hybrid attention**: 28 windowed layers + 4 global layers at positions [7, 15, 23, 31]

### Vision Buckets

Vision buckets represent the **number of images**, not sequence lengths. Default is `buckets=[2]` which provides headroom for vllm image preprocessing that may resize images beyond the compiled `pixels_per_image`.

### Key Files

| File | Description | Lines |
|------|-------------|-------|
| `modeling_qwen2_5_vl.py` | VL orchestrator (config, forward, state dict) | ~370 |
| `modeling_qwen2_5_vl_text.py` | Text decoder (attention, MLP, decoder layers) | ~370 |
| `modeling_qwen2_5_vl_vision.py` | Vision encoder (windowed attn, SwiGLU, merger) | ~760 |

## Example Checkpoints

* [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) -- Primary target (7B)
* [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) -- Validated (3B)
* [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) -- Validated (72B, requires trn2.48xlarge)

## Testing

```bash
# Set paths (adjust for your environment)
export QWEN25VL_MODEL_PATH=/mnt/models/Qwen2.5-VL-7B-Instruct
export QWEN25VL_COMPILED_PATH=/mnt/models/qwen25vl_compiled

# Run all tests
pytest test/integration/test_model.py -v --capture=tee-sys

# Run specific test
pytest test/integration/test_model.py::test_logit_validation -v
```

## Known Limitations

1. **Batch size > 1** requires the VLM batch>1 fix from branch [`fix/qwen3-vl-batch-size-gt1-v2`](https://github.com/jimburtoft/neuronx-distributed-inference/tree/fix/qwen3-vl-batch-size-gt1-v2) (3 patches to NxDI `image_to_text_model_wrapper.py`). Without it, batch>1 crashes.
2. **MLP kernel** is not compatible (intermediate_size/TP = 4736 exceeds SBUF threshold ~4096).
3. **Builtin TKG kernel** is not compatible with M-RoPE 3D rotary embeddings.
4. **Vision qkv_kernel** is not compatible (fused RMSNorm+QKV ISA kernel fails with vision encoder's RMSNorm epsilon type).
5. **Multi-bucket CTE minimum bucket**: Must be >= 512. Auto-generated small buckets (e.g., 128) cause TKG NKI kernel assertion failure (`sharded_S_ctx % 128 == 0` with LNC=2). Always set `context_encoding_buckets` explicitly.
6. **Video input** is not tested (the model architecturally supports it via temporal patches).

## Maintainer

Jim Burtoft, AWS

**Last Updated:** 2026-03-29
