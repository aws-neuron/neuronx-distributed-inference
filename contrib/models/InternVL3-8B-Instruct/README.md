# Contrib Model: OpenGVLab InternVL3-8B-Instruct

NeuronX Distributed Inference implementation for [OpenGVLab/InternVL3-8B-Instruct](https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct), a vision-language model with an InternViT-300M vision encoder, pixel shuffle MLP projector, and Qwen2.5-7B text backbone.

## Model Information

- **HuggingFace ID:** [`OpenGVLab/InternVL3-8B-Instruct`](https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct)
- **Model Type:** LLaVA-style VLM (ViT + MLP projector + LLM)
- **Parameters:** ~8B total (300M vision + 7B text + projector)
- **License:** MIT (Apache-2.0 for Qwen2.5 component)

## Architecture Details

### Text Backbone: Qwen2.5-7B

| Spec | Value |
|------|-------|
| **Layers** | 28 |
| **Hidden Size** | 3584 |
| **Head Dim** | 128 |
| **Attention Heads** | 28 |
| **KV Heads** | 4 (GQA, 7:1 ratio) |
| **Intermediate Size** | 18944 |
| **Vocabulary Size** | 151,674 |
| **Max Position Embeddings** | 32,768 |
| **Position Encoding** | RoPE (theta=1e6) |
| **Normalization** | RMSNorm |
| **Activation** | SiLU |
| **Tied Embeddings** | No |
| **QKV Bias** | Yes (Q, K, V); No (O) |

### Vision Encoder: InternViT-300M-448px-V2.5

| Spec | Value |
|------|-------|
| **Layers** | 24 |
| **Hidden Size** | 1024 |
| **Head Dim** | 64 |
| **Attention Heads** | 16 |
| **Patch Size** | 14 |
| **Image Size** | 448x448 |
| **Normalization** | LayerNorm |
| **Activation** | GELU |
| **Layer Scaling** | Yes |
| **Visual Tokens per Tile** | 256 (after pixel shuffle) |

### Projector: Pixel Shuffle + 2-Layer MLP

1. Pixel shuffle (0.5x downsample): 1024 patches -> 256 patches, 1024d -> 4096d
2. LayerNorm(4096) -> Linear(4096, 3584) -> GELU -> Linear(3584, 3584)
3. Output: [batch, 256, 3584] per tile

## Validation Results

**Validated:** 2026-04-24
**Configuration:** trn2.3xlarge, LNC=2, TP=4, batch_size=1, BF16

### Accuracy

| Test | Status | Result |
|------|--------|--------|
| CTE Logit Comparison | PASS | cosine=0.9984, top1=MATCH, top5=5/5 |
| Text-only Generation | PASS | "The capital of France is Paris." |
| Multimodal Generation | PASS | Coherent image descriptions |
| State Reset | PASS | Deterministic across runs |

### Performance (batch_size=1)

#### TP=4 (Recommended)

| seq_len | TTFT (ms) | TKG (tok/s) | TKG (ms/tok) | MM TTFT (ms) |
|---------|-----------|-------------|--------------|--------------|
| 2048 | 138 | **75.1** | 13.3 | 140 |
| 4096 | 230 | **58.9** | 17.0 | 248 |
| 8192 | 482 | **40.0** | 25.0 | 484 |
| 16384 | 1019 | **23.6** | 42.4 | 1008 |
| 32768 | 2438 | **11.4** | 87.7 | 2366 |

#### TP=2

| seq_len | TTFT (ms) | TKG (tok/s) | TKG (ms/tok) |
|---------|-----------|-------------|--------------|
| 2048 | 284 | **26.2** | 38.2 |

TP=4 is 2.05x faster TTFT and 2.87x faster TKG than TP=2.

Vision encoder latency: ~35 ms/tile (448x448).

## Usage

```python
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from modeling_internvl3 import InternVL3InferenceConfig, NeuronInternVL3ForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig

model_path = "/path/to/InternVL3-8B-Instruct/"
compiled_path = "/path/to/compiled/model/"

# Text NeuronConfig
text_neuron_config = NeuronConfig(
    tp_degree=4,
    max_batch_size=1,
    seq_len=4096,  # Recommended for single-image use
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=None,
    save_sharded_checkpoint=True,
)

# Vision NeuronConfig (must match text TP degree)
vision_neuron_config = NeuronConfig(
    tp_degree=4,
    max_batch_size=1,
    seq_len=256,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=None,
    buckets=[1],
    fused_qkv=True,
    save_sharded_checkpoint=True,
)

config = InternVL3InferenceConfig.from_pretrained(
    model_path,
    text_neuron_config=text_neuron_config,
    vision_neuron_config=vision_neuron_config,
)

# Create, compile, load
model = NeuronInternVL3ForCausalLM(model_path, config=config)
model.compile(compiled_path)
model.load(compiled_path)

# Text-only inference
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
input_ids = tokenizer("What is the capital of France?", return_tensors="pt").input_ids
position_ids = torch.arange(input_ids.shape[-1], dtype=torch.int32).unsqueeze(0)
seq_ids = torch.zeros(1, dtype=torch.int32)

outputs = model(input_ids=input_ids, position_ids=position_ids, seq_ids=seq_ids)

# Multimodal inference (with HuggingFaceGenerationAdapter)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
adapter = HuggingFaceGenerationAdapter(model)

# Build input with image tokens
IMG_CONTEXT_ID = 151667  # <IMG_CONTEXT>
IMG_START_ID = 151665    # <img>
IMG_END_ID = 151666      # </img>

text_ids = tokenizer("Describe this image:", return_tensors="pt").input_ids[0]
img_tokens = torch.full((256,), IMG_CONTEXT_ID, dtype=torch.long)
input_ids = torch.cat([
    text_ids,
    torch.tensor([IMG_START_ID]), img_tokens, torch.tensor([IMG_END_ID]),
]).unsqueeze(0)

pixel_values = torch.randn(1, 3, 448, 448)  # Replace with actual image

output = adapter.generate(
    input_ids=input_ids,
    attention_mask=torch.ones_like(input_ids),
    pixel_values=pixel_values,
    max_new_tokens=64,
    do_sample=False,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Recommended Configurations

| Use Case | seq_len | Notes |
|----------|---------|-------|
| Single image + short prompt | 2048 | Best throughput (75 tok/s) |
| Single image + conversation | 4096 | Good balance (59 tok/s) |
| Multi-image (2-3 images) | 8192 | Handles 3 full-res images |
| Long document understanding | 16384+ | Higher latency |

## Compatibility Matrix

| Instance/SDK | 2.29 | 2.28 and earlier |
|--------------|------|------------------|
| Trn2 (LNC=2, TP=4) | **PASS (recommended)** | Not tested |
| Trn2 (LNC=2, TP=2) | PASS (2.87x slower TKG) | Not tested |
| Trn1 | Not supported (NxDI 0.9.x is trn2-only) |
| Inf2 | Not supported |

## Known Limitations

1. **Batch size:** Only batch_size=1 supported. Batch>1 has a sampling_params shape issue with `NeuronBaseForImageToText`. Use vLLM for concurrent requests.
2. **V2PE:** Variable Visual Position Encoding is not implemented (not in HF transformers code). Standard position IDs are used. Accuracy is not affected for single-image use.
3. **Dynamic resolution:** Tile splitting is CPU-side. Current implementation supports 1 tile per image. Multi-tile support requires vision encoder bucket expansion.
4. **`trust_remote_code`:** Required for tokenizer/processor loading (no `-hf` variant exists).
5. **NKI kernels:** Opt-in NKI kernels are incompatible with the Qwen2.5-7B text backbone dimensions (`intermediate_size=18944`, `num_kv_heads=4`). This matches the NxDI team's own Qwen2-VL test config, which uses the same backbone and explicitly disables `mlp_kernel` and `attn_block_tkg`. See [NKI Kernel Details](#nki-kernel-details) below.

## NKI Kernel Details

All opt-in NKI kernel flags were tested at TP=2 and TP=4 on trn2.3xlarge LNC=2 (SDK 2.29). None improve performance. The failures are caused by Qwen2.5-7B's non-standard dimensions, not by model integration code.

**Root cause:** The NxDI test suite for Qwen2-VL (identical text backbone: `hidden=3584`, `heads=28`, `kv_heads=4`, `intermediate=18944`) sets `mlp_kernel_enabled=False` and `attn_block_tkg_nki_kernel_enabled=False`, only enabling `qkv_kernel_enabled=True`. Our testing confirms the same.

| Kernel | TP=4 | TP=2 | Constraint |
|--------|------|------|------------|
| `qkv_kernel_enabled` | Compiles, -4% TKG | Compiles, no gain | Only kernel compatible with these dimensions |
| `mlp_kernel_enabled` | Compiles, +10% TTFT | **FAIL**: floordiv-by-zero | `intermediate/TP` must satisfy NKI tiling divisibility. 18944/4=4736 is marginal; 18944/2=9472 fails. |
| `attn_block_tkg_nki_kernel_enabled` | **FAIL**: NCC_ISTP902 | **FAIL**: NCC_INKI016 | Requires exactly 1 KV head per rank. TP=2 gives 2 heads/rank. TP=4 gives 1 but hits compiler memory pressure issue. |

**Recommendation:** Use default config with no NKI kernel flags. The auto-enabled kernels (CTE flash attention for seq_len>=256, NKI argmax, NKI cumsum) are already active and do not require opt-in.

## File Structure

```
InternVL3-8B-Instruct/
  src/
    __init__.py
    modeling_internvl3.py          # Top-level VLM (NeuronBaseForImageToText)
    modeling_internvl3_text.py     # Text model (NeuronBaseModel)
    modeling_internvl3_vision.py   # Vision encoder (torch_neuronx.trace)
  compile_internvl3_vlm.py        # Compile + smoke test script
  nki_benchmark.py                # NKI kernel benchmark script (TP=4)
  tp2_nki_sweep.py                # TP=2 NKI kernel sweep script
  accuracy_test.py                # Accuracy validation script
  validate_cte.py                 # CTE validation (logit comparison)
  validate_tkg.py                 # TKG validation (generation quality)
  scale_test.py                   # Scaling benchmarks
  README.md                       # This file
```

## Weight Mapping

| HuggingFace Key | NxDI Key |
|-----------------|----------|
| `language_model.model.layers.{i}.*` | `layers.{i}.*` |
| `language_model.model.embed_tokens.weight` | `embed_tokens.weight` |
| `language_model.model.norm.weight` | `norm.weight` |
| `language_model.lm_head.weight` | `lm_head.weight` |
| `vision_model.*` | Loaded separately into vision encoder |
| `mlp1.*` | Loaded into vision projector |

## Maintainer

Agent intern-vl (Opencode)
