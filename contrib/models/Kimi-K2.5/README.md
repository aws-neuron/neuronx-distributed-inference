# Contrib Model: Kimi-K2.5 (Multimodal)

NeuronX Distributed Inference implementation of Moonshot AI's Kimi-K2.5 — a native multimodal agentic model with MoonViT vision encoder.

## Model Information

- **HuggingFace ID:** `moonshotai/Kimi-K2.5`
- **Model Type:** Multimodal (image + text) Mixture of Experts decoder
- **Architecture:** Kimi-K2 text decoder + MoonViT-400M vision encoder
- **License:** Check HuggingFace model card

## Architecture Details

### Text Decoder (same as Kimi-K2)

| Parameter | Value |
|-----------|-------|
| Total parameters | ~1,017B |
| Active parameters per token | ~32B |
| Hidden size | 7168 |
| Attention heads | 128 |
| Layers | 61 |
| Vocabulary size | 163840 |
| Routed experts | 384 (8 active per token) |
| Shared experts | 1 per MoE layer |
| Dense layers | 1 (layer 0) |
| Attention type | Multi-Latent Attention (MLA) |
| Router activation | Sigmoid with `e_score_correction_bias` |

### MoonViT Vision Encoder

| Parameter | Value |
|-----------|-------|
| Architecture | ViT with 2D RoPE |
| Layers | 27 |
| Hidden size | 1152 |
| Attention heads | 16 |
| MLP hidden | 4304 |
| Parameters | ~400M (466M with projector) |
| Patch size | 14×14 |
| Patch merging | 2×2 (4 patches → 1 token) |
| Projector | PatchMergerMLP → 7168 |

### Vision-Text Fusion

- **Method:** scatter_by_index_put (Llama4/Pixtral pattern)
- **Mechanism:** Vision embeddings replace text embeddings at placeholder positions via `index_put_`
- **Integration point:** `NeuronBaseModel.get_model_output()` → `encode_vision_to_input()`

### K2.5 Weight Format

K2.5 uses a different weight format than K2:
- **Expert weights:** INT4 compressed-tensors (pack-quantized, group_size=32, symmetric)
  → Dequantized to BF16 → Re-quantized to FP8 per-channel for Neuron
- **Non-expert weights:** BF16 (attention, shared experts, norms, embeddings, lm_head)
- **Key prefix:** `language_model.model.` (stripped to match K2 format)
- **Vision keys:** `vision_tower.*`, `mm_projector.*` (filtered for text-only model)

## Validation Results

**Validated:** 2026-04-25 (SDK 2.29)
**Configuration:** TP=64, EP=1, LNC=2, batch_size=1, seq_len=512, FP8 per-channel

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles and loads on trn2.48xlarge |
| Multimodal Generation | PASS | Generates coherent image description |
| Vision A/B Test | PASS | Real vision ≠ zero vision (max logit diff: 15.2) |
| Coherence | PASS | No repetition, natural text |
| Throughput | PASS | 45.9 tok/s at BS=1 (LNC=2) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **TKG throughput** | **45.9 tok/s** |
| **TPOT (per-token latency)** | 21.4 ms |
| **E2E throughput** | 26.3 tok/s (128 tokens) |
| **CTE latency** | 2,094 ms |
| **CTE vision overhead** | 1.1 ms (negligible) |
| **MoonViT latency** | 35.5 ms |
| **TTFT** | ~2,130 ms (CTE + MoonViT) |
| **Model load time** | ~79 min (weight sharding) |
| **Compile time** | ~10 min (CTE ~5 min, TKG ~5 min) |

### Benchmark Details (10 iterations, 3 warmup)

| Component | Mean | p50 | Std |
|-----------|------|-----|-----|
| MoonViT vision encoder | 35.5 ms | 35.5 ms | 0.1 ms |
| CTE (with vision) | 2,094.6 ms | 2,094.5 ms | 0.4 ms |
| CTE (text-only) | 2,093.5 ms | 2,093.5 ms | 0.3 ms |
| TKG per-token | 21.4 ms | 21.3 ms | 0.4 ms |
| End-to-end (CTE+TKG) | 4,863.2 ms | 4,864.7 ms | 12.1 ms |

### Accuracy Validation (vs GPU reference)

| Metric | Value |
|--------|-------|
| Vision encoder cosine similarity | 0.9995 |
| Token match rate (vs vLLM H100) | 3.1% (expected for 1T MoE) |
| Semantic quality | Both describe same image correctly |

Token-level divergence is expected: FP8 vs BF16 quantization + 384-expert MoE routing causes different expert selections that cascade through autoregressive generation. Both outputs are semantically equivalent and correctly describe the input image.

## Usage

```python
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("src")))

from modeling_kimi_k2 import NeuronKimiK2ForCausalLM, NeuronKimiK2Model
from modeling_kimi_k25 import (
    apply_k25_patches,
    apply_k25_checkpoint_patch,
    build_k25_config,
    create_text_only_model_dir,
    BOS_TOKEN_ID, IM_USER_TOKEN_ID, IM_END_TOKEN_ID,
    IM_ASSISTANT_TOKEN_ID, MEDIA_PLACEHOLDER_TOKEN_ID,
)

model_path = "/path/to/Kimi-K2.5"
text_model_dir = "/path/to/Kimi-K2.5-text"
compiled_path = "/path/to/compiled"
vision_emb_path = "/path/to/moonvit_448_real_embeddings.pt"

# 1. Create text-only model directory
create_text_only_model_dir(model_path, text_model_dir)

# 2. Apply patches BEFORE model init
apply_k25_patches(NeuronKimiK2ForCausalLM, NeuronKimiK2Model, ep_degree=1)

# 3. Build config
config = build_k25_config(text_model_dir, tp_degree=64, ep_degree=1, lnc=2)

# 4. Initialize, patch, compile, load
model = NeuronKimiK2ForCausalLM(text_model_dir, config=config)
apply_k25_checkpoint_patch(model)
model.compile(compiled_path)  # ~10 min
model.load(compiled_path)     # ~79 min

# 5. Load pre-computed vision embeddings
vision_emb = torch.load(vision_emb_path, map_location="cpu").to(torch.bfloat16)

# 6. Build multimodal prompt
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

n_vision = vision_emb.shape[0]  # 256
text_ids = tokenizer.encode("Describe this image in detail.")
input_ids_list = (
    [BOS_TOKEN_ID, IM_USER_TOKEN_ID]
    + [MEDIA_PLACEHOLDER_TOKEN_ID] * n_vision
    + text_ids
    + [IM_END_TOKEN_ID, IM_ASSISTANT_TOKEN_ID]
)

# 7. Prepare vision tensors (must be [1, seq_len, ...])
seq_len = 512
ve = torch.zeros(1, seq_len, 7168, dtype=torch.bfloat16)
vm = torch.full((1, seq_len, 1), fill_value=seq_len - 1, dtype=torch.int32)
for i in range(n_vision):
    ve[0, i] = vision_emb[i]
    vm[0, i, 0] = i + 2  # positions after BOS + im_user

# 8. Run inference (see test/integration/test_model.py for full generation loop)
```

**Important:** Run with environment variables:
```bash
NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 python your_script.py
```

## Pre-computing MoonViT Embeddings

All 64 Neuron cores are consumed by the text decoder (TP=64). MoonViT must be run **before** loading the text decoder:

```python
import torch
import torch_neuronx
from moonvit import NeuronMoonViTWrapper, load_vision_weights, precompute_rope_real
from modeling_kimi_k25 import preprocess_image, precompute_rope_tables
from PIL import Image

# Create and trace MoonViT
model = NeuronMoonViTWrapper(patch_h=32, patch_w=32)  # 448x448
model = load_vision_weights(model, "/path/to/Kimi-K2.5", 32, 32)
model = model.to(torch.bfloat16).eval()

# Precompute RoPE
cos_table, sin_table = precompute_rope_real(72, 512, 512)
rope_cos = cos_table[:32, :32].reshape(-1, 36).to(torch.bfloat16)
rope_sin = sin_table[:32, :32].reshape(-1, 36).to(torch.bfloat16)

# Preprocess image
image = Image.open("test_image.jpg")
pixel_values, grid_thw, n_merged = preprocess_image(image, 448)

# Trace on Neuron
model_neuron = torch_neuronx.trace(
    model, (pixel_values, rope_cos, rope_sin),
    compiler_args=["--model-type", "transformer", "--auto-cast", "none"],
)
torch.jit.save(model_neuron, "moonvit_448.pt")

# Pre-compute embeddings
with torch.no_grad():
    vision_output = model_neuron(pixel_values, rope_cos, rope_sin)
torch.save(vision_output.to(torch.bfloat16), "moonvit_448_real_embeddings.pt")
```

## Compatibility Matrix

| Instance / SDK Version | 2.29 | 2.28 | 2.27 and earlier |
|------------------------|------|------|------------------|
| trn2.48xlarge (LNC=2, TP=64, EP=1) | **Working (45.9 tok/s)** | Not tested | Not tested |
| trn2.48xlarge (LNC=2, TP=32, EP=2) | Not recommended* | Not tested | Not tested |
| trn2.3xlarge | Not supported (needs TP=64) | Not supported | Not supported |
| inf2 | Not supported | Not supported | Not supported |

\*EP=2 has known blockwise CTE kernel regression in SDK 2.29 (see K2 contrib notes).

## Testing

Run integration tests on a trn2.48xlarge:

```bash
# Activate Neuron venv (SDK 2.29)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install tiktoken  # Required for K2.5 tokenizer

# Run tests
NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
    pytest test/integration/test_model.py -v --capture=tee-sys
```

Or run standalone:

```bash
NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
    python test/integration/test_model.py
```

**Note:** Compilation takes ~10 min, model loading takes ~79 min (dominated by weight sharding across 64 ranks). The first run will compile NEFFs; subsequent runs reuse cached NEFFs.

## Prerequisites

1. **Model weights:** Download from HuggingFace (~555 GB):
   ```bash
   huggingface-cli download moonshotai/Kimi-K2.5 \
       --local-dir /mnt/nvme/models/Kimi-K2.5
   ```

2. **Pre-computed vision embeddings:** Trace MoonViT and pre-compute embeddings before loading the text decoder (see "Pre-computing MoonViT Embeddings" above).

3. **Storage:** At least 600 GB for model weights + 50 GB for compiled NEFFs. NVMe RAID recommended for faster loading.

4. **Host RAM:** At least 2 TB (safetensors mmap can use significant virtual memory during weight sharding).

5. **tiktoken package:** Required for K2.5 tokenizer:
   ```bash
   pip install tiktoken
   ```

## Key Implementation Details

### Vision Embedding Fusion (`encode_vision_to_input`)

Uses `scatter_by_index_put` following the Llama4/Pixtral pattern:
- `vision_embeddings`: `[BS, seq_len, 7168]` — real vision data packed at front, zeros padding rest
- `vision_mask`: `[BS, seq_len, 1]` — integer position indices, `fill_value=seq_len-1` for padding
- The function clones `inputs_embeds` and uses `index_put_` to scatter vision embeddings at their target positions

### CRITICAL: `pad_inputs()` Silent Replacement

`ModelWrapper.pad_inputs()` (model_wrapper.py:791-809) silently replaces vision tensors with dummy zeros when their sequence dimension doesn't match the padded sequence length. Vision tensors **must** be provided at `[BS, seq_len, ...]` size to avoid being replaced.

### ImageToTextModelWrapper Tracing

The standard `ImageToTextModelWrapper` provides zero-filled vision inputs for tracing, which the Neuron XLA compiler may optimize away. `K25ImageToTextModelWrapper` overrides `input_generator()` to use ones-like inputs, matching NxDI's proven `test_scatter.py` pattern.

### MoonViT on Neuron

MoonViT uses real-number decomposition of 2D complex RoPE and eager attention (no flash_attn) for Neuron compatibility. The 400M parameter encoder processes a 448×448 image in 35.5ms on a single Neuron core.

## Example Checkpoints

* [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)

## Known Limitations

- **Pre-computed vision required:** All 64 Neuron cores are used by the text decoder. MoonViT cannot run after text decoder loading. Pre-compute embeddings first.

- **On-device sampling (ODS):** Disabled. The model returns logits `[BS, 1, 163840]`, not token indices, due to a known ODS compatibility issue.

- **Single image per inference:** Fixed to one 448×448 image. Variable resolution requires retracing MoonViT.

- **Batching:** BS=1 only. Same bandwidth-bound limitation as K2 (MoE expert weight loads dominate).

- **seq_len=512:** Maximum 512 tokens total (256 vision + text + generation). Larger seq_len causes HBM OOM with TP=64 EP=1. See HBM Memory Bottleneck section.

### HBM Memory Bottleneck

| seq_len | TKG scratchpad | Total per HBM bank | Headroom (of 23.363 GB) |
|---------|---------------|-------------------|------------------------|
| 128 | ~3.0 GB | ~20.9 GB | ~2.5 GB |
| 512 | ~4.1 GB | ~21.9 GB | ~1.4 GB |
| 1024 | ~5.5 GB | ~23.4 GB | ~0 GB |

## Relationship to Kimi-K2

This is an extension of the Kimi-K2 text-only NxDI contrib (PR #131). Key differences:

| Aspect | K2 | K2.5 |
|--------|-----|------|
| Modality | Text-only | Multimodal (image + text) |
| Config | TP=32, EP=2 | TP=64, EP=1 |
| Quantization | Blockwise FP8 (native) | INT4 → BF16 → FP8 per-channel |
| Weight format | K2 safetensors | K2.5 compressed-tensors |
| TKG throughput | 6.0 tok/s | 45.9 tok/s |
| Vision encoder | N/A | MoonViT-400M (35.5 ms) |

The 7.6x throughput improvement (6.0 → 45.9 tok/s) comes from TP=64 EP=1 (vs TP=32 EP=2), which eliminates inter-EP communication overhead and gives each core more bandwidth.

## Maintainer

Annapurna Labs

**Last Updated:** 2026-04-25
