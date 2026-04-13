# Contrib Model: HunyuanVideo-1.5

Text-to-video generation using the 8.33B HunyuanVideo-1.5 DiT transformer on AWS Trainium, with TP=4 parallelism and NKI flash attention.

## Model Information

- **HuggingFace ID:** `tencent/HunyuanVideo-1.5` (DiT + VAE), `Qwen/Qwen2.5-VL-7B-Instruct` (text encoder)
- **Model Type:** Multi-component diffusion pipeline (DiT transformer + 3D Causal VAE + LLM text encoder + byT5 glyph encoder)
- **Parameters:** ~8.33B (DiT), ~7B (LLM, CPU), ~0.3B (VAE), ~0.3B (byT5)
- **Architecture:** 54 double-stream DiT blocks, hidden_size=2048, 16 attention heads, head_dim=128, patch_size=[1,1,1], flow matching with Euler scheduler
- **License:** Tencent Hunyuan Community License

## Validation Results

**Validated:** 2026-04-12
**Instance:** trn2.3xlarge (LNC=2, 4 logical NeuronCores, 24 GB HBM per core pair)
**SDK:** Neuron SDK 2.28 (neuronx-cc 2.22, torch-neuronx 2.9.0)

### Benchmark Results

| Component | Latency | Method | Notes |
|-----------|---------|--------|-------|
| byT5 encode | 4.4ms | torch_neuronx.trace() | cos_sim=1.000 vs CPU |
| DiT per-step (no CFG) | 328ms | parallel_model_trace() TP=4 | NKI flash attention |
| DiT per-step (with CFG) | 653ms | 2x sequential B=1 passes | guidance_scale=6.0 |
| VAE decode (tiled) | 8.5s | torch_neuronx.trace() | 45 tiles x 177ms, cos_sim=0.997 |
| LLM encode (Qwen2.5-VL 7B) | 13.9s | CPU (bf16) | Cached negative embeddings save ~14s |
| **E2E with CFG (50 steps)** | **~55s** | | Photorealistic output |
| **E2E no CFG (50 steps)** | **~25s** | | |

### Accuracy Validation

| Component | Metric | Value |
|-----------|--------|-------|
| byT5 encoder | Cosine similarity vs CPU | 1.000 |
| byT5 mapper | Cosine similarity vs CPU | 0.9999 |
| VAE per-tile | Cosine similarity vs CPU | 0.999971 |
| VAE full decode | Cosine similarity vs CPU | 0.997 |
| E2E video | Visual quality | Photorealistic (verified across multiple prompts) |

## Usage

### 1. Prerequisites

```bash
# SSH into trn2.3xlarge instance
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Clone HunyuanVideo repo (needed for model definitions)
git clone https://github.com/Tencent/HunyuanVideo-1.5.git
# If GitHub repo is unavailable, download from HuggingFace:
# huggingface-cli download tencent/HunyuanVideo-1.5 --local-dir HunyuanVideo-1.5

# Install dependencies
pip install peft diffusers

# Download model weights (~50 GB total)
huggingface-cli download tencent/HunyuanVideo-1.5 --local-dir models/HunyuanVideo-1.5
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
huggingface-cli download google/byt5-small --local-dir models/byt5-small
```

### 2. Set Environment Variables

```bash
export HUNYUAN_REPO_DIR=./HunyuanVideo-1.5
export HUNYUAN_MODELS_DIR=./models
export HUNYUAN_COMPILED_DIR=./compiled
export HUNYUAN_ATTN_MODE=torch
export NEURON_RT_NUM_CORES=4
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export XLA_DISABLE_FUNCTIONALIZATION=1
export NEURON_FUSE_SOFTMAX=1
```

### 3. Compile Models (First Run Only)

```bash
# Compile DiT transformer (TP=4, ~2 minutes)
python src/recompile_dit_masked.py

# Compile VAE decoder (~12 minutes)
python src/compile_vae_neuron.py

# Trace byT5 encoder (~30 seconds)
python src/trace_byt5.py

# Pre-cache negative embeddings for CFG (~15 seconds)
python src/cache_neg_embeddings.py
```

### 4. Generate Video

```bash
python src/e2e_pipeline.py --steps 50 \
    --prompt "A golden retriever running in a sunny meadow"
```

Output: 5 frames at 480x848 resolution saved as PNG + MP4.

## Pipeline Architecture

```
                    [Prompt]
                       |
           +-----------+-----------+
           |                       |
    Qwen2.5-VL 7B (CPU)    byT5-small (Neuron)
    extract hidden_states[-3]   glyph features
           |                       |
           +--------> CPU Preprocessor <--------+
                   (patch embed, timestep embed,
                    token refiner, RoPE, reorder)
                           |
                    [6 tensor inputs]
                           |
                 DiT Backbone (Neuron TP=4)
                 54 double-stream blocks
                 NKI flash attention
                 328ms/step
                           |
                     x 50 Euler steps (CPU scheduler)
                           |
                    Unpatchify (CPU)
                           |
                 VAE Decode (Neuron, tiled)
                 45 tiles x 177ms = 8.5s
                           |
                    [480x848 Video Frames]
```

## Required Code Patches

HunyuanVideo-1.5 requires 5 patches for Neuron compatibility:

1. **Dtype mismatch** (`token_refiner.py:277`): `mask.float()` -> `mask.to(x.dtype)` (bf16 required)
2. **FlexAttention not supported**: Replace `flex_block_attn_func` with `F.scaled_dot_product_attention`
3. **NaN from SDPA mask expansion**: Use key-only mask `[B, 1, 1, L]` instead of `[B, 1, L, L]`
4. **byT5 output dtype**: Cast byT5 output to bf16 before passing to DiT
5. **RoPE rotate_half**: HunyuanVideo uses interleaved-pair rotation `(x[2i], x[2i+1])`, NOT LLaMA-style half-split `(x[i], x[i+D/2])`. The cos/sin from `get_nd_rotary_pos_embed(use_real=True)` use `repeat_interleave(2)`. Using the wrong `rotate_half` destroys spatial coherence.

Patches 1-4 are applied automatically by setting `HUNYUAN_ATTN_MODE=torch`. Patch 5 is implemented in `dit_wrapper.py`.

## Compatibility Matrix

| Instance Type | SDK 2.28 | SDK 2.29 |
|--------------|----------|----------|
| trn2.3xlarge | **VALIDATED** | Not tested |
| trn2.48xlarge | Not tested | Not tested |
| Inf2 | Not supported (TP=4 required) | Not supported |
| Trn1 | Not supported (TP=4 + LNC=2 required) | Not supported |

## Example Checkpoints

* [tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) -- DiT transformer + 3D Causal VAE
* [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) -- LLM text encoder (CPU)
* [google/byt5-small](https://huggingface.co/google/byt5-small) -- Glyph text encoder

## Testing Instructions

```bash
# Set environment variables (see Usage section above)

# Run component accuracy tests (byT5 + VAE):
pytest test/integration/test_model.py -k "ByT5 or VAE" -v --capture=tee-sys

# Run E2E pipeline tests (separately, to avoid NeuronCore contention):
pytest test/integration/test_model.py -k "E2E" -v --capture=tee-sys
```

**Note:** Run component and E2E tests separately. In-process Neuron model loads
(byT5/VAE) claim NeuronCores, which prevents E2E subprocess tests from
initializing their own Neuron runtime.

The test suite includes:
- **byT5 accuracy**: Cosine similarity >= 0.999 vs CPU reference (encoder + mapper)
- **VAE tile accuracy**: Output finiteness, shape validation, and spatial upsampling (16x)
- **E2E generation**: Produces 5 frames at 480x848 resolution (2-step quick test)
- **E2E performance**: 50-step generation completes within 120s wall-clock (no CFG, includes model loading)

## File Structure

```
HunyuanVideo-1.5/
├── README.md
├── src/
│   ├── __init__.py
│   ├── dit_tp_wrapper.py          # DiT TP=4 with NKI flash attention (699 LOC)
│   ├── dit_wrapper.py             # CPU preprocessor for tracing (366 LOC)
│   ├── e2e_pipeline.py            # Full E2E T2V pipeline (908 LOC)
│   ├── compile_vae_neuron.py      # VAE compilation with monkey-patches
│   ├── tiled_vae_decode.py        # Tiled VAE decode runtime
│   ├── trace_byt5.py              # byT5 encoder tracing
│   ├── cache_neg_embeddings.py    # Negative embedding pre-cache
│   └── recompile_dit_masked.py    # DiT compilation script
├── test/
│   └── integration/
│       └── test_model.py          # Accuracy + performance tests
├── samples/
│   └── neuron/                    # Sample output frames (trn2.3xlarge)
│       ├── frame_0000.png
│       ├── frame_0002.png
│       └── frame_0004.png
└── examples/
    └── generate_video.py          # Simplified generation entry point
```

## Technical Details

### DiT Forward Signature (6 tensor inputs, all preprocessed on CPU)

| # | Input | Shape | Description |
|---|-------|-------|-------------|
| 1 | img | [B, L_img, 2048] | Patch-embedded video latents |
| 2 | txt | [B, L_txt, 2048] | Refined + reordered text tokens |
| 3 | vec | [B, 2048] | Timestep + text pool embedding |
| 4 | txt_mask | [B, L_txt] | Text attention mask |
| 5 | freqs_cos | [L_img, 128] | RoPE cosine (3D positional) |
| 6 | freqs_sin | [L_img, 128] | RoPE sine (3D positional) |

### NKI Flash Attention

Uses `attention_isa_kernel` from `neuronxcc.nki._private_kernels.attention` for O(n) memory attention. Sequence lengths are padded to multiples of 2048 (NKI_SEQ_TILE_SHARDED) for LNC=2 compatibility.

### CFG (Classifier-Free Guidance)

The 480p_t2v model uses traditional CFG with `guidance_scale=6.0`. Two sequential B=1 forward passes per step (unconditional + conditional). Pre-caching the negative (empty string) embeddings saves ~14s per run.

### Compiler Flags

```
--model-type=transformer -O1 --auto-cast=none --enable-saturate-infinity
--enable-mixed-precision-accumulation
```

### VAE Monkey-Patches

Three patches are required for the 3D Causal VAE to compile on Neuron:

1. **CausalConv3d**: Replace `F.pad(mode='replicate')` with `torch.cat` (temporal) + `F.pad(mode='constant')` (spatial)
2. **swish**: `inplace=True` -> `inplace=False` (Neuron rejects inplace SiLU)
3. **prepare_causal_attention_mask**: Python loop -> vectorized `torch.arange` + broadcasting

## Known Issues

1. **LLM runs on CPU**: Qwen2.5-VL 7B is too large for single-core Neuron trace. Requires TP=2+ which needs chunked loading. Current approach: CPU bf16 (~14s encode time).

2. **VAE `replication_pad3d`**: The 3D Causal VAE uses `F.pad(mode='replicate')` which triggers compiler assertion `NCC_IDLO901`. Workaround: monkey-patch with constant padding.

3. **B=2 batched CFG is slower**: Compiling DiT at B=2 to batch uncond+cond in a single call results in 4.2x slower execution due to HBM bandwidth saturation. Sequential B=1 is optimal.

4. **First-run compilation**: DiT ~2 min, VAE ~12 min, byT5 ~30s. NEFF cache makes subsequent runs instant.

## Maintainer

Jim Burtoft
