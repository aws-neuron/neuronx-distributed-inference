# Contrib Model: LTX-2.3

LTX-2.3 22B parameter DiT audio-video diffusion transformer running on AWS Trainium 2 via NxD Inference. Generates synchronized video + audio from text prompts, with optional image-to-video conditioning.

## Model Information

- **HuggingFace ID:** [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3)
- **Model Type:** DiT (Diffusion Transformer) for joint audio-video generation (text-to-video and image-to-video)
- **Parameters:** 22B (BF16) — 48 transformer blocks, 32 heads, 4096 video dim, 2048 audio dim
- **Architecture:** Bidirectional audio-video cross-attention, gated attention, QK-RMSNorm, split RoPE, flow matching
- **License:** See HuggingFace model card
- **Framework:** Native [`ltx-core`](https://github.com/Lightricks/LTX-2) (not Diffusers)

## Validation Results

**Validated:** 2026-03-09
**Instance:** trn2.3xlarge (TP=4, LNC=2, 4 logical NeuronCores)
**SDK:** Neuron SDK 2.28, PyTorch 2.9, Deep Learning AMI Neuron (Ubuntu 24.04) 20260227

### Accuracy Validation

| Component | Metric | Value | Notes |
|-----------|--------|-------|-------|
| Single forward pass (video) | Cosine similarity | 0.999947 | sigma=1.0, noise input |
| Single forward pass (audio) | Cosine similarity | 0.999867 | sigma=1.0, noise input |
| 8-step denoised latent (real text) | Cosine similarity | 0.972 | Same Gemma 3 text, same seed |

All accuracy numbers measured against CPU reference (unsharded BF16, native ltx-core model).

### Benchmark Results

| Stage | Time | Notes |
|-------|------|-------|
| CPU component loading | 24.9s | LTXModel, VideoDecoder, AudioDecoder, Vocoder, EmbeddingsProcessor |
| Gemma3 encoder loading (4 ranks) | 362s | Pre-sharded weights, NEFF rehydration (cold start) |
| Text encoding — Neuron Gemma3 (warm) | 0.6s | Encoder forward pass only (644ms), with tensorizer-optimized compiler flags |
| Text encoding — Neuron Gemma3 (E2E warm) | ~1.3s | Including tokenization + post-processing |
| Text encoding — Neuron Gemma3 (warmup) | 16.3s | First forward pass on NeuronCores |
| Text encoding — CPU fallback | ~162s | Without Neuron compilation |
| Gemma3 unload | 2.6s | Explicit NRT resource cleanup |
| Neuron backbone weight loading | 70s | Pre-sharded weights, 4×9.3 GB rank files |
| Neuron backbone NEFF loading | 84s | Compiled model loaded onto 4 NeuronCores |
| **Denoising step (warm, steps 2-8)** | **0.3s** | **Steady-state per-step latency** |
| Denoising step (cold, step 1) | 174.6s | Includes Neuron device initialization |
| DiT warmup (1st inference) | 138.5s | Forces NRT to load NEFF onto cores |
| Total denoising (8 steps) | 176.9s | 22.1s/step average (dominated by cold start) |
| Video decode (CPU) | 7.2s | 25 frames @ 384×512 |
| Video decode (CPU, upscaled) | 33.3s | 49 frames @ 768×1024 |
| Audio decode (CPU) | 2.5s | Stereo WAV, 48kHz |
| Spatial upscaler (CPU) | 0.7s | 498M params, (1,128,4,12,16) → (1,128,4,24,32) |
| Temporal upscaler (CPU) | 0.4s | 131M params, (1,128,4,24,32) → (1,128,7,24,32) |

Note: Gemma3 and DiT backbone share the same 4 NeuronCores and are loaded sequentially. The cold start latency (NEFF rehydration) is a one-time cost when the compiled model is first loaded onto a fresh instance. Subsequent generations reuse the loaded model.

### Per-Step Performance Detail

Warm denoising steps (2-8) at 384×512, 25 frames, with all CPU optimizations applied:

| Component | Time | % of Step |
|-----------|------|-----------|
| CPU Preprocess | 33.1 ms | 11.9% |
| Neuron Backbone | 244.1 ms | 87.4% |
| Euler Step | 2.1 ms | 0.7% |
| **Total per step** | **279.3 ms** | 100% |

Two CPU preprocessing optimizations are applied:

1. **AdaLN deduplication**: Computes the timestep embedding MLP once per unique sigma value instead of per-token (768 tokens in T2V mode). Reduces AdaLN time from 54ms to 7ms.
2. **Step-invariant caching**: RoPE embeddings, context projection (caption_projection Linear), and attention masks are constant across denoising steps. Computed once on step 1 and reused for steps 2-8. Saves ~57ms on the first optimization pass (context projection dominates at ~55ms), with combined CPU preprocessing reduced from 96.5ms (baseline) to 39.0ms (60% reduction).

Overall per-step improvement vs unoptimized baseline: 330ms to 279.3ms (15.4% reduction).

### Two-Stage Pipeline Benchmarks

| Stage | Time | Notes |
|-------|------|-------|
| Stage 1 backbone loading (4 ranks) | 83.6s | Pre-sharded weights from full-res sharding |
| Stage 1 warmup (1st inference) | 107.9s | Forces NRT to load half-res NEFF |
| **S1 denoising step (warm, steps 2-8)** | **0.3s** | **192×256 latent, VIDEO_SEQ=192** |
| S1 denoising step (cold, step 1) | 142.1s | Includes Neuron device initialization |
| S1 total (8 steps) | 143.9s | 18.0s/step average |
| Spatial upscaler loading (CPU) | 8.1s | 498M params |
| Spatial upsample (CPU) | 0.3s | (1,128,4,6,8) → (1,128,4,12,16) |
| Stage 2 backbone loading (4 ranks) | 17.2s | Same pre-sharded weights, cached |
| Stage 2 warmup (1st inference) | 110.6s | Forces NRT to load full-res NEFF |
| **S2 denoising step (warm, steps 2-3)** | **0.3s** | **384×512 latent, VIDEO_SEQ=768** |
| S2 denoising step (cold, step 1) | 143.0s | Includes Neuron device initialization |
| S2 total (3 steps) | 143.7s | 47.9s/step average |
| **Combined denoising (S1+S2)** | **287.6s** | **2.7s actual compute after warmup** |

Two-stage mode generates at half resolution (192×256) with 8 denoising steps, spatially upscales x2, then refines at full resolution (384×512) with 3 additional steps. The same backbone weights are used for both stages — only the compiled shapes differ.

### trn2.48xlarge Full Benchmark (Two-Phase Pipeline)

**Validated:** 2026-03-16
**Instance:** trn2.48xlarge (TP=4/16, LNC=2, 32 logical NeuronCores)
**SDK:** Neuron SDK 2.27, PyTorch 2.9, Deep Learning AMI Neuron (Ubuntu 24.04) 20260126
**Resolution:** 512×768 → 1024×1536, 121 frames, Image-to-Video

Two-phase execution is required because the NRT communicator cannot change TP degree mid-process:
- **Phase 1** (TP=4): Gemma3 text encoding + S1 denoising (8 steps at 512×768)
- **Phase 2** (TP=16): Spatial upsample → S2 denoising (3 steps at 1024×1536) → Neuron VAE decode

| Component | Time | Notes |
|-----------|------|-------|
| **S1 denoising (8 steps, 512×768)** | **14.4s** | **1.8s/step, TP=4** |
| **S2 denoising (3 steps, 1024×1536)** | **21.7s** | **7.2s/step, TP=16** |
| **Combined denoising** | **36.0s** | S1 + S2 |
| **VAE decode (Neuron, tiled)** | **23.5s** | 33 tiles @ 610ms/tile (TP=4), 3.4s first-tile warmup |
| VAE decode (CPU, reference) | 78.7s | 3.3x slower than Neuron |
| Audio decode (CPU) | 2.2s | Stereo WAV |
| Spatial upsample (CPU) | 1.8s | 498M params |
| Compilation (total) | ~33 min | Encoder 68s + S1 133s + S2 338s + VAE 570s |

**VAE Decoder**: The LTX-2.3 video decoder is compiled at TP=4 with 4×16 latent tiles (128×512 pixels). After Phase 2 unloads the TP=16 S2 backbone, the TP=4 VAE loads onto the freed NeuronCores (4.6s load time). Tiled decode uses overlap blending (overlap_h=1 latent) for seamless spatial reconstruction at arbitrary resolutions.

### Component Distribution

| Component | Location | Notes |
|-----------|----------|-------|
| DiT transformer (48 blocks) | **Neuron** (TP=4) | ~11 GB/rank HBM |
| Gemma 3 12B text encoder | **Neuron** (TP=4) or CPU | Shares NeuronCores with DiT, sequential execution |
| VideoDecoder | CPU | Per-channel statistics normalization |
| AudioDecoder + Vocoder | CPU | Float32 for vocoder accuracy |
| Spatial/Temporal upscalers | CPU | Sub-second each |
| EmbeddingsProcessor | CPU | Connectors + feature extraction |

Both the DiT backbone and Gemma3 encoder are compiled for TP=4 and share the same 4 NeuronCores. They execute sequentially: Gemma3 loads, encodes text, and unloads; then the DiT backbone loads and runs the 8-step denoising loop. CPU fallback for Gemma3 is available but ~25x slower.

## Usage

### Prerequisites

```bash
# On trn2.3xlarge with Deep Learning AMI Neuron (Ubuntu 24.04) 20260227
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Install ltx-core
pip install git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core

# Download model weights
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-distilled.safetensors \
  --local-dir /home/ubuntu/models/LTX-2.3/

# Download Gemma 3 12B text encoder
huggingface-cli download google/gemma-3-12b-it \
  --local-dir /home/ubuntu/models/gemma-3-12b

# Download upscaler weights (for --upscale, spatial x2 + temporal x2)
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --local-dir /home/ubuntu/models/LTX-2.3/upscalers/
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-temporal-upscaler-x2-1.0.safetensors \
  --local-dir /home/ubuntu/models/LTX-2.3/upscalers/
```

### Step 1: Compile the DiT Backbone

```bash
# Full-resolution backbone (384x512, VIDEO_SEQ=768)
NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  torchrun --nproc_per_node=4 src/compile_transformer.py
```

Compilation takes approximately 60 seconds. The compiled model is saved to `compiler_workdir_tp4_lnc2_v2/tp_0.pt` (8.7 GB).

For two-stage mode, also compile the half-resolution backbone:

```bash
# Half-resolution backbone (192x256, VIDEO_SEQ=192) — for two-stage mode
NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  torchrun --nproc_per_node=4 src/compile_transformer_halfres.py
```

This compiles the same architecture at the half-res latent shape (4×6×8 instead of 4×12×16). Output: `compiler_workdir_tp4_lnc2_halfres/tp_0.pt` (~8.7 GB).

### Step 2: Pre-shard Backbone Weights

Pre-sharding avoids loading the full 41 GB safetensors file during generation:

```bash
python3 src/shard_backbone_weights.py \
  --model-path /home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
  --output-dir /home/ubuntu/backbone_sharded
```

This produces 4 rank files (~9.3 GB each) that are loaded directly during generation.

### Step 3: Compile and Shard Gemma3 Encoder (Recommended)

Compiling Gemma3 for Neuron eliminates the ~162s CPU text encoding bottleneck:

```bash
# Compile the encoder graph
NEURON_FUSE_SOFTMAX=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  python3 src/compile_gemma3.py \
    --compile-dir /home/ubuntu/gemma3_encoder_compiled

# Pre-shard weights for fast loading (~5.9 GB per rank)
python3 src/shard_gemma3_weights.py \
  --gemma-path /home/ubuntu/models/gemma-3-12b \
  --output-dir /home/ubuntu/gemma3_encoder_sharded
```

The Gemma3 encoder uses tensorizer-optimized compiler flags (`--enable-ccop-compute-overlap`, `--cc-pipeline-tiling-factor=1`, `--vectorize-strided-dma`, `--enable-scalar-dge-vectorization`) that achieve 3.1x faster inference (644ms vs 2000ms) compared to the original precision flags.

### Step 4: Generate Video + Audio

```bash
# With Neuron-compiled Gemma3 (recommended, fastest)
python3 src/generate_ltx23.py \
  --neuron-gemma \
  --gemma-path /home/ubuntu/models/gemma-3-12b \
  --gemma-compiled-dir /home/ubuntu/gemma3_encoder_compiled \
  --gemma-sharded-dir /home/ubuntu/gemma3_encoder_sharded \
  --backbone-sharded-dir /home/ubuntu/backbone_sharded \
  --prompt "A golden retriever puppy runs across a sunny green meadow"

# With upscaling (384x512 @ 25 frames -> 768x1024 @ 49 frames)
python3 src/generate_ltx23.py \
  --neuron-gemma \
  --gemma-path /home/ubuntu/models/gemma-3-12b \
  --gemma-compiled-dir /home/ubuntu/gemma3_encoder_compiled \
  --gemma-sharded-dir /home/ubuntu/gemma3_encoder_sharded \
  --backbone-sharded-dir /home/ubuntu/backbone_sharded \
  --prompt "A golden retriever puppy runs across a sunny green meadow" \
  --upscale

# With CPU Gemma3 (slower, no compilation needed)
python3 src/generate_ltx23.py \
  --gemma-path /home/ubuntu/models/gemma-3-12b \
  --backbone-sharded-dir /home/ubuntu/backbone_sharded \
  --prompt "A golden retriever puppy runs across a sunny green meadow"

# Quick test with random embeddings (no Gemma needed)
python3 src/generate_ltx23.py --no-text-encoder
```

#### Image-to-Video Generation

Add `--image` to condition the video on an input photograph. Frame 0 is encoded from the image and preserved throughout denoising; subsequent frames are generated to match the prompt while maintaining visual consistency with the input.

```bash
# Image-to-video with Neuron Gemma3
python3 src/generate_ltx23.py \
  --neuron-gemma \
  --gemma-path /home/ubuntu/models/gemma-3-12b \
  --gemma-compiled-dir /home/ubuntu/gemma3_encoder_compiled \
  --gemma-sharded-dir /home/ubuntu/gemma3_encoder_sharded \
  --backbone-sharded-dir /home/ubuntu/backbone_sharded \
  --prompt "The woman turns and smiles warmly at the camera" \
  --image /path/to/photo.png
```

The image encoder uses the ltx-core `VideoEncoder` loaded from the same safetensors checkpoint (no additional downloads needed). No recompilation of the DiT backbone is required — the same compiled model handles both T2V and I2V since the tensor shapes are identical.

#### Two-Stage Generation (Refinement Denoising)

Two-stage mode follows the LTX-2.3 `DistilledPipeline` reference: generate at half resolution (192×256) with 8 steps, spatially upsample x2, then refine at full resolution (384×512) with 3 additional denoising steps. This produces sharper output than single-stage generation. Requires the half-res backbone compilation from Step 1.

```bash
# Two-stage with Neuron Gemma3
python3 src/generate_ltx23.py \
  --neuron-gemma \
  --gemma-path /home/ubuntu/models/gemma-3-12b \
  --gemma-compiled-dir /home/ubuntu/gemma3_encoder_compiled \
  --gemma-sharded-dir /home/ubuntu/gemma3_encoder_sharded \
  --backbone-sharded-dir /home/ubuntu/backbone_sharded \
  --prompt "A golden retriever puppy runs across a sunny green meadow" \
  --two-stage \
  --halfres-compiled-dir /home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_halfres
```

The pipeline sequence: Gemma3 encode → unload → half-res DiT (8 steps) → unload → spatial upsample x2 → full-res DiT (3 refinement steps) → unload → VAE decode. The same pre-sharded backbone weights are used for both stages (the model weights are identical; only the compiled shapes differ). The spatial upscaler weights are downloaded as part of the prerequisites.

Output: PNG frames, MP4 video (if ffmpeg available), WAV audio.

## Compatibility Matrix

| Instance/Version | SDK 2.27 | SDK 2.28 |
|------------------|----------|----------|
| trn2.3xlarge (TP=4, LNC=2) | — | VALIDATED |
| trn2.48xlarge (TP=4/16, LNC=2) | VALIDATED | — |

## Example Checkpoints

* [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) — `ltx-2.3-22b-distilled.safetensors` (8-step distilled, 43 GB)
* [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) — Text encoder (23 GB)

## Testing Instructions

```bash
# Ensure model is downloaded and backbone is compiled (see Usage above), then:
cd contrib/models/LTX-2.3

MODEL_PATH=/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
COMPILED_MODEL_PATH=/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_v2 \
  pytest test/integration/test_model.py -v -s
```

Tests validate:
- Model loads successfully with weight injection
- Forward pass produces valid (non-NaN) output
- Cosine similarity >= 0.999 vs CPU reference
- Per-step latency measurement

## Architecture Details

### Backbone Signature

The compiled backbone takes 24 flat tensors (for XLA tracing compatibility):

| Index | Shape | Description |
|-------|-------|-------------|
| 0 | (1, 768, 4096) | Video hidden states |
| 1 | (1, 26, 2048) | Audio hidden states |
| 2-3 | (1, 256, 4096/2048) | Text encoder context (video/audio) |
| 4-5 | (1, seq, 9*dim) | AdaLN timestep embeddings |
| 6-7 | (1, seq, dim) | Embedded timesteps |
| 8-11 | (1, 1, ...) | Cross-attention AdaLN scale/shift/gate |
| 12-19 | (1, heads, seq, dim/2) | RoPE cos/sin (self-attn + cross-attn) |
| 20-21 | (1, 256) | Encoder attention masks (additive) |
| 22-23 | (1, 1, ...) | Prompt timestep embeddings |

### TP Sharding Pattern

- **ColumnParallel**: Q, K, V projections, FFN gate/up projections, gate_logits
- **RowParallel**: Attention output projection, FFN down projection
- **DistributedRMSNorm**: QK-norm (q_norm, k_norm) with all-reduce for global variance
- **SPMDRank**: Per-rank RoPE slicing via `torch.index_select`

### Compiler Flags

**DiT backbone:**
```
--model-type=transformer -O1 --auto-cast matmult --lnc 2
--tensorizer-options='--enable-ccop-compute-overlap'
--enable-fast-loading-neuron-binaries
```

**Optional DiT flag** — adding `--vectorize-strided-dma` to the DiT tensorizer options gives a 1.9% backbone speedup (244.1ms → 239.4ms per step) but reduces single-pass cosine similarity from 0.9999 to 0.996 due to reordered BF16 accumulations. Not enabled by default. Other tensorizer flags tested: `--cc-pipeline-tiling-factor` hurts DiT performance at all values tested (1/2/4/8), and `--enable-scalar-dge-vectorization` is neutral.

**Gemma3 encoder** (tensorizer-optimized, 3.1x faster than original):
```
--model-type=transformer -O1 --auto-cast=none --lnc=2
--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1
  --vectorize-strided-dma --enable-scalar-dge-vectorization'
```

Environment: `NEURON_FUSE_SOFTMAX=1`, `NEURON_CUSTOM_SILU=1`, `NEURON_RT_STOCHASTIC_ROUNDING_EN=0`

## Known Issues

- **Cold start latency**: The DiT warmup pass takes ~139s and the first denoising step takes ~175s due to Neuron device initialization. Subsequent steps run at ~0.3s each. Gemma3 encoder NEFF rehydration adds ~362s on first load (one-time per instance).
- **CPU text encoding fallback**: Without Neuron-compiled Gemma3, text encoding takes ~162s on CPU. Use `--neuron-gemma` for 0.6s warm text encoding (~270x faster).
- **Two-stage cold start**: Two-stage mode loads two separate Neuron backbones sequentially (half-res and full-res), each with its own NEFF warmup. Total cold start overhead is ~2x single-stage.
- **BF16 TP accumulation**: The 0.972 cosine similarity over 8 denoising steps (vs CPU) is due to normal BF16 rounding across TP=4 ranks. Single forward pass accuracy is 0.9999.
- **No EFA**: The trn2.3xlarge single-instance setup does not use EFA for inter-node communication. NCCL/OFI warnings about EFA can be safely ignored.
- **CPU video decode bottleneck**: At 384×512 (25 frames), the CPU video decoder takes ~4.7s — over half of warm E2E time. At higher resolutions (1024×1536, 121 frames), CPU decode takes 78.7s. The TP=4 tiled Neuron VAE decoder reduces this to 23.5s (3.3x speedup). Use `--vae-compiled-dir` in `run_phase2.py` to enable Neuron decode. The tiled approach compiles the decoder at 4×16 latent (128×512 pixels) — H×W ≤ 64 is the SRAM limit — and decodes via overlapping spatial tiles with linear blending.

## Source Files

| File | Purpose |
|------|---------|
| `src/modeling_ltx23.py` | Core backbone: TP sharding, DistributedRMSNorm, SDPA replacement, TransformerArgs construction |
| `src/modeling_gemma3_encoder.py` | Custom Gemma3 encoder-only model: returns all 49 hidden states stacked, no KV cache |
| `src/pipeline.py` | NeuronTransformerWrapper: CPU preprocessing, backbone routing, mask handling, AdaLN deduplication, step-invariant caching |
| `src/compile_transformer.py` | DiT backbone compilation script — full-res 384×512 (torchrun --nproc_per_node=4) |
| `src/compile_transformer_halfres.py` | DiT backbone compilation script — half-res 192×256 for two-stage mode |
| `src/compile_gemma3.py` | Gemma3 encoder compilation script (parallel_model_trace) |
| `src/shard_gemma3_weights.py` | Pre-shard Gemma3 weights to per-rank files for fast loading |
| `src/shard_backbone_weights.py` | Pre-shard DiT backbone weights to per-rank files for fast loading |
| `src/load_with_weights.py` | DiT backbone weight sharding and injection utilities |
| `src/generate_ltx23.py` | E2E generation pipeline (text encoding, single/two-stage denoising, VAE decode, upscaling, image-to-video) |
| `src/run_phase2.py` | Phase 2 standalone script: spatial upsample + S2 denoising + Neuron/CPU VAE decode |
| `src/modeling_vae_23.py` | TP-sharded LTX-2.3 VAE decoder (~560 lines), ColumnRowParallelConv3d, CausalConv3d |
| `src/compile_vae_23.py` | VAE decoder compilation script (TP=4, 4×16 tile, 121 frames) |
| `src/tiled_vae_decode_23.py` | Tiled decode with overlap blending for arbitrary resolutions |
| `src/compile_benchmark.py` | Full benchmark compilation script (encoder + S1 + S2 backbone) |
| `src/application.py` | NeuronLTX23Application compositor for NxDI Application path |
