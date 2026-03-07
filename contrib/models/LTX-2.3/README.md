# Contrib Model: LTX-2.3

LTX-2.3 22B parameter DiT audio-video diffusion transformer running on AWS Trainium 2 via NxD Inference. Generates synchronized video + audio from text prompts.

## Model Information

- **HuggingFace ID:** [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3)
- **Model Type:** DiT (Diffusion Transformer) for joint audio-video generation
- **Parameters:** 22B (BF16) — 48 transformer blocks, 32 heads, 4096 video dim, 2048 audio dim
- **Architecture:** Bidirectional audio-video cross-attention, gated attention, QK-RMSNorm, split RoPE, flow matching
- **License:** See HuggingFace model card
- **Framework:** Native [`ltx-core`](https://github.com/Lightricks/LTX-2) (not Diffusers)

## Validation Results

**Validated:** 2026-03-07
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
| CPU component loading | 21.8s | LTXModel, VideoDecoder, AudioDecoder, Vocoder, EmbeddingsProcessor |
| Neuron backbone loading (4 ranks) | 128.6s | 4135 weights per rank, 9.3 GB compiled model |
| Text encoding (Gemma 3 12B) | 162.0s | CPU, single prompt |
| Denoising step (warm) | 228.7ms | Steps 3-8 after warmup (avg of 10 runs) |
| Denoising step (cold, step 1) | 180.2s | Includes Neuron device initialization |
| Denoising step (warmup, step 2) | 229.9s | Second pass warmup |
| Total denoising (8 steps) | 412.1s | Dominated by cold start |
| Spatial upscaler (CPU) | 0.6s | 498M params, (1,128,4,12,16) -> (1,128,4,24,32) |
| Temporal upscaler (CPU) | 0.4s | 131M params, (1,128,4,24,32) -> (1,128,7,24,32) |
| Video decode (CPU, no upscale) | ~8s | 25 frames @ 384x512 |
| Video decode (CPU, with upscale) | 32.4s | 49 frames @ 768x1024 |
| Audio decode (CPU) | 2.3s | Stereo WAV, 48kHz |

### Component Distribution

| Component | Location | Notes |
|-----------|----------|-------|
| DiT transformer (48 blocks) | **Neuron** (TP=4) | ~11 GB/rank HBM |
| Gemma 3 12B text encoder | CPU | 23 GB system RAM |
| VideoDecoder | CPU | Per-channel statistics normalization |
| AudioDecoder + Vocoder | CPU | Float32 for vocoder accuracy |
| Spatial/Temporal upscalers | CPU | Sub-second each |
| EmbeddingsProcessor | CPU | Connectors + feature extraction |

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

# Download upscaler weights (optional)
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --local-dir /home/ubuntu/models/LTX-2.3/upscalers/
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-temporal-upscaler-x2-1.0.safetensors \
  --local-dir /home/ubuntu/models/LTX-2.3/upscalers/
```

### Step 1: Compile the Backbone

```bash
NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  torchrun --nproc_per_node=4 src/compile_transformer.py
```

Compilation takes approximately 30-60 minutes. The compiled model is saved to `compiler_workdir_tp4_lnc2_v2/tp_0.pt` (9.3 GB).

### Step 2: Generate Video + Audio

```bash
# With real text encoder
python3 src/generate_ltx23.py \
  --gemma-path /home/ubuntu/models/gemma-3-12b \
  --prompt "A golden retriever puppy runs across a sunny green meadow"

# With upscaling (384x512 @ 25 frames -> 768x1024 @ 49 frames)
python3 src/generate_ltx23.py \
  --gemma-path /home/ubuntu/models/gemma-3-12b \
  --prompt "A golden retriever puppy runs across a sunny green meadow" \
  --upscale

# Quick test with random embeddings (no Gemma needed)
python3 src/generate_ltx23.py --no-text-encoder
```

Output: PNG frames, MP4 video (if ffmpeg available), WAV audio.

## Compatibility Matrix

| Instance/Version | SDK 2.28 |
|------------------|----------|
| trn2.3xlarge (TP=4, LNC=2) | VALIDATED |

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

```
--model-type=transformer -O1 --auto-cast matmult --lnc 2
--tensorizer-options='--enable-ccop-compute-overlap'
--enable-fast-loading-neuron-binaries
```

Environment: `NEURON_FUSE_SOFTMAX=1`, `NEURON_CUSTOM_SILU=1`, `NEURON_RT_STOCHASTIC_ROUNDING_EN=0`

## Known Issues

- **Cold start latency**: First two denoising steps are slow (~180s + ~230s) due to Neuron device initialization and warmup. Subsequent steps run at ~300ms each.
- **CPU bottleneck**: Text encoding (Gemma 3 12B on CPU) takes ~162s. This dominates total generation time for single-request workloads.
- **Single-stage only**: This submission includes Stage 1 generation with optional latent upscaling but not Stage 2 refinement denoising. Stage 2 requires recompiling the backbone at a larger latent shape and merging distilled LoRA weights.
- **BF16 TP accumulation**: The 0.972 cosine similarity over 8 denoising steps (vs CPU) is due to normal BF16 rounding across TP=4 ranks. Single forward pass accuracy is 0.9999.
- **No EFA**: The trn2.3xlarge single-instance setup does not use EFA for inter-node communication. NCCL/OFI warnings about EFA can be safely ignored.

## Source Files

| File | Purpose |
|------|---------|
| `src/modeling_ltx23.py` | Core backbone: TP sharding, DistributedRMSNorm, SDPA replacement, TransformerArgs construction |
| `src/pipeline.py` | NeuronTransformerWrapper: CPU preprocessing, backbone routing, mask handling |
| `src/compile_transformer.py` | Compilation script (torchrun --nproc_per_node=4) |
| `src/load_with_weights.py` | Weight sharding and injection utilities |
| `src/generate_ltx23.py` | E2E generation pipeline (text encoding, denoising, VAE decode, upscaling) |
