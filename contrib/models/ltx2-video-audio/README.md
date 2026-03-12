# LTX-2 on AWS Trainium (NxDI)

Port of the [Lightricks LTX-2](https://huggingface.co/Lightricks/LTX-2) 19B-parameter audio-video diffusion model to AWS Neuron (Trainium/Inferentia) using the NxDI (NeuronX Distributed Inference) framework.

## Architecture

LTX-2 generates synchronized video + audio from text prompts. The model has three main components:

| Component | Parameters | Runs on | Notes |
|-----------|-----------|---------|-------|
| Text encoder (Gemma 3-12B) | 12B | **Neuron TP=4** | Compiled with custom encoder |
| DiT transformer (48 blocks) | ~6B | **Neuron TP=4** | The denoising bottleneck |
| Video VAE decoder | ~1B | **Neuron TP=4** | Tiled decode with rectangular 4x16 tiles |
| Audio VAE decoder + vocoder | ~0.1B | CPU | Run once per generation |

Both the text encoder and DiT transformer backbone are compiled for Neuron. They coexist on the same 4 NeuronCores (TP=4) and execute sequentially: text encoding then denoising.

The video VAE decoder is also compiled for Neuron using tensor-parallel tiled decoding. The optimal tile shape is 4x16 latent (128x512 pixels), which fits within the 64-element SRAM budget while minimizing tile count for large resolutions.

## Performance

Tested on trn2.3xlarge (1 NeuronDevice, 4 logical NeuronCores with LNC=2) with Neuron SDK 2.28:

| Metric | Neuron (trn2) | GPU (g5.12xlarge) |
|--------|--------------|-------------------|
| Generation time (warm, CFG, 8 steps) | ~22s | ~48s |
| First generation (includes warmup) | ~64s | ~48s |

The Neuron pipeline produces nearly identical output to the GPU reference.

## Compatibility Matrix

| Instance Type | Neuron SDK | Status |
|--------------|-----------|--------|
| trn2.3xlarge | 2.28 | Tested |
| trn2.3xlarge | 2.27 | Tested |
| Inf2 | — | Not tested |
| Trn1 | — | Not tested (requires TP=4 with LNC=2) |

## Example Checkpoints

* [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) — HuggingFace Hub (downloaded automatically by the scripts)

## Requirements

- **Instance**: trn2.3xlarge (sa-east-1 or ap-southeast-4)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 24.04) 20260227 (SDK 2.28)
- **Python env**: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/`
- **Diffusers**: 0.37.0.dev0 (install from git main)
- **Disk**: ~100GB for model weights + compilation cache

## Quick Start

### 1. Instance Setup

```bash
# SSH into your trn2.3xlarge instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Install diffusers from git (LTX-2 requires dev version)
pip install git+https://github.com/huggingface/diffusers.git
pip install imageio imageio-ffmpeg

# Upload this package
# scp -r ltx2-neuron/ ubuntu@<instance-ip>:/home/ubuntu/
```

### 2. Compile Models (First Run Only)

```bash
cd /home/ubuntu/ltx2-neuron/src

# Compile the DiT transformer backbone (~2 minutes)
NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  python compile_gemma3.py

# Pre-shard Gemma3 weights for fast loading (~2 minutes)
python shard_gemma3_weights.py

# Compile VAE decoder for Neuron (~30 minutes on trn2.48xlarge)
# Default: 4x16 tile (128x512 px), optimal for 1024x1536 output
NEURON_RT_VISIBLE_CORES=0-3 python compile_vae.py \
  --tp-degree 4 --height 128 --width 512 \
  --output-dir /home/ubuntu/ltx2_vae_tp4_128x512
```

The DiT backbone is compiled automatically on first use by the E2E script.

### 3. Generate Video + Audio

```bash
cd /home/ubuntu/ltx2-neuron/examples

NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  python neuron_e2e.py
```

### 4. Notebook

For an interactive walkthrough, use the Jupyter notebook:

```bash
cd /home/ubuntu/ltx2-neuron/notebooks
jupyter notebook ltx2_neuron_inference.ipynb
```

## Testing

Run integration tests (requires compiled models — see Quick Start steps 1-2 first):

```bash
cd /home/ubuntu/ltx2-video-audio

# With pytest
NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  pytest test/integration/test_model.py -v --capture=tee-sys

# Or standalone
NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  python test/integration/test_model.py
```

The test suite includes:
- **Smoke test**: Pipeline loads without errors
- **Generation test**: Produces expected number of frames at correct resolution
- **Accuracy test**: SSIM comparison between Neuron output and GPU reference frames (threshold: SSIM > 0.7)
- **Performance test**: Warm generation completes within 120s

## File Structure

```
ltx2-video-audio/
├── README.md
├── src/                              # Core NxDI package
│   ├── __init__.py
│   ├── modeling_ltx2.py              # DiT backbone: TP sharding, SPMD, config
│   ├── modeling_vae.py               # VAE decoder: TP Conv3d, tiled compilation
│   ├── modeling_gemma3_encoder.py    # Gemma3 text encoder for Neuron
│   ├── application.py                # NeuronLTX2Application orchestrator
│   ├── pipeline.py                   # NeuronTransformerWrapper (CFG batch-split)
│   ├── compile_gemma3.py             # Gemma3 encoder compilation script
│   ├── compile_vae.py                # VAE decoder compilation script
│   ├── tiled_vae_decode.py           # Tiled VAE decode runtime (standalone + library)
│   ├── shard_gemma3_weights.py       # Pre-shard Gemma3 weights to disk
│   └── generate_ltx2.py             # CLI entry point with argument parsing
├── test/
│   └── integration/
│       └── test_model.py             # Accuracy + performance integration tests
├── notebooks/
│   └── ltx2_neuron_inference.ipynb   # Step-by-step compile + generate notebook
├── examples/
│   ├── neuron_e2e.py                 # Full E2E: load + generate (both on Neuron)
│   └── gpu_generate.py              # GPU reference generation script
└── samples/
    ├── neuron/                       # Output from Neuron (trn2.3xlarge)
    │   ├── frame_0000.png
    │   ├── frame_0012.png
    │   └── frame_0024.png
    └── gpu/                          # GPU reference (g5.12xlarge, same seed)
        ├── frame_0000.png
        ├── frame_0012.png
        └── frame_0024.png
```

## Generation Settings

Both Neuron and GPU samples were generated with identical settings:

```python
prompt = ("A golden retriever puppy runs across a sunny green meadow, "
          "its ears flapping in the wind. The camera follows from a low angle. "
          "Birds chirp in the background.")
height = 384
width = 512
num_frames = 25
num_inference_steps = 8
guidance_scale = 4.0       # CFG (pipeline default)
max_sequence_length = 1024  # Pipeline default
seed = 42
model = "Lightricks/LTX-2"
```

## Technical Details

### CFG (Classifier-Free Guidance) on Neuron

With `guidance_scale=4.0`, the pipeline runs the text encoder twice (positive + negative prompt) and the DiT backbone twice per denoising step (uncond + cond). Since the Neuron backbone is compiled for `batch_size=1`, the `NeuronTransformerWrapper` handles CFG by splitting the batch and calling the backbone twice per step, then concatenating results.

### 22-Input Transformer Signature

The compiled Neuron DiT model takes 22 positional tensor arguments, all preprocessed on CPU:

| # | Input | Shape | Description |
|---|-------|-------|-------------|
| 1 | hidden_states | (1, 768, 4096) | Video latents after proj_in |
| 2 | audio_hidden_states | (1, 26, 2048) | Audio latents after audio_proj_in |
| 3 | encoder_hidden_states | (1, 1024, 4096) | Text embeddings for video |
| 4 | audio_encoder_hidden_states | (1, 1024, 2048) | Text embeddings for audio |
| 5-8 | temb, temb_audio, embedded_ts, audio_embedded_ts | various | Time embeddings |
| 9-12 | cross-attn scale/shift/gate | various | Cross-attention conditioning |
| 13-16 | video/audio rotary cos/sin | various | Self-attention RoPE |
| 17-20 | cross-attn video/audio rotary cos/sin | various | Cross-attention RoPE |
| 21-22 | encoder/audio_attention_mask | (1, 1, 1024) | Additive attention bias |

### Critical Implementation Details

- **SPMDRank RoPE**: Uses NxD `SPMDRank` module (not a Python int) for per-rank RoPE slicing. A Python int gets baked as constant 0 during SPMD XLA tracing, causing all ranks to apply the same RoPE shard.

- **DistributedRMSNorm**: QK-norm uses all-reduce across TP ranks to compute global variance.

- **BMM SDPA**: Replaces `torch.nn.functional.scaled_dot_product_attention` with explicit BMM operations for Neuron XLA compatibility.

- **RoPE BF16 cast**: RoPE modules return float32 for numerical precision. Tensors must be cast to bfloat16 before passing to the compiled Neuron model.

- **Additive attention mask**: Binary masks are converted to additive bias format (-10000 for masked positions) before compilation.

- **Pre-sharded weights**: Gemma3 weights are pre-sharded to disk (~5.5 GB per rank) using `.contiguous().clone()` to avoid serializing the full unsharded storage backing sliced tensors.

### Compiler Flags

```
--model-type=transformer -O1 --auto-cast=none --enable-saturate-infinity
--enable-mixed-precision-accumulation --lnc=2
--tensorizer-options='--enable-ccop-compute-overlap'
```

VAE decoder uses different compiler flags:
```
--model-type=unet-inference -O1 --auto-cast none --enable-fast-loading-neuron-binaries
```

### Neuron VAE Decoder (Tiled Decode)

The LTX-2 VAE decoder (128 input channels, 3D convolutions with 3 upsampler blocks) cannot be compiled at full resolution due to SRAM limits. Instead, we compile it at a small tile size and decode the full image by tiling with overlap blending.

**Compilation boundary**: `H_latent * W_latent <= 64` elements. Above this, the compiler fails with `NCC_IGCA030` (SRAM allocation failure).

**Optimal tile**: 4x16 latent (128x512 pixels) — maximizes spatial area within the budget.

**Tiled decode performance (1024x1536, 121 frames, trn2.48xlarge)**:

| Tile | Overlap (h×w) | Tiles | Neuron | CPU | Speedup | Cos Sim |
|------|--------------|-------|--------|-----|---------|---------|
| 8×8 | 4×4 | 77 | 111.5s | 100s | 0.90x | 0.901 |
| **4×16** | **1×0** | **33** | **42.4s** | **99s** | **2.3x** | **0.892** |
| 4×16 | 0×0 | 24 | 30.9s | 98s | 3.2x | 0.872 |

Recommended configuration: `--tile-latent-h 4 --tile-latent-w 16 --overlap-h 1 --overlap-w 0`

Environment variables:
```bash
NEURON_FUSE_SOFTMAX=1
NEURON_CUSTOM_SILU=1
NEURON_RT_STOCHASTIC_ROUNDING_EN=0
```
