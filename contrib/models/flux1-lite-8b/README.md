# Contrib Model: FLUX.1-lite-8B-alpha

FLUX.1-lite-8B-alpha image generation model running on AWS Neuron using NxDI's first-party FLUX.1 implementation with zero code modifications.

## Model Information

- **HuggingFace ID:** `Freepik/flux.1-lite-8B-alpha`
- **Model Type:** Diffusion transformer (DiT) for text-to-image generation
- **Parameters:** ~8B (BF16)
- **Architecture:** 8 double-stream MMDiT blocks + 38 single-stream DiT blocks, CLIP + T5-XXL text encoders, 16-channel VAE, FlowMatchEulerDiscrete scheduler
- **License:** Check HuggingFace model card (gated model, requires access approval)

## Key Finding: Native NxDI FLUX.1 Compatibility

**FLUX.1-lite-8B-alpha is architecturally identical to FLUX.1-dev** with only the number of double-stream blocks reduced (8 vs 19). All other components are the same:

| Component | FLUX.1-dev | FLUX.1-lite-8B | Same? |
|-----------|-----------|----------------|-------|
| Double-stream (MMDiT) blocks | 19 | 8 | Different |
| Single-stream (DiT) blocks | 38 | 38 | Same |
| Attention heads | 24 | 24 | Same |
| Attention head dim | 128 | 128 | Same |
| Joint attention dim | 4096 | 4096 | Same |
| Text encoders | CLIP + T5-XXL | CLIP + T5-XXL | Same |
| VAE latent channels | 16 | 16 | Same |
| RoPE axes_dim | (16, 56, 56) | (16, 56, 56) | Same |
| Pipeline class | FluxPipeline | FluxPipeline | Same |
| Scheduler | FlowMatchEulerDiscrete | FlowMatchEulerDiscrete | Same |
| guidance_embeds | True | True | Same |

Because NxDI's FLUX.1 implementation reads `num_layers` and `num_single_layers` from the model's `config.json` at runtime (via `load_diffusers_config()`), it automatically adapts to FLUX.1-lite's configuration. **No custom modeling code is needed.**

This contrib provides:
- A standalone generation script (`src/generate_flux_lite.py`) for 1024x1024
- A high-resolution script (`src/generate_flux_lite_highres.py`) for 2048x2048 and 4096x4096
- Integration tests validating correct operation on Neuron
- Benchmark results demonstrating the performance benefit of the lighter architecture

## Validation Results

**Validated:** 2026-04-28
**Instance:** trn2.3xlarge (LNC=2, 4 logical cores)
**SDK:** Neuron SDK 2.29 (DLAMI 20260410), PyTorch 2.9, NxD Inference 0.9

### Benchmark Results (1024x1024, 25 steps, guidance_scale=3.5)

| Metric | Value |
|--------|-------|
| Resolution | 1024x1024 |
| Inference steps | 25 |
| TP Degree | 4 |
| CFG | Guidance distillation (single forward pass/step) |
| E2E generation time | 5.91s avg |
| Pipeline steps/sec | 4.23 |
| Backbone forward/sec | 4.49 |
| Compilation time | ~128s (CLIP 69s + T5 5s + backbone 53s + VAE ~2s) |

## High-Resolution Generation (2K, 4K)

> **Note:** The original FLUX.1-lite-8B model was trained and validated at 1024x1024 only.
> The original FLUX.1-dev/schnell models do not natively support 2K or 4K resolution either.
> High-resolution generation is an extrapolation beyond the training distribution — image
> quality may differ from native resolution. This capability is primarily useful for
> customers who have fine-tuned their own models at higher resolutions or want to evaluate
> the architecture's scaling behavior.

### Results Summary

| Resolution | Tokens | Latency | Instance | Strategy |
|-----------|--------|---------|----------|----------|
| 1024x1024 | 4,096 | **5.91s** | trn2.3xlarge | TP=4 |
| 2048x2048 | 16,384 | **31.53s** | trn2.3xlarge | TP=4 + tiled VAE (4 tiles) |
| 4096x4096 | 65,536 | **107.25s** | trn2.48xlarge | TP=4, CP=4 + tiled VAE (25 tiles) |

### How It Works

**2048x2048 (16,384 tokens):**
- The backbone (transformer) is compiled directly at 2K resolution — the self-attention operates over 16,384 tokens with TP=4, which fits in HBM on trn2.3xlarge (24 GB/core with LNC=2).
- The VAE decoder exceeds the 5M instruction limit at 2K, so it is compiled at 1024x1024 and the 256x256 latent is decoded with 4 overlapping tiles (128x128 each, 16px overlap).
- No context parallelism needed. Same hardware as 1K.

**4096x4096 (65,536 tokens):**
- The 65,536-token self-attention exceeds per-core HBM capacity on trn2.3xlarge even with TP=4.
- Solution: **Context Parallelism (CP=4)** splits the sequence across 4 groups, giving each shard 16,384 tokens (identical to the working 2K case).
- Configuration: `TP=4, CP=4, world_size=16` on trn2.48xlarge using 16 of 64 logical cores.
- The VAE decoder is compiled at 1024x1024 and decodes the 512x512 latent with 25 overlapping tiles.
- **CRITICAL**: Must set `NEURON_RT_VISIBLE_CORES=0-15` to prevent the runtime from detecting all 64 cores and creating a 64-rank collective communicator (which deadlocks).

### High-Resolution Usage

```bash
# 2K on trn2.3xlarge (same instance as 1K):
python src/generate_flux_lite_highres.py \
    --checkpoint_dir /shared/flux1-lite-8b \
    --height 2048 --width 2048 \
    --save_image --save_results

# 4K on trn2.48xlarge (requires NEURON_RT_VISIBLE_CORES):
NEURON_RT_VISIBLE_CORES=0-15 python src/generate_flux_lite_highres.py \
    --checkpoint_dir /shared/flux1-lite-8b \
    --height 4096 --width 4096 \
    --save_image --save_results
```

### 4K Timing Breakdown

| Phase | Time | Notes |
|-------|------|-------|
| Compilation (from scratch) | ~39 min | Backbone 25.8 min + VAE 7.3 min + encoders <1 min |
| Compilation (from cache) | 0s | NEFFs cached in compile_workdir |
| Model loading (cold) | ~40 min | 8B params across 16 cores from disk |
| Model loading (warm) | ~11s | NEFFs already in device memory |
| Backbone (25 steps) | 99.5s | 3.98s/step |
| Tiled VAE decode (25 tiles) | 7.3s | 0.29s/tile |
| **Total generation** | **107.25s** | Steady state (after warmup) |

## Usage

```python
import torch
from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication,
    create_flux_config,
    get_flux_parallelism_config,
)

MODEL_PATH = "/shared/flux1-lite-8b/"
COMPILE_DIR = "/tmp/flux-lite/compiled/"

# Configure (reads num_layers=8 from model's config.json automatically)
world_size = get_flux_parallelism_config(backbone_tp_degree=4)
clip_cfg, t5_cfg, backbone_cfg, decoder_cfg = create_flux_config(
    MODEL_PATH, world_size, backbone_tp_degree=4,
    dtype=torch.bfloat16, height=1024, width=1024,
)

# Create application
app = NeuronFluxApplication(
    model_path=MODEL_PATH,
    text_encoder_config=clip_cfg,
    text_encoder2_config=t5_cfg,
    backbone_config=backbone_cfg,
    decoder_config=decoder_cfg,
    height=1024, width=1024,
)

# Compile + load
app.compile(COMPILE_DIR)
app.load(COMPILE_DIR)

# Generate
image = app(
    "A cat holding a sign that says hello world",
    height=1024, width=1024,
    guidance_scale=3.5,
    num_inference_steps=25,
).images[0]
image.save("output.png")
```

Or use the provided script:

```bash
python src/generate_flux_lite.py \
    --checkpoint_dir /shared/flux1-lite-8b \
    --compile_workdir /tmp/flux-lite/compiled/ \
    --prompt "A cat holding a sign that says hello world" \
    --height 1024 --width 1024 \
    --num_inference_steps 25 \
    --save_image
```

## Setup

```bash
# Activate NxDI environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install diffusers (not pre-installed in NxDI venv)
pip install diffusers transformers accelerate sentencepiece protobuf

# Download model (requires HuggingFace token with access)
huggingface-cli download Freepik/flux.1-lite-8B-alpha \
    --local-dir /shared/flux1-lite-8b
```

## Compatibility Matrix

| Instance | Resolution | SDK 2.29 | SDK 2.30 |
|----------|-----------|----------|----------|
| trn2.3xlarge (LNC=2, TP=4) | 1024x1024 | VALIDATED | VALIDATED |
| trn2.3xlarge (LNC=2, TP=4) | 2048x2048 | VALIDATED | VALIDATED |
| trn2.48xlarge (LNC=2, TP=4, CP=4) | 4096x4096 | Not tested | VALIDATED |

## Example Checkpoints

* [Freepik/flux.1-lite-8B-alpha](https://huggingface.co/Freepik/flux.1-lite-8B-alpha)

## Testing Instructions

```bash
# Set model path
export FLUX_LITE_MODEL_PATH=/shared/flux1-lite-8b/

# Run with pytest
cd contrib/models/flux1-lite-8b/
pytest test/integration/test_model.py -v

# Or standalone
python test/integration/test_model.py
```

## Known Issues

- The NxDI venv (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`) does not include `diffusers` by default. Install it with pip before running.
- `attention_cte` kernel warnings about batch size x seqlen_q x seqlen_k appear during inference. These are informational and do not affect output quality.

## Sample Output

![FLUX.1-lite output](samples/flux_lite_cat_hello_world.png)

*"A cat holding a sign that says hello world" -- 1024x1024, 25 steps, guidance_scale=3.5*
