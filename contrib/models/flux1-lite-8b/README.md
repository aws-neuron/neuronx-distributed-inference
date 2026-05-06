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
- A standalone generation script (`src/generate_flux_lite.py`)
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

| Instance/Version | SDK 2.29 | SDK 2.28 |
|------------------|----------|----------|
| trn2.3xlarge (LNC=2, TP=4) | VALIDATED | Not tested |

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
