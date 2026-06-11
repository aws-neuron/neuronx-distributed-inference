# Contrib Model: Cosmos3-Text2Image

NeuronX Distributed Inference implementation of NVIDIA Cosmos3 omnimodal
Mixture-of-Transformers (MoT) for text-to-image generation.

## Model Information

- **Models:** Cosmos3-Nano (16B), Cosmos3-Super-Text2Image (65B)
- **HuggingFace ID:** `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super-Text2Image`
- **Model Type:** Diffusion Transformer (MoT architecture)
- **Task:** Text-to-Image Generation (512x512)
- **License:** Check HuggingFace model card

## Architecture Details

Cosmos3 uses a **Mixture-of-Transformers (MoT)** architecture:
- Dual-stream processing: text (understanding) and vision (generation) pathways
- Joint MMDiT-style attention: text uses causal self-attention, vision attends bidirectionally to all tokens
- Separate SwiGLU MLPs per stream (text MLP + generation MLP in each layer)
- M-RoPE (Multimodal Rotary Position Embedding) with 3 axes (T, H, W)
- QK normalization (per-head RMSNorm)
- GQA (Grouped Query Attention)
- VAE: AutoencoderKLWan with 48 latent channels, patch_size=2, spatial_compression=16
- Scheduler: UniPCMultistepScheduler (35 steps, flow matching)
- CFG scale: 6.0

| | Cosmos3-Nano | Cosmos3-Super |
|--|--|--|
| Parameters | 16B | 65B |
| hidden_size | 4096 | 5120 |
| intermediate_size | 12288 | 25600 |
| Layers | 36 | 64 |
| Q Heads | 32 | 64 |
| KV Heads | 8 | 8 |
| Instance | trn2.3xlarge | trn2.48xlarge |
| TP Degree | 4 | 8 |

## Validation Results

**Validated:** 2026-06-11
**SDK:** 2.30 (torch-neuronx 2.9.0.2.14.27725)

### Cosmos3-Nano (trn2.3xlarge, TP=4)

| Metric | Value |
|--------|-------|
| Backbone latency | 33.4 ms/call |
| E2E generation (35 steps) | **2.79s** |
| Per-step latency | 78.2 ms |
| VAE decode | 50 ms |
| Image quality | High fidelity, prompt-aligned |

### Cosmos3-Super-Text2Image (trn2.48xlarge, TP=8)

| Metric | Value |
|--------|-------|
| Backbone latency | 79.5 ms/call |
| E2E generation (35 steps) | **5.81s** |
| Per-step latency | 164.6 ms |
| VAE decode | 50 ms |
| Image quality | High fidelity, prompt-aligned |

**Status:** VALIDATED (both variants)

## Setup

### Prerequisites

- AWS Neuron SDK 2.30+ (DLAMI `Deep Learning AMI Neuron (Ubuntu 24.04) 20260522`)
- trn2.3xlarge (Nano) or trn2.48xlarge (Super)
- `diffusers >= 0.39.0.dev0` (for Cosmos3 VAE support)

### Environment

```bash
# Activate pre-installed environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

# Install diffusers from source (needed for Cosmos3 VAE)
pip install git+https://github.com/huggingface/diffusers.git
```

### Download Model

```bash
# Nano (33 GB)
huggingface-cli download nvidia/Cosmos3-Nano --local-dir /home/ubuntu/Cosmos3-Nano

# Super (124 GB)
huggingface-cli download nvidia/Cosmos3-Super-Text2Image --local-dir /home/ubuntu/Cosmos3-Super-Text2Image
```

## Usage

### 1. Compile Backbone

```bash
# Nano (TP=4, ~3 min compile)
python examples/compile.py \
    --model-path /home/ubuntu/Cosmos3-Nano \
    --tp 4 \
    --output /home/ubuntu/compiled_cosmos3_nano

# Super (TP=8, ~5 min compile)
python examples/compile.py \
    --model-path /home/ubuntu/Cosmos3-Super-Text2Image \
    --tp 8 \
    --output /home/ubuntu/compiled_cosmos3_super
```

### 2. Compile VAE

```bash
python examples/compile_vae.py \
    --model-path /home/ubuntu/Cosmos3-Nano \
    --output /home/ubuntu/compiled_vae/vae_decoder.pt
```

Note: The same compiled VAE works for both Nano and Super (same architecture).

### 3. Generate Images

```bash
# Nano
python examples/generate.py \
    --model-path /home/ubuntu/Cosmos3-Nano \
    --compiled-path /home/ubuntu/compiled_cosmos3_nano \
    --vae-path /home/ubuntu/compiled_vae/vae_decoder.pt \
    --tp 4 \
    --prompt "A cat sitting on a windowsill watching birds" \
    --output cat.png

# Super
python examples/generate.py \
    --model-path /home/ubuntu/Cosmos3-Super-Text2Image \
    --compiled-path /home/ubuntu/compiled_cosmos3_super \
    --vae-path /home/ubuntu/compiled_vae/vae_decoder.pt \
    --tp 8 \
    --prompt "A majestic snow-covered mountain at sunrise with golden light" \
    --output mountain.png
```

### Python API

```python
import torch
import torch_neuronx
from src.modeling_cosmos3 import (
    Cosmos3BackboneInferenceConfig,
    NeuronCosmos3BackboneApplication,
)
from src.pipeline import (
    build_position_ids, denoise, denormalize_latents, tokenize_prompt,
)
from neuronx_distributed_inference.models.config import NeuronConfig
from transformers import AutoTokenizer
from diffusers import UniPCMultistepScheduler

# Configure (Nano example)
neuron_config = NeuronConfig(tp_degree=4, world_size=4, torch_dtype=torch.bfloat16)
config = Cosmos3BackboneInferenceConfig(neuron_config=neuron_config)
config.max_text_len = 256
config.num_vision_patches = 256

# Load
app = NeuronCosmos3BackboneApplication(
    model_path="/home/ubuntu/Cosmos3-Nano/transformer", config=config
)
app.load("/home/ubuntu/compiled_cosmos3_nano")

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/Cosmos3-Nano", trust_remote_code=True)
cond_ids, cond_len = tokenize_prompt(tokenizer, "A sunset over the ocean")
uncond_ids, uncond_len = tokenize_prompt(tokenizer, "", negative=True)

# Position IDs
cond_pos = build_position_ids(256, cond_len, T=1, pH=16, pW=16)
uncond_pos = build_position_ids(256, uncond_len, T=1, pH=16, pW=16)

# Generate
latents = torch.randn(1, 48, 1, 32, 32, dtype=torch.float32)
scheduler = UniPCMultistepScheduler.from_pretrained("/home/ubuntu/Cosmos3-Nano", subfolder="scheduler")

latents = denoise(app, cond_ids, uncond_ids, cond_pos, uncond_pos, scheduler, latents)
latents = denormalize_latents(latents, "/home/ubuntu/Cosmos3-Nano/vae/config.json")

# Decode with VAE
vae = torch.jit.load("/home/ubuntu/compiled_vae/vae_decoder.pt")
pixels = vae(latents.float())  # [1, 3, 1, 512, 512]
```

## Testing

```bash
# Run integration tests
export COSMOS3_MODEL_PATH=/home/ubuntu/Cosmos3-Nano
export COSMOS3_COMPILED_PATH=/home/ubuntu/compiled_cosmos3_nano
export COSMOS3_VAE_PATH=/home/ubuntu/compiled_vae/vae_decoder.pt
export COSMOS3_TP_DEGREE=4

pytest test/integration/test_model.py --capture=tee-sys -v

# Or run manually:
python test/integration/test_model.py
```

## Key Implementation Notes

1. **Channel ordering in patchify/unpatchify**: Uses spatial-first, channels-last
   `(p_h, p_w, C)` matching the reference einsum `"cthpwq->thwpqc"`. Getting this
   wrong produces a 16x16 repeating tile pattern.

2. **Temporal margin**: Vision position IDs use `actual_text_len + 15000` as temporal
   offset, matching `unified_3d_mrope_temporal_modality_margin=15000` in the reference.

3. **Tokenization**: Must use the full chat template format with system prompt +
   resolution template + special tokens (eos + vision_start).

4. **Warmup both CFG paths**: The backbone must be warmed up with both conditional and
   unconditional inputs before timing. Without this, first-call overhead adds ~2.6s.

5. **Super model compilation**: Models with > 36 layers require `--verify-hlo=false`
   and `--modular-flow-mac-threshold=10` to avoid pre-partition HBM verification failure.

## Compatibility Matrix

| Instance | Nano (TP=4) | Super (TP=8) |
|----------|-------------|--------------|
| trn2.3xlarge (LNC=2) | **Working** | N/A (HBM limit) |
| trn2.48xlarge (LNC=2) | Working | **Working** |

## Maintainer

Annapurna Labs

**Last Updated:** 2026-06-11
