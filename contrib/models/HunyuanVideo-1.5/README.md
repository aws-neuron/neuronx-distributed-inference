# HunyuanVideo-1.5 on AWS Neuron

## Model Overview

HunyuanVideo-1.5 is an 8.3B parameter text-to-video diffusion model by Tencent. It generates short video clips from text prompts using a multi-stage pipeline:

1. **Text encoding**: A Qwen2.5-VL-7B language model and a ByT5-small encoder extract text features, which are refined and reordered into a unified conditioning sequence.
2. **Denoising**: A 54-block MMDiT (Multi-Modal Diffusion Transformer) iteratively denoises a latent video representation over 50 steps with classifier-free guidance.
3. **VAE decoding**: An 871M parameter 3D convolutional VAE decoder converts the denoised latents into pixel-space video frames.

| | |
|---|---|
| Architecture | MMDiT (54 blocks) + DCAE VAE |
| Total parameters | ~16B (8.3B backbone + 7B text encoder + 871M VAE) |
| Output | 33 frames at 480×640 (~1.1s at 30fps) |
| Precision | bfloat16 |
| Source | [hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v) |

### Components

| Component | Parameters | Role |
|-----------|-----------|------|
| Qwen2.5-VL-7B | 7B | Multi-modal text encoder (MLLM) |
| ByT5-small | 217M | Byte-level text encoder for glyph/text rendering |
| Token refiner | 141M | Refines MLLM embeddings conditioned on timestep |
| Transformer backbone | 8.3B | 54-block MMDiT, denoises latent video (TP=8) |
| VAE decoder | 871M | 3D Conv decoder, converts latents to 480p video |

## Neuron Port

This is a complete port of HunyuanVideo-1.5 for AWS Neuron (Trainium/Inferentia). All components run on NeuronCores — there is no CPU fallback.

### What was needed to make it work

The transformer backbone and text encoders compiled for Neuron without modification. The VAE decoder required three model rewrites to work around compiler limitations:

1. **CausalConv3d padding**: Replaced `replicate` pad with `constant` pad to avoid an XLA concatenation op that exceeded on-chip SBUF memory.
2. **DCAE upsample**: Decomposed an 8-dimensional tensor reshape into three sequential 6D operations to avoid an XLA tracer crash.
3. **Mid-block attention**: Added spatial 2× downsampling before attention and bilinear upsampling after, reducing the token count from 10,800 to 2,700 to stay within the compiler's instruction limit.

Additionally, `torch.sort` (used in token reordering) is unsupported on trn2 and was replaced with `torch.topk` with a position-preserving tiebreaker.

Details: see `PERFORMANCE.md` for benchmarks, source code in `modeling_hunyuan_video15_vae.py`.

## Performance

### Neuron vs CPU

Measured on trn2.48xlarge. Transformer backbone at TP=8 (8 NeuronCores). CPU estimates based on 45s per forward pass for the 8.3B backbone (conservative).

| Component | CPU (estimated) | Neuron (TP=8) | Speedup |
|-----------|----------------|---------------|---------|
| Transformer backbone (100 passes) | ~75 min | 2.3 min | **33×** |
| Text encoding | ~2 min | 34s | **3.5×** |
| VAE decode | 99s | 29s | **3.4×** |
| **Total (1.1s 480p video)** | **~79 min** | **3.4 min** | **~23×** |

### TP Scaling

| TP Degree | NeuronCores | Per Forward Pass | 50-step Denoising | vs TP=2 |
|-----------|-------------|-----------------|-------------------|---------|
| 2 | 2 | 3.98s | 6.6 min | 1.0× |
| 8 | 8 | 1.38s | 2.3 min | 2.9× |
| 16 | 16 | 1.40s | 2.3 min | 2.8× |

TP=8 is the sweet spot. TP=16 shows no improvement due to communication overhead at 1 attention head per core.

### Neuron VAE vs CPU VAE

The VAE can optionally run on CPU to skip the 4.5-hour VAE compilation:

| | All Neuron (TP=8) | Neuron + CPU VAE | Δ |
|--|-------------------|-----------------|---|
| Denoising | 137s | 137s | — |
| VAE decode | 29s | 99s | +70s |
| **Total** | **3.4 min** | **4.5 min** | **+35%** |
| Compile time | ~4.5 hrs | ~10 min | **96% less** |

### Compilation Time

| Component | Time |
|-----------|------|
| Transformer backbone (TP=8) | ~5 min |
| Qwen2.5-VL encoder (TP=2) | ~2 min |
| ByT5 + refiner + reorder | ~2 min |
| Weight extraction | ~5s |
| **Subtotal (no VAE)** | **~10 min** |
| VAE decoder (24 shards) | ~4.5 hrs |
| **Total** | **~4.5 hrs** |

## Getting Started

### Prerequisites

- **Instance**: trn2.48xlarge (or trn1.32xlarge — NEFFs must be recompiled per architecture)
- **OS**: Ubuntu 22.04

### 1. Install Neuron SDK

```bash
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com jammy main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

sudo apt-get update -y
sudo apt-get install -y aws-neuronx-dkms aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools

pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    torch-neuronx neuronx-cc neuronx-distributed neuronx-distributed-inference \
    transformers diffusers accelerate huggingface_hub sentencepiece protobuf
```

Verify: `neuron-ls` should show 16 NeuronDevices.

### 2. Download Model Weights (~32 GB)

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v')
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')
snapshot_download('google/byt5-small')
"
```

### 3. Compile

```bash
# Everything (~4.5 hrs)
python3 compile_all.py --output_dir .

# Everything except VAE (~10 min) — use --vae_on_cpu at inference
python3 compile_all.py --output_dir . --skip_vae

# Individual components
python3 compile_all.py --output_dir . --only transformer   # ~5 min
python3 compile_all.py --output_dir . --only qwen          # ~2 min
python3 compile_all.py --output_dir . --only traced        # ~2 min
python3 compile_all.py --output_dir . --only weights       # ~5s
python3 compile_all.py --output_dir . --only vae           # ~4.5 hrs
```

### 4. Run Inference

```bash
# All on Neuron
python3 run_inference.py --prompt "A cat walking on a beach at sunset, realistic, 4K"

# With CPU VAE (if VAE shards not compiled)
python3 run_inference.py --prompt "A cat walking on a beach at sunset" --vae_on_cpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | (see source) | Text description of the video |
| `--output_dir` | `pipeline_output` | Output directory for PNG frames |
| `--steps` | 50 | Denoising steps |
| `--cfg` | 3.0 | Classifier-free guidance scale |
| `--seed` | 42 | Random seed |
| `--compiled_dir` | (script dir) | Path to compiled NEFFs |
| `--vae_on_cpu` | off | Run VAE on CPU (no VAE compilation needed) |

Output: 33 PNG frames at 480×640 in the output directory.

## File Reference

| File | Description |
|------|-------------|
| `run_inference.py` | End-to-end inference pipeline |
| `compile_all.py` | Compiles all models into Neuron NEFFs |
| `modeling_hunyuan_video15_transformer.py` | Transformer backbone (NxDI, TP) |
| `modeling_hunyuan_video15_vae.py` | VAE decoder (patches + 24-shard compilation + runtime) |
| `modeling_hunyuan_video15_text.py` | ByT5 encoder, token refiner, token reorder |
| `modeling_qwen2vl_encoder.py` | Qwen2.5-VL text encoder (NxDI, TP) |
| `neuron_full_backbone.py` | TP transformer block implementation |
| `PERFORMANCE.md` | Detailed benchmarks and TP scaling analysis |
