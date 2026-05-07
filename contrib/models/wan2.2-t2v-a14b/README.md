# Contrib Model: Wan 2.2 T2V-A14B

NeuronX Distributed Inference implementation of [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) for text-to-video generation.

![Sample output](assets/sample_output.png)
*"A cat walking on a beach at sunset" — frame 6 of 13*

## Model Information

- **HuggingFace ID:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- **Model Type:** Diffusion Transformer (T2V)
- **Parameters:** 14.29B (×2 backbones: T1 high-noise, T2 low-noise)
- **License:** Apache 2.0
- **Framework:** HuggingFace Diffusers

## Architecture Details

| Property | Value |
|----------|-------|
| Backbone | WanTransformer3DModel |
| Num Layers | 40 per backbone |
| Num Attention Heads | 40 |
| Hidden Dim | 5120 |
| FFN Dim | 13824 |
| Head Dim | 128 |
| Patch Size | (1, 2, 2) |
| Text Encoder | UMT5 (4096-dim) |
| VAE | AutoencoderKLWan |
| Scheduler | UniPCMultistepScheduler |
| Parallelism | Context Parallel (CP=4) |
| Precision | BF16 |

## Neuron Core Usage

| Component | Cores | HBM/Core | Role |
|-----------|-------|----------|------|
| T1 first half (blocks 0–19) | 0–3 | ~17 GB | High-noise denoising (t=999→875) |
| T1 second half (blocks 20–39) | 4–7 | ~16 GB | High-noise denoising (t=999→875) |
| T2 first half (blocks 0–19) | 8–11 | ~17 GB | Low-noise denoising (t=875→0) |
| T2 second half (blocks 20–39) | 12–15 | ~16 GB | Low-noise denoising (t=875→0) |
| Text encoder (UMT5, TP=2) | 16–17 | ~13 GB | Prompt encoding |
| VAE decoder (hybrid) | 20–21 | ~17 GB | Latent → pixels |
| **Total** | **19 cores** | **~300 GB** | |

Instance: trn2.48xlarge (64 cores, 6 TB HBM). 45 cores idle.

## Performance

| Metric | Value |
|--------|-------|
| Denoising (50 steps) | 340s (6.8s/step) |
| VAE decode | 53s |
| Total pipeline | ~7 min |
| Video output | 13 frames, 480×832, RGB |

## Equivalence

Tested against CPU fp32 reference (HF Diffusers `WanTransformer3DModel`):

| Check | Result |
|-------|--------|
| Overall cosine (video) | 0.982 |
| Text encoder (per-component) | 1.000 |
| Transformer T1 (single step) | 0.999 |
| Transformer T2 (single step) | 0.999 |
| VAE decoder | 0.999 |
| Trajectory stability | No divergence |
| Semantic (5 prompts) | 5/5 pass |

## Quick Start

### 1. Download weights

```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers --cache-dir /mnt/work/.cache
```

### 2. Compile

```bash
# Backbone (4 halves, ~25 min total)
for HALF in first second; do
  for SUB in transformer transformer_2; do
    NEURON_RT_VISIBLE_CORES=0-3 HALF=$HALF TRANSFORMER_SUBFOLDER=$SUB \
      python3 scripts/compile_backbone.py
  done
done

# VAE decoder (~2 hours)
NEURON_RT_VISIBLE_CORES=20-21 python3 scripts/compile_vae.py
```

### 3. Run inference

```bash
python3 scripts/run_inference.py
```

Output frames saved to `neuron_output_allcp/frames/`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAN_MODEL_PATH` | HF cache path | HF model directory |
| `WAN_PORT_DIR` | Project root | Compiled NEFFs location |
| `COMPILED_PATH` | `<PORT_DIR>/compiled_cp_*` | Per-half NEFF output |
| `VAE_SAVE_DIR` | `<PORT_DIR>/compiled_vae_new` | Compiled VAE output |
| `CACHE_DIR` | `/mnt/work/.cache` | HuggingFace cache |

## Testing

```bash
pytest contrib/models/wan2.2-t2v-a14b/test/integration/ -v --capture=tee-sys
```

## Maintainer

Neuron Agentic Development — Annapurna Labs

**Last Updated:** 2026-05-07
