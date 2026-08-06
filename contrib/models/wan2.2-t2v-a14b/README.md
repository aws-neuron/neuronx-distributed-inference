# Wan 2.2 T2V-A14B — Neuron Port

Context-parallel (CP=4) port of [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) for AWS Trainium2.

Generates 13-frame 480×832 videos from text prompts on a single trn2.48xlarge instance.

![Sample output](assets/sample_output.png)
*"A cat walking on a beach at sunset" — frame 6 of 13*

## Model

| Property | Value |
|----------|-------|
| Architecture | WanTransformer3DModel (diffusion transformer) |
| Parameters | 14.29B (×2 models: T1 for high-noise, T2 for low-noise timesteps) |
| HF Model ID | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| Framework | HuggingFace Diffusers |
| Precision | BF16 |
| Parallelism | Context Parallel (CP=4), sequence split across ranks |

## Neuron Core Usage

| Component | Cores | HBM/Core | Role |
|-----------|-------|----------|------|
| T1 first half (blocks 0–19) | 0–3 | ~17 GB | High-noise denoising |
| T1 second half (blocks 20–39) | 4–7 | ~16 GB | High-noise denoising |
| T2 first half (blocks 0–19) | 8–11 | ~17 GB | Low-noise denoising |
| T2 second half (blocks 20–39) | 12–15 | ~16 GB | Low-noise denoising |
| Text encoder (UMT5, TP=2) | 16–17 | ~13 GB | Prompt encoding |
| VAE decoder | 20–21 | ~17 GB | Latent → pixels |
| **Total** | **19 cores** | **~300 GB** | |

Instance: trn2.48xlarge (64 cores, 6 TB HBM total). 45 cores idle.

## Performance

| Metric | Value |
|--------|-------|
| Denoising (50 steps) | 340s (6.8s/step) |
| VAE decode | 53s |
| Total pipeline | ~7 min |
| Video output | 13 frames, 480×832, RGB |

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
      python3 compile_backbone.py
  done
done

# VAE decoder (~2 hours)
NEURON_RT_VISIBLE_CORES=20-21 python3 compile_vae.py
```

### 3. Run inference

```bash
python3 run_inference.py
```

Output frames saved to `neuron_output_allcp/frames/`.

## Environment Variables

All paths are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WAN_MODEL_PATH` | `/mnt/work/.cache/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/...` | HF model directory |
| `WAN_PORT_DIR` | `/mnt/work/wan2.2-pr` | Project root (compiled NEFFs, scripts) |
| `COMPILED_PATH` | `<WAN_PORT_DIR>/compiled_cp_<subfolder>_<half>` | Compiled NEFF output |
| `VAE_SAVE_DIR` | `<WAN_PORT_DIR>/compiled_vae_new` | Compiled VAE output |
| `CACHE_DIR` | `/mnt/work/.cache` | HuggingFace cache |

## File Structure

```
├── nxdi_wan/
│   ├── modeling_wan_cp.py       # Neuron CP transformer (475 lines)
│   ├── application_cp.py        # NxDI compile/load/forward wrapper
│   └── application_umt5.py      # Text encoder wrapper
├── compile_backbone.py          # Compile T1/T2 halves
├── compile_vae.py               # Compile VAE decoder
├── worker.py                    # CP worker subprocess
├── vae_decode_hybrid.py         # Hybrid CPU+Neuron VAE decode
├── run_inference.py             # Full pipeline orchestrator
└── assets/
    └── sample_output.png        # Example generated frame
```

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
