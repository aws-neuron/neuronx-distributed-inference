# WAN 2.1 Text-to-Video 1.3B — Neuron Port

NeuronX implementation of [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers).
All neural network components (T5 encoder, transformer backbone, VAE decoder) run on AWS Trainium.

## Prerequisites

- trn2.48xlarge instance with Neuron SDK
- Python packages: `diffusers`, `transformers`, `torch_neuronx`, `neuronx_distributed`
- Model weights downloaded and `WAN_MODEL_DIR` set:
  ```bash
  export WAN_MODEL_DIR=/home/ubuntu/models/wan2.1-t2v-1.3b
  ```

## Compilation

NEFFs must be compiled before inference. Each backbone NEFF is compiled for a fixed frame count.

| Component | Command | Time |
|-----------|---------|------|
| T5 + 13f backbone + VAE | `NEURON_RT_VISIBLE_CORES=0-7 python3 scripts/trace_all.py --component=all` | ~65 min |
| 49f backbone (CP=8) | `NEURON_RT_VISIBLE_CORES=0-7 torchrun --nproc_per_node=8 scripts/compile_backbone_cp8.py` | ~95 min |

To compile individual components:
```bash
# T5 only (~2 min)
NEURON_RT_VISIBLE_CORES=0-3 NEURON_RT_VIRTUAL_CORE_SIZE=1 python3 scripts/trace_all.py --component=t5

# 13f backbone only (~48 min)
NEURON_RT_VISIBLE_CORES=0-3 python3 scripts/trace_all.py --component=backbone

# VAE blocks only (~15 min)
NEURON_RT_VISIBLE_CORES=0-3 python3 scripts/trace_all.py --component=vae
```

## Inference

### 13 Frames (~43s, all on Neuron)

```bash
# Step 1: T5 encoding (LNC=1, separate process)
NEURON_RT_VISIBLE_CORES=0-3 NEURON_RT_VIRTUAL_CORE_SIZE=1 \
  python3 scripts/run_pipeline.py --step=t5 --prompt="a cat on a beach at sunset"

# Step 2: Backbone + VAE decode
NEURON_RT_VISIBLE_CORES=0-3 \
  python3 scripts/run_pipeline.py --step=generate --seed=42
```

Output: 13 frames at 480×832 in `output/`.

### 49 Frames (~102s, no crosshatch)

```bash
# Step 1: T5 encoding (same as above)
NEURON_RT_VISIBLE_CORES=0-3 NEURON_RT_VIRTUAL_CORE_SIZE=1 \
  python3 scripts/run_pipeline.py --step=t5 --prompt="a cat on a beach at sunset"

# Step 2: Backbone denoising (CP=8, cores 0-7)
NEURON_RT_VISIBLE_CORES=0-7 \
  torchrun --nproc_per_node=8 scripts/validate_backbone_cp8.py

# Step 3: VAE decode (cores 8-15, reads latents from /tmp/latents_49f.pt)
NEURON_RT_VISIBLE_CORES=8-15 \
  python3 scripts/decode_vae.py --latents=/tmp/latents_49f.pt --num-frames=13
```

Output: 49 frames at 480×832 in `output_49f/`.

## File Structure

```
├── src/
│   ├── modeling_wan.py          # Standalone model, VAE wrappers, XLA compat
│   └── __init__.py
├── scripts/
│   ├── run_pipeline.py          # E2E 13-frame inference
│   ├── trace_all.py             # Compile T5, backbone, VAE blocks
│   ├── compile_backbone_cp8.py  # Compile 49-frame CP=8 backbone
│   └── validate_backbone_cp8.py # 49-frame backbone denoising
├── compiled_model/              # Pre-compiled NEFFs
│   ├── t5/
│   ├── vae_blocks/              # 55 blocks (frame-count-agnostic)
│   ├── 13f/backbone/
│   └── 49f/backbone_cp8/        # 16 NEFFs (2 per rank × 8)
└── test/integration/test_model.py
```

## Key Concepts

- **Pre-computed RoPE:** XLA silently drops inputs used only for `.shape`. RoPE is computed on CPU and passed as an explicit input.
- **Block-by-block VAE:** The causal 3D VAE can't be traced as one unit. Each block is traced with cache I/O and chained at runtime. VAE blocks are frame-count-agnostic.
- **CP=8 for 49 frames:** The 5M instruction limit prevents compiling the full backbone at seq_len=20,280. Context parallelism splits the sequence across 8 ranks, with the model split into 2 NEFFs of 15 blocks each.
- **Different core sets:** T5 (LNC=1) and backbone (LNC=2) can't share a process. For 49 frames, backbone uses cores 0-7 and VAE uses cores 8-15.
- **Fixed frame count per NEFF:** Backbone NEFFs are shape-specific. Different frame counts require recompilation. This is standard for Neuron/XLA (Flux on NxDI does the same for different resolutions).

## Performance

| Config | Backbone | VAE | Total | Cores | vs CPU |
|--------|----------|-----|-------|-------|--------|
| 13 frames | 36.5s | 6.6s | 43s | 2 | 1.4× |
| 49 frames | 70.7s | 31.8s | 102s | 10 | 13.5× |

## Known Limitations

- **Crosshatch at 13 frames:** VAE temporal upsampling artifact at low frame counts. Use ≥49 frames for clean output.
- **Instruction limit:** Backbone at seq_len>6,240 exceeds 5M limit at TP=1. Requires CP for longer sequences.
- **LNC mismatch:** T5 and backbone must run in separate processes.

## Validation

| Component | Cosine vs CPU |
|-----------|--------------|
| Backbone (single step) | 0.9998 |
| T5 encoder | 0.998 |
| VAE (per block) | ≥0.9998 |
| VAE (full decode) | 1.0006 |

## Compatibility

| Instance | 13 Frames | 49 Frames |
|----------|-----------|-----------|
| trn2.48xlarge | ✅ | ✅ |

**Last Updated:** 2026-03-23
