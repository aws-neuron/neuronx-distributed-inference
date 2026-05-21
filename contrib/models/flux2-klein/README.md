# Contrib Model: FLUX.2-klein-base-9B

FLUX.2-klein-base-9B image generation model running on AWS Neuron with NxD Inference tensor parallelism.

## Model Information

- **HuggingFace ID:** `black-forest-labs/FLUX.2-klein-base-9B`
- **Model Type:** Diffusion transformer (DiT) for text-to-image generation
- **Parameters:** ~9.08B (BF16)
- **Architecture:** 8 double-stream MMDiT blocks + 24 single-stream DiT blocks, 4D RoPE, SwiGLU, pre-computed modulation, Qwen3-8B text encoder
- **License:** Check HuggingFace model card (gated model, requires access approval)

## Key Architecture Notes

FLUX.2-klein differs from FLUX.1 in several important ways:

1. **Pre-computed modulation**: Modulation parameters are computed once from the timestep embedding and shared across all blocks (FLUX.1 computes per-block).
2. **Fused QKV+MLP**: Single-stream blocks use `Flux2ParallelSelfAttention` with a fused `to_qkv_mlp_proj` linear. For NxDI, this is split into separate TP-sharded Q/K/V and MLP projections.
3. **SwiGLU activation**: Uses `SiLU(x1) * x2` instead of GELU.
4. **Qwen3-8B text encoder**: Single encoder producing multi-layer hidden states (layers 9, 18, 27 concatenated for `joint_attention_dim=12288`), replacing FLUX.1's CLIP+T5 dual encoder.
5. **32 latent channels** (packed to 128), 4D RoPE with `axes_dims=(32,32,32,32)`, `theta=2000`.
6. **Classic CFG**: Two forward passes (positive + negative prompt) rather than guidance distillation.

## Validation Results

**Validated:** Yes
**Instance:** trn2.3xlarge (LNC=2, 4 logical cores)
**SDK:** Neuron SDK 2.29 (DLAMI 20260410), PyTorch 2.9, NxD Inference 0.9

### Benchmark Results (1024x1024, 30 steps, guidance_scale=4.0)

| Metric | Value |
|--------|-------|
| Resolution | 1024x1024 |
| Inference steps | 30 |
| TP Degree | 4 |
| CFG | Classic (2 forward passes per step) |
| E2E generation time | 31.09s +/- 0.08s |
| Pipeline steps/sec | 0.96 |
| Per-step latency | 1036ms |
| Backbone forward/sec | 1.93 (2 CFG passes/step) |
| Compilation time | ~135s (2.3 min) |
| Model load time | ~20s |

### Accuracy Validation

| Metric | Value |
|--------|-------|
| Backbone cosine similarity (Neuron vs CPU) | 0.9987 |
| Max absolute difference | 0.15 |
| Mean absolute difference | 0.022 |

## Usage

```python
import torch
from flux2_klein.src.application import (
    NeuronFlux2KleinApplication,
    create_flux2_klein_config,
)

MODEL_PATH = "/shared/flux2-klein/"
COMPILE_DIR = "/tmp/flux2_klein/compiled/"

# Configure
backbone_config = create_flux2_klein_config(
    model_path=MODEL_PATH,
    backbone_tp_degree=4,  # trn2.3xlarge LNC=2
    dtype=torch.bfloat16,
    height=1024,
    width=1024,
)

# Create application
app = NeuronFlux2KleinApplication(
    model_path=MODEL_PATH,
    backbone_config=backbone_config,
)

# Compile (first time only, ~2-3 min)
app.compile(COMPILE_DIR)

# Load compiled model
app.load(COMPILE_DIR)

# Generate
result = app(
    prompt="A cat holding a sign that says hello world",
    negative_prompt="",
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=4.0,
)
result.images[0].save("output.png")
```

## Compatibility Matrix

| Instance | SDK 2.29 | SDK 2.28 |
|----------|----------|----------|
| trn2.3xlarge (TP=4, LNC=2) | Validated | Not tested |
| trn2.48xlarge (TP=4, LNC=2) | Not tested | Not tested |

## Example Checkpoints

* [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (gated, requires access approval)

## Testing Instructions

```bash
# Set environment variables
export FLUX2_MODEL_PATH=/shared/flux2-klein/
export FLUX2_COMPILE_DIR=/tmp/flux2_klein_test/
export FLUX2_TP_DEGREE=4

# Run integration tests
cd contrib/models/flux2-klein/
pytest test/integration/test_model.py -v

# Or run standalone
python test/integration/test_model.py
```

## Design Decisions

### Why NxDI instead of torch_neuronx.trace()?

A single `Flux2SingleTransformerBlock` generates 7.4M instructions (exceeding the 5M compiler limit) due to the fused `to_qkv_mlp_proj: Linear(4096, 36864)`. With NxDI tensor parallelism (TP=4), each rank handles only 1/4 of the attention heads and MLP hidden dimensions, bringing the instruction count well within limits.

### Why is the text encoder on CPU?

Qwen3-8B (8.19B parameters) is too large for efficient Neuron compilation as a single traced model. Since it only runs once per image (not in the denoising loop), the ~5s CPU execution time is negligible compared to the 30-step denoising (~31s).

### Why split the fused QKV+MLP projection?

The HuggingFace `Flux2ParallelSelfAttention` uses a single massive `Linear(4096, 36864)` that fuses Q, K, V projections with MLP input. For TP sharding, we split this into separate `ColumnParallelLinear` layers for Q/K/V and MLP, each independently sharded across TP ranks.

### Why split SwiGLU linear_in into gate + value?

FLUX.2-klein uses SwiGLU activation (`SiLU(gate) * value`) in its feed-forward networks, where the HuggingFace implementation fuses gate and value into a single `linear_in` weight of shape `[2*inner_dim, dim]`. With `ColumnParallelLinear`, each TP rank receives a contiguous 1/N partition of the output dimension. If the fused weight is sharded directly, the SwiGLU split on each rank won't correspond to correct gate/value pairs. The fix is to split `linear_in` into two separate `ColumnParallelLinear` layers (`linear_in_gate` and `linear_in_value`), so each is independently and correctly sharded. The same pattern applies to the single block's fused MLP projection.

## Known Issues

- **Gated model**: Requires HuggingFace access approval before use
- **CPU text encoding**: Qwen3-8B runs on CPU (~5s per prompt)
- **First compilation**: Takes ~2-3 minutes on trn2.3xlarge
- **Memory**: The 9B transformer requires all 4 cores with TP=4 on trn2.3xlarge

## Maintainer

Jim Burtoft
