# V-JEPA 2.1 on AWS Trainium

V-JEPA 2.1 (Video Joint-Embedding Predictive Architecture) is Meta's self-supervised video foundation model. It learns visual representations by predicting masked video segments in a learned representation space, rather than pixel space. V-JEPA 2.1 extends V-JEPA 2 with knowledge distillation from a ViT-Gigantic teacher, enabling smaller student encoders (ViT-Base, ViT-Large) to achieve strong performance.

This port targets inference on AWS Trainium (trn2) using `torch_neuronx.trace()`.

## Model Information

- **Source**: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- **Paper**: [Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- **Model Type**: Self-supervised Vision Transformer (ViT) encoder + predictor
- **Architecture**: ViT with 3D-RoPE, mask-denoising pretraining, hierarchical multi-layer output, modality embeddings (image/video)
- **License**: MIT (vjepa2 repo)

### Available Checkpoints

| Model | Params | Embed Dim | Depth | Heads | Resolution | Teacher |
|-------|--------|-----------|-------|-------|------------|---------|
| V-JEPA 2.1 ViT-B/16 | 86M | 768 | 12 | 12 | 384 | ViT-G distillation |
| V-JEPA 2.1 ViT-L/16 | 300M | 1024 | 24 | 16 | 384 | ViT-G distillation |
| V-JEPA 2.1 ViT-g/16 | 1B | 1408 | 40 | 22 | 384 | Self-supervised |
| V-JEPA 2.1 ViT-G/16 | 1.8B | 1664 | 48 | 26 | 384 | Self-supervised |

## Architecture Overview

V-JEPA 2.1 consists of:

1. **Encoder** (`VisionTransformer`): Processes video frames patchified into 2×16×16 tubelets. Uses 3D-RoPE for spatiotemporal position encoding. Outputs hierarchical features from multiple intermediate layers (e.g., layers [5, 11, 17, 23] for ViT-L depth=24).

2. **Predictor** (`VisionTransformerPredictor`): Takes encoder features + learnable mask tokens and predicts representations of masked patches. Uses multi-layer hierarchical input from the encoder via a learned projection.

3. **Attentive Pooler** (optional, for classification): Cross-attention pooling over encoder features for downstream classification tasks.

Key differences from V-JEPA 2:
- Hierarchical multi-layer output with per-layer norms (`norms_block`)
- Modality embeddings (separate for image vs video input)
- `img_temporal_dim_size` for handling single-frame image inputs with tubelet_size=1
- Distillation-aware predictor with `n_output_distillation` controlling which layers contribute
- `interpolate_rope` for resolution-flexible RoPE

## Inference Approach

For inference on Trainium, we use `torch_neuronx.trace()` on the encoder. The encoder is the primary component needed for downstream tasks (classification, VQA, feature extraction). The predictor is only needed for pretraining and action anticipation tasks.

### Why `torch_neuronx.trace()` (not NxDI)

- The encoder is a standard ViT without KV cache or autoregressive decoding
- At 384×384 resolution with 64 frames: seq_len = (64/2) × (384/16)² = 32 × 576 = 18,432 tokens per clip
- For single-frame image inference: seq_len = 576 tokens (trivially fits)
- For short video clips (16 frames): seq_len = 8 × 576 = 4,608 tokens
- NxDI's KV cache and flash attention infrastructure is unnecessary for non-autoregressive models
- `torch_neuronx.trace()` is simpler and sufficient for encoder-only inference

### Compilation Strategy

- Trace the encoder with a fixed input shape (batch, channels, frames, height, width)
- Use `torch_neuronx.trace()` with example inputs
- For variable-length video, compile multiple buckets or pad to max length

## Usage

```python
import torch
import torch_neuronx

# Load encoder (CPU reference)
from src.modeling_jepa21 import build_vjepa21_encoder

encoder = build_vjepa21_encoder(
    arch="vit_large",
    img_size=384,
    num_frames=16,
    pretrained=False,  # set True when checkpoint available
)
encoder.eval()

# Example: single image input (B, C, T, H, W)
image_input = torch.randn(1, 3, 1, 384, 384)
with torch.no_grad():
    features = encoder(image_input)
# features shape: (1, 576, 1024) for ViT-L

# Example: video input
video_input = torch.randn(1, 3, 16, 384, 384)
with torch.no_grad():
    features = encoder(video_input)
# features shape: (1, 4608, 1024) for ViT-L with 16 frames
```

### Neuron Compilation (on Trainium instance)

```python
import torch
import torch_neuronx
from src.modeling_jepa21 import build_vjepa21_encoder

encoder = build_vjepa21_encoder(arch="vit_large", img_size=384, num_frames=16, use_sdpa=False)
encoder.eval().bfloat16()

example_input = torch.randn(1, 3, 16, 384, 384, dtype=torch.bfloat16)
traced = torch_neuronx.trace(encoder, example_input, compiler_args=["--auto-cast", "none"])
traced.save("vjepa21_vitl_16f_384.pt")
```

## Compatibility Matrix

| Instance | SDK | Model | Frames | Resolution | Dtype | Compile | Cosine Sim | Latency (median) |
|----------|-----|-------|--------|------------|-------|---------|------------|-------------------|
| trn2.3xlarge | 2.27 (torch-neuronx 2.9.0, neuronx-cc 2.24.5133) | ViT-B (86M) | 16 | 384×384 | BF16 | ✅ PASS | 0.9998 | 164.5 ms |
| trn2.3xlarge | 2.27 (torch-neuronx 2.9.0, neuronx-cc 2.24.5133) | ViT-L (300M) | 16 | 384×384 | BF16 | ✅ PASS | 0.9999 | 437.4 ms |
| inf2.xlarge | — | — | — | — | — | Not tested | — | — |

## Example Checkpoints

* V-JEPA 2.1 weights are loaded via `torch.hub` from Meta's servers (see `hubconf.py` in vjepa2 repo)

## Testing Instructions

```bash
# CPU-only tests (runs on MacBook)
cd contrib/models/jepa-2-1
pytest test/ -v

# On Trainium instance
pytest test/integration/test_model.py -v
```

## Known Issues

- `use_sdpa=False` is required for Neuron compilation — `F.scaled_dot_product_attention` is not supported by `torch_neuronx.trace()`. The manual attention fallback (`q @ k.T * scale → softmax → @ v`) works correctly.
- BF16 softmax promotes to FP32 on CPU; `.to(v.dtype)` cast added after softmax to maintain dtype consistency.
- 3D-RoPE uses a duplicated frequency pattern (known upstream bug, preserved for checkpoint compatibility)
- `timm` is required as a dependency for `drop_path` (replaced with inline implementation)
- Full 64-frame ViT-G inference may require TP>1 on Trainium due to memory
- Conv3d, `torch.arange`, and `repeat_interleave` all compile successfully on neuronx-cc 2.24.5133
