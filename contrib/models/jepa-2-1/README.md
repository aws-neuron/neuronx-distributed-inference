# V-JEPA 2.1 on AWS Trainium

V-JEPA 2.1 (Video Joint-Embedding Predictive Architecture) is Meta's self-supervised video foundation model. It learns visual representations by predicting masked video segments in a learned representation space, rather than pixel space. V-JEPA 2.1 extends V-JEPA 2 with knowledge distillation from a ViT-Gigantic teacher.

This port targets inference on AWS Trainium (trn2) using `torch_neuronx.trace()`.

## Model Information

- **Source**: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- **Paper**: [arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985)
- **License**: MIT

| Model | Params | Depth | Heads | Resolution | Neuron Status |
|-------|--------|-------|-------|------------|---------------|
| ViT-B/16 | 86M | 12 | 12 | 384 | ✅ Compiled & benchmarked |
| ViT-L/16 | 300M | 24 | 16 | 384 | ✅ Compiled & benchmarked |
| ViT-g/16 | 1B | 40 | 22 | 384 | ❌ Host OOM during compilation (needs >124GB RAM) |
| ViT-G/16 | 1.8B | 48 | 26 | 384 | ❌ Not attempted (blocked by ViT-g) |

## Benchmark Results

trn2.3xlarge, BF16, 16 frames, 384×384, `torch_neuronx.trace()` with `--auto-cast none`:

| Model | Single-core Latency | Cosine Sim vs CPU | DataParallel (2 NCs) | Throughput |
|-------|--------------------|--------------------|----------------------|------------|
| ViT-B (86M) | 164.5 ms | 0.9998 | 83.2 ms/clip | 12.0 clips/sec |
| ViT-L (300M) | 437.4 ms | 0.9999 | 219.8 ms/clip | 4.5 clips/sec |

Real-time video processing (16 frames @ 30fps = 0.53s of video):
- ViT-B: **3.2x real-time** (single-core), **6.4x real-time** (DataParallel)
- ViT-L: **1.2x real-time** (single-core), **2.4x real-time** (DataParallel)

## Usage

### CPU Inference

```python
from src.modeling_jepa21 import build_vjepa21_encoder
import torch

encoder = build_vjepa21_encoder(arch="vit_large", img_size=384, num_frames=16, pretrained=False)
encoder.eval()

video = torch.randn(1, 3, 16, 384, 384)
with torch.no_grad():
    features = encoder(video)  # (1, 4608, 1024)
```

### Neuron Compilation

```python
import torch, torch_neuronx
from src.modeling_jepa21 import build_vjepa21_encoder

encoder = build_vjepa21_encoder(arch="vit_large", img_size=384, num_frames=16, use_sdpa=False)
encoder.eval().bfloat16()

x = torch.randn(1, 3, 16, 384, 384, dtype=torch.bfloat16)
traced = torch_neuronx.trace(encoder, x, compiler_args=["--auto-cast", "none"])
traced.save("vjepa21_vitl_16f_384.pt")
```

### DataParallel (2x throughput on trn2.3xlarge)

```python
import torch, torch_neuronx

traced = torch.jit.load("vjepa21_vitl_16f_384.pt")
model_dp = torch_neuronx.DataParallel(traced)

batch = torch.randn(4, 3, 16, 384, 384, dtype=torch.bfloat16)
output = model_dp(batch)  # distributes across 2 NeuronCores
```

## Key Requirements for Neuron Compilation

- `use_sdpa=False` — SDPA is not supported by `torch_neuronx.trace()`
- `.bfloat16()` model and inputs — trn2 NeuronCores are optimized for BF16
- `--auto-cast none` — avoids unpredictable compiler auto-cast behavior
- Conv3d, `torch.arange`, `repeat_interleave` all compile natively

## Known Issues

- ViT-g (1B) and ViT-G (1.8B) cannot compile on trn2.3xlarge (124GB RAM) — the neuronx-cc compiler OOMs on the 40+ layer graph. NxDI `ModuleMarkerStartWrapper`/`EndWrapper` markers were added but `torch_neuronx.trace()` does not respect them for graph splitting. Next step: use `parallel_model_trace` from NxD, or compile on a larger instance.
- NKI flash attention (`attention_isa_kernel`) is integrated but slower than compiler-generated attention at 4608 tokens. Reserved for 64-frame inference (18K tokens) where it will be beneficial.
- BF16 softmax promotes to FP32 on CPU; `.to(v.dtype)` cast added after softmax.

## Testing

```bash
# CPU-only tests
pytest test/ -v

# On Trainium instance
pytest test/integration/test_model.py -v
```
