# Contrib Model: V-JEPA 2.1

V-JEPA 2.1 (Video Joint-Embedding Predictive Architecture) is Meta's self-supervised video foundation model. It learns visual representations by predicting masked video segments in a learned representation space, rather than pixel space. This is a vision encoder — not a causal language model — compiled for inference on AWS Trainium via `torch_neuronx.trace()`.

## Model Family

| Model | Source | Params | Instance | Neuron Status |
|-------|--------|--------|----------|---------------|
| **ViT-B/16** | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | 86M | trn2.3xlarge | ✅ Compiled & benchmarked |
| **ViT-L/16** | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | 300M | trn2.3xlarge | ✅ Compiled & benchmarked |
| **ViT-g/16** | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | 1.01B | trn2.48xlarge | ✅ Compiled & benchmarked |
| **ViT-G/16** | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | 1.8B | trn2.48xlarge | ✅ Image compiled; ❌ Video exceeds graph limit |

**License:** [MIT](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE)
**Paper:** [arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985)

## Architecture Details

| Feature | Value |
|---------|-------|
| Type | Vision Transformer encoder (not a causal LM) |
| Patch Embedding | Conv3d tubelets (patch_size=16, tubelet_size=2) |
| Position Encoding | 3D RoPE (separate depth/height/width rotations) |
| Attention | Bidirectional multi-head attention (no KV cache) |
| Normalization | LayerNorm |
| Activation | GELU (encoder), SiLU (predictor) |
| Hierarchical Output | Normed features from 4 intermediate layers |
| Modality Embeddings | Separate learned embeddings for image vs video |
| Image Path | Conv3d with tubelet_size=1 for single-frame input |

### Model Configurations

| Arch | embed_dim | depth | num_heads | head_dim | mlp_ratio | Tokens (16f, 384²) |
|------|-----------|-------|-----------|----------|-----------|---------------------|
| vit_base | 768 | 12 | 12 | 64 | 4.0 | 4,608 |
| vit_large | 1024 | 24 | 16 | 64 | 4.0 | 4,608 |
| vit_giant | 1408 | 40 | 22 | 64 | 48/11 | 4,608 |
| vit_gigantic | 1664 | 48 | 26 | 64 | 64/13 | 4,608 |

### Unique Architecture Features

- **3D RoPE:** Head dimension split into depth/height/width slices (d_dim=h_dim=w_dim=20 for head_dim=64, 4 dims unrotated). Uses `repeat_interleave` layout.
- **Hierarchical output:** Normed features from intermediate layers (e.g., [5,11,17,23] for depth=24). Inference returns only the last layer's normed output by default.
- **Modality embeddings:** Separate learned embeddings added after patch embedding for image vs video inputs.
- **interpolate_rope:** Scales RoPE positions for resolution flexibility beyond the pretrained grid size.
- **Pretrained weight loading:** Checkpoints loaded via `torch.hub.load_state_dict_from_url` from Meta's servers. Distilled models (ViT-B, ViT-L) use key `ema_encoder`; self-supervised (ViT-g, ViT-G) use `target_encoder`.

## Test Results

### Unit Tests (CPU)

| Test Module | Tests | Status |
|-------------|-------|--------|
| test_encoder.py — Construction | 4 | 4/4 PASS |
| test_encoder.py — Forward | 6 | 6/6 PASS |
| test_encoder.py — Components | 4 | 4/4 PASS |
| **Total** | **14** | **14/14 PASS** |

### Integration Tests (trn2.3xlarge, 2 NeuronCores)

| Test | Status | Notes |
|------|--------|-------|
| Trace ViT-B image (1 frame) | PASS | Output shape (1, 576, 768) |
| Trace ViT-B video (16 frames) | PASS | Output shape (1, 4608, 768) |
| Neuron vs CPU accuracy (ViT-B) | PASS | `neuron_allclose` rtol=0.01 |
| Trace ViT-L image (1 frame) | PASS | Output shape (1, 576, 1024) |

### Pretrained Weight Validation (BF16 Neuron vs FP32 CPU)

Validated with official Meta pretrained weights downloaded from `dl.fbaipublicfiles.com/vjepa2/`. Cosine similarity measured between BF16 Neuron output and FP32 CPU reference on identical inputs (seed=42).

| Model | Input | Cosine Similarity | Status |
|-------|-------|-------------------|--------|
| ViT-B (86M) | Image (1×3×1×384×384) | 0.9999 | PASS |
| ViT-B (86M) | Video (1×3×16×384×384) | 1.0000 | PASS |
| ViT-L (300M) | Image (1×3×1×384×384) | 0.9999 | PASS |
| ViT-L (300M) | Video (1×3×16×384×384) | 1.0002 | PASS |
| ViT-g (1.01B) | Image (1×3×1×384×384) | 0.9999 | PASS |
| ViT-g (1.01B) | Video (1×3×16×384×384) | 1.0001 | PASS |
| ViT-G (1.8B) | Image (1×3×1×384×384) | 0.9998 | PASS |

No NaN or Inf values in any output. Feature statistics (mean, std, norm) match closely between CPU and Neuron.

## Performance Benchmarks

**Pretrained weights, BF16, 384×384, `torch_neuronx.trace()` with `--auto-cast none`.** ViT-B/L on trn2.3xlarge (2 NeuronCores); ViT-g/G on trn2.48xlarge (2 NeuronCores). All measurements from 100 timed iterations after 10 warmup runs.

### Single NeuronCore Latency (batch=1)

| Model | Input | Median (ms) | p5 (ms) | p95 (ms) |
|-------|-------|-------------|---------|----------|
| ViT-B (86M) | Image (1 frame) | 4.4 | 4.4 | 4.5 |
| ViT-B (86M) | Video (16 frames) | 247.4 | 247.3 | 248.3 |
| ViT-L (300M) | Image (1 frame) | 11.6 | 11.6 | 11.7 |
| ViT-L (300M) | Video (16 frames) | 741.8 | 741.5 | 742.5 |
| ViT-g (1.01B) | Image (1 frame) | 28.0 | 27.9 | 28.1 |
| ViT-g (1.01B) | Video (16 frames) | 1029.5 | 1029.4 | 1029.7 |
| ViT-G (1.8B) | Image (1 frame) | 49.8 | 49.8 | 49.9 |

Sub-millisecond variance — deterministic Neuron execution.

### DataParallel Throughput (2 NeuronCores)

| Model | Input | Per-clip Latency | Throughput |
|-------|-------|-----------------|------------|
| ViT-B (86M) | Image | 5.2 ms | 383 clips/sec |
| ViT-B (86M) | Video (16f) | 249.5 ms | 8.0 clips/sec |
| ViT-L (300M) | Image | 12.5 ms | 160 clips/sec |
| ViT-L (300M) | Video (16f) | 744.1 ms | 2.7 clips/sec |
| ViT-g (1.01B) | Image | 29.1 ms | 68.8 clips/sec |
| ViT-g (1.01B) | Video (16f) | 1032.1 ms | 1.9 clips/sec |
| ViT-G (1.8B) | Image | 51.1 ms | 39.1 clips/sec |

### Real-Time Video Processing (16 frames @ 30fps = 0.53s of video)

| Model | Single NC | DataParallel (2 NCs) |
|-------|-----------|----------------------|
| ViT-B | 2.1x real-time | 4.3x real-time |
| ViT-L | 0.7x real-time | 1.4x real-time |
| ViT-g | 0.5x real-time | 1.0x real-time |

### Timing Summary

| Operation | Time |
|-----------|------|
| ViT-B compilation | ~8 min |
| ViT-L compilation | ~18 min |
| ViT-g compilation (image) | ~7 min |
| ViT-g compilation (video) | ~51 min |
| ViT-G compilation (image) | ~11 min |
| ViT-B video inference (single NC) | 247.4 ms |
| ViT-L video inference (single NC) | 741.8 ms |
| ViT-g video inference (single NC) | 1029.5 ms |

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

encoder = build_vjepa21_encoder(
    arch="vit_large", img_size=384, num_frames=16,
    use_sdpa=False, pretrained=False,
)
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

### With Pretrained Weights

```python
encoder = build_vjepa21_encoder(arch="vit_large", img_size=384, num_frames=16, pretrained=True)
# Downloads from https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
```

## Demos

### Neuron Smoke Test (`demo_neuron.py`)

Runs pretrained ViT-B on both CPU (FP32) and Neuron (BF16), compares feature embeddings. Serves as a quick validation that the Neuron port is working correctly. No external dependencies beyond torch-neuronx.

```bash
# On a Neuron instance (trn2/inf2):
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python demo_neuron.py                        # synthetic video (no extra deps)
python demo_neuron.py path/to/video.mp4      # your own video (needs decord, pillow)
```

Expected output:
```
Using synthetic video (moving circle, no dependencies needed)
Input shape: torch.Size([1, 3, 16, 384, 384])

Loading pretrained ViT-B (CPU, FP32)...
CPU output: shape=torch.Size([1, 4608, 768]), norm=2042.1

Tracing for Neuron (BF16)...
Compilation: 416.1s
Neuron output: shape=torch.Size([1, 4608, 768]), norm=2046.2
Latency: 248.0ms

Cosine similarity (CPU FP32 vs Neuron BF16): 1.000502  [PASS]
```

### Video Classification (`demo_classify.py`)

Classifies a video using a finetuned V-JEPA 2 model on Something-Something v2 (174 action classes). Runs on CPU — no Neuron hardware needed.

```bash
pip install transformers accelerate torchvision decord
python demo_classify.py                          # Big Buck Bunny sample (CC-BY-3.0)
python demo_classify.py path/to/video.mp4        # your own video
```

Note: This demo uses the HuggingFace `VJEPA2ForVideoClassification` model (V-JEPA 2, not 2.1) to demonstrate what the encoder features can do. The Neuron port (`modeling_jepa21.py`) is the V-JEPA 2.1 encoder only.

## Caveats

1. **`use_sdpa=False` required** — `F.scaled_dot_product_attention` is not supported by `torch_neuronx.trace()`. Must use manual attention path.

2. **BF16 model and inputs required** — Cast model with `.bfloat16()` and use BF16 input tensors. Use `--auto-cast none` compiler flag.

3. **ViT-g/ViT-G require trn2.48xlarge for compilation** — neuronx-cc uses >130GB host RAM for 40+ layer graphs, exceeding trn2.3xlarge's 124GB. Compiled models run on any trn2 instance (inference uses <30GB host RAM).

4. **NKI flash attention slower at short sequences** — The `attention_isa_kernel` is optimized for 16K+ tokens. At 4,608 tokens (16 frames), it's ~80% slower than compiler-generated attention. Use `use_nki_flash=False` for 16-frame inference.

5. **BF16 softmax dtype** — Softmax promotes BF16→FP32 on CPU, causing dtype mismatch. Fixed with `.to(v.dtype)` after softmax (already handled in code).

6. **Not a causal LM** — This is a vision encoder. It does not use NxDI's `NeuronBaseForCausalLM`, KV cache, token generation, or vLLM integration. Compilation uses `torch_neuronx.trace()` directly.

7. **ViT-G video exceeds single-graph instruction limit** — ViT-G with 16 frames generates 17.8M instructions, exceeding neuronx-cc's 10M limit (`NCC_EXTP004`). Requires `parallel_model_trace` to split across NeuronCores. ViT-G image (1 frame) compiles fine.

## Compatibility Matrix

| Instance | NeuronCores | Status | Notes |
|----------|-------------|--------|-------|
| trn2.3xlarge | 2 | **PASS** | ViT-B and ViT-L |
| trn2.48xlarge | 64 | **PASS** | ViT-g and ViT-G (compilation requires >130GB RAM) |

### Minimum Requirements (ViT-L)

| Resource | Requirement |
|----------|------------|
| HBM | 96 GB (1 Neuron device) |
| System RAM | 124 GB (compilation) |
| Instance | trn2.3xlarge |
| Compiled model size | 1.1 GB (ViT-L .pt file) |

### SDK Configuration

| Component | Version |
|-----------|---------|
| torch-neuronx | 2.9.0 |
| neuronx-cc | 2.24.5133 |
| NxDI | 0.9.17334 |
| Python | 3.12 |
| Venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` |

## Testing

### Unit Tests (CPU only, no device needed)

```bash
cd contrib/models/jepa-2-1/
pytest test/unit/ -v
```

Tests: construction (4), forward pass (6), components (4) = **14 tests**.

### Integration Tests (needs Neuron hardware)

```bash
cd contrib/models/jepa-2-1/
pytest test/integration/test_model.py -v
```

Tests: trace ViT-B image/video, Neuron vs CPU accuracy, trace ViT-L image = **4 tests**.

## Key Porting Challenges

1. **Self-contained port:** All upstream imports from `vjepa2` replaced with inline implementations. No dependency on the upstream repo at runtime.

2. **3D RoPE with `repeat_interleave`:** V-JEPA 2.1 uses `repeat_interleave` (not `repeat`) for RoPE frequency expansion. This compiles natively on Neuron.

3. **Conv3d tubelet embedding:** 3D convolution for video patch embedding compiles natively — no decomposition into 2D convolutions needed.

4. **Modular compilation markers:** `ModuleMarkerStartWrapper`/`EndWrapper` from NxDI added for future graph splitting. Currently only respected by `parallel_model_trace`, not `torch_neuronx.trace()`.

5. **NKI flash attention integration:** Integrated `attention_isa_kernel` with correct tensor layouts (q/k: `(B*H, d, seqlen)`, v: `(B*H, seqlen, d)`). Works correctly but slower at short sequences.

## Example Checkpoints

Pretrained weights are downloaded automatically when `pretrained=True`:

| Arch | Checkpoint | Size | Key |
|------|-----------|------|-----|
| vit_base | `vjepa2_1_vitb_dist_vitG_384.pt` | ~350 MB | `ema_encoder` |
| vit_large | `vjepa2_1_vitl_dist_vitG_384.pt` | ~1.2 GB | `ema_encoder` |
| vit_giant | `vjepa2_1_vitg_384.pt` | ~4 GB | `target_encoder` |
| vit_gigantic | `vjepa2_1_vitG_384.pt` | ~7 GB | `target_encoder` |

Source: `https://dl.fbaipublicfiles.com/vjepa2/`

## Maintainer

Community contribution

**Last Updated:** 2026-04-29
