## Description

NxDI contrib implementation of [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2), Meta's self-supervised video foundation model. V-JEPA 2.1 is a Vision Transformer encoder that learns visual representations by predicting masked video segments in representation space. This is a vision encoder — not a causal language model — compiled for inference on AWS Trainium via `torch_neuronx.trace()`.

Key architecture features ported:
* **3D RoPE:** Separate depth/height/width rotations on head_dim slices, using `repeat_interleave` layout
* **Conv3d tubelet embedding:** 3D convolution for video patch embedding (patch_size=16, tubelet_size=2)
* **Hierarchical output:** Normed features from 4 intermediate layers
* **Modality embeddings:** Separate learned embeddings for image vs video inputs
* **NKI flash attention:** Integrated `attention_isa_kernel` (reserved for 64-frame / 18K token inference)
* **Modular compilation markers:** `ModuleMarkerStartWrapper`/`EndWrapper` for future graph splitting

## Model Information
* **Model Name:** V-JEPA 2.1 (vit_base, vit_large, vit_giant, vit_gigantic)
* **Model Architecture:** Vision Transformer encoder with 3D RoPE (86M–1.8B params)
* **Purpose:** Self-supervised video representation learning (feature extraction, not text generation)
* **Source:** [https://github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
* **License:** [MIT](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE)

## Checklist

### Required Components
* **Accuracy Test** (`test/integration/test_model.py`)
  * Integration test validates Neuron vs CPU accuracy via `neuron_allclose` (rtol=0.01)
  * Test can compile and run the model on Neuron (validated on trn2.3xlarge)
  * Pretrained weight validation: cosine similarity 0.9998–1.0002 across all configurations (ViT-B/L/g/G, image/video)
* **README.md** with the following sections:
  * Usage Example: CPU inference, Neuron compilation, DataParallel
  * Compatibility Matrix: trn2.3xlarge with SDK 2.28
  * Example Checkpoints: Meta's pretrained weights (auto-download)
  * Testing Instructions: Commands to run unit and integration test suites
  * Performance Benchmarks: Latency, throughput, DataParallel scaling
* **Source Code** (`src/`)
  * `modeling_jepa21.py` (~700 lines): Self-contained encoder implementation, no upstream imports
  * Properly structured in the contrib folder hierarchy

### Optional Components
* **Unit Tests** (CPU-based, no Neuron device required)
  * `test_encoder.py` — Construction: 4/4 PASS (ViT-B/L/g construction, invalid arch)
  * `test_encoder.py` — Forward: 6/6 PASS (video/image/batch shapes, hierarchical output, determinism, resolution)
  * `test_encoder.py` — Components: 4/4 PASS (PatchEmbed3D, RoPEAttention, Block)

### Not Applicable (vision encoder, not causal LM)
* vLLM Integration — not applicable (not a text generation model)
* TPOT/TTFT benchmarks — not applicable (no token generation)
* Logit divergence test — not applicable (no autoregressive decoding)
* On-device sampling — not applicable

### Demos
* **`demo_neuron.py`** — Neuron smoke test: runs pretrained ViT-B on CPU (FP32) and Neuron (BF16), compares embeddings. Cosine similarity 1.0005 [PASS]. Compilation 416s, latency 248ms.
* **`demo_classify.py`** — CPU video classification demo using HuggingFace V-JEPA 2 finetuned on SSv2 (174 action classes). Downloads Big Buck Bunny (CC-BY-3.0) as sample.

## Folder Structure

```
contrib/models/jepa-2-1/
├── README.md
├── PR_README.md                      # PR description (paste into GitHub PR body)
├── AGENT.md                          # Technical reference for coding agents
├── demo_neuron.py                    # Neuron smoke test (pretrained ViT-B, CPU vs Neuron)
├── demo_classify.py                  # CPU video classification demo (HF V-JEPA 2 + SSv2)
├── pyproject.toml
├── src/
│   ├── __init__.py
│   └── modeling_jepa21.py            # Self-contained encoder (3D RoPE, Conv3d, NKI flash)
└── test/
    ├── __init__.py
    ├── unit/
    │   ├── __init__.py
    │   └── test_encoder.py           # CPU-only: construction, forward, components
    └── integration/
        ├── __init__.py
        └── test_model.py             # Neuron: trace, accuracy, ViT-B/L
```

## Testing

### How to run the test suite

**Unit tests (CPU only, no Neuron device needed):**

```bash
cd contrib/models/jepa-2-1/
pytest test/unit/ -v
```

Expected: **14/14 PASS** (construction: 4, forward: 6, components: 4)

**Integration tests (needs Neuron hardware, trn2.3xlarge):**

```bash
cd contrib/models/jepa-2-1/
pytest test/integration/test_model.py -v
```

Expected: **4/4 PASS** (trace ViT-B image/video, Neuron vs CPU accuracy, trace ViT-L image)

### Accuracy validation

Neuron output is validated against CPU reference using `neuron_allclose`:

```python
from torch_neuronx.testing.validation import neuron_allclose
result = neuron_allclose(neuron_output, cpu_output, rtol=0.01, atol=1e-5)
assert result.allclose
```

Cosine similarity between BF16 Neuron output and FP32 CPU reference:
- ViT-B: 0.9998
- ViT-L: 0.9999
- ViT-g: 0.9999 (image), 1.0001 (video)
- ViT-G: 0.9998 (image)

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

### Accuracy (BF16 Neuron vs FP32 CPU, pretrained weights)

| Model | Input | Cosine Similarity | Status |
|-------|-------|-------------------|--------|
| ViT-B (86M) | Image (1 frame) | 0.9999 | PASS |
| ViT-B (86M) | Video (16 frames) | 1.0000 | PASS |
| ViT-L (300M) | Image (1 frame) | 0.9999 | PASS |
| ViT-L (300M) | Video (16 frames) | 1.0002 | PASS |
| ViT-g (1.01B) | Image (1 frame) | 0.9999 | PASS |
| ViT-g (1.01B) | Video (16 frames) | 1.0001 | PASS |
| ViT-G (1.8B) | Image (1 frame) | 0.9998 | PASS |

## Performance Benchmarks

**Pretrained weights, BF16, 384×384.** ViT-B/L on trn2.3xlarge; ViT-g/G on trn2.48xlarge. 100 timed iterations after 10 warmup.

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

## Compatibility

Tested with:
* **Neuron SDK Version(s):** 2.28
* **Instance Type(s):** trn2.3xlarge, trn2.48xlarge
* **PyTorch Version:** 2.9.0
* **Python Version:** 3.12
* **neuronx-cc Version:** 2.24.5133
* **NxDI Version:** 0.9.17334
* **torch-neuronx Version:** 2.9.0

| Instance | NeuronCores | Status | Notes |
|----------|-------------|--------|-------|
| trn2.3xlarge | 2 | **PASS** | ViT-B and ViT-L compiled and benchmarked |
| trn2.48xlarge | 64 | **PASS** | ViT-g and ViT-G compiled and benchmarked (ViT-G video exceeds graph limit) |

### Minimum Requirements (ViT-L)

| Resource | Requirement |
|----------|------------|
| HBM | 96 GB (1 Neuron device) |
| System RAM | 124 GB (compilation) |
| Instance | trn2.3xlarge |
| Compiled model | 1.1 GB (.pt file) |

## Additional Information

### Key Porting Challenges
1. **Self-contained port:** All upstream imports from Meta's `vjepa2` repo replaced with inline implementations (~700 lines). No runtime dependency on the upstream repo.
2. **SDPA not supported:** `F.scaled_dot_product_attention` is not supported by `torch_neuronx.trace()`. Replaced with manual `Q @ K^T * scale → softmax → @ V` path.
3. **3D RoPE with `repeat_interleave`:** V-JEPA 2.1 uses `repeat_interleave` (not `repeat`) for RoPE frequency expansion. Compiles natively on Neuron — no workaround needed.
4. **Conv3d tubelet embedding:** 3D convolution compiles natively. No decomposition into 2D convolutions needed.
5. **BF16 softmax dtype:** Softmax promotes BF16→FP32 on CPU, causing dtype mismatch with V tensor. Fixed with `.to(v.dtype)` after softmax.
6. **NKI flash attention integration:** Integrated `attention_isa_kernel` with correct tensor layouts. Works correctly but ~80% slower at 4,608 tokens (designed for 16K+). Reserved for 64-frame inference.
7. **Modular compilation markers:** Added `ModuleMarkerStartWrapper`/`EndWrapper` from NxDI, but `torch_neuronx.trace()` does not respect them for graph splitting. They are only respected by `parallel_model_trace` from NxD.

### Known Limitations
* ViT-g (1B) and ViT-G (1.8B) require trn2.48xlarge for compilation (>130GB host RAM needed); compiled models run on any trn2 instance
* ViT-G video (16 frames) exceeds neuronx-cc's 10M instruction limit (17.8M instructions); requires `parallel_model_trace` to split across NeuronCores
* `use_sdpa=False` is required (SDPA not supported by `torch_neuronx.trace()`)
* NKI flash attention is slower than compiler-generated attention at 4,608 tokens (16 frames)
* Modular compilation markers are not respected by `torch_neuronx.trace()` — need `parallel_model_trace` for graph splitting
* Not a causal LM — no vLLM integration, no KV cache, no token generation
* Pretrained weight download requires network access to `dl.fbaipublicfiles.com`

### Future Work
* ViT-G video (16 frames) via `parallel_model_trace` to split the 17.8M-instruction graph across NeuronCores
* 64-frame inference (18,432 tokens) where NKI flash attention becomes beneficial
* Downstream tasks: attentive pooler for classification, predictor for action anticipation

By submitting this PR, I confirm that:
* I have read and followed the contributing guidelines
* This is a community contribution and may have limited testing compared to officially-supported models
* The code follows best practices and is well-documented
* All required components listed above are included
