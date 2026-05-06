# Contrib Model: S3Diff

S3Diff one-step 4x super-resolution on AWS Neuron using `torch_neuronx.trace()`.

## Model Information

- **HuggingFace ID:** `zhangap/S3Diff` (weights), base model `stabilityai/sd-turbo`
- **Model Type:** One-step diffusion model for image super-resolution
- **Parameters:** ~1.3B total (~2 GB on disk)
- **Architecture:** SD-Turbo UNet with degradation-guided dynamic LoRA modulation (DEResNet encoder, CLIP text encoder, VAE encoder/decoder with per-layer LoRA, UNet with per-layer LoRA)
- **Paper:** "Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors" (ECCV 2024)
- **License:** Check model cards for SD-Turbo and S3Diff

## Key Architecture Notes

S3Diff is unusual among diffusion models:

1. **Single denoising step**: Only one UNet forward pass per image (at t=999), making it extremely fast.
2. **Dynamic LoRA modulation**: A DEResNet encoder estimates input degradation and produces per-layer LoRA scaling matrices. These `[rank, rank]` modulation matrices are injected between `lora_A` and `lora_B` via einsum operations, conditioning the UNet on the specific degradation pattern of each input.
3. **Two LoRA ranks**: VAE uses rank=16 (6 blocks), UNet uses rank=32 (10 blocks).
4. **Small model**: Total size ~2 GB, fits on a single NeuronCore with no tensor parallelism needed.

This contrib uses `torch_neuronx.trace()` rather than NxDI tensor parallelism, which is appropriate for the model's small size and non-autoregressive architecture.

## Validation Results

**Validated:** 2026-04-28
**Instance:** trn2.3xlarge (LNC=2)
**SDK:** Neuron SDK 2.29 (DLAMI 20260410), PyTorch 2.9

### Benchmark Results (128x128 -> 512x512, single step)

| Component | Time |
|-----------|------|
| DEResNet | 3.8ms |
| Modulation (CPU) | 0.5ms |
| VAE Encode | 83.2ms |
| UNet x2 (CFG) | 218.8ms |
| VAE Decode | 164.6ms |
| **Total** | **0.471s** |

| Metric | Value |
|--------|-------|
| Resolution | 128x128 -> 512x512 (4x SR) |
| Inference steps | 1 (one-step model) |
| Warm generation time | 0.544s |
| Throughput | ~1.8 img/s |
| Total compile time | ~21 min |
| CPU baseline | 11.53s |
| Speedup vs CPU | ~21x |

### Accuracy Validation

Visual quality validated against CPU reference output. The model produces high-quality 4x upscaled images with correct degradation-aware enhancement.

## Usage

```python
from S3Diff.src.modeling_s3diff import S3DiffNeuronPipeline
from PIL import Image

pipeline = S3DiffNeuronPipeline(
    sd_turbo_path="/shared/sd-turbo/",
    s3diff_weights_path="/shared/s3diff/s3diff.pkl",
    de_net_path="/shared/s3diff/de_net.pth",
    compile_dir="/tmp/s3diff/compiled/",
    lr_size=128,
)
pipeline.load()
pipeline.compile()

lr_image = Image.open("input_128x128.png").convert("RGB")
sr_image = pipeline(lr_image)
sr_image.save("output_512x512.png")
```

Or use the provided script:

```bash
python src/generate_s3diff.py \
    --download \
    --input_image input.png \
    --output_image output.png \
    --compile_dir /tmp/s3diff/compiled/
```

## Setup

```bash
# Activate NxDI environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install dependencies
pip install diffusers transformers peft accelerate torchvision

# Download weights
python src/generate_s3diff.py --download

# Or manually:
# SD-Turbo: huggingface-cli download stabilityai/sd-turbo --local-dir /shared/sd-turbo/
# S3Diff: huggingface-cli download zhangap/S3Diff --local-dir /shared/s3diff/
# DEResNet: git clone https://github.com/ArcticHare105/S3Diff.git /tmp/s3diff_repo
#           cp /tmp/s3diff_repo/assets/mm-realsr/de_net.pth /shared/s3diff/
```

## Compatibility Matrix

| Instance/Version | SDK 2.29 | SDK 2.28 |
|------------------|----------|----------|
| trn2.3xlarge | VALIDATED | Not tested |

## Example Checkpoints

* [zhangap/S3Diff](https://huggingface.co/zhangap/S3Diff) -- S3Diff LoRA weights
* [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) -- Base SD-Turbo model

## Testing Instructions

```bash
export SD_TURBO_PATH=/shared/sd-turbo/
export S3DIFF_WEIGHTS=/shared/s3diff/s3diff.pkl
export DE_NET_WEIGHTS=/shared/s3diff/de_net.pth

cd contrib/models/S3Diff/
pytest test/integration/test_model.py -v

# Or standalone
python test/integration/test_model.py
```

## Known Issues

- **Fixed resolution only (128x128 -> 512x512)**: This implementation uses `torch_neuronx.trace()` which compiles static tensor shapes. Each input resolution requires a separate compilation. The pipeline is validated at 128x128 input only.
- **Higher resolutions (256->1024, 512->2048, etc.) produce degraded or NaN output** when using trace(). The LoRA modulation einsum operations accumulate BF16 rounding errors at larger spatial dimensions. For multi-resolution or high-resolution (1K/2K/4K) super-resolution, use `torch.compile(backend="neuron")` on the UNet's `Transformer2DModel` blocks instead, with latent tiling for resolutions above 1K. See [xniwangaws/NeuronStuff/s3diff-benchmark](https://github.com/xniwangaws/NeuronStuff/tree/main/s3diff-benchmark) for a validated multi-resolution implementation using PyTorch Native (6.14s @ 1K, 60s @ 2K, 303s @ 4K).
- **LoRA + `--auto-cast=matmult` produces NaN**: The LoRA modulation einsum operations are numerically unstable when `--auto-cast=matmult` casts them to BF16. The VAE encoder, UNet, and VAE decoder all use `--model-type=unet-inference` instead, which avoids this issue at 128x128. Only DEResNet and text encoder (no LoRA) use `--auto-cast=matmult`.
- **Compilation time**: ~21 minutes total (UNet is the slowest at ~12 min). Compiled models are cached for reuse.
- **CFG is sequential**: Two separate UNet passes (positive + negative prompt), not batched. Batching with batch_size=2 would halve UNet wall time but requires recompilation.
- **Neuron runtime HBM**: Once loaded, compiled models stay in HBM even if the Python object is deleted (within the same process). Plan memory accordingly.
