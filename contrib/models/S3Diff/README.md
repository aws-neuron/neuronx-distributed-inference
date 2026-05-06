# Contrib Model: S3Diff

S3Diff one-step 4x super-resolution on AWS Neuron using `torch_neuronx.trace()`.
Supports arbitrary input resolutions via tiling with Gaussian blending.

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
5. **Arbitrary resolution via tiling**: All components are compiled at a fixed tile size (512x512 pixels). Images larger than this are split into overlapping tiles, processed independently, and blended with Gaussian weights for seamless output.

This contrib uses `torch_neuronx.trace()` rather than NxDI tensor parallelism, which is appropriate for the model's small size and non-autoregressive architecture.

## Validation Results

**Validated:** 2026-05-06
**Instance:** trn2.3xlarge (LNC=2)
**SDK:** Neuron SDK 2.29 (DLAMI 20260410), PyTorch 2.9

### Benchmark Results (multi-resolution, single step)

| Input Size | Output Size | Tiles | Time | Throughput |
|-----------|-------------|-------|------|------------|
| 128x128 | 512x512 | 1 | 0.545s | 1.8 img/s |
| 256x256 | 1024x1024 | 9 | 4.809s | 0.21 img/s |
| 512x512 | 2048x2048 | 25 | 13.346s | 0.075 img/s |

### Component Timing (single tile, 512x512 output)

| Component | Time |
|-----------|------|
| DEResNet | 3.8ms |
| Modulation (CPU) | 0.5ms |
| VAE Encode | 83.2ms |
| UNet x2 (CFG) | 218.8ms |
| VAE Decode | 164.6ms |
| **Total (single tile)** | **0.471s** |

| Metric | Value |
|--------|-------|
| Inference steps | 1 (one-step model) |
| Total compile time | ~21 min |
| CPU baseline (128->512) | 11.53s |
| Speedup vs CPU | ~21x |

### Accuracy Validation

Visual quality validated against CPU reference output. The model produces high-quality 4x upscaled images with correct degradation-aware enhancement. Tiled outputs show pixel std > 20 (indicating good visual detail) with seamless blending at tile boundaries.

## Usage

```python
from S3Diff.src.modeling_s3diff import S3DiffNeuronPipeline
from PIL import Image

pipeline = S3DiffNeuronPipeline(
    sd_turbo_path="/shared/sd-turbo/",
    s3diff_weights_path="/shared/s3diff/s3diff.pkl",
    de_net_path="/shared/s3diff/de_net.pth",
    compile_dir="/tmp/s3diff/compiled/",
    lr_size=128,       # DEResNet fixed input (always 128)
    tile_size=512,     # HR tile size (default)
    tile_overlap=128,  # Overlap for blending (default)
)
pipeline.load()
pipeline.compile()

# Works with any input size -- tiling is automatic
lr_image = Image.open("input.png").convert("RGB")
sr_image = pipeline(lr_image)  # 4x upscaled output
sr_image.save("output.png")
```

Or use the provided script:

```bash
# 128x128 -> 512x512 (single tile, fast)
python src/generate_s3diff.py \
    --input_image input_128.png \
    --output_image output_512.png

# 256x256 -> 1024x1024 (tiled)
python src/generate_s3diff.py \
    --input_image input_256.png \
    --output_image output_1024.png

# Custom tile settings
python src/generate_s3diff.py \
    --input_image input.png \
    --output_image output.png \
    --tile_size 512 \
    --tile_overlap 128
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

## Tiling Design

For images whose HR output (input x 4) exceeds 512x512 pixels, the pipeline automatically:

1. **Upscales** the input image to HR resolution via bicubic interpolation
2. **Splits** the HR image into overlapping 512x512 tiles (128px overlap by default)
3. **Processes** each tile independently through VAE encode -> UNet -> VAE decode
4. **Blends** tile outputs using Gaussian weights (smooth center-to-edge falloff)

This approach avoids recompilation for different resolutions and produces seamless outputs. The degradation estimation (DEResNet) runs once on the full image resized to 128x128, producing global modulation parameters shared across all tiles.

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

- **LoRA + `--auto-cast=matmult` produces NaN**: The LoRA modulation einsum operations are numerically unstable when `--auto-cast=matmult` casts them to BF16. All components with LoRA use `--model-type=unet-inference` instead. Only DEResNet and text encoder (no LoRA) use `--auto-cast=matmult`.
- **Compilation time**: ~21 minutes total (UNet is the slowest at ~12 min). Compiled models are cached for reuse.
- **CFG is sequential**: Two separate UNet passes (positive + negative prompt), not batched. Batching with batch_size=2 would halve UNet wall time but requires recompilation.
- **Neuron runtime HBM**: Once loaded, compiled models stay in HBM even if the Python object is deleted (within the same process). Plan memory accordingly.
- **Tiling artifacts at very high resolution**: At 4K+ output, very minor blending seams may be visible in uniform regions. Increasing `--tile_overlap` to 192 or 256 reduces this at the cost of more tiles.
