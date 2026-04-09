# Contrib Model: Wan2.2-TI2V-5B

NeuronX adaptation of [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) for AWS Trainium2 inference. Supports text-to-video (T2V) and image-to-video (I2V) generation at multiple resolutions.

## Model Information

- **HuggingFace ID:** `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- **Model Type:** Diffusion model for text/image-to-video generation
- **Architecture:** Multi-component (UMT5 Text Encoder + DiT Transformer + 3D VAE)
- **License:** Check HuggingFace model card

## Architecture Details

| Component | Model | Parameters | Neuron Parallelism |
|-----------|-------|------------|-------------------|
| Text Encoder | UMT5 | ~4.7B | TP=4, world_size=8 |
| Transformer | DiT-based diffusion | ~5B | TP=4, CP=2 or CFG Parallel, world_size=8 |
| VAE Decoder | Conv3D, rolling cache | ~300M | Single device, bfloat16 |
| VAE Encoder | Conv3D (I2V only) | ~300M | CPU (Neuron bug NCC_IBIR158) |

Key parameters:
- **Denoising steps**: 50 (default)
- **Context Parallel (CP)**: Splits sequence across 2 ranks, K/V all-gather in self-attention
- **CFG Parallel**: Splits batch (cond/uncond), no K/V communication, ~13% faster for most resolutions
- **Rolling Cache**: Stateful temporal caching for flicker-free video, ~960MB on-device

## Performance

| Resolution | Frames | Trn2 CP (s) | Trn2 CFG (s) | H100 (s) | Decoder |
|-----------|--------|-------------|--------------|----------|---------|
| 512x384 | 81 | 20.67 | **18.32** | 16.13 | stateful rolling |
| 512x384 | 121 | 30.07 | **26.44** | 24.48 | stateful rolling |
| 640x480 | 81 | **33.20** | 34.10 | 26.06 | stateful rolling |
| 640x480 | 121 | 49.29 | **45.15** | 39.67 | stateful rolling |
| 1280x704 | 81 | 163.99 | **155.06** | 87.66 | tiled |
| 1280x704 | 121 | 255.07 | **243.71** | 143.20 | tiled |

Test: trn2.48xlarge, 50 denoising steps.

## Prerequisites

- **Instance**: trn2.48xlarge (64 NeuronCores, 1.5TB device memory)
- **Virtual env**: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
  - PyTorch 2.9, neuronx-cc 2.22, neuronx-distributed 0.16
- **NVMe**: Mount RAID at `/opt/dlami/nvme/` (run `src/setup_nvme.sh`)

## Usage

### 1. Setup

```bash
# Mount NVMe RAID
sudo bash src/setup_nvme.sh

# Activate virtual environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model

```bash
python src/cache_hf_model.py
```

### 3. Compile All Components

```bash
# Context Parallel (default)
bash src/compile.sh

# CFG Parallel (recommended, faster for most resolutions)
CFG_PARALLEL=1 bash src/compile.sh

# Custom output directory
bash src/compile.sh /path/to/output /path/to/compiler_workdir
```

Compilation takes ~30-60 minutes.

### 4. Run Inference

```bash
# Text-to-Video (T2V)
NEURON_RT_NUM_CORES=8 PYTHONPATH=src:$PYTHONPATH python src/run_wan2.2_ti2v.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b \
    --prompt "A cat walks on the grass, realistic" \
    --output output.mp4

# Image-to-Video (I2V)
NEURON_RT_NUM_CORES=8 PYTHONPATH=src:$PYTHONPATH python src/run_wan2.2_ti2v.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b \
    --image assets/cat.png \
    --prompt "A cat walks on the grass, realistic" \
    --output output_i2v.mp4
```

## Compatibility Matrix

| Instance/Version | 2.22+ (PyTorch 2.9) | 2.21 and earlier |
|------------------|---------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested | Not tested |
| Trn1 | Not tested | Not tested |
| Inf2 | Not supported | Not supported |

## Testing

```bash
# Run integration tests
PYTHONPATH=src:$PYTHONPATH pytest test/integration/ --capture=tee-sys -v
```

## Key Implementation Notes

1. **Context Parallel & CFG Parallel**: Two parallelism strategies for the transformer. CFG Parallel batches cond+uncond prompts into single forward pass, avoiding K/V all-gather.
2. **local_rms_norm**: Workaround for Neuron compiler bug with DistributedRMSNorm. Computes RMSNorm locally on each rank's shard.
3. **Stateful Rolling Cache**: VAE decoder's 34 `feat_cache` tensors stay on-device (HBM) between calls via input-output aliasing, eliminating ~960MB host-device roundtrip per call.
4. **Tiled Spatial Decode**: For 720P+, the decoder is compiled at small tile resolution and tiles the full-resolution latent with overlap blending.
5. **VAE Encoder on CPU**: Due to Neuron compiler bug NCC_IBIR158 in Conv3D tensorizer. Runs once per video, negligible overhead.
6. **bfloat16 Decoder**: Halves memory bandwidth for Conv3D-dominated decoder.

## File Structure

```
Wan2.2-TI2V/
  README.md
  requirements.txt
  assets/
    cat.png                               # Test input image (for I2V)
  src/
    run_wan2.2_ti2v.py                    # Main inference script (T2V and I2V)
    neuron_commons.py                     # Decoder/encoder wrappers, attention utilities
    neuron_parallel_utils.py              # TP sharding utilities for UMT5
    distributed_rmsnorm.py                # Distributed RMSNorm (reference, unused due to bug)
    compile_transformer.py                # Transformer (TP=4, CP=2 or CFG Parallel)
    compile_text_encoder.py               # Text encoder (ModelBuilder API, TP=4)
    compile_decoder_rolling.py            # VAE decoder with rolling cache (default)
    compile_decoder.py                    # VAE decoder with external feat_cache (legacy)
    compile_decoder_nocache.py            # VAE decoder without cache
    compile_encoder.py                    # VAE encoder (unused due to NCC_IBIR158)
    cache_hf_model.py                     # Download model
    compile.sh                            # Master compilation script
    setup_nvme.sh                         # NVMe RAID setup
  test/
    integration/
      test_model.py                       # Integration tests
    unit/
```

## Example Checkpoints

* [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-09
