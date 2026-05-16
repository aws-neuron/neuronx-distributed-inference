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
- **CFG Parallel**: Splits batch (cond/uncond), no K/V communication, ~11-13% faster for most resolutions (default)
- **Rolling Cache**: Stateful temporal caching for flicker-free video, ~960MB on-device

## Performance

Re-measured on trn2.48xlarge under the **bf16 TP all-reduce default** (`WAN_ALLREDUCE_BF16=1`),
50 denoising steps, CFG Parallel (TP=4 × DP=2). H100 column is single H100 on p5.48xlarge with
`diffusers` eager mode (bf16, FA2/SDPA), measured at the same time.

| Resolution | Frames | Trn2 CFG bf16 (s) | H100 eager (s) | Decoder |
|-----------|-------:|------------------:|---------------:|---------|
| 512×384   | 81  | **15.26**  | 16.57 | stateful rolling |
| 640×480   | 121 | **39.40**  | 86.34 | stateful rolling |
| 1280×704  | 81  | **139.58** (denoise 67.8 s + tiled decode 70.8 s) | 89.08 | tiled (3×3, overlap=4) |

Older fp32-reduce numbers and resolutions not re-measured here (e.g. 512×384×121,
640×480×81, 1280×704×121) live in the PR description history; toggle with
`WAN_ALLREDUCE_BF16=0` to reproduce them. The 1280×704 case is dominated by the
tile decoder, not the transformer — single-tile decode at that resolution exceeds
the compiler's 5 M instruction limit.

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
# CFG Parallel (default, recommended, fastest for most resolutions)
bash src/compile.sh

# Context Parallel
CP=1 bash src/compile.sh

# Custom output directory
bash src/compile.sh /path/to/output /path/to/compiler_workdir
```

Compilation takes ~30-60 minutes.

### 4. Run Inference

```bash
# Text-to-Video (T2V) - auto-detects CFG or CP from compiled models
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

Note: The run script auto-detects CFG Parallel (`transformer_cfg/`) or Context Parallel (`transformer/`) from compiled models directory.

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

**Last Updated:** 2026-04-13
