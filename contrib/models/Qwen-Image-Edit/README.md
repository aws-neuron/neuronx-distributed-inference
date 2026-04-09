# Contrib Model: Qwen-Image-Edit

NeuronX adaptation of [alibaba-pai/Qwen-Image-Edit-2509](https://huggingface.co/alibaba-pai/Qwen-Image-Edit-2509) for AWS Trainium2 inference.

## Model Information

- **HuggingFace ID:** `alibaba-pai/Qwen-Image-Edit-2509`
- **Model Type:** Diffusion model for image editing
- **Architecture:** Multi-component (Qwen2.5-VL Vision Encoder + Language Model + QwenImageTransformer2DModel + 3D VAE)
- **License:** Check HuggingFace model card

## Architecture Details

| Component | Model | Parameters | Neuron Parallelism |
|-----------|-------|------------|-------------------|
| Vision Encoder | Qwen2.5-VL ViT (32 blocks) | ~1.4B | TP=4, float32 (or CPU) |
| Language Model | Qwen2.5-VL LM (28 layers) | ~7B | TP=4, world_size=8 (or CPU) |
| Transformer | QwenImageTransformer2DModel | ~20.4B | TP=4-8, various parallelism modes |
| VAE | 3D AutoencoderKL (causal) | ~300M | Single device, tiled processing |

Key parameters:
- **Transformer**: 48 attention heads, head_dim=128, inner_dim=6144
- **Text Hidden Size**: 3584 (Qwen2.5-VL)
- **Dual-stream blocks**: 20 (separate text/image norms+FFN, joint attention)
- **Single-stream blocks**: 40 (concatenated text+image, parallel MLP+attention)

## Performance

6 compilation APIs with different parallelism strategies:

| Version | Parallelism | Attention | Per Step | Notes |
|---------|------------|-----------|----------|-------|
| **V3 CFG** | TP=4, DP=2 | NKI Flash | **~0.75s** | Fastest, recommended |
| V3 CP | TP=4, CP=2 | NKI Flash | ~0.77s | Context Parallel |
| V1 Flash | TP=8 | NKI Flash | ~1.2s | NKI kernel |
| V2 Flash | TP=8 | NKI Flash | ~1.2s | ModelBuilder + NKI |
| V2 | TP=8 | Standard SDPA | ~1.2s | ModelBuilder |
| V1 | TP=8 | Standard SDPA | ~2.4s | Baseline |

Test: 1024x1024 output, guidance_scale=4.5, 50 steps, trn2.48xlarge.

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
# Compile V3 CFG (recommended, fastest)
bash src/compile.sh v3_cfg

# Compile V3 CP (Context Parallel)
bash src/compile.sh v3_cp

# Compile all versions
bash src/compile.sh

# Custom dimensions:
# bash src/compile.sh <version> <height> <width> <image_size> <tp_degree> <max_seq_len> <patch_mult> <batch_size>
```

Compilation takes ~60-120 minutes total depending on version.

### 4. Run Inference

```bash
NEURON_RT_NUM_CORES=8 PYTHONPATH=src:$PYTHONPATH python src/run_qwen_image_edit.py \
    --image assets/image1.png \
    --prompt "change the sky to sunset" \
    --version v3_cfg \
    --output output.png
```

## Compatibility Matrix

| Instance/Version | 2.22+ (PyTorch 2.9) | 2.21 and earlier |
|------------------|---------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested | Not tested |
| Trn1 | Not tested | Not tested |
| Inf2 | Not supported | Not supported |

## Testing

```bash
# Run component tests
PYTHONPATH=src:$PYTHONPATH pytest test/integration/ --capture=tee-sys -v

# Run all tests manually
PYTHONPATH=src:$PYTHONPATH python test/integration/run_all_tests.py
```

## Key Implementation Notes

1. **Modulation Layer Sharding**: Uses `ColumnParallelLinear(gather_output=True)` to reduce memory from ~17GB to ~5.2GB per shard.
2. **RoPE Without Complex Numbers**: Neuron doesn't support C64; uses (cos, sin) tuples instead.
3. **M-RoPE Position IDs**: 3D position indices (temporal, height, width) for multimodal tokens.
4. **VAE Interpolation**: Replaces `nearest-exact` with `nearest` for Neuron compatibility.
5. **CFG Parallel**: Batches negative + positive prompts into single forward pass for ~6% speedup over CP.
6. **NKI Flash Attention**: Custom NKI kernel for Trainium2, requires `XLA_DISABLE_FUNCTIONALIZATION=1`.

## File Structure

```
Qwen-Image-Edit/
  README.md
  requirements.txt
  assets/
    image1.png, image2.png            # Test input images
  src/
    run_qwen_image_edit.py            # Main inference script
    neuron_commons.py                 # NeuronTextEncoderWrapper, SDPA implementations
    neuron_parallel_utils.py          # TP sharding utilities
    neuron_rope.py                    # Neuron-compatible RoPE
    autoencoder_kl_qwenimage_neuron.py  # Neuron-compatible 3D VAE
    compile_transformer.py            # V1 transformer (TP=8)
    compile_transformer_v1_flash.py   # V1 Flash (NKI)
    compile_transformer_v2.py         # V2 (ModelBuilder)
    compile_transformer_v2_flash.py   # V2 Flash (ModelBuilder + NKI)
    compile_transformer_v3_cp.py      # V3 Context Parallel (TP=4, CP=2)
    compile_transformer_v3_cfg.py     # V3 CFG Parallel (TP=4, DP=2)
    compile_language_model_v3.py      # Language Model V3 (TP=4)
    compile_vision_encoder_v3.py      # Vision Encoder V3 (TP=4)
    compile_text_encoder.py           # Vision encoder single-device
    compile_vae.py                    # 3D VAE encoder/decoder
    cache_hf_model.py                 # Download model
    compile.sh                        # Master compilation script
    setup_nvme.sh                     # NVMe RAID setup
  test/
    integration/
      run_all_tests.py                # Master test runner
      test_vae.py                     # VAE tests
      test_transformer.py             # Transformer tests
      test_text_encoder.py            # Text encoder tests
      test_component_comparison.py    # Neuron vs CPU comparison
      test_language_model_simple.py   # Language model tests
      test_multimodal.py              # Multi-image tests
    unit/
```

## Example Checkpoints

* [alibaba-pai/Qwen-Image-Edit-2509](https://huggingface.co/alibaba-pai/Qwen-Image-Edit-2509)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-09
