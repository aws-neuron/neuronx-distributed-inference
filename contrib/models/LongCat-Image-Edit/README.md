# Contrib Model: LongCat-Image-Edit

NeuronX adaptation of [meituan-longcat/LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) for AWS Trainium2 inference.

## Model Information

- **HuggingFace ID:** `meituan-longcat/LongCat-Image-Edit`
- **Model Type:** FLUX-style diffusion model for image editing
- **Architecture:** Multi-component (Vision Encoder + Language Model + FLUX Transformer + VAE)
- **License:** Check HuggingFace model card

## Architecture Details

LongCat-Image-Edit is a FLUX-style image editing model with the following components:

| Component | Model | Neuron Parallelism |
|-----------|-------|-------------------|
| Vision Encoder | Qwen2.5-VL ViT (32 blocks) | TP=4, float32 |
| Language Model | Qwen2.5-VL LM (28 layers) | TP=4, world_size=8 |
| Transformer (CP) | LongCatImageTransformer2DModel (10 dual + 20 single stream) | TP=4, CP=2, world_size=8 |
| Transformer (CFG) | LongCatImageTransformer2DModel (10 dual + 20 single stream) | TP=4, DP=2, world_size=8, batch=2 |
| VAE | 2D AutoencoderKL | Single device (1024x1024, no tiling) |

Key parameters:
- **Attention Heads:** 24, head_dim=128, inner_dim=3072
- **Text Hidden Size:** 3584 (Qwen2.5-VL)
- **In Channels:** 64 (packed latents)
- **Dual-stream blocks:** 10 (separate text/image norms+FFN, joint attention)
- **Single-stream blocks:** 20 (concatenated text+image, parallel MLP+attention)

## Performance

| Machine | Config | Total Time | Per Step | Quality |
|---------|--------|------------|----------|---------|
| **Trn2** (trn2.48xlarge) | All Neuron, **CFG Parallel** | **20.39s** | 0.41s | Good |
| **Trn2** (trn2.48xlarge) | All Neuron, Context Parallel | 22.39s | 0.45s | Good |
| **H100** (single GPU, bf16) | Full GPU | 23.61s | 0.47s | Reference |

Test: 1024x1024 output, guidance_scale=4.5, 50 steps.

## CFG Parallel vs Context Parallel

Both modes use TP=4, world_size=8 on the same hardware:

| Aspect | Context Parallel (CP) | CFG Parallel |
|--------|----------------------|--------------|
| Scatter dimension | dim=1 (sequence) | dim=0 (batch) |
| Calls per step | 2 (neg + pos sequential) | 1 (neg + pos batched) |
| K/V All-Gather | Yes (every attention layer) | No |
| Compile batch_size | 1 | 2 |
| Best for | guidance_scale = 1 (no CFG) | guidance_scale > 1 (~9% faster) |

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
# Compile with CFG Parallel (default, recommended, fastest)
bash src/compile.sh

# Compile with Context Parallel
bash src/compile.sh cp

# Custom dimensions:
# bash src/compile.sh [cfg|cp] <height> <width> <image_size> <max_seq_len>
# bash src/compile.sh cfg 1024 1024 448 1024
```

Compilation takes ~60-90 minutes total. Compiled models are saved to `/opt/dlami/nvme/compiled_models/`.

### 4. Run Inference

```bash
# CFG Parallel (default, recommended, fastest)
NEURON_RT_NUM_CORES=8 PYTHONPATH=src:$PYTHONPATH python src/run_longcat_image_edit.py \
    --image assets/test.png \
    --prompt "change the cat to a dog" \
    --seed 43 \
    --output output.png

# Context Parallel
NEURON_RT_NUM_CORES=8 PYTHONPATH=src:$PYTHONPATH python src/run_longcat_image_edit.py \
    --image assets/test.png \
    --prompt "change the cat to a dog" \
    --seed 43 \
    --use_cp \
    --output output.png
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Input image path |
| `--prompt` | (required) | Edit instruction |
| `--output` | `output_edited.png` | Output image path |
| `--height` | 1024 | Output height |
| `--width` | 1024 | Output width |
| `--num_inference_steps` | 50 | Denoising steps |
| `--guidance_scale` | 4.5 | Guidance scale |
| `--seed` | 42 | Random seed |
| `--use_cfg_parallel` | true | Use CFG Parallel transformer (default, fastest) |
| `--use_cp` | false | Use Context Parallel instead of CFG |
| `--cpu_vision_encoder` | false | Use CPU vision encoder for better accuracy |
| `--warmup` | false | Run warmup inference first |
| `--compiled_models_dir` | `/opt/dlami/nvme/compiled_models` | Path to compiled models |

## Compatibility Matrix

| Instance/Version | 2.22+ (PyTorch 2.9) | 2.21 and earlier |
|------------------|---------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested | Not tested |
| Trn1 | Not tested | Not tested |
| Inf2 | Not supported | Not supported |

## Testing

Run integration test (requires Trn2 instance with compiled models):

```bash
# Full test (compile + inference + validate output)
PYTHONPATH=src:$PYTHONPATH pytest test/integration/test_model.py --capture=tee-sys -v

# Or run manually:
cd contrib/models/LongCat-Image-Edit
PYTHONPATH=src:$PYTHONPATH python test/integration/test_model.py
```

## Key Implementation Notes

1. **M-RoPE position IDs**: Must use original model's `get_rope_index()` method for correct 3D position IDs. Custom reimplementation produces wrong results.
2. **VL processor resolution**: Must match between compiled model and inference. CPU VE mode uses default resolution.
3. **Text sequence length**: `text_seq_len=1024` required (770-838 tokens typical for image editing prompts).
4. **VAE**: Compiled for full 1024x1024 output to avoid tile seam artifacts.
5. **Vision Encoder**: Uses native `F.scaled_dot_product_attention` (no monkey-patching) for accuracy.
6. **NKI Flash Attention**: Used for FLUX transformer attention (both dual-stream and single-stream blocks).

## File Structure

```
LongCat-Image-Edit/
  README.md
  requirements.txt
  assets/
    test.png                          # Test input image
  src/
    run_longcat_image_edit.py         # Main Neuron inference script
    neuron_commons.py                 # NeuronTextEncoderWrapper, NKI attention
    neuron_parallel_utils.py          # FLUX-specific TP sharding
    neuron_rope.py                    # 3-axis RoPE pre-computation
    compile_transformer.py            # FLUX transformer (TP=4, CP=2)
    compile_transformer_cfg.py        # FLUX transformer (TP=4, DP=2, CFG Parallel)
    compile_vae.py                    # 2D AutoencoderKL (1024x1024)
    compile_vision_encoder.py         # Qwen2.5-VL ViT (TP=4)
    compile_language_model.py         # Qwen2.5-VL LM (TP=4)
    cache_hf_model.py                 # Download model + install diffusers
    compile.sh                        # Master compilation script
    setup_nvme.sh                     # NVMe RAID setup
  test/
    integration/
      test_model.py                   # Integration test
    unit/
```

## Example Checkpoints

* [meituan-longcat/LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-09
