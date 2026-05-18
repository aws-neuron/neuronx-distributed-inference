# Contrib Model: QwenImageEditLoRA

Multi-LoRA runtime adapter switching for the Qwen-Image-Edit-2511 diffusion transformer on AWS Trainium2.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen-Image-Edit-2511`
- **LoRA Adapter:** `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`
- **Model Type:** MMDiT (Multimodal Diffusion Transformer) — 60-layer joint image+text transformer
- **Parameters:** ~28B total (transformer backbone ~20.4B)
- **Architecture:** Joint attention (24 heads, head_dim=128), GEGLU MLP, AdaLN-Zero modulation, NKI Flash Attention
- **License:** See HuggingFace model cards for respective licenses

## Validation Results

**Validated:** 2026-05-18
**Instance:** trn2.3xlarge (LNC=2, TP=4)
**SDK:** Neuron SDK 2.29, PyTorch 2.9

### Benchmark Results

| Resolution | Per-Step Latency | 28-Step Generation |
|-----------|------------------|-------------------|
| 512x512 | 335 ms | 9.4s |
| 768x768 | 805 ms | 22.6s |
| 1024x1024 | 1,541 ms | 43.2s |
| 1536x1536 | 4,294 ms | 120.2s |

Full pipeline with CFG (1024x1024, 28 steps, 56 transformer calls):
- **Transformer + LoRA:** 87.3s
- **Adapter switch time:** <1 ms (zero recompilation)

### Accuracy Validation

| Test | Result |
|------|--------|
| Baseline determinism (zeroed LoRA) | max_diff < 1e-5 between runs |
| LoRA injection changes output | max_diff > 1e-5 confirmed |
| Zeroing LoRA restores baseline | max_diff < 1e-5 from original |
| LoRA overhead | <1% (87.33s vs 86.45s base) |

The model uses `write_to_neuron_buffer()` for runtime adapter switching. Accuracy is validated by confirming:
1. Deterministic outputs with fixed inputs (no stochastic variance)
2. LoRA injection measurably changes output (aliasing works)
3. Removing LoRA perfectly restores the base model output

## Usage

### Compile

```bash
# Activate Neuron venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install diffusers transformers accelerate

# Compile (downloads model from HuggingFace)
python -m src.modeling_qwen_image_edit_lora \
    --model_id Qwen/Qwen-Image-Edit-2511 \
    --height 1024 --width 1024 \
    --compiled_models_dir ./compiled_models \
    --max_loras 4 --max_rank 16

# Or with local model path:
python -m src.modeling_qwen_image_edit_lora \
    --model_id /path/to/Qwen-Image-Edit-2511 \
    --height 1024 --width 1024 \
    --compiled_models_dir ./compiled_models \
    --max_loras 4 --max_rank 16
```

### Run Inference

```python
import torch
from neuronx_distributed import NxDModel

# Load compiled model
traced_model = NxDModel.load("compiled_models/transformer_multi_lora/nxd_model.pt")

# Load sharded weights
sharded_checkpoints = []
for rank in range(4):
    ckpt = torch.load(f"compiled_models/transformer_multi_lora/weights/tp{rank}_weights.pt")
    sharded_checkpoints.append(ckpt)

traced_model.set_weights(sharded_checkpoints)
traced_model.to_neuron()

# Run inference
output = traced_model(hidden_states, encoder_hidden_states, timestep, img_rope, txt_rope)
```

### Switch LoRA Adapter at Runtime

```python
# Inject adapter weights (< 1ms, zero recompilation)
for key, tensor in adapter_tensors.items():
    traced_model.write_to_neuron_buffer(tensor, key, rank=0)

# Run with new adapter active
output = traced_model(hidden_states, encoder_hidden_states, timestep, img_rope, txt_rope)

# Remove adapter (zero all LoRA buffers)
for key in lora_keys:
    zeros = torch.zeros_like(checkpoints[0][key])
    for rank in range(4):
        traced_model.write_to_neuron_buffer(zeros, key, rank)
```

### Verify Aliasing

```bash
python -m src.modeling_qwen_image_edit_lora \
    --model_id Qwen/Qwen-Image-Edit-2511 \
    --compiled_models_dir ./compiled_models \
    --test-aliasing
```

## Compatibility Matrix

| Instance | SDK 2.29 | SDK 2.28 |
|----------|----------|----------|
| trn2.3xlarge (LNC=2, TP=4) | VALIDATED | Not tested |
| trn2.48xlarge | Not tested | Not tested |

## Example Checkpoints

* [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) — Base model
* [fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) — 96-pose camera control LoRA (rank 16)

## Testing Instructions

```bash
# Set model path (local or HuggingFace ID)
export MODEL_ID=/path/to/Qwen-Image-Edit-2511
export COMPILED_MODELS_DIR=/tmp/qwen_edit_lora_test

# Run integration tests
cd contrib/models/QwenImageEditLoRA
pytest test/integration/test_model.py -v

# Note: Tests compile at 512x512 for speed. Full 1024x1024 compilation tested separately.
```

## Architecture Details

### Multi-LoRA Design

The model uses NxD ModelBuilder's `enable_aliasing=True` to alias all model parameters (including 1680 LoRA active buffers) in device HBM. This enables `write_to_neuron_buffer()` to update LoRA weights between inference calls without recompilation.

**LoRA targets (14 per block × 60 blocks = 840 pairs = 1680 tensors):**
- Attention: to_q, to_k, to_v, to_out.0, add_q_proj, add_k_proj, add_v_proj, to_add_out
- MLP: img_mlp.net.0.proj, img_mlp.net.2, txt_mlp.net.0.proj, txt_mlp.net.2
- Modulation: img_mod.1, txt_mod.1

### Compilation Configuration

- **TP=4**: ColumnParallelLinear for Q/K/V, RowParallelLinear for output projections
- **NKI Flash Attention**: Fused Q@K^T → softmax → @V kernel (ISA-level)
- **Sequential CFG**: batch_size=1, two passes per denoising step
- **Compiler args**: `--model-type=transformer -O1 --auto-cast=none --enable-fast-loading-neuron-binaries`

## Known Issues

- Each resolution requires a separate compilation (no dynamic shapes). Pre-compile for target resolutions.
- The model uses `local_files_only=True` by default for local paths. Set `MODEL_ID` to a HuggingFace ID for automatic download.
- Initial model load on fresh DLAMI instances may take 3-5 minutes (library rehydration).

## Maintainer

Jim Burtoft
