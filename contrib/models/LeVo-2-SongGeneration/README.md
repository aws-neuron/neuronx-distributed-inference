# Contrib Model: LeVo 2 (SongGeneration v2)

Text-to-music generation on AWS Trainium2 using Tencent's SongGeneration v2 (LeVo 2) -- a hybrid LLM-Diffusion audio pipeline that generates stereo 48kHz music with vocals from lyrics and text descriptions. Supports both v2-medium (2.83B) and v2-large (5.12B) model variants.

## Model Information

- **HuggingFace ID:** `lglg666/SongGeneration-v2-medium` (v2-medium), `lglg666/SongGeneration-v2-large` (v2-large)
- **Shared Assets:** `lglg666/SongGeneration-Runtime` (diffusion, VAE, tokenizer, prompts)
- **Model Type:** Multi-stage audio generation pipeline (LLM + Diffusion + VAE)
- **Parameters:** v2-medium: ~2.83B (FP32/BF16), v2-large: ~5.12B (FP32/BF16); shared diffusion: ~1.1B, VAE: ~169M
- **Architecture:** Dual-Llama AR LM (primary + secondary) with delayed codebook pattern, GPT2-RoPE CFM diffusion backbone (16L), Stable Audio VAE decoder
- **Output:** Stereo 48kHz WAV audio
- **License:** Check [HuggingFace model card](https://huggingface.co/lglg666/SongGeneration-v2-medium)

### Model Variants

| Variant | Total Params | Primary Layers | Dim | Heads | Secondary Layers |
|---------|-------------|----------------|-----|-------|-----------------|
| v2-medium | 2.83B | 28 | 1536 | 12 | 12 |
| v2-large | 5.12B | 36 | 2048 | 16 | 12 |

Both variants share the same diffusion backbone (GPT2-RoPE, 1.1B) and VAE decoder (169M).

## Architecture

LeVo 2 uses a three-stage pipeline:

| Stage | Component | Params | Neuron Compilation | Key Innovation |
|-------|-----------|--------|-------------------|----------------|
| 1. LeLM | Dual-Llama AR (primary + secondary) | 2.83B / 5.12B | `ModelBuilder` (on-device KV) | Prefill + `torch.scatter` KV cache in HBM |
| 2. Diffusion | GPT2-RoPE CFM (16L) | 1.1B | `torch_neuronx.trace()` | Rewritten RoPE (no complex numbers) |
| 3. VAE | Stable Audio decoder | 169M | `torch_neuronx.trace()` | `weight_norm` removal pre-trace |

### Key Differences from v1 SongGeneration

| Feature | v1 | v2 |
|---------|----|----|
| Model sizes | 1 (base, 2.83B) | 2 (v2-medium 2.83B, v2-large 5.12B) |
| Batch size | Fixed B=1 | Configurable B=1..N |
| Conditioning | PREFILL_LEN=602 | PREFILL_LEN=952 |
| rope_theta (primary) | 100000 | 500000 |
| Musicality tokens | No | Yes (`[Musicality-very-high]` prefix) |
| Model loading | `get_lm_model(cfg)` | `get_lm_model(cfg, version='v2')` |

### On-Device KV Cache

The LeLM transformers use on-device KV caching via `neuronx_distributed.ModelBuilder`. Instead of passing KV cache tensors as model inputs/outputs each autoregressive step (PCIe round-trip), the cache is stored as `register_buffer` on the model and updated in-place with `torch.scatter`. This keeps the cache in Neuron HBM.

### Prefill Optimization

The first 952 condition-prepend tokens (description=600 + prompt_audio=252 + type_info=100) are processed in a single Neuron call via a dedicated "prefill" NEFF, rather than one-at-a-time through the decode NEFF.

### Neuron-Specific Adaptations

- **RoPE rewrite:** `torch.view_as_complex` / `torch.view_as_real` replaced with explicit sin/cos rotation (XLA compatible)
- **Flash Attention disabled:** `use_flash_attn_2=False` (CUDA-only feature)
- **CUDA-to-CPU patches:** All `torch.cuda` calls redirected to CPU (upstream codebase assumes CUDA)
- **weight_norm removal:** `torch.nn.utils.remove_weight_norm` applied to VAE before tracing
- **GPT2 diffusion fp32:** The GPT2 diffusion backbone **must** be traced with `--auto-cast none` (full FP32). Using `--auto-cast matmult` causes severe numerical degradation (cosine similarity drops from 1.0 to 0.64 vs CPU) which compounds across 10 Euler solver steps into garbled audio. The VAE can safely use `--auto-cast matmult`.
- **Musicality tokens:** v2 adds `[Musicality-very-high]` prefix to style descriptions for quality control

## Validation Results

**Validated:** 2026-04-06
**Instance:** trn2.3xlarge (LNC=2, 4 NeuronCores)
**SDK:** Neuron SDK 2.28 (DLAMI 20260227), PyTorch 2.9

### Component Accuracy (shared across variants)

| Component | Metric | Value | Threshold |
|-----------|--------|-------|-----------|
| GPT2 diffusion (fp32) | Cosine similarity vs CPU | >0.9999 | > 0.98 |
| GPT2 diffusion (fp32) | Max diff vs CPU | <0.001 | < 0.01 |
| VAE decoder | Cosine similarity vs CPU | >0.9999 | > 0.98 |
| VAE decoder | SNR vs CPU | > 40 dB | > 20 dB |

### Benchmark Results: v2-medium

| Config | Total E2E | LeLM Steps | Steps/s | ms/step |
|--------|-----------|------------|---------|---------|
| 5s audio, B=1, TP=1 | 21.8s | 1327 | 61.5 | 56.5 |
| 30s audio, B=1, TP=1 | 61.3s | 1952 | 32.6 | -- |

Per-step breakdown (5s audio):
- Primary (28L): 38.3 ms/step
- Secondary (12L): 18.2 ms/step

### Benchmark Results: v2-large

| Config | Total E2E | LeLM Steps | Steps/s | ms/step |
|--------|-----------|------------|---------|---------|
| 5s audio, B=1, TP=1 | 37.3s | 1327 | 35.8 | 97.0 |
| 30s audio, B=1, TP=1 | 75.3s | 1669 | 22.7 | -- |

Per-step breakdown (5s audio):
- Primary (36L, dim=2048, 16H): 71.0 ms/step
- Secondary (12L): 26.0 ms/step

### Batch Size Results (v2-medium, 5s audio, TP=1)

| Metric | B=1 | B=2 |
|--------|-----|-----|
| Total inference | 22.2s | 30.8s |
| Songs generated | 1 | 2 |
| Wall time per song | 22.2s | 15.4s |
| Throughput improvement | -- | 1.44x |

### GPU Comparison (L40S)

Benchmarked on g6e.2xlarge (1x NVIDIA L40S, 48GB VRAM).

| Config | Neuron (trn2) | GPU (L40S) |
|--------|---------------|------------|
| v2-medium 5s | 21.8s | 15.95s |
| v2-medium 30s | 61.3s | 42.89s |
| v2-large 5s | 37.3s | 18.95s |
| v2-large 30s | 75.3s | 49.27s |

## Usage

### Prerequisites

1. Clone the SongGeneration repository:
   ```bash
   git clone https://github.com/tencent-ailab/songgeneration.git /mnt/models/songgeneration
   cd /mnt/models/songgeneration
   git lfs pull --include='tools/new_auto_prompt.pt'
   ```

2. Download model weights:
   ```bash
   pip install huggingface_hub
   python -c "
   from huggingface_hub import snapshot_download
   # v2-medium
   snapshot_download('lglg666/SongGeneration-v2-medium',
                     local_dir='/mnt/models/levo/weights/v2-medium',
                     ignore_patterns=['*.md'])
   # v2-large
   snapshot_download('lglg666/SongGeneration-v2-large',
                     local_dir='/mnt/models/levo/weights/v2-large',
                     ignore_patterns=['*.md'])
   # Shared runtime assets (diffusion, VAE, tokenizer, prompts)
   snapshot_download('lglg666/SongGeneration-Runtime',
                     local_dir='/mnt/models/levo/runtime',
                     ignore_patterns=['*.md'])
   "
   ```

3. Set up symlinks and paths:
   ```bash
   cd /mnt/models/songgeneration
   ln -sf /mnt/models/levo/runtime/third_party third_party
   ln -sf /mnt/models/levo/runtime/ckpt ckpt
   mkdir -p conf && cp codeclm/conf/vocab.yaml conf/vocab.yaml
   # Symlink third_party into Flow1dVAE as well
   ln -sf /mnt/models/levo/runtime/third_party codeclm/tokenizer/Flow1dVAE/third_party
   ```

4. Activate Neuron environment and install dependencies:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
   pip install accelerate flashy alias-free-torch descript-audio-codec \
       k-diffusion vector-quantize-pytorch einops-exts x-transformers \
       diffusers==0.37.0 peft==0.18.0 lightning openunmix
   pip install protobuf==5.29.3  # Must be after descript-audio-codec
   export PYTHONPATH="/mnt/models/songgeneration/codeclm/tokenizer/:/mnt/models/songgeneration:/mnt/models/songgeneration/codeclm/tokenizer/Flow1dVAE/:$PYTHONPATH"
   ```

5. Apply patches (required on each new instance):
   ```bash
   # SequenceSummary stub
   UTILS_FILE=$(python3 -c "import transformers.modeling_utils; print(transformers.modeling_utils.__file__)")
   echo '
   class SequenceSummary:
       pass' >> "$UTILS_FILE"

   # Flash attention import fix
   find /mnt/models/songgeneration/codeclm/ -name "*.py" -exec sed -i "s/is_flash_attn_available/is_flash_attn_2_available/g" {} +

   # Remove transformers version assertion
   sed -i "/assert.*transformers.*version/d" /mnt/models/songgeneration/codeclm/models/levo.py
   ```

### Compile and Generate (v2-medium)

```python
from modeling_levo2 import LeVo2Neuron, LeVo2Config

config = LeVo2Config.v2_medium(
    model_path="/mnt/models/levo/weights/v2-medium/model.pt",
    config_path="/mnt/models/levo/weights/v2-medium/config.yaml",
    safetensors_path="/mnt/models/levo/runtime/ckpt/model_septoken/model_2.safetensors",
    prompt_path="/mnt/models/levo/runtime/ckpt/encode-s12k.pt",
    codeclm_path="/mnt/models/songgeneration",
    default_duration_sec=5.0,
)

pipeline = LeVo2Neuron(config)
pipeline.compile()  # ~20 min first time

# Override with English prompts
import torch
prompt_data = torch.load(
    '/mnt/models/songgeneration/tools/new_auto_prompt.pt',
    map_location='cpu', weights_only=False
)
pipeline._prompt_data = {
    g: prompt_data[g]['en']
    for g in prompt_data
    if isinstance(prompt_data[g], dict) and 'en' in prompt_data[g]
}

pipeline.warmup()

# Generate
audio, sr = pipeline.generate(
    lyrics="[intro-short] ; [verse] Sunlight breaks through morning haze ; [chorus] Sing along ; [outro-short]",
    descriptions="pop, uplifting, piano",
    genre="Pop",
    duration_sec=5.0,
)

# Save as WAV
import scipy.io.wavfile
import numpy as np
audio_np = audio.squeeze(0).float().cpu().numpy().T
audio_np = np.clip(audio_np, -1.0, 1.0)
audio_int16 = (audio_np * 32767).astype(np.int16)
scipy.io.wavfile.write("output.wav", sr, audio_int16)
```

### Compile and Generate (v2-large)

```python
from modeling_levo2 import LeVo2Neuron, LeVo2Config

config = LeVo2Config.v2_large(
    model_path="/mnt/models/levo/weights/v2-large/model.pt",
    config_path="/mnt/models/levo/weights/v2-large/config.yaml",
    safetensors_path="/mnt/models/levo/runtime/ckpt/model_septoken/model_2.safetensors",
    prompt_path="/mnt/models/levo/runtime/ckpt/encode-s12k.pt",
    codeclm_path="/mnt/models/songgeneration",
    default_duration_sec=5.0,
)

pipeline = LeVo2Neuron(config)
pipeline.compile()  # ~25 min first time
# ... (same prompt loading and generation as above)
```

### Lyrics Format

The model expects structured lyrics with section tags separated by ` ; ` and lines separated by `.`:

```
[intro-short] ; [verse] First line of verse.Second line of verse ; [chorus] Chorus line one.Chorus line two ; [outro-short]
```

**Structure tags:** `[verse]`, `[chorus]`, `[bridge]`, `[intro-short/medium/long]`, `[outro-short/medium/long]`, `[inst-short/medium/long]`, `[silence]`

**Language:** The model generates vocals in the language of the lyrics. Use English lyrics for English vocals, Chinese for Chinese. The prompt audio language should match (use `new_auto_prompt.pt` with `['en']` or `['zh']` key).

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.27 |
|----------|----------|----------|
| trn2.3xlarge (LNC=2, v2-medium) | VALIDATED | Not tested |
| trn2.3xlarge (LNC=2, v2-large) | VALIDATED | Not tested |
| trn2.48xlarge | Not tested | Not tested |

## Example Checkpoints

* [lglg666/SongGeneration-v2-medium](https://huggingface.co/lglg666/SongGeneration-v2-medium)
* [lglg666/SongGeneration-v2-large](https://huggingface.co/lglg666/SongGeneration-v2-large)
* [lglg666/SongGeneration-Runtime](https://huggingface.co/lglg666/SongGeneration-Runtime) (shared assets)

## Testing Instructions

```bash
# On a trn2.3xlarge with model weights and codeclm source:
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
cd /mnt/models/songgeneration

# Set paths
export PYTHONPATH="$(pwd)/codeclm/tokenizer/:$(pwd):$(pwd)/codeclm/tokenizer/Flow1dVAE/:$PYTHONPATH"
export CODECLM_PATH=/mnt/models/songgeneration

# Test v2-medium (compile from scratch, ~30 min):
LEVO2_VARIANT=v2-medium \
LEVO2_MODEL_PATH=/mnt/models/levo/weights/v2-medium/model.pt \
LEVO2_CONFIG_PATH=/mnt/models/levo/weights/v2-medium/config.yaml \
LEVO2_SAFETENSORS_PATH=/mnt/models/levo/runtime/ckpt/model_septoken/model_2.safetensors \
LEVO2_PROMPT_PATH=/mnt/models/levo/runtime/ckpt/encode-s12k.pt \
pytest contrib/models/LeVo-2-SongGeneration/test/integration/test_model.py -v --timeout=3600

# Test v2-large:
LEVO2_VARIANT=v2-large \
LEVO2_MODEL_PATH=/mnt/models/levo/weights/v2-large/model.pt \
LEVO2_CONFIG_PATH=/mnt/models/levo/weights/v2-large/config.yaml \
LEVO2_SAFETENSORS_PATH=/mnt/models/levo/runtime/ckpt/model_septoken/model_2.safetensors \
LEVO2_PROMPT_PATH=/mnt/models/levo/runtime/ckpt/encode-s12k.pt \
pytest contrib/models/LeVo-2-SongGeneration/test/integration/test_model.py -v --timeout=3600

# Or run standalone:
python contrib/models/LeVo-2-SongGeneration/test/integration/test_model.py
```

## Known Issues

1. **GPT2 diffusion MUST use `--auto-cast none`:** The iterative Euler solver (10 steps) amplifies per-step numerical errors exponentially. With `--auto-cast matmult`, cosine similarity drops to 0.64 vs CPU, producing garbled audio. With `--auto-cast none`, cosine similarity is >0.9999.

2. **Language-aware prompt audio is essential:** The `encode-s12k.pt` prompt file provides per-language prompt audio tokens. Always use the correct language key matching the lyrics language.

3. **Duration affects compilation:** The GPT2 and VAE components are traced at a fixed frame count (`T_frames = duration_sec * 25`). Changing duration requires recompilation. The LeLM models support variable lengths up to `max_seq_len`.

4. **torchaudio WAV saving:** The Neuron DLAMI's torchaudio may lack codec support for WAV saving. Use `scipy.io.wavfile` instead.

5. **First-run library rehydration:** The first import of torch-neuronx/transformers on a fresh DLAMI instance can take 2-5 minutes due to lazy package decompression.

6. **Compilation not cached across sessions:** ModelBuilder does not persist compiled NEFFs between Python sessions. Each run recompiles (~20 min for v2-medium, ~25 min for v2-large). Use `save()`/`load()` to avoid recompilation.

7. **Musicality prefix required:** v2 models expect `[Musicality-very-high]` prefix on style descriptions. The pipeline adds this automatically if missing.

## Maintainer

@jimburtoft
