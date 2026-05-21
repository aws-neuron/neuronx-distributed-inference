# Contrib Model: Qwen2.5-Omni-7B

NeuronX Distributed Inference implementation of [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) with full multimodal support: text generation, image understanding, audio understanding, and text-to-speech.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen2.5-Omni-7B`
- **Model Type:** Multimodal encoder-decoder (Thinker + Vision + Audio + Talker + Token2Wav)
- **Architecture:** Qwen2-based text backbone with vision/audio encoders and speech synthesis
- **License:** Check HuggingFace model card

## Architecture Details

| Component | Runtime | TP | Parameters |
|-----------|---------|-----|------------|
| Thinker (text) | Neuron | 4 | hidden=3584, heads=28, kv_heads=4, layers=28 |
| Vision encoder | Neuron | 4 | embed=1280, heads=16, depth=32, SwiGLU MLP |
| Audio encoder | CPU+Neuron | 4 | d_model=1280, heads=20, layers=32, chunked attention |
| Talker | Neuron | 4 | hidden=896, heads=12, kv_heads=4, head_dim=128, layers=24, vocab=8448, fused embed (8448→896) |
| Token2Wav | CPU+Neuron (fp32) | N/A | DiT: dim=1024, 22 blocks (Neuron); BigVGAN: 6 upsample stages (CPU) |

**Total state dict keys:** 2448 (Text: 339, Vision: 518, Audio: 489, Talker: 293, Token2Wav: 809)

Key features:
- **Thinker**: Architecturally identical to Qwen2.5-7B; reuses `NeuronQwen2ForCausalLM` with state-dict prefix remapping (28 heads / 4 TP = 7 per rank, 4 kv_heads / 4 TP = 1 per rank)
- **Vision encoder**: SwiGLU MLP, RMSNorm, separate QKV projections, PatchMerger (16 heads / 4 TP = 4 per rank)
- **Audio encoder**: Whisper-style with chunked attention. Hybrid CPU+Neuron: Conv1d frontend + chunking on CPU, 32 transformer layers on Neuron (20 heads / 4 TP = 5 per rank), AvgPool + LayerNorm + projection on CPU
- **Talker**: Neuron-compiled with fused embedding (embed_tokens 8448→3584 + thinker_to_talker_proj 3584→896 collapsed into 8448→896), explicit head_dim=128, 3D mRoPE, per-step thinker state injection via vision_embeddings (12 heads / 4 TP = 3 per rank, 4 kv_heads / 4 TP = 1 per rank). Auto-pads vision_embeddings to max_context_length for compiled bucket compatibility.
- **Token2Wav**: DiT transformer core (22 blocks) on Neuron + BigVGAN vocoder on CPU, ODE sampling (Runge-Kutta 4, 10 steps), float32. Split architecture: CPU preprocessing (ECAPA-TDNN, codec embed, input embed, rotary) + Neuron transformer core + CPU ODE solver + CPU BigVGAN. Automatic CPU fallback when mel_len exceeds compiled max.

## Prerequisites

- **Instance**: trn2.48xlarge or trn2.xlarge (4+ NeuronCores sufficient)
- **Weights**: Download from [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — the example scripts auto-download via `huggingface_hub.snapshot_download` on first run.
- **Python dependencies** (on top of the NxDI venv):
  ```bash
  pip install soundfile          # writes WAV output in generate_qwen25_omni_speech.py
  pip install qwen-omni-utils[decord]   # process_mm_info() in generate_qwen25_omni.py for image/audio/video inputs
  ```
- **Pin all three Neuron models to the same core group**. Set `NEURON_RT_VISIBLE_CORES=0-3` before launching the speech pipeline so the Thinker (TP=4), Talker (TP=4), and the single-device Token2Wav DiT all live on the same four NeuronCores. Without this the Neuron runtime places the DiT NEFF on a different core group and every DiT forward pays a cross-group scheduling penalty (~30% slower). `examples/generate_qwen25_omni_speech.py` already sets this via `os.environ.setdefault` before any Neuron module is imported; if you embed the pipeline in your own entrypoint, do the same.
  ```bash
  export NEURON_RT_VISIBLE_CORES=0-3
  python examples/generate_qwen25_omni_speech.py
  ```

## Usage

### Text-only (Thinker)

```python
import sys
from pathlib import Path

# Make this contrib package's src/ importable (flat, per upstream contrib convention).
sys.path.insert(0, str(Path("contrib/models/Qwen2.5-Omni-7B/src").resolve()))
import _upstream_compat  # noqa: F401  (applies hf_adapter bug fix)

import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from modeling_qwen25_omni import (
    NeuronQwen25OmniForCausalLM,
    Qwen25OmniInferenceConfig,
)

model_path = "/path/to/Qwen2.5-Omni-7B/"
compiled_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    seq_len=4096,
    max_context_length=4096,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.6, top_k=20, top_p=0.95
    ),
)

config = Qwen25OmniInferenceConfig(
    neuron_config, load_config=load_pretrained_config(model_path)
)

model = NeuronQwen25OmniForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
adapter = HuggingFaceGenerationAdapter(model, tokenizer)
output = adapter.generate("What is quantum computing?", max_new_tokens=256)
```

### Multimodal (Vision + Audio + Speech)

```python
from modeling_qwen25_omni import (
    NeuronQwen25OmniMultimodalForCausalLM,
    Qwen25OmniMultimodalInferenceConfig,
)

# After loading text model, enable multimodal components:
# model.enable_audio_encoder(audio_state_dict)
# model.compile_audio_encoder("/path/to/compiled_audio/")  # compile Neuron transformer
# model.load_audio_encoder("/path/to/compiled_audio/")     # load compiled model
# model.enable_talker(talker_state_dict)
# model.enable_token2wav(token2wav_state_dict, speaker_dict_path="spk_dict.pt")
#
# Full multimodal pipeline:
#   Thinker generates text -> hidden states passed to Talker
#   -> Talker generates codec tokens -> Token2Wav generates waveform
```

## vLLM Integration

Qwen2.5-Omni can be served via [vllm-neuron](https://github.com/aws-neuron/vllm-neuron) for text-only inference. A patch is required for the nested config structure.

### Setup

```bash
# 1. Install vllm-neuron
pip install vllm-neuron

# 2. Apply the Qwen2.5-Omni patch
python perf_test/apply_vllm_neuron_patch_qwen25omni.py
```

### Serving

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-Omni-7B \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 4,
            "fused_qkv": false,
            "flash_decoding_enabled": false,
            "sequence_parallel_enabled": false,
            "batch_size": 32,
            "ctx_batch_size": 1,
            "tkg_batch_size": 32,
            "max_context_length": 4096,
            "seq_len": 4096,
            "is_continuous_batching": true,
            "enable_bucketing": true,
            "async_mode": true,
            "on_device_sampling_config": {
                "do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95
            }
        }
    }'
```

### Key vLLM Patch Changes

The patch (`perf_test/apply_vllm_neuron_patch_qwen25omni.py`) modifies vllm-neuron to:
- Extract text config from nested `thinker_config.text_config`
- Map `Qwen2_5OmniModel` architecture to `qwen2_5_omni` model type
- Handle layer count extraction for nested config

See `perf_test/3_bench_qwen25_omni_7b.sh` for full benchmark configurations.

## Performance

Text-only benchmark (trn2, BF16, TP=4):

| Config | TPOT (ms) | Output tok/s | Notes |
|--------|-----------|--------------|-------|
| BS=1, non-CB, greedy | ~11-13 | ~77-90 | Tested with chat template |
| BS=4, CB, c=4 | TBD | TBD | vLLM serving |

Model load time: ~15s (from compiled artifacts on NVMe).

Audio encoder performance (CPU frontend + CPU postprocessor, no Neuron transformer):

| Audio Length | Mel Frames | Frontend | Postprocessor |
|-------------|-----------|----------|---------------|
| 1s | ~100 | ~20ms | included |
| 3s | ~300 | ~22ms | included |
| 10s | ~1000 | ~33ms | included |
| 30s | ~3000 | ~34ms | included |

### End-to-End Multimodal (CPU inference, trn2.48xlarge)

| Test | Input | Output | Time |
|------|-------|--------|------|
| Text → Text | "What is the capital of France?" | Correct answer (Paris) | 15.1s |
| Image + Text → Text | Synthetic image (shapes) + description prompt | Correctly identified red square, blue circle, yellow circle, green triangle | 59.5s |
| Audio + Text → Text | 440Hz sine wave + "What do you hear?" | Text response generated | 12.1s |
| Text → Speech | "Say hello and tell me the weather" | Text + audio waveform (14.2s audio) | 197.2s |

### Speech Pipeline Profiling (CPU inference, trn2.48xlarge)

Per-component measured breakdown for text-to-speech (14.1s audio output):

| Component | Time | % of Total | RTF | Notes |
|-----------|------|------------|-----|-------|
| Thinker (7B) | 31.0s | 12% | — | 59 text tokens, ~1.9 tok/s on CPU |
| Talker (690M) | 103.3s | 41% | 7.3x | Autoregressive codec token generation, 24 layers |
| Token2Wav (DiT+BigVGAN) | 117.9s | 47% | 8.4x | 22 DiT blocks × 10 ODE steps × 2 (CFG) = 440 forward passes |
| **Total** | **252.1s** | **100%** | **17.9x** | Generating 14.1s audio takes 252.1s on CPU |

### Full Neuron Speech Pipeline (trn2.48xlarge, TP=4, BF16)

End-to-end speech synthesis with all components on Neuron (9.1s audio output):

| Component | Time | Notes |
|-----------|------|-------|
| Thinker (7B, Neuron TP=4) | 0.3s | 24 text tokens |
| CPU hidden state extraction | ~3s | HF model forward for thinker states |
| Talker (690M, Neuron TP=4) | 2.1s | 454 codec tokens, per-step thinker injection |
| Token2Wav (Neuron DiT + CPU BigVGAN) | 9.9s | 22 DiT blocks × 10 ODE steps |
| **Total** | **~15s** | **9.1s audio, RTF ~1.7x** |

### Neuron vs CPU Speedup (trn2.48xlarge, TP=4, BF16)

| Component | CPU Time | Neuron Time | Speedup | Notes |
|-----------|----------|-------------|---------|-------|
| Thinker (7B) | 30.4s | 0.47s | **64.7x** | TPOT 10.2ms |
| Talker (690M) | 98.1s | 2.0s (500 tokens) | **49.1x** | TPOT 4.0ms |
| Token2Wav DiT (85M) | 24.1s | 3.8s | **6.3x** | 22 blocks × 10 ODE steps, batch=2 (CFG) |
| Token2Wav BigVGAN | 2.8s | 2.8s (CPU) | 1x | Stays on CPU |
| **Total** | **267.9s** | **~15s** | **~18x** | All Neuron components active |

Token2Wav component breakdown (300 codec tokens / 6.0s audio):

| Config | CPU | Neuron DiT | Speedup |
|--------|-----|-----------|---------|
| DiT only (22 blocks, 10 ODE steps) | 24.1s | 3.8s | 6.3x |
| Token2Wav end-to-end | 13.7s | 5.2s | 2.7x |
| DiT core single forward (batch=2, mel_len=1024) | 592ms | 60ms | 9.8x |

Key observations:
- **Full Neuron speech pipeline** verified end-to-end: Thinker → Talker → Token2Wav all on Neuron, producing real human speech
- Thinker and Talker achieve **49-65x speedup** on Neuron
- Token2Wav DiT achieves **6.3x speedup** (9.8x for isolated transformer core)
- BigVGAN vocoder (CPU) is now the remaining bottleneck
- **Per-step thinker state injection**: Talker v2 adds thinker_reply_part[step] embedding at each autoregressive step, matching HF behavior
- **Vision embeddings auto-padding**: Compiled Neuron models require fixed bucket shapes; vision_embeddings are auto-padded to max_context_length
- Split architecture for Token2Wav: CPU preprocessing (ECAPA-TDNN, codec/input embed, rotary, block_diff) + Neuron transformer core (22 blocks + norm + proj)
- Overcame XLA tracing limitations: in-place slice assignment in DiTAttention (→ torch.cat), SDPA dispatch (→ explicit matmul), ECAPA-TDNN/codec embed issues (→ kept on CPU)
- Automatic CPU fallback when mel_len exceeds compiled DiT max

## Compatibility Matrix

| Instance/Version | 2.23+ (PyTorch 2.9) | 2.22 and earlier |
|------------------|---------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested (TP=4) | Not tested |
| Trn2 (trn2.xlarge) | Supported (TP=4) | Not tested |
| Trn1 (trn1.32xlarge) | Should work (TP=4, 4 NeuronCores) | Not tested |
| Inf2 (inf2.48xlarge) | Should work (TP=4) | Not tested |

## Testing

Verified on trn2.48xlarge with real Qwen2.5-Omni-7B weights:

- **Imports**: All model classes import successfully
- **Config**: TP=4 head divisibility verified (Thinker 7/1, Audio 5, Vision 4 per rank)
- **State dict**: All 2448 keys converted correctly (text=339, audio=489, vision=518, talker=293, token2wav=809)
- **Audio CPU**: Frontend+postprocessor 1s=20ms, 30s=34ms
- **Talker CPU**: 1351M params loaded in ~10s, codec tokens verified
- **Text generation (TP=4)**: Compile + load + generate working, TPOT ~11-13ms, correct outputs verified

```bash
# End-to-end test (compile + load + generate)
python /tmp/test_qwen25_omni_tp4.py
```

## Key Implementation Notes

1. **TP=4 for all Neuron components**: Thinker (28 heads/4=7 per rank), Vision (16 heads/4=4), Audio (20 heads/4=5). All heads divisible by 4.
2. **Audio encoder hybrid architecture**: Conv1d frontend + chunking on CPU, 32 transformer layers on Neuron with TP=4, AvgPool + LayerNorm + projection on CPU. Asymmetric attention bias (q/v have bias, k has none) handled via ColumnParallelLinear.
3. **Talker on Neuron**: Non-standard head_dim (128 != 896/12), 3D mRoPE with per-step thinker-state injection, ~690M params. Uses ImageToTextModelWrapper with 24 positional args. Fused embedding (embed_tokens 8448→3584 + proj 3584→896 collapsed into 8448→896). Per-step thinker reply states injected via vision_embeddings during token generation. Vision embeddings auto-padded to max_context_length for compiled bucket compatibility. TPOT 4.0ms.
4. **Token2Wav split architecture**: DiT transformer core (22 blocks) on Neuron via torch_neuronx.trace(). CPU preprocessing: ECAPA-TDNN speaker encoder, codec embedding (repeat_interleave), input embedding, rotary embedding, block_diff mask. CPU postprocessing: ODE solver (RK4, 10 steps), BigVGAN vocoder. Float32 for ODE precision. XLA fixes: DiTAttention in-place slice assignment → torch.cat, SDPA dispatch → explicit matmul attention, float additive attention mask. Automatic CPU fallback when mel_len exceeds compiled max.
5. **Speaker support**: `spk_dict.pt` contains per-speaker conditioning (Ethan, Chelsie)
6. **State dict prefix remapping**: `thinker.model.*` -> `model.*`, `thinker.lm_head.*` -> `lm_head.*`, `thinker.visual.*` -> `visual.*`, `thinker.audio_tower.*` -> `frontend.*`/`transformer.*`/`postprocessor.*`

## Example Checkpoints

* [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-15
