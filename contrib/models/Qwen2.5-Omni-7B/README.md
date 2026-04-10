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
| Thinker (text) | Neuron | 32 | hidden=3584, heads=28, kv_heads=4, layers=28 |
| Vision encoder | Neuron | 32 | embed=1280, heads=16, depth=32, SwiGLU MLP |
| Audio encoder | CPU | N/A | d_model=1280, heads=20, layers=32, chunked attention |
| Talker | CPU | N/A | hidden=896, heads=12, kv_heads=4, layers=24, vocab=8448 |
| Token2Wav | CPU (fp32) | N/A | DiT: dim=1024, 22 blocks; BigVGAN: 6 upsample stages |

**Total state dict keys:** 2448 (Text: 339, Vision: 518, Audio: 489, Talker: 293, Token2Wav: 809)

Key features:
- **Thinker**: Architecturally identical to Qwen2.5-7B; reuses `NeuronQwen2ForCausalLM` with state-dict prefix remapping
- **Vision encoder**: SwiGLU MLP, RMSNorm, separate QKV projections, PatchMerger
- **Audio encoder**: Whisper-style with chunked attention, runs on CPU (20 heads not TP-divisible by 32)
- **Talker**: Wraps HF's `Qwen2_5OmniTalkerForConditionalGeneration` for autoregressive codec token generation
- **Token2Wav**: DiT + BigVGAN vocoder with ODE sampling (Runge-Kutta 4), requires float32

## Prerequisites

- **Instance**: trn2.48xlarge (32 NeuronCores)
- **Weights**: Download from [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

## Usage

### Text-only (Thinker)

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni import (
    NeuronQwen25OmniForCausalLM,
    Qwen25OmniInferenceConfig,
)

model_path = "/path/to/Qwen2.5-Omni-7B/"
compiled_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=32,
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
from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni import (
    NeuronQwen25OmniMultimodalForCausalLM,
    Qwen25OmniMultimodalInferenceConfig,
)

# After loading text model, enable multimodal components:
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
    --tensor-parallel-size 32 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 32,
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

Text-only benchmark (trn2.48xlarge, BF16, TP=32):

| Config | TTFT (ms) | TPOT (ms) | Output tok/s |
|--------|-----------|-----------|--------------|
| BS=1, non-CB | 39.51 | 5.28 | 176.76 |
| BS=32, CB, c=32 | 283.44 | 17.31 | 1559.13 |

Audio encoder performance (CPU):

| Audio Length | Mel Frames | Latency |
|-------------|-----------|---------|
| 1s | ~100 | 0.14s |
| 30s | ~3000 | 2.85s |

## Compatibility Matrix

| Instance/Version | 2.23+ (PyTorch 2.9) | 2.22 and earlier |
|------------------|---------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested | Not tested |
| Trn1 | Not supported (text decoder needs TP=32) | Not supported |
| Inf2 | Not supported | Not supported |

## Testing

State dict conversion and weight loading verified on Trn2 with real Qwen2.5-Omni-7B weights:

```bash
# Full multimodal verification (all 2448 keys, 0 missing)
python /tmp/test_full_multimodal_all.py

# Individual component tests
python /tmp/test_talker.py
python /tmp/test_token2wav.py
```

## Key Implementation Notes

1. **State dict prefix remapping**: `thinker.model.*` -> `model.*`, `thinker.lm_head.*` -> `lm_head.*`, `thinker.visual.*` -> `visual.*`, `thinker.audio_tower.*` -> `audio_tower.*`
2. **Audio encoder on CPU**: 20 attention heads not divisible by 32 TP; variable-length chunked attention incompatible with Neuron compilation
3. **Talker on CPU**: 12 attention heads not divisible by 32 TP; autoregressive generation with KV cache
4. **Token2Wav in float32**: ODE solver (Runge-Kutta 4) requires float32 precision
5. **Speaker support**: `spk_dict.pt` contains per-speaker conditioning (Ethan, Chelsie)

## Example Checkpoints

* [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-10
