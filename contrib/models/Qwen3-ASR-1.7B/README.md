# Contrib Model: Qwen3-ASR-1.7B

NeuronX Distributed Inference implementation of [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), a speech-to-text model with Whisper-like audio encoder and Qwen3 decoder.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen3-ASR-1.7B`
- **Model Type:** Encoder-Decoder ASR (Audio encoder + Qwen3 text decoder)
- **Architecture:** `Qwen3ASRForConditionalGeneration`
- **Parameters:** ~1.7B total (encoder: ~300M, decoder: ~1.4B)
- **License:** Apache 2.0

## Architecture Details

- **Audio Encoder:** 24 transformer layers, d_model=1024, 16 heads (Whisper-like with Conv2D frontend)
- **Text Decoder:** 28 Qwen3 layers, hidden_size=2048, GQA 16/8, head_dim=128, QK-norm, mRoPE
- **Vocabulary:** 151,936 tokens
- **Max Position Embeddings:** 65,536 (KV cache: configurable, 1024 sufficient for most ASR)
- **Audio Rate:** ~13 tokens per second of audio

## Validation Results

**Validated:** 2026-05-09  
**Configuration:** TP=4, batch_size=1, N_POSITIONS=1024, bfloat16  
**Instance:** trn2.3xlarge (LNC=2)

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads and generates tokens |
| E2E Accuracy | PASS | EXACT MATCH with CPU reference |
| WER (LibriSpeech test-clean, 50 samples) | PASS | 3.06% (published: 1.63%) |
| Silence Handling | PASS | Empty output for non-speech audio |
| Long Audio (30s) | PASS | Correct transcription |

### Performance Metrics (trn2.3xlarge, TP=4)

| Metric | Value |
|--------|-------|
| TTFT (5s audio) | 27.5ms |
| TTFT (30s audio) | 39.9ms |
| TPOT | 4.9ms |
| E2E Latency (10s audio) | 240ms |
| RTF (30s audio) | 0.020x (50x real-time) |
| Throughput | 194 tok/s |
| Audio throughput | 49.7 audio-sec/wall-sec |

### DP=2 Throughput (TP=2 per instance)

| Config | Aggregate Throughput |
|--------|---------------------|
| TP=4 single stream | 29.8 audio-sec/wall-sec |
| TP=2 x DP=2 | ~46.2 audio-sec/wall-sec |

## Usage

### 1. Compile the Model

```python
import torch
from src.modeling_qwen3_asr import NeuronQwen3ASRForCausalLM, create_inference_config
from src.audio_encoder import trace_encoder

model_path = "Qwen/Qwen3-ASR-1.7B"  # or local path
compiled_path = "/path/to/compiled/"
encoder_dir = "/path/to/compiled/encoder/"

# Compile text decoder
config = create_inference_config(model_path, tp_degree=4, n_positions=1024)
model = NeuronQwen3ASRForCausalLM(compiled_path, config)
model.compile(compiled_path)

# Trace audio encoder (3 bucket sizes: 5s, 10s, 30s)
trace_encoder(model_path, encoder_dir, buckets=[500, 1000, 3000])
```

### 2. Run Inference

```python
import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, WhisperFeatureExtractor

from src.modeling_qwen3_asr import (
    NeuronQwen3ASRForCausalLM, create_inference_config,
    get_encoder_output_length, AUDIO_PAD_ID, AUDIO_START_ID, AUDIO_END_ID,
    IM_START_ID, IM_END_ID, EOS_ID,
)
from src.audio_encoder import load_encoders, select_bucket

model_path = "Qwen/Qwen3-ASR-1.7B"
compiled_path = "/path/to/compiled/"
encoder_dir = "/path/to/compiled/encoder/"

# Load model
config = create_inference_config(model_path, tp_degree=4, n_positions=1024)
model = NeuronQwen3ASRForCausalLM(compiled_path, config)
model.load(compiled_path)

# Load encoder
encoders = load_encoders(encoder_dir)

# Load tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

# Process audio
audio, sr = sf.read("audio.wav")
audio = audio.astype(np.float32)

mel = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
mel_len = int(mel["attention_mask"][0].sum().item())
bucket_T = select_bucket(mel_len)
N_tokens = get_encoder_output_length(mel_len)

mel_input = mel["input_features"][0][:, :bucket_T]
if mel_input.shape[1] < bucket_T:
    mel_input = torch.nn.functional.pad(mel_input, (0, bucket_T - mel_input.shape[1]))

# Encode audio
with torch.no_grad():
    audio_embeddings = encoders[bucket_T](mel_input)[:N_tokens]

# Build input sequence
prefix = [IM_START_ID, 8948, 198, IM_END_ID, 198, IM_START_ID, 872, 198, AUDIO_START_ID]
audio_ids = [AUDIO_PAD_ID] * N_tokens
suffix = [AUDIO_END_ID, IM_END_ID, 198, IM_START_ID, 77091, 198]
input_ids = torch.tensor([prefix + audio_ids + suffix], dtype=torch.long)

# Generate (see test/integration/test_model.py for full decode loop)
# ... autoregressive decode with model.forward() ...

# Output format: "language English<asr_text>transcription text<|im_end|>"
```

### 3. vLLM Serving (OpenAI-compatible API)

See [`vllm/README.md`](./vllm/README.md) for full setup instructions including patches to vllm-neuron.

Quick start (after applying patches):

```bash
export NEURON_COMPILED_ARTIFACTS='/path/to/compiled/qwen3_asr_vl_text_tp4'
export NEURON_ENCODER_PATH='/path/to/compiled/qwen3_asr_encoder'
export NEURON_RT_VISIBLE_CORES='0-3'
bash vllm/start-vllm-server.sh
```

Then transcribe via API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-ASR-1.7B", "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "<base64_wav>", "format": "wav"}}]}], "max_tokens": 256}'
```

## Key Implementation Notes

1. **rope_scaling must use "rope_type": "default"** (NOT "mrope") - mRoPE is applied externally via `rotary_position_ids`
2. **rotary_position_ids must be int/long** (NOT float) - computed from attention_mask.long().cumsum()
3. **sampling_params must be torch.zeros(1, 3)** even when on_device_sampling is disabled
4. **Encoder tracing: DO NOT use inline_weights_to_neff=True** - causes accuracy regression
5. **Batching limitation**: `scatter_by_index_put` in NxDI assumes BS=1 for multimodal prefill. Use DP for throughput.

## Compatibility Matrix

| Instance/SDK | SDK 2.29 | SDK 2.28 |
|--------------|----------|----------|
| trn2.3xlarge | VALIDATED | Not tested |
| trn2.48xlarge | Expected to work | Not tested |
| trn1.32xlarge | Not supported (NxDI 0.9 drops trn1) | May work with NxDI 0.7 |

## Testing

```bash
# Run integration tests (requires compiled model and encoder on Neuron instance)
pytest contrib/models/Qwen3-ASR-1.7B/test/integration/test_model.py -v --capture=tee-sys
```

## Example Checkpoints

* [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

## Maintainer

Jim Burtoft (jimburtoft)

**Last Updated:** 2026-05-09
