# Contrib Model: voxtral-mini-3B

Mistral AI's Voxtral Mini 3B (mistralai/Voxtral-Mini-3B-2507) audio-language model for NxD Inference on AWS Neuron (Trainium2 and Inferentia2).

This is a multimodal encoder-decoder model with a Whisper-like audio encoder, a linear projector, and a Llama-based LLM backbone. It supports both text-only and audio+text generation (transcription, audio understanding, chat).

## Model Information

| Field | Value |
|-------|-------|
| **HuggingFace ID** | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) |
| **Model Type** | Audio-Language (Encoder + Projector + Decoder) |
| **Parameters** | 4B total (637M encoder + 25M projector + 3.3B LLM) |
| **License** | Apache 2.0 |
| **Architecture** | 32 encoder layers + 30 decoder layers, 3072 hidden, 32 Q heads / 8 KV heads |

## Architecture Details

- **Audio encoder**: Whisper-like `VoxtralEncoder` -- 637M params, 32 layers, 20 heads, 1280 hidden, 128 mel bins, max 30s audio. Input: `[1, 128, 3000]` mel spectrogram -> Output: `[1, 1500, 1280]`
- **Projector**: `VoxtralMultiModalProjector` -- Linear(5120 -> 3072) + GELU + Linear(3072 -> 3072), 25M params. Packs 4 adjacent encoder hidden states (`[375, 5120]`) into text space (`[375, 3072]`)
- **LLM backbone**: `LlamaForCausalLM` -- 3.3B params, 30 layers, hidden=3072, 32 Q heads / 8 KV heads, vocab=131072, head_dim=128, rope_theta=1e8
- **Audio token injection**: Scatter-based embedding injection via `encode_vision_to_input` (NxDI's `scatter_by_index_put`), not cross-attention

### Key design decisions in this implementation

1. **Decomposed pipeline**: Audio encoder traced separately with `torch_neuronx.trace()`, projector runs on CPU, LLM compiled with NxDI's `ImageToTextModelWrapper`
2. **Reuses NxDI Llama**: `VoxtralTextModel` extends `NeuronLlamaModel` -- no custom attention/MLP needed
3. **State dict delegation**: `convert_hf_to_neuron_state_dict` delegates to `NeuronLlamaForCausalLM` with `config.text_config`
4. **PixtralInferenceConfig reuse**: Voxtral's encoder+projector+LLM maps cleanly to Pixtral's vision+LLM config pattern
5. **Hardware-aware compiler args**: Auto-detects trn2 for `--lnc=2` flag, omits on trn1/inf2
6. **None-safe forward**: `_get_model_outputs` handles `None` vision args for text-only generation

## Validation Results

| Test | Result |
|------|--------|
| Text Generation | Deterministic, correct answers (greedy) |
| Transcription WER (vs CPU) | 8.7% (trn2), 0.0% (Small-24B) |
| Audio + Text Understanding | Validated on TED talk audio |

## Performance Metrics

### trn2.3xlarge (TP=1, LNC=2, BF16)

| Component | Latency |
|-----------|---------|
| Audio encoder | 405ms |
| Projector | 14ms |
| TTFT | 418ms |
| Throughput | **58.5 tok/s** |

### inf2.xlarge (TP=1, BF16)

| Workload | Median tok/s |
|----------|-------------|
| text-short | 28.4 |
| text-long | 25.5 |
| audio-short | 27.4 |
| transcribe | 15.4 |

### Cost Comparison

| Platform | Instance | $/M tokens |
|----------|----------|-----------|
| GPU | g6e.4xlarge | $5.44 |
| Neuron | trn2.3xlarge | $18.28 |
| Neuron | inf2.xlarge | **$7.42** |

## Usage

```python
import os
import sys
import torch

# Add the contrib src directory to the Python path
sys.path.insert(0, "/path/to/contrib/models/voxtral-mini-3B/src")

from modeling_voxtral import NeuronApplicationVoxtral

MODEL_PATH = "/home/ubuntu/models/voxtral-mini-3B"
COMPILED_PATH = "/home/ubuntu/compiled_models/voxtral-mini-3B"

# Create application
app = NeuronApplicationVoxtral(
    model_path=MODEL_PATH,
    tp_degree=1,
    seq_len=2048,
    n_positions=4096,
    dtype=torch.bfloat16,
)

# Compile (one-time, ~10 min for audio encoder + text decoder)
if not os.path.exists(COMPILED_PATH):
    app.compile(COMPILED_PATH)

# Load from compiled checkpoint (~30s)
app.load(COMPILED_PATH)

# Text-only generation
text = app.generate("What is the capital of France?")
print(text)

# Audio transcription
transcription = app.transcribe("path/to/audio.wav")
print(transcription)

# Audio understanding
transcription = app.transcribe(
    "path/to/audio.wav",
    prompt="Summarize the main topics discussed in this audio.",
)
print(transcription)
```

## Compatibility Matrix

| Instance Type | SDK Version | TP Degree | Dtype | Status |
|--------------|-------------|-----------|-------|--------|
| trn2.3xlarge | 2.28 | 1 | bfloat16 | Validated |
| trn2.3xlarge | 2.28 | 4 | bfloat16 | Validated (Small-24B) |
| inf2.xlarge | 2.28 | 1 | bfloat16 | Validated |
| inf2.8xlarge | 2.28 | 1 | bfloat16 | Expected compatible |

**Notes**:
- TP=1 is recommended for Mini-3B (3.3B LLM fits on a single NeuronCore).
- TP=4 is needed for Small-24B (23.6B LLM backbone).
- Audio encoder is always TP=1 (traced separately, runs on a single core).
- `inline_weights_to_neff=False` is required for the audio encoder to avoid `Descriptor limit reached` errors.

## Testing

### Prerequisites

```bash
# Activate the pre-installed PyTorch inference environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Install additional dependencies
pip install mistral_common[audio] pytest
```

### Download model weights

```bash
# Using huggingface-cli (requires HF token for gated model)
huggingface-cli download mistralai/Voxtral-Mini-3B-2507 \
    --local-dir /home/ubuntu/models/voxtral-mini-3B
```

### Run integration tests

```bash
# From the voxtral-mini-3B directory
export VOXTRAL_MODEL_PATH=/home/ubuntu/models/voxtral-mini-3B
export VOXTRAL_COMPILED_PATH=/home/ubuntu/compiled_models/voxtral-mini-3B

pytest test/integration/test_model.py -v

# Or run directly
python test/integration/test_model.py
```

### Test details

The integration test:
1. Compiles the model (audio encoder + text decoder) if not already compiled
2. Loads all components (audio encoder, projector, text decoder)
3. Validates text-only generation (correctness + determinism)
4. Validates audio transcription (non-empty output)
5. Measures latency for both text and audio generation

## Voxtral Model Family

| Model | HF ID | Parameters | LLM Backbone | TP Degree |
|-------|-------|------------|-------------|-----------|
| **Mini-3B** | `mistralai/Voxtral-Mini-3B-2507` | 4B | Llama 3.3B | 1 |
| Small-24B | `mistralai/Voxtral-Small-24B-2507` | 24B | Llama 23.6B | 4 |

Both models share the same audio encoder (637M) and architecture pattern. The Small-24B variant uses a larger LLM backbone and projector, requiring TP=4 on trn2.3xlarge.

## Dependencies

- `transformers` >= 4.54.0 (for `VoxtralForConditionalGeneration`)
- `mistral_common[audio]` (for `processor.apply_chat_template` with audio)
- `neuronx-distributed-inference` (NxDI base classes)
- `torch-neuronx` (for `torch_neuronx.trace`)

## Maintainer

Jim Burtoft (jimburtoft)

## Last Updated

2026-03-27
