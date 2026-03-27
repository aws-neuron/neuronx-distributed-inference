# Contrib Model: whisper-large-v3-turbo

OpenAI Whisper Large V3 Turbo (openai/whisper-large-v3-turbo) speech-to-text model for NxD Inference on AWS Neuron (Trainium2 and Inferentia2).

This is an encoder-decoder model with separate encoder and decoder compilation. It uses the [OpenAI Whisper](https://github.com/openai/whisper) package as its base class (not HuggingFace Transformers).

## Model Information

| Field | Value |
|-------|-------|
| **HuggingFace ID** | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| **Model Type** | Encoder-Decoder (Speech-to-Text) |
| **Parameters** | 809M |
| **License** | MIT |
| **Architecture** | 32 encoder layers + 4 decoder layers (turbo), 1280 hidden, 20 heads |

## Architecture Details

- **Encoder**: 32-layer bidirectional transformer with Conv1D frontend (128 mel bins -> 1280 hidden)
- **Decoder**: 4-layer causal transformer with cross-attention to encoder output
- **Key optimizations in this implementation**:
  1. **Cross-attention K/V cache**: Skip redundant K/V projections during decode (~2.5x decode speedup, saves 19.7B FLOPs/token)
  2. **Fused QKV projections**: 3 matmuls -> 1 for self-attention
  3. **NKI flash attention (encoder)**: Bidirectional flash attention for all 32 encoder layers
  4. **NKI fused Conv1D+GELU (encoder)**: Fused conv1d kernel for encoder frontend (optional, graceful fallback)
  5. **LNC flag**: Compiler args pass `--lnc=` for LNC=1 support on trn2
  6. **Batch size >1**: Batched decode with per-sample positional embedding and logit extraction

## Validation Results

| Test | Result |
|------|--------|
| Transcription Accuracy | 0% WER on reference audio |
| Cosine Similarity | N/A (speech model, validated by WER) |

## Performance Metrics

### Single-Stream (BS=1, trn2.3xlarge, LNC=2, bfloat16)

| Audio Duration | Latency | Real-Time Factor |
|---------------|---------|-----------------|
| 5.0s | 180.2ms | 27.8x |
| 15.0s | 229.1ms | 65.5x |
| 30.0s | 462.9ms | 64.8x |
| 90.0s | 1102.2ms | 81.7x |

### Batched (BS=8, trn2.3xlarge, LNC=2, bfloat16)

| Audio Duration | Batch Latency | Per-Sample | Throughput |
|---------------|--------------|------------|------------|
| 5.0s | 630.2ms | 78.8ms | 12.69 audio-sec/wall-sec |
| 30.0s | 675.5ms | 84.4ms | 11.84 audio-sec/wall-sec |
| 90.0s | 675.0ms | 84.4ms | 11.85 audio-sec/wall-sec |

### Data Parallel (DP=4 x BS=8, trn2.3xlarge, LNC=2, bfloat16)

| Audio Duration | Aggregate Throughput |
|---------------|---------------------|
| 5.0s | **46.65 audio-sec/wall-sec** |
| 30.0s | **43.75 audio-sec/wall-sec** |
| 90.0s | **43.27 audio-sec/wall-sec** |

## Usage

```python
import os
import sys
import torch

# Add the contrib src directory to the Python path
sys.path.insert(0, "/path/to/contrib/models/whisper-large-v3-turbo/src")

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from modeling_whisper import WhisperInferenceConfig, NeuronApplicationWhisper

DTYPE = torch.bfloat16
BATCH_SIZE = 1
TP_DEGREE = 1
MODEL_PATH = "/home/ubuntu/models/whisper-large-v3-turbo/"
COMPILED_MODEL_PATH = "/home/ubuntu/compiled_models/whisper-large-v3-turbo/"

# Define configs
neuron_config = NeuronConfig(
    batch_size=BATCH_SIZE,
    torch_dtype=DTYPE,
    tp_degree=TP_DEGREE,
)
inference_config = WhisperInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(MODEL_PATH),
)

# Compile model (one-time, ~75s)
if not os.path.exists(COMPILED_MODEL_PATH):
    neuron_model = NeuronApplicationWhisper(MODEL_PATH, config=inference_config)
    neuron_model.compile(COMPILED_MODEL_PATH)

# Load from compiled checkpoint (~8s)
neuron_model = NeuronApplicationWhisper(COMPILED_MODEL_PATH, config=inference_config)
neuron_model.load(COMPILED_MODEL_PATH)

# Transcribe an audio file
result = neuron_model.transcribe("audio-sample.mp3", verbose=True)
print(result["text"])
```

## Compatibility Matrix

| Instance Type | SDK Version | TP Degree | Dtype | Status |
|--------------|-------------|-----------|-------|--------|
| trn2.3xlarge | 2.28 | 1 | bfloat16 | Validated |
| trn2.3xlarge | 2.28 | 1 | float16 | Validated |
| inf2.xlarge | 2.28 | 1 | bfloat16 | Expected compatible |
| inf2.xlarge | 2.28 | 1 | float16 | Expected compatible |

**Notes**:
- TP=1 is recommended. Whisper (809M params) fits on a single NeuronCore.
- Higher TP degrees are supported for head-sharding but provide no benefit for this model size.
- For maximum throughput on trn2.3xlarge, use DP=4 x BS=8 with LNC=2 (4 independent model instances).
- Each batch size requires separate compilation (BS is baked into the traced graph).

## Testing

### Prerequisites

```bash
pip install openai-whisper pytest
```

### Run integration tests

```bash
# From the whisper-large-v3-turbo directory
pytest test/integration/test_model.py -v

# Or run manually
python test/integration/test_model.py
```

### Test details

The integration test:
1. Compiles the model (encoder + decoder) if not already compiled
2. Loads the compiled model
3. Transcribes a reference audio file
4. Validates that the transcription produces non-empty text
5. Measures transcription latency

## Example Checkpoints

- [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (809M, 4 decoder layers, recommended)
- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) (1.5B, 32 decoder layers)

## Dependencies

- `openai-whisper` (provides base `Whisper` class and decoding loop)
- `transformers` (for `WhisperModel.from_pretrained` weight loading and `sinusoids`)
- `neuronx-distributed-inference` (NxDI base classes, model wrapper, config)
- `nkilib` (optional, for fused Conv1D+GELU kernel)

## Maintainer

Jim Burtoft (jimburtoft)

## Last Updated

2026-03-26
