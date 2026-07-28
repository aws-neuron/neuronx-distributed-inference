# Contrib Model: Shrutam-2

Multilingual Indic automatic speech recognition (ASR) on AWS Neuron. Three-stage pipeline: Conformer encoder + SMEAR-MoE projector + Llama decoder supporting 12 Indian languages.

**Maintainer:** Jim Burtoft (@jimburtoft)

## Model Information

- **HuggingFace ID:** [`bharatgenai/Shrutam-2`](https://huggingface.co/bharatgenai/Shrutam-2)
- **Model Type:** Encoder-decoder ASR (Conformer + MoE projector + autoregressive LLM)
- **Parameters:** ~1.9B total
  - Conformer encoder: 607.7M (FP32 weights, BF16 compute via auto-cast)
  - SMEAR-MoE projector: 50.4M (8 experts, FP32 weights, BF16 compute)
  - LLM decoder: ~1.2B (LlamaForCausalLM, BF16)
- **Architecture:**
  - Conformer: 24 layers, d_model=1024, 8 heads, head_dim=128, ff_dim=4096, conv_kernel=9, relative positional encoding
  - SMEAR-MoE: 8 experts (2-layer MLP each: 1024→2048→2048), utterance-level soft routing via einsum weight merge
  - LLM: LlamaForCausalLM, 16 layers, hidden_size=2048, 32 attention heads, 8 KV heads (GQA 4:1), head_dim=64, vocab=128016, tie_word_embeddings=true
- **Languages:** Hindi, Tamil, Telugu, Bengali, Kannada, Malayalam, Marathi, Gujarati, Odia, Punjabi, Assamese, Urdu
- **License:** BharatGen Non-Commercial License
- **Reference:** [arXiv:2601.19451](https://arxiv.org/abs/2601.19451)

## Validation Results

**Validated:** 2026-04-23
**Instance:** trn2.3xlarge (LNC=2, 4 logical NeuronCores)
**SDK:** Neuron SDK 2.29, NxDI 0.9.x, PyTorch 2.9

### Benchmark Results

#### Single-Core Performance (BS=1)

| Metric | Value |
|--------|-------|
| Conformer encoder latency | 9.0 ms (10s audio) |
| SMEAR-MoE projector latency | 1.6 ms |
| Audio TTFT (encoder + projector) | ~12 ms |
| LLM decode throughput | 113 tok/s |
| E2E median latency | 237 ms |
| Real-time factor | 30x |
| Throughput | 20.8 audio-seconds/s |

#### Data-Parallel Performance (DP=4, trn2.3xlarge)

| Metric | Value |
|--------|-------|
| Aggregate throughput | 61.1 audio-seconds/s |
| Aggregate decode | 370 tok/s |
| Scaling efficiency | 73% (2.9x from 4 cores) |
| E2E median latency | 302 ms per core |

#### NEFF Sizes

| Component | NEFF Size |
|-----------|-----------|
| Conformer encoder (BS=1) | 1,862 MB |
| SMEAR-MoE projector (BS=1) | 156 MB |
| LLM decoder (BS=1, TP=1) | 16.7 MB |
| **Total** | **~2,034 MB** |

### Accuracy Validation

Measured against CPU reference using FLEURS test samples (20 samples: 10 Hindi, 5 Tamil, 5 Telugu).

#### Encoder Numerical Accuracy

| Metric | Value |
|--------|-------|
| Cosine similarity (Neuron vs CPU) | 0.9985 |
| Max absolute error | < 0.01 |

#### SMEAR Numerical Accuracy

| Metric | Value |
|--------|-------|
| Cosine similarity (Neuron vs CPU) | ~1.0 |

#### Word Error Rate (WER) vs CPU

| Language | CPU avg WER | Neuron avg WER | Delta |
|----------|-------------|----------------|-------|
| Hindi (excl outliers) | 10.6% | 12.2% | +1.6% |
| Tamil | 26.7% | 27.7% | +1.0% |
| Telugu | 14.2% | 15.3% | +1.1% |
| **Overall (18/20 samples)** | **16.1%** | **17.4%** | **+1.3%** |

Note: Neuron uses greedy decoding. The original CPU pipeline uses beam search (num_beams=4). One sample (hi_08) requires beam search for correct output and produces hallucinated text under greedy decoding. Using `repetition_penalty=1.3` mitigates most hallucination artifacts.

## Usage

### Prerequisites

Download and extract the Shrutam-2 checkpoint from HuggingFace:

```bash
# Download from https://huggingface.co/bharatgenai/Shrutam-2
# Expected files:
#   encoder.pt        - Conformer encoder weights (FP32, ~2.5 GB)
#   model.pt          - Downsampler + SMEAR weights (FP32, ~5.2 GB)
#   llm/              - Llama decoder directory:
#     config.json
#     model.safetensors
#     tokenizer.json
#     tokenizer_config.json
```

### Step 1: Trace Encoder and SMEAR

```python
from modeling_shrutam2 import trace_encoder, trace_smear

# Trace Conformer encoder (~5-10 min)
trace_encoder(
    encoder_weights_path="/mnt/models/encoder.pt",
    model_pt_path="/mnt/models/model.pt",
    output_path="/mnt/models/encoder_neuron.pt",
    batch_size=1,
    audio_seconds=10.0,
    lnc=2,
)

# Trace SMEAR-MoE projector (~2-3 min)
trace_smear(
    model_pt_path="/mnt/models/model.pt",
    output_path="/mnt/models/smear_neuron.pt",
    batch_size=1,
    seq_len=126,  # ceil(1001/8) for 10s audio
    lnc=2,
)
```

### Step 2: Compile LLM Decoder

```python
from modeling_shrutam2 import build_llm_model

model, config = build_llm_model(
    llm_path="/mnt/models/Shrutam-2-hf/llm",
    tp_degree=1,
    batch_size=1,
    seq_len=2048,
    n_positions=4096,
    lnc=2,
)
model.compile("/mnt/models/compiled/shrutam2_decoder_tp1")
```

### Step 3: Run End-to-End Pipeline

```python
from modeling_shrutam2 import Shrutam2Pipeline

pipeline = Shrutam2Pipeline(
    encoder_neff_path="/mnt/models/encoder_neuron.pt",
    smear_neff_path="/mnt/models/smear_neuron.pt",
    llm_compiled_path="/mnt/models/compiled/shrutam2_decoder_tp1",
    llm_path="/mnt/models/Shrutam-2-hf/llm",
    tp_degree=1,
    batch_size=1,
    lnc=2,
)

result = pipeline.transcribe(
    "audio.wav",
    prompt="Transcribe speech to Hindi text.",
    max_new_tokens=200,
    repetition_penalty=1.3,
)
print(result["text"])
# Output: Hindi transcription of the audio
```

### Language-Specific Prompts

```python
# Hindi
result = pipeline.transcribe("audio.wav", prompt="Transcribe speech to Hindi text.")

# Tamil
result = pipeline.transcribe("audio.wav", prompt="Transcribe speech to Tamil text.")

# Telugu
result = pipeline.transcribe("audio.wav", prompt="Transcribe speech to Telugu text.")
```

## Compatibility Matrix

| Instance Type | SDK 2.29 | SDK 2.28 |
|---------------|----------|----------|
| trn2.3xlarge (LNC=2, TP=1) | VALIDATED | Not tested |
| trn2.48xlarge | Not tested | Not tested |
| inf2.xlarge | Not tested | Not tested |

## Example Checkpoints

* [bharatgenai/Shrutam-2](https://huggingface.co/bharatgenai/Shrutam-2)

## Testing Instructions

```bash
# Set environment variables pointing to model artifacts
export SHRUTAM2_ENCODER_WEIGHTS=/mnt/models/encoder.pt
export SHRUTAM2_MODEL_WEIGHTS=/mnt/models/model.pt
export SHRUTAM2_LLM_PATH=/mnt/models/Shrutam-2-hf/llm
export SHRUTAM2_ENCODER_NEFF=/mnt/models/encoder_neuron.pt
export SHRUTAM2_SMEAR_NEFF=/mnt/models/smear_neuron.pt
export SHRUTAM2_LLM_COMPILED=/mnt/models/compiled/shrutam2_decoder_tp1
export SHRUTAM2_TEST_AUDIO=/mnt/models/test_audio  # optional: real FLEURS samples

# Run all tests
pytest contrib/models/Shrutam-2/test/integration/test_model.py -v --timeout=900

# Run individual test classes
pytest contrib/models/Shrutam-2/test/integration/test_model.py::TestConformerEncoder -v
pytest contrib/models/Shrutam-2/test/integration/test_model.py::TestSMEARProjector -v
pytest contrib/models/Shrutam-2/test/integration/test_model.py::TestLLMDecoder -v
pytest contrib/models/Shrutam-2/test/integration/test_model.py::TestEndToEndPipeline -v
```

## Known Issues

1. **Greedy vs beam search gap:** The original CPU pipeline uses `num_beams=4`. NxDI does not support beam search for this model. Using greedy decoding with `repetition_penalty=1.3` produces comparable results for most samples, with ~5% of samples (beam-search-dependent) potentially producing hallucinated output.

2. **Fixed audio duration:** The Conformer encoder is traced for a fixed 10-second input shape. Audio shorter than 10s is zero-padded with proper attention masking. Audio longer than 10s is truncated. Multi-duration NEFFs (30s, 60s) can be traced for longer audio.

3. **SMEAR einsum scaling at batch size > 1:** The SMEAR projector's einsum-based weight merge scales poorly at batch sizes > 1 (34x slower at BS=8 vs BS=1). Workaround: run SMEAR at BS=1 in a loop for batched inference (75% latency reduction vs batched einsum).

4. **Single-core only (TP=1):** The LLM decoder is compiled at TP=1. The model is small enough (~1.2B) to fit on a single NeuronCore at LNC=2 (24 GB HBM). TP>1 is not needed and would add communication overhead.

5. **No torchaudio dependency:** Audio loading uses `soundfile` instead of `torchaudio` (which requires CUDA libraries not available on Neuron instances). Install with: `pip install soundfile`.
