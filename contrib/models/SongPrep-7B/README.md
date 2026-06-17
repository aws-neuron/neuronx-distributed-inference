# Contrib Model: SongPrep-7B

Song structure parsing and lyrics transcription with timestamps on AWS Neuron (Trainium2).

## Model Information

- **HuggingFace ID:** [`tencent/SongPrep-7B`](https://huggingface.co/tencent/SongPrep-7B)
- **Model Type:** Two-stage pipeline (audio encoder + decoder-only transformer)
- **Parameters:** ~7.5B total (329.5M encoder + ~7B decoder, BF16)
- **Architecture:** MuCodec audio encoder (Wav2Vec2-Conformer + 1-RVQ) + Qwen2 decoder (GQA, RoPE, SiLU)
- **License:** Apache 2.0
- **Paper:** [SongPrep: AI-Assisted Song Pre-Production](https://github.com/tencent-ailab/SongPrep)
- **Maintainer:** Jim Burtoft

## Overview

SongPrep-7B takes raw audio and produces structured lyrics with section labels and timestamps:

```
[verse][0.00:15.23]I'm looking for a new love, a new love
[chorus][15.23:30.45]Can you hear me calling out your name
```

The pipeline has two stages:
1. **MuCodec Encoder** (329.5M params, FP32): Converts audio waveform to discrete codec tokens at 25 tokens/second. Uses a Wav2Vec2-Conformer backbone with a single-codebook RVQ quantizer (16384 entries).
2. **Qwen2 Decoder** (7B params, BF16): Takes codec tokens as input and generates structured text with section labels (`[verse]`, `[chorus]`, etc.) and timestamps.

### Neuron Implementation

- **MuCodec**: Split pipeline — MelSTFT preprocessing runs on CPU (uses `torch.stft` which is not traceable due to overlapping window strides), Conformer+RVQ backbone traced to Neuron via `torch_neuronx.trace()` with `--auto-cast=matmult`.
- **Qwen2**: Compiled via NxD Inference with `on_device_sampling_config=None` (CPU-side sampling required because the extended vocabulary of 168,040 tokens exceeds the on-device sampling NKI kernel's per-partition limit).

## Validation Results

**Validated:** 2026-04-09
**Instance:** trn2.3xlarge (LNC=2, 4 logical cores)
**SDK:** Neuron SDK 2.27, PyTorch 2.9

### Benchmark Results

| Audio Duration | MuCodec Latency | Qwen2 Throughput | Generated Tokens | Total Pipeline |
|---------------|----------------|-----------------|-----------------|---------------|
| 10s | 0.089s | 26.3 tok/s | varies | < 0.1s + generation |
| 30s | 0.125s | 24.5 tok/s | varies | < 0.2s + generation |
| 60s | 0.244s | 21.0 tok/s | varies | < 0.3s + generation |

MuCodec encoding runs at 112-246x realtime. The total pipeline time is dominated by the Qwen2 decoder, which generates at 21-26 tok/s.

**Estimated real-world performance:** A typical 3-minute song completes in 10-21s (9-18x realtime), depending on output length.

### Accuracy Validation

| Component | Metric | Result |
|-----------|--------|--------|
| MuCodec encoder | Codec token match (Neuron vs CPU) | 96.8% (242/250 tokens, 10s audio) |
| Qwen2 decoder | Token match (Neuron vs CPU, greedy) | 100% (first 200 tokens identical) |

MuCodec token mismatches are expected with `--auto-cast=matmult` — small floating-point differences in the Conformer occasionally push vectors to different codebook entries. This does not meaningfully affect downstream lyrics quality.

## Usage

### Prerequisites

1. Download the model weights:
   ```bash
   huggingface-cli download tencent/SongPrep-7B --local-dir /mnt/models/SongPrep-7B
   ```

2. Clone the SongPrep repository (needed for MuCodec model definitions):
   ```bash
   git clone https://github.com/tencent-ailab/SongPrep /mnt/models/SongPrep
   ```

3. Install dependencies:
   ```bash
   pip install soundfile omegaconf
   ```

### Step 1: Trace MuCodec Encoder

```python
from src.modeling_songprep import trace_mucodec_encoder

trace_mucodec_encoder(
    model_path="/mnt/models/SongPrep-7B",
    output_path="/mnt/models/mucodec_neuron.pt",
    compiler_args=["--auto-cast", "matmult"],
)
```

### Step 2: Compile Qwen2 Decoder

```python
from src.modeling_songprep import SongPrepNeuronConfig, compile_qwen2

config = SongPrepNeuronConfig(
    model_path="/mnt/models/SongPrep-7B",
    tp_degree=2,
)
compile_qwen2(
    model_path="/mnt/models/SongPrep-7B",
    output_path="/mnt/models/qwen2-compiled",
    config=config,
)
```

### Step 3: Run Pipeline

```python
from src.modeling_songprep import SongPrepNeuronConfig, SongPrepPipeline

config = SongPrepNeuronConfig(
    model_path="/mnt/models/SongPrep-7B",
    mucodec_neff_path="/mnt/models/mucodec_neuron.pt",
    qwen2_compiled_path="/mnt/models/qwen2-compiled",
    tp_degree=2,
)

pipeline = SongPrepPipeline(config)
pipeline.load()

result = pipeline.run("/path/to/audio.wav")
print(result["lyrics"])
# Output: [verse][0.00:15.23]I'm looking for a new love...
```

## Compatibility Matrix

| Instance | SDK 2.27 | SDK 2.28 |
|----------|----------|----------|
| trn2.3xlarge (TP=2, LNC=2) | VALIDATED | Not tested |

### Configuration Notes

- **TP=2** is used because Qwen2's 4 KV heads trigger `GQA.CONVERT_TO_MHA` at TP=2 (works correctly). TP=4 with LNC=1 would enable native GQA but was not tested.
- **`on_device_sampling_config=None`** is required — the extended vocabulary (168,040 tokens) exceeds the on-device sampling NKI kernel's `max8` operation limit of 16,384 elements per partition.
- **`--auto-cast=matmult`** is required for the MuCodec encoder (FP32 model) to achieve reasonable performance on Neuron.

## Example Checkpoints

* [tencent/SongPrep-7B](https://huggingface.co/tencent/SongPrep-7B) — Model weights (14.5 GB, includes `mucodec.safetensors` + Qwen2 shards)

## Testing Instructions

```bash
# Set environment variables
export SONGPREP_MODEL_PATH=/mnt/models/SongPrep-7B
export SONGPREP_REPO_PATH=/mnt/models/SongPrep
export SONGPREP_MUCODEC_NEFF=/mnt/models/mucodec_neuron.pt
export SONGPREP_QWEN2_COMPILED=/mnt/models/qwen2-compiled

# Run tests
pytest test/integration/test_model.py -v --timeout=600
```

## Known Issues

1. **MelSTFT not traceable on Neuron**: The `torch.stft` operation uses `aten::as_strided` with overlapping window strides that XLA cannot lower. Workaround: run MelSTFT on CPU (~7ms overhead, negligible vs total pipeline time).

2. **Large vocabulary blocks vLLM-neuron**: The on-device sampling NKI kernel's `max8` operation is limited to 16,384 elements per partition. With `vocab_size=168,040` and TP=2, that's 84,020 elements/partition — exceeding the limit. Workaround: use NxD Inference directly with `on_device_sampling_config=None`.

3. **`import torch_neuronx` must precede `torch.jit.load()`**: When loading a traced MuCodec NEFF in the same process as NxD Inference, the Neuron model class registration requires `import torch_neuronx` before calling `torch.jit.load()`.

4. **SongPrep source dependency**: The MuCodec model definitions (`mucodec/generate_1rvq.py`, `mucodec/model_1rvq.py`) are imported from the SongPrep repository at runtime. The repo must be cloned and available on the Python path.

5. **`weight_norm` must be removed before tracing**: The RVQ quantizer uses `weight_norm` on Conv1d layers. These parametrizations must be removed before `torch_neuronx.trace()` to avoid compilation failures.
