# Contrib Model: Kokoro-82M

NeuronX implementation of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), an 82M-parameter text-to-speech model based on StyleTTS 2 and ISTFTNet. This is the first non-LLM contrib model -- it uses `torch_neuronx.trace()` with a 3-part architecture split to handle XLA tracer incompatibilities.

## Model Information

- **HuggingFace ID:** `hexgrad/Kokoro-82M`
- **Model Type:** Text-to-speech (parallel, non-autoregressive)
- **Parameters:** 82M
- **Architecture:** ALBERT (duration prediction) + StyleTTS 2 decoder + ISTFTNet vocoder
- **Output:** 24kHz mono audio
- **License:** Apache 2.0

## Architecture Details

Kokoro-82M generates an entire mel spectrogram in one forward pass (not autoregressive). The Neuron implementation splits the decoder into 3 traced parts to work around XLA tracer limitations:

| Part | Runs On | Latency | Function |
|------|---------|---------|----------|
| Part A | Neuron | ~0.8ms | Encoder + decode blocks 0-2 |
| Part B1 | Neuron | ~0.8ms | Decode block 3 (ConvTranspose1d decomposed) |
| har precompute | CPU | ~2.9ms | f0 -> harmonics (F.interpolate not traceable) |
| Part B2 | Neuron | ~16ms | Generator body (ISTFTNet) |

### Workarounds Applied

1. **CustomSTFT boolean mask** -> `torch.where` (XLA tracer crash)
2. **UpSample1d F.interpolate** -> `torch.repeat_interleave` (XLA drops input tensors)
3. **SineGen random ops** -> deterministic zeros (tracing compatibility)
4. **ConvTranspose1d(groups=1090)** -> element-wise decomposition (NCC_ITEN404 bug)
5. **Generator F.interpolate(scale_factor=300)** -> precompute `har` on CPU

### Bucketing

Input length varies by text. The model uses bucket-based compilation: compile at fixed sizes (32, 64, 96, 128, 160, 192 frames) and pad inputs with replicate padding to the nearest bucket. Each frame = 12.5ms of audio.

## Benchmark Results

**Validated:** 2026-03-11
**SDK:** Neuron SDK 2.27 (Deep Learning AMI Neuron Ubuntu 24.04 20260126)

### trn2.3xlarge (LNC=2, 4 NeuronCores)

#### Decoder Latency (P50, 100 iterations)

| Bucket | Audio | P50 Latency | Real-Time Factor | vs CPU Speedup |
|--------|-------|-------------|-----------------|---------------|
| 32 | 0.4s | 6.71ms | 60x | 12.6x |
| 64 | 0.8s | 11.24ms | 71x | 13.1x |
| 96 | 1.2s | 15.49ms | 77x | 14.0x |
| 128 | 1.6s | 20.47ms | 78x | 14.7x |
| 160 | 2.0s | 24.97ms | 80x | 15.0x |
| 192 | 2.4s | 30.68ms | 78x | 14.3x |

### inf2.xlarge (2 NeuronCores)

Requires `-O1` compiler flag due to Generator State Buffer constraints on inf2. Buckets 64 and 192 fail to compile (SB overflow even at `-O1`).

#### Decoder Latency (P50, 100 iterations)

| Bucket | Audio | P50 Latency | Real-Time Factor | vs CPU Speedup |
|--------|-------|-------------|-----------------|---------------|
| 32 | 0.4s | 7.26ms | 55x | 43.2x |
| 96 | 1.2s | 25.61ms | 47x | 35.4x |
| 128 | 1.6s | 31.17ms | 51x | 39.6x |
| 160 | 2.0s | 40.98ms | 49x | 38.3x |

Note: inf2.xlarge CPU baseline is ~1.3x real-time (4 vCPUs), so the Neuron speedup vs CPU is much larger than on trn2 (which has faster host CPUs).

### Accuracy

| Instance | Cosine vs CPU | SNR vs CPU | bf16 autocast error |
|----------|--------------|------------|-------------------|
| trn2.3xlarge | 0.985 | 15.0 dB | 0.0 (exact) |
| inf2.xlarge | 0.993 | — | 0.0 (exact) |

### NEFF Sizes

| Part | Size |
|------|------|
| Part A | 93.0 MB |
| Part B1 | 10.7 MB |
| Part B2 | 31-34 MB |
| Total (per bucket) | ~137 MB |

All 6 bucket copies fit on one NeuronCore (~822 MB, well within 24 GB HBM).

## Usage

```python
from kokoro_neuron import KokoroNeuron

# Compile and save (one-time, ~60s per bucket)
model = KokoroNeuron()
model.compile(buckets=[64, 128, 192])
model.save("compiled_models")

# Load pre-compiled models
model = KokoroNeuron.load("compiled_models")
model.warmup()

# Generate speech (short text, single chunk)
audio = model.generate("Hello, this is Kokoro running on Neuron!", voice="af_heart")
# audio: numpy array, float32, 24kHz

# Generate speech (long text, auto-chunked with crossfade stitching)
long_text = """The quick brown fox jumps over the lazy dog. This sentence is just
the beginning of a much longer passage that demonstrates Kokoro's ability to handle
arbitrarily long text inputs by automatically splitting at sentence boundaries."""
audio = model.generate(long_text, voice="af_heart")

# With timing information
audio, timings = model.generate_timed("Performance test.")
print(f"Decoder: {timings['total']*1000:.1f}ms, {timings['audio_duration']:.2f}s audio")
print(f"Chunks: {timings['num_chunks']}")
```

### Long-Form Generation

`generate()` automatically handles text of any length by delegating to KPipeline's
phoneme-aware text chunking (splits at sentence/clause boundaries: `!.?` > `:;` > `,`).
Each chunk is decoded on Neuron, then chunks are stitched with a 25ms linear crossfade
to eliminate click artifacts at boundaries.

For streaming (low-latency, process-as-you-go), use the generator API:

```python
for audio_chunk in model.generate_stream("Very long text...", voice="af_heart"):
    play(audio_chunk)  # each chunk is a numpy array at 24kHz
```

### Saving Audio

```python
import scipy.io.wavfile as wavfile
import numpy as np

audio = model.generate("Hello world!")
audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
wavfile.write("output.wav", 24000, audio_int16)
```

## Compatibility Matrix

| Instance | Supported | Notes |
|----------|-----------|-------|
| trn2.3xlarge | Yes | Primary target, 60-80x real-time, all buckets |
| trn2.48xlarge | Yes (untested) | Should work, more cores for DP |
| inf2.xlarge | Yes (`-O1`) | 47-55x real-time, buckets 32/96/128/160 only |
| inf2.8xlarge | Yes (untested, `-O1`) | Same as inf2.xlarge, more cores for DP |

**inf2 note:** Requires `-O1` compiler optimization flag. The Generator's Conv1d ops need 310KB SB at `-O2`, exceeding inf2's 196KB limit. At `-O1`, the compiler uses a different allocation strategy that fits for most bucket sizes. Buckets 64 and 192 fail to compile on inf2 even with `-O1`.

| Neuron SDK | Status |
|------------|--------|
| 2.27+ | Tested |
| 2.26 and earlier | Untested |

## System Requirements

### Python packages

```
kokoro>=0.8
misaki[en]>=0.8
espeak-ng  # system package, not pip
torch-neuronx>=2.9
neuronx-cc>=2.22
```

### System packages

```bash
sudo apt-get install -y espeak-ng
```

### Pre-installed environment (recommended)

On the Deep Learning AMI Neuron (Ubuntu 24.04):

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
pip install kokoro misaki[en]
sudo apt-get install -y espeak-ng
```

## Testing

```bash
# Set up compiled model directory (optional, tests will compile if needed)
export KOKORO_COMPILED_DIR=/home/ubuntu/compiled_models_kokoro

# Run with pytest
cd contrib/models/Kokoro-82M/test/integration
pytest test_model.py -v --capture=tee-sys

# Or run standalone
python test_model.py
```

## Known Limitations

1. **Per-chunk max: 192 frames (~2.4s audio) on trn2, 160 frames (~2.0s audio) on inf2** -- Generator State Buffer limits per Neuron inference call. Long text is automatically split into chunks and stitched (see Long-Form Generation above).
2. **inf2 requires `-O1` compiler flag** -- Default `-O2` causes SB overflow. Buckets 64 and 192 fail on inf2 even at `-O1`.
3. **CPU preprocessing required** -- ALBERT duration prediction and harmonic precomputation run on CPU. These add ~5-15ms depending on text length but are not the bottleneck.
4. **Single-utterance batch size** -- Model traces at bs=1. Use DataParallel for throughput scaling.
5. **F.interpolate not traceable** -- The `har` computation uses F.interpolate(scale_factor=300) which the XLA tracer drops silently. Workaround: precompute on CPU.
6. **No cross-chunk prosody** -- Each chunk gets an independent style vector from the voice pack, selected by its phoneme count. Prosody does not carry across chunk boundaries (same limitation as the official Kokoro package).

## Example Checkpoints

* [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) -- automatically downloaded via `huggingface_hub`

## Maintainer

Jim Burtoft

**Last Updated:** 2026-03-11
