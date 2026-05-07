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

Input length varies by text. The model uses bucket-based compilation: compile at fixed sizes and pad inputs with replicate padding to the nearest bucket. Each frame = 25ms of audio (HOP_SIZE=600 at 24kHz).

Default buckets: `[64, 128, 192, 512, 768, 1024]` -- covering audio from ~1.6s to ~25.6s per chunk.

**Dead zones:** Certain bucket ranges fail to compile due to State Buffer overflow in Part B2 (Generator). Two Conv1d tensors (~15MB each) cannot share the 29MB SB at these sizes, but the compiler finds different tiling strategies outside them:

| Range | Status |
|-------|--------|
| 32-192 | Works |
| 256-384 | Dead zone (SB overflow) |
| 512-1312 | Works |
| 1344-2624 | Dead zone (SB overflow) |
| 2688+ | Works (tested to 3072) |

## Benchmark Results

**Validated:** 2026-03-12
**SDK:** Neuron SDK 2.27 (Deep Learning AMI Neuron Ubuntu 24.04 20260126)

### trn2.3xlarge (LNC=2, 4 NeuronCores)

#### Decoder Latency (P50, 100 iterations)

| Bucket | Audio | P50 Latency | Real-Time Factor | vs CPU Speedup |
|--------|-------|-------------|-----------------|---------------|
| 64 | 1.6s | 9.54ms | 168x | 24x |
| 128 | 3.2s | 17.70ms | 181x | 24x |
| 192 | 4.8s | 26.76ms | 179x | 26x |
| 512 | 12.8s | 79.64ms | 161x | 22x |
| 768 | 19.2s | 149.30ms | 129x | 19x |
| 1024 | 25.6s | 174.20ms | 147x | 21x |

CPU baseline on trn2 host: ~6-7x real-time (trn2.3xlarge has 16 vCPUs).

#### End-to-End (including CPU G2P + ALBERT)

| Text | Audio | Decoder | Total | RTF (decoder) | RTF (total) |
|------|-------|---------|-------|---------------|-------------|
| 58 chars | 4.0s | 31ms | 995ms | 128x | 4x |
| 212 chars | 13.9s | 162ms | 2.4s | 85x | 6x |
| 559 chars | 33.1s | 286ms | 5.6s | 116x | 6x |

Note: End-to-end RTF is dominated by CPU stages (ALBERT duration prediction). The Neuron decoder alone runs at 85-181x real-time; the CPU bottleneck brings effective throughput to ~4-6x real-time for single-request latency.

### inf2.xlarge (2 NeuronCores)

Requires `-O1` compiler flag due to Generator State Buffer constraints on inf2. Buckets 64 and 192 fail to compile (SB overflow even at `-O1`).

Note: inf2 benchmarks below use the old HOP_SIZE=300 and need re-validation. The real-time factors should approximately double with the corrected HOP_SIZE=600.

#### Decoder Latency (P50, 100 iterations, HOP_SIZE=300 -- needs re-validation)

| Bucket | Audio | P50 Latency | Real-Time Factor | vs CPU Speedup |
|--------|-------|-------------|-----------------|---------------|
| 32 | 0.8s | 7.26ms | 110x* | — |
| 96 | 2.4s | 25.61ms | 94x* | — |
| 128 | 3.2s | 31.17ms | 103x* | — |
| 160 | 4.0s | 40.98ms | 98x* | — |

*Estimated 2x correction from HOP_SIZE fix; actual values need re-measurement on inf2.

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

All 6 bucket copies fit on one NeuronCore (well within 24 GB HBM).

## Usage

```python
from kokoro_neuron import KokoroNeuron

# Compile and save (one-time, ~60s per bucket)
model = KokoroNeuron()
model.compile(buckets=[64, 128, 192, 512, 768, 1024])
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
| trn2.3xlarge | Yes | Primary target, 129-181x real-time, 6 default buckets |
| trn2.48xlarge | Yes (untested) | Should work, more cores for DP |
| inf2.xlarge | Yes (`-O1`) | ~94-110x real-time*, buckets 32/96/128/160 only |
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

1. **Bucket dead zones on trn2** -- Buckets 256-384 and 1344-2624 fail to compile (SB overflow). Default buckets skip these zones. Long text is automatically sub-chunked to fit available buckets (see Long-Form Generation above).
2. **inf2 requires `-O1` compiler flag** -- Default `-O2` causes SB overflow. Buckets 64 and 192 fail on inf2 even at `-O1`.
3. **CPU preprocessing bottleneck** -- ALBERT duration prediction runs on CPU and dominates end-to-end latency (~1-5s depending on text length). The Neuron decoder is 85-181x real-time but end-to-end is ~4-6x real-time due to CPU stages.
4. **Single-utterance batch size** -- Model traces at bs=1. Use DataParallel for throughput scaling.
5. **F.interpolate not traceable** -- The `har` computation uses F.interpolate(scale_factor=300) which the XLA tracer drops silently. Workaround: precompute on CPU.
6. **No cross-chunk prosody** -- Each chunk gets an independent style vector from the voice pack, selected by its phoneme count. Prosody does not carry across chunk boundaries (same limitation as the official Kokoro package).

## Example Checkpoints

* [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) -- automatically downloaded via `huggingface_hub`

## Maintainer

Jim Burtoft

**Last Updated:** 2026-03-12
