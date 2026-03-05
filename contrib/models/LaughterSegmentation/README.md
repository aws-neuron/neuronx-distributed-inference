# Contrib Model: LaughterSegmentation

Laughter detection on AWS Inferentia2 using `torch_neuronx.trace()`.

## Model Information

- **HuggingFace ID:** `omine-me/LaughterSegmentation`
- **Model Type:** Wav2Vec2-based audio frame classifier
- **Parameters:** ~315M (FP32)
- **Base Model:** `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- **License:** Check HuggingFace model card

## Architecture Details

The model uses `Wav2Vec2ForAudioFrameClassification` to classify each audio frame as laughter or not-laughter. It takes 7-second audio windows at 16 kHz (112,000 samples) and outputs 349 per-frame binary predictions.

This model uses `torch_neuronx.trace()` (not NxD Inference) since it is an encoder-only classification model rather than an autoregressive LLM.

## Validation Results

**Validated:** 2026-03-04
**Instance:** inf2.xlarge (1 Inferentia2 chip, 2 NeuronCores)
**SDK:** Neuron SDK 2.28, PyTorch 2.9

### Benchmark Results (Single Core)

| Batch Size | Mean Latency | Throughput | Real-Time Factor |
|-----------|-------------|-----------|-----------------|
| 1 | 18.44 ms | 54.2 W/s | 380x |
| 2 | 21.27 ms | 94.0 W/s | 658x |
| 4 | 42.05 ms | 95.1 W/s | 666x |
| 8 | 83.93 ms | 95.3 W/s | 667x |

### DataParallel Results (Full Instance, 2 Cores)

| Configuration | Throughput | Real-Time Factor | Speedup |
|--------------|-----------|-----------------|---------|
| Single core (BS=2) | 94.0 W/s | 658x | 1.0x |
| DataParallel (2 cores) | 175.2 W/s | 1226x | 1.86x |

### Accuracy Validation

| Input | Cosine Similarity | Frame Agreement |
|-------|------------------|----------------|
| Random normal | 1.000000 | 100.00% |
| Quiet noise | 1.000000 | 100.00% |
| Loud signal | 1.000000 | 100.00% |
| Sine 440 Hz | 1.000000 | 100.00% |
| Silence | 0.999999 | 100.00% |

**Status:** VALIDATED

## Usage

The included notebook (`laughter_neuron_inf2.ipynb`) contains the complete workflow:

1. Download model weights from HuggingFace
2. Remove `weight_norm` parametrizations (required for SDK 2.28+)
3. Compile with `torch_neuronx.trace()` across multiple batch sizes
4. Benchmark single-core and DataParallel throughput
5. Validate accuracy against CPU reference
6. Run end-to-end inference on sample audio

```bash
# On an inf2 or trn2 instance with DLAMI
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
pip install jupyter safetensors librosa scipy pydub
jupyter notebook laughter_neuron_inf2.ipynb
```

### Known Issues

**weight_norm crash on SDK 2.28+**: Wav2Vec2 uses `weight_norm` on `pos_conv_embed.conv`, which crashes `torch_neuronx.trace()`. The notebook strips parametrizations before tracing:

```python
import torch.nn.utils.parametrize as parametrize
for name, module in model.named_modules():
    if hasattr(module, "parametrizations"):
        for param_name in list(module.parametrizations.keys()):
            parametrize.remove_parametrizations(module, param_name)
```

### trn2 Usage

To run on trn2.3xlarge, uncomment the `--lnc` compiler arg in the compilation cell.

## Compatibility Matrix

| Instance/Version | SDK 2.28 | SDK 2.27 and earlier |
|------------------|----------|---------------------|
| inf2.xlarge | VALIDATED | Not tested |
| trn2.3xlarge | Tested (see project notes) | Not tested |

## Example Checkpoints

* [omine-me/LaughterSegmentation](https://huggingface.co/omine-me/LaughterSegmentation)

## Maintainer

Community contribution

**Last Updated:** 2026-03-04
