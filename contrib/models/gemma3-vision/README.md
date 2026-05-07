# Contrib Model: Google Gemma3 VLM models

NeuronX Distributed Inference implementationn for Google Gemma3 VLM (Vision-Language Model) based on the HuggingFace Transformers Gemma3 architecture with SigLIP vision encoder.

## Model Information

- **HuggingFace IDs:**
    * [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)
    * [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it)
    * [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it)
- **Model Type:** LLaVA-style VLM with fixed-resolution SigLIP vision encode (400M) and Transformer-based LLM backbone.
- **License:** Check HuggingFace model card

## Architecture Details

LLM backbones (text models):

| Spec | Gemma 3 4B | Gemma 3 12B | Gemma 3 27B |
|---|---:|---:|---:|
| **Layers** | 34 | 48 | 62 |
| **Hidden Size** | 2560 | 3840 | 5376 |
| **Head Dim** | 256 | 256 | 128 |
| **Attention Heads** | 8 | 16 | 32 |
| **KV Heads** | 4 | 8 | 16 |
| **Intermediate Size** | 10240 | 15360 | 21504 |
| **Vocabulary size** | 32,064 | 32,064 | 32,064 |
| **Max Position Embeddings** | 131,072 | 131,072 | 131,072 |
| **Position Encoding** | RoPE | RoPE | RoPE |
| **Normalization** | RMSNorm | RMSNorm | RMSNorm |
| **Activation type** | GELU | GELU | GELU |
| **Context length** | 128K | 128K | 128K |

The 400M-parameter fixed-resolution SigLIP vision encoder is shared by all models:

| Spec | SigLIP vision tower |
|---|---:|
| **Layers** | 27 |
| **Hidden Size** | 1152 |
| **Head Dim** | 72 |
| **Attention Heads** | 16 |
| **KV Heads** | 16 |
| **Intermediate Size** | 4304 |
| **Activation type** | GELU |
| **Number of multi-modal tokens per image** | 256 |

## Validation Results

**Validated:** 2026-02-05  
**Configuration:** Trn1, TP=8, batch_size=1, seq_len=1024, float16, 1 image per sample

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS  | 100.0% match |
| Logits Matching | ⚠️ PARTIAL | ~56.2% match |

### Performance Metrics

| Metric | Value |
|--------|-------|
| E2E Throughput | 360.4 tokens/s |
| CTE Throughput | 49563.7 tokens/s |
| TKG Throughput | 223.8 tokens/s |

**Status:** ✅ GOOD

**Note:** Low token matching is due to sampling divergence at close probability tokens, not model incorrectness.

## Usage

```python
import torch

from gemma3_vision.modeling_gemma3 import NeuronGemma3ForConditionalGeneration
from gemma3_vision.utils import create_neuron_config

model_path = "/path/to/hf/artifacts"
compiled_model_path = "/path/to/compiled/artifacts"

# Create Neuron configuration
nrn_config = create_neuron_config(
    hf_config_path=config_file_path,
    text_batch_size=1,
    vision_batch_size=1,  # num_images_per_sample * batch_size
    total_max_seq_len=1024,
    torch_dtype=torch.bfloat16,
    lnc=1,  # Logical NC config
    tp_degree=8
)

# Initialize model
nrn_model = NeuronGemma3ForConditionalGeneration(
    model_path=model_path,
    config=nrn_config
)

# Compile and load
nrn_model.compile(compiled_model_path.as_posix())
nrn_model.load(compiled_model_path.as_posix())

# Generate (see integration test for full example)
```

## Compatibility Matrix

| Instance/Version | 2.27 | 2.26 and earlier |
|------------------|-------|------------------|
| Trn2             | ✅ Working | Not tested |
| Trn1             | ✅ Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/gemma3-vision/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/gemma3-vision
python3 -m test.integration.test_model
```

## Example Checkpoints

* [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)
* [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it)
* [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it)