# Contrib Model: InternVL3-8B-Instruct

InternVL3-8B-Instruct is a vision-language model (VLM) running on AWS Trainium2 via NxD Inference. It supports both text-only and multimodal (text + image) inference using the NeuronBaseForImageToText framework.

**Maintainer:** Jim Burtoft ([@jimburtoft](https://github.com/jimburtoft))

## Model Information

- **HuggingFace ID:** [`OpenGVLab/InternVL3-8B-Instruct`](https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct)
- **Model Type:** Vision-language model (decoder-only transformer with vision encoder)
- **Parameters:** ~8B total (InternViT-300M vision encoder + Qwen2.5-7B text backbone)
- **Architecture:** GQA (28 heads / 4 KV heads), RoPE, RMSNorm (text); LayerNorm, GELU, absolute position embeddings (vision); pixel shuffle downsampling + 2-layer MLP projector
- **License:** MIT (Apache-2.0 for Qwen2.5 component)
- **Precision:** BF16

### Architecture Overview

| Component | Details |
|-----------|---------|
| **Vision encoder** | InternViT-300M-448px-V2.5: 24 layers, hidden=1024, 16 heads, patch_size=14, image_size=448 |
| **Projector** | Pixel shuffle (downsample_ratio=0.5, 1024→256 tokens) + LayerNorm + Linear(4096, 3584) + GELU + Linear(3584, 3584) |
| **Text backbone** | Qwen2.5-7B: 28 layers, hidden=3584, intermediate=18944, GQA (28/4), vocab=151674, tie_word_embeddings=False |

## Validation Results

**Validated:** 2026-04-28
**Instance:** trn2.3xlarge (LNC=2, TP=4)
**SDK:** Neuron SDK 2.29 (NxDI 0.9.17334, neuronx-cc 2.24.5133.0, PyTorch 2.9)

### Benchmark Results

#### Performance (TP=4, batch_size=1)

| Sequence Length | TTFT (ms) | TKG Throughput (tok/s) |
|----------------|-----------|------------------------|
| 2048 | 138 | 75.1 |
| 4096 | 230 | 58.9 |
| 8192 | 482 | 40.0 |
| 16384 | 1019 | 23.6 |
| 32768 | 2438 | 11.4 |

Vision encoder latency: 34.5 ms per 448x448 tile (batch=1).

#### GPU Comparison (1x NVIDIA L40S, BF16, SDPA)

| Metric | GPU (L40S) | Neuron (trn2.3xlarge TP=4) | Speedup |
|--------|------------|---------------------------|---------|
| TTFT (2048 input tokens) | 153.5 ms | 138 ms | 1.11x |
| Output tok/s (BS=1) | 40.5 | 75.1 | **1.85x** |

### Accuracy Validation

| Test | Status | Metrics |
|------|--------|---------|
| CTE logit comparison (vs CPU FP32) | PASS | cosine=0.9984, top-1 match, top-5 5/5, top-10 8/10 |
| TKG text generation | PASS | Correct, coherent output ("The capital of France is Paris.") |
| Multimodal generation | PASS | Vision encoder + text pipeline end-to-end working |

## Usage

### Prerequisites

```bash
# Activate NxDI environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Download model
huggingface-cli download OpenGVLab/InternVL3-8B-Instruct --local-dir /mnt/models/InternVL3-8B-Instruct/
```

### Compile and Run

```python
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Add contrib src to path
sys.path.insert(0, str(Path("contrib/models/InternVL3-8B-Instruct/src")))

from modeling_internvl3 import NeuronInternVL3ForCausalLM, InternVL3InferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig

MODEL_PATH = "/mnt/models/InternVL3-8B-Instruct/"
COMPILED_PATH = "/mnt/models/neuron_models/InternVL3-8B-Instruct/"

# Configure
text_neuron_config = NeuronConfig(
    tp_degree=4,
    max_batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=None,
    save_sharded_checkpoint=True,
)
vision_neuron_config = NeuronConfig(
    tp_degree=4,
    max_batch_size=1,
    seq_len=256,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=None,
    buckets=[1],
    fused_qkv=True,
    save_sharded_checkpoint=True,
)

config = InternVL3InferenceConfig.from_pretrained(
    MODEL_PATH,
    text_neuron_config=text_neuron_config,
    vision_neuron_config=vision_neuron_config,
)

# Compile (first time only)
model = NeuronInternVL3ForCausalLM(MODEL_PATH, config=config)
model.compile(COMPILED_PATH)

# Load and generate
model.load(COMPILED_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

outputs = model(input_ids=input_ids)
next_token = outputs.logits[0, -1].argmax().item()
print(f"{prompt} {tokenizer.decode([next_token])}")
# Output: The capital of France is Paris
```

### Multimodal Inference

```python
# Build input with vision tokens
IMG_CONTEXT_ID = 151667  # <IMG_CONTEXT>
IMG_START_ID = 151665    # <img>
IMG_END_ID = 151666      # </img>

text_ids = tokenizer("Describe this image:", return_tensors="pt").input_ids[0]
img_tokens = torch.full((256,), IMG_CONTEXT_ID, dtype=torch.long)

input_ids = torch.cat([
    text_ids,
    torch.tensor([IMG_START_ID]),
    img_tokens,
    torch.tensor([IMG_END_ID]),
]).unsqueeze(0)

# Pixel values for a single 448x448 tile
pixel_values = preprocess_image(image)  # [1, 3, 448, 448]

outputs = model(input_ids=input_ids, pixel_values=pixel_values)
```

## Compatibility Matrix

| Instance | SDK 2.29 | SDK 2.28 |
|----------|----------|----------|
| trn2.3xlarge (LNC=2, TP=4) | **VALIDATED** | Not tested |

## Example Checkpoints

* [OpenGVLab/InternVL3-8B-Instruct](https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct)

## Testing Instructions

```bash
# Ensure model is compiled first (see Usage above), then:
cd contrib/models/InternVL3-8B-Instruct/
pytest test/integration/test_model.py -v --tb=short
```

## Implementation Notes

### Three-File VLM Architecture

The model uses the NxDI `NeuronBaseForImageToText` framework with three files:

- `src/modeling_internvl3.py` — Top-level VLM orchestrating vision + text
- `src/modeling_internvl3_text.py` — Text model (Qwen2.5-7B) with vision embedding injection via `scatter_by_index_put()`
- `src/modeling_internvl3_vision.py` — Vision encoder (InternViT-300M) compiled via `torch_neuronx.trace()` with pixel shuffle and MLP projector

### Weight Mapping

| HuggingFace Key | NxDI Key |
|-----------------|----------|
| `language_model.model.layers.{i}.*` | `layers.{i}.*` |
| `language_model.model.embed_tokens.weight` | `embed_tokens.weight` |
| `language_model.model.norm.weight` | `norm.weight` |
| `language_model.lm_head.weight` | `lm_head.weight` |
| `vision_model.*` | Vision encoder (separate NEFF) |
| `mlp1.*` | Projector (part of vision NEFF) |

### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<IMG_CONTEXT>` | 151667 | Visual token placeholder in text sequence |
| `<img>` | 151665 | Image region start marker |
| `</img>` | 151666 | Image region end marker |
| `<|im_end|>` | 151645 | EOS token |

## Known Issues

- **V2PE not implemented**: Variable Visual Position Encoding (described in InternVL2.5/3 papers) is not implemented in the HuggingFace model code and is not included here. Standard position IDs are used. Accuracy validation passes without V2PE.
- **Batch size > 1**: Single-request batch inference (batch_size > 1) has a known issue with sampling_params shape. Use vLLM for multi-request concurrent serving.
- **trust_remote_code**: The HuggingFace tokenizer requires `trust_remote_code=True`. The NxDI model code reads config.json directly and does not require it.
- **NKI kernels**: Not applicable for this model. Qwen2.5-7B's `intermediate_size=18944` is incompatible with NxDI NKI `mlp_kernel` and `attn_block_tkg` kernels at tested TP degrees.

## vLLM Integration

This model can be served through vLLM-neuron with patches to the vllm-neuron worker. See the [NxD Inference vLLM User Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html) for general vLLM setup. InternVL3 requires modifications to vllm-neuron's model loader and runner to register the custom architecture. Contact the maintainer for patch details.
