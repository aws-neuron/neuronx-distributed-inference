# Gemma3-Vision Model

Support for Google Gemma3-Vision VLM (Vision-Language Model) based on the HuggingFace Transformers Gemma3 architecture with SigLIP vision encoder.

## Architecture

Gemma3-Vision is a multimodal model that combines:
- **Text Model**: Gemma3 language model with sliding window attention
- **Vision Encoder**: SigLIP vision transformer with average pooling
- **Multimodal Projector**: Linear projection to align vision and text spaces

The model uses a dual configuration architecture with separate NeuronConfig instances for text and vision components.

## Usage

### Text + Image Generation

```python
import torch
from transformers import AutoTokenizer, AutoProcessor, GenerationConfig
from PIL import Image

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama4.utils.input_processor import (
    prepare_generation_inputs_hf
)
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)

from gemma3_vision import (
    NeuronGemma3ForCausalLM,
    Gemma3InferenceConfig,
)

model_path = "/home/ubuntu/models/google/gemma-3-27b-it/"
compiled_model_path = "/home/ubuntu/neuron-models/gemma-3-27b-it/"
image_path = "/path/to/image.jpg"

# Create dual configs
text_config = NeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
    fused_qkv=True,
    attn_kernel_enabled=True,
    enable_bucketing=True,
    context_encoding_buckets=[2048],
    token_generation_buckets=[2048],
    is_continuous_batching=True,
    ctx_batch_size=1,
)

vision_config = NeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
    fused_qkv=False,  # SigLIP requires separate QKV
    attn_kernel_enabled=True,
    enable_bucketing=True,
    buckets=[1],  # Auto-bucketing for vision
    is_continuous_batching=True,
    ctx_batch_size=1,
)

# Initialize model
config = Gemma3InferenceConfig(
    text_neuron_config=text_config,
    vision_neuron_config=vision_config,
    load_config=load_pretrained_config(model_path),
)

model = NeuronGemma3ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Prepare inputs
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
processor = AutoProcessor.from_pretrained(model_path)
generation_config = GenerationConfig.from_pretrained(model_path)

text_prompt = "Describe this image"
input_ids, attention_mask, pixel_values, vision_mask = prepare_generation_inputs_hf(
    text_prompt, image_path, processor, 'user', config
)

# Generate
generation_model = HuggingFaceGenerationAdapter(model)
outputs = generation_model.generate(
    input_ids,
    generation_config=generation_config,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    vision_mask=vision_mask.to(torch.bool),
    max_new_tokens=100,
)

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(output_text[0])
```

### Text-Only Generation

```python
# Same setup as above, but prepare inputs without image
text_prompt = "What is the capital of France?"
input_ids, attention_mask, _, _ = prepare_generation_inputs_hf(
    text_prompt, None, processor, 'user'
)

outputs = generation_model.generate(
    input_ids,
    generation_config=generation_config,
    attention_mask=attention_mask,
    max_new_tokens=100,
)

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(output_text[0])
```

## Compatibility Matrix

### Neuron SDK Versions and Instance Types

|Instance/Version	|2.27.1+	|2.26 and earlier   |
|---	|---	|---	|
|Trn2	|Working	|Not tested	|
|Trn1	|Working	|Not compatible (API breaking changes)	|
|Inf2	|Working	|Not tested	|

### Supported Features

|Feature	|Status|Notes|
|---	|---	|---	|
|Tensor Parallelism	|:white_check_mark:	|Tested with TP=8|
|Sequence Parallelism	|:x:	|Not supported|
|Context Parallelism	|:x:	|Not supported|
|Expert Parallelism	|Not applicable	||
|QKV Fusion	|:white_check_mark:	|Text model only|
|Continuous Batching	|:white_check_mark:	||
|On-Device Sampling	|:white_check_mark:	||
|Async Mode	|:white_check_mark:	||
|Bucketing	|:white_check_mark:	|Dual bucketing for text/vision|
|Weight Quantization	|:white_check_mark:	|Excludes vision components|
|Activation Quantization	|:x:	|Not supported|
|KV Cache Quantization	|:x:	|Not supported|
|Flash Decoding	|:x:	|Not supported|
|Prefix Caching	|:x:	|Not supported|
|Paged Attention	|:x:	|Not supported|
|Chunked Prefill	|:x:	|Not supported|
|Speculation	|:x:	|Not supported|
|Attention Kernels	|:white_check_mark:	|Context encoding only|

## Architecture Details

### Dual Configuration

Gemma3-Vision requires separate NeuronConfig instances for text and vision:

- **Text Config**: `fused_qkv=True`, bucketing for variable sequence lengths
- **Vision Config**: `fused_qkv=False`, auto-bucketing from 1024 to seq_len

This is necessary because SigLIP vision encoder has different architectural requirements than the Gemma3 text model.

### Vision Encoder

The vision encoder uses:
- **SigLIP**: Vision transformer with layer normalization
- **Average Pooling**: Reduces patch embeddings to fixed number of tokens
- **Linear Projection**: Projects vision embeddings to text model's hidden size

### Quantization

When using quantization, the following components must be excluded:
- `multi_modal_projector`: Vision-to-text projection layer
- `vision_tower`: Entire SigLIP encoder
- All `self_attn` layers in the language model
- `lm_head`: Final output projection

### Compiler Optimization Levels

- Vision encoder: `-O1` (faster compilation)
- Context encoding: `-O1` (balanced)
- Token generation: `-O2` (maximum optimization)

## Example Checkpoints

* https://huggingface.co/google/gemma-3-27b-it

## Testing

Run integration tests to validate model accuracy and performance:

```bash
cd /home/ubuntu/nxdi-gemma3-contribution/contrib/models/gemma3-vision && PYTHONPATH="src:/home/ubuntu/nxdi-gemma3-contribution/src:$PYTHONPATH" uv run python -m test.integration.test_model
```

Run all tests (integration + unit):

```bash
pytest contrib/models/gemma3-vision/test/ --capture=tee-sys
```
