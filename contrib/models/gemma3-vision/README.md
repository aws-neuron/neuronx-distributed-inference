# Contrib Model: Gemma3-Vision

Support for Google Gemma3-Vision VLM (Vision-Language Model) based on the HuggingFace Transformers Gemma3 architecture with SigLIP vision encoder.

## Model Information

- **HuggingFace ID:** [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it)
- **Model Type:** Transformer decoder with a SigLIP vision encoder
- **License:** Check HuggingFace model card

## Usage

### Prerequisites

Download the Gemma-3-27b-it model from HuggingFace:

```bash
huggingface-cli download google/gemma-3-27b-it --local-dir /home/ubuntu/models/google/gemma-3-27b-it/
```

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
    seq_len=1024,
    torch_dtype=torch.bfloat16,
    fused_qkv=True,
    attn_kernel_enabled=True,
    enable_bucketing=True,
    context_encoding_buckets=[1024],
    token_generation_buckets=[1024],
    is_continuous_batching=True,
    ctx_batch_size=1,
)

vision_config = NeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=1024,
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

## Testing

Run integration tests:

```bash
export PYTHONPATH="/home/ubuntu/nxdi-gemma3-contribution/contrib/models/gemma3-vision/src:$PYTHONPATH"
pytest contrib/models/gemma3-vision/test/integration/test_model.py --capture=tee-sys
```

Run all tests (integration + unit):

```bash
pytest contrib/models/gemma3-vision/test/ --capture=tee-sys
```

## Example Checkpoints

* gemma-3-27b-it

## Maintainer

AWS Generative AI Innovation Center

**Last Updated:** 2026-02-05
