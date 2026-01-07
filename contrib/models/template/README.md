# Contrib Model Example/Template: Llama (Text)

Support for Llama text models from the Llama 2 and Llama 3 collections.

## Usage

```
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

model_path = "/home/ubuntu/models/Llama-3.2-1B/"
compiled_model_path = "/home/ubuntu/neuron_models/Llama-3.2-1B/"

prompts = ["The color of the sky is"]

# Init Neuron model, HuggingFace tokenizer, and HuggingFace generation config.
neuron_config = NeuronConfig(
    tp_degree=32,
    batch_size=1,
    max_context_length=128,
    seq_len=128,
    on_device_sampling_config=OnDeviceSamplingConfig(),
)
config = LlamaInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)
model = NeuronLlamaForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
generation_config = GenerationConfig.from_pretrained(model_path)

# Run generation with HuggingFaceGenerationAdapter.
generation_model = HuggingFaceGenerationAdapter(model)
inputs = tokenizer(prompts, padding=True, return_tensors="pt")
outputs = generation_model.generate(
    inputs.input_ids,
    generation_config=generation_config,
    attention_mask=inputs.attention_mask,
    max_length=model.neuron_config.max_length,
)
output_tokens = tokenizer.batch_decode(
    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Generated outputs:")
for i, output_token in enumerate(output_tokens):
    print(f"Output {i}: {output_token}")
```

## Compatibility Matrix

This matrix shows which Neuron SDK versions and instance types are tested with this model.

|Instance/Version	|2.24	|2.23 and earlier   |
|---	|---	|---	|
|Trn2	|Not tested	|Not tested	|
|Trn1	|Working	|Not tested	|
|Inf2	|Not working	|Not tested	|

## Example Checkpoints

* https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
* https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
* https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

## Testing

The following command runs a set of end-to-end integration tests that compile the model and run it on Neuron to validate that it’s accurate.

```
pytest contrib/models/template/test/test_model.py --capture=tee-sys
```