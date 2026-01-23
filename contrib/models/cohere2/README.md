# Cohere Command R7B and Command A Models

Support for Cohere Command text models based on the HuggingFace Transformers Cohere2 architecture.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

from cohere2 import Cohere2NeuronConfig, Cohere2InferenceConfig, NeuronCohere2ForCausalLM

model_path = "/home/ubuntu/models/c4ai-command-r7b-12-2024/"
compiled_model_path = "/home/ubuntu/neuron-models/c4ai-command-r7b-12-2024/"

prompts = ["The color of the sky is"]

# Init Neuron model, HuggingFace tokenizer, and HuggingFace generation config.
neuron_config = Cohere2NeuronConfig(
    tp_degree=2,
    batch_size=1,
    max_context_length=128,
    seq_len=128,
    on_device_sampling_config=OnDeviceSamplingConfig(),
)
config = Cohere2InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)
model = NeuronCohere2ForCausalLM(model_path, config)
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

|Instance/Version	|2.25	|2.24 and earlier   |
|---	|---	|---	|
|Trn2	|Not tested	|Not tested	|
|Trn1	|Working	|Not tested	|
|Inf2	|Not tested	|Not tested	|

This matrix shows which Neuron inference features are supported with this model.

|Feature	|Status|
|---	|---	|
|Tensor Parallelism	|:white_check_mark:	|
|Sequence Parallelism	|:white_check_mark:	|
|Context Parallelism	|:x:	|
|Expert Parallelism	|Not applicable	|
|QKV Fusion	|:white_check_mark:	|
|Continous Batching	|:white_check_mark:	|
|On-Device Sampling	|:white_check_mark:	|
|Async Mode	|:white_check_mark:	|
|Bucketing	|:white_check_mark:	|
|Weight Quantization	|:white_check_mark:	|
|Activation Quantization	|:x:	|
|KV Cache Quantization	|:white_check_mark:	|
|KV Cache Tiling	|Not tested	|
|Flash Decoding	|:x:	|
|Fused QKV	|:white_check_mark:	|
|Prefix Caching	|:x:	|
|Paged Attention	|:x:	|
|Chunked Prefill	|:x:	|
|LoRA	|Not tested	|
|Speculation	|:x:	|
|Kernels	|:warning: - cf. Below	|

Supported kernels include:
* FlashAttention kernel (context encoding only)
* QKV kernel (QKV kernel with NBSD layout not tested)
* MLP kernel
* Quantized MLP kernel

As the model uses layer normalization, RMSNorm kernels are not applicable. As the model uses parallel 
attention and MLP layers, fused residual kernels are not applicable.

## Example Checkpoints

* https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024
* https://huggingface.co/CohereLabs/c4ai-command-a-03-2025

## Testing

The following command runs a set of end-to-end integration tests that compile the model and run it on Neuron to validate that it’s accurate and performant.

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/contrib/models/cohere2/src"
pytest contrib/models/cohere2/test/integration/test_model.py --capture=tee-sys
```

**Note:** In HuggingFace Transformers, the `HybridCache` KV-cache manager for hybrid SWA/global-attention models had a bug in 
its sliding-window update (see [Issue 37574](https://github.com/huggingface/transformers/issues/37574)) that was fixed 
in v4.52. To get the integration tests to pass, use the fixed KV-cache manager at `contrib/models/cohere2/src/cohere2/hybrid_kv_cache_manager.py`.
