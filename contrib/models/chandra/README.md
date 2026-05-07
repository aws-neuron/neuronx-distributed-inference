# Contrib Model: Chandra OCR VLM

NeuronX Distributed Inference validation of Chandra, a Qwen3-VL-8B fine-tune for OCR with layout preservation.

> **Note:** This is a **full multimodal VLM validation** with actual image input, not text-backbone-only. Chandra uses NxDI's built-in `NeuronQwen3VLForCausalLM` + `ImageToTextInferenceConfig` pipeline -- no custom modeling code is required.

## Model Information

- **HuggingFace ID:** `datalab-to/chandra`
- **Model Type:** Vision-Language Model (Qwen3-VL architecture)
- **Parameters:** ~9B (BF16)
- **Architecture:** Qwen3-VL with ViT vision encoder + decoder-only transformer text backbone, GQA (32 Q / 8 KV heads), M-RoPE, SwiGLU MLP
- **License:** Check [HuggingFace model card](https://huggingface.co/datalab-to/chandra)
- **Use Case:** OCR with layout preservation -- outputs structured markdown/HTML from document images

## Validation Results

**Validated:** 2026-03-07
**Instance:** trn2.3xlarge (LNC=2, 4 logical NeuronCores)
**SDK:** Neuron SDK 2.28 (DLAMI 20260227), PyTorch 2.9, Python 3.12

### Benchmark Results

| Configuration | Throughput | Notes |
|--------------|-----------|-------|
| batch_size=1, single request | 76.8 tok/s | Steady state, handwritten form (1623 tokens) |
| batch_size=1, single request | 73.3 tok/s | Steady state, benchmark document (278 tokens) |
| batch_size=4, 4 concurrent | 120.9 tok/s (peak) | Requires NxDI batch fix (see Known Issues) |

### Accuracy Validation

Chandra achieves 83.1% on olmOCR-bench (state of the art as of 2026-03). On Neuron, the model produces identical OCR output to GPU -- the Neuron compilation does not degrade accuracy since the model runs in native BF16 with `cast_type=as-declared`.

Validation was performed with real document images (handwritten forms, benchmark documents) producing structured markdown output with layout preservation.

## Usage

### Quick Start (vLLM)

```python
import os
os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"

from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor

MODEL_PATH = "/path/to/chandra"

# Neuron configs for trn2.3xlarge (LNC=2, tp=4)
text_neuron_config = {
    "batch_size": 1, "ctx_batch_size": 1, "tkg_batch_size": 1,
    "seq_len": 8192, "max_context_length": 8192,
    "enable_bucketing": True,
    "context_encoding_buckets": [1024, 4096, 8192],
    "token_generation_buckets": [1024, 4096, 8192],
    "world_size": 4, "tp_degree": 4,
    "torch_dtype": "bfloat16",
    "rpl_reduce_dtype": "bfloat16", "attention_dtype": "bfloat16",
    "cast_type": "as-declared",
    "logical_neuron_cores": 2, "cc_pipeline_tiling_factor": 2,
    "fused_qkv": True,
    "qkv_kernel_enabled": True, "mlp_kernel_enabled": False,
    "attn_kernel_enabled": True,
}
vision_neuron_config = {
    "batch_size": 1, "seq_len": 4096, "max_context_length": 4096,
    "enable_bucketing": True, "buckets": [1024, 4096],
    "world_size": 4, "tp_degree": 4,
    "torch_dtype": "bfloat16", "rpl_reduce_dtype": "bfloat16",
    "cast_type": "as-declared",
    "logical_neuron_cores": 2, "cc_pipeline_tiling_factor": 2,
    "fused_qkv": True,
    "attn_kernel_enabled": False, "mlp_kernel_enabled": False,
}

llm = LLM(
    model=MODEL_PATH, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=4, max_num_seqs=1, max_model_len=8192,
    additional_config=dict(override_neuron_config=dict(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
    )),
    limit_mm_per_prompt={"image": 1},
    enable_prefix_caching=False, enable_chunked_prefill=False,
)

# Run OCR on an image
processor = AutoProcessor.from_pretrained(MODEL_PATH)
image = Image.open("document.png").convert("RGB")

# Resize if needed (max 1024px long side to fit vision bucket)
w, h = image.size
if max(w, h) > 1024:
    scale = 1024 / max(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Convert the document in this image to markdown. Preserve the layout and formatting."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = llm.generate(
    [{"prompt": prompt, "multi_modal_data": {"image": [image]}}],
    SamplingParams(top_k=1, max_tokens=4096, temperature=0.0),
)
print(outputs[0].outputs[0].text)
```

### Using Helper Module

```python
from src.modeling_chandra import load_chandra_vllm, run_chandra_ocr
from PIL import Image
from transformers import AutoProcessor

MODEL_PATH = "/path/to/chandra"

llm = load_chandra_vllm(model_path=MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
image = Image.open("document.png").convert("RGB")

result = run_chandra_ocr(llm, image, processor)
print(f"Tokens: {result['num_tokens']}, Speed: {result['tokens_per_sec']:.1f} tok/s")
print(result['text'])
```

### vLLM Online Serving

```bash
VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference vllm serve /path/to/chandra \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --max-num-seqs 1 \
    --max-model-len 8192 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --limit_mm_per_prompt '{"image": 1}' \
    --additional-config '{"override_neuron_config": {"text_neuron_config": {"batch_size": 1, "ctx_batch_size": 1, "tkg_batch_size": 1, "seq_len": 8192, "max_context_length": 8192, "enable_bucketing": true, "context_encoding_buckets": [1024, 4096, 8192], "token_generation_buckets": [1024, 4096, 8192], "world_size": 4, "tp_degree": 4, "torch_dtype": "bfloat16", "rpl_reduce_dtype": "bfloat16", "attention_dtype": "bfloat16", "cast_type": "as-declared", "logical_neuron_cores": 2, "cc_pipeline_tiling_factor": 2, "fused_qkv": true, "qkv_kernel_enabled": true, "mlp_kernel_enabled": false, "attn_kernel_enabled": true}, "vision_neuron_config": {"batch_size": 1, "seq_len": 4096, "max_context_length": 4096, "enable_bucketing": true, "buckets": [1024, 4096], "world_size": 4, "tp_degree": 4, "torch_dtype": "bfloat16", "rpl_reduce_dtype": "bfloat16", "cast_type": "as-declared", "logical_neuron_cores": 2, "cc_pipeline_tiling_factor": 2, "fused_qkv": true, "attn_kernel_enabled": false, "mlp_kernel_enabled": false}}}' \
    --port 8000
```

Then send requests via OpenAI-compatible API:

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

with open("document.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="/path/to/chandra",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        {"type": "text", "text": "Convert the document in this image to markdown."},
    ]}],
    max_tokens=4096, temperature=0.0,
)
print(response.choices[0].message.content)
```

## Compatibility Matrix

| Instance/Version | SDK 2.28 | SDK 2.27 |
|------------------|----------|----------|
| trn2.3xlarge (LNC=2, tp=4) | VALIDATED | Not tested |
| inf2.8xlarge (tp=2) | Not viable (13.8 tok/s) | Not tested |
| Trn1 | Not tested | Not tested |

## Example Checkpoints

* [datalab-to/chandra](https://huggingface.co/datalab-to/chandra) (9B BF16, ~17 GB)

## Testing Instructions

### Prerequisites

1. A trn2.3xlarge instance with Neuron SDK 2.28 (DLAMI 20260227)
2. Download the model: `huggingface-cli download datalab-to/chandra --local-dir ~/models/chandra`
3. Activate the venv: `source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate`

### Run Tests

```bash
# Set model path (default: ~/models/chandra)
export CHANDRA_MODEL_PATH=~/models/chandra

# Run with pytest
cd contrib/models/chandra
VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference \
    pytest test/integration/test_model.py -v --capture=tee-sys

# Or run directly
VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference \
    python test/integration/test_model.py
```

### Expected Output

First run includes compilation (~7 min). Subsequent runs use the compile cache (~60s load).

```
test_smoke_load              PASS
test_config_valid            PASS
test_image_resize            PASS
test_ocr_generates_output    PASS  (generates tokens from image input)
test_ocr_accuracy            PASS  (recognizes text in synthetic image)
test_throughput              PASS  (>50 tok/s steady state)
test_output_not_repetitive   PASS  (coherent, non-degenerate output)
```

## Known Issues

1. **mlp_kernel_enabled must be False on trn2.3xlarge (LNC=2)**: The fused MLP TKG kernel exceeds SBUF capacity under LNC=2 (requested 24576 bytes, free 9908). Use `mlp_kernel_enabled=False`.

2. **Vision encoder kernels not supported**: `attn_kernel_enabled` and `mlp_kernel_enabled` must both be False in `vision_neuron_config` for Qwen3-VL.

3. **Image patch limit**: Images must produce <=4096 patches to fit the vision encoder bucket. Resize images to ~1024px on the long side. A 1901x1224 image produces 8968 patches and fails.

4. **batch_size > 1 requires NxDI patches**: Three bugs in NxDI prevent batch_size > 1 for Qwen3-VL. With patches applied, batch_size=4 achieves 120.9 tok/s. See [batch bug fix branch](https://github.com/jimburtoft/neuronx-distributed-inference/tree/fix/qwen3-vl-batch-size-gt1) for details.

5. **prefix_caching and chunked_prefill must be disabled**: These features are not supported with NxDI VLM models.

6. **First compilation takes ~7 minutes**: Subsequent runs use the compile cache at `/var/tmp/neuron-compile-cache/`. If compilation is interrupted, delete failed cache entries before retrying.

7. **inf2 not viable**: inf2.8xlarge achieves only 13.8 tok/s due to ISA kernel compiler bugs on the trn1 target and fundamentally slower hardware.

## Maintainer

Jim Burtoft

**Last Updated:** 2026-03-07
