# Contrib Model: Qwen3.5-2B

NeuronX Distributed Inference implementation of Qwen3.5-2B, a 2B parameter dense model from Alibaba Cloud with a hybrid DeltaNet + GQA attention architecture.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen3.5-2B`
- **Model Type:** Decoder-only hybrid DeltaNet/GQA transformer
- **Parameters:** ~2B (BF16)
- **Architecture:** 24 layers (18 DeltaNet linear attention + 6 standard GQA), dense SwiGLU MLP, partial RoPE, tied embeddings
- **License:** Apache 2.0

### Key Architecture Details

| Feature | Value |
|---------|-------|
| Layers | 24 (18 DeltaNet + 6 GQA, pattern: [3 DeltaNet + 1 GQA] x 6) |
| Hidden Size | 2048 |
| GQA Attention | 8 Q heads, 2 KV heads, head_dim=256 |
| DeltaNet Attention | 16 value heads, 16 key heads, k_dim=v_dim=128 |
| MLP | Dense SwiGLU (intermediate_size=6144) |
| Position Encoding | Partial RoPE (25% of head_dim), mRoPE for VL |
| Vocabulary | 248,320 |
| Tied Embeddings | Yes |

The DeltaNet layers use linear recurrent attention (gated delta rule) instead of softmax attention, requiring custom NKI kernels for execution on Neuron. A fused single-kernel chunked forward handles context encoding (CTE), while a per-token recurrent kernel handles token generation (TKG).

## Validation Results

**Validated:** 2026-04-23
**Instance:** trn2.3xlarge (TP=4, LNC=2)
**SDK:** Neuron SDK 2.29, PyTorch 2.9, NKI 0.3.0

### Benchmark Results

All benchmarks on trn2.3xlarge, TP=4, LNC=2, BF16. Chat-formatted prompt (~19 input tokens). Throughput is total tokens/sec across all batch items.

#### Batch Size Scaling (seq_len=128)

| Batch Size | TTFT (ms) | Throughput (tok/s) | Per-Request (tok/s) |
|:----------:|:---------:|:------------------:|:-------------------:|
| 1 | 157.8 | 114.5 | 114.5 |
| 2 | 72.0 | 233.1 | 116.5 |
| 4 | 104.4 | 329.6 | 82.4 |
| 8 | 185.6 | 409.5 | 51.2 |

#### Sequence Length Scaling (BS=1)

| seq_len | TTFT (ms) | Throughput (tok/s) |
|:-------:|:---------:|:------------------:|
| 128 | 157.8 | 114.5 |
| 512 | 54.3 | 138.1 |
| 1024 | 102.7 | 125.3 |
| 2048 | 199.7 | 106.5 |
| 4096 | 401.7 | 80.3 |

### Accuracy Validation

9/9 integration tests pass. Accuracy is validated through:

1. **First-token logit comparison** against pre-computed CPU BF16 reference logits:
   - Cosine similarity: 0.9156 (threshold: 0.85) on TP shard 0
   - Top-1 token agreement: True (both CPU and Neuron predict "Paris")
   - Top-5 overlap: 4/5 (threshold: 3)

2. **Multi-prompt coherence tests** with chat-formatted prompts:
   - Factual Q&A: "What is the capital of France?" produces correct answer
   - Code generation: "Write a Python fibonacci function" produces valid code
   - Knowledge: "What is the largest ocean on Earth?" produces correct answer
   - List generation: "List two ingredients for a chocolate cake" produces valid list

**Note on multi-token logit validation:** DeltaNet layers (18 of 24) use NKI linear recurrent kernels that produce higher BF16 numerical divergence than standard GQA. Autoregressive sequences diverge after the first generated token, making multi-token `logit_validation()` inapplicable. The first-token logits are validated where CPU and Neuron process identical input prefixes. Additionally, the model outputs TP-sharded logits (vocab/tp_degree) because `ModelWrapper` does not call `_gather_along_dim`, so comparison uses the TP shard 0 slice.

## Usage

```python
import json
import os
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

model_path = "/path/to/Qwen3.5-2B"
compiled_path = "/scratch/qwen35_2b_traced/"

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    save_sharded_checkpoint=True,
)

# Read config.json directly (model_type 'qwen3_5' may not be
# registered in all transformers versions)
with open(os.path.join(model_path, "config.json")) as f:
    hf_config = json.load(f)
text_config = hf_config.get("text_config", hf_config)
config_dict = dict(text_config)
config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)

config = Qwen35InferenceConfig(
    neuron_config=neuron_config,
    **config_dict,
)

# Compile
model = NeuronQwen35ForCausalLM(model_path, config)
model.compile(compiled_path)

# Load
model = NeuronQwen35ForCausalLM(compiled_path)
model.load(compiled_path)

# Generate with chat template (recommended)
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
gen_config = GenerationConfig(
    do_sample=True, top_k=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

messages = [{"role": "user", "content": "What is the capital of France?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, padding=True, return_tensors="pt")
gen_model = HuggingFaceGenerationAdapter(model)
outputs = gen_model.generate(
    inputs.input_ids,
    generation_config=gen_config,
    attention_mask=inputs.attention_mask,
    max_new_tokens=80,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Note:** Qwen3.5-2B is a chat model. Use `tokenizer.apply_chat_template()` for best results. Raw text prompts may produce echoey output.

**Note on `seq_len`:** The `seq_len` parameter is the total sequence budget (input + generated tokens). Do not pad inputs to `max_length=seq_len`. Use `padding=True` for automatic minimal padding.

## Compatibility Matrix

| Instance | TP | SDK 2.29 | SDK 2.28 |
|----------|-----|----------|----------|
| trn2.3xlarge (LNC=2) | 4 | VALIDATED | Not tested |

### Tested Configurations (trn2.3xlarge, TP=4, LNC=2)

| Batch Size | seq_len | Status |
|:----------:|:-------:|:------:|
| 1 | 128 | VALIDATED |
| 2 | 128 | VALIDATED |
| 4 | 128 | VALIDATED |
| 8 | 128 | VALIDATED |
| 1 | 512 | VALIDATED |
| 1 | 1024 | VALIDATED |
| 1 | 2048 | VALIDATED |
| 1 | 4096 | VALIDATED |
| 2 | 1024 | VALIDATED |
| 4 | 512 | VALIDATED |

## Example Checkpoints

* [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (BF16, ~4 GB)

## Testing Instructions

### Unit Tests (CPU only)

```bash
cd contrib/models/Qwen3.5-2B/
pytest test/unit/ -v
```

### Integration Tests (requires trn2 instance)

```bash
cd contrib/models/Qwen3.5-2B/
# Activate SDK 2.29 environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

QWEN35_MODEL_PATH=/mnt/models/Qwen3.5-2B \
QWEN35_COMPILED_PATH=/mnt/models/qwen35_2b_traced \
QWEN35_LOGIT_COMPILED_PATH=/mnt/models/qwen35_2b_traced_logits \
QWEN35_REF_LOGITS_PATH=/mnt/models/qwen35_2b_cpu_reference_logits_bf16.pt \
pytest test/integration/test_model.py --capture=tee-sys -v
```

Environment variables:
- `QWEN35_MODEL_PATH` — Path to HF model weights (required)
- `QWEN35_COMPILED_PATH` — Path for compiled artifacts (default: `/tmp/qwen35_2b_traced`)
- `QWEN35_LOGIT_COMPILED_PATH` — Path to model compiled with `output_logits=True` for logit validation (optional; test skips if not provided)
- `QWEN35_REF_LOGITS_PATH` — Path to pre-computed CPU BF16 reference logits for logit validation (optional; test skips if not provided)
- `QWEN35_TP_DEGREE` — Tensor parallelism degree (default: 4)
- `QWEN35_SEQ_LEN` — Max sequence length (default: 128)

#### Generating CPU Reference Logits

The `qwen3_5` model type requires `transformers>=5.0`. Generate BF16 reference logits in a separate environment:

```bash
python3 -m venv /tmp/cpu_ref_venv && source /tmp/cpu_ref_venv/bin/activate
pip install torch transformers accelerate
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
model = AutoModelForCausalLM.from_pretrained('/path/to/Qwen3.5-2B', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('/path/to/Qwen3.5-2B')
inputs = tokenizer('The capital of France is', return_tensors='pt')
gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=16, min_new_tokens=16,
    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
with torch.no_grad():
    out = model.generate(inputs.input_ids, generation_config=gen_cfg,
        return_dict_in_generate=True, output_scores=True)
torch.save({'expected_logits': torch.stack(out.scores)[:16,:,:],
    'input_ids': inputs.input_ids, 'prompt': 'The capital of France is'},
    '/path/to/qwen35_2b_cpu_reference_logits_bf16.pt')
"
deactivate
```

#### Compiling with output_logits for Logit Validation

The logit validation test requires a separate compiled model with `output_logits=True`. After compiling the standard model, compile a second copy:

```python
neuron_config = NeuronConfig(
    tp_degree=4, batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
    seq_len=128, torch_dtype=torch.bfloat16, logical_nc_config=2,
    enable_bucketing=False, flash_decoding_enabled=False,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    save_sharded_checkpoint=True, output_logits=True,  # <-- enables logit capture
)
```

## Known Issues

1. **SDK 2.29+ required:** The NKI DeltaNet kernels require NKI 0.3.0 (SDK 2.29).

2. **PyTorch chunked forward hits compiler ICE on 2B dimensions:** The `_chunk_forward` path creates 5D tensors that trigger neuronx-cc codegen crash (NCC_INLA001). The fused NKI kernel is the default and required CTE path. Controlled via `USE_NKI_FUSED` env var (defaults to enabled).

3. **No mini model test:** DeltaNet layers require NKI kernels that only execute on Neuron devices. All integration tests require a trn2 instance with full model weights.

4. **Chat template required for quality output:** Raw text prompts produce echoey/repetitive output. Always use `tokenizer.apply_chat_template()`.

## Maintainer

Jim Burtoft ([@jimburtoft](https://github.com/jimburtoft))
