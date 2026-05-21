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

**Validated:** 2026-05-21
**Instance:** trn2.3xlarge (LNC=2)
**SDK:** Neuron SDK 2.29.1, PyTorch 2.9, NKI 0.3.0

### Recommended Configuration: TP=1, BS=2, DP=4

The optimal configuration for maximum throughput on trn2.3xlarge is **TP=1 with DP=4** (one model per core, 4 independent processes). Each core delivers the same per-sequence throughput as TP=4 while enabling 4x aggregate throughput.

| Config | TTFT | Per-Core Throughput | Aggregate | Concurrent Sequences |
|--------|:----:|:-------------------:|:---------:|:--------------------:|
| TP=1 BS=2 DP=4 | 362 ms | 104.5 tok/s | **418.7 tok/s** | 8 |
| TP=4 BS=1 | 307 ms | 104.2 tok/s | 104.2 tok/s | 1 |
| TP=2 BS=1 DP=2 | 294 ms | 91.2 tok/s | ~182 tok/s | 2 |

### Benchmark Results (TP=1, BS=2, DP=4)

All benchmarks on trn2.3xlarge, LNC=2, BF16, seq_len=128. Each core runs independently with `NEURON_RT_VISIBLE_CORES=<n>`.

| Core | TTFT (ms) | Aggregate Throughput (tok/s) |
|:----:|:---------:|:----------------------------:|
| 0 | 362.1 | 104.6 |
| 1 | 363.2 | 104.9 |
| 2 | 361.5 | 105.5 |
| 3 | 361.1 | 103.7 |
| **Total** | — | **418.7** |

Zero contention across cores — each core delivers identical throughput when running concurrently.

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

### Quick Start (TP=1, single core)

```python
import json
import os
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

model_path = "/path/to/Qwen3.5-2B"
compiled_path = "/scratch/qwen35_2b_tp1/"

# Default: TP=1, BS=2 (optimal for throughput with DP=4)
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=2,
    ctx_batch_size=2,
    tkg_batch_size=2,
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

# Compile (one-time, ~34 min at TP=1)
model = NeuronQwen35ForCausalLM(model_path, config)
model.compile(compiled_path)

# Load (pin to a single core)
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
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
# Pad to BS=2 (required minimum batch size at TP=1)
if inputs.input_ids.shape[0] < 2:
    inputs["input_ids"] = inputs.input_ids.repeat(2, 1)
    inputs["attention_mask"] = inputs.attention_mask.repeat(2, 1)

gen_model = HuggingFaceGenerationAdapter(model)
outputs = gen_model.generate(
    inputs.input_ids,
    generation_config=gen_config,
    attention_mask=inputs.attention_mask,
    max_new_tokens=80,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### DP=4 Deployment (maximum throughput)

For maximum throughput (418 tok/s), run 4 independent processes:

```bash
# Compile once (shared artifacts):
python compile.py  # produces /scratch/qwen35_2b_tp1/

# Run 4 processes (one per core):
for core in 0 1 2 3; do
    NEURON_RT_VISIBLE_CORES=$core python serve.py --port $((8000+core)) &
done
```

Each process serves 2 concurrent sequences (BS=2) independently. Total: 8 concurrent sequences at 418 tok/s aggregate.

### Alternative: TP=4 (lowest latency)

For lowest per-request latency (307ms TTFT), use TP=4 with BS=1:

```python
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
```

**Note:** Qwen3.5-2B is a chat model. Use `tokenizer.apply_chat_template()` for best results. Raw text prompts may produce echoey output.

**Note on `seq_len`:** The `seq_len` parameter is the total sequence budget (input + generated tokens). Do not pad inputs to `max_length=seq_len`. Use `padding=True` for automatic minimal padding.

## Compatibility Matrix

| Instance | TP | DP | SDK 2.29.1 | Notes |
|----------|-----|-----|------------|-------|
| trn2.3xlarge (LNC=2) | 1 | 4 | **VALIDATED** | Recommended: 418 tok/s aggregate |
| trn2.3xlarge (LNC=2) | 2 | 2 | VALIDATED | Balanced: ~182 tok/s |
| trn2.3xlarge (LNC=2) | 4 | 1 | VALIDATED | Lowest latency: 307ms TTFT |

### Tested Configurations (trn2.3xlarge, LNC=2)

| TP | Batch Size | seq_len | Status | Notes |
|:--:|:----------:|:-------:|:------:|-------|
| 1 | 2 | 128 | VALIDATED | Default config |
| 4 | 1 | 128 | VALIDATED | |
| 4 | 2 | 128 | VALIDATED | |
| 4 | 4 | 128 | VALIDATED | |
| 4 | 8 | 128 | VALIDATED | |
| 4 | 1 | 512 | VALIDATED | |
| 4 | 1 | 1024 | VALIDATED | |
| 4 | 1 | 2048 | VALIDATED | |
| 4 | 1 | 4096 | VALIDATED | |
| 2 | 1 | 1024 | VALIDATED | |
| 1 | 1 | any | BLOCKED | Compiler exit code 70 |

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

5. **VL grid 64 (1024 vision tokens, 1024x1024 image) -- FIXED.** The empty-reply behaviour for `grid_size=64` was caused by `vision_mask` padding entries using `n_active_tokens - 1` as the scatter index, which equals the last real input token position. The downstream `index_put_` then scattered zero vision embeddings to that position, zeroing the last token's text embedding. The fix uses a large sentinel value (`2**30`) for all padding entries in `vision_mask`, which `pad_inputs()` clamps to `padded_seq_len - 1` (always a throw-away pad slot). No recompile needed for text-only; VL pipeline requires recompile with updated `input_generator()`.

## High-Resolution Images: Chunked Vision Encoding (2K and 4K)

The vision encoder supports images up to **4096x4096** (65,536 patches) via automatic chunked processing. When the input sequence exceeds the largest compiled bucket (8192), the forward pass splits it into N chunks of max_bucket size, processes each chunk independently through the compiled encoder, then concatenates the merged outputs.

### How It Works

| Image Resolution | Vision Patches | Processing Mode | Chunks | Text Tokens |
|-----------------|---------------|-----------------|--------|-------------|
| 448x448 | 784 | Single bucket (1024) | 1 | 196 |
| 1024x1024 | 4,096 | Single bucket (4096) | 1 | 1,024 |
| 2048x2048 | 16,384 | Chunked | 2 x 8192 | 4,096 |
| 4096x4096 | 65,536 | Chunked | 8 x 8192 | 16,384 |

### Quality Tradeoffs

Chunking means each chunk only has self-attention within its own patches (no cross-chunk vision attention). This is acceptable because:
- The spatial merger (2x2 → 1 token) operates on local neighborhoods, unaffected by chunking
- Rotary position embeddings are per-token, so positional information is preserved
- The text decoder sees ALL merged vision tokens with full cross-attention
- Tested: model correctly identifies quadrant colors, spatial layout, and uniform colors at 4K

### Compilation Requirements

```bash
# Vision encoder buckets (compile once, reuse for all resolutions):
# seq=256 (~2 min), seq=1024 (~2 min), seq=4096 (~10 min), seq=8192 (~30 min)
python compile_vision_all_buckets.py

# Text decoder (choose seq_len based on max image size):
# For 2K images: seq_len=8192 (~10 min compile)
# For 4K images: seq_len=32768 (~54 min compile)
```

### Performance (trn2.3xlarge, TP=4)

| Resolution | TTFT | TKG Throughput |
|-----------|------|----------------|
| 448x448 | ~1s | 10 tok/s |
| 1024x1024 | ~1.5s | 16 tok/s |
| 2048x2048 | ~5s | 10 tok/s |
| 4096x4096 | ~18s | 2.5 tok/s |

### Alternative: Layer-by-Layer Compilation

For seq_len=16384 (2048x2048 images without chunking), layer-by-layer compilation compiles each of the 24 transformer blocks independently. This preserves full self-attention across all patches but requires 24 separate NEFFs and sequential execution with CPU round-trips between blocks. See `load_layerwise()` in the vision model wrapper.
## Maintainer

Jim Burtoft ([@jimburtoft](https://github.com/jimburtoft))
