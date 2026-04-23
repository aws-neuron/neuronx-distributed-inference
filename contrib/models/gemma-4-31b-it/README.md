# Contrib Model: Gemma 4 31B IT

NeuronX Distributed Inference implementation of Google's Gemma 4 31B Instruct, supporting both text-only and vision-language (VLM) inference.

## Model Information

- **HuggingFace ID:** [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it)
- **Model Type:** Multimodal (text + vision) transformer
- **Parameters:** 31B
- **License:** Check HuggingFace model card

## Architecture Details

Gemma 4 31B has several unique features compared to Gemma 3 and other standard decoder models:

| Feature | Description |
|---------|-------------|
| **Heterogeneous layers** | SWA layers (head_dim=256, 16 KV heads) and Global layers (head_dim=512, 4 KV heads) |
| **attention_k_eq_v** | Global layers share K/V projections (V = K before normalization) |
| **QK normalization** | RMSNorm on Q and K after projection |
| **V normalization** | RMSNorm without learnable scale on V states |
| **layer_scalar** | Per-layer learned multiplicative factor |
| **final_logit_softcapping** | `30 * tanh(logits / 30)` after lm_head |
| **Partial RoPE** | Global layers: `partial_rotary_factor=0.25` (128 of 512 dims rotated) |
| **4-norm decoder** | `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm` |
| **Scaled embeddings** | `embed * sqrt(hidden_size)` |

### Vision Encoder (Gemma4Vision)

| Feature | Description |
|---------|-------------|
| **Architecture** | 27 layers, hidden_size=1152, intermediate_size=4304, head_dim=72 |
| **Attention** | 16 MHA heads, bidirectional (no causal mask) |
| **2D RoPE** | theta=100, head_dim split into x/y halves (36 dims each) |
| **Pooler** | Spatial average pooling (3x3 kernel), sqrt(1152) scaling -> 64 tokens for 384x384 |
| **Projector** | RMSNorm(no scale) + Linear(1152->5376) |
| **Output** | 64 vision tokens per 384x384 image, projected to text hidden_size (5376) |

**Note:** A custom NKI flash attention kernel (`nki_flash_attn_large_d.py`) supports head_dim up to 512 by tiling the QK contraction dimension in chunks of 128. It is automatically enabled via `ndxi_patch.py` when `apply_patch()` is called. The text-only path works with or without the kernel.

## Validation Results

**Validated:** 2026-04-23
**Configuration:** TP=4, batch_size=1, bfloat16, trn2.3xlarge (LNC=2)
**SDK:** 2.29 (NxDI 0.9, neuronx-cc 2.24). Also validated on SDK 2.28.

### Text-Only Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles and loads successfully |
| Token Matching | PASS | Greedy token match vs HF CPU reference |
| Chat Generation | PASS | "What is 2 + 2?" -> "2 + 2 = 4" |
| Coherence | PASS | Coherent haiku generation |
| Logit Correlation | PASS | Pearson r = 0.980 vs HF CPU reference |

### VLM Results

| Test | Status | Result |
|------|--------|--------|
| Vision Encoder CPU Match | PASS | Cosine similarity 0.9995 vs HF CPU |
| Text-only Prefill | PASS | Correct text generation without images |
| Vision Prefill | PASS | Vision encoder runs, embeddings scattered correctly |
| Full Generation Loop | PASS | Vision prefill -> multi-step token generation |
| Coherent Description | PASS | Generates multi-sentence image descriptions |
| Text After Vision | PASS | Text-only works after vision queries (no KV corruption) |

### NKI Flash Attention Kernel Results

**Standalone kernel validation (5/5 tests pass):**

| Test | Cosine Similarity |
|------|-------------------|
| d=256 causal | 0.999989 |
| d=256 causal + SW=256 | 0.999990 |
| d=512 causal + GQA(4:1) | 0.999999 |
| d=256 seq=1024 | 0.999993 |
| d=128 baseline | 0.999991 |

**End-to-end text generation (with NKI kernel):**

| Test | Status | Result |
|------|--------|--------|
| "The capital of France is" | PASS | "Paris" |
| "What is 2 + 2?" | PASS | "4" |
| "The largest planet..." | PASS | "Jupiter" |

**Prompts tested:**
- "The capital of France is" -> "Paris" (greedy match)
- "What is 2 + 2?" -> "2 + 2 = 4" (chat template)
- "Write a haiku about the ocean" -> coherent haiku

### Performance Metrics

**Configuration:** TP=4, batch_size=1, bfloat16
**Instance:** trn2.3xlarge (LNC=2, 4 logical cores)

| Metric | Text-Only | VLM (with image) |
|--------|-----------|-------------------|
| TTFT | ~65 ms | ~180-270 ms |
| TPOT | ~31 ms | ~30 ms |
| Throughput | ~32 tok/s | ~33 tok/s |
| Vision Encoder | N/A | ~18.5 ms |
| Compile Time | ~10 min | ~10 min (text) + ~30 s (vision) |
| Load Time | ~15 s | ~15 s (text) + ~0.3 s (vision) |

### Sequence Length Support

Tested up to seq_len=8192 on trn2.3xlarge (LNC=2, TP=4, batch=1, BF16, greedy).
All correctness tests pass at every tested length.

| seq_len | Compile | Load | KV cache/core | TTFT P50 | TPOT P50 | tok/s | Status |
|---------|---------|------|---------------|----------|----------|-------|--------|
| 512 | ~10 min | 15s | 0.11 GB | 65 ms | 30.9 ms | 32.4 | PASS |
| 4096 | ~15 min | 17s | 0.86 GB | 999 ms | 31.3 ms | 32.0 | PASS |
| 8192 | ~28 min | 72s | 1.72 GB | 17.0 s | 33.5 ms | 29.8 | PASS |
| 16384 | OOM | - | 3.44 GB (est.) | - | - | - | CTE compiler OOM |

- TPOT is nearly constant across sequence lengths (TKG is independent of context length).
- 16384 fails during CTE compilation (host RAM OOM on 128 GB system RAM, not HBM).
- Larger instances (trn2.48xlarge) may support 16384+.

**Status:** VALIDATED

## Prerequisites

- **Neuron SDK 2.29** (DLAMI: `Deep Learning AMI Neuron (Ubuntu 24.04) 20260410`)
  - Also works on SDK 2.28 (DLAMI 20260227) for trn1 compatibility
- **transformers >= 4.57.0**
- **transformers/utils/fx.py shim** (only needed if using transformers >= 5.0):

```python
# Create at: <venv>/lib/python3.12/site-packages/transformers/utils/fx.py
class HFTracer:
    pass
```

## Usage: Text-Only

```python
import json
import os
import torch
from transformers import AutoTokenizer

from src.modeling_gemma4 import (
    NeuronGemma4ForCausalLM,
    Gemma4InferenceConfig,
    Gemma4NeuronConfig,
)

model_path = "/path/to/gemma-4-31b-it"
compiled_model_path = "/path/to/compiled"

# Configure
neuron_config = Gemma4NeuronConfig(
    tp_degree=4,
    batch_size=1,
    max_batch_size=1,
    seq_len=4096,  # max context length (tested up to 8192)
    on_device_sampling_config=None,
    torch_dtype=torch.bfloat16,
    fused_qkv=False,
    attn_kernel_enabled=False,
)

def load_config_fn(config_obj):
    with open(os.path.join(model_path, "config.json")) as f:
        config_dict = json.load(f)
    for k, v in config_dict.items():
        setattr(config_obj, k, v)

config = Gemma4InferenceConfig(
    neuron_config=neuron_config,
    load_config=load_config_fn,
)

# Compile (first time only)
model = NeuronGemma4ForCausalLM(model_path, config)
model.compile(compiled_model_path)

# Load onto Neuron
model = NeuronGemma4ForCausalLM(model_path, config)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
prompt = "The capital of France is"
ids = tokenizer.encode(prompt, add_special_tokens=True)
prompt_len = len(ids)
seq_len = 4096  # must match neuron_config.seq_len

input_ids = torch.zeros(1, seq_len, dtype=torch.int32)
input_ids[0, :prompt_len] = torch.tensor(ids, dtype=torch.int32)
attention_mask = torch.zeros(1, seq_len, dtype=torch.int32)
attention_mask[0, :prompt_len] = 1

# IMPORTANT: padding positions must have position_id=0, not sequential values.
# The base class extracts hidden states at max(position_ids).
position_ids = torch.zeros(1, seq_len, dtype=torch.long)
position_ids[0, :prompt_len] = torch.arange(prompt_len, dtype=torch.long)

# Prefill (CTE)
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

# Token generation loop
token_id = outputs.tokens[0].item()
generated = [token_id]
cur_pos = prompt_len
for _ in range(99):
    tkg_in = torch.tensor([[token_id]], dtype=torch.long)
    am_len = cur_pos + 1
    tkg_mask = torch.cat([
        torch.ones(1, am_len, dtype=torch.long),
        torch.zeros(1, seq_len - am_len, dtype=torch.long),
    ], dim=1)
    out = model(
        input_ids=tkg_in,
        attention_mask=tkg_mask,
        position_ids=torch.tensor([[cur_pos]], dtype=torch.long),
    )
    cur_pos += 1
    token_id = out.tokens[0].item()
    generated.append(token_id)
    if token_id == tokenizer.eos_token_id:
        break
print(tokenizer.decode(generated, skip_special_tokens=True))
```

See `test/integration/test_model.py` for a complete generation example with chat template support.

## Usage: VLM (Vision + Language)

```python
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

# Apply NxDI monkey-patches BEFORE importing model classes
from src.ndxi_patch import apply_patch
apply_patch()

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from src.modeling_gemma4_vlm import (
    NeuronGemma4ForConditionalGeneration,
    Gemma4VLMInferenceConfig,
    load_pretrained_config,
)

model_path = "/path/to/gemma-4-31b-it"
compiled_path = "/path/to/compiled-vlm"

# Configure text and vision neuron configs
text_neuron_config = NeuronConfig(
    batch_size=1, seq_len=512, torch_dtype=torch.bfloat16,
    rpl_reduce_dtype=torch.float32, logical_nc_config=2,
    tp_degree=4, world_size=4, skip_sharding=False,
    save_sharded_checkpoint=True, enable_bucketing=True,
    context_encoding_buckets=[512], token_generation_buckets=[512],
    fused_qkv=False, attn_kernel_enabled=False,
    on_device_sampling_config=OnDeviceSamplingConfig(
        dynamic=False, do_sample=False, deterministic=True,
        top_k=1, global_topk=256,
    ),
    output_logits=True,
)
vision_neuron_config = NeuronConfig(
    batch_size=1, seq_len=512, torch_dtype=torch.bfloat16,
    rpl_reduce_dtype=torch.float32, logical_nc_config=2,
    tp_degree=4, world_size=4, skip_sharding=False,
    save_sharded_checkpoint=True, enable_bucketing=True,
    buckets=[1],
)
config = Gemma4VLMInferenceConfig(
    text_neuron_config=text_neuron_config,
    vision_neuron_config=vision_neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile (first time only)
model = NeuronGemma4ForConditionalGeneration(model_path=model_path, config=config)
model.compile(compiled_path)

# Load
model = NeuronGemma4ForConditionalGeneration(model_path=model_path, config=config)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right')

# --- Prepare image patches ---
IMAGE_TOKEN_ID, BOI_TOKEN_ID, EOI_TOKEN_ID = 258880, 255999, 258882
MM_TOKENS = 64  # for 384x384 image
MAX_SEQ_LEN = 512

image = Image.open("photo.jpg").convert("RGB").resize((384, 384))
arr = np.array(image).astype(np.float32) / 255.0
arr = 2.0 * (arr - 0.5)  # normalize to [-1, 1]

patch_size, nps = 16, 24
patches = arr.reshape(nps, patch_size, nps, patch_size, 3)
patches = patches.transpose(0, 2, 1, 3, 4).reshape(nps * nps, patch_size**2 * 3)
pixel_values = torch.from_numpy(patches).unsqueeze(0).to(torch.bfloat16)

position_ids = torch.zeros(1, nps * nps, 2, dtype=torch.long)
for i in range(nps):
    for j in range(nps):
        position_ids[:, i * nps + j, 0] = j  # x
        position_ids[:, i * nps + j, 1] = i  # y

# --- Build prompt (HF Gemma4 chat format) ---
question = "What is in this image?"
bos = tokenizer.bos_token_id
ids = (
    [bos]
    + tokenizer.encode("<|turn>", add_special_tokens=False)
    + tokenizer.encode("user", add_special_tokens=False)
    + tokenizer.encode("\n\n\n", add_special_tokens=False)
    + [BOI_TOKEN_ID] + [IMAGE_TOKEN_ID] * MM_TOKENS + [EOI_TOKEN_ID]
    + tokenizer.encode("\n\n", add_special_tokens=False)
    + tokenizer.encode(question, add_special_tokens=False)
    + tokenizer.encode("<turn|>", add_special_tokens=False)
    + tokenizer.encode("\n", add_special_tokens=False)
    + tokenizer.encode("<|turn>", add_special_tokens=False)
    + tokenizer.encode("model", add_special_tokens=False)
    + tokenizer.encode("\n", add_special_tokens=False)
    + tokenizer.encode("<|channel>", add_special_tokens=False)
    + tokenizer.encode("thought", add_special_tokens=False)
    + tokenizer.encode("\n", add_special_tokens=False)
    + tokenizer.encode("<channel|>", add_special_tokens=False)
)
seq_len = len(ids)
input_ids = torch.tensor([ids], dtype=torch.int32)
pad_len = MAX_SEQ_LEN - seq_len
input_ids = torch.cat([input_ids, torch.zeros(1, pad_len, dtype=torch.int32)], dim=1)
attention_mask = torch.cat([
    torch.ones(1, seq_len, dtype=torch.int32),
    torch.zeros(1, pad_len, dtype=torch.int32),
], dim=1)
vision_mask = (input_ids == IMAGE_TOKEN_ID).to(torch.bool)

# --- Run inference ---
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        pixel_position_ids=position_ids,
        vision_mask=vision_mask,
    )
    # Token generation loop
    token_id = outputs.tokens[0].item()
    generated = [token_id]
    for _ in range(49):
        tkg_in = torch.tensor([[token_id]], dtype=torch.long)
        tkg_len = seq_len + len(generated)
        tkg_mask = torch.cat([
            torch.ones(1, tkg_len, dtype=torch.long),
            torch.zeros(1, MAX_SEQ_LEN - tkg_len, dtype=torch.long),
        ], dim=1)
        out = model(input_ids=tkg_in, attention_mask=tkg_mask)
        token_id = out.tokens[0].item()
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
    print(tokenizer.decode(generated, skip_special_tokens=True))
```

## Compatibility Matrix

| Instance Type | SDK 2.29 | SDK 2.28 |
|---------------|----------|----------|
| trn2.3xlarge (TP=4, LNC=2) | **VALIDATED** (text + VLM, seq_len up to 8192) | VALIDATED (text + VLM, seq_len up to 2048) |
| trn2.48xlarge | Not tested (may support seq_len=16384+) | Not tested |
| trn1 | Not tested (SDK 2.28 likely needed) | Not tested |

**Notes:**
- Requires TP=4 on trn2.3xlarge with LNC=2 (default). Global layers have 4 KV heads, requiring TP <= 4.
- `fused_qkv=False` required (heterogeneous Q/K/V shapes per layer type).
- `attn_kernel_enabled=False` in NeuronConfig (the standard NxDI kernel doesn't support head_dim > 128). The custom NKI kernel in `nki_flash_attn_large_d.py` is applied separately via `ndxi_patch.py`.
- SDK 2.29 NxDI 0.9 moved `create_sampler` to `modules.generation.sampling`; the code handles both paths automatically.

## Testing

Run integration tests:

```bash
# Set environment variables (optional, defaults shown)
export GEMMA4_MODEL_PATH=/mnt/models/gemma-4-31b-it
export GEMMA4_COMPILED_PATH=/mnt/models/gemma4-compiled
export GEMMA4_TP_DEGREE=4

# Run with pytest
pytest nxdi_contrib_models/models/gemma-4-31b-it/test/integration/test_model.py --capture=tee-sys

# Or run standalone
cd nxdi_contrib_models/models/gemma-4-31b-it
python test/integration/test_model.py
```

## Example Checkpoints

- [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) (59 GB, 2 safetensor shards)

## Known Limitations

- **No bidirectional vision attention**: HF Gemma4 uses bidirectional attention for vision tokens in SWA layers (`or_mask`). This implementation uses standard causal masking for all tokens. Vision quality may be slightly degraded but generation is coherent.
- **Fixed image resolution**: Currently supports 384x384 images (64 vision tokens). Dynamic resolution requires different bucket configurations.
- **Prompt format required**: VLM inference requires the specific HF Gemma4 chat template format (see Usage: VLM above). Incorrect prompt formatting produces garbage output.
- **VLM monkey-patches**: The VLM requires `ndxi_patch.py` to be applied before model creation due to API differences across NxDI versions.
- **NKI kernel**: The custom NKI flash attention kernel works on SDK 2.28 (NKI 0.2.x) but produces incorrect output on SDK 2.29 (NKI 0.3.0) due to ISA behavioral changes. Text-only inference works correctly without the kernel on both SDK versions (uses decomposed attention).
- **seq_len=16384**: CTE compilation runs out of host memory on trn2.3xlarge (128 GB RAM). Max tested: 8192.
- **Position IDs**: Padding positions must have `position_id=0`. Using sequential IDs for padding tokens causes garbage CTE output (see Usage example above).

## Maintainer

Community contribution

**Last Updated:** 2026-04-23
