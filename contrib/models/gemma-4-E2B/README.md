# Contrib Model: Gemma 4 E2B

NeuronX Distributed Inference implementation of Google's Gemma 4 E2B (Everything to Bytes) model.

## Model Information

- **HuggingFace ID:** `google/gemma-4-E2B`
- **Model Type:** Decoder-only transformer with vision encoder (VLM)
- **Parameters:** 5.1B total, 2.3B effective (PLE embedding table accounts for the difference)
- **Native Dtype:** BF16
- **License:** Check HuggingFace model card

## Architecture Details

### Text Decoder
- 35 layers, hidden_size=1536, 8 attention heads, 1 KV head
- Heterogeneous layers: SWA (head_dim=256, sliding_window=512) and full attention (head_dim=512) every 5th layer
- Per-Layer Embeddings (PLE): shared embedding table [262144, 8960] with per-layer gated projections
- KV cache sharing: layers 15-34 reuse K/V from donor layers (20 shared layers)
- Double-wide MLP: layers 15-34 have intermediate_size=12288 (2x the 6144 of layers 0-14)
- QK normalization via RMSNorm (standard weight scaling)
- Final logit softcapping: `30 * tanh(logits / 30)`
- No v_norm, no attention_k_eq_v (unlike the 31B variant)

### Vision Encoder
- 16 layers, hidden_size=768, 12 attention heads, head_dim=64
- patch_size=16, pooling_kernel_size=3
- Multidimensional RoPE (theta=100)
- Produces 280 soft tokens per image
- Linear projector: [1536, 768]

## Validation Results

**Validated:** 2026-04-07
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16, KV cache sharing enabled

### Text Decoder Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| BF16 + KV Sharing Accuracy | PASS | Cosine similarity 0.999999 vs CPU reference |
| Chat Generation | PASS | Coherent responses to factual and creative prompts |

### Performance (TP=1, batch=1, trn2.3xlarge LNC=2)

| Metric | Value |
|--------|-------|
| TTFT (CTE, bucket=128) | 27.3 ms |
| TPOT (TKG) | 10.4 ms |
| Throughput | 96 tok/s |

## Known Limitations

### VLM Compilation Blocked by NCC_ITEN404

The vision-language model (VLM) pipeline is architecturally complete but cannot currently be compiled due to an internal compiler error (`NCC_ITEN404`) in `neuronx-cc` 2.23. The error occurs during the `TensorInitialization` pass when compiling the context encoding (CTE) NEFF with vision inputs.

- **Text-only inference works correctly** -- compile and run without vision inputs
- **VLM code is included** (`modeling_gemma4_e2b_vlm.py`, `modeling_gemma4_vision.py`) and ready for testing once the compiler issue is resolved
- The same vision encoder architecture compiles and runs successfully on the Gemma4-31B variant

## Usage

### Text-Only Inference

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

import sys
sys.path.insert(0, "src")
from ndxi_patch import apply_patch
apply_patch()

from modeling_gemma4_e2b import NeuronGemma4E2BForCausalLM, Gemma4E2BInferenceConfig

model_path = "/path/to/gemma-4-E2B/"
compiled_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    max_batch_size=1,
    max_length=1024,
    seq_len=128,
    torch_dtype=torch.bfloat16,
    attn_kernel_enabled=False,
)

config = Gemma4E2BInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

model = NeuronGemma4E2BForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# ... (see integration test for full example)
```

### VLM Inference (Pending Compiler Fix)

```python
from modeling_gemma4_e2b_vlm import (
    NeuronGemma4E2BForConditionalGeneration,
    Gemma4E2BVLMInferenceConfig,
    load_pretrained_config,
)

# Text neuron config
text_neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    max_batch_size=1,
    max_length=512,
    seq_len=384,  # Must fit 280 image tokens + prompt
    torch_dtype=torch.bfloat16,
    attn_kernel_enabled=False,
)

# Vision neuron config
vision_neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    max_batch_size=1,
    seq_len=1,
    max_length=1,
    torch_dtype=torch.bfloat16,
    attn_kernel_enabled=False,
)

config = Gemma4E2BVLMInferenceConfig(
    text_neuron_config=text_neuron_config,
    vision_neuron_config=vision_neuron_config,
    load_config=load_pretrained_config(model_path),
)

model = NeuronGemma4E2BForConditionalGeneration(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)
```

## Compatibility Matrix

| Instance/Version | SDK 2.28+ | SDK 2.27 |
|------------------|-----------|----------|
| Trn2 (text-only) | PASS | Not tested |
| Trn2 (VLM) | Blocked (NCC_ITEN404) | Blocked (NCC_EVRF023) |

## File Structure

```
gemma-4-E2B/
├── README.md
├── src/
│   ├── __init__.py                     # Text + VLM exports
│   ├── modeling_gemma4_e2b.py          # Text decoder (~1750 lines)
│   ├── modeling_gemma4_e2b_vlm.py      # VLM wrapper (~857 lines)
│   ├── modeling_gemma4_vision.py       # Vision encoder (~770 lines)
│   └── ndxi_patch.py                  # NxDI compatibility patches
└── test/
    └── integration/
        └── test_model.py              # Text-only integration tests
```

## Testing

Run integration tests:

```bash
pytest contrib/models/gemma-4-E2B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/gemma-4-E2B
python3 test/integration/test_model.py
```

## Example Checkpoints

* google/gemma-4-E2B

## Maintainer

Community contribution

**Last Updated:** 2026-04-08
