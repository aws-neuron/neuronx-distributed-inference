# Contrib Model: Gemma 4 31B IT

NeuronX Distributed Inference implementation of Google's Gemma 4 31B Instruct (text-only decoder).

## Model Information

- **HuggingFace ID:** [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it)
- **Model Type:** Decoder-only transformer (text-only; VLM vision encoder not yet ported)
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

**Note:** Flash attention kernel (NKI) is not supported because head_dim > 128. This implementation uses decomposed attention (`attn_kernel_enabled=False`).

## Validation Results

**Validated:** 2026-04-03
**Configuration:** TP=4, batch_size=1, seq_len=256 (128 context + 128 generation), bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles and loads successfully |
| Token Matching | PASS | Greedy token match vs HF CPU reference |
| Chat Generation | PASS | "What is 2 + 2?" -> "2 + 2 = 4" |
| Coherence | PASS | Coherent haiku generation |
| Logit Correlation | PASS | Pearson r = 0.980 vs HF CPU reference |

**Prompts tested:**
- "The capital of France is" -> "Paris" (greedy match)
- "What is 2 + 2?" -> "2 + 2 = 4" (chat template)
- "Write a haiku about the ocean" -> coherent haiku

### Performance Metrics

**Configuration:** TP=4, batch_size=1, bfloat16
**Instance:** trn2.3xlarge (LNC=2, 4 logical cores)

| Metric | Value |
|--------|-------|
| TTFT | ~66 ms |
| TPOT | ~30.6 ms |
| Throughput | ~32.2 tok/s |
| Compile Time | ~120 s |
| Load Time | ~42 s |

**Status:** VALIDATED

## Prerequisites

- **Neuron SDK 2.28+** (DLAMI: `Deep Learning AMI Neuron (Ubuntu 24.04) 20260227`)
- **transformers >= 5.5.0** (required for `Gemma4ForConditionalGeneration`)
- **transformers/utils/fx.py shim**: NxD imports `transformers.utils.fx.HFTracer` which was removed in transformers 5.x. Create a shim file:

```python
# Create at: <venv>/lib/python3.12/site-packages/transformers/utils/fx.py
class HFTracer:
    pass
```

## Usage

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
    seq_len=256,  # context_len + gen_len
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

# Generate (see test/integration/test_model.py for full generation loop)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... pad inputs to seq_len, run prefill + token generation loop
```

See `test/integration/test_model.py` for a complete generation example with chat template support.

## Compatibility Matrix

| Instance Type | SDK 2.28+ | SDK 2.27 and earlier |
|---------------|-----------|----------------------|
| trn2.3xlarge (TP=4, LNC=2) | VALIDATED | Not tested |
| trn2.48xlarge | Not tested | Not tested |
| Inf2 | N/A | N/A |

**Notes:**
- Requires TP=4 on trn2.3xlarge with LNC=2 (default). Global layers have 4 KV heads, requiring TP <= 4.
- `fused_qkv=False` required (heterogeneous Q/K/V shapes per layer type).
- `attn_kernel_enabled=False` required (head_dim > 128 exceeds NKI flash attention limit).

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

- **Text-only**: Vision encoder (Gemma4Vision) is not yet ported. VLM image+text inference is not supported.
- **No flash attention**: head_dim > 128 requires decomposed attention path.
- **transformers shim required**: NxD depends on `transformers.utils.fx.HFTracer` which is absent in transformers 5.x.

## Maintainer

Community contribution

**Last Updated:** 2026-04-03
