# Contrib Model: PerceptronAI Isaac-0.2-2B-Preview VLM

NeuronX Distributed Inference implementation for the PerceptronAI Isaac-0.2-2B-Preview Vision-Language Model. Isaac combines a Qwen3 text backbone with a SigLIP2 vision encoder and 2-layer MLP projector with pixel shuffle.

## Model Information

- **HuggingFace ID:** [`PerceptronAI/Isaac-0.2-2B-Preview`](https://huggingface.co/PerceptronAI/Isaac-0.2-2B-Preview)
- **Model Type:** VLM with SigLIP2 vision encoder, pixel shuffle, MLP projector, and Qwen3 text decoder
- **License:** CC-BY-NC-4.0 (non-commercial)
- **Requires:** `trust_remote_code=True`

## Architecture Details

### Text Backbone (Qwen3)

| Spec | Isaac 2B |
|---|---:|
| **Layers** | 28 |
| **Hidden Size** | 2048 |
| **Head Dim** | 128 |
| **Attention Heads** | 16 |
| **KV Heads** | 8 |
| **Intermediate Size** | 6144 |
| **Vocabulary Size** | 151,936 |
| **Max Position Embeddings** | 40,960 |
| **Position Encoding** | RoPE (mRoPE-capable) |
| **Normalization** | RMSNorm |
| **Activation** | SiLU |
| **Total Parameters** | 2.57B |

### SigLIP2 Vision Encoder

| Spec | Value |
|---|---:|
| **Layers** | 27 |
| **Hidden Size** | 1152 |
| **Head Dim** | 72 |
| **Attention Heads** | 16 |
| **KV Heads** | 16 |
| **Intermediate Size** | 4304 |
| **Activation** | GELU (approximate) |
| **Image Size** | 256×256 |
| **Patch Size** | 16 |
| **Pixel Shuffle Scale** | 2 |
| **Vision Tokens per Image** | 64 |

### MLP Projector

| Spec | Value |
|---|---:|
| **Layer 1** | Linear(4608 → 18432, no bias) + SiLU |
| **Layer 2** | Linear(18432 → 2048, no bias) |
| **Parameters** | ~122M |

## Validation Results

**Validated:** 2026-04-30
**Configuration:** trn2.3xlarge, TP=1, batch_size=1, seq_len=1024, bfloat16

### Accuracy

| Test | Status | Result |
|------|--------|--------|
| Text logit cosine (5 prompts) | PASS | avg 0.99998 vs CPU ref |
| Top-1 token match | PASS | 100% match (8/8 prompts) |
| Image+text generation | PASS | Coherent descriptions |
| TP=2 accuracy | PASS | cosine 0.99997 |
| TP=4 accuracy | PASS | cosine 0.99997 |

### Performance (trn2.3xlarge, TP=1, BS=1)

| Metric | seq_len=1024 | seq_len=4096 |
|--------|-------------|-------------|
| **TKG Throughput** | 110.7 tok/s | 94.0 tok/s |
| **TPOT** | 9.0 ms | 10.6 ms |
| **TTFT** | 9.0 ms | 10.6 ms |
| **Image+text tok/s** | 108.7 tok/s | 93.1 tok/s |
| **Projected DP=4** | ~443 tok/s | ~376 tok/s |

**Compilation time:** ~196s (one-time, seq_len=1024)

## Usage

```python
import torch
from transformers import AutoConfig, AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from isaac_neuron.modeling_isaac import (
    NeuronIsaacForConditionalGeneration,
    IsaacInferenceConfig,
)

model_path = "/path/to/Isaac-0.2-2B-Preview"
compiled_path = "/path/to/compiled/model"

# Configure
text_config = NeuronConfig(
    batch_size=1,
    seq_len=1024,
    torch_dtype=torch.bfloat16,
    tp_degree=1,
    is_continuous_batching=True,
    ctx_batch_size=1,
    enable_bucketing=True,
    context_encoding_buckets=[1024],
    token_generation_buckets=[1024],
    on_device_sampling_config=OnDeviceSamplingConfig(
        dynamic=True, do_sample=True, deterministic=True,
        top_k=1, global_topk=256, top_k_kernel_enabled=True,
    ),
    attn_kernel_enabled=True,  # CTE flash attention
    fused_qkv=False,
    mlp_kernel_enabled=False,
)

vision_config = NeuronConfig(
    batch_size=1, seq_len=1024, torch_dtype=torch.bfloat16,
    tp_degree=1, is_continuous_batching=True, ctx_batch_size=1,
    enable_bucketing=True, buckets=[1],
)

hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config = IsaacInferenceConfig(
    text_neuron_config=text_config,
    vision_neuron_config=vision_config,
    load_config=load_pretrained_config(hf_config=hf_config),
)
config.image_token_index = 151655  # <|image_pad|>

# Compile and load
model = NeuronIsaacForConditionalGeneration(model_path, config)
model.compile(compiled_path, debug=False)
model.load(compiled_path)

# Generate (see integration tests for full examples)
```

## Compatibility Matrix

| Instance/Version | SDK 2.29 | SDK 2.28 and earlier |
|------------------|----------|----------------------|
| trn2.3xlarge (TP=1) | Tested | Not tested |
| trn2.3xlarge (TP=2) | Tested | Not tested |
| trn2.3xlarge (TP=4) | Tested | Not tested |
| trn1 | Not tested | Not tested |
| inf2 | Not tested | Not tested |

## Known Limitations

- **Batch size:** Only BS=1 supported (NxDI VLM framework limitation, shared with all VLM contribs)
- **MLP NKI kernel:** Not compatible at TP=1 (intermediate=6144 exceeds SBUF capacity). Use default kernels.
- **QKV NKI kernel:** Not compatible (Q/K layernorm incompatible with fused QKV kernel)
- **Image size:** Fixed at 256×256 (64 vision tokens per image)
- **License:** CC-BY-NC-4.0 — non-commercial use only

## Testing

Run integration tests:

```bash
# Set up environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export PYTHONPATH=/path/to/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH

# Run validation
cd contrib/models/Isaac-0.2-2B
python test/integration/run_isaac.py
```

## Module Structure

```
contrib/models/Isaac-0.2-2B/
├── README.md
├── src/
│   └── isaac_neuron/
│       ├── __init__.py
│       ├── modeling_isaac.py          # VLM orchestrator + config + state dict mapping
│       ├── modeling_isaac_text.py     # Text model (NeuronBaseModel + Qwen3 layers)
│       ├── modeling_isaac_vision.py   # Vision wrapper + MLP projector + pixel shuffle
│       ├── ndxi_patch.py             # SDK 2.29 compatibility patches
│       ├── utils.py                  # QKV fusion + pixel shuffle utilities
│       └── siglip/
│           ├── modeling_siglip.py    # SigLIP2 vision encoder
│           └── layers.py            # OutputChannelParallelConv2d
└── test/
    └── integration/
        ├── run_isaac.py              # Main compilation + generation test
        ├── benchmark.py              # Formal benchmark script
        ├── test_tp.py                # TP=2/4 validation
        ├── validate_text_logits.py   # Text logit validation vs CPU
        ├── validate_tkg.py           # TKG multi-token validation
        ├── validate_image_text.py    # Image+text E2E validation
        └── validate_vision_encoder.py # Vision encoder sanity checks
```

## Example Checkpoint

* [`PerceptronAI/Isaac-0.2-2B-Preview`](https://huggingface.co/PerceptronAI/Isaac-0.2-2B-Preview)
