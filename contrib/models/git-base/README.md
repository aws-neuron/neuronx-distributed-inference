# Contrib Model: Git-base

NeuronX Distributed Inference implementation of Microsoft's Git (Generative Image-to-text Transformer).

## Model Information

- **HuggingFace ID:** `microsoft/git-base`
- **Model Type:** Vision-language model
- **Parameters:** ~125M (text decoder) + ~86M (CLIP ViT-B/16 vision encoder)
- **License:** MIT

## Architecture Details

Git uses a BERT-style text decoder with several distinguishing features:

- **Post-LayerNorm residual blocks** (BERT-style, not pre-LN like GPT/LLaMA): LayerNorm is applied after the residual addition, not before
- **Learned absolute position embeddings** (no rotary embeddings)
- **Embedding LayerNorm** applied after combining token + position embeddings
- **Separate Q/K/V projections with bias** in all attention and MLP layers
- **GELU activation** in MLP
- **No final layer norm** (post-LN per block handles normalization)
- **CLIP ViT-B/16 vision encoder** compiled on Neuron, projects to text hidden size via visual_projection

Key dimensions (git-base):
- hidden_size: 768
- num_attention_heads: 12
- num_hidden_layers: 6
- intermediate_size: 3072
- vocab_size: 30522
- vision: CLIP ViT-B/16@224, 197 vision tokens (196 patches + 1 CLS)

## Available Implementations

### 1. Text-only (`modeling_git.py`)
Only the text decoder is compiled on Neuron. Vision encoder runs on CPU or is skipped entirely.

### 2. Full vision+text (`modeling_git_vision.py`)
Both CLIP vision encoder and text decoder are compiled as separate NEFFs on Neuron. Uses `NeuronBaseForImageToText` infrastructure.

## Validation Results

### Text-only
**Validated:** 2026-03-16
**Configuration:** TP=1, batch_size=1, seq_len=128, bf16

| Test | Status | Result |
|------|--------|--------|
| Compilation | PASS | Compiler status PASS |
| Token Matching | PASS | **100% match** (3/3 tokens) |
| TTFT | PASS | 1.28ms avg |
| Throughput | PASS | 728.05 tokens/sec |

### Vision+Text (full Neuron)
**Validated:** 2026-03-19
**Configuration:** TP=1, batch_size=1, text_seq=256, vision_seq=197, fp32

| Test | Status | Result |
|------|--------|--------|
| Compilation | PASS | Text + Vision NEFFs compiled in 95s |
| Token Matching (TF) | PASS | **95.13% avg** (5 random images, 20 tokens each) |
| Token Matching (Greedy) | INFO | 46.41% avg (2/5 images 100%, cascading divergence on others) |

## Usage (Vision+Text)

```python
import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from src.modeling_git_vision import (
    GitVisionInferenceConfig,
    NeuronGitForCausalLMVision,
)

model_path = "/path/to/git-base/"
compiled_path = "/path/to/compiled_vision/"

text_nc = NeuronConfig(
    tp_degree=1, batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
    max_context_length=256, seq_len=256, n_active_tokens=1,
    torch_dtype=torch.float32, padding_side="right",
    enable_bucketing=True, save_sharded_checkpoint=True,
)
vision_nc = NeuronConfig(
    tp_degree=1, batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
    seq_len=197, n_active_tokens=197,
    torch_dtype=torch.float32, padding_side="right",
    enable_bucketing=False, buckets=[197], save_sharded_checkpoint=True,
)

config = GitVisionInferenceConfig.from_pretrained(
    model_path, text_neuron_config=text_nc, vision_neuron_config=vision_nc,
)
model = NeuronGitForCausalLMVision(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

# Image captioning with forward_atomic_prefill + TKG loop
# See contrib/run_token_match_git_vision.py for full example
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Performance

Profiled on trn1.32xlarge (single NeuronCore utilization):

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| Throughput | - | 714.4 tok/s |
| MBU (Memory) | 11.7% | 12.4% |
| MFU (Compute) | 4.3% | 0.1% |

*Text-only mode: Batch size 1, sequence length 128, BF16 precision, TP=1*

## Testing

Run integration tests:

```bash
pytest contrib/models/git-base/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/git-base
python3 test/integration/test_model.py
```

## Example Checkpoints

* microsoft/git-base

## Maintainer

Neuroboros Team - Annapurna Labs
