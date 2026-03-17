# Contrib Model: Git-base

NeuronX Distributed Inference implementation of Microsoft's Git (Generative Image-to-text Transformer) text decoder.

## Model Information

- **HuggingFace ID:** `microsoft/git-base`
- **Model Type:** Vision-language model (text decoder compiled on Neuron)
- **Parameters:** ~125M (text decoder)
- **License:** MIT

## Architecture Details

Git uses a BERT-style text decoder with several distinguishing features:

- **Post-LayerNorm residual blocks** (BERT-style, not pre-LN like GPT/LLaMA): LayerNorm is applied after the residual addition, not before
- **Learned absolute position embeddings** (no rotary embeddings)
- **Embedding LayerNorm** applied after combining token + position embeddings
- **Separate Q/K/V projections with bias** in all attention and MLP layers
- **GELU activation** in MLP
- **No final layer norm** (post-LN per block handles normalization)
- **Vision encoder** (CLIP-style ViT) runs on CPU; only text decoder is compiled on Neuron

Key dimensions (git-base):
- hidden_size: 768
- num_attention_heads: 12
- num_hidden_layers: 6
- intermediate_size: 3072
- vocab_size: 30522

## Validation Results

**Validated:** 2026-03-16
**Configuration:** TP=1, batch_size=1, seq_len=128, bf16

| Test | Status | Result |
|------|--------|--------|
| Compilation | PASS | Compiler status PASS |
| Token Matching | PASS | **100% match** (3/3 tokens) |
| TTFT | PASS | 1.28ms avg |
| Throughput | PASS | 728.05 tokens/sec |

Note: Git-base is a vision-language model. Text-only generation quickly hits EOS since the model expects image context. The text decoder compiles and runs correctly on Neuron.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

from src.modeling_git import NeuronGitForCausalLM, GitInferenceConfig

model_path = "/path/to/git-base/"
compiled_model_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    max_context_length=128,
    torch_dtype=torch.bfloat16,
)

config = GitInferenceConfig.from_pretrained(
    model_path, neuron_config=neuron_config,
)

model = NeuronGitForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("a photo of", return_tensors="pt")
# ... generate with manual forward loop (see integration test)
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

*Batch size 1, sequence length 128, BF16 precision, TP=1*

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
