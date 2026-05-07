# Contrib Model: Ouro-1.4B

NeuronX Distributed Inference implementation of Ouro-1.4B, a Universal Transformer model.

## Model Information

- **HuggingFace ID:** `Ouro-1.4B`
- **Model Type:** Decoder-only Universal Transformer
- **Parameters:** 1.4B
- **License:** Apache-2.0

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 2048 |
| Num Attention Heads | 16 (MHA) |
| Num HF Hidden Layers | 24 |
| Total UT Steps | 4 |
| Unrolled Physical Layers | 96 (24 x 4) |
| Vocab Size | 32000 |
| Max Position Embeddings | 2048 |
| Head Dimension | 128 |
| Intermediate Size | 5632 |
| Position Encoding | RoPE (theta=10000) |
| Normalization | RMSNorm (dual pre+post sandwich) |
| Activation | SiLU (SwiGLU) |

### Key Implementation Notes

- **Universal Transformer loop:** The model runs 4 UT steps over 24 layers, sharing weights across steps. For NXDI, this is unrolled into 96 physical layers with duplicated weights so that NXDI can iterate them in a single pass.
- **Dual layer norms:** Each decoder layer applies pre-norm + post-norm sandwich for both attention and MLP blocks (4 RMSNorms per layer).
- **Intermediate norm:** An additional RMSNorm is applied at UT step boundaries (every 24 layers, except the last group which uses the final norm).
- **Separate KV cache per UT step:** 96 total cache slots, one per unrolled layer.
- **Weight conversion:** HF's 24-layer weights are duplicated 4x during conversion. Intermediate norm weights come from the model's final norm.

## Validation Results

**Validated:** 2026-03-10
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **87.0% average** (7/10 prompts at 100%) |
| Teacher-Forced Match | PASS | **98.0% average** |

### Greedy Match Details

7 of 10 prompts achieve 100% greedy match. The 3 prompts with divergence (80%, 50%, 40%) occur on high-entropy continuations where BF16 precision causes cascading differences. All prompts achieve >= 95% teacher-forced match.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

from src.modeling_ouro import NeuronOuroForCausalLM, OuroInferenceConfig

model_path = "/path/to/Ouro-1.4B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = OuroInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronOuroForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# ... (see integration test for full example)
```

## Performance

Profiled on trn1.32xlarge (single NeuronCore utilization):

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| Throughput | - | 23.3 tok/s |
| MBU (Memory) | 19.0% | 15.1% |
| MFU (Compute) | 10.6% | 0.1% |

*Batch size 1, sequence length 128, BF16 precision, TP=1*
## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/Ouro-1.4B/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* Ouro-1.4B

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-10
