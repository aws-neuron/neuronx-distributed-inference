# Contrib Model: bitnet-b1.58-2B-4T

NeuronX Distributed Inference implementation of microsoft/BitNet-b1.58-2B-4T, a Llama-variant with ternary quantized weights.

## Model Information

- **HuggingFace ID:** `microsoft/BitNet-b1.58-2B-4T`
- **Model Type:** Decoder-only transformer with ternary weights
- **Parameters:** 2B (ternary quantized, 1.58 bits per weight)
- **License:** MIT

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 2560 |
| Num Attention Heads | 20 (GQA) |
| Num KV Heads | 5 |
| Num Hidden Layers | 30 |
| Head Dimension | 128 |
| Vocab Size | 128,256 |
| Max Position Embeddings | 4,096 |
| Intermediate Size | 6912 |
| Position Encoding | RoPE (theta=500,000) |
| Normalization | RMSNorm |
| Activation | ReLU squared (relu2) |
| Tied Embeddings | Yes |

### Key Implementation Notes

- **Ternary weight unpacking:** Weights are stored as packed uint8 (4 values per byte, values: -1/0/+1). Unpacked during `convert_hf_to_neuron_state_dict` and scaled by per-tensor `weight_scale`.
- **Sub-norm fusion:** Both `attn_sub_norm` (before o_proj) and `ffn_sub_norm` (before down_proj) have their gamma fused into the following linear layer's weights. At runtime, `_TPAwareUnitRMSNorm` applies unit RMSNorm with TP-aware all-reduce for correct RMS computation.
- **ReLU squared activation:** Uses `relu2` (ReLU(x)^2) instead of SiLU/SwiGLU.
- **Tied word embeddings:** `lm_head` shares weights with `embed_tokens`, handled via `update_state_dict_for_tied_weights`.
- **KV replication:** When `num_kv_heads % tp_degree != 0`, KV heads are replicated via `repeat_interleave` for CONVERT_TO_MHA compatibility.

## Validation Results

**Validated:** 2026-03-13
**Configuration:** TP=2, batch_size=1, seq_len=256, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **70.9% average** (4/10 prompts at 100%) |
| Teacher-Forced Match | PASS | **97.2% average** |

### Teacher-Forced Match Details

Per-prompt results (10 prompts, 32 tokens each):
- 4 prompts: 100% TF, 100% greedy
- 3 prompts: 96.9% TF
- 2 prompts: 90-94% TF
- 1 prompt: 96.9% TF

BF16 precision causes greedy divergence on some prompts. Teacher-forced match confirms the model is functionally correct.

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

from src.modeling_bitnet import NeuronBitNetForCausalLM, BitNetInferenceConfig

model_path = "/path/to/BitNet-b1.58-2B-4T/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=256,
    torch_dtype=torch.bfloat16,
)

config = BitNetInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronBitNetForCausalLM(model_path, config)
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
| Throughput | - | 26.7 tok/s |
| MBU (Memory) | 3.2% | 8.8% |
| MFU (Compute) | 3.0% | 0.0% |

*Batch size 1, sequence length 256, BF16 precision, TP=2*
## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1 (32xl)     | Working (TP=2) | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests (requires trn1.32xlarge):

```bash
pytest contrib/models/bitnet-b1.58-2B-4T/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* microsoft/BitNet-b1.58-2B-4T

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-13
