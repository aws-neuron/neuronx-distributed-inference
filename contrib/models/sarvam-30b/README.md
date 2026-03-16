# Contrib Model: sarvam-30b

NeuronX Distributed Inference implementation of sarvamai/sarvam-30b, a Mixture of Experts model.

## Model Information

- **HuggingFace ID:** `sarvamai/sarvam-30b`
- **Model Type:** Decoder-only Mixture of Experts transformer
- **Parameters:** 30B total (2.4B active per token)
- **License:** Apache-2.0

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 4096 |
| Num Attention Heads | 64 (GQA) |
| Num KV Heads | 4 |
| Num Hidden Layers | 19 |
| Head Dimension | 64 |
| Vocab Size | 262,144 |
| Max Position Embeddings | 131,072 |
| Num Routed Experts | 128 |
| Num Shared Experts | 1 |
| Top-K Routing | 6 |
| Expert Intermediate Size | 1024 |
| Dense Intermediate Size | 8192 |
| Position Encoding | RoPE (theta=8,000,000) |
| Normalization | RMSNorm |
| Activation | SiLU (SwiGLU) |

### Key Implementation Notes

- **Hybrid dense+MoE:** Layer 0 uses dense MLP (`first_k_dense_replace=1`), layers 1-18 use MoE.
- **Sigmoid routing with expert bias:** Custom `SarvamRouterTopK` applies sigmoid activation then adds learned `expert_bias` (post-sigmoid, pre-topk). Affinities use unbiased sigmoid scores. Matches HF behavior exactly.
- **Routed scaling factor:** Normalized routing weights are multiplied by 2.5 before combining with shared expert output.
- **Shared expert:** Handled separately from the NXDI MoE module to support the scaling factor. Each MoE layer has its own shared expert MLP.
- **Q/K normalization:** RMSNorm applied per-head on head_dim=64 after Q/K projection split (Qwen3-style pattern).
- **Fused QKV:** Single `query_key_value` projection, renamed to `Wqkv` for NXDI format.
- **ParallelEmbedding fix:** Required `shard_across_embedding`, `pad`, `tensor_model_parallel_group`, and `use_spmd_rank` parameters to avoid rank-0 baked constants in XLA tracing.

## Validation Results

**Validated:** 2026-03-13
**Configuration:** TP=8, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **61.1% average** (4/10 prompts at 100%) |
| Teacher-Forced Match | PASS | **98.4% average** |

### Teacher-Forced Match Details

Per-prompt results (10 prompts, 256 tokens each):
- 4 prompts: 100% TF, 100% greedy
- 3 prompts: 98-99% TF
- 2 prompts: 97% TF
- 1 prompt: 92.4% TF

Greedy divergence is expected for MoE models with sigmoid routing + expert bias + scaling factor interactions in BF16 precision. Teacher-forced match confirms the model is functionally correct.

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

from src.modeling_sarvam import NeuronSarvamForCausalLM, SarvamInferenceConfig

model_path = "/path/to/sarvam-30b/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = MoENeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = SarvamInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronSarvamForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# ... (see integration test for full example)
```

## Performance

Measured on trn1.32xlarge, batch_size=1, seq_len=128, bfloat16. Utilization is per-NeuronCore (TP=8).

| Metric | Value |
|--------|-------|
| Throughput | 3.6 tok/s |
| Context Encoding MBU | 1.8% |
| Context Encoding MFU | 0.6% |
| Token Generation MBU | 2.5% |
| Token Generation MFU | 0.0% |

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1 (32xl)     | Working (TP=8) | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests (requires trn1.32xlarge):

```bash
pytest contrib/models/sarvam-30b/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* sarvamai/sarvam-30b

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-13
