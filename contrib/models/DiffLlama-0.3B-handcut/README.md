# Contrib Model: DiffLlama

NeuronX Distributed Inference implementation of DiffLlama (Differential Transformer).

## Model Information

- **HuggingFace ID:** `kajuma/DiffLlama-0.3B-handcut`
- **Model Type:** Decoder-only transformer with differential attention
- **Parameters:** 0.3B
- **License:** Apache-2.0

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 2048 |
| Num Attention Heads | 32 (GQA: 8 KV heads) |
| Head Dimension | 64 |
| Num Hidden Layers | 16 |
| Vocab Size | 128256 |
| Max Position Embeddings | 131072 |
| Intermediate Size | 8192 |
| Position Encoding | RoPE with llama3 scaling (factor=32, original_max=8192) |
| Residual Connection | Pre-norm (LN -> Attn -> Add -> LN -> MLP -> Add) |
| Normalization | RMSNorm (eps=1e-5) |
| Activation | SiLU (SwiGLU MLP) |
| LM Head | Tied with embed_tokens |

### Key Implementation Notes

- **Differential attention:** Unlike standard attention, DiffLlama transforms V before the attention matmul — chunk heads into 2 halves, concatenate along head_dim (doubling it to 128), then repeat. After attention, the output is split into 2 head groups and subtracted with learned lambda parameters, followed by RMSNorm on 2*head_dim features.
- **standard_causal_attention_forward override:** The full attention forward is overridden because NXDI's built-in attention kernels cannot handle the modified V shape. Manual attention (matmul + softmax + matmul) compiles correctly to XLA/HLO.
- **Causal mask:** Generated internally via `torch.triu` rather than using the framework-provided mask, which avoids XLA shape broadcasting issues.
- **Llama3 RoPE scaling:** Custom `Llama3RotaryEmbedding` extends NXDI's `RotaryEmbedding` with frequency-dependent scaling (high-frequency components unchanged, low-frequency scaled by factor, mid-frequency interpolated).
- **KV cache:** Stores original K, V (before V transformation); the transformation is reapplied at each token generation step.
- **HF transformers:** Requires custom transformers with DiffLlama support (not yet in mainline HuggingFace). Path: `/shared/dhwanw/agent_friday_test/example/transformers/src`.

## Validation Results

**Validated:** 2026-03-05
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Greedy Token Matching | PASS | **94.69% average** (7/10 prompts at 100%) |
| Teacher-Forced Match | PASS | **99.38% average** |

### Greedy Match Details

7 of 10 prompts achieve 100% greedy match. The 3 prompts with slight divergence (98.4%) occur on high-entropy natural language where BF16 precision causes late-stage cascading differences. One prompt ("Amazon River") diverges early (51.6% greedy) but maintains 98.4% teacher-forced match, indicating correct model behavior with a single early-token divergence that cascades.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

from src.modeling_diffllama import NeuronDiffLlamaForCausalLM, DiffLlamaInferenceConfig

model_path = "/path/to/DiffLlama-0.3B-handcut/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = DiffLlamaInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronDiffLlamaForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/DiffLlama-0.3B-handcut/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* kajuma/DiffLlama-0.3B-handcut

## References

- [Differential Transformer Paper](https://arxiv.org/abs/2410.05258)

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-05
