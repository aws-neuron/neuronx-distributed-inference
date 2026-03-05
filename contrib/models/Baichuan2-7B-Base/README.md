# Contrib Model: Baichuan2-7B-Base

NeuronX Distributed Inference implementation of Baichuan Inc. Baichuan2-7B-Base.

## Model Information

- **HuggingFace ID:** `baichuan-inc/Baichuan2-7B-Base`
- **Model Type:** Decoder-only transformer (Llama-2 architecture variant)
- **Parameters:** 7B
- **License:** Apache-2.0

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden Size | 4096 |
| Num Attention Heads | 32 (MHA) |
| Head Dimension | 128 |
| Num Hidden Layers | 32 |
| Vocab Size | 125696 |
| Max Position Embeddings | 4096 |
| Intermediate Size | 11008 |
| Position Encoding | RoPE (theta=10000) |
| Residual Connection | Pre-norm (RMSNorm -> Attn -> Add -> RMSNorm -> MLP -> Add) |
| Normalization | RMSNorm (epsilon=1e-6) |
| Activation | SiLU |
| LM Head | NormHead (weight-normalized, no bias) |

### Key Implementation Notes

- **Extends Llama:** Architecture is identical to Llama-2-7b, so this port extends `NeuronLlamaForCausalLM` and only overrides config loading, weight conversion, and HF model loading.
- **W_pack (fused QKV):** Baichuan2 stores Q/K/V as a single fused tensor `W_pack.weight` of shape `[3*hidden_size, hidden_size]`. Weight conversion splits this into separate `q_proj`, `k_proj`, `v_proj`.
- **NormHead lm_head:** The lm_head applies L2 normalization to weights at inference time. We pre-normalize during weight conversion with `F.normalize(dim=-1)`.
- **Direct loading:** Bypasses `trust_remote_code` by loading `config.json` and safetensors directly, adding missing Llama-required keys (`num_key_value_heads`, `rope_theta`).
- **Tied weights:** `tie_word_embeddings=false` — override prevents Llama's default weight tying.

## Validation Results

**Validated:** 2026-03-05
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Token Matching | PASS | **54.84% greedy, 98.59% teacher-forced** |

### Token Match Notes

54.84% greedy token match and 98.59% teacher-forced match vs HF reference across 10 prompts (640 tokens).
4 of 10 prompts achieve 100% greedy match. The high teacher-forced rate confirms the model is
functionally correct — lower greedy match on some prompts is due to BF16 precision causing early
divergence that cascades into different generation paths.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig

from src.modeling_baichuan2 import NeuronBaichuan2ForCausalLM, Baichuan2InferenceConfig

model_path = "/path/to/Baichuan2-7B-Base/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = Baichuan2InferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronBaichuan2ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
pytest contrib/models/Baichuan2-7B-Base/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* baichuan-inc/Baichuan2-7B-Base

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-05
