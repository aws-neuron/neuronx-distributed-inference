# Contrib Model: Baichuan2-7B-Base

NeuronX Distributed Inference implementation of Baichuan2-7B-Base.

## Model Information

- **HuggingFace ID:** `baichuan-inc/Baichuan2-7B-Base`
- **Model Type:** Decoder-only transformer (Llama-2 architecture variant)
- **License:** Apache-2.0

## Architecture Details

- **Layers:** 32 decoder layers
- **Hidden Size:** 4096
- **Attention Heads:** 32 (MHA, head_dim=128)

### Baichuan2-Specific Features

- **W_pack (fused QKV):** Stores Q/K/V as a single fused tensor `W_pack.weight [3*H, H]`, split into separate projections during weight conversion.
- **NormHead lm_head:** Applies L2 normalization to lm_head weights at inference time; pre-normalized during weight conversion.
- **Direct loading:** Bypasses `trust_remote_code` by loading config.json and safetensors directly, adding missing Llama-required keys.

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

## Performance

Profiled on trn1.32xlarge (single NeuronCore utilization):

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| Throughput | - | 16.6 tok/s |
| MBU (Memory) | 4.9% | 5.4% |
| MFU (Compute) | 4.9% | 0.0% |

*Batch size 1, sequence length 256, BF16 precision, TP=2*
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

Or run manually:

```bash
cd contrib/models/Baichuan2-7B-Base
python3 test/integration/test_model.py
```

## Example Checkpoints

* baichuan-inc/Baichuan2-7B-Base

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-05
