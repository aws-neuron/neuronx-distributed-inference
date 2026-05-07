# Contrib Model: Orion 14B Chat

NeuronX Distributed Inference implementation of Orion-14B-Chat from OrionStar AI.

## Model Information

- **HuggingFace ID:** `OrionStarAI/Orion-14B-Chat`
- **Model Type:** Decoder-only transformer (Llama-based with modifications)
- **Parameters:** ~14B
- **License:** Orion Community License

## Architecture Details

- **Layers:** 40 decoder layers
- **Hidden Size:** 5120
- **Attention Heads:** 40
- **KV Heads:** 40
- **Intermediate Size:** 15360
- **Vocabulary:** 84,608 tokens
- **Max Position Embeddings:** 4096
- **Position Encoding:** RoPE
- **Normalization:** RMSNorm
- **Activation:** SwiGLU

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=8, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match (3/3 tokens)** |
| TTFT (P50) | ✅ PASS | 25.80ms (threshold: 100ms) |
| Throughput | ✅ PASS | 38.00 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 25.80ms |
| Token Generation (P50) | 26.00ms per token |
| Throughput | 38.00 tokens/s |

**Status:** ✅ EXCELLENT - Perfect accuracy, outstanding performance

### Device Profiling Metrics

**Configuration:** TP=8, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-20

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.19 | 0.00 |
| MBU (%) | 0.38 | 0.58 |
| HFU (%) | 0.19 | 0.00 |
| Execution Time (us) | 0.03 | 0.01 |
| HBM Read | 3.79 GB | 3.54 GB |
| HBM Write | 180.11 MB | 3.87 MB |

**Throughput:** 38.90 tok/s | **Compile Time:** 521.68s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_orion import OrionForCausalLM, OrionInferenceConfig

model_path = "/path/to/orion-14b-chat/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = OrionInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = OrionForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# ... (see integration test for full example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest nxdi_contrib_models/models/orion-14b-chat/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/orion-14b-chat
python3 test/integration/test_model.py
```

## Example Checkpoints

* OrionStarAI/Orion-14B-Chat

## Notes

- Llama-based architecture with custom modifications
- Excellent performance: 38 tokens/second
- Perfect token matching with HF reference
- Multilingual support (Chinese, English, Japanese, Korean)

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
