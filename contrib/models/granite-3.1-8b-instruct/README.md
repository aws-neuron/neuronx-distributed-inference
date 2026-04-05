# Contrib Model: granite 3.1 8b instruct

NeuronX Distributed Inference implementation of granite 3.1 8b instruct.

## Model Information

- **HuggingFace ID:** `ibm-granite/granite-3.1-8b-instruct`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Layers:** 32
- **Hidden Size:** 4096
- **Attention Heads:** 32
- **Key-Value Heads:** 8 (GQA)
- **Vocabulary:** 49152
- **Max Position Embeddings:** 131072

### Granite-Specific Scaling Factors

Granite uses custom scaling factors that differ from standard Llama:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_multiplier` | 12.0 | Scales input embeddings after lookup |
| `attention_multiplier` | 0.0078125 (1/head_dim) | Custom attention scaling instead of 1/√head_dim |
| `residual_multiplier` | 0.22 | Scales residual connections |
| `logits_scaling` | 16.0 | Divides output logits |

## Validation Results

**Validated:** 2026-02-06  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** (64/64 tokens) |
| TTFT (P50) | ✅ PASS | ~20ms (threshold: 100ms) |
| Throughput | ✅ PASS | ~100 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | ~20ms |
| Throughput | ~100 tokens/s |

**Status:** ✅ VALIDATED

## Critical Implementation Notes

This implementation includes critical fixes for Granite's custom scaling:

1. **Attention Multiplier Fix**: The `prep_qkv_tensors` method in `NeuronGraniteAttention` applies a correction factor to Q tensors to convert from the standard `1/√head_dim` scaling to Granite's `attention_multiplier`.

2. **Embedding Multiplier**: Applied in `get_model_output` after embedding lookup (not to weights, to handle tied embeddings correctly).

3. **Logits Scaling**: Applied via `ScaledColumnParallelLinear` which divides output by `logits_scaling`.

4. **Residual Multiplier**: Applied in `NeuronGraniteDecoderLayer` to scale residual connections.

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.29 | 0.00 |
| MBU (%) | 0.54 | 0.61 |
| HFU (%) | 0.31 | 0.01 |
| Execution Time (us) | 0.04 | 0.03 |
| HBM Read | 8.30 GB | 8.19 GB |
| HBM Write | 100.60 MB | 3.28 MB |

**Throughput:** 25.37 tok/s | **Compile Time:** 317.64s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_granite import NeuronGraniteForCausalLM, GraniteInferenceConfig

model_path = "/path/to/granite-3.1-8b-instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = GraniteInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronGraniteForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
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
pytest nxdi_contrib_models/models/granite-3.1-8b-instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/granite-3.1-8b-instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* ibm-granite/granite-3.1-8b-instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
