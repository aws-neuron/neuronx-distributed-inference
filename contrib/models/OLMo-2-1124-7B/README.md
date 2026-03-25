# Contrib Model: OLMo 2 1124 7B

NeuronX Distributed Inference implementation of OLMo 2 1124 7B.

## Model Information

- **HuggingFace ID:** `allenai/OLMo-2-1124-7B`
- **Model Type:** Decoder-only transformer
- **Parameters:** ~7B
- **License:** Apache 2.0

## Architecture Details

- **Layers:** 32 decoder layers
- **Hidden Size:** 4096
- **Attention Heads:** 32
- **Key-Value Heads:** 32
- **Head Dimension:** 128
- **Intermediate Size:** 11008
- **Vocabulary:** 100,352 tokens
- **Max Position Embeddings:** 4096
- **Position Encoding:** RoPE (theta=500000)
- **Normalization:** RMSNorm
- **Activation:** SiLU (SwiGLU)

### OLMo2-Specific Features

1. **Post-layer normalization**: RMSNorm applied AFTER attention and MLP (not before like LLaMA)
2. **Q-K normalization**: RMSNorm on Q and K projections BEFORE reshaping to heads

## Validation Results

**Validated:** 2026-02-05  
**Configuration:** TP=8, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** |
| TTFT (P50) | ✅ PASS | ~55ms (threshold: 100ms) |
| Throughput | ✅ PASS | ~18 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | ~55ms |
| Throughput | ~18 tokens/s |

**Status:** ✅ VALIDATED

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_olmo2 import (
    NeuronOlmo2ForCausalLM,
    Olmo2InferenceConfig,
    Olmo2NeuronConfig,
)

model_path = "/path/to/OLMo-2-1124-7B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = Olmo2NeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = Olmo2InferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronOlmo2ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

### Device Profiling Metrics

**Configuration:** TP=8, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.14 | 0.00 |
| MBU (%) | 0.29 | 0.53 |
| HFU (%) | 0.16 | 0.00 |
| Execution Time (us) | 0.02 | 0.01 |
| HBM Read | 1.87 GB | 1.73 GB |
| HBM Write | 82.95 MB | 1.02 MB |

**Throughput:** 57.01 tok/s | **Compile Time:** 369.97s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Implementation Notes

### Q-K Normalization with Tensor Parallelism

This model uses Q-K normalization where RMSNorm is applied to Q and K projections BEFORE reshaping to heads. This requires special handling with tensor parallelism (TP > 1):

**The Challenge:**
- Q/K projections are sharded across TP ranks (4096 → 512 per rank with TP=8)
- RMSNorm variance must be computed over the FULL dimension (4096), not the sharded dimension (512)
- Naive implementation computes variance over sharded dimension, causing incorrect normalization

**The Solution:**
The `ShardedRMSNorm` class uses an all-reduce to compute variance correctly:
1. Compute local sum of squares (not mean) over sharded dimension
2. All-reduce across TP ranks to get global sum of squares
3. Divide by FULL dimension size to get correct variance
4. Apply normalization with the correct variance

This fix was critical for achieving 100% token match accuracy with TP=8.

See `NEURON_PORT_DEBUGGING_GUIDE.md` for detailed documentation of this issue and solution.

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest nxdi_contrib_models/models/OLMo-2-1124-7B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/OLMo-2-1124-7B
python3 test/integration/test_model.py
```

## Example Checkpoints

* allenai/OLMo-2-1124-7B

## Notes

- Post-layer normalization architecture (different from LLaMA's pre-norm)
- Q-K RMSNorm requires special handling for tensor parallelism
- Perfect accuracy validation (100% token match with TP=8)

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-05
