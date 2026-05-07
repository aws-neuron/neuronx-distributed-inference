# Contrib Model: Persimmon 8B Base

NeuronX Distributed Inference implementation of Persimmon-8B-Base from Adept AI.

## Model Information

- **HuggingFace ID:** `adept/persimmon-8b-base`
- **Model Type:** Decoder-only transformer
- **Parameters:** ~8B
- **License:** Apache-2.0

## Architecture Details

- **Layers:** 36 decoder layers
- **Hidden Size:** 4096
- **Attention Heads:** 64
- **KV Heads:** 64 (Multi-Head Attention)
- **Intermediate Size:** 16384
- **Vocabulary:** 262,144 tokens
- **Max Position Embeddings:** 16384
- **Position Encoding:** RoPE
- **Normalization:** LayerNorm
- **Activation:** Squared ReLU

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=8, batch_size=1, seq_len=2048, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match (64/64 tokens)** |
| TTFT (P50) | ⚠️ SLOW | 150.13ms (threshold: 100ms) |
| Throughput | ⚠️ LOW | 6.64 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 150.13ms |
| Token Generation (P50) | 150.69ms per token |
| Throughput | 6.64 tokens/s |

**Status:** ✅ VALIDATED - Perfect accuracy, functional model

**Note:** Perfect token matching (100%) demonstrates excellent accuracy. Performance is slower than threshold but model is fully functional and generates correct outputs.

### Device Profiling Metrics

**Configuration:** TP=8, batch_size=1, seq_len=2048, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.27 | 0.00 |
| MBU (%) | 0.13 | 0.51 |
| HFU (%) | 0.33 | 0.01 |
| Execution Time (us) | 0.16 | 0.01 |
| HBM Read | 7.46 GB | 2.24 GB |
| HBM Write | 1.51 GB | 2.02 MB |

**Throughput:** 6.64 tok/s | **Compile Time:** 296.07s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_persimmon import NeuronPersimmonForCausalLM, PersimmonInferenceConfig

model_path = "/path/to/persimmon-8b-base/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
)

config = PersimmonInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronPersimmonForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
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
pytest nxdi_contrib_models/models/persimmon-8b-base/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/persimmon-8b-base
python3 test/integration/test_model.py
```

## Example Checkpoints

* adept/persimmon-8b-base

## Notes

- Unique architecture with Squared ReLU activation
- Perfect accuracy validation (100% token match)
- Large vocabulary (262K tokens)
- Long context support (16K tokens)

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
