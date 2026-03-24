# Contrib Model: opt 1.3b

NeuronX Distributed Inference implementation of opt 1.3b.

## Model Information

- **HuggingFace ID:** `opt-1.3b`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=1, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ⚠️ PARTIAL | **81.2% match** |
| Throughput | ✅ PASS | 79.00 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 79.00 tokens/s |


**Status:** ✅ GOOD

### Device Profiling Metrics

**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-20

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.28 | 0.00 |
| MBU (%) | 0.55 | 0.48 |
| HFU (%) | 0.32 | 0.02 |
| Execution Time (us) | 0.01 | 0.01 |
| HBM Read | 2.65 GB | 2.65 GB |
| HBM Write | 52.12 MB | 496.0 KB |

**Throughput:** 78.09 tok/s | **Compile Time:** 74.34s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_opt_1_3b import Neuronopt13bForCausalLM, opt13bInferenceConfig

model_path = "/path/to/opt-1.3b/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = opt13bInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuronopt13bForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/opt-1.3b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/opt-1.3b
python3 test/integration/test_model.py
```

## Example Checkpoints

* opt-1.3b

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
