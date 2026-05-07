# Contrib Model: starcoder2 3b

NeuronX Distributed Inference implementation of starcoder2 3b.

## Model Information

- **HuggingFace ID:** `starcoder2-3b`
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
| Token Matching | ⚠️ PARTIAL | **91.2% match** |
| Throughput | ✅ PASS | 19.50 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 19.50 tokens/s |


**Status:** ✅ GOOD

### Device Profiling Metrics

**Configuration:** TP=1, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.66 | 0.00 |
| MBU (%) | 0.31 | 0.61 |
| HFU (%) | 0.68 | 0.00 |
| Execution Time (us) | 0.05 | 0.02 |
| HBM Read | 6.28 GB | 6.08 GB |
| HBM Write | 218.27 MB | 1.70 MB |

**Throughput:** 19.53 tok/s | **Compile Time:** 355.17s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_starcoder2_3b import Neuronstarcoder23bForCausalLM, starcoder23bInferenceConfig

model_path = "/path/to/starcoder2-3b/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = starcoder23bInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuronstarcoder23bForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/starcoder2-3b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/starcoder2-3b
python3 test/integration/test_model.py
```

## Example Checkpoints

* starcoder2-3b

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
