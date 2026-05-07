# Contrib Model: biogpt

NeuronX Distributed Inference implementation of biogpt.

## Model Information

- **HuggingFace ID:** `biogpt`
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
| Token Matching | ✅ PASS | **100.0% match** |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=1, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.34 | 0.00 |
| MBU (%) | 0.19 | 0.38 |
| HFU (%) | 0.40 | 0.03 |
| Execution Time (us) | 0.01 | 0.00 |
| HBM Read | 749.99 MB | 741.80 MB |
| HBM Write | 102.14 MB | 317.0 KB |

**Throughput:** 87.29 tok/s | **Compile Time:** 98.19s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_biogpt import NeuronbiogptForCausalLM, biogptInferenceConfig

model_path = "/path/to/biogpt/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = biogptInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronbiogptForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/biogpt/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/biogpt
python3 test/integration/test_model.py
```

## Example Checkpoints

* biogpt

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
