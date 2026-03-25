# Contrib Model: internlm3 8b instruct

NeuronX Distributed Inference implementation of internlm3 8b instruct.

## Model Information

- **HuggingFace ID:** `internlm3-8b-instruct`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=None, seq_len=None, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100.0% match** |
| TTFT (P50) | ✅ PASS | 42.82ms (threshold: 100ms) |
| Throughput | ✅ PASS | 29.31 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 42.82ms |
| Throughput | 29.31 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-20

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.35 | 0.00 |
| MBU (%) | 0.19 | 0.59 |
| HFU (%) | 0.37 | 0.00 |
| Execution Time (us) | 0.13 | 0.03 |
| HBM Read | 9.01 GB | 8.30 GB |
| HBM Write | 668.06 MB | 4.57 MB |

**Throughput:** 8.95 tok/s | **Compile Time:** 275.75s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_internlm3_8b_instruct import Neuroninternlm38binstructForCausalLM, internlm38binstructInferenceConfig

model_path = "/path/to/internlm3-8b-instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = internlm38binstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuroninternlm38binstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/internlm3-8b-instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/internlm3-8b-instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* internlm3-8b-instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
