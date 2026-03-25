# Contrib Model: gpt bigcode santacoder

NeuronX Distributed Inference implementation of gpt bigcode santacoder.

## Model Information

- **HuggingFace ID:** `None`
- **Model Type:** Decoder-only transformer
- **License:** {'model_license': 'BigCode OpenRAIL-M', 'port_license': 'Apache-2.0'}

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=1, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ⚠️ PARTIAL | **80.0% match** |
| Throughput | ✅ PASS | 45.37 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 45.37 tokens/s |


**Status:** ✅ GOOD

### Device Profiling Metrics

**Configuration:** TP=1, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.39 | 0.00 |
| MBU (%) | 0.20 | 0.62 |
| HFU (%) | 0.41 | 0.00 |
| Execution Time (us) | 0.03 | 0.01 |
| HBM Read | 2.35 GB | 2.25 GB |
| HBM Write | 111.61 MB | 221.7 KB |

**Throughput:** 48.01 tok/s | **Compile Time:** 108.65s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_gpt_bigcode_santacoder import NeurongptbigcodesantacoderForCausalLM, gptbigcodesantacoderInferenceConfig

model_path = "/path/to/gpt_bigcode-santacoder/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = gptbigcodesantacoderInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeurongptbigcodesantacoderForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/gpt_bigcode-santacoder/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/gpt_bigcode-santacoder
python3 test/integration/test_model.py
```

## Example Checkpoints

* None

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
