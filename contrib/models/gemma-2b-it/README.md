# Contrib Model: gemma 2b it

NeuronX Distributed Inference implementation of gemma 2b it.

## Model Information

- **HuggingFace ID:** `google/gemma-2b-it`
- **Model Type:** Decoder-only transformer
- **License:** Gemma Terms of Use (Google)

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=1, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100.0% match** |
| Throughput | ✅ PASS | 25.24 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 25.24 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=1, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.40 | 0.00 |
| MBU (%) | 0.37 | 0.61 |
| HFU (%) | 0.40 | 0.00 |
| Execution Time (us) | 0.06 | 0.02 |
| HBM Read | 7.62 GB | 5.02 GB |
| HBM Write | 911.35 MB | 1.09 MB |

**Throughput:** 25.30 tok/s | **Compile Time:** 317.32s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_gemma_2b_it import Neurongemma2bitForCausalLM, gemma2bitInferenceConfig

model_path = "/path/to/gemma-2b-it/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = gemma2bitInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neurongemma2bitForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/gemma-2b-it/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/gemma-2b-it
python3 test/integration/test_model.py
```

## Example Checkpoints

* google/gemma-2b-it

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
