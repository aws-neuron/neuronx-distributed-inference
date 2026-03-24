# Contrib Model: EXAONE 4.0 1.2B

NeuronX Distributed Inference implementation of EXAONE 4.0 1.2B.

## Model Information

- **HuggingFace ID:** `EXAONE-4.0-1.2B`
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
| MFU (%) | 0.08 | 0.00 |
| MBU (%) | 0.15 | 0.55 |
| HFU (%) | 0.09 | 0.00 |
| Execution Time (us) | 0.16 | 0.01 |
| HBM Read | 9.53 GB | 2.59 GB |
| HBM Write | 96.88 MB | 716.8 KB |

**Throughput:** 6.40 tok/s | **Compile Time:** 452.61s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_exaone_4_0_1_2b import NeuronEXAONE4012BForCausalLM, EXAONE4012BInferenceConfig

model_path = "/path/to/EXAONE-4.0-1.2B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = EXAONE4012BInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronEXAONE4012BForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/EXAONE-4.0-1.2B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/EXAONE-4.0-1.2B
python3 test/integration/test_model.py
```

## Example Checkpoints

* EXAONE-4.0-1.2B

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
