# Contrib Model: ERNIE 4.5 0.3B PT

NeuronX Distributed Inference implementation of ERNIE 4.5 0.3B PT.

## Model Information

- **HuggingFace ID:** `ERNIE-4.5-0.3B-PT`
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
| MFU (%) | 0.43 | 0.00 |
| MBU (%) | 0.24 | 0.58 |
| HFU (%) | 0.47 | 0.01 |
| Execution Time (us) | 0.01 | 0.00 |
| HBM Read | 722.59 MB | 732.68 MB |
| HBM Write | 10.90 MB | 956.4 KB |

**Throughput:** 120.23 tok/s | **Compile Time:** 133.99s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_ernie_4_5_0_3b_pt import NeuronERNIE4503BPTForCausalLM, ERNIE4503BPTInferenceConfig

model_path = "/path/to/ERNIE-4.5-0.3B-PT/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = ERNIE4503BPTInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronERNIE4503BPTForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/ERNIE-4.5-0.3B-PT/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/ERNIE-4.5-0.3B-PT
python3 test/integration/test_model.py
```

## Example Checkpoints

* ERNIE-4.5-0.3B-PT

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
