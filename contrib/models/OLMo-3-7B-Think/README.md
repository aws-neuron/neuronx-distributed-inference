# Contrib Model: OLMo 3 7B Think

NeuronX Distributed Inference implementation of OLMo 3 7B Think.

## Model Information

- **HuggingFace ID:** `allenai/OLMo-3-7B-Think`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Layers:** Check model config
- **Hidden Size:** Check model config
- **Attention Heads:** Check model config
- **Vocabulary:** Check model config

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Cosine Similarity | ✅ PASS | **0.9975** |
| Top-1 Accuracy | ✅ PASS | **100%** |

**Status:** EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-20

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.17 | 0.00 |
| MBU (%) | 0.32 | 0.34 |
| HFU (%) | 0.19 | 0.02 |
| Execution Time (us) | 0.05 | 0.06 |
| HBM Read | 7.03 GB | 7.97 GB |
| HBM Write | 107.01 MB | 3.23 MB |

**Throughput:** 17.99 tok/s | **Compile Time:** 226.27s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_olmo_3_7b_think import Model, Config

model_path = "/path/to/OLMo-3-7B-Think/"
compiled_model_path = "/path/to/compiled/"

# Configure and use model
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
pytest nxdi_contrib_models/models/OLMo-3-7B-Think/test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

* allenai/OLMo-3-7B-Think

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-30
