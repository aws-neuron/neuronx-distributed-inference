# Contrib Model: Seed OSS 36B Instruct

NeuronX Distributed Inference implementation of Seed OSS 36B Instruct.

## Model Information

- **HuggingFace ID:** `Seed-OSS-36B-Instruct`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=8, batch_size=None, seq_len=None, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100.0% match** |
| TTFT (P50) | ✅ PASS | 50.97ms (threshold: 100ms) |
| Throughput | ✅ PASS | 27.66 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 50.97ms |
| Throughput | 27.66 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=8, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.38 | 0.00 |
| MBU (%) | 0.21 | 0.60 |
| HFU (%) | 0.40 | 0.01 |
| Execution Time (us) | 0.13 | 0.04 |
| HBM Read | 9.92 GB | 8.87 GB |
| HBM Write | 1.06 GB | 5.78 MB |

**Throughput:** 7.59 tok/s | **Compile Time:** 1007.27s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_seed_oss_36b_instruct import NeuronSeedOSS36BInstructForCausalLM, SeedOSS36BInstructInferenceConfig

model_path = "/path/to/Seed-OSS-36B-Instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=8,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = SeedOSS36BInstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronSeedOSS36BInstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Seed-OSS-36B-Instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Seed-OSS-36B-Instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* Seed-OSS-36B-Instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
