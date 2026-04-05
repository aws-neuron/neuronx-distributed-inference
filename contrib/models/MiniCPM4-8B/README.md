# Contrib Model: MiniCPM4 8B

NeuronX Distributed Inference implementation of MiniCPM4 8B.

## Model Information

- **HuggingFace ID:** `MiniCPM4-8B`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100.0% match** |
| Throughput | ✅ PASS | 22.80 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 22.80 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.38 | 0.00 |
| MBU (%) | 0.19 | 0.60 |
| HFU (%) | 0.40 | 0.02 |
| Execution Time (us) | 0.11 | 0.03 |
| HBM Read | 8.41 GB | 7.90 GB |
| HBM Write | 340.11 MB | 2.57 MB |

**Throughput:** 9.95 tok/s | **Compile Time:** 274.78s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_minicpm4_8b import NeuronMiniCPM48BForCausalLM, MiniCPM48BInferenceConfig

model_path = "/path/to/MiniCPM4-8B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = MiniCPM48BInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronMiniCPM48BForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/MiniCPM4-8B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/MiniCPM4-8B
python3 test/integration/test_model.py
```

## Example Checkpoints

* MiniCPM4-8B

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
