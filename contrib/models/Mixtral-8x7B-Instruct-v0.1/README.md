# Contrib Model: Mixtral 8x7B Instruct v0.1

NeuronX Distributed Inference implementation of Mixtral 8x7B Instruct v0.1.

## Model Information

- **HuggingFace ID:** `Mixtral-8x7B-Instruct-v0.1`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=5, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100.0% match** |
| Throughput | ⚠️ SLOW | 5.28 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 5.28 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=8, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-20

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.21 | 0.00 |
| MBU (%) | 0.38 | 0.99 |
| HFU (%) | 0.21 | 0.00 |
| Execution Time (us) | 0.08 | 0.01 |
| HBM Read | 11.94 GB | 6.01 GB |
| HBM Write | 260.47 MB | 2.21 MB |

**Throughput:** 12.64 tok/s | **Compile Time:** 1208.41s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_mixtral_8x7b_instruct_v0_1 import NeuronMixtral8x7BInstructv01ForCausalLM, Mixtral8x7BInstructv01InferenceConfig

model_path = "/path/to/Mixtral-8x7B-Instruct-v0.1/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=5,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = Mixtral8x7BInstructv01InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronMixtral8x7BInstructv01ForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Mixtral-8x7B-Instruct-v0.1/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Mixtral-8x7B-Instruct-v0.1
python3 test/integration/test_model.py
```

## Example Checkpoints

* Mixtral-8x7B-Instruct-v0.1

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
