# Contrib Model: helium 1 2b

NeuronX Distributed Inference implementation of helium 1 2b.

## Model Information

- **HuggingFace ID:** `kyutai/helium-1-2b`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Parameters:** 2B

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ⚠️ PARTIAL | **82.2% match** |
| Throughput | ✅ PASS | 42.00 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 42.00 tokens/s |


**Status:** ✅ GOOD

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-20

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.43 | 0.00 |
| MBU (%) | 0.24 | 0.59 |
| HFU (%) | 0.46 | 0.01 |
| Execution Time (us) | 0.02 | 0.01 |
| HBM Read | 2.10 GB | 1.93 GB |
| HBM Write | 222.09 MB | 2.10 MB |

**Throughput:** 40.09 tok/s | **Compile Time:** 193.53s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_helium_1_2b import Neuronhelium12bForCausalLM, helium12bInferenceConfig

model_path = "/path/to/helium-1-2b/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = helium12bInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuronhelium12bForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/helium-1-2b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/helium-1-2b
python3 test/integration/test_model.py
```

## Example Checkpoints

* kyutai/helium-1-2b

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
