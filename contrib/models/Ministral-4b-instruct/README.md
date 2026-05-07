# Contrib Model: Ministral 4b instruct

NeuronX Distributed Inference implementation of Ministral 4b instruct.

## Model Information

- **HuggingFace ID:** `mistralai/Ministral-4b-instruct`
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
| TTFT (P50) | ✅ PASS | 5.00ms (threshold: 100ms) |
| Throughput | ✅ PASS | 45.35 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 5.00ms |
| Throughput | 45.35 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.42 | 0.00 |
| MBU (%) | 0.23 | 0.61 |
| HFU (%) | 0.45 | 0.00 |
| Execution Time (us) | 0.06 | 0.02 |
| HBM Read | 4.81 GB | 4.33 GB |
| HBM Write | 478.84 MB | 2.40 MB |

**Throughput:** 17.24 tok/s | **Compile Time:** 254.38s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_ministral_4b_instruct import NeuronMinistral4binstructForCausalLM, Ministral4binstructInferenceConfig

model_path = "/path/to/Ministral-4b-instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = Ministral4binstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronMinistral4binstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Ministral-4b-instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Ministral-4b-instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* mistralai/Ministral-4b-instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
