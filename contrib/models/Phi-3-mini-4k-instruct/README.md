# Contrib Model: Phi 3 mini 4k instruct

NeuronX Distributed Inference implementation of Phi 3 mini 4k instruct.

## Model Information

- **HuggingFace ID:** `Phi-3-mini-4k-instruct`
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
| MFU (%) | 0.50 | 0.00 |
| MBU (%) | 0.27 | 0.49 |
| HFU (%) | 0.52 | 0.01 |
| Execution Time (us) | 0.08 | 0.04 |
| HBM Read | 8.40 GB | 7.65 GB |
| HBM Write | 730.79 MB | 718.1 KB |

**Throughput:** 11.91 tok/s | **Compile Time:** 834.02s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_phi_3_mini_4k_instruct import NeuronPhi3mini4kinstructForCausalLM, Phi3mini4kinstructInferenceConfig

model_path = "/path/to/Phi-3-mini-4k-instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = Phi3mini4kinstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronPhi3mini4kinstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Phi-3-mini-4k-instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Phi-3-mini-4k-instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* Phi-3-mini-4k-instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
