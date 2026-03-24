# Contrib Model: Apertus 8B Instruct 2509

NeuronX Distributed Inference implementation of Apertus 8B Instruct 2509.

## Model Information

- **HuggingFace ID:** `swiss-ai/Apertus-8B-Instruct-2509`
- **Model Type:** Decoder-only transformer
- **License:** See HuggingFace model page

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ⚠️ PARTIAL | **84.7% match** |


**Status:** ✅ GOOD

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.47 | 0.00 |
| MBU (%) | 0.25 | 0.59 |
| HFU (%) | 0.48 | 0.00 |
| Execution Time (us) | 0.09 | 0.03 |
| HBM Read | 8.33 GB | 7.56 GB |
| HBM Write | 516.55 MB | 3.22 MB |

**Throughput:** 11.50 tok/s | **Compile Time:** 359.84s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_apertus_8b_instruct_2509 import NeuronApertus8BInstruct2509ForCausalLM, Apertus8BInstruct2509InferenceConfig

model_path = "/path/to/Apertus-8B-Instruct-2509/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = Apertus8BInstruct2509InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronApertus8BInstruct2509ForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Apertus-8B-Instruct-2509/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Apertus-8B-Instruct-2509
python3 test/integration/test_model.py
```

## Example Checkpoints

* swiss-ai/Apertus-8B-Instruct-2509

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
