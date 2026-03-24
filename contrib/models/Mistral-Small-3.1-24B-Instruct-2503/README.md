# Contrib Model: Mistral Small 3.1 24B Instruct 2503

NeuronX Distributed Inference implementation of Mistral Small 3.1 24B Instruct 2503.

## Model Information

- **HuggingFace ID:** `Mistral-Small-3.1-24B-Instruct-2503`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=8, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **96.2% match** |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=8, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.38 | 0.00 |
| MBU (%) | 0.21 | 0.60 |
| HFU (%) | 0.40 | 0.00 |
| Execution Time (us) | 0.08 | 0.02 |
| HBM Read | 6.41 GB | 5.74 GB |
| HBM Write | 670.20 MB | 3.59 MB |

**Throughput:** 11.92 tok/s | **Compile Time:** 384.88s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_mistral_small_3_1_24b_instruct_2503 import NeuronMistralSmall3124BInstruct2503ForCausalLM, MistralSmall3124BInstruct2503InferenceConfig

model_path = "/path/to/Mistral-Small-3.1-24B-Instruct-2503/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=8,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = MistralSmall3124BInstruct2503InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronMistralSmall3124BInstruct2503ForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Mistral-Small-3.1-24B-Instruct-2503/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Mistral-Small-3.1-24B-Instruct-2503
python3 test/integration/test_model.py
```

## Example Checkpoints

* Mistral-Small-3.1-24B-Instruct-2503

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
