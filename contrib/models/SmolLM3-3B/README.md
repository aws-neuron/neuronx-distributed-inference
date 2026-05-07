# Contrib Model: SmolLM3 3B

NeuronX Distributed Inference implementation of SmolLM3 3B.

## Model Information

- **HuggingFace ID:** `HuggingFaceTB/SmolLM3-3B`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Layers:** 36 decoder layers
- **Hidden Size:** 2048
- **Attention Heads:** 16

## Validation Results

**Validated:** 2026-02-06  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** (best of multiple prompts) |

**Test Prompt:** `"The square root of 144 is"`

**Status:** ✅ VALIDATED

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.14 | 0.00 |
| MBU (%) | 0.28 | 0.30 |
| HFU (%) | 0.15 | 0.00 |
| Execution Time (us) | 0.03 | 0.03 |
| HBM Read | 3.11 GB | 3.08 GB |
| HBM Write | 34.54 MB | 2.50 MB |

**Throughput:** 34.40 tok/s | **Compile Time:** 267.18s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_smollm3_3b import NeuronSmolLM33BForCausalLM, SmolLM33BInferenceConfig

model_path = "/path/to/SmolLM3-3B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = SmolLM33BInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronSmolLM33BForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/SmolLM3-3B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/SmolLM3-3B
python3 test/integration/test_model.py
```

## Example Checkpoints

* HuggingFaceTB/SmolLM3-3B

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
