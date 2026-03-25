# Contrib Model: Falcon H1 0.5B Instruct

NeuronX Distributed Inference implementation of Falcon H1 0.5B Instruct.

## Model Information

- **HuggingFace ID:** `Falcon-H1-0.5B-Instruct`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details


## Validation Results

**Validated:** 2026-02-06  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model compiles and loads |
| Token Matching | ❌ FAIL | **0% match** (Mamba SSM implementation needs work) |

**Status:** ⚠️ NEEDS WORK

### Notes

This is a Mamba2 + Attention hybrid model with complex SSM (State Space Model) computation.
The current implementation compiles but produces incorrect outputs due to differences in the
Mamba SSM computation between the HuggingFace implementation and the NeuronX port.

**Known Issues:**
- Mamba SSM chunked computation not matching HuggingFace exactly
- MuP (Maximal Update Parameterization) multipliers may need verification
- Complex state management for SSM not fully implemented

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.00 | 0.00 |
| MBU (%) | 0.01 | 0.27 |
| HFU (%) | 0.01 | 0.01 |
| Execution Time (us) | 0.15 | 0.01 |
| HBM Read | 576.79 MB | 553.79 MB |
| HBM Write | 104.68 MB | 3.26 MB |

**Throughput:** 7.37 tok/s | **Compile Time:** 357.80s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_falcon_h1_0_5b_instruct import NeuronFalconH105BInstructForCausalLM, FalconH105BInstructInferenceConfig

model_path = "/path/to/Falcon-H1-0.5B-Instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=0,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = FalconH105BInstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronFalconH105BInstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Falcon-H1-0.5B-Instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Falcon-H1-0.5B-Instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* Falcon-H1-0.5B-Instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
