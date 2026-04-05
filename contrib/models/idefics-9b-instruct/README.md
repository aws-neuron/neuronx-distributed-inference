# Contrib Model: idefics 9b instruct

NeuronX Distributed Inference implementation of idefics 9b instruct.

> **Note:** This implementation has been validated using the **text backbone only**. Vision/image modalities are implemented but not yet verified.

## Model Information

- **HuggingFace ID:** `HuggingFaceM4/idefics-9b-instruct`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Layers:** Check model config
- **Hidden Size:** Check model config
- **Attention Heads:** Check model config
- **Vocabulary:** Check model config
- **Max Position Embeddings:** Check model config

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100.0% match** |
| TTFT (P50) | ✅ PASS | 74.93ms (threshold: 100ms) |
| Throughput | ✅ PASS | 13.10 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 74.93ms |
| Throughput | 13.10 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-21

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.25 | 0.00 |
| MBU (%) | 0.47 | 0.42 |
| HFU (%) | 0.25 | 0.01 |
| Execution Time (us) | 0.07 | 0.08 |
| HBM Read | 13.97 GB | 13.29 GB |
| HBM Write | 330.92 MB | 2.00 MB |

**Throughput:** 13.40 tok/s | **Compile Time:** 755.21s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_idefics_9b_instruct import Neuronidefics9binstructForCausalLM, idefics9binstructInferenceConfig

model_path = "/path/to/idefics-9b-instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = idefics9binstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuronidefics9binstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/idefics-9b-instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/idefics-9b-instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* HuggingFaceM4/idefics-9b-instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
