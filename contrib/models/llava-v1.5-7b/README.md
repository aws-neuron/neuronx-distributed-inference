# Contrib Model: llava v1.5 7b

NeuronX Distributed Inference implementation of llava v1.5 7b.

> **Note:** This implementation has been validated using the **text backbone only**. Vision/image modalities are implemented but not yet verified.

## Model Information

- **HuggingFace ID:** `llava-hf/llava-v1.5-7b-hf`
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
| TTFT (P50) | ⚠️ SLOW | 111.16ms (threshold: 100ms) |
| Throughput | ⚠️ SLOW | 9.00 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 111.16ms |
| Throughput | 9.00 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-20

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.17 | 0.00 |
| MBU (%) | 0.32 | 0.33 |
| HFU (%) | 0.18 | 0.00 |
| Execution Time (us) | 0.05 | 0.05 |
| HBM Read | 6.75 GB | 6.65 GB |
| HBM Write | 140.23 MB | 2.86 MB |

**Throughput:** 18.36 tok/s | **Compile Time:** 365.50s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_llava_v1_5_7b import Neuronllavav157bForCausalLM, llavav157bInferenceConfig

model_path = "/path/to/llava-v1.5-7b/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = llavav157bInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuronllavav157bForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/llava-v1.5-7b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/llava-v1.5-7b
python3 test/integration/test_model.py
```

## Example Checkpoints

* llava-hf/llava-v1.5-7b-hf

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
