# Contrib Model: recurrentgemma 2b it

NeuronX Distributed Inference implementation of recurrentgemma 2b it.

## Model Information

- **HuggingFace ID:** `google/recurrentgemma-2b-it`
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
| TTFT (P50) | ✅ PASS | 29.11ms (threshold: 100ms) |
| Throughput | ✅ PASS | 33.79 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 29.11ms |
| Throughput | 33.79 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-20

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.20 | 0.00 |
| MBU (%) | 0.47 | 0.62 |
| HFU (%) | 0.21 | 0.00 |
| Execution Time (us) | 0.03 | 0.02 |
| HBM Read | 5.41 GB | 5.37 GB |
| HBM Write | 44.76 MB | 1.11 MB |

**Throughput:** 34.13 tok/s | **Compile Time:** 133.12s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_recurrentgemma_2b_it import Neuronrecurrentgemma2bitForCausalLM, recurrentgemma2bitInferenceConfig

model_path = "/path/to/recurrentgemma-2b-it/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = recurrentgemma2bitInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = Neuronrecurrentgemma2bitForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/recurrentgemma-2b-it/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/recurrentgemma-2b-it
python3 test/integration/test_model.py
```

## Example Checkpoints

* google/recurrentgemma-2b-it

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
