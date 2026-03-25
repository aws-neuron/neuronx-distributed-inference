# Contrib Model: Llama 2 7b hf

NeuronX Distributed Inference implementation of Llama 2 7b hf.

## Model Information

- **HuggingFace ID:** `meta-llama/Llama-2-7b-hf`
- **Model Type:** Decoder-only transformer
- **License:** Llama 2 Community License Agreement

## Architecture Details

- **Layers:** 32 decoder layers
- **Hidden Size:** 4096
- **Attention Heads:** 32

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=None, seq_len=None, None

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100.0% match** |
| TTFT (P50) | ⚠️ SLOW | 100.00ms (threshold: 100ms) |
| Throughput | ⚠️ SLOW | 10.00 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 100.00ms |
| Throughput | 10.00 tokens/s |


**Status:** ✅ EXCELLENT

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=512, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.40 | 0.00 |
| MBU (%) | 0.20 | 0.33 |
| HFU (%) | 0.42 | 0.00 |
| Execution Time (us) | 0.09 | 0.05 |
| HBM Read | 7.12 GB | 6.75 GB |
| HBM Write | 584.02 MB | 2.86 MB |

**Throughput:** 12.54 tok/s | **Compile Time:** 362.00s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_llama_2_7b_hf import NeuronLlama27bhfForCausalLM, Llama27bhfInferenceConfig

model_path = "/path/to/Llama-2-7b-hf/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = Llama27bhfInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronLlama27bhfForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Llama-2-7b-hf/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Llama-2-7b-hf
python3 test/integration/test_model.py
```

## Example Checkpoints

* meta-llama/Llama-2-7b-hf

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
