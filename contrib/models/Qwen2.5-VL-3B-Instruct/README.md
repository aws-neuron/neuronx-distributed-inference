# Contrib Model: Qwen2.5 VL 3B Instruct

NeuronX Distributed Inference implementation of Qwen2.5 VL 3B Instruct.

> **Note:** This implementation has been validated using the **text backbone only**. Vision/image modalities are implemented but not yet verified.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen2.5-VL-3B-Instruct`
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
| Token Matching | ⚠️ LOW | **67.2% match** |
| TTFT (P50) | ✅ PASS | 29.82ms (threshold: 100ms) |
| Throughput | ✅ PASS | 38.20 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 29.82ms |
| Throughput | 38.20 tokens/s |


**Status:** ✅ GOOD

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-21

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.13 | 0.00 |
| MBU (%) | 0.27 | 0.29 |
| HFU (%) | 0.15 | 0.02 |
| Execution Time (us) | 0.03 | 0.03 |
| HBM Read | 3.15 GB | 3.09 GB |
| HBM Write | 62.86 MB | 3.35 MB |

**Throughput:** 32.98 tok/s | **Compile Time:** 224.93s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_qwen2_5_vl_3b_instruct import NeuronQwen25VL3BInstructForCausalLM, Qwen25VL3BInstructInferenceConfig

model_path = "/path/to/Qwen2.5-VL-3B-Instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = Qwen25VL3BInstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronQwen25VL3BInstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Qwen2.5-VL-3B-Instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Qwen2.5-VL-3B-Instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* Qwen/Qwen2.5-VL-3B-Instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
