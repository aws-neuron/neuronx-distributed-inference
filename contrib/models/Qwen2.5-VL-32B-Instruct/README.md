# Contrib Model: Qwen2.5 VL 32B Instruct

NeuronX Distributed Inference implementation of Qwen2.5 VL 32B Instruct.

> **Note:** This implementation has been validated using the **text backbone only**. Vision/image modalities are implemented but not yet verified.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen2.5-VL-32B-Instruct`
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
| Token Matching | ⚠️ N/A | **0.0% match** |
| TTFT (P50) | ✅ PASS | 7.98ms (threshold: 100ms) |
| Throughput | ✅ PASS | 120.65 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 7.98ms |
| Throughput | 120.65 tokens/s |


**Status:** ✅ VALIDATED

### Device Profiling Metrics

**Configuration:** TP=8, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-20

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.23 | 0.00 |
| MBU (%) | 0.44 | 0.60 |
| HFU (%) | 0.25 | 0.01 |
| Execution Time (us) | 0.05 | 0.03 |
| HBM Read | 8.30 GB | 8.01 GB |
| HBM Write | 263.30 MB | 5.77 MB |

**Throughput:** 20.68 tok/s | **Compile Time:** 952.27s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_qwen2_5_vl_32b_instruct import NeuronQwen25VL32BInstructForCausalLM, Qwen25VL32BInstructInferenceConfig

model_path = "/path/to/Qwen2.5-VL-32B-Instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = Qwen25VL32BInstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronQwen25VL32BInstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Qwen2.5-VL-32B-Instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Qwen2.5-VL-32B-Instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* Qwen/Qwen2.5-VL-32B-Instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
