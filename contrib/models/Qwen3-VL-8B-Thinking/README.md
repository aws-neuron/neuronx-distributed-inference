# Contrib Model: Qwen3 VL 8B Thinking

NeuronX Distributed Inference implementation of Qwen3 VL 8B Thinking.

> **Note:** This implementation has been validated using the **text backbone only**. Vision/image modalities are implemented but not yet verified.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen3-VL-8B-Thinking`
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
| TTFT (P50) | ✅ PASS | 93.57ms (threshold: 100ms) |
| Throughput | ✅ PASS | 10.66 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 93.57ms |
| Throughput | 10.66 tokens/s |


**Status:** ✅ VALIDATED

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.28 | 0.00 |
| MBU (%) | 0.54 | 0.59 |
| HFU (%) | 0.31 | 0.03 |
| Execution Time (us) | 0.04 | 0.03 |
| HBM Read | 7.72 GB | 7.58 GB |
| HBM Write | 129.44 MB | 3.49 MB |

**Throughput:** 26.99 tok/s | **Compile Time:** 317.64s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_qwen3_vl_8b_thinking import NeuronQwen3VL8BThinkingForCausalLM, Qwen3VL8BThinkingInferenceConfig

model_path = "/path/to/Qwen3-VL-8B-Thinking/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = Qwen3VL8BThinkingInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronQwen3VL8BThinkingForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Qwen3-VL-8B-Thinking/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Qwen3-VL-8B-Thinking
python3 test/integration/test_model.py
```

## Example Checkpoints

* Qwen/Qwen3-VL-8B-Thinking

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
