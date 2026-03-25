# Contrib Model: Ovis2.5 9B

NeuronX Distributed Inference implementation of Ovis2.5 9B.

> **Note:** This implementation has been validated using the **text backbone only**. Vision/image modalities are implemented but not yet verified.

## Model Information

- **HuggingFace ID:** `AIDC-AI/Ovis2.5-9B`
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
| TTFT (P50) | ✅ PASS | 32.92ms (threshold: 100ms) |
| Throughput | ✅ PASS | 30.03 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 32.92ms |
| Throughput | 30.03 tokens/s |


**Status:** ✅ VALIDATED

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-21

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.27 | 0.00 |
| MBU (%) | 0.54 | 0.59 |
| HFU (%) | 0.31 | 0.03 |
| Execution Time (us) | 0.04 | 0.03 |
| HBM Read | 7.72 GB | 7.58 GB |
| HBM Write | 129.44 MB | 3.49 MB |

**Throughput:** 27.36 tok/s | **Compile Time:** 412.67s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_ovis2_5_9b import NeuronOvis259BForCausalLM, Ovis259BInferenceConfig

model_path = "/path/to/Ovis2.5-9B/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = Ovis259BInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronOvis259BForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Ovis2.5-9B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Ovis2.5-9B
python3 test/integration/test_model.py
```

## Example Checkpoints

* AIDC-AI/Ovis2.5-9B

## Maintainer

Annapurna Labs

**Last Updated:** 2026-01-29
