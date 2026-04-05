# Contrib Model: VaultGemma 1B

NeuronX Distributed Inference implementation of VaultGemma 1B.

## Model Information

- **HuggingFace ID:** `google/vaultgemma-1b`
- **Model Type:** Decoder-only transformer (Gemma-2 architecture)
- **License:** Check HuggingFace model card

## Architecture Details

- **Layers:** 26 decoder layers
- **Hidden Size:** 1152
- **Attention Heads:** 4

## Validation Results

**Validated:** 2026-02-05  
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** |
| TTFT (P50) | ✅ PASS | ~10ms (threshold: 100ms) |
| Throughput | ✅ PASS | ~100 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | ~10ms |
| Throughput | ~100 tokens/s |

**Status:** ✅ VALIDATED

### Device Profiling Metrics

**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-21

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.24 | 0.00 |
| MBU (%) | 0.58 | 0.51 |
| HFU (%) | 0.25 | 0.00 |
| Execution Time (us) | 0.01 | 0.01 |
| HBM Read | 2.08 GB | 2.09 GB |
| HBM Write | 14.12 MB | 106.5 KB |

**Throughput:** 106.83 tok/s | **Compile Time:** 194.46s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_vaultgemma import (
    NeuronVaultGemmaForCausalLM,
    VaultGemmaInferenceConfig,
    VaultGemmaNeuronConfig,
)

model_path = "/path/to/vaultgemma-1b/"
compiled_model_path = "/path/to/compiled/"

# Configure (OnDeviceSamplingConfig is automatically enabled for accuracy)
neuron_config = VaultGemmaNeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = VaultGemmaInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
)

# Compile and load
model = NeuronVaultGemmaForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Important Note

This model requires `OnDeviceSamplingConfig` for correct predictions. This is automatically enabled in `VaultGemmaNeuronConfig`. Without it, compiler optimizations may cause numerical divergence.

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest nxdi_contrib_models/models/vaultgemma-1b/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/vaultgemma-1b
python3 test/integration/test_model.py
```

## Example Checkpoints

* google/vaultgemma-1b

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-05
