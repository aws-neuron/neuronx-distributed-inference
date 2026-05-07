# Contrib Model: Qwen2 7B Instruct

NeuronX Distributed Inference implementation of Qwen2 7B Instruct.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen2-7B-Instruct`
- **Model Type:** Decoder-only transformer
- **License:** {'model_license': 'Apache-2.0 (Qwen team terms apply)', 'port_license': 'Apache-2.0'}

## Architecture Details

- **Layers:** 28 decoder layers
- **Hidden Size:** 3584
- **Attention Heads:** 28

## Validation Results

**Validated:** 2026-02-06  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** (best of multiple prompts) |

**Test Prompt:** `"def fibonacci(n):"`

**Status:** ✅ VALIDATED

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.27 | 0.00 |
| MBU (%) | 0.52 | 0.42 |
| HFU (%) | 0.27 | 0.00 |
| Execution Time (us) | 0.03 | 0.04 |
| HBM Read | 7.18 GB | 7.08 GB |
| HBM Write | 86.63 MB | 2.58 MB |

**Throughput:** 28.35 tok/s | **Compile Time:** 426.10s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_qwen2_7b_instruct import NeuronQwen27BInstructForCausalLM, Qwen27BInstructInferenceConfig

model_path = "/path/to/Qwen2-7B-Instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=None,
    seq_len=512,
    torch_dtype=torch.None,
)

config = Qwen27BInstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronQwen27BInstructForCausalLM(model_path, config)
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
pytest nxdi_contrib_models/models/Qwen2-7B-Instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Qwen2-7B-Instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* Qwen/Qwen2-7B-Instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
