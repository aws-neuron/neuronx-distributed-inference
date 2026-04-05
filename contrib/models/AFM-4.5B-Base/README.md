# Contrib Model: AFM 4.5B Base (Arcee)

NeuronX Distributed Inference implementation of AFM 4.5B Base (Arcee architecture).

## Model Information

- **HuggingFace ID:** `arcee-ai/AFM-4.5B-Base`
- **Model Type:** Decoder-only transformer
- **Architecture:** Arcee (similar to LLaMA with YaRN RoPE and ReLU² activation)
- **License:** Check HuggingFace model card

## Architecture Details

- **Hidden Size:** 2560
- **Num Layers:** 36
- **Attention Heads:** 20 (Q), 4 (KV) - Grouped Query Attention
- **Head Dim:** 128
- **Intermediate Size:** 18432
- **Vocab Size:** 128004
- **Max Position Embeddings:** 65536
- **RoPE Scaling:** YaRN (factor=20, original_max_pos=4096)
- **Activation:** ReLU² (relu(x).pow(2))

## Validation Results

**Validated:** 2026-02-06
**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100% match** |

### Key Implementation Notes

1. **YaRN RoPE Scaling:** Implements the YaRN (Yet another RoPE extensioN) mechanism for extended context support (65k tokens)
2. **ReLU² Activation:** Uses `relu(x).pow(2)` instead of SwiGLU - only `up_proj` and `down_proj` (no `gate_proj`)
3. **State Dict Conversion:** QKV projections are separate in HF, combined into `qkv_proj` for Neuron

### Device Profiling Metrics

**Configuration:** TP=1, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-18

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.28 | 0.00 |
| MBU (%) | 0.58 | 0.61 |
| HFU (%) | 0.31 | 0.03 |
| Execution Time (us) | 0.04 | 0.03 |
| HBM Read | 9.23 GB | 8.60 GB |
| HBM Write | 218.55 MB | 2.03 MB |

**Throughput:** 24.92 tok/s | **Compile Time:** 588.37s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_afm import NeuronAFMForCausalLM, AFMInferenceConfig

model_path = "/path/to/AFM-4.5B-Base/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = AFMInferenceConfig(
    neuron_config=neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronAFMForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("1+1=", return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
# Output: 1+1=2, 2+2=4, 3+3=6, 4+4=8, ...
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest nxdi_contrib_models/models/AFM-4.5B-Base/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/AFM-4.5B-Base
python3 test/integration/test_model.py
```

## Example Checkpoints

* arcee-ai/AFM-4.5B-Base

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
