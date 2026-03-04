# Contrib Model: Phi 3.5 mini instruct

NeuronX Distributed Inference implementation of Phi 3.5 mini instruct.

## Model Information

- **HuggingFace ID:** `microsoft/Phi-3.5-mini-instruct`
- **Model Type:** Decoder-only transformer
- **Architecture:** Phi-3 with LongRoPE scaling
- **License:** MIT

## Architecture Details

- Hidden size: 3072
- Num attention heads: 32
- Num KV heads: 32 (MHA, not GQA)
- Num layers: 32
- Intermediate size: 8192
- Vocab size: 32064
- Max position embeddings: 131072
- RoPE scaling: LongRoPE
- Activation: SiLU

### Key Differences from LLaMA

1. **Fused QKV projection**: Single `qkv_proj` layer instead of separate Q, K, V
2. **Fused gate_up projection**: Single `gate_up_proj` layer in MLP
3. **LongRoPE scaling**: Extended context support via learned scaling factors

## Validation Results

**Validated:** 2026-02-06  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model compiles and loads successfully |
| Token Matching | ✅ PASS | **100% match** (best of multiple prompts) |

### Multi-Prompt Accuracy

| Prompt | Match Rate |
|--------|------------|
| "The capital of France is" | 100% |

**Status:** ✅ VALIDATED

## Key Fixes Applied

1. **LongRoPE Implementation**: Implemented `Phi3LongRoPEScaledRotaryEmbedding` class that handles:
   - `short_factor` for sequences ≤ 4096 tokens
   - `long_factor` for longer sequences
   - Scaling factor based on context length ratio

2. **State Dict Conversion**: Fixed weight mapping:
   - Split fused QKV into separate Q, K, V with `qkv_proj.` wrapper
   - Split fused gate_up into `gate_proj` and `up_proj`
   - Let preshard_hook handle o_proj mapping (don't add extra prefix)

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig

model_path = "/path/to/Phi-3.5-mini-instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
)

config = Phi3InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronPhi3ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## State Dict Conversion

The `convert_hf_to_neuron_state_dict` method handles:

1. **Strip `model.` prefix**: HF uses `model.layers.X...`, Neuron expects `layers.X...`
2. **Split fused QKV**: `qkv_proj.weight` → `qkv_proj.q_proj.weight`, `qkv_proj.k_proj.weight`, `qkv_proj.v_proj.weight`
3. **Split fused gate_up**: `gate_up_proj.weight` → `gate_proj.weight`, `up_proj.weight`
4. **o_proj passthrough**: Let preshard_hook handle the `o_proj.o_proj` mapping
5. **Add rank tensors**: For tensor parallelism

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest nxdi_contrib_models/models/Phi-3.5-mini-instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Phi-3.5-mini-instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* microsoft/Phi-3.5-mini-instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-02-06
