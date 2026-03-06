# Contrib Model: GLM-4.5 MoE

NeuronX Distributed Inference implementation of [GLM-4.5 MoE](https://huggingface.co/zai-org/GLM-4.5-Air) — a Mixture-of-Experts language model from ZhipuAI / Tsinghua University with unique architectural features including partial RoPE, sigmoid routing with group selection, and shared experts.

## Model Information

- **HuggingFace ID:** `zai-org/GLM-4.5-Air`
- **Model Type:** Decoder-only MoE transformer (`Glm4MoeForCausalLM`)
- **Architecture:** 46 layers, hidden size 4096, 128 routed experts, 2 shared experts
- **Parameters:** ~70B total, ~9B active per token
- **License:** [GLM-4 License](https://huggingface.co/zai-org/GLM-4.5-Air)

## Architecture Details

GLM-4.5 MoE has several differences from standard MoE models that required custom implementations:

| Feature | GLM-4.5 MoE | Standard MoE (e.g. Qwen3MoE) |
|---|---|---|
| RoPE | Partial (first 50% of head_dim) | Full |
| QKV Bias | Yes (`attention_bias=True`) | No |
| Router activation | Sigmoid | Softmax |
| Routing | Group-limited top-k | Top-k |
| Correction bias | `e_score_correction_bias` | None |
| Weight normalization | `norm_topk_prob` + `routed_scaling_factor` | Simple softmax |
| Shared experts | `n_shared_experts=1` (always active) | 0 or variable |
| First `k` layers | Dense MLP (`first_k_dense_replace`) | All MoE |

### Full Architecture (GLM-4.5 Air)

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 46 |
| `hidden_size` | 4096 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 2 |
| `head_dim` | 128 |
| `partial_rotary_factor` | 0.5 (rotary_dim = 64) |
| `attention_bias` | True |
| `n_routed_experts` | 128 |
| `num_experts_per_tok` | 8 |
| `n_shared_experts` | 1 |
| `first_k_dense_replace` | 1 |
| `moe_intermediate_size` | 2048 |
| `intermediate_size` | 16384 (dense layers) |
| `n_group` | 8 |
| `topk_group` | 4 |
| `vocab_size` | 151552 |
| `max_position_embeddings` | 131072 |

## Validation Results

**Tested with:** Reduced 2-layer config (`hidden_size=512`, `n_routed_experts=8`, random weights) on `trn2.3xlarge` (LNC=2, 96 GB Neuron memory)  
**Configuration:** TP=2 (LNC=2), `batch_size=1`, `seq_len=128`, `bfloat16`  
**Date:** 2026-03-06

> Note: Full model validation requires a larger Trn2 instance (e.g. `trn2.48xlarge`) for the 70B full model.
> The integration test uses a reduced random-weight model to verify model structure, compilation, and logit accuracy
> without requiring the full checkpoint or large hardware.

### Test Results

| Test | Status | Notes |
|------|--------|-------|
| Model compilation | ✅ PASS | Reduced config (2L, h=512), TP=2, `trn2.3xlarge` |
| Model load | ✅ PASS | |
| Logit accuracy (`check_accuracy_logits_v2`) | ✅ PASS | `divergence_difference_tol=0.001` |
| Unit: router top-k (10 tests) | ✅ PASS | CPU-only |
| Unit: partial RoPE (8 tests) | ✅ PASS | CPU-only |
| Unit: decoder layer dispatch (15 tests) | ✅ PASS | CPU-only |
| **Total** | **✅ 53/53 PASS** | |

## Usage

### Compile and Run

```python
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

# Add src to path (or install as package)
import sys
sys.path.insert(0, "contrib/models/glm4_moe/src")
from glm4_moe.modeling_glm4_moe import Glm4MoeInferenceConfig, NeuronGlm4MoeForCausalLM

model_path = "/path/to/GLM-4.5-Air"          # HuggingFace checkpoint
compiled_model_path = "/path/to/compiled"     # Neuron compiled artifacts

# 1. Configure
neuron_config = MoENeuronConfig(
    tp_degree=32,
    moe_tp_degree=32,
    moe_ep_degree=1,
    batch_size=1,
    seq_len=4096,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False, top_k=1),
    fused_qkv=True,
    flash_decoding_enabled=True,
)

inference_config = Glm4MoeInferenceConfig(
    neuron_config=neuron_config,
    load_config=load_pretrained_config(model_path),
)

# 2. Compile (run once, ~hours)
model = NeuronGlm4MoeForCausalLM(model_path, inference_config)
model.compile(compiled_model_path)

# 3. Load compiled model
model.load(compiled_model_path)

# 4. Generate
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
adapter = HuggingFaceGenerationAdapter(model)

prompt = "Explain mixture-of-experts routing in one paragraph."
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = adapter.generate(
        **inputs,
        generation_config=GenerationConfig(do_sample=False, max_new_tokens=200),
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Using the Demo Script

```bash
cd contrib/models/glm4_moe

# Compile and run generation demo
python examples/generation_glm4_moe_demo.py \
    --model-path /path/to/GLM-4.5-Air \
    --compiled-model-path /path/to/compiled \
    --tp-degree 32 \
    --seq-len 4096
```

## Compatibility Matrix

| Instance / NxDI Version | 2.21+ | 2.20 | 2.19 and earlier |
|---|---|---|---|
| Trn2 (`trn2.48xlarge`, 512 NCs) | ✅ Recommended | Not tested | Not supported |
| Trn2 (`trn2.3xlarge`, 4 NCs) | ✅ Tested (reduced config, 2026-03-06) | Not tested | Not supported |
| Trn1 (`trn1.32xlarge`, 64 NCs) | Not tested | Not tested | Not supported |
| Inf2 | Not tested | Not tested | Not supported |

> **Minimum requirements:** `transformers>=4.56.0` (for `Glm4MoeForCausalLM`), AWS Neuron SDK 2.21+

## Testing

### Unit Tests (CPU, no Neuron hardware required)

```bash
cd contrib/models/glm4_moe
pip install pytest

# Router routing logic
pytest test/unit/test_router.py -v

# Partial RoPE and QK norm logic
pytest test/unit/test_attention.py -v

# Dense vs MoE layer dispatch
pytest test/unit/test_decoder.py -v

# All unit tests
pytest test/unit/ -v
```

### Integration Tests (requires Trn1/Trn2 with ≥2 NeuronCores)

```bash
cd contrib/models/glm4_moe

# Reduced config (~2 min compile), TP=2
pytest test/integration/test_model.py -v -s

# Run manually (standalone, no pytest)
python test/integration/test_model.py
```

> **Note:** On `trn2.3xlarge` (LNC=2), do not set `NEURON_RT_NUM_CORES`. The test uses `tp_degree=2` which
> maps automatically to the available NeuronCores.

The integration test:
1. Creates a tiny 2-layer random-weight model (no HuggingFace download needed)
2. Compiles it on Neuron (fast due to small model size)
3. Runs `check_accuracy_logits_v2` to compare Neuron logits against HuggingFace CPU logits

## Example Checkpoints

- [`zai-org/GLM-4.5-Air`](https://huggingface.co/zai-org/GLM-4.5-Air) — Full 70B model (128 experts, 46 layers)

## Known Limitations

- `zai-org/GLM-4.7-Flash` uses `Glm4MoeLiteForCausalLM` (different architecture, not supported)
- Flash decoding requires Trn2 for optimal performance; Trn1 falls back to standard decoding
- `e_score_correction_bias` is loaded from checkpoint as a frozen buffer (not trained during fine-tuning)

## Maintainer

Community contribution — PRs welcome.

**Last Updated:** 2026-03-06
