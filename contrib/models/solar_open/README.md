# Contrib Model: Solar Open 100B MoE

NeuronX Distributed Inference implementation of [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B), a 100B Mixture-of-Experts language model.

## Model Information

- **HuggingFace ID:** `upstage/Solar-Open-100B`
- **Model Type:** Decoder-only MoE transformer
- **Architecture:** 64 routed experts + 1 shared expert per layer, top-2 routing
- **Parameters:** ~100B total, ~22B active per token
- **License:** Check HuggingFace model card

> **Note:** Solar Open is **not** available in the `transformers` library. The model config and weights are loaded directly from the HuggingFace checkpoint using custom loaders (`load_solar_open_config`).

## Architecture Details

Solar Open shares the same MoE routing architecture as GLM-4.5 MoE, with the following key differences:

| Property | Solar Open | GLM-4.5 MoE |
|----------|-----------|-------------|
| `partial_rotary_factor` | 1.0 (full RoPE) | < 1.0 (partial RoPE) |
| `attention_bias` | False | True |
| `use_qk_norm` | False | True |
| `first_k_dense_replace` | **0** (ALL layers MoE) | > 0 (some dense layers) |
| `rope_scaling` | None or `yarn` | None |
| In `transformers` | ❌ No | ✅ Yes |

### MoE Configuration (100B model)

- `n_routed_experts`: 64
- `n_shared_experts`: 1
- `num_experts_per_tok`: 2 (top-2 routing)
- `n_group`: 8, `topk_group`: 2
- `norm_topk_prob`: True
- `routed_scaling_factor`: 1.0
- Router: sigmoid + group-limited routing + `e_score_correction_bias`

### Expert Parallelism Limitation

> ⚠️ **EP (Expert Parallelism) is currently limited to `moe_ep_degree=1`** due to a known issue with the MoE EP group initialization when `n_group > 1`. Use TP-only parallelism for now.

Recommended production config: `tp_degree=32, moe_tp_degree=4, moe_ep_degree=8` (requires trn2.48xlarge or equivalent).

## Hardware Requirements

| Configuration | Instance |
|--------------|----------|
| Development / testing | trn1.32xlarge (32 NeuronCores) |
| Production (100B, seq_len=65536) | trn2.48xlarge (128 NeuronCores) |

## Usage

```python
import sys
sys.path.insert(0, "contrib/models/solar_open/src")

import torch
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from solar_open.modeling_solar_open import (
    SolarOpenInferenceConfig,
    NeuronSolarOpenForCausalLM,
    load_solar_open_config,
)

model_path = "/path/to/upstage/Solar-Open-100B"
traced_model_path = "/path/to/traced_model"

neuron_config = MoENeuronConfig(
    tp_degree=32,
    moe_tp_degree=4,
    moe_ep_degree=8,
    batch_size=4,
    seq_len=65536,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.6, top_k=20, top_p=0.95
    ),
    fused_qkv=True,
    qkv_kernel_enabled=True,
    attn_kernel_enabled=True,
)

config = SolarOpenInferenceConfig(
    neuron_config,
    load_config=load_solar_open_config(model_path),
)

# Compile
model = NeuronSolarOpenForCausalLM(model_path, config)
model.compile(traced_model_path)

# Load and run
model = NeuronSolarOpenForCausalLM(traced_model_path)
model.load(traced_model_path)
```

See `examples/generation_solar_open_demo.py` for a full end-to-end example, or `../../examples/generation_solar_open.py` for the production benchmark script.

## Testing

### Unit Tests (CPU, no Neuron hardware required)

```bash
cd contrib/models/solar_open
source /path/to/neuronx_venv/bin/activate
python -m pytest test/unit/ -v
```

### Integration Tests (requires Neuron hardware)

```bash
cd contrib/models/solar_open
python -m pytest test/integration/ -v --capture=tee-sys
```

Integration tests compile a 2-layer tiny random model and verify:
1. **Smoke test** — model compiles and loads without error
2. **Output shape** — generated token IDs have correct shape
3. **Determinism** — same input produces same output across runs

## Compatibility Matrix

| Instance | NxDI Version | Status |
|----------|-------------|--------|
| trn1.32xlarge | 2.20+ | ✅ Validated (unit tests) |
| trn2.48xlarge | 2.20+ | 🔧 Integration pending |
| Inf2 | Any | Not tested |

## Maintainer

Contributed by: gmkim (lifelongeeek)

**Last Updated:** 2026-03-06
