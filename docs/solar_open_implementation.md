# Solar Open MoE — NXD Inference Implementation

## Overview

This document describes the implementation of Solar Open MoE (`SolarOpenForCausalLM`) inference support in `neuronx-distributed-inference`.

Solar Open is a Mixture-of-Experts language model that is **not** registered in the `transformers` library (requires `trust_remote_code`). The NXD implementation uses `GLM-4.5 MoE` as the primary template, adapted for Solar Open's unique architecture.

---

## Architecture Differences from GLM-4.5 MoE

| Feature | GLM-4.5 MoE | Solar Open |
|---------|-------------|------------|
| `partial_rotary_factor` | 0.5 (half RoPE) | 1.0 (full RoPE) |
| `attention_bias` | True | False |
| `use_qk_norm` | Configurable | False |
| `first_k_dense_replace` | N > 0 (some dense layers) | 0 (all MoE) |
| Expert weight format | Per-expert `{e}.gate_proj.weight`, `{e}.up_proj.weight`, `{e}.down_proj.weight` | Pre-fused 3D tensors: `experts.gate_up_proj [E, 2I, H]`, `experts.down_proj [E, H, I]` |
| HF registration | `transformers.Glm4MoeForCausalLM` | Not in transformers (custom) |

---

## Key Files

| File | Description |
|------|-------------|
| `src/neuronx_distributed_inference/models/solar_open/__init__.py` | Module init |
| `src/neuronx_distributed_inference/models/solar_open/modeling_solar_open.py` | Main implementation |
| `examples/generation_solar_open_demo.py` | Generation demo script |
| `test_solar_open_accuracy.py` | Accuracy test (CPU reference vs Neuron) |
| `create_solar_open_tiny_random.py` | Creates tiny random test model |
| `solar_open_tiny_random/` | Tiny random model checkpoint |
| `solar_open_tiny_random_traced/` | Compiled Neuron model |

---

## Implementation Details

### `modeling_solar_open.py`

#### Classes

- **`NeuronSolarOpenRouter`** — GroupLimitedRouter with sigmoid activation, `e_score_correction_bias`, `norm_topk_prob`, and `routed_scaling_factor`. Identical to `NeuronGlm4MoeRouter`.

- **`initialize_solar_open_moe_module`** — Creates `MoE(router, ExpertMLPsV2, SharedExperts)`. All layers are MoE (no dense branch).

- **`NeuronSolarOpenAttention`** — Full RoPE (`rotary_dim = head_dim`), no bias, no QK norm.

- **`NeuronSolarOpenDecoderLayer`** — Always MoE (no `is_moe_layer` check needed).

- **`NeuronSolarOpenModel`** — Standard `NeuronBaseModel` with `ParallelEmbedding`, decoder layers, `RMSNorm`, and `lm_head`.

- **`NeuronSolarOpenForCausalLM`** — `NeuronBaseForCausalLM` wrapper. `load_hf_model` loads safetensors directly (not via AutoConfig).

- **`SolarOpenInferenceConfig`** — Extends `InferenceConfig`:
  - Sets `num_local_experts = n_routed_experts`
  - Overrides `intermediate_size = moe_intermediate_size` (used by ExpertMLPsV2)
  - Sets `output_attentions = False`, `output_hidden_states = False`, `is_encoder_decoder = False` (transformers defaults)
  - FP32 router, `normalize_top_k_affinities = False`

- **`load_solar_open_config`** — Custom config loader that reads `config.json` directly (bypasses `AutoConfig.from_pretrained`). Sets `_name_or_path` so `checkpoint_loader_fn` can find safetensors.

#### State Dict Conversion

The critical difference from GLM-4.5:

```
HF Solar Open:
  mlp.experts.gate_up_proj  [E, 2*I, H]   ← 3D pre-fused, NO .weight suffix
  mlp.experts.down_proj      [E, H, I]     ← 3D pre-fused, NO .weight suffix

NXD target:
  mlp.expert_mlps.mlp_op.gate_up_proj.weight  [E, H, 2*I]   ← permute(0,2,1)
  mlp.expert_mlps.mlp_op.down_proj.weight      [E, I, H]     ← permute(0,2,1)
```

**Conversion**: just `permute(0, 2, 1)` — no expert-loop fusion needed.

---

## Config Loader Pattern

Because `solar_open` is not registered in transformers, `AutoConfig.from_pretrained` fails. The solution:

```python
config = SolarOpenInferenceConfig(
    neuron_config,
    load_config=load_solar_open_config(model_path),
)
```

`load_solar_open_config` reads `config.json` directly and sets all required attributes.

---

## Tiny Random Test Model

Created by `create_solar_open_tiny_random.py`:

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 32 |
| `num_hidden_layers` | 2 |
| `num_attention_heads` | 4 |
| `num_key_value_heads` | 2 |
| `head_dim` | 8 |
| `moe_intermediate_size` | 8 |
| `n_routed_experts` | 8 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 4 |
| `vocab_size` | 196608 |
| Total parameters | 12,603,568 |

---

## Neuron Compilation

Compiled with `tp_degree=2`, `moe_tp_degree=1`, `moe_ep_degree=1`, `seq_len=64`, `max_context_length=48`, `bfloat16`, greedy decoding (`top_k=1`).

Compiler flags:
```
--enable-saturate-infinity --enable-mixed-precision-accumulation
--model-type transformer -O1
--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'
--auto-cast=none
--internal-enable-dge-levels vector_dynamic_offsets
--internal-hlo2tensorizer-options='--verify-hlo=true'
```

---

## Notes

1. **Safetensors must be copied to traced path**: When loading with `model.load(traced_model_path)`, the checkpoint loader looks for safetensors. Copy `model.safetensors` from original to traced path (the demo does this automatically during compile).

2. **e_score_correction_bias dtype**: Saved as `float32` in the checkpoint, auto-converted to `bfloat16` on load (warning is expected).

3. **Redundant keys removed**: `o_proj.weight` and `Wqkv.weight` appear in the trace's weight removal list — this is expected behavior from the neuronx weight sharding.
