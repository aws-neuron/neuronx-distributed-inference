# GLM-4.5 MoE Implementation for NeuronX Distributed Inference

## Overview

This document describes the implementation of GLM-4.5 MoE (Mixture of Experts) model inference
on AWS Neuron hardware using the `neuronx-distributed-inference` package.

## Architecture

### GLM-4.5 MoE vs Qwen3MoE differences

| Feature | Qwen3MoE | GLM-4.5 MoE |
|---------|----------|-------------|
| RoPE | Full head_dim | Partial: first `head_dim × partial_rotary_factor` dims |
| Attention bias | No | Yes (Q/K/V projections have bias) |
| QK normalization | No | Yes (`use_qk_norm=True`) |
| Dense layers | All MoE | First `first_k_dense_replace` layers use dense MLP |
| Shared experts | No | Yes (`n_shared_experts=1`) |
| Router activation | Softmax | Sigmoid |
| Router correction | None | `e_score_correction_bias` per-expert |
| Routing scale | 1.0 | `routed_scaling_factor=2.5` |

### Key Configuration Fields (glm4_moe_tiny_random)

```
hidden_size: 16
num_hidden_layers: 2
num_attention_heads: 4
num_key_value_heads: 2
head_dim: 64
partial_rotary_factor: 0.5   -> rotary_dim = 32
attention_bias: true
use_qk_norm: true
first_k_dense_replace: 1     -> layer 0 is dense, layer 1 is MoE
n_routed_experts: 16
n_shared_experts: 1
num_experts_per_tok: 2
moe_intermediate_size: 64
intermediate_size: 128       -> dense layer intermediate size
norm_topk_prob: true
routed_scaling_factor: 2.5
n_group: 1
topk_group: 1
```

## Implementation Files

### `src/neuronx_distributed_inference/models/glm4_moe/modeling_glm4_moe.py`

Main model implementation. Key components:

#### `NeuronGlm4MoeRouter(GroupLimitedRouter)`
Custom router with GLM-4.5-specific logic:
- Registers `e_score_correction_bias` buffer (per-expert, FP32)
- `noaux_tc_top_k()`: group-limited top-k with bias-corrected selection, original-score weighting, normalization, and scaling
- Returns `(router_logits, full_affinities, topk_idx)` with pre-scaled weights

#### `NeuronGlm4MoeAttention(NeuronAttentionBase)`
- Partial RoPE: creates `RotaryEmbedding(dim=rotary_dim)` where `rotary_dim = head_dim × partial_rotary_factor`
- Overrides `apply_rotary_embedding()` to split Q/K, apply RoPE to first `rotary_dim` dims, concatenate back
- Optional QK norm via `q_layernorm` and `k_layernorm` (RMSNorm per head)
- QKV bias support via `qkv_bias=True`

#### `NeuronGlm4MoeDenseMLP`
Standard GLU MLP for dense layers (layers 0..first_k_dense_replace-1):
- `gate_proj` + `up_proj` via `ColumnParallelLinear`
- `down_proj` via `RowParallelLinear`
- Activation: SiLU

#### `NeuronGlm4MoeDecoderLayer`
- Branches per `is_moe_layer = layer_idx >= first_k_dense_replace`
- Dense layers: `NeuronGlm4MoeDenseMLP`
- MoE layers: `MoE(router=NeuronGlm4MoeRouter, expert_mlps=ExpertMLPsV2, shared_experts=SharedExperts)`

#### `Glm4MoeInferenceConfig(InferenceConfig)`
- Sets `num_local_experts = n_routed_experts`
- Saves `dense_intermediate_size` (original `intermediate_size`)
- Sets `intermediate_size = moe_intermediate_size` for MoE/shared experts
- Configures router in FP32
- Disables normalize_top_k_affinities (handled by router)

#### `convert_glm4_moe_hf_to_neuron_state_dict()`
State dict conversion:
1. Adds `rank_util.rank` and per-layer `self_attn.rank_util.rank` tensors
2. Renames `q_norm.weight` → `q_layernorm.weight`, `k_norm.weight` → `k_layernorm.weight`
3. Fuses `{q,k,v}_proj.{weight,bias}` → `Wqkv.{weight,bias}`
4. For MoE layers:
   - `mlp.gate.weight` → `mlp.router.linear_router.weight`
   - `mlp.gate.e_score_correction_bias` → `mlp.router.e_score_correction_bias`
   - Per-expert `gate_proj[I,H].T` + `up_proj[I,H].T` → fused `gate_up_proj[E,H,2I]`
   - Per-expert `down_proj[H,I].T` → `down_proj[E,I,H]`
   - Shared expert keys pass through as-is

## Bug Fix: `hf_adapter.py`

### Problem
`hf_adapter.py` referenced `tensor_capture_hook` variable that was never defined (NameError),
and also unconditionally passed `tensor_capture_hook` to `model.forward()` which only accepts
it for multimodal models.

### Fix
1. Added `tensor_capture_hook = kwargs.get("tensor_capture_hook", None)` to resolve the NameError
2. Removed unconditional `"tensor_capture_hook": tensor_capture_hook` from `model_inputs`
3. Added `import inspect`
4. Added conditional pass-through: only include `tensor_capture_hook` in model_inputs if the
   model's `forward()` signature accepts it (for multimodal models like Pixtral)

```python
if tensor_capture_hook is not None:
    fwd_params = inspect.signature(self.neuron_model.forward).parameters
    if "tensor_capture_hook" in fwd_params:
        model_inputs["tensor_capture_hook"] = tensor_capture_hook
```
