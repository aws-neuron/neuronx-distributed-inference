# Diff vs PR #106 (gemma-4-31B-IT)

This port shares the Gemma 4 attention / norm / softcap / RoPE machinery
with [PR #106](https://github.com/aws-neuron/neuronx-distributed-inference/pull/106)
(Jim Burtoft, gemma-4-31B-IT). The intent of this diff is to make review
easy by listing exactly what is **identical**, what is **adapted**, and
what is **new for the 26B-A4B MoE variant**.

## Summary

| Category | File | Status |
|---|---|---|
| NKI sliding-window flash attention (head_dim=256) | `src/nki_flash_attn_d256_swa.py` | **Identical** to PR #106 |
| NKI flash attention for head_dim>128 | `src/nki_flash_attn_large_d.py` | **Identical** to PR #106 |
| NxDI runtime patches | `src/ndxi_patch.py` | **PR #106 + 1-line relative-import fix** |
| Modeling | `src/modeling_gemma4_neuron.py` | **Adapted** (text-only; adds MoE block + router) |
| Configuration shim | `src/configuration_gemma4_neuron.py` | New (was inline in PR #106) |
| Vision / VLM | – | **Not ported** (text-only) |

## File-by-file

### `src/nki_flash_attn_d256_swa.py`, `src/nki_flash_attn_large_d.py`

Verbatim copies of PR #106 kernels. Head dimensions on the 26B-A4B variant
match the 31B-IT (SWA layers head_dim=256, global head_dim=512, GQA 2:1)
so no kernel changes are required.

### `src/ndxi_patch.py`

Imports the NKI flash-attention kernel through a relative import so the
patch module is self-contained inside this port directory:

```python
# Prefer relative import when this module ships inside the src/ package.
from .nki_flash_attn_large_d import flash_attn_large_d
```

Behaviour is otherwise unchanged from PR #106.

### `src/modeling_gemma4_neuron.py`

**Reused 1:1 from PR #106 (renamed only):**

- `Gemma4RMSNorm`, `Gemma4VNorm` — RMSNorm flavours.
- `Gemma4ScaledEmbedding` — `embed * sqrt(hidden_size)`.
- `SoftcappedLMHead` — `cap * tanh(x / cap)` with `cap=30.0` in fp32.
- `Gemma4KVCacheManager` — per-layer heterogeneous KV shapes.
- `NeuronGemma4Attention` — partial RoPE for global, K=V at weight level,
  NKI d=256 SWA prefill, post-projection v_norm.
- Q-norm pre-scaling trick in the state-dict converter (cancels NxDI's
  automatic `1/sqrt(head_dim)`).

**26B-A4B-specific additions:**

- `NeuronGemma4Router` — FP32 softmax + top-k + renormalise + per-expert
  learned scale. Reads `scale` and `per_expert_scale` learned tensors.
- `NeuronGemma4MoEBlock` — thin wrapper around NxDI `initialize_moe_module`
  that consumes the gemma4 router's `top_k_index` / `top_k_weights`.
- `NeuronGemma4DecoderLayer` — **parallel-MoE layout**:
  - dense MLP and MoE branch run on the **post-norm residual** in
    parallel (HF source lines 1429–1441).
  - `mlp_branch + moe_branch` ⇒ `layer_scalar`-multiplied final residual.
  - **Dual-input MoE forward**: the router sees the *raw* residual while
    the experts see `post_feedforward_layernorm_2(residual)`. Necessary
    to match the HF reference; the two pre-norm streams differ.
- `convert_hf_to_neuron_state_dict` — extended for MoE:
  - Stacks per-expert `gate_up_proj.weight` and `down_proj.weight` to
    shape `[num_experts, ...]` for `moe_v2`.
  - Renames the gemma4 router weight (`gating.weight` ⇒
    `router.weight`).
  - Wires the shared-expert weights through the dense MLP path
    (`shared_experts.{gate,up,down}_proj` ⇒ `mlp.{gate,up,down}_proj`).
  - Pre-scales `q_layernorm.weight` by `sqrt(head_dim)` (PR #106's
    trick, kept for parity).

**Config knobs that differ from a stock NxDI MoE:**

- `disable_normalize_top_k_affinities=True` — gemma4 already renormalises
  + applies `per_expert_scale` inside the custom router; we want NxDI to
  consume our affinities verbatim.
- `router_dtype="float32"`, `router_act_fn="softmax"` — match HF
  reference; underlying NxDI `RouterConfig` reads these for typing.
- `glu_mlp=True`, `glu_type="glu"` — gemma4 expert MLP is gated.

### `src/configuration_gemma4_neuron.py`

Lightweight HF-style config dataclass split out for static parsing.
PR #106 keeps its config inline. Splitting it lets external tools read
`hidden_size` / `num_experts` / `top_k` without importing NxDI.

### Test layout

`test/integration/test_model.py` mirrors PR #106's layout but is reduced
to a Stage 1 / Stage 2 / Stage 3 smoke runner (compile dense, compile
MoE, generate ≤ 8 tokens). Token-match accuracy is a follow-up.

## What is **not** in this PR (deferred)

- Vision / audio towers — text-only port. Use PR #106 / #109 for VLM.
- Token-match accuracy validation vs HF reference (sampling, chat
  template, longer prompts).
- `seq_len > 256` — round 4 only validated 256. Longer sequence compile
  is a follow-up.
- vLLM serving notebook (PR #106 has one).
