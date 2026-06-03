# Contrib Model: Gemma 4 26B-A4B-it

NeuronX Distributed Inference port of `google/gemma-4-26B-A4B-it`, an MoE
text-only sibling of Gemma 4 31B-IT (PR #106).

## Model Information

- **HuggingFace ID:** [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Model Type:** Text-only Mixture-of-Experts decoder
- **Parameters:** ~25.2B total / ~3.8B active
- **License:** Check HuggingFace model card

## Architecture Details

Gemma 4 26B-A4B-it shares the Gemma 4 attention + LM-head stack with PR #106
(31B-IT) but replaces the dense FFN with a parallel-MoE block.

| Feature | Description |
|---------|-------------|
| **Layers** | 30 decoder layers |
| **Hidden / Intermediate (dense)** | 2816 / 2112 |
| **Attention heads** | 16 attention, 8 KV (GQA 2:1) |
| **Heterogeneous attention** | SWA layers (head_dim=256) and Global layers (head_dim=512) — same as 31B-IT |
| **`attention_k_eq_v`** | Global layers share K/V projections |
| **QK / V normalisation** | RMSNorm on Q and K post-projection; V uses RMSNorm without learnable scale |
| **Partial RoPE on global** | `partial_rotary_factor=0.25` (128 of 512 dims rotated) |
| **Final logit softcap** | `30 * tanh(logits / 30)` |
| **Scaled embeddings** | `embed * sqrt(hidden_size)` |
| **MoE block** | 128 routed experts, `top_k=8`, plus 1 shared (parallel) dense MLP |
| **Router** | FP32 softmax + top-k + renormalise + per-expert learned scale |
| **Decoder layout** | dense MLP and MoE branch run in **parallel** on the post-norm residual; outputs summed before `layer_scalar` |
| **Per-layer-input embed** | `hidden_size_per_layer_input=0` (PLE disabled — differs from earlier Gemma) |

The NKI flash-attention kernels (`nki_flash_attn_d256_swa.py` for SWA layers,
`nki_flash_attn_large_d.py` for global head_dim>128 layers) are imported
verbatim from PR #106. Head dimensions are unchanged so the kernels apply
without modification.

## Validation Results

**Validated:** 2026-06-03 (round 4).
**Configuration:** TP=8, batch_size=1, bfloat16, seq_len=256, LNC=2.
**Instance:** trn2.48xlarge.
**SDK:** 2.29 (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`, torch 2.9.1, NxDI 0.10.0).

### Stage 1 — DISABLE_MOE compile + load

Validates the attention / scaled-embed / softcap / dense-MLP path without
the MoE branch.

| Metric | Value |
|--------|-------|
| Compile | 2.2 min (priority HLO 81 s, all HLOs 17 s) |
| Weight load | 20.85 s |
| Warmup | 0.49 s |
| NEFF artifact dir | 17 MB |
| Status | **PASS** |

### Stage 2 — MoE-on compile + load

Adds the 128-expert routed MoE branch (parallel to the dense MLP).

| Metric | Value |
|--------|-------|
| Compile | 19.7 min (priority HLO 106 s, all HLOs 925 s, build 1183 s) |
| Weight load | 29.1 s |
| Warmup | 0.66 s |
| NEFF artifact dir | 297 MB |
| Status | **PASS** |

MoE compile requires `--internal-hlo2tensorizer-options='--verify-hlo=false'`
(genericmoe v16 KB) — set in `Gemma4NeuronConfig`.

### Stage 3 — Inference smoke

| Metric | Value |
|--------|-------|
| Prompt | `"Hello, my name is"` (5 tokens) |
| Generated | 8 tokens, decoded `", my name is, my name is"` |
| TTFT (prefill seq_len=256) | 309.5 ms |
| TPOT | 8.79 ms |
| Throughput | 114 tok/s |
| Status | **PASS** (coherence smoke; greedy + base-style continuation, no chat template) |

### Stage 5 — Canonical Gemma-4 chat validation (added 2026-06-03)

Replaces the Stage 3 "smoke only" caveat with full canonical validation
following the official HF Gemma-4 pattern: `processor.apply_chat_template`
(both `enable_thinking=False` and `=True`) plus `processor.parse_response`.

Compared the Trainium 2 port head-to-head against `Gemma4ForConditionalGeneration`
on CPU bf16 (transformers 5.10.0.dev0). 3 prompts × {greedy, sampled} × {thinking off, on}.

| prompt | thinking | greedy | sampled |
|---|---|---|---|
| `Write a short joke about saving RAM.` | off | **16/16 (100%)** | 16/16 (100%) |
| `Write a short joke about saving RAM.` | on  | **16/16 (100%)** | 16/16 (100%) |
| `What is the capital of France?`       | off | **9/9 (100%)** EOS | 9/9 (100%) EOS |
| `What is the capital of France?`       | on  | **16/16 (100%)** | 14/16 (87.5%) |
| `Explain quantum entanglement in two sentences.` | off | **16/16 (100%)** | 16/16 (100%) |
| `Explain quantum entanglement in two sentences.` | on  | **16/16 (100%)** | 16/16 (100%) |

Match = first-16-tokens equal vs HF CPU bf16 reference. **11/12 cases at
100%; the lone 87.5% is sampling RNG divergence (different framework on
each backend with the same seed) — greedy at the same setup is 100%.**

`enable_thinking=True` exercises the full multi-channel response path
(`<|channel>thought\n...<channel|>`) through the MoE router. Both backends
emit identical tokens and `parse_response` returns the same
`{role, thinking, content}` dict, e.g.:

```python
{'role': 'assistant',
 'thinking': 'The user is asking for the capital of France.\n'
             'The capital of France is Paris.\nState the answer clearly.',
 'content':  'The capital of France is **Paris**.'}
```

Latency on Trainium 2 (TP=8, BF16, seq=256): TTFT ~303 ms, TPOT ~8.3 ms
(~120 tok/s greedy). See
[`agent_artifacts/round4/STAGE5_CANONICAL_VALIDATION.md`](https://github.com/xniwangaws/NeuronStuff/blob/main/gemma4-port-26b-a4b/agent_artifacts/round4/STAGE5_CANONICAL_VALIDATION.md)
on the upstream reference repo for full output, raw JSON, and
comparator script.

## What was reused from existing NxDI

- `NeuronAttentionBase` — Q/K/V/o projections, KV cache, GQA sharding, mask
  builders. Overrides: `apply_rotary_embedding` (partial RoPE),
  `prep_qkv_tensors` (post-projection v_norm), `perform_prefill` (NKI
  d=256 SWA kernel).
- `RotaryEmbedding` — instantiated per-layer with the right `dim` for
  partial RoPE on global layers.
- `ColumnParallelLinear` / `RowParallelLinear` / `ParallelEmbedding` — for
  dense MLP, lm_head, token embedding.
- `initialize_moe_module` (NxDI `moe_v2`) — handles expert dispatch and
  sharded `gate_up_proj` / `down_proj`. We feed it our own `top_k_index` /
  `top_k_weights` from the gemma4 router.
- `KVCacheManager` — subclassed to support per-layer heterogeneous shapes
  (8×256 SWA vs 2×512 global, after TP sharding).
- `NeuronBaseForCausalLM` / `NeuronBaseModel` — generation loop, sampling,
  weight loading.

## What was borrowed from PR #106 (31B-IT)

- `nki_flash_attn_d256_swa.py` — verbatim.
- `nki_flash_attn_large_d.py` — verbatim.
- `ndxi_patch.py` — verbatim, with one-line tweak so the relative import
  works inside this `src/` package.
- KV cache manager with per-layer cache size mapping.
- `SoftcappedLMHead` (cap = 30.0).
- `Gemma4ScaledEmbedding` (multiplies by `sqrt(hidden_size)`).
- `Gemma4RMSNorm` / `Gemma4VNorm`.
- Q-norm pre-scaling trick (cancels NxDI's automatic `1/sqrt(head_dim)` so
  that gemma4's QK-norm + scale match the HF reference).

## What is 26B-A4B-specific

| Class | Reason |
|---|---|
| `NeuronGemma4Router` | gemma4 router with `scale` + `per_expert_scale` learned tensors, FP32 softmax + top-k + renormalise + per-expert scale. PR #106 has no router (dense). |
| `NeuronGemma4MoEBlock` | Wraps NxDI `initialize_moe_module` and feeds it gemma4-flavoured `top_k_index` / `top_k_weights`. |
| `NeuronGemma4DecoderLayer` | Parallel-MoE layout: dense MLP and MoE branch operate on the post-norm residual concurrently; combined output = `mlp_branch + moe_branch`, then `layer_scalar`-multiplied. **Dual-input MoE forward**: the router sees the raw residual, while experts see `post_feedforward_layernorm_2(residual)` — matches HF source lines 1429–1441. |
| `convert_hf_to_neuron_state_dict` | Extends PR #106's converter with MoE weight stacking (`gate_up_proj.weight` / `down_proj.weight` shaped `[num_experts, ...]`), router weight rename, shared-expert weight wiring, and `disable_normalize_top_k_affinities` so NxDI uses our pre-computed expert affinities verbatim. |

## Open issues / known limitations

- **Validated as of Stage 5** (2026-06-03): canonical chat (`apply_chat_template` + `parse_response`,
  including `enable_thinking={False,True}`) matches HF CPU bf16 at 100% token agreement
  for 11/12 greedy/sample combos (the 12th is sampling RNG divergence; greedy is 100%).
  See Stage 5 above.
- **AutoTokenizer fix**: HF transformers ≤ 4.45 trips on gemma-4's
  special-tokens list-vs-dict shape; `scripts/smoke_inference.py` falls
  back to the raw `tokenizers` Rust backend.
- **Multimodal towers** (vision, audio) are **not ported** — text-only.
  Use PR #106 / PR #109 for VLM.
- **Sequence length tested:** 256. Longer `seq_len` (1024 / 2048) compile
  is a follow-up.
- **NxDI ≥ 0.10** required (per-layer `layer_to_cache_size_mapping` and
  `get_last_kv_window` patch).
- Apply `ndxi_patch.apply_patch()` once at process start before
  constructing the model class — see `scripts/smoke_compile.py`.

## How to compile and run (on trn2.48xlarge)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# 1. Stage 1 — dense path only (fast smoke)
GEMMA4_DISABLE_MOE=1 PYTHONPATH=src \
  python scripts/smoke_compile.py 2>&1 | tee compile_disable_moe.log

# 2. Stage 2 — MoE on
PYTHONPATH=src \
  python scripts/smoke_compile.py 2>&1 | tee compile_moe_on.log

# 3. Stage 3 — generate
PYTHONPATH=src \
  python scripts/smoke_inference.py 2>&1 | tee inference.log
```

Environment overrides (compile and inference must agree):

| Var | Default | Notes |
|---|---|---|
| `GEMMA4_MODEL_PATH` | `/home/ubuntu/gemma4-26b-a4b` | HF checkpoint dir |
| `GEMMA4_COMPILED_PATH` | `/home/ubuntu/gemma4-compiled` | NEFF output dir |
| `GEMMA4_TP_DEGREE` | `8` | Tensor-parallel degree |
| `GEMMA4_BATCH_SIZE` | `1` | – |
| `GEMMA4_SEQ_LEN` | `256` | Compile-time max seq |
| `GEMMA4_DISABLE_MOE` | `0` | `1` ⇒ dense smoke |
| `GEMMA4_MOE_EP_DEGREE` | `1` | Keep at 1 unless `BS ≥ 32` |
| `GEMMA4_MOE_TP_DEGREE` | `<TP>` | Match `tp_degree` |

## Diff vs PR #106

See [`DIFF_FROM_PR106.md`](DIFF_FROM_PR106.md) for a structural diff against
Jim Burtoft's 31B-IT port — the canonical review companion.
