# MiMo-V2.5-Pro on Trn2 — Long-Context & Small-Batch Optimization Design

> **STATUS: partly superseded — read this first.**
>
> This was the *planning* document, written before any code existed. Phase 1
> (SWA cache truncation) was subsequently implemented as
> `MIMO_SWA_KV_TRUNCATION`, and measurement changed two of the conclusions
> below. What actually held up:
>
> - **Phase 1 works as designed.** Truncating the 60 SWA layers to 128 slots cut
>   KV at `seq_len=4096` from 21.1 GB to 3.6 GB per rank; 4K compiles and loads
>   at ~20.6/24 GB. The mechanism analysis in §2/M1 is accurate.
> - **The HBM framing in §1 is no longer the binding constraint.** The claim that
>   `seq_len=1024` OOMs came from a pre-BF16-attn measurement and does not
>   reproduce; 1024 loads on the stock recipe.
> - **The real blocker is output quality, which this doc did not anticipate.**
>   Prompts past ~480 tokens degrade — first answering fluently but wrongly
>   (~520), then collapsing into repetition (≥568) — independently of `seq_len`
>   and of the truncation flag, on a graph proven structurally identical to the
>   working 512 recipe. So making longer context *fit* does not make it *usable*.
>   See "Long context and output degeneration" in README.md for the measured
>   numbers. Everything below about reaching 8K/128K is therefore necessary but
>   not sufficient.
> - **Phase 2 (DP attention) is the wrong lever for Pro**, for a reason §2/M2
>   underweighted: it replicates attention *weights* per DP group (+4.1 GB/rank
>   at dp=8) and Pro's weights already sit near the limit. It overflows where
>   cache truncation fits.
>
> Kept for the mechanism survey and file:line references, which remain useful for
> any future 128K attempt.

**Goal (user requirements):**
1. Raise `seq_len` from the current **512** to **≥8K, ideally 128K**.
2. Allow **batch_size < 48** (currently forced to 48), so a smaller batch can buy longer context.

This doc maps the root causes to concrete NxDI mechanisms that **already exist**
and are used by other models (Llama4, gpt_oss, qwen3-moe), with file:line
evidence, then proposes a phased plan. No code is changed yet.

---

## 1. Why seq_len is stuck at 512 and BS is stuck at 48

Three independent multipliers blow up HBM, all traceable to the same choice
("use all 64 cores with TP=64 while the model has only 8 KV heads and 384
experts"):

### 1a. CONVERT_TO_MHA replicates KV heads 8→64 (8× KV cache)
MiMo has `num_key_value_heads=8` but we run `tp_degree=64`. The contrib model
forces MHA conversion:
- `contrib/.../modeling_mimo_v2.py:395` `self.use_gqa_convert_to_mha = tp_degree > self.attn_num_kv_heads` (64>8 → True)
- `:399` `self._kv_replication_factor = self.attn_num_heads // self.attn_num_kv_heads` (128/8 = **16×** at the weight level; per-rank the cache ends up with 64 KV heads vs the natural 8, i.e. **8× more KV cache** than a batch-parallel layout would need)
- `:415-416` every projection binds to `parallel_state.get_tensor_model_parallel_group()` — the **full** TP=64 group, so this is hard-wired.

### 1b. V is padded 128→192 in the cache (1.5× on V)
- `contrib/.../modeling_mimo_v2.py:644-647` pads `value_states_for_cache` from `v_head_dim=128` to `head_dim=192` before storing. So the V half of the cache carries 50% dead padding.

### 1c. All 70 layers store the full sequence — the SWA layers waste 60/70 of it
MiMo is **hybrid attention**: 10 full-attention layers + **60 sliding-window
layers with `sliding_window=128`**. A sliding-window layer only needs the last
128 tokens of KV, but MiMo allocates a full-`seq_len` cache for every layer:
- installed `models/mimo_v2/modeling_mimo_v2.py:996-999`:
  ```
  # NOTE: Do NOT set self.sliding_window here because it affects KV cache size globally.
  # Setting has_mixed_attn = True enables proper mask creation without affecting cache size.
  self.has_mixed_attn = True
  ```
  → `self.sliding_window=None`, `self.layer_to_cache_size_mapping=None` (defaults
  `model_base.py:116-117`), so it falls into the plain `KVCacheManager` else-branch
  (`model_base.py:190`) and **every layer gets a full `max_length` cache**.

### 1d. BS ≥ 48 is required by moe_ep_degree > 1 at decode
- README/config: `BS * top_k / num_experts >= 1.0` when `moe_ep_degree>1` →
  `BS >= 384/8 = 48`. This is the expert-parallel decode requirement that every
  EP rank receive ≥1 token for a well-formed all-to-all (see `moe`/`moe_v2`).

### Quantified HBM (real 8 KV heads, K+V, bf16, V padded to 192)

| context | BS | all 70 layers full | **SWA truncated** (10 full @ seq + 60 @ 128) |
|--------:|---:|-------------------:|---------------------------------------------:|
| 512     | 48 | 10.6 GB            | —                                            |
| 8K      | 48 | 169 GB             | **26 GB**                                    |
| 8K      | 8  | 28 GB              | **4.4 GB**                                   |
| 128K    | 48 | 2706 GB            | **389 GB**                                   |
| 128K    | 8  | 451 GB             | **65 GB**                                    |
| 128K    | 1  | 56 GB              | **8 GB**                                     |

(These are the *natural* 8-KV-head numbers. Today's CONVERT_TO_MHA multiplies
the attention side by another ~8×.) HBM budget: 64×24 = 1536 GB total, ~1 TB
consumed by weights, leaving ~300–400 GB for KV.

**Reading the table against the goals:**
- **8K** is reachable *today's cache layout* just by lowering BS (BS=8 → 28 GB).
  The blocker for 8K is really the **BS≥48** rule (1d), not the cache design.
- **128K** is impossible with the full-every-layer layout at any batch that also
  keeps EP=64 (2706 GB @ BS48, 451 GB @ BS8). It only becomes feasible with
  **SWA cache truncation** (1c): 389 GB @ BS48, 65 GB @ BS8, 8 GB @ BS1.
  → For 128K, SWA truncation is **necessary**, not optional.

---

## 2. The mechanisms already exist in NxDI (with model precedents)

### M1. Sliding-window / hybrid per-layer KV cache — **strong precedent**
The contiguous `KVCacheManager` supports heterogeneous per-layer cache lengths
via `layer_to_cache_size_mapping`:
- `modules/kvcache/kv_cache_manager.py:217-228` builds per-layer `k_shapes`/`v_shapes` from the mapping.
- helper `modules/kvcache/utils.py:507-516` `get_layer_to_kv_cache_size_mapping_for_mixed_attn(local, global, is_layer_locals)`.
- **Llama4** uses exactly this: `models/llama4/modeling_llama4_text.py:626` builds the mapping (chunk-size for local layers, seq_len for global) and passes it to `KVCacheManager` (`:650`).
- **gpt_oss** has a bespoke `GptOssKVCacheManager` (`modules/kvcache/gpt_oss_kv_cache_manager.py:88-116`) that stores `sliding_window` tokens for SWA layers and `max_length` for full layers — and even a separate DP degree for SWA layers.

MiMo would need to set `layer_to_cache_size_mapping` (128 for the 60 SWA layers,
seq_len for the 10 full layers) and pass it to the KV manager — instead of the
current `sliding_window=None` mask-only approach.

### M2. Data-parallel attention (attention TP < global TP, no KV replication) — **strong precedent**
- Config fields (`config.py:363-367`): `cp_degree`, `attention_dp_degree`, plus MoE's `moe_tp_degree`/`moe_ep_degree` — attention parallelism is **decoupled** from MoE parallelism.
- With `attention_dp_degree=8`, decode attention runs at TP=64/8=**8** (= the 8 KV heads → **no CONVERT_TO_MHA**), and KV cache batch shrinks: `config.py:514-515` `kv_cache_batch_size = tkg_batch_size // attention_dp_degree`. A `DataParallelKVCacheManager` slices the cache per DP rank (`model_base.py:185-186`).
- `NeuronAttentionBase` already consumes `get_data_parallel_attention_tp_group()` for the decode projections (`attention_base.py:188-190, 385-393`); CTE/prefill stays full-TP (`:389-391`) with no `cp_degree` required.
- **Working reference config** (same TP=64 class, MoE): `test/integration/tp64/models/qwen3moe/neuron_configs/bs16_sl10k_optimized.json` uses `tp_degree=64, moe_tp_degree=2, moe_ep_degree=32, attention_dp_degree=8, cp_degree=16, sequence_parallel_enabled, strided_context_parallel_kernel_enabled, ...` — i.e. **BS=16, seq 10K** already validated on tp64 with DP attention + CP + EP.

**Caveat:** `attention_dp_degree` is *batch* data parallelism (decode-only), and
its `DataParallelKVCacheManager` path does **not** plumb
`layer_to_cache_size_mapping` (`model_base.py:185-186` vs `:190`). So **M1 and
M2 are currently mutually exclusive** in the stock code — combining them (DP
attention *and* per-layer SWA cache) would need new work.

**Caveat 2:** MiMo's custom attention overrides `init_gqa_properties` to a no-op
(`contrib/.../modeling_mimo_v2.py:372-382`) and hand-rolls projections on the
full TP group, precisely because its Q/K head_dim=192 ≠ V head_dim=128 breaks
the base's fused GQA QKV. So MiMo does **not** currently consume the DP-attention
group even though the base class supports it. Wiring M2 into MiMo means
re-implementing the CTE/TKG projection split for its asymmetric head dims.

### M3. Context parallelism for prefill — **already partly wired in MiMo**
- MiMo's attention has a CP forward path: `contrib/.../modeling_mimo_v2.py:532-624` splits Q/K/V + mask along the sequence dim when `cp_degree>1` (prefill only). This shards the *prefill* sequence across ranks (helps prefill latency / prefill activation memory at long context) but does **not** shrink the decode KV cache.

### M4. Lower BS by changing MoE parallelism
The BS≥48 rule is tied to `moe_ep_degree>1`. Options to break it (need
validation): raise `moe_tp_degree` and lower `moe_ep_degree` (fewer EP ranks →
lower the `BS ≥ num_experts/top_k`… actually the bound is on top_k/num_experts,
independent of ep_degree — see open question Q1 below), use `capacity_factor` /
token-dropping, or a non-EP MoE kernel. The qwen3moe reference above runs
`moe_ep_degree=32` at **BS=16**, which suggests the "BS≥48" bound is **not**
fundamental to EP>1 in general and may be specific to MiMo's
`moe_ep_degree=64`/kernel choice. **This is the highest-value thing to verify
first** (see Phase 0).

---

## 3. Open questions to resolve before coding

- **Q1 (blocking for goal 2) — RESOLVED as "unimplemented case, not a hard
  limit":** The `NotImplementedError: Selective Loading with Expert parallelism`
  lives in the base `neuronx_distributed` library's `ExpertMLPsV2.forward_selective_loading`
  (not vendored in this repo). The gate is exactly "average tokens per expert
  = `BS*top_k/num_experts` ≥ 1"; below it, some EP ranks get 0 tokens and that
  decode path simply isn't written — it raises rather than asserting a hardware
  impossibility. So a zero-token-rank (capacity/padding) scheme *could* handle
  BS<48, but the selective-loading TKG kernel doesn't. **In-repo, the only
  documented way to run BS<48 while sharded across 64 cores is
  `moe_ep_degree=1 + moe_tp_degree=64`** (README:335) — but for MiMo
  `moe_tp=64` shrinks the per-rank MoE intermediate (2048/64 = 32 rows) below
  the 128-row FP8 blockwise scale block, collapsing per-channel scale
  (README:325-331), so it needs an **all-BF16 MoE checkpoint**. `capacity_factor`
  is a wired config field (`config.py:806,810` → `moe.py:31`, the v1
  capacity/drop-token wrapper) that *might* bypass the gate, but the deciding
  code is in the base library and unverifiable from here. **Action:** either
  test `capacity_factor` + the v1 `moe.py` path at BS<48, or accept
  `moe_ep=1/moe_tp=64` + a BF16 MoE checkpoint for low-BS long-context.
- **Q2:** Can `layer_to_cache_size_mapping` (M1) coexist with `fused_qkv=False`
  and MiMo's asymmetric head dims + attention-sink bias? Llama4/gpt_oss don't
  have asymmetric head dims, so the cache-shape helpers may assume symmetric
  head_dim.
- **Q3:** Does SWA truncation interact correctly with the attention-sink bias
  (MiMo adds a learnable sink column on SWA layers) and the V-pad-to-192
  workaround?

---

## 4. Phased plan (lowest risk / highest leverage first)

**Phase 0 — free wins, no modeling change (validate first):**
- Test **BS=8 or 16** with the *current* layout at seq_len 1024/2048. If the
  BS≥48 rule can be relaxed (Q1), 8K may already fit (28 GB @ BS8, full layers).
  This directly serves goal 2 and partially goal 1.
- Confirm the seq1024 NEFF (already compiling) actually loads — the README
  "1024 OOMs" note predates the BF16-attn recipe and may be stale.

**Phase 1 — SWA cache truncation (goal 1, biggest single lever for 128K):**
- Two proven in-tree implementations to copy:
  - **Llama4 style** — build `layer_to_cache_size_mapping` = [seq_len for the 10
    full layers, 128 for the 60 SWA layers] and pass it to the contiguous
    `KVCacheManager` (`modeling_llama4_text.py:626,650`; allocation
    `kv_cache_manager.py:217-228`; helper `kvcache/utils.py:507-516`
    `get_layer_to_kv_cache_size_mapping_for_mixed_attn`).
  - **gpt_oss style** — a bespoke per-layer manager `GptOssKVCacheManager`
    (`gpt_oss_kv_cache_manager.py:97-116`) that allocates
    `get_kernel_cache_size_bucket(sliding_window)` for SWA layers and
    `max_length` for full layers, and even carries a separate DP degree for SWA
    layers (relevant to combining with Phase 2).
  MiMo today sets `has_mixed_attn=True` but leaves `sliding_window=None` /
  `layer_to_cache_size_mapping=None`, so all 70 layers get full caches
  (`models/mimo_v2/modeling_mimo_v2.py:996-999`). The change is to populate the
  mapping. Cuts 128K KV from 2706→389 GB (BS48) / 451→65 GB (BS8).
- Resolve Q2/Q3 (asymmetric head dim 192/128 + sink bias under per-layer cache;
  Llama4/gpt_oss have symmetric head_dim, so the cache-shape helpers may need
  the V-pad-to-192 workaround threaded through).

**Phase 2 — DP attention to kill the 8× KV replication (goal 1+2):**
- Wire `attention_dp_degree` into MiMo's custom `_init_projections` so decode
  attention uses `get_data_parallel_attention_tp_group()` (TP=8 = 8 KV heads,
  zero replication) while MoE stays EP=64. Mirror the base's CTE(full-TP)/
  TKG(DP) split for asymmetric head dims. Reference config: qwen3moe
  bs16_sl10k_optimized.json.
- Note M1+M2 are mutually exclusive in stock code; combining SWA truncation with
  DP attention is net-new work (the DP KV manager would need per-layer sizing).

**Phase 3 — combine + push to 128K:**
- Full-layer KV at 128K even after SWA truncation is the remaining cost (10
  layers). Add CP (M3) for prefill of very long inputs, and pick the BS that
  fits (BS=1–8 for 128K).

---

## 4b. Iterate on MiMo-V2.5 (PR #148), not Pro, for faster turnaround

Pro is slow to compile/load/warmup (70 layers, 384 experts, ~200 s load, long
DeepGEMM warmup). **MiMo-V2.5** (PR #148, `contrib/MiMo-V2.5`, also whn09) is
architecturally the same family and every optimization here applies identically,
but it is smaller and iterates faster. Verified V2.5 config (from
`s3://datalab/xiaomi/models/MiMo-V2.5-Neuron-FP8/config.json`):

| param | MiMo-V2.5 | MiMo-V2.5-Pro |
|-------|-----------|---------------|
| layers | **48** (16 full + 32 SWA) | 70 (10 full + 60 SWA) |
| num_key_value_heads (full / SWA) | **4 / 8** | 8 / 8 |
| num_attention_heads | 64 | 128 |
| experts / top_k | 256 / 8 | 384 / 8 |
| BS≥ bound (num_experts/top_k) | **32** | 48 |
| sliding_window | 128 | 128 |
| head_dim / v_head_dim | 192 / 128 | 192 / 128 |

Same three problems apply: TP=64 > KV heads (4/8) → CONVERT_TO_MHA (replication
factor 64/4 = **16×** on full layers, even worse than Pro); 32 SWA layers store
full seq_len; BS forced to ≥32. So V2.5 is a faithful, faster testbed — validate
Phase 0/1/2 on V2.5 first, then port the identical changes to Pro. V2.5 also
already runs on vLLM without Pro's issue-#31 garble, making end-to-end
correctness checks easier.

## 5. Summary

| Goal | Blocker | Mechanism (exists) | Precedent | Phase |
|------|---------|--------------------|-----------|-------|
| BS < 48 | `moe_ep_degree>1` decode bound | lower moe_ep / raise moe_tp / capacity_factor | qwen3moe tp64 @ BS16 | 0 |
| 8K seq | mostly BS≥48 | lower BS, current cache | qwen3moe sl10k | 0–1 |
| 128K seq | 70 full-length layers | `layer_to_cache_size_mapping` SWA truncation | Llama4, gpt_oss | 1 |
| kill 8× KV replication | MiMo hardcodes full TP + CONVERT_TO_MHA | `attention_dp_degree` | qwen3moe DP=8 | 2 |

Everything needed is present in NxDI and proven on other models; the work is
adapting MiMo's *custom* attention/KV code (which currently bypasses these
mechanisms because of its asymmetric head dims) to consume them.
