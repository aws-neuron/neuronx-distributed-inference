# Qwen3.6 27B Full FP8 — Issues Encountered and Fixes

Consolidated catalog of every issue hit during the full-FP8 / 256K hybrid-APC
work and how each one was resolved. Branch: `codex/full-fp8-qwen36`.

Source-of-truth detail (with exact log lines, PIDs, artifact paths) lives in:

- [QWEN36_FP8_TIERFIX_VALIDATION_20260526.md](./QWEN36_FP8_TIERFIX_VALIDATION_20260526.md) — full chronological log
- [HYBRID_APC_PRODUCTION_PLAN.md](./HYBRID_APC_PRODUCTION_PLAN.md) — production bucket strategy
- [profile_artifacts/qwen36_256k_fp8_sparse_runtime_20260525/ERROR_LOG.md](../../../../profile_artifacts/qwen36_256k_fp8_sparse_runtime_20260525/ERROR_LOG.md) — runtime load failures
- [AGENTS.md](../../../../AGENTS.md) — error-logging contract and measurement rules

This document is the index. Each entry: **what broke → why → what we changed → verification**.

---

## Table of Contents

1. [Quantization & Checkpoint Conversion](#1-quantization--checkpoint-conversion)
2. [Neuron Compiler Failures](#2-neuron-compiler-failures-neuronx-cc)
3. [vLLM / Hybrid APC / Scheduler](#3-vllm--hybrid-apc--scheduler)
4. [Runtime Load & Memory (NRT_RESOURCE / scratchpad / HBM)](#4-runtime-load--memory-nrt_resource--scratchpad--hbm)
5. [Custom NKI Kernel (`qwen_segcte256`)](#5-custom-nki-kernel-qwen_segcte256)
6. [Validation Harness & Measurement Bugs](#6-validation-harness--measurement-bugs)
7. [Tooling, Sync, Shell, SSH](#7-tooling-sync-shell-ssh)
8. [Lessons Codified in `AGENTS.md`](#8-lessons-codified-in-agentsmd)

---

## 1. Quantization & Checkpoint Conversion

### 1.1 MLP-only FP8 scope was insufficient for "full FP8"

- **Symptom:** Original path only quantized MLP layers; attention, DeltaNet projections, and fused QKV stayed BF16.
- **Cause:** `_mlp_only_modules_to_not_convert` excluded entire `self_attn` and `linear_attn` modules; checkpoint rewrite only handled MLP scale tensors.
- **Fix:** Added `fp8_full` mode in [qwen36_27b_compile_fp8.py](../test/integration/qwen36_27b_compile_fp8.py); broadened module selector to all Linear matmuls (MLP + attention + DeltaNet `in_proj_*` / `out_proj`); kept embeddings, norms, rotary, `lm_head`, DeltaNet `conv1d`/`A_log`/`dt_bias` in BF16.
- **Verification:** Unit tests in [test_qwen36_compile_fp8_config.py](../test/unit/test_qwen36_compile_fp8_config.py).

### 1.2 Scale tensors not transformed alongside weights

- **Symptom:** Loading FP8 artifact failed because scale tensors didn't match the transformed weights (Q/gate split, fused QKV concat, DeltaNet QKV TP reorder).
- **Cause:** Checkpoint converter in [modeling_qwen35.py](../src/modeling_qwen35.py) only transformed `.weight`; FP8 needs the matching `.scale` to follow the same reorder/split/concat.
- **Fix:** Added scale-aware transforms in the converter for: q_proj weight/scale split → `output_gate_proj`, fused `Wqkv` weight+scale creation, and DeltaNet `in_proj_qkv.weight/scale` TP reorder using identical permutation. FP8 concat uses `view(torch.int8)` round-trip because PyTorch rejects direct `torch.float8_e4m3fn` concat.
- **Verification:** [test_weight_conversion.py](../test/unit/test_weight_conversion.py).

---

## 2. Neuron Compiler Failures (`neuronx-cc`)

### 2.1 `NCC_ITIN902 TensorInitialization` / `AffineIV doesn't appear in params or loopnest`

- **Symptom:** Compiler internal error during NEFF tensorization on specific 2D prefix-caching bucket pairs:
  - `cte=256, prefix=16384`
  - `cte=1024, prefix=1024`
  - `cte=2048, prefix=2048`
  - `cte=4096, prefix=4096`
- **Cause:** Compiler bug in `neuronx-cc` lowering on power-of-two square shapes and the small-active/large-prefix corner. AWS log itself says "open a Neuron SDK issue."
- **Fix:** Avoid those exact shapes. Use safe CTE ladder `512, 768, 1536, 3072` (256-aligned, non-square) and limit prefix-bucket granularity.
- **Verification:** `cte512_768_1536_3072_pfx16k` compile completed with `COMPILE_DONE` and 0 `NCC_ITIN902`.

### 2.2 Combined dense + long-prefix artifact — `[F137] neuronx-cc forcibly killed (-9)`

- **Symptom:** Compiling all five long-prefix pairs (`3072:0,32k,64k,128k,256k`) in one run was killed by OOM on the largest buckets (`bk3`, `bk4`).
- **Cause:** Compile-host RAM pressure when multiple HLOs compile in parallel for very large shapes.
- **Fix:** Split tiers into separate compile runs. Implemented sequential orchestrator in [tmp_compile_qwen256k_fp8_full_prod_three_prefix_tiers_hostlogits.sh](../../../../tmp_compile_qwen256k_fp8_full_prod_three_prefix_tiers_hostlogits.sh).
- **Verification:** Three tiered artifacts (`pfx32k_64k`, `pfx128k`, `pfx256k`) all reached `COMPILE_DONE` on the same host once compiled sequentially.

### 2.3 Bash script wrote artifact paths with spaces (`tkg32768 131072 262144`)

- **Symptom:** Orchestrator created malformed paths because the launcher used `${ARR[*]}` instead of joining with `_`.
- **Cause:** Bash array word-splitting in label construction.
- **Fix:** Use explicit `IFS=_` join in the helper script.
- **Verification:** Relaunch produced underscore-only paths.

### 2.4 `head_dim must be <= 128 (got 256)` — `NCC_INKI016`

- **Symptom:** AWS Neuron 2.30 `attention_segmented_cte` kernel hard-asserts `head_dim <= 128`; Qwen3.6 uses `head_dim=256`.
- **Cause:** Kernel was not designed for 256-wide head dim.
- **Fix:** Wrote custom `qwen_segcte256` kernel that splits Q/K into two 128-wide D tiles and accumulates `Q_lo@K_lo + Q_hi@K_hi` into one PSUM before softmax. See [Section 5](#5-custom-nki-kernel-qwen_segcte256).
- **Verification:** Custom kernel BIR-compiled cleanly for production shape `q=(2,3072,256)`, `k/v=(1024,1,256,256)`, `prior_seg_size=32768`.

### 2.5 `NCC_INLA001 Allocated memory out of bound (128x402724)` — SBUF scratch too large

- **Symptom:** First version of the custom segmented CTE kernel compiled HLO but exceeded SBUF in the backend.
- **Cause:** Each Q group held its own K/V segment buffers + scratch live simultaneously; per-group block-dim allocation × 24 Q groups blew SBUF.
- **Fix:** Two-stage kernel rewrite in [fused_segmented_attention_256.py](../../../../src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/fused_segmented_attention_256.py):
  1. Allocate one reusable Q-group window instead of `block_dim=[num_grps]`.
  2. Stream active CTE in 512-token chunks through the same online-softmax accumulator.
  3. Cap packed Q loads to 4 groups (instead of 8) for `head_dim=256`.
- **Verification:** Production-shape BIR scratch dropped from `402724` to `31360`, under the 32767 SBUF free-dim limit. Full compile produced `COMPILE_DONE`.

---

## 3. vLLM / Hybrid APC / Scheduler

### 3.1 64 GiB KV cache estimate on 96 GiB Trn2 (model wouldn't even start)

- **Symptom:** vLLM rejected 256K context with `64.0 GiB KV cache needed, 39.12 GiB available`.
- **Cause:** vLLM-Neuron runner created `FullAttentionSpec` for all 64 layers. Qwen3.6 is hybrid — only 16 of 64 layers are full-attention; the other 48 are DeltaNet (no token-long KV).
- **Fix:** Patched `get_kv_cache_spec` in [qwen36_hybrid_apc_scheduler_patch.py](../vllm/qwen36_hybrid_apc_scheduler_patch.py:129) to report KV only for the 16 full-attention layers, with local KV heads per TP rank.
- **Verification:** Server log: `Using Qwen hybrid KV-cache spec for 16/64 attention layers`, `GPU KV cache size: 262,400 tokens`.

### 3.2 Warm prefix-cache continuation crashed with "no `hybrid_full_input_ids`"

- **Symptom:** Second request reusing a cached prefix died because runner received suffix-only tokens without the full prompt context the GDN path needs.
- **Cause:** Scheduler metadata didn't carry full `all_token_ids` through suffix-prefill requests; runner's strict guard rightly rejected.
- **Fix:** Scheduler patch now attaches `full_input_ids` only when `num_computed_tokens > 0` (cached continuation), not for the first cold chunk; async prep bridge unpacks it back to `hybrid_full_input_ids` and slices to active suffix length.
- **Verification:** [test_hybrid_apc_manager.py](../test/unit/test_hybrid_apc_manager.py) + working 8k→16k→18432 cold/warm exactness on TRN2.

### 3.3 `request_prefix_len` polluted by generated tokens

- **Symptom:** During decode, vLLM `request.num_tokens` grows past the original prompt length; that leaked into APC metadata and made cold vs warm runs schedule differently.
- **Cause:** Metadata used `request.num_tokens` instead of the original prompt length.
- **Fix:** Cap `request_prefix_len` to the prompt-only token count.
- **Verification:** Cold and warm 8k runs now schedule identically (`prompt_len=8192 restore_len=6144 suffix_len=2048`).

### 3.4 Dummy token `0` (`!!!!`) leaked into output during chunked prefill

- **Symptom:** Cold output started with two `0` tokens before real decoding; warm output started correctly.
- **Cause:** vLLM-Neuron host-logits sampling appended `sampled_token_ids` from incomplete chunked-prefill rows. Earlier mask attempt was on the wrong path (`_sample_on_device` instead of `_generate_model_runner_output` for `hostlogits` artifacts).
- **Fix:** Added `_generate_model_runner_output` wrapper that masks incomplete-prefill rows before vLLM appends sampled IDs to request state. Used scalar `.item()` to work on Neuron/XLA tensors.
- **Verification:** 8k cold and warm now both emit `[3817, 7840, 9197, 4590]`. Test coverage added.

### 3.5 Chunked prefill above 16k prefix exceeded compiled prefix bucket

- **Symptom:** Cold 32k prompt failed with `Prefill len 512 with prefix len 16896 exceeds compiled 2D buckets... largest prefix bucket 16384`.
- **Cause:** vLLM's chunked-prefill continuation presents (active_chunk, computed_prefix) shapes to NxDI's 2D bucket selector; with `pfx16k`, a 32k prompt eventually reaches prefix `16896`.
- **Fix:** Two approaches:
  1. Runtime cap on backed prefix reads (`QWEN36_HYBRID_APC_MAX_BACKED_PREFIX_READ_LEN`).
  2. Split production artifacts by prefix tier; route long contexts to the long artifact.
- **Verification:** Tiered split (`pfx16k` for short, `pfx32k_64k`, `pfx128k`, `pfx256k` for long) compiled and validated for context up to 128K. 256K required the custom kernel (see §5).

### 3.6 Sparse 2D bucket support needed — runtime assumed full Cartesian grid

- **Symptom:** Wanting sparse pairs like `cte=3072 × prefix=262144` only (without the failing `cte=512 × prefix=262144`) wasn't possible — runtime did `bucket_idx = prefill_index * len(prefix_buckets) + prefix_index`.
- **Cause:** NxDI runtime hard-assumed a rectangular bucket grid.
- **Fix:** Added `context_encoding_bucket_pairs` config + sparse-pair-aware runtime selection in [model_wrapper.py](../../../../src/neuronx_distributed_inference/models/model_wrapper.py:1126) and [autobucketing.py](../../../../src/neuronx_distributed_inference/modules/autobucketing.py:162). Wired through compile script and vLLM serving config.
- **Verification:** Unit tests in [test_autobucketing.py](../../../../test/unit/modules/test_autobucketing.py) and [test_prefix_caching_bucket_selection.py](../../../../test/unit/models/test_prefix_caching_bucket_selection.py).

### 3.7 Async sample called before any `execute_model()` (V1 scheduler)

- **Symptom:** With async scheduling, `sample_tokens()` was invoked once before any cached logits existed; Neuron runner raised.
- **Cause:** vLLM-Neuron runner had no "no output yet" guard like the GPU path.
- **Fix:** Added no-output guard in the runner wrapper.

### 3.8 Contract mismatch: `expected 24/29 tensors, got 15`

- **Symptom:** With prefix caching disabled at vLLM level but Hybrid APC enabled, the model wrapper got only 15 mandatory tensors while artifact expected 29.
- **Cause:** Compiled artifact's input contract is fixed at trace time. Runtime config flipping `is_prefix_caching` off without recompiling broke the contract.
- **Fix:**
  1. Server script preserves the compiled `is_prefix_caching` contract from `neuron_config.json` even when vLLM-level prefix caching is off.
  2. Qwen wrapper expands 15-tensor runtime input to 24/29-tensor traced form by padding with inert MRoPE/vision/tile tensors.
- **Verification:** Unit coverage in [test_qwen36_model_aliases.py](../test/unit/test_qwen36_model_aliases.py).

---

## 4. Runtime Load & Memory (`NRT_RESOURCE` / scratchpad / HBM)

### 4.1 Combined sparse artifact failed to load on `trn2.3xlarge`

- **Symptom:** Artifact compiled fine, but TRN2 load failed with `Failed to allocate 1.000GB ... usage: shared scratchpad` at `_tp0_bk36` (long-prefix NEFF).
- **Cause:** Trn2.3xlarge has 96 GiB total but in four 24 GiB HBM banks under LNC=2. Per-bank usage hit `~22-24 GiB` (tensors + scratchpad) before runtime needed another aligned 1 GiB allocation. The "combined" artifact loaded **every** compiled CTE×prefix NEFF at once.
- **Fix:** Physical split into multiple artifacts; route requests to the smallest artifact that covers the prefix tier.
- **Verification:** `pfx32k_64k` and `pfx128k` loaded and ran end-to-end after the split.

### 4.2 Runtime bucket-override JSON didn't reduce loaded NEFFs

- **Symptom:** Setting `--context-encoding-bucket-pairs 512:0 512:512` at runtime still failed at `_tp0_bk36` load.
- **Cause:** Saved `model.pt` references all compiled workdir NEFFs; runtime overrides control routing, not which NEFFs get staged.
- **Fix:** Per §4.1 — split artifacts physically; runtime overrides alone are insufficient.

### 4.3 `NEURON_SCRATCHPAD_PAGE_SIZE=2048` did not help

- **Symptom:** Tried larger scratchpad page size to relieve alignment pressure; still failed with `Failed to allocate 2.000GB`.
- **Cause:** Total scratchpad footprint, not just alignment fragmentation.
- **Fix:** Abandon page-size-only mitigation for over-broad artifacts; compile narrower artifacts.

### 4.4 Initial three-tier artifacts compiled but failed to load

- **Symptom:** `pfx32k_64k`, `pfx128k`, `pfx256k` all compiled with `seq_len=262144`, `pa_num_blocks=1024`, `tkg=[32768,131072,262144]` — and all failed `NRT_RESOURCE` at load.
- **Cause:** "Tiered" by prefix only; every tier still paid the full 256K cache and 3 TKG buckets.
- **Fix:** True tier-specific budgets in [tmp_compile_qwen256k_fp8_full_prod_three_prefix_tiers_hostlogits.sh](../../../../tmp_compile_qwen256k_fp8_full_prod_three_prefix_tiers_hostlogits.sh):
  - `pfx32k_64k`: `seq_len=65536`, `pa_num_blocks=256`, `tkg=[32768,65536]`, keep dense `3072:0`.
  - `pfx128k`: `seq_len=131072`, `pa_num_blocks=512`, `tkg=[131072]`, omit dense `3072:0`.
  - `pfx256k`: `seq_len=262144`, `pa_num_blocks=1024`, `tkg=[262144]`, omit dense `3072:0`.
- **Verification:** All three tierfix artifacts loaded; `pfx32k_64k` and `pfx128k` passed prefill + chat.

### 4.5 Device profiling caused `NRT_RESOURCE` on `pfx256k` load

- **Symptom:** First 256K runtime validation died because `NEURON_RT_INSPECT_DEVICE_PROFILE=1` reserved `2.348 GB HBM per NC`, pushing per-bank load over the edge.
- **Cause:** Device profiler adds non-trivial HBM tax.
- **Fix:** Run validation without `NEURON_RT_INSPECT_DEVICE_PROFILE`. Profile separately on smaller artifacts or with reduced sampling.

### 4.6 Null block (vLLM adds 1 reserved block) — `pa_num_blocks=1024` was too small

- **Symptom:** vLLM logs showed `num_gpu_blocks` becoming `1025` after the runtime adds a reserved null block, but compiled artifact only had 1024 physical blocks.
- **Cause:** Off-by-one between compile-time `pa_num_blocks` and runtime "user-usable + null" convention.
- **Fix:** Compile with `pa_num_blocks=1025`. Validation runners now treat compiled count as physical (includes null) and set `num_gpu_blocks_override` to `compiled - 1`.
- **Verification:** Compile config logged `pa_num_blocks=1025, pa_min_blocks=1024, pa_headroom_blocks=1`. Updated [qwen36_hybrid_apc_context_sweep.py](../../../../validation_scripts/qwen36_hybrid_apc_context_sweep.py) + [qwen36_offline_decode_bench.py](../../../../validation_scripts/qwen36_offline_decode_bench.py).

---

## 5. Custom NKI Kernel (`qwen_segcte256`)

Required because AWS Neuron 2.30 `attention_segmented_cte` rejects `head_dim > 128`.

### 5.1 `dma_copy dst partition dimension 256 exceeds maximum 128`

- **Symptom:** BIR compile failed when loading K cache: K SBUF tile shape `(256, 512)` violated the 128-partition rule.
- **Cause:** Tried to keep `head_dim=256` on the partition axis.
- **Fix:** Load each 256-token KV block as two 128-token halves: temp `(128, 128)`, transpose each, write into 128-token offset inside K tile.

### 5.2 `unsupported expression` — list comprehensions

- **Symptom:** `[(k_lo[i], k_hi[i]) for i in range(...)]` rejected by NKI specialization.
- **Cause:** NKI front-end doesn't accept Python list comprehensions inside kernel helpers.
- **Fix:** Build the list with explicit `for ... append`.

### 5.3 `failed to resolve name 'x::0.shape'`

- **Symptom:** After splitting K into `(lo, hi)` pair, old metadata lookup `k_sbuf[0].shape[1]` returned `.shape` from the pair tuple.
- **Fix:** Branch the metadata lookup to use `k_sbuf[0][0].shape[1]` on the split-K path.

### 5.4 `dma_transpose dst.shape must match transposed src.shape`

- **Symptom:** Q load pattern used `ac.d=256` as D extent while destination was 128.
- **Fix:** Use 128-wide D extent in source pattern: `[[ac.d, num_f], [1,1], [1,1], [1,128]]`.

### 5.5 `reduce_one_batch` signature mismatch

- **Symptom:** Compile failed with `batch_idx * sb_p * num_grps` where `batch_idx` was an object.
- **Cause:** Copied call signature didn't match installed Neuron 2.30 helper's argument order.
- **Fix:** Call with explicit keyword arguments matching the installed helper.

### 5.6 `NCC_INLA001 Allocated memory out of bound (128x402724)`

- See [§2.5](#25-ncc_inla001-allocated-memory-out-of-bound-128x402724--sbuf-scratch-too-large). The fix (group-window aliasing + active streaming + Q-pack cap) reduced production-shape SBUF scratch from `402724` to `31360`.

### 5.7 `_exp_impl` partial-sum slot index out of range

- **Symptom:** Active-streaming variant tried to index exp partial-sum slot 1 when each 512-token chunk only allocated slot 0.
- **Cause:** Active attention config still referenced full active KV length per chunk instead of per-chunk view.
- **Fix:** Specialize the active attention config per chunk with that chunk's `global KV end`, retain global K start via `kv_section_idx`.

### 5.8 Runtime `scalar DGE out-of-bound access` at 256K prefill (PA-blocks)

- **Symptom:** Compile passed, model loaded, KV initialized, then context-encoding NEFF crashed mid-execution with repeated scalar-DGE OOB.
- **First hypothesis tested:** vLLM adds a null block (`1025` physical), but artifact had `pa_num_blocks=1024`.
- **Fix attempted:** Recompiled with `pa_num_blocks=1025`. **Did not fix it** — runtime still hit DGE OOB on the new artifact.

### 5.9 Runtime DGE OOB — root cause: final partial active chunk reads past block table

- **Symptom:** Even with `pa1025`, the 261,888-token prefill failed in `context_encoding_model/_tp0_bk0` with scalar-DGE OOB.
- **Cause:** Active stream loop always processed 6 full sections per CTE bucket, even when the final real active chunk was only 768 tokens. At the end of the 256K prompt, the kernel read block-table offsets beyond the 1024-entry table.
- **Fix:** Pad the kernel's internal block table by the CTE active block count (1024 → 1036 entries). Padded active stream loads resolve to block 0 instead of reading past the table.
- **Verification:** Bound-fix artifact compiled (`COMPILE_DONE`), and the no-device-profile 256K runtime validation passed:
  - Cold 261,888 prefill: `551.97s`
  - Warm refill (16-token suffix on shared 261,872-token prefix): `10.76s`
  - Cold throughput: `474.46 tok/s`; warm refill throughput: `24,342.68 tok/s`
  - Real-token + token-range checks: passed
  - Host RSS peak: `35.31 GiB`; Neuron active allocation peak: `~28 GiB`; high-water counter: `58 GiB`

### 5.10 Block-table active-block-fill (necessary but not sufficient)

- **Symptom:** Earlier hypothesis was that `block_table` had `-1` entries for the active suffix.
- **Fix attempted:** Fill active block ids from `slot_mapping // pa_block_size` before NKI dispatch. Aligned with AWS docs on `nisa.dma_copy` dynamic addressing.
- **Result:** Helped the 8K smoke test but did NOT fix the 256K case. The real bug was §5.9.

### 5.11 Production envelope and fail-closed hardening

- **Finding:** The bound-fix 256K artifact has strong validation evidence, but only for the exact serving envelope: 256K context, `pa_num_blocks=1025`, one `cte3072:pfx262144` bucket, `qwen_segcte256` segment size 512, batch/concurrency 1, backed prefix reads, non-KVP, and non-transposed K cache.
- **Risk:** Enabling Hybrid APC outside the blessed vLLM launcher could previously fall back to local prompt hashing or synthetic attention block refs.
- **Fix:** When `use_hybrid_apc_manager=True`, `Qwen35InferenceConfig` now defaults to requiring vLLM metadata and attention block refs, with local hash fallback disabled. Validation-only flows can still opt back into local fallback explicitly.
- **Risk:** The generic ModelWrapper used absolute Hybrid APC control positions (`args[25]`) for restore-active detection.
- **Fix:** Restore-active detection now reads from the final five Hybrid APC control args, so future pre-control extras do not silently misbucket CTE.
- **Risk:** `qwen_segcte256` still exposed KVP and transposed-K branches that were not validated for production and contained NKI 0.3-sensitive HBM output/intermediate patterns.
- **Fix:** `qwen_segcte256` now raises immediately for `kvp_offset`/KVP or `k_pre_transposed=True`. The validated production path remains the non-KVP, non-transposed K path used by `attention_base.py`.

---

## 6. Validation Harness & Measurement Bugs

### 6.1 TPOT measured from streamed content chunks, not tokens

- **Symptom:** Reported TPOT was `~109 ms/chunk` at 16K context (with 16 generated tokens → only 8 streamed content chunks), masking real decode speed.
- **Fix:** [qwen36_chat_completion_context_bench.py](../../../../validation_scripts/qwen36_chat_completion_context_bench.py) now requests `stream_options: {"include_usage": true}` and computes `token_tpot_seconds` from `usage.completion_tokens`. Old chunk metric preserved as `content_chunk_tpot_seconds`.
- **Verification:** Corrected 16k pfx16k measurement: `~50-52 ms/token`, `~19-20 decode tok/s`.

### 6.2 "Warm prefill" was actually full-prompt cache replay

- **Symptom:** Sub-second warm runs were misinterpreted as refill speed.
- **Cause:** [qwen36_hybrid_apc_context_sweep.py](../../../../validation_scripts/qwen36_hybrid_apc_context_sweep.py) generated the exact same prompt twice — that's an exact cache hit, not a refill.
- **Fix:** Default warm mode now: warm shared prefix + suffix A, then measure shared prefix + suffix B.
- **Verification:** Corrected 16k partial refill: `0.91s` for a 16,368-token shared prefix → ~`18k tok/s` reuse rate.

### 6.3 Sweep accepted dummy token `0` (`!!!!`) as "real" output

- **Symptom:** Validator's `vocab_size=248044` check passed because token `0` was within range, masking the chunked-prefill output leak.
- **Fix:** Tighter validation: explicitly fail if all generated tokens equal the configured dummy id, regardless of vocab bounds. Then use `usage.completion_tokens` + tokenizer/AutoConfig vocab fallback for true range check.

### 6.4 Hardcoded `seq_len=262144, pa_num_blocks=1024` for every tier

- **Symptom:** Three-tier validation runner forced 256K cache shape on the 64K and 128K tiers, causing `NRT_RESOURCE`.
- **Fix:** [tmp_run_qwen256k_fp8_tierfix_validation.sh](../../../../tmp_run_qwen256k_fp8_tierfix_validation.sh) now uses per-tier `(seq_len, pa_num_blocks, tkg buckets)`.

### 6.5 Chat wrapper passed `--pa-num-blocks` to server script (unknown arg)

- **Symptom:** `start_vllm_server.sh` rejected `--pa-num-blocks`.
- **Fix:** Pass `--pa-num-blocks` only to offline benchmarks; server gets `--num-gpu-blocks-override` via the appropriate path.

### 6.6 Memory sampler could hang the wrapper if benchmark never started

- **Symptom:** With `--stop-when-no-match`, sampler waited forever if vLLM died during startup.
- **Fix:** Sampler ignores its own PID, handles SIGTERM/SIGINT to write summary JSON; wrapper explicitly stops sampler per phase rather than relying on regex disappearance. Sampler regex broadened to match vLLM server processes during startup.

### 6.7 `start_vllm_server.sh` forced `ENABLE_PREFIX_CACHING=1` when `--enable-hybrid-apc`

- **Symptom:** Couldn't test "Hybrid APC on, prefix-cache reads off" because flags were coupled.
- **Fix:** Split controls: `ENABLE_PREFIX_CACHING`, `ENABLE_HYBRID_APC`, `HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS`, `HYBRID_APC_ENABLE_BACKED_PREFIX_READS`, and `QWEN36_HYBRID_APC_INSTALL_PATCH` are now independent.

### 6.8 Validator's `vocab_size` check rejected legitimate model tokens

- **Symptom:** Model emitted token `248068`, valid for the loaded model (`vocab_size=248320`) but above the tokenizer's base `vocab_size=248044`.
- **Fix:** Use `max(tokenizer.vocab_size, len(tokenizer), AutoConfig.vocab_size)` as the upper bound.

---

## 7. Tooling, Sync, Shell, SSH

### 7.1 Stale remote code (no `--context-encoding-bucket-pairs`)

- **Symptom:** Remote compile script lacked sparse-pair CLI flag even though local repo had it.
- **Fix:** Sync the compile entrypoint along with runtime code; `bash -n` + `py_compile` checks before launching.

### 7.2 `scp` multi-file → wrong directory

- **Symptom:** Multi-file `scp` of mixed sources landed extra copies in the last directory.
- **Fix:** Use explicit per-file destinations or `rsync -R`. Cleaned the misplaced copies and removed them.

### 7.3 Local zsh expanded `*` in remote command

- **Symptom:** `ssh host "find /path/*"` failed locally because zsh tried to glob the path on the Mac.
- **Fix:** Quote remote command bodies; use single quotes around the SSH command argument.

### 7.4 `rsync --info=stats2` rejected by macOS BSD rsync

- **Fix:** Use portable `--stats`.

### 7.5 SSH `Permission denied (publickey)` for EC2-to-EC2 transfers

- **Symptom:** Source EC2 had no key for destination.
- **Fix:** Three options used at various times:
  1. SSH agent forwarding from local `trainium.pem`.
  2. Temporary ed25519 key created on source, authorized on destination, removed after transfer.
  3. `scp -3` relay through local (slow — avoid for large artifacts).

### 7.6 `pkill -f` matched its own SSH command, killed the shell

- **Symptom:** Cleanup SSH exited 255 with no output because broad `pgrep -f` pattern matched the SSH command line itself.
- **Fix:** Use explicit PIDs from prior status or narrower patterns; never use `pkill -f` patterns that could match the controlling shell.

### 7.7 Remote `python` not on PATH

- **Symptom:** Status/parsing commands failed with `python: command not found`.
- **Fix:** Use `python3` for remote helpers; activate Neuron venv for actual runtime work.

### 7.8 Overlay venv missing PyTorch / `libneuronpjrt-path`

- **Symptom:** Neuron 2.30 overlay venv had `nki 0.4` but no PyTorch; later, `torch_xla` failed to find the base venv's `libneuronpjrt-path` helper.
- **Fix:** Compile launcher adds base venv `site-packages` behind overlay, and base venv `bin` to `PATH`.

### 7.9 Backgrounded shell ate `$ROOT` (`tee /run.log`)

- **Symptom:** Wrapper backgrounded too broadly; nested var expansion broke; ended up writing to `/run.log` and python wasn't on PATH.
- **Fix:** Cleaner wrapper structure: start sampler separately as a tracked nohup PID; run benchmark in main subshell with explicit env activation.

### 7.10 TRN2 SSH banner timeout / port unreachable during heavy compile

- **Symptom:** SSH banner exchange timed out, then later TCP itself stopped. Local AWS CLI had stale credentials so couldn't inspect instance state.
- **Mitigation:** Use light periodic probes (not long live-tails) during heavy compiles; keep heartbeat automation as the resume signal.

---

## 8. Lessons Codified in `AGENTS.md`

Two operational rules added to the repo's [AGENTS.md](../../../../AGENTS.md):

1. **Error-logging contract:** every error gets logged with what failed, exact error text, how we got there, hypothesis, fix, and verification — enough detail that another agent can reconstruct it.
2. **Measurement discipline:**
   - TPOT must come from `usage.completion_tokens` (request `stream_options.include_usage`), not streamed content chunks.
   - "Warm refill" requires a shared prefix + different suffix; identical prompts only measure exact cache hits.
   - Record artifact, CTE buckets, prefix buckets, and whether backed prefix reads were enabled with every reported number.

---

## Final State Summary

| Tier | Artifact | Status |
|---|---|---|
| 16K | `cte512_768_1536_3072_pfx16k` | **Production-validated** — chat + multi-turn smoke pass |
| 64K | `pfx32k_64k_pa256` | **Loads + runs**, prefill + chat pass |
| 128K | `pfx128k_pa512` | **Loads + runs**, prefill + chat pass |
| 256K | `pfx256k_segcte512stream_qpack4_boundfix_pa1025` | **Validated only for the exact gated config** — cold `551.97s`, warm refill `10.76s`, real tokens validated |

**Open work before general "production-ready":** repeat 256K runs (×3-5), full OpenAI server path test on `pfx256k`, multi-turn chat at long context, soak/load test, and fresh validation for any other bucket, KVP, transposed K cache, sliding-window, or multi-seq serving configuration.
