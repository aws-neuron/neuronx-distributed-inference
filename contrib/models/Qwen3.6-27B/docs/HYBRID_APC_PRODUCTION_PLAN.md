# Qwen3.6 Hybrid APC Production Plan

## Build Order

```text
1. Production hybrid APC correctness
2. Dynamic CTE bucket serving
3. Block-size, bucket, and HBM tuning
4. GDN state dtype and memory optimization
5. Decode-side improvements
6. Kernel fusion and speculative decode
```

Do not start with FP8 recurrent cache, MTP, EAGLE, Medusa, flash decode, KV
tiling, or deeper GDN kernel fusion. Those add scheduler and rollback
complexity before the cache contract is correct.

## Target Cache Object

```text
HybridPrefixCheckpoint
  cumulative_prefix_hash
  token_ids_hash
  cache_salt / tenant key
  prefix_length_at_boundary

  attention:
    per-attention-layer KV block refs

  gdn:
    per-GDN-layer recurrent_state checkpoint
    per-GDN-layer conv_state checkpoint

  metadata:
    dtype
    layout_version
    model_revision
    ref_count
    last_access_time
    valid_state_mask
```

The usable hit is the deepest cumulative-prefix boundary where all required
state exists:

```text
usable_hit_len =
  intersection(
    attention_KV_full_block_hit,
    all_GDN_recurrent_prefix_checkpoint_hits,
    all_GDN_conv_prefix_checkpoint_hits
  )
```

If attention KV hits 16K but GDN state only hits 12K, suffix prefill must resume
from 12K.

## Qwen3.6 GDN State

At every reusable cumulative-prefix boundary, cache:

```text
recurrent_state: [num_local_value_heads, key_dim, value_dim]
conv_state:      [conv_dim, conv_kernel_size - 1]
```

Initial dtype policy:

```text
attention KV:        bfloat16
GDN conv_state:      bfloat16
GDN recurrent_state: float32
```

Conv state is small but correctness-critical. Recurrent state dominates GDN
cache memory and should remain FP32 until BF16 exactness is proven.

## Restore Flow

For prompt length `P` and hybrid hit length `H`:

```text
cached prefix:  tokens [0, H)
suffix prefill: tokens [H, P)
decode:         tokens [P, ...)
```

Serving path:

```text
1. vLLM hashes prompt blocks.
2. Hybrid APC computes usable H.
3. Restore attention block table for [0, H).
4. Restore GDN recurrent_state at H.
5. Restore GDN conv_state at H.
6. Send only suffix tokens [H, P) to Neuron CTE.
7. Position IDs start at H.
8. Attention suffix attends to cached KV plus new suffix KV.
9. GDN recurrence starts from restored recurrent_state.
10. GDN conv starts from restored conv_state.
11. Store new boundary checkpoints for newly completed blocks.
12. Decode uses final restored and updated state.
```

## Sprint Plan

### Sprint 1: Correctness Foundation

Build:

```text
HybridAPCManager
GDN recurrent/conv prefix-boundary checkpoint cache
hybrid hit intersection
partial-prefix restore path
FP32 recurrent cache option
correctness tests
```

Success criteria:

```text
warm full-prefix output == cold output
partial-prefix output == cold output
attention-only false hit cannot happen
concurrent requests do not leak state
```

Current v0 branch status:

```text
implemented:
  HybridAPCMetadataStore for cumulative-prefix checkpoint metadata
  bounded model-side HybridGDNCheckpointCache tensor bank
  model restore/commit slot inputs
  use_hybrid_apc_manager initialization without the old guard
  v0 launcher validation requiring checkpoint interval == block size
  async prefix-caching bridge for scheduler-supplied restore/commit tensors
  request finish/cancel lifecycle callbacks for checkpoint refcounts
  Trainium exactness and HBM validation harness

still required before production:
  vLLM scheduler integration that computes cumulative-prefix hashes and slots
  Trainium execution of cold/warm exactness harness on compiled artifacts
  production cancellation/eviction callback wiring from vLLM events
  long-context HBM sweep to choose checkpoint slot count and commit policy
  larger production prefix buckets for 32K+ warm reuse
```

Production prefix-bucket plan:

```text
Previous 256K FP8 artifact was correct only up to its compiled prefix bucket
coverage:
  prefix_buckets = [256, 512, 1024, 2048, 4096, 8192, 16384]

32K/64K/128K contexts can still run on the 256K artifact, but warm APC reuse
above 16K must replay the remainder. This is correct but slower.

Production strategy is one sparse 2D CTE/prefix artifact, not two separate
models:

  dense fast path:
    CTE buckets    = [512, 768, 1536, 3072]
    prefix buckets = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

  long-prefix fallback:
    [CTE 3072, prefix 65536]
    [CTE 3072, prefix 131072]
    [CTE 3072, prefix 262144]

The dense fast path is for common short/normal cached prefixes and preserves
prefill speed by avoiding unnecessary padding to 3072. The sparse long-prefix
fallback enables 64K/128K/256K prefix reuse without compiling the full CTE x
prefix Cartesian grid that triggers Neuron compiler tensorization failures.
```

Implementation notes:

```text
compile flag:
  --context-encoding-bucket-pairs ACTIVE:PREFIX ...

runtime behavior:
  Prefix-caching CTE bucket selection now chooses the smallest actual compiled
  [active_tokens, prefix_tokens] pair that can serve the request, instead of
  assuming every CTE bucket exists for every prefix bucket.

serving behavior:
  vLLM override config forwards context_encoding_bucket_pairs so loaded
  artifacts use the same sparse matrix they were compiled with.
```

## Fixed Bug Record: Neuron Tensorization Failure on Full 2D Prefix Grid

```text
What failed:
  256K FP8 full Hybrid APC compile with pfx256k and multiple CTE buckets.

How we got there:
  Host: ubuntu@16.26.202.235
  Script:
    tmp_compile_qwen256k_fp8_full_cte512_768_1536_3072_pfx256k_hostlogits.sh
  Key args:
    --seq-len 262144
    --max-context-length 262144
    --cte-buckets 512 768 1536 3072
    --prefix-buckets 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144
    --weight-dtype fp8_full
    --enable-prefix-caching
    --enable-hybrid-apc
    --enable-vllm-chunked-prefill

Exact error:
  NCC_ITIN902 TensorInitialization error:
    AffineIV doesn't appear in params or loopnest

Failed generated buckets:
  bk9  = [CTE 512, prefix 65536]
  bk10 = [CTE 512, prefix 131072]
  bk21 = [CTE 768, prefix 65536]
  bk22 = [CTE 768, prefix 131072]

Root cause hypothesis:
  HLO generation succeeds, then neuronx-cc fails inside internal tensorization
  for some small-active-token / large-prefix-token 2D prefix-cache shapes. This
  is a Neuron compiler lowering bug, not disk pressure and not an invalid model
  config.

Fix:
  Stop compiling the full Cartesian product. Add explicit sparse
  context_encoding_bucket_pairs and route runtime selection over the actual
  compiled pair list.

Mitigation shape set:
  Dense fast path only up to 32K prefix for all production CTE buckets:
    [512/768/1536/3072] x [0..32768]
  Long-prefix fallback only on largest CTE bucket:
    [3072, 65536], [3072, 131072], [3072, 262144]

Verification:
  Unit/config tests passed:
    38 local contrib tests passed
    86 remote Neuron-env focused tests passed
  Sparse high-prefix probe compile started with 7 CTE HLOs and no NCC_ITIN902
  observed at HLO generation time; final NEFF compile result must still be
  checked before treating the sparse artifact as production-ready.
```

## Fixed Bug Record: Invalid Fast Warm Prefill

This bug is useful to showcase because the first symptom looked like excellent
performance, but the warm path was not executing the same model semantics as
cold prefill.

```text
Symptom:
  Warm prefill appeared sub-second, but cold/warm generated token IDs diverged.
  Cold also leaked placeholder token IDs:
    cold = [0, 0, 3817, 7840]
    warm = [3817, 7840, 9197, 4590]

Root causes:
  1. vLLM attention prefix hits could exceed the deepest GDN checkpoint that
     was actually available.
  2. Scheduler metadata used request token counts that could include generated
     tokens instead of prompt-only tokens.
  3. Incomplete chunked-prefill rows in the host-logits path could append
     placeholder sampled IDs as real generated tokens.

Fix:
  1. Cap vLLM prefix-cache reads to the largest GDN-backed checkpoint.
  2. Build Hybrid APC metadata from prompt-only length/token IDs.
  3. Mask incomplete chunked-prefill sampled IDs to -1 before vLLM appends
     them to request state.

Evidence after fix:
  8K cold/warm exactness passed:
    cold = [3817, 7840, 9197, 4590]
    warm = [3817, 7840, 9197, 4590]
    repeat_exact = true

  Warm prefill became slower than the invalid shortcut, but correct:
    cold ~= 15.26s
    warm ~= 4.95s
```

### Sprint 2: Dynamic CTE Buckets

Build:

```text
multi-bucket CTE artifact path
runtime suffix bucket selection
262K TP=4 [256] artifact
block_size 128/256 comparison
```

Success criteria:

```text
short prompts retain 1.5x-2.3x latency gain
262K TP=4 [256] loads
TP=4 beats TP=8 unless TP=4 cannot load
```

### Sprint 3: Memory and HBM Tuning

Build:

```text
GDN recurrent state slot accounting
eviction/ref-count policy
FP32 vs BF16 recurrent experiment
attention KV memory report
hybrid cache memory dashboard
```

### Sprint 4: Decode Optimization

Build:

```text
lower-overhead GDN state gather/scatter
decode microbenchmarks
batch-slot reuse optimization
possibly fused recurrent step
```

## Test Matrix

Correctness:

```text
cold vs warm exact token IDs
partial-prefix exact match
non-block-aligned shared prefix floors to full block
attention hit with missing GDN state falls back
conv-state restore failure test by zeroing conv state
multi-hit chat simulation
mixed cold/warm continuous batching
long-context warm hit at 128K and 262K
```

Performance:

```text
Context length: 256, 512, 2K, 8K, 32K, 128K, 262K
Block size:    64, 128, 256
CTE buckets:   [256], [512], [256,512], [256,512,1024]
TP:            4, and 8 only if HBM/load requires it
Cache mode:    no APC, attention APC only, hybrid APC
GDN dtype:     recurrent FP32, recurrent BF16 experiment
Workloads:     single request, repeated system prompt, chat, long-doc QA
```

Immediate Trainium experiments:

```text
262K TP=4, block_size=256, CTE buckets [256]
262K TP=4, block_size=128, CTE buckets [256]
128K TP=4, block_size=128, CTE buckets [256,512]
128K TP=4, block_size=256, CTE buckets [256,512]
```
