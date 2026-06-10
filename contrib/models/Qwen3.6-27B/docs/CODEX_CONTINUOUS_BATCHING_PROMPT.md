# Codex Prompt — Enable Continuous Batching for Qwen3.6-27B

## Context

vLLM v1 already does continuous batching. The bottleneck is on the Neuron
side: the current MTP artifact (and baseline v3) was compiled with
`tkg_batch_size=1`, meaning the device can only execute one decode stream
per forward call regardless of how many vLLM tries to schedule.

To enable real continuous batching: recompile with `tkg_batch_size > 1` so
the device-side decode graph processes multiple sequences in parallel.

Current compile harness has it hardcoded:
- `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_mtp.py:79-81`
  - `batch_size=1, ctx_batch_size=1, tkg_batch_size=1`

vLLM start script ALREADY wires `tkg_batch_size = MAX_NUM_SEQS` via
`--override-neuron-config`. This override only takes effect if the
underlying NEFF was compiled with the matching batch size. **No runtime
override of compile-time batch dimension is possible.**

## Goal

Compile and validate a continuous-batching artifact with `tkg_batch_size=8`
(and matching `batch_size=8`). Demonstrate aggregate throughput scaling
with `max-num-seqs=8` on real workloads. Document HBM peak.

## Phase A: Compile harness update (target 0.5 day)

A.1 Modify `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_mtp.py`
to accept a CLI `--tkg-batch-size` argument (default 1 for backward compat).
Apply to NeuronConfig:
```python
batch_size = args.tkg_batch_size
ctx_batch_size = 1   # prefill stays single-stream per CTE call
tkg_batch_size = args.tkg_batch_size
```

A.2 Add `--max-num-seqs` and `--max-model-len` mirrors if not already present.

A.3 Reduce `seq_len` for the first batched run to keep HBM in budget:
- batch=8, seq_len=16384 → KV cache ~8 GB, total HBM ~63 GB (fits)
- batch=8, seq_len=32768 → KV cache ~16 GB, total HBM ~71 GB (fits)
- batch=8, seq_len=65536 → KV cache ~34 GB, total HBM ~89 GB (tight)

Start with batch=8, seq_len=32768. Validate. Push to 65536 only if HBM
budget allows.

A.4 Document in the compile harness which compile flags need to match the
vLLM `--max-num-seqs` value at serve time.

## Phase B: Compile + load (target 0.5 day)

B.1 Compile artifact:
```bash
python contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_mtp.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-path /opt/dlami/nvme/qwen_artifacts/qwen36_27b_32k_fp8_batch8_run1 \
  --seq-len 32768 \
  --cte-bucket 512 \
  --tp-degree 4 \
  --logical-nc-config 2 \
  --tkg-batch-size 8 \
  --load-after-compile
```

Expected compile time: ~22 min. Slightly longer than batch=1 due to bigger
TKG graph.

B.2 Load artifact on hardware. Verify load succeeds and HBM peak after
load is below 80 GB (leaves headroom for activations during inference).

B.3 If NRT_RESOURCE (HBM blew up): drop seq_len to 16384 or batch to 4.
Report which tensor exceeded budget.

## Phase C: Single-stream regression check (target 0.25 day)

C.1 Bring up vLLM server with `--max-num-seqs 8` pointing to the new
artifact:
```bash
bash contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_32k_fp8_batch8_run1 \
  --max-num-seqs 8 \
  --seq-len 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

C.2 Single-stream smoke (one request at a time):
- Math: 17 × 23 should return 391
- 762-token MGS prompt: coherent output

C.3 Single-stream perf (one request at a time):
- 4K prompt, 128-token decode: measure prefill tok/s + decode tok/s
- Compare against baseline v3 (~418 prefill, ~27 decode)
- Expect: equivalent prefill, slightly lower decode (batch=8 graph has
  some per-step overhead even at active batch=1). Acceptable: within 10%.

If decode regresses more than 20% from baseline v3: there's a configuration
issue. Investigate before continuing.

## Phase D: Aggregate throughput measurement (target 0.5 day)

D.1 Use existing `validation_scripts/qwen36_27b_vllm_concurrency_eval.py`
or write a small async harness if needed. Test at:
- concurrency=1 (baseline)
- concurrency=2
- concurrency=4
- concurrency=8 (full batch)
- concurrency=16 (tests queueing behavior)

D.2 Two prompt distributions:
- "Short": 1K prompts, 128 token decode (chat-like workload)
- "Medium": 8K prompts, 256 token decode (RAG-like workload)

D.3 Capture for each (concurrency, prompt_len) point:
- Aggregate input tok/s
- Aggregate output tok/s
- Per-stream input tok/s
- Per-stream output tok/s
- P50 / P95 TTFT
- P50 / P95 inter-token latency (TPOT)
- HBM peak (neuron-monitor during run)

D.4 Expected scaling pattern:
- concurrency=1: ~baseline single-stream
- concurrency=4: ~2-3× aggregate (sub-linear because per-stream slows)
- concurrency=8: ~3-5× aggregate at the batch ceiling
- concurrency=16: queued, aggregate same as 8 but TTFT spikes

If concurrency=8 aggregate is NOT 3× the concurrency=1 number: batching
isn't actually happening at the device. Verify by checking
neuron-monitor: should see batch=8 graph activity, not batch=1.

## Phase E: APC interaction (target 0.25 day)

E.1 Add a shared prefix to the prompts (system message + variable user
turn). Repeat the concurrency=8 measurement.

E.2 Expected: APC hit rate ≥ 50% across the 8 concurrent streams (they
share the system prompt). Aggregate prefill should jump significantly on
warm streams.

E.3 If APC hit rate stays low when streams share a prefix: APC cache is
being evicted across concurrent streams. Investigate cache size limits.

## Phase F: Documentation (target 0.25 day)

F.1 Update `OPTIMIZATION_ARC.md` with the continuous-batching results:
- Add a "Continuous batching" row to the "What worked" table
- Update the "Hardware utilization" section with the aggregate numbers
- Update the "How this compares to NVIDIA" table with aggregate Trainium
  numbers (the Millstone H100 page shows aggregate at 5 concurrent)

F.2 Create `vllm/CONTINUOUS_BATCHING.md` with:
- Compile flags required
- vLLM serve flags required
- Measured throughput curve (concurrency 1-16)
- HBM budget table by (batch, seq_len)
- APC interaction notes

## Hard constraints

1. Do not modify baseline v3 artifact. Tag the new artifact as
   `qwen36-27b-continuous-batching-v1` if all gates pass.
2. Commit + push after each phase.
3. Maximum compile attempts: 3. Each ~22 min.
4. If HBM exceeds 92 GB at batch=8: drop seq_len to 16384 and retry.
5. Do not enable MTP speculation in this artifact (defer to PR #4). Spec
   decoding + continuous batching together is harder; tackle one at a time.

## Expected outcomes

| Outcome | Probability | Meaning |
|---|---|---|
| batch=8 compiles, loads, scales to 3-5× aggregate | 60% | Best case; ship as PR #2 (continuous batching baseline) |
| batch=8 compiles but scales less than 2× | 20% | Diagnose: probably KV cache contention or scheduler overhead |
| HBM blowup, must drop to batch=4 or seq_len=16K | 15% | Acceptable fallback; still 2-3× aggregate |
| Quality regression at batch>1 | 5% | Bug in hybrid cache at batch>1; investigate |

Begin with Phase A. Report after Phase B. Do not chain phases.

## Why this is high priority

Currently single-stream decode is 27 tok/s. After continuous batching
with batch=8:
- Aggregate decode probably 150-200 tok/s (4-7× single)
- This is the metric that maps to "production serving capacity"
- Without it, you cannot answer "how many users can one instance serve?"
- Without it, the cost-per-token comparison vs H100 cannot be made

This is the prerequisite for the multi-instance scaling discussion and
for any honest production-deployment claims.
