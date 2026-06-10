# MTP Batched Accept Fix

## Problem

The OpenAI-compatible server in `scripts/openai_compat_server.py` discards
fused-spec accepted tokens beyond the first. The decode loop calls
`_token_scalar(out.tokens)` which returns only `tokens[0]`, then feeds that
token back as the next input. Result: host advances 1 token per Python loop
iteration even when the device accepted multiple via MTP speculation.

Observed effect on `qwen36_27b_128k_fp8_mtp_run2` artifact:
- Expected decode: 2.0-2.5x baseline (NVIDIA's published MTP gain for length=2)
- Actual decode: 1.6x baseline (44 tok/s vs 27 baseline)
- Gap is purely host-loop, not device compute

## Fix

Patch: `mtp_batched_accept.patch`

Changes:
1. Add `_accepted_tokens(tokens, vocab_size, pad_id)` helper that scans the
   fused-spec output tensor and returns the prefix of in-vocab non-pad tokens.
2. Rewrite the decode loop as a `while` loop with bootstrap + iterations:
   - Bootstrap: commit `first_token` from prefill at position `prompt_tokens`.
   - Each iteration: feed `new_ids[-1]` at position
     `prompt_tokens + len(new_ids) - 1`, then commit all accepted tokens
     returned by the device.
   - Stop on EOS, max_tokens cap, or invalid token id.
3. No model recompile required. No NeuronConfig changes.

## Apply

From repo root on branch `codex/qwen36-mtp-vllm-apc`:

```bash
git apply docs/patches/mtp_batched_accept.patch
# or, if line numbers shifted:
git apply --3way docs/patches/mtp_batched_accept.patch
```

Verify by inspection:

```bash
grep -n "_accepted_tokens\|while len(new_ids)" \
  contrib/models/Qwen3.6-27B/scripts/openai_compat_server.py
```

Should show the helper definition near the top and the new while-loop in the
decode path.

## Validation gates (in order)

Run against the existing `qwen36_27b_128k_fp8_mtp_run2` artifact.

### Gate 1: Smoke
Math prompt returns 391 with coherent text. No invalid token errors. Same
behavior as before the patch.

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-128k-fp8-mtp","messages":[
        {"role":"user","content":"What is 17 * 23?"}],
       "max_tokens":32}'
```

Expect output containing `391`.

### Gate 2: Greedy parity
Same 5 fixed prompts before and after patch. Greedy decode (top_k=1).
Token-by-token output should be **identical** between pre-patch and post-patch
because the patch only changes how the host loop consumes the device output,
not the math.

If mismatch: bug in `_accepted_tokens` (likely missing pad sentinel or
off-by-one). Investigate before measuring perf.

### Gate 3: Decode tok/s
Same benchmarks as the MTP results doc:
- 32-token prompt, 128-token completion: expect decode tok/s ~50-60 (vs 41.6)
- 28-token prompt, 256-token completion: expect decode tok/s ~55-65 (vs 44.3)
- 3959-token prompt, 128-token completion: expect decode tok/s ~55-65 (vs 45.2)

If decode is unchanged from previous MTP measurements: spec is not actually
accepting multiple tokens per forward. Verify by logging
`len(accepted)` distribution during a 200-token generation; expect mean ≥ 1.5.

### Gate 4: Long-context coherence
16K-token prompt, 256-token completion. Output should be coherent and not
contain any invalid tokens. Same quality as pre-patch.

## Expected speedup

| Workload | Before patch | After patch | Mechanism |
|---|---:|---:|---|
| 32-tok / 128-out decode | 41.6 tok/s | **~55-65 tok/s** | Consume 2 accepted per forward |
| 28-tok / 256-out decode | 44.3 tok/s | **~55-65 tok/s** | Sustained spec acceptance |
| 4K / 128-out decode | 45.2 tok/s | **~55-65 tok/s** | Same |

Combined with baseline v3 (27 tok/s) → MTP after patch (~55-65) is **2.0-2.4x
total decode speedup**, matching NVIDIA's published number for spec length=2.

## What this does NOT do

- Does not change prefill speed (still ~420 tok/s flat across contexts)
- Does not change model quality (same math, same tokens, same logits)
- Does not change vLLM bridge (custom OpenAI server only)
- Does not change cache management
- Does not require artifact recompile

## Followups after this lands

1. Tag artifact + branch as `qwen36-27b-mtp-v2` with the new tok/s numbers
2. Apply the same batched-accept logic to the vLLM-Neuron decode path (once
   the v1 MTP registry gap is fixed)
3. Investigate speculation length=3 (currently length=2 in the artifact)
4. Measure acceptance rate distribution; if mean < 1.5, MTP head quality is
   the limit, not the host loop
