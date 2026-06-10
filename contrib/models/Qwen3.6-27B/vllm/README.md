# Qwen3.6-27B vLLM on Neuron

This folder contains the first-pass vLLM integration helpers for the
Qwen3.6-27B contrib model.

The current goal is **vLLM serving through the Neuron/NxDI plugin** for the
validated Qwen3.6 artifact, including long prompts through vLLM's native
chunked-prefill scheduler.

## Which vLLM Neuron Package?

Use the vLLM-on-Neuron environment that matches the installed Neuron SDK first.
For SDK 2.29, the AWS Neuron guide lists the NxDI/vLLM plugin stack as
`vLLM 0.16.0` with plugin version `0.5.0`. The
`vllm-project/vllm-neuron` repository is useful source/reference material, but
its README currently describes a beta plugin path tied to older `vLLM 0.11.0`
and SDK 2.26.1. Do not downgrade the working SDK 2.29 environment just to use
that repository.

On a DLAMI, prefer the preinstalled vLLM/Neuron environment when available. If
the instance does not have one, install the Neuron-compatible vLLM plugin/fork
using the current AWS guide, then run the contrib registry patch below.

## What Works First

- Register the contrib `qwen3_5` text model with the NxDI model registry inside
  the vLLM environment.
- Start vLLM with `VLLM_PLUGINS=neuron`.
- Load a small-context model or a precompiled artifact with
  `NEURON_COMPILED_ARTIFACTS`.
- Run a short OpenAI-compatible smoke prompt.

## Hybrid APC Production Boundary

Qwen3.6-27B is a hybrid model, so attention prefix caching alone is not a
complete production APC contract. The current stack has strong serving
primitives for attention-only models: block KV, block tables, slot mapping,
prefix caching, continuous batching, chunked prefill, and decode. For hybrid
attention plus GDN recurrence, the attention KV path is production-shaped, but
the GDN state path is still model-specific glue.

Current readiness:

| Layer | Current support | Production readiness |
| --- | ---: | ---: |
| Attention KV cache | Good | High on the existing NxDI/vLLM block-KV path |
| vLLM APC for attention blocks | Working baseline | Medium/high |
| GDN recurrent state cache | Implemented locally | Low/medium |
| GDN conv state cache | Implemented locally | Low/medium |
| Hybrid APC across attention + GDN | Not fully implemented | Low |
| Continuous batching with exact hybrid prefix reuse | Not supported by the local manager | Low |
| Speculation, FP8 cache, tiling, flash decode with hybrid state | Explicitly rejected by the local manager | Low |

The `HybridDeltaNetCacheManager` is therefore a contrib-local static/stateful
cache manager, not a production hybrid APC manager. It proves the model can
preserve recurrent and conv state, but it is batch-row based rather than
vLLM block-hash, refcount, eviction, and tenant-isolation based.

Production hybrid APC must define the usable prefix as the intersection of:

1. attention KV block hit;
2. GDN recurrent prefix-boundary checkpoint hit;
3. GDN conv prefix-boundary checkpoint hit.

For each GDN layer, the reusable checkpoint object needs:

```text
recurrent_state: [local_value_heads, key_dim, value_dim]
conv_state:      [conv_dim, conv_kernel_size - 1]
```

The recurrent state should stay FP32 for exact cold-vs-warm agreement until
BF16 equivalence is proven. Conv state can follow the model-compatible dtype,
but exactness still needs token-level validation. If the attention APC hit lands
inside a GDN checkpoint interval, restore the nearest earlier full GDN
checkpoint, replay the residual tokens, then run the suffix.

The launchers expose `--enable-hybrid-apc` and explicit hybrid cache dtype
knobs. In the current v0 implementation, `use_hybrid_apc_manager=True` creates
a bounded GDN checkpoint-slot bank and adds restore/commit tensors to the model
signature. The serving request-prep path must still fill those tensors from the
vLLM/NxDI cumulative-prefix hash lifecycle; otherwise the default zero masks run
as attention KV plus normal active-row GDN state with no GDN checkpoint reuse.
For v0, `gdn_checkpoint_interval` must equal the vLLM block size.

The production server launcher enables strict hybrid APC metadata by default.
That means request prep must provide vLLM/NxDI cumulative prefix hashes and real
attention block refs; local token-hash fallback is reserved for controlled
validation via `--allow-hybrid-apc-local-hash-fallback`. The live scheduler
integration should pass the full prompt before suffix slicing using
`hybrid_full_input_ids`/`full_input_ids`, attach `vllm_attention_hit_len`, pass
`cumulative_hashes_by_prefix_len`, and pass actual attention block refs at
commit time through `actual_attention_block_refs` or
`hybrid_actual_attention_block_refs`. Attention KV eviction should call the
model/store `on_attention_block_evicted` callback so GDN checkpoints do not
outlive the KV blocks they depend on.

## Chunked Prefill Note

The Neuron plugin disables vLLM chunked prefill by default and installs a custom
continuous-batching scheduler. For this Qwen3.6 artifact we need vLLM's native
chunked-prefill scheduler so prompts longer than the 512-token context graph are
fed to the precompiled model in 512-token chunks. The launcher sets
`DISABLE_NEURON_CUSTOM_SCHEDULER=1` when `--enable-vllm-chunked-prefill` is
passed. It also launches with `--generation-config vllm` so model
`generation_config.json` does not silently override deterministic sampling
defaults.

## Install The Contrib Registry Patch

Activate the vLLM/Neuron environment on the instance, then run:

```bash
cd /home/ubuntu/inferentia-gdn
contrib/models/Qwen3.6-27B/vllm/install_qwen36_vllm.sh
```

If your vLLM environment is not in a standard location:

```bash
contrib/models/Qwen3.6-27B/vllm/install_qwen36_vllm.sh \
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
```

The installer only patches the active environment. It does not modify core repo
files.

## Start vLLM

Small-context compile/load path:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --max-model-len 512 \
  --port 8000
```

Precompiled artifact path:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-buckets 128,256,512 \
  --port 8000
```

Cold-prefill bucket waste is the first performance target. CTE buckets must stay
128-aligned because the fused DeltaNet CTE path operates in 128-token chunks.
Use one of the explicit profiles when compiling artifacts:

```bash
# Short-prompt latency
--cte-bucket-profile short     # [128,256,512,1024]

# General production
--cte-bucket-profile general   # [256,512,1024,2048]

# Long-context artifact
--cte-bucket-profile long      # [4096,8192,16384,32768]

# 262K load experiment
--cte-bucket-profile 262k      # [256]
```

`--cold-zero-conv-fast-path` is only for a cold-only CTE artifact whose suffix
prefill always starts at position 0. Leave it disabled for APC or partial-prefix
serving because restored GDN conv state must be consumed exactly.

Long-prompt precompiled artifact path:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-buckets 256,512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --port 8000
```

Native vLLM prefix-cache experiment:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-buckets 256,512 \
  --block-size 128 \
  --enable-vllm-chunked-prefill \
  --enable-prefix-caching \
  --gdn-checkpoint-interval 256 \
  --hybrid-gdn-recurrent-cache-dtype float32 \
  --hybrid-gdn-conv-cache-dtype bfloat16 \
  --mamba-cache-mode all \
  --mamba-ssm-cache-dtype float32 \
  --port 8000
```

Treat this as an experiment, not a production mode, until validation passes.
Standard vLLM APC reuses attention KV blocks; Qwen3.6 also needs DeltaNet
recurrent state and conv state as prefix-boundary checkpoints keyed by the
cumulative prefix hash. If native APC does not produce exact greedy matches and
a clear warm-hit speedup, the next step is a hybrid APC path that restores those
GDN checkpoints alongside attention KV.

For APC experiments, do not treat `256` as the only block size. It can be useful
for long-context amortization, but it is coarse for chat-style prefix reuse.
Run explicit sweeps at `64` and `128`; include `32` when hit granularity matters
enough to justify possible block-table/layout overhead. Keep the GDN checkpoint
interval separate from the attention block size.

Immediate Trainium experiments:

```text
262K TP=4, block_size=256, CTE buckets [256]
262K TP=4, block_size=128, CTE buckets [256]
128K TP=4, block_size=128, CTE buckets [256,512]
128K TP=4, block_size=256, CTE buckets [256,512]
```

Production chat proxy:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --port 8001
```

Then expose the guarded OpenAI-compatible endpoint on port 8000:

```bash
python contrib/models/Qwen3.6-27B/vllm/qwen36_chat_proxy.py \
  --backend-url http://127.0.0.1:8001 \
  --port 8000
```

The proxy forces `chat_template_kwargs={"enable_thinking": false}` for
`/v1/chat/completions` by default. It rejects raw `/v1/completions` because raw
prompts bypass the Qwen chat template and can pollute the hybrid model state.
It also hoists `system` and `developer` messages to a single leading `system`
message because the Qwen chat template rejects system messages that appear later
in the conversation. Start the proxy with `--allow-thinking` to allow a
request-level toggle while keeping the default non-thinking path. Supported
toggles include `enable_thinking=true`, `thinking=true`,
`thinking={"enabled": true}`, `reasoning_effort=low|medium|high`, and native
`chat_template_kwargs={"enable_thinking": true}`. Use `--allow-completions` only
for explicit debugging.

Offline long-prompt smoke:

```bash
python contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --chat \
  --prompt "$(python - <<'PY'
print('Summarize this document in one paragraph. ' + 'Neuron inference ' * 700)
PY
)"
```

Offline token-exact prefix-cache validation:

```bash
python validation_scripts/qwen36_vllm_prefix_cache_offline.py \
  --repo-root /home/ubuntu/inferentia-gdn \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 128 \
  --enable-vllm-chunked-prefill \
  --mamba-cache-mode all
```

Offline partial-prefix validation:

```bash
python validation_scripts/qwen36_vllm_prefix_cache_partial_offline.py \
  --repo-root /home/ubuntu/inferentia-gdn \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 128 \
  --enable-vllm-chunked-prefill \
  --mamba-cache-mode all
```

Server-side prefix-cache validation through the guarded proxy:

```bash
python validation_scripts/qwen36_prefix_cache_validation.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3.6-27b-neuron-128k-fp8-mlp
```

The acceptance gate is strict: repeated greedy calls must produce identical
output, and warm-hit latency should be materially lower than cold-fill latency.
For hybrid Qwen3.6, prefix-cache validation is not complete until the GDN
recurrent/conv state behavior is proven, not just attention KV cache hits.

Hybrid APC exactness and HBM harness:

```bash
python validation_scripts/qwen36_hybrid_apc_validation.py exactness \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_hybrid_apc \
  --seq-len 2048 \
  --cte-buckets 256,512 \
  --block-size 256 \
  --gdn-checkpoint-interval 256 \
  --enable-vllm-chunked-prefill

python validation_scripts/qwen36_hybrid_apc_validation.py hbm \
  --context-lens 131072 262144 \
  --checkpoint-intervals 128 256 512
```

Native APC validation run on Trn2 with the FP8 128K artifact:

- server exact-repeat, `~10.8K` prompt tokens: `26.68s` cold to `1.67s` warm,
  `16.0x` speedup, exact greedy text match;
- offline exact-repeat, token IDs exposed: `26.19s` cold to `2.38s` warm,
  `11.0x` speedup, exact greedy token-ID match;
- offline partial-prefix reuse, token IDs exposed: `25.52s` no-cache target to
  `1.70s` APC target after a different shared-prefix warmup request, `15.0x`
  speedup, exact greedy token-ID match.
- server hardening, exact repeat: `25.38s` cold to `1.55s` warm, `16.35x`
  speedup, exact text match;
- server hardening, cross-prefix reuse after unrelated prefix: `25.17s` cold to
  `1.36s` warm, exact text match;
- shared-prefix concurrency at 1/2/4 requests returned all requested markers
  exactly; the artifact still queues because it is compiled for `max_num_seqs=1`.

Validation run on Trn2 with the FP8 128K artifact:

- state-reset artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1`;
- OpenAI-compatible `/v1/chat/completions` behind the proxy passes focused
  quality checks without callers passing `chat_template_kwargs`;
- repeated short-after-long validation passes after 32K and 64K requests,
  confirming DeltaNet recurrent/conv state is reset for new requests;
- 32K and 64K needle retrieval prompts return all expected codes;
- measured prefill is `404-428 tok/s` from 512 through 64K prompt tokens;
- measured decode is `26.3-26.6 tok/s`;
- peak Neuron device memory is about `53.25 GB` decimal for the 64K eval.

Raw `/v1/completions` prompts are not chat-templated and can pollute the hybrid
state if sent directly to the backend. Keep the backend private and expose the
proxy on the public port for production calls.

4K BF16 Hybrid APC boundary/server probes:

```bash
# Artifact/config audit before spending a Trn2 run. This flags oversized PA
# blocks, low block headroom, strict-gate boundary pressure, and nki_chunked CTE.
python validation_scripts/qwen36_artifact_config_audit.py \
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_4096_bf16_hybrid_apc_nki_chunked_prefix4096_ctx2_tkg2_r7i_20260520T082342Z \
  --compile-log /home/ubuntu/validation_logs/hybrid_apc_real_tokens/qwen36_4k_bf16_hybrid_apc_nki_chunked_prefix4096_20260520T082342Z_compile.log

# Boundary-aligned APC proof. Run this directly against vLLM or a proxy started
# with --allow-completions because exact token-ID prompt lengths are required.
python validation_scripts/qwen36_openai_boundary_apc_probe.py \
  --base-url http://127.0.0.1:8000 \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --lengths 256,512,1024,2048,4096 \
  --repeats 3 \
  --require-prefix-cache-query \
  --output-jsonl /home/ubuntu/validation_logs/hybrid_apc_real_tokens/boundary_apc_probe.jsonl

# Cold prefill ctx-batch utilization check. Compare --concurrency 1 and 2 with
# --unique-per-request to avoid warm-cache reuse.
python validation_scripts/qwen36_chat_completion_context_bench.py \
  --base-url http://127.0.0.1:8000 \
  --model /home/ubuntu/models/Qwen3.6-27B \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --lengths 4096 \
  --turns 8 \
  --repeats 3 \
  --concurrency 2 \
  --unique-per-request \
  --no-stream \
  --output-json /home/ubuntu/validation_logs/hybrid_apc_real_tokens/chat_4k_concurrency2.json
```

4K BF16 compile controls for the current investigation:

```bash
# Single-request cold-prefill latency control: smaller PA blocks, usable block
# headroom, and fused DeltaNet CTE. Use a fresh compiled path and workdir.
python contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py \
  --repo-root /home/ubuntu/inferentia-gdn-experimental \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --compiled-path /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_4096_bf16_hybrid_apc_fused_block32_ctx1 \
  --base-compile-work-dir /mnt/trainium_artifacts/qwen_artifacts/_work_qwen36_4k_fused_block32_ctx1 \
  --weight-dtype bf16_control \
  --seq-len 4096 \
  --max-context-length 4096 \
  --cte-buckets 256,512,1024,2048,4096 \
  --prefix-buckets 4096 \
  --block-size 32 \
  --pa-headroom-blocks 64 \
  --tp-degree 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --skip-warmup \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --enable-vllm-chunked-prefill \
  --deltanet-cte-backend fused \
  --gdn-checkpoint-interval 32 \
  --max-gdn-checkpoint-slots 160 \
  --hybrid-apc-require-vllm-metadata \
  --hybrid-apc-enable-backed-prefix-reads
```

The `block_size=32` control follows Neuron's prefix-cache performance guidance,
but it also increases the number of prefix boundaries the strict Hybrid APC gate
must prove. Without boundary chunk commits, a full 4096-token prompt has 128
possible attention-hit boundaries at block size 32, so `max_gdn_checkpoint_slots`
must be sized accordingly or the safe gate will keep skipping APC reads.

## Offline Smoke

```bash
python contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-buckets 128,256,512 \
  --chat \
  --prompt "What is 17 * 23? Answer with the number only."
```

## Next Milestone

For cold-prefill latency, fix bucket waste before speculative decode or cache
quantization. The serving entrypoints now support multi-bucket CTE artifacts,
text-only CTE inputs, compact CTE masks, context-batch profiles, and attention
tile overrides.

For warm-prefix production APC, the required contract remains a unified
prefix-cache object whose attention KV, GDN recurrent state, and GDN conv state
are jointly addressable, evictable, restorable, and exact under continuous
batching.

Recommended order:

1. Dynamic CTE buckets: start with `[128,256,512]` for 2K short-prompt tests,
   `[256,512]` for 128K, and `[256]` for the 262K TP=4 load experiment.
2. Fused GDN CTE path validation: qwen chunked-prefill should use fused
   DeltaNet with restored initial state by default.
3. Text-only CTE and compact-mask validation: no full dummy vision reductions
   and no dense 4D causal masks in normal text serving.
4. Hybrid APC exactness: cold vs warm greedy token IDs, partial-prefix reuse,
   multi-hit chat history, continuous batching movement, and eviction pressure.
5. Attention block-size sweeps at `64` and `128`, with `32` included for
   granularity-sensitive chat workloads.
6. FP8 KV/cache only after the BF16/FP32 baseline is exact.
7. MTP/spec decode after recurrent-state rollback semantics are explicit.
