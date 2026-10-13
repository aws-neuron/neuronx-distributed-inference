# Qwen3.6-27B vLLM on Neuron

This folder contains the first-pass vLLM integration helpers for the
Qwen3.6-27B contrib model.

The current goal is **vLLM serving through the Neuron/NxDI plugin** for the
validated coherent Qwen3.6 artifact. The validated fast long-context path uses
the compiled Neuron-native chunking contract captured in the artifact config;
the launcher should mirror that contract instead of relying on generic vLLM
chunk slicing.

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

Current 256K precompiled artifact path:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /mnt/trainium_artifacts/qwen_artifacts/qwen36_256k_fp8_loadfix_lmheadbf16_gatesbf16_kvbf16_qkvnki_segmented_cte512_gdnseg512_cte2048_pfx256k_pa1025_slots64_20260608T195113Z_256k_loadfix_segcte2048_chatfix_hostsampling_kkt_hier_scan7 \
  --max-model-len 262144 \
  --seq-len 262144 \
  --cte-bucket 2048 \
  --block-size 256 \
  --num-gpu-blocks-override 1024 \
  --port 8000
```

Prefix-cache experiment:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /mnt/trainium_artifacts/qwen_artifacts/qwen36_256k_fp8_loadfix_lmheadbf16_gatesbf16_kvbf16_qkvnki_segmented_cte512_gdnseg512_cte2048_pfx256k_pa1025_slots64_20260608T195113Z_256k_loadfix_segcte2048_chatfix_hostsampling_kkt_hier_scan7 \
  --max-model-len 262144 \
  --seq-len 262144 \
  --cte-bucket 2048 \
  --block-size 256 \
  --enable-prefix-caching \
  --gdn-checkpoint-interval 256 \
  --hybrid-gdn-recurrent-cache-dtype float32 \
  --hybrid-gdn-conv-cache-dtype bfloat16 \
  --mamba-cache-mode all \
  --mamba-ssm-cache-dtype float32 \
  --num-gpu-blocks-override 1024 \
  --port 8000
```

Treat this as an experiment, not a production mode, until validation passes.
Standard vLLM APC reuses attention KV blocks; Qwen3.6 also needs DeltaNet
recurrent state and conv state as prefix-boundary checkpoints keyed by the
cumulative prefix hash. If native APC does not produce exact greedy matches and
a clear warm-hit speedup, the next step is a hybrid APC path that restores those
GDN checkpoints alongside attention KV.

Production chat proxy:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /mnt/trainium_artifacts/qwen_artifacts/qwen36_256k_fp8_loadfix_lmheadbf16_gatesbf16_kvbf16_qkvnki_segmented_cte512_gdnseg512_cte2048_pfx256k_pa1025_slots64_20260608T195113Z_256k_loadfix_segcte2048_chatfix_hostsampling_kkt_hier_scan7 \
  --max-model-len 262144 \
  --seq-len 262144 \
  --cte-bucket 2048 \
  --block-size 256 \
  --num-gpu-blocks-override 1024 \
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
  --compiled-artifacts /mnt/trainium_artifacts/qwen_artifacts/qwen36_256k_fp8_loadfix_lmheadbf16_gatesbf16_kvbf16_qkvnki_segmented_cte512_gdnseg512_cte2048_pfx256k_pa1025_slots64_20260608T195113Z_256k_loadfix_segcte2048_chatfix_hostsampling_kkt_hier_scan7 \
  --max-model-len 262144 \
  --seq-len 262144 \
  --cte-bucket 2048 \
  --block-size 256 \
  --chat \
  --prompt "$(python - <<'PY'
print('Summarize this document in one paragraph. ' + 'Neuron inference ' * 700)
PY
)"
```

Optional Hybrid APC validation should be artifact-specific. The acceptance gate
is strict: repeated greedy calls must produce identical output, warm-hit latency
should be materially lower than cold-fill latency, and GDN recurrent/conv state
must be proven exact alongside attention KV cache hits. Attention-only prefix
cache hits are not sufficient for this hybrid model.

Current validation run on Trn2 with the 256K loadfix artifact:

- 16K native-chunk run: `16,374` prompt tokens, `6.8379s` TTFT,
  `2,394.6 tok/s` usage-accounted, `pass=true`, thinking enabled.
- Long-context native-chunk run: `242,864` `usage.prompt_tokens`, `235.9819s`
  TTFT, `1,029.2 tok/s` usage-accounted, `pass=true`, thinking enabled.
- The same long-context run has tokenizer-estimated prompt length `253,899`,
  which gives `1,075.9 tok/s`; keep that separate from usage-accounted
  throughput.
- `log_scan_empty.txt` contains no invalid-token, fallback, NaN, NRT, or
  traceback markers.

Raw `/v1/completions` prompts are not chat-templated and can pollute the hybrid
state if sent directly to the backend. Keep the backend private and expose the
proxy on the public port for production calls.

## Offline Smoke

```bash
python contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /mnt/trainium_artifacts/qwen_artifacts/qwen36_256k_fp8_loadfix_lmheadbf16_gatesbf16_kvbf16_qkvnki_segmented_cte512_gdnseg512_cte2048_pfx256k_pa1025_slots64_20260608T195113Z_256k_loadfix_segcte2048_chatfix_hostsampling_kkt_hier_scan7 \
  --max-model-len 262144 \
  --seq-len 262144 \
  --cte-buckets 2048 \
  --chat \
  --prompt "What is 17 * 23? Answer with the number only."
```

## Next Milestone

For warm-prefix production APC, the required contract remains a unified
prefix-cache object whose attention KV, GDN recurrent state, and GDN conv state
are jointly addressable, evictable, restorable, and exact. Speculation, FP8
cache variants, resident row-IO experiments, and continuous-batching extensions
are intentionally outside this baseline contribution.
