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
  --cte-bucket 512 \
  --port 8000
```

Long-prompt precompiled artifact path:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
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
  --cte-bucket 512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --port 8000
```

Treat this as an experiment, not a production mode, until validation passes.
Standard vLLM APC reuses attention KV blocks; Qwen3.6 also needs DeltaNet
recurrent state and conv state at block boundaries. If native APC does not
produce exact greedy matches and a clear warm-hit speedup, the next step is a
hybrid APC path that caches those GDN states alongside attention KV.

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
in the conversation. Use `--allow-thinking` or `--allow-completions` only for
explicit debugging.

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
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --mamba-cache-mode align
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
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --mamba-cache-mode align
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

## Offline Smoke

```bash
python contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --chat \
  --prompt "What is 17 * 23? Answer with the number only."
```

## Next Milestone

Validate native vLLM prefix caching with the token-exact offline harness. If it
does not pass, implement hybrid APC by saving/restoring DeltaNet recurrent and
conv state at block boundaries.
