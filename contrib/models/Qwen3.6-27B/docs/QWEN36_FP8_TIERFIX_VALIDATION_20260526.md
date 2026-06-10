# Qwen3.6 27B FP8 Tierfix Validation - 2026-05-26

This note records the 2026-05-26 TRN2 validation of the three prefix-tier FP8
artifacts and the current blocker for 256K prefix serving.

Raw result JSON is stored at:

```text
profile_artifacts/qwen36_fp8_tierfix_validation_20260526/summary_partial_with_pfx256_failure.json
```

Remote validation root:

```text
/home/ubuntu/validation_logs/fp8_256k/tierfix_validation_20260526T152617Z
```

Test host:

```text
ubuntu@16.50.61.215
instance: trn2.3xlarge
logical-neuroncore-config: 2
```

## Artifact Results

### 32K/64K Prefix Tier

Artifact:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_64k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx32k_64k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx32k_64k_pa256_slots64_tkg32768_65536_async_20260526T132620Z_tierfix_pfx32k_64k
```

Compiled limits:

```text
seq_len=65536
max_context_length=65536
pa_num_blocks=256
pa_block_size=256
prefix_buckets=[32768, 65536]
context_encoding_bucket_pairs=[[3072, 0], [3072, 32768], [3072, 65536]]
token_generation_buckets=[32768, 65536]
```

Prefill:

| target tokens | cold prefill | cold TPS | warm refill | warm refill TPS | real tokens |
| --- | ---: | ---: | ---: | ---: | --- |
| 32768 | 60.596s | 540.77 | 5.480s | 5979.19 | pass |
| 65280 | 121.736s | 536.24 | 5.667s | 11519.06 | pass |

Chat/decode:

| target tokens | run | TTFT | TPOT | decode TPS | completion tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| 32768 | cold | 59.913s | 79.67ms | 12.55 | 64 |
| 32768 | repeat | 5.549s | 78.36ms | 12.76 | 64 |
| 65280 | cold | 67.959s | 83.66ms | 11.95 | 64 |
| 65280 | repeat | 5.785s | 83.50ms | 11.98 | 64 |

Runtime evidence:

```text
vLLM reported GPU KV cache size: 65,792 tokens
max concurrency for 65,536 tokens: 1.00x
peak host RSS during chat: 34.04 GiB
```

### 128K Prefix Tier

Artifact:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_128k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx128k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx128k_pa512_slots64_tkg131072_async_20260526T132620Z_tierfix_pfx128k
```

Compiled limits:

```text
seq_len=131072
max_context_length=131072
pa_num_blocks=512
pa_block_size=256
prefix_buckets=[131072]
context_encoding_bucket_pairs=[[3072, 131072]]
token_generation_buckets=[131072]
```

Prefill:

| target tokens | cold prefill | cold TPS | warm refill | warm refill TPS | real tokens |
| --- | ---: | ---: | ---: | ---: | --- |
| 130816 | 298.355s | 438.46 | 6.972s | 18762.74 | pass |

Chat/decode:

| target tokens | run | TTFT | TPOT | decode TPS | completion tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| 130816 | cold | 298.550s | 173.88ms | 5.75 | 64 |
| 130816 | repeat | 7.250s | 173.41ms | 5.77 | 64 |

Runtime evidence:

```text
vLLM reported GPU KV cache size: 131,328 tokens
max concurrency for 131,072 tokens: 1.00x
peak host RSS during chat: 33.75 GiB
```

### 256K Prefix Tier

Artifact:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1024_slots64_tkg262144_async_20260526T132620Z_tierfix_pfx256k
```

Compiled limits:

```text
seq_len=262144
max_context_length=262144
pa_num_blocks=1024
pa_block_size=256
prefix_buckets=[262144]
context_encoding_bucket_pairs=[[3072, 262144]]
token_generation_buckets=[262144]
```

Result: failed during Neuron Runtime load before validation could run.

Exact failure:

```text
NRT_RESOURCE in nrt_load_util
Failed to allocate 1.000GB (alignment: 4.000MB, usage: shared scratchpad)
Failed to load NN:
  .../context_encoding_model/_tp0_bk0/model.MODULE_c3eddc16a94d9c7dfe80+5c498585.neff
err: 4
Failed to create logical core info for subgraph 0 to 1
Failed to stage graph to NeuronCore
Failed to load collectives for model
```

TDRV memory table at failure:

```text
per-HBM TOTAL:       22.056GB
Model Tensors:       12.052GB
Shared Scratchpad:   10.000GB
Failed next alloc:    1.000GB shared scratchpad
```

Retry with debug scratchpad placement disabled:

```text
NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE=0
root=/home/ubuntu/validation_logs/fp8_256k/tierfix_validation_pfx256_dbg0_20260526T160225Z
```

Result: still failed with the same `NRT_RESOURCE` class.

```text
per-HBM TOTAL:       22.056GB
Model Tensors:       12.052GB
Shared Scratchpad:    7.000GB on one logical core + 3.000GB on sibling
Failed next alloc:    1.000GB shared scratchpad
```

Probe with smaller runtime scratchpad page:

```text
NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE=0
NEURON_SCRATCHPAD_PAGE_SIZE=512
root=/home/ubuntu/validation_logs/fp8_256k/pfx256_pagesize512_probe_20260526T160430Z
```

Result: still failed.

```text
NRT_RESOURCE in nrt_load_util
Failed to allocate 512.000MB (alignment: 4.000MB, usage: shared scratchpad)
per-HBM TOTAL:       23.056GB
Model Tensors:       12.052GB
Shared Scratchpad:   11.000GB
```

## Why 256K Prefix Fails

The current 256K prefix artifact is not failing because GDN attention KV cache
needs a full-attention 256K cache. It fails earlier: Neuron Runtime cannot load
the 256K context-encoding NEFF because the compiled NEFF's model tensors plus
shared scratchpad exceed the usable HBM slice for that logical placement.

AWS Neuron's device-memory documentation describes HBM usage categories such as
model tensors, shared scratchpad, non-shared scratchpad, DMA rings, and runtime
allocations. It also documents that scratchpad page size must be coordinated
between compile-time `NEURON_CC_FLAGS=--hbm-scratchpad-page-size=...` and
runtime `NEURON_SCRATCHPAD_PAGE_SIZE=...`; changing only runtime placement/page
size is not guaranteed to repair a NEFF whose compiled scratchpad layout is too
large. See:

```text
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/device-memory.html
```

AWS Trainium2 documentation lists 96 GiB of device memory per Trainium2 chip,
but this validation shows the failing pfx256 CTE NEFF is constrained by the
per-HBM/logical-placement allocation shown in the TDRV table, not by aggregate
host RAM or the headline chip memory number. See:

```text
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html
```

Current hypothesis:

```text
The pfx256 context-encoding NEFF at [CTE 3072, prefix 262144] has too much
compiled tensor + shared scratchpad footprint for trn2.3xlarge LNC2 placement.
Runtime-only tweaks did not reduce that footprint enough. Fixing it requires a
new compile with lower pfx256 CTE scratchpad/tensor footprint, a different
tiling/page-size compile, or avoiding the pfx262144 CTE NEFF.
```

## Can We Use 128K Prefix and Infer 256K Context?

Not with the current 128K artifact.

The current 128K artifact is a 128K-total artifact:

```text
max_context_length=131072
seq_len=131072
pa_num_blocks=512
token_generation_buckets=[131072]
```

It cannot serve or decode a 256K context because the compiled position range,
KV capacity, and token-generation bucket stop at 131072.

A separate 256K-total artifact with only a 128K prefix bucket is a valid next
mitigation to test:

```text
seq_len=262144
max_context_length=262144
pa_num_blocks=1024
prefix_buckets=[131072]
context_encoding_bucket_pairs=[[3072, 131072]]
token_generation_buckets=[262144]
```

That would be semantically valid for 256K context if it loads, but it changes
the caching behavior:

```text
cached reusable prefix: up to 128K
remaining prompt suffix: replay/refill up to the requested context length
decode positions: up to 256K, because max_context_length and tkg are 256K
```

So 128K prefix is not a replacement for 256K context. It is a cache boundary
inside a 256K-capable artifact. It should be correct, but slower than true
pfx256 reuse for prompts where the reusable shared prefix is above 128K.

Risk:

```text
This still needs a 256K token-generation/KV-capable artifact. The pfx128 CTE
NEFF may avoid the pfx256 shared-scratchpad load failure, but the 256K decode
and PA footprint still must be compiled and load-tested before we call it
production-ready.
```

## Robust 256K Prefix Fix Under Test

The robust fix is to keep the 256K prefix bucket but stop compiling the
prefix-attention CTE as one monolithic `[active_tokens, prefix_tokens]` score
tensor.

Implementation:

```text
src/neuronx_distributed_inference/models/config.py
  NeuronConfig.prefix_cte_attention_chunk_size

src/neuronx_distributed_inference/modules/attention/attention_base.py
  NeuronAttentionBase.perform_prefix_prefill_chunked_prior()

contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py
  --prefix-cte-attention-chunk-size
```

Behavior:

```text
If prefix_cte_attention_chunk_size is set and prior_len exceeds it, prefix CTE
attention streams cached-prefix K/V in fixed chunks and combines the chunks with
online softmax. This avoids materializing the full [Q, prefix] score tensor.
The compiled bucket can still be [CTE 3072, prefix 262144].
```

Why this is the robust path:

```text
The failed pfx256 compile produced an 11GB page-aligned scratchpad requirement
for the pfx256 context-encoding NEFF. 32K/64K prefix-tier artifacts already
compiled and loaded. Streaming pfx256 as eight 32K chunks should bound live
attention-score memory near the proven smaller prefix shapes while preserving
correct full-256K prefix semantics.
```

Compile probe started:

```text
host: ubuntu@16.51.90.254
pid: 59247
artifact:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_stream32k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_stream32k_pa1024_slots64_tkg262144_async_20260526T164116Z_pfx256_stream32k
workdir:
  /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_pfx256k_stream32k_cte3072_pfx256k_stream32k_pa1024_tkg262144_20260526T164116Z_pfx256_stream32k
log:
  /home/ubuntu/validation_logs/fp8_256k/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_stream32k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_stream32k_pa1024_slots64_tkg262144_async_20260526T164116Z_pfx256_stream32k_compile.log
key args:
  --seq-len 262144
  --max-context-length 262144
  --prefix-buckets 262144
  --context-encoding-bucket-pairs 3072:262144
  --token-generation-buckets 262144
  --pa-num-blocks 1024
  --prefix-cte-attention-chunk-size 32768
```

Validation so far:

```text
Local config test:
  python3 -m pytest contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py \
    -k 'prefix_cte_attention_chunk_size or sparse_context_encoding_bucket_pairs_are_forwarded'
  result: 2 passed

Remote attention test:
  NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -m pytest \
    test/unit/modules/attention/test_attention_base.py \
    -k 'prefix_prefill_chunked_prior or prefix_prefill_sharded_flash_attn or prefix_prefill_unsharded_flash_attn'
  result: 6 passed

Compile status at start:
  HLO generation completed for context_encoding_model and token_generation_model.
  neuronx-cc priority compilation started with no NCC_ITIN902/NRT_RESOURCE in
  the main log at the time this note was updated.
```

## Error Log

### Remote `rg` Missing

```text
What failed:
  Remote repo inspection command using `rg`.

How it failed:
  bash: line 1: rg: command not found

How we got there:
  Host ubuntu@16.50.61.215 did not have ripgrep installed.

Hypothesis:
  The TRN2 instance image lacks the local developer tooling installed on the
  Mac workspace.

Fix:
  Switched remote inspection to `find`, `grep`, `sed`, and `python3`.

Verification:
  Remote config inspection completed and printed the 128K/256K neuron_config
  limits recorded in this note.
```

### Launcher PID Redirection

```text
What failed:
  Initial background validation launcher.

How it failed:
  bash: line 1: ${PID}: ambiguous redirect

How we got there:
  A shell grouping/variable expansion issue while starting the long validation
  command and writing the PID file.

Hypothesis:
  The PID variable was expanded in the wrong shell context.

Fix:
  Manually wrote the detected validation PID to:
    /home/ubuntu/validation_logs/fp8_256k/tierfix_validation_20260526T152617Z/run.pid

Verification:
  The validation continued and produced the raw summary JSON stored in this
  repo.
```

### Remote `python` Missing

```text
What failed:
  Remote JSON/config parsing helper invoked as `python`.

How it failed:
  bash: line 1: python: command not found

How we got there:
  The remote instance exposes Python as `python3`, not `python`.

Hypothesis:
  No `python` compatibility symlink on the remote image.

Fix:
  Reran the helper with `python3`.

Verification:
  Parsed artifact config fields successfully.
```

### Local Attention Unit Test Missing `torch_xla`

```text
What failed:
  Local focused attention unit test.

Command:
  python3 -m pytest test/unit/modules/attention/test_attention_base.py \
    -k 'prefix_prefill_chunked_prior or prefix_prefill_sharded_flash_attn or prefix_prefill_unsharded_flash_attn'

How it failed:
  ModuleNotFoundError: No module named 'torch_xla'

How we got there:
  The Mac workspace Python environment does not include torch_xla.

Hypothesis:
  Local environment is not the Neuron inference venv.

Fix:
  Synced the changed files to ubuntu@16.51.90.254 and reran in the Neuron venv.

Verification:
  Remote test passed with 6 selected tests after setting
  NEURON_PLATFORM_TARGET_OVERRIDE=trn2.
```

### Remote Attention Unit Test Platform Override

```text
What failed:
  First remote focused attention unit test on ubuntu@16.51.90.254.

How it failed:
  RuntimeError: Unsupported Platform - r7i.24xlarge
  If you want to compile on CPU, please supply a compiler target argument.

How we got there:
  The compile host is a CPU/cross-compile instance. Importing Neuron/NxD modules
  without a platform override caused torch_neuronx to infer the host platform
  instead of the target Trainium platform.

Hypothesis:
  Neuron unit tests that import NxD need NEURON_PLATFORM_TARGET_OVERRIDE when
  running on non-Trainium compile hosts.

Fix:
  Reran with:
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2

Verification:
  6 selected attention prefix-prefill tests passed.
```

### 256K Prefix Runtime Load Failure

```text
What failed:
  256K pfx256 artifact prefill/runtime load.

How it failed:
  NRT_RESOURCE in nrt_load_util:
    Failed to allocate 1.000GB shared scratchpad
    Failed to load context_encoding_model/_tp0_bk0/model...neff, err: 4

How we got there:
  Artifact:
    qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_...
  Inputs:
    seq_len=262144
    pa_num_blocks=1024
    length=261888
    CTE/prefix pair=[3072, 262144]
    token_generation_buckets=[262144]

Hypothesis:
  The pfx256 context-encoding NEFF compiled tensor + shared scratchpad footprint
  exceeds the usable per-HBM allocation for the logical NeuronCore placement.

Fix attempted:
  Retried with:
    NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE=0
  Then probed:
    NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE=0
    NEURON_SCRATCHPAD_PAGE_SIZE=512

Verification:
  Both retries still failed with `NRT_RESOURCE`, so the remaining blocker is a
  compiled NEFF footprint issue, not just a runtime placement knob.
```

### Python-Level 256K Prefix Chunking Did Not Reduce NEFF Memory

```text
What failed:
  Robust pfx256 mitigation probe using Python-level prefix attention chunking.

How it failed:
  The compile itself completed, but the context-encoding NEFF memory footprint
  did not improve:
    COMPILE_DONE
    context HBM: 24.101GB
    total page-aligned scratchpad: 11.000000GB

How we got there:
  Host:
    ubuntu@16.51.90.254
  Artifact:
    /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_stream32k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_stream32k_pa1024_slots64_tkg262144_async_20260526T164116Z_pfx256_stream32k
  Workdir:
    /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_pfx256k_stream32k_cte3072_pfx256k_stream32k_pa1024_tkg262144_20260526T164116Z_pfx256_stream32k
  Inputs:
    --seq-len 262144
    --max-context-length 262144
    --cte-buckets 3072
    --prefix-buckets 262144
    --context-encoding-bucket-pairs 3072:262144
    --token-generation-buckets 262144
    --pa-num-blocks 1024
    --prefix-cte-attention-chunk-size 32768

Hypothesis:
  XLA/Neuron static tracing still lowered the Python chunk loop into a graph
  with the same large flat prefix attention footprint. This confirms that
  chunking must happen inside an NKI kernel or via a newer segmented CTE kernel,
  not in regular PyTorch graph code.

Fix or mitigation applied:
  Do not treat the pfx256_stream32k artifact as the production fix. Web/docs
  research points to Neuron 2.30 NKI Library `Attention Segmented CTE` and
  `KV-Parallel Segmented Prefill` as the next production-grade path because
  they process block KV/prefix cache in configurable segments inside the kernel.

Verification:
  Pending. Need either:
    1. Upgrade/overlay a Neuron 2.30 NKI Library containing segmented CTE and
       wire prefix CTE to that kernel; or
    2. Write a custom NKI segmented prefix attention kernel if the library
       kernel is unavailable in our runtime.
```

### Neuron 2.30 Segmented CTE Overlay Inspection

```text
What failed:
  First SSH inspection command after creating the Neuron 2.30 segmented CTE
  overlay on ubuntu@16.51.90.254.

How it failed:
  The command exited 1 because it used `set -o pipefail` with:
    find "$NKILIB_DIR" -maxdepth 5 -type f | grep -E "attention.*(seg|prefill|cte).*\.py$|kv.*prefill.*\.py$"
  The `find` maxdepth/pattern missed the files under
  src/nkilib_src/nkilib/core/attention, so `grep` returned no matches.

How we got there:
  Host:
    ubuntu@16.51.90.254
  Overlay venv:
    /home/ubuntu/venvs/neuron_230_segmented_cte
  Source checkout:
    /home/ubuntu/nki-library-2.30
  Branch:
    2.30_release
  Installed Python packages:
    nki==0.4.0+25940409122.gd30719f9
    neuronx-cc==2.25.3371.0+f524f7f8

Root cause:
  Inspection-command bug, not an overlay setup failure.

Fix:
  Reran inspection with the overlay activated and direct Python imports.

Verification:
  Confirmed:
    IMPORT_OK nkilib.core.attention.attention_segmented_cte
    IMPORT_OK nkilib.core.attention.kv_parallel_segmented_prefill
    attention_segmented_cte signature accepts block KV cache, block_tables,
    prior_tokens, block_size, and prior_seg_size.
```

### Local Segmented CTE Search/Syntax Checks

```text
What failed:
  Local source search for k-cache transposition references.

How it failed:
  Command exited 2:
    rg: src/modeling_qwen35.py: No such file or directory (os error 2)

How we got there:
  I searched `src/modeling_qwen35.py`, but this repository stores the Qwen model
  file at:
    contrib/models/Qwen3.6-27B/src/modeling_qwen35.py

Root cause:
  Wrong local path in the search command.

Fix:
  Reran with `rg --files` and then searched existing paths under `src` and
  `contrib/models/Qwen3.6-27B`.

Verification:
  Found the relevant `k_cache_transposed` references and confirmed block KV
  cache disables transposed K cache.

What failed:
  First local Python syntax command:
    python -m py_compile ...

How it failed:
  zsh:1: command not found: python

How we got there:
  The local Mac shell exposes `python3` but not `python`.

Fix:
  Reran:
    python3 -m py_compile ...

Verification:
  Syntax compile passed for:
    src/neuronx_distributed_inference/modules/attention/attention_base.py
    src/neuronx_distributed_inference/models/config.py
    src/neuronx_distributed_inference/models/model_wrapper.py
    contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py
    contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py
  Focused config tests passed:
    23 passed, 3 subtests passed
```

### Segmented CTE Overlay Wiring and Compile Launch

```text
What failed:
  Remote diff preview after syncing segmented CTE files to ubuntu@16.51.90.254.

How it failed:
  The command exited 141 because it ran `git diff ... | head -240` under
  `set -o pipefail`; `head` closed the pipe and `git diff` received SIGPIPE.

How we got there:
  Files had already been installed into:
    /home/ubuntu/inferentia-gdn-fused-noclamp-4340808
  The failing command was only a preview step after install.

Root cause:
  Shell preview mistake, not a sync failure.

Fix:
  Reran remote status/syntax checks without piping through `head`.

Verification:
  Remote `py_compile` passed for the synced files.

What failed:
  First remote focused tests in the Neuron 2.30 overlay venv.

How it failed:
  /home/ubuntu/venvs/neuron_230_segmented_cte/bin/python:
    No module named pytest

How we got there:
  The overlay venv was intentionally minimal and only installed newer
  nki/neuronx-cc.

Root cause:
  Missing test dependency in the overlay venv.

Fix:
  Ran focused unit tests in the base Neuron venv instead:
    /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16

Verification:
  Remote tests passed:
    42 passed, 3 subtests passed

What failed:
  First overlay import check for the synced attention module.

How it failed:
  ModuleNotFoundError: No module named 'torch'

How we got there:
  The overlay venv had nki 0.4 / neuronx-cc 2.25 but did not inherit the base
  Neuron venv's PyTorch/NxD packages.

Root cause:
  `python -m venv --system-site-packages` does not inherit packages installed
  inside another venv.

Fix:
  Added a `.pth` file in the overlay site-packages pointing to:
    /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/lib/python3.12/site-packages

Verification:
  Overlay then imported PyTorch, but exposed the next PATH issue below.

What failed:
  Overlay import after adding base site-packages.

How it failed:
  FileNotFoundError:
    [Errno 2] No such file or directory: 'libneuronpjrt-path'

How we got there:
  torch_xla imported from the base venv site-packages, but the overlay
  activation did not include the base venv `bin` directory on PATH.

Root cause:
  Base Neuron helper executables were not visible when using overlay Python.

Fix:
  Updated the compile launcher to export:
    PATH="${NEURON_VENV}/bin:${BASE_NEURON_VENV}/bin:${PATH}"

Verification:
  Overlay import check passed:
    TORCH 2.9.1+cu128
    NKI 0.4.0+25940409122.gd30719f9
    NEURONXCC 2.25.3371.0+f524f7f8
    SEGMENTED_KERNEL True

What failed:
  Potential segmented CTE compile-sample invalidity for
  [active=3072, prefix=262144] with `pa_num_blocks=1024`.

How it would fail:
  The generated sample `slot_mapping` would write active KV at positions
  262144..265215, past the 256K cache capacity, before segmented CTE reads
  active KV from block cache.

How we got there:
  Existing prefix CTE sampled `computed_context_lens=prefix_bucket`; this was
  fine for flat `attention_cte` because active KV was passed separately, but
  segmented CTE reads active KV from the updated block cache.

Root cause:
  The sample value for `computed_context_lens` was not constrained to
  `max_context_length - active_bucket` for segmented CTE.

Fix:
  In `model_wrapper.py`, for context-encoding segmented CTE samples, keep the
  bucket shape at 262144 but set the sample prior to:
    min(prefix_bucket, max_context_length - n_active_tokens)
  For the pfx256/cte3072 trace this is:
    computed_context_lens=259072

Verification:
  The segmented CTE compile got through both context HLOs and the
  token-generation HLO without sample OOB or import errors.

What was cleaned up:
  Removed the two known-bad pfx256 probes before launching the new compile:
    qwen36_27b_256k_..._prod_pfx256k_..._20260526T132620Z_tierfix_pfx256k
    qwen36_27b_256k_..._prod_pfx256k_stream32k_..._20260526T164116Z_pfx256_stream32k
  plus their `_nxd_model_workdir_*` directories.

Why:
  They were already proven not production-ready:
    pfx256 tierfix hit runtime load NRT_RESOURCE.
    pfx256_stream32k compiled but kept the same large HBM/scratch footprint.

Verification:
  Free disk on /mnt/trainium_artifacts increased from 35GB to 93GB.

Current compile:
  Host:
    ubuntu@16.51.90.254
  PID:
    65885
  Artifact:
    /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte32k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_segcte32k_pa1024_slots64_tkg262144_async_20260526T174252Z
  Log:
    /home/ubuntu/validation_logs/fp8_256k/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte32k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_segcte32k_pa1024_slots64_tkg262144_async_20260526T174252Z_compile.log
  Key flags:
    --context-encoding-bucket-pairs 3072:262144
    --prefix-cte-attention-backend segmented_cte
    --prefix-cte-attention-segment-size 32768
    --pa-num-blocks 1024
  Status:
    HLO generation completed for both context_encoding_model traces and the
    token_generation_model trace. neuronx-cc compilation is running.
```

### Segmented CTE Compile Completed but pfx256 Footprint Still Has Flat Gather

```text
What failed:
  The pfx256 segmented CTE compile completed, but it did not eliminate the
  high-footprint 256K-prefix context NEFF.

How it failed:
  Compile status:
    COMPILE_DONE
  Artifact:
    /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte32k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_segcte32k_pa1024_slots64_tkg262144_async_20260526T174252Z
  Context bucket summaries:
    context_encoding_model/_tp0_bk0:
      Total estimated HBM usage: 13.65GB
      Total page-aligned scratchpad: 1.500000GB
    context_encoding_model/_tp0_bk1:
      Total estimated HBM usage: 24.10GB
      Total page-aligned scratchpad: 11.000000GB
  Token-generation summary:
    token_generation_model/_tp0_bk0:
      Total estimated HBM usage: 12.42GB
      Total page-aligned scratchpad: 0.500000GB

How we got there:
  Host:
    ubuntu@16.51.90.254
  Key compile args:
    --context-encoding-bucket-pairs 3072:262144
    --prefix-cte-attention-backend segmented_cte
    --prefix-cte-attention-segment-size 32768
    --pa-num-blocks 1024
  Overlay:
    nki==0.4.0+25940409122.gd30719f9
    neuronx-cc==2.25.3371.0+f524f7f8

Evidence:
  `neuron_config.json` in the artifact records:
    prefix_cte_attention_backend=segmented_cte
    prefix_cte_attention_segment_size=32768
  But `context_encoding_model/_tp0_bk1/log-neuron-cc.txt` still contains
  large indirect loads from:
    get_kv_by_layer_id/_get_block_cache_and_reshape_bhsd/aten.index_select
  with cache-shaped tensors such as:
    bfloat16 (1025, 65536)
  That means the long-prefix trace still materialized the flattened block-cache
  gather before/alongside the segmented CTE path.

Root cause / hypothesis:
  The integration still calls `kv_mgr.get_kv_by_layer_id(**kwargs)` before the
  segmented CTE pre-update path. For prefix caching, that method gathers block
  KV into flat BHSD prior tensors through `_get_block_cache_and_reshape_bhsd`.
  Those flattened gathers remain in the HLO and dominate the 256K-prefix NEFF,
  so using `attention_segmented_cte` later is not enough.

Fix or next mitigation:
  The robust fix is to add a true raw-block-cache prefix path for segmented
  CTE:
    1. In context encoding when `prefix_cte_attention_backend=segmented_cte`,
       do not call `kv_mgr.get_kv_by_layer_id` for prefix prior.
    2. Fetch raw per-layer block KV via `kv_mgr._fetch_cache(...)` or a clean
       public wrapper.
    3. Pre-update active K/V into raw block KV.
    4. Call `attention_segmented_cte` with raw block KV, `active_block_table`,
       and `computed_context_lens`.
    5. Return the updated raw block KV and skip the old flat prior path.

Verification:
  Not fixed yet. The completed artifact should not be treated as the pfx256
  production fix. It can be transferred only for confirmation, but based on the
  compile footprint it is expected to have the same runtime-load risk as the
  previous pfx256 artifact.
```

### Raw Block Segmented CTE Fix Applied

```text
What failed:
  Web/docs review found that the previous segmented CTE integration did not
  match the official block-KV contract. The Qwen hybrid prefill path still
  called `get_kv_by_layer_id`, which flattened prefix blocks before attention.

How it failed:
  The pfx256 segmented CTE artifact compiled, but `log-neuron-cc.txt` still
  showed `_get_block_cache_and_reshape_bhsd/aten.index_select` in the pfx256
  context bucket and HBM reached 24.10GB per core with 11GB page-aligned
  scratchpad.

How we got there:
  Branch:
    codex/full-fp8-qwen36
  Backend:
    prefix_cte_attention_backend=segmented_cte
  Bucket:
    context_encoding_bucket_pairs=3072:262144
  The base attention code had a segmented CTE call, but Qwen's hybrid path
  pre-fetched `past_key_values` through `QwenHybridBlockKVCacheManager.get_cache`
  and then used `perform_qwen_chunked_prefill` over flat selected prefix KV.

Root cause / hypothesis:
  Official Neuron docs say NxDI prefix caching uses block KV, but the default
  prefix-caching flow gathers block KV into a flat layout before attention.
  Neuron 2.30 adds `Attention Segmented CTE` and `KV-Parallel Segmented
  Prefill` kernels specifically for block-based KV cache. Therefore the fix is
  not another bucket shape; it is avoiding the flat gather entirely for the
  segmented CTE path.

Fix applied:
  - Added `BlockKVCacheManager.get_raw_kv_by_layer_id()` to return block-layout
    KV without `_get_block_cache_and_reshape_bhsd`.
  - Changed `QwenHybridBlockKVCacheManager.get_cache()` so segmented context
    prefix buckets return raw block KV for full-attention layers.
  - Changed Qwen chunked prefill so `prefix_cte_attention_backend=segmented_cte`
    pre-updates active K/V into raw block cache and calls
    `attention_segmented_cte` with `active_block_table` and
    `computed_context_lens`.
  - Changed Qwen cache update to accept already-updated raw block KV and skip a
    second block-cache update.
  - Fixed the base attention segmented path so it no longer requires flat
    `past_key_value` before dispatching to segmented CTE.

Verification:
  Pending local unit tests and a new pfx256 compile. Expected compile evidence
  for success:
    - no `_get_block_cache_and_reshape_bhsd/aten.index_select` in pfx256
      context HLO/logs;
    - pfx256 context HBM below the per-core 24GB limit with materially smaller
      scratchpad than the failed 24.10GB / 11GB artifact.
```

### Raw Block Segmented CTE Compile Blocked by head_dim=256

```text
What failed:
  Fresh pfx256 raw-block segmented CTE compile failed during HLO generation.

How it failed:
  Host:
    ubuntu@16.51.90.254
  PID:
    69740
  Log:
    /home/ubuntu/validation_logs/fp8_256k/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_rawsegcte32k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_rawsegcte32k_pa1024_slots64_tkg262144_async_20260526T183314Z_compile.log
  Artifact target:
    /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_rawsegcte32k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_rawsegcte32k_pa1024_slots64_tkg262144_async_20260526T183314Z
  Exact error:
    AssertionError: error: failed to compile NKI kernel:
    Collected 1 different diagnostics:
    - [x2] error: assertion failed: [INTERNAL_ERROR] [NCC_INKI016]
      Kernel validation exception: head_dim must be <= 128 (got 256).
      Larger head_dim not yet supported. - Please check the validation
      message and adjust kernel inputs accordingly

How we got there:
  Command launched `tmp_compile_qwen256k_fp8_full_prod_prefix_tier_hostlogits.sh`
  with:
    TIER_NAME=pfx256k_rawsegcte32k
    PREFIX_BUCKETS_STR=262144
    PAIR_ARGS_STR=3072:262144
    CTE_BUCKETS_STR=3072
    TKG_BUCKETS_STR=262144
    PA_NUM_BLOCKS=1024
    PREFIX_CTE_ATTENTION_BACKEND=segmented_cte
    PREFIX_CTE_ATTENTION_SEGMENT_SIZE=32768
    NEURON_VENV=/home/ubuntu/venvs/neuron_230_segmented_cte
    NKI_LIBRARY_SRC=/home/ubuntu/nki-library-2.30/src/nkilib_src

Root cause / hypothesis:
  Proven root cause for this failure: Neuron 2.30 `attention_segmented_cte`
  hard-validates `head_dim <= 128`, while Qwen3.6 27B has attention
  `head_dim=256`.
  Evidence:
    /home/ubuntu/nki-library-2.30/src/nkilib_src/nkilib/core/attention/attention_segmented_cte.py
    contains:
      kernel_assert(head_dim <= 128, ...)
  The bundled NKI model test config for `qwen3_235b` uses `d_head=128`, so
  the official segmented CTE Qwen coverage does not cover this 27B head_dim.

Fix or mitigation:
  The raw-block segmented CTE integration is correct structurally, but the
  official kernel cannot support this model without a head_dim=256 variant.
  Viable next options are:
    1. Build a Qwen-specific head_dim=256 segmented CTE kernel that accumulates
       QK over two 128-wide D tiles before softmax, then computes PV over the
       full 256-wide V. This is the robust fix if we require a true 256K prefix
       bucket.
    2. Use the existing production-safe tier strategy with <=128K prefix buckets
       and route 256K-context requests through a lower prefix bucket, accepting
       extra refill work.
    3. Open/escalate an AWS Neuron issue requesting head_dim=256 support in
       `attention_segmented_cte`.

Verification:
  Compile did not complete. Do not retry this exact raw-block segmented CTE
  compile until the head_dim=256 kernel limitation is addressed.
```

### Qwen head_dim=256 Segmented CTE Kernel Bring-Up

```text
What failed:
  The first Qwen-specific segmented CTE prototype was a direct copy of the
  Neuron 2.30 segmented CTE kernel with only the top-level head_dim validator
  relaxed.

How it failed:
  Host:
    ubuntu@16.51.90.254
  Remote repo:
    /home/ubuntu/inferentia-gdn-fused-noclamp-4340808
  Probe:
    Offline NKI compile_to_bir with q=(2,256,256),
    k/v_cache=(8,1,256,256), block_size=256, prior_seg_size=512,
    tp_q=True, tp_out=False, target=trn2.
  Exact errors hit and fixed:
    1. dma_copy dst partition dimension 256 exceeds maximum 128
       at attention_segmented_cte_256.py load_kv_cache.
       Cause: copied kernel still loaded K as one (head_dim, K_TILE) tile.
       Fix: split K into low/high (128, K_TILE) tiles.
    2. unsupported expression on list comprehensions creating K tile pairs.
       Cause: NKI specialization rejected Python list comprehensions.
       Fix: build the list with explicit for/append meta-programming.
    3. failed to resolve name 'x::0.shape' from k_sbuf[0].shape.
       Cause: split K tile entries are Python pairs, not NKI tensors.
       Fix: read K_TILE_SIZE from k_sbuf[0][0].shape for head_dim=256.
    4. dma_copy dst partition dimension 256 exceeds maximum 128 on the
       temporary non-transposed K block load.
       Cause: temp was shaped (block_size, 128), and block_size=256 became
       the partition dimension.
       Fix: load each K block in 128-token by 128-dim chunks.
    5. dma_copy src/dst element mismatch src=32768 dst=16384.
       Cause: source access pattern still selected full D=256 for a 128-wide
       destination.
       Fix: use HBM source pattern [[head_dim, 128], [1, 128]] for each
       128-token by 128-dim K chunk.
    6. dma_transpose Q shape mismatch: source D=256, destination D=128.
       Cause: split Q source pattern used full D as the transposed extent.
       Fix: keep token stride at ac.d but set the transposed D count to 128:
       [[ac.d, num_f], [1, 1], [1, 1], [1, 128]].
    7. reduce_one_batch batch_idx typed as object.
       Cause: copied call used an older helper signature and passed output
       tensors where Neuron 2.30 expects batch_idx/grp_start/grp_end.
       Fix: call reduce_one_batch with batch_idx=0, grp_start=0,
       grp_end=n_grps, d=head_dim, num_grps=n_grps, sb_p=sb_p.

Fix implemented:
  Added a Qwen-specific NKI package:
    src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/

  The kernel keeps V and output on legal free dimensions and only splits the
  QK contraction:
    logits = Q_lo @ K_lo + Q_hi @ K_hi

  This follows the documented NKI matmul rule that contraction dimensions
  larger than 128 must be accumulated through multiple nc_matmul writes to the
  same PSUM tile.

Verification:
  Local syntax:
    python3 -m py_compile attention_base.py attention_segmented_cte_256.py
    fused_segmented_attention_256.py

  Remote syntax:
    PYCOMPILE_OK on ubuntu@16.51.90.254 under
    /home/ubuntu/venvs/neuron_230_segmented_cte with Neuron 2.30 NKI source.

  Remote NKI BIR probe:
    BIR_OK for q=(2,256,256), k/v=(8,1,256,256), prior_seg_size=512.

  Remote production-shape NKI BIR probe:
    BIR_Q3072_OK for q=(2,3072,256), k/v=(1024,1,256,256),
    block_size=256, prior_seg_size=32768, pa_num_blocks=1024.
    Reported scratch:
      sb_scratch_sizes=[402724]
      psum_scratch_sizes=[15360]

Remaining work:
  This validates NKI front-end/BIR legality for the target bucket geometry.
  Full model compile and runtime numerical validation are still required before
  calling the pfx256 artifact production-ready.
```

### pfx256 segcte256d32k Full Compile Failed on SBUF Scratch Allocation

```text
What failed:
  Full Qwen3.6 27B FP8 pfx256k compile with the Qwen head_dim=256 segmented
  CTE kernel failed during neuronx-cc compilation of context_encoding_model.

How it failed:
  Host:
    ubuntu@16.51.90.254
  PID:
    76788
  Log:
    /home/ubuntu/validation_logs/fp8_256k/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte256d32k_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1024_slots64_tkg262144_async_20260526T191006Z_compile.log
  Workdir:
    /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_pfx256k_segcte256d32k_cte3072_pfx256k_pa1024_tkg262144_20260526T191006Z
  Failing buckets:
    context_encoding_model/_tp0_bk0
    context_encoding_model/_tp0_bk1
  Exact compiler error:
    [INTERNAL_ERROR] [NCC_INLA001] Unhandled exception with message:
    Allocated memory out of bound
    {scratch_sb_for_inst__I-361}@SB<0,0>(128x402724)
    #Internal DebugInfo: <scratch_sb_for_inst__I-361||[128, 402724]>
  Exit:
    neuronx-cc returned non-zero exit status 70.

How we got there:
  Compile command used:
    TIER_NAME=pfx256k_segcte256d32k
    PREFIX_BUCKETS_STR=262144
    PAIR_ARGS_STR=3072:262144
    CTE_BUCKETS_STR=3072
    TKG_BUCKETS_STR=262144
    PA_NUM_BLOCKS=1024
    PREFIX_CTE_ATTENTION_BACKEND=segmented_cte
    PREFIX_CTE_ATTENTION_SEGMENT_SIZE=32768
    NEURON_VENV=/home/ubuntu/venvs/neuron_230_segmented_cte
    NKI_LIBRARY_SRC=/home/ubuntu/nki-library-2.30/src/nkilib_src

Root cause / hypothesis:
  Proven:
    The custom head_dim=256 NKI kernel is BIR-legal for the target shape, but
    the backend rejects the generated context CTE kernel because its live SBUF
    scratch allocation is too large: 128x402724.
  Best current hypothesis:
    The first head_dim=256 kernel keeps too many per-segment K/V and per-Q-group
    attention buffers live in SBUF. Splitting D into two 128-wide K/Q tiles fixed
    the head_dim validator, but doubled K-side live storage and still inherited
    the upstream segmented-CTE allocation style that materializes too much segment
    state at once.

Additional probes:
  Offline BIR probes after failure showed scratch is still high even with lower
  segment sizes:
    q=3072, segment=8192  -> sb_scratch_sizes=[206116]
    q=3072, segment=4096  -> sb_scratch_sizes=[116064]
    q=3072, segment=2048  -> sb_scratch_sizes=[107840]
    q=3072, segment=512   -> sb_scratch_sizes=[107808]
    q=512,  segment=512   -> sb_scratch_sizes=[54208]
  These are BIR-legal but still likely too high for backend SBUF placement.

Fix or mitigation:
  Do not retry the same pfx256 segcte256d32k compile.
  The next robust kernel fix is to reduce live SBUF, not only segment length:
    - stream K/V tiles through the QK and PV loops instead of holding an entire
      prior segment in SBUF;
    - allocate MM1/MM2 scratch per Q group or a small group window instead of
      block_dim=[num_grps] for all 3072 active tokens;
    - keep only the running softmax stats/output persistent across segments.
  This is a second-stage kernel rewrite. The current kernel fixed the head_dim
  problem but is not production-ready for pfx256 because of SBUF pressure.

Verification:
  The compile failed. No artifact was produced. The heartbeat monitor was
  stopped after recording this failure.
```

### pfx256 Kernel Rewrite: Active CTE Streaming + Q-Pack Cap

```text
What failed:
  The first qwen_segcte256 kernel fixed head_dim=256 front-end legality, but
  full pfx256 compile failed because the context CTE kernel needed an illegal
  live SBUF allocation:
    {scratch_sb_for_inst__I-361}@SB<0,0>(128x402724)

How we got there:
  Host:
    ubuntu@16.51.90.254
  Branch:
    codex/full-fp8-qwen36
  Files:
    src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/fused_segmented_attention_256.py
    src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/attention_segmented_cte_256.py
  Target compile shape:
    CTE bucket 3072, prefix bucket 262144, PA blocks 1024, head_dim 256,
    FP8 full model, hybrid APC, segmented CTE backend.

Errors encountered while fixing:
  1. NKI front-end rejected list comprehensions inside the kernel.
     Command:
       Remote inline compile_kernel_to_nir probe with
       q=(2,256,256), k/v=(8,1,256,256), prior_seg_size=512.
     Exact pattern:
       unsupported expression on list comprehensions for mm1_masked_row,
       exp_sb_row, mm1_copy_row, mm1_affine_select_output_row, exp_tp_row,
       and _repeat_ref.
     Root cause:
       NKI kernels do not accept those Python list-comprehension expressions.
     Fix:
       Replaced each list comprehension with an explicit for-loop and append.
     Verification:
       The same BIR probe passed:
         BIR_SMALL_OK
         sb_scratch_sizes=[30592, 30592]
         psum_scratch_sizes=[9216, 9216]

  2. First active-streaming BIR hit an out-of-bound tensor access.
     Command:
       Remote inline compile_kernel_to_nir probe with
       q=(2,3072,256), k/v=(1024,1,256,256), block_tables=(1,1024),
       prior_seg_size=512.
     Exact error:
       assertion failed: Out-of-bound access for tensor `unnamed` on dimension
       1: index 1 exceed dimension size of 1.
       Called from fused_segmented_attention_256.py in _exp_impl().
     Root cause / hypothesis:
       The active stream allocated exp/running partial-sum columns for one
       512-token chunk, but ac.seqlen_k_active_updated still described the full
       3072-token active range, so _exp_impl tried to address chunk index 1 in
       a one-column buffer.
     Fix:
       Rebuild ac/atp per active stream chunk with
       seqlen_k_active_updated=next_section_offset_active, while preserving the
       global K position through SectionParams.kv_section_idx.
     Verification:
       The q=3072, segment=512 BIR probe advanced past _exp_impl and compiled.

  3. A docs/inspection helper import failed while probing NKI internals.
     Command:
       Import nki.framework.torch_xla in
       /home/ubuntu/venvs/neuron_230_segmented_cte.
     Exact error:
       FileNotFoundError: [Errno 2] No such file or directory:
       'libneuronpjrt-path'
     Root cause / hypothesis:
       Importing torch_xla through the overlay venv initialized torch_neuronx
       without the base Neuron runtime path.
     Fix:
       Avoid that inspection path for BIR probes; import NkiTensor from
       nki.language.tensor and shared_hbm from nki.language.buffers.
     Verification:
       BIR probes compiled with the direct NKI imports.

Fix implemented:
  The robust simple fix is not a larger prefix segment. It is a smaller live
  working set:
    - alias per-Q-group temporary SBUF buffers to one reusable group window;
    - stream active CTE K/V through the same bounded K/V SBUF window used by
      prior-prefix segments;
    - keep only running max/sum/output persistent across active/prior segments;
    - cap Q group packing to 4 groups for head_dim=256.

  The compile must use:
    PREFIX_CTE_ATTENTION_BACKEND=segmented_cte
    PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512
    CTE_BUCKETS_STR=3072
    PAIR_ARGS_STR=3072:262144
    PREFIX_BUCKETS_STR=262144
    PA_NUM_BLOCKS=1024

Verification:
  Local syntax:
    python3 -m py_compile \
      src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/fused_segmented_attention_256.py \
      src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/attention_segmented_cte_256.py

  Remote syntax:
    REMOTE_PYCOMPILE_OK on ubuntu@16.51.90.254 under
    /home/ubuntu/venvs/neuron_230_segmented_cte.

  Remote BIR scratch results after the rewrite:
    q=3072, prior_seg_size=512:
      BIR_STREAMACTIVE_Q3072_SEG512_QPACK4_OK
      sb_scratch_sizes=[31360, 31360]
      psum_scratch_sizes=[9216, 9216]

    q=3072, prior_seg_size=1024:
      sb_scratch_sizes=[35488, 35488]

    q=3072, prior_seg_size=2048:
      sb_scratch_sizes=[43680, 43680]

    q=3072, prior_seg_size=4096:
      sb_scratch_sizes=[60096, 60096]

Conclusion:
  For Trn2 head_dim=256 with the current NKI layout, prior_seg_size=512 is the
  only verified segment size under the documented SBUF free-dimension limit
  of 32767. The previous 32k segment path and the 1024+ segment probes remain
  unsafe. Full model compile and runtime validation are still required before
  marking the pfx256 artifact production-ready.
```

### pfx256 segcte512stream Full Compile Launched

```text
What changed:
  Launched the full model compile using the verified active-streaming kernel
  shape instead of the failed 32k-segment kernel.

Host:
  ubuntu@16.51.90.254

Compile PID:
  84525

Artifact target:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1024_slots64_tkg262144_async_20260526T195604Z

Workdir:
  /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_pfx256k_segcte512stream_qpack4_cte3072_pfx256k_pa1024_tkg262144_20260526T195604Z

Log:
  /home/ubuntu/validation_logs/fp8_256k/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1024_slots64_tkg262144_async_20260526T195604Z_compile.log

PID file:
  /home/ubuntu/validation_logs/fp8_256k/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1024_slots64_tkg262144_async_20260526T195604Z_compile.pid

Inputs / flags:
  TIER_NAME=pfx256k_segcte512stream_qpack4
  PREFIX_BUCKETS_STR=262144
  PAIR_ARGS_STR=3072:262144
  CTE_BUCKETS_STR=3072
  TKG_BUCKETS_STR=262144
  SEQ_LEN=262144
  MAX_CONTEXT_LENGTH=262144
  PA_NUM_BLOCKS=1024
  OMIT_ZERO_PREFIX_PAIR=1
  PREFIX_CTE_ATTENTION_BACKEND=segmented_cte
  PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512
  NEURON_VENV=/home/ubuntu/venvs/neuron_230_segmented_cte
  BASE_NEURON_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16
  NKI_LIBRARY_SRC=/home/ubuntu/nki-library-2.30/src/nkilib_src

Initial status:
  HLO generation completed for context_encoding_model and token_generation_model.
  neuronx-cc started compiling token_generation_model.

Profiling note:
  NKI profiling cannot replace compile preflight because it requires an
  executed NEFF/NTFF pair. The pre-compile gate for this issue is BIR scratch
  sizing. After COMPILE_DONE, runtime validation should be run with:
    NEURON_RT_INSPECT_ENABLE=1
    NEURON_RT_INSPECT_DEVICE_PROFILE=1
    NEURON_RT_INSPECT_OUTPUT_DIR=<validation output>/inspect
  Then identify the qwen_segcte256 NEFF, capture with profile-nth-exec=2, and
  save neuron-profile/neuron-explorer summary-json beside the cold/warm
  prefill, TTFT, TPOT, and memory results.
```

### Runtime Validation Attempt with Device Profiling Failed

```text
What failed:
  First pfx256 segcte512stream runtime validation on the TRN2 validation host
  failed during context_encoding_model NEFF load.

How it failed:
  Host:
    ubuntu@16.50.61.215
  Output root:
    /home/ubuntu/validation_logs/fp8_256k/pfx256_segcte512_runtime_20260526T202502Z
  Phase:
    pfx256k_prefill
  Status:
    pfx256k_prefill.status = 1
  Exact runtime errors:
    Failed to allocate 1.000GB (alignment: 4.000MB, usage: shared scratchpad)
    on ND 0:NC 2
    device profiling is enabled, which requires 2.348GB of HBM per NC. Either
    reduce the the model size to enable profiling, or disable profiling.
    Failed to allocate virtual scratchpad!
    Failed to load NN:
    /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_pfx256k_segcte512stream_qpack4_cte3072_pfx256k_pa1024_tkg262144_20260526T195604Z/context_encoding_model/_tp0_bk0/model.MODULE_dc595ea41a524c32e935+86f42f0e.neff
    Failure: NRT_RESOURCE in nrt_load_util
    RuntimeError: Engine core initialization failed.

How we got there:
  Artifact:
    /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1024_slots64_tkg262144_async_20260526T195604Z
  Validation flags:
    seq_len=262144
    pa_num_blocks=1024
    cte_buckets=3072
    token_generation_buckets=262144
    context_encoding_bucket_pairs=3072:262144
    max_tokens=1
    length=261888
  Profiling environment enabled:
    NEURON_RT_INSPECT_ENABLE=1
    NEURON_RT_INSPECT_DEVICE_PROFILE=1
    NEURON_RT_INSPECT_OUTPUT_DIR=/home/ubuntu/validation_logs/fp8_256k/pfx256_segcte512_runtime_20260526T202502Z/inspect

Memory evidence:
  Runtime table for the failing HBM group showed:
    Model tensors: 12.052GB
    Shared scratchpad: 6.000GB
    Profiler buffers: 4.758GB total, 2.379GB per NC
    Total shown on HBM group: 22.814GB

Root cause / hypothesis:
  Proven:
    Device profiling itself adds enough HBM pressure to prevent the 256K context
    NEFF from loading. The error explicitly names profiler buffers and says to
    disable profiling or reduce model size.
  Not proven:
    This does not prove the artifact fails without profiling. The profiler
    overhead is the immediate blocker for this attempt.

Fix / mitigation:
  Rerun the same pfx256 validation without NEURON_RT_INSPECT_DEVICE_PROFILE.
  Keep memory sampling enabled via neuron_memory_sampler. If runtime validation
  passes, profile a smaller context/shorter profile variant or capture profiling
  from a reduced-shape NEFF because full 256K device profiling does not fit.

Verification:
  Pending rerun without device profiling.
```

### Runtime Validation Without Device Profiling Failed with DGE OOB

```text
What failed:
  The no-profile pfx256 segcte512stream runtime validation failed during the
  261888-token context prefill execution after the artifact loaded.

How it failed:
  Host:
    ubuntu@16.50.61.215
  Output root:
    /home/ubuntu/validation_logs/fp8_256k/pfx256_segcte512_runtime_noprofile_20260526T202721Z
  Wrapper PID:
    21303
  Context sweep PID:
    21309
  Phase:
    pfx256k_prefill
  Log:
    /home/ubuntu/validation_logs/fp8_256k/pfx256_segcte512_runtime_noprofile_20260526T202721Z/pfx256k_prefill.log
  Exact runtime errors:
    TDRV:exec_process_custom_notification nd0:nc2:h_model.id1006:
    Received notification generated at runtime: failed to run scatter/gather
    (indirect memory copy via scalar DGE), due to out-of-bound access.
    model name =
    /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_pfx256k_segcte512stream_qpack4_cte3072_pfx256k_pa1024_tkg262144_20260526T195604Z/context_encoding_model/_tp0_bk0/model.MODULE_dc595ea41a524c32e935+86f42f0e.neff.

    NMGR:kmgr_exec_worker_do_work Async request 88 failed for model
    .../context_encoding_model/_tp0_bk0/model.MODULE_dc595ea41a524c32e935+86f42f0e.neff
    on vnc 1 with status 1006

    NMGR:kmgr_async_exec_default_exec_status_callback Exec id 88 for model
    10006 on worker 1 failed with fatal status 1006... aborting.

    /opt/workspace/KaenaRuntime/kmgr/kmgr_async_exec.cc:34:
    void kmgr_async_exec_default_exec_status_callback(...):
    Assertion `0' failed.

    ERROR Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.

How we got there:
  Artifact:
    /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1024_slots64_tkg262144_async_20260526T195604Z
  Runtime flags:
    seq_len=262144
    max_model_len=262144
    pa_num_blocks=1024
    block_size=256
    gdn_checkpoint_interval=256
    max_gdn_checkpoint_slots=64
    cte_buckets=3072
    token_generation_buckets=262144
    context_encoding_bucket_pairs=3072:262144
    lengths=261888
    max_tokens=1
    suffix_tokens=16
    require_real_tokens=true
  Runtime environment:
    NEURON_RT_INSPECT_ENABLE=0
    NEURON_RT_INSPECT_DEVICE_PROFILE unset
  Kernel/compile path:
    PREFIX_CTE_ATTENTION_BACKEND=segmented_cte
    PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512
    TIER_NAME=pfx256k_segcte512stream_qpack4

Memory evidence:
  The artifact loaded before execution. The memory sampler showed the runtime
  had dropped to present-only bookkeeping after the fatal DGE error, not a
  NRT_RESOURCE allocation failure:
    latest host RSS: about 1.08 GiB for qwen36_hybrid_apc_context_sweep.py
    neuron present: about 6.46GB total
    latest total bytes: 0
  This separates this failure from the earlier device-profiling HBM failure.

Root cause / hypothesis:
  Proven:
    The 256K pfx artifact compiles and loads without device profiling, but the
    context_encoding_model NEFF issues an out-of-bound scalar DGE access during
    execution.
  Best current hypothesis:
    The qwen_segcte256 segmented CTE kernel has a runtime address calculation
    bug for the actual long-prefix path. The likely fault is in the mapping of
    block-table, prior segment, active segment, or kv_section_idx offsets for
    the 261888-token request. BIR scratch sizing and compile legality did not
    catch it because the address goes out of range only with real runtime block
    tables and long-prefix execution.

Fix / mitigation applied:
  Stopped the failed validation wrapper and context sweep on ubuntu@16.50.61.215
  to free Neuron resources:
    kill -TERM 21309 / children through wrapper PID 21303, then kill stale
    sampler PID 21308.

Next mitigation:
  Do not retry the same pfx256 segcte512stream artifact as production. Build a
  targeted runtime/addressing debug path for qwen_segcte256:
    1. Reproduce with a smaller debug prefix artifact or a reduced long-prefix
       request that still uses segmented_cte address math.
    2. Instrument or assert the block table index, prior segment start, active
       stream start, kv_section_idx, and max addressed block before DGE loads.
    3. Patch the segmented CTE offset mapping, then rerun BIR preflight and a
       no-profile runtime prefill before enabling any profiling.

Verification:
  Validation did not complete. No prefill, TTFT, TPOT, or chat metrics were
  produced for this artifact.
```

### Null-Block PA Count Mismatch Hypothesis for DGE OOB

```text
Additional evidence:
  The failing pfx256 artifact was compiled with:
    pa_num_blocks=1024
    block_size=256
    max_context_length=262144
  Runtime vLLM logs showed:
    Adding 1 to num_gpu_blocks_override (1024 -> 1025) to account for null
    block allocation
    User provided pa_num_blocks (1024) matching original
    --num-gpu-blocks-override intent. Incrementing pa_num_blocks to 1025 to
    match the increment for a null block in vllm.

Why this matters:
  For vLLM, the user-intended usable block count for 256K at block size 256 is
  1024. vLLM adds one reserved null block, so the physical block-KV cache needs
  1025 blocks. The current artifact was compiled as pa1024, so a block-table
  value of 1024 can be legal to vLLM but out of bounds for the compiled NEFF's
  raw block-KV cache. That matches the observed scalar-DGE OOB in
  context_encoding_model.

Root cause / hypothesis update:
  Best current hypothesis is now a PA physical-block sizing mismatch, not
  scratch/HBM pressure. The qwen_segcte256 kernel may still need address tests,
  but the first robust/simple fix to try is compiling the artifact with 1025
  physical PA blocks while running vLLM with 1024 usable blocks.

Fix applied to validation scripts:
  Updated validation_scripts/qwen36_hybrid_apc_context_sweep.py and
  validation_scripts/qwen36_offline_decode_bench.py so artifact pa_num_blocks
  is treated as physical block count. When the artifact uses block KV or prefix
  caching and has more blocks than the minimum usable request, validation passes
  artifact_pa_num_blocks - 1 as vLLM's num_gpu_blocks_override.

Next mitigation:
  Compile a replacement pfx256 segcte512stream artifact with:
    PA_NUM_BLOCKS=1025
    PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512
    CTE_BUCKETS_STR=3072
    PAIR_ARGS_STR=3072:262144
    PREFIX_BUCKETS_STR=262144
  Then validate it with user-usable pa override 1024 so vLLM adds the null
  block back to 1025.
```

### PA1025 Relaunch Setup Errors and Correction

```text
What failed:
  First corrected relaunch attempt on ubuntu@16.50.61.215 used:
    TIER_NAME=pfx256k_segcte512stream_qpack4_pafix
    PA_NUM_BLOCKS=1025
    PREFIX_CTE_ATTENTION_BACKEND=segmented_cte
    PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512
    PAIR_ARGS_STR=3072:262144

How it failed:
  The remote helper script was stale and hardcoded:
    --pa-num-blocks 1024
  The resulting process PID 28426 was running a pa1024 compile even though the
  environment requested PA_NUM_BLOCKS=1025. The log showed:
    CONTEXT_TRACE_SHAPE ... "pa_num_blocks": 1024, "pa_min_blocks": 1024,
    "pa_headroom_blocks": 0

How we got there:
  The local helper had already been updated to include _pa${PA_NUM_BLOCKS} in
  the artifact name and to pass --pa-num-blocks "${PA_NUM_BLOCKS}", but that
  helper had not been synced to ubuntu@16.50.61.215.

Fix / mitigation applied:
  Stopped PID 28426 before useful compilation work continued, synced the local
  helper to:
    /home/ubuntu/inferentia-gdn-fused-noclamp-4340808/tmp_compile_qwen256k_fp8_full_prod_prefix_tier_hostlogits.sh
  Verified the synced helper contains:
    BASE=..._pa${PA_NUM_BLOCKS}_...
    --pa-num-blocks "${PA_NUM_BLOCKS}"

Verification:
  Relaunch produced a _pa1025_ artifact name.
```

```text
What failed:
  The next relaunch on ubuntu@16.50.61.215 failed before compilation started.

Exact error:
  qwen36_27b_compile_fp8.py: error: unrecognized arguments:
    --omit-zero-prefix-pair
    --prefix-cte-attention-backend segmented_cte
    --prefix-cte-attention-segment-size 512

How we got there:
  Remote repo branch was codex/full-fp8-qwen36 at 03e7e3a, but
  contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py was
  stale relative to the local full-FP8 branch work. The runtime modules already
  had segmented_cte support, but the compile entrypoint did not expose the
  required CLI flags.

Fix / mitigation applied:
  Synced the local compile entrypoint to:
    /home/ubuntu/inferentia-gdn-fused-noclamp-4340808/contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py
  Verified with:
    python3 -m py_compile contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py
    grep for omit-zero-prefix, prefix-cte-attention, and segmented_cte.

Verification:
  Corrected relaunch started as PID 29224 with:
    artifact:
      /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_pafix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260526T205447Z
    log:
      /home/ubuntu/validation_logs/fp8_256k/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_pafix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260526T205447Z_compile.log
  The compile log now shows:
    CONTEXT_TRACE_SHAPE ... "pa_num_blocks": 1025,
    "pa_min_blocks": 1024,
    "pa_headroom_blocks": 1,
    "prefix_cte_attention_backend": "segmented_cte",
    "prefix_cte_attention_segment_size": 512
  and then enters HLO generation for context_encoding_model.
```

```text
What failed:
  The PA1025 relaunch at 20260526T205447Z reached HLO tracing but failed in
  Python before neuronx-cc compilation.

Exact error:
  AttributeError: 'QwenHybridBlockKVCacheManager' object has no attribute
  'get_raw_kv_by_layer_id'. Did you mean: 'get_kv_by_layer_id'?

Evidence:
  PID/log:
    /home/ubuntu/validation_logs/fp8_256k/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_pafix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260526T205447Z_compile.pid
    /home/ubuntu/validation_logs/fp8_256k/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_pafix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260526T205447Z_compile.log
  Stack:
    modeling_qwen35.py:get_cache -> self.get_raw_kv_by_layer_id(...)
    torch.nn.Module.__getattr__ raised AttributeError.

How we got there:
  The Qwen model file and attention path expected the newer raw block-KV cache
  accessor, but the remote
  src/neuronx_distributed_inference/modules/kvcache/block_kv_cache_manager.py
  had not been synced with the matching full-FP8 branch changes.

Fix / mitigation applied:
  Synced the matching local cache/runtime files to ubuntu@16.50.61.215:
    src/neuronx_distributed_inference/modules/kvcache/block_kv_cache_manager.py
    src/neuronx_distributed_inference/models/config.py
    src/neuronx_distributed_inference/models/model_wrapper.py
    src/neuronx_distributed_inference/modules/async_execution.py
    src/neuronx_distributed_inference/modules/autobucketing.py
    src/neuronx_distributed_inference/modules/attention/attention_base.py
    src/neuronx_distributed_inference/modules/attention/nki_kernels/
  Verified with py_compile and confirmed:
    def get_raw_kv_by_layer_id(self, idx, kvcache_buffer=None, **kwargs)

Verification:
  Relaunched as PID 30174 with artifact:
    /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_pafix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260526T205813Z
  Current log shows:
    CONTEXT_TRACE_SHAPE ... "pa_num_blocks": 1025,
    "pa_headroom_blocks": 1,
    "prefix_cte_attention_backend": "segmented_cte",
    "prefix_cte_attention_segment_size": 512
    Finished generating HLO for context_encoding_model
    Started loading module token_generation_model
```

```text
Operator error:
  During the remote sync fix, one multi-file scp command targeted the attention
  directory for all source files. It created extra inert copies under:
    src/neuronx_distributed_inference/modules/attention/config.py
    src/neuronx_distributed_inference/modules/attention/model_wrapper.py
    src/neuronx_distributed_inference/modules/attention/async_execution.py
    src/neuronx_distributed_inference/modules/attention/autobucketing.py

Impact / hypothesis:
  These files are not imported by the current attention package path, but they
  are remote workspace clutter and should be removed after explicit approval or
  during the next cleanup pass.

Fix / mitigation applied:
  Re-copied each file to its correct destination. No compile path depends on
  the accidental files.
```

### PA1025 pfx256 Runtime Validation Failed with DGE OOB

```text
What failed:
  No-device-profile runtime validation of the corrected PA1025 pfx256k
  segmented CTE artifact on TRN2 ubuntu@16.50.61.215.

Artifact:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_pafix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260526T205813Z

Validation output root:
  /home/ubuntu/validation_logs/fp8_256k/pfx256_pa1025_runtime_noprofile_20260527T034323Z

Inputs / flags:
  validation_scripts/qwen36_hybrid_apc_context_sweep.py
    --lengths 261888
    --max-tokens 16
    --suffix-tokens 16
    --seq-len 262144
    --max-model-len 262144
    --cte-buckets 3072
    --context-encoding-bucket-pairs 3072:262144
    --token-generation-buckets 262144
    --async-mode
    --block-size 256
    --gdn-checkpoint-interval 256
    --max-gdn-checkpoint-slots 64
    --gdn-recurrent-cache-dtype float32
    --gdn-conv-cache-dtype bfloat16
    --require-real-tokens
  Device profiling was explicitly disabled:
    unset NEURON_RT_INSPECT_ENABLE
    unset NEURON_RT_INSPECT_DEVICE_PROFILE
    unset NEURON_RT_INSPECT_OUTPUT_DIR

Observed runtime context:
  Engine loaded the compiled artifact successfully.
  vLLM reported:
    GPU KV cache size: 262,400 tokens
    Maximum concurrency for 262,144 tokens per request: 1.00x
  Prompt execution started for the 261888-token request.

Exact error:
  At 2026-05-27T03:53:20Z, the context_encoding_model NEFF repeatedly emitted:
    TDRV:exec_process_custom_notification ... failed to run scatter/gather
    (indirect memory copy via scalar DGE), due to out-of-bound access.
    model name =
    /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_pfx256k_segcte512stream_qpack4_pafix_cte3072_pfx256k_pa1025_tkg262144_20260526T205813Z/context_encoding_model/_tp0_bk0/model.MODULE_30b568c5d3faaeced212+b0ee5af3.neff
  The first repeated failures were on Neuron cores nc4/nc6 and then expanded
  across other cores.

Memory evidence:
  This was not an NRT_RESOURCE/OOM failure. The memory sampler captured:
    peak_host_rss_gib: 35.310752868652344
    peak_neuron_by_category_gib.total: 157.06698608398438
    peak_neuron_by_category_gib.present: 7.32489013671875
  Note: peak_neuron_total_gib in the sampler summary sums peak/present/total
  categories and should not be used as a real HBM footprint.

Root cause / hypothesis update:
  The null-block PA mismatch was not the root cause for the 256K runtime
  failure. Compiling with 1025 physical PA blocks fixed the compile/load shape
  and provided physical capacity for the null block, but the actual long-prefix
  segmented CTE path still generates an out-of-range scalar-DGE address at
  runtime. The best current hypothesis is now a qwen_segcte256 address-mapping
  bug for the pfx256k context_encoding bucket, likely in block-table indexing,
  prior-segment offset, active-stream offset, or kv_section_idx mapping inside
  the custom segmented CTE kernel.

Fix / mitigation applied:
  Stopped the failed validation run and sampler after the DGE OOB:
    wrapper PID: 31812
    sampler PID: 31814
    context sweep PID: 31815
    EngineCore PID: 31872
  Verified those PIDs were no longer present afterward.

Remaining blocker:
  The PA1025 pfx256k artifact is not runtime-valid and is not production-ready.
  Do not run OpenAI/server TTFT/TPOT validation on this artifact until the
  segmented CTE 256K address calculation is fixed or replaced.

Next mitigation:
  Build a targeted qwen_segcte256 debug/fix path:
    1. Reproduce with a small diagnostic harness that exercises the same
       segmented CTE addressing with controlled block_table values.
    2. Add bounds checks or debug-side assertions for physical block id,
       kv_head/block offset, prior segment start, active segment start, and
       kv_section_idx before the DGE loads.
    3. Patch the qwen_segcte256 NKI address mapping, then recompile the
       pfx256k bucket and rerun no-device-profile runtime validation.
```

```text
Operator/status-check errors encountered during this validation:

1. A status check command exited 127 because it used `python` in a remote
   non-login shell where only the activated venv process had `python` on PATH.
   The validation process itself was unaffected. Mitigation: subsequent status
   parsing used `python3` or an activated venv.

2. The first cleanup command used a broad pgrep pattern:
     qwen36_hybrid_apc_context_sweep|VLLM::EngineCore|neuron_memory_sampler
   and matched its own SSH-side shell command, causing the SSH cleanup command
   to exit 255 before printing post-cleanup status. Mitigation: reran cleanup
   with explicit known PIDs 31812, 31814, 31815, and 31872, then verified they
   were no longer running.
```

### Segmented CTE Active Block-Table Fill Fix

```text
What failed:
  Follow-up investigation of the PA1025 pfx256k runtime DGE OOB found that the
  segmented CTE kernel reads the active suffix K/V from the raw paged KV cache
  through active_block_table. If active suffix logical block-table entries are
  still unset, the NKI kernel can consume an invalid block id for scalar DGE.

Evidence / how we got there:
  Artifact:
    /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_pafix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260526T205813Z
  Compile trace shape:
    context_encoding_bucket_pairs=[[3072,262144]]
    pa_num_blocks=1025
    pa_min_blocks=1024
    pa_headroom_blocks=1
    prefix_cte_attention_backend=segmented_cte
    prefix_cte_attention_segment_size=512
  Runtime loaded the artifact with pa_num_blocks=1025, then failed inside the
  context_encoding_model NEFF with:
    failed to run scatter/gather (indirect memory copy via scalar DGE),
    due to out-of-bound access
  There were no debug `pad-pre`, `pad-post`, or `qwen-cte-call` lines in the
  failed validation log because QWEN36_HYBRID_APC_DEBUG was not enabled.

Root cause / best current hypothesis:
  BlockKVCacheManager writes the active suffix K/V into the raw block cache by
  slot_mapping. The qwen_segcte256 path then reads active K/V from the raw block
  cache by active_block_table. For segmented CTE, active_block_table must contain
  physical block ids for logical active-suffix blocks as well as prefix blocks.
  If those active entries remain -1 or otherwise unset, the NKI kernel casts the
  block table to uint32 and can form a huge scalar-DGE HBM offset. That matches
  the observed runtime-only scalar DGE OOB after successful load.

Fix / mitigation applied locally:
  Patched:
    src/neuronx_distributed_inference/models/model_wrapper.py
  Added segmented-CTE-only input preprocessing in `_pad_prefix_caching_inputs`:
    - derive active logical block positions from computed_context_lens + token
      index
    - derive active physical block ids from slot_mapping // pa_block_size
    - fill those active logical block-table entries before masking/padding
    - include active tokens when sizing the segmented CTE block table
  This leaves the non-segmented attention_cte path unchanged.

Test added:
    test/unit/models/test_prefix_caching_bucket_selection.py
    test_segmented_cte_padding_fills_active_block_table_from_slots
  The focused case starts with block_table [[0, 1, 2, -1]], prefix_len=768,
  suffix_len=48, pa_block_size=256, and slot_mapping in physical block 4. The
  expected padded block table is [[0, 1, 2, 4]].

Local verification:
  Command:
    python3 -m py_compile src/neuronx_distributed_inference/models/model_wrapper.py test/unit/models/test_prefix_caching_bucket_selection.py
  Result:
    pass

Local test environment errors:
  Command:
    python3 -m pytest test/unit/models/test_prefix_caching_bucket_selection.py -q
  Result:
    exit 2 during collection
  Exact error:
    ModuleNotFoundError: No module named 'neuronx_distributed_inference'
  Mitigation:
    reran with PYTHONPATH=src

  Command:
    PYTHONPATH=src python3 -m pytest test/unit/models/test_prefix_caching_bucket_selection.py -q
  Result:
    exit 2 during collection
  Exact error:
    ModuleNotFoundError: No module named 'neuronx_distributed'
  Root cause / hypothesis:
    The local Mac environment lacks the Neuron/NxD Python dependency needed for
    this test module. This is an environment dependency issue, not a syntax
    failure; py_compile passed locally.

Next verification:
  Sync the patch to TRN2 ubuntu@16.50.61.215, run py_compile and the focused
  pytest in the Neuron venv, then rerun a no-device-profile pfx256 validation
  with QWEN36_HYBRID_APC_DEBUG=1. If the shorter debug validation passes, rerun
  the original 261888-token validation against the same compiled artifact.
```

### Active Block-Table Fill Validation Results

```text
What passed:
  Remote syntax/unit validation on TRN2 ubuntu@16.50.61.215 after syncing:
    src/neuronx_distributed_inference/models/model_wrapper.py
    test/unit/models/test_prefix_caching_bucket_selection.py
    contrib/models/Qwen3.6-27B/docs/QWEN36_FP8_TIERFIX_VALIDATION_20260526.md

Command:
  cd /home/ubuntu/inferentia-gdn-fused-noclamp-4340808
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
  PYTHONPATH=src python -m py_compile \
    src/neuronx_distributed_inference/models/model_wrapper.py \
    test/unit/models/test_prefix_caching_bucket_selection.py
  PYTHONPATH=src python -m pytest \
    test/unit/models/test_prefix_caching_bucket_selection.py -q

Result:
  35 passed, 46 warnings in 5.33s

What passed at runtime:
  Short debug validation with the PA1025 pfx256k artifact and the local
  active-block-table fill patch completed without DGE OOB.

Artifact:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_pafix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260526T205813Z

Output root:
  /home/ubuntu/validation_logs/fp8_256k/pfx256_pa1025_activeblockfix_short_20260527T_test

Inputs:
  --lengths 8192
  --max-tokens 1
  --suffix-tokens 16
  --seq-len 262144
  --max-model-len 262144
  --cte-buckets 3072
  --context-encoding-bucket-pairs 3072:262144
  --token-generation-buckets 262144
  --async-mode
  --block-size 256
  --gdn-checkpoint-interval 256
  --max-gdn-checkpoint-slots 64
  --gdn-recurrent-cache-dtype float32
  --gdn-conv-cache-dtype bfloat16
  --require-real-tokens
  QWEN36_HYBRID_APC_DEBUG=1

Short-run evidence:
  The debug trace showed the active/prefix block table now includes the active
  physical block range. Examples:
    prefix_len=6144, slot_mapping max=8447, block_table max=32
    prefix_len=6144, slot_mapping max=18687, block_table max=72
  The 8192-token run completed:
    cold elapsed: 14.63105383799848s
    warm elapsed: 4.818916980999347s
    real_tokens_passed: true

What still failed:
  Full 261888-token validation with the same artifact and patch still failed
  inside the context_encoding_model NEFF with scalar DGE OOB.

Output root:
  /home/ubuntu/validation_logs/fp8_256k/pfx256_pa1025_activeblockfix_full_20260527T0524Z

Inputs:
  --lengths 261888
  --max-tokens 16
  --suffix-tokens 16
  --seq-len 262144
  --max-model-len 262144
  --cte-buckets 3072
  --context-encoding-bucket-pairs 3072:262144
  --token-generation-buckets 262144
  --async-mode
  --block-size 256
  --gdn-checkpoint-interval 256
  --max-gdn-checkpoint-slots 64
  --gdn-recurrent-cache-dtype float32
  --gdn-conv-cache-dtype bfloat16
  --require-real-tokens
  Device profiling and QWEN36_HYBRID_APC_DEBUG were disabled for the full run.

Exact error:
  First repeated failures at run.log lines 2592+:
    2026-May-27 05:21:35.021738 ... ERROR TDRV:exec_process_custom_notification
    nd0:nc6:h_model.id1005: Received notification generated at runtime:
    failed to run scatter/gather (indirect memory copy via scalar DGE),
    due to out-of-bound access. model name =
    /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_pfx256k_segcte512stream_qpack4_pafix_cte3072_pfx256k_pa1025_tkg262144_20260526T205813Z/context_encoding_model/_tp0_bk0/model.MODULE_30b568c5d3faaeced212+b0ee5af3.neff.
  The same error appeared on nc4, nc5, nc6, nc7 and later nc0/nc1/nc2/nc3.
  The runtime also reported:
    TDRV:exec_request_process_errors [ND 0][NC 6] Out of bounds access on model ...
    NMGR:dlr_exec_wait Execution completed with err: 1006. mode->h_nn=1008, lnc=2

Core dump evidence:
  Neuron generated NRT_EXEC_OOB dumps:
    /tmp/neuron-core-dump/dt-20260527-051233-cid-d99e36ea74c263ca
      i-05d3f024966df11d5-nd0-nc4-pid-39738-tid-39861-lid-1
      i-05d3f024966df11d5-nd0-nc6-pid-39738-tid-39862-lid-2
      i-05d3f024966df11d5-nd0-nc2-pid-39738-tid-39863-lid-3

Memory evidence:
  This was not a Neuron load OOM/NRT_RESOURCE failure.
  Memory summary:
    peak_host_rss_gib: 34.55003356933594
    peak_neuron_by_category_gib.present: 6.589611053466797
    peak_neuron_by_category_gib.total: 157.06698608398438
  As before, the sampler's peak_neuron_total_gib sums sysfs categories and is
  not a single real HBM allocation.

Root cause / hypothesis update:
  The active-block-table fill is necessary and fixes a real input-prep hazard,
  but it is not sufficient for the pfx256/261888 path. The remaining scalar DGE
  OOB is likely inside qwen_segcte256 address generation for high prior segment
  indices, for example:
    - prior segment block-table offset when prefix_len approaches 256K
    - the first/last partial-prior segment around a 512-token segment boundary
    - segment index to block-table index arithmetic in the NKI kernel
    - kv_section_idx or KV-head/block offset at high logical block ids
  This is now confirmed as a kernel/addressing bug, not a PA1025 capacity issue
  and not just missing active physical block ids.

Fix / mitigation applied:
  Stopped the failed full validation and sampler:
    sampler PID: 39613
    wrapper bash PID: 39684
    context sweep PID: 39692
    EngineCore PID: 39738
  PID 39738 became a short-lived defunct EngineCore while neuron-dump wrote
  NRT_EXEC_OOB dumps. No qwen36 sweep/sampler process remained afterward.

Remaining blocker:
  The PA1025 pfx256k segmented CTE artifact is still not runtime-valid for
  261888-token / 256K-context serving. It must not be called production-ready.

Next mitigation:
  Add high-prefix debug instrumentation or a CPU/NKI address simulator for
  qwen_segcte256 and binary-search the failing prefix length with the pfx256
  artifact. The short 8K smoke is not enough; test lengths should bracket the
  failure, e.g. 32768, 65536, 131072, 196608, 229376, and 261888, with debug
  enabled only around the final failing CTE chunk.
```

```text
Operator errors during the active-block-table validation:

1. The first full-run wrapper backgrounded too broad a shell command and lost
   ROOT/PATH state. It printed:
     tee: /run.log: Permission denied
     bash: line 1: python: command not found
   The validation did not start. An orphaned sampler PID 39303 was killed.
   Mitigation: reran with explicit absolute output paths and separate sampler
   launch.

2. The first separate sampler launch quoted ROOT incorrectly inside nested
   local/remote shell expansion. It printed:
     mkdir: missing operand
     bash: line 1: /sampler.pid: Permission denied
   No validation ran from that command. Mitigation: relaunched sampler with
   literal absolute paths.
```

### Root Cause Found: Final Partial Active Chunk Reads Past Block Table

```text
What failed:
  The PA1025 pfx256k artifact still emitted scalar DGE OOB at 261888 tokens even
  after the Python active-block-table fill. The 8192-token smoke passed, which
  meant the remaining bug was specific to high-prefix / end-of-context address
  generation.

Code path:
  src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/
    attention_segmented_cte_256.py
    fused_segmented_attention_256.py

Root cause:
  In qwen_segcte256 active streaming, the compiled 3072-token CTE bucket is
  split into six 512-token active stream sections:
    active_stream_tokens = 512
    num_active_stream_sections = ceil(3072 / 512) = 6
    num_blocks_per_active_stream = 512 / 256 = 2

  For the final real chunk of a 261888-token prompt:
    prior_tokens = 261120
    real active_len = 768
    active_block_offset = prior_tokens // 256 = 1020

  The compiled active-stream loop still loads all six bucket sections, so the
  block-table offsets are:
    1020, 1022, 1024, 1026, 1028, 1030

  A real pfx256 block_table has 1024 entries. The older internal padding only
  padded to 1026 entries for the prior-segment one-past read:
    padded_width = (1024 // 2 + 1) * 2 = 1026

  Therefore active sections 4 and 5 can read block-table offsets 1028/1030,
  outside the internally padded table. That exactly matches AWS Neuron's DGE
  docs: scalar/vector DGE offsets must still resolve to valid tensor addresses,
  otherwise runtime reports out-of-bound scatter/gather.

Fix applied locally:
  Patched `attention_segmented_cte_256.py` to pad the internal block table for
  both hazards:
    padded_width_for_prior = one extra prior segment
    padded_width_for_active_stream = max_blocks_per_seq + seqlen_q // block_size
    padded_width = rounded max of both

  For pfx256 cte3072 this pads from 1024 to 1036 entries, so out-of-range
  compiled active-stream sections read zero block ids from the padded tail
  instead of DGE-reading past the block table. Block id 0 is the existing null
  block, so this matches the intended padding semantics.

Why this aligns with docs:
  The NKI/DGE docs allow dynamic/scalar-offset DMA patterns, but the program is
  responsible for keeping the dynamic address inside the tensor. Padding the
  source table before the scalar DGE access is the simple robust fix; relying on
  masks after the DMA is too late because the OOB happens during the DMA
  descriptor execution.

Verification so far:
  Local syntax:
    python3 -m py_compile \
      src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/attention_segmented_cte_256.py
    result: pass

Remaining work:
  Sync to TRN2, run remote py_compile, recompile the pfx256 segmented CTE
  artifact, then rerun the 261888-token no-device-profile validation. The old
  PA1025 artifact cannot be fixed in place because this change is inside the
  compiled NKI kernel.
```

### Bound-Fix PFX256 Runtime Validation Passed

```text
Artifact:
  /mnt/trainium_artifacts/qwen_artifacts/
    qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_boundfix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260527T052822Z

Validation root:
  /home/ubuntu/validation_logs/fp8_256k/pfx256_boundfix_runtime_20260527T0552Z

Inputs:
  length: 261888 prompt tokens
  max_tokens: 16
  seq_len/max_model_len: 262144
  cte/prefix pair: 3072:262144
  token_generation_bucket: 262144
  pa_num_blocks: 1025
  backend: segmented_cte
  segment_size: 512
  profiling: disabled

Result:
  passed: true
  real_tokens_passed: true
  token_range_passed: true
  non_dummy_generated_token_count: 48
  unique_generated_token_count: 41

Timings:
  cold prefill+decode: 551.9684613459976s
  prefix warmup prefill+decode: 551.5538088760004s
  measured warm/refill+decode: 10.758389775000978s
  cold effective prompt throughput: 474.46189110402327 tokens/s
  warm/refill effective prompt throughput: 24342.67631839693 tokens/s

Memory summary:
  peak_host_rss_gib: 35.314510345458984
  peak_neuron_by_category_gib.present: 6.291294097900391
  peak_neuron_by_category_gib.total: 159.61318969726562

Notes:
  The sysfs Neuron memory sampler aggregates categories and logical cores; the
  `present` category is the most useful live resident counter from this sampler.
  The larger `total` and `peak` aggregates are not single-device HBM usage.

Monitor-side error encountered:
  Command:
    ssh ... 'python - <<PY ...'
  Failure:
    bash: line 1: python: command not found
  Cause:
    The remote default PATH has python3 but not python.
  Fix:
    Reran the final JSON/memory parser with python3.
  Verification:
    Parser completed and reported the metrics above.
```

### Production-Readiness Suite Launch

```text
Suite root:
  /home/ubuntu/validation_logs/fp8_256k/prod_readiness_boundfix_20260527T064706Z

Suite launcher:
  tmp_run_qwen256k_fp8_prod_readiness_boundfix.sh

Suite PID:
  49273

Planned stages:
  1. offline_repeat_256k:
       lengths 261888,261888,261888
       validates repeated full-context cold/warmup/warm-refill behavior.
  2. offline_context_sweep:
       lengths 32768,65536,131072,261888
       validates shorter contexts and one more full-context pass.
  3. server startup:
       vLLM backend on 127.0.0.1:8001
       guarded chat proxy on 127.0.0.1:8000
  4. server_chat_apc:
       OpenAI-compatible chat, multi-turn, semantic smoke, partial/warm APC.
  5. server_context_bench:
       OpenAI-compatible chat TTFT/TPOT sweep at
       32768,65536,131072,261888 prompt-token targets.

Launch error encountered:
  Initial attempt used one very long inline ssh command to create and launch the
  suite. The local zsh shell rejected it before any remote execution:
    zsh:166: unmatched "

How we got there:
  The inline command nested local zsh quoting, remote bash quoting, heredoc
  content, and JSON-like grep patterns in one shell string.

Root cause:
  Operator-side brittle quoting in the launch command, not a Neuron/runtime
  failure and not a remote host issue.

Fix:
  Added `tmp_run_qwen256k_fp8_prod_readiness_boundfix.sh`, copied it to the
  TRN2 repo, and launched it with:
    nohup env ROOT=/home/ubuntu/validation_logs/fp8_256k/prod_readiness_boundfix_20260527T064706Z \
      ./tmp_run_qwen256k_fp8_prod_readiness_boundfix.sh

Verification:
  Suite log shows:
    [2026-05-27T06:47:06+00:00] START offline_repeat_256k
  PID 49273 is running. The first stage loaded the artifact and began the
  offline repeated 256K validation.
```

### Production-Readiness Suite Results And Harness Fix

```text
Primary suite root:
  /home/ubuntu/validation_logs/fp8_256k/prod_readiness_boundfix_20260527T064706Z

Corrected server context bench root:
  /home/ubuntu/validation_logs/fp8_256k/prod_readiness_boundfix_contextbench_capped_20260527T083606Z

Offline repeated 256K:
  passed: true
  rows:
    261888 tokens: cold 551.9598s, warm/refill 10.7489s, warm TPS 24364.23
    261888 tokens: cold 551.4770s, warm/refill 10.7508s, warm TPS 24359.92
    261888 tokens: cold 551.4248s, warm/refill 10.7529s, warm TPS 24355.10

Offline context sweep:
  passed: true
  rows:
    32768 tokens:  cold 56.9797s,  warm/refill 7.3258s,  warm TPS 4472.95
    65536 tokens:  cold 115.5351s, warm/refill 7.8311s,  warm TPS 8368.69
    131072 tokens: cold 241.2197s, warm/refill 8.7931s,  warm TPS 14906.21
    261888 tokens: cold 551.6429s, warm/refill 10.7701s, warm TPS 24316.17

Server chat/APC:
  all_status_ok: true
  semantic_smoke_passed: true
  warm_full_initial_seconds: 54.9422
  warm_full_repeat_avg_seconds: 5.7894
  warm_full_speedup: 9.4901
  partial_cold_reference_seconds: 55.8281
  partial_warm_beta_avg_seconds: 7.3978
  partial_reference_speedup: 7.5466
  multi_turn_repeat_exact_text: true
  multi_turn_avg_seconds: 4.8914

Original server-context-bench error:
  What failed:
    server_context_bench stage in the primary suite.
  How it failed:
    The bench process stopped making progress after this tokenizer warning:
      Token indices sequence length is longer than the specified maximum sequence length for this model (426209 > 262144). Running this sequence through the model will result in indexing errors
    It had already completed 32768, 65536, and 131072 rows successfully.
    The stuck child was manually terminated, so the suite recorded:
      [2026-05-27T08:33:26+00:00] END server_context_bench rc=143
  How we got there:
    validation_scripts/qwen36_chat_completion_context_bench.py was run with:
      --lengths 32768,65536,131072,261888 --turns 8 --repeats 1
    The old prompt builder doubled filler repetitions until it exceeded the
    target, which created a transient 426209-token chat-template probe for the
    261888-token target.
  Root cause:
    Validation harness bug, not a Neuron runtime/model failure. The prompt
    builder used exponential overshoot probes that are too large near the
    262144-token model limit.
  Fix:
    Updated _make_messages in validation_scripts/qwen36_chat_completion_context_bench.py
    to estimate filler repetitions from one-repeat token delta and correct
    downward instead of doubling past the target. Synced the fixed script to
    TRN2 and reran only server startup + server_context_bench.
  Verification:
    python3 -m py_compile validation_scripts/qwen36_chat_completion_context_bench.py
    passed locally.
    Corrected context bench passed:
      32768 target:  prompt 32764,  status 200, TTFT 57.0513s,  completion 16
      65536 target:  prompt 65524,  status 200, TTFT 66.4657s,  completion 16
      131072 target: prompt 131070, status 200, TTFT 132.5198s, completion 9
      261888 target: prompt 261876, status 200, TTFT 319.2527s, completion 16

Memory summaries:
  Primary server peak host RSS: 35.3254 GiB
  Corrected server peak host RSS: 35.3525 GiB
  Corrected server live Neuron `present` peak: 10.7526 GiB from sampler

Monitor/tooling errors encountered:
  write_stdin failed when attempting to interrupt old tail sessions:
    stdin is closed for this session; rerun exec_command with tty=true to keep stdin open
  This was a local monitoring-tool state issue. It did not affect remote
  validation. The old remote suite had already exited and the corrected rerun
  used a new tail session.

  A local sandboxed ps probe failed:
    zsh:1: operation not permitted: ps
  This was local sandboxing, not a repo or remote failure. Remote process
  checks were done through ssh instead.

  The command used to stop the completed remote live tail returned ssh exit
  code 255 with no stderr:
    ssh ... 'pkill -f "tail -n 80 -F .*prod_readiness_boundfix_contextbench_capped_20260527T083606Z" || true'
  Hypothesis:
    pkill matched and terminated the remote tail/ssh session while the command
    was still attached, so ssh reported disconnect as 255.
  Verification:
    The tail session then reported `Process exited with code 255`; validation
    had already completed and the server had already shut down cleanly.
```
