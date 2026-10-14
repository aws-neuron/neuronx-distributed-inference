# vLLM PR #42270 model files (MiMo-V2.5 FP8 fused-qkv loader fix)

`mimo_v2.py` and `mimo_v2_mtp.py` from vLLM PR
[#42270](https://github.com/vllm-project/vllm/pull/42270)
("[models] MiMo V2: Pro fused-QKV FP8 loader + fix SWA wrong-data on V2.5 base"),
branch `amd-satre:mimo-v25-pro-fp8-qkv-loader`, fetched 2026-07-21.

## Why these are here

Stock vLLM (0.25.1 **and** nightly) crashes loading MiMo-V2.5 FP8 in
`_shard_fp8_qkv_proj` (`RuntimeError: size of tensor a (1856) must match b
(1792)`): a sliding-window layer's fused-qkv group is 1856 rows = 14.5 128-row
FP8 scale blocks, so per-KV-head scale slicing misaligns. This PR replaces the
loader with `_mimo_v2_copy_paired_qkv_fp8`, which "dequantizes and requantizes
local shards for TP configurations that cut through 128-row FP8 scale blocks,
such as TP4". As of 2026-07 the PR is still **open** (merge conflicts / awaiting
review), so it is not in any released image.

## Verified (2026-07-21, single-node 8xH100, nightly + these files)

Overwriting nightly's two files with these makes `--tensor-parallel-size 8
--enable-expert-parallel` (the reference command) **load and generate correct
output** — coherent short answers and a full 500-token B-tree explanation with
no gibberish/collapse. `run_vllm_h100.sh` bind-mounts this dir and `cp`s the
files in at container start. Delete this dir and revert run_vllm_h100.sh once
the PR merges into the image.
