# Contrib Model: MiMo-V2.5-Pro

NeuronX Distributed Inference implementation of [XiaomiMiMo/MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro).

## Model Information

- **HuggingFace ID:** `XiaomiMiMo/MiMo-V2.5-Pro`
- **Model Type:** Decoder-only MoE transformer with hybrid attention
- **Architecture:** Custom MoE with full + sliding window attention
- **License:** Check HuggingFace model card

## Architecture Details

| Parameter | Value |
|-----------|-------|
| Hidden Size | 6144 |
| Layers | 70 |
| Attention Heads | 128 Q |
| KV Heads (full & sliding window) | 8 |
| Q/K Head Dim | 192 |
| V Head Dim | 128 |
| Experts | 384 routed (top-8 routing), no shared expert |
| Expert Intermediate | 2048 |
| Dense MLP Intermediate (layer 0) | 16,384 |
| Vocab Size | 152,576 |
| RoPE | Partial (33.4% → 64 of 192 dims), theta=10M (full) / 10K (SWA) |
| Sliding Window | 128 |
| Max Position | 1,048,576 (1M) |
| Attention Projection | `fused_qkv` (single `qkv_proj.weight`) |

Key features:
- **Hybrid Attention**: 10 full attention layers (0, 7, 15, 23, 31, 39, 47, 55, 62, 69) + 60 sliding window layers, per `hybrid_layer_pattern`
- **Asymmetric Head Dims**: Q/K use head_dim=192, V uses v_head_dim=128
- **Attention Sink Bias**: Learnable per-head bias on sliding window layers only (`add_swa_attention_sink_bias=True`, `add_full_attention_sink_bias=False`)
- **Sigmoid Router + noaux_tc**: `sigmoid(logits) + e_score_correction_bias` is used to pick top-8 experts; unbiased `sigmoid(logits)` becomes the affinity weights. `n_group=1, topk_group=1` degenerates group-limited routing to plain noaux_tc.
- **attention_value_scale = 0.612**: HF reference multiplies `value_states` by this before `softmax(QK^T) × V` (NOT applied post-attention); the NxDI port matches.

## Status (work-in-progress)

**This port compiles cleanly and serves via vLLM on Trn2. The shipping recipe is BF16 attention + FP8 MoE at `seq_len=256` — verified to produce coherent output end-to-end via `smoke_generate_mimo_v2.py`.** Last updated 2026-04-29.

### Why BF16 attn + FP8 MoE

Pro's attention weights have `abs_mean ≈ 0.00124`, roughly 4× smaller than V2.5 (256 experts). Under an all-FP8 recipe, the NKI blockwise FP8 accumulator on attention q/k/v at this magnitude drifts the logits across 70 layers and produces prompt-dependent gibberish (`"The capital of France is\n# 1000000000000000"`, `"Once upon a time in a small village there lived\n# 0000000000..."`, etc.). Dequantizing q/k/v to BF16 before the matmul restores coherent output. MoE experts (scales `≈ 2.3e-5`, similarly small) can stay FP8.

Verified end-to-end: `smoke_generate_mimo_v2.py` with a minimal chat template returns a well-formed reasoning trace that correctly identifies the model ("As MiMo, based on Xiaomi's self-developed large model..."). `preprocess_mimo_v2_fp8.py` emits BF16 q/k/v directly so no separate step is required.

GPU stacks (sglang on H100/H200) run the same OCP FP8 checkpoint correctly because they always dequantize FP8 → BF16 before the matmul. The issue is specific to Neuron's direct-FP8 compute path on small-magnitude tensors. Kimi PR #131 observes similar FP8 degradation on Flash and recommends SDK 2.28.

### Cost and constraints

- **HBM headroom.** BF16 q/k/v adds ~2 GB per rank. `seq_len=1024` OOMs on load (the previous attempt failed allocating ~40 MB for rdh/alltoall rings after per-rank tensors already reached 20.9/24 GB). `seq_len=256` frees enough full-attention softmax scratch to fit.
- **Short context.** At `seq_len=256`, Pro's own chat template with the default system prompt is already 260 tokens. Longer context needs a different HBM plan (cross-instance TP/PP, or a larger instance).
- `BS * top_k / num_experts >= 1.0` required when `moe_ep_degree > 1` at decode (else `NotImplementedError`). With `num_experts=384, top_k=8` this forces `BS >= 48`.
- `n_routed_experts=384 = 2^7 × 3` → `384 / ep_degree` is never a power of 2 (6, 12, 24, 48, 96, 192, 384). Kimi PR #131 says NKI `_bwmm_shard_on_block_nki_call` on SDK 2.29 has "depressed logits with EP=2" and recommends SDK 2.28.

### Recipes tried that did not work

- **All-FP8 attention (`modules_to_not_convert` without q/k/v).** Drifts as described above. Known broken; `preprocess_mimo_v2_fp8.py` no longer emits it.
- **`use_torch_block_wise=True`** (PyTorch-fallback blockwise matmul for higher accumulator precision): compile+shard succeeded after ~2 h, but `model.load()` crashed with `status=4 Allocation Failure` — the fallback path raises HBM demand even when scoped to MoE.

### Next experiments queued

- **BF16-attn at `seq_len=512`** (needs a tighter HBM plan — smaller batch, different EP ratio, or shrinking the default system prompt).
- **Cross-instance BF16** via pipeline/tensor parallelism on 2× Trn2 (single-instance HBM cannot hold full BF16 Pro).
- **Selective BF16 only on MoE `gate_up_proj`** (smallest expert scales) while keeping `down_proj` FP8 — another axis to probe if attn drift returns at longer contexts.
- **SDK 2.28 venv** test once installed, per Kimi PR #131.

## Prerequisites

- **Instance**: trn2.48xlarge (128 physical NeuronCores, logical_nc_config=2 → 64 logical cores)
- **Neuron SDK**: 2.29 (Python 3.12, PyTorch 2.9)
- **Venv**: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16` (used by preprocess, smoke, and vLLM serving alike; ships with the DLAMI and is where `0_setup.sh` installs the patched `vllm-neuron`).
- **Disk**: ~3 TB free under `/opt/dlami/nvme` (the HF FP8 checkpoint is ~962 GB, the Neuron-FP8 preprocessed output is ~1 TB, and `save_sharded_checkpoint=true` writes another ~300-1000 GB per compiled config (varies with recipe)).

### NVMe mount

The Trn2 DLAMI ships with four local NVMe SSDs that are assembled into a
RAID0 array at `/opt/dlami/nvme`. After a reboot the mount is **NOT**
reassembled automatically — you must re-mount manually before the paths
below resolve:

```bash
lsblk                            # confirm you see nvme0n1..nvme3n1 devices
sudo mdadm --assemble /dev/md0 /dev/nvme[0-3]n1 2>/dev/null || true
sudo mount /dev/md0 /opt/dlami/nvme
df -h /opt/dlami/nvme            # should show ~6.9 TB total
```

If `mdadm --assemble` says the array is already assembled, the mount
step alone is enough. If `/dev/md0` doesn't exist, the array was never
created on this instance — run `/opt/dlami/setup-nvme.sh` (or the
DLAMI's built-in helper; consult `ls /opt/dlami/*.sh`) before mounting.

## Quick Start (FP8 on Trn2)

End-to-end recipe to go from a fresh trn2.48xlarge to a working vLLM OpenAI server serving MiMo-V2.5-Pro FP8. First-time compile takes ~45-60 minutes; subsequent runs hit the neuronx-cc cache and start in a few minutes.

```bash
# 1. Clone this repo on the Trn2 instance
cd $HOME
git clone <your-fork>/neuronx-distributed-inference.git
cd neuronx-distributed-inference
git checkout contrib/MiMo-V2.5-Pro          # the branch this README lives on

# 2. Download the HuggingFace FP8 checkpoint (~290 GB). Any HF-compatible
#    downloader works; huggingface-cli example:
huggingface-cli download XiaomiMiMo/MiMo-V2.5-Pro \
    --local-dir /opt/dlami/nvme/models/MiMo-V2.5-Pro

# 3. Preprocess HF FP8 -> Neuron-FP8 (BF16 attn, FP8 MoE). ~20 min, ~24 GB
#    peak RAM. The preprocess dequants q/k/v to BF16 in one pass — see
#    "Checkpoint Preparation" below for why BF16 attn is the only recipe.
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
python contrib/models/MiMo-V2.5-Pro/src/conversion_script/preprocess_mimo_v2_fp8.py \
    --hf_model_path /opt/dlami/nvme/models/MiMo-V2.5-Pro \
    --save_path     /opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8 \
    --tp_degree 64

# 4. (Optional) sanity-check the Neuron-FP8 checkpoint without vLLM
#    ~90 min first compile; subsequent runs ~60s to load the pre-sharded NEFF.
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
python contrib/models/MiMo-V2.5-Pro/perf_test/smoke_compile_mimo_v2.py  # compile
python contrib/models/MiMo-V2.5-Pro/perf_test/smoke_generate_mimo_v2.py # 20-token generate

# 5. Install vllm-neuron with the contrib registration patch
bash contrib/models/MiMo-V2.5-Pro/perf_test/0_setup.sh

# 6. Start vLLM serving MiMo-V2.5-Pro FP8 (first compile ~60 min; subsequent ~3 min)
bash contrib/models/MiMo-V2.5-Pro/perf_test/bench_mimo_v2.sh
```

The bench script runs one configuration (BS=48,
`moe_tp_degree=1 / moe_ep_degree=64`) at three concurrency levels (1, 16, 48)
and logs results under `/opt/dlami/nvme/logs/bench_results/mimo_v2_5_pro/`.

### Keeping a server up for ad-hoc testing

`bench_mimo_v2.sh` is a one-shot wrapper (launch server → sanity →
3 bench runs → teardown). If you want a long-running server to iterate
against, use the three underlying scripts separately:

```bash
# Terminal 1: launch the server in the foreground (Ctrl-C to stop).
bash contrib/models/MiMo-V2.5-Pro/perf_test/start_vllm_server.sh

# Terminal 2: once "Application startup complete." prints, sanity-check:
bash contrib/models/MiMo-V2.5-Pro/perf_test/sanity_check.sh

# Run a single bench pass with a chosen concurrency:
CONCURRENCY=16 NUM_PROMPTS=128 \
    bash contrib/models/MiMo-V2.5-Pro/perf_test/run_bench_single.sh
```

`bench_mimo_v2.sh` composes exactly these three pieces; use whichever
is more convenient.

### Environment variables

`0_setup.sh` prints these at the end; setting them explicitly makes the
smoke / bench / manual-launch paths all behave the same. All of them have
sensible defaults in the scripts — export them only if you want to
override or if you plan to launch vLLM outside of `bench_mimo_v2.sh`.

**Required (at least for manual `vllm api_server` launches):**

| Variable | Purpose |
|---|---|
| `NXDI_CONTRIB_MIMO_V2_FLASH_SRC` | Path to `contrib/models/MiMo-V2.5-Pro/src/`. `vllm-neuron`'s registration hook reads it to plug `NeuronMiMoV2ForCausalLM` into NxDI's `MODEL_TYPES` table. The `_FLASH_` suffix is kept for backward compatibility with the shared registration hook that also serves V2-Flash and V2.5. |
| `MIMO_V2_FLASH_PATH` | Preprocessed Neuron-FP8 checkpoint dir (the `--save_path` output from preprocess). Same naming rationale as above. |

**Optional (recommended):**

| Variable | Default | Purpose |
|---|---|---|
| `NEURON_COMPILED_ARTIFACTS` | `/opt/dlami/nvme/compiled/mimo_v2_5_pro_bs48_moetp1_ep64_fp8_vllm` | Where vLLM writes the NEFF + per-rank sharded weights. Default points at a persistent path under `/opt/dlami/nvme/compiled/` so multiple configs don't collide and runs after the nightly reboot can reuse the sharded weights. vLLM's fallback is `<checkpoint>/neuron-compiled-artifacts/<hash>/` which buries output inside the checkpoint dir. |
| `BASE_COMPILE_WORK_DIR` | `/opt/dlami/nvme/tmp/nxd_model/<basename of NEURON_COMPILED_ARTIFACTS>` | NxDI's HLO / NEFF staging workdir. Default is `/tmp/nxd_model/`, which is wiped by the nightly Trn2 reboot and can silently corrupt parallel compiles that share a basename; the pinned value lives on persistent storage and is unique per config. |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `7200` | First-time compile of Pro's 384-expert MoE is ~60 min TKG + ~15 min CTE + ~30 min shard, well past vLLM's default. |

For a quick `curl` sanity check while the server is up:

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8",
         "messages": [{"role": "user", "content": "Hello! Introduce yourself in one sentence."}],
         "max_tokens": 64, "temperature": 0.0}' | python3 -m json.tool
```

Output quality is currently prompt-dependent under the FP8 recipe (see
Status). A successful sanity check confirms the serving path works; it
does not yet confirm that all prompts produce coherent text.

## Checkpoint Preparation

The HuggingFace checkpoint ships as block-wise OCP FP8 (E4M3, ±448 range), which is not directly compatible with Neuron FP8 (IEEE-754 E4M3, ±240 range). Two preprocess scripts are provided:

### Recommended: FP8 → Neuron-FP8 (streaming)

`src/conversion_script/preprocess_mimo_v2_fp8.py` performs a per-layer streaming rescale from OCP FP8 to Neuron FP8 (per-row scales for attention Q/K/V and layer-0 dense MLP; blockwise scales for MoE experts). `o_proj` is listed in HF's `quantization_config.ignored_layers` and is kept BF16 on the Neuron side (it binds to a plain `RowParallelLinear`, not `QuantizedRowParallel`). Output is ~1 TB across 70 per-layer safetensors shards.

```bash
python contrib/models/MiMo-V2.5-Pro/src/conversion_script/preprocess_mimo_v2_fp8.py \
    --hf_model_path /path/to/MiMo-V2.5-Pro \
    --save_path     /path/to/MiMo-V2.5-Pro-Neuron-FP8 \
    --tp_degree 64
```

Peak RAM during preprocessing is ~24 GB; total runtime ~20 minutes on a trn2.48xlarge instance.

### Why q/k/v are BF16 in the preprocessed output

Pro's attention weights have `abs_mean ≈ 0.00124`, roughly 4× smaller than V2.5 (256 experts). The NKI blockwise FP8 accumulator at this magnitude drifts the logits across 70 layers and produces gibberish output — `"The capital of France is\n# 1000000000000000"`, `"Once upon a time in a small village there lived\n# 0000000000..."`, etc. Dequantizing q/k/v to BF16 while keeping MoE experts FP8 restores coherent output (verified on 2026-04-29 via `smoke_generate_mimo_v2.py`).

The preprocess handles this in a single pass: `split_qkv_fused()` unfuses Pro's `qkv_proj` into per-proj BF16 tensors directly, and the Flash-style per-proj fallback path dequants via `_dequant_attn_to_bf16()`. The checkpoint emitted by preprocess has no `q_proj.scale` / `k_proj.scale` / `v_proj.scale` entries. Compile-time `modules_to_not_convert` must therefore include `q_proj`, `k_proj`, `v_proj` so NxDI routes them through a plain `ColumnParallelLinear` rather than the FP8 `QuantizedColumnParallel` path — `smoke_compile_mimo_v2.py` already does this.

### Fallback: FP8 → BF16

`src/conversion_script/preprocess_mimo_v2_fp8.py` dequantizes the entire checkpoint to BF16. Output is ~290 GB; BF16 is numerically equivalent to the published HF FP8 weights and is useful as a known-good reference. Throughput is ~2× worse than the FP8 path because every attention/MLP matmul operates on full BF16 weights.

## Usage

```python
import sys
from pathlib import Path

# Make this contrib package's src/ importable (flat, per upstream contrib convention).
sys.path.insert(0, str(Path("contrib/models/MiMo-V2.5-Pro/src").resolve()))

import torch
from transformers import AutoConfig, AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter

from modeling_mimo_v2 import NeuronMiMoV2ForCausalLM, MiMoV2InferenceConfig

model_path = "/path/to/MiMo-V2.5-Pro-Neuron-FP8/"
compiled_path = "/path/to/compiled/"

# Recommended recipe: BF16 attn + FP8 MoE.
#   moe_tp_degree = 1, moe_ep_degree = 64
#   q_proj/k_proj/v_proj in modules_to_not_convert (BF16; preprocess
#       emits BF16 for q/k/v, no separate step needed)
#   seq_len = 256 (HBM-tight with BF16 attn; see Status)
# See "FP8 Configuration Notes" below for why other moe_tp/ep ratios
# collapse.
neuron_config = MoENeuronConfig(
    tp_degree=64,
    ep_degree=1,          # keep outer EP = 1; only MoE-internal EP varies
    moe_tp_degree=1,
    moe_ep_degree=64,
    batch_size=48,        # must be >= num_experts / top_k = 384 / 8 = 48
    max_batch_size=48,
    ctx_batch_size=1,
    tkg_batch_size=48,
    seq_len=256,          # HBM is tight with BF16 attn; seq_len=1024 OOMs
    n_active_tokens=128,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,
    capacity_factor=1.0,
    glu_mlp=True,
    fused_qkv=False,      # required: asymmetric Q/K (192) vs V (128) head dims
    router_config={"act_fn": "sigmoid", "dtype": "float32"},
    blockwise_matmul_config={
        "use_shard_on_block_dynamic_while": True,
        "block_sharding_strategy": "PING_PONG",
    },
    save_sharded_checkpoint=True,
    quantized=True,
    quantized_checkpoints_path=model_path,
    quantization_dtype="f8e4m3",
    quantization_type="blockwise_symmetric",
    quantization_block_axis=[1, 2],
    quantization_block_size=[128, 128],
    modules_to_not_convert=[
        "embed_tokens", "lm_head", "norm", "router", "o_proj",
        "q_proj", "k_proj", "v_proj",  # BF16 attn — preprocess emits BF16
    ],
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.6, top_k=20, top_p=0.95,
    ),
)

# trust_remote_code is required by Flash's HF config; pre-load via AutoConfig
# and pass to NxDI so load_pretrained_config does not re-load without the flag.
hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config = MiMoV2InferenceConfig(
    neuron_config, load_config=load_pretrained_config(hf_config=hf_config),
)

model = NeuronMiMoV2ForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
adapter = HuggingFaceGenerationAdapter(model)
inputs = tokenizer(["Hello, how are you?"] * 32, return_tensors="pt", padding=True)
output = adapter.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=128,
)
```

For a minimal end-to-end smoke test that bypasses vLLM, see:

- `perf_test/smoke_compile_mimo_v2.py` — compile + load (STAGE=instantiate|compile|load|all, DRY_RUN, SKIP_WARMUP)
- `perf_test/smoke_generate_mimo_v2.py` — 20-token generation via HuggingFaceGenerationAdapter

Both default to the recommended FP8 recipe (`moe_tp=1`, `moe_ep=64`).

## FP8 Configuration Notes

### moe_tp_degree = 1, moe_ep_degree = 64

**Why**: at `moe_tp_degree=64` each rank owns 1/64 of the intermediate dim, which for MiMo-V2.5-Pro (MoE intermediate = 2048) is 32 rows — **below the 128-row blockwise scale block**. NxDI's `_setup_for_scale` detects `weight_shape[axis] < block_size` and collapses the per-rank scale dim to 1, losing per-channel FP8 scale granularity. The resulting drift compounds across Pro's 69 MoE layers and manifests as output collapse ("helpful helpful helpful ...") after roughly 30 decode tokens.

`moe_tp_degree=1, moe_ep_degree=64` keeps each expert's weights and blockwise scales intact on a single rank (6 experts per rank for Pro's 384 experts), which preserves per-channel scale. On V2.5 (256 experts) this recipe yields coherent output; on V2.5-Pro it still exhibits prompt-dependent drift (see Status).

Intermediate ratios (`moe_tp=32/ep=2`, `moe_tp=16/ep=4`) have been empirically tested and still produce gibberish, so `moe_tp=1/moe_ep=64` is the only currently-usable moe_tp/ep combination.

### batch_size >= 48

NxDI's TKG (token generation) path refuses Expert Parallelism when `batch_size < num_experts / top_k`. For Pro that is 384 / 8 = 48, so the smallest working BS on the FP8 path is 48. BS=1 latency demos are not possible on FP8; use the BF16 checkpoint with `moe_tp=64, moe_ep=1, batch_size=1` for single-stream latency measurements.

### outer ep_degree = 1

`MoENeuronConfig.ep_degree` is the **full-model** expert-parallel factor. Setting it to anything > 1 multiplies `world_size` to `tp_degree * ep_degree`, which on a 64-NC Trn2 overflows the device (ranks beyond 63 have no backing hardware, sharded-checkpoint size grows linearly, and load fails). The MoE-internal expert parallelism is controlled exclusively by `moe_ep_degree` — keep `ep_degree=1` at the outer level.

## vLLM Integration

MiMo-V2.5-Pro can be served via [vllm-neuron](https://github.com/aws-neuron/vllm-neuron). A contrib registration patch is required to plug the NxDI modeling code into vllm-neuron's lookup tables.

### Setup

```bash
# The setup script clones vllm-project/vllm-neuron at release-0.5.0, applies
# the contrib registration patch, installs it editable, and downloads
# Pro Neuron-FP8 weights from S3 (set MIMO_V2_FLASH_PATH to override).
bash contrib/models/MiMo-V2.5-Pro/perf_test/0_setup.sh
```

The patch (`perf_test/vllm-neuron-patch.patch`) touches `vllm_neuron/worker/neuronx_distributed_model_loader.py`. It adds a `_register_contrib_models()` hook that, when `NXDI_CONTRIB_MIMO_V2_FLASH_SRC` is set, registers `NeuronMiMoV2ForCausalLM` into NxDI's `MODEL_TYPES` under keys `mimov2flash` **and** `mimov2pro`, **and** overrides vLLM's built-in `MiMoV2FlashForCausalLM` / `MiMoV2ProForCausalLM` (GPU-only stubs) in `ModelRegistry` with the Neuron wrapper so ModelConfig validation accepts either architecture. No upstream vLLM or NxDI source is modified. The checkpoint's `config.json` must set `architectures` to `["MiMoV2ProForCausalLM"]` (or `MiMoV2FlashForCausalLM` for V2.5); the preprocess script takes care of this.

### Serving (FP8, recommended)

Use `perf_test/start_vllm_server.sh` for a foreground launch (stays up until Ctrl-C), or `perf_test/bench_mimo_v2.sh` for the one-shot launch → sanity → bench → teardown flow. Both scripts bake in the full `override_neuron_config` (TP=64, moe_tp=1, moe_ep=64, BS=48, CB + bucketing, blockwise FP8 MoE with `PING_PONG`, on-device sampling), the required env vars, and the persistent compile-artifact path. See "Keeping a server up for ad-hoc testing" above for the three-terminal workflow.

```bash
# One-shot launch + bench + teardown (~2 h on cold cache, ~5 min on warm cache).
bash contrib/models/MiMo-V2.5-Pro/perf_test/bench_mimo_v2.sh

# Or keep the server up for interactive work:
bash contrib/models/MiMo-V2.5-Pro/perf_test/start_vllm_server.sh
```

See "Environment variables" above for all the knobs (`NEURON_COMPILED_ARTIFACTS`, `BASE_COMPILE_WORK_DIR`, etc.) and their defaults.

> **Note on the shipped vLLM scripts:** the current `start_vllm_server.sh` still uses `seq_len=1024` and does not list `q_proj/k_proj/v_proj` in `modules_to_not_convert`. Coupled with a BF16-attn preprocessed checkpoint this runs correctly (NxDI just sees BF16 tensors where it expected FP8 and casts them as-is) but at a longer context than the BF16-attn recipe has been HBM-validated for. If you hit a `status=4 Allocation Failure` on load, drop `seq_len` / `max_model_len` / `context_encoding_buckets` / `token_generation_buckets` to 256 and add `q_proj/k_proj/v_proj` to `modules_to_not_convert` to match the smoke-verified configuration. The bench numbers below were taken on the older all-FP8 checkpoint and have not been re-measured since the preprocess switched to BF16 attn.

### vllm-neuron patch summary

The patch is applied to vllm-neuron 0.5.0 and:

- Patches `AutoConfig.from_pretrained` to default `trust_remote_code=True` so NxDI's `hf_adapter.load_config` can load the `MiMoV2Config` custom code that ships with the checkpoint.
- Registers `NeuronMiMoV2ForCausalLM` into NxDI's `MODEL_TYPES` under `mimov2flash` and `mimov2pro` so the NxDI loader resolves either model_type to the contrib Neuron wrapper.
- Overrides vLLM's built-in `MiMoV2FlashForCausalLM` and `MiMoV2ProForCausalLM` GPU stubs in `ModelRegistry`, since vLLM's ModelConfig validator rejects any architecture not in its registry and the Neuron path never instantiates vLLM's stub class anyway.

## Performance

> The throughput numbers below were captured on 2026-04-29 against a pre-BF16-attn checkpoint (all-FP8, `seq_len=1024`). They are historical — the shipping recipe is BF16 attn + FP8 MoE at `seq_len=256` and has not yet been re-benchmarked. The numbers are kept here for infra validation (continuous batching + bucketing + on-device sampling + vllm-neuron plugin all wire up end-to-end) and order-of-magnitude reference.

### vLLM Serving (trn2.48xlarge, historical all-FP8 run, BS=48, TP=64, moe_tp=1/moe_ep=64, CB + bucketing, `seq_len=1024`)

Input/output: 900/90 tokens (`vllm bench serve --dataset-name random`), `on_device_sampling_config={do_sample:true, temperature:0.6, top_k:20, top_p:0.95}`.

| Concurrency | Total tok/s | Output tok/s | TTFT median (ms) | TTFT P99 (ms) | TPOT median (ms) |
|-------------|-------------|--------------|------------------|---------------|------------------|
| 1  | 47  | 4.3  | 1,392  | 1,393  | 220 |
| 16 | 391 | 35.6 | 2,361  | 17,394 | 422 |
| 48 | 606 | 55   | 7,322  | 54,413 | 752 |

Per-stream ITL median holds at ~220 ms across all concurrency levels; TPOT/TTFT growth at higher concurrency comes from continuous-batching queue pressure, not per-step compute.

> Expected BF16-attn delta: only q/k/v go from FP8 to BF16 (MoE is unchanged), so steady-state throughput should be within a few percent. TTFT should drop proportionally with `seq_len` (256 vs 1024 prefill tokens).

> **Compile time:** the first Pro compile on SDK 2.29 is ~60 minutes for the TKG NEFF and ~15 minutes for the CTE NEFF; subsequent runs with the same `override_neuron_config` hit the neuronx-cc cache and start in ~1-2 minutes. `save_sharded_checkpoint=true` additionally persists per-rank FP8 shards under `<compiled-path>/weights/`, letting future `load()` calls skip the ~10-minute shard_checkpoint pass. First full server launch (compile + shard + warmup) is ~2 hours wall-clock.

## Compatibility Matrix

| Instance | Neuron SDK 2.29+ (PyTorch 2.9) | 2.21 and earlier |
|----------|--------------------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested | Not tested |
| Trn1 | Not supported (requires 64 logical cores via logical_nc_config=2) | Not supported |
| Inf2 | Not supported | Not supported |

## Testing

```bash
pytest contrib/models/MiMo-V2.5-Pro/test/integration/test_model.py -v
```

## Key Implementation Notes

1. **Hybrid Attention**: `hybrid_layer_pattern` list determines full vs sliding window per layer; the modeling code constructs one `NeuronMiMoV2Attention` per layer with the correct `is_sliding_window` flag and rope_theta.
2. **CONVERT_TO_MHA**: When `tp_degree > num_kv_heads` (64 > 4 full / 64 > 8 SWA), K/V are replicated to `num_attention_heads` (64) during state-dict conversion; this applies to both `.weight` and the per-row `.scale` on the FP8 path.
3. **Attention Sink Bias**: Learnable per-head bias added as an extra "sink" column to attention scores in sliding window layers (not added in full-attention layers). Per-rank slicing of the bias happens inside `forward()` based on `parallel_state.get_tensor_model_parallel_rank()`.
4. **FP8 Path Caveats**:
   - Must use `moe_tp_degree=1, moe_ep_degree=64` (see "FP8 Configuration Notes" above).
   - Must use `batch_size >= 48` (NxDI EP>1 requirement, `384 / 8 = 48`).
   - Must keep outer `ep_degree=1` (only `moe_ep_degree` should vary).
   - Several runtime monkey-patches (router bias, blockwise scale stride, 2D per-channel, EP scale handling) are installed automatically in `NeuronMiMoV2ForCausalLM.__init__` when `quantized=True`; the BF16 path is untouched.

## Example Checkpoints

* [XiaomiMiMo/MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) — HF FP8 source checkpoint

## Maintainer

Henan Wang (whn09)

**Last Updated:** 2026-04-29
