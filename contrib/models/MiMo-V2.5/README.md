# Contrib Model: MiMo-V2.5

NeuronX Distributed Inference implementation of [XiaomiMiMo/MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5). MiMo-V2.5 supersedes the earlier MiMo-V2-Flash release with the same decoder-only MoE architecture, an updated tokenizer, and a multimodal (vision + audio) head that the NxDI language path does not use.

## Model Information

- **HuggingFace ID:** `XiaomiMiMo/MiMo-V2.5`
- **Model Type:** Decoder-only MoE transformer with hybrid (full + SWA) attention
- **License:** Check HuggingFace model card

## Architecture Details

| Parameter | Value |
|-----------|-------|
| Hidden Size | 4096 |
| Layers | 48 (layer 0 dense, layers 1–47 MoE) |
| Q Heads | 64 |
| KV Heads (full attn) | 4 |
| KV Heads (sliding window) | 8 |
| Q/K Head Dim | 192 |
| V Head Dim | 128 |
| Experts | 256 (top-8 routing) |
| Expert Intermediate | 2048 |
| Vocab Size | 152,576 |
| RoPE | Partial (64 of 192 head dims = 33.4%), theta=5M (full) / 10K (SWA) |
| Sliding Window | 128 |
| Max Position | 262,144 |

Key features:
- **Hybrid Attention**: 9 full attention layers (0, 5, 11, 17, 23, 29, 35, 41, 47) + 39 sliding window layers (positions driven by `hybrid_layer_pattern`).
- **Asymmetric Head Dims**: Q/K use 192, V uses 128. Plus asymmetric `num_kv_heads` between full (4) and SWA (8) layers.
- **Fused QKV on disk, split on Neuron**: the HF checkpoint ships `qkv_proj.weight` fused (`attention_projection_layout="fused_qkv"`); the NxDI modeling code keeps separate `q_proj`/`k_proj`/`v_proj` linears, so the preprocess script slices the fused tensor back into three per-proj tensors (see "Checkpoint Preparation").
- **Attention Sink Bias**: Learnable per-head bias on sliding window layers only (`add_swa_attention_sink_bias=true`, `add_full_attention_sink_bias=false`).
- **Sigmoid Router + noaux_tc**: `e_score_correction_bias` added to sigmoid scores before top-k selection; unbiased scores become the affinity weights.
- **attention_value_scale = 0.707**: HF MiMo-V2 multiplies `value_states` by this before the attention softmax × V (NOT applied to attn_output); the NxDI model matches.

## Prerequisites

- **Instance**: trn2.48xlarge (128 NeuronCores, logical_nc_config=2 → 64 logical cores)
- **Neuron SDK**: 2.29 (Python 3.12, PyTorch 2.9)
- **Venv**: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16` (ships with the DLAMI; has NxDI, vllm-neuron, and `huggingface_hub`/`s5cmd`).
- **Disk**: ~900 GB free under `/opt/dlami/nvme` (HF FP8 checkpoint ~295 GB, Neuron-FP8 preprocessed output ~310 GB, and `save_sharded_checkpoint=true` writes another ~300 GB of per-rank sharded weights per compiled config). The DLAMI creates a 6.9 TB RAID0 at `/dev/md0` across the instance-store NVMes but does **not** add it to `/etc/fstab`, so it is not mounted automatically after a reboot. Before running any of the steps below, remount it if needed:

  ```bash
  # If /opt/dlami/nvme appears empty after an overnight reboot, the md0 array
  # is still intact and just needs to be remounted:
  mount | grep -q /opt/dlami/nvme || sudo mount /dev/md0 /opt/dlami/nvme
  df -h /opt/dlami/nvme   # should show ~6.9 TB
  ```

## Quick Start (FP8 on Trn2)

End-to-end recipe to go from a fresh trn2.48xlarge to a working vLLM OpenAI server serving MiMo-V2.5 FP8. First-time compile takes ~30 minutes; subsequent runs hit the neuronx-cc cache and start in a few minutes.

```bash
# 1. Clone this repo on the Trn2 instance
cd $HOME
git clone <your-fork>/neuronx-distributed-inference.git
cd neuronx-distributed-inference
git checkout contrib/MiMo-V2.5          # the branch this README lives on

# 2. Download the HuggingFace FP8 checkpoint (~295 GB).
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
huggingface-cli download XiaomiMiMo/MiMo-V2.5 \
    --local-dir /opt/dlami/nvme/models/MiMo-V2.5 --max-workers 16

# 3. Preprocess HF FP8 -> Neuron FP8 (~16 min, ~15 GB peak RAM)
python contrib/models/MiMo-V2.5/src/conversion_script/preprocess_mimo_v2_5_fp8.py \
    --hf_model_path /opt/dlami/nvme/models/MiMo-V2.5 \
    --save_path     /opt/dlami/nvme/models/MiMo-V2.5-Neuron-FP8 \
    --tp_degree 64

# 4. (Optional) sanity-check the Neuron-FP8 checkpoint without vLLM
#    ~30 min first compile (priority HLO + CE HLO + 27 min shard_checkpoint
#    for 64 ranks); subsequent runs ~30s to load the pre-sharded NEFF.
python contrib/models/MiMo-V2.5/perf_test/smoke_compile_mimo_v2_5.py  # compile + shard
python contrib/models/MiMo-V2.5/perf_test/smoke_generate_mimo_v2_5.py # 20-token generate

# 5. Install vllm-neuron with the contrib registration patch
bash contrib/models/MiMo-V2.5/perf_test/0_setup.sh

# 6. Start vLLM serving MiMo-V2.5 FP8
bash contrib/models/MiMo-V2.5/perf_test/bench_mimo_v2_5.sh
```

The bench script runs one configuration (BS=32,
`moe_tp_degree=1 / moe_ep_degree=64`) at three concurrency levels (1, 16, 32) and logs results under
`/opt/dlami/nvme/logs/bench_results/mimo_v2_5/`.

### Keeping a server up for ad-hoc testing

`bench_mimo_v2_5.sh` is a one-shot wrapper (launch server → sanity →
3 bench runs → teardown). If you want a long-running server to iterate
against, use the three underlying scripts separately:

```bash
# Terminal 1: launch the server in the foreground (Ctrl-C to stop).
bash contrib/models/MiMo-V2.5/perf_test/start_vllm_server.sh

# Terminal 2: once "Application startup complete." prints, sanity-check:
bash contrib/models/MiMo-V2.5/perf_test/sanity_check.sh

# Run a single bench pass with a chosen concurrency:
CONCURRENCY=16 NUM_PROMPTS=128 \
    bash contrib/models/MiMo-V2.5/perf_test/run_bench_single.sh
```

`bench_mimo_v2_5.sh` composes exactly these three pieces; use whichever
is more convenient.

### Environment variables

`0_setup.sh` prints these at the end; setting them explicitly makes the
smoke / bench / manual-launch paths all behave the same. All of them have
sensible defaults in the scripts — export them only if you want to
override or if you plan to launch vLLM outside of `bench_mimo_v2_5.sh`.

**Required (at least for manual `vllm api_server` launches):**

| Variable | Purpose |
|---|---|
| `NXDI_CONTRIB_MIMO_V2_5_SRC` | Path to `contrib/models/MiMo-V2.5/src/`. `vllm-neuron`'s registration hook reads it to plug `NeuronMiMoV2ForCausalLM` into NxDI's `MODEL_TYPES` table. |
| `NXDI_CONTRIB_MIMO_V2_FLASH_SRC` | Alias of `NXDI_CONTRIB_MIMO_V2_5_SRC` — same value. vLLM's builtin arch validator only knows `MiMoV2FlashForCausalLM`, so preprocess rewrites the checkpoint's `architectures` to that name and we re-use the Flash registration key (`mimov2flash`) in vllm-neuron's lookup table. |
| `MIMO_V2_5_PATH` | Preprocessed Neuron-FP8 checkpoint dir (the `--save_path` output from preprocess). |

**Optional (recommended):**

| Variable | Default | Purpose |
|---|---|---|
| `NEURON_COMPILED_ARTIFACTS` | `/opt/dlami/nvme/compiled/mimo_v2_5_bs32_moetp1_ep64_fp8_vllm` | Where vLLM writes the NEFF + per-rank sharded weights. Default points at a persistent path under `/opt/dlami/nvme/compiled/` so multiple configs don't collide and runs after the nightly reboot can reuse the sharded weights. vLLM's fallback is `<checkpoint>/neuron-compiled-artifacts/<hash>/` which buries output inside the checkpoint dir. |
| `BASE_COMPILE_WORK_DIR` | `/opt/dlami/nvme/tmp/nxd_model/<basename of NEURON_COMPILED_ARTIFACTS>` | NxDI's HLO / NEFF staging workdir. Default is `/tmp/nxd_model/`, which is wiped by the nightly Trn2 reboot and can silently corrupt parallel compiles that share a basename; the pinned value lives on persistent storage and is unique per config. |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `7200` | First-time compile of V2.5's 256-expert MoE is ~30 min dominated by `shard_checkpoint`, well past vLLM's default. |

For a quick `curl` sanity check while the server is up:

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "/opt/dlami/nvme/models/MiMo-V2.5-Neuron-FP8",
         "messages": [{"role": "user", "content": "Hello! Introduce yourself in one sentence."}],
         "max_tokens": 64, "temperature": 0.0}' | python3 -m json.tool
```

If you get fluent sentence-ending output on a 30+ token generation, the
FP8 path is working correctly. If you see repetition collapse
("helpful helpful helpful..."), double-check that `moe_tp_degree=1`,
`moe_ep_degree=64`, `batch_size>=32`, and that you are loading the
preprocessed Neuron-FP8 checkpoint (not the raw HF FP8 directory).

## Checkpoint Preparation

The HuggingFace checkpoint ships as block-wise OCP FP8 (E4M3, ±448 range), which is not directly compatible with Neuron FP8 (IEEE-754 E4M3, ±240 range). `src/conversion_script/preprocess_mimo_v2_5_fp8.py` performs a per-layer streaming rescale: per-row scales for attention Q/K/V (after fused-qkv split) and the layer-0 dense MLP; blockwise 128×128 scales for MoE experts. `o_proj` is listed in HF's `quantization_config.ignored_layers` and is kept BF16 on the Neuron side (it binds to a plain `RowParallelLinear`, not `QuantizedRowParallel`). Output is ~310 GB across 48 per-layer safetensors shards.

```bash
python contrib/models/MiMo-V2.5/src/conversion_script/preprocess_mimo_v2_5_fp8.py \
    --hf_model_path /path/to/MiMo-V2.5 \
    --save_path     /path/to/MiMo-V2.5-Neuron-FP8 \
    --tp_degree 64
```

Peak RAM during preprocessing is ~15 GB; total runtime ~16 minutes on a trn2.48xlarge instance.

### V2.5-specific: fused qkv_proj split into 4 interleaved groups

The HF checkpoint advertises `q_proj.weight` / `k_proj.weight` / `v_proj.weight` in its safetensors index, but the actual LFS objects on the Hub only carry a single fused `self_attn.qkv_proj.weight` tensor. NxDI's MiMoV2Attention hard-codes separate Q/K/V `ColumnParallelLinear` modules, so the preprocess script splits the fused tensor back into three per-proj tensors.

The fused layout is **not** `[all_Q | all_K | all_V]`. It is **4 interleaved groups** (the group count equals the full-attention `num_key_value_heads = 4`), each packing `hpg` Q heads, `kpg` K heads, and `kpg` V heads contiguously:

    group g (g = 0..3):
        rows [g*R      : g*R + qg]          = Q heads [g*hpg : (g+1)*hpg]
        rows [g*R + qg : g*R + qg + kg]     = K heads [g*kpg : (g+1)*kpg]
        rows [g*R + qg + kg : g*R + R]      = V heads [g*kpg : (g+1)*kpg]

    where hpg = num_q_heads / 4, kpg = num_kv_heads / 4,
          qg = hpg * 192, kg = kpg * 192, vg = kpg * 128,
          R  = qg + kg + vg

For **full-attention layers** this gives `hpg=16, kpg=1, R=3392, total=13568` rows with 108 scale blocks (includes 2 phantom rows from `ceil(192/128)=2`). For **SWA layers** (`num_kv_heads=8`), `hpg=16, kpg=2, R=3712, total=14848` rows with 116 scale blocks (no phantom, since `kg=384` is 128-aligned). Layer 0 (dense) is still attention-FP8 and follows the full-layer layout.

Any preprocess approach that treats the fused tensor as a plain `[Q|K|V]` concatenation produces garbled outputs — Q/K/V rows land in the wrong per-head slots after the split.

## Usage

```python
import sys
from pathlib import Path

# Make this contrib package's src/ importable (flat, per upstream contrib convention).
sys.path.insert(0, str(Path("contrib/models/MiMo-V2.5/src").resolve()))

import torch
from transformers import AutoConfig, AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter

from modeling_mimo_v2 import NeuronMiMoV2ForCausalLM, MiMoV2InferenceConfig

model_path = "/path/to/MiMo-V2.5-Neuron-FP8/"
compiled_path = "/path/to/compiled/"

# Recommended FP8 recipe:
#   moe_tp_degree = 1, moe_ep_degree = 64
# See "FP8 Configuration Notes" below for why other moe_tp/ep ratios collapse.
neuron_config = MoENeuronConfig(
    tp_degree=64,
    ep_degree=1,          # keep outer EP = 1; only MoE-internal EP varies
    moe_tp_degree=1,
    moe_ep_degree=64,
    batch_size=32,        # must be >= num_experts / top_k = 256 / 8 = 32
    max_batch_size=32,
    ctx_batch_size=1,
    tkg_batch_size=32,
    seq_len=1024,
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
    ],
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.6, top_k=20, top_p=0.95,
    ),
)

# trust_remote_code is required by MiMo-V2's HF config; pre-load via AutoConfig
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

- `perf_test/smoke_compile_mimo_v2_5.py` — compile + load (STAGE=instantiate|compile|load|all, DRY_RUN, SKIP_WARMUP)
- `perf_test/smoke_generate_mimo_v2_5.py` — 20-token generation via HuggingFaceGenerationAdapter

Both default to the recommended FP8 recipe (`moe_tp=1`, `moe_ep=64`).

## FP8 Configuration Notes

### moe_tp_degree = 1, moe_ep_degree = 64

**Why**: at `moe_tp_degree=64` each rank owns 1/64 of the intermediate dim, which for MiMo-V2.5 (MoE intermediate = 2048) is 32 rows — **below the 128-row blockwise scale block**. NxDI's `_setup_for_scale` detects `weight_shape[axis] < block_size` and collapses the per-rank scale dim to 1, losing per-channel FP8 scale granularity. The resulting drift compounds across MiMo-V2.5's 47 MoE layers and manifests as output collapse ("helpful helpful helpful ...") after roughly 30 decode tokens.

`moe_tp_degree=1, moe_ep_degree=64` keeps each expert's weights and blockwise scales intact on a single rank (4 experts per rank), which preserves per-channel scale and produces correct output even on long multi-turn prompts.

Intermediate ratios (`moe_tp=32/ep=2` or `moe_tp=16/ep=4`) have been empirically tested and still produce gibberish, so this is the only currently-supported moe_tp/ep combination for MiMo-V2.5 FP8.

### batch_size >= 32

NxDI's TKG (token generation) path refuses Expert Parallelism when `batch_size < num_experts / top_k`. For MiMo-V2.5 that is 256 / 8 = 32, so the smallest working BS on the FP8 path is 32. BS=1 latency demos are not currently possible on FP8; use the BF16 checkpoint with `moe_tp=64, moe_ep=1, batch_size=1` for single-stream latency measurements.

### outer ep_degree = 1

`MoENeuronConfig.ep_degree` is the **full-model** expert-parallel factor. Setting it to anything > 1 multiplies `world_size` to `tp_degree * ep_degree`, which on a 64-NC Trn2 overflows the device (ranks beyond 63 have no backing hardware, sharded-checkpoint size grows linearly, and load fails). The MoE-internal expert parallelism is controlled exclusively by `moe_ep_degree` — keep `ep_degree=1` at the outer level.

## vLLM Integration

MiMo-V2.5 can be served via [vllm-neuron](https://github.com/aws-neuron/vllm-neuron). A contrib registration patch is required to plug the NxDI modeling code into vllm-neuron's lookup tables.

### Setup

```bash
# The setup script clones vllm-project/vllm-neuron at release-0.5.0, applies
# the contrib registration patch, installs it editable, and downloads
# MiMo-V2.5 FP8 weights from HuggingFace (~295 GB; skipped if already present).
bash contrib/models/MiMo-V2.5/perf_test/0_setup.sh
```

`perf_test/vllm-neuron-patch.patch` adds a `_register_contrib_models()` hook to `vllm_neuron/worker/neuronx_distributed_model_loader.py`. When `NXDI_CONTRIB_MIMO_V2_5_SRC` is set, it registers `NeuronMiMoV2ForCausalLM` into NxDI's `MODEL_TYPES` under the key `mimov2` **and** registers the `MiMoV2ForCausalLM` architecture into vLLM's `ModelRegistry`. The hook also patches `AutoConfig.from_pretrained` to default `trust_remote_code=True` so NxDI's `load_pretrained_config` can read the V2.5 config. No upstream vLLM or NxDI source is modified.

### Serving (FP8, recommended)

```bash
export NXDI_CONTRIB_MIMO_V2_5_SRC=/path/to/neuronx-distributed-inference/contrib/models/MiMo-V2.5/src
export MIMO_V2_5_PATH=/path/to/MiMo-V2.5-Neuron-FP8
# First-time compile of MiMo-V2.5's 256-expert MoE takes 30-60 minutes.
export VLLM_ENGINE_READY_TIMEOUT_S=7200
# Optional: isolate compile cache per config so parallel MiMo-V2.5/Pro/etc. compiles
# don't race on the default /var/tmp/neuron-compile-cache lock files.
export NEURON_COMPILED_ARTIFACTS=/path/to/compiled/mimo_v2_5_bs32_moetp1_ep64_fp8

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MIMO_V2_5_PATH" \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 64,
            "logical_nc_config": 2,
            "fused_qkv": false,
            "sequence_parallel_enabled": false,
            "glu_mlp": true,
            "normalize_top_k_affinities": true,
            "save_sharded_checkpoint": true,
            "router_config": {"act_fn": "sigmoid", "dtype": "float32"},
            "quantized": true,
            "quantized_checkpoints_path": "/path/to/MiMo-V2.5-Neuron-FP8",
            "quantization_dtype": "f8e4m3",
            "quantization_type": "blockwise_symmetric",
            "quantization_block_axis": [1, 2],
            "quantization_block_size": [128, 128],
            "modules_to_not_convert": ["embed_tokens", "lm_head", "norm", "router", "o_proj"],
            "blockwise_matmul_config": {"use_shard_on_block_dynamic_while": true, "block_sharding_strategy": "PING_PONG"},
            "moe_tp_degree": 1,
            "moe_ep_degree": 64,
            "batch_size": 32,
            "ctx_batch_size": 1,
            "tkg_batch_size": 32,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": true,
            "enable_bucketing": true,
            "context_encoding_buckets": [1024],
            "token_generation_buckets": [1024],
            "async_mode": true,
            "on_device_sampling_config": {
                "do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95
            }
        }
    }'
```

See `perf_test/bench_mimo_v2_5.sh` for the full benchmark recipe at BS=32.

### Testing the vLLM server

Once `/v1/models` returns 200 (first-compile takes ~30 min; subsequent starts ~3 min), hit `/v1/chat/completions`. MiMo-V2.5's chat template expects the `<|im_start|>...<|im_end|>` ChatML format — vLLM applies it automatically when you use the chat endpoint, so just send a standard messages array:

```bash
MODEL=/opt/dlami/nvme/models/MiMo-V2.5-Neuron-FP8

# 1. Short sanity — should return a one-line MiMo self-introduction.
curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",
         \"messages\":[{\"role\":\"user\",\"content\":\"Hello! Introduce yourself in one sentence.\"}],
         \"max_tokens\":64}" | python3 -m json.tool

# 2. Long output — check for repetition collapse / gibberish on 500+ tokens.
curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",
         \"messages\":[{\"role\":\"user\",\"content\":\"Explain the B-tree data structure in detail, including how insertions and deletions preserve balance.\"}],
         \"max_tokens\":800}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
```

If you see a coherent MiMo introduction and a multi-paragraph technical explanation, the FP8 path is working end-to-end. Output collapse ("helpful helpful helpful ...") on either prompt indicates a broken FP8 recipe — re-check that `moe_tp_degree=1`, `moe_ep_degree=64`, `batch_size>=32`, and that the server is pointed at the Neuron-FP8 preprocessed directory (not the raw HF one).

**Note on sampling determinism**: `on_device_sampling_config.do_sample=true` is the recommended setting; request-level `temperature` is ignored (sampling params are baked into the NEFF at compile time).

### vllm-neuron patch summary

The patch is applied to vllm-neuron 0.5.0 and:

- Maps the `MiMoV2ForCausalLM` architecture to MiMo-V2.5's model loader (reusing the Qwen2-family loader path, which MiMo-V2.5's tokenizer inherits from).
- Passes `hf_config` from vLLM into `load_pretrained_config` so NxDI does not re-load the config without `trust_remote_code=True`.
- Replaces vllm-neuron's internal `AutoModelForCausalLM.from_pretrained` call with `huggingface_hub.snapshot_download`, which is the only path that works for `trust_remote_code=True` models when no GPU is available for HF's CUDA-gated FP8 quantizer.

## Performance

> Benchmark numbers will be added once a stable bench run completes on the FP8 recipe. Preliminary single-stream sanity test produces fluent MiMo self-introduction output on the recipe below (`moe_tp=1, moe_ep=64, batch_size=32`).

> **Compile time:** the first MiMo-V2.5 compile on SDK 2.29 is ~30 minutes (TKG + CE HLO compilation, weight layout optimization, then `shard_checkpoint` for 64 ranks which dominates at ~27 minutes). Subsequent runs with the same `override_neuron_config` hit the neuronx-cc cache and the NEFF loads in ~1 minute. `save_sharded_checkpoint=true` persists per-rank FP8 shards under `<compiled-path>/weights/`, letting future `load()` calls skip the `shard_checkpoint` pass entirely.

## Compatibility Matrix

| Instance | Neuron SDK 2.29+ (PyTorch 2.9) | 2.21 and earlier |
|----------|--------------------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested | Not tested |
| Trn1 | Not supported (requires 64 logical cores via logical_nc_config=2) | Not supported |
| Inf2 | Not supported | Not supported |

## Testing

```bash
pytest contrib/models/MiMo-V2.5/test/integration/test_model.py -v
```

## Key Implementation Notes

1. **Hybrid Attention**: `hybrid_layer_pattern` list determines full vs sliding window per layer; the modeling code constructs one `NeuronMiMoV2Attention` per layer with the correct `is_sliding_window` flag and rope_theta.
2. **CONVERT_TO_MHA**: When `tp_degree > num_kv_heads` (64 > 4 full / 64 > 8 SWA), K/V are replicated to `num_attention_heads` (64) during state-dict conversion; this applies to both `.weight` and the per-row `.scale` on the FP8 path.
3. **Attention Sink Bias**: Learnable per-head bias added as an extra "sink" column to attention scores in sliding window layers (not added in full-attention layers). Per-rank slicing of the bias happens inside `forward()` based on `parallel_state.get_tensor_model_parallel_rank()`.
4. **Fused qkv split in preprocess**: V2.5's HF checkpoint stores `self_attn.qkv_proj.weight` as 4 interleaved Q/K/V groups (see "Checkpoint Preparation" above). The preprocess script must slice these groups — naïve `[Q|K|V]` concat slicing produces garbage outputs.
5. **weight_map rebuild**: V2.5's `model.safetensors.index.json` references legacy `model_N-00001-of-00002.safetensors` filenames that do not match the actual `model_pp0_epN_shardM.safetensors` objects on disk. `LazyWeightMap` scans the on-disk shards at startup and rebuilds `weight_map` directly from each file's manifest; the inconsistent index is ignored.
6. **FP8 Path Caveats**:
   - Must use `moe_tp_degree=1, moe_ep_degree=64` (see "FP8 Configuration Notes" above).
   - Must use `batch_size >= 32` (NxDI EP>1 requirement).
   - Must keep outer `ep_degree=1` (only `moe_ep_degree` should vary).
   - Several runtime monkey-patches (router bias, blockwise scale stride, 2D per-channel, EP scale handling) are installed automatically in `NeuronMiMoV2ForCausalLM.__init__` when `quantized=True`; the BF16 path is untouched.

## Example Checkpoints

* [XiaomiMiMo/MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) — HF FP8 source checkpoint

## Maintainer

Henan Wang (whn09)

**Last Updated:** 2026-04-28
