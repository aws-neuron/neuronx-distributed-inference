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

## Prerequisites

- **Instance**: trn2.48xlarge (32 NeuronCores, logical_nc_config=2 → 64 logical cores)
- **Neuron SDK**: 2.29 (Python 3.12, PyTorch 2.9)
- **Venvs**: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` (for preprocess + NxDI direct smoke), `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16` (for vLLM serving). Both ship with the DLAMI.
- **Disk**: ~3 TB free under `/opt/dlami/nvme` (the HF FP8 checkpoint is ~962 GB, the Neuron-FP8 preprocessed output is ~1 TB, and `save_sharded_checkpoint=true` writes another ~300-1000 GB per compiled config (varies with recipe)).

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

# 3. Preprocess HF FP8 -> Neuron FP8 (~20 min, ~24 GB peak RAM)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python contrib/models/MiMo-V2.5-Pro/src/conversion_script/preprocess_mimo_v2_fp8.py \
    --hf_model_path /opt/dlami/nvme/models/MiMo-V2.5-Pro \
    --save_path     /opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8 \
    --tp_degree 64

# 4. (Optional) sanity-check the Neuron-FP8 checkpoint without vLLM
#    ~45 min first compile; subsequent runs ~30s to load the pre-sharded NEFF.
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
python contrib/models/MiMo-V2.5-Pro/perf_test/smoke_compile_mimo_v2.py  # compile
python contrib/models/MiMo-V2.5-Pro/perf_test/smoke_generate_mimo_v2.py # 20-token generate

# 5. Install vllm-neuron with the contrib registration patch
bash contrib/models/MiMo-V2.5-Pro/perf_test/0_setup.sh

# 6. Start vLLM serving MiMo-V2.5-Pro FP8 (first compile ~60 min; subsequent ~3 min)
bash contrib/models/MiMo-V2.5-Pro/perf_test/bench_mimo_v2.sh
```

The bench script runs two configurations (BS=32 and BS=128, both
`moe_tp_degree=X / moe_ep_degree=Y (see bench script)`) and logs results under
`/tmp/bench_results/mimo_v25_pro/`.

For a quick `curl` sanity check while the server is up:

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8",
         "messages": [{"role": "user", "content": "Hello! Introduce yourself in one sentence."}],
         "max_tokens": 64, "temperature": 0.0}' | python3 -m json.tool
```

If you get fluent sentence-ending output on a 30+ token generation, the
FP8 path is working correctly. If you see repetition collapse
("helpful helpful helpful..."), double-check that `moe_tp_degree=1`,
`moe_ep_degree=64`, `batch_size>=32`, and that you are loading the
preprocessed Neuron-FP8 checkpoint (not the raw HF FP8 directory).

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

**Why**: at `moe_tp_degree=64` each rank owns 1/64 of the intermediate dim, which for Flash (MoE intermediate = 2048) is 32 rows — **below the 128-row blockwise scale block**. NxDI's `_setup_for_scale` detects `weight_shape[axis] < block_size` and collapses the per-rank scale dim to 1, losing per-channel FP8 scale granularity. The resulting drift compounds across Flash's 47 MoE layers and manifests as output collapse ("helpful helpful helpful ...") after roughly 30 decode tokens.

`moe_tp_degree=1, moe_ep_degree=64` keeps each expert's weights and blockwise scales intact on a single rank (4 experts per rank), which preserves per-channel scale and produces correct output even on long multi-turn prompts.

Intermediate ratios (`moe_tp=32/ep=2` or `moe_tp=16/ep=4`) have been empirically tested and still produce gibberish, so this is the only currently-supported moe_tp/ep combination for MiMo-V2.5-Pro FP8.

### batch_size >= 32

NxDI's TKG (token generation) path refuses Expert Parallelism when `batch_size < num_experts / top_k`. For Flash that is 256 / 8 = 32, so the smallest working BS on the FP8 path is 32. BS=1 latency demos are not currently possible on FP8; use the BF16 checkpoint with `moe_tp=64, moe_ep=1, batch_size=1` for single-stream latency measurements.

### outer ep_degree = 1

`MoENeuronConfig.ep_degree` is the **full-model** expert-parallel factor. Setting it to anything > 1 multiplies `world_size` to `tp_degree * ep_degree`, which on a 64-NC Trn2 overflows the device (ranks beyond 63 have no backing hardware, sharded-checkpoint size grows linearly, and load fails). The MoE-internal expert parallelism is controlled exclusively by `moe_ep_degree` — keep `ep_degree=1` at the outer level.

## vLLM Integration

MiMo-V2.5-Pro can be served via [vllm-neuron](https://github.com/aws-neuron/vllm-neuron). A contrib registration patch is required to plug the NxDI modeling code into vllm-neuron's lookup tables.

### Setup

```bash
# The setup script clones vllm-project/vllm-neuron at release-0.5.0, applies
# the contrib registration patch, installs it editable, and downloads Flash
# weights (BF16 by default; set MIMO_V2_FLASH_PATH to override).
bash contrib/models/MiMo-V2.5-Pro/perf_test/0_setup.sh
```

The patch (`perf_test/vllm-neuron-patch.patch`) is 40 lines and only touches `vllm_neuron/__init__.py`. It adds a `_register_contrib_models()` hook that, when `NXDI_CONTRIB_MIMO_V2_FLASH_SRC` is set, registers `NeuronMiMoV2ForCausalLM` into NxDI's `MODEL_TYPES` under the key `mimo_v2` **and** registers the `MiMoV2ForCausalLM` architecture into vLLM's `ModelRegistry`. No upstream vLLM or NxDI source is modified.

### Serving (FP8, recommended)

```bash
export NXDI_CONTRIB_MIMO_V2_FLASH_SRC=/path/to/neuronx-distributed-inference/contrib/models/MiMo-V2.5-Pro/src
export MIMO_V2_FLASH_PATH=/path/to/MiMo-V2.5-Pro-Neuron-FP8
# First-time compile of Flash's 256-expert MoE takes 30-60 minutes.
export VLLM_ENGINE_READY_TIMEOUT_S=7200
# Optional: isolate compile cache per config so parallel Flash/Pro/etc. compiles
# don't race on the default /var/tmp/neuron-compile-cache lock files.
export NEURON_COMPILED_ARTIFACTS=/path/to/compiled/mimo_v25_pro_bs32_moetp1_ep64_fp8

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MIMO_V2_FLASH_PATH" \
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
            "quantized_checkpoints_path": "/path/to/MiMo-V2.5-Pro-Neuron-FP8",
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

See `perf_test/bench_mimo_v2.sh` for the full benchmark recipe at BS=32 and BS=128.

### vllm-neuron patch summary

The patch is applied to vllm-neuron 0.5.0 and:

- Maps the `MiMoV2ForCausalLM` architecture to Flash's model loader (reusing the Qwen2-family loader path, which Flash's tokenizer inherits from).
- Passes `hf_config` from vLLM into `load_pretrained_config` so NxDI does not re-load the config without `trust_remote_code=True`.
- Replaces vllm-neuron's internal `AutoModelForCausalLM.from_pretrained` call with `huggingface_hub.snapshot_download`, which is the only path that works for `trust_remote_code=True` models when no GPU is available for HF's CUDA-gated FP8 quantizer.

## Performance

> These numbers are from the earlier BF16 recipe (pre-FP8 rollout). FP8 numbers will be added once a stable bench run completes on the new recipe; preliminary single-stream qualitative tests show fluent multi-sentence output on long Chinese chat prompts with `moe_tp=1, moe_ep=64, batch_size=32`.

### Standalone NxDI (trn2.48xlarge, BF16, TP=64, EP=64)

| Batch Size | Throughput (tok/s) |
|------------|-------------------|
| 1 | 29.92 |
| 8 | 215.94 |
| 32 | 649.14 |

### vLLM Serving (trn2.48xlarge, BF16, BS=32, TP=64/EP=64, CB)

Input/output: 900/90 tokens (random dataset)

| Concurrency | Throughput (tok/s) | TPOT (ms) | TTFT (ms) |
|-------------|-------------------|-----------|-----------|
| 1 | 27.98 | 33.65 | 222 |
| 16 | 224.57 | 64.95 | 570 |
| 32 | 302.61 | 90.23 | 1351 |

> **Compile time:** the first Flash compile on SDK 2.29 is ~30-60 minutes for the TKG NEFF and similar for the CTE NEFF. Subsequent runs with the same `override_neuron_config` hit the neuronx-cc cache and start in ~1-2 minutes. `save_sharded_checkpoint=true` additionally persists per-rank FP8 shards under `<compiled-path>/weights/`, letting future `load()` calls skip the ~10-minute shard_checkpoint pass.

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
   - Must use `batch_size >= 32` (NxDI EP>1 requirement).
   - Must keep outer `ep_degree=1` (only `moe_ep_degree` should vary).
   - Several runtime monkey-patches (router bias, blockwise scale stride, 2D per-channel, EP scale handling) are installed automatically in `NeuronMiMoV2ForCausalLM.__init__` when `quantized=True`; the BF16 path is untouched.

## Example Checkpoints

* [XiaomiMiMo/MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) — HF FP8 source checkpoint

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-25
