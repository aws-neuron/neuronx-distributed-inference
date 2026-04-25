# Contrib Model: MiniMax-M2 / M2.7

NeuronX Distributed Inference implementation of the MiniMax-M2 family on Trn2.

- **Reference checkpoint used for validation:** [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
- Works with any `MiniMaxM2ForCausalLM` variant (M2 / M2.7 / any minor version) — the config schema is stable across M2 / M2.7.

## Model Information

- **HuggingFace ID:** `MiniMaxAI/MiniMax-M2.7` (and compatible M2 siblings)
- **Model Type:** Decoder-only MoE transformer with uniform GQA attention
- **Architecture:** Custom MoE with sigmoid routing, `e_score_correction_bias` (noaux_tc), per-layer QK RMSNorm
- **License:** Check HuggingFace model card

## Architecture Details

| Parameter | Value |
|-----------|-------|
| Hidden Size | 3072 |
| Layers | 62 |
| Attention Heads | 48 Q / 8 KV (GQA) |
| Head Dim | 128 (Q=K=V; uniform, no asymmetry) |
| Experts | 256 (top-8 routing) |
| Expert Intermediate | 1536 |
| Vocab Size | 200,064 |
| RoPE | Partial (rotary_dim=64 of head_dim=128), theta=5M |
| Max Position | 204,800 |

Key features:
- **Uniform GQA** (no hybrid attention / sliding window / sink bias — M2 is structurally simpler than Flash).
- **QK RMSNorm**: Per-layer RMSNorm applied on Q and K after projection, before RoPE (uses Neuron-native `RmsNorm.apply` for CE/TKG consistency).
- **Sigmoid router + noaux_tc**: `e_score_correction_bias` added to the sigmoid scores before top-k selection; unbiased scores become the expert-affinity weights.
- **FP8-native**: Routed experts ship in blockwise FP8 (128×128 blocks). Per-row FP8 for attention Q/K/V/O after preprocess, which converts the HF OCP FP8 to Neuron FP8.

## Prerequisites

- **Instance**: trn2.48xlarge (32 NeuronCores, logical_nc_config=2 → 64 logical cores)
- **Neuron SDK**: 2.29 (Python 3.12, PyTorch 2.9)
- **Venvs**: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` (for preprocess + direct NxDI smoke), `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16` (for vLLM serving). Both ship with the DLAMI.
- **Disk**: ~500 GB free under `/opt/dlami/nvme` (HF FP8 checkpoint ~215 GB, Neuron-FP8 preprocessed output ~230 GB, plus `save_sharded_checkpoint` writes another ~140 GB per compiled config).

## Quick Start (FP8 on Trn2)

End-to-end recipe. First-time compile is ~25 minutes; subsequent runs hit the neuronx-cc cache and start in a few minutes.

```bash
# 1. Clone this repo on the Trn2 instance
cd $HOME
git clone <your-fork>/neuronx-distributed-inference.git
cd neuronx-distributed-inference
git checkout contrib/MiniMax-M2

# 2. Download the HuggingFace FP8 checkpoint (~215 GB)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
huggingface-cli download MiniMaxAI/MiniMax-M2.7 \
    --local-dir /opt/dlami/nvme/models/MiniMax-M2.7

# 3. Preprocess HF FP8 -> Neuron FP8 (~13 min, ~15 GB peak RAM)
python contrib/models/MiniMax-M2/src/conversion_script/preprocess_minimax_m2_fp8.py \
    --hf_model_path /opt/dlami/nvme/models/MiniMax-M2.7 \
    --save_path     /opt/dlami/nvme/models/MiniMax-M2.7-Neuron-FP8 \
    --tp_degree 64

# 4. (Optional) sanity-check without vLLM (~25 min first compile, then ~20s to load)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
python contrib/models/MiniMax-M2/perf_test/smoke_compile_minimax_m2.py
python contrib/models/MiniMax-M2/perf_test/smoke_generate_minimax_m2.py

# 5. Install vllm-neuron with the contrib registration patch
bash contrib/models/MiniMax-M2/perf_test/0_setup.sh

# 6. Start vLLM + bench (BS=32/moe_ep=64, BS=128/moe_ep=64)
bash contrib/models/MiniMax-M2/perf_test/bench_minimax_m2.sh
```

The bench script runs two configurations (BS=32 and BS=128, both `moe_tp_degree=1 / moe_ep_degree=64`) and logs results under `/tmp/bench_results/minimax_m2/`.

Quick `curl` sanity check once the server is up:

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "/opt/dlami/nvme/models/MiniMax-M2.7-Neuron-FP8",
         "messages": [{"role": "user", "content": "Hello! Introduce yourself in one sentence."}],
         "max_tokens": 64, "temperature": 0.7}' | python3 -m json.tool
```

If you see fluent sentence output on a 50+ token generation, the FP8 path is working correctly. If you see repetition collapse (single-token loops like "helpful helpful helpful..."), double-check that `moe_tp_degree=1`, `moe_ep_degree=64`, `batch_size>=32`, and that you're loading the preprocessed Neuron-FP8 checkpoint (not the raw HF FP8 directory).

## Checkpoint Preparation

The HuggingFace checkpoint ships as block-wise OCP FP8 (E4M3, ±448 range), which is not directly compatible with Neuron FP8 (IEEE-754 E4M3, ±240 range). The preprocess script in `src/conversion_script/preprocess_minimax_m2_fp8.py` rescales it:

- **Attention q/k/v/o**: OCP FP8 blockwise → Neuron FP8 per-row. Per-row scales are used because at TP=64 each rank's output dim is <128, which would collapse a blockwise scale to a singleton. A `_apply_2d_per_channel_fix` monkey-patch installed at compile time routes the 2D weights through PER_CHANNEL_SYMMETRIC to match.
- **MoE experts**: w1/w3 fused into packed `gate_up_proj [num_experts, hidden, 2*IM]`, w2 stacked into `down_proj [num_experts, IM, hidden]`. Scales stay blockwise.
- **Router gate + `e_score_correction_bias`**: renamed into the NxDI router namespace (`block_sparse_moe.router.linear_router.weight` and `...router.e_score_correction_bias`).
- **Norms + embed_tokens + lm_head**: passed through BF16.

Output layout:
```
save_path/
  config.json, tokenizer.*, chat_template.jinja
  configuration_minimax_m2.py, modeling_minimax_m2.py  (trust_remote_code)
  model.safetensors.index.json
  model_extras.safetensors                              (embed/norm/lm_head)
  model_layer{N}.safetensors                            (one per decoder layer, N=0..61)
```

Runtime characteristics: ~15 GB peak RAM, ~13 minutes total on trn2.48xlarge.

## FP8 Configuration Notes

Three non-obvious constraints on Trn2, identical to the Flash FP8 path and for the same underlying reasons:

1. **`moe_tp_degree=1, moe_ep_degree=64` is the only working FP8 ratio.** At `moe_tp=64` each rank's intermediate slice is 24 rows (<128 blockwise block), and NxDI's `_setup_for_scale` collapses the per-rank scale to a singleton — losing per-channel FP8 scale granularity. The resulting drift compounds across M2's 62 MoE layers and manifests as output collapse after ~30 decode tokens. `moe_tp=1, moe_ep=64` keeps each expert's weight + blockwise scale intact on a single rank and produces correct output.

2. **`batch_size >= 32` on the FP8 path.** NxDI's TKG path refuses Expert Parallelism when `batch_size < num_experts / top_k = 256 / 8 = 32`. BS=1 single-stream latency demos on FP8 are not possible.

3. **Keep outer `ep_degree=1`.** `MoENeuronConfig.ep_degree` is the full-model expert-parallel factor and multiplies `world_size` to `tp_degree * ep_degree`. At `world_size > 64` on a 64-NC Trn2, sharded-checkpoint size grows linearly, ranks beyond 63 have no backing hardware, and load fails. MoE EP is controlled exclusively via `moe_ep_degree`.

The bench and smoke scripts have all three pinned correctly; the items above matter only if you're hand-crafting a `MoENeuronConfig`.

## vLLM Integration

MiniMax-M2 can be served via [vllm-neuron](https://github.com/aws-neuron/vllm-neuron). A contrib registration patch (`perf_test/vllm-neuron-patch.patch`) is required to plug the NxDI modeling code into vllm-neuron's lookup tables.

### Setup

```bash
# The setup script clones vllm-project/vllm-neuron at release-0.5.0, applies
# the contrib registration patch, installs it editable, and fetches the HF
# FP8 checkpoint (or skips if already present). It also prints the
# preprocess command if the Neuron-FP8 output dir is empty.
bash contrib/models/MiniMax-M2/perf_test/0_setup.sh
```

### Serving (FP8, recommended)

The bench script already starts a vLLM server at port 8000 with the right config; to start one manually:

```bash
export NXDI_CONTRIB_MINIMAX_M2_SRC=/path/to/neuronx-distributed-inference/contrib/models/MiniMax-M2/src
export MINIMAX_M2_PATH=/path/to/MiniMax-M2.7-Neuron-FP8
export VLLM_ENGINE_READY_TIMEOUT_S=7200
# Optional: isolate compile cache per config so parallel M2/Flash/Pro compiles
# don't race on the default /var/tmp/neuron-compile-cache lock files.
export NEURON_COMPILED_ARTIFACTS=/path/to/compiled/minimax_m2_bs32_moetp1_ep64_fp8

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MINIMAX_M2_PATH" \
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
            "moe_mask_padded_tokens": true,
            "disable_numeric_cc_token": true,
            "save_sharded_checkpoint": true,
            "router_config": {"act_fn": "sigmoid", "dtype": "float32"},
            "quantized": true,
            "quantized_checkpoints_path": "/path/to/MiniMax-M2.7-Neuron-FP8",
            "quantization_dtype": "f8e4m3",
            "quantization_type": "blockwise_symmetric",
            "quantization_block_axis": [1, 2],
            "quantization_block_size": [128, 128],
            "modules_to_not_convert": ["embed_tokens", "lm_head", "norm", "router"],
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

### vllm-neuron patch summary

The patch is applied to vllm-neuron 0.5.0 and:

- Registers `NeuronMiniMaxM2ForCausalLM` into NxDI's `MODEL_TYPES` under `minimax_m2` when `NXDI_CONTRIB_MINIMAX_M2_SRC` points at this contrib package's `src/`.
- Passes `hf_config` from vLLM into `load_pretrained_config` so NxDI does not re-load the config without `trust_remote_code=True`.
- Replaces vllm-neuron's internal `AutoModelForCausalLM.from_pretrained` with `huggingface_hub.snapshot_download`, which is the only path that works for `trust_remote_code=True` models when no GPU is available for HF's CUDA-gated FP8 quantizer.

## Testing

```bash
pytest contrib/models/MiniMax-M2/test/integration/test_model.py -v
```

## Key Implementation Notes

1. **QK Norm**: `MiniMaxM2QKNorm` uses Neuron-native `RmsNorm.apply` (not hand-rolled pow/mean/rsqrt). Hand-rolled PyTorch RMSNorm compiles into different HLO in CE vs TG and produces incorrect TG results.
2. **Router Bias**: `RouterTopKWithBias` stores `e_score_correction_bias` as an `nn.Parameter` initialised to `torch.arange(num_experts, dtype=torch.bfloat16)`. Two non-obvious reasons:
   - `register_buffer` (zeros) gets constant-folded by XLA and the checkpoint bias never binds at inference time.
   - `dtype=float32` triggers a silent dtype mismatch in the NxDI loader's `LayoutTransformation`, which then drops the weight.
3. **CONVERT_TO_MHA**: When `tp_degree > num_kv_heads` (64 > 8), K/V are replicated to `num_attention_heads` (48) during state-dict conversion; on the FP8 path this applies to the per-row `.scale` tensors in lockstep with the weights.
4. **FP8 Runtime Patches** (installed in `NeuronMiniMaxM2ForCausalLM.__init__` when `quantized=True`, idempotent):
   - `_apply_ep_scale_fix` — don't EP-shard `[1,1,W]` singleton scales.
   - `_apply_blockwise_scale_stride_fix` — force `partition_stride=1` for `BLOCKWISE_SYMMETRIC` to avoid strided-split failures when per-rank weight is smaller than a 128-wide scale block.
   - `_apply_2d_per_channel_fix` — flip q_config from `BLOCKWISE_SYMMETRIC` to `PER_CHANNEL_SYMMETRIC` for 2D attention weights at layer-swap time.
5. **`save_quantized_state_dict` override**: short-circuits the HF re-quantize path (which requires CUDA and materialises a ~600 GB BF16 copy) when the preprocess-produced Neuron-FP8 index is already on disk.

## Compatibility Matrix

| Instance | Neuron SDK 2.29+ (PyTorch 2.9) | 2.21 and earlier |
|----------|--------------------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested | Not tested |
| Trn1 | Not supported (requires 64 logical cores via logical_nc_config=2) | Not supported |
| Inf2 | Not supported | Not supported |

## Example Checkpoints

* [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
* [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)  (same config schema, compatible preprocess)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-25
