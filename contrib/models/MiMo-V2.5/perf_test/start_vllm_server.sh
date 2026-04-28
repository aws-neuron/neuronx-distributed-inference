#!/bin/bash
# Start the MiMo-V2.5 FP8 vLLM OpenAI-compatible server in the foreground.
#
# The server stays up until you Ctrl-C it. Use sanity_check.sh and
# run_bench_single.sh in a separate shell to exercise / benchmark it.
# bench_mimo_v2_5.sh calls this script under the hood for its one-shot
# launch + bench + teardown flow.
#
# Recipe: TP=64, moe_tp=1/moe_ep=64, BS=32, continuous batching + bucketing.
# moe_tp=1/moe_ep=64 keeps each expert's weights and blockwise FP8 scales
# intact on a single rank (4 experts/rank), avoiding the per-rank scale
# collapse that comes from moe_tp=64 when intermediate=2048 is TP-sharded
# below the 128-row scale block boundary.
#
# NxDI's TKG path refuses Expert Parallelism with BS < num_experts/top_k
# (256 / 8 = 32), so BS=32 is the smallest working batch size on the FP8
# path. BS=1 single-stream latency is not currently supported on V2.5 FP8.

set -e

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

MODEL_PATH="${MIMO_V2_5_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Neuron-FP8}"
PORT="${PORT:-8000}"

# Contrib package src. vllm-neuron's registration hook reads these env vars
# to plug NeuronMiMoV2ForCausalLM into NxDI's MODEL_TYPES table.
: "${NXDI_CONTRIB_MIMO_V2_5_SRC:=$(cd "$(dirname "$0")/.." && pwd)/src}"
export NXDI_CONTRIB_MIMO_V2_5_SRC
# vLLM 0.16's builtin arch validator knows MiMoV2FlashForCausalLM but not
# MiMoV2ForCausalLM. Preprocess rewrites the checkpoint's config.json
# architectures to the Flash name, and we reuse the Flash registration
# key in vllm-neuron (MODEL_TYPES['mimov2flash']). The modeling module
# (modeling_mimo_v2) and class (NeuronMiMoV2ForCausalLM) are shared.
export NXDI_CONTRIB_MIMO_V2_FLASH_SRC="$NXDI_CONTRIB_MIMO_V2_5_SRC"

# Persistent compile-artifact location (NEFF + per-rank sharded weights).
# Setting this overrides vLLM's fallback of <checkpoint>/neuron-compiled-artifacts/<hash>/.
: "${NEURON_COMPILED_ARTIFACTS:=/opt/dlami/nvme/compiled/mimo_v2_5_bs32_moetp1_ep64_fp8_vllm}"
export NEURON_COMPILED_ARTIFACTS
# NxDI HLO/NEFF staging directory, pinned to persistent storage so it
# survives the nightly Trn2 reboot and a unique per-config subdir.
: "${BASE_COMPILE_WORK_DIR:=/opt/dlami/nvme/tmp/nxd_model/$(basename "$NEURON_COMPILED_ARTIFACTS")}"
export BASE_COMPILE_WORK_DIR
mkdir -p "$BASE_COMPILE_WORK_DIR"

# First-time compile of V2.5's 256-expert MoE takes ~30 min (HLO + shard).
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

echo "=========================================="
echo "Starting MiMo-V2.5 FP8 vLLM server"
echo "=========================================="
echo "  Model path:              $MODEL_PATH"
echo "  Port:                    $PORT"
echo "  Compiled artifacts:      $NEURON_COMPILED_ARTIFACTS"
echo "  Compile work dir:        $BASE_COMPILE_WORK_DIR"
echo "  NXDI_CONTRIB_MIMO_V2_5_SRC: $NXDI_CONTRIB_MIMO_V2_5_SRC"
echo ""

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port "$PORT" \
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
            "quantized_checkpoints_path": "'"$MODEL_PATH"'",
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
