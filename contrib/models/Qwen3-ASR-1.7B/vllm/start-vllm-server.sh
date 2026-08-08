#!/bin/bash
# Start vLLM server for Qwen3-ASR-1.7B on Neuron
#
# Prerequisites:
#   - Pre-compiled encoder NEFFs in NEURON_ENCODER_PATH
#   - Pre-compiled text decoder in NEURON_COMPILED_ARTIFACTS
#   - vllm-neuron installed with Qwen3-ASR patches applied

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"
PORT="${PORT:-8000}"
TP_DEGREE="${TP_DEGREE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

# Verify environment variables
if [ -z "$NEURON_COMPILED_ARTIFACTS" ]; then
    echo "ERROR: NEURON_COMPILED_ARTIFACTS not set"
    echo "  Set to path containing compiled text decoder (e.g., /mnt/models/compiled/qwen3_asr_vl_text_tp4)"
    exit 1
fi

if [ -z "$NEURON_ENCODER_PATH" ]; then
    echo "ERROR: NEURON_ENCODER_PATH not set"
    echo "  Set to path containing encoder NEFFs (encoder_T500.pt, encoder_T1000.pt, encoder_T3000.pt)"
    exit 1
fi

if [ -z "$NEURON_RT_VISIBLE_CORES" ]; then
    export NEURON_RT_VISIBLE_CORES="0-$((TP_DEGREE - 1))"
    echo "NEURON_RT_VISIBLE_CORES not set, defaulting to: $NEURON_RT_VISIBLE_CORES"
fi

echo "Starting vLLM server for Qwen3-ASR-1.7B"
echo "  Model: $MODEL_PATH"
echo "  TP degree: $TP_DEGREE"
echo "  Port: $PORT"
echo "  Compiled artifacts: $NEURON_COMPILED_ARTIFACTS"
echo "  Encoder path: $NEURON_ENCODER_PATH"
echo "  Visible cores: $NEURON_RT_VISIBLE_CORES"
echo ""

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP_DEGREE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --port "$PORT" \
    --trust-remote-code \
    --additional-config "{\"override_neuron_config\": {\"text_neuron_config\": {\"tp_degree\": $TP_DEGREE, \"batch_size\": 1, \"n_positions\": $MAX_MODEL_LEN, \"seq_len\": $MAX_MODEL_LEN}}}"
