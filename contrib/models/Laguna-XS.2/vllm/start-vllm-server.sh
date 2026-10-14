#!/bin/bash
# Start vLLM server for Laguna-XS.2 on Neuron
#
# Prerequisites:
#   - trn2.3xlarge (or larger) with SDK 2.29
#   - Model weights at $LAGUNA_MODEL_PATH (default: /mnt/models/Laguna-XS.2)
#   - NxDI source with Laguna contrib
#   - Pre-compiled and pre-sharded artifacts at $NEURON_COMPILED_ARTIFACTS:
#       model.pt, neuron_config.json, weights/tp{0..3}_sharded_checkpoint.safetensors
#     Without pre-sharded weights, model loading will OOM on 128GB hosts.
#
# Usage:
#   cd /path/to/neuronx-distributed-inference
#   bash contrib/models/Laguna-XS.2/vllm/start-vllm-server.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTRIB_DIR="$(dirname "$SCRIPT_DIR")"
NXDI_DIR="$(dirname "$(dirname "$(dirname "$CONTRIB_DIR")")")"

# Activate vLLM venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

# Set paths
export PYTHONPATH="${NXDI_DIR}/src:${CONTRIB_DIR}:${PYTHONPATH}"
export VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference'
export NEURON_COMPILED_ARTIFACTS="${NEURON_COMPILED_ARTIFACTS:-/mnt/models/laguna-vllm-compiled}"

# Configuration
MODEL_PATH="${LAGUNA_MODEL_PATH:-/mnt/models/Laguna-XS.2}"
TP_DEGREE="${LAGUNA_TP_DEGREE:-4}"
MAX_MODEL_LEN="${LAGUNA_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${LAGUNA_MAX_NUM_SEQS:-4}"
PORT="${LAGUNA_VLLM_PORT:-8000}"

echo ""
echo "Starting vLLM server for Laguna-XS.2"
echo "  Model: ${MODEL_PATH}"
echo "  TP: ${TP_DEGREE}"
echo "  Max seq len: ${MAX_MODEL_LEN}"
echo "  Max concurrent: ${MAX_NUM_SEQS}"
echo "  Port: ${PORT}"
echo "  Compiled artifacts: ${NEURON_COMPILED_ARTIFACTS}"
echo ""

python "${CONTRIB_DIR}/vllm/serve_laguna.py" \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${TP_DEGREE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --port "${PORT}" \
    --trust-remote-code \
    "$@"
