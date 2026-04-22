#!/bin/bash
# Setup for MiniMax-M2 vLLM benchmarking on Trn2.
#
# Clones upstream vllm-project/vllm-neuron at release-0.5.0 and applies
# vllm-neuron-patch.patch, which adds a runtime registration hook so the
# contrib NeuronMiniMaxM2ForCausalLM is plugged into NxDI's MODEL_TYPES
# at vllm-neuron plugin init time. vLLM's ModelRegistry already recognizes
# MiniMaxM2ForCausalLM so no vLLM-side registration is needed.
set -e

echo "=========================================="
echo "Setup: vllm-neuron + MiniMax-M2 weights"
echo "=========================================="

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

PATCH_FILE="$(cd "$(dirname "$0")" && pwd)/vllm-neuron-patch.patch"

echo ""
echo "[1/2] Installing vllm-neuron (release-0.5.0) with the contrib registration patch..."

if [ ! -d /tmp/vllm-neuron ]; then
    git clone --branch release-0.5.0 https://github.com/vllm-project/vllm-neuron.git /tmp/vllm-neuron
fi

cd /tmp/vllm-neuron

if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "  Applied $PATCH_FILE"
else
    echo "  Patch already applied or conflicts; continuing."
fi

pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
pip install s5cmd

python3 -c "import vllm_neuron; print('vllm-neuron installed:', vllm_neuron.__file__)"

echo ""
echo "[2/2] Downloading MiniMax-M2 BF16 weights..."

MINIMAX_PATH="${MINIMAX_M2_PATH:-/opt/dlami/nvme/models/MiniMax-M2-BF16}"
if [ -d "$MINIMAX_PATH" ] && [ "$(ls "$MINIMAX_PATH"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  MiniMax weights already exist at $MINIMAX_PATH, skipping download"
else
    echo "  Downloading BF16 weights from your S3 bucket (edit the URI if needed)..."
    mkdir -p "$MINIMAX_PATH"
    s5cmd cp "s3://datalab/minimax/model_hf/MiniMax-M2-BF16/**" "$MINIMAX_PATH/"
    echo "  Download complete: $(du -sh $MINIMAX_PATH | cut -f1)"
fi

CONTRIB_SRC="$(cd "$(dirname "$0")/.." && pwd)/src"

echo ""
echo "Setup complete. Before running the benchmark, export:"
echo "  export MINIMAX_M2_PATH=$MINIMAX_PATH"
echo "  export NXDI_CONTRIB_MINIMAX_M2_SRC=$CONTRIB_SRC"
