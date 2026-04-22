#!/bin/bash
# Setup for MiniMax-M2 vLLM benchmarking on Trn2.
set -e

echo "=========================================="
echo "Setup: vllm-neuron + MiniMax-M2 weights"
echo "=========================================="

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

echo ""
echo "[1/2] Installing vllm-neuron with the MiMo/MiniMax patch..."

if [ ! -d /tmp/vllm-neuron ]; then
    git clone --branch feature/mimo-support https://github.com/whn09/vllm-neuron.git /tmp/vllm-neuron
fi
cd /tmp/vllm-neuron
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

echo ""
echo "Setup complete. Set MINIMAX_M2_PATH=$MINIMAX_PATH before running the benchmark."
