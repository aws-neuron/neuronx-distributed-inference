#!/bin/bash
# Setup for MiMo-V2-Flash vLLM benchmarking on Trn2.
set -e

echo "=========================================="
echo "Setup: vllm-neuron + MiMo-V2-Flash weights"
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
echo "[2/2] Downloading MiMo-V2-Flash BF16 weights..."

MIMO_PATH="${MIMO_V2_FLASH_PATH:-/opt/dlami/nvme/models/MiMo-V2-Flash-BF16}"
if [ -d "$MIMO_PATH" ] && [ "$(ls "$MIMO_PATH"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  MiMo weights already exist at $MIMO_PATH, skipping download"
else
    echo "  Downloading BF16 weights from your S3 bucket (edit the URI if needed)..."
    mkdir -p "$MIMO_PATH"
    s5cmd cp "s3://datalab/xiaomi/models/MiMo-V2-Flash-BF16/**" "$MIMO_PATH/"
    echo "  Download complete: $(du -sh $MIMO_PATH | cut -f1)"
fi

echo ""
echo "Setup complete. Set MIMO_V2_FLASH_PATH=$MIMO_PATH before running the benchmark."
