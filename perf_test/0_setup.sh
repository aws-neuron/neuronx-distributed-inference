#!/bin/bash
set -e

echo "=========================================="
echo "Setup: vllm-neuron + model weights"
echo "=========================================="

# --- 1. Install vllm-neuron from fork with MiMo support ---
echo ""
echo "[1/3] Installing vllm-neuron (feature/mimo-support branch)..."

# Use the NxDI venv as base
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Clone and install vllm-neuron
if [ ! -d /tmp/vllm-neuron ]; then
    git clone --branch feature/mimo-support https://github.com/whn09/vllm-neuron.git /tmp/vllm-neuron
fi
cd /tmp/vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
pip install s5cmd

# Verify installation
python3 -c "import vllm_neuron; print('vllm-neuron installed:', vllm_neuron.__file__)"
vllm --version 2>/dev/null || echo "vllm CLI check done"

# --- 2. Download MiMo-V2-Flash BF16 weights ---
echo ""
echo "[2/3] Downloading MiMo-V2-Flash BF16 weights..."

MIMO_PATH="/opt/dlami/nvme/models/MiMo-V2-Flash-BF16"
if [ -d "$MIMO_PATH" ] && [ "$(ls $MIMO_PATH/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  MiMo weights already exist at $MIMO_PATH, skipping download"
else
    echo "  Downloading from s3://datalab/xiaomi/models/MiMo-V2-Flash-BF16/ ..."
    mkdir -p "$MIMO_PATH"
    s5cmd cp "s3://datalab/xiaomi/models/MiMo-V2-Flash-BF16/**" "$MIMO_PATH/"
    echo "  Download complete: $(du -sh $MIMO_PATH | cut -f1)"
fi

# --- 3. Verify MiniMax-M2 BF16 weights ---
echo ""
echo "[3/3] Verifying MiniMax-M2 BF16 weights..."

MINIMAX_PATH="/opt/dlami/nvme/models/MiniMax-M2-BF16"
if [ -d "$MINIMAX_PATH" ] && [ "$(ls $MINIMAX_PATH/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  MiniMax weights exist at $MINIMAX_PATH: $(du -sh $MINIMAX_PATH | cut -f1)"
else
    echo "  Downloading from s3://datalab/minimax/model_hf/MiniMax-M2-BF16/ ..."
    mkdir -p "$MINIMAX_PATH"
    s5cmd cp "s3://datalab/minimax/model_hf/MiniMax-M2-BF16/**" "$MINIMAX_PATH/"
    echo "  Download complete: $(du -sh $MINIMAX_PATH | cut -f1)"
fi

# --- Summary ---
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "  vllm-neuron: /tmp/vllm-neuron (feature/mimo-support)"
echo "  MiMo weights: $MIMO_PATH"
echo "  MiniMax weights: $MINIMAX_PATH"
echo "  Disk usage: $(df -h /opt/dlami/nvme | tail -1 | awk '{print $3, "used /", $2, "(" $5 ")"}')"
