#!/bin/bash
# Setup script for GPU benchmark of Isaac-0.2-2B
# Run on a fresh GPU DLAMI (g6e.xlarge with L40S)
#
# Usage:
#   bash setup_gpu.sh

set -e

echo "=== Isaac GPU Benchmark Setup ==="

# Use the PyTorch 2.7 virtual environment from DLAMI
echo "Setting up Python environment..."
source /opt/dlami/nvme/pytorch-2.7/bin/activate 2>/dev/null || {
    echo "DLAMI venv not found, using system Python..."
    python3 -m venv ~/gpu_bench_env
    source ~/gpu_bench_env/bin/activate
}

# Install vLLM and dependencies
echo "Installing vLLM..."
pip install -U vllm transformers torch pillow requests 2>&1 | tail -5

# Download model (Isaac requires trust_remote_code)
echo "Downloading Isaac-0.2-2B-Preview..."
pip install -U "huggingface_hub[cli]" 2>&1 | tail -3
huggingface-cli download PerceptronAI/Isaac-0.2-2B-Preview --local-dir ~/Isaac-0.2-2B-Preview

echo ""
echo "=== Setup complete ==="
echo "To run benchmark:"
echo "  python benchmark_gpu.py --model ~/Isaac-0.2-2B-Preview"
