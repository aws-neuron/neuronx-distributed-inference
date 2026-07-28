#!/bin/bash
# YuE on AWS Neuron -- Setup Script
#
# Prerequisites:
#   - trn2 instance (trn2.3xlarge or larger)
#   - Deep Learning AMI Neuron (Ubuntu 24.04) 20260227 or later
#   - At least 300GB EBS storage
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [MODEL_DIR]
#
# MODEL_DIR defaults to /mnt/models

set -euo pipefail

MODEL_DIR="${1:-/mnt/models}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13"

echo "=== YuE on Neuron: Setup ==="
echo "Model directory: ${MODEL_DIR}"
echo "Script directory: ${SCRIPT_DIR}"
echo "Neuron venv: ${VENV}"
echo ""

# Activate Neuron venv
source "${VENV}/bin/activate"

# Install extra dependencies
echo "Installing extra Python packages..."
pip install -q soundfile omegaconf einops descript-audiotools 'protobuf>=3.20,<5'

# Create model directory
sudo mkdir -p "${MODEL_DIR}"
sudo chown ubuntu:ubuntu "${MODEL_DIR}"
cd "${MODEL_DIR}"

# Download models from HuggingFace
echo ""
echo "Downloading YuE S1 (7B) model..."
if [ ! -d "YuE-s1-7B-anneal-en-cot" ]; then
    huggingface-cli download m-a-p/YuE-s1-7B-anneal-en-cot \
        --local-dir YuE-s1-7B-anneal-en-cot \
        --local-dir-use-symlinks False
else
    echo "  Already exists, skipping."
fi

echo "Downloading YuE S2 (1B) model..."
if [ ! -d "YuE-s2-1B-general" ]; then
    huggingface-cli download m-a-p/YuE-s2-1B-general \
        --local-dir YuE-s2-1B-general \
        --local-dir-use-symlinks False
else
    echo "  Already exists, skipping."
fi

echo "Downloading xcodec_mini inference model..."
if [ ! -d "xcodec_mini_infer" ]; then
    huggingface-cli download m-a-p/xcodec_mini_infer \
        --local-dir xcodec_mini_infer \
        --local-dir-use-symlinks False
else
    echo "  Already exists, skipping."
fi

echo "Cloning YuE repository (inference code + tokenizer)..."
if [ ! -d "YuE" ]; then
    git clone https://github.com/multimodal-art-projection/YuE.git
else
    echo "  Already exists, skipping."
fi

# Create compiled model directories
mkdir -p compiled/s1_tp2_bs2_ctx2048 compiled/s1_tp2_bs2_ctx2048_nki
mkdir -p compiled/s2_tp1 compiled/s2_tp1_nki compiled/s2_tp1_nki_bs2

# Copy scripts and sample files to MODEL_DIR
echo ""
echo "Copying scripts to ${MODEL_DIR}/..."
cp "${SCRIPT_DIR}/src/yue_e2e_neuron.py" "${MODEL_DIR}/"
cp "${SCRIPT_DIR}/src/yue_stage1_worker.py" "${MODEL_DIR}/"
cp "${SCRIPT_DIR}/src/yue_stage2_worker.py" "${MODEL_DIR}/"
cp "${SCRIPT_DIR}/src/nki_mlp_patch.py" "${MODEL_DIR}/"
cp "${SCRIPT_DIR}/genre.txt" "${MODEL_DIR}/"
cp "${SCRIPT_DIR}/lyrics.txt" "${MODEL_DIR}/"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Models downloaded to: ${MODEL_DIR}"
echo "Scripts copied to: ${MODEL_DIR}"
echo ""
echo "Quick start (default -- KV-cache optimization, no NKI):"
echo "  source ${VENV}/bin/activate"
echo "  cd ${MODEL_DIR}"
echo "  python yue_e2e_neuron.py --genre_txt genre.txt --lyrics_txt lyrics.txt --seed 123"
echo ""
echo "With NKI kernel optimization (20% faster, requires recompile):"
echo "  python yue_e2e_neuron.py --genre_txt genre.txt --lyrics_txt lyrics.txt --seed 123 --nki-kernels"
echo ""
echo "With S2 batching (process 2 chunks simultaneously):"
echo "  python yue_e2e_neuron.py --genre_txt genre.txt --lyrics_txt lyrics.txt --seed 123 --nki-kernels --s2-batch-size 2"
echo ""
echo "First run will compile the models (~7 min). Subsequent runs use --skip-compile."
echo ""
echo "Options:"
echo "  --nki-kernels          Enable NKI MLP TKG fused kernels (20% speedup)"
echo "  --s2-batch-size N      S2 batch size (default: 1, try 2 for best pipeline time)"
echo "  --s1-tp-degree N       S1 TP degree (default: 2, use 1 for LNC=1)"
echo "  --no-cfg               Disable CFG (faster but lower quality vocals)"
echo "  --skip-compile         Skip compilation (use pre-compiled models)"
echo "  --guidance-scale-first Float, CFG scale for first segment (default: 1.5)"
echo "  --guidance-scale-rest  Float, CFG scale for subsequent segments (default: 1.2)"
echo "  --max_new_tokens       Max tokens per segment (default: 3000)"
echo "  --run_n_segments       Number of lyrics segments to generate (default: 2)"
echo "  --seed                 Random seed (default: 42)"
