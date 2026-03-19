#!/bin/bash
# Restore DeepSeek V3 NXDI artifacts from S3 after spot instance reclamation.
#
# Prerequisites:
#   - trn2.48xlarge with NXDI AMI
#   - AWS CLI configured (instance role or credentials)
#
# This script restores code, compiled model, sharded weights, and optionally HF weights.
# After restoring, run the benchmark with --skip-compile for fast startup (~8 min).
#
# Usage:
#   bash scripts/restore_from_s3.sh [BUCKET_NAME]

set -euo pipefail

BUCKET="${1:-deepseek-v3-nxdi-artifacts}"
PREFIX="deepseek-v3-0324"

echo "=== Restoring DeepSeek V3 NXDI from S3 ==="
echo "Bucket: s3://$BUCKET"

# Optimize S3 transfer settings for high-bandwidth instances
aws configure set default.s3.max_concurrent_requests 100
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.preferred_transfer_client crt
aws configure set default.s3.target_bandwidth 100Gb/s

# 1. Mount NVMe scratch
echo ""
echo "[1/7] Setting up NVMe scratch ..."
if ! mountpoint -q /scratch; then
    sudo mkfs.ext4 -L scratch /dev/nvme1n1
    sudo mkdir -p /scratch
    sudo mount /dev/nvme1n1 /scratch
    sudo chown ubuntu:ubuntu /scratch
    echo "  Mounted /dev/nvme1n1 at /scratch"
else
    echo "  /scratch already mounted"
fi

# 2. Set up NVMe swap (400GB on second NVMe drive)
echo ""
echo "[2/7] Setting up NVMe swap ..."
if ! mountpoint -q /scratch2; then
    sudo mkfs.ext4 -L scratch2 /dev/nvme2n1
    sudo mkdir -p /scratch2
    sudo mount /dev/nvme2n1 /scratch2
    sudo chown ubuntu:ubuntu /scratch2
    echo "  Mounted /dev/nvme2n1 at /scratch2"
else
    echo "  /scratch2 already mounted"
fi
if ! swapon --show | grep -q /scratch2/swapfile; then
    sudo fallocate -l 400G /scratch2/swapfile
    sudo chmod 600 /scratch2/swapfile
    sudo mkswap /scratch2/swapfile
    sudo swapon /scratch2/swapfile
    echo "  400GB NVMe swap enabled on /scratch2"
else
    echo "  NVMe swap already active"
fi

# 3. Restore source code
echo ""
echo "[3/7] Restoring source code ..."
aws s3 cp "s3://$BUCKET/$PREFIX/deepseek_nxdi_code.tar.gz" /tmp/
tar -xzf /tmp/deepseek_nxdi_code.tar.gz -C ~/environment/
rm /tmp/deepseek_nxdi_code.tar.gz
echo "  Restored to ~/environment/deepseek-nxdi/"

# 4. Restore compiled model (NEFFs)
echo ""
echo "[4/7] Restoring compiled model ..."
mkdir -p /scratch/deepseek_v3_traced
aws s3 sync "s3://$BUCKET/$PREFIX/traced_model/" /scratch/deepseek_v3_traced/
echo "  Restored to /scratch/deepseek_v3_traced/"

# 5. Restore compile cache
echo ""
echo "[5/7] Restoring Neuron compile cache ..."
if aws s3 ls "s3://$BUCKET/$PREFIX/neuron_compile_cache.tar.gz" >/dev/null 2>&1; then
    aws s3 cp "s3://$BUCKET/$PREFIX/neuron_compile_cache.tar.gz" /tmp/
    tar -xzf /tmp/neuron_compile_cache.tar.gz -C /var/tmp/
    rm /tmp/neuron_compile_cache.tar.gz
    echo "  Restored to /var/tmp/neuron-compile-cache/"
else
    echo "  No compile cache in S3 (will recompile if needed)"
fi

# 6. Restore sharded weights (if they exist)
echo ""
echo "[6/7] Checking for sharded weights ..."
NSHARDS=$(aws s3 ls "s3://$BUCKET/$PREFIX/sharded_weights/" 2>/dev/null | grep -c safetensors || true)
if [ "$NSHARDS" -gt 0 ]; then
    echo "  Found $NSHARDS sharded checkpoints (~1.37TB). Downloading ..."
    mkdir -p /scratch/deepseek_v3_traced/weights
    aws s3 sync "s3://$BUCKET/$PREFIX/sharded_weights/" /scratch/deepseek_v3_traced/weights/ \
        --no-progress
    echo "  Downloaded to /scratch/deepseek_v3_traced/weights/"
else
    echo "  No sharded weights in S3."
    echo "  You will need HF weights and must run without --skip-compile (~4.5 hours)."
fi

# 7. Restore HF weights (if they exist and aren't already on disk)
echo ""
echo "[7/7] Checking for HF weights ..."
HF_WEIGHTS_DIR="$HOME/environment/models/DeepSeek-V3-0324-FP8"
NHF=$(aws s3 ls "s3://$BUCKET/$PREFIX/hf_weights/" 2>/dev/null | grep -c safetensors || true)
if [ -d "$HF_WEIGHTS_DIR" ] && [ "$(ls "$HF_WEIGHTS_DIR"/*.safetensors 2>/dev/null | wc -l)" -ge 100 ]; then
    echo "  HF weights already on disk at $HF_WEIGHTS_DIR"
elif [ "$NHF" -gt 0 ]; then
    echo "  Found $NHF files in S3. Downloading HF weights (~642GB) ..."
    mkdir -p "$HF_WEIGHTS_DIR"
    aws s3 sync "s3://$BUCKET/$PREFIX/hf_weights/" "$HF_WEIGHTS_DIR/" --no-progress
    echo "  Downloaded to $HF_WEIGHTS_DIR/"
else
    echo "  No HF weights in S3 or on disk."
    echo "  Download from HuggingFace if needed for recompilation:"
    echo "    huggingface-cli download deepseek-ai/DeepSeek-V3-0324 --local-dir $HF_WEIGHTS_DIR"
fi

# Install code into NXDI venv
echo ""
echo "[Install] Installing code into NXDI venv ..."
cd ~/environment/deepseek-nxdi && bash install_deepseek.sh

echo ""
echo "=== Restore complete ==="
echo ""
echo "Next steps:"
if [ "$NSHARDS" -gt 0 ]; then
    echo "  # Sharded weights available — fast load path (~8 min):"
    echo "  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate"
    echo "  python examples/generation_deepseek_v3.py \\"
    echo "      --traced-model-path /scratch/deepseek_v3_traced --skip-compile \\"
    echo "      --tp-degree 64 --seq-len 512 --batch-size 1"
else
    echo "  # No sharded weights — full run needed (~4.5 hours):"
    echo "  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate"
    echo "  python examples/generation_deepseek_v3.py \\"
    echo "      --model-path ~/environment/models/DeepSeek-V3-0324-FP8 \\"
    echo "      --traced-model-path /scratch/deepseek_v3_traced \\"
    echo "      --tp-degree 64 --seq-len 512 --batch-size 1"
fi
