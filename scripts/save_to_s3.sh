#!/bin/bash
# Save DeepSeek V3 NXDI artifacts to S3 for spot instance recovery.
#
# What gets saved:
#   - Compiled model (NEFFs, tokenizer, config) — saves 12 min compilation
#   - Per-rank sharded checkpoints — saves 3.5 hour weight sharding
#   - HF weights (optional) — saves 40 min download from HuggingFace
#   - Neuron compile cache — may avoid recompilation
#   - Source code + PLAN file — everything needed to rebuild
#
# Usage:
#   bash scripts/save_to_s3.sh [BUCKET_NAME]

set -euo pipefail

BUCKET="${1:-deepseek-v3-nxdi-artifacts}"
REGION="${AWS_DEFAULT_REGION:-us-east-2}"
TRACED_DIR="/scratch/deepseek_v3_traced"
HF_WEIGHTS_DIR="$HOME/environment/models/DeepSeek-V3-0324-FP8"
COMPILE_CACHE="/var/tmp/neuron-compile-cache"
CODE_DIR="$HOME/environment/deepseek-nxdi"
PREFIX="deepseek-v3-0324"

echo "=== DeepSeek V3 NXDI → S3 ==="
echo "Bucket: s3://$BUCKET"
echo "Region: $REGION"

# Optimize S3 transfer settings for high-bandwidth instances
aws configure set default.s3.max_concurrent_requests 100
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.preferred_transfer_client crt
aws configure set default.s3.target_bandwidth 100Gb/s

# Create bucket if it doesn't exist
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    echo "Creating bucket s3://$BUCKET ..."
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION"
fi

# 1. Compiled model (NEFFs + tokenizer)
if [ -d "$TRACED_DIR" ]; then
    echo ""
    echo "[1/5] Uploading compiled model from $TRACED_DIR ..."
    aws s3 sync "$TRACED_DIR" "s3://$BUCKET/$PREFIX/traced_model/" \
        --exclude "weights/*"
    echo "  Done. $(du -sh "$TRACED_DIR" --exclude=weights 2>/dev/null | cut -f1 || echo 'N/A')"
else
    echo "[1/5] SKIP: No compiled model at $TRACED_DIR"
fi

# 2. Per-rank sharded checkpoints (if they exist)
if [ -d "$TRACED_DIR/weights" ]; then
    NSHARDS=$(ls "$TRACED_DIR/weights/"tp*_sharded_checkpoint.safetensors 2>/dev/null | wc -l)
    if [ "$NSHARDS" -gt 0 ]; then
        echo ""
        echo "[2/5] Uploading $NSHARDS per-rank sharded checkpoints (~1.37TB) ..."
        aws s3 sync "$TRACED_DIR/weights/" "s3://$BUCKET/$PREFIX/sharded_weights/" \
            --no-progress
        echo "  Done. $(du -sh "$TRACED_DIR/weights" | cut -f1)"
    else
        echo "[2/5] SKIP: No sharded checkpoints in $TRACED_DIR/weights/"
    fi
else
    echo "[2/5] SKIP: No weights/ dir (use save_sharded_checkpoint=True next time)"
fi

# 3. HF weights (original FP8 model from HuggingFace)
if [ -d "$HF_WEIGHTS_DIR" ]; then
    echo ""
    echo "[3/5] Uploading HF weights from $HF_WEIGHTS_DIR (~642GB) ..."
    aws s3 sync "$HF_WEIGHTS_DIR" "s3://$BUCKET/$PREFIX/hf_weights/" \
        --no-progress
    echo "  Done. $(du -sh "$HF_WEIGHTS_DIR" | cut -f1)"
else
    echo "[3/5] SKIP: No HF weights at $HF_WEIGHTS_DIR"
fi

# 4. Neuron compile cache
if [ -d "$COMPILE_CACHE" ]; then
    echo ""
    echo "[4/5] Uploading Neuron compile cache ..."
    tar -czf /tmp/neuron_compile_cache.tar.gz -C /var/tmp neuron-compile-cache/
    aws s3 cp /tmp/neuron_compile_cache.tar.gz "s3://$BUCKET/$PREFIX/neuron_compile_cache.tar.gz"
    rm /tmp/neuron_compile_cache.tar.gz
    echo "  Done. $(du -sh "$COMPILE_CACHE" | cut -f1)"
else
    echo "[4/5] SKIP: No compile cache at $COMPILE_CACHE"
fi

# 5. Source code + PLAN
echo ""
echo "[5/5] Uploading source code + PLAN ..."
tar -czf /tmp/deepseek_nxdi_code.tar.gz -C "$HOME/environment" deepseek-nxdi/
aws s3 cp /tmp/deepseek_nxdi_code.tar.gz "s3://$BUCKET/$PREFIX/deepseek_nxdi_code.tar.gz"
aws s3 cp "$CODE_DIR/PLAN_deepseek_v3.md" "s3://$BUCKET/$PREFIX/PLAN_deepseek_v3.md"
rm /tmp/deepseek_nxdi_code.tar.gz
echo "  Done."

echo ""
echo "=== Upload complete ==="
echo "To restore on a new instance:"
echo "  bash scripts/restore_from_s3.sh $BUCKET"
