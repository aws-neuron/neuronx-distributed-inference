#!/bin/bash

# Compile LongCat-Image-Edit for Neuron (trn2.48xlarge)
#
# Components:
#   1. VAE: 2D AutoencoderKL (standard FLUX VAE)
#   2. Transformer: FLUX-style with TP=4, CP=2 (10 dual + 20 single stream blocks)
#   3. Vision Encoder: Qwen2.5-VL ViT with TP=4 (same as Qwen reference)
#   4. Language Model: Qwen2.5-VL LM with TP=4 (same as Qwen reference)
#
# Usage:
#   ./compile.sh                    # Compile CP (Context Parallel) with defaults
#   ./compile.sh cfg                # Compile CFG (CFG Parallel, recommended when guidance_scale > 1)
#   ./compile.sh cp 1024 1024 448 512  # Custom dimensions with CP
#   ./compile.sh cfg 1024 1024 448 512 # Custom dimensions with CFG

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"
COMPILED_MODELS_DIR="/opt/dlami/nvme/compiled_models"
COMPILER_WORKDIR="/opt/dlami/nvme/compiler_workdir"

# VAE compiled for full output size (no tiling needed, avoids seam artifacts)
VAE_TILE_SIZE=1024

# Check if first argument is mode selector
MODE="cp"
if [[ "$1" == "cp" || "$1" == "cfg" ]]; then
    MODE="$1"
    shift
fi

# Parse arguments
HEIGHT=${1:-1024}
WIDTH=${2:-1024}
IMAGE_SIZE=${3:-448}
MAX_SEQ_LEN=${4:-1024}
BATCH_SIZE=${5:-1}

echo "============================================"
echo "LongCat-Image-Edit Compilation for Neuron"
echo "============================================"
echo "Output Size: ${HEIGHT}x${WIDTH}"
echo "VAE Tile Size: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE}"
echo "Vision Encoder Image Size: ${IMAGE_SIZE}"
echo "Max Sequence Length: ${MAX_SEQ_LEN}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Mode: ${MODE}"
if [[ "$MODE" == "cfg" ]]; then
    echo "Transformer: FLUX-style, TP=4, DP=2 (CFG Parallel, world_size=8)"
else
    echo "Transformer: FLUX-style, TP=4, CP=2 (Context Parallel, world_size=8)"
fi
echo ""

# Step 1: Download model and install dependencies
echo "[Step 1/5] Downloading model and installing dependencies..."
pip install -r "${SCRIPT_DIR}/../requirements.txt" --quiet
python ${SCRIPT_DIR}/cache_hf_model.py
echo "Model downloaded successfully!"
echo ""

# Step 2: Compile VAE (single device, ~5 min)
echo "[Step 2/5] Compiling VAE (2D AutoencoderKL)..."
echo "  Tile size: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE}"
python ${SCRIPT_DIR}/compile_vae.py \
    --height ${VAE_TILE_SIZE} \
    --width ${VAE_TILE_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "VAE compiled!"
echo ""

# Step 3: Compile Transformer (TP=4, world_size=8)
if [[ "$MODE" == "cfg" ]]; then
    echo "[Step 3/5] Compiling FLUX Transformer (CFG Parallel, TP=4, DP=2)..."
    neuron_parallel_compile python ${SCRIPT_DIR}/compile_transformer_cfg.py \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --tp_degree 4 \
        --world_size 8 \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --compiled_models_dir ${COMPILED_MODELS_DIR} \
        --compiler_workdir ${COMPILER_WORKDIR}
    echo "CFG Transformer compiled!"
else
    echo "[Step 3/5] Compiling FLUX Transformer (Context Parallel, TP=4, CP=2)..."
    neuron_parallel_compile python ${SCRIPT_DIR}/compile_transformer.py \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --tp_degree 4 \
        --world_size 8 \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --batch_size ${BATCH_SIZE} \
        --compiled_models_dir ${COMPILED_MODELS_DIR} \
        --compiler_workdir ${COMPILER_WORKDIR}
    echo "CP Transformer compiled!"
fi
echo ""

# Step 4: Compile Vision Encoder (TP=4, ~10 min)
echo "[Step 4/5] Compiling Vision Encoder (TP=4, float32)..."
python ${SCRIPT_DIR}/compile_vision_encoder.py \
    --image_size ${IMAGE_SIZE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "Vision Encoder compiled!"
echo ""

# Step 5: Compile Language Model (TP=4, ~15 min)
echo "[Step 5/5] Compiling Language Model (TP=4)..."
neuron_parallel_compile python ${SCRIPT_DIR}/compile_language_model.py \
    --max_sequence_length ${MAX_SEQ_LEN} \
    --batch_size ${BATCH_SIZE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "Language Model compiled!"
echo ""

echo "============================================"
echo "Compilation Complete!"
echo "============================================"
echo ""
echo "Compiled models saved to: ${COMPILED_MODELS_DIR}/"
echo "  - vae_encoder/ (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE})"
echo "  - vae_decoder/ (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE})"
if [[ "$MODE" == "cfg" ]]; then
    echo "  - transformer_cfg/ (TP=4, DP=2, CFG Parallel, output: ${HEIGHT}x${WIDTH}, batch=2)"
else
    echo "  - transformer/ (TP=4, CP=2, output: ${HEIGHT}x${WIDTH})"
fi
echo "  - vision_encoder/ (TP=4, float32)"
echo "  - language_model/ (TP=4)"
echo ""
echo "To run inference:"
if [[ "$MODE" == "cfg" ]]; then
    echo "  # CFG Parallel (recommended when guidance_scale > 1):"
    echo "  NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \\"
    echo "      --image input.jpg \\"
    echo "      --prompt \"your edit instruction\" \\"
    echo "      --use_cfg_parallel --warmup"
    echo ""
    echo "  Note: CFG Parallel batches negative+positive prompts for ~2x denoising speedup"
else
    echo "  # Context Parallel:"
    echo "  NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \\"
    echo "      --image input.jpg \\"
    echo "      --prompt \"your edit instruction\" \\"
    echo "      --warmup"
fi
