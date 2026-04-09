#!/bin/bash
# Wan2.2 TI2V Compilation Script
#
# Compiles all models for Wan2.2 text/image-to-video on Trainium2.
# Transformer: TP=4, CP=2 (Context Parallel), world_size=8
#
# Usage:
#   ./compile.sh                    # Use default directories
#   ./compile.sh /path/to/output    # Custom output directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"

# Fix nearest-exact -> nearest for Trainium2 compatibility
DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
VAE_FILE="${DIFFUSERS_PATH}/models/autoencoders/autoencoder_kl_wan.py"
if grep -q 'nearest-exact' "${VAE_FILE}" 2>/dev/null; then
    echo "Patching autoencoder_kl_wan.py: nearest-exact -> nearest"
    sed -i 's/nearest-exact/nearest/g' "${VAE_FILE}"
fi

# Configuration
COMPILED_MODELS_DIR="${1:-/opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b}"
COMPILER_WORKDIR="${2:-/opt/dlami/nvme/compiler_workdir_wan2.2_ti2v_5b}"

# Video settings (should match inference)
HEIGHT=512
WIDTH=512
NUM_FRAMES=81
MAX_SEQUENCE_LENGTH=512

# Parallelism
TP_DEGREE=4
WORLD_SIZE=8  # TP=4 x CP=2

echo "=============================================="
echo "Wan2.2 TI2V Compilation"
echo "=============================================="
echo "Output: ${COMPILED_MODELS_DIR}"
echo "Compiler workdir: ${COMPILER_WORKDIR}"
echo "Resolution: ${HEIGHT}x${WIDTH}, Frames: ${NUM_FRAMES}"
echo "TP degree: ${TP_DEGREE}, World size: ${WORLD_SIZE}"
echo "=============================================="

# Create directories
mkdir -p "${COMPILED_MODELS_DIR}"
mkdir -p "${COMPILER_WORKDIR}"

# Step 1: Cache HuggingFace model (if not already cached)
echo ""
echo "[Step 1/4] Caching HuggingFace model..."
python ${SCRIPT_DIR}/cache_hf_model.py

# Step 2: Compile Text Encoder (TP=4 to match transformer)
# At inference time, the 4 TP checkpoints are duplicated for 2 CP ranks → 8 total
echo ""
echo "[Step 2/4] Compiling Text Encoder (TP=${TP_DEGREE}, world_size=${WORLD_SIZE})..."
python ${SCRIPT_DIR}/compile_text_encoder.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE}

# Step 3: Compile Transformer (TP=4, CP=2 or CFG Parallel)
# Set CFG_PARALLEL=1 to use CFG Parallel (batch=2, no K/V gather) instead of Context Parallel
CFG_PARALLEL="${CFG_PARALLEL:-0}"
CFG_FLAG=""
if [ "${CFG_PARALLEL}" = "1" ]; then
    CFG_FLAG="--cfg_parallel"
    echo ""
    echo "[Step 3/4] Compiling Transformer (TP=${TP_DEGREE}, CFG Parallel, batch=2)..."
else
    echo ""
    echo "[Step 3/4] Compiling Transformer (TP=${TP_DEGREE}, CP=2)..."
fi
python ${SCRIPT_DIR}/compile_transformer.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE} \
    ${CFG_FLAG}

# Step 4: Compile Decoder (Rolling Cache) + post_quant_conv
# Rolling cache carries temporal context between chunks for flicker-free video
# post_quant_conv (float32) is also compiled here
echo ""
echo "[Step 4/4] Compiling Decoder (Rolling Cache, bfloat16) + post_quant_conv..."
python ${SCRIPT_DIR}/compile_decoder_rolling.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --decoder_frames 2 \
    --tp_degree ${WORLD_SIZE} \
    --world_size ${WORLD_SIZE}

# Note: VAE Encoder is NOT compiled to Neuron due to a Neuron compiler bug
# (NCC_IBIR158) in the Conv3D tensorizer at 256x256 spatial resolution.
# For I2V mode, the encoder runs on CPU (runs once per video, negligible overhead).

echo ""
echo "=============================================="
echo "Compilation Complete!"
echo "=============================================="
echo "Models saved to: ${COMPILED_MODELS_DIR}"
echo ""
echo "To run T2V inference:"
echo "  python run_wan2.2_ti2v.py \\"
echo "    --compiled_models_dir ${COMPILED_MODELS_DIR} \\"
echo "    --prompt 'A cat walks on the grass, realistic'"
echo ""
echo "To run I2V inference:"
echo "  python run_wan2.2_ti2v.py \\"
echo "    --compiled_models_dir ${COMPILED_MODELS_DIR} \\"
echo "    --image input.png \\"
echo "    --prompt 'A cat walks on the grass, realistic'"
echo "=============================================="
