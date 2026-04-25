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

if [ ! -d $HOME/vllm-neuron ]; then
    git clone --branch release-0.5.0 https://github.com/vllm-project/vllm-neuron.git $HOME/vllm-neuron
fi

cd $HOME/vllm-neuron

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
echo "[2/2] Fetching MiniMax-M2.7 FP8 weights (HuggingFace)..."

# Source HF checkpoint (FP8 OCP, ~215 GB). Preprocessing this into
# Neuron-FP8 via src/conversion_script/preprocess_minimax_m2_fp8.py is
# a separate step (~13 min); the bench script expects the preprocessed
# output at $MINIMAX_M2_PATH.
HF_PATH="${MINIMAX_M2_HF_PATH:-/opt/dlami/nvme/models/MiniMax-M2.7}"
if [ -d "$HF_PATH" ] && [ "$(ls "$HF_PATH"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  HF weights already at $HF_PATH, skipping download"
else
    echo "  Downloading HF FP8 weights (this takes ~5 min at S3 speeds)..."
    huggingface-cli download MiniMaxAI/MiniMax-M2.7 --local-dir "$HF_PATH"
    echo "  Download complete: $(du -sh $HF_PATH | cut -f1)"
fi

MINIMAX_PATH="${MINIMAX_M2_PATH:-/opt/dlami/nvme/models/MiniMax-M2.7-Neuron-FP8}"
if [ -d "$MINIMAX_PATH" ] && [ "$(ls "$MINIMAX_PATH"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Neuron-FP8 checkpoint already exists at $MINIMAX_PATH"
else
    echo ""
    echo "Next step (not run automatically): preprocess HF -> Neuron-FP8"
    echo "  python contrib/models/MiniMax-M2/src/conversion_script/preprocess_minimax_m2_fp8.py \\"
    echo "    --hf_model_path $HF_PATH \\"
    echo "    --save_path     $MINIMAX_PATH \\"
    echo "    --tp_degree 64"
fi

CONTRIB_SRC="$(cd "$(dirname "$0")/.." && pwd)/src"

echo ""
echo "Setup complete. Before running the benchmark, export:"
echo "  export MINIMAX_M2_PATH=$MINIMAX_PATH"
echo "  export NXDI_CONTRIB_MINIMAX_M2_SRC=$CONTRIB_SRC"
