#!/bin/bash
# Setup for MiMo-V2.5 vLLM benchmarking on Trn2.
#
# Clones upstream vllm-project/vllm-neuron at release-0.5.0 and applies
# vllm-neuron-patch.patch, which adds a runtime registration hook so the
# contrib NeuronMiMoV2ForCausalLM is plugged into both NxDI's MODEL_TYPES
# (under the key "mimov2") and vLLM's ModelRegistry (as
# MiMoV2ForCausalLM) at vllm-neuron plugin init time.
#
# Then downloads XiaomiMiMo/MiMo-V2.5 from HuggingFace (FP8 blockwise, ~320 GB).
set -e

echo "=========================================="
echo "Setup: vllm-neuron + MiMo-V2.5 weights"
echo "=========================================="

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

PATCH_FILE="$(cd "$(dirname "$0")" && pwd)/vllm-neuron-patch.patch"

echo ""
echo "[1/2] Installing vllm-neuron (release-0.5.0) with the contrib registration patch..."

if [ ! -d $HOME/vllm-neuron ]; then
    git clone --branch release-0.5.0 https://github.com/vllm-project/vllm-neuron.git $HOME/vllm-neuron
fi

cd $HOME/vllm-neuron

# Apply patch (idempotent via `git apply --check` first).
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "  Applied $PATCH_FILE"
else
    echo "  Patch already applied or conflicts; continuing."
fi

pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .

python3 -c "import vllm_neuron; print('vllm-neuron installed:', vllm_neuron.__file__)"

echo ""
echo "[2/2] Downloading MiMo-V2.5 FP8 weights from HuggingFace..."

MIMO_PATH="${MIMO_V2_5_PATH:-/opt/dlami/nvme/models/MiMo-V2.5}"
if [ -d "$MIMO_PATH" ] && [ "$(ls "$MIMO_PATH"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  MiMo-V2.5 weights already exist at $MIMO_PATH, skipping download"
else
    mkdir -p "$MIMO_PATH"
    huggingface-cli download XiaomiMiMo/MiMo-V2.5 --local-dir "$MIMO_PATH" --max-workers 16
    echo "  Download complete: $(du -sh $MIMO_PATH | cut -f1)"
fi

CONTRIB_SRC="$(cd "$(dirname "$0")/.." && pwd)/src"

echo ""
echo "Next, preprocess the FP8 checkpoint for Neuron (~15 min, ~15 GB peak RAM):"
echo "  python $CONTRIB_SRC/conversion_script/preprocess_mimo_v2_5_fp8.py \\"
echo "      --hf_model_path $MIMO_PATH \\"
echo "      --save_path     ${MIMO_PATH}-Neuron-FP8 \\"
echo "      --tp_degree 64"
echo ""
echo "Then before running the benchmark, export:"
echo "  export MIMO_V2_5_PATH=${MIMO_PATH}-Neuron-FP8"
echo "  export NXDI_CONTRIB_MIMO_V2_5_SRC=$CONTRIB_SRC"
