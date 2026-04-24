#!/bin/bash
# Setup for MiMo-V2-Flash vLLM benchmarking on Trn2.
#
# This clones upstream vllm-project/vllm-neuron at release-0.5.0 (which already
# has the mimov2flash -> mimo_v2_flash model_type rewrite), then applies
# vllm-neuron-patch.patch to add a runtime registration hook so the contrib
# NeuronMiMoV2ForCausalLM is plugged into both NxDI's MODEL_TYPES and vLLM's
# ModelRegistry at vllm-neuron plugin init time.
set -e

echo "=========================================="
echo "Setup: vllm-neuron + MiMo-V2-Flash weights"
echo "=========================================="

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

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

# Figure out where this contrib package's src/ lives so the registration hook
# can add it to sys.path inside vllm-neuron.
CONTRIB_SRC="$(cd "$(dirname "$0")/.." && pwd)/src"

echo ""
echo "Setup complete. Before running the benchmark, export:"
echo "  export MIMO_V2_FLASH_PATH=$MIMO_PATH"
echo "  export NXDI_CONTRIB_MIMO_V2_FLASH_SRC=$CONTRIB_SRC"
