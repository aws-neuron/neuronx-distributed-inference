#!/bin/bash
# Setup for MiMo-V2.5-Pro vLLM benchmarking on Trn2.
#
# This clones upstream vllm-project/vllm-neuron at release-0.5.0 (which already
# has the mimov2flash -> mimo_v2 model_type rewrite), then applies
# vllm-neuron-patch.patch to add a runtime registration hook so the contrib
# NeuronMiMoV2ForCausalLM is plugged into both NxDI's MODEL_TYPES and vLLM's
# ModelRegistry at vllm-neuron plugin init time.
set -e

echo "=========================================="
echo "Setup: vllm-neuron + MiMo-V2.5-Pro weights"
echo "=========================================="

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

PATCH_FILE="$(cd "$(dirname "$0")" && pwd)/vllm-neuron-patch.patch"

echo ""
echo "[1/2] Installing vllm-neuron (release-0.5.0) with the contrib registration patch..."

if [ ! -d $HOME/vllm-neuron ]; then
    git clone --branch release-0.5.0 https://github.com/vllm-project/vllm-neuron.git $HOME/vllm-neuron
fi

cd $HOME/vllm-neuron

# Apply patch. Distinguish three cases so a corrupt/conflicting patch is a
# hard error rather than being silently skipped (a malformed hunk header once
# caused this to no-op, leaving the contrib model unregistered and vLLM unable
# to load MiMo-V2.5-Pro):
#   - applies cleanly            -> apply it
#   - already applied (reverse)  -> skip, fine
#   - neither                    -> abort with a clear message
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "  Applied $PATCH_FILE"
elif git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "  Patch already applied; skipping."
else
    echo "  ERROR: $PATCH_FILE does not apply cleanly and is not already applied." >&2
    echo "  Refusing to continue with an unpatched vllm-neuron (MiMo would fail to load)." >&2
    git apply --check "$PATCH_FILE"   # surface the real error, then abort via set -e
    exit 1
fi

pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
pip install s5cmd

python3 -c "import vllm_neuron; print('vllm-neuron installed:', vllm_neuron.__file__)"

echo ""
echo "[2/2] Downloading MiMo-V2.5-Pro Neuron-FP8 weights..."

MIMO_PATH="${MIMO_V2_FLASH_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8}"
if [ -d "$MIMO_PATH" ] && [ "$(ls "$MIMO_PATH"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  MiMo weights already exist at $MIMO_PATH, skipping download"
else
    echo "  Downloading Neuron-FP8 weights from your S3 bucket (edit the URI if needed)..."
    mkdir -p "$MIMO_PATH"
    s5cmd cp "s3://datalab/xiaomi/models/MiMo-V2.5-Pro-Neuron-FP8/**" "$MIMO_PATH/"
    echo "  Download complete: $(du -sh $MIMO_PATH | cut -f1)"
fi

# Figure out where this contrib package's src/ lives so the registration hook
# can add it to sys.path inside vllm-neuron.
CONTRIB_SRC="$(cd "$(dirname "$0")/.." && pwd)/src"

echo ""
echo "Setup complete. Before running the benchmark, export:"
echo "  export MIMO_V2_FLASH_PATH=$MIMO_PATH"
echo "  export NXDI_CONTRIB_MIMO_V2_FLASH_SRC=$CONTRIB_SRC"
