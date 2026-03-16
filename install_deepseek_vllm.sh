#!/usr/bin/env bash
# Install DeepSeek V3 model code into the vLLM venv's NXDI package.
#
# The vLLM venv is separate from the NXDI venv and has its own copy of NXDI.
# This script copies the model code there and registers it.
#
# Usage:
#   cd ~/environment/deepseek-nxdi
#   bash install_deepseek_vllm.sh            # auto-detects vLLM venv
#   bash install_deepseek_vllm.sh /path/to/vllm_venv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Locate the vLLM venv ─────────────────────────────────────────────
VENV="${1:-}"
if [[ -z "$VENV" ]]; then
    for candidate in \
        /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13 \
        /opt/aws_neuronx_venv_pytorch_inference_vllm_0_12 \
        /opt/aws_neuronx_venv_pytorch_inference_vllm_0_11 \
    ; do
        if [[ -f "$candidate/bin/activate" ]]; then
            VENV="$candidate"
            break
        fi
    done
fi

if [[ -z "$VENV" || ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: Could not find vLLM venv. Pass the path as an argument:"
    echo "  bash install_deepseek_vllm.sh /path/to/vllm_venv"
    exit 1
fi

# Detect the NXDI package inside the vLLM venv
NXDI_ROOT=$("$VENV/bin/python" -c "
import neuronx_distributed_inference as nxdi
import os
print(os.path.dirname(nxdi.__file__))
" 2>/dev/null || true)

if [[ -z "$NXDI_ROOT" || ! -d "$NXDI_ROOT" ]]; then
    PYTHON_VERSION=$("$VENV/bin/python" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
    NXDI_ROOT="$VENV/lib/$PYTHON_VERSION/site-packages/neuronx_distributed_inference"
fi

if [[ ! -d "$NXDI_ROOT" ]]; then
    echo "ERROR: NXDI package not found in vLLM venv at $NXDI_ROOT"
    exit 1
fi

NXDI_MODELS="$NXDI_ROOT/models"
NXDI_UTILS="$NXDI_ROOT/utils"

echo "vLLM venv : $VENV"
echo "NXDI root : $NXDI_ROOT"

# ── 1. Copy model code ───────────────────────────────────────────────
echo ""
echo "Installing DeepSeek V3 model code into vLLM venv..."
mkdir -p "$NXDI_MODELS/deepseek"
cp "$SCRIPT_DIR/src/neuronx_distributed_inference/models/deepseek/__init__.py" "$NXDI_MODELS/deepseek/"
cp "$SCRIPT_DIR/src/neuronx_distributed_inference/models/deepseek/modeling_deepseek.py" "$NXDI_MODELS/deepseek/"
cp "$SCRIPT_DIR/src/neuronx_distributed_inference/models/deepseek/rope_util.py" "$NXDI_MODELS/deepseek/"
echo "  -> $NXDI_MODELS/deepseek/"

# ── 2. Register model in constants.py ────────────────────────────────
# NOTE: vLLM parses architecture "DeepseekV3ForCausalLM" -> model_type "deepseekv3" (no underscore)
# So we register as BOTH "deepseek_v3" (for direct NXDI use) and "deepseekv3" (for vLLM)
echo ""
CONSTANTS="$NXDI_UTILS/constants.py"
if grep -q "NeuronDeepseekV3ForCausalLM" "$CONSTANTS"; then
    echo "DeepSeek V3 already registered in constants.py"
else
    echo "Registering DeepSeek V3 in constants.py..."
    sed -i '/^END_TO_END_MODEL/i from neuronx_distributed_inference.models.deepseek.modeling_deepseek import NeuronDeepseekV3ForCausalLM' "$CONSTANTS"
    sed -i '/^}/i\    "deepseek_v3": {"causal-lm": NeuronDeepseekV3ForCausalLM},' "$CONSTANTS"
    sed -i '/^}/i\    "deepseekv3": {"causal-lm": NeuronDeepseekV3ForCausalLM},' "$CONSTANTS"
    echo "  -> Updated $CONSTANTS"
fi

# Ensure "deepseekv3" (no underscore) is also registered for vLLM compatibility
if ! grep -q '"deepseekv3"' "$CONSTANTS"; then
    echo "Adding vLLM-compatible 'deepseekv3' key..."
    sed -i '/^}/i\    "deepseekv3": {"causal-lm": NeuronDeepseekV3ForCausalLM},' "$CONSTANTS"
    echo "  -> Added 'deepseekv3' entry"
fi

# ── 3. Verify imports ────────────────────────────────────────────────
echo ""
echo "Verifying imports in vLLM venv..."
"$VENV/bin/python" -c "
from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    NeuronDeepseekV3ForCausalLM,
)
print('  DeepSeek V3 imports OK in vLLM venv')
"

echo ""
echo "============================================================"
echo " vLLM installation complete!"
echo "============================================================"
echo ""
echo " Serving command:"
echo "   source $VENV/bin/activate"
echo "   VLLM_PLUGINS=neuron vllm serve /path/to/DeepSeek-V3 \\"
echo "       --tensor-parallel-size 32 --max-model-len 4096 --max-num-seqs 1 \\"
echo "       --trust-remote-code --dtype bfloat16 \\"
echo "       --no-enable-prefix-caching --no-enable-chunked-prefill"
echo ""
echo " IMPORTANT: Use VLLM_PLUGINS=neuron if optimum_neuron is also installed"
echo " IMPORTANT: vLLM parses 'DeepseekV3ForCausalLM' -> model type 'deepseekv3' (no underscore)"
echo ""
