#!/usr/bin/env bash
# Install DeepSeek V3 model code into the NXDI venv and register it.
#
# Usage:
#   cd ~/environment/deepseek-nxdi
#   bash install_deepseek.sh            # auto-detects the NXDI venv
#   bash install_deepseek.sh /path/to/venv  # or specify it explicitly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Locate the NXDI venv ─────────────────────────────────────────────
VENV="${1:-}"
if [[ -z "$VENV" ]]; then
    for candidate in \
        /opt/aws_neuronx_venv_pytorch_2_11_nxd_inference \
        /opt/aws_neuronx_venv_pytorch_2_10_nxd_inference \
        /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference \
        /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference \
    ; do
        if [[ -f "$candidate/bin/activate" ]]; then
            VENV="$candidate"
            break
        fi
    done
fi

if [[ -z "$VENV" || ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: Could not find NXDI venv. Pass the path as an argument:"
    echo "  bash install_deepseek.sh /path/to/venv"
    exit 1
fi

# Detect the NXDI package root directory
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
    echo "ERROR: NXDI package not found at $NXDI_ROOT"
    exit 1
fi

NXDI_MODELS="$NXDI_ROOT/models"
NXDI_UTILS="$NXDI_ROOT/utils"

echo "NXDI venv : $VENV"
echo "NXDI root : $NXDI_ROOT"
echo "Models dir: $NXDI_MODELS"

# ── 1. Copy model code into NXDI ─────────────────────────────────────
echo ""
echo "Installing DeepSeek V3 model code..."
mkdir -p "$NXDI_MODELS/deepseek"
cp "$SCRIPT_DIR/src/neuronx_distributed_inference/models/deepseek/__init__.py" "$NXDI_MODELS/deepseek/"
cp "$SCRIPT_DIR/src/neuronx_distributed_inference/models/deepseek/modeling_deepseek.py" "$NXDI_MODELS/deepseek/"
cp "$SCRIPT_DIR/src/neuronx_distributed_inference/models/deepseek/rope_util.py" "$NXDI_MODELS/deepseek/"
echo "  -> $NXDI_MODELS/deepseek/"

# ── 2. Register model in constants.py ────────────────────────────────
echo ""
CONSTANTS="$NXDI_UTILS/constants.py"
if grep -q "NeuronDeepseekV3ForCausalLM" "$CONSTANTS"; then
    echo "DeepSeek V3 already registered in constants.py"
else
    echo "Registering DeepSeek V3 in constants.py..."
    # Add import line before the first constant definition
    sed -i '/^END_TO_END_MODEL/i from neuronx_distributed_inference.models.deepseek.modeling_deepseek import NeuronDeepseekV3ForCausalLM' "$CONSTANTS"
    # Add to MODEL_TYPES dict before the closing brace
    sed -i '/^}/i\    "deepseek_v3": {"causal-lm": NeuronDeepseekV3ForCausalLM},' "$CONSTANTS"
    echo "  -> Updated $CONSTANTS"
fi

# ── 3. Register model in inference_demo.py ────────────────────────────
echo ""
DEMO="$NXDI_ROOT/inference_demo.py"
if [[ -f "$DEMO" ]]; then
    if grep -q "NeuronDeepseekV3ForCausalLM" "$DEMO"; then
        echo "DeepSeek V3 already registered in inference_demo.py"
    else
        echo "Registering DeepSeek V3 in inference_demo.py..."
        # Add import after the last model import line
        sed -i '/from neuronx_distributed_inference.models.qwen3_moe/a from neuronx_distributed_inference.models.deepseek.modeling_deepseek import NeuronDeepseekV3ForCausalLM' "$DEMO"
        # Add to MODEL_TYPES dict before closing brace
        sed -i '/^MODEL_TYPES = {/,/^}/ s/^}/    "deepseek_v3": {"causal-lm": NeuronDeepseekV3ForCausalLM},\n}/' "$DEMO"
        echo "  -> Updated $DEMO"
    fi
else
    echo "  inference_demo.py not found at $DEMO (skipping)"
fi

# ── 4. Ensure __init__.py files exist for test imports ───────────────
echo ""
echo "Setting up test structure..."
for d in test test/unit test/unit/models test/unit/models/deepseek; do
    touch "$SCRIPT_DIR/$d/__init__.py" 2>/dev/null || true
done

# ── 5. Verify imports ────────────────────────────────────────────────
echo ""
echo "Verifying imports..."
source "$VENV/bin/activate"
python -c "
from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3DenseMLP,
    DeepseekV3InferenceConfig,
    DeepseekV3Router,
    NeuronDeepseekV3DecoderLayer,
    NeuronDeepseekV3ForCausalLM,
    NeuronDeepseekV3Model,
    convert_deepseek_v3_hf_to_neuron_state_dict,
)
print('  DeepSeek V3 imports OK')
"

# ── 6. Print next steps ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Installation complete!"
echo "============================================================"
echo ""
echo " To use:"
echo "   source $VENV/bin/activate"
echo "   cd $SCRIPT_DIR"
echo ""
echo " Run unit tests (CPU, no device needed):"
echo "   python -m pytest test/unit/models/deepseek/ -v"
echo ""
echo " Run logit matching test (needs Neuron device, tp=2):"
echo "   python examples/test_logit_matching.py"
echo ""
echo " Full logit matching (multi-step, needs Neuron device, tp=2):"
echo "   python examples/test_logit_matching_full.py"
echo ""
echo " Full model inference (DeepSeek V3 671B, trn2.48xlarge):"
echo "   python examples/generation_deepseek_v3.py \\"
echo "       --model-path /path/to/DeepSeek-V3 \\"
echo "       --traced-model-path /tmp/deepseek_v3_traced \\"
echo "       --tp-degree 32 --seq-len 4096"
echo ""
echo " vLLM serving (after also running: bash install_deepseek_vllm.sh):"
echo "   VLLM_PLUGINS=neuron vllm serve /path/to/DeepSeek-V3 \\"
echo "       --tensor-parallel-size 32 --max-model-len 4096 --max-num-seqs 1 \\"
echo "       --trust-remote-code --dtype bfloat16 \\"
echo "       --no-enable-prefix-caching --no-enable-chunked-prefill"
echo ""
