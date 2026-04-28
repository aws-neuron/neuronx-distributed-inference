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

# Resolve repo-relative paths up front — we cd into $HOME/vllm-neuron below,
# after which $0's relative form would no longer resolve.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/vllm-neuron-patch.patch"
CONTRIB_SRC="$(cd "$SCRIPT_DIR/.." && pwd)/src"

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

NEURON_FP8_PATH="${MIMO_PATH}-Neuron-FP8"
COMPILED_PATH="/opt/dlami/nvme/compiled/mimo_v2_5_bs32_moetp1_ep64_fp8_vllm"

echo ""
echo "========================================================================"
echo "Next steps"
echo "========================================================================"
echo ""
echo "1. Preprocess the FP8 checkpoint for Neuron (~16 min, ~15 GB peak RAM):"
echo ""
echo "     python $CONTRIB_SRC/conversion_script/preprocess_mimo_v2_5_fp8.py \\"
echo "         --hf_model_path $MIMO_PATH \\"
echo "         --save_path     $NEURON_FP8_PATH \\"
echo "         --tp_degree 64"
echo ""
echo "2. Export the environment variables used by the smoke / bench scripts:"
echo ""
echo "     # --- Required ---"
echo "     # Contrib package src (registers NeuronMiMoV2ForCausalLM with vllm-neuron)."
echo "     export NXDI_CONTRIB_MIMO_V2_5_SRC=$CONTRIB_SRC"
echo "     # vLLM's builtin arch validator only knows MiMoV2FlashForCausalLM, so the"
echo "     # preprocess rewrites the checkpoint's config.json architectures to that"
echo "     # name. Alias V2.5 src to the Flash env var so vllm-neuron's contrib hook"
echo "     # registers mimov2flash -> our V2.5 NeuronMiMoV2ForCausalLM class."
echo "     export NXDI_CONTRIB_MIMO_V2_FLASH_SRC=\"\$NXDI_CONTRIB_MIMO_V2_5_SRC\""
echo "     # Preprocessed Neuron-FP8 checkpoint."
echo "     export MIMO_V2_5_PATH=$NEURON_FP8_PATH"
echo ""
echo "     # --- Optional (recommended) ---"
echo "     # vLLM compiles into <checkpoint>/neuron-compiled-artifacts/<hash>/ by"
echo "     # default. Pin it to a persistent shared location so multiple configs"
echo "     # don't collide and you can reuse the NEFF / sharded weights across runs."
echo "     export NEURON_COMPILED_ARTIFACTS=$COMPILED_PATH"
echo "     # NxDI's HLO/NEFF staging workdir (.hlo_module.pb etc). Default is"
echo "     # /tmp/nxd_model/<compile-name>/; on Trn2 /tmp is wiped by the nightly"
echo "     # reboot, and parallel compiles sharing the same basename silently"
echo "     # overwrite each other's staged HLOs. Pin to a unique per-config"
echo "     # directory under persistent storage."
echo "     export BASE_COMPILE_WORK_DIR=/opt/dlami/nvme/tmp/nxd_model/\$(basename $COMPILED_PATH)"
echo "     # First-time compile of V2.5's 256-expert MoE takes ~30 min (NEFF HLO +"
echo "     # shard_checkpoint for 64 ranks). Extend vLLM's ready timeout."
echo "     export VLLM_ENGINE_READY_TIMEOUT_S=7200"
echo ""
echo "3. Run the benchmark:"
echo ""
echo "     bash $SCRIPT_DIR/bench_mimo_v2_5.sh"
echo ""
