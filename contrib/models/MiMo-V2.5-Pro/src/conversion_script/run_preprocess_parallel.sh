#!/bin/bash
# Parallel wrapper around preprocess_mimo_v2_parallel.py.
#
# Each worker dequants one MoE layer at a time (peak ~25 GB per layer on
# V2.5-Pro's 6144 hidden / 384 experts / 2048 intermediate shape). 12
# workers stay under ~300 GB CPU RAM on a 2 TB box while keeping the
# 192-core CPU busy. On a trn2.48xl that brings total wall time from
# ~30 min (serial) to ~5-6 min.
#
# Env:
#   HF_MODEL_PATH    raw HF checkpoint (default: /opt/dlami/nvme/models/MiMo-V2.5-Pro)
#   SAVE_PATH        output Neuron checkpoint (default: /opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8)
#   TP_DEGREE        tensor-parallel degree used at compile time (default: 64)
#   N_WORKERS        concurrent layer workers (default: 12)
#   VENV             venv with torch + safetensors + contrib pkg on sys.path
set -e

VENV=${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}
source "$VENV/bin/activate"

HF_MODEL_PATH=${HF_MODEL_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Pro}
SAVE_PATH=${SAVE_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8}
TP_DEGREE=${TP_DEGREE:-64}
N_WORKERS=${N_WORKERS:-12}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

exec python3 "$SCRIPT_DIR/preprocess_mimo_v2_parallel.py" \
    --hf_model_path "$HF_MODEL_PATH" \
    --save_path "$SAVE_PATH" \
    --tp_degree "$TP_DEGREE" \
    --workers "$N_WORKERS"
