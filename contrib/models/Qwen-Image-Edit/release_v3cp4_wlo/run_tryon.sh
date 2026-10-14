#!/bin/bash
# V3 CP=4 + WLO try-on inference (TP=4 × CP=4 = world 16, 16 cores). ~4.7s E2E.
# Cloth (image 1) onto model (image 2); lower-body garment replaced.
#
# KEY runtime env vs CP=2: QIE_WORLD_SIZE=16 and NEURON_RT_NUM_CORES=16 (uses 16 cores).
set -euo pipefail
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd "$(dirname "$0")/.."   # repo root

export NEURON_RT_NUM_CORES=16
export QIE_WORLD_SIZE=16
export PYTHONPATH=src:${PYTHONPATH:-}

OUT_DIR="${QIE_OUT_DIR:-/opt/dlami/nvme/compiled_models_qwen_image_edit_step4000_896x1184_cp4}"
OUT_IMG="${1:-release_v3cp4_wlo/tryon_cp4.png}"
CLOTH="${QIE_CLOTH:-cloth/1686634914e5521d5145f5c95c1b4ee70560881686.jpg}"
MODEL="${QIE_MODEL_IMG:-input_img/1764042352dfda7d588f8da62b2d3aea69d3889bcb.webp}"

python src/run_qwen_image_edit.py \
  --images "$CLOTH" "$MODEL" \
  --prompt "让图2的模特换上图1的下装" \
  --negative_prompt "" \
  --output "$OUT_IMG" \
  --height 1184 --width 896 --image_h 448 --image_w 336 \
  --patch_multiplier 3 --max_sequence_length 1024 \
  --num_inference_steps 8 --true_cfg_scale 1.0 --seed 42 \
  --compiled_models_dir "$OUT_DIR" \
  --use_v3_cp \
  2>&1
