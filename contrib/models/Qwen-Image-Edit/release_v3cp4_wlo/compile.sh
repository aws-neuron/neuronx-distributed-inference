#!/bin/bash
# V3 CP=4 + WLO compile (TP=4 × CP=4 = world 16, uses 16 of the chip's 32 logical cores).
#
# This is the fast configuration: doubling cores vs the world=8 baseline (CP=2) halves
# the transformer step (793ms → 411ms) because QIE is compute-bound. Plus WLO (bit-exact).
# E2E 4.72s — beats H100 vLLM-Omni's 4.99s. Output is correct (see tryon_cp4.png).
#
# Three components must ALL be compiled at world=16:
#   - transformer:  --tp_degree 4 --world_size 16  (cp_degree = 16/4 = 4)
#   - vision:       VISION_WORLD_SIZE=16
#   - language model: LM_WORLD_SIZE=16  (keep --max_sequence_length 1024 to match runtime)
# VAE is single-device (world-agnostic) — symlinked from the CP=2 build.
set -euo pipefail
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd "$(dirname "$0")/.."   # repo root

export PYTHONPATH=src:${PYTHONPATH:-}
export QIE_ALLREDUCE_BF16=1 QIE_OPT_LEVEL=2 QIE_CC_TILING=4 QIE_WLO=1

MODEL_PATH="${QIE_MODEL_PATH:-/home/ubuntu/checkpoints/Qwen-Image-Edit-2509-step4000}"
DST="${QIE_OUT_DIR:-/opt/dlami/nvme/compiled_models_qwen_image_edit_step4000_896x1184_cp4}"
VAE_SRC="${QIE_VAE_SRC:-/opt/dlami/nvme/compiled_models_qwen_image_edit_step4000_896x1184_vaebatch6}"

echo "=== transformer TP=4 CP=4 world=16 (+WLO) ==="
python src/compile_transformer_v3_cp.py \
  --model_path "$MODEL_PATH" \
  --height 1184 --width 896 --max_sequence_length 512 \
  --patch_multiplier 3 --tp_degree 4 --world_size 16 --batch_size 1 \
  --compiled_models_dir "$DST" --compiler_workdir /opt/dlami/nvme/cw_cp4 2>&1 | tail -4

echo "=== vision encoder world=16 ==="
VISION_WORLD_SIZE=16 python src/compile_vision_encoder_v3.py \
  --model_path "$MODEL_PATH" \
  --compiled_models_dir "$DST" --compiler_workdir /opt/dlami/nvme/cw_vis16 2>&1 | tail -3

echo "=== language model world=16 (max_seq 1024) ==="
LM_WORLD_SIZE=16 python src/compile_language_model_v3.py \
  --model_path "$MODEL_PATH" --max_sequence_length 1024 \
  --compiled_models_dir "$DST" --compiler_workdir /opt/dlami/nvme/cw_lm16 2>&1 | tail -3

echo "=== symlink VAE (single device) ==="
for c in vae_encoder vae_decoder quant_conv post_quant_conv vae_config.json; do
  [ -e "$VAE_SRC/$c" ] && ln -sfn "$VAE_SRC/$c" "$DST/$c"
done
echo "DONE — compiled to $DST"
