#!/usr/bin/env python3
"""
LTX-2.3 Half-Resolution Compilation for Stage 1
=================================================
Compiles the LTX-2.3 22B DiT transformer backbone for HALF resolution
(192x256) as required by the distilled model's two-stage pipeline.

The distilled model generates at half resolution in Stage 1, then
upscales to full resolution in Stage 2.

Half-res latent grid: 6x8 (192/32 x 256/32)
Video tokens: 4 frames * 6 * 8 = 192

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
    torchrun --nproc_per_node=4 compile_transformer_halfres.py
"""

import compile_transformer

# Override constants for half-resolution Stage 1
compile_transformer.VIDEO_SEQ = 192  # 4 frames * 6h * 8w (192x256 resolution)
compile_transformer.LATENT_H = 6  # 192 / 32
compile_transformer.LATENT_W = 8  # 256 / 32
compile_transformer.COMPILE_DIR = (
    "/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_halfres"
)

if __name__ == "__main__":
    compile_transformer.main()
