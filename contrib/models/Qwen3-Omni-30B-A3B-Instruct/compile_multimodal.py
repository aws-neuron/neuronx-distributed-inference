#!/usr/bin/env python3
"""Compile Qwen3-Omni multimodal model (text MoE + vision encoder) for Neuron.

Both text and vision models use TP=16 with LNC=2, running on 32 physical cores.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    NEURON_RT_VISIBLE_CORES=0-31 python compile_multimodal.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from modeling_qwen3_omni import (
    NeuronQwen3OmniForCausalLM,
    Qwen3OmniInferenceConfig,
    load_qwen3_omni_multimodal_config,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig, NeuronConfig

MODEL_PATH = "/home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct"
COMPILED_PATH = "/home/ubuntu/traced_model/Qwen3-Omni-multimodal"
TP_DEGREE = 16

text_neuron_config = MoENeuronConfig(
    tp_degree=TP_DEGREE,
    batch_size=1,
    seq_len=4096,
    max_context_length=2048,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config={"top_k": 1, "do_sample": False},
    blockwise_matmul_config={"use_torch_block_wise": True},
)

vision_neuron_config = NeuronConfig(
    tp_degree=TP_DEGREE,
    batch_size=1,
    seq_len=4096,
    torch_dtype=torch.bfloat16,
)

config = Qwen3OmniInferenceConfig(
    text_neuron_config=text_neuron_config,
    vision_neuron_config=vision_neuron_config,
    load_config=load_qwen3_omni_multimodal_config(MODEL_PATH),
)

model = NeuronQwen3OmniForCausalLM(MODEL_PATH, config)

print(f"Compiling multimodal model with TP={TP_DEGREE} ...")
t0 = time.perf_counter()
model.compile(COMPILED_PATH)
elapsed = time.perf_counter() - t0
print(f"Compilation complete in {elapsed:.1f}s")
print(f"Compiled model saved to: {COMPILED_PATH}")
