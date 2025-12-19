"""
The Hugging Face 128E FP8 checkpoint cannot be directly used for inference on Neuron. we need to run this code to preprocess the
model checkpoints from Hugging Face to run Llama 4 FP8 model on Neuron.

Steps to preprocess:

1. The FP8 format Neuron supports (E4M3) is the FP8_EXP4 (IEEE-754), which differs from the OCP FP8 E4M3/e4m3fn format commonly
available on GPUs. This requires rescaling the checkpoint to make it compatible with Neuron. One of the main difference is that
on Neuron FP8E4M3(FP8_EXP4) datatype, the range is +/-240 while for the OCP FP8 E4M3/e4m3fn, the range is +/-448. Value outside
+/-240 on Neuron devices, for E4M3 might result into NaNs.

2. We need to ensure that the Gate/Up projections for the expert router MLPs are fused, which is not the case with the Hugging
Face checkpoints. This also requires fusing the scales to prevent issues during weight dequantization.
"""

import gc
import json
import os

import torch

from neuronx_distributed_inference.modules.checkpoint import (
    load_state_dict,
    save_state_dict_safetensors,
)

hf_fp8_model_path = ""
save_model_path = ""


def rescale(initial_weight, initial_scale):
    FP8_SCALING_FACTOR = 448.0 / 240.0
    initial_weight_bf16 = initial_weight.bfloat16()

    final_weight_bf16 = initial_weight_bf16 / FP8_SCALING_FACTOR
    final_scale = initial_scale * FP8_SCALING_FACTOR
    return final_weight_bf16.to(torch.float8_e4m3fn), final_scale


state_dict = load_state_dict(hf_fp8_model_path)

with open(os.path.join(hf_fp8_model_path, "config.json"), "r") as f:
    config = json.load(f)

text_config = config["text_config"]

W_DTYPE = torch.float8_e4m3fn
S_DTYPE = torch.bfloat16
NUM_LAYERS = text_config["num_hidden_layers"]
moe_intermediate_size = text_config["intermediate_size"]
hidden_size = text_config["hidden_size"]
num_experts = text_config["num_local_experts"]


state_dict_keys = set(state_dict.keys())

for layer_n in range(NUM_LAYERS):
    prefix = f"language_model.model.layers.{layer_n}."
    if prefix + "feed_forward.shared_expert.up_proj.weight" in state_dict_keys:

        gate_weight, up_weight, down_weight = (
            torch.empty((0, moe_intermediate_size), dtype=W_DTYPE),
            torch.empty((0, moe_intermediate_size), dtype=W_DTYPE),
            torch.empty((0, hidden_size), dtype=W_DTYPE),
        )
        gate_scale, up_scale, down_scale = (
            torch.empty((0, 1), dtype=S_DTYPE),
            torch.empty((0, 1), dtype=S_DTYPE),
            torch.empty((0, 1), dtype=S_DTYPE),
        )

        for expert in range(num_experts):
            print(f"layer : {layer_n} start expert {expert}", end="...")
            gate_weight = torch.cat(
                (
                    gate_weight,
                    state_dict[f"{prefix}feed_forward.experts.{expert}.gate_proj.weight"].T,
                ),
                dim=0,
            )
            gate_scale = torch.cat(
                (
                    gate_scale,
                    state_dict[f"{prefix}feed_forward.experts.{expert}.gate_proj.weight_scale"],
                ),
                dim=0,
            )
            state_dict.pop(f"{prefix}feed_forward.experts.{expert}.gate_proj.weight")
            state_dict.pop(f"{prefix}feed_forward.experts.{expert}.gate_proj.weight_scale")

            up_weight = torch.cat(
                (up_weight, state_dict[f"{prefix}feed_forward.experts.{expert}.up_proj.weight"].T),
                dim=0,
            )
            up_scale = torch.cat(
                (
                    up_scale,
                    state_dict[f"{prefix}feed_forward.experts.{expert}.up_proj.weight_scale"],
                ),
                dim=0,
            )
            state_dict.pop(f"{prefix}feed_forward.experts.{expert}.up_proj.weight")
            state_dict.pop(f"{prefix}feed_forward.experts.{expert}.up_proj.weight_scale")

            down_weight = torch.cat(
                (
                    down_weight,
                    state_dict[f"{prefix}feed_forward.experts.{expert}.down_proj.weight"].T,
                ),
                dim=0,
            )
            down_scale = torch.cat(
                (
                    down_scale,
                    state_dict[f"{prefix}feed_forward.experts.{expert}.down_proj.weight_scale"],
                ),
                dim=0,
            )
            state_dict.pop(f"{prefix}feed_forward.experts.{expert}.down_proj.weight")
            state_dict.pop(f"{prefix}feed_forward.experts.{expert}.down_proj.weight_scale")

            gc.collect()

            print(f"Done expert {expert}")

        gate_up_key = [
            f"{prefix}feed_forward.experts.gate_up_proj",
            f"{prefix}feed_forward.experts.gate_up_proj.scale",
        ]
        down_key = [
            f"{prefix}feed_forward.experts.down_proj",
            f"{prefix}feed_forward.experts.down_proj.scale",
        ]

        state_dict[down_key[0]] = down_weight.view(num_experts, moe_intermediate_size, hidden_size)
        state_dict[down_key[1]] = down_scale.view(num_experts, 1, -1).to(torch.float32)

        state_dict[gate_up_key[0]] = torch.cat(
            (
                gate_weight.view(num_experts, hidden_size, -1),
                up_weight.view(num_experts, hidden_size, -1),
            ),
            dim=2,
        )
        state_dict[gate_up_key[1]] = torch.cat(
            (gate_scale.view(num_experts, 1, -1), up_scale.view(num_experts, 1, -1)), dim=2
        ).to(torch.float32)

        state_dict[gate_up_key[0]], state_dict[gate_up_key[1]] = rescale(
            state_dict[gate_up_key[0]], state_dict[gate_up_key[1]]
        )
        state_dict[down_key[0]], state_dict[down_key[1]] = rescale(
            state_dict[down_key[0]], state_dict[down_key[1]]
        )

        gc.collect()

os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
save_state_dict_safetensors(state_dict, save_model_path)
