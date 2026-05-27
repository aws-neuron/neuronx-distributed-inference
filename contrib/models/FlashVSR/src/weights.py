# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Weight conversion for FlashVSR DiT on AWS Trainium.

Converts DiffSynth Studio / HuggingFace diffusers weights to the NxDI Neuron
module naming convention used by NeuronFlashVSRDiT.

Source format: FlashVSR safetensors (diffusion_pytorch_model_streaming_dmd.safetensors)
  DiffSynth/Native Wan naming:
    blocks.N.self_attn.q.weight -> blocks.N.self_attn.to_q.weight
    blocks.N.ffn.0.weight -> blocks.N.ffn_gelu_proj.weight
    blocks.N.modulation -> blocks.N.scale_shift_table
    head.head.weight -> proj_out.weight
    text_embedding.0.weight -> condition_embedder.text_embedder_linear_1.weight

Model: JunhaoZhuang/FlashVSR-v1.1 (Wan 2.1 1.3B variant)
"""

import math
import re
import torch
from collections import OrderedDict


def pad_attention_weights_for_tp(
    state_dict: OrderedDict,
    num_heads: int = 12,
    head_dim: int = 128,
    tp_degree: int = 1,
) -> OrderedDict:
    """Pad attention Q/K/V/O weights for TP head padding.

    When num_heads is not divisible by tp_degree, the model pads heads to the
    next multiple. This pads weights to match compiled model shapes.
    """
    dim = num_heads * head_dim
    padded_heads = math.ceil(num_heads / tp_degree) * tp_degree
    padded_dim = padded_heads * head_dim

    if padded_dim == dim:
        return state_dict

    result = OrderedDict()

    for key, value in state_dict.items():
        if re.search(r"\.(self_attn|cross_attn)\.to_(q|k|v)\.weight$", key):
            if value.shape[0] == dim:
                padded = torch.zeros(padded_dim, value.shape[1], dtype=value.dtype)
                padded[:dim, :] = value
                result[key] = padded
                continue

        elif re.search(r"\.(self_attn|cross_attn)\.to_(q|k|v)\.bias$", key):
            if value.shape[0] == dim:
                padded = torch.zeros(padded_dim, dtype=value.dtype)
                padded[:dim] = value
                result[key] = padded
                continue

        elif re.search(r"\.(self_attn|cross_attn)\.to_out\.weight$", key):
            if value.shape[1] == dim and value.shape[0] == dim:
                padded = torch.zeros(dim, padded_dim, dtype=value.dtype)
                padded[:, :dim] = value
                result[key] = padded
                continue

        elif re.search(r"\.(self_attn|cross_attn)\.norm_(q|k)\.weight$", key):
            if value.shape[0] == dim:
                padded = torch.ones(padded_dim, dtype=value.dtype)
                padded[:dim] = value
                result[key] = padded
                continue

        result[key] = value

    return result


def convert_diffsynth_to_neuron_state_dict(state_dict: dict) -> OrderedDict:
    """Convert DiffSynth Studio / Native Wan weights to NxDI Neuron format."""
    neuron_sd = OrderedDict()

    for key, value in state_dict.items():
        new_key = key

        # Self-attention Q/K/V: add 'to_' prefix
        new_key = re.sub(r"\.self_attn\.(q|k|v)\.", r".self_attn.to_\1.", new_key)
        new_key = new_key.replace(".self_attn.o.", ".self_attn.to_out.")

        # Cross-attention Q/K/V/O
        new_key = re.sub(r"\.cross_attn\.(q|k|v)\.", r".cross_attn.to_\1.", new_key)
        new_key = new_key.replace(".cross_attn.o.", ".cross_attn.to_out.")

        # FFN layers
        new_key = new_key.replace(".ffn.0.", ".ffn_gelu_proj.")
        new_key = new_key.replace(".ffn.2.", ".ffn_out.")

        # Block modulation -> scale_shift_table
        if "blocks." in new_key:
            new_key = new_key.replace(".modulation", ".scale_shift_table")

        # Output head
        new_key = new_key.replace("head.head.", "proj_out.")
        if new_key == "head.modulation":
            new_key = "scale_shift_table"

        # Condition embedder
        new_key = new_key.replace(
            "text_embedding.0.", "condition_embedder.text_embedder_linear_1."
        )
        new_key = new_key.replace(
            "text_embedding.2.", "condition_embedder.text_embedder_linear_2."
        )
        new_key = new_key.replace(
            "time_embedding.0.", "condition_embedder.time_embedder_linear_1."
        )
        new_key = new_key.replace(
            "time_embedding.2.", "condition_embedder.time_embedder_linear_2."
        )
        new_key = new_key.replace("time_projection.1.", "condition_embedder.time_proj.")

        neuron_sd[new_key] = value.clone().detach().contiguous()

    return neuron_sd


def convert_diffusers_to_neuron_state_dict(state_dict: dict) -> OrderedDict:
    """Convert HuggingFace diffusers WanTransformer3DModel weights to NxDI format."""
    neuron_sd = OrderedDict()

    for key, value in state_dict.items():
        new_key = key

        new_key = new_key.replace(".attn1.", ".self_attn.")
        new_key = new_key.replace(".attn2.", ".cross_attn.")
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".ffn.net.0.proj.", ".ffn_gelu_proj.")
        new_key = new_key.replace(".ffn.net.2.", ".ffn_out.")
        new_key = new_key.replace(
            "condition_embedder.text_embedder.linear_1.",
            "condition_embedder.text_embedder_linear_1.",
        )
        new_key = new_key.replace(
            "condition_embedder.text_embedder.linear_2.",
            "condition_embedder.text_embedder_linear_2.",
        )
        new_key = new_key.replace(
            "condition_embedder.time_embedder.linear_1.",
            "condition_embedder.time_embedder_linear_1.",
        )
        new_key = new_key.replace(
            "condition_embedder.time_embedder.linear_2.",
            "condition_embedder.time_embedder_linear_2.",
        )

        neuron_sd[new_key] = value.clone().detach().contiguous()

    return neuron_sd


def detect_format_and_convert(state_dict: dict, tp_degree: int = 1) -> OrderedDict:
    """Auto-detect weight format and convert to NxDI Neuron format.

    Detection: 'head.head.weight' or 'text_embedding.*' -> DiffSynth format.
    """
    is_native = "head.head.weight" in state_dict or any(
        k.startswith("text_embedding.") for k in state_dict
    )

    if is_native:
        result = convert_diffsynth_to_neuron_state_dict(state_dict)
    else:
        result = convert_diffusers_to_neuron_state_dict(state_dict)

    if tp_degree > 1:
        result = pad_attention_weights_for_tp(
            result,
            num_heads=12,
            head_dim=128,
            tp_degree=tp_degree,
        )

    return result
