# Copyright 2025 © Amazon.com and Affiliates
"""Utility functions for Isaac NxDI contrib model."""

from collections import OrderedDict
import gc

import torch
from neuronx_distributed_inference.models.config import NeuronConfig


StateDict = OrderedDict[str, torch.FloatTensor]


def _helper_concat_and_delete_qkv(
    state_dict: StateDict, prefix: str, attr: str
) -> None:
    """Concatenate Q, K, V weights into fused Wqkv tensor and delete originals."""
    full_state_key_q_proj = f"{prefix}.qkv_proj.q_proj.{attr}"
    full_state_key_k_proj = f"{prefix}.qkv_proj.k_proj.{attr}"
    full_state_key_v_proj = f"{prefix}.qkv_proj.v_proj.{attr}"

    if (
        full_state_key_q_proj in state_dict
        and full_state_key_k_proj in state_dict
        and full_state_key_v_proj in state_dict
    ):
        state_dict[f"{prefix}.qkv_proj.Wqkv.{attr}"] = torch.cat(
            [
                state_dict[full_state_key_q_proj],
                state_dict[full_state_key_k_proj],
                state_dict[full_state_key_v_proj],
            ],
            dim=0,
        )
        del state_dict[full_state_key_q_proj]
        del state_dict[full_state_key_k_proj]
        del state_dict[full_state_key_v_proj]


def convert_state_dict_to_fused_qkv(
    state_dict: StateDict,
    num_layers: int,
    neuron_config: NeuronConfig,
    prefix: str,
) -> StateDict:
    """Convert separate Q, K, V weights to fused QKV format for all layers."""
    for layer_num in range(num_layers):
        layer_prefix = prefix.format(layer_num=layer_num)
        _helper_concat_and_delete_qkv(state_dict, layer_prefix, "weight")
        _helper_concat_and_delete_qkv(state_dict, layer_prefix, "bias")
        is_qkv_quantized = (
            neuron_config.quantized_mlp_kernel_enabled or neuron_config.quantized
        ) and f"{layer_prefix}.qkv_proj.q_proj.scale" in state_dict
        if is_qkv_quantized:
            _helper_concat_and_delete_qkv(state_dict, layer_prefix, "scale")

    gc.collect()
    return state_dict


def pixel_shuffle_varlen(hidden_states: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Apply pixel shuffle (channel concatenation) to vision encoder output.

    This is a deterministic CPU-side operation that merges scale x scale patches
    by concatenating along the channel dimension.

    Isaac's pixel shuffle:
    - Input:  (batch, num_patches, hidden_dim) where num_patches = (H/p * W/p)
    - After reshape to (batch, H/p, W/p, hidden_dim)
    - Group scale x scale patches and concatenate channels
    - Output: (batch, num_patches / scale^2, hidden_dim * scale^2)

    For Isaac: hidden_dim=1152, scale=2 -> output hidden_dim=4608

    Args:
        hidden_states: Vision encoder output of shape (batch, num_patches, hidden_dim)
        scale: Pixel shuffle scale factor (default: 2)

    Returns:
        Shuffled tensor of shape (batch, num_patches // scale^2, hidden_dim * scale^2)
    """
    batch_size, num_patches, hidden_dim = hidden_states.shape

    # Compute spatial dimensions
    h = w = int(num_patches**0.5)
    assert h * w == num_patches, f"num_patches {num_patches} is not a perfect square"
    assert h % scale == 0 and w % scale == 0, (
        f"Spatial dims ({h}, {w}) not divisible by scale {scale}"
    )

    # Reshape to spatial: (batch, h, w, hidden_dim)
    hidden_states = hidden_states.view(batch_size, h, w, hidden_dim)

    # Group into scale x scale blocks
    new_h = h // scale
    new_w = w // scale
    hidden_states = hidden_states.view(
        batch_size, new_h, scale, new_w, scale, hidden_dim
    )

    # Rearrange: (batch, new_h, new_w, scale, scale, hidden_dim)
    hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Concatenate channels: (batch, new_h * new_w, hidden_dim * scale^2)
    hidden_states = hidden_states.view(
        batch_size, new_h * new_w, hidden_dim * scale * scale
    )

    return hidden_states
