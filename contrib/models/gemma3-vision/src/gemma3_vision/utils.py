from collections import OrderedDict
import gc

import torch
from neuronx_distributed_inference.models.config import NeuronConfig


StateDict = OrderedDict[str, torch.FloatTensor]


def _helper_concat_and_delete_qkv(state_dict: StateDict, prefix: str, attr: str) -> None:
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
            dim=0
        )
        del state_dict[full_state_key_q_proj]
        del state_dict[full_state_key_k_proj]
        del state_dict[full_state_key_v_proj]


def convert_state_dict_to_fused_qkv(
        state_dict: StateDict,
        num_layers: int,
        neuron_config: NeuronConfig,
        prefix: str
        ) -> StateDict:
    for layer_num in range(num_layers):
        layer_prefix = prefix.format(layer_num=layer_num)
        _helper_concat_and_delete_qkv(state_dict, layer_prefix, "weight")
        _helper_concat_and_delete_qkv(state_dict, layer_prefix, "bias")
        is_qkv_quantized = (
            (neuron_config.quantized_mlp_kernel_enabled or neuron_config.quantized) and \
            f"{layer_prefix}.qkv_proj.q_proj.scale" in state_dict
        )
        if is_qkv_quantized:
            _helper_concat_and_delete_qkv(state_dict, layer_prefix, "scale")

    gc.collect()
    return state_dict
