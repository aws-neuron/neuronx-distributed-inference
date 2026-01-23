import gc
from typing import Any, Dict

from neuronx_distributed.parallel_layers.mappings import gather_from_sequence_parallel_region
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    _helper_concat_and_delete_qkv, 
    get_modules_to_not_convert
    )
from neuronx_distributed_inference.modules.attention.gqa import GroupQueryAttention_QKV


class GroupQueryAttentionQKVWithoutRMSKernel(GroupQueryAttention_QKV):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rms_norm_eps = 1e-05 # Dummy value
        # fused_rmsnorm is forced to False since Cohere2 use regular LayerNorm
        self.fused_rmsnorm = False


def convert_state_dict_to_fused_qkv(llama_state_dict: Dict[str, Any], cfg: InferenceConfig) -> Dict[str, Any]:
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):  # noqa: E741
        _helper_concat_and_delete_qkv(llama_state_dict, l, "weight")
        if (cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized) and f"self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(llama_state_dict, l, "scale")

    gc.collect()

    return llama_state_dict
