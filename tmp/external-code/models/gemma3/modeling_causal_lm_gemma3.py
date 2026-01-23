# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

import math
from typing import Dict, List, Optional

import torch
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM

from models.gemma3.modeling_gemma3_text import NeuronGemma3TextModel
from models.utils import (
    convert_state_dict_to_fused_qkv,
    StateDict
)

class TextGemma3InferenceConfig(InferenceConfig):

    def __init__(
        self,
        neuron_config: NeuronConfig,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(
            neuron_config=neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )

        # NeuronLlamaMLP expects the activation type to be at text_config.hidden_act
        # Enable to fully reuse NeuronLlamaMLP
        if not hasattr(self, "hidden_act"):
            self.hidden_act = self.hidden_activation
            del self.hidden_activation

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim", # for gemma3, head_dim != hidden_size // num_attention_heads
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "query_pre_attn_scalar",
            "rope_scaling",
            "sliding_window",
        ]


class NeuronTextGemma3ForCausalLM(NeuronBaseForCausalLM):

    _model_cls = NeuronGemma3TextModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Gemma3ForCausalLM
        return Gemma3ForCausalLM.from_pretrained(model_path, **kwargs) # nosec B615

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: StateDict) -> None:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: StateDict, inference_config: InferenceConfig) -> StateDict:
        neuron_config = inference_config.neuron_config
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
            ".self_attn.q_norm.": ".self_attn.q_layernorm.",
            ".self_attn.k_norm.": ".self_attn.k_layernorm.",
        }

        # At the time of writing, NxDI (Neuron 2.26) attention layer does not provide a simple way to use a custom 
        # scaling factor for raw attention scores (QK^T) while ensuring all optimizations (e.g. kernels) remain available 
        # To work around this, we fuse the scaling factor into the weights (knowing that  the attention layer will use the 
        # default math.sqrt(inference_config.head_dim) value)
        default_qk_scaling_factor_inv = math.sqrt(float(inference_config.query_pre_attn_scalar))
        gemma_qk_scaling_factor = 1.0 / math.sqrt(float(inference_config.head_dim))
        gamma = math.sqrt(gemma_qk_scaling_factor * default_qk_scaling_factor_inv)

        new_state_dict = {}
        for key, weights in state_dict.items():
            if 'vision_tower.' in key:
                continue
            if 'language_model.model.' in key:
                key = key.replace('language_model.model.', "")
            for atten_key in attention_keys:
                if atten_key in key:
                    replacement_atten_key = attention_keys[atten_key]
                    key = key.replace(atten_key, replacement_atten_key)
                    break
            if key.endswith((".q_proj.weight", ".k_proj.weight")):
                orig_dtype = weights.dtype
                weights = (weights.to(dtype=torch.float32) * gamma).to(dtype=orig_dtype)
            new_state_dict[key] = weights

        if neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(
                state_dict=new_state_dict, 
                num_layers=inference_config.num_hidden_layers,
                neuron_config=inference_config.neuron_config,
                prefix="layers.{layer_num}.self_attn"
                )

        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.rank_util.rank"] = torch.arange(0, neuron_config.local_ranks_size)

        tp_degree = neuron_config.tp_degree
        for i in range(inference_config.num_hidden_layers):
            new_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return TextGemma3InferenceConfig
