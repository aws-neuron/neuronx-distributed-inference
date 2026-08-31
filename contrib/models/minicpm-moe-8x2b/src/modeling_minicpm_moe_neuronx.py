#!/usr/bin/env python3
"""
MiniCPM-MoE NeuronX Implementation
Port of openbmb/MiniCPM-MoE-8x2B for AWS NeuronX hardware
Following Qwen3-MoE pattern exactly.
"""

import gc
import math
import warnings
from typing import List, Optional, Tuple

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding, ColumnParallelLinear
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

from transformers import AutoModelForCausalLM


class MiniCPMRMSNorm(nn.Module):
    """RMSNorm for MiniCPM-MoE"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_rmsnorm_cls():
    return MiniCPMRMSNorm if cpu_mode() else CustomRMSNorm


class ScaledParallelEmbedding(nn.Module):
    """Wrapper that applies MiniCPM embedding scaling."""
    def __init__(self, embedding, scale_emb):
        super().__init__()
        self.embedding = embedding
        self.scale_emb = scale_emb
        # Forward weight attribute for checkpoint loading
        self.weight = embedding.weight
    
    def forward(self, input_ids):
        return self.embedding(input_ids) * self.scale_emb


def convert_minicpm_moe_hf_to_neuron_state_dict(neuron_state_dict, config):
    """Convert MiniCPM-MoE HuggingFace state dict to NeuronX format."""
    
    neuron_state_dict["rank_util.rank"] = torch.arange(0, config.neuron_config.tp_degree, dtype=torch.int32)

    # Rename embed_tokens.weight to embed_tokens.embedding.weight for ScaledParallelEmbedding
    if "embed_tokens.weight" in neuron_state_dict:
        neuron_state_dict["embed_tokens.embedding.weight"] = neuron_state_dict["embed_tokens.weight"]
        del neuron_state_dict["embed_tokens.weight"]

    for l in range(config.num_hidden_layers):
        neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # Router weights
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.mlp.gate.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

        # Expert weights transformation
        intermediate_size, hidden_size = neuron_state_dict[f"layers.{l}.mlp.experts.0.w1.weight"].shape
        device = neuron_state_dict[f"layers.{l}.mlp.experts.0.w1.weight"].device
        dtype = neuron_state_dict[f"layers.{l}.mlp.experts.0.w1.weight"].dtype

        gate_up_proj = torch.empty(config.num_experts, hidden_size, 2 * intermediate_size, dtype=dtype, device=device)
        down_proj = torch.empty(config.num_experts, intermediate_size, hidden_size, dtype=dtype, device=device)
        
        for e in range(config.num_experts):
            gate_proj_weights = neuron_state_dict[f"layers.{l}.mlp.experts.{e}.w1.weight"].T.detach().clone()
            up_proj_weights = neuron_state_dict[f"layers.{l}.mlp.experts.{e}.w3.weight"].T.detach().clone()
            down_proj_weights = neuron_state_dict[f"layers.{l}.mlp.experts.{e}.w2.weight"].T.detach().clone()

            gate_up_proj[e, :, :intermediate_size] = gate_proj_weights
            gate_up_proj[e, :, intermediate_size:] = up_proj_weights
            down_proj[e] = down_proj_weights

            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.w1.weight"]
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.w2.weight"]
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.w3.weight"]
        
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj
        gc.collect()

    return neuron_state_dict


class MiniCPMMoEInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        # Calculate head_dim before super().__init__ since it validates required attrs
        if 'hidden_size' in kwargs and 'num_attention_heads' in kwargs:
            kwargs['head_dim'] = kwargs['hidden_size'] // kwargs['num_attention_heads']
        
        super().__init__(*args, **kwargs)
        self.num_local_experts = self.num_experts
        self.n_shared_experts = 0
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"
        self.neuron_config.glu_mlp = True
        # Required attributes for inference
        self.output_attentions = False
        self.output_hidden_states = False
        self.use_return_dict = True

    def get_required_attributes(self) -> List[str]:
        return ["head_dim", "hidden_size", "intermediate_size", "num_attention_heads", 
                "num_hidden_layers", "num_key_value_heads", "num_experts", 
                "num_experts_per_tok", "vocab_size", "max_position_embeddings", 
                "rms_norm_eps", "rope_theta", "scale_emb", "scale_depth", "dim_model_base"]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config=None, **kwargs):
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config_dict = {
            'vocab_size': hf_config.vocab_size,
            'hidden_size': hf_config.hidden_size,
            'intermediate_size': hf_config.intermediate_size,
            'num_hidden_layers': hf_config.num_hidden_layers,
            'num_attention_heads': hf_config.num_attention_heads,
            'num_key_value_heads': hf_config.num_key_value_heads,
            'num_experts': hf_config.num_experts,
            'num_experts_per_tok': hf_config.num_experts_per_tok,
            'max_position_embeddings': hf_config.max_position_embeddings,
            'rms_norm_eps': hf_config.rms_norm_eps,
            'rope_theta': getattr(hf_config, 'rope_theta', 10000.0),
            'scale_emb': hf_config.scale_emb,
            'dim_model_base': hf_config.dim_model_base,
            'scale_depth': hf_config.scale_depth,
            'hidden_act': hf_config.hidden_act,
            'pad_token_id': getattr(hf_config, 'pad_token_id', 0),
            'tie_word_embeddings': getattr(hf_config, 'tie_word_embeddings', True),
        }
        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronMiniCPMMoEAttention(NeuronAttentionBase):
    def __init__(self, config: MiniCPMMoEInferenceConfig):
        rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
        )


class NeuronMiniCPMMoEDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPMMoEInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronMiniCPMMoEAttention(config=config)
        self.mlp = initialize_moe_module(config=config)
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Attention with 4-tuple unpacking
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        
        # MoE - use [0] to get tensor from tuple
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        
        # Return 5-tuple like Qwen3-MoE
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronMiniCPMMoEModel(NeuronBaseModel):
    """Following Qwen3-MoE pattern - lm_head inside Model"""
    
    def setup_attr_for_model(self, config: MiniCPMMoEInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.scale_emb = config.scale_emb

    def init_model(self, config: MiniCPMMoEInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.dim_model_base = config.dim_model_base

        base_embedding = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.embed_tokens = ScaledParallelEmbedding(base_embedding, self.scale_emb)
        self.layers = nn.ModuleList([
            NeuronMiniCPMMoEDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
            pad=True,
        )

    def get_input_embeddings(self, input_ids):
        """Apply MiniCPM embedding scaling."""
        return self.embed_tokens(input_ids)


class NeuronMiniCPMMoEForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronMiniCPMMoEModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return MiniCPMMoEInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[6:] if key.startswith('model.') else key
            new_state_dict[new_key] = value
        return convert_minicpm_moe_hf_to_neuron_state_dict(new_state_dict, config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        # Handle both direct and wrapped embedding
        if "embed_tokens.embedding.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.embedding.weight"].clone()
        elif "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
