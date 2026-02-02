# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch XGLM model for NXD inference."""

"""
XGLM Architecture:
- Sinusoidal positional embeddings (NOT RoPE)
- Standard Multi-Head Attention (16 heads for 564M)
- GELU activation in MLP
- Pre-LayerNorm (norm before attention/MLP)
- Scaled word embeddings (sqrt(d_model))

Reference: transformers/src/transformers/models/xglm/modeling_xglm.py
"""
import math
import os
import json
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_layernorm_cls():
    """Return LayerNorm class - CustomRMSNorm for Neuron, standard for CPU."""
    if cpu_mode():
        return nn.LayerNorm
    return CustomRMSNorm


class XGLMSinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings for XGLM.
    Produces embeddings of any length with offset=2 for padding.
    """
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)
        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """Build sinusoidal embeddings matching tensor2tensor implementation."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = position_ids.size()
        position_ids = position_ids + self.offset
        max_pos = 2 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos, self.embedding_dim, self.padding_idx)
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()


class XGLMNeuronConfig(NeuronConfig):
    """NeuronConfig for XGLM with custom attention class."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronXGLMAttention


class XGLMInferenceConfig(InferenceConfig):
    """Inference configuration for XGLM model."""

    def add_derived_config(self):
        self.num_cores_per_group = 1
        # XGLM uses standard MHA, not GQA
        if not hasattr(self, 'num_key_value_heads') or self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        # Required attributes for inference
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "pad_token_id",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[XGLMNeuronConfig]:
        return XGLMNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load configuration from pretrained model directory."""
        neuron_config = kwargs.pop("neuron_config", None)
        config_path = os.path.join(model_path, "config.json")
        
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Map XGLM config names to standard names
        config_dict = {
            "hidden_size": hf_config.get("d_model", 1024),
            "num_attention_heads": hf_config.get("attention_heads", 16),
            "num_hidden_layers": hf_config.get("num_layers", 24),
            "num_key_value_heads": hf_config.get("attention_heads", 16),  # MHA
            "vocab_size": hf_config.get("vocab_size", 256008),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 2048),
            "intermediate_size": hf_config.get("ffn_dim", 4096),
            "pad_token_id": hf_config.get("pad_token_id", 1),
            "bos_token_id": hf_config.get("bos_token_id", 0),
            "eos_token_id": hf_config.get("eos_token_id", 2),
            "activation_function": hf_config.get("activation_function", "gelu"),
            "scale_embedding": hf_config.get("scale_embedding", True),
            "layer_norm_eps": 1e-5,
            # Required for inference
            "output_attentions": False,
            "output_hidden_states": False,
            "use_return_dict": True,
        }
        config_dict.update(kwargs)
        
        if neuron_config is None:
            neuron_config = cls.get_neuron_config_cls()()
        
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronXGLMAttention(NeuronAttentionBase):
    """
    XGLM Attention using NeuronAttentionBase.
    XGLM uses standard MHA without RoPE (sinusoidal embeddings added at embedding layer).
    """
    def __init__(self, config: XGLMInferenceConfig):
        head_dim = config.hidden_size // config.num_attention_heads
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=None,  # XGLM uses sinusoidal, not RoPE
            qkv_bias=True,
            o_bias=True,
        )


class NeuronXGLMMLP(nn.Module):
    """
    XGLM MLP with GELU activation.
    Architecture: fc1 -> GELU -> fc2
    """
    def __init__(self, config: XGLMInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.activation_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states, None


class NeuronXGLMDecoderLayer(nn.Module):
    """
    XGLM Decoder Layer with Pre-LayerNorm architecture.
    Order: LN -> Attention -> Residual -> LN -> MLP -> Residual
    """
    def __init__(self, config: XGLMInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronXGLMAttention(config)
        self.mlp = NeuronXGLMMLP(config)
        
        # Pre-LayerNorm: norm before attention and MLP
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Self Attention - unpack tuple return
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronXGLMModel(NeuronBaseModel):
    """
    XGLM Model for NeuronX with sinusoidal positional embeddings.
    """

    def setup_attr_for_model(self, config: XGLMInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        # XGLM specific
        self.scale_embedding = getattr(config, 'scale_embedding', True)
        self.embed_scale = math.sqrt(config.hidden_size) if self.scale_embedding else 1.0

    def init_model(self, config: XGLMInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        
        # Sinusoidal positional embeddings
        self.embed_positions = XGLMSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            config.pad_token_id,
        )
        
        self.layers = nn.ModuleList(
            [NeuronXGLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
        )

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ):
        """
        Override get_model_output to add sinusoidal positional embeddings.
        XGLM adds position embeddings to token embeddings before passing through layers.
        """
        batch_size, seq_length = input_ids.shape[:2]
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # Scale embeddings by sqrt(d_model) as per XGLM
            inputs_embeds = inputs_embeds * self.embed_scale
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        # Add sinusoidal positional embeddings
        position_embeddings = self.embed_positions(position_ids, past_key_values_length)
        hidden_states = inputs_embeds + position_embeddings.to(inputs_embeds.device)
        
        # Process through decoder layers
        next_decoder_cache = ()
        cos_cache = None
        sin_cache = None
        
        cache_size = self.n_positions
        if not is_for_context_encoding:
            if past_key_values is None:
                past_key_values = self.kv_mgr.get_cache(
                    seq_ids=seq_ids,
                    seq_len=cache_size,
                    is_for_context_encoding=is_for_context_encoding,
                    **kwargs,
                )
        
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                kv_mgr=self.kv_mgr,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                **kwargs,
            )
            
            hidden_states = layer_outputs[0]
            kv = layer_outputs[1]
            next_decoder_cache += (kv,)
            cos_cache, sin_cache = layer_outputs[2:4]
        
        if update_cache:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                **kwargs,
            )
        
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states, next_decoder_cache


class NeuronXGLMForCausalLM(NeuronBaseForCausalLM):
    """XGLM Causal LM for NeuronX inference."""
    
    _model_cls = NeuronXGLMModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HuggingFace model state dict."""
        from transformers import XGLMForCausalLM
        return XGLMForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace XGLM state dict to NeuronX format.
        
        Key transformations:
        1. Remove 'model.' prefix from keys
        2. Add rank_util.rank tensors for tensor parallelism
        3. Map layer names to match NeuronX expectations
        """
        neuron_config = config.neuron_config
        neuron_state_dict = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # Remove 'model.' prefix
            if new_key.startswith("model."):
                new_key = new_key[6:]
            
            # Map XGLM layer names to NeuronX names
            # embed_tokens -> embed_tokens (same)
            # embed_positions -> embed_positions (same)
            # layers.X.self_attn.q_proj -> layers.X.self_attn.q_proj (same)
            # layers.X.self_attn.k_proj -> layers.X.self_attn.k_proj (same)
            # layers.X.self_attn.v_proj -> layers.X.self_attn.v_proj (same)
            # layers.X.self_attn.out_proj -> layers.X.self_attn.o_proj
            # layers.X.self_attn_layer_norm -> layers.X.self_attn_layer_norm (same)
            # layers.X.fc1 -> layers.X.mlp.fc1
            # layers.X.fc2 -> layers.X.mlp.fc2
            # layers.X.final_layer_norm -> layers.X.final_layer_norm (same)
            # layer_norm -> layer_norm (same)
            # lm_head -> lm_head (same)
            
            # Map out_proj to o_proj
            new_key = new_key.replace(".out_proj.", ".o_proj.")
            
            # Map fc1/fc2 to mlp.fc1/mlp.fc2
            if ".fc1." in new_key:
                new_key = new_key.replace(".fc1.", ".mlp.fc1.")
            if ".fc2." in new_key:
                new_key = new_key.replace(".fc2.", ".mlp.fc2.")
            
            neuron_state_dict[new_key] = value
        
        # Add rank utilities for tensor parallelism
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )
        
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights between embed_tokens and lm_head."""
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return XGLMInferenceConfig
