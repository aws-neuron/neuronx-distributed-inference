# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch CLIPVision model for LLaVA NXD inference."""

from typing import Optional, Tuple, Union

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    OutputChannelParallelConv2d,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn
from transformers import CLIPPreTrainedModel, CLIPVisionConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase


class NeuronCLIPVisionModel(CLIPPreTrainedModel):
    """
    The vision_model is replaced by NeuronCLIPVisionTransformer
    """

    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = NeuronCLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
        )


class NeuronCLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = NeuronCLIPVisionEmbeddings(config)
        self.pre_layrnorm = LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = NeuronCLIPEncoder(config)
        self.post_layernorm = LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class NeuronCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        if parallel_state.model_parallel_is_initialized():
            self.patch_embedding = OutputChannelParallelConv2d(
                in_channels=config.num_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=False,
                dtype=config.torch_dtype,
                partition_pad=True,
            )

            self.position_embedding = ParallelEmbedding(
                self.num_positions,
                self.embed_dim,
                dtype=config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
            )
        else:
            self.patch_embedding = nn.Conv2d(
                in_channels=config.num_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=False,
            )

            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class NeuronCLIPEncoder(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [NeuronCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self, inputs_embeds, output_hidden_states: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        bath_size, seq_len, _ = hidden_states.shape
        attention_mask = torch.ones(
            (bath_size, 1, seq_len, seq_len), dtype=torch.bool, device=hidden_states.device
        )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(hidden_states, attention_mask)

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)


class NeuronCLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = NeuronCLIPAttention(config)
        self.layer_norm1 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = NeuronCLIPMLP(config)
        self.layer_norm2 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.tensor,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class NeuronCLIPAttention(NeuronAttentionBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.tp_degree = config.tp_degree
        self.torch_dtype = config.torch_dtype
        self.fused_qkv = False
        self.clip_qkv = None
        self.bias = True

        self.o_proj_layer_name = "out_proj"

        self.init_gqa_properties()


class NeuronCLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]

        if parallel_state.model_parallel_is_initialized():
            self.fc1 = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                bias=True,
                gather_output=False,
                dtype=config.torch_dtype,
                pad=True,
            )

            self.fc2 = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                bias=True,
                input_is_parallel=True,
                dtype=config.torch_dtype,
                pad=True,
            )
        else:
            self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
            self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
