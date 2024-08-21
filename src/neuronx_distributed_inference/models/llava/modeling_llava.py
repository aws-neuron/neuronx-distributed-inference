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
""" PyTorch LLaVA model for NXD inference."""
from typing import Optional, Tuple, Union

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from torch import nn
from transformers import LlavaConfig, LlavaPreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
)

from neuronx_distributed_inference.models.config import NeuronConfig, PretrainedConfigAdapter
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaModel
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel

from .model_wrapper_llava import ModelWrapperLlava
from .modeling_clip import NeuronCLIPVisionModel


class LlavaConfigAdapter(PretrainedConfigAdapter, LlavaConfig):
    def __init__(self, neuron_config: NeuronConfig = None, **kwargs):
        super().__init__(neuron_config, **kwargs)
        self.text_config.attn_cls = "NeuronLlamaAttention"
        self.text_config._attn_implementation = "eager"

        # Move self.text_config.* to  self.*
        # We will directly use self as the configuration for llama model
        for key, value in self.text_config.__dict__.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        self.pad_token_id = self.text_config.pad_token_id
        self.bos_token_id = self.text_config.bos_token_id
        self.eos_token_id = self.text_config.eos_token_id
        self._attn_implementation = self.text_config._attn_implementation

    @classmethod
    def get_config_cls(cls):
        return NeuronConfig


class NeuronLlavaMultiModalProjector(nn.Module):
    """
    The linear layers are replaced with ColumnParallelLinear
    """

    def __init__(self, config: LlavaConfigAdapter):
        super().__init__()

        self.act = ACT2FN[config.projector_hidden_act]

        if parallel_state.model_parallel_is_initialized():
            self.linear_1 = ColumnParallelLinear(
                config.vision_config.hidden_size,
                config.text_config.hidden_size,
                bias=True,
                gather_output=False,
                dtype=config.torch_dtype,
                pad=True,
            )

            self.linear_2 = RowParallelLinear(
                config.text_config.hidden_size,
                config.text_config.hidden_size,
                bias=True,
                input_is_parallel=True,
                dtype=config.torch_dtype,
                pad=True,
            )
        else:
            self.linear_1 = nn.Linear(
                config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
            )
            self.linear_2 = nn.Linear(
                config.text_config.hidden_size, config.text_config.hidden_size, bias=True
            )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class NeuronLlavaModel(NeuronBaseModel, LlavaPreTrainedModel):
    def __init__(self, config: LlavaConfigAdapter):
        super().__init__(config, optimize_inference=False)

    def setup_attr_for_model(self, config: LlavaConfigAdapter):
        self.on_device_sampling = config.neuron_config.on_device_sampling
        self.tp_degree = config.neuron_config.tp_degree
        self.neuron_config = config.neuron_config

    def init_model(self, config: LlavaConfigAdapter):
        config.vision_config.tp_degree = config.neuron_config.tp_degree
        config.vision_config.torch_dtype = config.torch_dtype
        self.vision_tower = NeuronCLIPVisionModel(config.vision_config)
        self.multi_modal_projector = NeuronLlavaMultiModalProjector(config)
        self.language_model = NeuronLlamaModel(config)
        self.past_key_values = self.language_model.kv_mgr.past_key_values

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        text_embedding_indices,
        image_embedding_indices,
        inputs_embeds,
        input_ids,
    ):
        batch_size, sequence_length = input_ids.shape
        num_images, num_image_patches, embed_dim = image_features.shape

        final_embedding = torch.zeros(
            batch_size,
            sequence_length,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        text_embedding_indices = torch.unsqueeze(text_embedding_indices, dim=-1).expand(
            batch_size, -1, embed_dim
        )
        final_embedding.scatter_(1, text_embedding_indices, inputs_embeds)

        image_features = image_features.to(dtype=inputs_embeds.dtype).contiguous()
        # Todo: implement variable number of images in patches
        image_embedding_indices = image_embedding_indices[:, :num_image_patches]
        image_embedding_indices = torch.unsqueeze(image_embedding_indices, dim=-1).expand(
            batch_size, -1, embed_dim
        )
        final_embedding.scatter_(1, image_embedding_indices, image_features)

        return final_embedding

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        seq_ids: torch.IntTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_embedding_indices: Optional[torch.LongTensor] = None,
        image_embedding_indices: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)

                # The attention_mask and position_ids have been prepared in prepare_inputs_for_generation()
                # Here we only do static-shape embedding merge
                inputs_embeds = self._merge_input_ids_with_image_features(
                    image_features,
                    text_embedding_indices,
                    image_embedding_indices,
                    inputs_embeds,
                    input_ids,
                )

        # We set n_positions in language model again because self.n_positions may be changed
        # by DecoderModelInstance::get() in model_wrapper.py and this will be used in
        # NeuronBaseModel::_create_context_attn_mask() to create attention mask
        self.language_model.n_positions = self.n_positions
        outputs = self.language_model(
            input_ids, attention_mask, position_ids, seq_ids, inputs_embeds=inputs_embeds
        )

        return outputs


class NeuronLlavaForConditionalGeneration(NeuronBaseForCausalLM, LlavaPreTrainedModel):
    _STATE_DICT_MODEL_PREFIX = "language_model.model."
    _NEW_STATE_DICT_MODEL_PREFIX = "language_model."
    _model_cls = NeuronLlavaModel

    def __init__(self, model_path: str, config: LlavaConfigAdapter):
        super().__init__(model_path, config)
        self.torch_dtype = config.torch_dtype
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.image_token_index = config.image_token_index

    @staticmethod
    def load_hf_model(model_path):
        return LlavaForConditionalGeneration.from_pretrained(model_path)

    def get_model_wrapper_cls(self):
        return ModelWrapperLlava

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        **kwargs,
    ):
        # If kv_cache is populated, we are entering token generation mode
        # Only the last input_id is needed for next token generation
        if not self.kv_cache_populated:
            # In context generation mode, compute embedding indices
            (
                text_embedding_indices,
                image_embedding_indices,
                attention_mask,
            ) = self._prepare_embedding_indices(
                input_ids, attention_mask, self.neuron_config.max_context_length
            )

        # position_ids does not carry in each token generation loop
        # so we need to calculate again before each run
        position_ids = (attention_mask.long().cumsum(-1) - 1).masked_fill_(attention_mask == 0, 1)
        if self.kv_cache_populated:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids.amax(dim=-1, keepdim=True) + 1
            text_embedding_indices = torch.zeros_like(input_ids)
            image_embedding_indices = torch.zeros_like(input_ids)

        pixel_values = pixel_values.to(dtype=self.torch_dtype)

        seq_ids = torch.arange(input_ids.shape[0], device=input_ids.device)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "seq_ids": seq_ids,
            "llava_args": [pixel_values, text_embedding_indices, image_embedding_indices],
        }

        return model_inputs

    def _prepare_embedding_indices(self, input_ids, attention_mask, max_context_length):
        batch_size = input_ids.shape[0]
        num_image_patches = (self.image_size // self.patch_size) ** 2
        special_image_token_mask = input_ids == self.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)

        new_token_positions = (
            torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        )
        max_embed_dim = (
            num_special_image_tokens.max() * (num_image_patches - 1)
        ) + max_context_length

        text_embedding_indices = torch.full((batch_size, max_embed_dim), -1)
        image_embedding_indices = torch.full((batch_size, max_embed_dim), -1)
        final_attention_mask = torch.zeros(
            (batch_size, max_embed_dim), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # non_image_indices is computed per batch because its shape may be different in each batch
        # To write the new token position to the corresponding text_embedding_indices we need this for-loop
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # Truncate text token positions beyond sequence length
            cur_new_token_positions_mask = new_token_positions[batch_idx] < max_context_length
            cur_new_token_positions = new_token_positions[batch_idx][cur_new_token_positions_mask]
            cur_input_ids = cur_input_ids[cur_new_token_positions_mask]
            cur_attention_mask = attention_mask[batch_idx][cur_new_token_positions_mask]
            cur_valid_embedding_len = cur_new_token_positions[torch.sum(cur_attention_mask) - 1] + 1
            final_attention_mask[batch_idx, :cur_valid_embedding_len] = 1

            image_indices = torch.where(cur_input_ids == self.image_token_index)[0]

            # Text embedding indices
            # We can ignore special image token because it will be overwrite by actual image embeddings
            text_embedding_indices[
                batch_idx, : cur_new_token_positions.shape[0]
            ] = cur_new_token_positions

            # Image embedding indices
            for image_idx, pos_idx in enumerate(image_indices):
                left = image_idx * num_image_patches
                right = left + num_image_patches

                # +1 because the end index is not inclusive [start: end)
                embedding_end_idx = cur_new_token_positions[pos_idx] + 1
                embedding_start_idx = embedding_end_idx - num_image_patches
                image_embedding_indices[batch_idx, left:right] = torch.arange(
                    embedding_start_idx, embedding_end_idx
                )

        return text_embedding_indices, image_embedding_indices, final_attention_mask
