# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViT model for NxD Inference."""

import collections.abc
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.utils import torch_int

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.encoder_base import NeuronEncoderBase
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ViTInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        # some config moved from HF model init or forward arguments
        self.use_mask_token = kwargs.pop("use_mask_token", False)
        self.add_pooling_layer = kwargs.pop("add_pooling_layer", False)
        self.interpolate_pos_encoding = kwargs.pop("interpolate_pos_encoding", False)

        super().__init__(*args, **kwargs)

    def get_required_attributes(self) -> List[str]:
        # To validate if the config.json include all the configs we need in model.
        # Need to manually add what's required in below list
        return [
            "_name_or_path",
            "architectures",
            "attention_probs_dropout_prob",
            "encoder_stride",
            "hidden_act",
            "hidden_dropout_prob",
            "hidden_size",
            "image_size",
            "initializer_range",
            "intermediate_size",
            "layer_norm_eps",
            "model_type",
            "num_attention_heads",
            "num_channels",
            "num_hidden_layers",
            "patch_size",
            "qkv_bias",
            "transformers_version",
            "use_mask_token",
            "add_pooling_layer",
            "interpolate_pos_encoding",
        ]


class NeuronViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTInferenceConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        logger.info(f"use_mask_token {use_mask_token}")
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        )
        logger.info(f"self.mask_token {self.mask_token}")
        self.patch_embeddings = NeuronViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[torch.BoolTensor] = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class NeuronViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTInferenceConfig):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        # self.projection = OutputChannelParallelConv2d( # FIXME: in checkpoint bias is not sharded: Incorrect tensor shape at checkpoint keyprojection.bias: received 768, expected 24.
        #     in_channels=num_channels,
        #     out_channels=hidden_size,
        #     kernel_size=patch_size,
        #     stride=patch_size
        #     )

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: Optional[
            torch.BoolTensor
        ] = False,  # swap bool to torch.BoolTensor for Neuron
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class NeuronViTAttention(NeuronAttentionBase):
    def __init__(self, config: ViTInferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.tp_degree = config.neuron_config.tp_degree
        self.torch_dtype = config.neuron_config.torch_dtype
        self.fused_qkv = False
        self.clip_qkv = None
        self.bias = True

        self.o_proj_layer_name = "o_proj"

        self.init_gqa_properties()


class NeuronViTIntermediate(nn.Module):
    def __init__(self, config: ViTInferenceConfig) -> None:
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            raise ValueError(
                f"{config.hidden_act} is not supported. Choose from {list(ACT2FN.keys())}"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class NeuronViTOutput(nn.Module):
    def __init__(self, config: ViTInferenceConfig) -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class NeuronViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTInferenceConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # NeuronAttentionBase includes qkv project layers (CPL) and output project (RPL) layers
        # but HF separates into ViTSelfAttention which only has qkv, then an another ViTSelfOutput that has the output project layer
        self.attention = NeuronViTAttention(config)
        self.intermediate = NeuronViTIntermediate(config)
        self.output = NeuronViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in ViT, layernorm is applied before self-attention
        )

        # NeuronAttentionBases output tuple (attn_output, past_key_value, cos_cache, sin_cache)
        attention_output = self_attention_outputs[0]

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


class NeuronViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [NeuronViTLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Union[tuple, torch.Tensor]:
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs

        return hidden_states


class NeuronViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            gather_output=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class NeuronViTModel(NeuronEncoderBase):
    """
    Neuron version of HF ViTModel

    Difference from original: Move/remove input arguments that result in dynamic graph at runtime.

    - Move init argument `use_mask_token`, `add_pooling_layer`, and `interpolate_pos_encoding` to
     `config:ViTInferenceConfig`. All Default to `False`.

    - No `_prune_heads()` method. This is a feature in HF PretrainedModel that prune attention heads
     by `self.encoder.layer[layer].attention.prune_heads(heads)`. However, it is not used in all HF
     ViT models and not supported by `NeuronAttentionBase`.

    - The forward pass does not take `head_mask` input. This is the mask to nullify selected heads of
     the HF self-attention modules `ViTSelfAttention`. It is default to `None` in all HF ViT model.
     And `NeuronAttentionBase` does not support this.

    - The forward pass does not take `output_attentions` input. If set `True`, HF ViTModel outputs
     all raw attention weights after softmax. It is default to `None` in all HF ViT model and not
     supported by `NeuronAttentionBase`. And will increase the output tensor size and increase latency
     due to data transfer between devices.

    - The forward pass does not take `output_hidden_states` input. If set `True`, HF ViTModel outputs
     all hidden states of every ViT layer. It is default to `None` in all HF ViT model. And will increase
     the output tensor size and increase latency due to data transfer between devices.
    """

    def __init__(self, config: ViTInferenceConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = NeuronViTEmbeddings(config, use_mask_token=self.config.use_mask_token)
        self.encoder = NeuronViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = NeuronViTPooler(config) if self.config.add_pooling_layer else None

    def get_input_embeddings(self) -> NeuronViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[
            torch.BoolTensor
        ] = None,  # only used in ViTForMaskedImageModeling
    ) -> Tuple:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)
            raise TypeError(
                f"pixel_values is of dtype {pixel_values.dtype}, but weights are of dtype \
                            {self.embeddings.patch_embeddings.projection.weight.dtype}"
            )

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=self.config.interpolate_pos_encoding,
        )

        sequence_output = self.encoder(embedding_output)
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        head_outputs = (
            (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        )
        return head_outputs


class ModelWrapperViT(ModelWrapper):
    """
    Neuron ModelWrapper class for NeuronViTModel.
    Generates input shapes for trace and compilation. Disables bucketing.
    """

    def __init__(
        self,
        config: ViTInferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs
        )
        self.bucket_config = None  # Set to None because we don't have bucketing

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """
        Override ModelWrapper.input_generator().
        Generate a list of valid sample inputs containing one input list for each bucket.
        Different model may have a different set of input args.

        Returns:
            inputs (List[Tuple[torch.Tensor]]): Example input args for every bucket.
        """
        image = torch.ones(
            [
                self.neuron_config.batch_size,
                self.config.num_channels,
                self.config.image_size,
                self.config.image_size,
            ]
        )
        inputs = [(image,)]
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def forward(self, *args):
        """
        Override ModelWrapper.forward().
        """

        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        # convert int64 to int32 to improve compatibility with compiler; does not apply to cpu case
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)

        output = self._forward(*args)

        return output


class NeuronViTForImageEncoding(NeuronApplicationBase):
    """
    Neuron Application class for ViT image encoding case.
    Wraps NeuronViTModel with Neuron specific functionalities such as compile and load.
    """

    _model_cls = NeuronViTModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
        )
        # will only have one model one tag
        # after compilation, in /tmp/nxd_model,
        # you should only see one folder called f"self._model_cls.__name__"
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return ModelWrapperViT

    def forward(self, pixel_values):
        return self.models[0](pixel_values)

    def get_compiler_args(self):
        # Flag for model type
        compiler_args = "-O1 --model-type=transformer"
        # Add flags for cc-overlap
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        # Prevent auto-down casting when running with fp32
        if self.config.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        logger.info(f"{self._model_cls.__name__} compiler_args: {compiler_args}")
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        for layer in range(config.num_hidden_layers):
            state_dict[f"encoder.layer.{layer}.attention.qkv_proj.q_proj.weight"] = state_dict.pop(
                f"encoder.layer.{layer}.attention.attention.query.weight"
            )
            state_dict[f"encoder.layer.{layer}.attention.qkv_proj.q_proj.bias"] = state_dict.pop(
                f"encoder.layer.{layer}.attention.attention.query.bias"
            )
            state_dict[f"encoder.layer.{layer}.attention.qkv_proj.k_proj.weight"] = state_dict.pop(
                f"encoder.layer.{layer}.attention.attention.key.weight"
            )
            state_dict[f"encoder.layer.{layer}.attention.qkv_proj.k_proj.bias"] = state_dict.pop(
                f"encoder.layer.{layer}.attention.attention.key.bias"
            )
            state_dict[f"encoder.layer.{layer}.attention.qkv_proj.v_proj.weight"] = state_dict.pop(
                f"encoder.layer.{layer}.attention.attention.value.weight"
            )
            state_dict[f"encoder.layer.{layer}.attention.qkv_proj.v_proj.bias"] = state_dict.pop(
                f"encoder.layer.{layer}.attention.attention.value.bias"
            )

            state_dict[f"encoder.layer.{layer}.attention.o_proj.weight"] = state_dict.pop(
                f"encoder.layer.{layer}.attention.output.dense.weight"
            )
            state_dict[f"encoder.layer.{layer}.attention.o_proj.bias"] = state_dict.pop(
                f"encoder.layer.{layer}.attention.output.dense.bias"
            )

        return state_dict
