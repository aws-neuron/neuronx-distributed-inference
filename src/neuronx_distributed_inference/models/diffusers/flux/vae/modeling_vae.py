# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# This implementation is derived from the Diffusers and VAE library.
# The original codebase has been optimized and modified to achieve optimal performance
# characteristics when executed on Amazon Neuron devices.
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

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F, init

from diffusers.models.autoencoders.vae import Decoder
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance, ModelWrapper
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from typing import List, Tuple


# Replace torch.nn.GroupNorm with PatchedGroupNorm when running the decoder in bfloat16.
class PatchedGroupNorm(torch.nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": torch.float32}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        out_dtype = input.dtype
        return F.group_norm(input.to(torch.float32), self.num_groups, self.weight, self.bias, self.eps).to(out_dtype)

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )


class VAEDecoderInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_config = {
            "in_channels": self.latent_channels,
            "out_channels": self.out_channels,
            "up_block_types": self.up_block_types,
            "block_out_channels": self.block_out_channels,
            "layers_per_block": self.layers_per_block,
            "norm_num_groups": self.norm_num_groups,
            "act_fn": self.act_fn,
            "mid_block_add_attention": self.mid_block_add_attention
        }

    def get_required_attributes(self) -> List[str]:
        return [
            "latent_channels",
            "out_channels",
            "up_block_types",
            "block_out_channels",
            "layers_per_block",
            "norm_num_groups",
            "act_fn",
            "mid_block_add_attention",
            "height",
            "width",
            "transformer_in_channels",
        ]

    @property
    def vae_scale_factor(self):
        return 2 ** (len(self.block_out_channels) - 1)


class ModelWrapperVAEDecoder(ModelWrapper):

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs
        )
        self.bucket_config = None  # Set to None if you don't have bucketing

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        model_inputs = torch.rand(
            [
                1,
                self.config.transformer_in_channels // 4,
                self.config.height // self.config.vae_scale_factor,
                self.config.width // self.config.vae_scale_factor,
            ],
            dtype=self.config.neuron_config.torch_dtype,
        )
        inputs = [(model_inputs,)]
        return inputs

    def get_model_instance(self):
        # Create the model instance

        def _create_model():
            # Need to replace torch.nn.GroupNorm with PatchedGroupNorm when using bfloat16
            # It is because of sensitivity of norm operation to extremely small values.
            # We need to convert input to float32 before computation.
            if self.config.neuron_config.torch_dtype == torch.bfloat16:
                torch.nn.GroupNorm = PatchedGroupNorm
            model = self.model_cls(**self.config.decoder_config)
            model = model.to(self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        model_instance = BaseModelInstance(module_cls=_create_model, input_output_aliases={})

        return model_instance

    def forward(self, *args):
        """
        Override ModelWrapper.forward().
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )
        output = self._forward(*args)

        return output


class NeuronVAEDecoderApplication(NeuronApplicationBase):

    _model_cls = Decoder

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
        )

        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    def get_model_wrapper_cls(self):
        return ModelWrapperVAEDecoder

    def forward(self, model_inputs):
        return self.models[0](model_inputs)

    def get_compiler_args(self):
        compiler_args = "--model-type=unet-inference -O1"
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        new_load = {
            key.replace("decoder.", ""): state_dict[key]
            .to(config.neuron_config.torch_dtype)
            .clone()
            .detach()
            .contiguous()
            for key in list(state_dict.keys())
        }

        state_dict.update(new_load)
        return state_dict
