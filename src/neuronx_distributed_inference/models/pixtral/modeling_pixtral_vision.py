# coding=utf-8
# Copyright 2024 Mistral and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Pixtral model for NXD inference."""

import logging
import os
import copy
from typing import List, Optional, Tuple

import torch
from safetensors.torch import save_file
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.pixtral.modeling_pixtral import position_ids_in_meshgrid

import neuronx_distributed_inference.modules.autobucketing as autobucketing
from neuronx_distributed_inference.modules.padding import pad_tensor, unpad_tensor
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP, get_rmsnorm_cls
from neuronx_distributed_inference.models.llama4.modeling_llama4_vision import get_hw
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PixtralRotaryEmbedding(nn.Module):
    """
    The key with pixtral embedding is just that you have a frequency for each pixel positions.
    If you have height x width pixels (or embedding pixels), then the frequency used for ROPE
    is given by indexing the pre_computed frequency on the width and height.

    What you output is of dimension (batch, height * width, dim) with dim the embed dim.

    This simply means that for each image hidden state, you are going to add
    a corresponding positional embedding, based on its index in the grid.
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.rope_type = "default"
        self.dim = config.head_dim
        self.base = config.rope_theta
        max_patches_per_side = config.image_size // config.patch_size
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        h = torch.arange(max_patches_per_side, device=freqs.device)
        w = torch.arange(max_patches_per_side, device=freqs.device)

        freqs_h = torch.outer(h, freqs[::2]).float()
        freqs_w = torch.outer(w, freqs[1::2]).float()
        inv_freq = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
                freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
            ],
            dim=-1,
        ).reshape(-1, self.dim // 2)  # we reshape to only index on the position indexes, not tuple of indexes
        # Different from paper, but it uses a different permutation in order to obtain the same calculation

        # TODO maybe make it torch compatible later on. We can also just slice
        self.register_buffer("inv_freq", torch.cat((inv_freq, inv_freq), dim=-1), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        freqs = self.inv_freq[position_ids]

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            emb = freqs
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype).unsqueeze(0), sin.to(dtype=x.dtype).unsqueeze(0)


class NeuronPixtralImageAttention(NeuronAttentionBase):
    def __init__(self, config):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_cores_per_group=config.num_cores_per_group,
            sequence_parallel_enabled=False,
            rotary_emb=PixtralRotaryEmbedding(config)
        )


class NeuronPixtralAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.feed_forward = NeuronLlamaMLP(config)
        self.attention = NeuronPixtralImageAttention(config)
        self.ffn_norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, seq_len, seq_len)`.
                vision patches only attend to patches of the same image.
                So attention_mask value is 1 where patches are in the same image.
                And 0 where patches are not in the same image.
                Padded position should have 0 value.
        """
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states, _ = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NeuronPixtralTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(NeuronPixtralAttentionLayer(config))

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
            )

        return hidden_states


class NeuronLlavaMultiModalProjector(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = ColumnParallelLinear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = RowParallelLinear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
            reduce_dtype=config.neuron_config.rpl_reduce_dtype,
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class NeuronPixtralVisionModel(torch.nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config
        logger.info(f"in NeuronPixtralVisionModel self.vision_config {vars(self.vision_config)}")

        self.vision_patch_conv_linear = ColumnParallelLinear(
            self.vision_config.num_channels * self.vision_config.patch_size * self.vision_config.patch_size,
            self.vision_config.hidden_size,
            bias=False,
            gather_output=True,
            dtype=self.vision_config.neuron_config.torch_dtype,
        )

        # Vision model's rms_norm_eps is hardcoded in HF code and not included in HF config
        self.vision_config.rms_norm_eps = getattr(self.vision_config, "rms_norm_eps", 1e-5)

        self.vision_ln_pre = get_rmsnorm_cls()(
            self.vision_config.hidden_size,
            eps=config.vision_config.rms_norm_eps,
        )
        # For the vision model, pass in just the config.vision_config:InferenceConfig to avoid mixing up neuron_config from config.text_config
        self.vision_transformer = NeuronPixtralTransformer(self.vision_config)
        self.vision_patch_positional_embedding = PixtralRotaryEmbedding(self.vision_config)
        # multi_modal_projector need to read text model hidden_size, so we pass in the entire config to it
        self.multi_modal_projector = NeuronLlavaMultiModalProjector(self.config)

    def forward(
            self,
            patch_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
    ):
        patch_embeds = self.vision_patch_conv_linear(patch_embeds)
        patch_embeds = self.vision_ln_pre(patch_embeds)

        hidden_states = self.vision_transformer(
            patch_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logger.info(f"in NeuronPixtralVisionModel hidden_states {hidden_states.shape}")

        projected_vision_emb = self.multi_modal_projector(hidden_states)
        logger.info(f"in NeuronPixtralVisionModel projected_vision_emb {projected_vision_emb.shape}")

        return projected_vision_emb


class PixtralVisionModelWrapper(ModelWrapper):
    """
    Neuron ModelWrapper class for NeuronPixtralVisionModel.
    Generates input shapes for trace and compilation. Disables bucketing.
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = True,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx,
            pipeline_execution, return_ranked_to_cpu, model_init_kwargs
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """
        Override ModelWrapper.input_generator().
        Generate a list of valid sample inputs containing one input list for each bucket.
        Different model may have a different set of input args.

        Returns:
            inputs (List[Tuple[torch.Tensor]]): Example input args for every bucket.
        """
        inputs = []
        batch_size = self.config.vision_config.neuron_config.batch_size   # should be 1
        for bucket in self.config.vision_config.neuron_config.buckets:
            patch_embeds = torch.ones(
                [
                    batch_size,
                    bucket,
                    self.config.vision_config.num_channels * self.config.vision_config.patch_size * self.config.vision_config.patch_size,
                ],
                dtype=self.config.vision_config.neuron_config.torch_dtype
            )
            attention_mask = torch.ones(  # this is block attention mask, different from traditional text causal mask
                [
                    batch_size,
                    batch_size,
                    bucket,
                    bucket,
                ],
                dtype=torch.int32
            )
            position_ids = torch.ones(  # generated in meshgrid of each patchfied image
                [
                    bucket,
                ],
                dtype=torch.int32
            )
            inputs.append((patch_embeds, attention_mask, position_ids))

        return inputs

    def vision_patch_conv_unfold(self, x):
        k0, k1 = get_hw(self.config.vision_config.patch_size)

        bs, nc, r, c = x.shape  # torch.Size([1, 3, 512, 512])
        pr, pc = r // k0, c // k1  # 32, 32
        dr, dc = r % k0, c % k1
        # conv2d ignore pixels that does not fit in the strided kernels
        # in Pixtral case, kernel_size == stride, so we just clip off the remainer
        x = x[:, :, : (r - dr), : (c - dc)]
        x = x.reshape(bs, nc, pr, k0, pc, k1)  # after-op shape: [1, 3, 32, 16, 32, 16]
        x = x.permute(0, 1, 3, 5, 2, 4)  # after-op shape: [0, 3, 16, 16, 32, 32]
        x = x.reshape(bs, nc * k0 * k1, pr, pc)  # after-op shape: [1, 768, 32, 32]
        return x

    def patchify(self, pixel_values, image_sizes):
        # pixel_values: torch.Size([1, 3, 512, 512])
        # image_sizes: tensor([[512, 512]], dtype=torch.int32), torch.Size([1, 2])
        # pass images through initial convolution independently
        if len(pixel_values.shape) == 5:
            assert pixel_values.shape[0] == 1, "Vision encoder only supports BS=1"
            pixel_values = pixel_values.squeeze(0)

        # Replace conv2d with unfold + linear and move linear to neuron device
        patch_embeds = self.vision_patch_conv_unfold(pixel_values)  # after-op shape: [1, 768, 32, 32]

        # leave only patches with actual pixel_values, remove padding
        patch_embeds_list = [
            embed[..., : (size[0] // self.config.vision_config.patch_size), : (size[1] // self.config.vision_config.patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]  # [torch.Size([768, 32, 32]),]

        # flatten into a single sequence
        patch_embeds = torch.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)  # torch.Size([1, 1024, 768])

        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list, max_width=self.config.vision_config.image_size // self.config.vision_config.patch_size
        )  # torch.Size([1024])

        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )  # torch.Size([1, 1, 1024, 1024])

        return (patch_embeds.to(self.config.vision_config.neuron_config.torch_dtype),
                attention_mask.to(torch.int32),
                position_ids.to(torch.int32))

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def pad_inputs(self, patch_embeds, attention_mask, position_ids):
        target_len = self.get_target_bucket(patch_embeds)

        # pad patch_embeds with real value
        target_patch_embeds_size = [patch_embeds.shape[0], target_len, patch_embeds.shape[2]]
        padded_patch_embeds, self.original_patch_embed_slices = pad_tensor(patch_embeds, target_patch_embeds_size)

        # patch_embeds_padding_mask is of shape [bs, target_len, vision_hidden_size]
        # but the vision model returns vision_emb after multi_modal_projector
        # of shape [bs, target_len, text_hidden_size]
        self.original_patch_embed_slices[-1][-1] = self.config.text_config.hidden_size

        # pad attention_mask with -inf
        target_attention_mask_size = [attention_mask.shape[0],
                                      attention_mask.shape[1],
                                      target_len,
                                      target_len]
        # NeuronAttentionBase expects a bool attention mask, so padding positions should have value 0
        padded_attention_mask, _ = pad_tensor(attention_mask, target_attention_mask_size, pad_value=0)

        # pad position_ids with 0
        target_position_ids_size = [target_len]
        padded_position_ids, _ = pad_tensor(position_ids, target_position_ids_size, pad_value=0)

        return padded_patch_embeds, padded_attention_mask, padded_position_ids

    def get_target_bucket(
        self,
        patch_embeds,
    ) -> int:
        """
        Override ModelWrapper.get_target_bucket().
        Get the closest bucket size.

        Returns:
            int: target bucket size
        """

        # Largest bucket size must be more than any possible vision patch len
        # max_image_size is 1024, (1024 // 16) ** 2 = 4096 is  max_num_patches_per_image
        # max_num_image = 6. Therefore max_num_patches = 6 * 4096 = 24576

        patch_seq_len = patch_embeds.shape[1]
        # InferenceConfig would use seq_len if buckets are not specified. Validation is done there.
        largest_bucket = self.config.vision_config.neuron_config.buckets[-1]

        # validate the input patch_seq_len does not exceed largest bucket
        assert patch_seq_len <= largest_bucket, \
            f"Total number of image patches {patch_seq_len} exceeds largest bucket ({largest_bucket})"

        # return closest bucket
        for i, bucket in enumerate(self.config.vision_config.neuron_config.buckets):
            if patch_seq_len <= bucket:
                logger.info(f"Routing patch_seq_len {patch_seq_len} to bucket size {bucket}")
                return bucket

    def forward(self, pixel_values, image_sizes):
        """
        Override ModelWrapper.forward().
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        patch_embeds, attention_mask, position_ids = self.patchify(pixel_values, image_sizes)

        padded_patch_embeds, padded_attention_mask, padded_position_ids = self.pad_inputs(patch_embeds, attention_mask, position_ids)

        # convert int64 to int32 to improve compatibility with compiler; does not apply to cpu case
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(padded_patch_embeds, padded_attention_mask, padded_position_ids)
        vision_emb = self._forward(*args)

        vision_emb = unpad_tensor(vision_emb, self.original_patch_embed_slices)

        return vision_emb


class NeuronPixtralForImageEncoding(NeuronApplicationBase):
    """
    Neuron Application class for Pixtral image encoding case.
    Wraps NeuronPixtralVisionModel with Neuron specific functionalities such as compile and load.
    """

    _model_cls = NeuronPixtralVisionModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.vision_config.neuron_config.enable_bucketing:
            # neuron_config.buckets default to neuron_config.seq_len is not given. For vision we want to do auto-bucketing here
            if self.config.vision_config.neuron_config.buckets == [self.config.vision_config.neuron_config.seq_len] or \
                    self.config.vision_config.neuron_config.buckets is None:
                # 1024 vision seq len corresponds to a single 512x512 image. Smaller bucket size does not make sense in real life.
                if self.config.vision_config.neuron_config.seq_len > 1024:
                    self.config.vision_config.neuron_config.buckets = autobucketing.generate_buckets(
                        1024, self.config.vision_config.neuron_config.seq_len
                    )
                else:
                    self.config.vision_config.neuron_config.buckets = [self.config.vision_config.neuron_config.seq_len]

        self.neuron_config = copy.deepcopy(self.config.vision_config.neuron_config)

        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
            pipeline_execution=True,
            return_ranked_to_cpu=True
        )
        # will only have one model one tag
        # after compilation, in /tmp/nxd_model,
        # you should only see one folder called f"self._model_cls.__name__"
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return PixtralVisionModelWrapper

    def forward(self, pixel_values, image_sizes):
        return self.models[0](pixel_values, image_sizes)

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor=2 ' -O1 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"
        logger.info(f"{self._model_cls.__name__} compiler_args: {compiler_args}")
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        new_state_dict = {}
        for key, value in state_dict.items():
            vision_model_keys_mapping = {
                "vision_tower.": "vision_",
            }
            for pattern, replacement in vision_model_keys_mapping.items():
                if pattern in key:
                    key = key.replace(pattern, replacement)

            new_state_dict[key] = (
                value.clone()
                .detach()
                .contiguous()
                .to(inference_config.vision_config.neuron_config.torch_dtype)
            )

        new_state_dict["vision_patch_conv_linear.weight"] = state_dict.pop("vision_tower.patch_conv.weight").reshape(
            -1, inference_config.vision_config.num_channels * inference_config.vision_config.patch_size * inference_config.vision_config.patch_size
        )

        del state_dict
        return new_state_dict

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        # NeuronPixtralVisionModel combines the vision model and the multimodal projector
        # So we wrap the HF modules into one class

        from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel, PixtralVisionConfig
        from transformers.models.llava.modeling_llava import LlavaMultiModalProjector, LlavaConfig

        class hf_vision_model(torch.nn.Module):
            def __init__(self, model_path, **kwargs):
                super().__init__()

                # config
                self.hf_config = LlavaConfig.from_pretrained(model_path, **kwargs)
                hf_vision_config = PixtralVisionConfig(**vars(self.hf_config.vision_config))

                self.vision_tower = PixtralVisionModel(hf_vision_config)
                self.multi_modal_projector = LlavaMultiModalProjector(self.hf_config)

            def forward(self, pixel_values, image_sizes):
                image_outputs = self.vision_tower(pixel_values, image_sizes)
                hidden_state = image_outputs.last_hidden_state
                print(f"in original_vision_model hidden_state {hidden_state.shape}")

                projected_vision_emb = self.multi_modal_projector(hidden_state)
                print(f"in original_vision_model projected_vision_emb {projected_vision_emb.shape}")

                return projected_vision_emb

            def save_pretrained(self, save_model_path):
                self.hf_config.save_pretrained(save_model_path)
                save_file(self.state_dict(), os.path.join(save_model_path, "model.safetensors"))

        hf_model = hf_vision_model(model_path, **kwargs)

        return hf_model


def generate_block_attention_mask(patch_embeds_list, tensor):
    '''
    Copied from transformers.models.pixtral.modeling_pixtral.generate_block_attention_mask
    Difference is that this version outputs the boolean mask.
    Meaning that the positions to compute attention are represented by 0,
    the positions to skip attention computation are represented by 1.
    '''
    dtype = torch.int32
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = 0
    causal_mask = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device)

    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 1

    causal_mask = causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
    return causal_mask
