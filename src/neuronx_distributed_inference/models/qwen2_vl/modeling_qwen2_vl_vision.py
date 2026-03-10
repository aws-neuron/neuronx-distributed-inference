import os

import torch
from torch import nn
from torch.nn import LayerNorm
from safetensors.torch import save_file
from typing import List, Optional, Tuple
from transformers.activations import ACT2FN
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding, PatchEmbed

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.padding import pad_tensor, pad_with_first_batchline
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    apply_rotary_pos_emb,
)
from neuronx_distributed_inference.models.qwen2_vl.utils.vision_utils import (
    calculate_max_grid_size, get_image_dimensions
)
from neuronx_distributed_inference.models.qwen2_vl.utils.input_processor import prepare_generation_inputs_hf

import logging

logger = logging.getLogger(__name__)


class Qwen2VLVisionRotaryEmbedding(nn.Module):

    @torch.inference_mode()
    def forward(self, x, position_embeddings):
        cos, sin = position_embeddings
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6, dtype=dtype)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size, self.hidden_size, gather_output=False, dtype=dtype
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size, dim, input_is_parallel=True, dtype=dtype, reduce_dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(dim, hidden_dim, gather_output=False, dtype=dtype)
        self.act = ACT2FN[hidden_act]
        self.fc2 = RowParallelLinear(
            hidden_dim, dim, input_is_parallel=True, dtype=dtype, reduce_dtype=dtype
        )

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class NeuronQwen2VLAttention(NeuronAttentionBase):
    def __init__(self, config):
        super().__init__(
            config=config,
            hidden_size=config.embed_dim,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            head_dim=config.head_dim,
            num_cores_per_group=config.num_cores_per_group,
            sequence_parallel_enabled=False,
            rotary_emb=Qwen2VLVisionRotaryEmbedding(),
            qkv_bias=True,
            o_bias=True,
        )

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        self._position_embeddings = position_embeddings
        try:
            return super().forward(hidden_states, **kwargs)
        finally:
            self._position_embeddings = None

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, self._position_embeddings)
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
        return Q, K, cos_cache, sin_cache


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, vision_config) -> None:
        super().__init__()
        self.norm1 = LayerNorm(vision_config.embed_dim, eps=1e-6, dtype=vision_config.neuron_config.torch_dtype)
        self.norm2 = LayerNorm(vision_config.embed_dim, eps=1e-6, dtype=vision_config.neuron_config.torch_dtype)
        mlp_hidden_dim = int(vision_config.embed_dim * vision_config.mlp_ratio)
        self.attn = NeuronQwen2VLAttention(vision_config)
        self.mlp = VisionMlp(
            dim=vision_config.embed_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_act=vision_config.hidden_act,
            dtype=vision_config.neuron_config.torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        attn_output = self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
        )[0]
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class NeuronQwen2VisionModel(nn.Module):

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config

        self.spatial_merge_size = self.vision_config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=self.vision_config.patch_size,
            temporal_patch_size=self.vision_config.temporal_patch_size,
            in_channels=self.vision_config.in_channels,
            embed_dim=self.vision_config.embed_dim,
        ).to(self.vision_config.neuron_config.torch_dtype)

        head_dim = self.vision_config.embed_dim // self.vision_config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(self.vision_config) for _ in range(self.vision_config.depth)]
        )
        self.merger = PatchMerger(
            dim=self.vision_config.hidden_size,
            context_dim=self.vision_config.embed_dim,
            spatial_merge_size=self.vision_config.spatial_merge_size,
            dtype=self.vision_config.neuron_config.torch_dtype
        )

        # Calculate dynamic MAX_GRID_SIZE based on configured image dimensions
        # These can be configured via additional-config.json under vision_neuron_config:
        #   "vision_neuron_config": {
        #       "default_image_width": 1024,
        #       "default_image_height": 512,
        #       ...
        #   }
        # Falls back to constants (640x320) if not provided
        image_width, image_height = get_image_dimensions(self.vision_config.neuron_config)
        self.max_grid_size = calculate_max_grid_size(
            image_width,
            image_height,
            patch_size=self.vision_config.patch_size
        )
        logger.info(f"Calculated max_grid_size={self.max_grid_size} for image dimensions {image_width}x{image_height}")

        self.precomputed_rotary_pos_emb = self.rotary_pos_emb(self.max_grid_size)
        self.register_buffer(
            'rotary_pos_emb_cache',
            self.precomputed_rotary_pos_emb, persistent=False)

    def rot_pos_ids(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    def pad_to_text_seq_len(self, hidden_states):
        # pad to maximum seq len as we do not know the length of text tokens
        padded_length = self.config.neuron_config.seq_len
        hidden_states = hidden_states.to(self.config.text_config.neuron_config.torch_dtype)

        hidden_size = hidden_states.shape[-1]
        hidden_states, _ = pad_tensor(hidden_states, (padded_length, hidden_size), pad_value=0)

        # flatten vision outputs
        hidden_states = hidden_states.view(-1, hidden_size).unsqueeze(0)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        hidden_states = self.patch_embed(hidden_states)

        assert grid_thw[:, 1:].max() < self.max_grid_size, \
            f"Grid size {grid_thw[:, 1:].max()} exceeds max_grid_size {self.max_grid_size}. " \
            f"Increase default_image_width/height in vision_neuron_config."
        pos_ids = self.rot_pos_ids(grid_thw)
        rotary_pos_emb = self.rotary_pos_emb_cache[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        cos_emb = cos_emb.reshape(grid_thw.shape[0], -1, cos_emb.shape[-1])
        sin_emb = sin_emb.reshape(grid_thw.shape[0], -1, sin_emb.shape[-1])
        position_embeddings = (cos_emb, sin_emb)

        hidden_states = hidden_states.reshape(grid_thw.shape[0], -1, hidden_states.shape[-1])
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                position_embeddings)
        hidden_states_merger = self.merger(hidden_states)
        return self.pad_to_text_seq_len(hidden_states_merger)


class Qwen2VLVisionModelWrapper(ModelWrapper):
    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = False,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx,
            pipeline_execution, return_ranked_to_cpu, model_init_kwargs
        )
        # Calculate pixels_per_image once during initialization to avoid duplication
        image_width, image_height = get_image_dimensions(self.config.vision_config.neuron_config)
        resized_height, resized_width = smart_resize(width=image_width, height=image_height)
        self.pixels_per_image = (resized_height // self.config.vision_config.patch_size) * (resized_width // self.config.vision_config.patch_size)

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        inputs = []

        # Use pre-calculated pixels_per_image from __init__
        image_width, image_height = get_image_dimensions(self.config.vision_config.neuron_config)
        resized_height, resized_width = smart_resize(width=image_width, height=image_height)
        # bucket is num of images
        for bucket in self.config.vision_config.neuron_config.buckets:
            pixel_values = torch.ones(
                [
                    bucket * self.pixels_per_image,
                    self.config.vision_config.in_channels * self.config.vision_config.patch_size * self.config.vision_config.patch_size * self.config.vision_config.temporal_patch_size,
                ],
                dtype=self.config.vision_config.neuron_config.torch_dtype
            )
            grid_thw = torch.tensor(
                [[1,
                  resized_height // self.config.vision_config.patch_size,
                  resized_width // self.config.vision_config.patch_size]]).repeat(bucket, 1)

            inputs.append((pixel_values, grid_thw))

        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def get_padded_num_image(self, pixel_values):
        # pixel_values should be padded to vision model bucket
        buckets = self.config.vision_config.neuron_config.buckets

        for val in buckets:
            # pixel_values.shape: (pixels, num_of_patches)
            if val * self.pixels_per_image >= pixel_values.shape[0]:
                return val
        raise Exception(f"No bucket found for provided pixel_values with shape {pixel_values.shape[0]}. "
                        f"Calculated pixels_per_image={self.pixels_per_image}, buckets={buckets}")

    def forward(self, pixel_values, grid_thw):
        """
        Override ModelWrapper.forward().
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward")
        # Use pre-calculated pixels_per_image from __init__
        padded_num_image = self.get_padded_num_image(pixel_values)
        padded_pixel_values = pad_with_first_batchline(pixel_values, (padded_num_image * self.pixels_per_image, pixel_values.shape[1]))
        padded_grid_thw = pad_with_first_batchline(grid_thw, (padded_num_image, 3))
        output = self._forward(padded_pixel_values, padded_grid_thw)

        return output


class NeuronQwen2VLForImageEncoding(NeuronApplicationBase):
    """
    Neuron Application class for Qwen2VL image encoding case.
    Wraps NeuronQwen2VisionModel with Neuron specific functionalities such as compile and load.
    """

    _model_cls = NeuronQwen2VisionModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        # will only have one model one tag
        # after compilation, in /tmp/nxd_model,
        # you should only see one folder called f"self._model_cls.__name__"
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return Qwen2VLVisionModelWrapper

    def forward(self, pixel_values, grid_thw):
        return self.models[0](pixel_values, grid_thw)

    def get_compiler_args(self):
        compiler_args = "--auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor=2 ' -O1 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"
        logger.info(f'Compiling {self._model_cls.__name__} vision model with args: {compiler_args}')
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel, Qwen2VLConfig

        class hf_vision_model(torch.nn.Module):
            def __init__(self, model_path, **kwargs):
                super().__init__()
                # config
                self.hf_config = Qwen2VLConfig.from_pretrained(model_path, **kwargs)
                hf_vision_config = Qwen2VLConfig(**vars(self.hf_config.vision_config))
                self.visual = Qwen2VisionTransformerPretrainedModel._from_config(hf_vision_config)

            def forward(self, pixel_values, grid_thw):
                return self.visual(pixel_values, grid_thw)

            def save_pretrained(self, save_model_path):
                self.hf_config.save_pretrained(save_model_path)
                save_file(self.state_dict(), os.path.join(save_model_path, "model.safetensors"))

        hf_model = hf_vision_model(model_path, **kwargs)

        return hf_model

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        new_state_dict = {}
        for key, value in state_dict.items():
            if "visual." in key:
                key = key.replace("visual.", "")
                if ".attn.qkv." in key:
                    key = key.replace(".attn.qkv.", ".attn.qkv_proj.Wqkv.")
                elif ".attn.proj." in key:
                    key = key.replace(".attn.proj.", ".attn.o_proj.")
            new_state_dict[key] = (
                value.clone()
                .detach()
                .contiguous()
                .to(inference_config.vision_config.neuron_config.torch_dtype))

        del state_dict
        return new_state_dict

    @classmethod
    def get_config_cls(cls):
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLInferenceConfig
        return Qwen2VLInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        # Qwen2-VL only support batch size 1
        if len(prompts) > 1:
            raise NotImplementedError("Qwen2-VL currently only supports batch size 1")
        if isinstance(prompts, list):
            prompts = prompts[0]
        if images and isinstance(images, list) and isinstance(images[0], list):
            images = images[0]
        inputs = prepare_generation_inputs_hf(
            prompts, images, processor, role, config
        )
        vision_inputs = None
        if hasattr(inputs, "pixel_values") and hasattr(inputs, "image_grid_thw"):
            vision_inputs = {
                "pixel_values": inputs.pixel_values,
                "image_grid_thw": inputs.image_grid_thw
            }
        return inputs.input_ids, inputs.attention_mask, vision_inputs
