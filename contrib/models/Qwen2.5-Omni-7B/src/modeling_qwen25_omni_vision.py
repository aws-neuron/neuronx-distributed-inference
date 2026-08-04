# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni Vision Encoder for NXD inference.
#
# Differences from Qwen2-VL vision encoder:
#   - SwiGLU MLP (gate_proj, up_proj, down_proj) instead of simple FC1/FC2
#   - RMSNorm instead of LayerNorm
#   - Separate Q/K/V projections instead of fused QKV
#   - intermediate_size=3420 (not TP-divisible by 32, use nn.Linear for MLP)

"""Qwen2.5-Omni Vision Encoder for NXD inference."""

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    VisionRotaryEmbedding,
)

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import (
    EncoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import (
    Qwen2VLVisionRotaryEmbedding,
)
from neuronx_distributed_inference.models.qwen2_vl.utils.vision_utils import (
    calculate_max_grid_size,
    get_image_dimensions,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb
from neuronx_distributed_inference.modules.padding import (
    pad_tensor,
    pad_with_first_batchline,
)

logger = logging.getLogger(__name__)


class VisionRMSNorm(nn.Module):
    """RMSNorm for vision encoder (replaces LayerNorm used in Qwen2-VL)."""

    def __init__(self, hidden_size, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VisionSwiGLUMLP(nn.Module):
    """SwiGLU MLP for Qwen2.5-Omni vision encoder.

    Uses regular nn.Linear instead of ColumnParallelLinear/RowParallelLinear
    because intermediate_size=3420 is not divisible by common TP degrees (16, 32).
    The vision model is small enough that MLP weight replication is acceptable.
    """

    def __init__(self, dim, hidden_dim, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)
        # Cast to target dtype
        self.gate_proj = self.gate_proj.to(dtype)
        self.up_proj = self.up_proj.to(dtype)
        self.down_proj = self.down_proj.to(dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NeuronQwen25OmniVisionAttention(NeuronAttentionBase):
    """Vision attention with separate Q/K/V projections (not fused).

    Requires vision_config.neuron_config.fused_qkv = False.
    """

    def __init__(self, config):
        super().__init__(
            config=config,
            hidden_size=config.embed_dim,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            head_dim=config.embed_dim // config.num_heads,
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

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, self._position_embeddings)
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
        return Q, K, cos_cache, sin_cache


class Qwen25OmniVisionBlock(nn.Module):
    """Vision transformer block with RMSNorm and SwiGLU MLP."""

    def __init__(self, vision_config):
        super().__init__()
        dtype = vision_config.neuron_config.torch_dtype
        self.norm1 = VisionRMSNorm(
            vision_config.embed_dim, eps=1e-6, dtype=dtype
        )
        self.norm2 = VisionRMSNorm(
            vision_config.embed_dim, eps=1e-6, dtype=dtype
        )
        self.attn = NeuronQwen25OmniVisionAttention(vision_config)
        self.mlp = VisionSwiGLUMLP(
            dim=vision_config.embed_dim,
            hidden_dim=vision_config.intermediate_size,
            dtype=dtype,
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


class Qwen25OmniPatchMerger(nn.Module):
    """Patch merger with RMSNorm (Qwen2-VL uses LayerNorm).

    Merges spatial_merge_size^2 patches into one, projecting from
    embed_dim to out_hidden_size (text model hidden size).
    """

    def __init__(self, dim, context_dim, spatial_merge_size=2, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = VisionRMSNorm(context_dim, eps=1e-6, dtype=dtype)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=False,
                dtype=dtype,
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size,
                dim,
                input_is_parallel=True,
                dtype=dtype,
                reduce_dtype=dtype,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class NeuronQwen25OmniVisionModel(nn.Module):
    """Qwen2.5-Omni Vision Encoder on Neuron.

    Architecture is based on Qwen2-VL ViT but uses RMSNorm, SwiGLU MLP,
    and separate Q/K/V projections. Reuses the same PatchEmbed and
    VisionRotaryEmbedding from Qwen2-VL (identical parameters).
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config

        self.spatial_merge_size = self.vision_config.spatial_merge_size

        # Reuse Qwen2-VL PatchEmbed (same Conv3D architecture)
        self.patch_embed = PatchEmbed(
            patch_size=self.vision_config.patch_size,
            temporal_patch_size=self.vision_config.temporal_patch_size,
            in_channels=self.vision_config.in_channels,
            embed_dim=self.vision_config.embed_dim,
        ).to(self.vision_config.neuron_config.torch_dtype)

        head_dim = self.vision_config.embed_dim // self.vision_config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen25OmniVisionBlock(self.vision_config)
                for _ in range(self.vision_config.depth)
            ]
        )

        self.merger = Qwen25OmniPatchMerger(
            dim=self.vision_config.out_hidden_size,  # 3584 (text hidden size)
            context_dim=self.vision_config.embed_dim,  # 1280
            spatial_merge_size=self.vision_config.spatial_merge_size,
            dtype=self.vision_config.neuron_config.torch_dtype,
        )

        # Calculate dynamic MAX_GRID_SIZE based on configured image dimensions
        image_width, image_height = get_image_dimensions(
            self.vision_config.neuron_config
        )
        self.max_grid_size = calculate_max_grid_size(
            image_width,
            image_height,
            patch_size=self.vision_config.patch_size,
        )
        logger.info(
            f"Calculated max_grid_size={self.max_grid_size} for "
            f"image dimensions {image_width}x{image_height}"
        )

        self.precomputed_rotary_pos_emb = self.rotary_pos_emb(self.max_grid_size)
        self.register_buffer(
            "rotary_pos_emb_cache",
            self.precomputed_rotary_pos_emb,
            persistent=False,
        )

    def rot_pos_ids(self, grid_thw):
        """Compute rotary position IDs for patches (same algorithm as Qwen2-VL)."""
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
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            )
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    def pad_to_text_seq_len(self, hidden_states):
        """Pad vision outputs to text model sequence length."""
        padded_length = self.config.neuron_config.seq_len
        hidden_states = hidden_states.to(
            self.config.text_config.neuron_config.torch_dtype
        )

        hidden_size = hidden_states.shape[-1]
        hidden_states, _ = pad_tensor(
            hidden_states, (padded_length, hidden_size), pad_value=0
        )

        # Flatten vision outputs: (seq_len, hidden_size) -> (1, seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, hidden_size).unsqueeze(0)
        return hidden_states

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        assert grid_thw[:, 1:].max() < self.max_grid_size, (
            f"Grid size {grid_thw[:, 1:].max()} exceeds max_grid_size "
            f"{self.max_grid_size}. Increase default_image_width/height "
            f"in vision_neuron_config."
        )
        pos_ids = self.rot_pos_ids(grid_thw)
        rotary_pos_emb = self.rotary_pos_emb_cache[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        cos_emb = cos_emb.reshape(grid_thw.shape[0], -1, cos_emb.shape[-1])
        sin_emb = sin_emb.reshape(grid_thw.shape[0], -1, sin_emb.shape[-1])
        position_embeddings = (cos_emb, sin_emb)

        hidden_states = hidden_states.reshape(
            grid_thw.shape[0], -1, hidden_states.shape[-1]
        )
        for blk in self.blocks:
            hidden_states = blk(hidden_states, position_embeddings)
        hidden_states_merger = self.merger(hidden_states)
        return self.pad_to_text_seq_len(hidden_states_merger)


class Qwen25OmniVisionModelWrapper(ModelWrapper):
    """Model wrapper for Qwen2.5-Omni vision encoder.

    Handles bucketing on number of images, padding, and input generation.
    Same pattern as Qwen2VLVisionModelWrapper.
    """

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
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            smart_resize,
        )

        image_width, image_height = get_image_dimensions(
            self.config.vision_config.neuron_config
        )
        resized_height, resized_width = smart_resize(
            width=image_width, height=image_height
        )
        self.pixels_per_image = (
            resized_height // self.config.vision_config.patch_size
        ) * (resized_width // self.config.vision_config.patch_size)

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            smart_resize,
        )

        inputs = []
        image_width, image_height = get_image_dimensions(
            self.config.vision_config.neuron_config
        )
        resized_height, resized_width = smart_resize(
            width=image_width, height=image_height
        )
        vc = self.config.vision_config
        for bucket in vc.neuron_config.buckets:
            pixel_values = torch.ones(
                [
                    bucket * self.pixels_per_image,
                    vc.in_channels
                    * vc.patch_size
                    * vc.patch_size
                    * vc.temporal_patch_size,
                ],
                dtype=vc.neuron_config.torch_dtype,
            )
            grid_thw = torch.tensor(
                [
                    [
                        1,
                        resized_height // vc.patch_size,
                        resized_width // vc.patch_size,
                    ]
                ]
            ).repeat(bucket, 1)
            inputs.append((pixel_values, grid_thw))
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def get_padded_num_image(self, pixel_values):
        """Get the bucket size (number of images) for given pixel_values."""
        buckets = self.config.vision_config.neuron_config.buckets
        for val in buckets:
            if val * self.pixels_per_image >= pixel_values.shape[0]:
                return val
        raise Exception(
            f"No bucket found for pixel_values shape {pixel_values.shape[0]}. "
            f"pixels_per_image={self.pixels_per_image}, buckets={buckets}"
        )

    def forward(self, pixel_values, grid_thw):
        """Override ModelWrapper.forward() with padding to bucket size."""
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() "
                "before calling forward"
            )
        padded_num_image = self.get_padded_num_image(pixel_values)
        padded_pixel_values = pad_with_first_batchline(
            pixel_values,
            (padded_num_image * self.pixels_per_image, pixel_values.shape[1]),
        )
        padded_grid_thw = pad_with_first_batchline(
            grid_thw, (padded_num_image, 3)
        )
        output = self._forward(padded_pixel_values, padded_grid_thw)
        return output


class NeuronQwen25OmniForImageEncoding(NeuronApplicationBase):
    """Standalone Neuron Application for Qwen2.5-Omni image encoding.

    Wraps NeuronQwen25OmniVisionModel with compile/load functionality.
    Can be used independently or as part of the full multimodal model.
    """

    _model_cls = NeuronQwen25OmniVisionModel

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
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return Qwen25OmniVisionModelWrapper

    def forward(self, pixel_values, grid_thw):
        return self.models[0](pixel_values, grid_thw)

    def get_compiler_args(self):
        compiler_args = (
            "--auto-cast=none --model-type=transformer "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 ' -O1 "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        logger.info(
            f"Compiling {self._model_cls.__name__} vision model "
            f"with args: {compiler_args}"
        )
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the full Qwen2.5-Omni model (vision weights will be filtered
        in convert_hf_to_neuron_state_dict)."""
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        """Convert HF state dict to NxDI format for vision encoder.

        Handles:
          1. Filter to vision keys only (thinker.visual.* or visual.*)
          2. Strip prefix
          3. Rename separate Q/K/V attention keys to NeuronAttentionBase format
          4. Cast to target dtype
        """
        new_state_dict = {}
        dtype = inference_config.vision_config.neuron_config.torch_dtype

        for key, value in state_dict.items():
            # Accept both "thinker.visual." and "visual." prefixes
            if key.startswith("thinker.visual."):
                new_key = key[len("thinker.visual."):]
            elif key.startswith("visual."):
                new_key = key[len("visual."):]
            else:
                # Pass through non-vision keys unchanged
                new_state_dict[key] = value
                continue

            # Rename attention keys: separate Q/K/V -> NeuronAttentionBase format
            if ".attn.proj." in new_key:
                new_key = new_key.replace(".attn.proj.", ".attn.o_proj.")
            elif ".attn.q." in new_key:
                new_key = new_key.replace(
                    ".attn.q.", ".attn.qkv_proj.q_proj."
                )
            elif ".attn.k." in new_key:
                new_key = new_key.replace(
                    ".attn.k.", ".attn.qkv_proj.k_proj."
                )
            elif ".attn.v." in new_key:
                new_key = new_key.replace(
                    ".attn.v.", ".attn.qkv_proj.v_proj."
                )

            new_state_dict[new_key] = (
                value.clone().detach().contiguous().to(dtype)
            )

        del state_dict
        return new_state_dict

    @classmethod
    def get_config_cls(cls):
        from modeling_qwen25_omni import (
            Qwen25OmniMultimodalInferenceConfig,
        )

        return Qwen25OmniMultimodalInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        """Prepare input arguments for Qwen2.5-Omni vision model."""
        from neuronx_distributed_inference.models.qwen2_vl.utils.input_processor import (
            prepare_generation_inputs_hf,
        )

        if len(prompts) > 1:
            raise NotImplementedError(
                "Qwen2.5-Omni currently only supports batch size 1"
            )
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
                "image_grid_thw": inputs.image_grid_thw,
            }
        return inputs.input_ids, inputs.attention_mask, vision_inputs
