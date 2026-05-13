# coding=utf-8
# Copyright 2025 The Qwen Team. All rights reserved.
# Adapted for Qwen2.5-VL NxDI implementation.
#
# Vision encoder for Qwen2.5-VL:
# - RMSNorm (not LayerNorm like Qwen2-VL)
# - Gated SwiGLU MLP with bias=True (unique to Qwen2.5-VL)
# - Hybrid windowed/global attention (window_size=112, fullatt_block_indexes=[7,15,23,31])
# - 2D spatial rotary position embeddings (like Qwen2-VL, not learned like Qwen3-VL)
# - Patch merger with RMSNorm + GELU MLP
#
# Architecture pattern follows qwen3_vl: patch embed + pos embed on CPU in wrapper,
# compiled model only handles ViT blocks + merger. Bucketed on vision sequence length.

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

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
from neuronx_distributed_inference.modules.padding import (
    pad_tensor,
    pad_with_first_batchline,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb

logger = logging.getLogger("Neuron")


# =============================================================================
# Vision encoder components
# =============================================================================


class Qwen2_5_VLVisionRMSNorm(nn.Module):
    """RMSNorm for vision encoder (eps=1e-6)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class Qwen2_5_VLVisionRotaryEmbedding(nn.Module):
    """Pass-through rotary embedding that receives precomputed (cos, sin) from wrapper."""

    @torch.inference_mode()
    def forward(self, x, position_embeddings):
        cos, sin = position_embeddings
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class VisionRotaryEmbedding(nn.Module):
    """Precomputes rotary embeddings for vision encoder.
    Creates a lookup table of shape (max_grid_size, dim) that is indexed by position IDs."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5_VLVisionSwiGLUMLP(nn.Module):
    """Gated SwiGLU MLP with bias=True -- unique to Qwen2.5-VL vision encoder.
    gate_proj and up_proj are computed in parallel, then: silu(gate) * up -> down."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=True, gather_output=False, dtype=dtype
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=True, gather_output=False, dtype=dtype
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
            reduce_dtype=dtype,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2_5_VLVisionAttention(NeuronAttentionBase):
    """MHA (not GQA) vision attention with precomputed rotary embeddings.
    All 16 heads are attention heads (no KV head grouping)."""

    def __init__(self, config):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,  # MHA -- all heads are KV heads
            head_dim=config.head_dim,
            num_cores_per_group=config.num_cores_per_group,
            sequence_parallel_enabled=False,
            rotary_emb=Qwen2_5_VLVisionRotaryEmbedding(),
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


class Qwen2_5_VLVisionBlock(nn.Module):
    """Pre-norm vision transformer block: RMSNorm + Attention + RMSNorm + SwiGLU MLP."""

    def __init__(self, vision_config):
        super().__init__()
        dtype = vision_config.neuron_config.torch_dtype
        self.norm1 = Qwen2_5_VLVisionRMSNorm(
            vision_config.hidden_size, eps=1e-6, dtype=dtype
        )
        self.norm2 = Qwen2_5_VLVisionRMSNorm(
            vision_config.hidden_size, eps=1e-6, dtype=dtype
        )
        self.attn = Qwen2_5_VLVisionAttention(vision_config)
        self.mlp = Qwen2_5_VLVisionSwiGLUMLP(
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        attn_output = self.attn(
            self.norm1(hidden_states), position_embeddings=position_embeddings
        )[0]
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLPatchMerger(nn.Module):
    """Merge spatial_merge_size x spatial_merge_size patches into one token.
    Uses RMSNorm (unlike Qwen2-VL which uses LayerNorm)."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2_5_VLVisionRMSNorm(context_dim, eps=1e-6, dtype=dtype)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size, self.hidden_size, gather_output=False, dtype=dtype
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
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


# =============================================================================
# Windowed attention utilities (run on CPU in wrapper)
# =============================================================================


def get_window_index(grid_thw, window_size, spatial_merge_size, patch_size):
    """Compute window partition indices for windowed attention.

    For each image, partitions the merged-patch grid into spatial windows.
    Returns permutation indices that group tokens by window, and cumulative
    sequence lengths for both windowed and full attention.

    Args:
        grid_thw: Tensor of shape (num_images, 3) with [T, H, W] per image
        window_size: Pixel-space window size (e.g., 112)
        spatial_merge_size: Spatial merge factor (e.g., 2)
        patch_size: Patch size in pixels (e.g., 14)

    Returns:
        window_index: 1D tensor -- permutation to reorder tokens by window
        cu_window_seqlens: 1D tensor -- cumulative sequence lengths for windows
    """
    vit_merger_window_size = (
        window_size // spatial_merge_size // patch_size
    )  # typically 4

    all_window_indices = []
    all_window_seqlens = []

    spatial_merge_unit = spatial_merge_size * spatial_merge_size  # 4
    offset = 0

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size

        index = torch.arange(t * llm_grid_h * llm_grid_w).reshape(
            t, llm_grid_h, llm_grid_w
        )

        # Pad to window-aligned size
        pad_h = (
            vit_merger_window_size - llm_grid_h % vit_merger_window_size
        ) % vit_merger_window_size
        pad_w = (
            vit_merger_window_size - llm_grid_w % vit_merger_window_size
        ) % vit_merger_window_size

        if pad_h > 0 or pad_w > 0:
            index = F.pad(index, (0, pad_w, 0, pad_h), value=-100)

        padded_h = llm_grid_h + pad_h
        padded_w = llm_grid_w + pad_w
        num_windows_h = padded_h // vit_merger_window_size
        num_windows_w = padded_w // vit_merger_window_size

        # Reshape into windows
        index = index.reshape(
            t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index = index.permute(0, 1, 3, 2, 4)  # (T, nWh, nWw, wH, wW)
        index = index.reshape(
            t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )

        # Count valid tokens per window and build sequence lengths
        seqlens = (index != -100).sum(dim=[2, 3]).reshape(-1)  # (T * num_windows,)

        # Extract valid indices
        valid_mask = index != -100
        valid_indices = index[valid_mask]  # Flat list of valid indices

        all_window_indices.append(valid_indices + offset)
        all_window_seqlens.append(seqlens * spatial_merge_unit)

        offset += t * llm_grid_h * llm_grid_w

    window_index = torch.cat(all_window_indices)
    cu_window_seqlens_raw = torch.cat(all_window_seqlens)

    # Build cumulative sequence lengths
    cu_window_seqlens = torch.zeros(len(cu_window_seqlens_raw) + 1, dtype=torch.int32)
    cu_window_seqlens[1:] = cu_window_seqlens_raw.cumsum(0)

    return window_index, cu_window_seqlens


def compute_vision_rotary_pos_emb(grid_thw, spatial_merge_size, head_dim):
    """Compute 2D spatial rotary position embeddings for vision tokens.

    Args:
        grid_thw: Tensor of shape (num_images, 3) with [T, H, W]
        spatial_merge_size: Spatial merge factor (2)
        head_dim: Attention head dimension (80)

    Returns:
        cos, sin: Tensors of shape (num_images, max_seq_per_image, head_dim)
    """
    # Build position IDs per image
    pos_ids_list = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = (
            hpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .flatten()
        )

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = (
            wpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .flatten()
        )

        pos_ids_list.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids_list, dim=0)

    # Compute rotary embeddings
    max_grid_size = max(grid_thw[:, 1:].max().item(), 1)
    dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Build full embedding table
    freqs = torch.arange(max_grid_size, dtype=torch.float32).unsqueeze(
        1
    ) * inv_freq.unsqueeze(0)
    emb_cache = torch.cat((freqs, freqs), dim=-1)  # (max_grid_size, dim)

    # Index into cache
    rotary_pos_emb = emb_cache[pos_ids].flatten(1)  # (total_patches, head_dim)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    cos_emb = emb.cos()
    sin_emb = emb.sin()

    # Reshape per image batch
    cos_emb = cos_emb.reshape(grid_thw.shape[0], -1, cos_emb.shape[-1])
    sin_emb = sin_emb.reshape(grid_thw.shape[0], -1, sin_emb.shape[-1])

    return cos_emb, sin_emb


# =============================================================================
# Neuron vision model (compiled)
# =============================================================================


class NeuronQwen2_5_VLVisionModel(nn.Module):
    """Vision encoder that runs on Neuron.
    Patch embed, rotary pos embed, and window index computation happen on CPU in the wrapper.
    This model only contains the ViT blocks and patch merger."""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config

        dtype = self.vision_config.neuron_config.torch_dtype
        self.spatial_merge_size = self.vision_config.spatial_merge_size

        # Conv3D patch embedding
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionPatchEmbed,
        )

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=self.vision_config.patch_size,
            temporal_patch_size=self.vision_config.temporal_patch_size,
            in_channels=self.vision_config.in_chans,
            embed_dim=self.vision_config.hidden_size,
        ).to(dtype)

        # Rotary embedding (precomputed cache, same pattern as Qwen2-VL NxDI)
        head_dim = self.vision_config.hidden_size // self.vision_config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        # Calculate max grid size from default image dimensions
        default_w = getattr(
            self.vision_config.neuron_config, "default_image_width", 640
        )
        default_h = getattr(
            self.vision_config.neuron_config, "default_image_height", 640
        )
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        resized_h, resized_w = smart_resize(width=default_w, height=default_h)
        self.max_grid_size = (
            max(resized_h, resized_w) // self.vision_config.patch_size + 1
        )

        # Precompute and cache rotary embeddings
        precomputed = self.rotary_pos_emb(self.max_grid_size)
        self.register_buffer("rotary_pos_emb_cache", precomputed, persistent=False)

        # ViT blocks
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VLVisionBlock(self.vision_config)
                for _ in range(self.vision_config.depth)
            ]
        )

        # Patch merger: projects from vision hidden_size to text hidden_size
        self.merger = Qwen2_5_VLPatchMerger(
            dim=self.vision_config.out_hidden_size,
            context_dim=self.vision_config.hidden_size,
            spatial_merge_size=self.vision_config.spatial_merge_size,
            dtype=dtype,
        )

        # Store fullatt_block_indexes for windowed vs global attention routing
        self.fullatt_block_indexes = set(self.vision_config.fullatt_block_indexes)

    def rot_pos_ids(self, grid_thw):
        """Compute position IDs for rotary embedding indexing."""
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    def pad_to_text_seq_len(self, hidden_states):
        """Pad merged vision tokens to text sequence length."""
        padded_length = self.config.neuron_config.seq_len
        hidden_states = hidden_states.to(
            self.config.text_config.neuron_config.torch_dtype
        )
        hidden_size = hidden_states.shape[-1]
        hidden_states, _ = pad_tensor(
            hidden_states, (padded_length, hidden_size), pad_value=0
        )
        hidden_states = hidden_states.view(-1, hidden_size).unsqueeze(0)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """Forward pass through vision encoder.

        Args:
            hidden_states: Raw pixel patches.
                           Shape: (total_patches, in_chans * patch_size * patch_size * temporal_patch_size)
            grid_thw: Grid dimensions per image. Shape: (num_images, 3)

        Returns:
            Merged vision features padded to text seq_len.
            Shape: (1, seq_len, text_hidden_size)
        """
        # Patch embed: raw pixels -> hidden_size
        hidden_states = self.patch_embed(hidden_states)

        # Compute rotary position embeddings (via precomputed cache)
        pos_ids = self.rot_pos_ids(grid_thw)
        rotary_pos_emb = self.rotary_pos_emb_cache[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        cos_emb = cos_emb.reshape(grid_thw.shape[0], -1, cos_emb.shape[-1])
        sin_emb = sin_emb.reshape(grid_thw.shape[0], -1, sin_emb.shape[-1])
        position_embeddings = (cos_emb, sin_emb)

        # Reshape to (num_images, seq_per_image, hidden_size)
        hidden_states = hidden_states.reshape(
            grid_thw.shape[0], -1, hidden_states.shape[-1]
        )

        # Run through ViT blocks
        for blk in self.blocks:
            hidden_states = blk(hidden_states, position_embeddings)

        # Merge patches and project to text hidden dim
        hidden_states = self.merger(hidden_states)

        return self.pad_to_text_seq_len(hidden_states)


# =============================================================================
# Vision model wrapper (handles CPU preprocessing)
# =============================================================================


class Qwen2_5_VLVisionModelWrapper(ModelWrapper):
    """Wrapper that handles patch embedding, rotary position computation,
    and windowed attention preprocessing on CPU before passing to the
    compiled vision model on Neuron."""

    def __init__(
        self,
        config,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        pipeline_execution=True,
        return_ranked_to_cpu=False,
        model_init_kwargs={},
    ):
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
        # Calculate image dimensions for bucketing
        vc = self.config.vision_config
        default_w = getattr(vc.neuron_config, "default_image_width", 640)
        default_h = getattr(vc.neuron_config, "default_image_height", 640)
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        resized_h, resized_w = smart_resize(width=default_w, height=default_h)
        self.pixels_per_image = (resized_h // vc.patch_size) * (
            resized_w // vc.patch_size
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """Generate input shapes for each vision bucket (number of images)."""
        inputs = []
        vc = self.config.vision_config
        dtype = vc.neuron_config.torch_dtype

        default_w = getattr(vc.neuron_config, "default_image_width", 640)
        default_h = getattr(vc.neuron_config, "default_image_height", 640)
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        resized_h, resized_w = smart_resize(width=default_w, height=default_h)

        for bucket in vc.neuron_config.buckets:
            pixel_values = torch.ones(
                [
                    bucket * self.pixels_per_image,
                    vc.in_chans
                    * vc.patch_size
                    * vc.patch_size
                    * vc.temporal_patch_size,
                ],
                dtype=dtype,
            )
            grid_thw = torch.tensor(
                [[1, resized_h // vc.patch_size, resized_w // vc.patch_size]]
            ).repeat(bucket, 1)
            inputs.append((pixel_values, grid_thw))

        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def get_padded_num_image(self, pixel_values):
        buckets = self.config.vision_config.neuron_config.buckets
        for val in buckets:
            if val * self.pixels_per_image >= pixel_values.shape[0]:
                return val
        raise Exception(
            f"No bucket found for pixel_values with shape {pixel_values.shape[0]}. "
            f"pixels_per_image={self.pixels_per_image}, buckets={buckets}"
        )

    def forward(self, pixel_values, grid_thw):
        """Override forward: pad inputs and call compiled model."""
        if self.model is None:
            raise RuntimeError("Forward called before load()")

        padded_num_image = self.get_padded_num_image(pixel_values)
        padded_pixel_values = pad_with_first_batchline(
            pixel_values,
            (padded_num_image * self.pixels_per_image, pixel_values.shape[1]),
        )
        padded_grid_thw = pad_with_first_batchline(grid_thw, (padded_num_image, 3))
        return self._forward(padded_pixel_values, padded_grid_thw)


# =============================================================================
# NeuronApplication class
# =============================================================================


class NeuronQwen2_5_VLForImageEncoding(NeuronApplicationBase):
    """Neuron application class for Qwen2.5-VL vision encoder."""

    _model_cls = NeuronQwen2_5_VLVisionModel

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
        return Qwen2_5_VLVisionModelWrapper

    def forward(self, pixel_values, grid_thw):
        return self.models[0](pixel_values, grid_thw)

    def get_compiler_args(self):
        return (
            "--auto-cast=none --model-type=transformer "
            "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
            "-O1 --internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

        class HFVisionModel(nn.Module):
            def __init__(self, model_path, **kwargs):
                super().__init__()
                self.hf_config = Qwen2_5_VLConfig.from_pretrained(model_path, **kwargs)
                # Load just the vision model
                full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, **kwargs
                )
                self.visual = full_model.model.visual
                del full_model

            def forward(self, pixel_values, grid_thw):
                return self.visual(pixel_values, grid_thw)

        return HFVisionModel(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        """Convert HF vision state dict to Neuron format.
        Maps: visual.blocks.N.attn.qkv -> blocks.N.attn.qkv_proj.Wqkv
        Maps: visual.blocks.N.attn.proj -> blocks.N.attn.o_proj
        Preserves all non-vision keys for downstream text conversion."""
        new_state_dict = {}
        for key, value in state_dict.items():
            if "visual." in key:
                new_key = key.replace("visual.", "")
                if ".attn.qkv." in new_key:
                    new_key = new_key.replace(".attn.qkv.", ".attn.qkv_proj.Wqkv.")
                elif ".attn.proj." in new_key:
                    new_key = new_key.replace(".attn.proj.", ".attn.o_proj.")
            else:
                new_key = key
            new_state_dict[new_key] = (
                value.clone()
                .detach()
                .contiguous()
                .to(inference_config.vision_config.neuron_config.torch_dtype)
            )
        del state_dict
        return new_state_dict

    @classmethod
    def get_config_cls(cls):
        from src.modeling_qwen2_5_vl import Qwen2_5_VLInferenceConfig

        return Qwen2_5_VLInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        """Prepare inputs for generation -- single batch only for now."""
        if len(prompts) > 1:
            raise NotImplementedError("Qwen2.5-VL currently only supports batch size 1")
        if isinstance(prompts, list):
            prompts = prompts[0]
        if images and isinstance(images, list) and isinstance(images[0], list):
            images = images[0]

        # Build conversation format
        content = []
        if images:
            for img in images if isinstance(images, list) else [images]:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompts})

        messages = [{"role": role, "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        )

        vision_inputs = None
        if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
            vision_inputs = {
                "pixel_values": inputs.pixel_values,
                "image_grid_thw": inputs.image_grid_thw,
            }

        return inputs.input_ids, inputs.attention_mask, vision_inputs
