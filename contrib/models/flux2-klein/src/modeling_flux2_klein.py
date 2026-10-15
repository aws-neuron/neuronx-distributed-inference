# Copyright 2024 Black Forest Labs, The HuggingFace Team. All rights reserved.
#
# This implementation is derived from the Diffusers library and the NxD Inference
# FLUX implementation. It has been adapted for FLUX.2-klein-base-9B architecture
# and optimized for execution on Amazon Neuron devices with tensor parallelism.
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

"""
FLUX.2-klein-base-9B NxDI model implementation.

Architecture differences from FLUX.1:
- Pre-computed modulation (shared across all blocks, not per-block)
- SwiGLU activation in feed-forward (not GELU)
- Fused QKV+MLP in single-stream blocks (Flux2ParallelSelfAttention)
- 4D RoPE (T, H, W, L) with axes_dims=(32,32,32,32), theta=2000
- Single text encoder: Qwen3-8B (not CLIP+T5)
- 32 latent channels packed to 128 (not 16->64)
- Classic CFG (two forward passes) instead of guidance distillation
- Timestep embedding without pooled text projection
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
        SPMDRank,
    )
    from neuronx_distributed.parallel_layers.mappings import (
        gather_from_tensor_model_parallel_region_with_dim,
        reduce_from_tensor_model_parallel_region,
    )
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_group,
        get_tensor_model_parallel_size,
        get_world_group,
    )
    from neuronx_distributed_inference.models.diffusers.embeddings import (
        FluxPosEmbed,
        apply_rotary_emb,
    )
    from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
    from neuronx_distributed_inference.utils.distributed import get_dp_rank_spmd
    from nkilib.core.attention.attention_cte import attention_cte
    from neuronx_distributed.utils.utils import hardware
    from torch_neuronx.utils import get_platform_target
    from neuronx_distributed_inference.models.application_base import (
        NeuronApplicationBase,
    )
    from neuronx_distributed_inference.models.config import InferenceConfig
    from neuronx_distributed_inference.models.layer_boundary_marker import (
        ModuleMarkerEndWrapper,
        ModuleMarkerStartWrapper,
    )
    from neuronx_distributed_inference.models.model_wrapper import (
        BaseModelInstance,
        ModelWrapper,
    )

    NEURON_AVAILABLE = True
    _HARDWARE = hardware(get_platform_target())
except ImportError:
    NEURON_AVAILABLE = False
    _HARDWARE = None

logger = logging.getLogger(__name__)


# ============================================================
# NKI Flash Attention Wrapper
# ============================================================


def attention_wrapper_sharded(query, key, value):
    """
    NKI flash attention for FLUX.2-klein (bi-directional, no causal mask).

    Input shapes: query, key, value all [bs, n_head, seq_len, d_head]
    Output shape: [bs, n_head, q_len, d_head]
    """
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]

    q = query.reshape((bs * n_head, q_len, d_head))
    k = key.reshape((bs * n_head, k_len, d_head))
    v = value.reshape((bs * n_head, v_len, d_head))

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    use_sharded = vc_size == 2
    scale = 1 / math.sqrt(d_head)

    if use_sharded:
        attn_output = attention_cte[2](
            q,
            k,
            v,
            scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )
    else:
        attn_output = attention_cte(
            q,
            k,
            v,
            scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )

    return attn_output.reshape((bs, n_head, q_len, d_head))


# ============================================================
# FLUX.2-klein Modulation
# ============================================================


class NeuronFlux2Modulation(nn.Module):
    """
    Pre-computed modulation for FLUX.2-klein.

    Unlike FLUX.1 where each block has its own AdaLayerNormZero with
    an internal linear projection, FLUX.2 computes modulation ONCE from
    the timestep embedding and passes it to ALL blocks.

    Args:
        dim: Hidden dimension (4096 for Klein)
        mod_param_sets: Number of parameter sets per block type.
            - 2 for double blocks (shift/scale/gate for attn + shift/scale/gate for FF = 6 params)
            - 1 for single blocks (shift/scale/gate for fused attn+MLP = 3 params)
    """

    def __init__(self, dim: int, mod_param_sets: int = 1, reduce_dtype=torch.bfloat16):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.act = nn.SiLU()
        # Output: 3 * mod_param_sets * dim
        # For single: 3 * 1 * 4096 = 12288
        # For double: 3 * 2 * 4096 = 24576
        self.linear = ColumnParallelLinear(
            dim,
            3 * mod_param_sets * dim,
            bias=False,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temb: [B, dim] timestep embedding
        Returns:
            modulation params: [B, 3 * mod_param_sets * dim]
        """
        return self.linear(self.act(temb))

    @staticmethod
    def split(mod: torch.Tensor, n: int):
        """Split modulation output into n groups of (shift, scale, gate)."""
        # mod: [B, 3*n*dim] -> n groups of [B, 3*dim] -> each into (shift, scale, gate)
        chunks = mod.chunk(3 * n, dim=-1)
        groups = []
        for i in range(n):
            shift = chunks[3 * i]
            scale = chunks[3 * i + 1]
            gate = chunks[3 * i + 2]
            groups.append((shift, scale, gate))
        return groups


# ============================================================
# FLUX.2-klein Timestep Embedding
# ============================================================


class NeuronFlux2TimestepEmbedder(nn.Module):
    """Container for timestep MLP to match HF key naming."""

    def __init__(self, time_proj_dim, embedding_dim, reduce_dtype):
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            time_proj_dim,
            embedding_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.linear_2 = RowParallelLinear(
            embedding_dim,
            embedding_dim,
            bias=False,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
        )


class NeuronFlux2TimestepEmbedding(nn.Module):
    """
    Timestep + optional guidance embedding for FLUX.2-klein.

    Unlike FLUX.1 which also incorporates a pooled text projection,
    FLUX.2's time_guidance_embed only takes timestep and guidance.

    HF key structure:
        time_guidance_embed.timestep_embedder.linear_1.weight  [4096, 256]
        time_guidance_embed.timestep_embedder.linear_2.weight  [4096, 4096]
    """

    def __init__(self, embedding_dim: int, reduce_dtype=torch.bfloat16):
        super().__init__()
        self.time_proj_dim = 256
        self.timestep_embedder = NeuronFlux2TimestepEmbedder(
            self.time_proj_dim, embedding_dim, reduce_dtype
        )
        self.guidance_proj = None  # Klein base does not use guidance

    def get_timestep_sinusoidal(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional encoding for timesteps."""
        half_dim = self.time_proj_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
        )
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb.to(timesteps.dtype)

    def forward(
        self, timestep: torch.Tensor, guidance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t_emb = self.get_timestep_sinusoidal(timestep)
        temb = F.silu(self.timestep_embedder.linear_1(t_emb))
        temb = self.timestep_embedder.linear_2(temb)
        return temb


# ============================================================
# ============================================================
# FLUX.2-klein Feed-Forward (Double-stream blocks)
# ============================================================


class NeuronFlux2FeedForward(nn.Module):
    """
    SwiGLU-based feed-forward for FLUX.2-klein double-stream blocks.

    Architecture: SwiGLU(Linear_gate(x), Linear_value(x)) -> Linear_out

    HF key structure:
        ff.linear_in.weight  [inner_dim*2, dim]  (gate and value concatenated)
        ff.linear_out.weight [dim, inner_dim]

    For TP: We split linear_in into two ColumnParallelLinear (gate + value)
    to avoid the SwiGLU split crossing TP partition boundaries.
    """

    def __init__(self, dim: int, mlp_ratio: float = 3.0, reduce_dtype=torch.bfloat16):
        super().__init__()
        inner_dim = int(dim * mlp_ratio)

        # Two separate projections for SwiGLU gate and value
        # These will be loaded by splitting the fused linear_in weight
        self.linear_in_gate = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.linear_in_value = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.linear_out = RowParallelLinear(
            inner_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.linear_in_gate(x))
        value = self.linear_in_value(x)
        return self.linear_out(gate * value)


# ============================================================
# FLUX.2-klein Attention (shared between double and single blocks)
# ============================================================


class NeuronFlux2Attention(nn.Module):
    """
    Attention module for FLUX.2-klein double-stream blocks.

    Separate Q/K/V projections for image and text streams,
    concatenated for joint attention, then split back.

    Uses RMSNorm on Q/K (not LayerNorm as in FLUX.1).
    No bias on Q/K/V linears.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        added_kv_proj_dim: Optional[int] = None,
        bias: bool = False,
        eps: float = 1e-6,
        reduce_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim

        tp_degree = get_tensor_model_parallel_size()
        # Pad heads to be divisible by TP degree
        padded_heads = math.ceil(num_attention_heads / tp_degree) * tp_degree
        self.padded_inner_dim = padded_heads * attention_head_dim
        self.heads_per_rank = padded_heads // tp_degree

        # Image stream Q/K/V
        self.to_q = ColumnParallelLinear(
            dim,
            self.padded_inner_dim,
            bias=bias,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_k = ColumnParallelLinear(
            dim,
            self.padded_inner_dim,
            bias=bias,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_v = ColumnParallelLinear(
            dim,
            self.padded_inner_dim,
            bias=bias,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )

        # QK norm
        self.norm_q = CustomRMSNorm(attention_head_dim, eps=eps)
        self.norm_k = CustomRMSNorm(attention_head_dim, eps=eps)

        # Text stream Q/K/V (for double-stream blocks)
        self.added_kv_proj_dim = added_kv_proj_dim
        if added_kv_proj_dim is not None:
            self.add_q_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.padded_inner_dim,
                bias=bias,
                gather_output=False,
                reduce_dtype=reduce_dtype,
            )
            self.add_k_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.padded_inner_dim,
                bias=bias,
                gather_output=False,
                reduce_dtype=reduce_dtype,
            )
            self.add_v_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.padded_inner_dim,
                bias=bias,
                gather_output=False,
                reduce_dtype=reduce_dtype,
            )
            self.norm_added_q = CustomRMSNorm(attention_head_dim, eps=eps)
            self.norm_added_k = CustomRMSNorm(attention_head_dim, eps=eps)

            # Output projections
            # HF uses ModuleList for to_out (key: attn.to_out.0.weight)
            # but direct Linear for to_add_out (key: attn.to_add_out.weight)
            self.to_out = nn.ModuleList(
                [
                    RowParallelLinear(
                        self.padded_inner_dim,
                        dim,
                        bias=False,
                        input_is_parallel=True,
                        reduce_dtype=reduce_dtype,
                    ),
                ]
            )
            self.to_add_out = RowParallelLinear(
                self.padded_inner_dim,
                dim,
                bias=False,
                input_is_parallel=True,
                reduce_dtype=reduce_dtype,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = hidden_states.shape[0]
        head_dim = self.head_dim

        # Image Q/K/V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.view(batch_size, -1, self.heads_per_rank, head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.heads_per_rank, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads_per_rank, head_dim).transpose(
            1, 2
        )

        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            # Text Q/K/V
            add_query = self.add_q_proj(encoder_hidden_states)
            add_key = self.add_k_proj(encoder_hidden_states)
            add_value = self.add_v_proj(encoder_hidden_states)

            add_query = add_query.view(
                batch_size, -1, self.heads_per_rank, head_dim
            ).transpose(1, 2)
            add_key = add_key.view(
                batch_size, -1, self.heads_per_rank, head_dim
            ).transpose(1, 2)
            add_value = add_value.view(
                batch_size, -1, self.heads_per_rank, head_dim
            ).transpose(1, 2)

            add_query = self.norm_added_q(add_query)
            add_key = self.norm_added_k(add_key)

            # Concatenate text + image for joint attention BEFORE RoPE
            # image_rotary_emb has positions for [text_seq + img_seq]
            query = torch.cat([add_query, query], dim=2)
            key = torch.cat([add_key, key], dim=2)
            value = torch.cat([add_value, value], dim=2)

            # Apply RoPE to the full concatenated sequence
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            # Flash attention
            if _HARDWARE == hardware.TRN1:
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=False,
                )
            else:
                hidden_states = attention_wrapper_sharded(query, key, value)

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads_per_rank * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

            # Split back to text and image
            txt_len = encoder_hidden_states.shape[1]
            encoder_attn_out = hidden_states[:, :txt_len]
            hidden_attn_out = hidden_states[:, txt_len:]

            hidden_attn_out = self.to_out[0](hidden_attn_out)
            encoder_attn_out = self.to_add_out(encoder_attn_out)

            return hidden_attn_out, encoder_attn_out
        else:
            # Single-stream: no text stream (used when called from parallel attn)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            if _HARDWARE == hardware.TRN1:
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=False,
                )
            else:
                hidden_states = attention_wrapper_sharded(query, key, value)

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads_per_rank * head_dim
            )
            return hidden_states.to(query.dtype), None


# ============================================================
# FLUX.2-klein Double-Stream Transformer Block
# ============================================================


class NeuronFlux2TransformerBlock(nn.Module):
    """
    Double-stream transformer block for FLUX.2-klein.

    Unlike FLUX.1's NeuronFluxTransformerBlock which has internal AdaLayerNormZero,
    this block receives pre-computed modulation parameters from the parent model.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        reduce_dtype=torch.bfloat16,
    ):
        super().__init__()

        # Layer norms (no affine parameters -- modulated externally via shift/scale)
        self.norm1 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = LayerNorm(dim, elementwise_affine=False, eps=eps)

        # Joint attention
        self.attn = NeuronFlux2Attention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            added_kv_proj_dim=dim,
            bias=False,
            eps=eps,
            reduce_dtype=reduce_dtype,
        )

        # Feed-forward (SwiGLU, not GELU)
        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = NeuronFlux2FeedForward(
            dim=dim, mlp_ratio=mlp_ratio, reduce_dtype=reduce_dtype
        )

        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = NeuronFlux2FeedForward(
            dim=dim, mlp_ratio=mlp_ratio, reduce_dtype=reduce_dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [B, img_seq, dim] image features
            encoder_hidden_states: [B, txt_seq, dim] text features
            temb_mod_img: [B, 6*dim] pre-computed image modulation (2 sets of shift/scale/gate)
            temb_mod_txt: [B, 6*dim] pre-computed text modulation
            image_rotary_emb: RoPE frequencies
        """
        # Split modulation: 2 sets of (shift, scale, gate) for img and txt
        img_mods = NeuronFlux2Modulation.split(temb_mod_img, 2)
        txt_mods = NeuronFlux2Modulation.split(temb_mod_txt, 2)

        img_shift_attn, img_scale_attn, img_gate_attn = img_mods[0]
        img_shift_ff, img_scale_ff, img_gate_ff = img_mods[1]
        txt_shift_attn, txt_scale_attn, txt_gate_attn = txt_mods[0]
        txt_shift_ff, txt_scale_ff, txt_gate_ff = txt_mods[1]

        # Attention sub-block: image stream
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = (
            norm_hidden * (1 + img_scale_attn[:, None]) + img_shift_attn[:, None]
        )

        # Attention sub-block: text stream
        norm_encoder = self.norm1_context(encoder_hidden_states)
        norm_encoder = (
            norm_encoder * (1 + txt_scale_attn[:, None]) + txt_shift_attn[:, None]
        )

        # Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden,
            encoder_hidden_states=norm_encoder,
            image_rotary_emb=image_rotary_emb,
        )

        # Residual + gate for image
        hidden_states = hidden_states + img_gate_attn[:, None] * attn_output

        # FF sub-block: image stream
        norm_hidden = self.norm2(hidden_states)
        norm_hidden = norm_hidden * (1 + img_scale_ff[:, None]) + img_shift_ff[:, None]
        ff_output = self.ff(norm_hidden)
        hidden_states = hidden_states + img_gate_ff[:, None] * ff_output

        # Residual + gate for text
        encoder_hidden_states = (
            encoder_hidden_states + txt_gate_attn[:, None] * context_attn_output
        )

        # FF sub-block: text stream
        norm_encoder = self.norm2_context(encoder_hidden_states)
        norm_encoder = (
            norm_encoder * (1 + txt_scale_ff[:, None]) + txt_shift_ff[:, None]
        )
        context_ff_output = self.ff_context(norm_encoder)
        encoder_hidden_states = (
            encoder_hidden_states + txt_gate_ff[:, None] * context_ff_output
        )

        return encoder_hidden_states, hidden_states


# ============================================================
# FLUX.2-klein Single-Stream Transformer Block
# ============================================================


class NeuronFlux2SingleTransformerBlock(nn.Module):
    """
    Single-stream transformer block for FLUX.2-klein with fused attention+MLP.

    The key architectural feature is Flux2ParallelSelfAttention: a single large
    linear projects to Q, K, V, and MLP input simultaneously. The MLP uses SwiGLU.
    Attention output and MLP output are concatenated and projected together.

    For NxDI with tensor parallelism, we split this fused projection into separate
    TP-sharded Q/K/V and MLP projections to stay within compiler instruction limits.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        reduce_dtype=torch.bfloat16,
    ):
        super().__init__()

        self.norm = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        tp_degree = get_tensor_model_parallel_size()
        padded_heads = math.ceil(num_attention_heads / tp_degree) * tp_degree
        self.padded_inner_dim = padded_heads * attention_head_dim
        self.heads_per_rank = padded_heads // tp_degree
        self.head_dim = attention_head_dim

        # Separate Q/K/V projections (split from the fused to_qkv_mlp_proj)
        self.to_q = ColumnParallelLinear(
            dim,
            self.padded_inner_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_k = ColumnParallelLinear(
            dim,
            self.padded_inner_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_v = ColumnParallelLinear(
            dim,
            self.padded_inner_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )

        # QK norm
        self.norm_q = CustomRMSNorm(attention_head_dim, eps=eps)
        self.norm_k = CustomRMSNorm(attention_head_dim, eps=eps)

        # MLP input projection: split the fused to_qkv_mlp_proj's MLP part
        # into gate and value for correct TP sharding with SwiGLU
        self.proj_mlp_gate = ColumnParallelLinear(
            dim,
            self.mlp_hidden_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.proj_mlp_value = ColumnParallelLinear(
            dim,
            self.mlp_hidden_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )

        # Output: attention + MLP projected separately then summed
        # (Same pattern as FLUX.1 NxDI: split proj_out into attn + mlp parts)
        self.proj_out_attn = RowParallelLinear(
            self.padded_inner_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
            reduce_output=False,
        )
        self.proj_out_mlp = RowParallelLinear(
            self.mlp_hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
            reduce_output=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, seq_len, dim] (text + image concatenated)
            temb_mod: [B, 3*dim] pre-computed single-stream modulation
            image_rotary_emb: RoPE frequencies for the full sequence
        """
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # Modulation: 1 set of (shift, scale, gate)
        mods = NeuronFlux2Modulation.split(temb_mod, 1)
        shift, scale, gate = mods[0]

        # Norm + modulate
        norm_hidden = self.norm(hidden_states)
        norm_hidden = norm_hidden * (1 + scale[:, None]) + shift[:, None]

        # Parallel: Q/K/V projections + MLP input projection
        query = self.to_q(norm_hidden)
        key = self.to_k(norm_hidden)
        value = self.to_v(norm_hidden)

        query = query.view(
            batch_size, -1, self.heads_per_rank, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads_per_rank, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, -1, self.heads_per_rank, self.head_dim
        ).transpose(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # MLP branch (parallel with attention) - SwiGLU
        mlp_gate = F.silu(self.proj_mlp_gate(norm_hidden))
        mlp_value = self.proj_mlp_value(norm_hidden)
        mlp_hidden = mlp_gate * mlp_value

        # RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Attention
        if _HARDWARE == hardware.TRN1:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attn_output = attention_wrapper_sharded(query, key, value)

        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.heads_per_rank * self.head_dim
        )
        attn_output = attn_output.to(query.dtype)

        # Fused output projection: attention + MLP summed with single all-reduce
        out_attn = self.proj_out_attn(attn_output)
        out_mlp = self.proj_out_mlp(mlp_hidden)
        proj_out = reduce_from_tensor_model_parallel_region(
            out_attn + out_mlp,
            process_group=self.proj_out_attn.tensor_parallel_group,
        )

        hidden_states = gate[:, None] * proj_out
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================
# FLUX.2-klein Output Normalization
# ============================================================


class NeuronFlux2AdaLayerNormContinuous(nn.Module):
    """
    Continuous adaptive layer norm for the output layer.
    Produces scale/shift from the conditioning embedding (temb).
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_dim: int,
        eps: float = 1e-6,
        reduce_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=eps)
        self.linear = ColumnParallelLinear(
            conditioning_dim,
            embedding_dim * 2,
            bias=False,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.act(conditioning))
        scale, shift = emb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


# ============================================================
# FLUX.2-klein Transformer Model (Top-level)
# ============================================================


class NeuronFlux2KleinTransformer(nn.Module):
    """
    FLUX.2-klein-base-9B transformer backbone for NxD Inference.

    This model contains only the transformer blocks + output layers.
    Input embeddings (x_embedder, context_embedder), timestep embedding,
    modulation computation, and RoPE are all included in this compiled graph.

    Architecture:
    - 8 double-stream blocks (Flux2TransformerBlock)
    - 24 single-stream blocks (Flux2SingleTransformerBlock)
    - Pre-computed modulation (shared across all blocks of each type)
    - 4D RoPE (T, H, W, L)
    - No guidance embedding (Klein base)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        reduce_dtype = config.neuron_config.torch_dtype

        # Input projections (no bias in HF model)
        self.x_embedder = ColumnParallelLinear(
            config.in_channels,
            self.inner_dim,
            bias=False,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )
        self.context_embedder = ColumnParallelLinear(
            config.joint_attention_dim,
            self.inner_dim,
            bias=False,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )

        # Timestep embedding (no pooled text, no guidance for Klein base)
        self.time_guidance_embed = NeuronFlux2TimestepEmbedding(
            embedding_dim=self.inner_dim,
            reduce_dtype=reduce_dtype,
        )

        # Pre-computed modulation
        self.double_stream_modulation_img = NeuronFlux2Modulation(
            self.inner_dim,
            mod_param_sets=2,
            reduce_dtype=reduce_dtype,
        )
        self.double_stream_modulation_txt = NeuronFlux2Modulation(
            self.inner_dim,
            mod_param_sets=2,
            reduce_dtype=reduce_dtype,
        )
        self.single_stream_modulation = NeuronFlux2Modulation(
            self.inner_dim,
            mod_param_sets=1,
            reduce_dtype=reduce_dtype,
        )

        # Double-stream blocks (8 for Klein)
        self.transformer_blocks = nn.ModuleList(
            [
                NeuronFlux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    mlp_ratio=getattr(config, "mlp_ratio", 3.0),
                    reduce_dtype=reduce_dtype,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Single-stream blocks (24 for Klein)
        self.single_transformer_blocks = nn.ModuleList(
            [
                NeuronFlux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    mlp_ratio=getattr(config, "mlp_ratio", 3.0),
                    reduce_dtype=reduce_dtype,
                )
                for _ in range(config.num_single_layers)
            ]
        )

        # Output normalization and projection
        self.norm_out = NeuronFlux2AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            eps=1e-6,
            reduce_dtype=reduce_dtype,
        )
        self.proj_out = ColumnParallelLinear(
            self.inner_dim,
            config.patch_size * config.patch_size * config.out_channels,
            bias=False,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, img_seq, in_channels] packed latents
            encoder_hidden_states: [B, txt_seq, joint_attention_dim] text embeddings
            timestep: [B] diffusion timestep (0-1 range, will be scaled by 1000)
            image_rotary_emb: [total_seq, head_dim, 2] precomputed RoPE
        Returns:
            output: [B, img_seq, patch_size^2 * out_channels] denoised prediction
        """
        # Scale timestep
        timestep = timestep.to(self.config.neuron_config.torch_dtype) * 1000

        # Timestep embedding
        temb = self.time_guidance_embed(timestep, guidance=None)

        # Pre-compute all modulation parameters
        double_mod_img = self.double_stream_modulation_img(temb)
        double_mod_txt = self.double_stream_modulation_txt(temb)
        single_mod = self.single_stream_modulation(temb)

        # Input projections
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Double-stream blocks
        hidden_states, encoder_hidden_states = ModuleMarkerStartWrapper()(
            hidden_states, encoder_hidden_states
        )
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_mod_img,
                temb_mod_txt=double_mod_txt,
                image_rotary_emb=image_rotary_emb,
            )
        hidden_states, encoder_hidden_states = ModuleMarkerEndWrapper()(
            hidden_states, encoder_hidden_states
        )

        # Concatenate text + image for single-stream processing
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Single-stream blocks
        for idx, block in enumerate(self.single_transformer_blocks):
            if idx % 2 == 0:
                hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            hidden_states = block(
                hidden_states=hidden_states,
                temb_mod=single_mod,
                image_rotary_emb=image_rotary_emb,
            )
            if idx % 2 == 1:
                hidden_states = ModuleMarkerEndWrapper()(hidden_states)

        # Remove text tokens, keep only image tokens
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # Output normalization and projection
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        output = ModuleMarkerEndWrapper()(output)

        return output


# ============================================================
# NxDI Config and Application
# ============================================================


class Flux2KleinBackboneInferenceConfig(
    InferenceConfig if NEURON_AVAILABLE else object
):
    """Configuration for the FLUX.2-klein backbone."""

    def __init__(self, *args, **kwargs):
        if NEURON_AVAILABLE:
            super().__init__(*args, **kwargs)

    def get_required_attributes(self) -> List[str]:
        return [
            "attention_head_dim",
            "in_channels",
            "joint_attention_dim",
            "num_attention_heads",
            "num_layers",
            "num_single_layers",
            "patch_size",
            "out_channels",
            "height",
            "width",
        ]


class ModelWrapperFlux2KleinBackbone(ModelWrapper if NEURON_AVAILABLE else object):
    """Model wrapper for FLUX.2-klein backbone compilation."""

    def __init__(
        self,
        config,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        model_init_kwargs={},
    ):
        if NEURON_AVAILABLE:
            super().__init__(
                config,
                model_cls,
                tag,
                compiler_args,
                priority_model_idx,
                model_init_kwargs,
            )
        self.pos_embed = FluxPosEmbed(theta=2000, axes_dim=(32, 32, 32, 32))
        self.image_rotary_emb = None
        self.cache_image_rotary_emb = False

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """Generate example inputs for tracing."""
        in_channels = self.config.in_channels  # 128
        joint_attention_dim = self.config.joint_attention_dim  # 12288
        attention_head_dim = self.config.attention_head_dim  # 128
        # For FLUX.2-klein: latent is (B, H*W, 128) after pack
        # At 1024x1024: H=W=64 (1024/16), so img_seq = 4096
        vae_scale_factor = getattr(self.config, "vae_scale_factor", 16)
        num_patches = self.config.height * self.config.width // (vae_scale_factor**2)
        txt_seq_len = 512
        total_seq = num_patches + txt_seq_len

        dtype = self.config.neuron_config.torch_dtype

        model_inputs = (
            # hidden_states: [1, img_seq, in_channels]
            torch.randn([1, num_patches, in_channels], dtype=dtype),
            # encoder_hidden_states: [1, txt_seq, joint_attention_dim]
            torch.randn([1, txt_seq_len, joint_attention_dim], dtype=dtype),
            # timestep: [1]
            torch.randn([1], dtype=dtype),
            # image_rotary_emb: [total_seq, head_dim, 2]
            torch.randn([total_seq, attention_head_dim, 2], dtype=dtype),
        )

        return [model_inputs]

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        return_dict=False,
        **kwargs,
    ):
        """Override ModelWrapper.forward() to match Diffusers Flux2Transformer2DModel API."""
        if self.model is None:
            raise RuntimeError("Forward called before load. Call load() first.")

        if timestep is not None:
            timestep = timestep.to(self.config.neuron_config.torch_dtype)

        # Compute image_rotary_emb
        image_rotary_emb = self.image_rotary_emb
        if image_rotary_emb is None:
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]
            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = torch.stack(self.pos_embed(ids), dim=2).to(
                dtype=self.config.neuron_config.torch_dtype
            )

        if self.cache_image_rotary_emb:
            self.image_rotary_emb = image_rotary_emb

        output = self._forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            image_rotary_emb,
        )
        return output


class NeuronFlux2KleinBackboneApplication(
    NeuronApplicationBase if NEURON_AVAILABLE else object
):
    """NxDI Application for the FLUX.2-klein transformer backbone."""

    _model_cls = NeuronFlux2KleinTransformer

    def __init__(self, *args, **kwargs):
        if NEURON_AVAILABLE:
            super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperFlux2KleinBackbone

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    def get_model_wrapper_cls(self):
        return ModelWrapperFlux2KleinBackbone

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    @contextmanager
    def cache_context(self, name: str):
        """Cache context for matching Diffusers pipeline. Currently a no-op."""
        yield

    @contextmanager
    def image_rotary_emb_cache_context(self):
        self.model.cache_image_rotary_emb = True
        self.model.image_rotary_emb = None
        yield
        self.model.cache_image_rotary_emb = False
        self.model.image_rotary_emb = None

    def get_compiler_args(self):
        compiler_args = "--model-type=transformer -O1"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap'"
        compiler_args += (
            " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        if _HARDWARE == hardware.TRN2:
            os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        """
        Convert HuggingFace Flux2Transformer2DModel state dict to NxDI format.

        Key transformations:
        1. Single blocks: HF fused `attn.to_qkv_mlp_proj.weight` is split into
           to_q, to_k, to_v, proj_mlp_gate, proj_mlp_value.
        2. Single blocks: HF fused `attn.to_out.weight` is split into
           proj_out_attn and proj_out_mlp.
        3. Single blocks: `attn.norm_q/k` renamed to `norm_q/k`.
        4. Double blocks: HF `ff.linear_in.weight` is split into
           `ff.linear_in_gate.weight` and `ff.linear_in_value.weight`.
        5. Same for `ff_context.linear_in.weight`.

        All SwiGLU gate/value splits are needed so that TP sharding
        partitions gate and value weights together, preserving correctness.
        """
        inner_dim = config.num_attention_heads * config.attention_head_dim  # 4096
        mlp_hidden_dim = int(inner_dim * getattr(config, "mlp_ratio", 3.0))  # 12288

        new_state_dict = {}

        for key, value in state_dict.items():
            # ---- Single-stream block weight remapping ----
            if "single_transformer_blocks." in key:
                block_idx = key.split(".")[1]
                prefix = f"single_transformer_blocks.{block_idx}"

                if ".attn.to_qkv_mlp_proj.weight" in key:
                    # HF fused weight shape: [Q + K + V + MLP_gate + MLP_value, dim]
                    # = [4096 + 4096 + 4096 + 12288 + 12288, 4096] = [36864, 4096]
                    w = value
                    q_w = w[:inner_dim, :]
                    k_w = w[inner_dim : 2 * inner_dim, :]
                    v_w = w[2 * inner_dim : 3 * inner_dim, :]
                    mlp_gate_w = w[3 * inner_dim : 3 * inner_dim + mlp_hidden_dim, :]
                    mlp_value_w = w[3 * inner_dim + mlp_hidden_dim :, :]

                    new_state_dict[f"{prefix}.to_q.weight"] = q_w.clone()
                    new_state_dict[f"{prefix}.to_k.weight"] = k_w.clone()
                    new_state_dict[f"{prefix}.to_v.weight"] = v_w.clone()
                    new_state_dict[f"{prefix}.proj_mlp_gate.weight"] = (
                        mlp_gate_w.clone()
                    )
                    new_state_dict[f"{prefix}.proj_mlp_value.weight"] = (
                        mlp_value_w.clone()
                    )
                    continue

                if ".attn.to_out.weight" in key:
                    # HF fused output: [dim, attn_dim + mlp_dim]
                    # = [4096, 4096 + 12288] = [4096, 16384]
                    w = value
                    attn_w = w[:, :inner_dim]
                    mlp_w = w[:, inner_dim:]

                    new_state_dict[f"{prefix}.proj_out_attn.weight"] = attn_w.clone()
                    new_state_dict[f"{prefix}.proj_out_mlp.weight"] = mlp_w.clone()
                    continue

                # Rename QK norm from attn.norm_q/k to top-level norm_q/k
                if ".attn.norm_q.weight" in key:
                    new_state_dict[f"{prefix}.norm_q.weight"] = value.contiguous()
                    continue
                if ".attn.norm_k.weight" in key:
                    new_state_dict[f"{prefix}.norm_k.weight"] = value.contiguous()
                    continue

                # All other single block keys pass through

            # ---- Double-stream block: split SwiGLU linear_in ----
            if "transformer_blocks." in key:
                # ff.linear_in.weight [24576, 4096] -> gate [12288, 4096] + value [12288, 4096]
                if ".ff.linear_in.weight" in key:
                    block_prefix = key.rsplit(".ff.linear_in.weight", 1)[0]
                    w = value
                    gate_w = w[:mlp_hidden_dim, :]
                    value_w = w[mlp_hidden_dim:, :]
                    new_state_dict[f"{block_prefix}.ff.linear_in_gate.weight"] = (
                        gate_w.clone()
                    )
                    new_state_dict[f"{block_prefix}.ff.linear_in_value.weight"] = (
                        value_w.clone()
                    )
                    continue

                if ".ff_context.linear_in.weight" in key:
                    block_prefix = key.rsplit(".ff_context.linear_in.weight", 1)[0]
                    w = value
                    gate_w = w[:mlp_hidden_dim, :]
                    value_w = w[mlp_hidden_dim:, :]
                    new_state_dict[
                        f"{block_prefix}.ff_context.linear_in_gate.weight"
                    ] = gate_w.clone()
                    new_state_dict[
                        f"{block_prefix}.ff_context.linear_in_value.weight"
                    ] = value_w.clone()
                    continue

            # ---- All other keys pass through unchanged ----
            new_state_dict[key] = value.contiguous()

        return new_state_dict
