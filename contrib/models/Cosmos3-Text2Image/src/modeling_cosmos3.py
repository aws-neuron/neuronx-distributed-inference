# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cosmos3-Nano MoT (Mixture-of-Transformers) generation backbone for NxDI.
# Adapted from NxDI Flux implementation with Cosmos3-specific MoT dual-stream
# attention, separate MLPs, additive timestep conditioning, and GQA.

import logging
import math
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
    scatter_to_process_group_spmd,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group,
    get_tensor_model_parallel_size,
    get_world_group,
)

from neuronx_distributed_inference.models.diffusers.embeddings import (
    NeuronTimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_dp_rank_spmd

from nkilib.core.attention.attention_cte import attention_cte

from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.models.model_wrapper import (
    BaseModelInstance,
    ModelWrapper,
)

_HARDWARE = hardware(get_platform_target())

if not os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"):
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = get_platform_target()

logger = logging.getLogger(__name__)


# =============================================================================
# Attention Kernel Wrapper (bidirectional, from Flux)
# =============================================================================


def attention_wrapper_bidirectional(query, key, value):
    """
    Bidirectional attention using NKI attention_cte kernel.

    Input shapes: query, key, value all have shape [bs, n_head, seq_len, d_head]
    Output shape: [bs, n_head, q_len, d_head]

    Uses tp_q=True, tp_k=True to let the kernel handle transposes internally.
    """
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]

    q = query.reshape((bs * n_head, q_len, d_head))
    k = key.reshape((bs * n_head, k_len, d_head))
    v = value.reshape((bs * n_head, k_len, d_head))

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    use_sharded_attention_kernel = vc_size == 2
    scale = 1 / math.sqrt(d_head)

    if use_sharded_attention_kernel:
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

    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output


def attention_wrapper_causal(query, key, value):
    """
    Causal attention using NKI attention_cte kernel.

    Same as bidirectional but with causal_mask=True.
    Used for the text (understanding) pathway.
    """
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]

    q = query.reshape((bs * n_head, q_len, d_head))
    k = key.reshape((bs * n_head, k_len, d_head))
    v = value.reshape((bs * n_head, k_len, d_head))

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    use_sharded_attention_kernel = vc_size == 2
    scale = 1 / math.sqrt(d_head)

    if use_sharded_attention_kernel:
        attn_output = attention_cte[2](
            q,
            k,
            v,
            scale,
            causal_mask=True,
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
            causal_mask=True,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )

    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output


# =============================================================================
# M-RoPE (Multimodal Rotary Position Embedding)
# Matches Cosmos3VLTextRotaryEmbedding from diffusers reference implementation.
# =============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard half-rotation for RoPE: split last dim in half, negate+swap."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Cosmos3MRoPE(nn.Module):
    """
    M-RoPE for Cosmos3 matching Cosmos3VLTextRotaryEmbedding.

    Key differences from the previous (incorrect) implementation:
    1. Uses a SINGLE inv_freq basis computed over full head_dim (not per-axis)
    2. Applies interleaved M-RoPE mixing (T/H/W frequencies interleaved)
    3. Returns (cos, sin) each of shape [seq_len, head_dim]

    Position IDs: [seq_len, 3] -> (t, h, w) per token.
    """

    def __init__(
        self, head_dim: int, mrope_section: List[int], rope_theta: float = 5000000.0
    ):
        super().__init__()
        self.head_dim = head_dim
        self.mrope_section = mrope_section  # [24, 20, 20] for Cosmos3
        self.rope_theta = rope_theta
        # Single inv_freq basis over full head_dim
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Reorganize chunked [TTT...HHH...WWW] frequency layout into interleaved
        [THTHWHTHW...TT], preserving frequency continuity across the 3 grids.

        Uses scatter-based approach for XLA compatibility (no in-place mutation).

        Args:
            freqs: [3, seq_len, head_dim//2] - per-axis frequencies

        Returns:
            [seq_len, head_dim//2] - interleaved frequencies
        """
        # Build output by selecting from T, H, W based on interleaving pattern.
        # For mrope_section=[24, 20, 20], head_dim//2=64:
        #   positions 0,3,6,...,69 (step 3, 24 positions) -> T (freqs[0])
        #   positions 1,4,7,...,58 (step 3, 20 positions) -> H (freqs[1])
        #   positions 2,5,8,...,59 (step 3, 20 positions) -> W (freqs[2])
        #   Remaining positions use T (freqs[0])
        half_dim = freqs.shape[-1]
        seq_len = freqs.shape[1]

        # Build source index on CPU (constant, will be baked into XLA graph)
        # This determines which axis (0=T, 1=H, 2=W) each position draws from
        source_cpu = torch.zeros(half_dim, dtype=torch.long)
        for dim in range(1, 3):  # H=1, W=2
            length = self.mrope_section[dim] * 3
            for i in range(dim, length, 3):
                source_cpu[i] = dim

        # Move to device and expand for gather
        source = source_cpu.to(device=freqs.device)
        source_expanded = source.unsqueeze(0).expand(seq_len, -1)  # [seq_len, hd//2]

        # freqs: [3, seq_len, half_dim] -> permute to [seq_len, half_dim, 3]
        freqs_perm = freqs.permute(1, 2, 0)  # [seq_len, half_dim, 3]

        # Gather along last dim using source indices
        result = freqs_perm.gather(2, source_expanded.unsqueeze(-1)).squeeze(
            -1
        )  # [seq_len, half_dim]

        return result

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: [seq_len, 3] - (t, h, w) positions per token

        Returns:
            (cos, sin): each [seq_len, head_dim]
        """
        # position_ids: [seq_len, 3] -> transpose to [3, seq_len]
        pos = position_ids.T.float()  # [3, seq_len]

        # Move inv_freq to input device (critical for XLA tracing)
        inv_freq = self.inv_freq.to(device=position_ids.device)

        # inv_freq: [head_dim//2] -> expand to [3, head_dim//2, 1]
        inv_freq_expanded = inv_freq[None, :, None].expand(
            3, -1, 1
        )  # [3, head_dim//2, 1]
        pos_expanded = pos[:, None, :]  # [3, 1, seq_len]

        # freqs: [3, seq_len, head_dim//2]
        freqs = (inv_freq_expanded @ pos_expanded).transpose(
            1, 2
        )  # [3, seq_len, hd//2]

        # Apply interleaved M-RoPE mixing
        freqs_mixed = self._apply_interleaved_mrope(freqs)  # [seq_len, head_dim//2]

        # Expand to full head_dim by doubling (for rotate_half compatibility)
        emb = torch.cat((freqs_mixed, freqs_mixed), dim=-1)  # [seq_len, head_dim]

        return emb.cos(), emb.sin()


def apply_rotary_emb_cosmos3(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply M-RoPE to query or key tensor using rotate_half convention.

    Args:
        x: [B, H, S, D] - query or key
        cos: [S, D] - cosine component
        sin: [S, D] - sine component

    Returns:
        [B, H, S, D] - rotated tensor
    """
    # Broadcast cos/sin: [S, D] -> [1, 1, S, D]
    cos = cos[None, None]
    sin = sin[None, None]

    # Standard rotate_half application (matches reference _rotate_half)
    out = (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)
    return out


# =============================================================================
# NeuronCosmos3Attention (Joint MMDiT with GQA)
# =============================================================================


class NeuronCosmos3Attention(nn.Module):
    """
    Joint attention for Cosmos3 MoT with:
    - Separate Q/K/V projections per stream (text: to_q/to_k/to_v, gen: add_q/k/v_proj)
    - GQA (32 Q heads, 8 KV heads)
    - QK normalization (per-head RMSNorm)
    - Separate output projections (to_out, to_add_out)
    - Bidirectional attention (no causal mask)
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        tp_degree = get_tensor_model_parallel_size()

        # Pad heads for TP alignment
        self.padded_num_heads = math.ceil(num_attention_heads / tp_degree) * tp_degree
        self.padded_num_kv_heads = (
            math.ceil(num_key_value_heads / tp_degree) * tp_degree
        )
        self.padded_q_dim = self.padded_num_heads * head_dim
        self.padded_kv_dim = self.padded_num_kv_heads * head_dim

        # Per-TP-rank head counts
        self.heads_per_rank = self.padded_num_heads // tp_degree
        self.kv_heads_per_rank = self.padded_num_kv_heads // tp_degree

        # --- Text stream projections ---
        self.to_q = ColumnParallelLinear(
            hidden_size,
            self.padded_q_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_k = ColumnParallelLinear(
            hidden_size,
            self.padded_kv_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_v = ColumnParallelLinear(
            hidden_size,
            self.padded_kv_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_out = RowParallelLinear(
            self.padded_q_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
        )

        # --- Generation stream projections ---
        self.add_q_proj = ColumnParallelLinear(
            hidden_size,
            self.padded_q_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.add_k_proj = ColumnParallelLinear(
            hidden_size,
            self.padded_kv_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.add_v_proj = ColumnParallelLinear(
            hidden_size,
            self.padded_kv_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_add_out = RowParallelLinear(
            self.padded_q_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
        )

        # --- QK Normalization (per-head RMSNorm) ---
        self.norm_q = CustomRMSNorm(head_dim, eps=1e-6)
        self.norm_k = CustomRMSNorm(head_dim, eps=1e-6)
        self.norm_added_q = CustomRMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = CustomRMSNorm(head_dim, eps=1e-6)

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match Q heads for GQA. [B, kv_heads, S, D] -> [B, q_heads, S, D]"""
        if self.num_kv_groups == 1:
            return kv
        bs, n_kv_heads, seq_len, head_dim = kv.shape
        kv = kv[:, :, None, :, :].expand(
            bs, n_kv_heads, self.num_kv_groups, seq_len, head_dim
        )
        return kv.reshape(bs, n_kv_heads * self.num_kv_groups, seq_len, head_dim)

    def forward(
        self,
        text_hidden: torch.Tensor,
        gen_hidden: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dual-pathway attention matching reference Cosmos3AttnProcessor:
        - Text (understanding): CAUSAL self-attention over text tokens only
        - Generation: BIDIRECTIONAL attention over ALL tokens (text + gen)

        Args:
            text_hidden: [B, T_text, hidden_size] - normed text hidden states
            gen_hidden: [B, T_gen, hidden_size] - normed gen hidden states
            rotary_emb: (cos_und, sin_und, cos_gen, sin_gen)
                        cos/sin_und: [T_text, head_dim]
                        cos/sin_gen: [T_gen, head_dim]

        Returns:
            text_attn_out: [B, T_text, hidden_size]
            gen_attn_out: [B, T_gen, hidden_size]
        """
        batch_size = text_hidden.shape[0]
        text_len = text_hidden.shape[1]
        gen_len = gen_hidden.shape[1]

        cos_und, sin_und, cos_gen, sin_gen = rotary_emb

        # --- Project text stream ---
        text_q = self.to_q(text_hidden)  # [B, T_text, padded_q_dim/tp]
        text_k = self.to_k(text_hidden)  # [B, T_text, padded_kv_dim/tp]
        text_v = self.to_v(text_hidden)  # [B, T_text, padded_kv_dim/tp]

        # --- Project gen stream ---
        gen_q = self.add_q_proj(gen_hidden)
        gen_k = self.add_k_proj(gen_hidden)
        gen_v = self.add_v_proj(gen_hidden)

        # --- Reshape to [B, heads, S, head_dim] ---
        text_q = text_q.view(
            batch_size, text_len, self.heads_per_rank, self.head_dim
        ).transpose(1, 2)
        text_k = text_k.view(
            batch_size, text_len, self.kv_heads_per_rank, self.head_dim
        ).transpose(1, 2)
        text_v = text_v.view(
            batch_size, text_len, self.kv_heads_per_rank, self.head_dim
        ).transpose(1, 2)

        gen_q = gen_q.view(
            batch_size, gen_len, self.heads_per_rank, self.head_dim
        ).transpose(1, 2)
        gen_k = gen_k.view(
            batch_size, gen_len, self.kv_heads_per_rank, self.head_dim
        ).transpose(1, 2)
        gen_v = gen_v.view(
            batch_size, gen_len, self.kv_heads_per_rank, self.head_dim
        ).transpose(1, 2)

        # --- QK Normalization ---
        text_q = self.norm_q(text_q)
        text_k = self.norm_k(text_k)
        gen_q = self.norm_added_q(gen_q)
        gen_k = self.norm_added_k(gen_k)

        # --- Apply M-RoPE (separate per pathway) ---
        text_q = apply_rotary_emb_cosmos3(text_q, cos_und, sin_und)
        text_k = apply_rotary_emb_cosmos3(text_k, cos_und, sin_und)
        gen_q = apply_rotary_emb_cosmos3(gen_q, cos_gen, sin_gen)
        gen_k = apply_rotary_emb_cosmos3(gen_k, cos_gen, sin_gen)

        # --- GQA: expand KV heads to match Q heads ---
        text_k = self._repeat_kv(text_k)
        text_v = self._repeat_kv(text_v)
        gen_k = self._repeat_kv(gen_k)
        gen_v = self._repeat_kv(gen_v)

        # --- Text pathway: CAUSAL self-attention (text only) ---
        if _HARDWARE == hardware.TRN1:
            text_attn_output = F.scaled_dot_product_attention(
                text_q, text_k, text_v, dropout_p=0.0, is_causal=True
            )
        else:
            text_attn_output = attention_wrapper_causal(text_q, text_k, text_v)

        # --- Generation pathway: BIDIRECTIONAL attention to ALL tokens ---
        all_k = torch.cat([text_k, gen_k], dim=2)  # [B, heads, T_text+T_gen, D]
        all_v = torch.cat([text_v, gen_v], dim=2)

        if _HARDWARE == hardware.TRN1:
            gen_attn_output = F.scaled_dot_product_attention(
                gen_q, all_k, all_v, dropout_p=0.0, is_causal=False
            )
        else:
            gen_attn_output = attention_wrapper_bidirectional(gen_q, all_k, all_v)

        # --- Reshape outputs ---
        text_attn_output = text_attn_output.transpose(1, 2).reshape(
            batch_size, text_len, self.heads_per_rank * self.head_dim
        )
        gen_attn_output = gen_attn_output.transpose(1, 2).reshape(
            batch_size, gen_len, self.heads_per_rank * self.head_dim
        )

        # --- Separate output projections ---
        text_out = self.to_out(text_attn_output)
        gen_out = self.to_add_out(gen_attn_output)

        return text_out, gen_out


# =============================================================================
# SwiGLU MLP (used for both text and gen MLPs)
# =============================================================================


class NeuronCosmos3SwiGLU(nn.Module):
    """
    SwiGLU MLP: gate_proj (up), up_proj (gate), down_proj.
    Same as Llama/Qwen MLP structure.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# NeuronCosmos3MoTBlock (single MoT layer)
# =============================================================================


class NeuronCosmos3MoTBlock(nn.Module):
    """
    Single Cosmos3 MoT layer with:
    - Separate pre-attention LayerNorms per stream
    - Joint attention (MMDiT-style)
    - Separate post-attention LayerNorms per stream
    - Separate SwiGLU MLPs per stream (text: mlp, gen: mlp_moe_gen)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # --- Pre-attention norms (separate per stream) ---
        self.input_layernorm = CustomRMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm_moe_gen = CustomRMSNorm(hidden_size, eps=rms_norm_eps)

        # --- Joint attention ---
        self.self_attn = NeuronCosmos3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            reduce_dtype=reduce_dtype,
        )

        # --- Post-attention norms (separate per stream) ---
        self.post_attention_layernorm = CustomRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm_moe_gen = CustomRMSNorm(
            hidden_size, eps=rms_norm_eps
        )

        # --- Separate MLPs ---
        self.mlp = NeuronCosmos3SwiGLU(hidden_size, intermediate_size, reduce_dtype)
        self.mlp_moe_gen = NeuronCosmos3SwiGLU(
            hidden_size, intermediate_size, reduce_dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_len: int,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T_text + T_gen, hidden_size] - packed sequence
            text_len: number of text tokens (split point)
            rotary_emb: (cos_und, sin_und, cos_gen, sin_gen)
                        cos/sin_und: [T_text, head_dim]
                        cos/sin_gen: [T_gen, head_dim]

        Returns:
            [B, T_text + T_gen, hidden_size] - updated packed sequence
        """
        # --- Split streams ---
        text_hidden = hidden_states[:, :text_len, :]
        gen_hidden = hidden_states[:, text_len:, :]

        # --- Pre-norm ---
        text_normed = self.input_layernorm(text_hidden)
        gen_normed = self.input_layernorm_moe_gen(gen_hidden)

        # --- Joint attention ---
        text_attn_out, gen_attn_out = self.self_attn(
            text_normed, gen_normed, rotary_emb
        )

        # --- Residual ---
        text_hidden = text_hidden + text_attn_out
        gen_hidden = gen_hidden + gen_attn_out

        # --- Post-norm + separate MLPs ---
        text_hidden = text_hidden + self.mlp(self.post_attention_layernorm(text_hidden))
        gen_hidden = gen_hidden + self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(gen_hidden)
        )

        # --- Re-pack ---
        return torch.cat([text_hidden, gen_hidden], dim=1)


# =============================================================================
# NeuronCosmos3Transformer (full backbone)
# =============================================================================


class NeuronCosmos3Transformer(nn.Module):
    """
    Full Cosmos3-Nano MoT transformer backbone for generation.

    Architecture:
    - embed_tokens: shared text embedding
    - vae2llm (proj_in): project VAE latent patches to hidden_size
    - time_embedder: sinusoidal timestep -> MLP -> hidden_size (additive)
    - 36 MoT layers (NeuronCosmos3MoTBlock)
    - norm: final RMSNorm
    - llm2vae (proj_out): project hidden_size -> patch channels (velocity output)

    Forward signature:
        text_ids, vision_patches, timestep, position_ids -> velocity_prediction
    """

    def __init__(self, config: "Cosmos3BackboneInferenceConfig"):
        super().__init__()
        self.config = config

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_layers = config.num_hidden_layers
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim
        rms_norm_eps = config.rms_norm_eps
        vocab_size = config.vocab_size
        patch_channels = config.patch_channels  # 64 (16 channels * 2x2 spatial patch)

        reduce_dtype = config.neuron_config.torch_dtype

        self.data_parallel_group = get_data_parallel_group()
        self.global_rank = SPMDRank(world_size=get_world_group().size())
        self.cfg_parallel_enabled = getattr(config, "cfg_parallel_enabled", False)

        # --- Token embedding (text) ---
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # --- Patch projection (VAE latent -> hidden) ---
        # proj_in in the HF model: [patch_channels, hidden_size] with bias
        self.proj_in = ColumnParallelLinear(
            patch_channels,
            hidden_size,
            bias=True,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )

        # --- Timestep embedding (sinusoidal + MLP, additive) ---
        # Cosmos3 time_embedder: Timesteps(256) -> MLP(256 -> hidden_size)
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = NeuronTimestepEmbedding(
            in_channels=256,
            time_embed_dim=hidden_size,
            reduce_dtype=reduce_dtype,
        )

        # --- MoT Transformer layers ---
        self.layers = nn.ModuleList(
            [
                NeuronCosmos3MoTBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    rms_norm_eps=rms_norm_eps,
                    reduce_dtype=reduce_dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # --- Output projection ---
        # norm_moe_gen: final RMSNorm for generation stream output
        # (norm is for text stream / lm_head, not used here)
        self.norm_moe_gen = CustomRMSNorm(hidden_size, eps=rms_norm_eps)
        # proj_out: [hidden_size, patch_channels] with bias
        # Use ColumnParallelLinear with gather to shard across TP
        # Weight shape: [patch_channels, hidden_size] = [192, 4096]
        # Each rank: [48, 4096] input, gathered to [192] output
        self.proj_out = ColumnParallelLinear(
            hidden_size,
            patch_channels,
            bias=True,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        vision_patches: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_ids: [B, T_text] - tokenized text (int64)
            vision_patches: [B, T_gen, patch_channels] - patchified noisy latents
            timestep: [B] - diffusion timestep (float, 0 to 1)
            position_ids: [T_text + T_gen, 3] - M-RoPE positions (t, h, w)

        Returns:
            velocity: [B, T_gen, patch_channels] - predicted velocity
        """
        batch_size = text_ids.shape[0]
        text_len = text_ids.shape[1]
        gen_len = vision_patches.shape[1]

        # --- CFG Parallel: scatter inputs ---
        if self.cfg_parallel_enabled and batch_size == 2:
            dp_rank = get_dp_rank_spmd(
                global_rank=self.global_rank.get_rank(),
                tp_degree=get_tensor_model_parallel_size(),
            )
            text_ids = scatter_to_process_group_spmd(
                text_ids,
                partition_dim=0,
                rank=dp_rank,
                process_group=self.data_parallel_group,
            )
            vision_patches = scatter_to_process_group_spmd(
                vision_patches,
                partition_dim=0,
                rank=dp_rank,
                process_group=self.data_parallel_group,
            )
            timestep = scatter_to_process_group_spmd(
                timestep,
                partition_dim=0,
                rank=dp_rank,
                process_group=self.data_parallel_group,
            )
            batch_size = 1

        # --- 1. Embed text ---
        text_embeds = self.embed_tokens(text_ids)  # [B, T_text, hidden_size]

        # --- 2. Project vision patches ---
        vision_embeds = self.proj_in(vision_patches)  # [B, T_gen, hidden_size]

        # --- 3. Additive timestep conditioning on vision tokens ---
        timestep_proj = self.time_proj(timestep)  # [B, 256]
        t_emb = self.time_embedder(
            timestep_proj.to(vision_embeds.dtype)
        )  # [B, hidden_size]
        vision_embeds = vision_embeds + t_emb.unsqueeze(1)  # broadcast add

        # --- 4. Pack into single sequence ---
        hidden_states = torch.cat(
            [text_embeds, vision_embeds], dim=1
        )  # [B, T_text+T_gen, H]

        # --- 5. Compute M-RoPE and split into text/gen portions ---
        mrope = Cosmos3MRoPE(
            head_dim=self.config.head_dim,
            mrope_section=self.config.mrope_section,
            rope_theta=self.config.rope_theta,
        )
        # Returns (cos, sin) each [T_text + T_gen, head_dim]
        cos_full, sin_full = mrope(position_ids)
        cos_full = cos_full.to(
            dtype=self.config.neuron_config.torch_dtype, device=hidden_states.device
        )
        sin_full = sin_full.to(
            dtype=self.config.neuron_config.torch_dtype, device=hidden_states.device
        )
        # Split into text (understanding) and gen portions
        cos_und = cos_full[:text_len]  # [T_text, head_dim]
        sin_und = sin_full[:text_len]  # [T_text, head_dim]
        cos_gen = cos_full[text_len:]  # [T_gen, head_dim]
        sin_gen = sin_full[text_len:]  # [T_gen, head_dim]
        rotary_emb = (cos_und, sin_und, cos_gen, sin_gen)

        # --- 6. Run through MoT layers ---
        hidden_states, _ = ModuleMarkerStartWrapper()(hidden_states, hidden_states)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, text_len, rotary_emb)
            # Layer boundary markers for compiler optimization (every 2 layers)
            if i % 2 == 1 and i < len(self.layers) - 1:
                hidden_states = ModuleMarkerEndWrapper()(hidden_states)
                hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)

        # --- 7. Extract vision portion and project to velocity ---
        vision_output = hidden_states[:, text_len:, :]  # [B, T_gen, hidden_size]
        vision_output = self.norm_moe_gen(vision_output)
        velocity = self.proj_out(vision_output)  # [B, T_gen, patch_channels]

        # --- CFG Parallel: gather outputs ---
        if self.cfg_parallel_enabled:
            velocity = gather_from_tensor_model_parallel_region_with_dim(
                velocity,
                gather_dim=0,
                process_group=self.data_parallel_group,
            )

        return velocity


# =============================================================================
# Config
# =============================================================================


class Cosmos3BackboneInferenceConfig(InferenceConfig):
    """Config for the Cosmos3 generation backbone."""

    def __init__(self, *args, cfg_parallel_enabled: bool = False, **kwargs):
        # Set Cosmos3-Nano defaults BEFORE super().__init__ (which calls validate_config)
        self.hidden_size = kwargs.pop("hidden_size", 4096)
        self.intermediate_size = kwargs.pop("intermediate_size", 12288)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 36)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", 8)
        self.head_dim = kwargs.pop("head_dim", 128)
        self.rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-6)
        self.vocab_size = kwargs.pop("vocab_size", 151936)
        self.patch_channels = kwargs.pop(
            "patch_channels", 192
        )  # 48 latent_ch * 2*2 patch
        self.latent_channels = kwargs.pop("latent_channels", 48)
        self.rope_theta = kwargs.pop("rope_theta", 5000000.0)
        self.mrope_section = kwargs.pop("mrope_section", [24, 20, 20])
        self.cfg_parallel_enabled = cfg_parallel_enabled

        super().__init__(*args, **kwargs)

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "vocab_size",
            "patch_channels",
            "rope_theta",
            "mrope_section",
        ]


# =============================================================================
# Model Wrapper
# =============================================================================


class ModelWrapperCosmos3Backbone(ModelWrapper):
    """Wrapper for Cosmos3 backbone: handles input generation and forward dispatch."""

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        model_init_kwargs=None,
    ):
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            model_init_kwargs or {},
        )
        # For large models (Super, 64 layers), the NxDI framework appends
        # --verify-hlo=true which fails before partitioning for models > 24 GB/rank.
        # Replace verify-hlo=true with verify-hlo=false to skip the pre-partition check.
        if config.num_hidden_layers > 36:
            self.compiler_args = self.compiler_args.replace(
                "--verify-hlo=true", "--verify-hlo=false"
            )
            logger.info(
                f"Large model: disabled verify-hlo (compiler_args: {self.compiler_args})"
            )

        self.mrope = Cosmos3MRoPE(
            head_dim=config.head_dim,
            mrope_section=config.mrope_section,
            rope_theta=config.rope_theta,
        )

    def input_generator(self) -> List[Tuple[torch.Tensor, ...]]:
        """Generate example inputs for compilation."""
        dtype = self.config.neuron_config.torch_dtype
        text_len = self.config.max_text_len
        gen_len = self.config.num_vision_patches
        patch_channels = self.config.patch_channels

        batch_size = 2 if self.config.cfg_parallel_enabled else 1

        model_inputs = (
            # text_ids: [B, T_text]
            torch.zeros([batch_size, text_len], dtype=torch.long),
            # vision_patches: [B, T_gen, patch_channels]
            torch.randn([batch_size, gen_len, patch_channels], dtype=dtype),
            # timestep: [B]
            torch.randn([batch_size], dtype=dtype),
            # position_ids: [T_text + T_gen, 3]
            torch.zeros([text_len + gen_len, 3], dtype=torch.long),
        )
        return [model_inputs]

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        model_instance = BaseModelInstance(
            module_cls=_create_model, input_output_aliases={}
        )
        return model_instance

    def forward(self, text_ids, vision_patches, timestep, position_ids):
        """Override ModelWrapper.forward()."""
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")

        timestep = timestep.to(self.config.neuron_config.torch_dtype)

        output = self._forward(text_ids, vision_patches, timestep, position_ids)
        return output


# =============================================================================
# Application (compile/load infrastructure)
# =============================================================================


class NeuronCosmos3BackboneApplication(NeuronApplicationBase):
    """
    Application class for the Cosmos3 MoT backbone.
    Handles compilation, weight loading, and forward dispatch.
    """

    _model_cls = NeuronCosmos3Transformer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperCosmos3Backbone

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self):
        compiler_args = "--model-type=transformer -O1"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap'"
        compiler_args += " --auto-cast=none"
        # For large models (Super, 64 layers): force low MAC threshold so modular flow
        # always partitions the graph. Without this, the compiler may fail HBM verification
        # before partitioning kicks in.
        if self.config.num_hidden_layers > 36:
            compiler_args += (
                " --internal-hlo2tensorizer-options="
                "'--modular-flow-mac-threshold=10 --recursive-layer-det=false'"
            )

        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        if _HARDWARE == hardware.TRN2:
            os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """
        Convert HF Diffusers Cosmos3 state_dict to NxDI format.

        HF Diffusers keys (from transformer/ directory):
            layers.N.self_attn.to_q.weight
            layers.N.self_attn.to_k.weight
            layers.N.self_attn.to_v.weight
            layers.N.self_attn.to_out.weight
            layers.N.self_attn.norm_q.weight
            layers.N.self_attn.norm_k.weight
            layers.N.self_attn.add_q_proj.weight
            layers.N.self_attn.add_k_proj.weight
            layers.N.self_attn.add_v_proj.weight
            layers.N.self_attn.to_add_out.weight
            layers.N.self_attn.norm_added_q.weight
            layers.N.self_attn.norm_added_k.weight
            layers.N.input_layernorm.weight
            layers.N.input_layernorm_moe_gen.weight
            layers.N.post_attention_layernorm.weight
            layers.N.post_attention_layernorm_moe_gen.weight
            layers.N.mlp.gate_proj.weight
            layers.N.mlp.up_proj.weight
            layers.N.mlp.down_proj.weight
            layers.N.mlp_moe_gen.gate_proj.weight
            layers.N.mlp_moe_gen.up_proj.weight
            layers.N.mlp_moe_gen.down_proj.weight
            embed_tokens.weight
            proj_in.weight / proj_in.bias
            proj_out.weight / proj_out.bias
            time_embedder.linear_1.weight / bias
            time_embedder.linear_2.weight / bias

        NxDI keys (this model):
            Same structure -- we keep the HF naming since our module names match.
            Only need to:
            1. Map time_embedder.linear_1/2 -> time_embedder.linear_1/2
            2. Add global_rank tensor
            3. Ensure contiguous tensors
        """
        new_state_dict = {}

        # Key mapping from HF Diffusers to NxDI module names
        # Most keys map directly since our module structure mirrors HF
        key_mapping = {
            # Timestep embedder
            "time_embedder.linear_1.weight": "time_embedder.linear_1.weight",
            "time_embedder.linear_1.bias": "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight": "time_embedder.linear_2.weight",
            "time_embedder.linear_2.bias": "time_embedder.linear_2.bias",
        }

        # Keys to skip (not used in generation backbone)
        skip_prefixes = [
            "lm_head.",  # text generation head (reasoning only)
            "audio_",  # audio modality
            "action_",  # action modality
            "sound_",  # sound tokenizer
            "visual.",  # vision encoder (separate NEFF if needed)
        ]

        for key, value in state_dict.items():
            # Skip non-generation keys
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                continue

            # Apply key mapping if exists
            new_key = key_mapping.get(key, key)
            new_state_dict[new_key] = value.clone().detach().contiguous()

        # Add global rank tensor (required by NxDI parallel layers)
        new_state_dict["global_rank.rank"] = torch.arange(
            0, config.neuron_config.world_size, dtype=torch.int32
        )

        return new_state_dict
