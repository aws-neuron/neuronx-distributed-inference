"""
NeuronGemma3TextEncoder — Gemma 3-12B as an encoder-only model for LTX-2.3 text conditioning
==============================================================================================

LTX-2.3 uses Gemma 3-12B as a text encoder, NOT as a causal language model. The pipeline
calls the text encoder with output_hidden_states=True and collects ALL 49 hidden states
(embedding + 48 decoder layers), stacks them, normalizes, and flattens to produce
conditioning embeddings for the video and audio diffusion streams.

NxDI's built-in NeuronGemma3ForCausalLM cannot provide this because:
  - It only returns logits/tokens, not intermediate hidden states
  - It has KV cache machinery baked into the compiled graph
  - Its forward slices to the last token position

This module builds a CUSTOM encoder-only model that:
  1. Uses NxD parallel layers for TP-sharded attention, MLP, norms
  2. Runs all 48 decoder layers in a single forward pass
  3. Accumulates all hidden states and returns torch.stack(all_hidden_states, dim=-1)
  4. Has NO KV cache, NO lm_head, NO sampling
  5. Takes (input_ids, attention_mask) -> (B, seq_len, hidden_size, num_layers+1)

Architecture:
  Gemma3ScaledEmbedding -> 48 x Gemma3EncoderLayer -> Gemma3RMSNorm
  Output: torch.stack([embed_out, layer_0_out, ..., layer_47_out], dim=-1)

TP strategy:
  - Q, K, V projections: ColumnParallelLinear (shard output dim)
  - O projection: RowParallelLinear (shard input dim, all-reduce)
  - gate_proj, up_proj: ColumnParallelLinear
  - down_proj: RowParallelLinear
  - Norms: replicated (not sharded) -- they operate on the full hidden_size
  - Embedding: ParallelEmbedding (sharded across vocab)

Adapted from the LTX-2 contrib (contrib/ltx2-video-audio) for LTX-2.3.
"""

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)

logger = logging.getLogger(__name__)


# ── RMSNorm (Gemma3 variant: 1 + weight) ────────────────────────────────────


class Gemma3RMSNorm(nn.Module):
    """Gemma3-specific RMSNorm: uses (1.0 + weight) instead of just weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


# ── Scaled Embedding ────────────────────────────────────────────────────────


class Gemma3ScaledEmbedding(nn.Module):
    """Gemma3 embeddings scaled by sqrt(hidden_size)."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx, dtype):
        super().__init__()
        self.embed_scale = embedding_dim**0.5
        self.embedding = ParallelEmbedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            dtype=dtype,
            shard_across_embedding=True,
            pad=True,
        )

    def forward(self, input_ids):
        return self.embedding(input_ids) * self.embed_scale


# ── Rotary Position Embedding ───────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """Standard RoPE for Gemma3 (no sliding window variant needed for encoder)."""

    def __init__(self, dim, max_position_embeddings=131072, base=1_000_000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        """
        Args:
            position_ids: (batch_size, seq_len)
        Returns:
            cos, sin: (batch_size, seq_len, dim)
        """
        inv_freq = self.inv_freq.to(position_ids.device)
        pos = position_ids.unsqueeze(-1).float()
        freqs = pos * inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embedding to query and key tensors."""
    cos = cos.unsqueeze(1)  # (B, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ── Attention ───────────────────────────────────────────────────────────────


class Gemma3EncoderAttention(nn.Module):
    """Gemma3 attention for encoder-only use (no KV cache).

    Uses GQA with Q-K normalization. Keeps causal attention to match
    training behavior of the original Gemma3 weights.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_position_embeddings: int,
        query_pre_attn_scalar: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        # Scaling uses query_pre_attn_scalar, not head_dim
        self.scale = query_pre_attn_scalar**-0.5

        tp_size = get_tensor_model_parallel_size()

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            num_attention_heads * head_dim,
            bias=False,
            gather_output=False,
            dtype=dtype,
            pad=True,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=False,
            gather_output=False,
            dtype=dtype,
            pad=True,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=False,
            gather_output=False,
            dtype=dtype,
            pad=True,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )

        # Q-K normalization (Gemma3-specific)
        self.q_layernorm = Gemma3RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_layernorm = Gemma3RMSNorm(head_dim, eps=rms_norm_eps)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        self.num_heads_per_rank = num_attention_heads // tp_size
        self.num_kv_heads_per_rank = num_key_value_heads // tp_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(
            batch_size, seq_len, self.num_heads_per_rank, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim
        ).transpose(1, 2)

        # Q-K normalization (before RoPE)
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        # Apply RoPE
        cos, sin = self.rotary_emb(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat K, V for each query head group
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # BMM attention (Neuron-friendly, no SDPA)
        attn_weights = (
            torch.bmm(
                q.reshape(batch_size * self.num_heads_per_rank, seq_len, self.head_dim),
                k.reshape(
                    batch_size * self.num_heads_per_rank, seq_len, self.head_dim
                ).transpose(-1, -2),
            )
            * self.scale
        )

        attn_weights = attn_weights.view(
            batch_size, self.num_heads_per_rank, seq_len, seq_len
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax in float32 for numerical precision
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            hidden_states.dtype
        )

        attn_output = torch.bmm(
            attn_weights.reshape(
                batch_size * self.num_heads_per_rank, seq_len, seq_len
            ),
            v.reshape(batch_size * self.num_heads_per_rank, seq_len, self.head_dim),
        )

        attn_output = attn_output.view(
            batch_size, self.num_heads_per_rank, seq_len, self.head_dim
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        return self.o_proj(attn_output)


# ── MLP ─────────────────────────────────────────────────────────────────────


class Gemma3EncoderMLP(nn.Module):
    """Gemma3 MLP: gate_proj * act(up_proj) -> down_proj, with GELU(tanh)."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ── Decoder Layer (simplified for encoder use) ─────────────────────────────


class Gemma3EncoderLayer(nn.Module):
    """Single Gemma3 decoder layer adapted for encoder-only use (no KV cache).
    Four norms per layer (Gemma3-specific).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_position_embeddings: int,
        query_pre_attn_scalar: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.self_attn = Gemma3EncoderAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            query_pre_attn_scalar=query_pre_attn_scalar,
            dtype=dtype,
        )
        self.mlp = Gemma3EncoderMLP(hidden_size, intermediate_size, dtype)

        # Four norms (Gemma3-specific)
        self.input_layernorm = Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ── Full Encoder Model ──────────────────────────────────────────────────────


class Gemma3TextEncoderModel(nn.Module):
    """Gemma3 used as a text encoder: returns all hidden states stacked.

    This is the model that gets compiled to a Neuron graph. It takes
    (input_ids, attention_mask) and returns a single tensor of shape
    (B, seq_len, hidden_size, num_layers+1).

    For Gemma 3-12B:
      hidden_size = 3840, num_hidden_layers = 48
      Output: (B, seq_len, 3840, 49)

    Note on causal attention: Gemma3 was trained with causal (left-to-right)
    attention. We keep causal masking to produce hidden states consistent
    with the original model.
    """

    def __init__(
        self,
        vocab_size: int = 262208,
        hidden_size: int = 3840,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 256,
        intermediate_size: int = 15360,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        max_position_embeddings: int = 131072,
        query_pre_attn_scalar: int = 256,
        pad_token_id: int = 0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dtype = dtype

        self.embed_tokens = Gemma3ScaledEmbedding(
            vocab_size,
            hidden_size,
            pad_token_id,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                Gemma3EncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    rope_theta=rope_theta,
                    max_position_embeddings=max_position_embeddings,
                    query_pre_attn_scalar=query_pre_attn_scalar,
                    dtype=dtype,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.norm = Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, seq_len), int64
        attention_mask: torch.Tensor,  # (B, seq_len), int64 -- 1=real, 0=pad
    ) -> torch.Tensor:
        """
        Returns:
            stacked_hidden_states: (B, seq_len, hidden_size, num_layers+1)
        """
        batch_size, seq_len = input_ids.shape

        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Causal mask: (1, 1, seq_len, seq_len) with -inf for future positions
        causal_mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=input_ids.device,
                dtype=self.dtype,
            ),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Padding mask: (B, 1, 1, seq_len) with -inf for padded positions
        pad_mask = (1.0 - attention_mask.to(self.dtype)).unsqueeze(1).unsqueeze(
            2
        ) * float("-inf")
        pad_mask = torch.nan_to_num(pad_mask, nan=0.0)
        combined_mask = causal_mask + pad_mask

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        all_hidden_states = [hidden_states]

        for layer in self.layers:
            hidden_states = layer(hidden_states, combined_mask, position_ids)
            all_hidden_states.append(hidden_states)

        # Apply final norm (replaces last element, matching HF behavior)
        hidden_states = self.norm(hidden_states)
        all_hidden_states[-1] = hidden_states

        # Stack: (B, seq_len, hidden_size, num_layers+1)
        return torch.stack(all_hidden_states, dim=-1)


# ── State Dict Conversion ───────────────────────────────────────────────────


def convert_hf_gemma3_to_encoder_state_dict(
    hf_state_dict: dict, dtype: torch.dtype = torch.bfloat16
) -> dict:
    """Convert HuggingFace Gemma3 state dict to encoder format.

    Handles multiple HF key prefix formats:
      1. "base_text_encoder.language_model.model."  -- diffusers safetensors
      2. "model.language_model."  -- pipeline state_dict
      3. "language_model.model."  -- HF safetensors
      4. "model."  -- bare Gemma3ForCausalLM

    Key renames:
      embed_tokens.weight -> embed_tokens.embedding.weight (ParallelEmbedding)
      q_norm -> q_layernorm
      k_norm -> k_layernorm
    """
    encoder_state_dict = {}

    prefixes = [
        "base_text_encoder.language_model.model.",
        "model.language_model.",
        "language_model.model.",
        "model.",
    ]

    for key, value in hf_state_dict.items():
        new_key = None
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                break
        if new_key is None:
            continue

        # Skip lm_head, vision tower, projector
        if "lm_head" in new_key:
            continue
        if not (
            new_key.startswith("embed_tokens")
            or new_key.startswith("layers.")
            or new_key.startswith("norm.")
        ):
            continue

        # Rename embed_tokens for ParallelEmbedding wrapper
        if new_key == "embed_tokens.weight":
            encoder_state_dict["embed_tokens.embedding.weight"] = (
                value.detach().clone().to(dtype)
            )
            continue

        # Rename Q-K norm: q_norm -> q_layernorm, k_norm -> k_layernorm
        new_key = new_key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
        new_key = new_key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")

        encoder_state_dict[new_key] = value.detach().clone().to(dtype)

    return encoder_state_dict
