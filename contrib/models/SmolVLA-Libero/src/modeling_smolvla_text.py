"""
SmolVLA text & expert subgraphs
================================

Defines two compiled subgraphs:

    Subgraph #2  PrefixWrapper
        VLM 16-layer text decoder, fills KV cache.

        Input:
            vision_features  [B, 192, 960]  BF16   — from vision encoder
            lang_token_ids   [B, 48]        INT32
            state            [B, 32]        FP32
        Output (stacked across layers):
            prefix_keys      [16, B, 241, 5, 64]  BF16
            prefix_values    [16, B, 241, 5, 64]  BF16

    Subgraph #3  DenoiseStepWrapper
        One Euler step of the action expert.

        Input:
            noisy_actions    [B, 50, 32]              BF16
            timestep         [B]                      FP32  (scalar per batch)
            prefix_keys      [16, B, 241, 5, 64]      BF16
            prefix_values    [16, B, 241, 5, 64]      BF16
        Output:
            v_t              [B, 50, 32]              FP32

Expert layer alternation (config: self_attn_every_n_layers=2):
    Even layers (0, 2, ..., 14)   "self-attn"
        Q from suffix; K/V = concat(past_VLM_KV, suffix_KV) over seq dim.
        K/V dim = 320 (5 KV heads × 64). q_proj outputs 960 (15 q heads × 64).
        RoPE on Q and K with positions 241..290 (continuing prefix).
        Attention: Q[50] × K[291], full bidirectional within suffix.

    Odd layers (1, 3, ..., 15)   "cross-attn"
        Q from suffix expert hidden; K/V from cached VLM K/V re-projected
        through expert k_proj/v_proj (input 320 → output 320).
        RoPE on Q only with positions 0..49.
        Attention: Q[50] × K[241].

The interleaving exactly mirrors `forward_attn_layer` and
`forward_cross_attn_layer` in the lerobot SmolVLA source.

Sinusoidal timestep embedding runs INSIDE the compiled denoise graph.
The frequency table is a register_buffer (pre-computed in __init__), so
no torch.linspace/arange is invoked during forward — see nxdi_background.md
"Dynamic Constants".
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed_inference.models.model_wrapper import ModelWrapper

import config_constants as C
from neuron_action_head_base import NeuronDenoisingConfig


# ---------------------------------------------------------------------------
# Parallel-linear helpers (TP=1 fallback safe — see config_constants.py)
# ---------------------------------------------------------------------------

def _col(in_f: int, out_f: int, bias: bool = False) -> nn.Module:
    if parallel_state.model_parallel_is_initialized():
        return ColumnParallelLinear(
            in_f, out_f, bias=bias, gather_output=False,
            dtype=torch.bfloat16,
            tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        )
    return nn.Linear(in_f, out_f, bias=bias)


def _row(in_f: int, out_f: int, bias: bool = False) -> nn.Module:
    if parallel_state.model_parallel_is_initialized():
        return RowParallelLinear(
            in_f, out_f, bias=bias, input_is_parallel=True,
            dtype=torch.bfloat16,
            tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        )
    return nn.Linear(in_f, out_f, bias=bias)


def _embed(num_emb: int, dim: int) -> nn.Module:
    if parallel_state.model_parallel_is_initialized():
        return ParallelEmbedding(
            num_emb, dim,
            dtype=torch.bfloat16,
            shard_across_embedding=True,
        )
    return nn.Embedding(num_emb, dim)


# ---------------------------------------------------------------------------
# RoPE (Llama-style, applied to a [B, S, H, D] tensor)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   [B, S, H, D]
    cos: [B, S, D]    -> broadcast to [B, S, 1, D]
    sin: [B, S, D]
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (_rotate_half(x) * sin)


class _RoPECache(nn.Module):
    """Pre-computed RoPE cos/sin tables; indexed by position_ids at forward."""
    def __init__(self, head_dim: int, max_pos: int, base: float):
        super().__init__()
        # inv_freq: [head_dim // 2]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # positions: [max_pos]
        positions = torch.arange(max_pos, dtype=torch.float32)
        # freqs: [max_pos, head_dim // 2]
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        # emb: [max_pos, head_dim] (concat of [freqs, freqs] to match Llama convention)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.bfloat16), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, S]  -> cos/sin: [B, S, head_dim]
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return cos, sin


# ---------------------------------------------------------------------------
# Eager GQA attention (used by both VLM prefix and expert layers)
# ---------------------------------------------------------------------------

def _eager_gqa_attention(
    q: torch.Tensor,            # [B, S_q, H, D]
    k: torch.Tensor,            # [B, S_kv, KH, D]
    v: torch.Tensor,            # [B, S_kv, KH, D]
    attn_mask_2d: torch.Tensor, # [B, S_q, S_kv] BOOL
    head_dim: int,
) -> torch.Tensor:
    B, S_q, H, D = q.shape
    S_kv, KH = k.shape[1], k.shape[2]
    groups = H // KH

    # Repeat K/V to match Q heads (GQA expansion)
    k = k[:, :, :, None, :].expand(B, S_kv, KH, groups, D).reshape(B, S_kv, H, D)
    v = v[:, :, :, None, :].expand(B, S_kv, KH, groups, D).reshape(B, S_kv, H, D)

    # Compute in fp32 to match HF eager_attention_forward upcast (see modeling line 528-540)
    q32 = q.to(torch.float32).transpose(1, 2)   # [B, H, S_q, D]
    k32 = k.to(torch.float32).transpose(1, 2)   # [B, H, S_kv, D]
    attn = torch.matmul(q32, k32.transpose(2, 3)) * (head_dim ** -0.5)
    big_neg = torch.finfo(torch.float32).min
    # broadcast mask [B, S_q, S_kv] -> [B, 1, S_q, S_kv]
    attn = torch.where(attn_mask_2d.unsqueeze(1), attn, big_neg)
    probs = F.softmax(attn, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v.transpose(1, 2))  # [B, H, S_q, D]
    out = out.transpose(1, 2).reshape(B, S_q, H * D)
    return out


# ---------------------------------------------------------------------------
# RMS norm
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to fp32 for the variance computation (Llama convention)
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        var = x_f32.pow(2).mean(-1, keepdim=True)
        x_f32 = x_f32 * torch.rsqrt(var + self.eps)
        return (x_f32 * self.weight).to(in_dtype)


# ---------------------------------------------------------------------------
# Llama-style MLP (gated SiLU)
# ---------------------------------------------------------------------------

class _LlamaMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = _col(hidden, intermediate, bias=False)
        self.up_proj   = _col(hidden, intermediate, bias=False)
        self.down_proj = _row(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# VLM decoder layer (standard GQA self-attention, used in prefix pass only)
# ---------------------------------------------------------------------------

class _VLMSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        H, KH, D = C.VLM_NUM_HEADS, C.VLM_NUM_KV_HEADS, C.VLM_HEAD_DIM
        self.num_heads = H
        self.num_kv_heads = KH
        self.head_dim = D
        self.q_proj = _col(C.VLM_HIDDEN, H * D,  bias=False)
        self.k_proj = _col(C.VLM_HIDDEN, KH * D, bias=False)
        self.v_proj = _col(C.VLM_HIDDEN, KH * D, bias=False)
        self.o_proj = _row(H * D, C.VLM_HIDDEN, bias=False)


class VLMDecoderLayer(nn.Module):
    """
    LlamaDecoderLayer split apart so the prefix pass can return per-layer K/V
    in the same step as the residual update.
    """
    def __init__(self):
        super().__init__()
        self.input_layernorm = _RMSNorm(C.VLM_HIDDEN, C.VLM_RMS_NORM_EPS)
        self.self_attn = _VLMSelfAttention()
        self.post_attention_layernorm = _RMSNorm(C.VLM_HIDDEN, C.VLM_RMS_NORM_EPS)
        self.mlp = _LlamaMLP(C.VLM_HIDDEN, C.VLM_INTERMEDIATE)

    def forward(
        self,
        hidden_states: torch.Tensor,    # [B, 241, 960]
        cos: torch.Tensor,              # [B, 241, 64]
        sin: torch.Tensor,              # [B, 241, 64]
        attention_mask_2d: torch.Tensor,# [B, 241, 241] BOOL
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)

        B, S, _ = x.shape
        H, KH, D = self.self_attn.num_heads, self.self_attn.num_kv_heads, self.self_attn.head_dim
        q = self.self_attn.q_proj(x).view(B, S, H, D)
        k = self.self_attn.k_proj(x).view(B, S, KH, D)
        v = self.self_attn.v_proj(x).view(B, S, KH, D)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn = _eager_gqa_attention(q, k, v, attention_mask_2d, D)
        attn = attn.to(self.self_attn.o_proj.weight.dtype if hasattr(self.self_attn.o_proj, "weight") else attn.dtype)
        x = residual + self.self_attn.o_proj(attn)

        residual2 = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual2 + x

        return x, k, v


# ---------------------------------------------------------------------------
# Prefix model — the thing that goes into NEFF #2
# ---------------------------------------------------------------------------

class SmolVLAPrefixModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = _embed(C.VLM_VOCAB_SIZE, C.VLM_HIDDEN)
        self.layers = nn.ModuleList(
            [VLMDecoderLayer() for _ in range(C.VLM_NUM_LAYERS)]
        )
        self.norm = _RMSNorm(C.VLM_HIDDEN, C.VLM_RMS_NORM_EPS)
        # state_proj is owned by the prefix: state -> 1 prefix token
        self.state_proj = nn.Linear(C.MAX_STATE_DIM, C.VLM_HIDDEN, bias=True)

        # RoPE table over the largest position the prefix will see
        self.rope = _RoPECache(C.VLM_HEAD_DIM, C.FULL_LEN, C.VLM_ROPE_THETA)

        # Constants for embed scaling (sqrt(hidden) — see modeling_smolvla.py:684)
        self.register_buffer(
            "lang_emb_scale",
            torch.tensor(C.VLM_HIDDEN ** 0.5, dtype=torch.bfloat16),
            persistent=False,
        )

        # Block-attention markers for the prefix-LM cumsum mask:
        # image+lang are one block (mark=0 for all but the first), state is a
        # separate block (mark=1 at state position to start a new block).
        # Pad-aware variant: pad lang positions are skipped via the pad mask.
        att_marks = torch.zeros(C.PREFIX_LEN, dtype=torch.int64)
        att_marks[C.NUM_VISION_TOKENS_TOTAL + C.NUM_TEXT_TOKENS:] = 1
        self.register_buffer(
            "prefix_att_marks",
            att_marks.unsqueeze(0),                                        # [1, PREFIX_LEN]
            persistent=False,
        )
        # constant ones for the always-valid vision and state regions
        self.register_buffer(
            "vision_pad_const",
            torch.ones(1, C.NUM_VISION_TOKENS_TOTAL, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "state_pad_const",
            torch.ones(1, C.NUM_STATE_TOKENS, dtype=torch.bool),
            persistent=False,
        )

    def forward(
        self,
        vision_features: torch.Tensor,  # [B, 192, 960]  — already scaled by sqrt(hidden)
        lang_token_ids: torch.Tensor,   # [B, NUM_TEXT_TOKENS] INT32
        lang_mask: torch.Tensor,        # [B, NUM_TEXT_TOKENS] BOOL  (True = valid token)
        state: torch.Tensor,            # [B, 32]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = vision_features.shape[0]

        # 1. embed lang and scale
        lang_emb = self.embed_tokens(lang_token_ids).to(torch.bfloat16) * self.lang_emb_scale  # [B, S_lang, 960]

        # 2. project state and add token dim
        state_emb = self.state_proj(state.to(torch.bfloat16)).unsqueeze(1)                # [B, 1, 960]

        # 3. concat: image + lang + state
        prefix = torch.cat([vision_features, lang_emb, state_emb], dim=1)                  # [B, PREFIX_LEN, 960]

        # 4. Pad-aware position ids and attention mask.
        # Build full pad mask [B, PREFIX_LEN] = vision_ones | lang_mask | state_ones
        pad_mask = torch.cat([
            self.vision_pad_const.expand(B, -1),
            lang_mask.to(torch.bool),
            self.state_pad_const.expand(B, -1),
        ], dim=1)                                                                          # [B, PREFIX_LEN]

        # position_ids = cumsum(pad_mask) - 1, clamped at 0
        position_ids = torch.cumsum(pad_mask.to(torch.int64), dim=1) - 1
        position_ids = torch.clamp(position_ids, min=0)                                    # [B, PREFIX_LEN]
        cos, sin = self.rope(position_ids)                                                  # [B, PREFIX_LEN, 64]

        # 2D attention mask = (cumsum-prefix-LM mask) AND (pad outer product)
        att_marks = self.prefix_att_marks.expand(B, -1)                                     # [B, PREFIX_LEN]
        cumsum_att = torch.cumsum(att_marks, dim=1)                                         # [B, PREFIX_LEN]
        att_2d = cumsum_att.unsqueeze(1) <= cumsum_att.unsqueeze(2)                          # [B, PREFIX_LEN, PREFIX_LEN]
        pad_2d = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)                               # [B, PREFIX_LEN, PREFIX_LEN]
        attn_mask = att_2d & pad_2d                                                         # [B, PREFIX_LEN, PREFIX_LEN]

        # 5. 16 layers, collect per-layer K/V
        keys: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        x = prefix
        for layer in self.layers:
            x, k, v = layer(x, cos, sin, attn_mask)
            keys.append(k)
            values.append(v)

        # x is unused after final layer (we only need K/V from the prefix), but
        # the final norm matches the HF reference exactly. Keep it cheap-out:
        # the result is discarded.
        _ = self.norm(x)

        prefix_keys = torch.stack(keys, dim=0)        # [16, B, 241, 5, 64]
        prefix_values = torch.stack(values, dim=0)
        return prefix_keys, prefix_values


# ---------------------------------------------------------------------------
# Expert decoder layers: even index = self-attn,  odd index = cross-attn
# ---------------------------------------------------------------------------

class _ExpertSelfAttnLayer(nn.Module):
    """
    Even-indexed expert layer.

    Q from suffix expert hidden (720). K/V projected from suffix expert hidden
    (720 → 320), then concatenated with the past VLM K/V (320 each) along the
    seq dim. RoPE applied with prefix-continued positions (241..290) on Q and
    on the suffix-portion of K only — past K already has RoPE baked in from
    the prefix pass.
    """
    def __init__(self):
        super().__init__()
        H, KH, D = C.EXPERT_NUM_HEADS, C.EXPERT_NUM_KV_HEADS, C.EXPERT_HEAD_DIM
        self.head_dim = D
        self.num_heads = H
        self.num_kv_heads = KH

        self.input_layernorm = _RMSNorm(C.EXPERT_HIDDEN, C.VLM_RMS_NORM_EPS)
        self.q_proj = _col(C.EXPERT_HIDDEN, H * D,  bias=False)   # 720 -> 960
        self.k_proj = _col(C.EXPERT_HIDDEN, KH * D, bias=False)   # 720 -> 320
        self.v_proj = _col(C.EXPERT_HIDDEN, KH * D, bias=False)   # 720 -> 320
        self.o_proj = _row(H * D, C.EXPERT_HIDDEN, bias=False)    # 960 -> 720
        self.post_attention_layernorm = _RMSNorm(C.EXPERT_HIDDEN, C.VLM_RMS_NORM_EPS)
        self.mlp = _LlamaMLP(C.EXPERT_HIDDEN, C.EXPERT_INTERMEDIATE)

    def forward(
        self,
        suffix_hidden: torch.Tensor,    # [B, 50, 720]
        past_k: torch.Tensor,           # [B, 241, 5, 64]
        past_v: torch.Tensor,           # [B, 241, 5, 64]
        suffix_cos: torch.Tensor,       # [B, 50, 64]   — positions 241..290
        suffix_sin: torch.Tensor,       # [B, 50, 64]
        attention_mask_2d: torch.Tensor,# [B, 50, 291] BOOL
    ) -> torch.Tensor:
        residual = suffix_hidden
        x = self.input_layernorm(suffix_hidden)

        B, S = x.shape[:2]
        H, KH, D = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, D)
        k = self.k_proj(x).view(B, S, KH, D)
        v = self.v_proj(x).view(B, S, KH, D)

        q = _apply_rope(q, suffix_cos, suffix_sin)
        k = _apply_rope(k, suffix_cos, suffix_sin)

        # Concat past + new along seq dim -> 291 keys/values
        full_k = torch.cat([past_k, k], dim=1)
        full_v = torch.cat([past_v, v], dim=1)

        attn = _eager_gqa_attention(q, full_k, full_v, attention_mask_2d, D)
        x = residual + self.o_proj(attn.to(suffix_hidden.dtype))

        residual2 = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual2 + x


class _ExpertCrossAttnLayer(nn.Module):
    """
    Odd-indexed expert layer.

    Q from suffix expert hidden (720). K/V from cached VLM K/V re-projected
    through expert k_proj/v_proj (320 → 320). RoPE on Q only with positions
    0..49 (independent of prefix positions — see modeling_smolvla.py:365).
    """
    def __init__(self):
        super().__init__()
        H, KH, D = C.EXPERT_NUM_HEADS, C.EXPERT_NUM_KV_HEADS, C.EXPERT_HEAD_DIM
        self.head_dim = D
        self.num_heads = H
        self.num_kv_heads = KH

        self.input_layernorm = _RMSNorm(C.EXPERT_HIDDEN, C.VLM_RMS_NORM_EPS)
        self.q_proj = _col(C.EXPERT_HIDDEN, H * D,  bias=False)   # 720 -> 960
        self.k_proj = _col(C.EXPERT_KV_DIM,  KH * D, bias=False)  # 320 -> 320
        self.v_proj = _col(C.EXPERT_KV_DIM,  KH * D, bias=False)  # 320 -> 320
        self.o_proj = _row(H * D, C.EXPERT_HIDDEN, bias=False)    # 960 -> 720
        self.post_attention_layernorm = _RMSNorm(C.EXPERT_HIDDEN, C.VLM_RMS_NORM_EPS)
        self.mlp = _LlamaMLP(C.EXPERT_HIDDEN, C.EXPERT_INTERMEDIATE)

    def forward(
        self,
        suffix_hidden: torch.Tensor,    # [B, 50, 720]
        past_k: torch.Tensor,           # [B, 241, 5, 64]   — VLM cached K
        past_v: torch.Tensor,           # [B, 241, 5, 64]
        suffix_cos: torch.Tensor,       # [B, 50, 64]   — positions 0..49
        suffix_sin: torch.Tensor,       # [B, 50, 64]
        attention_mask_2d: torch.Tensor,# [B, 50, 241] BOOL
    ) -> torch.Tensor:
        residual = suffix_hidden
        x = self.input_layernorm(suffix_hidden)

        B, S = x.shape[:2]
        H, KH, D = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, D)
        q = _apply_rope(q, suffix_cos, suffix_sin)

        # Re-project past VLM K/V through expert k_proj/v_proj (320 -> 320)
        past_k_flat = past_k.reshape(B, C.PREFIX_LEN, KH * D)
        past_v_flat = past_v.reshape(B, C.PREFIX_LEN, KH * D)
        k = self.k_proj(past_k_flat).view(B, C.PREFIX_LEN, KH, D)
        v = self.v_proj(past_v_flat).view(B, C.PREFIX_LEN, KH, D)
        # No RoPE on K (VLM K already had prefix RoPE applied during prefix pass)

        attn = _eager_gqa_attention(q, k, v, attention_mask_2d, D)
        x = residual + self.o_proj(attn.to(suffix_hidden.dtype))

        residual2 = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual2 + x


# ---------------------------------------------------------------------------
# Suffix embedder + denoise step (compiled subgraph #3)
# ---------------------------------------------------------------------------

class _SinusoidalTimestepEmbedder(nn.Module):
    """
    Pure-Neuron sinusoidal positional embedding for the diffusion timestep.

    Instead of `torch.linspace` inside forward (which becomes a dynamic
    constant and bloats NEFF compile-time path names), the period table is
    computed once in __init__ and registered as a buffer.
    """
    def __init__(self, dim: int, min_period: float, max_period: float):
        super().__init__()
        assert dim % 2 == 0, "Sinusoidal embedding dim must be even."
        fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32)
        period = min_period * (max_period / min_period) ** fraction
        # angular_freq[i] = 2*pi / period[i]
        ang_freq = (2.0 * math.pi) / period
        self.register_buffer("angular_freq", ang_freq, persistent=False)
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time: [B] FP32 in [0, 1]   -> emb: [B, dim] BF16
        # angular_freq: [dim/2]
        x = time.unsqueeze(-1) * self.angular_freq.unsqueeze(0)  # [B, dim/2]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)     # [B, dim]
        return emb.to(torch.bfloat16)


class SmolVLADenoiseStep(nn.Module):
    """
    One Euler step of the action expert.

    Inputs are exactly the four tensors the compiled NEFF needs. CPU-side
    Euler integration calls this with new noisy_actions on each step.
    """
    def __init__(self):
        super().__init__()
        # Embed suffix: action_in_proj + sinusoidal time + action_time_mlp
        self.action_in_proj   = nn.Linear(C.MAX_ACTION_DIM, C.EXPERT_HIDDEN, bias=True)
        self.action_time_mlp_in  = nn.Linear(C.ACTION_TIME_MLP_IN_DIM, C.EXPERT_HIDDEN, bias=True)
        self.action_time_mlp_out = nn.Linear(C.EXPERT_HIDDEN, C.EXPERT_HIDDEN, bias=True)
        self.action_out_proj  = nn.Linear(C.EXPERT_HIDDEN, C.MAX_ACTION_DIM, bias=True)
        self.timestep_embedder = _SinusoidalTimestepEmbedder(
            C.TIMESTEP_EMBED_DIM, C.TIMESTEP_MIN_PERIOD, C.TIMESTEP_MAX_PERIOD
        )

        # 16 expert layers: even idx self-attn, odd idx cross-attn
        layers = []
        for i in range(C.EXPERT_NUM_LAYERS):
            if i % C.SELF_ATTN_EVERY_N_LAYERS == 0:
                layers.append(_ExpertSelfAttnLayer())
            else:
                layers.append(_ExpertCrossAttnLayer())
        self.layers = nn.ModuleList(layers)
        self.norm = _RMSNorm(C.EXPERT_HIDDEN, C.VLM_RMS_NORM_EPS)

        # RoPE caches for both layer types
        self.rope = _RoPECache(C.EXPERT_HEAD_DIM, C.FULL_LEN, C.VLM_ROPE_THETA)

        # Cumsum-based block-attention pattern over the FULL sequence.
        # prefix has [0]*(vis+lang) + [1]*1  (state starts a new block)
        # suffix has [1]*50           (each suffix token starts a new block)
        full_att_marks = torch.zeros(C.FULL_LEN, dtype=torch.int64)
        full_att_marks[C.NUM_VISION_TOKENS_TOTAL + C.NUM_TEXT_TOKENS:] = 1
        self.register_buffer(
            "full_att_marks",
            full_att_marks.unsqueeze(0),                                    # [1, FULL_LEN]
            persistent=False,
        )
        self.register_buffer(
            "suffix_pad_const",
            torch.ones(1, C.SUFFIX_LEN, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "suffix_arange",
            torch.arange(C.SUFFIX_LEN, dtype=torch.int64).unsqueeze(0),     # [1, 50]
            persistent=False,
        )

    def _embed_suffix(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        # noisy_actions: [B, 50, 32]  timestep: [B] fp32
        action_emb = self.action_in_proj(noisy_actions.to(torch.bfloat16))   # [B, 50, 720]
        time_emb = self.timestep_embedder(timestep)                           # [B, 720]
        time_emb = time_emb.unsqueeze(1).expand_as(action_emb)                # [B, 50, 720]
        cat = torch.cat([action_emb, time_emb], dim=-1)                       # [B, 50, 1440]
        x = self.action_time_mlp_in(cat)
        x = F.silu(x)
        x = self.action_time_mlp_out(x)
        return x   # [B, 50, 720]

    def forward(
        self,
        noisy_actions: torch.Tensor,    # [B, 50, 32]   FP32 in (from CPU)
        timestep: torch.Tensor,         # [B]           FP32
        prefix_keys: torch.Tensor,      # [L, B, PREFIX_LEN, 5, 64] BF16
        prefix_values: torch.Tensor,    # [L, B, PREFIX_LEN, 5, 64] BF16
        prefix_pad_mask: torch.Tensor,  # [B, PREFIX_LEN] BOOL
    ) -> torch.Tensor:                  # [B, 50, 32]   FP32

        B = noisy_actions.shape[0]
        suffix = self._embed_suffix(noisy_actions, timestep)   # [B, 50, hidden]

        # Suffix position_ids for self-attn = prefix_offset + 0..49 (per lerobot
        # `position_ids = prefix_offsets + cumsum(suffix_pad_masks) - 1`).
        # For cross-attn, RoPE on Q only with positions 0..49 (independent —
        # see modeling_smolvla.py:365).
        prefix_pad_b = prefix_pad_mask.to(torch.bool)
        prefix_offset = prefix_pad_b.to(torch.int64).sum(dim=1, keepdim=True)   # [B, 1]
        self_pos  = prefix_offset + self.suffix_arange.expand(B, -1)            # [B, 50]
        cross_pos = self.suffix_arange.expand(B, -1)                            # [B, 50]

        self_cos, self_sin   = self.rope(self_pos)
        cross_cos, cross_sin = self.rope(cross_pos)

        # Self-attn mask over [B, 50, FULL_LEN]: cumsum-block AND pad-2D.
        full_pad = torch.cat(
            [prefix_pad_b, self.suffix_pad_const.expand(B, -1)], dim=1,
        )                                                                       # [B, FULL_LEN]
        att_marks = self.full_att_marks.expand(B, -1)                           # [B, FULL_LEN]
        cumsum_att = torch.cumsum(att_marks, dim=1)                             # [B, FULL_LEN]
        att_2d = cumsum_att.unsqueeze(1) <= cumsum_att.unsqueeze(2)              # [B, FULL_LEN, FULL_LEN]
        pad_2d = full_pad.unsqueeze(1) & full_pad.unsqueeze(2)                   # [B, FULL_LEN, FULL_LEN]
        full_mask = att_2d & pad_2d                                              # [B, FULL_LEN, FULL_LEN]
        self_mask  = full_mask[:, C.PREFIX_LEN:, :]                              # [B, 50, FULL_LEN]
        cross_mask = prefix_pad_b.unsqueeze(1).expand(B, C.SUFFIX_LEN, -1)        # [B, 50, PREFIX_LEN]

        x = suffix
        for i, layer in enumerate(self.layers):
            past_k = prefix_keys[i]    # [B, 241, 5, 64]
            past_v = prefix_values[i]
            if i % C.SELF_ATTN_EVERY_N_LAYERS == 0:
                x = layer(x, past_k, past_v, self_cos, self_sin, self_mask)
            else:
                x = layer(x, past_k, past_v, cross_cos, cross_sin, cross_mask)

        x = self.norm(x)
        # The HF reference upcasts to fp32 before action_out_proj. Linear in
        # bf16 followed by fp32 cast preserves numeric accuracy adequately
        # (action_out_proj is a single 720->32 projection at the end of 16
        # layers of bf16 attention) and keeps the Linear dtype-matched.
        v_t = self.action_out_proj(x).to(torch.float32)
        return v_t


# ---------------------------------------------------------------------------
# Wrappers for ModelBuilder compilation
# ---------------------------------------------------------------------------

class SmolVLAPrefixWrapper(ModelWrapper):
    tag = "prefix"

    def __init__(self, config: NeuronDenoisingConfig):
        nn.Module.__init__(self)
        super().__init__(config=config, model_cls=type(self))
        self.config = config
        self.model = None

    def load_module(self):
        self.model = SmolVLAPrefixModel().bfloat16().eval()

    def forward(self, vision_features, lang_token_ids, lang_mask, state):
        return self.model(vision_features, lang_token_ids, lang_mask, state)

    def input_generator(self):
        B = self.config.neuron_config.batch_size
        return [(
            torch.zeros(B, C.NUM_VISION_TOKENS_TOTAL, C.VLM_HIDDEN, dtype=torch.bfloat16),
            torch.zeros(B, C.NUM_TEXT_TOKENS, dtype=torch.int32),
            torch.ones(B, C.NUM_TEXT_TOKENS, dtype=torch.bool),
            torch.zeros(B, C.MAX_STATE_DIM, dtype=torch.float32),
        )]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return super().load_state_dict(state_dict, strict=strict, **kwargs)


class SmolVLADenoiseWrapper(ModelWrapper):
    tag = "denoise_step"

    def __init__(self, config: NeuronDenoisingConfig):
        nn.Module.__init__(self)
        super().__init__(config=config, model_cls=type(self))
        self.config = config
        self.model = None

    def load_module(self):
        self.model = SmolVLADenoiseStep().bfloat16().eval()

    def forward(self, noisy_actions, timestep, prefix_keys, prefix_values, prefix_pad_mask):
        return self.model(noisy_actions, timestep, prefix_keys, prefix_values, prefix_pad_mask)

    def input_generator(self):
        B = self.config.neuron_config.batch_size
        return [(
            torch.zeros(B, C.ACTION_CHUNK_SIZE, C.MAX_ACTION_DIM, dtype=torch.float32),
            torch.zeros(B, dtype=torch.float32),
            torch.zeros(C.VLM_NUM_LAYERS, B, C.PREFIX_LEN, C.VLM_NUM_KV_HEADS, C.VLM_HEAD_DIM, dtype=torch.bfloat16),
            torch.zeros(C.VLM_NUM_LAYERS, B, C.PREFIX_LEN, C.VLM_NUM_KV_HEADS, C.VLM_HEAD_DIM, dtype=torch.bfloat16),
            torch.ones(B, C.PREFIX_LEN, dtype=torch.bool),
        )]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return super().load_state_dict(state_dict, strict=strict, **kwargs)
