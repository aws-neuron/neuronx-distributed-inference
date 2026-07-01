# coding=utf-8
# Copyright 2026 MiniMax AI and Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch MiniMax-M3 text backbone for NXD inference.

This is an MVP port of the MiniMax-M3 text decoder (vision tower and the
Lightning-Indexer-based block-sparse attention are not included). The sparse
attention layers are run as dense GQA — this matches the model's reference
behavior at moderate context lengths but does not realise the long-context
compute savings of MSA. Multi-Token Prediction (MTP) modules are skipped.

Architecture highlights (text backbone, see config.json `text_config`):
  - 60 decoder layers, 6144 hidden size
  - GQA: 64 query heads / 4 KV heads, head_dim=128
  - Partial RoPE: rotary_dim=64 over the first half of each head
  - Per-head Gemma-style RMSNorm on Q and K (weight initialised to zeros,
    scale is (1 + weight))
  - SwiGLU-OAI gated MLP with `alpha=1.702`, clamp at `limit=7.0`
  - Dense MLP (intermediate=12288) for the first 3 layers; MoE for the rest
  - MoE: 128 experts, top-4 sigmoid routing with a learned correction bias,
    + 1 shared expert (intermediate=3072), routed outputs scaled by 2.0
"""

import gc
import math
import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
)
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

from neuronx_distributed.modules.moe.routing import RouterTopK

from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

# Lower bound on per-rank intermediate size required by NxDI's shard-on-I
# blockwise matmul kernel (matches the M2 contrib constant).
SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP = 128
# Per-rank intermediate alignment required by the fused MoE TKG NKI kernel.
MOE_TKG_MK_INTERMEDIATE_PER_TP = 128


# -----------------------------------------------------------------------------
# Norm helpers
# -----------------------------------------------------------------------------
class MiniMaxM3GemmaRMSNorm(nn.Module):
    """Plain RMSNorm `x_norm * w` for use with M3.

    M3's HF reference uses Gemma-style `(1 + weight)` scaling. To stay
    compatible with NxDI's fused attention/QKV TKG kernels (which apply
    plain `x_norm * w`), the converter pre-adds +1.0 to every RMSNorm
    weight in the checkpoint. This module therefore performs the plain
    Llama-style multiply — the `+1` is baked into the loaded weights.
    Normalization is done in fp32 to match the HF reference, then cast
    back to the input dtype (Gemma3 ordering).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * self.weight.float()
        return x.to(in_dtype)


def get_rmsnorm_cls():
    """Standard RMSNorm class (CustomRMSNorm on device, HF-style on CPU)."""
    return CustomRMSNorm if not cpu_mode() else nn.LayerNorm


# -----------------------------------------------------------------------------
# Partial RoPE
# -----------------------------------------------------------------------------
class MiniMaxM3PartialRotaryEmbedding(nn.Module):
    """RoPE applied only over the first `rotary_dim` channels of each head.

    Mirrors `MiniMaxM3VLRotaryEmbedding` from transformers: concatenated (not
    interleaved) freqs, cos/sin shape `(B, S, rotary_dim)`.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_minimax_m3_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply partial RoPE to Q and K.

    cos/sin shape: `(B, S, rotary_dim)`. Q/K shape: `(B, H, S, head_dim)`.
    Only the first `rotary_dim` channels of each head are rotated.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    return (
        torch.cat([q_embed, q_pass], dim=-1),
        torch.cat([k_embed, k_pass], dim=-1),
    )


# -----------------------------------------------------------------------------
# MSA (MiniMax Sparse Attention) Lightning Indexer
# -----------------------------------------------------------------------------
class MiniMaxM3Indexer(nn.Module):
    """Lightning Indexer for MSA — selects per-query top-K key blocks.

    Mirrors HF's ``MiniMaxM3VLIndexer``:
      * ``q_proj``: hidden → ``index_n_heads * index_head_dim`` (4 * 128 = 512)
      * ``k_proj``: hidden → ``index_head_dim`` (128) — single indexer key head
      * ``q_norm``, ``k_norm``: RMSNorm(head_dim=128) applied per head
      * partial RoPE on the first ``index_head_dim`` channels of idx_q / idx_k

    Forward returns the sparse causal mask ``(B, num_heads, S_q, S_k)`` (0 for
    kept, -inf for dropped) so the caller can pass it straight to attention.
    ``num_heads`` is the full ``config.num_attention_heads`` — the indexer's
    per-block verdict is broadcast to every query head within the GQA group.

    The `+1` Gemma pre-shift on q_norm/k_norm weights is handled by the
    checkpoint converter, so we use plain RMSNorm here.
    """

    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        block_size: int,
        topk_blocks: int,
        local_blocks: int,
        num_attention_heads: int,
        rms_norm_eps: float,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.local_blocks = local_blocks
        self.num_attention_heads = num_attention_heads
        self.q_proj = ColumnParallelLinear(
            hidden_size, index_n_heads * index_head_dim, bias=False,
            gather_output=True, dtype=dtype,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, index_head_dim, bias=False,
            gather_output=True, dtype=dtype,
        )
        self.q_norm = get_rmsnorm_cls()(hidden_size=index_head_dim, eps=rms_norm_eps)
        self.k_norm = get_rmsnorm_cls()(hidden_size=index_head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Returns sparse causal mask ``(B, num_attention_heads, S_q, S_k)``."""
        B, S, _ = hidden_states.shape
        idx_q = self.q_proj(hidden_states).view(B, S, self.n_heads, self.head_dim)
        idx_q = self.q_norm(idx_q).transpose(1, 2)  # (B, n_heads, S, D_idx)
        idx_k = self.k_proj(hidden_states).view(B, S, 1, self.head_dim)
        idx_k = self.k_norm(idx_k).transpose(1, 2)  # (B, 1, S, D_idx)

        # partial RoPE on first head_dim channels of cos/sin
        idx_q, idx_k = apply_minimax_m3_rotary(
            idx_q, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim]
        )

        # Score qk in fp32
        k_len = idx_k.shape[2]
        num_key_blocks = (k_len + self.block_size - 1) // self.block_size
        pad = num_key_blocks * self.block_size - k_len

        scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))
        k_positions = torch.arange(k_len, device=idx_q.device)
        # Future mask
        token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
        scores = scores.masked_fill(token_future, float("-inf"))
        if pad:
            scores = F.pad(scores, (0, pad), value=float("-inf"))

        # Max-pool per block
        scores = scores.view(B, self.n_heads, S, num_key_blocks, self.block_size)
        block_scores = scores.amax(dim=-1)  # (B, n_heads, S, num_blocks)

        # Force local blocks (last N before query's own block) to always keep
        q_block = (position_ids // self.block_size)  # (B, S)
        if self.local_blocks > 0:
            local = torch.arange(self.local_blocks, device=idx_q.device)
            local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
            local_idx = local_idx.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            block_scores = block_scores.scatter(-1, local_idx, float("inf"))

        # Top-K blocks
        topk = min(self.topk_blocks, num_key_blocks)
        topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
        # Invalidate slots whose top-k score is still -inf
        topk_indices = topk_indices.masked_fill(topk_scores == float("-inf"), -1)

        # Expand top-K block indices to a `(B, num_att_heads, S_q, S_k)` mask
        safe = topk_indices.masked_fill(topk_indices < 0, num_key_blocks)
        bias = torch.full(
            (B, self.n_heads, S, num_key_blocks + 1),
            float("-inf"),
            device=idx_q.device,
            dtype=torch.float32,
        )
        bias.scatter_(-1, safe, 0.0)
        bias = bias[..., :num_key_blocks]

        # Repeat per-block to per-key, then broadcast per-idx-head to all attention heads
        block_keep = (bias == 0.0).repeat_interleave(self.block_size, dim=-1)[..., :k_len]
        block_keep = block_keep.repeat_interleave(
            self.num_attention_heads // self.n_heads, dim=1
        )  # (B, num_att_heads, S_q, S_k)

        # Compose with causal mask
        keep = block_keep & ~token_future
        # Emit additive mask
        min_val = torch.finfo(hidden_states.dtype).min
        mask = torch.zeros_like(keep, dtype=hidden_states.dtype).masked_fill(~keep, min_val)
        return mask


# -----------------------------------------------------------------------------
# Router with e_score_correction_bias (sigmoid + bias, MiniMax / DeepSeek style)
# -----------------------------------------------------------------------------
class RouterTopKWithBias(RouterTopK):
    """RouterTopK with `e_score_correction_bias` for sigmoid routing.

    Ported from the MiniMax-M2 contrib model. The bias affects which experts
    are selected (`topk(sigmoid(logits) + bias)`) but the un-biased sigmoid
    scores remain the affinity weights. Without the bias, ~75% of expert
    selections differ from the HF reference (bias values ~8.0-9.5 dominate
    the 0..1 sigmoid range).

    The bias is stored as ``nn.Parameter`` (not a buffer) so XLA tracing
    separates it from the NEFF and lets the checkpoint loader fill it in.
    Initialised with ``torch.arange(...)`` rather than zeros so the bias
    values are non-uniform — uniform values would let XLA fold the add into
    a no-op and drop the parameter from the HLO entirely.
    """

    def __init__(self, num_experts: int, *args, **kwargs):
        super().__init__(num_experts=num_experts, *args, **kwargs)
        # Float32 because HF M3's `MiniMaxM3VLTopKRouter` adds the bias to
        # `sigmoid(logits).float()` (fp32). The bias values in the checkpoint
        # are stored as fp32 too. Using bf16 here drops ~7 bits of precision
        # on the +bias add and may shift topk decisions.
        self.e_score_correction_bias = nn.Parameter(
            torch.arange(num_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)

        scores_for_choice = (
            expert_affinities.float() + self.e_score_correction_bias.unsqueeze(0)
        )
        _, expert_index = torch.topk(scores_for_choice, self.top_k, dim=-1)

        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
        expert_index = expert_index.detach().to(dtype=torch.long)

        return router_logits, expert_affinities, expert_index


def initialize_minimax_m3_moe_module(
    config: InferenceConfig, rmsnorm=None, init_tkg_module=False
):
    """Create the M3 MoE module with `RouterTopKWithBias` and 1 shared expert.

    Identical wiring to NxDI's ``initialize_moe_module`` except for the router
    swap and the explicit shared-experts handling.
    """
    from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
    from neuronx_distributed.modules.moe.model import MoE, MoEFusedTKGConfig
    from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig
    from neuronx_distributed.modules.moe.shared_experts import SharedExperts
    from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_process_group

    enabled_hybrid_sharding = config.neuron_config.hybrid_sharding_config is not None
    (
        moe_tkg_tp_group,
        moe_tkg_ep_group,
        moe_cte_tp_group,
        moe_cte_ep_group,
    ) = initialize_moe_process_group(config, enabled_hybrid_sharding)

    router = RouterTopKWithBias(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        dtype=config.neuron_config.router_config.dtype,
        act_fn=config.neuron_config.router_config.act_fn,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
        bias=False,
        apply_act_fn_over_topk=False,
        store_transposed_weights=init_tkg_module,
    )

    hidden_size_actual = getattr(config, "original_hidden_size", None)
    intermediate_size_actual = getattr(config, "original_intermediate_size", None)

    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_size_actual=hidden_size_actual,
            intermediate_size_actual=intermediate_size_actual,
            is_hidden_dim_shuffled=config.neuron_config.is_hidden_dim_shuffled,
            is_intermediate_dim_shuffled=config.neuron_config.is_intermediate_dim_shuffled,
            top_k=config.num_experts_per_tok,
            hidden_act=config.hidden_act,
            bias=False,
            glu_mlp=config.neuron_config.glu_mlp,
            glu_type=config.neuron_config.glu_type,
            hidden_act_scaling_factor=config.neuron_config.hidden_act_scaling_factor,
            hidden_act_bias=config.neuron_config.hidden_act_bias,
            use_index_calc_kernel=config.neuron_config.use_index_calc_kernel,
            gate_clamp_upper_limit=config.neuron_config.gate_clamp_upper_limit,
            gate_clamp_lower_limit=config.neuron_config.gate_clamp_lower_limit,
            up_clamp_upper_limit=config.neuron_config.up_clamp_upper_limit,
            up_clamp_lower_limit=config.neuron_config.up_clamp_lower_limit,
            early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation,
            normalize_top_k_affinities=config.neuron_config.normalize_top_k_affinities,
            enable_spmd_rank=config.neuron_config.blockwise_matmul_config.parallelize_token_to_block_mapping,
        ),
        blockwise_matmul_config=config.neuron_config.blockwise_matmul_config,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        dtype=config.neuron_config.torch_dtype,
        is_prefill=config.neuron_config.is_prefill_stage,
        enabled_hybrid_sharding=enabled_hybrid_sharding,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        expert_model_parallel_group=parallel_state.get_expert_model_parallel_group(),
        cte_tensor_model_parallel_group=moe_cte_tp_group,
        cte_expert_model_parallel_group=moe_cte_ep_group,
        tkg_tensor_model_parallel_group=moe_tkg_tp_group,
        tkg_expert_model_parallel_group=moe_tkg_ep_group,
    )

    # NxDI's SharedExperts hard-codes `act_fn(gate) * up` — that's plain
    # SwiGLU/GLU, NOT M3's SwiGLU-OAI (`gate * sigmoid(α·gate) * (up + 1)`).
    # Setting hidden_act="sigmoid" on the routed experts (so they get the
    # SwiGLU-OAI formula via NxDI's SWIGLU + hidden_act_bias=1.0) made the
    # SharedExperts path equally wrong (`sigmoid(gate) * up`). Skip the
    # bundled shared-expert and let the decoder layer apply its own
    # `MiniMaxM3DenseMLP` for the shared branch.
    shared_experts = None

    if init_tkg_module:
        tkg_config = MoEFusedTKGConfig(
            quantized=config.neuron_config.quantized,
            moe_fused_kernel_enabled=config.neuron_config.moe_fused_nki_kernel_enabled,
            router_topk_kernel_enabled=config.neuron_config.router_topk_nki_kernel_enabled,
            expert_mlp_kernel_enabled=config.neuron_config.expert_mlp_nki_kernel_enabled,
            shared_mlp_kernel_enabled=config.neuron_config.shared_mlp_nki_kernel_enabled,
            norm_topk_prob=config.neuron_config.normalize_top_k_affinities,
            is_mxfp4_compute=config.neuron_config.is_mxfp4_compute,
            router_mm_dtype=config.neuron_config.router_config.dtype,
        )
    else:
        tkg_config = None

    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=shared_experts,
        rmsnorm=rmsnorm,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        return_router_logits=config.neuron_config.return_router_logits,
        sequence_dimension=1,
        init_tkg_module=init_tkg_module,
        tkg_config=tkg_config,
    )
    moe.eval()
    return moe


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
class MiniMaxM3NeuronConfig(MoENeuronConfig):
    """Neuron hardware config for the MiniMax-M3 text backbone.

    NOTE: ExpertMLPsV2 only supports `glu` and `swiglu` activations. The
    released M3 uses SwiGLU-OAI (clamped, with `alpha` and `(up+1)` bias).
    For the MVP we run the MoE experts as plain `swiglu` — accuracy will not
    match the HF reference exactly, but the architecture compiles and runs.
    The clamp values are still wired through so future GLU types added to
    ExpertMLPsV2 can opt back into the full M3 activation.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("glu_mlp", True)
        kwargs.setdefault("glu_type", "swiglu")
        # SwiGLU-OAI is reproduced as the SWIGLU branch of NxDI ExpertMLPsV2
        # with activation=sigmoid and hidden_act_bias=1.0:
        #   out = gate * sigmoid(alpha * gate) * (up + 1.0)
        # That's exactly the M3 SwiGLU-OAI formula. Setting alpha=1.702 and
        # clamp(±7) matches the released M3 reference.
        kwargs.setdefault("hidden_act_scaling_factor", 1.702)
        kwargs.setdefault("hidden_act_bias", 1.0)
        kwargs.setdefault("gate_clamp_upper_limit", 7.0)
        kwargs.setdefault("up_clamp_upper_limit", 7.0)
        kwargs.setdefault("up_clamp_lower_limit", -7.0)
        super().__init__(**kwargs)


class MiniMaxM3InferenceConfig(InferenceConfig):
    """
    InferenceConfig for the MiniMax-M3 text backbone.

    The HF release stores everything under `config["text_config"]`. We promote
    those fields to top-level attributes so the rest of the modeling stack can
    read them with `getattr(config, ...)`.
    """

    def __init__(self, *args, **kwargs):
        # If the caller passed `load_config=load_pretrained_config(model_path)`,
        # the loader will populate attributes from the raw `config.json`. We
        # promote `text_config` afterwards in `add_derived_config`.
        super().__init__(*args, **kwargs)

    def add_derived_config(self):
        # Promote text_config to top level if present (the released config has
        # everything nested under text_config for the VL wrapper).
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            tc = text_config if isinstance(text_config, dict) else text_config.__dict__
            for key, value in tc.items():
                if not hasattr(self, key) or getattr(self, key) in (None, []):
                    setattr(self, key, value)

        # HF defaults expected by model_base
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "return_dict"):
            self.return_dict = True

        # Pad token may not be set in M3 configs
        if not hasattr(self, "pad_token_id") or self.pad_token_id is None:
            self.pad_token_id = 0

        # MoE intermediate size is the per-expert FFN; dense layers use a
        # separate (much larger) intermediate.
        if not hasattr(self, "moe_intermediate_size"):
            self.moe_intermediate_size = self.intermediate_size

        if not hasattr(self, "dense_intermediate_size"):
            self.dense_intermediate_size = self.intermediate_size

        if not hasattr(self, "shared_intermediate_size"):
            self.shared_intermediate_size = self.intermediate_size

        # n_shared_experts: 1 shared expert sized at shared_intermediate_size
        if not hasattr(self, "n_shared_experts") or self.n_shared_experts is None:
            self.n_shared_experts = 0

        # MSA (MiniMax Sparse Attention) config — indexer params.
        # `sparse_attention_config` is a nested dict; promote its fields to
        # flat attributes so the modeling code can `getattr(config, ...)`.
        sac = getattr(self, "sparse_attention_config", None)
        if isinstance(sac, dict):
            self.sparse_attention_freq = sac.get("sparse_attention_freq", [0] * self.num_hidden_layers)
            self.index_n_heads = sac.get("sparse_num_index_heads", 4)
            self.index_head_dim = sac.get("sparse_index_dim", 128)
            self.index_block_size = sac.get("sparse_block_size", 128)
            self.index_topk_blocks = sac.get("sparse_topk_blocks", 16)
            self.index_local_blocks = sac.get("sparse_local_block", 1)
        else:
            self.sparse_attention_freq = [0] * self.num_hidden_layers

        # The MoE module reads `intermediate_size` for per-expert FFN size.
        # Keep `dense_intermediate_size` to size the dense layers.
        self.intermediate_size = self.moe_intermediate_size

        # Pad intermediate for shard-on-I blockwise matmul (M2 pattern).
        self.moe_intermediate_pad_size = 0
        self._maybe_pad_intermediate()

        # Make sure num_cores_per_group is set
        self.num_cores_per_group = 1

        # SwiGLU-OAI is `gate * sigmoid(alpha * gate) * (up + 1)`. With
        # NxDI's SWIGLU path and activation_fn=sigmoid + hidden_act_bias=1.0,
        # this becomes exactly that formula. Force `hidden_act` to "sigmoid"
        # so ACT2FN["sigmoid"] is used as the gate non-linearity.
        if not hasattr(self, "hidden_act") or self.hidden_act in (None, "swigluoai", "silu"):
            self.hidden_act = "sigmoid"

        # moe_layer_freq: per-layer 0/1 flag (0 = dense, 1 = MoE). When absent,
        # fall back to first_k_dense_replace=3 (the M3 default).
        if not hasattr(self, "moe_layer_freq") or self.moe_layer_freq is None:
            first_dense = getattr(self, "first_k_dense_replace", 3)
            self.moe_layer_freq = [
                0 if i < first_dense else 1 for i in range(self.num_hidden_layers)
            ]

        # Sigmoid routing + correction bias → tell ExpertMLPsV2 not to softmax
        if hasattr(self.neuron_config, "router_config"):
            self.neuron_config.router_config.dtype = torch.float32
            self.neuron_config.router_config.act_fn = "sigmoid"

        # Disable numeric CC token for MoE stability (M2 pattern).
        self.neuron_config.disable_numeric_cc_token = True

        # Enable fused MoE NKI kernel if the per-MoE-TP intermediate is aligned.
        self._enable_moe_fused_nki_kernel()

    def _maybe_pad_intermediate(self):
        """Pad intermediate_size so the shard-on-I blockwise kernel tiles cleanly."""
        moe_tp_degree = self.neuron_config.moe_tp_degree
        i_tp = self.intermediate_size // moe_tp_degree
        if getattr(
            self.neuron_config.blockwise_matmul_config,
            "use_shard_on_intermediate_dynamic_while",
            False,
        ):
            if i_tp % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded = (
                    math.ceil(i_tp / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                    * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                    * moe_tp_degree
                )
                self.moe_intermediate_pad_size = max(padded - self.intermediate_size, 0)
                self.intermediate_size = padded

    def _enable_moe_fused_nki_kernel(self):
        i_tp = self.intermediate_size // self.neuron_config.moe_tp_degree
        if getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", False):
            if i_tp % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0:
                self.moe_fused_nki_kernel_enabled = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "head_dim",
            "num_local_experts",
            "num_experts_per_tok",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MiniMaxM3NeuronConfig]:
        return MiniMaxM3NeuronConfig


# -----------------------------------------------------------------------------
# Attention with Gemma QK-norm + partial RoPE
# -----------------------------------------------------------------------------
class NeuronMiniMaxM3Attention(NeuronAttentionBase):
    """GQA with per-head Gemma RMSNorm on Q/K and partial RoPE."""

    def __init__(self, config: MiniMaxM3InferenceConfig, layer_idx: int):
        head_dim = config.head_dim
        rotary_dim = getattr(config, "rotary_dim", head_dim // 2)

        rotary_emb = MiniMaxM3PartialRotaryEmbedding(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # M3 uses Gemma-style RMSNorm `x_norm * (1 + w)`. The converter
        # pre-adds +1.0 to every loaded RMSNorm weight, so we use NxDI's
        # CustomRMSNorm (plain `x_norm * w`) and get the Gemma scale for
        # free — compatible with NxDI's fused TKG attention kernel which
        # also applies plain `x_norm * w`.
        q_layernorm = get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps)
        k_layernorm = get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            q_layernorm=q_layernorm,
            k_layernorm=k_layernorm,
            rms_norm_eps=config.rms_norm_eps,
        )

        self.layer_idx = layer_idx
        self.rotary_dim = rotary_dim

        # MSA indexer if this is a sparse layer
        self.is_sparse = bool(getattr(config, "sparse_attention_freq", [0]*60)[layer_idx])
        self.indexer = None
        if self.is_sparse:
            self.indexer = MiniMaxM3Indexer(
                hidden_size=config.hidden_size,
                index_n_heads=getattr(config, "index_n_heads", 4),
                index_head_dim=getattr(config, "index_head_dim", 128),
                block_size=getattr(config, "index_block_size", 128),
                topk_blocks=getattr(config, "index_topk_blocks", 16),
                local_blocks=getattr(config, "index_local_blocks", 1),
                num_attention_heads=config.num_attention_heads,
                rms_norm_eps=config.rms_norm_eps,
                dtype=config.neuron_config.torch_dtype,
            )

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronMiniMaxM3Attention must be initialized in a distributed env."
            )

    def apply_rotary_embedding(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        position_ids: torch.Tensor,
        cos_cache: Optional[torch.Tensor],
        sin_cache: Optional[torch.Tensor],
        use_polar_compatible_rope: bool = False,
    ):
        if self.rotary_emb is None:
            return Q, K, cos_cache, sin_cache
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)
        Q, K = apply_minimax_m3_rotary(Q, K, cos_cache, sin_cache)
        return Q, K, cos_cache, sin_cache


# -----------------------------------------------------------------------------
# Dense (SwiGLU-OAI) MLP used by the first 3 layers
# -----------------------------------------------------------------------------
class MiniMaxM3DenseMLP(nn.Module):
    """SwiGLU-OAI gated MLP.

    Output = down_proj((1 + up) * (gate * sigmoid(alpha * gate)))
    where gate is clamped at `swiglu_limit` (upper only) and up is clamped
    symmetrically.
    """

    def __init__(self, config: MiniMaxM3InferenceConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        self.hidden_size = config.hidden_size
        # Allow overriding intermediate_size — dense layers use
        # `dense_intermediate_size` (12288), shared experts in MoE layers use
        # `shared_intermediate_size` (3072), which is much smaller.
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.dense_intermediate_size
        )
        self.alpha = getattr(config, "swiglu_alpha", 1.702)
        self.limit = getattr(config, "swiglu_limit", 7.0)

        # Fused gate+up projection with **stride=2** — this is critical for
        # correct sharding across TP ranks. NxDI's `ColumnParallelLinear`
        # with `stride=2` splits the global `(2*I, H)` weight into
        # `2*TP` chunks along the output dim and gives each rank one gate
        # chunk + one up chunk (interleaved). Without stride=2 each rank
        # would get a contiguous slice of `[all_gate | all_up]` — which
        # at TP=64 means "all gate" or "all up" on every rank, and
        # `gate_up.chunk(2, dim=-1)` on the activation does not recover
        # (gate, up). NxDI's `ExpertFusedColumnParallelLinear` does the
        # same trick for the routed-expert path (`stride=2,
        # is_fused_gate_up=True`).
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
            stride=2,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor, *_, **__):
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return (self.down_proj((up + 1.0) * glu),)


# -----------------------------------------------------------------------------
# Decoder layer (dense or MoE based on moe_layer_freq[layer_idx])
# -----------------------------------------------------------------------------
class NeuronMiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, config: MiniMaxM3InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        is_moe_layer = bool(config.moe_layer_freq[layer_idx])
        self.is_moe = is_moe_layer

        self.self_attn = NeuronMiniMaxM3Attention(config, layer_idx)
        self.moe_fused_nki_kernel_enabled = getattr(
            config, "moe_fused_nki_kernel_enabled", False
        )

        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if is_moe_layer:
            # M3 MoE: 128 routed experts + 1 shared expert. We disable
            # `n_shared_experts` in the MoE module so NxDI doesn't bundle
            # its (wrong-activation) `SharedExperts` into the same forward;
            # instead we run a `MiniMaxM3DenseMLP` sized to
            # `shared_intermediate_size` as a sibling and add it after the
            # routed branch has been scaled by `routed_scaling_factor`.
            config.n_shared_experts = 0
            if self.moe_fused_nki_kernel_enabled:
                self.block_sparse_moe = initialize_minimax_m3_moe_module(
                    config=config,
                    rmsnorm=self.post_attention_layernorm,
                    init_tkg_module=True,
                )
            else:
                self.block_sparse_moe = initialize_minimax_m3_moe_module(config=config)

            shared_inter = getattr(config, "shared_intermediate_size", 0)
            if shared_inter and shared_inter > 0:
                self.shared_experts = MiniMaxM3DenseMLP(config, intermediate_size=shared_inter)
            else:
                self.shared_experts = None
        else:
            self.mlp = MiniMaxM3DenseMLP(config)

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)

        qkv_fused_rmsnorm = None
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # MSA: if this attention layer is sparse AND we're in prefill
        # (multi-token forward), run the indexer on the normalized hidden
        # states and build a block-sparse causal mask that overrides the
        # ordinary causal `attention_mask` from the caller. Decode steps
        # (S==1) fall back to dense causal — attention over KV cache stays
        # unchanged. This is a simplification for MVP; a full MSA impl
        # would need to score against cached indexer keys during decode.
        if (
            getattr(self.self_attn, "indexer", None) is not None
            and position_ids is not None
            and hidden_states.shape[1] > 1
        ):
            hs_for_idx = hidden_states
            if qkv_fused_rmsnorm is not None:
                hs_for_idx = self.input_layernorm(hidden_states)
            # Get cos/sin for the indexer's RoPE (uses same base as main RoPE)
            idx_cos, idx_sin = self.self_attn.rotary_emb(hs_for_idx, position_ids)
            attention_mask = self.self_attn.indexer(
                hs_for_idx, idx_cos, idx_sin, position_ids
            )

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        if self.is_moe:
            if not self.moe_fused_nki_kernel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)
            # M3 SwiGLU-OAI MoE forward:
            #   out = routed_scaling_factor * routed_experts(x) + shared_experts(x)
            # block_sparse_moe with `n_shared_experts=0` returns just the routed
            # branch in the first output, so scaling it is safe. shared_experts
            # is our `MiniMaxM3DenseMLP` (SwiGLU-OAI) at shared_intermediate_size.
            routed = self.block_sparse_moe(hidden_states, padding_mask)[0]
            mlp_out = routed * self.routed_scaling_factor
            if self.shared_experts is not None:
                mlp_out = mlp_out + self.shared_experts(hidden_states)[0]
            hidden_states = mlp_out
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)[0]

        hidden_states = residual + hidden_states
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# -----------------------------------------------------------------------------
# Model & CausalLM
# -----------------------------------------------------------------------------
class NeuronMiniMaxM3Model(NeuronBaseModel):
    def setup_attr_for_model(self, config: MiniMaxM3InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: MiniMaxM3InferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronMiniMaxM3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


def _dequantize_mxfp8_state_dict(state_dict: dict, block_size: int = 32) -> dict:
    """
    Dequantize MXFP8 weights to BF16 in-place.

    The MiniMax-M3-MXFP8 release stores each quantized weight as a pair:
      - `<name>.weight`:           float8_e4m3fn tensor of shape (M, N)
      - `<name>.weight_scale_inv`: uint8 (E8M0) MX scales of shape (M, N // 32)

    Each row of the original weight is divided into blocks of 32 elements;
    each block has one 8-bit unsigned-exponent scale. Dequantization is:
        bf16 = fp8.to(bf16) * 2 ** (scale_uint8 - 127)

    NxDI's mainstream MoE / GQA path doesn't yet consume on-device MXFP8
    GeMM, so we dequant once on host before sharding. This loses the HBM
    savings (weights occupy bf16 on device) but keeps the conversion path
    working with the rest of the pipeline. The disk and download cost saving
    from MXFP8 (~half of bf16) still applies.
    """
    scale_keys = [k for k in state_dict if k.endswith(".weight_scale_inv")]
    if not scale_keys:
        return state_dict

    logger.info("Dequantizing %d MXFP8 weights to BF16 (block_size=%d)...",
                len(scale_keys), block_size)

    for scale_key in scale_keys:
        weight_key = scale_key.replace(".weight_scale_inv", ".weight")
        if weight_key not in state_dict:
            del state_dict[scale_key]
            continue

        weight = state_dict[weight_key]
        scale = state_dict[scale_key]

        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            del state_dict[scale_key]
            continue

        # Expand scales along the block axis: scale[..., j] applies to
        # weight[..., j*block_size : (j+1)*block_size].
        # MXFP8 uses E8M0 (uint8) scales where actual_scale = 2 ** (e - 127).
        bf16 = weight.to(torch.float32)
        # Reshape last dim into (num_blocks, block_size) and broadcast scale.
        *prefix, n = bf16.shape
        if n % block_size != 0:
            # Some scale dims may not divide cleanly (e.g. final block padded);
            # fall back to per-element scaling of the prefix that does divide.
            n_blocks = n // block_size
            n_kept = n_blocks * block_size
            head = bf16[..., :n_kept].reshape(*prefix, n_blocks, block_size)
            scale_head = scale[..., :n_blocks].to(torch.int32) - 127
            head = torch.ldexp(head, scale_head.unsqueeze(-1)).reshape(*prefix, n_kept)
            bf16 = torch.cat([head, bf16[..., n_kept:]], dim=-1)
        else:
            n_blocks = n // block_size
            bf16 = bf16.reshape(*prefix, n_blocks, block_size)
            exp = scale.to(torch.int32) - 127
            bf16 = torch.ldexp(bf16, exp.unsqueeze(-1))
            bf16 = bf16.reshape(*prefix, n)

        state_dict[weight_key] = bf16.to(torch.bfloat16)
        del state_dict[scale_key]

    gc.collect()
    logger.info("MXFP8 dequantization complete.")
    return state_dict


def convert_minimax_m3_hf_to_neuron_state_dict(
    state_dict: dict,
    config: MiniMaxM3InferenceConfig,
) -> dict:
    """
    Convert a HuggingFace MiniMax-M3 (text) state dict to the Neuron layout.

    Key transformations:
      0. Dequantize MXFP8 weights to BF16 (if present).
      1. Strip the `language_model.model.` / `language_model.` prefix from the
         multimodal checkpoint so layers live at `layers.<i>.*`.
      2. Drop any vision-tower / projector / MTP weights.
      3. MoE: rename `block_sparse_moe.gate.weight` to
         `block_sparse_moe.router.linear_router.weight`, fuse per-expert
         `w1` (gate) and `w3` (up) into a single `gate_up_proj` of shape
         `[num_experts, hidden, 2 * intermediate]`, stack `w2` (down) into
         `down_proj` of shape `[num_experts, intermediate, hidden]`.
      4. Shared experts: move
         `block_sparse_moe.shared_experts.{gate,up,down}_proj` to
         the shape NxDI SharedExperts expects (col-parallel gate/up,
         row-parallel down).
      5. Drop the `e_score_correction_bias` (routing bias) for now — the MVP
         path does not consume it (router has no bias).
      6. Dense layers (first `first_k_dense_replace` layers): fuse
         `mlp.gate_proj` + `mlp.up_proj` into a single `mlp.gate_up_proj` so
         we can use a single ColumnParallelLinear.
      7. Add `rank_util.rank` tensors for tensor-parallel sharding.
    """
    # Dequantize MXFP8 if present. Pull block size from the HF quantization
    # config when available (M3-MXFP8 uses [1, 32]).
    quant_config = getattr(config, "quantization_config", None)
    block_size = 32
    if isinstance(quant_config, dict):
        wbs = quant_config.get("weight_block_size", [1, 32])
        if isinstance(wbs, (list, tuple)) and len(wbs) >= 2:
            block_size = wbs[1]
    _dequantize_mxfp8_state_dict(state_dict, block_size=block_size)
    neuron_config = config.neuron_config
    tp_degree = neuron_config.tp_degree
    num_layers = config.num_hidden_layers
    num_experts = config.num_local_experts
    hidden_size = config.hidden_size
    moe_inter = config.moe_intermediate_size
    dense_inter = config.dense_intermediate_size
    shared_inter = getattr(config, "shared_intermediate_size", 0)
    first_k_dense = getattr(config, "first_k_dense_replace", 3)
    moe_layer_freq = config.moe_layer_freq

    new_state = {}

    def _strip_prefix(k: str) -> Optional[str]:
        for prefix in ("language_model.model.", "language_model."):
            if k.startswith(prefix):
                return k[len(prefix):]
        if k.startswith("model."):
            return k[len("model."):]
        return k

    def _is_skip(k: str) -> bool:
        # vision_tower, multi_modal_projector, patch_merge_mlp, mtp, etc.
        # Indexer weights (`index_q_proj`, `index_k_proj`, `index_q_norm`,
        # `index_k_norm`) are KEPT — needed for MSA.
        for tag in (
            "vision_tower",
            "multi_modal_projector",
            "patch_merge_mlp",
            ".mtp.",
            "mtp.",
        ):
            if tag in k:
                return True
        return False

    # First pass: collect non-MoE / non-dense-MLP weights with renaming.
    expert_w_buf = {}  # layer -> {"w1": list of (e, tensor), "w2": ..., "w3": ...}
    dense_mlp_buf = {}  # layer -> {"gate": t, "up": t, "down": t}
    shared_mlp_buf = {}  # layer -> {"gate_proj": t, "up_proj": t, "down_proj": t}

    for raw_key, value in list(state_dict.items()):
        if _is_skip(raw_key):
            continue
        key = _strip_prefix(raw_key)
        if key is None:
            continue

        # Rename e_score_correction_bias to the RouterTopKWithBias parameter.
        if "block_sparse_moe.e_score_correction_bias" in key:
            new_key = key.replace(
                "block_sparse_moe.e_score_correction_bias",
                "block_sparse_moe.router.e_score_correction_bias",
            )
            # Cast to FP32 to match the RouterTopKWithBias init dtype. HF M3
            # adds bias to sigmoid(logits).float() in fp32; using bf16 drops
            # ~7 bits of precision in the +bias add and shifts topk choices.
            new_state[new_key] = value.detach().to(torch.float32).clone()
            continue

        # MoE expert weights: collect for later fusion.
        if "block_sparse_moe.experts." in key:
            # layers.<l>.block_sparse_moe.experts.<e>.<w1|w2|w3>.weight
            parts = key.split(".")
            l_idx = int(parts[1])
            e_idx = int(parts[4])
            w_name = parts[5]  # w1, w2, or w3
            expert_w_buf.setdefault(l_idx, {}).setdefault(w_name, [None] * num_experts)
            expert_w_buf[l_idx][w_name][e_idx] = value
            continue

        # Dense MLP gate/up/down: collect for fusion.
        if ".mlp.gate_proj.weight" in key:
            l_idx = int(key.split(".")[1])
            dense_mlp_buf.setdefault(l_idx, {})["gate"] = value
            continue
        if ".mlp.up_proj.weight" in key:
            l_idx = int(key.split(".")[1])
            dense_mlp_buf.setdefault(l_idx, {})["up"] = value
            continue
        if ".mlp.down_proj.weight" in key:
            l_idx = int(key.split(".")[1])
            dense_mlp_buf.setdefault(l_idx, {})["down"] = value
            continue

        # Rename MoE router weight.
        if "block_sparse_moe.gate.weight" in key:
            new_key = key.replace(
                "block_sparse_moe.gate.weight",
                "block_sparse_moe.router.linear_router.weight",
            )
            new_state[new_key] = value.detach().clone()
            continue

        # M3 stores per-head Q/K norms as `self_attn.q_norm` / `self_attn.k_norm`.
        # NeuronAttentionBase expects them as `q_layernorm` / `k_layernorm`.
        # Order matters — check `index_q_norm` before `q_norm`.
        if ".self_attn.index_q_norm." in key:
            new_key = key.replace(".self_attn.index_q_norm.", ".self_attn.indexer.q_norm.")
            new_state[new_key] = value.detach().clone()
            continue
        if ".self_attn.index_k_norm." in key:
            new_key = key.replace(".self_attn.index_k_norm.", ".self_attn.indexer.k_norm.")
            new_state[new_key] = value.detach().clone()
            continue
        if ".self_attn.index_q_proj." in key:
            new_key = key.replace(".self_attn.index_q_proj.", ".self_attn.indexer.q_proj.")
            new_state[new_key] = value.detach().clone()
            continue
        if ".self_attn.index_k_proj." in key:
            new_key = key.replace(".self_attn.index_k_proj.", ".self_attn.indexer.k_proj.")
            new_state[new_key] = value.detach().clone()
            continue
        if ".self_attn.q_norm." in key:
            new_key = key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
            new_state[new_key] = value.detach().clone()
            continue
        if ".self_attn.k_norm." in key:
            new_key = key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")
            new_state[new_key] = value.detach().clone()
            continue

        # Shared experts: M3 stores them as a sibling of the MoE under
        # `block_sparse_moe.shared_experts.{gate,up,down}_proj`. We move them
        # to a sibling of `block_sparse_moe` (a separate `shared_experts`
        # attribute on the decoder layer) and fuse gate/up into gate_up_proj
        # to match `MiniMaxM3DenseMLP`'s `ColumnParallelLinear(2*inter)`.
        if "block_sparse_moe.shared_experts." in key:
            parts = key.split(".")
            # key: layers.<l>.block_sparse_moe.shared_experts.<proj>.weight
            l_idx = int(parts[1])
            proj = parts[4]
            shared_mlp_buf.setdefault(l_idx, {})[proj] = value
            continue

        # Pass-through (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
        # input_layernorm, post_attention_layernorm, embed_tokens, lm_head,
        # final norm, etc).
        new_state[key] = value.detach().clone()

    # Fuse expert weights per MoE layer.
    for l_idx, w_dict in expert_w_buf.items():
        assert "w1" in w_dict and "w2" in w_dict and "w3" in w_dict, (
            f"Layer {l_idx}: missing expert weight tensors"
        )

        w1_list = w_dict["w1"]
        w2_list = w_dict["w2"]
        w3_list = w_dict["w3"]

        assert all(t is not None for t in w1_list), (
            f"Layer {l_idx}: gate (w1) weights have gaps across experts"
        )
        sample = w1_list[0]
        dtype = sample.dtype
        device = sample.device

        gate_up = torch.empty(
            num_experts, hidden_size, 2 * moe_inter, dtype=dtype, device=device
        )
        for e in range(num_experts):
            w1 = w1_list[e].T.detach().clone()  # (hidden, intermediate)
            w3 = w3_list[e].T.detach().clone()  # (hidden, intermediate)
            gate_up[e, :, :moe_inter] = w1
            gate_up[e, :, moe_inter:] = w3

        down = torch.empty(
            num_experts, moe_inter, hidden_size, dtype=dtype, device=device
        )
        for e in range(num_experts):
            down[e] = w2_list[e].T.detach().clone()

        new_state[
            f"layers.{l_idx}.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight"
        ] = gate_up
        new_state[
            f"layers.{l_idx}.block_sparse_moe.expert_mlps.mlp_op.down_proj.weight"
        ] = down

    # Fused dense MLP gate+up. The ColumnParallelLinear uses stride=2 so
    # the global `[gate | up]` layout shards correctly across ranks.
    for l_idx, parts in dense_mlp_buf.items():
        if "gate" not in parts or "up" not in parts or "down" not in parts:
            continue
        gate = parts["gate"].detach().clone()  # (dense_inter, hidden)
        up = parts["up"].detach().clone()
        gate_up = torch.cat([gate, up], dim=0)  # (2 * dense_inter, hidden)
        new_state[f"layers.{l_idx}.mlp.gate_up_proj.weight"] = gate_up
        new_state[f"layers.{l_idx}.mlp.down_proj.weight"] = parts["down"].detach().clone()

    # Fused shared expert gate+up (same stride=2 pattern).
    for l_idx, parts in shared_mlp_buf.items():
        if "gate_proj" not in parts or "up_proj" not in parts or "down_proj" not in parts:
            continue
        gate = parts["gate_proj"].detach().clone()
        up = parts["up_proj"].detach().clone()
        gate_up = torch.cat([gate, up], dim=0)
        new_state[f"layers.{l_idx}.shared_experts.gate_up_proj.weight"] = gate_up
        new_state[f"layers.{l_idx}.shared_experts.down_proj.weight"] = (
            parts["down_proj"].detach().clone()
        )

    # Rank utilities for TP.
    new_state["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
    for l in range(num_layers):
        new_state[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )

    # Optionally fuse Q/K/V into Wqkv. NxDI's GroupQueryAttention_QKV nests it
    # under `self_attn.qkv_proj.Wqkv` (M2 pattern), and the preshard hook
    # finds the source weights at `self_attn.{q,k,v}_proj` and renames them
    # under `qkv_proj`. With fused_qkv=True we pre-concat to avoid a host-side
    # split-on-load.
    if getattr(neuron_config, "fused_qkv", False):
        for l in range(num_layers):
            q_key = f"layers.{l}.self_attn.q_proj.weight"
            k_key = f"layers.{l}.self_attn.k_proj.weight"
            v_key = f"layers.{l}.self_attn.v_proj.weight"
            if q_key in new_state and k_key in new_state and v_key in new_state:
                fused = torch.cat(
                    [new_state[q_key], new_state[k_key], new_state[v_key]], dim=0
                )
                new_state[f"layers.{l}.self_attn.qkv_proj.Wqkv.weight"] = fused
                del new_state[q_key]
                del new_state[k_key]
                del new_state[v_key]

    # Gemma-style pre-shift: M3 uses RMSNorm with scale `(1 + weight)` for
    # every norm (input_layernorm, post_attention_layernorm, q_layernorm,
    # k_layernorm, final `norm.weight`). NxDI's fused TKG attention kernel
    # and `CustomRMSNorm` both apply plain `x_norm * w`, so we bake the
    # +1.0 into the loaded weights. Matches the trick in
    # `neuronx_distributed_inference.models.gemma3.modeling_gemma3` lines
    # 374, 393-394 (`state_dict["...norm.weight"] += 1.0`).
    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_layernorm.weight",
        ".k_layernorm.weight",
        ".indexer.q_norm.weight",
        ".indexer.k_norm.weight",
    )
    shifted = 0
    for k in list(new_state.keys()):
        if k.endswith(norm_suffixes) or k == "norm.weight":
            new_state[k] = new_state[k].detach().to(torch.float32) + 1.0
            new_state[k] = new_state[k].to(torch.bfloat16)
            shifted += 1
    print(f"[M3] Pre-shifted +1.0 on {shifted} RMSNorm weights")

    gc.collect()
    return new_state


class NeuronMiniMaxM3ForCausalLM(NeuronBaseForCausalLM):
    """MiniMax-M3 text backbone for causal LM on Neuron."""

    _model_cls = NeuronMiniMaxM3Model

    @staticmethod
    def load_hf_model(model_path: str, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @classmethod
    def get_config_cls(cls):
        return MiniMaxM3InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: MiniMaxM3InferenceConfig
    ) -> dict:
        return convert_minimax_m3_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        args = "--enable-saturate-infinity --enable-mixed-precision-accumulation"
        args += " --model-type transformer -O1"
        args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
        args += f" --lnc={self.config.neuron_config.logical_nc_config}"
        return args


__all__ = [
    "MiniMaxM3InferenceConfig",
    "MiniMaxM3NeuronConfig",
    "NeuronMiniMaxM3Attention",
    "MiniMaxM3DenseMLP",
    "NeuronMiniMaxM3DecoderLayer",
    "NeuronMiniMaxM3Model",
    "NeuronMiniMaxM3ForCausalLM",
]
