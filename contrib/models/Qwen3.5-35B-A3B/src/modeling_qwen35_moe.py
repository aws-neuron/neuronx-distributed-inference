"""
NxDI contrib: Qwen3.5-35B-A3B (qwen3_5_moe / qwen3_next)

Hybrid DeltaNet + Standard Attention + MoE architecture.
Based on NxDI Qwen3-MoE with custom DeltaNet layers.

30 of 40 layers use Gated DeltaNet (linear recurrent attention)
10 of 40 layers use standard GQA with KV cache + output gate
All 40 layers use sparse MoE (256 experts, top-8 + shared expert with sigmoid gate)

Architecture details:
- DeltaNet layers: separate in_proj_{qkv, z, a, b}, causal conv1d on QKV, gated delta rule
- Attention layers: q_proj doubled (Q + gate), partial RoPE (25% of head_dim), sigmoid output gate
- MoE: pre-fused expert weights, shared expert with sigmoid gate
- KV cache: NxDI KVCacheManager for attention layers; DeltaNet layers store recurrent+conv
  state as nn.Parameter buffers and return dummy KV tuples
"""

import gc
import math
import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch_neuronx.xla_impl.ops import nki_jit
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

try:
    from .nki_deltanet import deltanet_recurrent_fwd as _deltanet_nki_kernel
    from .nki_deltanet import deltanet_recurrent_fwd_state as _deltanet_nki_kernel_state
except ImportError:
    from nki_deltanet import deltanet_recurrent_fwd as _deltanet_nki_kernel
    from nki_deltanet import deltanet_recurrent_fwd_state as _deltanet_nki_kernel_state

# Custom NKI flash attention kernel for head_dim=256
# (standard NxDI kernel asserts head_dim <= 128)
# Opt-in only: set QWEN35_USE_FLASH_ATTN_D256=1 to enable
try:
    from .nki_flash_attn_d256 import flash_attn_d256 as _flash_attn_d256_raw
except ImportError:
    try:
        from nki_flash_attn_d256 import flash_attn_d256 as _flash_attn_d256_raw
    except ImportError:
        _flash_attn_d256_raw = None

if _flash_attn_d256_raw is not None:
    _flash_attn_d256_kernel = nki_jit()(_flash_attn_d256_raw)
else:
    _flash_attn_d256_kernel = None

_USE_FLASH_ATTN_D256 = os.environ.get("QWEN35_USE_FLASH_ATTN_D256", "0") == "1"

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
    MOE_TKG_MK_INTERMEDIATE_PER_TP,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    DecoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

logger = logging.getLogger(__name__)

_flash_fwd_call = nki_jit()(attention_isa_kernel)

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


def get_rmsnorm_cls():
    return Qwen3MoeRMSNorm if cpu_mode() else CustomRMSNorm


def l2norm(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)


# ============================================================
# Gated DeltaNet Module (Linear Recurrent Attention)
# ============================================================


class NeuronGatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention for Neuron.

    Replaces standard attention for 30 of 40 layers.
    Uses NKI kernel for context encoding (CTE) and PyTorch recurrent step for
    token generation (TKG).

    State carry-over between CTE and TKG is handled via nn.Parameter buffers
    and input_output_aliases:
    - recurrent_state_buffer: (B, num_v_heads, k_dim, v_dim) carries the
      recurrent state (delta rule memory matrix) across graphs.
    - conv_state_buffer: (B, conv_dim, kernel_size-1) carries the last 3
      tokens of QKV concatenation for causal conv1d context.

    DeltaNet layers return dummy (K, V) tuples so KVCacheManager can process
    them without crashing. The dummy values are never read back.

    HF weight layout:
    - in_proj_qkv.weight: (key_dim*2 + value_dim, hidden_size) = (8192, 2048)
    - in_proj_z.weight: (value_dim, hidden_size) = (4096, 2048)
    - in_proj_a.weight: (num_v_heads, hidden_size) = (32, 2048)
    - in_proj_b.weight: (num_v_heads, hidden_size) = (32, 2048)
    - conv1d.weight: (conv_dim, 1, conv_kernel_size) = (8192, 1, 4)
    - A_log: (num_v_heads,) = (32,)
    - dt_bias: (num_v_heads,) = (32,)
    - norm.weight: (head_v_dim,) = (128,)
    - out_proj.weight: (hidden_size, value_dim) = (2048, 4096)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        tc = config

        self.hidden_size = tc.hidden_size  # 2048
        self.num_v_heads = tc.linear_num_value_heads  # 32
        self.num_k_heads = tc.linear_num_key_heads  # 16
        self.head_k_dim = tc.linear_key_head_dim  # 128
        self.head_v_dim = tc.linear_value_head_dim  # 128
        self.key_dim = self.head_k_dim * self.num_k_heads  # 2048
        self.value_dim = self.head_v_dim * self.num_v_heads  # 4096
        self.conv_kernel_size = tc.linear_conv_kernel_dim  # 4
        self.layer_idx = layer_idx
        self.rms_norm_eps = tc.rms_norm_eps

        # KV cache dummy shape info (for returning proper-shaped zeros)
        # Must match KVCacheManager's per-rank shape: (B, num_kv_heads_per_rank, seq_len, head_dim)
        # With REPLICATE_TO_TP_DEGREE: raw KV heads (2) replicated to tp_degree (4), then /tp = 1 per rank
        self.head_dim = tc.head_dim  # 256
        tp_degree = tc.neuron_config.tp_degree
        raw_kv_heads = tc.num_key_value_heads
        # Replicate KV heads to tp_degree if fewer, then divide
        if raw_kv_heads < tp_degree:
            replicated_kv_heads = tp_degree  # REPLICATE_TO_TP_DEGREE strategy
        else:
            replicated_kv_heads = raw_kv_heads
        self.kv_heads_per_rank = replicated_kv_heads // tp_degree

        # Conv1d on concatenated QKV (NOT Z)
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 8192
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Input projections -- separate to match HF weight layout
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Decay parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        # Output norm and projection
        # Use standard RMSNorm (not CustomRMSNorm) since DeltaNet is custom code
        # and we need it to work in both CPU mode and Neuron tracing.
        # The Qwen3MoeRMSNorm is a plain PyTorch RMSNorm that works everywhere.
        self.norm = Qwen3MoeRMSNorm(self.head_v_dim, eps=self.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # ---- State buffers for CTE -> TKG carry-over ----
        # These are nn.Parameter(requires_grad=False) so they participate in
        # input_output_aliases, allowing the XLA runtime to alias the output
        # tensors back to the same HBM buffers across CTE and TKG graphs.
        #
        # Recurrent state: (B, num_v_heads, k_dim, v_dim) = (1, 32, 128, 128)
        # The NKI kernel outputs per-head (128, 128) in float32; we store as bf16
        # on HBM and cast at load/store time.
        #
        # Conv state: last (kernel_size - 1) = 3 tokens of the mixed tensor
        # (QKV concat before conv1d). Shape: (B, conv_dim, kernel_size - 1)
        # = (1, 8192, 3). Stores the last 3 tokens' mixed values so TKG can
        # compute conv1d correctly for the next token.
        #
        # Note: batch_size is determined at config time. We default to 1.
        # Both buffers are stored in the model's compute dtype (bf16).
        batch_size = getattr(config.neuron_config, "max_batch_size", 1)
        self.recurrent_state_buffer = nn.Parameter(
            torch.zeros(
                batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=config.neuron_config.torch_dtype,
            ),
            requires_grad=False,
        )
        self.conv_state_buffer = nn.Parameter(
            torch.zeros(
                batch_size,
                self.conv_dim,
                self.conv_kernel_size - 1,  # 3
                dtype=config.neuron_config.torch_dtype,
            ),
            requires_grad=False,
        )

    def _recurrent_step(self, query, key, value, g, beta, recurrent_state):
        """Single-step recurrent update for token generation.

        Args:
            query: (B, H, 1, k_dim)  [H = num_v_heads after K-head expansion]
            key:   (B, H, 1, k_dim)
            value: (B, H, 1, v_dim)
            g:     (B, H, 1) -- log-decay
            beta:  (B, H, 1) -- write gate
            recurrent_state: (B, H, k_dim, v_dim)

        Returns:
            output: (B, H, 1, v_dim)
            new_state: (B, H, k_dim, v_dim)
        """
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale

        q_t = query[:, :, 0]  # (B, H, k_dim)
        k_t = key[:, :, 0]
        v_t = value[:, :, 0]  # (B, H, v_dim)
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        beta_t = beta[:, :, 0].unsqueeze(-1)  # (B, H, 1)

        # Decay old state
        new_state = recurrent_state * g_t
        # Compute delta update
        kv_mem = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, v_dim)
        delta = (v_t - kv_mem) * beta_t
        new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # Read out
        output = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, v_dim)

        return output.unsqueeze(2), new_state

    def _nki_recurrent_forward(self, query, key, value, g, beta):
        """Full-sequence recurrent forward using NKI kernel for context encoding.

        Uses the _state variant kernel to also return the final recurrent state
        for CTE -> TKG carry-over.

        The NKI kernel processes a single (batch, head) pair's full sequence.
        We call it in a loop over B*H from PyTorch.

        Args:
            query: (B, H, S, k_dim) float32
            key:   (B, H, S, k_dim) float32
            value: (B, H, S, v_dim) float32
            g:     (B, H, S) float32 -- log-decay
            beta:  (B, H, S) float32 -- write gate

        Returns:
            output:      (B, H, S, v_dim) float32
            final_state: (B, H, k_dim, v_dim) float32 -- recurrent state after last token
        """
        # L2-normalize and scale
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        # Flatten (B, H) -> BH for looping
        BH = B * H
        query_flat = query.reshape(BH, S, k_dim).contiguous()
        key_flat = key.reshape(BH, S, k_dim).contiguous()
        value_flat = value.reshape(BH, S, v_dim).contiguous()

        # Expand g/beta from (B, H, S) to (BH, S, 128) for NKI --
        # tensor_scalar requires operand0 shape (P_MAX, 1) matching partition axis.
        g_flat = g.reshape(BH, S).unsqueeze(-1).expand(-1, -1, v_dim).contiguous()
        beta_flat = beta.reshape(BH, S).unsqueeze(-1).expand(-1, -1, v_dim).contiguous()

        # Call NKI kernel per (batch, head) pair -- 2D tensors (S, 128)
        outputs = []
        states = []
        for bh in range(BH):
            out_bh, state_bh = _deltanet_nki_kernel_state(
                query_flat[bh],  # (S, 128)
                key_flat[bh],  # (S, 128)
                value_flat[bh],  # (S, 128)
                g_flat[bh],  # (S, 128)
                beta_flat[bh],  # (S, 128)
            )
            outputs.append(out_bh)
            states.append(state_bh)  # (128, 128)

        # Stack back to (BH, S, v_dim) then reshape to (B, H, S, v_dim)
        output = torch.stack(outputs, dim=0)  # (BH, S, v_dim)
        output = output.reshape(B, H, S, v_dim)

        # Stack states to (BH, k_dim, v_dim) then reshape to (B, H, k_dim, v_dim)
        final_state = torch.stack(states, dim=0)  # (BH, k_dim, v_dim)
        final_state = final_state.reshape(B, H, k_dim, v_dim)

        return output, final_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        """Forward pass compatible with NxDI decoder layer interface.

        DeltaNet layers do NOT use KV cache. We return dummy (K, V) tuples
        with proper shapes so KVCacheManager.update_cache() can process them
        without crashing. The dummy values are zeros and will be written to
        the cache slots allocated for this layer but never read back.

        State carry-over:
        - recurrent_state_buffer: (B, 32, 128, 128) nn.Parameter, aliased via
          input_output_aliases. CTE writes final state; TKG reads it.
        - conv_state_buffer: (B, 8192, 3) nn.Parameter, aliased. CTE writes
          last 3 tokens of pre-silu mixed; TKG uses for conv1d context.

        CRITICAL: For CTE, NxDI pads input to bucket size (e.g., 128 tokens).
        DeltaNet has no attention mask -- the recurrence processes ALL positions.
        We must zero out padding positions before projection to prevent pad tokens
        from corrupting the recurrent state. The attention_mask from NxDI is
        typically (B, 1, 1, S) or (B, 1, S, S) with 0 for valid, large negative
        for padding.

        Returns:
            output: (B, S, hidden_size)
            dummy_kv: tuple(K_dummy, V_dummy) with proper shapes
            new_recurrent_state: (B, 32, 128, 128) updated recurrent state buffer
            new_conv_state: (B, 8192, 3) updated conv state buffer
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Determine mode: context encoding (prefill) vs token generation (decode)
        is_decode = past_key_value is not None

        # --- Mask padding tokens for DeltaNet ---
        # NxDI passes attention_mask as BOOLEAN (B, 1, S, S) where True=valid, False=pad.
        # DeltaNet has no attention mask in its recurrence -- padding tokens contaminate
        # the recurrent state. We zero out padding positions before projection.
        #
        # Save valid_mask_1d for later use to:
        #   1. Zero out g (decay) for padding positions -- otherwise padding tokens
        #      decay the recurrent state towards zero (exp(-1.3) ~ 0.27 per token)
        #   2. Save conv_state from the last 3 VALID positions, not last 3 absolute
        #      positions (which may be padding with right-padding)
        valid_mask_1d = None  # (B, S) float, 1.0 for valid, 0.0 for padding
        if attention_mask is not None and not is_decode:
            if attention_mask.dim() == 4:
                # Boolean 4D causal mask: (B, 1, S, S)
                # Use the LAST ROW to get per-position validity.
                pad_mask_1d = attention_mask[:, 0, -1, :]  # (B, S) bool
            elif attention_mask.dim() == 2:
                pad_mask_1d = attention_mask  # (B, S) bool or int
            else:
                pad_mask_1d = None

            if pad_mask_1d is not None:
                valid_mask = pad_mask_1d.to(hidden_states.dtype)  # (B, S) in bf16
                valid_mask_1d = valid_mask  # Save for later g masking and conv_state
                hidden_states = hidden_states * valid_mask.unsqueeze(-1)  # (B, S, D)

        # Project inputs
        qkv = self.in_proj_qkv(hidden_states)  # (B, S, 8192)
        z = self.in_proj_z(hidden_states)  # (B, S, 4096)
        b = self.in_proj_b(hidden_states)  # (B, S, 32)
        a = self.in_proj_a(hidden_states)  # (B, S, 32)

        # Split QKV
        query = qkv[..., : self.key_dim]  # (B, S, 2048)
        key = qkv[..., self.key_dim : self.key_dim * 2]  # (B, S, 2048)
        value = qkv[..., self.key_dim * 2 :]  # (B, S, 4096)

        # Causal Conv1d on QKV (NOT on Z)
        mixed = torch.cat([query, key, value], dim=-1)  # (B, S, 8192)
        mixed = mixed.transpose(1, 2)  # (B, 8192, S)

        if is_decode:
            # TKG: Use conv_state_buffer for causal conv1d context.
            # conv_state_buffer holds the last 3 tokens of pre-silu mixed from CTE/prev TKG.
            # We need 4 consecutive values for conv1d kernel_size=4.
            # Build window: [conv_state[:, :, 0:3], new_token] = (B, 8192, 4)
            conv_state = self.conv_state_buffer  # (B, 8192, 3)
            conv_input = torch.cat([conv_state, mixed], dim=-1)  # (B, 8192, 4)

            # Apply depthwise conv1d manually (kernel_size=4, groups=conv_dim):
            # out = sum_{k=0}^{3} w[:, k] * input[:, :, k]
            w = self.conv1d.weight.squeeze(1)  # (8192, 4)
            conv_out = torch.zeros_like(mixed)  # (B, 8192, 1)
            for k in range(4):
                conv_out = (
                    conv_out
                    + w[:, k].unsqueeze(0).unsqueeze(-1) * conv_input[:, :, k : k + 1]
                )
            mixed_post_conv = F.silu(conv_out)

            # Update conv state: shift left, append new token
            # new_conv_state = [conv_state[:, :, 1:3], mixed[:, :, 0:1]] = (B, 8192, 3)
            new_conv_state = torch.cat(
                [conv_state[:, :, 1:], mixed], dim=-1
            )  # (B, 8192, 3)
        else:
            # CTE: Use nn.Conv1d with built-in padding (proven correct).
            # self.conv1d has padding=kernel_size-1=3, which pads both sides symmetrically.
            # Truncating to [:, :, :seq_len] gives correct causal conv1d output.
            mixed_post_conv = F.silu(self.conv1d(mixed)[:, :, :seq_len])

            # Save last 3 VALID tokens' mixed values for conv_state.
            # With right-padding, valid tokens are at positions 0..n-1, padding at n..S-1.
            # mixed[:, :, -3:] would capture PADDING positions (all zeros) — WRONG.
            # Instead, find the number of valid tokens and gather the last 3.
            if valid_mask_1d is not None:
                # valid_mask_1d: (B, S) float, 1=valid, 0=padding
                # Count valid tokens per batch element
                num_valid = valid_mask_1d.sum(dim=-1, keepdim=True).long()  # (B, 1)
                # Indices for last 3 valid positions: [n-3, n-2, n-1]
                # Clamp to 0 to handle case where num_valid < 3
                idx_base = num_valid - 3  # (B, 1)
                idx_base = idx_base.clamp(min=0)
                offsets = torch.arange(3, device=mixed.device).unsqueeze(0)  # (1, 3)
                gather_idx = idx_base + offsets  # (B, 3)
                # Expand for gather: (B, conv_dim, 3)
                gather_idx = gather_idx.unsqueeze(1).expand(-1, self.conv_dim, -1)
                new_conv_state = torch.gather(mixed, 2, gather_idx)  # (B, 8192, 3)
            else:
                # No mask (shouldn't happen in CTE, but fallback)
                new_conv_state = mixed[:, :, -3:].contiguous()

            # IMPORTANT: Touch conv_state_buffer during CTE so XLA can find it
            # in the lowering context (it's aliased via input_output_aliases).
            # During CTE, we don't USE the old state (nn.Conv1d handles its own
            # padding), but the alias requires the parameter to be part of the
            # traced graph. Adding * 0 ensures the old buffer is read but has
            # no numeric effect on new_conv_state.
            new_conv_state = new_conv_state + self.conv_state_buffer * 0

        mixed_post_conv = mixed_post_conv.transpose(
            1, 2
        )  # (B, S, 8192) or (B, 1, 8192)
        query = mixed_post_conv[..., : self.key_dim]
        key = mixed_post_conv[..., self.key_dim : self.key_dim * 2]
        value = mixed_post_conv[..., self.key_dim * 2 :]

        # Reshape to heads
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # Compute gating
        beta = b.sigmoid()  # (B, S, num_v_heads)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Zero out g for padding positions.
        # g controls the decay factor: exp(g). For padding, g ≈ -1.3 → exp(g) ≈ 0.27.
        # With right-padding, 123 padding tokens after 5 valid tokens would decay state
        # by 0.27^123 ≈ 10^{-70}, effectively zeroing it. Setting g=0 for padding means
        # exp(0)=1, so the state is preserved unchanged through padding positions.
        if valid_mask_1d is not None:
            # valid_mask_1d: (B, S) float bf16, g: (B, S, num_v_heads) float32
            g = g * valid_mask_1d.float().unsqueeze(-1)  # Zero g for padding positions

        # Expand K heads to match V heads (16 -> 32) using expand+reshape
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads  # 2
            query = (
                query.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )
            key = (
                key.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )

        # Transpose to (B, H, S, dim) for delta rule
        query = query.transpose(1, 2).contiguous().float()
        key = key.transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        if is_decode:
            # Load recurrent state from buffer (bf16 -> f32)
            recurrent_state = self.recurrent_state_buffer.float()
            output, new_recurrent_state = self._recurrent_step(
                query, key, value, g, beta, recurrent_state
            )
        else:
            # Context encoding with NKI recurrent kernel -- returns (output, final_state)
            output, new_recurrent_state = self._nki_recurrent_forward(
                query, key, value, g, beta
            )
            # IMPORTANT: Touch recurrent_state_buffer during CTE so XLA can find it
            # in the lowering context (it's aliased via input_output_aliases).
            # During CTE, we don't USE the old state (NKI kernel starts from zero),
            # but the alias requires the parameter to be part of the traced graph.
            # Adding * 0 ensures the old buffer is read but has no numeric effect.
            new_recurrent_state = (
                new_recurrent_state + self.recurrent_state_buffer.float() * 0
            )

        # Cast recurrent state back to storage dtype (f32 -> bf16)
        new_recurrent_state = new_recurrent_state.to(hidden_states.dtype)

        # Back to (B, S, H, v_dim) then (B, S, value_dim)
        output = output.transpose(1, 2).contiguous().to(hidden_states.dtype)
        output = output.reshape(batch_size, seq_len, -1)

        # Gated RMSNorm + output projection
        # norm(output) * silu(z)
        z_flat = z.reshape(-1, self.head_v_dim)
        output_flat = output.reshape(-1, self.head_v_dim)
        output_flat = self.norm(output_flat) * F.silu(z_flat)
        output = output_flat.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(output)

        # Build dummy KV for KVCacheManager compatibility.
        dummy_k = torch.zeros(
            batch_size,
            self.kv_heads_per_rank,
            seq_len,
            self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dummy_v = torch.zeros(
            batch_size,
            self.kv_heads_per_rank,
            seq_len,
            self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dummy_kv = (dummy_k, dummy_v)

        return output, dummy_kv, new_recurrent_state, new_conv_state


# ============================================================
# Config
# ============================================================


class Qwen35MoeInferenceConfig(InferenceConfig):
    """Config for Qwen3.5-35B-A3B with hybrid DeltaNet + Attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model-specific attributes from text_config
        self.num_local_experts = self.num_experts
        # Shared expert: save intermediate_size for manual shared expert MLP
        self.shared_expert_intermediate_size = getattr(
            self, "shared_expert_intermediate_size", 512
        )
        # CRITICAL: Set n_shared_experts=0 for NxDI's MoE module.
        # We handle shared experts manually with sigmoid gating in the decoder layer.
        # NxDI adds shared expert output directly without gating, which is incorrect
        # for Qwen3.5 (requires sigmoid gate).
        self.n_shared_experts = 0
        self.intermediate_size = self.moe_intermediate_size

        # Attention output gate
        self.attn_output_gate = getattr(self, "attn_output_gate", True)

        # Partial RoPE
        self.partial_rotary_factor = getattr(self, "partial_rotary_factor", 0.25)
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)  # 64

        # Layer types for hybrid dispatch
        if not hasattr(self, "layer_types"):
            self.layer_types = []
            for _ in range(10):
                self.layer_types.extend(
                    [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ]
                )

        # Standard HF config attributes expected by NxDI base class
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False

        # DeltaNet-specific config
        if not hasattr(self, "linear_num_value_heads"):
            self.linear_num_value_heads = 32
        if not hasattr(self, "linear_num_key_heads"):
            self.linear_num_key_heads = 16
        if not hasattr(self, "linear_key_head_dim"):
            self.linear_key_head_dim = 128
        if not hasattr(self, "linear_value_head_dim"):
            self.linear_value_head_dim = 128
        if not hasattr(self, "linear_conv_kernel_dim"):
            self.linear_conv_kernel_dim = 4

        # MoE config
        self.maybe_pad_intermediate()
        self.enable_moe_fused_nki_kernel()
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"
        self.neuron_config.disable_numeric_cc_token = True
        self.neuron_config.normalize_top_k_affinities = True

    def maybe_pad_intermediate(self):
        moe_tp_degree = self.neuron_config.moe_tp_degree
        I_TP = self.moe_intermediate_size // moe_tp_degree
        if getattr(
            self.neuron_config.blockwise_matmul_config,
            "use_shard_on_intermediate_dynamic_while",
            False,
        ):
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded = (
                    math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                    * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                    * moe_tp_degree
                )
                self.moe_intermediate_pad_size = max(
                    padded - self.moe_intermediate_size, 0
                )
                self.moe_intermediate_size = padded

    def enable_moe_fused_nki_kernel(self):
        I_TP = self.moe_intermediate_size // self.neuron_config.moe_tp_degree
        if (
            getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", False)
            and I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0
        ):
            self.moe_fused_nki_kernel_enabled = True

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "num_attention_heads",
            "num_experts",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "vocab_size",
            # DeltaNet-specific
            "linear_num_value_heads",
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "layer_types",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


# ============================================================
# Attention (standard GQA for 10 of 40 layers)
# With output gate: q_proj is 2x sized, split into (query, gate)
# With partial RoPE: only first rope_dim dimensions get rotary
# ============================================================


class NeuronQwen35Attention(NeuronAttentionBase):
    """Attention with output gate and partial RoPE.

    Implements partial RoPE (25% of head_dim), per-head QK norm,
    and output gate (sigmoid gate applied BEFORE o_proj, matching reference).

    HF weight layout:
    - q_proj.weight: (num_heads * head_dim * 2, hidden_size) = (8192, 2048)
      First half is query, second half is gate
    - k_proj.weight: (num_kv_heads * head_dim, hidden_size) = (512, 2048)
    - v_proj.weight: (num_kv_heads * head_dim, hidden_size) = (512, 2048)
    - o_proj.weight: (hidden_size, num_heads * head_dim) = (2048, 4096)
    - q_norm.weight: (head_dim,) = (256,)
    - k_norm.weight: (head_dim,) = (256,)
    """

    def __init__(self, config: Qwen35MoeInferenceConfig):
        # Partial RoPE: create RotaryEmbedding with rope_dim (64), not full head_dim (256)
        self.rope_dim = config.rope_dim  # 64 = head_dim * partial_rotary_factor

        # Create QK norm modules first (will be passed to base class)
        rms_norm_eps = config.rms_norm_eps
        q_ln = get_rmsnorm_cls()(config.head_dim, rms_norm_eps)
        k_ln = get_rmsnorm_cls()(config.head_dim, rms_norm_eps)

        rotary_emb = RotaryEmbedding(
            self.rope_dim,  # Only 64 dims get rotary embedding
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=rms_norm_eps,
            use_qk_norm=False,
            q_layernorm=q_ln,
            k_layernorm=k_ln,
        )

        # Output gate projection: hidden_size -> num_heads * head_dim (4096)
        # This is populated from the second half of q_proj during state dict conversion.
        # Use ColumnParallelLinear so it gets sharded across TP ranks,
        # matching the per-rank attention output shape.
        self.output_gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            gather_output=False,  # Each rank keeps its shard
            bias=False,
        )

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        """Partial RoPE: only apply rotary embedding to first rope_dim dimensions.

        Q shape: (B, H, S, head_dim) where head_dim=256
        cos/sin shape: (B, S, rope_dim) where rope_dim=64 (from RotaryEmbedding(dim=64))

        Split Q/K along last dim into:
          q_rope (first 64 dims) -- apply RoPE
          q_pass (remaining 192 dims) -- pass through unchanged
        """
        from neuronx_distributed_inference.modules.attention.utils import (
            apply_rotary_pos_emb,
        )

        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

        # Split into rope and pass-through portions
        q_rope = Q[..., : self.rope_dim]  # (B, H, S, 64)
        q_pass = Q[..., self.rope_dim :]  # (B, H, S, 192)
        k_rope = K[..., : self.rope_dim]
        k_pass = K[..., self.rope_dim :]

        # Apply RoPE only to the rope portion
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos_cache, sin_cache)

        # Concatenate back
        Q = torch.cat([q_rope, q_pass], dim=-1)
        K = torch.cat([k_rope, k_pass], dim=-1)

        return Q, K, cos_cache, sin_cache

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask):
        """Override to handle head_dim=256 safely.

        The standard NxDI NKI flash attention kernel asserts head_dim <= 128.
        This override either:
        1. Uses a custom NKI kernel (flash_attn_d256) if explicitly opted in
           via QWEN35_USE_FLASH_ATTN_D256=1 env var, OR
        2. Forces the PyTorch softmax path by temporarily disabling attn_kernel

        The custom kernel is functional but ~2.4x slower than the PyTorch softmax
        path due to layout conversion overhead (BHSD -> BHDS permute+contiguous).
        It is included for reference and future optimization.
        """
        use_custom_kernel = (
            _USE_FLASH_ATTN_D256
            and _flash_attn_d256_kernel is not None
            and self.head_dim == 256
            and q_len % 512 == 0
            and q_len >= 512
        )
        if use_custom_kernel:
            # Kernel expects:
            #   Q: (bs, n_heads, 256, seq_q) -- BHDS
            #   K: (bs, nk_heads, 256, seq_k) -- BHDS
            #   V: (bs, nv_heads, seq_v, 256) -- BHSD
            #   O: (bs, n_heads, seq_q, 256) -- BHSD (pre-allocated output)
            # Input Q, K, V are all BHSD: (B, H, S, D)
            Q_kernel = (
                Q.permute(0, 1, 3, 2).contiguous().to(self.torch_dtype)
            )  # BHSD -> BHDS
            K_kernel = (
                K.permute(0, 1, 3, 2).contiguous().to(self.torch_dtype)
            )  # BHSD -> BHDS
            V_kernel = V.to(self.torch_dtype)  # already BHSD

            # Pre-allocate output tensor (nki_jit kernels don't support return)
            attn_output = torch.zeros(
                bsz,
                self.num_heads,
                q_len,
                self.head_dim,
                dtype=self.torch_dtype,
                device=Q.device,
            )

            # Launch with grid [bs, nk_heads] -- GQA handled inside kernel
            _flash_attn_d256_kernel[bsz, self.num_key_value_heads](
                Q_kernel,
                K_kernel,
                V_kernel,
                attn_output,
                use_causal_mask=(attention_mask is not None),
            )
            # Output is BHSD (B, n_heads, seq_q, 256)
            # Return NONE strategy to signal BHSD layout to the caller
            return attn_output, FlashAttentionStrategy.NONE

        # Default: force PyTorch softmax path for head_dim > 128
        # (standard NxDI NKI kernel asserts head_dim <= 128)
        if self.head_dim > 128:
            saved = self.attn_kernel_enabled
            self.attn_kernel_enabled = False
            result = super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)
            self.attn_kernel_enabled = saved
            return result
        return super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        active_mask=None,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        **kwargs,
    ):
        """Forward with output gate applied BEFORE o_proj.

        Override NeuronAttentionBase.forward() to insert the sigmoid gate
        between the attention output and o_proj, matching the HF reference:
          gate = sigmoid(gate_proj(pre_attn_hidden))
          attn_output = attn_output * gate
          attn_output = o_proj(attn_output)
        """
        bsz, q_len, _ = hidden_states.shape

        # Compute gate from input hidden states (before QKV projection)
        gate = self.output_gate_proj(hidden_states)  # (B, S, num_heads * head_dim)

        # Standard QKV prep (projections, QK norm, RoPE)
        Q, K, V, cos_cache, sin_cache, _residual = self.prep_qkv_tensors(
            position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
        )

        if past_key_value is None:
            # Context encoding (prefill)
            attn_output, _flash_strategy = self.perform_prefill(
                Q, K, V, q_len, bsz, attention_mask
            )
        else:
            # Token generation (decode)
            attn_output = self.compute_for_token_gen(
                Q, K, V, position_ids, past_key_value, attention_mask, active_mask
            )

        # attn_output is (B, H, S, head_dim) -- transpose to (B, S, H, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Apply sigmoid output gate BEFORE o_proj (matching HF reference)
        attn_output = attn_output * torch.sigmoid(gate)

        # Apply o_proj
        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

        past_key_value = (K, V)
        return attn_output, past_key_value, cos_cache, sin_cache


# ============================================================
# Sigmoid-Gated Shared Expert Wrapper
# ============================================================


class SigmoidGatedSharedExperts(nn.Module):
    """Wrapper around NxDI SharedExperts that adds Qwen3.5's sigmoid gate.

    NxDI's MoE._apply_shared_experts calls:
        shared_output = self.shared_experts(full_hidden_states, seq_len)
        output = output + shared_output

    This wrapper intercepts that call and multiplies the shared expert output
    by sigmoid(gate_linear(input)) before returning, so the MoE's addition
    already includes the sigmoid gating.

    The wrapped SharedExperts uses ColumnParallelLinear (gate/up) and
    RowParallelLinear(reduce_output=False) (down), so its output is TP-partial.
    The sigmoid gate (1, hidden_size) operates on full hidden states and produces
    a per-token scalar — multiplying TP-partial output by a scalar is correct.

    Weight layout:
      shared_experts.gate_proj.weight: (intermediate_size, hidden_size) = (512, 2048)
      shared_experts.up_proj.weight:   (intermediate_size, hidden_size) = (512, 2048)
      shared_experts.down_proj.weight: (hidden_size, intermediate_size) = (2048, 512)
      sigmoid_gate.weight:             (1, hidden_size) = (1, 2048)
    """

    def __init__(self, config):
        super().__init__()
        from neuronx_distributed.modules.moe.shared_experts import SharedExperts
        from neuronx_distributed.parallel_layers import parallel_state

        self.shared_experts = SharedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            num_shared_experts=1,
            hidden_act=config.hidden_act,
            dtype=config.neuron_config.torch_dtype,
            reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            fused_gate_up_projection=False,
            sequence_parallel_enabled=False,
        )

        # Sigmoid gate: linear(hidden_size -> 1) applied to full hidden states
        self.sigmoid_gate = nn.Linear(config.hidden_size, 1, bias=False)

    @property
    def sequence_parallel_enabled(self):
        """Expose sequence_parallel_enabled so MoE._apply_shared_experts works."""
        return self.shared_experts.sequence_parallel_enabled

    def preshard_hook(self, model_state_dict, prefix):
        """Delegate preshard to inner SharedExperts."""
        self.shared_experts.preshard_hook(model_state_dict, prefix)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute sigmoid-gated shared expert output.

        Called by MoE._apply_shared_experts as:
            shared_output = self.shared_experts(full_hidden_states, seq_len)

        Args:
            x: (T, H) flattened hidden states (full, not TP-partial)
            seq_len: sequence length

        Returns:
            output: (T, H) sigmoid-gated shared expert output (TP-partial from down_proj)
        """
        # Compute shared expert MLP output (TP-partial)
        shared_output = self.shared_experts(x, seq_len)

        # Apply sigmoid gate: sigmoid(x @ gate_weight.T) -> (T, 1)
        gate_value = torch.sigmoid(self.sigmoid_gate(x))  # (T, 1)
        return shared_output * gate_value


# ============================================================
# Decoder Layer (hybrid dispatch)
# ============================================================


class NeuronQwen35DecoderLayer(nn.Module):
    """Hybrid decoder layer: dispatches to DeltaNet or standard attention.

    Interface contract with NxDI get_model_output:
    - forward() receives: hidden_states, seq_ids, attention_mask, position_ids,
      past_key_value, active_mask, adapter_ids, cos_cache, sin_cache,
      rotary_position_ids, kv_mgr, get_kv_per_layer, update_kv_per_layer,
      idx, is_for_context_encoding, seq_len, residual, local_mask,
      windowed_context_encoding_window_idx, padding_mask, **kwargs
    - forward() returns: (hidden_states, present_key_value, cos_cache, sin_cache, None)
    """

    def __init__(self, config: Qwen35MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx
        self.config = config

        # Attention (DeltaNet or standard GQA)
        if self.layer_type == "linear_attention":
            self.linear_attn = NeuronGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = NeuronQwen35Attention(config=config)

        # MoE (all layers) -- uses NxDI's initialize_moe_module
        # n_shared_experts=0 so NxDI MoE creates no shared experts internally
        self.moe_fused_nki_kernel_enabled = getattr(
            config, "moe_fused_nki_kernel_enabled", False
        )
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config,
                rmsnorm=self.post_attention_layernorm,
                init_tkg_module=True,
            )
        else:
            self.mlp = initialize_moe_module(config=config)

        # Sigmoid-gated shared expert using NxDI's TP-sharded SharedExperts
        # Created separately (not via n_shared_experts) because Qwen3.5's
        # shared_expert_intermediate_size (512) differs from moe_intermediate_size (256).
        # Injected into MoE.shared_experts so _apply_shared_experts handles it.
        # NOTE: We assign directly to mlp.shared_experts (not self.sigmoid_gated_...)
        # so there's only ONE module tree path and weight keys match.
        self.mlp.shared_experts = SigmoidGatedSharedExperts(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        padding_mask=None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            # DeltaNet path -- returns (output, dummy_kv, new_recurrent, new_conv)
            attn_out, dummy_kv, new_rec_state, new_conv_state = self.linear_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
            hidden_states = residual + attn_out
            present_key_value = dummy_kv
            deltanet_states = (new_rec_state, new_conv_state)
            cos_cache, sin_cache = None, None
        else:
            deltanet_states = None
            # Standard attention path (gate is inside self_attn.forward())
            hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
            hidden_states = residual + hidden_states

        # MoE FFN (routed experts + sigmoid-gated shared expert via NxDI MoE)
        residual = hidden_states
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)

        moe_output = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + moe_output

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (
            hidden_states,
            present_key_value,
            cos_cache,
            sin_cache,
            None,
            deltanet_states,
        )
        return outputs


# ============================================================
# Model
# ============================================================


class NeuronQwen35MoeModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: Qwen35MoeInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen35MoeInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronQwen35DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )

    @property
    def _deltanet_state_params(self):
        """Return DeltaNet state nn.Parameters in alias order.

        Order: for each DeltaNet layer, (recurrent_state, conv_state).
        Used by Qwen35DecoderModelInstance to set up input_output_aliases.
        Returns fresh references each time (load_state_dict may replace .data).
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, "linear_attn"):
                params.append(layer.linear_attn.recurrent_state_buffer)
                params.append(layer.linear_attn.conv_state_buffer)
        return params

    def get_model_output(
        self,
        input_ids=None,
        seq_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        active_mask=None,
        inputs_embeds=None,
        prev_hidden=None,
        adapter_ids=None,
        rotary_position_ids=None,
        update_cache=False,
        is_for_context_encoding=False,
        vision_embeddings=None,
        vision_mask=None,
        local_attn_mask=None,
        windowed_context_encoding_window_idx=-1,
        padding_mask=None,
        **kwargs,
    ):
        """Override to collect DeltaNet state tensors from decoder layers.

        Calls the parent get_model_output logic but extracts the 6th element
        (deltanet_states) from each decoder layer's output and collects them
        into a flat list that will be appended to the model output.
        """
        batch_size, seq_length = input_ids.shape[:2]
        if self.config.neuron_config.layer_boundary_markers:
            input_ids = ModuleMarkerStartWrapper()(input_ids)

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        hidden_states = inputs_embeds

        # Get KV cache for TKG
        cache_size = self.n_positions
        if not is_for_context_encoding:
            if self.kv_mgr is not None:
                past_key_values = self.kv_mgr.get_cache(
                    seq_ids=seq_ids,
                    seq_len=cache_size,
                    is_for_context_encoding=is_for_context_encoding,
                    windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                    **kwargs,
                )

        # Decoder layers
        next_decoder_cache = ()
        deltanet_state_tensors = []  # Collect DeltaNet states
        cos_cache = None
        sin_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                rotary_position_ids=rotary_position_ids,
                kv_mgr=self.kv_mgr,
                get_kv_per_layer=False,
                update_kv_per_layer=False,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                residual=None,
                local_mask=local_attn_mask,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                padding_mask=padding_mask,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            kv = layer_outputs[1]
            next_decoder_cache += (kv,)
            cos_cache, sin_cache = layer_outputs[2:4]

            # Collect DeltaNet state tensors (element 5)
            deltanet_states = layer_outputs[5] if len(layer_outputs) > 5 else None
            if deltanet_states is not None:
                # deltanet_states = (new_recurrent_state, new_conv_state)
                deltanet_state_tensors.append(deltanet_states[0])  # recurrent
                deltanet_state_tensors.append(deltanet_states[1])  # conv

        # Update KV cache
        if update_cache:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # Store DeltaNet state tensors for forward() to append to output
        self._deltanet_updated_states = deltanet_state_tensors

        return (hidden_states, next_decoder_cache)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds=None,
        kv_cache=None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
    ):
        """Override base forward to append DeltaNet state tensors to output.

        The base flow builds: [result] + [logits?] + updated_kv_cache
        We add: + deltanet_state_tensors

        The input_output_aliases dict maps each DeltaNet state nn.Parameter
        to its output index, which is after the KV cache entries.
        """
        # Call parent forward to get the standard output
        # We can't call super().forward() because we need to inject deltanet
        # state tensors. Instead, replicate the relevant parts of the base forward.

        prev_hidden = self.set_none_if_empty(prev_hidden)
        adapter_ids = self.set_none_if_empty(adapter_ids)
        accepted_indices = self.set_none_if_empty(accepted_indices)
        current_length = self.set_none_if_empty(current_length)
        medusa_mask = self.set_none_if_empty(medusa_mask)
        scatter_index = self.set_none_if_empty(scatter_index)
        slot_mapping = self.set_none_if_empty(slot_mapping)
        active_block_table = self.set_none_if_empty(active_block_table)
        num_queries = self.set_none_if_empty(num_queries)
        computed_context_lens = self.set_none_if_empty(computed_context_lens)
        tile_q_indices = self.set_none_if_empty(tile_q_indices)
        tile_block_tables = self.set_none_if_empty(tile_block_tables)
        tile_masks = self.set_none_if_empty(tile_masks)
        inputs_embeds = self.set_none_if_empty(inputs_embeds)
        kv_cache = self.set_none_if_empty(kv_cache)
        active_mask = self.set_none_if_empty(active_mask)
        rotary_position_id = self.set_none_if_empty(rotary_position_id)
        vision_embeddings = self.set_none_if_empty(vision_embeddings)
        vision_mask = self.set_none_if_empty(vision_mask)

        is_for_context_encoding = position_ids.shape[-1] != 1 and not (
            hasattr(self.neuron_config, "speculation_length")
            and position_ids.shape[-1] == self.neuron_config.speculation_length
        )

        seq_ids = seq_ids.to(torch.int32)
        attn_mask = attention_mask

        hidden_states, updated_kv_cache = self.get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            update_cache=True,
            is_for_context_encoding=is_for_context_encoding,
            padding_mask=None,
            active_block_table=active_block_table,
            scatter_index=slot_mapping
            if getattr(self, "is_block_kv_layout", False)
            else scatter_index,
        )

        batch_size = input_ids.shape[0]
        if not getattr(self, "sliced_hidden", False):
            if not is_for_context_encoding:
                # Token generation: already (B, 1, H) from position_ids
                pass
            else:
                # Context encoding: take last valid position
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if hasattr(self.lm_head, "pad_size"):
            if self.lm_head.gather_output:
                rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
                world_size = 1
            else:
                from neuronx_distributed.parallel_layers import parallel_state

                rank_id = self.rank_util.get_rank()
                world_size = torch.distributed.get_world_size(
                    group=self.lm_head.tensor_parallel_group
                )
            from neuronx_distributed_inference.models.model_base import (
                mask_padded_logits,
            )

            logits = mask_padded_logits(
                logits, rank_id, world_size, pad_size=self.lm_head.pad_size
            )

        if self.on_device_sampling:
            res = self._sample_on_device(
                logits, sampling_params, False, is_for_context_encoding
            )
        else:
            res = logits

        outputs = [res]
        if self.neuron_config.output_logits:
            outputs += [logits]
        outputs += updated_kv_cache

        # Append DeltaNet state tensors (for input_output_aliases)
        if hasattr(self, "_deltanet_updated_states"):
            outputs += self._deltanet_updated_states

        return outputs


# ============================================================
# State Dict Converter
# ============================================================


def convert_qwen35_hf_to_neuron_state_dict(neuron_state_dict, config):
    """Convert HF Qwen3.5 weights to NxDI format.

    Weight mappings per layer type:

    DeltaNet layers (linear_attention):
      HF: layers.X.linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
          conv1d, A_log, dt_bias, norm, out_proj}
      NxDI: same names (no remapping needed)

    Full attention layers:
      HF: layers.X.self_attn.q_proj.weight: (8192, 2048) -- doubled for gate
      NxDI: layers.X.self_attn.Wqkv.weight (fused Q+K+V, gate separated)
             layers.X.self_attn.output_gate_proj.weight (gate part)
      HF: layers.X.self_attn.{k_proj, v_proj, o_proj, q_norm, k_norm}
      NxDI: layers.X.self_attn.{..., q_layernorm, k_layernorm}

    MoE (all layers):
      HF: layers.X.mlp.gate.weight -> NxDI: layers.X.mlp.router.linear_router.weight
      HF: layers.X.mlp.experts.gate_up_proj -> NxDI: layers.X.mlp.expert_mlps.mlp_op.gate_up_proj.weight
      HF: layers.X.mlp.experts.down_proj -> NxDI: layers.X.mlp.expert_mlps.mlp_op.down_proj.weight
      HF: layers.X.mlp.shared_expert.{gate,up,down}_proj -> NxDI: layers.X.mlp.shared_experts.shared_experts.{gate,up,down}_proj
      HF: layers.X.mlp.shared_expert_gate.weight -> NxDI: layers.X.mlp.shared_experts.sigmoid_gate.weight
    """
    # Add rank_util
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0,
        config.neuron_config.tp_degree,
        dtype=torch.int32,
    )

    # CRITICAL: Convert (1+weight) RMSNorm weights to standard RMSNorm weights.
    # Qwen3.5-MoE uses RMSNorm with `output = norm(x) * (1 + weight)` where weight
    # is initialized to zeros. Standard NxDI RMSNorm uses `output = norm(x) * weight`
    # where weight is initialized to ones. To convert: new_weight = old_weight + 1.0
    # This affects: input_layernorm, post_attention_layernorm, q_norm, k_norm, final norm
    # but NOT the DeltaNet internal RMSNormGated (which uses standard weight * norm(x))
    norm_keys_to_convert = []
    for l in range(config.num_hidden_layers):
        norm_keys_to_convert.append(f"layers.{l}.input_layernorm.weight")
        norm_keys_to_convert.append(f"layers.{l}.post_attention_layernorm.weight")
        if config.layer_types[l] == "full_attention":
            norm_keys_to_convert.append(f"layers.{l}.self_attn.q_norm.weight")
            norm_keys_to_convert.append(f"layers.{l}.self_attn.k_norm.weight")
    norm_keys_to_convert.append("norm.weight")

    for nk in norm_keys_to_convert:
        if nk in neuron_state_dict:
            old_val = neuron_state_dict[nk]
            neuron_state_dict[nk] = old_val.float() + 1.0

    for l in range(config.num_hidden_layers):
        layer_type = config.layer_types[l]

        # === Attention layers ===
        if layer_type == "full_attention":
            neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
                0,
                config.neuron_config.tp_degree,
                dtype=torch.int32,
            )

            # QK norms: q_norm -> q_layernorm, k_norm -> k_layernorm
            q_norm_key = f"layers.{l}.self_attn.q_norm.weight"
            k_norm_key = f"layers.{l}.self_attn.k_norm.weight"
            if q_norm_key in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
                    neuron_state_dict.pop(q_norm_key).detach().clone()
                )
            if k_norm_key in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
                    neuron_state_dict.pop(k_norm_key).detach().clone()
                )

            # q_proj is doubled: (8192, 2048) = (num_heads * head_dim * 2, hidden)
            # The weight is INTERLEAVED by head:
            #   [head0_query(256) | head0_gate(256) | head1_query(256) | head1_gate(256) | ...]
            # We need to deinterleave into separate query and gate weights.
            q_proj_key = f"layers.{l}.self_attn.q_proj.weight"
            if q_proj_key in neuron_state_dict:
                q_proj_w = neuron_state_dict.pop(q_proj_key)
                num_heads = config.num_attention_heads  # 16
                head_dim = config.head_dim  # 256
                # Reshape to (num_heads, head_dim*2, hidden_size)
                q_proj_w = q_proj_w.reshape(num_heads, head_dim * 2, config.hidden_size)
                # Split each head's output into query and gate
                query_w = q_proj_w[:, :head_dim, :]  # (16, 256, 2048)
                gate_w = q_proj_w[:, head_dim:, :]  # (16, 256, 2048)
                # Reshape back to (num_heads * head_dim, hidden_size)
                query_w = query_w.reshape(
                    num_heads * head_dim, config.hidden_size
                )  # (4096, 2048)
                gate_w = gate_w.reshape(
                    num_heads * head_dim, config.hidden_size
                )  # (4096, 2048)

                # Store query part back as q_proj for Wqkv fusion
                neuron_state_dict[q_proj_key] = query_w
                # Store gate weights for the output_gate_proj ColumnParallelLinear
                neuron_state_dict[f"layers.{l}.self_attn.output_gate_proj.weight"] = (
                    gate_w
                )

            # Fuse QKV
            if config.neuron_config.fused_qkv:
                q_key = f"layers.{l}.self_attn.q_proj.weight"
                k_key = f"layers.{l}.self_attn.k_proj.weight"
                v_key = f"layers.{l}.self_attn.v_proj.weight"
                if q_key in neuron_state_dict:
                    neuron_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = torch.cat(
                        [
                            neuron_state_dict[q_key],
                            neuron_state_dict[k_key],
                            neuron_state_dict[v_key],
                        ]
                    )
                    del neuron_state_dict[q_key]
                    del neuron_state_dict[k_key]
                    del neuron_state_dict[v_key]

        # === MoE weights ===
        # Router
        gate_key = f"layers.{l}.mlp.gate.weight"
        if gate_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
                neuron_state_dict.pop(gate_key).detach().clone()
            )

        # Fused expert weights
        # HF pre-fused: experts.gate_up_proj (E, 2*I, H) -- need transpose to NxDI (E, H, 2*I)
        # HF pre-fused: experts.down_proj (E, H, I) -- need transpose to NxDI (E, I, H)
        gate_up_key = f"layers.{l}.mlp.experts.gate_up_proj"
        down_key = f"layers.{l}.mlp.experts.down_proj"

        if gate_up_key in neuron_state_dict:
            w = neuron_state_dict.pop(gate_up_key).detach().clone()
            # Transpose: (E, 2*I, H) -> (E, H, 2*I)
            w = w.permute(0, 2, 1).contiguous()
            # Apply padding if needed
            pad_size = getattr(config, "moe_intermediate_pad_size", 0)
            if pad_size > 0:
                I = w.shape[2] // 2
                w = w.reshape(config.num_experts, config.hidden_size, 2, I)
                w = torch.nn.functional.pad(w, (0, pad_size))
                w = w.reshape(config.num_experts, config.hidden_size, -1)
            neuron_state_dict[
                f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
            ] = w

        if down_key in neuron_state_dict:
            w = neuron_state_dict.pop(down_key).detach().clone()
            # Transpose: (E, H, I) -> (E, I, H)
            w = w.permute(0, 2, 1).contiguous()
            # Apply padding if needed
            pad_size = getattr(config, "moe_intermediate_pad_size", 0)
            if pad_size > 0:
                w = torch.nn.functional.pad(w, (0, 0, 0, pad_size))
            neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = w

        # Shared expert weights (SigmoidGatedSharedExperts wrapping NxDI SharedExperts)
        # HF: mlp.shared_expert.{gate_proj, up_proj, down_proj}
        # NxDI: mlp.shared_experts.shared_experts.{gate_proj, up_proj, down_proj}
        # (mlp.shared_experts is SigmoidGatedSharedExperts, inner .shared_experts is NxDI SharedExperts)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            hf_key = f"layers.{l}.mlp.shared_expert.{proj}.weight"
            nxdi_key = f"layers.{l}.mlp.shared_experts.shared_experts.{proj}.weight"
            if hf_key in neuron_state_dict:
                neuron_state_dict[nxdi_key] = (
                    neuron_state_dict.pop(hf_key).detach().clone()
                )

        # Shared expert sigmoid gate: mlp.shared_expert_gate.weight
        # -> mlp.shared_experts.sigmoid_gate.weight
        seg_key = f"layers.{l}.mlp.shared_expert_gate.weight"
        if seg_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.mlp.shared_experts.sigmoid_gate.weight"] = (
                neuron_state_dict.pop(seg_key).detach().clone()
            )

        gc.collect()

    return neuron_state_dict


# ============================================================
# Custom ModelWrapper and DecoderModelInstance for DeltaNet state aliasing
# ============================================================


class Qwen35DecoderModelInstance(DecoderModelInstance):
    """Custom DecoderModelInstance that adds DeltaNet state buffers to input_output_aliases.

    After the standard KV cache aliases, we add aliases for each DeltaNet layer's
    recurrent_state_buffer and conv_state_buffer. This allows the XLA runtime to
    carry state between CTE and TKG graphs via shared HBM buffers.
    """

    def get(self, bucket_rank, **kwargs):
        """Override to add DeltaNet state aliases after KV cache aliases."""
        module, input_output_aliases = super().get(bucket_rank, **kwargs)

        # After super().get(), input_output_aliases maps KV cache params to
        # output indices starting from num_output_from_trace.
        # DeltaNet states go after all KV cache entries.
        num_output_from_trace = 1 if not self.neuron_config.output_logits else 2

        # Count KV cache entries
        if module.kv_mgr is not None:
            num_kv = len(module.kv_mgr.past_key_values)
        else:
            num_kv = 0

        # DeltaNet state aliases start after KV cache
        state_start_idx = num_output_from_trace + num_kv

        # Add aliases for DeltaNet state buffers
        if hasattr(module, "_deltanet_state_params"):
            for i, param in enumerate(module._deltanet_state_params):
                input_output_aliases[param] = state_start_idx + i

        return module, input_output_aliases


class Qwen35ModelWrapper(ModelWrapper):
    """Custom ModelWrapper that uses Qwen35DecoderModelInstance."""

    def get_model_instance(self):
        return Qwen35DecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )


# ============================================================
# Top-Level Model
# ============================================================


class NeuronQwen35MoeForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronQwen35MoeModel

    def get_model_wrapper_cls(self):
        """Return custom ModelWrapper with DeltaNet state aliasing."""
        return Qwen35ModelWrapper

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HF model weights.

        The model is a VL model (Qwen3_5MoeForConditionalGeneration) but we
        only need the text backbone. We load with AutoModelForCausalLM which
        will load the full model, then strip in convert_hf_to_neuron_state_dict.
        """
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Qwen35MoeInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Strip VL wrapper prefix and convert to NxDI format.

        The NxDI base class strips 'model.' prefix before calling this method.
        So HF keys like 'model.language_model.layers.X...' arrive as
        'language_model.layers.X...'. We strip the 'language_model.' prefix here.
        """
        new_sd = {}
        for k, v in state_dict.items():
            # After base class strips 'model.', VL wrapper keys start with 'language_model.'
            if k.startswith("language_model."):
                new_k = k.replace("language_model.", "", 1)
                new_sd[new_k] = v
            # Handle case where 'model.' was NOT stripped (e.g., called directly)
            elif k.startswith("model.language_model."):
                new_k = k.replace("model.language_model.", "", 1)
                new_sd[new_k] = v
            elif k.startswith("model.visual") or k.startswith("visual"):
                continue  # Skip vision encoder
            elif k.startswith("model."):
                new_sd[k.replace("model.", "", 1)] = v
            elif k.startswith("mtp."):
                continue  # Skip MTP
            elif k.startswith("lm_head."):
                new_sd[k] = v
            else:
                new_sd[k] = v

        return convert_qwen35_hf_to_neuron_state_dict(new_sd, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def _copy_past_key_values(self, outputs):
        """Override to also copy DeltaNet state buffers on CPU.

        On Neuron, input_output_aliases handles this automatically.
        On CPU, we must manually copy the output tensors back to the
        nn.Parameter .data attributes on both CTE and TKG models.
        """
        # First, call parent to copy KV cache
        super()._copy_past_key_values(outputs)

        # Then copy DeltaNet state buffers
        # The output layout is: [result] + [logits?] + kv_cache + deltanet_states
        num_output_from_trace = 1
        if (
            self.neuron_config.output_logits
            and self.neuron_config.on_device_sampling_config
        ):
            num_output_from_trace = 2

        # Count KV cache entries
        if (
            hasattr(self, "token_generation_model")
            and self.token_generation_model is not None
        ):
            tkg_model = self.token_generation_model.model
            cte_model = self.context_encoding_model.model
        else:
            return

        if tkg_model.kv_mgr is not None:
            num_kv = len(tkg_model.kv_mgr.past_key_values)
        else:
            num_kv = 0

        # DeltaNet states start after KV cache
        state_start = num_output_from_trace + num_kv

        # Get the state params from both models
        tkg_params = getattr(tkg_model, "_deltanet_state_params", [])
        cte_params = getattr(cte_model, "_deltanet_state_params", [])

        if len(tkg_params) > 0 and state_start + len(tkg_params) <= len(outputs):
            for i, (tkg_param, cte_param) in enumerate(zip(tkg_params, cte_params)):
                new_state = outputs[state_start + i]
                tkg_param.data = new_state
                cte_param.data = new_state

    def get_compiler_args(self):
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        else:
            optimization_level = "-O1"

        compiler_args = (
            "--enable-saturate-infinity "
            "--enable-mixed-precision-accumulation "
            f"--model-type transformer {optimization_level} "
            "--auto-cast=none "
            "--internal-enable-dge-levels vector_dynamic_offsets "
        )
        return compiler_args
