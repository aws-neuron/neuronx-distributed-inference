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

from nki_deltanet import deltanet_recurrent_fwd as _deltanet_nki_kernel
from nki_deltanet import deltanet_recurrent_fwd_state as _deltanet_nki_kernel_state
from nki_flash_attn_d256 import flash_attn_d256 as _flash_attn_d256_kernel

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
    NeuronAttentionBase,
    FlashAttentionStrategy,
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
    Uses a chunk-based linear recurrence instead of KV cache.

    V1 Design (stateless -- compiles but loses state between CTE and TKG):
    - CTE: chunk forward computes correct output for the prefill sequence.
    - TKG: recurrent step with ZERO initial state (no carry-over from CTE).
    - DeltaNet layers return dummy (K, V) tuples so KVCacheManager can process them.
    - No in-place buffer mutations (XLA trace safe).

    V2 TODO: Use input_output_aliases or repurpose KV cache slots to carry
    recurrent state and conv state between CTE and TKG.

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

    def _chunk_forward(self, query, key, value, g, beta, output_final_state=False):
        """Chunk-based forward for context encoding (prefill).

        V5: Uses the chunked formulation from the reference (torch_chunk_gated_delta_rule).
        Uses chunk_size=64, inter-chunk recurrent state propagation, and iterative
        correction loop within each chunk.

        NOTE: The iterative correction loop uses variable-width slice assignments
        which may cause accuracy issues under XLA tracing, but this approach
        compiles and produces reasonable (not perfect) results.

        Args:
            query: (B, H, S, k_dim) -- already in float32
            key:   (B, H, S, k_dim) -- already in float32
            value: (B, H, S, v_dim) -- already in float32
            g:     (B, H, S) -- already in float32
            beta:  (B, H, S) -- already in float32
            output_final_state: if True, return final recurrent state

        Returns:
            output: (B, H, S, v_dim)
            last_recurrent_state: (B, H, k_dim, v_dim) or None
        """
        chunk_size = 64

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        # Pad to multiple of chunk_size
        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        # Reshape to chunks: (B, H, num_chunks, chunk_size, dim)
        num_chunks = total_seq_len // chunk_size
        query = query.reshape(B, H, num_chunks, chunk_size, k_dim)
        key = key.reshape(B, H, num_chunks, chunk_size, k_dim)
        value = value.reshape(B, H, num_chunks, chunk_size, v_dim)
        k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, k_dim)
        v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, v_dim)
        g = g.reshape(B, H, num_chunks, chunk_size)

        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=0,
        )

        # Cumulative decay within each chunk
        g = g.cumsum(dim=-1)
        decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()

        # Intra-chunk delta rule correction (iterative)
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

        # Corrected value within each chunk
        value = attn @ v_beta  # (B, H, num_chunks, chunk_size, v_dim)
        # Corrected key * cumdecay within each chunk
        k_cumdecay = attn @ (
            k_beta * g.exp().unsqueeze(-1)
        )  # (B, H, num_chunks, chunk_size, k_dim)

        # Inter-chunk recurrent state propagation
        last_recurrent_state = torch.zeros(
            B, H, k_dim, v_dim, dtype=query.dtype, device=query.device
        )
        core_attn_out = torch.zeros_like(value)
        mask2 = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=1,
        )

        for i in range(num_chunks):
            q_i = query[:, :, i]  # (B, H, chunk_size, k_dim)
            k_i = key[:, :, i]
            v_i = value[:, :, i]  # corrected value

            attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
                mask2, 0
            )

            # Inter-chunk: subtract contribution of old state from corrected value
            v_prime = (
                k_cumdecay[:, :, i] @ last_recurrent_state
            )  # (B, H, chunk_size, v_dim)
            v_new = v_i - v_prime

            # Inter-chunk: query reads from old state
            attn_inter = (
                q_i * g[:, :, i, :, None].exp()
            ) @ last_recurrent_state  # (B, H, chunk_size, v_dim)
            core_attn_out[:, :, i] = attn_inter + attn_i @ v_new

            # Update recurrent state: decay by last position's cumulative g, then add new info
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (
                    k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                ).transpose(-1, -2)
                @ v_new
            )

        # Reshape back and trim padding
        core_attn_out = core_attn_out.reshape(B, H, -1, v_dim)
        core_attn_out = core_attn_out[:, :, :S]

        if not output_final_state:
            last_recurrent_state = None

        return core_attn_out, last_recurrent_state

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

        State carry-over (V37):
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
        # CRITICAL (V38b): We also need to save valid_mask_1d for later use to:
        #   1. Zero out g (decay) for padding positions -- otherwise padding tokens
        #      decay the recurrent state towards zero (exp(-1.3) ≈ 0.27 per token)
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
        deltanet_fp32 = os.environ.get("DELTANET_FP32") == "1"
        if deltanet_fp32:
            hs_f32 = hidden_states.float()
            qkv = F.linear(hs_f32, self.in_proj_qkv.weight.float()).to(
                hidden_states.dtype
            )
            z = F.linear(hs_f32, self.in_proj_z.weight.float()).to(hidden_states.dtype)
            b = F.linear(hs_f32, self.in_proj_b.weight.float()).to(hidden_states.dtype)
            a = F.linear(hs_f32, self.in_proj_a.weight.float()).to(hidden_states.dtype)
        else:
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
            # CTE: Use nn.Conv1d with built-in padding (V36 approach -- proven correct).
            # self.conv1d has padding=kernel_size-1=3, which pads both sides symmetrically.
            # Truncating to [:, :, :seq_len] gives correct causal conv1d output.
            # This is IDENTICAL to V36 which produced correct "Paris" output.
            mixed_post_conv = F.silu(self.conv1d(mixed)[:, :, :seq_len])

            # CRITICAL (V38b): Save last 3 VALID tokens' mixed values for conv_state.
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

        # CRITICAL (V38b): Zero out g for padding positions.
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
        if deltanet_fp32:
            output = F.linear(output.float(), self.out_proj.weight.float()).to(
                hidden_states.dtype
            )
        else:
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

        # mRoPE (multimodal RoPE) for VL support
        # Extract from rope_parameters if present (HF config format)
        rope_params = getattr(self, "rope_parameters", {}) or {}
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self.mrope_interleaved = rope_params.get("mrope_interleaved", True)

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


class Qwen35MRoPEEmbedding(nn.Module):
    """Multimodal Rotary Position Embedding (mRoPE) for Qwen3.5.

    Handles 3D position information (temporal, height, width) for VL models.
    Position IDs have shape (3, batch_size, seq_len) for T/H/W dimensions.
    For text-only (2D position_ids), broadcasts to 3D with identical positions.

    Uses interleaved layout: THWTHW... (stride-3 indexing) matching HF reference.

    Based on Qwen3-VL-8B-Thinking contrib model, adapted for partial RoPE:
    - dim = rope_dim (64), not full head_dim (256)
    - mrope_section = [11, 11, 10] (total 32 = rope_dim / 2)
    - Output (cos, sin) shape: (batch_size, seq_len, rope_dim=64)
    """

    def __init__(self, config: Qwen35MoeInferenceConfig):
        super().__init__()
        self.dim = config.rope_dim  # 64 (partial RoPE)
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta

        # mRoPE specific configuration
        self.mrope_section = getattr(config, "mrope_section", [11, 11, 10])
        self.mrope_interleaved = getattr(config, "mrope_interleaved", True)

        # inv_freq: (rope_dim / 2,) = (32,) -- matches sum(mrope_section)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved mRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHW...TT], preserving frequency continuity.

        Args:
            freqs: (3, bs, seq_len, rope_dim // 2) - frequencies for T, H, W
            mrope_section: (3,) - sections for temporal, height, width

        Returns:
            freqs_t: (bs, seq_len, rope_dim // 2) - interleaved frequencies
        """
        freqs_t = freqs[0].clone()  # Start with temporal frequencies
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x, position_ids):
        """Compute mRoPE cos/sin embeddings.

        Args:
            x: Input tensor (for device/dtype only)
            position_ids: (3, batch_size, seq_len) or (batch_size, seq_len)

        Returns:
            cos: (batch_size, seq_len, rope_dim=64)
            sin: (batch_size, seq_len, rope_dim=64)
        """
        # Expand to 3D if needed
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # (rope_dim/2,) -> (3, batch_size, rope_dim/2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )

        # (3, batch_size, seq_len) -> (3, batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].float()

        # Compute frequencies per dimension: (3, bs, rope_dim/2, seq_len) -> (3, bs, seq_len, rope_dim/2)
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)

            # Apply interleaved mRoPE
            if self.mrope_interleaved:
                freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            else:
                freqs = freqs[0]

            # Double to rope_dim: (bs, seq_len, rope_dim/2) -> (bs, seq_len, rope_dim)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronQwen35Attention(NeuronAttentionBase):
    """Attention with output gate and partial RoPE.

    V3: Implements partial RoPE (25% of head_dim), per-head QK norm,
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
        # Partial RoPE: create mRoPE embedding with rope_dim (64)
        self.rope_dim = config.rope_dim  # 64 = head_dim * partial_rotary_factor

        # Create QK norm modules first (will be passed to base class)
        rms_norm_eps = config.rms_norm_eps
        q_ln = get_rmsnorm_cls()(config.head_dim, rms_norm_eps)
        k_ln = get_rmsnorm_cls()(config.head_dim, rms_norm_eps)

        # Partial RoPE: use standard RotaryEmbedding (identical to pre-mRoPE working code).
        # For VL with 3D mRoPE positions, cos/sin are pre-computed externally in
        # get_model_output() using Qwen35MRoPEEmbedding and passed as cos_cache/sin_cache.
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

        # Separate mRoPE module for VL 3D position_ids (not used as self.rotary_emb).
        # When rotary_position_ids is 3D (T/H/W), we pre-compute cos/sin with mRoPE
        # and pass them to prep_qkv_tensors via cos_cache/sin_cache.
        self.mrope_emb = Qwen35MRoPEEmbedding(config)

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
        """Override to handle head_dim=256 with custom NKI flash attention kernel.

        The standard NxDI NKI flash attention kernel asserts head_dim <= 128.
        We use our own flash_attn_d256 kernel which tiles the QK contraction
        into 2x128 chunks, giving ~2-3x TTFT improvement over the softmax fallback.

        Shape contract:
          Input Q:  (B, H, S, D=256) -- BHSD
          Input K:  (B, Hkv, S, D=256) -- BHSD
          Input V:  (B, Hkv, S, D=256) -- BHSD
          Kernel q: (B, H, D, S) -- BHDS (transposed)
          Kernel k: (B, Hkv, D, S) -- BHDS (transposed)
          Kernel v: (B, Hkv, S, D) -- BHSD (unchanged)
          Output:   (B, H, S, D) -- BHSD (same as NONE path)

        The kernel requires seq_len divisible by 512 (B_F tile size).
        For smaller seq_lens, fall back to softmax path.
        """
        if self.head_dim > 128 and q_len >= 512 and q_len % 512 == 0:
            # Reshape for our d=256 kernel: BHSD -> BHDS for Q and K
            q_kernel = Q.permute(0, 1, 3, 2).contiguous().to(self.torch_dtype)
            k_kernel = K.permute(0, 1, 3, 2).contiguous().to(self.torch_dtype)
            v_kernel = V.contiguous().to(self.torch_dtype)

            n_kv_heads = K.shape[1]
            grid_size = bsz * n_kv_heads

            # Kernel returns (B, H, S, D) -- BHSD, same as NONE path
            attn_output = _flash_attn_d256_kernel[grid_size](
                q_kernel, k_kernel, v_kernel, use_causal_mask=True
            )

            return attn_output, FlashAttentionStrategy.NONE

        if self.head_dim > 128:
            # Fallback for seq_lens not divisible by 512
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
        rotary_position_ids=None,
        **kwargs,
    ):
        """Forward with output gate applied BEFORE o_proj.

        Override NeuronAttentionBase.forward() to insert the sigmoid gate
        between the attention output and o_proj, matching the HF reference:
          gate = sigmoid(gate_proj(pre_attn_hidden))
          attn_output = attn_output * gate
          attn_output = o_proj(attn_output)

        Phase 2 mRoPE: cos_cache/sin_cache are pre-computed from 3D mRoPE
        position_ids in get_model_output() and passed through the decoder loop.
        When they arrive non-None, apply_rotary_embedding skips self.rotary_emb()
        and uses the pre-computed values directly. For TKG (cos/sin=None),
        self.rotary_emb computes from 2D position_ids as before.
        """
        bsz, q_len, _ = hidden_states.shape

        # Use standard 2D position_ids for prep_qkv_tensors.
        # When cos/sin are pre-computed (mRoPE), apply_rotary_embedding skips
        # self.rotary_emb() and uses them directly. When None (TKG),
        # self.rotary_emb computes from these 2D position_ids.
        rope_pos_ids = position_ids

        # Compute gate from input hidden states (before QKV projection)
        gate = self.output_gate_proj(hidden_states)  # (B, S, num_heads * head_dim)

        # Standard QKV prep (projections, QK norm, RoPE)
        Q, K, V, cos_cache, sin_cache, _residual = self.prep_qkv_tensors(
            rope_pos_ids,
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
        cos_cache=None,
        sin_cache=None,
        **kwargs,
    ):
        # V30 identity diagnostic: skip all layer computation, return input unchanged
        if os.environ.get("SKIP_LAYER_COMPUTE") == "1":
            bsz, seq_len, _ = hidden_states.shape
            # Create dummy KV cache matching NxDI expected shape:
            # (B, kv_heads_per_rank, seq_len, head_dim)
            tp = getattr(self.config.neuron_config, "tp_degree", 1)
            kv_heads_per_rank = max(self.config.num_key_value_heads // tp, 1)
            dummy_k = torch.zeros(
                bsz,
                kv_heads_per_rank,
                seq_len,
                self.config.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            dummy_v = torch.zeros_like(dummy_k)
            hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            hidden_states = ModuleMarkerEndWrapper()(hidden_states)
            return (hidden_states, (dummy_k, dummy_v), None, None, None, None)

        # V34: Test just input_layernorm + a single projection to isolate where divergence starts
        if (
            os.environ.get("DELTANET_PROJ_ONLY") == "1"
            and self.layer_type == "linear_attention"
        ):
            bsz, seq_len, _ = hidden_states.shape
            tp = getattr(self.config.neuron_config, "tp_degree", 1)
            kv_heads_per_rank = max(self.config.num_key_value_heads // tp, 1)
            dummy_k = torch.zeros(
                bsz,
                kv_heads_per_rank,
                seq_len,
                self.config.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            dummy_v = torch.zeros_like(dummy_k)

            hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            # Apply input_layernorm
            normed = self.input_layernorm(hidden_states)
            # Do one projection to test matmul
            proj_out = self.linear_attn.in_proj_qkv(normed)
            # Return a scaled version so the output is meaningful
            # Use mean of first 2048 dims as a scalar multiplier for the residual
            # This avoids having to do the full DeltaNet computation
            scale = proj_out[..., : self.hidden_size].mean(dim=-1, keepdim=True)
            hidden_states = (
                hidden_states + scale * 0.001
            )  # tiny perturbation from projection
            hidden_states = ModuleMarkerEndWrapper()(hidden_states)
            return (hidden_states, (dummy_k, dummy_v), None, None, None, None)

        residual = hidden_states

        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)

        # V30 diagnostic: SKIP_ATTN_COMPUTE skips attention, keeps MoE
        skip_attn = os.environ.get("SKIP_ATTN_COMPUTE") == "1"

        if self.layer_type == "linear_attention":
            if skip_attn:
                # Skip DeltaNet, just use residual
                hidden_states = residual
                bsz, seq_len, _ = hidden_states.shape
                tp = getattr(self.config.neuron_config, "tp_degree", 1)
                kv_heads_per_rank = max(self.config.num_key_value_heads // tp, 1)
                present_key_value = (
                    torch.zeros(
                        bsz,
                        kv_heads_per_rank,
                        seq_len,
                        self.config.head_dim,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                    torch.zeros(
                        bsz,
                        kv_heads_per_rank,
                        seq_len,
                        self.config.head_dim,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                )
                deltanet_states = None
            else:
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
            # Pass through cos/sin cache (pre-computed mRoPE from get_model_output)
            # instead of resetting to None, so subsequent GQA layers receive them.
        else:
            deltanet_states = None
            if skip_attn:
                hidden_states = residual
                bsz, seq_len, _ = hidden_states.shape
                tp = getattr(self.config.neuron_config, "tp_degree", 1)
                kv_heads_per_rank = max(self.config.num_key_value_heads // tp, 1)
                present_key_value = (
                    torch.zeros(
                        bsz,
                        kv_heads_per_rank,
                        seq_len,
                        self.config.head_dim,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                    torch.zeros(
                        bsz,
                        kv_heads_per_rank,
                        seq_len,
                        self.config.head_dim,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                )
                cos_cache, sin_cache = None, None
            else:
                # Standard attention path (V3: gate is inside self_attn.forward())
                hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    **kwargs,
                )
                hidden_states = residual + hidden_states

        # V30 diagnostic: SKIP_MOE_COMPUTE skips MoE, keeps attention
        skip_moe = os.environ.get("SKIP_MOE_COMPUTE") == "1"

        if skip_moe:
            # Skip MoE entirely, hidden_states stays as-is (attn residual)
            pass
        else:
            # MoE FFN (routed experts + sigmoid-gated shared expert via NxDI MoE)
            # NxDI's _apply_shared_experts calls self.mlp.shared_experts (our
            # SigmoidGatedSharedExperts wrapper) which adds sigmoid-gated shared
            # expert output to routed output BEFORE the all-reduce.
            residual = hidden_states
            if not self.moe_fused_nki_kernel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)

            is_speculative_decoding = (
                self.config.neuron_config.enable_fused_speculation
                and not self.config.neuron_config.is_prefill_stage
            )
            moe_output = self.mlp(
                hidden_states,
                padding_mask,
                is_speculative_decoding=is_speculative_decoding,
            )[0]
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

        # mRoPE embedding for VL: pre-computes cos/sin from 3D position_ids
        # in get_model_output() before the decoder layer loop.
        self.mrope_emb = Qwen35MRoPEEmbedding(config)

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

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        """Scatter vision embeddings into text input embeddings at image token positions.

        Uses index_put_ to replace placeholder token embeddings with vision encoder output.

        Args:
            inputs_embeds: (batch_size, seq_len, hidden_size) -- text token embeddings
            vision_embeddings: (batch_size, n_vision_tokens, hidden_size) -- from vision encoder
            vision_mask: (batch_size, n_vision_tokens, 1) -- int32 position indices

        Returns:
            inputs_embeds with vision embeddings scattered in at the specified positions
        """
        _, max_positions, embedding_dim = inputs_embeds.shape
        h_new = inputs_embeds.clone()
        vision_flat = vision_embeddings.view(-1, embedding_dim)
        positions_flat = vision_mask.view(-1)
        h_new.view(-1, embedding_dim).index_put_(
            (positions_flat,), vision_flat, accumulate=False
        )
        return h_new

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

        # Vision embedding injection (scatter vision tokens into text embeddings)
        if (vision_embeddings is not None) and (vision_mask is not None):
            if vision_embeddings.dtype != self.config.neuron_config.torch_dtype:
                vision_embeddings = vision_embeddings.to(
                    self.config.neuron_config.torch_dtype
                )
            if is_for_context_encoding:
                inputs_embeds = self.encode_vision_to_input(
                    inputs_embeds, vision_embeddings, vision_mask
                )

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

        # Phase 2 mRoPE: Pre-compute cos/sin from 3D position_ids when available.
        # For CTE with VL content, rotary_position_ids is (3, B, S) with T/H/W positions.
        # For text-only CTE, it's (3, B, S) with T=H=W=sequential.
        # For TKG, it's None (set_none_if_empty converted torch.zeros((0,)) to None).
        # Pre-computing here ensures all layers receive the same mRoPE cos/sin,
        # and DeltaNet layers pass them through unchanged.
        if rotary_position_ids is not None and rotary_position_ids.ndim == 3:
            cos_cache, sin_cache = self.mrope_emb(inputs_embeds, rotary_position_ids)

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
            rotary_position_ids=rotary_position_id,
            update_cache=True,
            is_for_context_encoding=is_for_context_encoding,
            padding_mask=None,
            active_block_table=active_block_table,
            scatter_index=slot_mapping
            if getattr(self, "is_block_kv_layout", False)
            else scatter_index,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
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
            if "layers.0." in nk or nk == "norm.weight":
                print(
                    f"  [NORM FIX] {nk}: mean {old_val.float().mean():.4f} -> {neuron_state_dict[nk].mean():.4f}"
                )
        else:
            if "layers.0." in nk or nk == "norm.weight":
                print(f"  [NORM FIX] WARNING: key not found: {nk}")
                print(
                    f"    Available keys (sample): {[k for k in neuron_state_dict.keys() if 'norm' in k.lower()][:5]}"
                )

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
    """Custom ModelWrapper that uses Qwen35DecoderModelInstance.

    Overrides input_generator to add vision_embeddings and vision_mask
    as traced inputs for VL support.
    """

    def get_model_instance(self):
        return Qwen35DecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )

    def input_generator(self):
        """Generate inputs including mrope_position_ids, vision_embeddings, and vision_mask.

        Extends the base input_generator output:
        - Positions 7-20: empty tensors (unused NxDI slots)
        - Position 21: rotary_position_id = mrope_position_ids (3, BS, seq_len) for CTE,
                        empty (0,) for TKG
        - Position 22: vision_embeddings (BS, seq_len, hidden_size) for CTE,
                        empty (0,) for TKG
        - Position 23: vision_mask (BS, seq_len, 1) for CTE,
                        empty (0,) for TKG
        """
        base_inputs = super().input_generator()
        extended_inputs = []

        for bucket_inputs in base_inputs:
            input_ids = bucket_inputs[0]
            batch_size = input_ids.shape[0]
            n_active_tokens = input_ids.shape[1]

            is_cte = n_active_tokens > 1

            if is_cte:
                # Context encoding: properly-shaped inputs
                # mRoPE position IDs: (3, BS, seq_len) -- T/H/W all sequential for trace
                mrope_position_ids = (
                    torch.arange(0, n_active_tokens, dtype=torch.int32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(3, batch_size, -1)
                    .contiguous()
                )

                vision_embeddings = torch.zeros(
                    (batch_size, n_active_tokens, self.config.hidden_size),
                    dtype=self.config.neuron_config.torch_dtype,
                )
                vision_mask = torch.full(
                    (batch_size, n_active_tokens, 1),
                    fill_value=n_active_tokens
                    - 1,  # Safe fill: scatter to last position
                    dtype=torch.int32,
                )
            else:
                # Token generation: empty tensors
                mrope_position_ids = torch.zeros((0,), dtype=torch.int32)
                vision_embeddings = torch.zeros(
                    (0,), dtype=self.config.neuron_config.torch_dtype
                )
                vision_mask = torch.zeros((0,), dtype=torch.int32)

            # Base generates 7 args; pad to 21, then add mrope + vision
            padded = list(bucket_inputs)
            while len(padded) < 21:
                padded.append(torch.zeros((0,), dtype=torch.int32))
            padded.append(mrope_position_ids)  # position 21: rotary_position_id
            padded.append(vision_embeddings)  # position 22
            padded.append(vision_mask)  # position 23

            extended_inputs.append(tuple(padded))

        return extended_inputs

    def pad_inputs(self, *args, pad_type="first_fit"):
        """Override to pad mrope_position_ids and vision inputs to bucket size.

        CRITICAL FIX: The base class pad_inputs() (model_wrapper.py line 831)
        has a code path that REGENERATES vision embeddings as all-zeros when
        it detects 24 args with vision_mask shape != pad_length. This destroys
        the real vision data BEFORE our override gets to work on it.

        Solution: Save the ORIGINAL vision args (positions 21-23) BEFORE
        calling super().pad_inputs(), then use those originals for
        zero-extension padding afterward.
        """
        # Save original vision args BEFORE the base class destroys them
        orig_mrope = args[21] if len(args) >= 22 else None
        orig_vis_emb = args[22] if len(args) >= 23 else None
        orig_vis_mask = args[23] if len(args) >= 24 else None

        # Let base class pad positions 0-2 (input_ids, attention_mask, position_ids)
        # NOTE: base class will zero out positions 22-23, but we saved originals above
        padded_args = super().pad_inputs(*args, pad_type=pad_type)

        # Check if padding is needed (CTE only, when we have 24 args)
        if len(padded_args) >= 24 and orig_mrope is not None:
            padded_seq_len = padded_args[0].shape[1]
            batch_size = padded_args[0].shape[0]
            is_cte = padded_seq_len > 1

            if is_cte:
                # Use ORIGINALS (not the base-class-zeroed versions)
                current_mrope = orig_mrope
                current_vis_emb = orig_vis_emb
                current_vis_mask = orig_vis_mask

                # Pad mrope_position_ids: (3, BS, orig_len) -> (3, BS, padded_len)
                if (
                    current_mrope.ndim == 3
                    and current_mrope.shape[-1] != padded_seq_len
                ):
                    orig_len = current_mrope.shape[-1]
                    pad_size = padded_seq_len - orig_len
                    last_pos = current_mrope[:, :, -1:]  # (3, BS, 1)
                    pad_offsets = torch.arange(
                        1, pad_size + 1, dtype=current_mrope.dtype
                    )
                    pad_offsets = (
                        pad_offsets.unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
                    )
                    mrope_pad = last_pos + pad_offsets
                    mrope_position_ids = torch.cat([current_mrope, mrope_pad], dim=-1)
                elif current_mrope.ndim == 3:
                    mrope_position_ids = current_mrope
                else:
                    # Fallback: generate sequential (text-only tracing)
                    mrope_position_ids = (
                        torch.arange(0, padded_seq_len, dtype=torch.int32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(3, batch_size, -1)
                        .contiguous()
                    )

                # Pad vision_embeddings: (BS, orig_len, H) -> (BS, padded_len, H)
                # Extend with zeros (padding tokens have no vision content)
                if (
                    current_vis_emb is not None
                    and current_vis_emb.ndim == 3
                    and current_vis_emb.shape[1] < padded_seq_len
                ):
                    pad_emb = torch.zeros(
                        (
                            batch_size,
                            padded_seq_len - current_vis_emb.shape[1],
                            current_vis_emb.shape[2],
                        ),
                        dtype=current_vis_emb.dtype,
                    )
                    vision_embeddings = torch.cat([current_vis_emb, pad_emb], dim=1)
                elif current_vis_emb is not None and current_vis_emb.ndim == 3:
                    vision_embeddings = current_vis_emb[:, :padded_seq_len]
                else:
                    vision_embeddings = torch.zeros(
                        (batch_size, padded_seq_len, self.config.hidden_size),
                        dtype=self.config.neuron_config.torch_dtype,
                    )

                # Pad vision_mask: (BS, orig_len, 1) -> (BS, padded_len, 1)
                # Extend with padded_seq_len-1 (safe scatter target for padding)
                if (
                    current_vis_mask is not None
                    and current_vis_mask.ndim == 3
                    and current_vis_mask.shape[1] < padded_seq_len
                ):
                    pad_mask = torch.full(
                        (batch_size, padded_seq_len - current_vis_mask.shape[1], 1),
                        fill_value=padded_seq_len - 1,
                        dtype=torch.int32,
                    )
                    vision_mask = torch.cat([current_vis_mask, pad_mask], dim=1)
                elif current_vis_mask is not None and current_vis_mask.ndim == 3:
                    vision_mask = current_vis_mask[:, :padded_seq_len]
                else:
                    vision_mask = torch.full(
                        (batch_size, padded_seq_len, 1),
                        fill_value=padded_seq_len - 1,
                        dtype=torch.int32,
                    )

                padded_args = (
                    *padded_args[:21],
                    mrope_position_ids,
                    vision_embeddings,
                    vision_mask,
                )

                # Safety clamp: ensure all vision_mask entries are within valid range.
                # This is a no-op when fill_value is already seq_len-1, but protects
                # against any edge case where values exceed the tensor dimensions.
                padded_args = list(padded_args)
                padded_args[23] = padded_args[23].clamp(max=padded_seq_len - 1)
                padded_args = tuple(padded_args)

        return padded_args


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

    def get_required_kwargs(self):
        """Return extra kwargs that must be propagated through the HF generation loop.

        This ensures llava_args (vision_embeddings + vision_mask) flows from
        generate() -> prepare_inputs_for_generation() -> forward() -> _get_model_outputs().
        """
        return ["llava_args"]

    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden,
        adapter_ids,
        medusa_args,
        llava_args,
        slot_mapping=None,
        block_table=None,
        full_context_lens=None,
        computed_context_lens=None,
        tf_args=None,
    ):
        """Override to pass all 24 positional args explicitly.

        The model is traced with 24 positional args (from Qwen35ModelWrapper.input_generator).
        The base class splats *llava_args after 7 args, which puts vision inputs at positions
        7-8 instead of 22-23. We override to fill positions 7-20 with torch.empty(0) and
        place mrope_position_ids at 21 and vision inputs at 22-23.

        llava_args layout from VL generate():
            [0] vision_embeddings (BS, seq_len, hidden_size)
            [1] vision_mask (BS, seq_len, 1)
            [2] mrope_position_ids (3, batch, seq_len) -- optional

        For CTE: slot 21 = (3, B, S) mRoPE position IDs.
                 If not in llava_args, generate sequential IDs with T=H=W (text-only).
        For TKG: slot 21 = torch.zeros((0,)) → set_none_if_empty → None → uses 2D position_ids.
        """
        is_prefill = self._is_prefill(position_ids)

        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        # Extract vision inputs and mRoPE position IDs from llava_args.
        # llava_args layout: [vision_embeddings, vision_mask, mrope_position_ids (optional)]
        if llava_args and len(llava_args) >= 2:
            vision_embeddings = llava_args[0]
            vision_mask = llava_args[1]
            # mRoPE position IDs: (3, batch, seq_len) if provided
            if len(llava_args) >= 3:
                mrope_position_ids = llava_args[2]
            else:
                mrope_position_ids = None
        elif is_prefill:
            # Text-only CTE: generate dummy vision inputs matching compiled shape.
            # The compiled CTE expects (BS, seq_len, hidden_size) and (BS, seq_len, 1).
            # Use zeros for embeddings and seq_len-1 for mask (safe scatter target).
            # NOTE: Do NOT use large sentinel values (e.g., 2**30) as fill_value --
            # they cause DGE out-of-bounds crashes in the Neuron runtime.
            # Using seq_len-1 targets the last position (always a padding slot).
            vision_embeddings = torch.zeros(
                (batch_size, seq_len, self.config.hidden_size),
                dtype=self.config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                (batch_size, seq_len, 1),
                fill_value=seq_len - 1,
                dtype=torch.int32,
            )
            mrope_position_ids = None
        else:
            # TKG: empty tensors (no vision injection during decode)
            vision_embeddings = torch.zeros((0,), dtype=torch.float32)
            vision_mask = torch.zeros((0,), dtype=torch.int32)
            mrope_position_ids = None

        # For CTE: mRoPE position IDs at slot 21 must be (3, batch, seq_len).
        # If not provided (text-only), generate sequential IDs with T=H=W (identical axes).
        # For TKG: slot 21 = torch.empty(0) → set_none_if_empty → None → fallback to 2D position_ids.
        if is_prefill:
            if mrope_position_ids is None:
                mrope_position_ids = (
                    torch.arange(0, seq_len, dtype=torch.int32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(3, batch_size, -1)
                    .contiguous()
                )
        else:
            mrope_position_ids = torch.zeros((0,), dtype=torch.int32)

        # Build the 14 empty tensors for positions 7-20
        # Position 21 = mrope_position_ids, 22 = vision_embeddings, 23 = vision_mask
        empties = [torch.empty(0) for _ in range(14)]

        if self._is_prefill(position_ids):
            outputs = self.context_encoding_model(
                input_ids,  # 0
                attention_mask,  # 1
                position_ids,  # 2
                seq_ids,  # 3
                sampling_params,  # 4
                prev_hidden,  # 5
                adapter_ids,  # 6
                *empties,  # 7-20
                mrope_position_ids,  # 21
                vision_embeddings,  # 22
                vision_mask,  # 23
            )
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,  # 0
                attention_mask,  # 1
                position_ids,  # 2
                seq_ids,  # 3
                sampling_params,  # 4
                prev_hidden,  # 5
                adapter_ids,  # 6
                *empties,  # 7-20
                mrope_position_ids,  # 21
                vision_embeddings,  # 22
                vision_mask,  # 23
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

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
