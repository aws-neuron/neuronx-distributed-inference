# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NxDI contrib: Qwen3.5-4B (qwen3_5 -- dense model)

Hybrid DeltaNet + Standard Attention + Dense MLP architecture.

24 of 32 layers use Gated DeltaNet (linear recurrent attention)
8 of 32 layers use standard GQA with KV cache + output gate
All 32 layers use a dense SwiGLU MLP (intermediate_size=9216)

Architecture details:
- DeltaNet layers: separate in_proj_{qkv, z, a, b}, causal conv1d on QKV, gated delta rule
- Attention layers: q_proj doubled (Q + gate), partial RoPE (25% of head_dim), sigmoid output gate
- Dense MLP: standard SwiGLU (gate_proj, up_proj, down_proj) -- no MoE, no router, no experts
- KV cache: NxDI KVCacheManager for attention layers; DeltaNet layers store recurrent+conv
  state as nn.Parameter buffers and return dummy KV tuples
"""

import gc
import math
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode

try:
    from nki import jit as nki_jit  # NKI 0.3.0+ (SDK 2.29)
except ImportError:
    from torch_neuronx.xla_impl.ops import nki_jit  # NKI 0.2.x (SDK 2.28)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from src.nki_kernels.nki_deltanet import deltanet_recurrent_fwd as _deltanet_nki_kernel
from src.nki_kernels.nki_deltanet import (
    deltanet_recurrent_fwd_state as _deltanet_nki_kernel_state,
)
from src.nki_kernels.nki_deltanet_chunked import (
    deltanet_chunk_step as _deltanet_nki_chunk_step,
)
from src.nki_kernels.nki_deltanet_fused import (
    deltanet_fused_chunked_fwd as _deltanet_fused_kernel,
)
from src.nki_kernels.nki_deltanet_fused import (
    _make_lower_mask,
    _make_lower_mask_diag,
    _make_identity,
)

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    NeuronConfig,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    DecoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

logger = logging.getLogger(__name__)

try:
    _flash_fwd_call = nki_jit()(attention_isa_kernel)
except TypeError:
    from torch_neuronx.xla_impl.ops import nki_jit as _torch_xla_nki_jit

    _flash_fwd_call = _torch_xla_nki_jit()(attention_isa_kernel)

# Option B: Direct nkilib flash attention for head_dim > 128
USE_NKILIB_KERNEL = os.environ.get("USE_NKILIB_KERNEL", "0") == "1"

_nkilib_flash_attn = None
if USE_NKILIB_KERNEL:
    try:
        import neuronxcc.nki as _nki
        from neuronx_distributed_inference.modules.attention.attention_base import (
            peel_decorations as _peel_decorations,
            get_platform_target as _get_platform_target,
        )
        from neuronxcc.nki.compiler import (
            skip_middle_end_transformations as _skip_middle_end,
            enable_stack_allocator as _enable_stack_allocator,
        )

        import importlib

        _fork_path = "/home/ubuntu/nki-library-fork/nkilib_src"
        if os.path.isdir(_fork_path) and _fork_path not in sys.path:
            sys.path.insert(0, _fork_path)
        _to_remove = [k for k in sys.modules if k.startswith("nkilib")]
        for k in _to_remove:
            del sys.modules[k]
        import nki.language as _stub_nl
        import neuronxcc.nki.language as _real_nl

        for _attr in [
            "NKIObject",
            "float8_e4m3fn",
            "float8_e4m3fn_x4",
            "float8_e5m2_x4",
            "float4_e2m1fn_x4",
        ]:
            if not hasattr(_real_nl, _attr) and hasattr(_stub_nl, _attr):
                setattr(_real_nl, _attr, getattr(_stub_nl, _attr))
        from nkilib.core.attention.attention_cte import (
            attention_cte as _attention_cte_raw,
            _MAX_HEAD_DIM,
        )

        assert _MAX_HEAD_DIM == 256, (
            f"nkilib fork has _MAX_HEAD_DIM={_MAX_HEAD_DIM}, expected 256. "
            f"System nkilib may have been loaded instead of fork."
        )
        logger.info(
            f"Loaded nkilib attention_cte from fork (_MAX_HEAD_DIM={_MAX_HEAD_DIM})"
        )

        _raw_fn = _peel_decorations(_attention_cte_raw)
        _platform = _get_platform_target()
        _nkilib_flash_attn = _nki.jit(
            _raw_fn,
            mode="torchxla",
            platform_target=_platform,
            show_compiler_tb=True,
            debug_kernel=True,
        )
        _nkilib_flash_attn = _skip_middle_end(_nkilib_flash_attn)
        _nkilib_flash_attn = _enable_stack_allocator(
            _nkilib_flash_attn, log_level=logging.INFO
        )
        logger.info("Option B: nkilib flash attention loaded for head_dim > 128")
    except Exception as e:
        logger.warning(f"Option B: Failed to load nkilib flash attention: {e}")
        import traceback as _tb

        _tb.print_exc()
        _nkilib_flash_attn = None

# Option A: Detect if patch_attn_kernel was imported
NKILIB_PATCH_ACTIVE = False
try:
    from importlib import import_module as _import_module

    _attn_mod = _import_module("neuronxcc.nki._pre_prod_kernels.attn_fwd")
    if hasattr(_attn_mod, "_original_attention_nki_kernel_adapter"):
        NKILIB_PATCH_ACTIVE = True
        logger.info("Option A detected: _pre_prod_kernels patched with nkilib kernel")
except Exception:
    pass


# ============================================================
# Newton-Raphson Refined RMSNorm
# ============================================================
USE_NEWTON_RMSNORM = os.environ.get("USE_NEWTON_RMSNORM") == "1"
USE_PYTHON_RMSNORM = os.environ.get("USE_PYTHON_RMSNORM") == "1"


class NewtonRMSNorm(nn.Module):
    """RMSNorm with Newton-Raphson refined rsqrt for improved numerical accuracy."""

    def __init__(self, hidden_size=None, eps=1e-6):
        super().__init__()
        self.weight = None
        if hidden_size is not None:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        y = torch.rsqrt(variance + self.variance_epsilon)
        y = y * (3.0 - (variance + self.variance_epsilon) * y * y) * 0.5
        result = x * y
        if self.weight is not None:
            result = result * self.weight.float()
        return result.to(original_dtype)


def get_rmsnorm_cls():
    if cpu_mode() or USE_PYTHON_RMSNORM:
        return Qwen3MoeRMSNorm
    return NewtonRMSNorm if USE_NEWTON_RMSNORM else CustomRMSNorm


def l2norm(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)


# ============================================================
# Gated DeltaNet Module (Linear Recurrent Attention)
# ============================================================


class NeuronGatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention for Neuron.

    Replaces standard attention for 24 of 32 layers in Qwen3.5-4B.
    Uses a chunk-based linear recurrence instead of KV cache.

    HF weight layout (4B dense):
    - in_proj_qkv.weight: (key_dim*2 + value_dim, hidden_size) = (8192, 2560)
    - in_proj_z.weight: (value_dim, hidden_size) = (4096, 2560)
    - in_proj_a.weight: (num_v_heads, hidden_size) = (32, 2560)
    - in_proj_b.weight: (num_v_heads, hidden_size) = (32, 2560)
    - conv1d.weight: (conv_dim, 1, conv_kernel_size) = (8192, 1, 4)
    - A_log: (num_v_heads,) = (32,)
    - dt_bias: (num_v_heads,) = (32,)
    - norm.weight: (head_v_dim,) = (128,)
    - out_proj.weight: (hidden_size, value_dim) = (2560, 4096)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        tc = config

        self.hidden_size = tc.hidden_size  # 5120
        self.num_v_heads = tc.linear_num_value_heads  # 48
        self.num_k_heads = tc.linear_num_key_heads  # 16
        self.head_k_dim = tc.linear_key_head_dim  # 128
        self.head_v_dim = tc.linear_value_head_dim  # 128
        self.key_dim = self.head_k_dim * self.num_k_heads  # 2048
        self.value_dim = self.head_v_dim * self.num_v_heads  # 6144
        self.conv_kernel_size = tc.linear_conv_kernel_dim  # 4
        self.layer_idx = layer_idx
        self.rms_norm_eps = tc.rms_norm_eps

        # KV cache dummy shape info
        self.head_dim = tc.head_dim  # 256
        tp_degree = tc.neuron_config.tp_degree
        raw_kv_heads = tc.num_key_value_heads
        if raw_kv_heads < tp_degree:
            replicated_kv_heads = tp_degree
        else:
            replicated_kv_heads = raw_kv_heads
        self.kv_heads_per_rank = replicated_kv_heads // tp_degree

        # Conv1d on concatenated QKV (NOT Z)
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 10240
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Input projections (nn.Linear — NOT sharded by NxDI TP, replicated on all ranks)
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
        self.norm = Qwen3MoeRMSNorm(self.head_v_dim, eps=self.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # State buffers for CTE -> TKG carry-over
        alloc_batch_size = getattr(config.neuron_config, "max_batch_size", 1)
        self._phase_batch_size = getattr(config.neuron_config, "batch_size", 1)
        self.recurrent_state_buffer = nn.Parameter(
            torch.zeros(
                alloc_batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=config.neuron_config.torch_dtype,
            ),
            requires_grad=False,
        )
        self.conv_state_buffer = nn.Parameter(
            torch.zeros(
                alloc_batch_size,
                self.conv_dim,
                self.conv_kernel_size - 1,
                dtype=config.neuron_config.torch_dtype,
            ),
            requires_grad=False,
        )

    def _recurrent_step(self, query, key, value, g, beta, recurrent_state):
        """Single-step recurrent update for token generation."""
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale

        q_t = query[:, :, 0]
        k_t = key[:, :, 0]
        v_t = value[:, :, 0]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, 0].unsqueeze(-1)

        new_state = recurrent_state * g_t
        kv_mem = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)

        return output.unsqueeze(2), new_state

    def _nki_recurrent_forward(self, query, key, value, g, beta):
        """Full-sequence recurrent forward using NKI kernel for context encoding."""
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        BH = B * H
        query_flat = query.reshape(BH, S, k_dim).contiguous()
        key_flat = key.reshape(BH, S, k_dim).contiguous()
        value_flat = value.reshape(BH, S, v_dim).contiguous()

        g_flat = g.reshape(BH, S).unsqueeze(-1).expand(-1, -1, v_dim).contiguous()
        beta_flat = beta.reshape(BH, S).unsqueeze(-1).expand(-1, -1, v_dim).contiguous()

        outputs = []
        states = []
        for bh in range(BH):
            out_bh, state_bh = _deltanet_nki_kernel_state(
                query_flat[bh],
                key_flat[bh],
                value_flat[bh],
                g_flat[bh],
                beta_flat[bh],
            )
            outputs.append(out_bh)
            states.append(state_bh)

        output = torch.stack(outputs, dim=0)
        output = output.reshape(B, H, S, v_dim)

        final_state = torch.stack(states, dim=0)
        final_state = final_state.reshape(B, H, k_dim, v_dim)

        return output, final_state

    def _nki_chunked_forward(
        self, query, key, value, g, beta, output_final_state=False
    ):
        """Chunked NKI kernel forward for context encoding (prefill)."""
        chunk_size = 128

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size

        num_chunks = total_seq_len // chunk_size
        g_reshaped = g.reshape(B, H, num_chunks, chunk_size)
        g_cs = g_reshaped.cumsum(dim=-1)
        g_last_per_chunk = g_cs[:, :, :, -1:]
        g_last_expanded = g_last_per_chunk.expand(-1, -1, -1, chunk_size)

        query_chunks = query.reshape(B, H, num_chunks, chunk_size, k_dim)
        key_chunks = key.reshape(B, H, num_chunks, chunk_size, k_dim)
        value_chunks = value.reshape(B, H, num_chunks, chunk_size, v_dim)

        beta_chunks = (
            beta.reshape(B, H, num_chunks, chunk_size)
            .unsqueeze(-1)
            .expand(-1, -1, -1, -1, v_dim)
        )
        gc_chunks = g_cs.unsqueeze(-1).expand(-1, -1, -1, -1, v_dim)
        gl_chunks = g_last_expanded.unsqueeze(-1).expand(-1, -1, -1, -1, v_dim)

        BH = B * H
        query_chunks = query_chunks.reshape(
            BH, num_chunks, chunk_size, k_dim
        ).contiguous()
        key_chunks = key_chunks.reshape(BH, num_chunks, chunk_size, k_dim).contiguous()
        value_chunks = value_chunks.reshape(
            BH, num_chunks, chunk_size, v_dim
        ).contiguous()
        beta_chunks = beta_chunks.reshape(
            BH, num_chunks, chunk_size, v_dim
        ).contiguous()
        gc_chunks = gc_chunks.reshape(BH, num_chunks, chunk_size, v_dim).contiguous()
        gl_chunks = gl_chunks.reshape(BH, num_chunks, chunk_size, v_dim).contiguous()

        device = query.device
        lower_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=device),
            diagonal=-1,
        )
        identity_mat = torch.eye(chunk_size, dtype=torch.float32, device=device)
        lower_mask_diag = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=device),
            diagonal=0,
        )

        all_outputs = []
        all_states = []
        for bh in range(BH):
            state = torch.zeros(k_dim, v_dim, dtype=torch.float32, device=device)

            head_chunks = []
            for c_idx in range(num_chunks):
                q_chunk = query_chunks[bh, c_idx].contiguous()
                k_chunk = key_chunks[bh, c_idx].contiguous()
                v_chunk = value_chunks[bh, c_idx].contiguous()
                beta_chunk = beta_chunks[bh, c_idx].contiguous()
                gc_chunk = gc_chunks[bh, c_idx].contiguous()
                gl_chunk = gl_chunks[bh, c_idx].contiguous()

                out_chunk, state = _deltanet_nki_chunk_step(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    beta_chunk,
                    gc_chunk,
                    gl_chunk,
                    state,
                    lower_mask,
                    identity_mat,
                    lower_mask_diag,
                )
                head_chunks.append(out_chunk)

            head_output = torch.cat(head_chunks, dim=0)
            all_outputs.append(head_output)
            all_states.append(state)

        output = torch.stack(all_outputs, dim=0)
        output = output.reshape(B, H, total_seq_len, v_dim)
        output = output[:, :, :S]

        if output_final_state:
            final_state = torch.stack(all_states, dim=0)
            last_recurrent_state = final_state.reshape(B, H, k_dim, v_dim)
        else:
            last_recurrent_state = None

        return output, last_recurrent_state

    def _fused_chunked_forward(
        self, query, key, value, g, beta, output_final_state=False
    ):
        """Fused single-kernel chunked forward for CTE — SSD-style.

        Processes all chunks in a single NKI kernel call per (B,H) pair.
        State persists in SBUF across chunks (no HBM round-trips).
        Cumsum of g computed in-kernel via tensor_tensor_scan.

        This is the optimized version of _nki_chunked_forward with:
          1. Single kernel call per (B,H) instead of B*H*num_chunks
          2. State in SBUF across all chunks (biggest perf win)
          3. In-kernel cumsum (avoids PyTorch cumsum overhead)
          4. tensor_scalar for broadcasts (no explicit loops)
        """
        chunk_size = 128

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        # Pad sequence to multiple of chunk_size
        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size

        BH = B * H
        # Flatten to (BH, S, dim) for per-(b,h) kernel calls
        query_flat = query.reshape(BH, total_seq_len, k_dim).contiguous()
        key_flat = key.reshape(BH, total_seq_len, k_dim).contiguous()
        value_flat = value.reshape(BH, total_seq_len, v_dim).contiguous()

        # g and beta: (BH, S) -> (BH, S, 1) for the kernel's (S, 1) input layout
        g_flat = g.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()
        beta_flat = beta.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()

        # Create constant mask tensors (shared across all B*H calls)
        device = query.device
        lower_mask = torch.tensor(
            _make_lower_mask(), dtype=torch.float32, device=device
        )
        identity_mat = torch.tensor(
            _make_identity(), dtype=torch.float32, device=device
        )
        lower_mask_diag = torch.tensor(
            _make_lower_mask_diag(), dtype=torch.float32, device=device
        )

        all_outputs = []
        all_states = []
        for bh in range(BH):
            out_bh, state_bh = _deltanet_fused_kernel(
                query_flat[bh],  # (S, 128)
                key_flat[bh],  # (S, 128)
                value_flat[bh],  # (S, 128)
                g_flat[bh],  # (S, 1) — RAW g, not cumsum
                beta_flat[bh],  # (S, 1) — sigmoid(b)
                lower_mask,  # (128, 128)
                identity_mat,  # (128, 128)
                lower_mask_diag,  # (128, 128)
            )
            all_outputs.append(out_bh)
            all_states.append(state_bh)

        output = torch.stack(all_outputs, dim=0)
        output = output.reshape(B, H, total_seq_len, v_dim)
        output = output[:, :, :S]

        if output_final_state:
            final_state = torch.stack(all_states, dim=0)
            last_recurrent_state = final_state.reshape(B, H, k_dim, v_dim)
        else:
            last_recurrent_state = None

        return output, last_recurrent_state

    def _sequential_forward(self, query, key, value, g, beta, output_final_state=False):
        """Sequential full-sequence gated delta rule for CTE.

        Uses the same per-step recurrence as _recurrent_step but loops over the
        full sequence.  Avoids the slice-assignment loop in _chunk_forward that
        may compile incorrectly on Neuron/XLA.
        """
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        state = query.new_zeros(B, H, k_dim, v_dim)
        all_outputs = []
        for t in range(S):
            q_t = query[:, :, t]  # (B, H, K)
            k_t = key[:, :, t]  # (B, H, K)
            v_t = value[:, :, t]  # (B, H, V)
            beta_t = beta[:, :, t].unsqueeze(-1)  # (B, H, 1)
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

            # Gated delta rule
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, V)
            delta = (v_t - kv_mem) * beta_t  # (B, H, V)
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # (B, H, K, V)

            o_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, V)
            all_outputs.append(o_t.unsqueeze(2))

        output = torch.cat(all_outputs, dim=2)  # (B, H, S, V)
        final_state = state if output_final_state else None
        return output, final_state

    def _chunk_forward(self, query, key, value, g, beta, output_final_state=False):
        """Chunk-based forward for context encoding (prefill)."""
        chunk_size = 64

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

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

        g = g.cumsum(dim=-1)
        decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()

        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        last_recurrent_state = torch.zeros(
            B, H, k_dim, v_dim, dtype=query.dtype, device=query.device
        )
        core_attn_out = torch.zeros_like(value)
        mask2 = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=1,
        )

        for i in range(num_chunks):
            q_i = query[:, :, i]
            k_i = key[:, :, i]
            v_i = value[:, :, i]

            attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
                mask2, 0
            )

            v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
            v_new = v_i - v_prime

            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn_i @ v_new

            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (
                    k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                ).transpose(-1, -2)
                @ v_new
            )

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
        """Forward pass compatible with NxDI decoder layer interface."""
        batch_size, seq_len, _ = hidden_states.shape

        seq_ids = kwargs.get("seq_ids", None)
        is_decode = past_key_value is not None

        # Padding mask for DeltaNet: [B, S, 1] with 1.0 for real tokens, 0.0 for padding.
        # Passed from get_model_output where it's computed from input_ids != pad_token_id.
        # Embeddings are already zeroed for padding tokens; this mask additionally
        # zeros the decay gate so the recurrent state is preserved unchanged
        # through padding positions (no spurious decay).
        valid_mask_1d = kwargs.get("deltanet_padding_mask", None)  # [B, S, 1] or None

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
            qkv = self.in_proj_qkv(hidden_states)
            z = self.in_proj_z(hidden_states)
            b = self.in_proj_b(hidden_states)
            a = self.in_proj_a(hidden_states)

        # Split QKV
        query = qkv[..., : self.key_dim]
        key = qkv[..., self.key_dim : self.key_dim * 2]
        value = qkv[..., self.key_dim * 2 :]

        # Causal Conv1d on QKV
        mixed = torch.cat([query, key, value], dim=-1)
        mixed = mixed.transpose(1, 2)

        if is_decode:
            if seq_ids is not None:
                conv_state = torch.index_select(self.conv_state_buffer, 0, seq_ids)
            else:
                conv_state = self.conv_state_buffer[:batch_size]
            conv_input = torch.cat([conv_state, mixed], dim=-1)

            w = self.conv1d.weight.squeeze(1)
            conv_out = torch.zeros_like(mixed)
            for k in range(4):
                conv_out = (
                    conv_out
                    + w[:, k].unsqueeze(0).unsqueeze(-1) * conv_input[:, :, k : k + 1]
                )
            mixed_post_conv = F.silu(conv_out)

            new_conv_state = torch.cat([conv_state[:, :, 1:], mixed], dim=-1)
            alloc_bs = self.conv_state_buffer.shape[0]
            if seq_ids is not None:
                # BS=1 optimization: scatter to index 0 of size-1 buffer = direct replacement
                # Add buffer dependency for input_output_alias
                new_conv_state = (
                    new_conv_state.to(self.conv_state_buffer.dtype)
                    + self.conv_state_buffer * 0
                )
            elif batch_size < alloc_bs:
                pad_size = alloc_bs - batch_size
                new_conv_state = torch.cat(
                    [
                        new_conv_state,
                        self.conv_state_buffer[batch_size:] * 0,
                    ],
                    dim=0,
                )
            else:
                new_conv_state = new_conv_state + self.conv_state_buffer * 0
        else:
            mixed_post_conv = F.silu(self.conv1d(mixed)[:, :, :seq_len])

            if valid_mask_1d is not None:
                # valid_mask_1d is [B, S, 1]; count valid tokens per batch
                num_valid = (
                    valid_mask_1d.squeeze(-1).sum(dim=-1, keepdim=True).long()
                )  # [B, 1]
                idx_base = num_valid - 3
                idx_base = idx_base.clamp(min=0)
                offsets = torch.arange(3, device=mixed.device).unsqueeze(0)
                gather_idx = idx_base + offsets  # [B, 3]
                gather_idx = gather_idx.unsqueeze(1).expand(-1, self.conv_dim, -1)
                new_conv_state = torch.gather(mixed, 2, gather_idx)
            else:
                new_conv_state = mixed[:, :, -3:].contiguous()

            alloc_bs = self.conv_state_buffer.shape[0]
            if seq_ids is not None:
                # BS=1 optimization: scatter to index 0 = direct replacement
                new_conv_state = (
                    new_conv_state.to(self.conv_state_buffer.dtype)
                    + self.conv_state_buffer * 0
                )
            elif batch_size < alloc_bs:
                pad_size = alloc_bs - batch_size
                new_conv_state = torch.cat(
                    [
                        new_conv_state,
                        torch.zeros(
                            pad_size,
                            self.conv_dim,
                            self.conv_kernel_size - 1,
                            dtype=new_conv_state.dtype,
                            device=new_conv_state.device,
                        ),
                    ],
                    dim=0,
                )
                new_conv_state = new_conv_state + self.conv_state_buffer * 0
            else:
                new_conv_state = new_conv_state + self.conv_state_buffer * 0

        mixed_post_conv = mixed_post_conv.transpose(1, 2)

        # Zero out conv1d output for padding positions.
        # Conv1d with kernel_size=4 leaks real token info into the first
        # few padding positions.  Zeroing here ensures Q, K, V are exactly
        # zero for all padding positions so the recurrence is unaffected.
        if valid_mask_1d is not None:
            mixed_post_conv = (
                mixed_post_conv * valid_mask_1d
            )  # [B, S, conv_dim] * [B, S, 1]

        query = mixed_post_conv[..., : self.key_dim]
        key = mixed_post_conv[..., self.key_dim : self.key_dim * 2]
        value = mixed_post_conv[..., self.key_dim * 2 :]

        # Reshape to heads
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # Compute gating
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if valid_mask_1d is not None:
            # Zero g for padding → alpha=exp(0)=1 → state preserved through padding
            # Zero beta for padding → no state update from padding tokens
            mask_2d = valid_mask_1d.squeeze(-1).float()  # [B, S]
            g = g * mask_2d.unsqueeze(-1)
            beta = beta * mask_2d.unsqueeze(-1)

        # Expand K heads to match V heads (16 -> 48) using expand+reshape
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads  # 3
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

        # Transpose to (B, H, S, dim)
        query = query.transpose(1, 2).contiguous().float()
        key = key.transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        if is_decode:
            # TKG: single-step recurrent update
            if seq_ids is not None:
                recurrent_state = torch.index_select(
                    self.recurrent_state_buffer, 0, seq_ids
                ).float()
            else:
                recurrent_state = self.recurrent_state_buffer[:batch_size].float()

            output, new_state = self._recurrent_step(
                query, key, value, g, beta, recurrent_state
            )
            new_state_bf16 = new_state.to(self.recurrent_state_buffer.dtype)
            alloc_bs = self.recurrent_state_buffer.shape[0]
            if seq_ids is not None:
                # BS=1 optimization: scatter to index 0 of size-1 buffer = direct replacement
                # Add buffer dependency for input_output_alias
                new_rec_state = new_state_bf16 + self.recurrent_state_buffer * 0
            elif batch_size < alloc_bs:
                new_rec_state = torch.cat(
                    [
                        new_state_bf16,
                        self.recurrent_state_buffer[batch_size:] * 0,
                    ],
                    dim=0,
                )
            else:
                new_rec_state = new_state_bf16 + self.recurrent_state_buffer * 0
        else:
            # CTE: fused NKI kernel by default (PyTorch _chunk_forward can hit
            # neuronx-cc codegen ICE NCC_INLA001 with these DeltaNet dimensions).
            # Override with env vars for debugging/benchmarking.
            use_nki_fused = os.environ.get("USE_NKI_FUSED", "1") != "0"
            use_nki_chunked = os.environ.get("USE_NKI_CHUNKED") == "1"
            use_nki = os.environ.get("USE_NKI") == "1"
            use_sequential = os.environ.get("DELTANET_SEQUENTIAL") == "1"
            use_pytorch_chunk = os.environ.get("USE_PYTORCH_CHUNK") == "1"

            if use_pytorch_chunk:
                output, final_state = self._chunk_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_nki_chunked:
                output, final_state = self._nki_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_nki:
                output, final_state = self._nki_recurrent_forward(
                    query, key, value, g, beta
                )
            elif use_sequential:
                output, final_state = self._sequential_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_nki_fused:
                output, final_state = self._fused_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            else:
                output, final_state = self._fused_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )

            if final_state is not None:
                final_state_bf16 = final_state.to(self.recurrent_state_buffer.dtype)
                alloc_bs = self.recurrent_state_buffer.shape[0]
                if seq_ids is not None:
                    # BS=1 optimization: scatter to index 0 of size-1 buffer = direct replacement
                    # Add buffer dependency for input_output_alias
                    new_rec_state = final_state_bf16 + self.recurrent_state_buffer * 0
                elif batch_size < alloc_bs:
                    new_rec_state = torch.cat(
                        [
                            final_state_bf16,
                            torch.zeros(
                                alloc_bs - batch_size,
                                self.num_v_heads,
                                self.head_k_dim,
                                self.head_v_dim,
                                dtype=final_state_bf16.dtype,
                                device=final_state_bf16.device,
                            ),
                        ],
                        dim=0,
                    )
                    new_rec_state = new_rec_state + self.recurrent_state_buffer * 0
                else:
                    new_rec_state = final_state_bf16 + self.recurrent_state_buffer * 0
            else:
                new_rec_state = self.recurrent_state_buffer * 1

        # Output: norm, gate, project
        output = output.to(hidden_states.dtype)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = self.norm(output)
        z_gate = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = output * F.silu(z_gate)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(output)

        # Return dummy KV for KVCacheManager
        dummy_k = torch.zeros(
            batch_size,
            self.kv_heads_per_rank,
            seq_len,
            self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dummy_v = torch.zeros_like(dummy_k)

        return output, (dummy_k, dummy_v), new_rec_state, new_conv_state


# ============================================================
# InferenceConfig (Dense -- no MoE)
# ============================================================


class Qwen35InferenceConfig(InferenceConfig):
    """Config for Qwen3.5-4B (dense) with hybrid DeltaNet + Attention."""

    def __init__(self, *args, **kwargs):
        # Set defaults BEFORE super().__init__() because it calls validate_config()
        # which checks get_required_attributes(). These can be overridden by
        # kwargs or load_config.

        # Layer types for hybrid dispatch: [3 DeltaNet + 1 GQA] repeated.
        if "layer_types" not in kwargs and not any(
            hasattr(a, "layer_types") for a in args if hasattr(a, "__dict__")
        ):
            num_layers = kwargs.get("num_hidden_layers", 32)
            if num_layers % 4 != 0:
                raise ValueError(
                    f"Qwen3.5 hybrid layer count must be divisible by 4, got {num_layers}"
                )
            layer_types = []
            for _ in range(num_layers // 4):
                layer_types.extend(
                    [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ]
                )
            kwargs.setdefault("layer_types", layer_types)

        # DeltaNet-specific config defaults
        kwargs.setdefault("linear_num_value_heads", 32)
        kwargs.setdefault("linear_num_key_heads", 16)
        kwargs.setdefault("linear_key_head_dim", 128)
        kwargs.setdefault("linear_value_head_dim", 128)
        kwargs.setdefault("linear_conv_kernel_dim", 4)

        super().__init__(*args, **kwargs)

        # Attention output gate
        self.attn_output_gate = getattr(self, "attn_output_gate", True)

        # Partial RoPE
        self.partial_rotary_factor = getattr(self, "partial_rotary_factor", 0.25)
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)  # 64

        # mRoPE (multimodal RoPE) for VL support
        rope_params = getattr(self, "rope_parameters", {}) or {}
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self.mrope_interleaved = rope_params.get("mrope_interleaved", True)

        # Standard HF config attributes expected by NxDI
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
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
        return NeuronConfig


# ============================================================
# Attention (standard GQA for 16 of 64 layers)
# With output gate: q_proj is 2x sized, split into (query, gate)
# With partial RoPE: only first rope_dim dimensions get rotary
# ============================================================


class Qwen35MRoPEEmbedding(nn.Module):
    """Multimodal Rotary Position Embedding (mRoPE) for Qwen3.5.

    Handles 3D position information (temporal, height, width) for VL models.
    Position IDs have shape (3, batch_size, seq_len) for T/H/W dimensions.
    For text-only (2D position_ids), broadcasts to 3D with identical positions.
    """

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim  # 256
        self.rope_dim = config.rope_dim  # 64
        self.mrope_section = config.mrope_section  # [11, 11, 10]
        self.mrope_interleaved = getattr(config, "mrope_interleaved", True)
        self.rope_theta = config.rope_theta

        # Validate mrope_section sums to rope_dim // 2 = 32
        assert sum(self.mrope_section) == self.rope_dim // 2, (
            f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, "
            f"expected {self.rope_dim // 2}"
        )

    def forward(self, x, position_ids_3d):
        """Compute cos/sin from 3D position IDs.

        Args:
            x: hidden_states (for device/dtype inference)
            position_ids_3d: (3, batch_size, seq_len) -- T, H, W positions

        Returns:
            cos: (batch_size, seq_len, rope_dim)
            sin: (batch_size, seq_len, rope_dim)
        """
        device = x.device
        dtype = torch.float32

        sections = self.mrope_section  # [11, 11, 10]
        cos_parts = []
        sin_parts = []

        freq_offset = 0
        for axis_idx, section_size in enumerate(sections):
            pos = position_ids_3d[axis_idx].float()  # (batch, seq_len)

            dim_pairs = section_size  # number of (cos, sin) pairs for this axis
            freqs = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(0, dim_pairs * 2, 2, dtype=dtype, device=device)
                    / (self.rope_dim)
                )
            )  # (dim_pairs,)

            # freqs: (dim_pairs,), pos: (B, S) -> angles: (B, S, dim_pairs)
            angles = pos.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)

            cos_parts.append(angles.cos())
            sin_parts.append(angles.sin())

        # Concatenate: (B, S, 32)
        cos = torch.cat(cos_parts, dim=-1)
        sin = torch.cat(sin_parts, dim=-1)

        if self.mrope_interleaved:
            # Interleave to (B, S, 64): [c0, c0, c1, c1, ...] for rotate_half
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        else:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

        return cos, sin


class NeuronQwen35Attention(NeuronAttentionBase):
    """Standard GQA attention for Qwen3.5 with output gate and partial RoPE.

    16 Q heads, 4 KV heads (4:1 GQA), head_dim=256 for the 4B dense model.
    q_proj is doubled (query + gate), split at load time.
    Only first rope_dim=64 of head_dim=256 gets rotary encoding.

    Uses NeuronAttentionBase infrastructure for QKV projection, KV cache,
    RoPE, and attention computation. Overrides forward() to insert the
    sigmoid output gate between attention output and o_proj.
    """

    def __init__(self, config):
        # Partial RoPE: create mRoPE embedding with rope_dim (64)
        self.rope_dim = config.rope_dim  # 64 = head_dim * partial_rotary_factor

        # Create QK norm modules (will be passed to base class)
        rms_norm_eps = config.rms_norm_eps
        q_ln = get_rmsnorm_cls()(config.head_dim, rms_norm_eps)
        k_ln = get_rmsnorm_cls()(config.head_dim, rms_norm_eps)

        # Partial RoPE: use standard RotaryEmbedding.
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

        # Separate mRoPE module for VL 3D position_ids
        self.mrope_emb = Qwen35MRoPEEmbedding(config)

        # Output gate projection: hidden_size -> num_heads * head_dim
        # Populated from the second half of q_proj during state dict conversion.
        self.output_gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            gather_output=False,
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
        Q_orig_dtype = Q.dtype
        q_rope = Q[..., : self.rope_dim]  # (B, H, S, 64)
        q_pass = Q[..., self.rope_dim :]  # (B, H, S, 192)
        k_rope = K[..., : self.rope_dim]
        k_pass = K[..., self.rope_dim :]

        # Apply RoPE only to the rope portion
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos_cache, sin_cache)

        # Concatenate back (ensure bf16 is maintained)
        Q = torch.cat([q_rope, q_pass], dim=-1).to(Q_orig_dtype)
        K = torch.cat([k_rope, k_pass], dim=-1).to(Q_orig_dtype)

        return Q, K, cos_cache, sin_cache

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask=None):
        """Prefill path with NKI flash attention for head_dim=256."""
        head_dim = Q.shape[-1]

        # Option B: nkilib flash attention for head_dim > 128
        if _nkilib_flash_attn is not None:
            q_contig = Q.contiguous()
            k_contig = K.contiguous()
            v_contig = V.contiguous()
            scale = 1.0 / math.sqrt(head_dim)
            result = _nkilib_flash_attn(
                q_contig, k_contig, v_contig, scale=scale, use_causal_mask=True
            )
            return result, None

        # Option A: kernel patched globally
        if NKILIB_PATCH_ACTIVE:
            return _flash_fwd_call(Q, K, V, use_causal_mask=True), None

        # Fallback: softmax path (use 3D tensors to avoid compiler ICE with 4D patterns)
        if head_dim > 128:
            # GQA: expand K/V heads to match Q heads
            num_q_heads = Q.shape[1]
            num_kv_heads = K.shape[1]
            if num_q_heads != num_kv_heads:
                kv_rep = num_q_heads // num_kv_heads
                K = (
                    K.unsqueeze(2)
                    .expand(-1, -1, kv_rep, -1, -1)
                    .reshape(bsz, num_q_heads, q_len, head_dim)
                )
                V = (
                    V.unsqueeze(2)
                    .expand(-1, -1, kv_rep, -1, -1)
                    .reshape(bsz, num_q_heads, q_len, head_dim)
                )
            # Reshape to 3D (B*H, S, d) to avoid neuronx-cc codegen ICE with 4D
            # attention weight tensors (NCC_INLA001: Expected 2D tensor but got 4D AP)
            Q_3d = Q.reshape(bsz * num_q_heads, q_len, head_dim)
            K_3d = K.reshape(bsz * num_q_heads, q_len, head_dim)
            V_3d = V.reshape(bsz * num_q_heads, q_len, head_dim)
            attn_weights = torch.bmm(Q_3d, K_3d.transpose(-1, -2)) / math.sqrt(head_dim)
            # Build causal mask for 3D: (1, S, S) broadcast over B*H
            causal_mask = torch.triu(
                torch.full(
                    (q_len, q_len),
                    -65504.0,
                    dtype=attn_weights.dtype,
                    device=attn_weights.device,
                ),
                diagonal=1,
            ).unsqueeze(0)
            attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                Q.dtype
            )
            attn_output = torch.bmm(attn_weights, V_3d)
            # Reshape back to 4D (B, H, S, d)
            return attn_output.reshape(bsz, num_q_heads, q_len, head_dim), None

        return _flash_fwd_call(Q, K, V, use_causal_mask=True), None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        adapter_ids=None,
        active_mask=None,
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

        # Use standard 2D position_ids for prep_qkv_tensors.
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
            tkg_mask = attention_mask
            if tkg_mask is not None and tkg_mask.ndim == 2:
                tkg_mask = tkg_mask.unsqueeze(1).unsqueeze(2)  # (B, S) -> (B, 1, 1, S)
            attn_output = self.compute_for_token_gen(
                Q, K, V, position_ids, past_key_value, tkg_mask, active_mask
            )

        # attn_output is (B, H, S, head_dim) -- transpose to (B, S, H*head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Apply sigmoid output gate BEFORE o_proj (matching HF reference)
        attn_output = attn_output * torch.sigmoid(gate)

        # Apply o_proj
        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

        # Ensure K, V are in model dtype (bf16) for KV cache update
        # (prevents mixed-precision dynamic-update-slice in neuronx-cc)
        K = K.to(self.torch_dtype)
        V = V.to(self.torch_dtype)
        past_key_value = (K, V)
        return attn_output, past_key_value, cos_cache, sin_cache


# ============================================================
# Dense MLP (replaces MoE)
# ============================================================


class Qwen35MLP(nn.Module):
    """Dense SwiGLU MLP for Qwen3.5-4B.

    gate_proj: hidden_size -> intermediate_size (2560 -> 9216)
    up_proj:   hidden_size -> intermediate_size (2560 -> 9216)
    down_proj: intermediate_size -> hidden_size (9216 -> 2560)

    output = down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, config):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
        )

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = F.silu(gate) * up
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


# ============================================================
# Decoder Layer (hybrid dispatch -- DeltaNet or GQA + Dense MLP)
# ============================================================


class NeuronQwen35DecoderLayer(nn.Module):
    """Hybrid decoder layer: dispatches to DeltaNet or standard attention.
    Uses dense MLP for all layers (no MoE).
    """

    def __init__(self, config: Qwen35InferenceConfig, layer_idx: int):
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

        # Dense MLP (all layers)
        self.mlp = Qwen35MLP(config)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
        residual = hidden_states

        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            # DeltaNet path
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
        else:
            deltanet_states = None
            # Standard attention path
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

        # Dense MLP FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

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


class NeuronQwen35Model(NeuronBaseModel):
    def setup_attr_for_model(self, config: Qwen35InferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen35InferenceConfig):
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

        # mRoPE embedding for VL
        self.mrope_emb = Qwen35MRoPEEmbedding(config)

    @property
    def _deltanet_state_params(self):
        """Return DeltaNet state nn.Parameters in alias order."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "linear_attn"):
                params.append(layer.linear_attn.recurrent_state_buffer)
                params.append(layer.linear_attn.conv_state_buffer)
        return params

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        """Scatter vision embeddings into text input embeddings at image token positions."""
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
        """Override to collect DeltaNet state tensors from decoder layers."""
        batch_size, seq_length = input_ids.shape[:2]
        if self.config.neuron_config.layer_boundary_markers:
            input_ids = ModuleMarkerStartWrapper()(input_ids)

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # CRITICAL: Zero out embeddings for padding tokens so DeltaNet recurrence
        # is not polluted. DeltaNet has no attention mask -- it processes all
        # sequence positions through a linear recurrence.  Padding tokens have
        # real embedding vectors which corrupt the recurrence state.
        # The mask is [B, S, 1] float with 1.0 for real tokens, 0.0 for padding.
        deltanet_padding_mask = (
            (input_ids != self.padding_idx).unsqueeze(-1).to(inputs_embeds.dtype)
        )
        if is_for_context_encoding:
            inputs_embeds = inputs_embeds * deltanet_padding_mask

        # Vision embedding injection
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
        deltanet_state_tensors = []
        cos_cache = None
        sin_cache = None

        # Convert 2D attention_mask to 4D causal mask for CTE
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and is_for_context_encoding
        ):
            causal = torch.ones(
                (seq_length, seq_length),
                dtype=torch.bool,
                device=attention_mask.device,
            ).tril()
            padding_4d = attention_mask[:, None, None, :].to(torch.bool)
            attention_mask = (causal[None, None, :, :] & padding_4d).to(
                attention_mask.dtype
            )

        # Pre-compute mRoPE cos/sin
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
                deltanet_padding_mask=deltanet_padding_mask,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            kv = layer_outputs[1]
            next_decoder_cache += (kv,)
            cos_cache, sin_cache = layer_outputs[2:4]

            # Collect DeltaNet state tensors
            deltanet_states = layer_outputs[5] if len(layer_outputs) > 5 else None
            if deltanet_states is not None:
                deltanet_state_tensors.append(deltanet_states[0])
                deltanet_state_tensors.append(deltanet_states[1])

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
        """Override base forward to append DeltaNet state tensors to output."""
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
                pass
            else:
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
# State Dict Converter (Dense -- no MoE weight handling)
# ============================================================


def convert_qwen35_hf_to_neuron_state_dict(neuron_state_dict, config):
    """Convert HF Qwen3.5 (dense) weights to NxDI format.

    Weight mappings per layer type:

    DeltaNet layers (linear_attention):
      HF: layers.X.linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
          conv1d, A_log, dt_bias, norm, out_proj}
      NxDI: same names (no remapping needed)

    Full attention layers:
      HF: layers.X.self_attn.q_proj.weight: (num_heads*head_dim*2, hidden) -- doubled for gate
      NxDI: layers.X.self_attn.Wqkv.weight (fused Q+K+V, gate separated)
             layers.X.self_attn.output_gate_proj.weight (gate part)
      HF: layers.X.self_attn.{k_proj, v_proj, o_proj, q_norm, k_norm}
      NxDI: layers.X.self_attn.{..., q_layernorm, k_layernorm}

    Dense MLP (all layers):
      HF: layers.X.mlp.{gate_proj, up_proj, down_proj}.weight
      NxDI: layers.X.mlp.{gate_proj, up_proj, down_proj}.weight (same names)
    """
    # Add rank_util
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0,
        config.neuron_config.tp_degree,
        dtype=torch.int32,
    )

    # CRITICAL: Convert (1+weight) RMSNorm weights to standard RMSNorm weights.
    # Qwen3.5 uses RMSNorm with `output = norm(x) * (1 + weight)` where weight
    # is initialized to zeros. Standard NxDI RMSNorm uses `output = norm(x) * weight`
    # where weight is initialized to ones. To convert: new_weight = old_weight + 1.0
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
                logger.debug(
                    f"[NORM FIX] {nk}: mean {old_val.float().mean():.4f} -> {neuron_state_dict[nk].mean():.4f}"
                )
        else:
            if "layers.0." in nk or nk == "norm.weight":
                logger.warning(f"[NORM FIX] key not found: {nk}")

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

            # q_proj is doubled: (num_heads * head_dim * 2, hidden_size)
            # INTERLEAVED: [head0_query(head_dim) | head0_gate(head_dim) | head1_query(head_dim) | ...]
            q_proj_key = f"layers.{l}.self_attn.q_proj.weight"
            if q_proj_key in neuron_state_dict:
                q_proj_w = neuron_state_dict.pop(q_proj_key)
                num_heads = config.num_attention_heads
                head_dim = config.head_dim
                q_proj_w = q_proj_w.reshape(num_heads, head_dim * 2, config.hidden_size)
                query_w = q_proj_w[:, :head_dim, :]
                gate_w = q_proj_w[:, head_dim:, :]
                query_w = query_w.reshape(num_heads * head_dim, config.hidden_size)
                gate_w = gate_w.reshape(num_heads * head_dim, config.hidden_size)

                neuron_state_dict[q_proj_key] = query_w
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

        # Dense MLP: no weight conversion needed -- HF and NxDI use same names
        # HF: layers.X.mlp.{gate_proj, up_proj, down_proj}.weight
        # NxDI: layers.X.mlp.{gate_proj, up_proj, down_proj}.weight

        gc.collect()

    return neuron_state_dict


# ============================================================
# Custom ModelWrapper and DecoderModelInstance for DeltaNet state aliasing
# ============================================================


class Qwen35DecoderModelInstance(DecoderModelInstance):
    """Custom DecoderModelInstance that adds DeltaNet state buffers to input_output_aliases."""

    def get(self, bucket_rank, **kwargs):
        """Override to add DeltaNet state aliases after KV cache aliases."""
        module, input_output_aliases = super().get(bucket_rank, **kwargs)

        num_output_from_trace = 1 if not self.neuron_config.output_logits else 2

        if module.kv_mgr is not None:
            num_kv = len(module.kv_mgr.past_key_values)
        else:
            num_kv = 0

        state_start_idx = num_output_from_trace + num_kv

        if hasattr(module, "_deltanet_state_params"):
            for i, param in enumerate(module._deltanet_state_params):
                input_output_aliases[param] = state_start_idx + i

        return module, input_output_aliases


class Qwen35ModelWrapper(ModelWrapper):
    """Custom ModelWrapper for VL support with mRoPE and vision inputs."""

    def get_model_instance(self):
        return Qwen35DecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )

    def input_generator(self):
        """Generate inputs including mrope_position_ids, vision_embeddings, and vision_mask."""
        base_inputs = super().input_generator()
        extended_inputs = []

        for bucket_inputs in base_inputs:
            input_ids = bucket_inputs[0]
            batch_size = input_ids.shape[0]
            n_active_tokens = input_ids.shape[1]

            is_cte = n_active_tokens > 1

            if is_cte:
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
                    fill_value=n_active_tokens - 1,
                    dtype=torch.int32,
                )
            else:
                mrope_position_ids = torch.zeros((0,), dtype=torch.int32)
                vision_embeddings = torch.zeros(
                    (0,), dtype=self.config.neuron_config.torch_dtype
                )
                vision_mask = torch.zeros((0,), dtype=torch.int32)

            padded = list(bucket_inputs)
            while len(padded) < 21:
                padded.append(torch.zeros((0,), dtype=torch.int32))
            padded.append(mrope_position_ids)  # position 21
            padded.append(vision_embeddings)  # position 22
            padded.append(vision_mask)  # position 23

            extended_inputs.append(tuple(padded))

        return extended_inputs

    def pad_inputs(self, *args, pad_type="first_fit"):
        """Override to pad mrope_position_ids and vision inputs to bucket size."""
        orig_mrope = args[21] if len(args) >= 22 else None
        orig_vis_emb = args[22] if len(args) >= 23 else None
        orig_vis_mask = args[23] if len(args) >= 24 else None

        padded_args = super().pad_inputs(*args, pad_type=pad_type)

        if len(padded_args) >= 24 and orig_mrope is not None:
            padded_seq_len = padded_args[0].shape[1]
            batch_size = padded_args[0].shape[0]
            is_cte = padded_seq_len > 1

            if is_cte:
                current_mrope = orig_mrope
                current_vis_emb = orig_vis_emb
                current_vis_mask = orig_vis_mask

                if (
                    current_mrope.ndim == 3
                    and current_mrope.shape[-1] != padded_seq_len
                ):
                    orig_len = current_mrope.shape[-1]
                    pad_size = padded_seq_len - orig_len
                    last_pos = current_mrope[:, :, -1:]
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
                    mrope_position_ids = (
                        torch.arange(0, padded_seq_len, dtype=torch.int32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(3, batch_size, -1)
                        .contiguous()
                    )

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

                padded_args = list(padded_args)
                padded_args[23] = padded_args[23].clamp(max=padded_seq_len - 1)
                padded_args = tuple(padded_args)

        return padded_args


# ============================================================
# Top-Level Model
# ============================================================


class NeuronQwen35ForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronQwen35Model

    def get_model_wrapper_cls(self):
        """Return custom ModelWrapper with DeltaNet state aliasing."""
        return Qwen35ModelWrapper

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HF model weights.

        The model is a VL model (Qwen3_5ForConditionalGeneration) but we
        only need the text backbone.
        """
        from transformers import AutoModelForCausalLM

        kwargs.setdefault("trust_remote_code", True)
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Qwen35InferenceConfig

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Copy embed_tokens weight to lm_head for tied embeddings."""
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Strip VL wrapper prefix and convert to NxDI format."""
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("language_model."):
                new_k = k.replace("language_model.", "", 1)
                new_sd[new_k] = v
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

        if (
            getattr(config, "tie_word_embeddings", False)
            and "lm_head.weight" not in new_sd
            and "embed_tokens.weight" in new_sd
        ):
            new_sd["lm_head.weight"] = new_sd["embed_tokens.weight"]

        return convert_qwen35_hf_to_neuron_state_dict(new_sd, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def _copy_past_key_values(self, outputs):
        """Override to also copy DeltaNet state buffers on CPU."""
        super()._copy_past_key_values(outputs)

        num_output_from_trace = 1
        if (
            self.neuron_config.output_logits
            and self.neuron_config.on_device_sampling_config
        ):
            num_output_from_trace = 2

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

        state_start = num_output_from_trace + num_kv

        tkg_params = getattr(tkg_model, "_deltanet_state_params", [])
        cte_params = getattr(cte_model, "_deltanet_state_params", [])

        if len(tkg_params) > 0 and state_start + len(tkg_params) <= len(outputs):
            for i, (tkg_param, cte_param) in enumerate(zip(tkg_params, cte_params)):
                new_state = outputs[state_start + i]
                tkg_param.data = new_state
                cte_param.data = new_state

    def get_required_kwargs(self):
        """Return extra kwargs for HF generation loop."""
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
        """Override to pass all 24 positional args explicitly."""
        is_prefill = self._is_prefill(position_ids)

        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        if llava_args and len(llava_args) >= 2:
            vision_embeddings = llava_args[0]
            vision_mask = llava_args[1]
            if len(llava_args) >= 3:
                mrope_position_ids = llava_args[2]
            else:
                mrope_position_ids = None
        elif is_prefill:
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
            vision_embeddings = torch.zeros((0,), dtype=torch.float32)
            vision_mask = torch.zeros((0,), dtype=torch.int32)
            mrope_position_ids = None

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

        empties = [torch.empty(0) for _ in range(14)]

        if self._is_prefill(position_ids):
            ctx_bs = self.context_encoding_model.neuron_config.batch_size
            output_logits = []

            for cb in range(0, batch_size, ctx_bs):
                cb_end = min(cb + ctx_bs, batch_size)
                actual_chunk = cb_end - cb

                chunk_input_ids = input_ids[cb:cb_end]
                chunk_attn_mask = attention_mask[cb:cb_end]
                chunk_pos_ids = position_ids[cb:cb_end]
                chunk_seq_ids = seq_ids[cb:cb_end]
                chunk_sampling = sampling_params[cb:cb_end]
                chunk_prev_hidden = (
                    prev_hidden[cb:cb_end]
                    if prev_hidden is not None
                    and hasattr(prev_hidden, "ndim")
                    and prev_hidden.ndim > 0
                    and prev_hidden.shape[0] > 0
                    else prev_hidden
                )
                chunk_adapter_ids = (
                    adapter_ids[cb:cb_end]
                    if adapter_ids is not None
                    and hasattr(adapter_ids, "ndim")
                    and adapter_ids.ndim > 0
                    and adapter_ids.shape[0] > 0
                    else adapter_ids
                )

                if mrope_position_ids.ndim == 3:
                    chunk_mrope = mrope_position_ids[:, cb:cb_end, :]
                else:
                    chunk_mrope = mrope_position_ids

                if vision_embeddings.ndim == 3:
                    chunk_vis_emb = vision_embeddings[cb:cb_end]
                    chunk_vis_mask = vision_mask[cb:cb_end]
                else:
                    chunk_vis_emb = vision_embeddings
                    chunk_vis_mask = vision_mask

                if actual_chunk < ctx_bs:
                    pad_n = ctx_bs - actual_chunk
                    chunk_input_ids = torch.cat(
                        [chunk_input_ids, chunk_input_ids[:1].expand(pad_n, -1)], dim=0
                    )
                    chunk_attn_mask = torch.cat(
                        [chunk_attn_mask, chunk_attn_mask[:1].expand(pad_n, -1)], dim=0
                    )
                    chunk_pos_ids = torch.cat(
                        [chunk_pos_ids, chunk_pos_ids[:1].expand(pad_n, -1)], dim=0
                    )
                    pad_seq = torch.arange(
                        batch_size, batch_size + pad_n, dtype=chunk_seq_ids.dtype
                    )
                    chunk_seq_ids = torch.cat([chunk_seq_ids, pad_seq], dim=0)
                    chunk_sampling = torch.cat(
                        [chunk_sampling, chunk_sampling[:1].expand(pad_n, -1)], dim=0
                    )
                    if (
                        chunk_prev_hidden is not None
                        and hasattr(chunk_prev_hidden, "ndim")
                        and chunk_prev_hidden.ndim > 0
                        and chunk_prev_hidden.shape[0] > 0
                    ):
                        chunk_prev_hidden = torch.cat(
                            [
                                chunk_prev_hidden,
                                chunk_prev_hidden[:1].expand(pad_n, -1),
                            ],
                            dim=0,
                        )
                    if (
                        chunk_adapter_ids is not None
                        and hasattr(chunk_adapter_ids, "ndim")
                        and chunk_adapter_ids.ndim > 0
                        and chunk_adapter_ids.shape[0] > 0
                    ):
                        chunk_adapter_ids = torch.cat(
                            [
                                chunk_adapter_ids,
                                chunk_adapter_ids[:1].expand(pad_n, -1),
                            ],
                            dim=0,
                        )
                    if chunk_mrope.ndim == 3:
                        chunk_mrope = torch.cat(
                            [chunk_mrope, chunk_mrope[:, :1, :].expand(-1, pad_n, -1)],
                            dim=1,
                        )
                    if chunk_vis_emb.ndim == 3:
                        chunk_vis_emb = torch.cat(
                            [
                                chunk_vis_emb,
                                torch.zeros(
                                    (pad_n,) + chunk_vis_emb.shape[1:],
                                    dtype=chunk_vis_emb.dtype,
                                ),
                            ],
                            dim=0,
                        )
                        chunk_vis_mask = torch.cat(
                            [
                                chunk_vis_mask,
                                torch.full(
                                    (pad_n,) + chunk_vis_mask.shape[1:],
                                    fill_value=seq_len - 1,
                                    dtype=chunk_vis_mask.dtype,
                                ),
                            ],
                            dim=0,
                        )

                chunk_out = self.context_encoding_model(
                    chunk_input_ids,
                    chunk_attn_mask,
                    chunk_pos_ids,
                    chunk_seq_ids,
                    chunk_sampling,
                    chunk_prev_hidden,
                    chunk_adapter_ids,
                    *empties,
                    chunk_mrope,
                    chunk_vis_emb,
                    chunk_vis_mask,
                )
                if actual_chunk < ctx_bs:
                    chunk_out = chunk_out[:actual_chunk]
                output_logits.append(chunk_out)

            outputs = (
                torch.cat(output_logits, dim=0)
                if len(output_logits) > 1
                else output_logits[0]
            )
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                *empties,
                mrope_position_ids,
                vision_embeddings,
                vision_mask,
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
        )
        return compiler_args
