# coding=utf-8
# Copyright 2025 Zyphra and contributors. All rights reserved.
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
PyTorch ZAYA1-base model for NxD Inference.

NxDI contrib model for ZAYA1-base (Zyphra/ZAYA1-base).
Novel MoE architecture: 800M active / 8.84B total params, 80 layers,
CCA attention (even layers), non-linear MLP router with MoD (odd layers).

Architecture:
  - 80 layers alternating: attention ('a') / MoE (16 experts)
  - CCA (Causal Cross-Attention) with L2-normalized QK, Conv1d, temp scaling
  - Non-linear MLP router with EDA (exponential depth averaging)
  - MoD (Mixture of Depths) skip expert
  - Per-layer residual scaling (learnable scale + bias)
  - Partial RoPE (partial_rotary_factor=0.5)
  - Tied word embeddings (lm_head = embed_tokens)
"""

import gc
import math
import os
from typing import List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# NeuronX distributed imports
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.utils import cpu_mode

# NeuronX distributed inference imports
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
)
from neuronx_distributed_inference.models.model_wrapper import (
    DecoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import (
    initialize_moe_module as initialize_moe_module_v2,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# NKI flash attention — lazy import for CPU compatibility.
# nkilib is only available on Neuron instances with neuronx-cc installed.
#
# We use the NxDI internal NKI kernel (_flash_fwd_call_nki from
# attention_base.py) instead of nkilib.core.attention.attention_cte directly.
# The nkilib kernel's @nki.jit decorator has a platform detection bug during
# SPMD tracing (nl.num_programs returns None). NxDI's adapter uses
# peel_decorations + re-decoration with mode='torchxla' which fixes this.
_nki_flash_fwd = None
_nki_has_new_kernel = False

# MLP ISA kernel — lazy import for CPU compatibility.
# These are private Neuron compiler kernels that fuse gate+up+silu+down into
# a single ISA-level kernel, avoiding intermediate activation HBM round-trips.
_mlp_isa_kernel = None
_mlp_isa_available = None  # None = not yet checked, True/False = cached result


def _get_nki_flash_fwd():
    """Lazy import of NxDI's internal NKI CTE attention kernel.

    Returns the kernel function, or None if not available (CPU mode).
    Uses the same kernel NxDI's attention_base.py uses for flash attention.
    This is already proven to work with NxD SPMD tracing.
    """
    global _nki_flash_fwd, _nki_has_new_kernel
    if _nki_flash_fwd is not None:
        return _nki_flash_fwd
    try:
        from neuronx_distributed_inference.modules.attention.attention_base import (
            _flash_fwd_call_nki,
            _has_new_kernel,
        )

        if _flash_fwd_call_nki is not None and _has_new_kernel:
            _nki_flash_fwd = _flash_fwd_call_nki
            _nki_has_new_kernel = _has_new_kernel
            return _nki_flash_fwd
        return None
    except (ImportError, Exception) as e:
        import warnings

        warnings.warn(f"Failed to import NxDI NKI flash attention: {e}")
        return None


def _get_mlp_isa_kernel():
    """Lazy import of the Neuron compiler's MLP ISA kernel.

    Returns (mlp_isa_kernel_fn, NormType, nki_jit, nc, reduce_fn) tuple,
    or None if not available (CPU mode).
    """
    global _mlp_isa_kernel, _mlp_isa_available
    if _mlp_isa_available is True:
        return _mlp_isa_kernel
    if _mlp_isa_available is False:
        return None
    try:
        from neuronxcc.nki._private_kernels.mlp import mlp_isa_kernel
        from neuronxcc.nki._pre_prod_kernels import NormType
        from neuronxcc.nki.language import nc
        from torch_neuronx.xla_impl.ops import nki_jit
        from neuronx_distributed.parallel_layers.mappings import (
            reduce_from_tensor_model_parallel_region,
        )

        _mlp_isa_kernel = (
            mlp_isa_kernel,
            NormType,
            nki_jit,
            nc,
            reduce_from_tensor_model_parallel_region,
        )
        _mlp_isa_available = True
        return _mlp_isa_kernel
    except (ImportError, Exception) as e:
        import warnings

        warnings.warn(f"MLP ISA kernel not available: {e}")
        _mlp_isa_available = False
        return None


# Transformers imports — requires Zyphra's custom fork:
#   pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya"
# ZayaForCausalLM is imported lazily in load_hf_model() to avoid jit_fuser
# issues when the HF model class isn't needed (e.g., during model construction).


def _get_hf_rmsnorm_cls():
    """Lazily import HFZayaRMSNorm to avoid triggering @jit_fuser at module load."""

    # The HF modeling_zaya.py has @jit_fuser decorators that fail with torch.jit.script
    # in certain environments. Use a simple CPU-compatible RMSNorm instead.
    class SimpleRMSNorm(nn.Module):
        """Standard RMSNorm for CPU mode (avoids HF jit_fuser import issues)."""

        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return self.weight * hidden_states.to(input_dtype)

    return SimpleRMSNorm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_rmsnorm_cls():
    """Return the appropriate RMSNorm implementation."""
    if cpu_mode():
        return _get_hf_rmsnorm_cls()
    return CustomRMSNorm


def swiglu(y):
    """SwiGLU activation: split in half, silu(first) * second."""
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ZayaNeuronConfig(MoENeuronConfig):
    """NeuronConfig subclass for ZAYA1-base.

    Disables fused NKI kernels that are incompatible with CCA attention
    and the non-linear MLP router. MLP ISA kernel is available but disabled
    by default (benchmarked at -4.2% vs native at batch=1 due to weight-read
    dominated workload at this model size).
    """

    def __init__(self, **kwargs):
        # Extract mlp_kernel_enabled before super().__init__ consumes kwargs
        mlp_kernel = kwargs.pop("mlp_kernel_enabled", False)
        super().__init__(**kwargs)
        # CCA attention is NOT compatible with block TKG/CTE mega-kernels
        self.attn_block_tkg_nki_kernel_enabled = False
        self.block_cte_nki_kernel_enabled = False
        # Non-linear MLP router is NOT compatible with fused MoE NKI kernel
        self.moe_fused_nki_kernel_enabled = False
        # MLP ISA kernel for expert MLPs (fuses gate+silu+up+down)
        self.mlp_kernel_enabled = mlp_kernel


class ZayaInferenceConfig(InferenceConfig):
    """Inference config for ZAYA1-base.

    Reads all ZAYA-specific fields from the HuggingFace config and exposes
    them for the NxDI model. Per-layer config lists (cca_num_q_heads,
    ffn_hidden_size_list, etc.) are preserved as-is.
    """

    def add_derived_config(self):
        """Set derived attributes after config loading."""
        # TP-aware: num_cores_per_group must match tp_degree for proper
        # GQA head distribution across NeuronCores.
        self.num_cores_per_group = getattr(self, "neuron_config", None)
        if self.num_cores_per_group is not None:
            self.num_cores_per_group = self.neuron_config.tp_degree
        else:
            self.num_cores_per_group = 1

        # head_dim is always hidden_size // num_attention_heads = 2048 // 16 = 128
        if not hasattr(self, "head_dim") or self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Required by NeuronBaseForCausalLM._setup_func_config
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False

        # --- ZAYA-specific per-layer lists ---
        # These come from ZayaConfig and must be preserved for per-layer model construction.
        # zaya_layers: list of 80 items, 'a' (attention) or int (MoE expert count)
        if not hasattr(self, "zaya_layers"):
            self.zaya_layers = ["a", 16] * 40  # default: alternating att/moe

        if not hasattr(self, "cca_num_q_heads"):
            self.cca_num_q_heads = [8, 0] * 40

        if not hasattr(self, "num_query_groups_list"):
            self.num_query_groups_list = [2, 0] * 40

        if not hasattr(self, "ffn_hidden_size_list"):
            self.ffn_hidden_size_list = [0, 4096] * 40

        if not hasattr(self, "zaya_mlp_expansion"):
            self.zaya_mlp_expansion = [0, 256] * 40

        # MoE config fields expected by the model
        if not hasattr(self, "moe_router_topk"):
            self.moe_router_topk = 1

        if not hasattr(self, "zaya_use_mod"):
            self.zaya_use_mod = True

        if not hasattr(self, "zaya_use_eda"):
            self.zaya_use_eda = True

        if not hasattr(self, "zaya_high_prec"):
            self.zaya_high_prec = True

        if not hasattr(self, "partial_rotary_factor"):
            self.partial_rotary_factor = 0.5

        if not hasattr(self, "scale_residual_merge"):
            self.scale_residual_merge = True

        if not hasattr(self, "norm_epsilon"):
            self.norm_epsilon = getattr(self, "rms_norm_eps", 1e-5)

        # Map rms_norm_eps -> norm_epsilon for compatibility
        if not hasattr(self, "rms_norm_eps"):
            self.rms_norm_eps = self.norm_epsilon

        # Expert config
        if not hasattr(self, "num_local_experts"):
            self.num_local_experts = 16

        if not hasattr(self, "num_experts_per_tok"):
            self.num_experts_per_tok = 1

        # Activation / bias
        if not hasattr(self, "add_bias_linear"):
            self.add_bias_linear = False

        if not hasattr(self, "gated_linear_unit"):
            self.gated_linear_unit = True

        if not hasattr(self, "attention_bias"):
            self.attention_bias = False

        if not hasattr(self, "cca"):
            self.cca = True

        if not hasattr(self, "tie_word_embeddings"):
            self.tie_word_embeddings = True

        # The "logical" layer count for ZAYA is len(zaya_layers) = 80,
        # NOT num_hidden_layers (which HF config sets to 120).
        # NxDI uses num_hidden_layers for the layer loop in NeuronBaseModel.
        # Override to 80 so the model builds the right number of decoder layers.
        self.num_hidden_layers = len(self.zaya_layers)

        # MoE neuron config: set GLU params and router config
        if hasattr(self, "neuron_config"):
            self.neuron_config.glu_mlp = True
            self.neuron_config.glu_type = "swiglu"
            # Router uses bf16 softmax (not float32) to match HF ZAYA behavior.
            # The custom ZayaRouter handles its own routing, but MoE V2 init
            # reads this config for the placeholder RouterTopK.
            self.neuron_config.normalize_top_k_affinities = False
            # EP=1: all 16 experts replicated on every rank, expert intermediate
            # dims sharded across TP ranks. This is required because ZAYA has
            # 17 logical experts (16 + skip) which is not evenly divisible by TP=2.
            self.neuron_config.moe_tp_degree = 1
            self.neuron_config.moe_ep_degree = 1

        # NxDI MoE V2 requires config.intermediate_size for expert MLPs.
        # ZAYA uses per-layer ffn_hidden_size_list; MoE V2 reads config.intermediate_size.
        # Set it to the MoE layer's ffn_hidden_size (4096 for ZAYA1-base).
        # Note: this is the gate+up fused size, so intermediate_size = ffn_hidden_size // 2
        # since glu_mlp=True makes ExpertMLPsV2 expect the un-fused intermediate size.
        if not hasattr(self, "intermediate_size"):
            # ffn_hidden_size_list[1] is the first MoE layer's FFN size = 4096
            # For SwiGLU, NxDI expects intermediate_size = ffn_hidden_size / 2
            # But ZAYA's fc1 weight is already [hidden, ffn_hidden_size] = [2048, 4096]
            # and the output is split in half for SwiGLU.
            # ExpertMLPsV2 with glu_mlp=True creates gate_up_proj: [hidden, 2*intermediate]
            # So we need intermediate_size = ffn_hidden_size / 2 = 2048
            self.intermediate_size = self.ffn_hidden_size_list[1] // 2

        # Required by MoE V2: n_shared_experts
        if not hasattr(self, "n_shared_experts"):
            self.n_shared_experts = 0

        # Required by MoE V2: hidden_act
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"  # SwiGLU uses silu as the gate activation

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "zaya_layers",
            "cca_num_q_heads",
            "num_query_groups_list",
            "ffn_hidden_size_list",
            "zaya_mlp_expansion",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[ZayaNeuronConfig]:
        return ZayaNeuronConfig


# ---------------------------------------------------------------------------
# Residual Scaling
# ---------------------------------------------------------------------------


class ResidualScaling(nn.Module):
    """Per-layer learnable residual scaling: scale * (x + bias).

    Layer 0 has no residual_scale/residual_bias (only hidden_states_*).
    All other layers have both.
    """

    def __init__(self, hidden_size, not_first_layer=True):
        super().__init__()
        self.not_first_layer = not_first_layer
        self.hidden_states_scale = nn.Parameter(torch.ones(hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(hidden_size))

        if self.not_first_layer:
            self.residual_scale = nn.Parameter(torch.ones(hidden_size))
            self.residual_bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, residual, hidden_states):
        hidden_states = (
            hidden_states + self.hidden_states_bias.expand(1, 1, -1)
        ) * self.hidden_states_scale.expand(1, 1, -1)
        if self.not_first_layer:
            residual = (
                residual + self.residual_bias.expand(1, 1, -1)
            ) * self.residual_scale.expand(1, 1, -1)
        return residual, hidden_states


# ---------------------------------------------------------------------------
# Manual Conv1d — avoids NKI kernel inliner (NCC_ITEN404 workaround)
# ---------------------------------------------------------------------------


class ManualConv1d(nn.Module):
    """Drop-in replacement for nn.Conv1d that avoids NKI kernel insertion.

    The Neuron compiler's NKI Conv1d kernel inliner (InlineNativeKernels)
    crashes with NCC_ITEN404 when the HLO graph has certain all-gather +
    Conv1d patterns. This module implements the same computation using basic
    tensor ops (element-wise multiply, matmul) that don't trigger NKI
    kernel specialization.

    Supports:
      - depthwise conv (groups == in_channels): element-wise multiply + sum
      - grouped conv (1 < groups < in_channels): block-diagonal matmul per
        time position
      - standard conv (groups == 1): matmul per time position

    Stores weight and bias with the same parameter names as nn.Conv1d so
    state_dict is fully compatible (no key remapping needed).

    Only supports stride=1, padding=0, dilation=1 (which is all ZAYA uses).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert in_channels == out_channels, (
            "ManualConv1d only supports in==out channels"
        )
        assert kernel_size in (2, 3), (
            f"ManualConv1d only supports kernel_size 2 or 3, got {kernel_size}"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.channels_per_group = in_channels // groups

        # Same parameter layout as nn.Conv1d:
        # weight: [out_channels, in_channels/groups, kernel_size]
        # bias: [out_channels]
        self.weight = nn.Parameter(
            torch.empty(out_channels, self.channels_per_group, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize with same defaults as nn.Conv1d (Kaiming uniform)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Args:
            x: [B, C, L] input tensor

        Returns:
            [B, C, L - kernel_size + 1] output tensor
        """
        if self.groups == self.in_channels:
            # Depthwise conv: each channel independently.
            # weight shape: [C, 1, K] -> squeeze to [C, K]
            w = self.weight.squeeze(1)  # [C, K]
            if self.kernel_size == 2:
                # out[:, c, t] = w[c, 0] * x[:, c, t] + w[c, 1] * x[:, c, t+1]
                out = w[None, :, 0:1] * x[:, :, :-1] + w[None, :, 1:2] * x[:, :, 1:]
            else:  # kernel_size == 3
                out = (
                    w[None, :, 0:1] * x[:, :, :-2]
                    + w[None, :, 1:2] * x[:, :, 1:-1]
                    + w[None, :, 2:3] * x[:, :, 2:]
                )
        else:
            # Grouped conv: process all time positions via batched matmul.
            # weight: [C_out, C_in/G, K] for grouped conv.
            # For each time position t, we gather K adjacent slices and do a
            # block-diagonal matmul across groups.
            B, C, L = x.shape
            L_out = L - self.kernel_size + 1
            G = self.groups
            cpg = self.channels_per_group

            # Build weight matrix for one time step: [G, cpg, cpg * K]
            # Reshape weight [C_out, cpg, K] -> [G, cpg, cpg, K] -> [G, cpg, cpg*K]
            # Actually weight is [C_out, cpg, K] = [G*cpg, cpg, K]
            w = self.weight.view(G, cpg, cpg, self.kernel_size)  # [G, cpg, cpg, K]
            # We need to contract over (cpg, K) dimensions.
            # Reshape to [G, cpg, cpg * K] for matmul with stacked input.
            w_flat = w.reshape(G, cpg, cpg * self.kernel_size)  # [G, cpg, cpg*K]

            # Stack input slices: for each time step t, gather x[:, :, t:t+K]
            # x: [B, C, L] -> [B, G, cpg, L]
            x_grouped = x.view(B, G, cpg, L)

            # Gather all kernel-width windows: [B, G, cpg, L_out, K]
            slices = []
            for ki in range(self.kernel_size):
                slices.append(x_grouped[:, :, :, ki : ki + L_out])
            x_windows = torch.stack(slices, dim=-1)  # [B, G, cpg, L_out, K]

            # Reshape for matmul: [B, G, L_out, cpg*K]
            x_windows = x_windows.permute(0, 1, 3, 2, 4).reshape(
                B, G, L_out, cpg * self.kernel_size
            )

            # Batched matmul: [B, G, L_out, cpg*K] @ [G, cpg*K, cpg]^T -> [B, G, L_out, cpg]
            # w_flat is [G, cpg, cpg*K], we need [G, cpg*K, cpg] for the right side
            # Actually: out = x_windows @ w_flat^T  (w_flat is [G, cpg, cpg*K])
            # [B, G, L_out, cpg*K] @ [1, G, cpg*K, cpg] -> [B, G, L_out, cpg]
            out = torch.matmul(
                x_windows, w_flat.transpose(-1, -2).unsqueeze(0)
            )  # [B, G, L_out, cpg]

            # Reshape back: [B, G, cpg, L_out] -> [B, C, L_out]
            out = out.permute(0, 1, 3, 2).reshape(B, C, L_out)

        if self.bias is not None:
            out = out + self.bias[None, :, None]

        return out


# ---------------------------------------------------------------------------
# CCA (Causal Cross-Attention) Module
# ---------------------------------------------------------------------------


class CCA(nn.Module):
    """CCA attention mechanism — full implementation from HF source.

    This is NOT using NeuronAttentionBase because CCA has a fundamentally
    different QKV computation (Conv1d, L2-norm, temperature) that doesn't
    map to the standard QKV projection + RoPE pipeline.

    For token generation (seq_len=1), the Conv1d state from prior positions
    is not available (it would require stateful caching across XLA-traced
    calls, which is complex). Instead, during TKG we use only the mean-based
    Q/K (which already captures the token's projection information) and
    zero out the conv contribution. This is a known simplification that
    Task 008 (CCA NKI optimization) will improve by implementing proper
    conv state caching through the input_output_aliases mechanism.

    For the value (v2) computation, we similarly use zero for the time-shifted
    hidden state during TKG. The v1 (non-shifted) part is computed correctly.
    """

    def __init__(
        self,
        config,
        cca_num_kv_heads: int = 2,
        cca_num_q_heads: int = 8,
        cca_num_heads: int = 16,
        hidden_size: Optional[int] = None,
        cca_time0: int = 2,
        cca_time1: int = 2,
        layer_number: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

        self.hidden_size = int(hidden_size or config.hidden_size)
        self.cca_time0 = cca_time0
        self.cca_time1 = cca_time1
        self.padding0 = cca_time0 - 1
        self.padding1 = cca_time1 - 1
        self.total_padding = self.padding0 + self.padding1

        # Global (full) head counts
        self.num_kv_heads_global = int(cca_num_kv_heads)
        self.num_q_heads_global = int(cca_num_q_heads)
        self.num_heads = int(cca_num_heads)

        # TP-aware per-rank head counts
        tp_degree = (
            getattr(config.neuron_config, "tp_degree", 1)
            if hasattr(config, "neuron_config")
            else 1
        )
        self.tp_degree = tp_degree
        self.num_kv_heads = self.num_kv_heads_global // tp_degree
        self.num_q_heads = self.num_q_heads_global // tp_degree

        # Geometry (per-rank)
        self.head_dim = self.hidden_size // self.num_heads
        self.latent_k_dim = self.num_kv_heads * self.head_dim  # per-rank
        self.latent_q_dim = self.num_q_heads * self.head_dim  # per-rank
        self.sqrt_head_dim = float(np.sqrt(self.head_dim))
        self.gqa_groups = self.num_q_heads // self.num_kv_heads

        # Global dims
        self.latent_q_dim_global = self.num_q_heads_global * self.head_dim
        self.latent_k_dim_global = self.num_kv_heads_global * self.head_dim
        latent_q_dim_global = self.latent_q_dim_global
        latent_k_dim_global = self.latent_k_dim_global

        # Projections — ColumnParallelLinear with gather_output=True.
        # The all-gather after matmul produces GLOBAL Q/K so the Conv1d
        # can run at global dimensions without scatter/gather hacks.
        # After conv, per-rank channels are extracted via index_select.
        self.linear_q = ColumnParallelLinear(
            self.hidden_size,
            latent_q_dim_global,
            bias=False,
            gather_output=True,
            dtype=getattr(config.neuron_config, "torch_dtype", torch.bfloat16),
        )
        self.linear_k = ColumnParallelLinear(
            self.hidden_size,
            latent_k_dim_global,
            bias=False,
            gather_output=True,
            dtype=getattr(config.neuron_config, "torch_dtype", torch.bfloat16),
        )
        # Value projections: plain nn.Linear (NOT ColumnParallelLinear).
        # val_proj1 produces all of V head 0 (from current hidden states),
        # val_proj2 produces all of V head 1 (from time-shifted hidden states).
        # Each has output dim = latent_k_dim_global // 2 = one full KV head.
        # With TP, we can't split within a head, so we replicate both projections
        # on all ranks and slice per-rank KV heads in forward().
        self.val_proj1 = nn.Linear(
            self.hidden_size,
            latent_k_dim_global // 2,
            bias=False,
        )
        self.val_proj2 = nn.Linear(
            self.hidden_size,
            latent_k_dim_global // 2,
            bias=False,
        )

        # Depthwise + grouped conv along sequence.
        # IMPORTANT: Conv1d is constructed at GLOBAL dimensions (not per-rank)
        # because NxD's weight loader doesn't auto-shard plain Conv1d modules.
        # The per-rank slicing happens in forward() via narrow().
        in_out_ch_global = (
            self.num_q_heads_global * self.head_dim
            + self.num_kv_heads_global * self.head_dim
        )
        self.in_out_ch_global = in_out_ch_global
        in_out_ch = self.latent_k_dim + self.latent_q_dim  # per-rank
        self.in_out_ch = in_out_ch
        num_groups_global_conv0 = (
            in_out_ch_global  # depthwise: each channel independent
        )
        num_groups_global_conv1 = (
            self.num_kv_heads_global + self.num_q_heads_global
        )  # grouped
        # ManualConv1d replaces nn.Conv1d to avoid NKI kernel inliner
        # (NCC_ITEN404 compiler bug). Same weight/bias parameter names for
        # state_dict compatibility.
        self.conv_qk = nn.Sequential(
            ManualConv1d(
                in_channels=in_out_ch_global,
                out_channels=in_out_ch_global,
                kernel_size=self.cca_time0,
                groups=num_groups_global_conv0,
            ),
            ManualConv1d(
                in_channels=in_out_ch_global,
                out_channels=in_out_ch_global,
                kernel_size=self.cca_time1,
                groups=num_groups_global_conv1,
            ),
        )

        # Per-KV-head temperature — stored at GLOBAL dimensions for weight loading.
        # Per-rank slicing happens in forward().
        self.temp = nn.Parameter(torch.zeros(self.num_kv_heads_global))

        # SPMD-compatible rank identification.
        # parallel_state.get_tensor_model_parallel_rank() returns a CONSTANT during
        # SPMD tracing (always rank 0), which gets baked into ALL ranks' compiled NEFFs.
        # SPMDRank uses a weight-sharding trick: the rank is a model parameter that gets
        # different values per-rank through checkpoint sharding at load time.
        if self.tp_degree > 1:
            self.rank_util = SPMDRank(world_size=self.tp_degree)

    def _get_rank(self):
        """Get current TP rank as a tensor (SPMD-compatible).

        Uses SPMDRank which stores rank as a model parameter — each rank loads
        a different value via checkpoint sharding. This is the ONLY correct way
        to get per-rank behavior in SPMD-traced models on Neuron.

        DO NOT use parallel_state.get_tensor_model_parallel_rank() — it returns
        a constant (0) during tracing that gets baked into all ranks' NEFFs.
        """
        if self.tp_degree <= 1:
            return 0
        return self.rank_util.get_rank()  # returns torch.Tensor([rank_id])

    def _extract_per_rank_qk(self, qk_global):
        """Extract per-rank Q and K channels from global QK tensor after conv.

        Uses the proven NxDI split_along_dim pattern: compute per-rank indices
        via tensor arithmetic with SPMDRank, then torch.index_select.

        Args:
            qk_global: [S, B, in_out_ch_global] — global Q/K after conv
                       Layout: [Q_all(latent_q_global), K_all(latent_k_global)]

        Returns:
            q_per_rank: [S, B, latent_q_dim] — per-rank Q channels
            k_per_rank: [S, B, latent_k_dim] — per-rank K channels
        """
        if self.tp_degree <= 1:
            q = qk_global[..., : self.latent_q_dim_global]
            k = qk_global[..., self.latent_q_dim_global :]
            return q, k

        rank = self._get_rank()  # tensor([rank_id])

        # Q: extract per-rank chunk from global Q (first latent_q_dim_global channels)
        q_offset = (rank * self.latent_q_dim).to(torch.long)
        q_indices = (
            torch.arange(self.latent_q_dim, device=qk_global.device, dtype=torch.long)
            + q_offset
        )
        q_per_rank = torch.index_select(qk_global, -1, q_indices)

        # K: extract per-rank chunk from global K (last latent_k_dim_global channels)
        k_offset = (self.latent_q_dim_global + rank * self.latent_k_dim).to(torch.long)
        k_indices = (
            torch.arange(self.latent_k_dim, device=qk_global.device, dtype=torch.long)
            + k_offset
        )
        k_per_rank = torch.index_select(qk_global, -1, k_indices)

        return q_per_rank, k_per_rank

    def _extract_per_rank_heads(self, tensor, num_heads_per_rank, dim=2):
        """Extract per-rank heads from a global-sized head tensor.

        Args:
            tensor: [..., num_heads_global, ...] at dimension `dim`
            num_heads_per_rank: number of heads this rank owns
            dim: the head dimension

        Returns:
            per-rank slice: [..., num_heads_per_rank, ...]
        """
        if self.tp_degree <= 1:
            return tensor

        rank = self._get_rank()
        offset = (rank * num_heads_per_rank).to(torch.long)
        indices = (
            torch.arange(num_heads_per_rank, device=tensor.device, dtype=torch.long)
            + offset
        )
        return torch.index_select(tensor, dim, indices)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        cca_mask=None,
        is_for_context_encoding: bool = True,
        conv_state: Optional[torch.Tensor] = None,
        prev_hs_cache: Optional[torch.Tensor] = None,
    ):
        """
        CCA forward pass.

        For context encoding (prefill): full conv over sequence, then save
        the last conv_kernel_size positions of qk_packed as conv_state, and
        the last token's hidden state as prev_hs for future TKG steps.

        For token generation (seq_len=1): use conv_state from previous step
        as left context for Conv1d, and prev_hs_cache as time-shifted hidden
        state for V2 computation.

        TP strategy: Q/K projections use gather_output=True, so conv operates
        at GLOBAL dimensions. After conv, per-rank Q/K are extracted via
        index_select (proven NxDI pattern). No scatter/gather needed.

        Args:
            hidden_states: [B, S, H] (HF layout)
            past_key_values: not used directly (KV cache managed externally)
            cca_mask: optional mask for padding in prefill
            is_for_context_encoding: True for prefill, False for token gen
            conv_state: [B, in_out_ch_global, 2] conv state from prior step (for TKG)
            prev_hs_cache: [B, hidden_size] prior hidden state (for TKG V2)

        Returns:
            query: [B, S, num_q_heads_per_rank*head_dim]
            key:   [B, S, num_kv_heads_per_rank*head_dim]
            value: [B, S, num_kv_heads_per_rank*head_dim]
            updated_conv_state: [B, in_out_ch_global, 2] updated conv state
            updated_prev_hs: [B, hidden_size] updated prev_hs
        """
        batch_size, seq_length, _ = hidden_states.shape

        if cca_mask is not None and seq_length > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * cca_mask[:, :, None]).to(dtype)

        # Switch to [S, B, H]
        hs = hidden_states.transpose(0, 1).contiguous()

        # Time-shifted stream for v2
        if not is_for_context_encoding and prev_hs_cache is not None:
            # TKG: use cached prev_hs from prior step
            hs_d = prev_hs_cache.unsqueeze(0)  # [1, B, H]
        else:
            # Prefill: standard shift-by-1 with zero padding
            hs_d = F.pad(hs[:-1], pad=(0, 0, 0, 0, 1, 0))

        # Updated prev_hs: save the last token's hidden state for TKG V2.
        # NOTE: During prefill with NxDI padding, hs[-1] may be a zero-padded
        # position. This means the first TKG step's V2 computation uses zeros
        # instead of the last real token's hs. This self-corrects after 1 step.
        # A proper fix would select the last real token using cca_mask, but
        # that adds complexity. The impact is minimal (1 token of slightly
        # degraded V2 quality).
        updated_prev_hs = hs[-1, :, :]  # [B, H]
        # NOTE: Do NOT add + prev_hs_cache * 0 here. Adding param * 0 inside
        # the layer computation corrupts CTE output when combined with
        # input_output_aliases. The force-read is handled in forward() instead.

        # Q/K projections — with gather_output=True, output is GLOBAL dims:
        # q: [S, B, latent_q_dim_global], k: [S, B, latent_k_dim_global]
        q = self.linear_q(hs)
        k = self.linear_k(hs)
        qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, in_out_ch_global]

        # Pre-mean tensors — computed at GLOBAL dims, then sliced per-rank
        query_pre = qk_packed0[..., : self.latent_q_dim_global].view(
            *qk_packed0.shape[:2], self.num_q_heads_global, self.head_dim
        )
        key_pre = qk_packed0[..., self.latent_q_dim_global :].view(
            *qk_packed0.shape[:2], self.num_kv_heads_global, self.head_dim
        )
        key_pre = (
            key_pre.unsqueeze(-2)
            .repeat(1, 1, 1, self.gqa_groups, 1)
            .view(*qk_packed0.shape[:2], self.num_q_heads_global, self.head_dim)
        )

        qk_mean_q_global = (query_pre + key_pre) / 2
        qk_mean_k_global = qk_mean_q_global.view(
            *qk_mean_q_global.shape[:2], self.num_kv_heads_global, self.gqa_groups, -1
        ).mean(dim=-2)

        # Per-rank mean slices (using proven index_select pattern)
        qk_mean_q = self._extract_per_rank_heads(
            qk_mean_q_global, self.num_q_heads, dim=2
        )
        qk_mean_k = self._extract_per_rank_heads(
            qk_mean_k_global, self.num_kv_heads, dim=2
        )

        if not is_for_context_encoding and conv_state is not None:
            # TKG: use cached conv_state as left context
            # qk_packed0: [1, B, E_global] -> [B, E_global, 1]
            qk_current = qk_packed0.permute(1, 2, 0)  # [B, E_global, 1]
            # conv_state: [B, E_global, 2] from prior step (global dims)
            # Concatenate: [B, E_global, 2] + [B, E_global, 1] = [B, E_global, 3]
            qk_packed_cat = torch.cat([conv_state, qk_current], dim=-1)
            qk_packed3_global = self.conv_qk(qk_packed_cat).permute(
                2, 0, 1
            )  # [1, B, E_global]

            # Update conv state: take last 2 positions from cat input
            updated_conv_state = qk_packed_cat[
                :, :, 1:
            ].contiguous()  # [B, E_global, 2]
        else:
            # Prefill: standard causal-padded conv
            qk_packed1 = qk_packed0.permute(1, 2, 0)  # [B, E_global, S]
            qk_packed2 = F.pad(qk_packed1, (self.total_padding, 0))
            qk_packed3_global = self.conv_qk(qk_packed2).permute(
                2, 0, 1
            )  # [S, B, E_global]

            # Save conv state: last conv_kernel_size positions of qk_packed (global).
            if seq_length >= self.cca_time0:
                updated_conv_state = qk_packed1[
                    :, :, -self.cca_time0 :
                ].contiguous()  # [B, E_global, 2]
            else:
                updated_conv_state = F.pad(
                    qk_packed1, (self.cca_time0 - seq_length, 0)
                ).contiguous()  # [B, E_global, 2]

            # NOTE: Do NOT add + conv_state * 0 here. Adding param * 0 inside
            # the layer computation corrupts CTE output when combined with
            # input_output_aliases. The force-read is handled in forward().

        # Extract per-rank Q/K from global conv output
        q_conv, k_conv = self._extract_per_rank_qk(qk_packed3_global)

        # Build queries/keys from per-rank conv output + per-rank means
        query = (
            q_conv.view(*q_conv.shape[:2], self.num_q_heads, self.head_dim) + qk_mean_q
        )
        key = (
            k_conv.view(*k_conv.shape[:2], self.num_kv_heads, self.head_dim) + qk_mean_k
        )

        # Values from two time streams (replicated on all ranks)
        v1 = self.val_proj1(hs)  # [S, B, latent_k_dim_global // 2] = full KV head 0
        v2 = self.val_proj2(hs_d)  # [S, B, latent_k_dim_global // 2] = full KV head 1
        # Concatenate to get all KV heads: [S, B, latent_k_dim_global]
        value_full = torch.cat([v1, v2], dim=-1).contiguous()
        # Reshape to [S, B, num_kv_heads_global, head_dim]
        value_full = value_full.view(
            *hs.shape[:2], self.num_kv_heads_global, self.head_dim
        )
        # Per-rank KV head slice
        value = self._extract_per_rank_heads(value_full, self.num_kv_heads, dim=2)

        # L2-normalize per head, then scale
        query_norm = query.norm(p=2, dim=-1, keepdim=True)
        key_norm = key.norm(p=2, dim=-1, keepdim=True)

        # Temperature: per-rank slice from global temp parameter
        temp_per_rank = self._extract_per_rank_heads(
            self.temp, self.num_kv_heads, dim=0
        )

        key = (key * (self.sqrt_head_dim / key_norm)) * temp_per_rank[
            None, None
        ].unsqueeze(-1)
        query = query * (self.sqrt_head_dim / query_norm)

        # Flatten head axis, return to HF layout [B, S, ...]
        query = (
            query.view(*query.shape[:2], self.num_q_heads * self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        key = (
            key.view(*key.shape[:2], self.num_kv_heads * self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        value = (
            value.view(*value.shape[:2], self.num_kv_heads * self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        return query, key, value, updated_conv_state, updated_prev_hs


# ---------------------------------------------------------------------------
# Attention Layer
# ---------------------------------------------------------------------------


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply partial RoPE: only the first `rotary_dim` dimensions."""
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class NeuronZayaAttention(nn.Module):
    """ZAYA CCA-based attention layer.

    This wraps the CCA module and adds the standard attention computation
    (QK^T, softmax, AV, o_proj) with partial RoPE.

    NOT inheriting from NeuronAttentionBase because CCA has fundamentally
    different QKV computation. Task 008 may refactor this.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        # TP-aware per-rank head counts
        tp_degree = (
            getattr(config.neuron_config, "tp_degree", 1)
            if hasattr(config, "neuron_config")
            else 1
        )
        self.tp_degree = tp_degree
        self.num_q_heads_per_rank = config.cca_num_q_heads[layer_idx] // tp_degree
        self.num_kv_heads_per_rank = (
            config.num_query_groups_list[layer_idx] // tp_degree
        )
        self.num_key_value_groups_per_rank = (
            self.num_q_heads_per_rank // self.num_kv_heads_per_rank
        )

        # Global CCA Q head count (for o_proj sizing)
        cca_num_q_heads_global = config.cca_num_q_heads[layer_idx]

        # CCA produces compressed Q (num_q_heads = num_heads // 2)
        # RowParallelLinear: input is already sharded (per-rank Q heads), output is all-reduced to full hidden_size
        self.o_proj = RowParallelLinear(
            cca_num_q_heads_global
            * self.head_dim,  # global input dim (NxD slices to per-rank)
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=getattr(config.neuron_config, "torch_dtype", torch.bfloat16),
        )

        self.qkv = CCA(
            config=config,
            cca_num_q_heads=config.cca_num_q_heads[layer_idx],
            cca_num_kv_heads=config.num_query_groups_list[layer_idx],
            cca_num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            layer_number=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        cos_cache=None,
        sin_cache=None,
        cca_mask=None,
        is_for_context_encoding: bool = True,
        conv_state: Optional[torch.Tensor] = None,
        prev_hs_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        query_states, key_states, value_states, updated_conv_state, updated_prev_hs = (
            self.qkv(
                hidden_states,
                past_key_value,
                cca_mask,
                is_for_context_encoding=is_for_context_encoding,
                conv_state=conv_state,
                prev_hs_cache=prev_hs_cache,
            )
        )

        query_states = query_states.view(
            batch_size, seq_length, self.num_q_heads_per_rank, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.num_kv_heads_per_rank, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.num_kv_heads_per_rank, self.head_dim
        ).transpose(1, 2)

        # Apply partial RoPE
        if cos_cache is not None and sin_cache is not None:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos_cache, sin_cache
            )

        # Save KV for cache BEFORE GQA repeat (BHSD format)
        # Cast to bf16 to match KV cache allocation
        present_kv = (key_states.to(torch.bfloat16), value_states.to(torch.bfloat16))

        # ---- Token generation: attend to full KV cache ----
        # During token generation (seq_length=1), past_key_value contains the
        # full KV cache from all previous positions. We must attend to it.
        # NxDI decomposes this into prior (cache) + active (current) attention.
        is_token_gen = past_key_value is not None and seq_length == 1

        if is_token_gen:
            # Prior KV from cache: [B, num_kv_heads, cache_size, head_dim]
            K_prior = past_key_value[0]
            V_prior = past_key_value[1]

            # GQA repeat for prior KV (num_kv_heads -> num_q_heads per rank)
            K_prior_expanded = repeat_kv(K_prior, self.num_key_value_groups_per_rank)
            V_prior_expanded = repeat_kv(V_prior, self.num_key_value_groups_per_rank)

            # Prior scores: Q @ K_prior^T / sqrt(d)
            # Q: [B, num_q_heads_per_rank, 1, head_dim], K_prior: [B, num_q_heads_per_rank, cache_size, head_dim]
            prior_scores = torch.matmul(
                query_states, K_prior_expanded.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            # Apply attention_mask to prior scores (boolean mask from NxDI)
            # attention_mask shape: [B, num_kv_heads_per_rank, 1, cache_size] — True for valid
            # We need to expand to num_q_heads_per_rank
            if attention_mask is not None:
                prior_attn_mask = attention_mask.expand(
                    -1, self.num_q_heads_per_rank, -1, -1
                )
                prior_scores = torch.where(
                    prior_attn_mask, prior_scores, torch.finfo(prior_scores.dtype).min
                )
            prior_scores = prior_scores.to(torch.float32)

            # Active scores: Q @ K_active^T / sqrt(d)
            K_active = repeat_kv(key_states, self.num_key_value_groups_per_rank)
            V_active = repeat_kv(value_states, self.num_key_value_groups_per_rank)
            active_scores = torch.matmul(
                query_states, K_active.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            active_scores = active_scores.to(torch.float32)

            # Decomposed softmax: softmax over [prior_scores, active_scores]
            # Compute max across both for numerical stability
            max_prior = prior_scores.max(dim=-1, keepdim=True).values
            max_active = active_scores.max(dim=-1, keepdim=True).values
            max_all = torch.maximum(max_prior, max_active)

            exp_prior = torch.exp(prior_scores - max_all)
            exp_active = torch.exp(active_scores - max_all)
            sum_exp = exp_prior.sum(dim=-1, keepdim=True) + exp_active.sum(
                dim=-1, keepdim=True
            )

            softmax_prior = (exp_prior / sum_exp).to(query_states.dtype)
            softmax_active = (exp_active / sum_exp).to(query_states.dtype)

            attn_output = torch.matmul(softmax_prior, V_prior_expanded) + torch.matmul(
                softmax_active, V_active
            )
        else:
            # ---- Context encoding (prefill): causal attention ----
            # Try NxDI internal NKI flash attention first.
            # Falls back to manual attention if not available (CPU mode).
            # Set ZAYA_DISABLE_NKI=1 to force manual attention (for A/B benchmarks).
            nki_disabled = os.environ.get("ZAYA_DISABLE_NKI", "0") == "1"
            nki_kernel = None if (cpu_mode() or nki_disabled) else _get_nki_flash_fwd()

            if nki_kernel is not None:
                # NxDI NKI flash attention path — handles GQA natively.
                #
                # attention_nki_kernel_adapter expects (same layout as nkilib):
                #   Q: (B*H_q, seqlen, d_head)   [tp_q=True]
                #   K: (B*H_kv, d_head, seqlen)   [tp_k=False]
                #   V: (B*H_kv, seqlen, d_head)
                # GQA: B*H_q % B*H_kv == 0 → kernel auto-maps Q heads to KV groups.
                #
                # Current shapes after CCA + RoPE:
                #   query_states: [B, num_q_heads_per_rank, S, head_dim]
                #   key_states:   [B, num_kv_heads_per_rank, S, head_dim]
                #   value_states: [B, num_kv_heads_per_rank, S, head_dim]

                bsz = query_states.shape[0]
                sq = query_states.shape[2]
                d = query_states.shape[3]

                # Reshape: fold batch and heads into dim 0
                q_cte = query_states.reshape(
                    bsz * self.num_q_heads_per_rank, sq, d
                )  # (B*H_q, S, D)
                k_cte = key_states.permute(0, 1, 3, 2).reshape(
                    bsz * self.num_kv_heads_per_rank, d, sq
                )  # (B*H_kv, D, S)
                v_cte = value_states.reshape(
                    bsz * self.num_kv_heads_per_rank, sq, d
                )  # (B*H_kv, S, D)

                # Scale: 1/sqrt(d_head). CCA already L2-normalized Q/K and
                # applied temperature scaling, but RoPE was applied AFTER
                # normalization, so the standard 1/sqrt(d) scale is still needed.
                scale = 1.0 / math.sqrt(d)

                # Use LNC grid for multi-core sharding within each TP rank.
                # Following NxDI pattern: grid = (nc(logical_nc_config),)
                from neuronxcc.nki.language import nc

                logical_nc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
                grid = (nc(logical_nc),)

                attn_output = nki_kernel[grid](
                    q_cte,
                    k_cte,
                    v_cte,
                    scale,
                    do_out_tp=False,
                    tp_q=True,
                    tp_k=False,
                    use_dma_transpose=True,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
                # Output: (B*H_q, S, D) → reshape back to [B, H_q, S, D]
                attn_output = attn_output.reshape(bsz, self.num_q_heads_per_rank, sq, d)
            else:
                # Fallback: manual attention (CPU mode or nkilib unavailable)
                # GQA: repeat KV heads (per-rank)
                key_states_expanded = repeat_kv(
                    key_states, self.num_key_value_groups_per_rank
                )
                value_states_expanded = repeat_kv(
                    value_states, self.num_key_value_groups_per_rank
                )

                attn_weights = torch.matmul(
                    query_states, key_states_expanded.transpose(2, 3)
                ) / math.sqrt(self.head_dim)

                # Build proper 4D causal mask (lower triangular)
                if seq_length > 1:
                    causal_mask = torch.full(
                        (seq_length, seq_length),
                        torch.finfo(attn_weights.dtype).min,
                        device=attn_weights.device,
                        dtype=attn_weights.dtype,
                    )
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

                    # Mask out padded KEY positions
                    if cca_mask is not None:
                        pad_mask = (1.0 - cca_mask).unsqueeze(1).unsqueeze(2)
                        pad_mask = pad_mask * torch.finfo(attn_weights.dtype).min
                        causal_mask = causal_mask + pad_mask.to(causal_mask.dtype)

                    attn_weights = attn_weights + causal_mask

                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attn_output = torch.matmul(attn_weights, value_states_expanded)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            batch_size, seq_length, self.num_q_heads_per_rank * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, present_kv, updated_conv_state, updated_prev_hs


# ---------------------------------------------------------------------------
# MLP Expert
# ---------------------------------------------------------------------------


class ZayaMLP(nn.Module):
    """Single expert MLP with SwiGLU activation.

    Uses separate gate_proj + up_proj (not fused) for ISA kernel compatibility.
    The checkpoint has fused linear_fc1 weights — these are split during weight
    conversion in convert_zaya_hf_to_neuron_state_dict().

    gate_proj: [hidden_size -> intermediate_size]  (ColumnParallelLinear)
    up_proj:   [hidden_size -> intermediate_size]  (ColumnParallelLinear)
    down_proj: [intermediate_size -> hidden_size]  (RowParallelLinear)

    With TP=2: intermediate_size is sharded across ranks. RowParallelLinear
    does all-reduce on the output.

    When mlp_kernel_enabled is True and running on Neuron, the mlp_isa_kernel
    fuses gate+silu+up+down into a single ISA-level kernel, avoiding
    intermediate activation HBM round-trips.
    """

    def __init__(self, config, ffn_hidden_size):
        super().__init__()
        self.config = config
        self.gated_linear_unit = getattr(config, "gated_linear_unit", True)

        if self.gated_linear_unit:
            intermediate_size = ffn_hidden_size // 2
        else:
            intermediate_size = ffn_hidden_size

        dtype = (
            getattr(config.neuron_config, "torch_dtype", torch.bfloat16)
            if hasattr(config, "neuron_config")
            else torch.bfloat16
        )

        self.hidden_size = config.hidden_size
        self.mlp_kernel_enabled = getattr(
            getattr(config, "neuron_config", None), "mlp_kernel_enabled", False
        )

        # Separate gate and up projections for ISA kernel compatibility.
        # The ISA kernel requires separate gate_w and up_w tensors.
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )

        # Pre-transpose weights for ISA kernel (Llama pattern).
        # The kernel expects [in, out/TP] layout instead of [out/TP, in].
        if self.mlp_kernel_enabled:
            try:
                from neuronx_distributed_inference.modules.attention.utils import (
                    transpose_parallel_linear_layer,
                )

                self.gate_proj.weight = transpose_parallel_linear_layer(
                    self.gate_proj.weight
                )
                self.up_proj.weight = transpose_parallel_linear_layer(
                    self.up_proj.weight
                )
                self.down_proj.weight = transpose_parallel_linear_layer(
                    self.down_proj.weight
                )
            except ImportError:
                import warnings

                warnings.warn(
                    "Could not import transpose_parallel_linear_layer; "
                    "MLP ISA kernel will fall back to native MLP."
                )
                self.mlp_kernel_enabled = False

    def forward(self, hidden_states):
        if (
            self.mlp_kernel_enabled
            and not cpu_mode()
            and hidden_states.device.type != "cpu"
        ):
            return self._kernel_mlp(hidden_states)
        return self._native_mlp(hidden_states)

    def _native_mlp(self, hidden_states):
        """Standard PyTorch MLP (fallback path)."""
        gate_out = F.silu(self.gate_proj(hidden_states))
        up_out = self.up_proj(hidden_states)
        output = self.down_proj(gate_out * up_out)
        return output, None  # (output, bias=None)

    def _kernel_mlp(self, hidden_states):
        """Fused MLP via mlp_isa_kernel (Neuron ISA kernel).

        Fuses gate+silu+up*gate+down into a single kernel launch.
        Weights are already pre-transposed to [in, out/TP] layout.
        """
        kernel_pack = _get_mlp_isa_kernel()
        if kernel_pack is None:
            return self._native_mlp(hidden_states)

        mlp_isa_kernel, NormType, nki_jit, nc, reduce_fn = kernel_pack

        # Ensure 3D input: [batch, seq, hidden]
        orig_shape = hidden_states.shape
        if len(orig_shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)

        # No fused RMSNorm — pass zeros for ln_w
        ln_w = torch.zeros(
            size=(1, hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        output_tensor = torch.zeros(
            size=(hidden_states.shape[0], hidden_states.shape[1], self.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Weights are already pre-transposed in __init__
        gate_w = self.gate_proj.weight.data
        up_w = self.up_proj.weight.data
        down_w = self.down_proj.weight.data

        # Grid: LNC=2 on trn2
        grid = (nc(2),)

        nki_jit()(mlp_isa_kernel)[grid](
            hidden_states,
            ln_w,
            gate_w,
            up_w,
            down_w,
            output_tensor,
            kernel_name="MLP",
            norm_type=NormType.NO_NORM,
        )

        # All-reduce across TP ranks (RowParallelLinear equivalent)
        output_tensor = reduce_fn(output_tensor)

        # Restore original shape
        if len(orig_shape) == 2:
            output_tensor = output_tensor.squeeze(0)

        return output_tensor, None  # (output, bias=None)


class SequentialMLP(nn.Module):
    """Sequential expert dispatch — executes each expert one at a time.

    Task 009-010 will replace this with NxDI ExpertMLPsV2 for parallelism.
    """

    def __init__(self, num_local_experts: int, config, ffn_hidden_size: int):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.local_experts = nn.ModuleList(
            [ZayaMLP(config, ffn_hidden_size) for _ in range(num_local_experts)]
        )

    def forward(self, permuted_hidden_states, tokens_per_expert):
        if self.num_local_experts == 1:
            return self.local_experts[0](permuted_hidden_states)

        tokens_per_expert = tokens_per_expert.tolist()
        tokens_list = torch.split(permuted_hidden_states, tokens_per_expert)

        output_list = []
        for expert, tokens in zip(self.local_experts, tokens_list):
            out, _ = expert(tokens)
            output_list.append(out)

        return torch.cat(output_list, dim=0), None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ZayaRouter(nn.Module):
    """Non-linear MLP router with EDA and MoD.

    Key features:
    - Down-projection -> optional EDA -> RMSNorm -> 3-layer MLP -> softmax
    - Top-k selection with balancing biases
    - MoD: last expert index is "skip" (passthrough)
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        num_moe_experts: int,
        moe_router_topk: int,
        mlp_expansion: int,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = int(hidden_size or config.hidden_size)

        self.use_mod = bool(getattr(config, "zaya_use_mod", True))
        self.num_experts = (num_moe_experts + 1) if self.use_mod else num_moe_experts
        self.topk = int(moe_router_topk)
        self.mlp_expansion = int(mlp_expansion)

        # Down projection (has bias — hardcoded in HF model)
        self.down_proj = nn.Linear(self.hidden_size, self.mlp_expansion, bias=True)

        # EDA (exponential depth averaging) — disabled for first MoE layer
        zaya_first_layer = 1  # first MoE layer index in the 80-layer scheme
        use_eda_cfg = bool(getattr(config, "zaya_use_eda", True))
        self.use_eda = use_eda_cfg and (layer_idx != zaya_first_layer)

        self.rmsnorm_eda = get_rmsnorm_cls()(self.mlp_expansion, eps=1e-6)
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(self.mlp_expansion))

        # 3-layer MLP router: Linear -> GELU -> Linear -> GELU -> Linear
        D = self.mlp_expansion
        E = self.num_experts
        self.router_mlp = nn.Sequential(
            nn.Linear(D, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
            nn.GELU(),
            nn.Linear(D, E, bias=False),
        )

        # Balancing biases (parameter instead of buffer for XLA compatibility)
        balancing_biases = torch.zeros(self.num_experts, dtype=torch.float32)
        if self.use_mod:
            balancing_biases[-1] = -1.0
        self.balancing_biases = nn.Parameter(balancing_biases, requires_grad=False)

    def forward(self, hidden_states, router_states=None):
        """
        Args:
            hidden_states: [B, S, H]
            router_states: [B, S, D] from previous MoE layer (for EDA)

        Returns:
            route_prob: [B*S, topk]
            expert_choice: [B*S, topk]
            router_hidden_states_next: [B, S, D]
        """
        B, S, _ = hidden_states.shape

        hs = self.down_proj(hidden_states)

        if self.use_eda and router_states is not None:
            hs = hs + router_states * self.router_states_scale

        router_hidden_states_next = hs[:, -S:].clone()

        hs_norm = self.rmsnorm_eda(hs)
        logits = self.router_mlp(hs_norm)
        # Match HF: softmax in native dtype (bf16), NOT float32.
        # Different precision → different expert routing decisions across 40 layers.
        expert_prob = torch.softmax(logits, dim=-1)

        # Match HF: detach + explicit float32 cast for biased selection.
        # balancing_biases may have been cast to bf16 by NxDI weight loader,
        # so we explicitly .float() them here.
        biased = expert_prob.detach().to(torch.float32) + self.balancing_biases.float()
        _, expert_choice_t = torch.topk(biased, self.topk, dim=-1)

        route_prob = torch.gather(expert_prob, dim=2, index=expert_choice_t)
        return (
            route_prob.reshape(-1, self.topk),
            expert_choice_t.reshape(-1, self.topk),
            router_hidden_states_next,
        )


# ---------------------------------------------------------------------------
# MoE Block
# ---------------------------------------------------------------------------


class ZayaBlock(nn.Module):
    """MoE block: Router + SequentialMLP experts."""

    def __init__(
        self,
        config,
        layer_idx: int,
        num_moe_experts: int,
        mlp_expansion: int,
        ffn_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.num_moe_experts = num_moe_experts
        self.use_mod = bool(getattr(config, "zaya_use_mod", True))

        self.router = ZayaRouter(
            config=config,
            layer_idx=layer_idx,
            num_moe_experts=num_moe_experts,
            moe_router_topk=getattr(config, "moe_router_topk", 1),
            mlp_expansion=mlp_expansion,
            hidden_size=config.hidden_size,
        )
        self.experts = SequentialMLP(num_moe_experts, config, ffn_hidden_size)

    def forward(self, hidden_states, prev_router_hidden_states=None):
        route_prob, expert_choice, prev_router_hidden_states = self.router(
            hidden_states, router_states=prev_router_hidden_states
        )

        batch_size, seq_length, emb_dim = hidden_states.shape
        num_tokens = batch_size * seq_length
        hidden_states_flat = hidden_states.view(num_tokens, emb_dim)
        # expert_choice: [num_tokens, topk=1] -> [num_tokens]
        indices_flat = expert_choice.view(num_tokens)

        # XLA-compatible static expert dispatch:
        # Process each expert with a mask (no dynamic indexing / bincount / sort).
        # For topk=1, each token goes to exactly one expert.
        total_experts = self.router.num_experts  # num_moe_experts + 1 if MoD
        num_real_experts = self.num_moe_experts

        expert_output = torch.zeros_like(hidden_states_flat)

        for expert_idx in range(num_real_experts):
            # Mask: 1 where this expert is selected, 0 elsewhere
            expert_mask = (
                (indices_flat == expert_idx).unsqueeze(-1).to(hidden_states_flat.dtype)
            )
            # Run expert on all tokens (masked zeros for non-selected)
            expert_input = hidden_states_flat * expert_mask
            expert_out, _ = self.experts.local_experts[expert_idx](expert_input)
            expert_output = expert_output + expert_out * expert_mask

        # MoD skip expert: tokens routed to the skip expert (last index)
        # just pass through (identity — already handled since expert_output
        # starts as zeros and those tokens get no expert contribution,
        # but we need to add the passthrough)
        if self.use_mod:
            skip_mask = (
                (indices_flat == (total_experts - 1))
                .unsqueeze(-1)
                .to(hidden_states_flat.dtype)
            )
            expert_output = expert_output + hidden_states_flat * skip_mask

        expert_output = expert_output.view(batch_size, seq_length, emb_dim)
        probs = route_prob.view(batch_size, seq_length)
        expert_output = expert_output * probs.unsqueeze(-1)

        return expert_output, prev_router_hidden_states


# ---------------------------------------------------------------------------
# Decoder Layers
# ---------------------------------------------------------------------------


class NeuronZayaAttentionLayer(nn.Module):
    """Attention decoder layer: ResidualScaling -> RMSNorm -> CCA Attention.

    Conforms to NxDI decoder layer interface:
    Input: (hidden_states, seq_ids=, attention_mask=, position_ids=,
            past_key_value=, cos_cache=, sin_cache=, residual=, **kwargs)
    Output: (hidden_states, kv_cache, cos_cache, sin_cache, residual)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = NeuronZayaAttention(config, layer_idx)
        self.input_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.norm_epsilon)
        self.rotary_emb = ZayaRotaryEmbedding(config)

        if getattr(config, "scale_residual_merge", True):
            self.res_scale = ResidualScaling(
                config.hidden_size, not_first_layer=(layer_idx != 0)
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        cos_cache=None,
        sin_cache=None,
        residual=None,
        cca_mask=None,
        is_for_context_encoding=True,
        conv_state=None,
        prev_hs_cache=None,
        **kwargs,
    ):
        # Always call res_scale — it applies hidden_states_scale/bias even when
        # residual is None (layer 0). The HF model calls it unconditionally.
        if hasattr(self, "res_scale"):
            residual, hidden_states = self.res_scale(residual, hidden_states)

        if residual is None:
            residual = hidden_states
        else:
            residual = hidden_states + residual

        hidden_states = self.input_norm(residual.to(dtype=self.input_norm.weight.dtype))

        # Compute RoPE if not cached
        if cos_cache is None and position_ids is not None:
            cos_cache, sin_cache = self.rotary_emb(hidden_states, position_ids)

        hidden_states, present_kv, updated_conv_state, updated_prev_hs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            cca_mask=cca_mask,
            is_for_context_encoding=is_for_context_encoding,
            conv_state=conv_state,
            prev_hs_cache=prev_hs_cache,
        )

        # Return 7-tuple: (hidden_states, kv_cache, cos_cache, sin_cache, residual,
        #                   updated_conv_state, updated_prev_hs)
        return (
            hidden_states,
            present_kv,
            cos_cache,
            sin_cache,
            residual,
            updated_conv_state,
            updated_prev_hs,
        )


class _RouterStateHolder:
    """Simple holder for EDA router hidden states that threads between MoE layers.

    This avoids storing a parent model reference on nn.Module (which would
    create circular references and infinite recursion in module traversal).
    """

    def __init__(self):
        self.prev_router_hidden_states = None

    def reset(self):
        self.prev_router_hidden_states = None


# Global holder instance — safe because XLA tracing is single-threaded
_router_state = _RouterStateHolder()


class NeuronZayaMoELayer(nn.Module):
    """MoE decoder layer: ResidualScaling -> RMSNorm -> ZayaBlock (router + experts).

    Conforms to NxDI decoder layer interface.
    Uses _router_state global for EDA state threading between MoE layers.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Per-layer config from the lists
        num_moe_experts = config.zaya_layers[layer_idx]  # int, e.g. 16
        mlp_expansion = config.zaya_mlp_expansion[layer_idx]
        ffn_hidden_size = config.ffn_hidden_size_list[layer_idx]

        self.zaya_block = ZayaBlock(
            config=config,
            layer_idx=layer_idx,
            num_moe_experts=num_moe_experts,
            mlp_expansion=mlp_expansion,
            ffn_hidden_size=ffn_hidden_size,
        )
        self.input_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.norm_epsilon)

        if getattr(config, "scale_residual_merge", True):
            self.res_scale = ResidualScaling(config.hidden_size, not_first_layer=True)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        cos_cache=None,
        sin_cache=None,
        residual=None,
        **kwargs,
    ):
        # Always call res_scale — it applies hidden_states_scale/bias even when
        # residual is None. The HF model calls it unconditionally.
        if hasattr(self, "res_scale"):
            residual, hidden_states = self.res_scale(residual, hidden_states)

        if residual is None:
            residual = hidden_states
        else:
            residual = hidden_states + residual

        hidden_states = self.input_norm(residual.to(dtype=self.input_norm.weight.dtype))

        # Get/set router state from global holder
        prev_rhs = _router_state.prev_router_hidden_states
        hidden_states, new_rhs = self.zaya_block(hidden_states, prev_rhs)
        _router_state.prev_router_hidden_states = new_rhs

        # Return 5-tuple: (hidden_states, kv_cache, cos_cache, sin_cache, residual)
        # MoE layers don't have attention, but the base KV cache manager expects
        # a (K, V) tuple for every layer. Return dummy KV tensors with the correct
        # shape: (batch, num_kv_heads_per_rank, seq_len, head_dim) in BHSD format.
        # Under TP, num_kv_heads is divided by tp_degree.
        tp_degree = (
            getattr(self.config.neuron_config, "tp_degree", 1)
            if hasattr(self.config, "neuron_config")
            else 1
        )
        num_kv_heads_per_rank = self.config.num_key_value_heads // tp_degree
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        kv_dummy = (
            torch.zeros(
                batch_size,
                num_kv_heads_per_rank,
                seq_len,
                self.config.head_dim,
                device=hidden_states.device,
                dtype=torch.bfloat16,
            ),
            torch.zeros(
                batch_size,
                num_kv_heads_per_rank,
                seq_len,
                self.config.head_dim,
                device=hidden_states.device,
                dtype=torch.bfloat16,
            ),
        )
        return hidden_states, kv_dummy, cos_cache, sin_cache, residual


# ---------------------------------------------------------------------------
# Rotary Embedding (partial RoPE)
# ---------------------------------------------------------------------------


class ZayaRotaryEmbedding(nn.Module):
    """Partial RoPE with partial_rotary_factor=0.5.

    Only applies RoPE to the first half of head_dim.
    """

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        partial_factor = getattr(config, "partial_rotary_factor", 0.5)
        self.rotary_dim = int(self.head_dim * partial_factor)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# NeuronBaseModel Implementation
# ---------------------------------------------------------------------------


class NeuronZayaModel(NeuronBaseModel):
    """NxDI traced model for ZAYA1-base.

    Builds the 80-layer alternating attention/MoE architecture with
    per-layer residual scaling and partial RoPE.

    Overrides get_model_output() because ZAYA requires a top-level
    ResidualScaling before the final residual merge + norm, and also
    needs to reset the global _router_state at the start of each
    forward pass for EDA state threading between MoE layers.
    """

    def setup_attr_for_model(self, config: ZayaInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: ZayaInferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )

        # Build 80 layers alternating attention / MoE
        layers = []
        self._attn_layer_indices = []  # Track which layer indices are attention
        for layer_idx in range(len(config.zaya_layers)):
            layer_type = config.zaya_layers[layer_idx]
            if isinstance(layer_type, str) and layer_type == "a":
                layers.append(NeuronZayaAttentionLayer(config, layer_idx))
                self._attn_layer_indices.append(layer_idx)
            elif isinstance(layer_type, int):
                layers.append(NeuronZayaMoELayer(config, layer_idx))
            else:
                raise ValueError(
                    f"Unknown layer type at index {layer_idx}: {layer_type}"
                )
        self.layers = nn.ModuleList(layers)

        # --- CCA state caches (persistent across TKG steps via input_output_aliases) ---
        # One conv_state and one prev_hs per attention layer (40 layers).
        # These are nn.Parameter objects (requires_grad=False) so they participate
        # in the NxD input_output_aliases protocol for in-place state updates.
        num_attn_layers = len(self._attn_layer_indices)
        dtype = config.neuron_config.torch_dtype
        batch_size = config.neuron_config.max_batch_size
        conv_kernel_size = getattr(config, "cca_time0", 2)

        # Compute in_out_ch at GLOBAL dimensions (same on all ranks).
        # Conv1d operates on global data with gather_output=True on Q/K projections.
        # Conv state is stored at global dimensions too.
        # For ZAYA1-base: 8*128 + 2*128 = 1280 (same for any TP degree)
        cca_num_heads = getattr(config, "cca_num_heads", 16)
        head_dim = config.hidden_size // cca_num_heads
        first_attn_idx = self._attn_layer_indices[0]
        cca_num_q_heads = config.cca_num_q_heads[first_attn_idx]
        cca_num_kv_heads = config.num_query_groups_list[first_attn_idx]
        in_out_ch_global = cca_num_q_heads * head_dim + cca_num_kv_heads * head_dim

        self.cca_conv_states = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        batch_size, in_out_ch_global, conv_kernel_size, dtype=dtype
                    ),
                    requires_grad=False,
                )
                for _ in range(num_attn_layers)
            ]
        )
        self.cca_prev_hs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(batch_size, config.hidden_size, dtype=dtype),
                    requires_grad=False,
                )
                for _ in range(num_attn_layers)
            ]
        )

        # Top-level residual scaling (applied before final norm)
        if getattr(config, "scale_residual_merge", True):
            self.res_scale = ResidualScaling(
                config.hidden_size,
                not_first_layer=True,  # top-level always has residual_scale
            )

        # Final norm
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.norm_epsilon)

        # Rotary embedding
        self.rotary_emb = ZayaRotaryEmbedding(config)

        # LM head (tied with embed_tokens — handled in update_state_dict_for_tied_weights)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

        # Do NOT set qkv_kernel_fuse_residual_add — we handle the final
        # residual merge ourselves in get_model_output with proper res_scale.
        config.neuron_config.qkv_kernel_fuse_residual_add = False

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Custom get_model_output for ZAYA1-base.

        This override is needed because:
        1. ZAYA has a top-level ResidualScaling that must be applied before
           the final residual merge + norm. The base implementation only does
           `hidden_states = residual + hidden_states` without scaling.
        2. The global _router_state must be reset at the start of each pass
           to ensure clean EDA state threading between MoE layers.
        """
        batch_size, seq_length = input_ids.shape[:2]

        # Reset router state for EDA threading
        _router_state.reset()

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        hidden_states = self.process_sequence_parallel_hidden_states(
            inputs_embeds, seq_length, kwargs.get("active_block_table", None)
        )

        # Derive cca_mask from position_ids: 2D [B, S] with 1.0 for real tokens,
        # 0.0 for padding. NxDI pads input_ids to bucket size — padding tokens get
        # real embeddings that leak into CCA convolutions without this mask.
        # For token generation (seq_length == 1), no mask is needed (CCA skips it).
        if seq_length > 1 and position_ids is not None:
            max_positions = torch.argmax(position_ids, dim=1)
            seq_indices = torch.arange(
                seq_length, device=position_ids.device
            ).unsqueeze(0)
            cca_mask = (seq_indices <= max_positions.unsqueeze(1)).to(
                hidden_states.dtype
            )
        else:
            cca_mask = None

        # KV cache handling
        update_kv_per_layer = update_cache and (
            self.neuron_config.layer_boundary_markers
            or (
                self.neuron_config.attn_block_tkg_nki_kernel_cache_update
                and not is_for_context_encoding
            )
        )

        next_decoder_cache = [] if update_kv_per_layer else ()
        cos_cache = None
        sin_cache = None

        cache_size = self.n_positions
        get_kv_per_layer = False

        # Retrieve KV cache for token generation (not context encoding first pass)
        if not is_for_context_encoding or windowed_context_encoding_window_idx >= 1:
            if not self.config.neuron_config.layer_boundary_markers:
                past_key_values = self.kv_mgr.get_cache(
                    seq_ids=seq_ids,
                    seq_len=cache_size,
                    is_for_context_encoding=is_for_context_encoding,
                    windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                    **kwargs,
                )
            else:
                get_kv_per_layer = True

        residual = None

        # --- CCA state: read current state from nn.Parameters ---
        # Build lists of current conv_states and prev_hs for each attention layer.
        # These will be updated during the layer loop and returned for aliasing.
        attn_layer_counter = 0
        updated_conv_states = []
        updated_prev_hs_list = []

        # --- Layer loop ---
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            is_attn_layer = isinstance(decoder_layer, NeuronZayaAttentionLayer)

            # Pass CCA state params to layers for TKG conv_state input.
            # During CTE (prefill), params are passed but not used by the
            # layer's CTE code path. During TKG, they provide conv_state
            # and prev_hs_cache input from the prior step.
            extra_kwargs = {}
            if is_attn_layer:
                extra_kwargs["conv_state"] = self.cca_conv_states[attn_layer_counter]
                extra_kwargs["prev_hs_cache"] = self.cca_prev_hs[attn_layer_counter]

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
                get_kv_per_layer=get_kv_per_layer,
                update_kv_per_layer=update_kv_per_layer,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                residual=residual,
                local_mask=local_attn_mask,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                padding_mask=padding_mask,
                cca_mask=cca_mask,
                **extra_kwargs,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            kv = layer_outputs[1]
            if update_kv_per_layer:
                next_decoder_cache += kv
            else:
                next_decoder_cache += (kv,)
            cos_cache, sin_cache = layer_outputs[2:4]
            residual = layer_outputs[4]

            # Collect CCA state updates from attention layers.
            if is_attn_layer:
                # Attention layers return 7-tuple with updated_conv_state and updated_prev_hs
                updated_conv_state = layer_outputs[5]
                updated_prev_hs = layer_outputs[6]
                updated_conv_states.append(updated_conv_state)
                updated_prev_hs_list.append(updated_prev_hs)
                attn_layer_counter += 1

        # --- Top-level residual scaling + merge ---
        # This is the key difference from the base: ZAYA applies learned
        # scaling to both residual and hidden_states before merging.
        # Without this, residual_scale=0.045 is not applied and logits
        # are ~22x too large.
        if hasattr(self, "res_scale") and residual is not None:
            residual, hidden_states = self.res_scale(residual, hidden_states)
            hidden_states = residual + hidden_states
        elif residual is not None:
            hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)

        # Collect updated CCA states for aliasing.
        # Order: conv_states[0..39], then prev_hs[0..39]
        cca_states = updated_conv_states + updated_prev_hs_list

        # NOTE: KV cache update is NOT done here. It is handled in the
        # overridden forward() method (mllama pattern), which gives us
        # control over the output list construction: [res] + kv_cache + cca_states.
        return (hidden_states, next_decoder_cache, cca_states)

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
        """Override NeuronBaseModel.forward() to handle CCA state output.

        This follows the mllama pattern: get_model_output() returns 3 values
        (hidden_states, next_decoder_cache, cca_states), and this method
        constructs the output list as [res] + updated_kv_cache + cca_states.

        This separation ensures CCA states are NOT mixed into the KV cache
        update path, which caused output corruption in Test C.
        """
        # Workaround: NxD does not support kwargs in traced functions.
        # Convert empty tensors back to None for optional params.
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

        is_for_context_encoding = 1 < input_ids.shape[-1] != self.speculation_length
        is_for_speculation = input_ids.shape[-1] == self.speculation_length

        cache_size = self.n_positions

        # Prepare attention mask
        attn_mask = self.create_attn_mask(
            attention_mask,
            is_for_context_encoding,
            is_for_speculation,
            position_ids=position_ids,
        )

        active_mask = None

        # FlashDecoding masks (not used for ZAYA, but keep for completeness)
        active_mask_2d = None

        # Create padding mask
        padding_mask = self.create_padding_mask(position_ids)

        # Call get_model_output() which returns 3 values (mllama pattern)
        hidden_states, past_key_values, cca_states = self.get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            prev_hidden=prev_hidden,
            is_for_context_encoding=is_for_context_encoding,
            update_cache=True,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            padding_mask=padding_mask,
        )

        # KV cache update — done here, separate from CCA states
        updated_kv_cache = self.kv_mgr.update_cache(
            is_for_context_encoding=is_for_context_encoding,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=past_key_values,
            seq_len=cache_size,
        )

        # Extract last token's hidden state
        batch_size = input_ids.shape[0]
        if self.padding_side == "left":
            index = torch.tensor(
                [hidden_states.shape[1] - 1], device=hidden_states.device
            )
            index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            if not (
                position_ids.shape[-1] == self.speculation_length
                or position_ids.shape[-1] == 1
            ):
                # Context encoding — gather last real token
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        # LM head
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.on_device_sampling:
            res = self.sampler(
                logits[:, -1, :],
                sampling_params,
                rank_id=self.rank_util.get_rank(),
            )
            res = res.to(torch.int32)
        else:
            res = logits

        # CCA state aliased output:
        # CTE and TKG are compiled separately with different input shapes.
        # CTE (is_for_context_encoding=True): Use raw param identity reads.
        #   Returning computed CCA states corrupts CTE output (discovered via
        #   Tests C-G). Raw identity means CCA params stay at initial values
        #   (zeros) after CTE. First TKG token will have wrong conv_state,
        #   but this self-corrects from the 2nd TKG token onwards.
        # TKG (is_for_context_encoding=False): Use computed CCA states from
        #   layers with + param * 0 to force param read. This writes the
        #   updated conv_state/prev_hs back to the parameter for the next step.
        if is_for_context_encoding:
            # CTE: raw param identity — avoids corruption
            cca_output = []
            for p in self.cca_conv_states:
                cca_output.append(p + torch.zeros_like(p))
            for p in self.cca_prev_hs:
                cca_output.append(p + torch.zeros_like(p))
        else:
            # TKG: computed CCA states + param * 0 for aliasing
            cca_output = []
            num_attn = len(self.cca_conv_states)
            for i in range(num_attn):
                cca_output.append(cca_states[i] + self.cca_conv_states[i] * 0)
            for i in range(num_attn):
                cca_output.append(cca_states[num_attn + i] + self.cca_prev_hs[i] * 0)

        # Construct output: [res] + updated_kv_cache + cca_output
        outputs = [res] + updated_kv_cache + cca_output

        return outputs


# ---------------------------------------------------------------------------
# Weight Conversion
# ---------------------------------------------------------------------------


def convert_zaya_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
    """Convert ZAYA1-base HuggingFace state dict to NxDI format.

    The NxDI base class strips the 'model.' prefix before calling this function.
    So HF keys arrive as 'layers.{l}.*', 'embed_tokens.*', etc.

    Key transformations:
    1. Rename 'final_norm.weight' -> 'norm.weight'
    2. Split fused 'linear_fc1.weight' -> 'gate_proj.weight' + 'up_proj.weight'
       (checkpoint has [4096, 2048] gate+up fused; split into [2048, 2048] each)
    3. Rename 'linear_fc2.weight' -> 'down_proj.weight'
    4. Add rank utilities for tensor parallelism (SPMDRank arange per CCA)
    5. Add zero-initialized CCA state tensors at GLOBAL dimensions (conv operates
       at global dims via gather_output=True on Q/K projections)
    6. Pass through all other weights as-is — ColumnParallelLinear/RowParallelLinear
       weights are auto-sharded by NxD's parallel layer mechanism.
    7. Conv1d weights, temperature, and val_proj are at GLOBAL dimensions (plain
       nn.Conv1d/nn.Linear/nn.Parameter — NOT auto-sharded). Per-rank head
       extraction happens in CCA.forward() via index_select.
    """
    neuron_state_dict = {}
    tp_degree = config.neuron_config.tp_degree

    # Compute per-rank CCA dimensions for Conv1d slicing
    attn_layer_indices = [
        i
        for i, lt in enumerate(config.zaya_layers)
        if isinstance(lt, str) and lt == "a"
    ]
    cca_num_heads = getattr(config, "cca_num_heads", 16)
    head_dim = config.hidden_size // cca_num_heads
    first_attn_idx = attn_layer_indices[0]
    cca_num_q_heads_global = config.cca_num_q_heads[first_attn_idx]
    cca_num_kv_heads_global = config.num_query_groups_list[first_attn_idx]
    in_out_ch_global = (
        cca_num_q_heads_global * head_dim + cca_num_kv_heads_global * head_dim
    )

    for key, value in state_dict.items():
        new_key = key

        # Rename final_norm -> norm (NxDI convention)
        if key == "final_norm.weight":
            new_key = "norm.weight"

        # Split fused linear_fc1 (gate+up) into separate gate_proj and up_proj.
        # Checkpoint shape: [ffn_hidden_size, hidden_size] = [4096, 2048]
        # First half is gate weights, second half is up weights.
        if "linear_fc1.weight" in key:
            w = value.detach().clone()
            mid = w.shape[0] // 2
            gate_key = key.replace("linear_fc1.weight", "gate_proj.weight")
            up_key = key.replace("linear_fc1.weight", "up_proj.weight")
            neuron_state_dict[gate_key] = w[:mid, :]  # [2048, 2048]
            neuron_state_dict[up_key] = w[mid:, :]  # [2048, 2048]
            continue

        # Rename linear_fc2 -> down_proj
        if "linear_fc2.weight" in key:
            new_key = key.replace("linear_fc2.weight", "down_proj.weight")

        neuron_state_dict[new_key] = value.detach().clone()

    # Add rank utilities for tensor parallelism
    neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

    # Add SPMDRank tensors for each CCA module's rank_util.
    # Each CCA is at: layers.{attn_idx}.self_attn.qkv.rank_util.rank
    # These get sharded per-rank at load time (rank 0 gets [0], rank 1 gets [1]).
    if tp_degree > 1:
        for attn_idx in attn_layer_indices:
            neuron_state_dict[f"layers.{attn_idx}.self_attn.qkv.rank_util.rank"] = (
                torch.arange(0, tp_degree, dtype=torch.int32)
            )

    # Add zero-initialized CCA state tensors at GLOBAL dimensions.
    # Conv1d operates at global dims (Q/K projections use gather_output=True),
    # so conv_state is global too. Same on all ranks — not sharded.
    dtype = config.neuron_config.torch_dtype
    batch_size = config.neuron_config.max_batch_size
    conv_kernel_size = getattr(config, "cca_time0", 2)

    for i in range(len(attn_layer_indices)):
        neuron_state_dict[f"cca_conv_states.{i}"] = torch.zeros(
            batch_size, in_out_ch_global, conv_kernel_size, dtype=dtype
        )
        neuron_state_dict[f"cca_prev_hs.{i}"] = torch.zeros(
            batch_size, config.hidden_size, dtype=dtype
        )

    return neuron_state_dict


# ---------------------------------------------------------------------------
# Model Wrapper + Instance (for CCA state input_output_aliases)
# ---------------------------------------------------------------------------


class ZayaDecoderModelInstance(DecoderModelInstance):
    """Custom model instance that adds CCA conv_states and prev_hs to
    input_output_aliases, enabling persistent state across TKG steps.

    This follows the same pattern as MMDecoderModelInstance (mllama) which
    adds vision_key_values aliases beyond the standard KV cache.
    """

    def get(self, bucket_rank, **kwargs):
        # Call parent's get() to set up standard KV cache aliases
        self.module, self.input_output_aliases = super().get(bucket_rank, **kwargs)

        # Count existing outputs: res [+ logits] + KV caches
        past_key_values = self.module.kv_mgr.past_key_values
        num_output_from_trace = (
            1 if not self.neuron_config.output_logits else 2
        ) + len(past_key_values)

        # Add conv_states aliases (one per attention layer)
        for i in range(len(self.module.cca_conv_states)):
            self.input_output_aliases[self.module.cca_conv_states[i]] = (
                num_output_from_trace + i
            )
        num_output_from_trace += len(self.module.cca_conv_states)

        # Add prev_hs aliases (one per attention layer)
        for i in range(len(self.module.cca_prev_hs)):
            self.input_output_aliases[self.module.cca_prev_hs[i]] = (
                num_output_from_trace + i
            )

        return self.module, self.input_output_aliases


class ZayaModelWrapper(ModelWrapper):
    """Custom model wrapper that returns ZayaDecoderModelInstance."""

    def get_model_instance(self):
        return ZayaDecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )


# ---------------------------------------------------------------------------
# Top-Level Model
# ---------------------------------------------------------------------------


class NeuronZayaForCausalLM(NeuronBaseForCausalLM):
    """NxDI wrapper for ZAYA1-base Causal Language Model.

    Provides the 4 required interfaces:
    1. _model_cls -> NeuronZayaModel
    2. load_hf_model() -> load from Zyphra/ZAYA1-base
    3. get_config_cls() -> ZayaInferenceConfig
    4. convert_hf_to_neuron_state_dict() -> weight mapping
    """

    _model_cls = NeuronZayaModel

    def get_model_wrapper_cls(self):
        """Return custom model wrapper with CCA state aliasing."""
        return ZayaModelWrapper

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace ZAYA1-base model.

        Requires Zyphra's custom transformers fork:
            pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya"

        Note: The Zyphra fork uses @jit_fuser which calls torch.jit.script.
        This crashes in the Neuron environment because torch.jit.script cannot
        inspect the source of builtin functions. We temporarily disable
        torch.jit.script during import and restore it afterward to avoid
        interfering with compilation.
        """
        import torch

        _real_jit_script = torch.jit.script
        torch.jit.script = (
            lambda fn=None, *a, **kw: fn if fn is not None else (lambda f: f)
        )
        try:
            from transformers.models.zaya.modeling_zaya import ZayaForCausalLM

            model = ZayaForCausalLM.from_pretrained(model_path, **kwargs)
        finally:
            torch.jit.script = _real_jit_script
        return model

    @classmethod
    def get_config_cls(cls):
        return ZayaInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        return convert_zaya_hf_to_neuron_state_dict(state_dict, config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied word embeddings: lm_head.weight = embed_tokens.weight."""
        if "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    def get_compiler_args(self):
        """Compiler arguments for ZAYA1-base on Neuron.

        ManualConv1d replaces nn.Conv1d, so NKI Conv1d kernel insertion
        is no longer relevant. Using transformer model type for better
        optimization of the attention patterns.
        Uses matmult auto-cast for bf16 performance.
        """
        compiler_args = "--model-type=transformer -O1"
        compiler_args += " --auto-cast=matmult"
        return compiler_args
