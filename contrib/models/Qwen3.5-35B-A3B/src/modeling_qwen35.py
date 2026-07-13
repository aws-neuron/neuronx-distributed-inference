"""
NxDI contrib: Qwen3.5-27B / Qwen3.6-27B (qwen3_5 -- dense model)

Supports both Qwen3.5-27B and Qwen3.6-27B. These models share identical
architecture (qwen3_5 model_type). Qwen3.6-27B is a post-training update
with improved agentic coding and thinking preservation -- no architecture
changes, only weight differences.

Hybrid DeltaNet + Standard Attention + Dense MLP architecture.
Adapted from Qwen3.5-35B-A3B (MoE) -- MoE removed, dense MLP added.

48 of 64 layers use Gated DeltaNet (linear recurrent attention)
16 of 64 layers use standard GQA with KV cache + output gate
All 64 layers use a dense SwiGLU MLP (intermediate_size=17408)

Architecture details:
- DeltaNet layers: separate in_proj_{qkv, z, a, b}, causal conv1d on QKV, gated delta rule
- Attention layers: q_proj doubled (Q + gate), partial RoPE (25% of head_dim), sigmoid output gate
- Dense MLP: standard SwiGLU (gate_proj, up_proj, down_proj) -- no MoE, no router, no experts
- KV cache: NxDI KVCacheManager for attention layers; DeltaNet layers store recurrent+conv
  state as nn.Parameter buffers and return dummy KV tuples

Config compatibility notes:
- Qwen3.6-27B adds output_gate_type="swish" to text_config. This field is
  unused by both HF transformers and this NxDI code (gate uses sigmoid, as
  confirmed across transformers v4.57.6, v5.6.0, and GitHub main). Safe to ignore.
"""

import copy
import gc
import json
import math
import logging
import os
import re
import sys
import time
from typing import Any, Hashable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
try:
    from neuronx_distributed_inference.modules.async_execution import (
        cancel_hybrid_apc_request,
        finish_hybrid_apc_request,
        prepare_hybrid_apc_model_inputs,
        prepare_hybrid_apc_request_for_execution,
    )
except ImportError:
    def cancel_hybrid_apc_request(*a, **kw):
        return None

    def finish_hybrid_apc_request(*a, **kw):
        return None

    def prepare_hybrid_apc_model_inputs(*a, **kw):
        return None

    def prepare_hybrid_apc_request_for_execution(*a, **kw):
        return None
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
from neuronx_distributed.parallel_layers.mappings import _gather_along_dim
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
from src.nki_kernels.nki_deltanet import (
    deltanet_recurrent_step_batched as _deltanet_nki_step_batched,
)
from src.nki_kernels.nki_deltanet_chunked import (
    deltanet_chunk_step as _deltanet_nki_chunk_step,
)
from src.nki_kernels.nki_deltanet_fused import (
    deltanet_autocp_affine_sequence as _deltanet_autocp_affine_sequence,
    deltanet_autocp_apply_output as _deltanet_autocp_apply_output,
    deltanet_autocp_prefix_apply_output as _deltanet_autocp_prefix_apply_output,
    deltanet_autocp_state_summary_sequence as _deltanet_autocp_state_summary_sequence,
    deltanet_autocp_state_prefix as _deltanet_autocp_state_prefix,
    deltanet_fused_chunked_fwd as _deltanet_fused_kernel,
    deltanet_fused_chunked_fwd_multihead as _deltanet_fused_multihead_kernel,
)
from src.nki_kernels.nki_deltanet_fused_legacy import (
    deltanet_fused_chunked_fwd as _deltanet_fused_legacy_direct_kernel,
)
from src.nki_kernels.nki_deltanet_fused import (
    _make_lower_mask,
    _make_lower_mask_diag,
    _make_identity,
)
try:
    import nki as _nkilib_nki
    from nkilib.core.qkv.qkv import qkv as _nkilib_qkv
    from nkilib.core.utils.common_types import (
        NormType as _NkilibNormType,
        QKVOutputLayout as _NkilibQKVOutputLayout,
        QuantizationType as _NkilibQuantizationType,
    )

    _qwen_gate_projection_kernel = _nkilib_nki.jit(_nkilib_qkv)
except Exception:
    _NkilibNormType = None
    _NkilibQKVOutputLayout = None
    _NkilibQuantizationType = None
    _qwen_gate_projection_kernel = None

try:
    from src.nki_kernels.qwen_qk_norm_rope import (
        qwen_qk_norm_partial_rope_kernel as _qwen_qk_norm_partial_rope_kernel,
    )
except Exception:
    _qwen_qk_norm_partial_rope_kernel = None
from src.hybrid_apc import (
    HybridAPCMetadataStore,
    HybridAPCSchedulerBridge,
    HybridAPCSlotAllocator,
)


def _infer_neuron_lnc(default: int = 1) -> int:
    flags = os.environ.get("NEURON_CC_FLAGS", "")
    match = re.search(r"(?:^|\s)--lnc(?:=|\s+)(\d+)", flags)
    if match is None:
        return default
    return max(1, int(match.group(1)))


def _resolve_deltanet_multihead_group_size(total_heads: int) -> int:
    lnc = _infer_neuron_lnc()
    raw_group_size = os.environ.get("QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE")
    if raw_group_size is None:
        requested_group_size = 2 if lnc >= 2 else 1
    else:
        requested_group_size = max(1, int(raw_group_size))
        if requested_group_size > lnc:
            raise ValueError(
                f"QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE={requested_group_size} "
                f"requires NEURON_CC_FLAGS --lnc >= {requested_group_size}; "
                f"inferred lnc={lnc}"
            )
    return max(1, min(total_heads, requested_group_size))


def _deltanet_multihead_launch_spec(num_heads: int):
    """Return the launch spec for a grouped multihead DeltaNet CTE kernel.

    The legacy ``kernel[2]`` launch only covers two programs.  For larger
    grouped launches we need an SPMD axis distributed over the available NCs,
    while each program still handles exactly one flattened (batch, head) row.
    """
    lnc = _infer_neuron_lnc()
    if num_heads <= lnc:
        return num_heads
    if os.environ.get("QWEN36_DELTANET_MULTIHEAD_SPMD", "1") == "0":
        raise ValueError(
            "QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE exceeds inferred LNC but "
            "QWEN36_DELTANET_MULTIHEAD_SPMD=0; "
            f"group_size={num_heads}, inferred_lnc={lnc}"
        )

    import nki.language as _nl  # Imported lazily so CPU-only unit stubs still load.

    if not hasattr(_nl, "spmd_dim") or not hasattr(_nl, "nc"):
        if os.environ.get("QWEN36_DELTANET_MULTIHEAD_GRID_FALLBACK", "0") == "1":
            return (num_heads, 1)
        raise ValueError(
            "QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE exceeds inferred LNC, but "
            "this NKI runtime does not expose spmd_dim/nc; "
            f"group_size={num_heads}, inferred_lnc={lnc}"
        )
    return (_nl.spmd_dim(num_heads, _nl.nc(lnc)),)


def _qwen35_grouped_prefix_attention(
    Q,
    K_cache,
    V_cache,
    query_positions,
    cache_positions,
    key_valid_mask=None,
):
    """GQA-native prefix attention without materializing repeated KV heads."""
    B, q_heads, q_len, head_dim = Q.shape
    kv_heads = K_cache.shape[1]
    if q_heads % kv_heads != 0:
        raise ValueError(
            "Qwen grouped prefix attention requires q_heads to be divisible "
            f"by kv_heads, got q_heads={q_heads}, kv_heads={kv_heads}."
        )

    q_per_kv = q_heads // kv_heads
    if cache_positions.ndim == 4:
        cache_positions = cache_positions.reshape(B, -1)
    elif cache_positions.ndim != 2:
        raise ValueError(
            "cache_positions must have shape (B, K) or (B, 1, 1, K), "
            f"got {tuple(cache_positions.shape)}."
        )

    if key_valid_mask is not None:
        if key_valid_mask.ndim == 4:
            key_valid_mask = key_valid_mask.reshape(B, -1)
        elif key_valid_mask.ndim != 2:
            raise ValueError(
                "key_valid_mask must have shape (B, K) or (B, 1, 1, K), "
                f"got {tuple(key_valid_mask.shape)}."
            )

    q_grouped = Q.reshape(B, kv_heads, q_per_kv, q_len, head_dim)
    k_grouped = K_cache.transpose(-1, -2).unsqueeze(2)
    attn_weights = torch.matmul(q_grouped, k_grouped) / math.sqrt(head_dim)

    causal_mask = cache_positions[:, None, None, None, :] <= query_positions[
        :, None, None, :, None
    ]
    if key_valid_mask is not None:
        causal_mask = causal_mask & key_valid_mask[:, None, None, None, :]
    attn_weights = attn_weights.masked_fill(~causal_mask, -65504.0)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)

    attn_output = torch.matmul(attn_weights, V_cache.unsqueeze(2))
    return attn_output.reshape(B, q_heads, q_len, head_dim)


def _qwen35_expanded_prefix_attention(
    Q,
    K_cache,
    V_cache,
    query_positions,
    cache_positions,
    key_valid_mask=None,
):
    B, q_heads, q_len, head_dim = Q.shape
    kv_heads = K_cache.shape[1]
    cache_len = K_cache.shape[2]

    if q_heads != kv_heads:
        kv_rep = q_heads // kv_heads
        K_full = (
            K_cache.unsqueeze(2)
            .expand(-1, -1, kv_rep, -1, -1)
            .reshape(B, q_heads, cache_len, head_dim)
        )
        V_full = (
            V_cache.unsqueeze(2)
            .expand(-1, -1, kv_rep, -1, -1)
            .reshape(B, q_heads, cache_len, head_dim)
        )
    else:
        K_full = K_cache
        V_full = V_cache

    attn_weights = torch.matmul(Q, K_full.transpose(-1, -2)) / math.sqrt(head_dim)
    causal_mask = cache_positions <= query_positions[:, None, :, None]
    if key_valid_mask is not None:
        causal_mask = causal_mask & key_valid_mask
    attn_weights = attn_weights.masked_fill(~causal_mask, -65504.0)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
    return torch.matmul(attn_weights, V_full)


def _qwen36_prefix_attention_impl() -> str:
    raw = os.environ.get("QWEN36_PREFIX_ATTENTION_IMPL", "grouped").strip().lower()
    aliases = {
        "grouped": "grouped",
        "current": "grouped",
        "expanded": "expanded",
        "legacy": "expanded",
        "legacy_expanded": "expanded",
    }
    if raw not in aliases:
        raise ValueError(
            "QWEN36_PREFIX_ATTENTION_IMPL must be grouped/current or "
            f"expanded/legacy, got {raw!r}"
        )
    return aliases[raw]


def _resolve_deltanet_autocp_lnc(num_chunks: int) -> int:
    lnc = _infer_neuron_lnc()
    raw_lnc = os.environ.get("QWEN36_DELTANET_AUTOCP_LNC")
    if raw_lnc is None:
        launch_lnc = 2 if lnc >= 2 and num_chunks % 2 == 0 else 1
    else:
        launch_lnc = max(1, int(raw_lnc))
        if launch_lnc > lnc:
            raise ValueError(
                f"QWEN36_DELTANET_AUTOCP_LNC={launch_lnc} requires "
                f"NEURON_CC_FLAGS --lnc >= {launch_lnc}; inferred lnc={lnc}"
            )
    if launch_lnc not in (1, 2):
        raise ValueError(
            f"QWEN36_DELTANET_AUTOCP_LNC must be 1 or 2, got {launch_lnc}"
        )
    if num_chunks % launch_lnc != 0:
        raise ValueError(
            "QWEN36_DELTANET_AUTOCP_CTE requires the number of 128-token "
            f"chunks to be divisible by launch LNC; chunks={num_chunks}, "
            f"launch_lnc={launch_lnc}"
        )
    return launch_lnc


def _deltanet_autocp_affine_launch_spec(num_chunks: int, launch_lnc: int):
    """Return a SPMD affine launch grid, falling back to legacy LNC split.

    Bare ``kernel[2]`` launches only two logical cores. For AutoCP affine
    generation we need one independent program per 128-token chunk, sharded
    across those logical cores. NKI represents that as a SPMD grid dimension
    with an attached NC distribution.
    """
    if os.environ.get("QWEN36_DELTANET_AUTOCP_SPMD_AFFINE", "1") == "0":
        return launch_lnc
    import nki.language as _nl  # Imported lazily so CPU-only unit stubs still load.

    if not hasattr(_nl, "spmd_dim") or not hasattr(_nl, "nc"):
        return launch_lnc

    if launch_lnc == 2:
        return (_nl.spmd_dim(num_chunks, _nl.nc(2)), 1)
    return (num_chunks, 1)


def _resolve_deltanet_autocp_cp_chunks(num_chunks: int) -> int:
    cp_chunks = max(1, int(os.environ.get("QWEN36_DELTANET_AUTOCP_CP_CHUNKS", "4")))
    if num_chunks % cp_chunks != 0:
        raise ValueError(
            "QWEN36_DELTANET_COMPACT_AUTOCP_CTE requires the number of "
            "128-token chunks to be divisible by QWEN36_DELTANET_AUTOCP_CP_CHUNKS; "
            f"chunks={num_chunks}, cp_chunks={cp_chunks}"
        )
    return cp_chunks

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
)
try:
    from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
    HAS_MOE_V2 = True
except ImportError:
    HAS_MOE_V2 = False


def _patch_nxd_shard_hidden_kernel():
    """Monkey-patch NxD's `_call_shard_hidden_kernel` to use the nkilib
    blockwise-matmul-shard-on-H NKI kernel present in SDK 2.29's `nkilib`
    package. Upstream NxD gates the fastest MoE blockwise path on a
    `neuronxcc.nki._private.blockwise_mm` module which is absent from the
    DLAMI; falling through, it raises NotImplementedError. But the same
    kernel exists at `nkilib.experimental.moe.forward.bwmm_shard_on_H.
    blockwise_mm_baseline_shard_hidden` — we wire that up here so the
    default LNC=2 forward path becomes available.
    """
    try:
        from nkilib.experimental.moe.forward.bwmm_shard_on_H import (
            blockwise_mm_baseline_shard_hidden as _nki_shard_H,
        )
    except ImportError:
        return  # nothing to do; NxD's fallback handling stays in effect

    from neuronx_distributed.modules.moe import blockwise as _nxd_blockwise

    if getattr(_nxd_blockwise, "_qwen35_shard_hidden_patched", False):
        return

    def _call_shard_hidden_kernel_patched(args):
        """Drop-in replacement matching the NxD `_call_shard_hidden_kernel`
        signature but using the nkilib forward-only kernel.

        The nkilib kernel returns (output, gate_up_activations_T, down_activations)
        — no `output=` kwarg — so we match that layout.
        """
        result = _nki_shard_H[2](
            hidden_states=args.hidden_states,
            expert_affinities_masked=args.expert_affinities_masked,
            gate_up_proj_weight=args.gate_up_proj_weight,
            down_proj_weight=args.down_proj_weight,
            block_size=args.block_size,
            token_position_to_id=args.token_position_to_id.to(dtype=torch.int32),
            block_to_expert=args.block_to_expert.to(dtype=torch.int32),
            gate_up_activations_T=args.gate_up_activations_T,
            down_activations=args.down_activations,
            skip_dma=args.skip_dma,
            is_tensor_update_accumulating=args.is_tensor_update_accumulating,
            expert_affinities_scaling_mode=args.expert_affinities_scaling_mode,
        )
        if isinstance(result, tuple) and len(result) == 3:
            return result
        # Some kernel builds return only the output tensor; adapt.
        return result, args.gate_up_activations_T, args.down_activations

    _nxd_blockwise._call_shard_hidden_kernel = _call_shard_hidden_kernel_patched
    _nxd_blockwise._qwen35_shard_hidden_patched = True


_patch_nxd_shard_hidden_kernel()
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    DecoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    move_heads_front,
    transpose_parallel_linear_layer,
)
try:
    from neuronx_distributed_inference.modules.attention.utils import (
        preprocess_quantized_linear_layer,
    )
except (ImportError, AttributeError):
    def preprocess_quantized_linear_layer(layer):
        return layer

from neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager import (
    BlockKVCacheManager,
)
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

logger = logging.getLogger(__name__)

try:
    from neuronxcc.nki._pre_prod_kernels import (
        NormType as _QKVNormType,
        QKVOutputLayout as _QKVOutputLayout,
        QuantizationType as _QKVQuantizationType,
    )
    from neuronxcc.nki._pre_prod_kernels.qkv_tkg_impl import (
        nki_qkv_projection_tkg_impl as _qkv_tkg_nki_kernel,
    )
except ImportError:
    _QKVNormType = None
    _QKVOutputLayout = None
    _QKVQuantizationType = None
    _qkv_tkg_nki_kernel = None

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
        os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", _get_platform_target())
        _nkilib_flash_attn = _nki.jit(
            _raw_fn,
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


class GDNAPCReusePlan(NamedTuple):
    """Exact hybrid-APC reuse plan for attention KV plus GDN checkpoints."""

    attention_hit_len: int
    recurrent_hit_len: int
    conv_hit_len: int
    reusable_prefix_len: int
    restore_checkpoint_prefix_len: int
    residual_replay_len: int
    suffix_len: int


def _non_negative_len(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _normalize_hybrid_cache_dtype(name: str, value, default: str) -> str:
    if value is None:
        value = default
    if isinstance(value, torch.dtype):
        if value == torch.float32:
            return "float32"
        if value == torch.bfloat16:
            return "bfloat16"
    normalized = str(value).lower()
    aliases = {
        "fp32": "float32",
        "float32": "float32",
        "torch.float32": "float32",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "torch.bfloat16": "bfloat16",
    }
    if normalized not in aliases:
        raise ValueError(
            f"{name} must be one of fp32/float32 or bf16/bfloat16, got {value}"
        )
    return aliases[normalized]


def _torch_dtype_from_hybrid_cache_dtype(value: str) -> torch.dtype:
    value = _normalize_hybrid_cache_dtype("hybrid cache dtype", value, "bfloat16")
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    raise AssertionError(f"unexpected hybrid cache dtype {value}")


def plan_gdn_apc_reuse(
    *,
    attention_hit_len: int,
    recurrent_hit_len: int,
    conv_hit_len: int,
    request_prefix_len: int,
    gdn_checkpoint_interval: int,
) -> GDNAPCReusePlan:
    """Plan exact prefix reuse for Qwen hybrid APC.

    Attention KV can be reused up to the vLLM APC hit, but DeltaNet can only
    resume exactly from a boundary with both recurrent and conv checkpoint
    state. When the attention hit is inside a GDN interval, restore the nearest
    earlier checkpoint and replay the residual tokens before running the suffix.
    """
    attention_hit_len = _non_negative_len("attention_hit_len", attention_hit_len)
    recurrent_hit_len = _non_negative_len("recurrent_hit_len", recurrent_hit_len)
    conv_hit_len = _non_negative_len("conv_hit_len", conv_hit_len)
    request_prefix_len = _non_negative_len("request_prefix_len", request_prefix_len)
    gdn_checkpoint_interval = int(gdn_checkpoint_interval)
    if gdn_checkpoint_interval <= 0:
        raise ValueError(
            f"gdn_checkpoint_interval must be positive, got {gdn_checkpoint_interval}"
        )

    reusable_prefix_len = min(
        attention_hit_len,
        recurrent_hit_len,
        conv_hit_len,
        request_prefix_len,
    )
    restore_checkpoint_prefix_len = (
        reusable_prefix_len // gdn_checkpoint_interval
    ) * gdn_checkpoint_interval
    residual_replay_len = reusable_prefix_len - restore_checkpoint_prefix_len
    suffix_len = request_prefix_len - reusable_prefix_len

    return GDNAPCReusePlan(
        attention_hit_len=attention_hit_len,
        recurrent_hit_len=recurrent_hit_len,
        conv_hit_len=conv_hit_len,
        reusable_prefix_len=reusable_prefix_len,
        restore_checkpoint_prefix_len=restore_checkpoint_prefix_len,
        residual_replay_len=residual_replay_len,
        suffix_len=suffix_len,
    )


# ============================================================
# Gated DeltaNet Module (Linear Recurrent Attention)
# ============================================================


class NeuronGatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention for Neuron.

    Replaces standard attention for 48 of 64 layers in Qwen3.5/3.6-27B.
    Uses a chunk-based linear recurrence instead of KV cache.

    HF weight layout (27B dense -- scaled dimensions):
    - in_proj_qkv.weight: (key_dim*2 + value_dim, hidden_size) = (10240, 5120)
    - in_proj_z.weight: (value_dim, hidden_size) = (6144, 5120)
    - in_proj_a.weight: (num_v_heads, hidden_size) = (48, 5120)
    - in_proj_b.weight: (num_v_heads, hidden_size) = (48, 5120)
    - conv1d.weight: (conv_dim, 1, conv_kernel_size) = (10240, 1, 4)
    - A_log: (num_v_heads,) = (48,)
    - dt_bias: (num_v_heads,) = (48,)
    - norm.weight: (head_v_dim,) = (128,)
    - out_proj.weight: (hidden_size, value_dim) = (5120, 6144)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        tc = config

        self.hidden_size = tc.hidden_size  # 5120
        self.tp_degree = tc.neuron_config.tp_degree
        self.global_num_v_heads = tc.linear_num_value_heads  # 48
        self.global_num_k_heads = tc.linear_num_key_heads  # 16
        self.head_k_dim = tc.linear_key_head_dim  # 128
        self.head_v_dim = tc.linear_value_head_dim  # 128
        if self.global_num_v_heads % self.tp_degree != 0:
            raise ValueError(
                f"linear_num_value_heads={self.global_num_v_heads} must be divisible "
                f"by tp_degree={self.tp_degree}"
            )
        if self.global_num_k_heads % self.tp_degree != 0:
            raise ValueError(
                f"linear_num_key_heads={self.global_num_k_heads} must be divisible "
                f"by tp_degree={self.tp_degree}"
            )
        self.num_v_heads = self.global_num_v_heads // self.tp_degree
        self.num_k_heads = self.global_num_k_heads // self.tp_degree
        self.global_key_dim = self.head_k_dim * self.global_num_k_heads  # 2048
        self.global_value_dim = self.head_v_dim * self.global_num_v_heads  # 6144
        self.key_dim = self.head_k_dim * self.num_k_heads  # 512 at TP=4
        self.value_dim = self.head_v_dim * self.num_v_heads  # 1536 at TP=4
        self.conv_kernel_size = tc.linear_conv_kernel_dim  # 4
        self.layer_idx = layer_idx
        self.rms_norm_eps = tc.rms_norm_eps
        self.use_hybrid_cache_manager = getattr(tc, "use_hybrid_cache_manager", False)
        self.use_hybrid_apc_manager = getattr(tc, "use_hybrid_apc_manager", False)
        self.use_qwen_hybrid_chunked_prefill = getattr(
            tc, "use_qwen_hybrid_chunked_prefill", False
        )
        self.use_qwen_hybrid_chunked_prefill_nki = getattr(
            tc, "use_qwen_hybrid_chunked_prefill_nki", False
        )
        self.use_qwen_deltanet_decode_nki = getattr(
            tc, "use_qwen_deltanet_decode_nki", False
        )
        self.use_cold_zero_conv_fast_path = getattr(
            tc, "use_cold_zero_conv_fast_path", False
        )

        # KV cache dummy shape info
        self.head_dim = tc.head_dim  # 256
        tp_degree = tc.neuron_config.tp_degree
        raw_kv_heads = tc.num_key_value_heads
        if raw_kv_heads < tp_degree:
            replicated_kv_heads = tp_degree
        else:
            replicated_kv_heads = raw_kv_heads
        self.kv_heads_per_rank = replicated_kv_heads // tp_degree

        # Conv1d on concatenated QKV (NOT Z).  Store the depthwise kernel in a
        # ColumnParallelLinear parameter container so NxD's checkpoint sharder
        # can split it by output channel.  Forward still uses it as Conv1d
        # weight after unsqueezing the singleton input-channel dimension.
        self.global_conv_dim = self.global_key_dim * 2 + self.global_value_dim  # 10240
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 2560 at TP=4
        self.conv1d_weight = ColumnParallelLinear(
            self.conv_kernel_size,
            self.global_conv_dim,
            bias=False,
            gather_output=False,
        )

        # Input/output projections are the large DeltaNet tensors.  Shard them
        # with tensor parallelism; convert_qwen35_hf_to_neuron_state_dict()
        # reorders in_proj_qkv into per-rank [Q_local | K_local | V_local]
        # blocks before NxD slices the output dimension.
        self.in_proj_qkv = ColumnParallelLinear(
            self.hidden_size,
            self.global_key_dim * 2 + self.global_value_dim,
            bias=False,
            gather_output=False,
        )
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size,
            self.global_value_dim,
            bias=False,
            gather_output=False,
        )
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )

        # Same parameter-container pattern for per-value-head decay vectors.
        # These are used as vectors in forward but sharded by output dim during
        # checkpoint conversion/loading.
        self.dt_bias_weight = ColumnParallelLinear(
            1,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )
        self.A_log_weight = ColumnParallelLinear(
            1,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )

        # Output norm and projection
        self.norm = Qwen3MoeRMSNorm(self.head_v_dim, eps=self.rms_norm_eps)
        self.out_proj = RowParallelLinear(
            self.global_value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        # State buffers for CTE -> TKG carry-over
        alloc_batch_size = getattr(config.neuron_config, "max_batch_size", 1)
        self._phase_batch_size = getattr(config.neuron_config, "batch_size", 1)
        recurrent_buffer_dtype = (
            _torch_dtype_from_hybrid_cache_dtype(config.hybrid_recurrent_cache_dtype)
            if self.use_hybrid_apc_manager
            else config.neuron_config.torch_dtype
        )
        conv_buffer_dtype = (
            _torch_dtype_from_hybrid_cache_dtype(config.hybrid_conv_cache_dtype)
            if self.use_hybrid_apc_manager
            else config.neuron_config.torch_dtype
        )
        self.recurrent_state_buffer = nn.Parameter(
            torch.zeros(
                alloc_batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=recurrent_buffer_dtype,
            ),
            requires_grad=False,
        )
        self.conv_state_buffer = nn.Parameter(
            torch.zeros(
                alloc_batch_size,
                self.conv_dim,
                self.conv_kernel_size - 1,
                dtype=conv_buffer_dtype,
            ),
            requires_grad=False,
        )

    def _conv1d_weight(self):
        return self.conv1d_weight.weight.unsqueeze(1)

    def _dt_bias(self):
        return self.dt_bias_weight.weight.squeeze(1)

    def _A_log(self):
        return self.A_log_weight.weight.squeeze(1)

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

    def _nki_recurrent_step(self, query, key, value, g, beta, recurrent_state):
        """Single-step recurrent update using the stateful NKI decode kernel."""
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        BH = B * H
        query_flat = query.reshape(BH, S, k_dim)[:, 0, :].contiguous()
        key_flat = key.reshape(BH, S, k_dim)[:, 0, :].contiguous()
        value_flat = value.reshape(BH, S, v_dim)[:, 0, :].contiguous()
        g_flat = g.reshape(BH, S)[:, 0:1].contiguous()
        beta_flat = beta.reshape(BH, S)[:, 0:1].contiguous()
        state_flat = recurrent_state.reshape(BH * k_dim, v_dim).contiguous()

        output_flat, state_flat_out = _deltanet_nki_step_batched(
            query_flat,
            key_flat,
            value_flat,
            g_flat,
            beta_flat,
            state_flat,
        )

        output = output_flat.reshape(B, H, S, v_dim)
        new_state = state_flat_out.reshape(B, H, k_dim, v_dim)

        return output, new_state

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
        self, query, key, value, g, beta, output_final_state=False, initial_state=None
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

        initial_state_flat = None
        if initial_state is not None:
            initial_state_flat = initial_state.reshape(BH, k_dim, v_dim).float().contiguous()

        all_outputs = []
        all_states = []
        for bh in range(BH):
            if initial_state_flat is None:
                state = torch.zeros(k_dim, v_dim, dtype=torch.float32, device=device)
            else:
                state = initial_state_flat[bh]

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
        self,
        query,
        key,
        value,
        g,
        beta,
        output_final_state=False,
        initial_state=None,
        _segment_disabled=False,
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

        initial_state is the restored GDN recurrent checkpoint for warm or
        partial-prefix suffix prefill. Cold prefill passes zeros.
        """
        chunk_size = int(os.environ.get("QWEN36_DELTANET_CHUNK_SIZE", "128"))
        if chunk_size not in (64, 128):
            raise ValueError(
                "QWEN36_DELTANET_CHUNK_SIZE must be 64 or 128 for fused CTE; "
                f"got {chunk_size}"
            )

        cte_impl = os.environ.get("QWEN36_DELTANET_CTE_IMPL", "current").lower()
        if cte_impl in ("legacy", "legacy_direct", "direct"):
            use_legacy_direct_cte = True
        elif cte_impl in ("current", "optimized"):
            use_legacy_direct_cte = False
        else:
            raise ValueError(
                "QWEN36_DELTANET_CTE_IMPL must be current or legacy_direct; "
                f"got {cte_impl!r}"
            )
        if use_legacy_direct_cte and chunk_size != 128:
            raise ValueError(
                "QWEN36_DELTANET_CTE_IMPL=legacy_direct requires "
                f"QWEN36_DELTANET_CHUNK_SIZE=128; got {chunk_size}"
            )

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]

        # Pad sequence to multiple of chunk_size
        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size

        segment_tokens = int(
            os.environ.get("QWEN36_DELTANET_FUSED_SEGMENT_TOKENS", "0") or "0"
        )
        if (
            not _segment_disabled
            and segment_tokens > 0
            and total_seq_len > segment_tokens
        ):
            if segment_tokens < chunk_size or segment_tokens % chunk_size != 0:
                raise ValueError(
                    "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS must be a positive "
                    "multiple of QWEN36_DELTANET_CHUNK_SIZE; "
                    f"got segment_tokens={segment_tokens}, chunk_size={chunk_size}"
                )
            segment_outputs = []
            state = initial_state
            for start in range(0, total_seq_len, segment_tokens):
                end = min(start + segment_tokens, total_seq_len)
                segment_output, state = self._fused_chunked_forward(
                    query[:, :, start:end, :],
                    key[:, :, start:end, :],
                    value[:, :, start:end, :],
                    g[:, :, start:end],
                    beta[:, :, start:end],
                    output_final_state=True,
                    initial_state=state,
                    _segment_disabled=True,
                )
                segment_outputs.append(segment_output)
            output = torch.cat(segment_outputs, dim=2)[:, :, :S, :]
            return output, state if output_final_state else None

        if use_legacy_direct_cte:
            query = l2norm(query, dim=-1)
            key = l2norm(key, dim=-1)
            query = query * (1.0 / (k_dim ** 0.5))

        BH = B * H
        # Flatten to (BH, S, dim). Grouped multihead launches are opt-in
        # because isolated validation must pass before using them in artifacts.
        query_flat = query.reshape(BH, total_seq_len, k_dim).contiguous()
        key_flat = key.reshape(BH, total_seq_len, k_dim).contiguous()
        value_flat = value.reshape(BH, total_seq_len, v_dim).contiguous()

        # g and beta: (BH, S) -> (BH, S, 1) for the kernel's (S, 1) input layout
        g_flat = g.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()
        beta_flat = beta.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()
        if initial_state is None:
            initial_state_flat = torch.zeros(
                BH, k_dim, v_dim, dtype=torch.float32, device=query.device
            )
        else:
            initial_state_flat = initial_state.reshape(BH, k_dim, v_dim).float().contiguous()

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

        use_multihead_cte = (
            not use_legacy_direct_cte
            and os.environ.get("QWEN36_DELTANET_MULTIHEAD_CTE", "1") != "0"
        )
        if use_multihead_cte:
            pair_outputs = []
            pair_states = []
            head_group_size = _resolve_deltanet_multihead_group_size(BH)
            for bh_start in range(0, BH, head_group_size):
                bh_end = min(bh_start + head_group_size, BH)
                launch_heads = bh_end - bh_start
                launch_spec = _deltanet_multihead_launch_spec(launch_heads)
                out_pair, state_pair = _deltanet_fused_multihead_kernel[launch_spec](
                    query_flat[bh_start:bh_end],  # (G, S, 128)
                    key_flat[bh_start:bh_end],  # (G, S, 128)
                    value_flat[bh_start:bh_end],  # (G, S, 128)
                    g_flat[bh_start:bh_end],  # (G, S, 1) — RAW g, not cumsum
                    beta_flat[bh_start:bh_end],  # (G, S, 1) — sigmoid(b)
                    initial_state_flat[bh_start:bh_end],
                    lower_mask,  # (128, 128)
                    identity_mat,  # (128, 128)
                    lower_mask_diag,  # (128, 128)
                )
                pair_outputs.append(out_pair)
                pair_states.append(state_pair)

            output = torch.cat(pair_outputs, dim=0)
            final_state = torch.cat(pair_states, dim=0)
        else:
            fused_singlehead_kernel = (
                _deltanet_fused_legacy_direct_kernel
                if use_legacy_direct_cte
                else _deltanet_fused_kernel
            )
            all_outputs = []
            all_states = []
            for bh in range(BH):
                out_bh, state_bh = fused_singlehead_kernel(
                    query_flat[bh],  # (S, 128)
                    key_flat[bh],  # (S, 128)
                    value_flat[bh],  # (S, 128)
                    g_flat[bh],  # (S, 1) — RAW g, not cumsum
                    beta_flat[bh],  # (S, 1) — sigmoid(b)
                    initial_state_flat[bh],  # (128, 128) recurrent checkpoint
                    lower_mask,  # (128, 128)
                    identity_mat,  # (128, 128)
                    lower_mask_diag,  # (128, 128)
                )
                all_outputs.append(out_bh)
                all_states.append(state_bh)

            output = torch.stack(all_outputs, dim=0)
            final_state = torch.stack(all_states, dim=0)

        output = output.reshape(B, H, total_seq_len, v_dim)
        output = output[:, :, :S]

        if output_final_state:
            last_recurrent_state = final_state.reshape(B, H, k_dim, v_dim)
        else:
            last_recurrent_state = None

        return output, last_recurrent_state

    def _compact_autocp_chunked_forward(
        self, query, key, value, g, beta, output_final_state=False, initial_state=None
    ):
        """Compact AutoCP CTE probe: prefix segment state summaries, replay segments.

        Compared with ``_autocp_chunked_forward``, this avoids materializing
        per-chunk output-affine tensors. It is intentionally opt-in because the
        first version reuses the existing recurrent fused kernel for segment
        replay; a later NKI replay kernel can collapse the segment loop.
        """
        chunk_size = 128

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        if k_dim != 128 or v_dim != 128:
            raise ValueError(
                "QWEN36_DELTANET_COMPACT_AUTOCP_CTE requires 128-wide "
                f"key/value heads; got k_dim={k_dim}, v_dim={v_dim}"
            )

        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size
        num_chunks = total_seq_len // chunk_size
        if num_chunks <= 0:
            raise ValueError("QWEN36_DELTANET_COMPACT_AUTOCP_CTE requires chunks")
        cp_chunks = _resolve_deltanet_autocp_cp_chunks(num_chunks)
        num_segments = num_chunks // cp_chunks
        launch_lnc = _resolve_deltanet_autocp_lnc(num_segments)
        summary_launch_spec = _deltanet_autocp_affine_launch_spec(
            num_segments,
            launch_lnc,
        )

        BH = B * H
        query_flat = query.reshape(BH, total_seq_len, k_dim).contiguous()
        key_flat = key.reshape(BH, total_seq_len, k_dim).contiguous()
        value_flat = value.reshape(BH, total_seq_len, v_dim).contiguous()
        g_flat = g.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()
        beta_flat = beta.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()
        if initial_state is None:
            initial_state_flat = torch.zeros(
                BH, k_dim, v_dim, dtype=torch.float32, device=query.device
            )
        else:
            initial_state_flat = initial_state.reshape(BH, k_dim, v_dim).float().contiguous()

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

        segment_len = cp_chunks * chunk_size
        all_outputs = []
        all_states = []
        for bh in range(BH):
            segment_matrix, segment_bias = (
                _deltanet_autocp_state_summary_sequence[summary_launch_spec](
                    key_flat[bh],
                    value_flat[bh],
                    g_flat[bh],
                    beta_flat[bh],
                    lower_mask,
                    identity_mat,
                )
            )
            segment_states, final_state = _deltanet_autocp_state_prefix(
                segment_matrix,
                segment_bias,
                initial_state_flat[bh],
            )

            q_segments = query_flat[bh].reshape(num_segments, segment_len, k_dim).contiguous()
            k_segments = key_flat[bh].reshape(num_segments, segment_len, k_dim).contiguous()
            v_segments = value_flat[bh].reshape(num_segments, segment_len, v_dim).contiguous()
            g_segments = g_flat[bh].reshape(num_segments, segment_len, 1).contiguous()
            beta_segments = beta_flat[bh].reshape(num_segments, segment_len, 1).contiguous()

            replay_group_size = _resolve_deltanet_multihead_group_size(num_segments)
            replay_outputs = []
            for segment_start in range(0, num_segments, replay_group_size):
                segment_end = min(segment_start + replay_group_size, num_segments)
                launch_segments = segment_end - segment_start
                replay_launch_spec = _deltanet_multihead_launch_spec(launch_segments)
                out_group, _ = _deltanet_fused_multihead_kernel[replay_launch_spec](
                    q_segments[segment_start:segment_end],
                    k_segments[segment_start:segment_end],
                    v_segments[segment_start:segment_end],
                    g_segments[segment_start:segment_end],
                    beta_segments[segment_start:segment_end],
                    segment_states[segment_start:segment_end],
                    lower_mask,
                    identity_mat,
                    lower_mask_diag,
                )
                replay_outputs.append(out_group)
            out_segments = torch.cat(replay_outputs, dim=0)

            all_outputs.append(out_segments.reshape(total_seq_len, v_dim))
            all_states.append(final_state)

        output = torch.stack(all_outputs, dim=0)
        output = output.reshape(B, H, total_seq_len, v_dim)
        output = output[:, :, :S]

        if output_final_state:
            final_state = torch.stack(all_states, dim=0)
            last_recurrent_state = final_state.reshape(B, H, k_dim, v_dim)
        else:
            last_recurrent_state = None

        return output, last_recurrent_state

    def _autocp_chunked_forward(
        self, query, key, value, g, beta, output_final_state=False, initial_state=None
    ):
        """FlashQLA-style AutoCP CTE path for exact GDN prefill probes.

        This path decomposes each 128-token chunk into an affine state transform,
        scans chunk states, then applies the per-chunk initial state to outputs.
        It is gated by QWEN36_DELTANET_AUTOCP_CTE while we measure whether the
        extra custom-call/HBM traffic beats the recurrent fused path.
        """
        chunk_size = 128

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        if k_dim != 128 or v_dim != 128:
            raise ValueError(
                "QWEN36_DELTANET_AUTOCP_CTE requires 128-wide key/value heads; "
                f"got k_dim={k_dim}, v_dim={v_dim}"
            )

        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size
        num_chunks = total_seq_len // chunk_size
        if num_chunks <= 0:
            raise ValueError("QWEN36_DELTANET_AUTOCP_CTE requires at least one chunk")
        launch_lnc = _resolve_deltanet_autocp_lnc(num_chunks)

        BH = B * H
        query_flat = query.reshape(BH, total_seq_len, k_dim).contiguous()
        key_flat = key.reshape(BH, total_seq_len, k_dim).contiguous()
        value_flat = value.reshape(BH, total_seq_len, v_dim).contiguous()
        g_flat = g.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()
        beta_flat = beta.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()
        if initial_state is None:
            initial_state_flat = torch.zeros(
                BH, k_dim, v_dim, dtype=torch.float32, device=query.device
            )
        else:
            initial_state_flat = initial_state.reshape(BH, k_dim, v_dim).float().contiguous()

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
        affine_launch_spec = _deltanet_autocp_affine_launch_spec(
            num_chunks,
            launch_lnc,
        )

        all_outputs = []
        all_states = []
        for bh in range(BH):
            output_base, output_state, state_matrix, state_bias = (
                _deltanet_autocp_affine_sequence[affine_launch_spec](
                    query_flat[bh],
                    key_flat[bh],
                    value_flat[bh],
                    g_flat[bh],
                    beta_flat[bh],
                    lower_mask,
                    identity_mat,
                    lower_mask_diag,
                )
            )
            if os.environ.get("QWEN36_DELTANET_AUTOCP_SPLIT_APPLY") == "1":
                chunk_states, final_state = _deltanet_autocp_state_prefix(
                    state_matrix,
                    state_bias,
                    initial_state_flat[bh],
                )
                out_bh = _deltanet_autocp_apply_output(
                    output_base,
                    output_state,
                    chunk_states,
                )
            else:
                out_bh, final_state = _deltanet_autocp_prefix_apply_output(
                    output_base,
                    output_state,
                    state_matrix,
                    state_bias,
                    initial_state_flat[bh],
                )
            all_outputs.append(out_bh)
            all_states.append(final_state)

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

    def _chunk_forward(
        self, query, key, value, g, beta, output_final_state=False, initial_state=None
    ):
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

        if initial_state is None:
            last_recurrent_state = torch.zeros(
                B, H, k_dim, v_dim, dtype=query.dtype, device=query.device
            )
        else:
            last_recurrent_state = initial_state.to(dtype=query.dtype)
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
        is_for_context_encoding = bool(kwargs.get("is_for_context_encoding", False))
        qwen_chunked_prefill_active = (
            self.use_qwen_hybrid_chunked_prefill
            and past_key_value is not None
            and seq_len > 1
        )
        is_decode = (
            past_key_value is not None
            and not qwen_chunked_prefill_active
            and not is_for_context_encoding
        )

        # Padding mask for DeltaNet: [B, S, 1] with 1.0 for real tokens, 0.0 for padding.
        # Passed from get_model_output where it's computed from input_ids != pad_token_id.
        # Embeddings are already zeroed for padding tokens; this mask additionally
        # zeros the decay gate so the recurrent state is preserved unchanged
        # through padding positions (no spurious decay).
        valid_mask_1d = kwargs.get("deltanet_padding_mask", None)  # [B, S, 1] or None
        static_hybrid_cache_active = self.use_hybrid_cache_manager
        recurrent_state_cache = None
        conv_state_cache = None
        if static_hybrid_cache_active and past_key_value is not None:
            recurrent_state_cache, conv_state_cache = past_key_value
        elif (
            self.use_hybrid_apc_manager
            and past_key_value is not None
            and len(past_key_value) == 2
            and getattr(past_key_value[0], "dim", lambda: 0)() == 4
            and getattr(past_key_value[1], "dim", lambda: 0)() == 3
            and past_key_value[0].shape[1:] == self.recurrent_state_buffer.shape[1:]
            and past_key_value[1].shape[1:] == self.conv_state_buffer.shape[1:]
        ):
            recurrent_state_cache, conv_state_cache = past_key_value

        # Project inputs
        deltanet_fp32 = os.environ.get("DELTANET_FP32") == "1"
        if deltanet_fp32 and isinstance(self.in_proj_qkv, nn.Linear):
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
            if conv_state_cache is not None:
                conv_state = conv_state_cache[:batch_size]
            elif seq_ids is not None:
                conv_state = torch.index_select(self.conv_state_buffer, 0, seq_ids)
            else:
                conv_state = self.conv_state_buffer[:batch_size]
            conv_input = torch.cat([conv_state, mixed], dim=-1)

            w = self._conv1d_weight().squeeze(1)
            if seq_len == 1:
                conv_out = (
                    conv_input[:, :, : self.conv_kernel_size] * w.unsqueeze(0)
                ).sum(dim=-1, keepdim=True)
            else:
                conv_out = torch.zeros_like(mixed)
                for k in range(self.conv_kernel_size):
                    conv_out = (
                        conv_out
                        + w[:, k].unsqueeze(0).unsqueeze(-1)
                        * conv_input[:, :, k : k + 1]
                    )
            mixed_post_conv = F.silu(conv_out)

            new_conv_state = torch.cat([conv_state[:, :, 1:], mixed], dim=-1)
            expected_state_len = self.conv_state_buffer.shape[-1]
            if new_conv_state.shape[-1] != expected_state_len:
                if new_conv_state.shape[-1] > expected_state_len:
                    new_conv_state = new_conv_state[:, :, -expected_state_len:]
                else:
                    new_conv_state = F.pad(
                        new_conv_state,
                        (expected_state_len - new_conv_state.shape[-1], 0),
                    )
            alloc_bs = self.conv_state_buffer.shape[0]
            if static_hybrid_cache_active:
                new_conv_state = new_conv_state.to(self.conv_state_buffer.dtype)
            elif seq_ids is not None:
                new_conv_state = _qwen36_update_state_rows_by_seq_ids(
                    self.conv_state_buffer,
                    new_conv_state.to(self.conv_state_buffer.dtype),
                    seq_ids,
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
            if (
                conv_state_cache is not None
                and (qwen_chunked_prefill_active or is_for_context_encoding)
            ):
                cold_prefill_from_zero = self.use_cold_zero_conv_fast_path
                if cold_prefill_from_zero:
                    mixed_post_conv = F.silu(
                        F.conv1d(
                            mixed,
                            self._conv1d_weight(),
                            bias=None,
                            padding=self.conv_kernel_size - 1,
                            groups=self.conv_dim,
                        )[:, :, :seq_len]
                    )
                    state_source = mixed
                else:
                    conv_state = conv_state_cache[:batch_size]
                    if position_ids is not None:
                        reset_mask = (position_ids[:, :1].long() == 0).to(
                            dtype=conv_state.dtype, device=conv_state.device
                        )
                        conv_state = conv_state * (
                            1.0 - reset_mask[:, :, None]
                        )
                    conv_input = torch.cat([conv_state, mixed], dim=-1)
                    w = self._conv1d_weight().squeeze(1)
                    conv_out = torch.zeros_like(mixed)
                    for k in range(self.conv_kernel_size):
                        conv_out = (
                            conv_out
                            + w[:, k].unsqueeze(0).unsqueeze(-1)
                            * conv_input[:, :, k : k + seq_len]
                        )
                    mixed_post_conv = F.silu(conv_out)
                    state_source = conv_input

                state_len = self.conv_kernel_size - 1
                if valid_mask_1d is not None:
                    num_valid = valid_mask_1d.squeeze(-1).sum(dim=-1, keepdim=True).long()
                    idx_base = (state_source.shape[-1] - seq_len + num_valid - state_len).clamp(min=0)
                    offsets = torch.arange(state_len, device=mixed.device).unsqueeze(0)
                    gather_idx = idx_base + offsets
                    gather_idx = gather_idx.unsqueeze(1).expand(-1, self.conv_dim, -1)
                    new_conv_state = torch.gather(state_source, 2, gather_idx)
                else:
                    new_conv_state = state_source[:, :, -state_len:].contiguous()
            else:
                mixed_post_conv = F.silu(
                    F.conv1d(
                        mixed,
                        self._conv1d_weight(),
                        bias=None,
                        padding=self.conv_kernel_size - 1,
                        groups=self.conv_dim,
                    )[:, :, :seq_len]
                )

                if valid_mask_1d is not None:
                    # valid_mask_1d is [B, S, 1]; count valid tokens per batch
                    state_len = self.conv_kernel_size - 1
                    num_valid = (
                        valid_mask_1d.squeeze(-1).sum(dim=-1, keepdim=True).long()
                    )  # [B, 1]
                    idx_base = num_valid - state_len
                    idx_base = idx_base.clamp(min=0)
                    offsets = torch.arange(state_len, device=mixed.device).unsqueeze(0)
                    gather_idx = idx_base + offsets  # [B, state_len]
                    gather_idx = gather_idx.unsqueeze(1).expand(-1, self.conv_dim, -1)
                    new_conv_state = torch.gather(mixed, 2, gather_idx)
                else:
                    new_conv_state = mixed[:, :, -self.conv_kernel_size + 1 :].contiguous()

            alloc_bs = self.conv_state_buffer.shape[0]
            if static_hybrid_cache_active:
                new_conv_state = new_conv_state.to(self.conv_state_buffer.dtype)
            elif seq_ids is not None:
                new_conv_state = _qwen36_update_state_rows_by_seq_ids(
                    self.conv_state_buffer,
                    new_conv_state.to(self.conv_state_buffer.dtype),
                    seq_ids,
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
        g = -self._A_log().float().exp() * F.softplus(a.float() + self._dt_bias())

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
            if recurrent_state_cache is not None:
                recurrent_state = recurrent_state_cache[:batch_size]
            elif seq_ids is not None:
                recurrent_state = torch.index_select(
                    self.recurrent_state_buffer, 0, seq_ids
                )
            else:
                recurrent_state = self.recurrent_state_buffer[:batch_size]

            use_nki_decode = (
                self.use_qwen_deltanet_decode_nki
                or os.environ.get("USE_NKI_DECODE") == "1"
            )
            if use_nki_decode and seq_len == 1:
                output, new_state = self._nki_recurrent_step(
                    query, key, value, g, beta, recurrent_state
                )
            else:
                output, new_state = self._recurrent_step(
                    query, key, value, g, beta, recurrent_state.float()
                )
            new_state_bf16 = new_state.to(self.recurrent_state_buffer.dtype)
            alloc_bs = self.recurrent_state_buffer.shape[0]
            if static_hybrid_cache_active:
                new_rec_state = new_state_bf16
            elif seq_ids is not None:
                new_rec_state = _qwen36_update_state_rows_by_seq_ids(
                    self.recurrent_state_buffer,
                    new_state_bf16,
                    seq_ids,
                )
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
            use_autocp_cte = os.environ.get("QWEN36_DELTANET_AUTOCP_CTE") == "1"
            use_compact_autocp_cte = (
                os.environ.get("QWEN36_DELTANET_COMPACT_AUTOCP_CTE") == "1"
            )

            if recurrent_state_cache is not None and (
                qwen_chunked_prefill_active or is_for_context_encoding
            ):
                initial_state = recurrent_state_cache[:batch_size].float()
                if position_ids is not None:
                    reset_mask = (position_ids[:, :1].long() == 0).to(
                        dtype=initial_state.dtype, device=initial_state.device
                    )
                    initial_state = initial_state * (1.0 - reset_mask[:, :, None, None])
                if use_autocp_cte and use_compact_autocp_cte:
                    output, final_state = self._compact_autocp_chunked_forward(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        output_final_state=True,
                        initial_state=initial_state,
                    )
                elif use_autocp_cte:
                    output, final_state = self._autocp_chunked_forward(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        output_final_state=True,
                        initial_state=initial_state,
                    )
                elif use_nki_chunked or (
                    self.use_qwen_hybrid_chunked_prefill_nki
                    and os.environ.get("USE_NKI_FUSED", "1") == "0"
                ):
                    output, final_state = self._nki_chunked_forward(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        output_final_state=True,
                        initial_state=initial_state,
                    )
                elif use_pytorch_chunk:
                    output, final_state = self._chunk_forward(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        output_final_state=True,
                        initial_state=initial_state,
                    )
                else:
                    output, final_state = self._fused_chunked_forward(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        output_final_state=True,
                        initial_state=initial_state,
                    )
            elif use_pytorch_chunk:
                output, final_state = self._chunk_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_autocp_cte and use_compact_autocp_cte:
                output, final_state = self._compact_autocp_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_autocp_cte:
                output, final_state = self._autocp_chunked_forward(
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
                if static_hybrid_cache_active:
                    new_rec_state = final_state_bf16
                elif seq_ids is not None:
                    new_rec_state = _qwen36_update_state_rows_by_seq_ids(
                        self.recurrent_state_buffer,
                        final_state_bf16,
                        seq_ids,
                    )
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

        if (
            is_for_context_encoding
            and not static_hybrid_cache_active
            and valid_mask_1d is not None
            and hasattr(valid_mask_1d, "numel")
            and valid_mask_1d.numel() > 0
        ):
            active_rows = _qwen36_active_state_rows(valid_mask_1d, seq_ids)
            new_conv_state = _qwen36_preserve_inactive_state_rows(
                new_conv_state,
                self.conv_state_buffer,
                active_rows,
            )
            new_rec_state = _qwen36_preserve_inactive_state_rows(
                new_rec_state,
                self.recurrent_state_buffer,
                active_rows,
            )

        # Output: norm, gate, project
        output = output.to(hidden_states.dtype)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = self.norm(output)
        z_gate = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = output * F.silu(z_gate)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(output)

        if static_hybrid_cache_active:
            return output, (new_rec_state, new_conv_state), new_rec_state, new_conv_state

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
    """Config for Qwen3.5/3.6-27B (dense) with hybrid DeltaNet + Attention."""

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Qwen35InferenceConfig":
        """Load Qwen3.5/Qwen3.6 text config from a pretrained model directory.

        Qwen3.6 stores the decoder settings under the top-level multimodal
        `text_config`. NxDI's text-only inference config expects those fields
        flattened onto the inference config itself.
        """
        neuron_config = kwargs.pop("neuron_config", None)
        if neuron_config is None:
            neuron_config = NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=128,
                torch_dtype=torch.bfloat16,
                save_sharded_checkpoint=True,
            )

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)

        text_config = config_dict.get("text_config", config_dict)
        rope_parameters = text_config.get("rope_parameters") or {}
        inference_config = dict(text_config)
        inference_config.setdefault("_name_or_path", model_path)
        # Preserve original model_type (may be qwen3_5_moe_text for MoE variants).
        inference_config.setdefault("model_type", text_config.get("model_type", "qwen3_5_text"))
        inference_config.setdefault("architectures", config_dict.get("architectures", []))
        inference_config.setdefault("tie_word_embeddings", config_dict.get("tie_word_embeddings", False))
        # For MoE variants: expose intermediate_size (needed by initialize_moe_module).
        # HF stores moe_intermediate_size and shared_expert_intermediate_size; we use
        # moe_intermediate_size as the per-expert MLP dim.
        if "moe_intermediate_size" in text_config and "intermediate_size" not in inference_config:
            inference_config["intermediate_size"] = text_config["moe_intermediate_size"]
        if "rope_theta" not in inference_config and "rope_theta" in rope_parameters:
            inference_config["rope_theta"] = rope_parameters["rope_theta"]
        if (
            "partial_rotary_factor" not in inference_config
            and "partial_rotary_factor" in rope_parameters
        ):
            inference_config["partial_rotary_factor"] = rope_parameters[
                "partial_rotary_factor"
            ]
        inference_config.update(kwargs)
        return cls(neuron_config=neuron_config, **inference_config)

    def __init__(self, *args, **kwargs):
        # Set defaults BEFORE super().__init__() because it calls validate_config()
        # which checks get_required_attributes(). These can be overridden by
        # kwargs or load_config.

        # ── MoE detection & defaults ──────────────────────────────────────────
        # Qwen3.5-35B-A3B (model_type=qwen3_5_moe_text) has:
        #   num_experts, num_experts_per_tok, moe_intermediate_size,
        #   shared_expert_intermediate_size, router_aux_loss_coef.
        # For those variants we expose the MoE fields that initialize_moe_module()
        # requires: num_local_experts, n_shared_experts, and use
        # moe_intermediate_size as the routed-experts intermediate dim.
        self._is_moe = str(kwargs.get("model_type", "")) == "qwen3_5_moe_text" or (
            "num_experts" in kwargs and "moe_intermediate_size" in kwargs
        )
        if self._is_moe:
            kwargs.setdefault("num_local_experts", kwargs.get("num_experts", 256))
            # Qwen3.5-MoE has a single shared expert with a per-token sigmoid gate.
            kwargs.setdefault("n_shared_experts", 1)
            # Route the moe_intermediate_size into the "intermediate_size" field
            # expected by NxDI's initialize_moe_module (which sizes the routed
            # experts). Shared expert uses shared_expert_intermediate_size, which
            # for A3B equals moe_intermediate_size (both 512).
            if "intermediate_size" not in kwargs and "moe_intermediate_size" in kwargs:
                kwargs["intermediate_size"] = kwargs["moe_intermediate_size"]
            kwargs.setdefault(
                "shared_expert_intermediate_size",
                kwargs.get("moe_intermediate_size", kwargs["intermediate_size"]),
            )

        # Layer types for hybrid dispatch: [3 DeltaNet + 1 GQA] repeated.
        if "layer_types" not in kwargs and not any(
            hasattr(a, "layer_types") for a in args if hasattr(a, "__dict__")
        ):
            num_layers = kwargs.get("num_hidden_layers", 64)
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
        kwargs.setdefault("linear_num_value_heads", 48)
        kwargs.setdefault("linear_num_key_heads", 16)
        kwargs.setdefault("linear_key_head_dim", 128)
        kwargs.setdefault("linear_value_head_dim", 128)
        kwargs.setdefault("linear_conv_kernel_dim", 4)
        kwargs.setdefault("use_hybrid_cache_manager", False)
        kwargs.setdefault("use_hybrid_apc_manager", False)
        kwargs.setdefault("use_qwen_hybrid_chunked_prefill", False)
        kwargs.setdefault("use_qwen_hybrid_chunked_prefill_nki", False)
        kwargs.setdefault("use_qwen_deltanet_decode_nki", False)
        kwargs.setdefault("gdn_checkpoint_interval", 256)
        kwargs.setdefault("max_gdn_checkpoint_slots", 8)
        kwargs.setdefault("hybrid_apc_layout_version", 1)
        kwargs.setdefault("hybrid_apc_allow_residual_replay", False)
        kwargs.setdefault("hybrid_apc_cache_salt", None)
        use_hybrid_apc_manager = bool(kwargs.get("use_hybrid_apc_manager", False))
        kwargs.setdefault(
            "hybrid_apc_require_vllm_metadata", use_hybrid_apc_manager
        )
        kwargs.setdefault(
            "hybrid_apc_allow_local_hash_fallback", not use_hybrid_apc_manager
        )
        kwargs.setdefault(
            "hybrid_apc_require_attention_block_refs", use_hybrid_apc_manager
        )
        kwargs.setdefault("hybrid_apc_reject_unbacked_attention_hits", True)
        kwargs.setdefault("hybrid_apc_disable_unbacked_prefix_reads", False)
        kwargs.setdefault("hybrid_apc_enable_backed_prefix_reads", False)
        kwargs.setdefault(
            "hybrid_apc_model_revision",
            kwargs.get("_name_or_path", kwargs.get("model_revision", "unknown")),
        )
        kwargs.setdefault(
            "hybrid_recurrent_cache_dtype",
            kwargs.get("gdn_recurrent_cache_dtype", "float32"),
        )
        kwargs.setdefault(
            "hybrid_conv_cache_dtype",
            kwargs.get("gdn_conv_cache_dtype", "bfloat16"),
        )
        kwargs.setdefault(
            "gdn_recurrent_cache_dtype", kwargs["hybrid_recurrent_cache_dtype"]
        )
        kwargs.setdefault("gdn_conv_cache_dtype", kwargs["hybrid_conv_cache_dtype"])
        kwargs.setdefault("hybrid_cache_mode", "all")
        kwargs.setdefault(
            "hybrid_cache_prefix_boundary_only",
            kwargs.get("hybrid_cache_block_boundary_only", True),
        )
        kwargs.setdefault(
            "hybrid_cache_block_boundary_only",
            kwargs["hybrid_cache_prefix_boundary_only"],
        )
        kwargs.setdefault("hybrid_cache_validate_exact", False)
        kwargs.setdefault("use_text_only_cte_inputs", True)
        kwargs.setdefault("use_compact_cte_attention_mask", True)
        kwargs.setdefault("use_cold_zero_conv_fast_path", False)
        kwargs.setdefault("disable_token_generation_wlo", False)

        super().__init__(*args, **kwargs)

        self.gdn_checkpoint_interval = int(self.gdn_checkpoint_interval)
        if self.gdn_checkpoint_interval <= 0:
            raise ValueError(
                "gdn_checkpoint_interval must be positive, "
                f"got {self.gdn_checkpoint_interval}"
            )
        self.max_gdn_checkpoint_slots = int(self.max_gdn_checkpoint_slots)
        if self.max_gdn_checkpoint_slots <= 0:
            raise ValueError(
                "max_gdn_checkpoint_slots must be positive, "
                f"got {self.max_gdn_checkpoint_slots}"
            )
        self.hybrid_apc_layout_version = int(self.hybrid_apc_layout_version)
        self.hybrid_recurrent_cache_dtype = _normalize_hybrid_cache_dtype(
            "hybrid_recurrent_cache_dtype",
            self.hybrid_recurrent_cache_dtype,
            "float32",
        )
        self.hybrid_conv_cache_dtype = _normalize_hybrid_cache_dtype(
            "hybrid_conv_cache_dtype",
            self.hybrid_conv_cache_dtype,
            "bfloat16",
        )
        self.gdn_recurrent_cache_dtype = self.hybrid_recurrent_cache_dtype
        self.gdn_conv_cache_dtype = self.hybrid_conv_cache_dtype
        self.hybrid_cache_block_boundary_only = (
            self.hybrid_cache_prefix_boundary_only
        )
        self.hybrid_apc_require_vllm_metadata = bool(
            self.hybrid_apc_require_vllm_metadata
        )
        self.hybrid_apc_allow_local_hash_fallback = bool(
            self.hybrid_apc_allow_local_hash_fallback
        )
        self.hybrid_apc_require_attention_block_refs = bool(
            self.hybrid_apc_require_attention_block_refs
        )
        self.hybrid_apc_reject_unbacked_attention_hits = bool(
            self.hybrid_apc_reject_unbacked_attention_hits
        )
        self.hybrid_apc_disable_unbacked_prefix_reads = bool(
            self.hybrid_apc_disable_unbacked_prefix_reads
        )
        if self.hybrid_apc_require_vllm_metadata:
            self.hybrid_apc_allow_local_hash_fallback = False
            self.hybrid_apc_require_attention_block_refs = True
            self.hybrid_apc_reject_unbacked_attention_hits = True
        if self.use_hybrid_cache_manager and self.use_hybrid_apc_manager:
            raise ValueError(
                "use_hybrid_cache_manager and use_hybrid_apc_manager are mutually exclusive"
            )
        if self.use_hybrid_apc_manager and self.hybrid_cache_mode != "all":
            raise ValueError("use_hybrid_apc_manager requires hybrid_cache_mode='all'")
        if self.use_hybrid_apc_manager:
            if self.hybrid_recurrent_cache_dtype != "float32":
                raise ValueError(
                    "use_hybrid_apc_manager requires float32 recurrent GDN "
                    "checkpoint cache state; bf16 checkpoint roundtrips are not "
                    "coherent for all-mode prefix caching"
                )
            pa_block_size = getattr(self.neuron_config, "pa_block_size", None)
            if pa_block_size is not None and self.gdn_checkpoint_interval != int(
                pa_block_size
            ):
                raise ValueError(
                    "use_hybrid_apc_manager v0 requires "
                    "gdn_checkpoint_interval == pa_block_size"
                )
            if self.hybrid_apc_allow_residual_replay:
                raise ValueError(
                    "hybrid_apc_allow_residual_replay is reserved for v1; "
                    "v0 restores only exact checkpoint boundaries"
                )

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
        # MoE variants (qwen3_5_moe_text) use MoENeuronConfig so that
        # initialize_moe_module() finds router_config / blockwise_matmul_config
        # / moe_tp_degree / etc. Dense variants keep the plain NeuronConfig path.
        # This is a class method but at call-time we don't have an instance;
        # callers construct NeuronConfig explicitly, so we return the correct
        # default for the common (dense) case here.
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

        if position_ids_3d.ndim == 2:
            position_ids_3d = position_ids_3d[None, ...].expand(
                3, position_ids_3d.shape[0], -1
            )

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rope_dim, 2, dtype=dtype, device=device)
                / self.rope_dim
            )
        )
        inv_freq = inv_freq[None, None, :, None].expand(
            3, position_ids_3d.shape[1], -1, 1
        )
        positions = position_ids_3d[:, :, None, :].float()
        freqs = (inv_freq.float() @ positions).transpose(2, 3)

        # Match HF Qwen3.6 mRoPE layout exactly: start from the temporal
        # frequencies, then splice H/W frequencies into interleaved positions.
        freqs_t = freqs[0]
        if self.mrope_interleaved:
            for dim, offset in enumerate((1, 2), start=1):
                length = self.mrope_section[dim] * 3
                idx = slice(offset, length, 3)
                freqs_t[..., idx] = freqs[dim, ..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)

        return cos, sin


class NeuronQwen35Attention(NeuronAttentionBase):
    """Standard GQA attention for Qwen3.5 with output gate and partial RoPE.

    24 Q heads, 4 KV heads (6:1 GQA), head_dim=256 for 27B dense.
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

        self.qwen_output_gate_nki_kernel_enabled = bool(
            getattr(config, "use_qwen_output_gate_nki", False)
            or os.environ.get("QWEN36_OUTPUT_GATE_NKI", "0") == "1"
        )
        self.qwen_qkv_gate_packed_enabled = bool(
            getattr(config, "use_qwen_qkv_gate_packed", False)
            or os.environ.get("QWEN36_QKV_GATE_PACKED", "0") == "1"
        )
        self.qwen_gated_o_proj_nki_kernel_enabled = bool(
            getattr(config, "use_qwen_gated_o_proj_nki", False)
            or os.environ.get("QWEN36_GATED_OUT_PROJ_NKI", "0") == "1"
        )
        if (
            self.qwen_output_gate_nki_kernel_enabled
            and self.qwen_qkv_gate_packed_enabled
        ):
            raise ValueError(
                "Qwen output-gate NKI and packed QKV+gate are mutually exclusive."
            )
        if self.qwen_output_gate_nki_kernel_enabled:
            if _qwen_gate_projection_kernel is None:
                raise ImportError(
                    "QWEN36_OUTPUT_GATE_NKI requires nkilib.core.qkv.qkv"
                )
            if getattr(config.neuron_config, "quantized", False):
                setattr(
                    self.output_gate_proj,
                    "post_create_quantized_module_hook",
                    preprocess_quantized_linear_layer,
                )
            else:
                self.output_gate_proj.weight = transpose_parallel_linear_layer(
                    self.output_gate_proj.weight
                )

        if self.qwen_qkv_gate_packed_enabled:
            if _qwen_gate_projection_kernel is None:
                raise ImportError(
                    "QWEN36_QKV_GATE_PACKED requires nkilib.core.qkv.qkv"
                )
            if not self.fused_qkv:
                raise ValueError("QWEN36_QKV_GATE_PACKED requires fused_qkv=True")
            self._enable_qwen_qkv_gate_packed_projection(config)

        self.qwen_qk_norm_rope_nki_kernel_enabled = bool(
            getattr(config, "use_qwen_qk_norm_rope_nki", False)
            or os.environ.get("QWEN36_QK_NORM_ROPE_NKI", "0") == "1"
        )
        if self.qwen_qk_norm_rope_nki_kernel_enabled:
            if _qwen_qk_norm_partial_rope_kernel is None:
                raise ImportError(
                    "QWEN36_QK_NORM_ROPE_NKI requires src.nki_kernels."
                    "qwen_qk_norm_rope"
                )
            if self.head_dim != 256 or self.rope_dim != 64:
                raise ValueError(
                    "Qwen Q/K norm+RoPE NKI kernel currently supports only "
                    f"head_dim=256 and rope_dim=64, got head_dim={self.head_dim}, "
                    f"rope_dim={self.rope_dim}"
                )

        self.qkv_tkg_nki_kernel_enabled = bool(
            getattr(config.neuron_config, "qkv_tkg_nki_kernel_enabled", False)
        ) and not bool(getattr(config.neuron_config, "is_prefill_stage", False))
        if self.qkv_tkg_nki_kernel_enabled:
            if _qkv_tkg_nki_kernel is None:
                raise ImportError(
                    "qkv_tkg_nki_kernel_enabled requires "
                    "neuronxcc.nki._pre_prod_kernels.qkv_tkg_impl"
                )
            if self.fused_qkv:
                raise ValueError(
                    "qkv_tkg_nki_kernel_enabled uses split q/k/v projections "
                    "and must not be combined with fused_qkv"
                )
            if self.qkv_proj_sp_enabled:
                raise ValueError(
                    "qkv_tkg_nki_kernel_enabled does not support sequence-parallel "
                    "QKV projection"
                )
            qkv_proj = self.get_qkv_proj()
            split_qkv_projections = (
                qkv_proj.q_proj,
                qkv_proj.k_proj,
                qkv_proj.v_proj,
            )
            for projection in split_qkv_projections:
                if not getattr(config.neuron_config, "quantized", False):
                    projection.weight = transpose_parallel_linear_layer(projection.weight)

    def _enable_qwen_qkv_gate_packed_projection(self, config):
        for attr_name in ("qkv_proj", "cte_qkv_proj", "tkg_qkv_proj"):
            qkv_proj = getattr(self, attr_name, None)
            if qkv_proj is not None and getattr(qkv_proj, "fused_qkv", False):
                self._replace_qkv_projection_with_qwen_qkvgate(qkv_proj, config)

    def _replace_qkv_projection_with_qwen_qkvgate(self, qkv_proj, config):
        if not hasattr(qkv_proj, "Wqkv"):
            raise ValueError("QWEN36_QKV_GATE_PACKED requires a fused Wqkv module")
        if not isinstance(qkv_proj.Wqkv, ColumnParallelLinear):
            raise ValueError(
                "QWEN36_QKV_GATE_PACKED currently supports ColumnParallelLinear Wqkv"
            )

        packed_q_heads = qkv_proj.num_attention_heads * 2
        packed_output_size = (
            packed_q_heads + 2 * qkv_proj.num_key_value_heads
        ) * qkv_proj.head_dim
        packed_wqkv = ColumnParallelLinear(
            qkv_proj.hidden_size,
            packed_output_size,
            bias=qkv_proj.bias,
            gather_output=qkv_proj.gather_output,
            dtype=qkv_proj.dtype,
            sequence_parallel_enabled=False,
            tensor_model_parallel_group=qkv_proj.tensor_model_parallel_group,
            rank_ordering=qkv_proj.rank_ordering,
        )
        if (
            (qkv_proj.qkv_kernel_enabled or qkv_proj.qkv_nki_kernel_enabled)
            and getattr(config.neuron_config, "quantized", False)
        ):
            setattr(
                packed_wqkv,
                "post_create_quantized_module_hook",
                preprocess_quantized_linear_layer,
            )
        elif qkv_proj.qkv_kernel_enabled or qkv_proj.qkv_nki_kernel_enabled:
            packed_wqkv.weight = transpose_parallel_linear_layer(packed_wqkv.weight)

        for param in (
            [packed_wqkv.weight, packed_wqkv.scale]
            if hasattr(packed_wqkv, "scale")
            else [packed_wqkv.weight]
        ):
            setattr(param, "fused_qkv", True)
            setattr(param, "num_attention_heads", packed_q_heads)
            setattr(param, "num_key_value_heads", qkv_proj.num_key_value_heads)
            setattr(param, "head_dim", qkv_proj.head_dim)
        if qkv_proj.bias:
            setattr(packed_wqkv.bias, "fused_qkv", True)
            setattr(packed_wqkv.bias, "num_attention_heads", packed_q_heads)
            setattr(packed_wqkv.bias, "num_key_value_heads", qkv_proj.num_key_value_heads)
            setattr(packed_wqkv.bias, "head_dim", qkv_proj.head_dim)

        qkv_proj.Wqkv = packed_wqkv
        qkv_proj.qwen_qkv_gate_packed = True
        qkv_proj.qwen_real_num_attention_heads = qkv_proj.num_attention_heads
        qkv_proj.qwen_packed_num_attention_heads = packed_q_heads

    @staticmethod
    def _apply_projection_scale(output, projection):
        scale = getattr(projection, "scale", None)
        if scale is None:
            return output
        scale_tensor = scale.data if hasattr(scale, "data") else scale
        if (
            scale_tensor.ndim == 2
            and scale_tensor.shape[0] == 128
            and scale_tensor.shape[1] == output.shape[-1]
        ):
            scale_tensor = scale_tensor[0]
        else:
            scale_tensor = scale_tensor.reshape(-1)
        if scale_tensor.numel() != output.shape[-1]:
            raise ValueError(
                "QKV TKG projection scale shape does not match output width: "
                f"scale={tuple(scale.shape)}, output={tuple(output.shape)}"
            )
        return output * scale_tensor.reshape(1, 1, output.shape[-1]).to(output.dtype)

    @staticmethod
    def _prepare_qkv_tkg_scale(scale_tensor, output_width):
        if (
            scale_tensor.ndim == 2
            and scale_tensor.shape[0] == 128
            and scale_tensor.shape[1] == output_width
        ):
            return scale_tensor.contiguous()
        if (
            scale_tensor.ndim == 2
            and scale_tensor.shape[0] == output_width
            and scale_tensor.shape[1] == 1
        ):
            return torch.broadcast_to(
                scale_tensor.transpose(0, 1),
                (128, output_width),
            ).contiguous()
        if (
            scale_tensor.ndim == 2
            and scale_tensor.shape[0] == 1
            and scale_tensor.shape[1] == output_width
        ):
            return torch.broadcast_to(scale_tensor, (128, output_width)).contiguous()
        if scale_tensor.numel() == output_width:
            return torch.broadcast_to(
                scale_tensor.reshape(1, output_width),
                (128, output_width),
            ).contiguous()
        raise ValueError(
            "QKV TKG projection scale shape does not match output width: "
            f"scale={tuple(scale_tensor.shape)}, output_width={output_width}"
        )

    def _run_split_qkv_tkg_projection(self, hidden_states, projection, local_heads):
        bias = (
            projection.bias.data.unsqueeze(0)
            if getattr(projection, "bias", None) is not None
            else None
        )
        weight = projection.weight.data
        if weight.shape[0] != self.hidden_size and weight.shape[1] == self.hidden_size:
            weight = weight.transpose(0, 1).contiguous()
        # The preprod QKV TKG kernel's LNC2 path reduces across pi0 and then
        # stores both programs to the same shared-HBM slice, which the current
        # NKI verifier rejects as an output dependency. Use the single-LNC
        # variant for this split projection until that kernel store is fixed.
        kernel = _qkv_tkg_nki_kernel[1]
        scale = getattr(projection, "scale", None)
        if scale is not None:
            scale_tensor = scale.data if hasattr(scale, "data") else scale
            qkv_w_scales = self._prepare_qkv_tkg_scale(
                scale_tensor,
                weight.shape[1],
            )
            quantization_type = getattr(_QKVQuantizationType, "ROW", None)
            if quantization_type is None:
                raise ValueError(
                    "qkv_tkg_nki_kernel_enabled requires ROW quantization support "
                    "when running quantized split-QKV projections"
                )
        else:
            qkv_w_scales = None
            quantization_type = _QKVQuantizationType.NONE

        output = kernel(
            hidden=hidden_states,
            qkv_w=weight,
            norm_w=None,
            fused_add=False,
            mlp_prev=None,
            attn_prev=None,
            d_head=self.head_dim,
            output_layout=_QKVOutputLayout.BSD,
            eps=self.rms_norm_eps,
            norm_type=_QKVNormType.NO_NORM,
            qkvInSB=False,
            qkv_bias=bias,
            norm_bias=None,
            hidden_actual=self.hidden_size,
            B=hidden_states.shape[0],
            S=hidden_states.shape[1],
            H=self.hidden_size,
            num_q_heads=local_heads,
            num_kv_heads=local_heads,
            quantization_type=quantization_type,
            qkv_w_scales=qkv_w_scales,
            qkv_in_scales=None,
        )
        if qkv_w_scales is not None:
            return output
        return self._apply_projection_scale(output, projection)

    def _prep_split_qkv_tkg_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        use_polar_compatible_rope=False,
    ):
        # NxDI traces a placeholder adapter_ids tensor even when no LoRA
        # adapters are active. Qwen3.6 serving here is non-LoRA, so the split
        # projection path intentionally ignores the placeholder.
        qkv_proj = self.get_qkv_proj()
        Q = self._run_split_qkv_tkg_projection(
            hidden_states,
            qkv_proj.q_proj,
            self.num_heads,
        )
        K = self._run_split_qkv_tkg_projection(
            hidden_states,
            qkv_proj.k_proj,
            self.num_key_value_heads,
        )
        V = self._run_split_qkv_tkg_projection(
            hidden_states,
            qkv_proj.v_proj,
            self.num_key_value_heads,
        )

        bsz, q_len, _ = hidden_states.size()
        Q = move_heads_front(
            Q,
            bsz,
            q_len,
            self.num_heads,
            self.head_dim,
            layernorm=self.q_layernorm,
            post_transpose_layernorm=self.post_transpose_layernorm,
        )
        K = move_heads_front(
            K,
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
            layernorm=self.k_layernorm,
            post_transpose_layernorm=self.post_transpose_layernorm,
        )
        V = move_heads_front(
            V,
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
            layernorm=None,
        )

        Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(
            Q,
            K,
            V,
            position_ids,
            cos_cache,
            sin_cache,
            use_polar_compatible_rope,
        )
        return Q, K, V, cos_cache, sin_cache, None

    def _should_use_qwen_output_gate_nki(self, q_len):
        return self.qwen_output_gate_nki_kernel_enabled

    def _should_use_qwen_qkv_gate_packed(self, q_len):
        return (
            self.qwen_qkv_gate_packed_enabled
            and not self.qkv_proj_sp_enabled
            and _qwen_gate_projection_kernel is not None
        )

    def _should_use_qwen_gated_o_proj_nki(self, q_len):
        o_proj = self.get_o_proj()
        return (
            self.qwen_gated_o_proj_nki_kernel_enabled
            and q_len > 1
            and hasattr(o_proj, "forward_gated")
        )

    def _output_gate_proj_nki(self, hidden_states):
        weight = self.output_gate_proj.weight.data
        bias = (
            self.output_gate_proj.bias.data.unsqueeze(0)
            if getattr(self.output_gate_proj, "bias", None) is not None
            else None
        )

        qkv_w_scale = None
        qkv_in_scale = None
        quantization_type = _NkilibQuantizationType.NONE
        gate_scale = getattr(self.output_gate_proj, "scale", None)
        if gate_scale is not None:
            qkv_w_scale = gate_scale.data
            gate_input_scale = getattr(self.output_gate_proj, "input_scale", None)
            qkv_in_scale = gate_input_scale.data if gate_input_scale is not None else None
            quantization_type = _NkilibQuantizationType.ROW
        elif getattr(self.config.neuron_config, "quantized", False):
            raise RuntimeError(
                "Qwen output-gate NKI path requires output_gate_proj.scale "
                "when running a quantized artifact."
            )

        return _qwen_gate_projection_kernel[self.logical_nc_config](
            input=hidden_states,
            fused_qkv_weights=weight,
            output_layout=_NkilibQKVOutputLayout.BSD,
            bias=bias,
            quantization_type=quantization_type,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
        )

    def _qkv_gate_packed_projection_nki(self, hidden_states):
        qkv_proj = self.get_qkv_proj()
        weight = qkv_proj.Wqkv.weight.data
        bias = (
            qkv_proj.Wqkv.bias.data.unsqueeze(0)
            if getattr(qkv_proj.Wqkv, "bias", None) is not None
            else None
        )

        qkv_w_scale = None
        qkv_in_scale = None
        quantization_type = _NkilibQuantizationType.NONE
        qkv_scale = getattr(qkv_proj.Wqkv, "scale", None)
        if qkv_scale is not None:
            qkv_w_scale = qkv_scale.data
            qkv_input_scale = getattr(qkv_proj.Wqkv, "input_scale", None)
            qkv_in_scale = qkv_input_scale.data if qkv_input_scale is not None else None
            quantization_type = _NkilibQuantizationType.ROW
        elif getattr(self.config.neuron_config, "quantized", False):
            raise RuntimeError(
                "Qwen packed QKV+gate path requires Wqkv.scale when running "
                "a quantized artifact."
            )

        packed = _qwen_gate_projection_kernel[self.logical_nc_config](
            input=hidden_states,
            fused_qkv_weights=weight,
            output_layout=_NkilibQKVOutputLayout.BSD,
            bias=bias,
            fused_residual_add=False,
            mlp_prev=None,
            attention_prev=None,
            fused_norm_type=_NkilibNormType.NO_NORM,
            gamma_norm_weights=None,
            norm_eps=self.rms_norm_eps,
            fused_rope=False,
            cos_cache=None,
            sin_cache=None,
            quantization_type=quantization_type,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            d_head=self.head_dim,
            num_q_heads=self.num_heads * 2,
            num_kv_heads=self.num_key_value_heads,
        )

        q_width = self.num_heads * self.head_dim
        gate_end = q_width * 2
        k_end = gate_end + self.num_key_value_heads * self.head_dim
        Q, gate, K, V = torch.tensor_split(
            packed,
            (q_width, gate_end, k_end),
            dim=2,
        )
        return Q, gate, K, V

    def _prep_qkv_gate_packed_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        use_polar_compatible_rope=False,
    ):
        Q, gate, K, V = self._qkv_gate_packed_projection_nki(hidden_states)

        bsz, q_len, _ = hidden_states.size()
        V = move_heads_front(
            V,
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
            layernorm=None,
        )
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)
        if (
            self._should_use_qwen_qk_norm_rope_nki(q_len)
            and cos_cache is not None
            and sin_cache is not None
        ):
            Q, K = _qwen_qk_norm_partial_rope_kernel[self.logical_nc_config](
                Q,
                K,
                self.q_layernorm.weight.data,
                self.k_layernorm.weight.data,
                cos_cache,
                sin_cache,
                self.rms_norm_eps,
            )
        else:
            Q = move_heads_front(
                Q,
                bsz,
                q_len,
                self.num_heads,
                self.head_dim,
                layernorm=self.q_layernorm,
                post_transpose_layernorm=self.post_transpose_layernorm,
            )
            K = move_heads_front(
                K,
                bsz,
                q_len,
                self.num_key_value_heads,
                self.head_dim,
                layernorm=self.k_layernorm,
                post_transpose_layernorm=self.post_transpose_layernorm,
            )
            Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(
                Q,
                K,
                V,
                position_ids,
                cos_cache,
                sin_cache,
                use_polar_compatible_rope,
            )
        return Q, K, V, gate, cos_cache, sin_cache, None

    def _should_use_qwen_qk_norm_rope_nki(self, q_len):
        return (
            self.qwen_qk_norm_rope_nki_kernel_enabled
            and q_len > 1
            and self.q_layernorm is not None
            and self.k_layernorm is not None
            and not self.qkv_proj_sp_enabled
        )

    def _prep_qkv_tensors_qwen_qk_norm_rope_nki(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
    ):
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states,
            rmsnorm=rmsnorm,
            adapter_ids=adapter_ids,
            residual=None,
        )

        bsz, q_len, _ = hidden_states.size()
        V = move_heads_front(
            V,
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
            layernorm=None,
        )
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)

        Q, K = _qwen_qk_norm_partial_rope_kernel[self.logical_nc_config](
            Q,
            K,
            self.q_layernorm.weight.data,
            self.k_layernorm.weight.data,
            cos_cache,
            sin_cache,
            self.rms_norm_eps,
        )
        return Q, K, V, cos_cache, sin_cache, residual

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

    def perform_qwen_chunked_prefill(
        self,
        Q,
        K,
        V,
        past_key_value,
        position_ids,
        attention_mask=None,
        kv_mgr=None,
        idx=None,
        active_block_table=None,
        computed_context_lens=None,
        scatter_index=None,
        kvcache_buffer=None,
    ):
        """Exact chunked CTE over full-cache or selected-prefix KV.

        For model-local chunked prefill, the current chunk K/V tensors are
        scattered into the full cache at absolute position_ids. For vLLM prefix
        reuse, BlockKVCacheManager returns selected prefix blocks already
        arranged as logical positions, so concatenate the current suffix K/V
        after that logical prefix.
        """
        k_cache, v_cache = past_key_value
        B, q_heads, q_len, head_dim = Q.shape
        kv_heads = K.shape[1]
        use_segmented_prefix_cte = (
            getattr(
                self.config.neuron_config,
                "prefix_cte_attention_backend",
                "attention_cte",
            )
            == "segmented_cte"
            and active_block_table is not None
            and getattr(active_block_table, "ndim", 0) > 1
        )
        if use_segmented_prefix_cte:
            if kv_mgr is None or idx is None or scatter_index is None:
                raise ValueError(
                    "segmented_cte Qwen prefix prefill requires kv_mgr, idx, "
                    "and scatter_index so active KV can be written to block KV."
                )
            updated_kv = kv_mgr.update_kv_by_layer_id(
                idx=idx,
                kv_per_layer=(K.to(self.torch_dtype), V.to(self.torch_dtype)),
                scatter_index=scatter_index,
                kvcache_buffer=kvcache_buffer,
            )
            attn_output, _flash_strategy = self.perform_prefix_prefill_segmented_cte(
                Q,
                q_len,
                B,
                updated_kv,
                active_block_table,
                computed_context_lens,
            )
            return attn_output.permute(0, 1, 3, 2).contiguous(), updated_kv

        if k_cache.shape[0] != B:
            # The cache is allocated at kv_cache_batch_size, while CTE can trace a
            # smaller active batch. Keep attention reshapes on the active batch.
            k_cache = k_cache[:B]
            v_cache = v_cache[:B]
        cache_len = k_cache.shape[2]

        pos = position_ids.long()
        selected_prefix_cache = cache_len < int(
            getattr(self.config.neuron_config, "seq_len", cache_len)
        )
        if selected_prefix_cache:
            k_cache = torch.cat([k_cache, K.to(k_cache.dtype)], dim=2)
            v_cache = torch.cat([v_cache, V.to(v_cache.dtype)], dim=2)
            prefix_positions = torch.arange(
                cache_len,
                device=position_ids.device,
                dtype=pos.dtype,
            ).view(1, -1).expand(B, -1)
            cache_positions = torch.cat([prefix_positions, pos], dim=1).view(
                B,
                1,
                1,
                -1,
            )
            prefix_valid = torch.ones(
                (B, cache_len),
                device=position_ids.device,
                dtype=torch.bool,
            )
            if (
                attention_mask is not None
                and attention_mask.ndim == 2
                and attention_mask.shape[1] == q_len
            ):
                active_valid = attention_mask.to(torch.bool)
            else:
                active_valid = torch.ones(
                    (B, q_len),
                    device=position_ids.device,
                    dtype=torch.bool,
                )
            key_valid_mask = torch.cat([prefix_valid, active_valid], dim=1).view(
                B,
                1,
                1,
                -1,
            )
            cache_len = k_cache.shape[2]
        else:
            k_index = pos[:, None, :, None].expand(B, kv_heads, q_len, head_dim)
            k_cache = torch.scatter(
                k_cache,
                dim=2,
                index=k_index,
                src=K.to(k_cache.dtype),
            )
            v_cache = torch.scatter(
                v_cache,
                dim=2,
                index=k_index,
                src=V.to(v_cache.dtype),
            )
            cache_positions = torch.arange(
                cache_len,
                device=position_ids.device,
                dtype=pos.dtype,
            ).view(1, 1, 1, -1)
            key_valid_mask = None

        prefix_attention_impl = _qwen36_prefix_attention_impl()
        if prefix_attention_impl == "grouped":
            attn_output = _qwen35_grouped_prefix_attention(
                Q,
                k_cache,
                v_cache,
                pos,
                cache_positions,
                key_valid_mask,
            )
        else:
            attn_output = _qwen35_expanded_prefix_attention(
                Q,
                k_cache,
                v_cache,
                pos,
                cache_positions,
                key_valid_mask,
            )
        return attn_output, None

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

        use_split_qkv_tkg = (
            self.qkv_tkg_nki_kernel_enabled
            and past_key_value is not None
            and q_len == 1
        )
        if self._should_use_qwen_qkv_gate_packed(q_len):
            Q, K, V, gate, cos_cache, sin_cache, _residual = (
                self._prep_qkv_gate_packed_tensors(
                    rope_pos_ids,
                    hidden_states,
                    past_key_value,
                    adapter_ids=adapter_ids,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                )
            )
        elif use_split_qkv_tkg:
            gate = (
                self._output_gate_proj_nki(hidden_states)
                if self._should_use_qwen_output_gate_nki(q_len)
                else self.output_gate_proj(hidden_states)
            )
            Q, K, V, cos_cache, sin_cache, _residual = (
                self._prep_split_qkv_tkg_tensors(
                    rope_pos_ids,
                    hidden_states,
                    past_key_value,
                    adapter_ids=adapter_ids,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                )
            )
        elif self.qkv_tkg_nki_kernel_enabled:
            raise ValueError(
                "qkv_tkg_nki_kernel_enabled is only valid for single-token "
                f"decode, got past_key_value={past_key_value is not None}, "
                f"q_len={q_len}"
            )
        else:
            # Compute gate from input hidden states (before QKV projection).
            if self._should_use_qwen_output_gate_nki(q_len):
                gate = self._output_gate_proj_nki(hidden_states)
            else:
                gate = self.output_gate_proj(hidden_states)

            # Standard QKV prep (projections, QK norm, RoPE)
            if self._should_use_qwen_qk_norm_rope_nki(q_len):
                Q, K, V, cos_cache, sin_cache, _residual = (
                    self._prep_qkv_tensors_qwen_qk_norm_rope_nki(
                        rope_pos_ids,
                        hidden_states,
                        past_key_value,
                        adapter_ids=adapter_ids,
                        cos_cache=cos_cache,
                        sin_cache=sin_cache,
                        rmsnorm=rmsnorm,
                    )
                )
            else:
                Q, K, V, cos_cache, sin_cache, _residual = self.prep_qkv_tensors(
                    rope_pos_ids,
                    hidden_states,
                    past_key_value,
                    adapter_ids=adapter_ids,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    rmsnorm=rmsnorm,
                )

        qwen_chunked_prefill_active = (
            past_key_value is not None
            and q_len > 1
            and getattr(self.config, "use_qwen_hybrid_chunked_prefill", False)
        )

        if past_key_value is None:
            # Context encoding (prefill)
            attn_output, _flash_strategy = self.perform_prefill(
                Q, K, V, q_len, bsz, attention_mask
            )
        elif qwen_chunked_prefill_active:
            attn_output, present_key_value = self.perform_qwen_chunked_prefill(
                Q,
                K,
                V,
                past_key_value,
                position_ids,
                attention_mask,
                kv_mgr=kwargs.get("kv_mgr"),
                idx=kwargs.get("idx"),
                active_block_table=kwargs.get("active_block_table"),
                computed_context_lens=kwargs.get("computed_context_lens"),
                scatter_index=kwargs.get("scatter_index"),
                kvcache_buffer=kwargs.get("kvcache_buffer"),
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

        o_proj = self.get_o_proj()
        if self._should_use_qwen_gated_o_proj_nki(q_len):
            attn_output = o_proj.forward_gated(attn_output, gate, adapter_ids=adapter_ids)
        else:
            # Apply sigmoid output gate BEFORE o_proj (matching HF reference)
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = o_proj(attn_output, adapter_ids=adapter_ids)

        # Ensure K, V are in model dtype (bf16) for KV cache update
        # (prevents mixed-precision dynamic-update-slice in neuronx-cc)
        K = K.to(self.torch_dtype)
        V = V.to(self.torch_dtype)
        if "present_key_value" not in locals() or present_key_value is None:
            present_key_value = (K, V)
        past_key_value = present_key_value
        return attn_output, past_key_value, cos_cache, sin_cache


# ============================================================
# Dense MLP (replaces MoE)
# ============================================================


class Qwen35MLP(nn.Module):
    """Dense SwiGLU MLP for Qwen3.5/3.6-27B.

    gate_proj: hidden_size -> intermediate_size (5120 -> 17408)
    up_proj:   hidden_size -> intermediate_size (5120 -> 17408)
    down_proj: intermediate_size -> hidden_size (17408 -> 5120)

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
# MoE MLP (Qwen3.5-MoE variants — qwen3_5_moe_text model_type)
# ============================================================


class Qwen35MoEBlock(nn.Module):
    """Qwen3.5-MoE sparse feed-forward with a shared expert gated by a per-token
    sigmoid, matching the HF reference `Qwen3_5MoeSparseMoeBlock`.

    Uses NxDI's `initialize_moe_module` for the routed experts + a plain shared
    MLP (SwiGLU). The shared-expert sigmoid gate lives here — NxDI's built-in
    SharedExperts sums shared_output into the routed output directly, without
    a per-token gate, so we override the shared path.
    """

    def __init__(self, config):
        super().__init__()
        if not HAS_MOE_V2:
            raise RuntimeError(
                "Qwen3.5-MoE variant needs neuronx_distributed_inference.modules.moe_v2, "
                "which is not present in the installed NxDI."
            )
        # NxDI's MoE module (routed experts + optional built-in shared experts).
        # We DISABLE the built-in shared branch (n_shared_experts=0 for the module)
        # and roll our own gated shared expert below.
        moe_config = copy.copy(config)  # shallow copy so we can override transiently
        moe_config.n_shared_experts = 0  # keep initialize_moe_module's SharedExperts off
        self.moe = initialize_moe_module(config=moe_config)

        # Sigmoid-gated shared expert with its own per-token gate. Uses column-
        # parallel + row-parallel linears, sharded on the intermediate dim.
        self.shared_intermediate_size = getattr(
            config, "shared_expert_intermediate_size",
            getattr(config, "moe_intermediate_size", config.intermediate_size),
        )
        self.shared_gate_proj = ColumnParallelLinear(
            config.hidden_size, self.shared_intermediate_size,
            bias=False, gather_output=False,
        )
        self.shared_up_proj = ColumnParallelLinear(
            config.hidden_size, self.shared_intermediate_size,
            bias=False, gather_output=False,
        )
        self.shared_down_proj = RowParallelLinear(
            self.shared_intermediate_size, config.hidden_size,
            bias=False, input_is_parallel=True,
        )
        # Per-token sigmoid gate for the shared expert output.
        # gate_output = sigmoid(shared_expert_gate(x)) * shared_mlp(x).
        # This is a scalar-output linear (1 output feature), which can't be
        # column-sharded, so it lives replicated on every rank.
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        # NxDI MoE forward returns (output, *aux) where output has same shape
        # as hidden_states.
        moe_output = self.moe(hidden_states)
        if isinstance(moe_output, (tuple, list)):
            moe_output = moe_output[0]

        # Shared expert (SwiGLU).
        gate = self.shared_gate_proj(hidden_states)
        up = self.shared_up_proj(hidden_states)
        shared = self.shared_down_proj(F.silu(gate) * up)

        # Per-token sigmoid gate on the shared expert output.
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        shared = shared_gate * shared

        return moe_output + shared


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

        # Dense MLP (all layers).  The reusable NxDI Llama MLP kernel supports
        # both CTE and TKG; keep RMSNorm separate for CTE so normalization stays
        # on the conservative high-precision path before FP8 GEMM quantization.
        self.mlp_kernel_enabled = bool(config.neuron_config.mlp_kernel_enabled)
        self.mlp_kernel_fused_rmsnorm = (
            self.mlp_kernel_enabled
            and not config.neuron_config.sequence_parallel_enabled
        )
        # MoE variants use a sparse feed-forward with routed experts + a
        # sigmoid-gated shared expert. Dense variants keep the plain SwiGLU MLP.
        self.is_moe = getattr(config, "_is_moe", False)
        if self.is_moe:
            self.mlp_kernel_enabled = False
            self.mlp = Qwen35MoEBlock(config)
        elif self.mlp_kernel_enabled:
            tensor_model_parallel_group = (
                parallel_state.get_tensor_model_parallel_group()
                if parallel_state.model_parallel_is_initialized()
                else None
            )
            self.mlp = NeuronLlamaMLP(config, tensor_model_parallel_group)
        else:
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
            deltanet_states = (
                None
                if getattr(self.config, "use_hybrid_cache_manager", False)
                else (new_rec_state, new_conv_state)
            )
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
        if self.mlp_kernel_enabled:
            use_fused_mlp_rmsnorm = (
                self.mlp_kernel_fused_rmsnorm
                and not bool(kwargs.get("is_for_context_encoding", False))
                and hidden_states.shape[1] == 1
            )
            if use_fused_mlp_rmsnorm:
                mlp_fused_rmsnorm = self.post_attention_layernorm
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
                mlp_fused_rmsnorm = None
            hidden_states, _ = self.mlp(hidden_states, rmsnorm=mlp_fused_rmsnorm)
        else:
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
# Hybrid Cache Manager (opt-in)
# ============================================================


class HybridDeltaNetCacheManager(KVCacheManager):
    """Opt-in local/static cache manager for Qwen hybrid dense models.

    This manager stores DeltaNet recurrent/conv state by batch row and delegates
    full-attention layers to the legacy KV manager. It is intentionally not a
    production vLLM APC manager: block ownership, prefix hashes, refcounts,
    eviction, continuous batching, and tenant isolation must remain in the
    vLLM/NxDI block-cache lifecycle.
    """

    def __init__(self, config: Qwen35InferenceConfig, num_kv_head, **kwargs):
        self.layer_types = list(config.layer_types)
        self._validate_hybrid_config(config)
        super().__init__(config, num_kv_head=num_kv_head, **kwargs)

        dtype = (
            config.neuron_config.attention_dtype
            if config.neuron_config.attention_dtype is not None
            else config.neuron_config.torch_dtype
        )
        cache_dtype = getattr(self, "cache_dtype", dtype)
        recurrent_cache_dtype = _torch_dtype_from_hybrid_cache_dtype(
            config.hybrid_recurrent_cache_dtype
        )
        conv_cache_dtype = _torch_dtype_from_hybrid_cache_dtype(
            config.hybrid_conv_cache_dtype
        )
        max_batch_size = (
            config.neuron_config.kv_cache_batch_size
            + config.neuron_config.kv_cache_padding_size
        )
        tp_degree = config.neuron_config.tp_degree
        if config.linear_num_value_heads % tp_degree != 0:
            raise ValueError(
                f"linear_num_value_heads={config.linear_num_value_heads} must be divisible "
                f"by tp_degree={tp_degree}"
            )
        if config.linear_num_key_heads % tp_degree != 0:
            raise ValueError(
                f"linear_num_key_heads={config.linear_num_key_heads} must be divisible "
                f"by tp_degree={tp_degree}"
            )
        local_num_value_heads = config.linear_num_value_heads // tp_degree
        local_num_key_heads = config.linear_num_key_heads // tp_degree
        recurrent_shape = [
            max_batch_size,
            local_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        ]
        conv_dim = (
            2 * local_num_key_heads * config.linear_key_head_dim
            + local_num_value_heads * config.linear_value_head_dim
        )
        conv_shape = [
            max_batch_size,
            conv_dim,
            config.linear_conv_kernel_dim - 1,
        ]

        params = []
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "linear_attention":
                params.append(
                    nn.Parameter(
                        torch.zeros(recurrent_shape, dtype=recurrent_cache_dtype),
                        requires_grad=False,
                    )
                )
                params.append(
                    nn.Parameter(
                        torch.zeros(conv_shape, dtype=conv_cache_dtype),
                        requires_grad=False,
                    )
                )
            else:
                k_shape = self.k_shapes[layer_idx] if hasattr(self, "k_shapes") else self.k_shape
                v_shape = self.v_shapes[layer_idx] if hasattr(self, "v_shapes") else self.v_shape
                params.append(
                    nn.Parameter(torch.zeros(k_shape, dtype=cache_dtype), requires_grad=False)
                )
                params.append(
                    nn.Parameter(torch.zeros(v_shape, dtype=cache_dtype), requires_grad=False)
                )

        self.past_key_values = nn.ParameterList(params)

    @staticmethod
    def _validate_hybrid_config(config: Qwen35InferenceConfig):
        nc = config.neuron_config
        unsupported = []
        if nc.is_block_kv_layout:
            unsupported.append("block KV layout")
        if getattr(nc, "kv_quant_config", None) is not None or getattr(nc, "kv_cache_quant", False):
            unsupported.append("KV cache quantization")
        if nc.enable_fused_speculation or nc.speculation_length > 0 or nc.is_medusa:
            unsupported.append("speculative decoding")
        if getattr(nc, "enable_eagle_speculation", False) or getattr(nc, "is_eagle_draft", False):
            unsupported.append("EAGLE speculation")
        if nc.flash_decoding_enabled:
            unsupported.append("flash decoding")
        if nc.attention_dp_degree > 1:
            unsupported.append("attention data parallelism")
        if nc.kv_cache_tiling:
            unsupported.append("KV cache tiling")
        if nc.padding_side != "right":
            unsupported.append("left padding")
        if nc.is_continuous_batching:
            unsupported.append("continuous batching")
        if unsupported:
            raise ValueError(
                "HybridDeltaNetCacheManager v1 does not support: "
                + ", ".join(unsupported)
            )

    def _is_deltanet_layer(self, idx: int) -> bool:
        return self.layer_types[idx] == "linear_attention"

    def get_seq_length(self, past_key_values=None):
        for idx, layer_type in enumerate(self.layer_types):
            if layer_type != "linear_attention":
                if past_key_values is None:
                    _, v_cache = self._fetch_cache(idx)
                elif len(past_key_values) == len(self.past_key_values):
                    v_cache = past_key_values[2 * idx + 1]
                else:
                    v_cache = past_key_values[idx][1]
                return v_cache.shape[2]
        return 0

    def get_deltanet_state_by_layer_id(self, idx, kvcache_buffer=None, seq_ids=None):
        recurrent_state, conv_state = self._fetch_cache(idx, kvcache_buffer)
        if seq_ids is not None:
            cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
            recurrent_state = torch.index_select(recurrent_state, dim=0, index=cache_idx)
            conv_state = torch.index_select(conv_state, dim=0, index=cache_idx)
        elif self.kv_cache_padding_size > 0:
            recurrent_state = recurrent_state[: -self.kv_cache_padding_size]
            conv_state = conv_state[: -self.kv_cache_padding_size]
        return recurrent_state, conv_state

    def get_cache(
        self,
        seq_len: int,
        skip_slice=False,
        kvcache_buffer=None,
        seq_ids=None,
        windowed_context_encoding_window_idx=-1,
        **kwargs,
    ):
        past_key_values = []
        for idx in range(len(self.past_key_values) // 2):
            if self._is_deltanet_layer(idx):
                past_key_values.append(
                    list(self.get_deltanet_state_by_layer_id(idx, kvcache_buffer, seq_ids))
                )
            else:
                past_key_values.append(
                    list(
                        self.get_kv_by_layer_id(
                            idx=idx,
                            skip_slice=skip_slice,
                            seq_len=seq_len,
                            kvcache_buffer=kvcache_buffer,
                            seq_ids=seq_ids,
                            windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                            **kwargs,
                        )
                    )
                )
        return past_key_values

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: torch.Tensor,
        position_ids: torch.Tensor,
        new_key_values: List[torch.Tensor],
        seq_len: int,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ):
        updated_cache = []
        for idx, kv_per_layer in enumerate(new_key_values):
            if self._is_deltanet_layer(idx):
                recurrent_state, conv_state = self.update_deltanet_state_by_layer_id(
                    idx=idx,
                    seq_ids=seq_ids,
                    state_per_layer=kv_per_layer,
                    kvcache_buffer=kvcache_buffer,
                )
            elif kwargs.get("qwen_chunked_prefill_update", False):
                recurrent_state, conv_state = self.update_qwen_chunked_kv_by_layer_id(
                    idx=idx,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    kv_per_layer=kv_per_layer,
                    kvcache_buffer=kvcache_buffer,
                    valid_mask=kwargs.get("qwen_chunked_valid_mask", None),
                )
            else:
                recurrent_state, conv_state = self.update_kv_by_layer_id(
                    idx=idx,
                    is_for_context_encoding=is_for_context_encoding,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    kv_per_layer=kv_per_layer,
                    seq_len=seq_len,
                    scatter_index=scatter_index,
                    kv_active_mask=kv_active_mask,
                    kvcache_buffer=kvcache_buffer,
                    windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                    **kwargs,
                )
            updated_cache.append(recurrent_state)
            updated_cache.append(conv_state)
        return updated_cache

    def update_qwen_chunked_kv_by_layer_id(
        self,
        idx: int,
        seq_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_per_layer: Tuple[torch.Tensor, torch.Tensor],
        kvcache_buffer=None,
        valid_mask=None,
    ):
        latest_k, latest_v = kv_per_layer
        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)
        latest_k = latest_k.to(k_cache.dtype)
        latest_v = latest_v.to(v_cache.dtype)

        if seq_ids is not None:
            cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
            selected_k = torch.index_select(k_cache, dim=0, index=cache_idx)
            selected_v = torch.index_select(v_cache, dim=0, index=cache_idx)
        else:
            cache_idx = None
            selected_k = k_cache[: latest_k.shape[0]]
            selected_v = v_cache[: latest_v.shape[0]]

        pos = position_ids.long()
        k_index = pos[:, None, :, None].expand_as(latest_k)
        v_index = pos[:, None, :, None].expand_as(latest_v)

        if valid_mask is not None:
            valid = valid_mask.to(torch.bool)[:, None, :, None]
            old_k = torch.gather(selected_k, dim=2, index=k_index)
            old_v = torch.gather(selected_v, dim=2, index=v_index)
            latest_k = torch.where(valid, latest_k, old_k)
            latest_v = torch.where(valid, latest_v, old_v)

        updated_k = torch.scatter(selected_k, dim=2, index=k_index, src=latest_k)
        updated_v = torch.scatter(selected_v, dim=2, index=v_index, src=latest_v)

        if cache_idx is not None:
            k_row_index = cache_idx.view(-1, 1, 1, 1).expand_as(updated_k)
            v_row_index = cache_idx.view(-1, 1, 1, 1).expand_as(updated_v)
            k_cache = torch.scatter(k_cache, dim=0, index=k_row_index, src=updated_k)
            v_cache = torch.scatter(v_cache, dim=0, index=v_row_index, src=updated_v)
            return k_cache, v_cache

        if updated_k.shape[0] == k_cache.shape[0]:
            return updated_k + k_cache * 0, updated_v + v_cache * 0

        pad_rows = k_cache.shape[0] - updated_k.shape[0]
        if pad_rows > 0:
            updated_k = torch.cat([updated_k, k_cache[updated_k.shape[0] :] * 0], dim=0)
            updated_v = torch.cat([updated_v, v_cache[updated_v.shape[0] :] * 0], dim=0)
        return updated_k + k_cache * 0, updated_v + v_cache * 0

    def update_deltanet_state_by_layer_id(
        self,
        idx: int,
        seq_ids: torch.Tensor,
        state_per_layer: Tuple[torch.Tensor, torch.Tensor],
        kvcache_buffer=None,
    ):
        latest_recurrent, latest_conv = state_per_layer
        recurrent_cache, conv_cache = self._fetch_cache(idx, kvcache_buffer)
        latest_recurrent = latest_recurrent.to(recurrent_cache.dtype)
        latest_conv = latest_conv.to(conv_cache.dtype)

        if latest_recurrent.shape[0] == recurrent_cache.shape[0] and seq_ids is None:
            return (
                latest_recurrent + recurrent_cache * 0,
                latest_conv + conv_cache * 0,
            )

        if seq_ids is not None:
            cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
            recurrent_index = cache_idx.view(-1, 1, 1, 1).expand_as(latest_recurrent)
            conv_index = cache_idx.view(-1, 1, 1).expand_as(latest_conv)
            recurrent_cache = torch.scatter(
                input=recurrent_cache,
                dim=0,
                index=recurrent_index,
                src=latest_recurrent,
            )
            conv_cache = torch.scatter(
                input=conv_cache,
                dim=0,
                index=conv_index,
                src=latest_conv,
            )
            return recurrent_cache, conv_cache

        pad_size = recurrent_cache.shape[0] - latest_recurrent.shape[0]
        if pad_size > 0:
            latest_recurrent = torch.cat(
                [latest_recurrent, recurrent_cache[latest_recurrent.shape[0] :] * 0],
                dim=0,
            )
            latest_conv = torch.cat(
                [latest_conv, conv_cache[latest_conv.shape[0] :] * 0],
                dim=0,
            )
        return latest_recurrent + recurrent_cache * 0, latest_conv + conv_cache * 0


class QwenHybridBlockKVCacheManager(BlockKVCacheManager):
    """Block KV manager that allocates real KV only for full-attention layers."""

    _LINEAR_PLACEHOLDER_SHAPE = (1, 1, 1, 1)

    def __init__(self, config: Qwen35InferenceConfig, num_kv_head, **kwargs):
        self.layer_types = list(config.layer_types)
        super().__init__(config, num_kv_head=num_kv_head, **kwargs)

        params = []
        for layer_type in self.layer_types:
            if layer_type == "full_attention":
                params.append(
                    nn.Parameter(
                        torch.zeros(self.k_shape, dtype=self.cache_dtype),
                        requires_grad=False,
                    )
                )
                params.append(
                    nn.Parameter(
                        torch.zeros(self.v_shape, dtype=self.cache_dtype),
                        requires_grad=False,
                    )
                )
            else:
                params.append(
                    nn.Parameter(
                        torch.zeros(
                            self._LINEAR_PLACEHOLDER_SHAPE,
                            dtype=self.cache_dtype,
                        ),
                        requires_grad=False,
                    )
                )
                params.append(
                    nn.Parameter(
                        torch.zeros(
                            self._LINEAR_PLACEHOLDER_SHAPE,
                            dtype=self.cache_dtype,
                        ),
                        requires_grad=False,
                    )
                )
        self.past_key_values = nn.ParameterList(params)

    def _is_attention_layer(self, idx: int) -> bool:
        return self.layer_types[idx] == "full_attention"

    def get_seq_length(self, past_key_values=None):
        for idx, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention":
                if past_key_values is None:
                    _, v_cache = self._fetch_cache(idx)
                elif len(past_key_values) == len(self.past_key_values):
                    v_cache = past_key_values[2 * idx + 1]
                else:
                    v_cache = past_key_values[idx][1]
                if v_cache.ndim >= 4 and v_cache.shape[1] == self.pa_block_size:
                    return self.pa_num_blocks * self.pa_block_size
                return v_cache.shape[2]
        return 0

    def get_cache(self, active_block_table=None, kvcache_buffer=None, **kwargs):
        past_key_values = []
        use_segmented_prefix_cte = (
            kwargs.get("is_for_context_encoding", False)
            and getattr(
                self.neuron_config,
                "prefix_cte_attention_backend",
                "attention_cte",
            )
            == "segmented_cte"
            and active_block_table is not None
            and getattr(active_block_table, "ndim", 0) > 1
        )
        for idx in range(len(self.past_key_values) // 2):
            if self._is_attention_layer(idx):
                if use_segmented_prefix_cte:
                    k_cache, v_cache = self.get_raw_kv_by_layer_id(
                        idx,
                        kvcache_buffer=kvcache_buffer,
                    )
                else:
                    k_cache, v_cache = self.get_kv_by_layer_id(
                        idx,
                        active_block_table,
                        kvcache_buffer=kvcache_buffer,
                        **kwargs,
                    )
            else:
                k_cache, v_cache = self._fetch_cache(
                    idx,
                    kvcache_buffer=kvcache_buffer,
                )
            past_key_values.append([k_cache, v_cache])
        return past_key_values

    def _is_raw_block_kv_pair(self, kv_per_layer: List[torch.Tensor]) -> bool:
        if len(kv_per_layer) != 2:
            return False
        k_cache, v_cache = kv_per_layer
        return (
            k_cache.ndim == 4
            and v_cache.ndim == 4
            and k_cache.shape[0] == self.pa_num_blocks + self._NUM_EXTRA_RESERVED_BLOCK
            and v_cache.shape[0] == self.pa_num_blocks + self._NUM_EXTRA_RESERVED_BLOCK
            and k_cache.shape[1] == self.pa_block_size
            and v_cache.shape[1] == self.pa_block_size
        )

    def update_cache(
        self,
        new_key_values: List[torch.Tensor],
        scatter_index=None,
        kvcache_buffer=None,
        **kwargs,
    ):
        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(new_key_values):
            if self._is_attention_layer(idx) and self._is_raw_block_kv_pair(
                kv_per_layer
            ):
                k_cache, v_cache = kv_per_layer
            elif self._is_attention_layer(idx):
                k_cache, v_cache = self.update_kv_by_layer_id(
                    idx=idx,
                    kv_per_layer=kv_per_layer,
                    scatter_index=scatter_index,
                    kvcache_buffer=kvcache_buffer,
                )
            else:
                k_cache, v_cache = self._fetch_cache(
                    idx,
                    kvcache_buffer=kvcache_buffer,
                )
                k_cache = k_cache * 1
                v_cache = v_cache * 1
            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)
        return updated_kv_cache


class HybridGDNCheckpointCache(nn.Module):
    """Bounded device-side GDN prefix checkpoint bank.

    Metadata owns prefix hashes, refcounts, and eviction. This module only owns
    recurrent/conv tensors addressed by checkpoint slot IDs supplied by the
    scheduler/request-prep path.
    """

    def __init__(self, config: Qwen35InferenceConfig):
        super().__init__()
        self.gdn_layer_ids = tuple(
            idx
            for idx, layer_type in enumerate(config.layer_types)
            if layer_type == "linear_attention"
        )
        if not self.gdn_layer_ids:
            raise ValueError("HybridGDNCheckpointCache requires GDN layers")
        self.layer_to_bank_index = {
            layer_id: bank_idx for bank_idx, layer_id in enumerate(self.gdn_layer_ids)
        }
        self.num_checkpoint_slots = int(config.max_gdn_checkpoint_slots)
        if self.num_checkpoint_slots <= 0:
            raise ValueError("max_gdn_checkpoint_slots must be positive")

        tp_degree = config.neuron_config.tp_degree
        if config.linear_num_value_heads % tp_degree != 0:
            raise ValueError("linear_num_value_heads must be divisible by tp_degree")
        if config.linear_num_key_heads % tp_degree != 0:
            raise ValueError("linear_num_key_heads must be divisible by tp_degree")

        self.local_num_value_heads = config.linear_num_value_heads // tp_degree
        self.local_num_key_heads = config.linear_num_key_heads // tp_degree
        self.key_dim = config.linear_key_head_dim
        self.value_dim = config.linear_value_head_dim
        self.conv_dim = (
            2 * self.local_num_key_heads * config.linear_key_head_dim
            + self.local_num_value_heads * config.linear_value_head_dim
        )
        self.conv_state_len = config.linear_conv_kernel_dim - 1
        self.recurrent_dtype = _torch_dtype_from_hybrid_cache_dtype(
            config.hybrid_recurrent_cache_dtype
        )
        self.conv_dtype = _torch_dtype_from_hybrid_cache_dtype(
            config.hybrid_conv_cache_dtype
        )

        recurrent_shape = (
            self.num_checkpoint_slots,
            self.local_num_value_heads,
            self.key_dim,
            self.value_dim,
        )
        conv_shape = (
            self.num_checkpoint_slots,
            self.conv_dim,
            self.conv_state_len,
        )
        self.recurrent_slots = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(recurrent_shape, dtype=self.recurrent_dtype),
                    requires_grad=False,
                )
                for _ in self.gdn_layer_ids
            ]
        )
        self.conv_slots = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(conv_shape, dtype=self.conv_dtype),
                    requires_grad=False,
                )
                for _ in self.gdn_layer_ids
            ]
        )

    @property
    def checkpoint_params(self):
        params = []
        for recurrent_slot, conv_slot in zip(self.recurrent_slots, self.conv_slots):
            params.append(recurrent_slot)
            params.append(conv_slot)
        return params

    def bytes_per_checkpoint_per_rank(self) -> int:
        recurrent_numel = (
            len(self.gdn_layer_ids)
            * self.local_num_value_heads
            * self.key_dim
            * self.value_dim
        )
        conv_numel = len(self.gdn_layer_ids) * self.conv_dim * self.conv_state_len
        recurrent_bytes = 4 if self.recurrent_dtype == torch.float32 else 2
        conv_bytes = 4 if self.conv_dtype == torch.float32 else 2
        return recurrent_numel * recurrent_bytes + conv_numel * conv_bytes

    def _safe_slot_ids(
        self,
        slot_ids: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        slot_ids = slot_ids.reshape(-1).long().clamp(
            min=0,
            max=self.num_checkpoint_slots - 1,
        )
        if batch_size is None:
            return slot_ids
        if slot_ids.shape[0] >= batch_size:
            return slot_ids[:batch_size]
        pad = torch.zeros(
            (batch_size - slot_ids.shape[0],),
            dtype=slot_ids.dtype,
            device=slot_ids.device,
        )
        return torch.cat([slot_ids, pad], dim=0)

    @staticmethod
    def _safe_bool_vector(
        mask: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = mask.reshape(-1).to(device=device, dtype=torch.bool)
        if mask.shape[0] >= batch_size:
            return mask[:batch_size]
        pad = torch.zeros(
            (batch_size - mask.shape[0],),
            dtype=torch.bool,
            device=device,
        )
        return torch.cat([mask, pad], dim=0)

    @staticmethod
    def _active_rows(
        state: torch.Tensor,
        seq_ids: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if seq_ids is not None and hasattr(seq_ids, "numel") and seq_ids.numel() > 0:
            safe_seq_ids = seq_ids.reshape(-1)[:batch_size].to(
                device=state.device,
                dtype=torch.long,
            )
            safe_seq_ids = safe_seq_ids.clamp(min=0, max=int(state.shape[0]) - 1)
            return torch.index_select(state, 0, safe_seq_ids)
        return state[:batch_size]

    def restore_to_active_rows(
        self,
        *,
        layers: nn.ModuleList,
        seq_ids: torch.Tensor | None,
        checkpoint_slot_ids: torch.Tensor | None,
        restore_mask: torch.Tensor | None,
        zero_inactive: bool = False,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]] | None:
        if checkpoint_slot_ids is None or restore_mask is None:
            return None
        batch_size = max(
            int(checkpoint_slot_ids.reshape(-1).shape[0]),
            int(restore_mask.reshape(-1).shape[0]),
        )
        if batch_size <= 0:
            return None
        slot_ids = self._safe_slot_ids(checkpoint_slot_ids, batch_size)
        restore_mask = self._safe_bool_vector(
            restore_mask,
            batch_size,
            slot_ids.device,
        )
        slot_ids = torch.where(restore_mask, slot_ids, torch.zeros_like(slot_ids))
        rec_mask = restore_mask.view(batch_size, 1, 1, 1)
        conv_mask = restore_mask.view(batch_size, 1, 1)

        restored = {}
        for bank_idx, layer_id in enumerate(self.gdn_layer_ids):
            linear_attn = layers[layer_id].linear_attn
            active_recurrent = self._active_rows(
                linear_attn.recurrent_state_buffer, seq_ids, batch_size
            )
            active_conv = self._active_rows(
                linear_attn.conv_state_buffer, seq_ids, batch_size
            )
            if zero_inactive:
                inactive_recurrent = torch.zeros_like(active_recurrent)
                inactive_conv = torch.zeros_like(active_conv)
            else:
                inactive_recurrent = active_recurrent
                inactive_conv = active_conv
            slot_recurrent = torch.index_select(
                self.recurrent_slots[bank_idx], 0, slot_ids
            ).to(active_recurrent.dtype)
            slot_conv = torch.index_select(self.conv_slots[bank_idx], 0, slot_ids).to(
                active_conv.dtype
            )
            _debug_qwen36_hybrid_gdn_state(
                "restore_slot_recurrent",
                slot_recurrent,
                layer_id=layer_id,
                bank_idx=bank_idx,
                slot_ids=slot_ids,
                mask=restore_mask,
                seq_ids=seq_ids,
            )
            _debug_qwen36_hybrid_gdn_state(
                "restore_slot_conv",
                slot_conv,
                layer_id=layer_id,
                bank_idx=bank_idx,
                slot_ids=slot_ids,
                mask=restore_mask,
                seq_ids=seq_ids,
            )
            restored_recurrent = torch.where(
                rec_mask, slot_recurrent, inactive_recurrent
            )
            restored_conv = torch.where(conv_mask, slot_conv, inactive_conv)
            _debug_qwen36_hybrid_gdn_state(
                "restore_active_recurrent",
                restored_recurrent,
                layer_id=layer_id,
                bank_idx=bank_idx,
                slot_ids=slot_ids,
                mask=restore_mask,
                seq_ids=seq_ids,
            )
            _debug_qwen36_hybrid_gdn_state(
                "restore_active_conv",
                restored_conv,
                layer_id=layer_id,
                bank_idx=bank_idx,
                slot_ids=slot_ids,
                mask=restore_mask,
                seq_ids=seq_ids,
            )
            restored[layer_id] = (restored_recurrent, restored_conv)
        return restored

    def commit_from_active_rows(
        self,
        *,
        layer_state_pairs: list[tuple[int, torch.Tensor, torch.Tensor]],
        seq_ids: torch.Tensor | None,
        checkpoint_slot_ids: torch.Tensor | None,
        commit_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if checkpoint_slot_ids is None or commit_mask is None:
            return self.identity_outputs()
        batch_size = max(
            int(checkpoint_slot_ids.reshape(-1).shape[0]),
            int(commit_mask.reshape(-1).shape[0]),
        )
        if batch_size <= 0:
            return self.identity_outputs()
        slot_ids = self._safe_slot_ids(checkpoint_slot_ids, batch_size)
        commit_mask = self._safe_bool_vector(
            commit_mask,
            batch_size,
            slot_ids.device,
        )
        slot_ids = torch.where(commit_mask, slot_ids, torch.zeros_like(slot_ids))
        rec_mask = commit_mask.view(batch_size, 1, 1, 1)
        conv_mask = commit_mask.view(batch_size, 1, 1)

        state_by_layer = {
            layer_id: (recurrent_state, conv_state)
            for layer_id, recurrent_state, conv_state in layer_state_pairs
        }

        def _commit_rows(slots, rows, row_mask):
            output = slots * 1
            slot_axis = torch.arange(
                slots.shape[0], dtype=slot_ids.dtype, device=slot_ids.device
            )
            broadcast_shape = (slots.shape[0],) + (1,) * (slots.ndim - 1)
            for row_idx in range(batch_size):
                write_mask = torch.logical_and(
                    row_mask[row_idx],
                    slot_axis == slot_ids[row_idx],
                ).view(broadcast_shape)
                row_value = rows[row_idx : row_idx + 1].expand_as(output)
                output = torch.where(write_mask, row_value, output)
            return output

        outputs = []
        for bank_idx, layer_id in enumerate(self.gdn_layer_ids):
            recurrent_slots = self.recurrent_slots[bank_idx]
            conv_slots = self.conv_slots[bank_idx]
            if layer_id not in state_by_layer:
                outputs.append(recurrent_slots * 1)
                outputs.append(conv_slots * 1)
                continue

            recurrent_state, conv_state = state_by_layer[layer_id]
            recurrent_rows = self._active_rows(recurrent_state, seq_ids, batch_size).to(
                recurrent_slots.dtype
            )
            conv_rows = self._active_rows(conv_state, seq_ids, batch_size).to(
                conv_slots.dtype
            )
            _debug_qwen36_hybrid_gdn_state(
                "commit_input_recurrent",
                recurrent_rows,
                layer_id=layer_id,
                bank_idx=bank_idx,
                slot_ids=slot_ids,
                mask=commit_mask,
                seq_ids=seq_ids,
            )
            _debug_qwen36_hybrid_gdn_state(
                "commit_input_conv",
                conv_rows,
                layer_id=layer_id,
                bank_idx=bank_idx,
                slot_ids=slot_ids,
                mask=commit_mask,
                seq_ids=seq_ids,
            )

            committed_recurrent = _commit_rows(
                recurrent_slots, recurrent_rows, commit_mask
            )
            committed_conv = _commit_rows(conv_slots, conv_rows, commit_mask)
            committed_recurrent_rows = torch.index_select(
                committed_recurrent, 0, slot_ids
            )
            committed_conv_rows = torch.index_select(committed_conv, 0, slot_ids)
            _debug_qwen36_hybrid_gdn_state(
                "commit_slot_recurrent",
                committed_recurrent_rows,
                layer_id=layer_id,
                bank_idx=bank_idx,
                slot_ids=slot_ids,
                mask=commit_mask,
                seq_ids=seq_ids,
            )
            _debug_qwen36_hybrid_gdn_state(
                "commit_slot_conv",
                committed_conv_rows,
                layer_id=layer_id,
                bank_idx=bank_idx,
                slot_ids=slot_ids,
                mask=commit_mask,
                seq_ids=seq_ids,
            )

            outputs.append(committed_recurrent)
            outputs.append(committed_conv)
        return outputs

    def identity_outputs(self) -> list[torch.Tensor]:
        return [param * 1 for param in self.checkpoint_params]


# ============================================================
# Model
# ============================================================


def _effective_lm_head_pad_size(lm_head, logits, config):
    pad_size = getattr(lm_head, "pad_size", None)
    if not pad_size:
        return pad_size

    if getattr(lm_head, "gather_output", False):
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is not None:
            return max(int(logits.shape[-1]) - int(vocab_size), 0)

    return pad_size


def _debug_tensor_minmax(tensor):
    if tensor is None or not hasattr(tensor, "numel") or tensor.numel() == 0:
        return "empty"
    flat = tensor.reshape(-1)
    return f"{int(flat.min().item())}:{int(flat.max().item())}"


def _debug_tensor_values(tensor, limit=8):
    if tensor is None or not hasattr(tensor, "numel") or tensor.numel() == 0:
        return []
    return tensor.reshape(-1)[:limit].tolist()


def _debug_tensor_shape(tensor):
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return tuple(tensor.shape)


def _normalize_qwen36_slot_mapping(slot_mapping, batch_size: int, active_tokens: int):
    if (
        slot_mapping is None
        or not hasattr(slot_mapping, "numel")
        or slot_mapping.numel() == 0
        or not hasattr(slot_mapping, "ndim")
    ):
        return slot_mapping
    if slot_mapping.ndim != 1:
        return slot_mapping

    batch_size = int(batch_size)
    active_tokens = int(active_tokens)
    total_slots = int(slot_mapping.numel())
    if batch_size > 0 and active_tokens > 0 and total_slots == batch_size * active_tokens:
        return slot_mapping.reshape(batch_size, active_tokens)
    if batch_size == 1:
        return slot_mapping.reshape(1, total_slots)
    return slot_mapping


def _use_legacy_tkg_args() -> bool:
    return os.environ.get("QWEN36_TKG_LEGACY_ARGS") == "1"


def _qwen36_config_flag(config, neuron_config, name: str, default: bool = False) -> bool:
    for owner in (config, neuron_config, getattr(config, "neuron_config", None)):
        value = getattr(owner, name, None)
        if value is not None:
            return bool(value)
    return bool(default)


def _use_expanded_hybrid_args_for_tag(config, tag: str) -> bool:
    if not _qwen36_config_flag(config, None, "use_hybrid_apc_manager"):
        return False
    # The legacy ABI experiment intentionally keeps both traced stages on the
    # older prefix-cache contract. Neuron prunes the extra CTE hybrid metadata
    # inputs from the serialized trace, so runtime must not send them either.
    if _use_legacy_tkg_args():
        return False
    if tag == CONTEXT_ENCODING_MODEL_TAG:
        return True
    if tag == TOKEN_GENERATION_MODEL_TAG:
        return True
    return False


def _qwen36_shape_entry_arg_count(entry) -> int | None:
    if isinstance(entry, str):
        try:
            entry = json.loads(entry)
        except Exception:
            return None
    if isinstance(entry, (list, tuple)):
        return len(entry)
    return None


def _qwen36_compiled_arg_count(model_wrapper) -> int | None:
    counts = []
    for owner in (
        model_wrapper,
        getattr(model_wrapper, "model", None),
        getattr(getattr(model_wrapper, "model", None), "nxd_model", None),
    ):
        shape_map = getattr(owner, "input_shape_map", None)
        keys = getattr(shape_map, "keys", None)
        if not callable(keys):
            continue
        try:
            iterable = keys()
        except Exception:
            continue
        for entry in iterable:
            count = _qwen36_shape_entry_arg_count(entry)
            if count is not None:
                counts.append(count)
    return max(counts) if counts else None


def _use_expanded_hybrid_args_for_wrapper(model_wrapper, tag: str) -> bool:
    compiled_arg_count = _qwen36_compiled_arg_count(model_wrapper)
    if compiled_arg_count is not None:
        return compiled_arg_count >= 29
    return _use_expanded_hybrid_args_for_tag(model_wrapper.config, tag)


def _qwen36_expected_arg_count(config, tag: str) -> int:
    return 29 if _use_expanded_hybrid_args_for_tag(config, tag) else 24


def _assert_qwen36_arg_count(stage: str, args, expected: int) -> None:
    actual = len(args)
    if actual != expected:
        raise RuntimeError(
            f"Qwen3.6 {stage} argument contract mismatch: "
            f"expected {expected} tensors, got {actual}"
        )


_QWEN36_PREFIX_ARG_NAMES = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "seq_ids",
    "sampling_params",
    "prev_hidden",
    "adapter_ids",
    "accepted_indices",
    "current_length",
    "medusa_mask",
    "scatter_index",
    "slot_mapping",
    "block_table",
    "num_queries",
    "computed_context_lens",
    "tile_q_indices",
    "tile_block_tables",
    "tile_masks",
    "inputs_embeds",
    "kv_cache",
    "active_mask",
)
_QWEN36_MROPE_VISION_ARG_NAMES = (
    "rotary_position_ids",
    "vision_embeddings",
    "vision_mask",
)
_QWEN36_HYBRID_APC_ARG_NAMES = (
    "hybrid_restore_slot_ids",
    "hybrid_restore_mask",
    "hybrid_restore_prefix_lens",
    "hybrid_commit_slot_ids",
    "hybrid_commit_mask",
)


def _empty_qwen36_arg():
    return torch.empty(0)


def _qwen36_arg_names(config, tag: str):
    names = list(_QWEN36_PREFIX_ARG_NAMES + _QWEN36_MROPE_VISION_ARG_NAMES)
    if _use_expanded_hybrid_args_for_tag(config, tag):
        names.extend(_QWEN36_HYBRID_APC_ARG_NAMES)
    return names


def _normalize_qwen36_prefix_args(prefix_args):
    args = list(prefix_args)
    if len(args) > len(_QWEN36_PREFIX_ARG_NAMES):
        raise RuntimeError(
            "Qwen3.6 prefix argument contract mismatch: "
            f"expected at most {len(_QWEN36_PREFIX_ARG_NAMES)} base tensors, "
            f"got {len(args)}"
        )
    while len(args) < len(_QWEN36_PREFIX_ARG_NAMES):
        args.append(_empty_qwen36_arg())
    return args


def _normalize_qwen36_hybrid_args(hybrid_args, batch_size):
    args = list(hybrid_args or ())
    while len(args) < len(_QWEN36_HYBRID_APC_ARG_NAMES):
        args.append(torch.zeros((batch_size,), dtype=torch.int32))
    if len(args) > len(_QWEN36_HYBRID_APC_ARG_NAMES):
        raise RuntimeError(
            "Qwen3.6 Hybrid APC argument contract mismatch: "
            f"expected {len(_QWEN36_HYBRID_APC_ARG_NAMES)} tensors, got {len(args)}"
        )
    return args


def _build_qwen36_stage_args(
    config,
    tag: str,
    prefix_args,
    mrope_position_ids,
    vision_embeddings,
    vision_mask,
    hybrid_args=None,
):
    args = _normalize_qwen36_prefix_args(prefix_args)
    args.extend([mrope_position_ids, vision_embeddings, vision_mask])
    if _use_expanded_hybrid_args_for_tag(config, tag):
        batch_size = args[0].shape[0]
        args.extend(_normalize_qwen36_hybrid_args(hybrid_args, batch_size))
    _assert_qwen36_arg_count(tag, args, _qwen36_expected_arg_count(config, tag))
    return args


def build_cte_args(
    config,
    prefix_args,
    mrope_position_ids,
    vision_embeddings,
    vision_mask,
    hybrid_args=None,
):
    return _build_qwen36_stage_args(
        config,
        CONTEXT_ENCODING_MODEL_TAG,
        prefix_args,
        mrope_position_ids,
        vision_embeddings,
        vision_mask,
        hybrid_args=hybrid_args,
    )


def build_tkg_args(
    config,
    prefix_args,
    mrope_position_ids,
    vision_embeddings,
    vision_mask,
    hybrid_args=None,
):
    return _build_qwen36_stage_args(
        config,
        TOKEN_GENERATION_MODEL_TAG,
        prefix_args,
        mrope_position_ids,
        vision_embeddings,
        vision_mask,
        hybrid_args=hybrid_args,
    )


def _debug_qwen36_arg_contract(stage: str, tag: str, config, args) -> None:
    if (
        os.environ.get("QWEN36_ARG_CONTRACT_DEBUG") != "1"
        and os.environ.get("QWEN36_HYBRID_APC_DEBUG") != "1"
    ):
        return

    names = _qwen36_arg_names(config, tag)
    print(
        f"[qwen36_arg_contract] stage={stage} tag={tag} argc={len(args)}",
        flush=True,
    )
    for idx, (name, value) in enumerate(zip(names, args)):
        shape = _debug_tensor_shape(value)
        dtype = getattr(value, "dtype", None)
        min_value = "empty"
        max_value = "empty"
        if value is not None and hasattr(value, "numel") and value.numel() > 0:
            try:
                flat = value.detach().reshape(-1) if hasattr(value, "detach") else value.reshape(-1)
                min_value = flat.min().item()
                max_value = flat.max().item()
            except Exception as exc:
                min_value = f"error:{type(exc).__name__}"
                max_value = f"error:{type(exc).__name__}"
        print(
            "[qwen36_arg_contract] "
            f"stage={stage} tag={tag} index={idx} name={name} "
            f"shape={shape} dtype={dtype} min={min_value} max={max_value}",
            flush=True,
        )


def _debug_qwen36_flat_values(value) -> str:
    if value is None:
        return "None"
    if not hasattr(value, "reshape"):
        return repr(value)
    try:
        flat = value.detach().reshape(-1) if hasattr(value, "detach") else value.reshape(-1)
        return repr(flat.tolist())
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def _debug_qwen36_hybrid_gdn_state(
    tag: str,
    tensor: torch.Tensor,
    *,
    layer_id: int,
    bank_idx: int,
    slot_ids: torch.Tensor,
    mask: torch.Tensor,
    seq_ids: torch.Tensor | None,
) -> None:
    if os.environ.get("QWEN36_HYBRID_GDN_STATE_DEBUG") != "1":
        return
    shape = _debug_tensor_shape(tensor)
    dtype = getattr(tensor, "dtype", None)
    total = 0
    finite_count = "error"
    nan_count = "error"
    posinf_count = "error"
    neginf_count = "error"
    max_abs = "error"
    mean_abs = "error"
    try:
        flat = tensor.detach().float().reshape(-1)
        total = int(flat.numel())
        if total > 0:
            finite = torch.isfinite(flat)
            finite_i = finite.to(torch.int32)
            finite_count = int(finite_i.sum().item())
            nan_count = int(torch.isnan(flat).to(torch.int32).sum().item())
            posinf_count = int(torch.isposinf(flat).to(torch.int32).sum().item())
            neginf_count = int(torch.isneginf(flat).to(torch.int32).sum().item())
            safe = torch.where(finite, flat, torch.zeros_like(flat)).abs()
            max_abs = float(safe.max().item())
            mean_abs = float((safe.sum() / max(finite_count, 1)).item())
        else:
            finite_count = 0
            nan_count = 0
            posinf_count = 0
            neginf_count = 0
            max_abs = "empty"
            mean_abs = "empty"
    except Exception as exc:
        finite_count = f"error:{type(exc).__name__}"
        nan_count = f"error:{type(exc).__name__}"
        posinf_count = f"error:{type(exc).__name__}"
        neginf_count = f"error:{type(exc).__name__}"
        max_abs = f"error:{type(exc).__name__}"
        mean_abs = f"error:{type(exc).__name__}"

    print(
        "[qwen36_hybrid_gdn_state] "
        f"tag={tag} layer={layer_id} bank={bank_idx} "
        f"slot_ids={_debug_qwen36_flat_values(slot_ids)} "
        f"mask={_debug_qwen36_flat_values(mask)} "
        f"seq_ids={_debug_qwen36_flat_values(seq_ids)} "
        f"shape={shape} dtype={dtype} finite={finite_count}/{total} "
        f"nan={nan_count} posinf={posinf_count} neginf={neginf_count} "
        f"max_abs={max_abs} mean_abs={mean_abs}",
        flush=True,
    )


def _validate_qwen36_tkg_input_ids(input_ids, vocab_size) -> None:
    if input_ids is None or not hasattr(input_ids, "numel") or input_ids.numel() == 0:
        raise ValueError("Qwen3.6 TKG input_ids must be a non-empty tensor")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "Qwen3.6 TKG input_ids must be int32 or int64, "
            f"got {input_ids.dtype}"
        )
    min_id = int(input_ids.min().item())
    max_id = int(input_ids.max().item())
    if min_id < 0:
        raise ValueError(f"Qwen3.6 TKG input_ids contains negative token id {min_id}")
    if vocab_size is not None and max_id >= int(vocab_size):
        raise ValueError(
            "Qwen3.6 TKG input_ids contains out-of-vocab token id "
            f"{max_id}; vocab_size={int(vocab_size)}"
        )


def _qwen36_query_lengths(full_context_lens, computed_context_lens) -> list[int] | None:
    if (
        full_context_lens is None
        or computed_context_lens is None
        or not hasattr(full_context_lens, "numel")
        or not hasattr(computed_context_lens, "numel")
        or full_context_lens.numel() == 0
        or computed_context_lens.numel() == 0
    ):
        return None
    full_values = full_context_lens.reshape(-1).to(torch.int64)
    computed_values = computed_context_lens.reshape(-1).to(torch.int64)
    count = min(int(full_values.numel()), int(computed_values.numel()))
    if count <= 0:
        return None
    return [
        max(0, int(full_values[idx].item()) - int(computed_values[idx].item()))
        for idx in range(count)
    ]


def _qwen36_prefill_has_incomplete_row(prefill_completion_state) -> bool:
    if prefill_completion_state is None:
        return False
    if hasattr(prefill_completion_state, "numel"):
        if prefill_completion_state.numel() == 0:
            return False
        return not bool(prefill_completion_state.reshape(-1).to(torch.bool).all().item())
    try:
        values = list(prefill_completion_state)
    except TypeError:
        return not bool(prefill_completion_state)
    return any(not bool(value) for value in values)


def _qwen36_hybrid_apc_mask_has_active_row(mask) -> bool:
    if mask is None:
        return False
    if hasattr(mask, "numel"):
        if mask.numel() == 0:
            return False
        try:
            return bool(mask.reshape(-1).to(torch.bool).any().item())
        except (RuntimeError, TypeError, ValueError):
            # If a non-empty control tensor cannot be inspected on the host, keep
            # the existing controls and avoid preparing the request twice.
            return True
    try:
        values = list(mask)
    except TypeError:
        return bool(mask)
    return any(bool(value) for value in values)


def _qwen36_hybrid_apc_controls_need_prepare(
    hybrid_restore_mask,
    hybrid_commit_mask,
) -> bool:
    return not (
        _qwen36_hybrid_apc_mask_has_active_row(hybrid_restore_mask)
        or _qwen36_hybrid_apc_mask_has_active_row(hybrid_commit_mask)
    )


def _qwen36_hybrid_apc_controls_materialized(
    hybrid_restore_mask,
    hybrid_restore_prefix_lens,
    hybrid_commit_mask,
) -> bool:
    return not _qwen36_hybrid_apc_controls_need_prepare(
        hybrid_restore_mask,
        hybrid_commit_mask,
    ) or _qwen36_hybrid_apc_mask_has_active_row(hybrid_restore_prefix_lens)


def _qwen36_is_prefill_request(
    input_ids,
    position_ids,
    *,
    full_context_lens=None,
    computed_context_lens=None,
    prefill_completion_state=None,
) -> bool:
    if _qwen36_prefill_has_incomplete_row(prefill_completion_state):
        return True

    query_lengths = _qwen36_query_lengths(full_context_lens, computed_context_lens)
    if (
        query_lengths is not None
        and len(query_lengths) > 1
        and input_ids.ndim >= 2
        and input_ids.shape[0] == 1
        and input_ids.shape[-1] == len(query_lengths)
    ):
        return any(query_len > 1 for query_len in query_lengths)

    # Warm prefix-cache suffixes may start at a nonzero position, but they are
    # still multi-token CTE requests. TKG must remain a one-token decode path.
    if input_ids.shape[-1] > 1:
        return True
    return position_ids.min().item() == 0


def _qwen36_deltanet_padding_mask(
    *,
    input_ids,
    inputs_embeds,
    attention_mask,
    padding_idx,
    is_for_context_encoding,
    hybrid_restore_mask=None,
    num_queries=None,
):
    if padding_idx is None:
        token_padding_mask = torch.ones(
            (*input_ids.shape, 1),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
    else:
        token_padding_mask = (
            (input_ids != padding_idx).unsqueeze(-1).to(inputs_embeds.dtype)
        )

    query_padding_mask = None
    if (
        is_for_context_encoding
        and num_queries is not None
        and hasattr(num_queries, "numel")
        and num_queries.numel() >= input_ids.shape[0]
    ):
        query_lens = num_queries.reshape(-1)[: input_ids.shape[0]].to(
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        positions = torch.arange(
            input_ids.shape[1],
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        query_padding_mask = (
            positions.unsqueeze(0) < query_lens.unsqueeze(1)
        ).unsqueeze(-1).to(inputs_embeds.dtype)

    if (
        is_for_context_encoding
        and query_padding_mask is not None
    ):
        deltanet_padding_mask = query_padding_mask
    elif (
        is_for_context_encoding
        and attention_mask is not None
        and attention_mask.ndim == 2
    ):
        attention_padding_mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        if attention_padding_mask.shape[1] == inputs_embeds.shape[1]:
            deltanet_padding_mask = attention_padding_mask
        else:
            deltanet_padding_mask = token_padding_mask
    else:
        deltanet_padding_mask = token_padding_mask

    if (
        is_for_context_encoding
        and hybrid_restore_mask is not None
        and hasattr(hybrid_restore_mask, "numel")
        and hybrid_restore_mask.numel() > 0
    ):
        restore_active = hybrid_restore_mask.reshape(-1).to(torch.bool)
        if restore_active.numel() < input_ids.shape[0]:
            restore_active = torch.cat(
                [
                    restore_active,
                    torch.zeros(
                        input_ids.shape[0] - restore_active.numel(),
                        dtype=torch.bool,
                        device=restore_active.device,
                    ),
                ],
                dim=0,
            )
        restore_active = restore_active[: input_ids.shape[0]].to(
            device=inputs_embeds.device
        ).view(-1, 1, 1)
        deltanet_padding_mask = torch.where(
            restore_active,
            token_padding_mask,
            deltanet_padding_mask,
        )
    return deltanet_padding_mask


def _qwen36_unpack_packed_decode_batch(
    *,
    input_ids,
    attention_mask,
    position_ids,
    seq_ids,
    adapter_ids,
    slot_mapping,
    full_context_lens,
    computed_context_lens,
):
    query_lengths = _qwen36_query_lengths(full_context_lens, computed_context_lens)
    if (
        query_lengths is None
        or len(query_lengths) <= 1
        or any(query_len > 1 for query_len in query_lengths)
        or input_ids.ndim < 2
        or input_ids.shape[0] != 1
        or input_ids.shape[-1] != len(query_lengths)
    ):
        return input_ids, attention_mask, position_ids, seq_ids, adapter_ids, slot_mapping

    batch_size = len(query_lengths)

    def _unpack_token_rows(value):
        if (
            value is not None
            and hasattr(value, "ndim")
            and value.ndim >= 2
            and value.shape[0] == 1
            and value.shape[1] == batch_size
        ):
            return value.reshape(batch_size, 1, *value.shape[2:]).contiguous()
        return value

    def _repair_batch_vector(value, *, fill_from_index: bool = False):
        if value is None or not hasattr(value, "numel") or value.numel() == 0:
            return value
        flattened = value.reshape(-1)
        if flattened.numel() == batch_size:
            return flattened
        if flattened.numel() == 1 and batch_size > 1:
            if fill_from_index:
                return torch.arange(
                    batch_size,
                    dtype=value.dtype,
                    device=value.device,
                )
            return flattened[:1].expand(batch_size).contiguous()
        return value

    input_ids = _unpack_token_rows(input_ids)
    position_ids = _unpack_token_rows(position_ids)
    slot_mapping = _unpack_token_rows(slot_mapping)
    if (
        attention_mask is not None
        and hasattr(attention_mask, "ndim")
        and attention_mask.ndim >= 2
        and attention_mask.shape[0] == 1
        and attention_mask.shape[1] == batch_size
        and computed_context_lens is not None
        and hasattr(computed_context_lens, "numel")
        and computed_context_lens.numel() >= batch_size
    ):
        context_lens = computed_context_lens.reshape(-1).to(torch.int64)[:batch_size]
        max_context_len = max(1, int(context_lens.max().item()))
        repaired_mask = torch.zeros(
            (batch_size, max_context_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        for row_idx, context_len in enumerate(context_lens):
            active_len = max(0, min(int(context_len.item()), max_context_len))
            if active_len:
                repaired_mask[row_idx, :active_len] = 1
        attention_mask = repaired_mask
    if (
        slot_mapping is not None
        and hasattr(slot_mapping, "ndim")
        and slot_mapping.ndim == 1
        and int(slot_mapping.numel()) == batch_size
    ):
        slot_mapping = slot_mapping.reshape(batch_size, 1).contiguous()
    seq_ids = _repair_batch_vector(seq_ids, fill_from_index=True)
    adapter_ids = _repair_batch_vector(adapter_ids)
    return input_ids, attention_mask, position_ids, seq_ids, adapter_ids, slot_mapping


def _qwen36_hashable_request_id(request_id: Any) -> Hashable:
    if isinstance(request_id, list):
        return tuple(request_id)
    try:
        hash(request_id)
    except TypeError:
        return repr(request_id)
    return request_id


def _qwen36_metadata_for_request(
    metadata_by_request_id,
    request_id,
) -> dict[str, Any] | None:
    if not isinstance(metadata_by_request_id, dict):
        return None
    normalized = _qwen36_hashable_request_id(request_id)
    metadata = metadata_by_request_id.get(normalized)
    if metadata is None and request_id is not None:
        metadata = metadata_by_request_id.get(str(request_id))
    return metadata if isinstance(metadata, dict) else None


def _qwen36_request_metadata_values(
    metadata_by_request_id,
    request_ids,
    key: str,
):
    if request_ids is None:
        return None
    if isinstance(request_ids, list):
        request_ids = tuple(request_ids)
    elif not isinstance(request_ids, tuple):
        request_ids = (request_ids,)

    values = []
    found = False
    for request_id in request_ids:
        metadata = _qwen36_metadata_for_request(metadata_by_request_id, request_id)
        value = metadata.get(key) if metadata is not None else None
        values.append(value)
        found = found or value is not None
    if not found:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _qwen36_request_ids_have_metadata(
    metadata_by_request_id,
    request_ids,
) -> bool:
    return any(
        _qwen36_request_metadata_values(
            metadata_by_request_id,
            request_ids,
            key,
        )
        is not None
        for key in (
            "cumulative_hashes_by_prefix_len",
            "attention_block_refs_by_prefix_len",
            "request_prefix_len",
            "vllm_attention_hit_len",
        )
    )


def _qwen36_select_vllm_hybrid_apc_request_ids(
    metadata_by_request_id,
    *request_id_groups,
):
    first_present = None
    for request_ids in request_id_groups:
        if request_ids is None:
            continue
        if first_present is None:
            first_present = request_ids
        if _qwen36_request_ids_have_metadata(metadata_by_request_id, request_ids):
            return request_ids
    return first_present


def _qwen36_flat_item_count(value: Any) -> int:
    if value is None:
        return 0
    if hasattr(value, "numel"):
        try:
            return int(value.reshape(-1).numel())
        except Exception:
            return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def _qwen36_pad_batch_repeat_first(value, target_batch):
    if value is None or not hasattr(value, "numel") or value.numel() == 0:
        return value
    if value.ndim == 0 or value.shape[0] >= target_batch:
        return value
    pad_n = target_batch - value.shape[0]
    return torch.cat([value, value[:1].expand(pad_n, *value.shape[1:])], dim=0)


def _qwen36_pad_batch_with_value(value, target_batch, fill_value):
    if value is None or not hasattr(value, "numel") or value.numel() == 0:
        return value
    if value.ndim == 0 or value.shape[0] >= target_batch:
        return value
    pad_shape = (target_batch - value.shape[0],) + tuple(value.shape[1:])
    pad = torch.full(pad_shape, fill_value, dtype=value.dtype, device=value.device)
    return torch.cat([value, pad], dim=0)


def _qwen36_pad_hybrid_restore_controls_for_dummy_cte_rows(
    restore_slot_ids,
    restore_mask,
    restore_prefix_lens,
    target_batch,
):
    return (
        _qwen36_pad_batch_with_value(restore_slot_ids, target_batch, 0),
        _qwen36_pad_batch_with_value(restore_mask, target_batch, 0),
        _qwen36_pad_batch_with_value(restore_prefix_lens, target_batch, 0),
    )


def _qwen36_update_state_rows_by_seq_ids(previous_state, new_rows, seq_ids):
    if (
        previous_state is None
        or new_rows is None
        or seq_ids is None
        or not hasattr(previous_state, "shape")
        or not hasattr(new_rows, "shape")
        or not hasattr(seq_ids, "numel")
        or previous_state.ndim != new_rows.ndim
        or previous_state.shape[1:] != new_rows.shape[1:]
        or previous_state.shape[0] <= 0
        or new_rows.shape[0] <= 0
        or seq_ids.numel() == 0
    ):
        return new_rows

    row_count = min(int(new_rows.shape[0]), int(seq_ids.reshape(-1).shape[0]))
    if row_count <= 0:
        return previous_state * 1

    output = previous_state * 1
    seq_ids_flat = seq_ids.reshape(-1)[:row_count].to(
        device=previous_state.device,
        dtype=torch.long,
    )
    slot_axis = torch.arange(
        int(previous_state.shape[0]),
        dtype=torch.long,
        device=previous_state.device,
    )
    broadcast_shape = (int(previous_state.shape[0]),) + (
        1,
    ) * (previous_state.ndim - 1)
    typed_rows = new_rows[:row_count].to(previous_state.dtype)
    for row_idx in range(row_count):
        seq_id = seq_ids_flat[row_idx]
        valid_seq = torch.logical_and(
            seq_id >= 0,
            seq_id < int(previous_state.shape[0]),
        )
        write_mask = torch.logical_and(valid_seq, slot_axis == seq_id).view(
            broadcast_shape
        )
        row_value = typed_rows[row_idx : row_idx + 1].expand_as(output)
        output = torch.where(write_mask, row_value, output)
    return output


def _qwen36_preserve_inactive_state_rows(new_state, previous_state, active_rows):
    if (
        new_state is None
        or previous_state is None
        or active_rows is None
        or not hasattr(new_state, "shape")
        or not hasattr(previous_state, "shape")
        or not hasattr(active_rows, "numel")
        or new_state.shape != previous_state.shape
        or active_rows.numel() == 0
    ):
        return new_state
    active_rows = active_rows.reshape(-1).to(device=new_state.device, dtype=torch.bool)
    row_count = min(int(active_rows.numel()), int(new_state.shape[0]))
    if row_count <= 0:
        return new_state
    if row_count < int(new_state.shape[0]):
        active_rows = torch.cat(
            [
                active_rows[:row_count],
                torch.ones(
                    int(new_state.shape[0]) - row_count,
                    dtype=torch.bool,
                    device=new_state.device,
                ),
            ],
            dim=0,
        )
    else:
        active_rows = active_rows[: int(new_state.shape[0])]
    view_shape = (int(new_state.shape[0]),) + (1,) * (new_state.ndim - 1)
    active_rows = active_rows.view(view_shape)
    return torch.where(active_rows, new_state, previous_state)


def _qwen36_active_state_rows(valid_mask_1d, seq_ids):
    if (
        valid_mask_1d is None
        or not hasattr(valid_mask_1d, "numel")
        or valid_mask_1d.numel() == 0
    ):
        return None
    active_rows = valid_mask_1d.squeeze(-1).to(torch.bool).any(dim=-1)
    if seq_ids is not None and hasattr(seq_ids, "numel") and seq_ids.numel() > 0:
        seq_active = seq_ids.reshape(-1).to(
            device=active_rows.device,
            dtype=torch.long,
        )[: active_rows.numel()] >= 0
        active_rows = active_rows & seq_active
    return active_rows


def _qwen36_request_ids_tuple(request_ids):
    if request_ids is None:
        return None
    if isinstance(request_ids, list):
        return tuple(request_ids)
    if isinstance(request_ids, tuple):
        return request_ids
    return (request_ids,)


def _qwen36_request_ids_from_hybrid_apc_records(records):
    if records is None:
        return None
    if isinstance(records, dict):
        records = (records,)
    elif isinstance(records, list):
        records = tuple(records)
    if not isinstance(records, tuple):
        return None
    request_ids = []
    for record in records:
        if not isinstance(record, dict):
            return None
        request_id = record.get("request_id")
        if request_id is None:
            return None
        request_ids.append(request_id)
    return tuple(request_ids) if request_ids else None


def _qwen36_max_seq_slots_for_request_ids(model, seq_ids, request_count):
    max_slots = int(request_count or 0)
    for owner in (
        model,
        getattr(model, "neuron_config", None),
        getattr(getattr(model, "context_encoding_model", None), "neuron_config", None),
        getattr(getattr(model, "token_generation_model", None), "neuron_config", None),
    ):
        for attr in ("batch_size", "max_batch_size", "max_num_seqs"):
            value = getattr(owner, attr, None)
            if value is None:
                continue
            try:
                max_slots = max(max_slots, int(value))
            except (TypeError, ValueError):
                pass
    if seq_ids is not None and hasattr(seq_ids, "numel") and seq_ids.numel() > 0:
        flat = seq_ids.reshape(-1)
        try:
            non_negative = flat[flat >= 0]
            if non_negative.numel() > 0:
                max_slots = max(max_slots, int(non_negative.max().item()) + 1)
        except Exception:
            pass
    return max(1, max_slots)


def _qwen36_stable_seq_ids_for_request_ids(model, seq_ids, request_ids):
    request_ids = _qwen36_request_ids_tuple(request_ids)
    if not request_ids:
        return seq_ids

    normalized_request_ids = tuple(
        _qwen36_hashable_request_id(request_id) for request_id in request_ids
    )
    if any(request_id is None for request_id in normalized_request_ids):
        return seq_ids

    slot_by_request = getattr(model, "_qwen36_hybrid_seq_slot_by_request", None)
    request_by_slot = getattr(model, "_qwen36_hybrid_request_by_seq_slot", None)
    if not isinstance(slot_by_request, dict) or not isinstance(request_by_slot, dict):
        slot_by_request = {}
        request_by_slot = {}
        setattr(model, "_qwen36_hybrid_seq_slot_by_request", slot_by_request)
        setattr(model, "_qwen36_hybrid_request_by_seq_slot", request_by_slot)

    max_slots = _qwen36_max_seq_slots_for_request_ids(
        model,
        seq_ids,
        len(normalized_request_ids),
    )
    active_request_ids = set(normalized_request_ids)
    for stale_slot, stale_owner in list(request_by_slot.items()):
        if stale_owner in active_request_ids:
            continue
        request_by_slot.pop(stale_slot, None)
        slot_by_request.pop(stale_owner, None)

    assigned_slots = []
    for request_id in normalized_request_ids:
        slot = slot_by_request.get(request_id)
        if slot is None or slot < 0 or slot >= max_slots:
            free_slots = [
                candidate
                for candidate in range(max_slots)
                if candidate not in request_by_slot
            ]
            if not free_slots:
                return seq_ids
            slot = free_slots[0]
            slot_by_request[request_id] = slot
            request_by_slot[slot] = request_id
        assigned_slots.append(slot)

    dtype = seq_ids.dtype if hasattr(seq_ids, "dtype") else torch.int32
    if seq_ids is not None and hasattr(seq_ids, "device"):
        device = seq_ids.device
    else:
        device = None
    kwargs = {"dtype": dtype}
    if device is not None:
        kwargs["device"] = device
    return torch.tensor(assigned_slots, **kwargs)


def _qwen36_select_vllm_hybrid_apc_request_ids_for_input(
    metadata_by_request_id,
    *,
    all_request_ids,
    new_request_ids,
    full_context_lens,
    computed_context_lens,
    prefill_completion_state,
):
    all_request_ids_tuple = _qwen36_request_ids_tuple(all_request_ids)
    logical_request_count = max(
        _qwen36_flat_item_count(full_context_lens),
        _qwen36_flat_item_count(computed_context_lens),
        _qwen36_flat_item_count(prefill_completion_state),
    )
    if (
        logical_request_count > 1
        and all_request_ids_tuple is not None
        and len(all_request_ids_tuple) == logical_request_count
    ):
        # Keep request identity aligned with the model row order. In mixed
        # cached/new prefill batches, scheduler "new" ids can be a strict
        # subset, but the metadata vectors still describe every model row.
        return all_request_ids_tuple
    return _qwen36_select_vllm_hybrid_apc_request_ids(
        metadata_by_request_id,
        new_request_ids,
        all_request_ids,
    )


def _qwen36_add_vllm_hybrid_apc_metadata(
    hybrid_apc_request_dict: dict[str, Any],
    *,
    request_ids,
    metadata_by_request_id,
) -> None:
    for key in (
        "cumulative_hashes_by_prefix_len",
        "attention_block_refs_by_prefix_len",
        "request_prefix_len",
        "vllm_attention_hit_len",
        "active_suffix_len",
        "full_input_ids",
    ):
        value = _qwen36_request_metadata_values(
            metadata_by_request_id,
            request_ids,
            key,
        )
        if value is not None:
            if key == "full_input_ids" and not isinstance(value, torch.Tensor):
                input_ids = hybrid_apc_request_dict.get("input_ids")
                dtype = (
                    input_ids.dtype
                    if isinstance(input_ids, torch.Tensor)
                    else torch.int64
                )
                device = (
                    input_ids.device
                    if isinstance(input_ids, torch.Tensor)
                    else None
                )
                value = torch.tensor([list(value)], dtype=dtype, device=device)
            hybrid_apc_request_dict[key] = value


def _debug_logits_stage(stage: str, tensor) -> None:
    if os.environ.get("QWEN36_LOGIT_STAGE_DEBUG") != "1":
        return
    if tensor is None or not hasattr(tensor, "numel"):
        print(
            f"[qwen36_logits_debug] stage={stage} tensor=none",
            flush=True,
        )
        return
    if tensor.numel() == 0:
        print(
            f"[qwen36_logits_debug] stage={stage} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} empty",
            flush=True,
        )
        return

    try:
        with torch.no_grad():
            flat = tensor.detach().reshape(-1)
            if torch.is_floating_point(flat):
                finite_mask = torch.isfinite(flat)
                finite_count = int(finite_mask.sum().item())
                nan_count = int(torch.isnan(flat).sum().item())
                posinf_count = int(
                    torch.logical_and(torch.isinf(flat), flat > 0).sum().item()
                )
                neginf_count = int(
                    torch.logical_and(torch.isinf(flat), flat < 0).sum().item()
                )
                if finite_count:
                    finite_flat = flat[finite_mask].float()
                    finite_min = float(finite_flat.min().item())
                    finite_max = float(finite_flat.max().item())
                else:
                    finite_min = "none"
                    finite_max = "none"
                print(
                    "[qwen36_logits_debug] "
                    f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                    f"device={tensor.device} numel={tensor.numel()} finite={finite_count} "
                    f"nan={nan_count} posinf={posinf_count} neginf={neginf_count} "
                    f"finite_min={finite_min} finite_max={finite_max}",
                    flush=True,
                )
            else:
                print(
                    "[qwen36_logits_debug] "
                    f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                    f"device={tensor.device} numel={tensor.numel()} "
                    f"minmax={_debug_tensor_minmax(tensor)}",
                    flush=True,
                )
    except Exception as exc:
        print(
            "[qwen36_logits_debug] "
            f"stage={stage} summary_error={type(exc).__name__}: {exc}",
            flush=True,
        )


def _qwen36_output_logits_for_return(logits, lm_head, neuron_config):
    if not (
        getattr(neuron_config, "output_logits", False)
        and getattr(neuron_config, "on_device_sampling_config", None) is not None
        and not getattr(lm_head, "gather_output", True)
    ):
        return logits
    return _gather_along_dim(
        logits,
        partition_dim=2,
        process_group=getattr(lm_head, "tensor_parallel_group", None),
    )


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

    def init_inference_optimization(self, config: Qwen35InferenceConfig):
        super().init_inference_optimization(config)
        if getattr(config, "use_hybrid_apc_manager", False):
            if getattr(config.neuron_config, "is_block_kv_layout", False):
                self.kv_mgr = QwenHybridBlockKVCacheManager(
                    config,
                    num_kv_head=self.num_key_value_heads,
                )
            self.hybrid_gdn_checkpoint_cache = HybridGDNCheckpointCache(config)
        elif getattr(config, "use_hybrid_cache_manager", False):
            self.kv_mgr = HybridDeltaNetCacheManager(
                config,
                num_kv_head=self.num_key_value_heads,
                global_rank=self.rank_util,
                attention_chunk_size=self.attention_chunk_size,
                sliding_window=self.sliding_window,
                windowed_context_encoding_size=self.windowed_context_encoding_size,
                layer_to_cache_size_mapping=self.layer_to_cache_size_mapping,
            )

    @property
    def _deltanet_state_params(self):
        """Return DeltaNet state nn.Parameters in alias order."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "linear_attn"):
                params.append(layer.linear_attn.recurrent_state_buffer)
                params.append(layer.linear_attn.conv_state_buffer)
        return params

    @property
    def _hybrid_gdn_checkpoint_params(self):
        if not hasattr(self, "hybrid_gdn_checkpoint_cache"):
            return []
        return self.hybrid_gdn_checkpoint_cache.checkpoint_params

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        """Scatter vision embeddings into text input embeddings at image-token
        positions, using exactly the Qwen3-VL upstream pattern.

        vision_embeddings: (1, seq_len, hidden). Real vision embeddings live in
            slots i < n_vis; pad slots i >= n_vis are ZEROS.
        vision_mask: (1, seq_len, 1) int32. Real slots (i < n_vis) hold image-token
            positions in [0, seq_len). Pad slots (i >= n_vis) hold seq_len-1
            (the last padding position of input_ids), so scatter writes zero to
            a single padding slot — safe because that position has
            attention_mask == 0.

        The scatter uses PyTorch index_put_(accumulate=False), matching
        `scatter_by_index_put` in neuronx_distributed_inference.models.llama4.
        """
        _, max_positions, embedding_dim = inputs_embeds.shape
        h_new = inputs_embeds.clone()
        vision_flat = vision_embeddings.reshape(-1, embedding_dim)
        positions_flat = vision_mask.reshape(-1)
        num_positions = positions_flat.shape[0]
        vision_flat = vision_flat[:num_positions]
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
        hybrid_restore_slot_ids=None,
        hybrid_restore_mask=None,
        hybrid_restore_prefix_lens=None,
        hybrid_commit_slot_ids=None,
        hybrid_commit_mask=None,
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
            if hasattr(self.kv_mgr, "get_seq_length"):
                past_key_values_length = self.kv_mgr.get_seq_length(past_key_values)
            else:
                past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # CRITICAL: Zero out embeddings for padding tokens so DeltaNet recurrence
        # is not polluted. DeltaNet has no attention mask -- it processes all
        # sequence positions through a linear recurrence.  Padding tokens have
        # real embedding vectors which corrupt the recurrence state.
        # The mask is [B, S, 1] float with 1.0 for real tokens, 0.0 for padding.
        deltanet_padding_mask = _qwen36_deltanet_padding_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            padding_idx=self.padding_idx,
            is_for_context_encoding=is_for_context_encoding,
            hybrid_restore_mask=hybrid_restore_mask,
            num_queries=kwargs.get("num_queries"),
        )
        if is_for_context_encoding:
            inputs_embeds = inputs_embeds * deltanet_padding_mask

        # Vision embedding injection. When use_text_only_cte_inputs=False we
        # always trace the scatter into the graph. The input generator makes
        # sure "dummy" (text-only) vision inputs are IDEMPOTENT: every pad slot
        # points to the same target position and carries the same value, so
        # scattering them repeatedly does not corrupt real embeddings.
        #
        # NOTE: the gate MUST be Python-static (trace-time) since the two
        # branches produce different graphs. `shape[1] != seq_length` is not a
        # reliable proxy because the input generator pads vision inputs to
        # seq_length; instead we look at the compile-time config flag.
        if (vision_embeddings is not None) and (vision_mask is not None):
            if vision_embeddings.dtype != self.config.neuron_config.torch_dtype:
                vision_embeddings = vision_embeddings.to(
                    self.config.neuron_config.torch_dtype
                )
            traced_with_vision = (
                not getattr(self.config, "use_text_only_cte_inputs", True)
                and vision_embeddings.ndim == 3
                and vision_mask.ndim == 3
                and vision_embeddings.shape[1] == seq_length
                and vision_mask.shape[1] == seq_length
            )
            if is_for_context_encoding and traced_with_vision:
                inputs_embeds = self.encode_vision_to_input(
                    inputs_embeds, vision_embeddings, vision_mask
                )
            elif is_for_context_encoding and vision_embeddings.numel() > 0:
                inputs_embeds = inputs_embeds + vision_embeddings.sum() * 0
                inputs_embeds = (
                    inputs_embeds + vision_mask.sum().to(inputs_embeds.dtype) * 0
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

        # Get KV cache for TKG and for model-local chunked CTE.
        use_qwen_chunked_prefill = (
            is_for_context_encoding
            and getattr(self.config, "use_qwen_hybrid_chunked_prefill", False)
        )
        active_block_table = kwargs.get("active_block_table", None)
        cte_has_prefix_blocks = (
            is_for_context_encoding
            and use_qwen_chunked_prefill
            and active_block_table is not None
            and getattr(active_block_table, "ndim", 0) > 1
        )
        cache_size = (
            self.config.neuron_config.seq_len
            if use_qwen_chunked_prefill
            else self.n_positions
        )
        if (not is_for_context_encoding) or cte_has_prefix_blocks:
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
        deltanet_layer_state_pairs = []
        cos_cache = None
        sin_cache = None
        restored_gdn_states = None
        if getattr(self.config, "use_hybrid_apc_manager", False) and hasattr(
            self, "hybrid_gdn_checkpoint_cache"
        ):
            if hybrid_restore_prefix_lens is not None and position_ids is not None:
                # Host-side request prep must set suffix position_ids to the
                # restored cumulative-prefix boundary. This is a no-op on
                # default zero masks, but it keeps the contract explicit.
                if (
                    not torch.jit.is_tracing()
                    and hybrid_restore_mask is not None
                    and bool(hybrid_restore_mask.to(torch.bool).any().item())
                ):
                    expected = hybrid_restore_prefix_lens.long()
                    observed = position_ids[:, 0].long()
                    if not torch.equal(observed, expected):
                        raise ValueError(
                            "hybrid APC restore prefix lens must match "
                            "position_ids[:, 0]"
                        )
            restored_gdn_states = (
                self.hybrid_gdn_checkpoint_cache.restore_to_active_rows(
                    layers=self.layers,
                    seq_ids=seq_ids,
                    checkpoint_slot_ids=hybrid_restore_slot_ids,
                    restore_mask=hybrid_restore_mask,
                    zero_inactive=(
                        is_for_context_encoding
                        and not _qwen36_hybrid_apc_mask_has_active_row(
                            hybrid_restore_prefix_lens
                        )
                    ),
                )
            )

        # Keep CTE masks compact on the Neuron paths. Qwen attention prefill
        # applies causal masking inside the attention kernel/path, while DeltaNet
        # consumes deltanet_padding_mask separately. Dense SxS masks are only a
        # small fallback path and are not viable for long-context CTE.
        use_compact_cte_attention_mask = getattr(
            self.config, "use_compact_cte_attention_mask", True
        )
        use_neuron_cte_attention = use_qwen_chunked_prefill or getattr(
            self.config.neuron_config, "is_block_kv_layout", False
        )
        # Convert 2D attention_mask to 4D causal mask for the small fallback path.
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and is_for_context_encoding
            and not use_compact_cte_attention_mask
            and not use_neuron_cte_attention
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
            if restored_gdn_states is not None and idx in restored_gdn_states:
                past_key_value = restored_gdn_states[idx]

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
                qwen_chunked_prefill_update=use_qwen_chunked_prefill,
                qwen_chunked_valid_mask=deltanet_padding_mask.squeeze(-1)
                if use_qwen_chunked_prefill
                else None,
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
                deltanet_layer_state_pairs.append(
                    (idx, deltanet_states[0], deltanet_states[1])
                )

        # Update KV cache
        if update_cache:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                qwen_chunked_prefill_update=use_qwen_chunked_prefill,
                qwen_chunked_valid_mask=deltanet_padding_mask.squeeze(-1)
                if use_qwen_chunked_prefill
                else None,
                **kwargs,
            )

        if getattr(self.config, "use_hybrid_apc_manager", False) and hasattr(
            self, "hybrid_gdn_checkpoint_cache"
        ):
            commit_during_tkg = bool(
                getattr(self.config, "hybrid_apc_commit_during_token_generation", False)
            )
            if not is_for_context_encoding and not commit_during_tkg:
                self._hybrid_gdn_checkpoint_updated_states = []
            else:
                self._hybrid_gdn_checkpoint_updated_states = (
                    self.hybrid_gdn_checkpoint_cache.commit_from_active_rows(
                        layer_state_pairs=deltanet_layer_state_pairs,
                        seq_ids=seq_ids,
                        checkpoint_slot_ids=hybrid_commit_slot_ids,
                        commit_mask=hybrid_commit_mask,
                    )
                )

        _debug_logits_stage("before_final_norm", hidden_states)
        hidden_states = self.norm(hidden_states)
        _debug_logits_stage("after_final_norm_full", hidden_states)

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
        hybrid_restore_slot_ids=None,
        hybrid_restore_mask=None,
        hybrid_restore_prefix_lens=None,
        hybrid_commit_slot_ids=None,
        hybrid_commit_mask=None,
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
        hybrid_restore_slot_ids = self.set_none_if_empty(hybrid_restore_slot_ids)
        hybrid_restore_mask = self.set_none_if_empty(hybrid_restore_mask)
        hybrid_restore_prefix_lens = self.set_none_if_empty(hybrid_restore_prefix_lens)
        hybrid_commit_slot_ids = self.set_none_if_empty(hybrid_commit_slot_ids)
        hybrid_commit_mask = self.set_none_if_empty(hybrid_commit_mask)

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
            hybrid_restore_slot_ids=hybrid_restore_slot_ids,
            hybrid_restore_mask=hybrid_restore_mask,
            hybrid_restore_prefix_lens=hybrid_restore_prefix_lens,
            hybrid_commit_slot_ids=hybrid_commit_slot_ids,
            hybrid_commit_mask=hybrid_commit_mask,
            num_queries=num_queries,
            computed_context_lens=computed_context_lens,
        )

        batch_size = input_ids.shape[0]
        if not getattr(self, "sliced_hidden", False):
            if not is_for_context_encoding:
                pass
            else:
                if getattr(self.config, "use_qwen_hybrid_chunked_prefill", False):
                    query_index = None
                    if (
                        num_queries is not None
                        and hasattr(num_queries, "numel")
                        and num_queries.numel() >= batch_size
                    ):
                        query_index = (
                            num_queries.reshape(-1)[:batch_size]
                            .to(device=input_ids.device, dtype=torch.long)
                            .view(batch_size, 1)
                            - 1
                        ).clamp(min=0)
                    token_index = None
                    if self.padding_idx is not None:
                        token_index = (
                            (input_ids != self.padding_idx)
                            .sum(dim=1, keepdim=True)
                            .long()
                            - 1
                        ).clamp(min=0)
                    if query_index is not None:
                        index = query_index
                    elif attention_mask is not None and attention_mask.ndim == 2:
                        attention_index = (
                            attention_mask.to(torch.long).sum(dim=1, keepdim=True)
                            - 1
                        ).clamp(min=0)
                        if (
                            hybrid_restore_mask is not None
                            and hasattr(hybrid_restore_mask, "numel")
                            and hybrid_restore_mask.numel() > 0
                        ):
                            restore_active = (
                                hybrid_restore_mask.reshape(-1).to(torch.bool).any()
                            )
                            index = torch.where(
                                restore_active,
                                token_index if token_index is not None else attention_index,
                                attention_index,
                            )
                        else:
                            index = attention_index
                    else:
                        index = (
                            token_index
                            if token_index is not None
                            else torch.full(
                                (batch_size, 1),
                                max(0, input_ids.shape[1] - 1),
                                dtype=torch.long,
                                device=input_ids.device,
                            )
                        )
                else:
                    index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        _debug_logits_stage("after_final_norm", hidden_states)
        _debug_logits_stage("selected_hidden_before_lm_head", hidden_states)
        _debug_logits_stage("lm_head_weight", getattr(self.lm_head, "weight", None))
        logits = self.lm_head(hidden_states)
        _debug_logits_stage("after_lm_head_pre_float", logits)
        logits = logits.float()
        _debug_logits_stage("after_lm_head", logits)

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
                logits,
                rank_id,
                world_size,
                pad_size=_effective_lm_head_pad_size(
                    self.lm_head, logits, self.config
                ),
            )
            _debug_logits_stage("after_mask_padded_logits", logits)

        if self.on_device_sampling:
            res = self._sample_on_device(
                logits, sampling_params, False, is_for_context_encoding
            )
        else:
            res = logits

        _debug_logits_stage("before_return_logits", logits)
        outputs = [res]
        if self.neuron_config.output_logits and self.on_device_sampling:
            outputs += [
                _qwen36_output_logits_for_return(
                    logits,
                    self.lm_head,
                    self.neuron_config,
                )
            ]
        _qwen36_validate_alias_output_counts(
            self,
            updated_kv_cache,
            is_for_context_encoding=is_for_context_encoding,
        )
        outputs += updated_kv_cache

        # Append DeltaNet state tensors (for input_output_aliases)
        if (
            not getattr(self.config, "use_hybrid_cache_manager", False)
            and hasattr(self, "_deltanet_updated_states")
        ):
            outputs += self._deltanet_updated_states
        if (
            getattr(self.config, "use_hybrid_apc_manager", False)
            and hasattr(self, "_hybrid_gdn_checkpoint_updated_states")
        ):
            outputs += self._hybrid_gdn_checkpoint_updated_states

        return outputs


# ============================================================
# State Dict Converter (Dense -- no MoE weight handling)
# ============================================================


_QWEN36_FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
)


def _qwen36_cat(tensors, dim=0):
    """Concatenate tensors, including FP8 tensors on builds without FP8 cat."""
    if tensors and tensors[0].dtype in _QWEN36_FP8_DTYPES:
        return torch.cat(
            [tensor.contiguous().view(torch.int8) for tensor in tensors],
            dim=dim,
        ).view(tensors[0].dtype)
    return torch.cat(tensors, dim=dim)


def convert_qwen35_hf_to_neuron_state_dict(neuron_state_dict, config):
    """Convert HF Qwen3.5/3.6-27B weights to NxDI format.

    Weight mappings per layer type:

    DeltaNet layers (linear_attention):
      HF: layers.X.linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
          conv1d, A_log, dt_bias, norm, out_proj}
      NxDI: projections keep names; conv1d/A_log/dt_bias are remapped into
            ColumnParallelLinear parameter containers so NxD can shard them.

    Full attention layers:
      HF: layers.X.self_attn.q_proj.weight: (12288, 5120) -- doubled for gate
      NxDI: layers.X.self_attn.Wqkv.weight (fused Q+K+V, gate separated)
             layers.X.self_attn.output_gate_proj.weight (gate part)
      HF: layers.X.self_attn.{k_proj, v_proj, o_proj, q_norm, k_norm}
      NxDI: layers.X.self_attn.{..., q_layernorm, k_layernorm}

    Dense MLP (all layers):
      HF: layers.X.mlp.{gate_proj, up_proj, down_proj}.weight
      NxDI: layers.X.mlp.{gate_proj, up_proj, down_proj}.weight (same names)

    FP8 quantized checkpoints carry one scale tensor next to each quantized
    weight. NxDI normalizes saved ``.weight_scale`` keys to model ``.scale``
    keys before this converter runs, so any Qwen-specific weight split/reorder/
    fusion below must apply the same transformation to the matching scale.
    """
    # Add rank_util
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0,
        config.neuron_config.tp_degree,
        dtype=torch.int32,
    )

    # ── Qwen3.5-MoE weight renames (only fires for MoE variants) ──────────
    # HF stores per-layer MoE weights as:
    #   mlp.gate.weight                       — router  (num_experts, hidden)
    #   mlp.experts.gate_up_proj              — routed  (num_experts, 2*I, H)
    #   mlp.experts.down_proj                 — routed  (num_experts, H, I)
    #   mlp.shared_expert.{gate,up,down}_proj.weight
    #   mlp.shared_expert_gate.weight
    # Our Qwen35MoEBlock uses:
    #   mlp.moe.router.linear_router.weight
    #   mlp.moe.expert_mlps.mlp_op.gate_up_proj.weight
    #   mlp.moe.expert_mlps.mlp_op.down_proj.weight
    #   mlp.shared_{gate,up,down}_proj.weight
    #   mlp.shared_expert_gate.weight  (unchanged)
    if getattr(config, "_is_moe", False):
        for l in range(config.num_hidden_layers):
            p = f"layers.{l}."
            # Transpose stacked-expert weights: HF stores
            #   gate_up_proj: (num_experts, 2*I, H)
            #   down_proj:    (num_experts, H, I)
            # NxDI's ExpertMLPsV2 stores
            #   gate_up_proj: (num_experts, H, 2*I)
            #   down_proj:    (num_experts, I, H)
            gu_key = p + "mlp.experts.gate_up_proj"
            if gu_key in neuron_state_dict:
                w = neuron_state_dict[gu_key]
                if w.ndim == 3:
                    neuron_state_dict[gu_key] = w.transpose(1, 2).contiguous()
            dp_key = p + "mlp.experts.down_proj"
            if dp_key in neuron_state_dict:
                w = neuron_state_dict[dp_key]
                if w.ndim == 3:
                    neuron_state_dict[dp_key] = w.transpose(1, 2).contiguous()

            renames = {
                p + "mlp.gate.weight":
                    p + "mlp.moe.router.linear_router.weight",
                p + "mlp.experts.gate_up_proj":
                    p + "mlp.moe.expert_mlps.mlp_op.gate_up_proj.weight",
                p + "mlp.experts.down_proj":
                    p + "mlp.moe.expert_mlps.mlp_op.down_proj.weight",
                p + "mlp.shared_expert.gate_proj.weight":
                    p + "mlp.shared_gate_proj.weight",
                p + "mlp.shared_expert.up_proj.weight":
                    p + "mlp.shared_up_proj.weight",
                p + "mlp.shared_expert.down_proj.weight":
                    p + "mlp.shared_down_proj.weight",
                # shared_expert_gate name is already correct
            }
            for old, new in renames.items():
                if old in neuron_state_dict:
                    neuron_state_dict[new] = neuron_state_dict.pop(old)

    def _reorder_deltanet_qkv_for_tp(qkv_weight: torch.Tensor) -> torch.Tensor:
        """Pack [Q_all | K_all | V_all] into per-rank Q/K/V blocks.

        ColumnParallelLinear slices the first dimension contiguously.  DeltaNet
        needs each rank to receive its local query, key, and value heads
        together, so the full HF tensor is repacked as:
        [rank0 Q | rank0 K | rank0 V | rank1 Q | rank1 K | rank1 V | ...].
        """
        tp_degree = config.neuron_config.tp_degree
        num_k_heads = config.linear_num_key_heads
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        if num_k_heads % tp_degree != 0:
            raise ValueError(
                f"linear_num_key_heads={num_k_heads} must be divisible by tp_degree={tp_degree}"
            )
        if num_v_heads % tp_degree != 0:
            raise ValueError(
                f"linear_num_value_heads={num_v_heads} must be divisible by tp_degree={tp_degree}"
            )

        key_dim = num_k_heads * head_k_dim
        value_dim = num_v_heads * head_v_dim
        q_weight = qkv_weight[:key_dim].reshape(num_k_heads, head_k_dim, -1)
        k_weight = qkv_weight[key_dim : 2 * key_dim].reshape(num_k_heads, head_k_dim, -1)
        v_weight = qkv_weight[2 * key_dim : 2 * key_dim + value_dim].reshape(
            num_v_heads, head_v_dim, -1
        )
        local_k_heads = num_k_heads // tp_degree
        local_v_heads = num_v_heads // tp_degree
        blocks = []
        for rank in range(tp_degree):
            blocks.append(
                q_weight[
                    rank * local_k_heads : (rank + 1) * local_k_heads
                ].reshape(-1, qkv_weight.shape[1])
            )
            blocks.append(
                k_weight[
                    rank * local_k_heads : (rank + 1) * local_k_heads
                ].reshape(-1, qkv_weight.shape[1])
            )
            blocks.append(
                v_weight[
                    rank * local_v_heads : (rank + 1) * local_v_heads
                ].reshape(-1, qkv_weight.shape[1])
            )
        return _qwen36_cat(blocks, dim=0).contiguous()

    def _reorder_deltanet_qkv_channels_for_tp(channel_tensor: torch.Tensor) -> torch.Tensor:
        """Repack a first-dimension Q/K/V channel tensor into TP rank blocks."""
        tp_degree = config.neuron_config.tp_degree
        num_k_heads = config.linear_num_key_heads
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        key_dim = num_k_heads * head_k_dim
        value_dim = num_v_heads * head_v_dim
        q_tensor = channel_tensor[:key_dim]
        k_tensor = channel_tensor[key_dim : 2 * key_dim]
        v_tensor = channel_tensor[2 * key_dim : 2 * key_dim + value_dim]
        local_key_dim = key_dim // tp_degree
        local_value_dim = value_dim // tp_degree
        blocks = []
        for rank in range(tp_degree):
            blocks.append(q_tensor[rank * local_key_dim : (rank + 1) * local_key_dim])
            blocks.append(k_tensor[rank * local_key_dim : (rank + 1) * local_key_dim])
            blocks.append(
                v_tensor[rank * local_value_dim : (rank + 1) * local_value_dim]
            )
        return _qwen36_cat(blocks, dim=0).contiguous()

    def _split_interleaved_q_proj_tensor(
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split interleaved Qwen q_proj tensor into query and output gate."""
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        trailing_shape = tensor.shape[1:]
        tensor = tensor.reshape(num_heads, head_dim * 2, *trailing_shape)
        query_tensor = tensor[:, :head_dim, ...].reshape(
            num_heads * head_dim,
            *trailing_shape,
        )
        gate_tensor = tensor[:, head_dim:, ...].reshape(
            num_heads * head_dim,
            *trailing_shape,
        )
        return query_tensor.contiguous(), gate_tensor.contiguous()

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

        # === DeltaNet layers ===
        if layer_type == "linear_attention":
            qkv_key = f"layers.{l}.linear_attn.in_proj_qkv.weight"
            if qkv_key in neuron_state_dict and config.neuron_config.tp_degree > 1:
                neuron_state_dict[qkv_key] = _reorder_deltanet_qkv_for_tp(
                    neuron_state_dict[qkv_key]
                )
            qkv_scale_key = f"layers.{l}.linear_attn.in_proj_qkv.scale"
            if qkv_scale_key in neuron_state_dict and config.neuron_config.tp_degree > 1:
                neuron_state_dict[qkv_scale_key] = _reorder_deltanet_qkv_channels_for_tp(
                    neuron_state_dict[qkv_scale_key]
                )

            conv_key = f"layers.{l}.linear_attn.conv1d.weight"
            conv_weight_key = f"layers.{l}.linear_attn.conv1d_weight.weight"
            conv_scale_key = f"layers.{l}.linear_attn.conv1d.scale"
            conv_weight_scale_key = f"layers.{l}.linear_attn.conv1d_weight.scale"
            if conv_key in neuron_state_dict:
                conv_weight = neuron_state_dict.pop(conv_key)
                if config.neuron_config.tp_degree > 1:
                    conv_weight = _reorder_deltanet_qkv_channels_for_tp(conv_weight)
                neuron_state_dict[conv_weight_key] = conv_weight.squeeze(1).contiguous()
            if conv_scale_key in neuron_state_dict:
                conv_scale = neuron_state_dict.pop(conv_scale_key)
                if config.neuron_config.tp_degree > 1:
                    conv_scale = _reorder_deltanet_qkv_channels_for_tp(conv_scale)
                neuron_state_dict[conv_weight_scale_key] = conv_scale.contiguous()

            for vector_name in ("A_log", "dt_bias"):
                vector_key = f"layers.{l}.linear_attn.{vector_name}"
                vector_weight_key = f"layers.{l}.linear_attn.{vector_name}_weight.weight"
                if vector_key in neuron_state_dict:
                    neuron_state_dict[vector_weight_key] = (
                        neuron_state_dict.pop(vector_key).reshape(-1, 1).contiguous()
                    )

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

            # q_proj is doubled: (12288, 5120) = (num_heads * head_dim * 2, hidden)
            # INTERLEAVED: [head0_query(256) | head0_gate(256) | head1_query(256) | ...]
            q_proj_key = f"layers.{l}.self_attn.q_proj.weight"
            q_proj_scale_key = f"layers.{l}.self_attn.q_proj.scale"
            if q_proj_key in neuron_state_dict:
                q_proj_w = neuron_state_dict.pop(q_proj_key)
                query_w, gate_w = _split_interleaved_q_proj_tensor(q_proj_w)

                neuron_state_dict[q_proj_key] = query_w
                neuron_state_dict[f"layers.{l}.self_attn.output_gate_proj.weight"] = (
                    gate_w
                )
                if q_proj_scale_key in neuron_state_dict:
                    q_proj_scale = neuron_state_dict.pop(q_proj_scale_key)
                    query_scale, gate_scale = _split_interleaved_q_proj_tensor(
                        q_proj_scale
                    )
                    neuron_state_dict[q_proj_scale_key] = query_scale
                    neuron_state_dict[f"layers.{l}.self_attn.output_gate_proj.scale"] = (
                        gate_scale
                    )

            # Fuse QKV
            if config.neuron_config.fused_qkv:
                q_key = f"layers.{l}.self_attn.q_proj.weight"
                k_key = f"layers.{l}.self_attn.k_proj.weight"
                v_key = f"layers.{l}.self_attn.v_proj.weight"
                gate_key = f"layers.{l}.self_attn.output_gate_proj.weight"
                pack_gate_in_qkv = bool(
                    getattr(config, "use_qwen_qkv_gate_packed", False)
                )
                if q_key in neuron_state_dict:
                    qkv_weight_parts = [neuron_state_dict[q_key]]
                    if pack_gate_in_qkv:
                        if gate_key not in neuron_state_dict:
                            raise ValueError(
                                f"Missing output-gate tensor for packed QKV: {gate_key}"
                            )
                        qkv_weight_parts.append(neuron_state_dict[gate_key])
                    qkv_weight_parts.extend(
                        [neuron_state_dict[k_key], neuron_state_dict[v_key]]
                    )
                    neuron_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = _qwen36_cat(
                        qkv_weight_parts
                    )
                    q_scale_key = f"layers.{l}.self_attn.q_proj.scale"
                    gate_scale_key = f"layers.{l}.self_attn.output_gate_proj.scale"
                    k_scale_key = f"layers.{l}.self_attn.k_proj.scale"
                    v_scale_key = f"layers.{l}.self_attn.v_proj.scale"
                    scale_keys = [q_scale_key]
                    if pack_gate_in_qkv:
                        scale_keys.append(gate_scale_key)
                    scale_keys.extend([k_scale_key, v_scale_key])
                    scale_keys_present = [key in neuron_state_dict for key in scale_keys]
                    if any(scale_keys_present):
                        if not all(scale_keys_present):
                            missing = [
                                key
                                for key, present in zip(scale_keys, scale_keys_present)
                                if not present
                            ]
                            raise ValueError(
                                f"Missing FP8 fused-QKV scale tensor(s): {missing}"
                            )
                        neuron_state_dict[f"layers.{l}.self_attn.Wqkv.scale"] = _qwen36_cat(
                            [neuron_state_dict[key] for key in scale_keys]
                        )
                        del neuron_state_dict[q_scale_key]
                        del neuron_state_dict[k_scale_key]
                        del neuron_state_dict[v_scale_key]
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


def _reassert_hybrid_gdn_checkpoint_param_dtypes(module):
    config = getattr(module, "config", None)
    if config is None:
        return

    recurrent_dtype = _torch_dtype_from_hybrid_cache_dtype(
        getattr(config, "hybrid_recurrent_cache_dtype", "float32")
    )
    conv_dtype = _torch_dtype_from_hybrid_cache_dtype(
        getattr(config, "hybrid_conv_cache_dtype", "bfloat16")
    )

    def _retarget(params, dtype):
        for param in params:
            if param.dtype != dtype:
                param.data = param.data.to(dtype)

    for layer in getattr(module, "layers", []):
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        recurrent_buffer = getattr(linear_attn, "recurrent_state_buffer", None)
        conv_buffer = getattr(linear_attn, "conv_state_buffer", None)
        if recurrent_buffer is not None and recurrent_buffer.dtype != recurrent_dtype:
            recurrent_buffer.data = recurrent_buffer.data.to(recurrent_dtype)
        if conv_buffer is not None and conv_buffer.dtype != conv_dtype:
            conv_buffer.data = conv_buffer.data.to(conv_dtype)

    cache = getattr(module, "hybrid_gdn_checkpoint_cache", None)
    if cache is not None:
        _retarget(cache.recurrent_slots, recurrent_dtype)
        _retarget(cache.conv_slots, conv_dtype)
        cache.recurrent_dtype = recurrent_dtype
        cache.conv_dtype = conv_dtype


def _qwen36_is_context_encoding_trace(
    n_active_tokens: int | None,
    neuron_config,
) -> bool:
    n_active_tokens = int(n_active_tokens or 0)
    speculation_length = getattr(neuron_config, "speculation_length", None)
    return n_active_tokens != 1 and not (
        speculation_length is not None and n_active_tokens == speculation_length
    )


def _qwen36_include_hybrid_gdn_checkpoint_outputs(
    config,
    *,
    is_for_context_encoding: bool | None = None,
    n_active_tokens: int | None = None,
    neuron_config=None,
) -> bool:
    if is_for_context_encoding is None:
        is_for_context_encoding = _qwen36_is_context_encoding_trace(
            n_active_tokens,
            neuron_config,
        )
    if not getattr(config, "use_hybrid_apc_manager", False):
        return True
    if is_for_context_encoding:
        return True
    return bool(getattr(config, "hybrid_apc_commit_during_token_generation", False))


def _qwen36_validate_alias_output_counts(
    module,
    updated_kv_cache,
    *,
    is_for_context_encoding: bool,
):
    kv_mgr = getattr(module, "kv_mgr", None)
    if kv_mgr is not None:
        expected_kv = len(kv_mgr.past_key_values)
    else:
        expected_kv = 0
    actual_kv = len(updated_kv_cache)
    if actual_kv != expected_kv:
        raise RuntimeError(
            "Qwen3.6 output alias count mismatch: "
            f"updated_kv_cache has {actual_kv} tensors but kv_mgr.past_key_values "
            f"has {expected_kv}"
        )

    expected_states = 0
    if not getattr(module.config, "use_hybrid_cache_manager", False):
        expected_states = len(getattr(module, "_deltanet_state_params", []))
    actual_states = len(getattr(module, "_deltanet_updated_states", []))
    if actual_states != expected_states:
        raise RuntimeError(
            "Qwen3.6 output alias count mismatch: "
            f"_deltanet_updated_states has {actual_states} tensors but "
            f"_deltanet_state_params has {expected_states}"
        )

    checkpoint_outputs_expected = _qwen36_include_hybrid_gdn_checkpoint_outputs(
        module.config,
        is_for_context_encoding=is_for_context_encoding,
    )
    expected_checkpoints = (
        len(getattr(module, "_hybrid_gdn_checkpoint_params", []))
        if checkpoint_outputs_expected
        else 0
    )
    actual_checkpoints = len(
        getattr(module, "_hybrid_gdn_checkpoint_updated_states", [])
    )
    if actual_checkpoints != expected_checkpoints:
        raise RuntimeError(
            "Qwen3.6 output alias count mismatch: "
            f"_hybrid_gdn_checkpoint_updated_states has {actual_checkpoints} tensors "
            f"but _hybrid_gdn_checkpoint_params expects {expected_checkpoints}"
        )


class Qwen35DecoderModelInstance(DecoderModelInstance):
    """Custom DecoderModelInstance that adds DeltaNet state buffers to input_output_aliases."""

    def load_module(self):
        super().load_module()
        _reassert_hybrid_gdn_checkpoint_param_dtypes(self.module)

    @staticmethod
    def _num_trace_outputs_before_aliases(neuron_config):
        if (
            getattr(neuron_config, "output_logits", False)
            and getattr(neuron_config, "on_device_sampling_config", None) is not None
        ):
            return 2
        return 1

    def get(self, bucket_rank, **kwargs):
        """Override to add DeltaNet state aliases after KV cache aliases."""
        module, input_output_aliases = super().get(bucket_rank, **kwargs)

        num_output_from_trace = self._num_trace_outputs_before_aliases(
            self.neuron_config
        )
        base_num_output_from_trace = 1 if not self.neuron_config.output_logits else 2
        if num_output_from_trace != base_num_output_from_trace:
            alias_shift = base_num_output_from_trace - num_output_from_trace
            for param in list(input_output_aliases.keys()):
                input_output_aliases[param] -= alias_shift

        if module.kv_mgr is not None:
            num_kv = len(module.kv_mgr.past_key_values)
        else:
            num_kv = 0

        state_start_idx = num_output_from_trace + num_kv

        if (
            not getattr(module.config, "use_hybrid_cache_manager", False)
            and hasattr(module, "_deltanet_state_params")
        ):
            for i, param in enumerate(module._deltanet_state_params):
                input_output_aliases[param] = state_start_idx + i

            checkpoint_start_idx = state_start_idx + len(module._deltanet_state_params)
            include_checkpoint_aliases = _qwen36_include_hybrid_gdn_checkpoint_outputs(
                module.config,
                n_active_tokens=getattr(module, "n_active_tokens", 0),
                neuron_config=self.neuron_config,
            )
            if include_checkpoint_aliases:
                for i, param in enumerate(
                    getattr(module, "_hybrid_gdn_checkpoint_params", [])
                ):
                    input_output_aliases[param] = checkpoint_start_idx + i

        return module, input_output_aliases


class Qwen35ModelWrapper(ModelWrapper):
    """Custom ModelWrapper for VL support with mRoPE and vision inputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qwen36_hybrid_apc_pending_input_dict = None
        self.hybrid_apc_store = None
        self.hybrid_apc_slot_allocator = None
        self.hybrid_apc_bridge = None
        self._init_hybrid_apc_scheduler_bridge()

    def _init_hybrid_apc_scheduler_bridge(self):
        if not _qwen36_config_flag(
            self.config,
            self.neuron_config,
            "use_hybrid_apc_manager",
        ):
            return

        required_gdn_layers = tuple(
            idx
            for idx, layer_type in enumerate(self.config.layer_types)
            if layer_type == "linear_attention"
        )
        if not required_gdn_layers:
            raise ValueError("hybrid APC requires at least one GDN layer")

        tp_rank = 0
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tp_rank = int(parallel_state.get_tensor_model_parallel_rank())
        except Exception:
            tp_rank = 0

        block_size = int(
            getattr(
                self.neuron_config,
                "pa_block_size",
                self.config.gdn_checkpoint_interval,
            )
        )
        self.hybrid_apc_store = HybridAPCMetadataStore(
            required_gdn_layers=required_gdn_layers,
            block_size=block_size,
            checkpoint_interval=self.config.gdn_checkpoint_interval,
            max_checkpoints=self.config.max_gdn_checkpoint_slots,
            layout_version=self.config.hybrid_apc_layout_version,
            model_revision=self.config.hybrid_apc_model_revision,
            tp_rank=tp_rank,
            recurrent_dtype=self.config.hybrid_recurrent_cache_dtype,
            conv_dtype=self.config.hybrid_conv_cache_dtype,
            allow_residual_replay=self.config.hybrid_apc_allow_residual_replay,
        )
        self.hybrid_apc_slot_allocator = HybridAPCSlotAllocator(
            self.config.max_gdn_checkpoint_slots
        )
        self.hybrid_apc_bridge = HybridAPCSchedulerBridge(
            store=self.hybrid_apc_store,
            slot_allocator=self.hybrid_apc_slot_allocator,
            cache_salt=self.config.hybrid_apc_cache_salt,
            model_revision=self.config.hybrid_apc_model_revision,
            layout_version=self.config.hybrid_apc_layout_version,
            tp_rank=tp_rank,
            recurrent_dtype=self.config.hybrid_recurrent_cache_dtype,
            conv_dtype=self.config.hybrid_conv_cache_dtype,
            allow_local_hash_fallback=self.config.hybrid_apc_allow_local_hash_fallback,
            require_attention_block_refs=self.config.hybrid_apc_require_attention_block_refs,
            reject_unbacked_attention_hits=(
                self.config.hybrid_apc_reject_unbacked_attention_hits
            ),
        )

    def ensure_hybrid_apc_scheduler_bridge(self):
        if not _qwen36_config_flag(
            self.config,
            self.neuron_config,
            "use_hybrid_apc_manager",
        ):
            return None
        if getattr(self, "hybrid_apc_bridge", None) is None:
            self._init_hybrid_apc_scheduler_bridge()
        return self.hybrid_apc_bridge

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

            is_cte = self.tag == CONTEXT_ENCODING_MODEL_TAG

            if is_cte:
                mrope_position_ids = (
                    torch.arange(0, n_active_tokens, dtype=torch.int32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(3, batch_size, -1)
                    .contiguous()
                )

                if getattr(self.config, "use_text_only_cte_inputs", True):
                    vision_embeddings = torch.zeros(
                        (0,), dtype=self.config.neuron_config.torch_dtype
                    )
                    vision_mask = torch.zeros((0,), dtype=torch.int32)
                else:
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

            hybrid_args = None
            if _use_expanded_hybrid_args_for_tag(self.config, self.tag):
                hybrid_args = (
                    torch.zeros((batch_size,), dtype=torch.int32),
                    torch.zeros((batch_size,), dtype=torch.int32),
                    torch.zeros((batch_size,), dtype=torch.int32),
                    torch.zeros((batch_size,), dtype=torch.int32),
                    torch.zeros((batch_size,), dtype=torch.int32),
                )

            if is_cte:
                padded = build_cte_args(
                    self.config,
                    bucket_inputs,
                    mrope_position_ids,
                    vision_embeddings,
                    vision_mask,
                    hybrid_args=hybrid_args,
                )
            else:
                padded = build_tkg_args(
                    self.config,
                    bucket_inputs,
                    mrope_position_ids,
                    vision_embeddings,
                    vision_mask,
                    hybrid_args=hybrid_args,
                )
            _debug_qwen36_arg_contract(
                "compile",
                self.tag,
                self.config,
                padded,
            )
            extended_inputs.append(tuple(padded))

        return extended_inputs

    def _prepare_hybrid_apc_pad_inputs(self, args):
        if (
            self.tag != CONTEXT_ENCODING_MODEL_TAG
            or len(args) < 29
            or not _qwen36_config_flag(
                self.config,
                self.neuron_config,
                "use_hybrid_apc_manager",
            )
            or _qwen36_hybrid_apc_controls_materialized(
                args[25],
                args[26],
                args[28],
            )
        ):
            return args

        computed_context_lens = args[14]
        num_queries = args[13]
        full_context_lens = (
            computed_context_lens + num_queries
            if hasattr(computed_context_lens, "shape") and hasattr(num_queries, "shape")
            else None
        )
        hybrid_apc_request_dict = {
            "input_ids": args[0],
            "attention_mask": args[1],
            "position_ids": args[2],
            "seq_ids": args[3],
            "sampling_params": args[4],
            "adapter_ids": args[6],
            "slot_mapping": args[11],
            "block_table": args[12],
            "num_queries": num_queries,
            "computed_context_lens": computed_context_lens,
        }
        if full_context_lens is not None:
            hybrid_apc_request_dict["full_context_lens"] = full_context_lens

        request_records = getattr(
            self,
            "_qwen36_vllm_hybrid_apc_request_records",
            None,
        )
        request_ids = _qwen36_request_ids_from_hybrid_apc_records(request_records)
        if request_records is not None:
            hybrid_apc_request_dict["hybrid_request_records"] = request_records
        if request_ids is None:
            request_ids = getattr(self, "_qwen36_vllm_request_ids", None)
        if request_ids is not None:
            if isinstance(request_ids, list):
                request_ids = tuple(request_ids)
            if isinstance(request_ids, tuple) and len(request_ids) == 1:
                hybrid_apc_request_dict["hybrid_request_id"] = request_ids[0]
            else:
                hybrid_apc_request_dict["hybrid_request_id"] = request_ids
        cached_request_ids = getattr(self, "_qwen36_vllm_cached_request_ids", None)
        if cached_request_ids is not None:
            hybrid_apc_request_dict["hybrid_cached_request_ids"] = cached_request_ids
        prefill_completion_state = getattr(
            self,
            "_qwen36_vllm_prefill_completion_state",
            None,
        )
        if prefill_completion_state is not None:
            hybrid_apc_request_dict[
                "hybrid_prefill_completion_state"
            ] = prefill_completion_state
        _qwen36_add_vllm_hybrid_apc_metadata(
            hybrid_apc_request_dict,
            request_ids=request_ids,
            metadata_by_request_id=getattr(
                self,
                "_qwen36_vllm_hybrid_apc_metadata_by_request_id",
                None,
            ),
        )

        prepared_inputs = prepare_hybrid_apc_request_for_execution(
            self,
            hybrid_apc_request_dict,
        )
        hybrid_args = prepare_hybrid_apc_model_inputs(self, prepared_inputs)
        if not hybrid_args:
            return args

        updated_args = list(args)
        for index, key in (
            (0, "input_ids"),
            (1, "attention_mask"),
            (2, "position_ids"),
            (3, "seq_ids"),
            (4, "sampling_params"),
            (6, "adapter_ids"),
            (11, "slot_mapping"),
            (12, "block_table"),
            (13, "num_queries"),
            (14, "computed_context_lens"),
        ):
            if key in prepared_inputs:
                updated_args[index] = prepared_inputs[key]
        if len(hybrid_args) == 14 and len(updated_args) >= 29:
            updated_args[15:29] = hybrid_args
        else:
            updated_args[24:29] = hybrid_args
        self._qwen36_hybrid_apc_pending_input_dict = hybrid_apc_request_dict
        return tuple(updated_args)

    def _forward_with_pad(self, *args):
        self._qwen36_hybrid_apc_pending_input_dict = None
        try:
            outputs = super()._forward_with_pad(*args)
        except Exception:
            pending = self._qwen36_hybrid_apc_pending_input_dict
            self._qwen36_hybrid_apc_pending_input_dict = None
            if pending is not None:
                cancel_hybrid_apc_request(pending)
            raise
        pending = self._qwen36_hybrid_apc_pending_input_dict
        self._qwen36_hybrid_apc_pending_input_dict = None
        if pending is not None:
            finish_hybrid_apc_request(pending)
        return outputs

    def pad_inputs(self, *args, pad_type="first_fit"):
        """Override to pad mrope_position_ids and vision inputs to bucket size."""
        args = self._prepare_hybrid_apc_pad_inputs(args)
        if (
            self.tag in (CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG)
            and len(args) == 15
            and self.is_prefix_caching
            and not getattr(
                getattr(self, "neuron_config", None),
                "enable_fused_speculation",
                False,
            )
            and not getattr(
                getattr(self, "neuron_config", None),
                "enable_eagle_speculation",
                False,
            )
        ):
            args = tuple(
                _normalize_qwen36_prefix_args(args)
                + [_empty_qwen36_arg(), _empty_qwen36_arg(), _empty_qwen36_arg()]
            )
        orig_mrope = args[21] if len(args) >= 22 else None
        orig_vis_emb = args[22] if len(args) >= 23 else None
        orig_vis_mask = args[23] if len(args) >= 24 else None
        if len(args) >= 29:
            orig_restore_slots = args[24]
            orig_restore_mask = args[25]
            orig_restore_prefix = args[26]
            orig_commit_slots = args[27]
            orig_commit_mask = args[28]
        elif (
            len(args) >= 20
            and _use_expanded_hybrid_args_for_tag(self.config, self.tag)
            and self.is_prefix_caching
            and not self.neuron_config.enable_fused_speculation
            and not self.neuron_config.enable_eagle_speculation
        ):
            orig_restore_slots = args[15]
            orig_restore_mask = args[16]
            orig_restore_prefix = args[17]
            orig_commit_slots = args[18]
            orig_commit_mask = args[19]
        else:
            orig_restore_slots = None
            orig_restore_mask = None
            orig_restore_prefix = None
            orig_commit_slots = None
            orig_commit_mask = None

        # Pre-pad/truncate vision args to match the target CTE bucket so the
        # upstream super().pad_inputs() shape check at model_wrapper.py:801
        # does not replace them with dummies. Bucket is determined by input
        # length; we compute it the same way upstream does.
        if (
            self.tag == CONTEXT_ENCODING_MODEL_TAG
            and len(args) >= 24
            and orig_vis_mask is not None
            and orig_vis_mask.ndim == 3
        ):
            try:
                target_bucket = self.get_target_bucket(*args, strategy=pad_type)
                if isinstance(target_bucket, list):
                    target_bucket = target_bucket[1]
                target_len = int(target_bucket)
            except Exception:
                target_len = None
            if target_len is not None:
                def _fit_seq(t, target, fill_dim0=False, fill_value=None):
                    if t is None or t.ndim != 3:
                        return t
                    cur = t.shape[1]
                    if cur == target:
                        return t
                    if cur < target:
                        pad_shape = list(t.shape)
                        pad_shape[1] = target - cur
                        if fill_value is not None:
                            pad = torch.full(pad_shape, fill_value=fill_value, dtype=t.dtype)
                        else:
                            pad = torch.zeros(pad_shape, dtype=t.dtype)
                        return torch.cat([t, pad], dim=1)
                    return t[:, :target].contiguous()

                new_vis_emb = _fit_seq(orig_vis_emb, target_len)
                new_vis_mask = _fit_seq(orig_vis_mask, target_len, fill_value=target_len - 1)
                if new_vis_emb is not None or new_vis_mask is not None:
                    args = list(args)
                    if new_vis_emb is not None:
                        args[22] = new_vis_emb
                    if new_vis_mask is not None:
                        args[23] = new_vis_mask
                    args = tuple(args)

        padded_args = super().pad_inputs(*args, pad_type=pad_type)

        if len(padded_args) >= 24 and orig_mrope is not None:
            padded_seq_len = padded_args[0].shape[1]
            batch_size = padded_args[0].shape[0]
            is_cte = self.tag == CONTEXT_ENCODING_MODEL_TAG

            if is_cte:
                current_mrope = orig_mrope
                current_vis_emb = orig_vis_emb
                current_vis_mask = orig_vis_mask

                if (
                    current_mrope.ndim == 3
                    and current_mrope.shape[-1] < padded_seq_len
                ):
                    pad_size = padded_seq_len - current_mrope.shape[-1]
                    last_pos = current_mrope[:, :, -1:]
                    # Padded tokens are masked out of the active CTE, so do not
                    # advance mRoPE into fake future positions.
                    mrope_pad = last_pos.expand(3, batch_size, pad_size)
                    mrope_position_ids = torch.cat([current_mrope, mrope_pad], dim=-1)
                elif (
                    current_mrope.ndim == 3
                    and current_mrope.shape[-1] > padded_seq_len
                ):
                    # Bucket smaller than caller-provided mrope; truncate.
                    mrope_position_ids = current_mrope[:, :, :padded_seq_len].contiguous()
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
                    # Qwen3-VL pad convention: pad slots of vision_embeddings
                    # are zeros; pad slots of vision_mask (below) point at
                    # padded_seq_len-1 which is guaranteed to be a padded
                    # (attention_mask==0) input position — so scatter writes
                    # zero to a masked slot with no downstream effect.
                    pad_len = padded_seq_len - current_vis_emb.shape[1]
                    pad_emb = torch.zeros(
                        (batch_size, pad_len, current_vis_emb.shape[2]),
                        dtype=current_vis_emb.dtype,
                    )
                    vision_embeddings = torch.cat([current_vis_emb, pad_emb], dim=1)
                elif current_vis_emb is not None and current_vis_emb.ndim == 3:
                    vision_embeddings = current_vis_emb[:, :padded_seq_len]
                elif getattr(self.config, "use_text_only_cte_inputs", True):
                    vision_embeddings = torch.zeros(
                        (0,), dtype=self.config.neuron_config.torch_dtype
                    )
                else:
                    # Dummy vision inputs for text-only calls when graph was
                    # traced with vision inputs. Zeros are fine because mask
                    # sends them all to the padding-position at padded_seq_len-1.
                    vision_embeddings = torch.zeros(
                        (batch_size, padded_seq_len, self.config.hidden_size),
                        dtype=self.config.neuron_config.torch_dtype,
                    )

                if (
                    current_vis_mask is not None
                    and current_vis_mask.ndim == 3
                    and current_vis_mask.shape[1] < padded_seq_len
                ):
                    # Qwen3-VL pad convention: pad slots of vision_mask point
                    # at padded_seq_len-1 (a padded input slot).
                    pad_len = padded_seq_len - current_vis_mask.shape[1]
                    pad_mask = torch.full(
                        (batch_size, pad_len, 1),
                        fill_value=padded_seq_len - 1,
                        dtype=torch.int32,
                    )
                    vision_mask = torch.cat([current_vis_mask, pad_mask], dim=1)
                elif current_vis_mask is not None and current_vis_mask.ndim == 3:
                    vision_mask = current_vis_mask[:, :padded_seq_len]
                elif getattr(self.config, "use_text_only_cte_inputs", True):
                    vision_mask = torch.zeros((0,), dtype=torch.int32)
                else:
                    # Dummy mask for text-only forward on a vision-traced graph.
                    # All slots target padded_seq_len-1 (the last position, always
                    # a padded/eos slot). Combined with zero vision_emb the scatter
                    # overwrites that one position with zeros — harmless because
                    # attention_mask=0 there.
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

                if vision_mask.ndim == 3:
                    padded_args = list(padded_args)
                    padded_args[23] = padded_args[23].clamp(max=padded_seq_len - 1)
                    padded_args = tuple(padded_args)

        if (
            len(padded_args) >= 24
            and _use_expanded_hybrid_args_for_tag(self.config, self.tag)
        ):
            padded_batch_size = padded_args[0].shape[0]

            def _pad_vector(value, dtype=torch.int32):
                if value is None or not hasattr(value, "ndim") or value.ndim == 0:
                    return torch.zeros((padded_batch_size,), dtype=dtype)
                value = value.to(dtype)
                if value.shape[0] == padded_batch_size:
                    return value
                if value.shape[0] > padded_batch_size:
                    return value[:padded_batch_size]
                pad = torch.zeros(
                    (padded_batch_size - value.shape[0],),
                    dtype=value.dtype,
                )
                return torch.cat([value, pad], dim=0)

            hybrid_args = (
                _pad_vector(orig_restore_slots),
                _pad_vector(orig_restore_mask),
                _pad_vector(orig_restore_prefix),
                _pad_vector(orig_commit_slots),
                _pad_vector(orig_commit_mask),
            )
            if len(padded_args) >= 29:
                padded_args = (*padded_args[:24], *hybrid_args)
            else:
                padded_args = (*padded_args, *hybrid_args)

        _assert_qwen36_arg_count(
            self.tag,
            padded_args,
            _qwen36_expected_arg_count(self.config, self.tag),
        )
        _debug_qwen36_arg_contract("pad", self.tag, self.config, padded_args)
        return padded_args


# ============================================================
# Top-Level Model
# ============================================================


class NeuronQwen35ForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronQwen35Model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_hybrid_apc_scheduler_bridge()

    def _init_hybrid_apc_scheduler_bridge(self):
        self.hybrid_apc_store = None
        self.hybrid_apc_slot_allocator = None
        self.hybrid_apc_bridge = None
        if not _qwen36_config_flag(
            self.config,
            self.neuron_config,
            "use_hybrid_apc_manager",
        ):
            return

        required_gdn_layers = tuple(
            idx
            for idx, layer_type in enumerate(self.config.layer_types)
            if layer_type == "linear_attention"
        )
        if not required_gdn_layers:
            raise ValueError("hybrid APC requires at least one GDN layer")

        tp_rank = 0
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tp_rank = int(parallel_state.get_tensor_model_parallel_rank())
        except Exception:
            tp_rank = 0

        block_size = int(
            getattr(
                self.neuron_config,
                "pa_block_size",
                self.config.gdn_checkpoint_interval,
            )
        )
        self.hybrid_apc_store = HybridAPCMetadataStore(
            required_gdn_layers=required_gdn_layers,
            block_size=block_size,
            checkpoint_interval=self.config.gdn_checkpoint_interval,
            max_checkpoints=self.config.max_gdn_checkpoint_slots,
            layout_version=self.config.hybrid_apc_layout_version,
            model_revision=self.config.hybrid_apc_model_revision,
            tp_rank=tp_rank,
            recurrent_dtype=self.config.hybrid_recurrent_cache_dtype,
            conv_dtype=self.config.hybrid_conv_cache_dtype,
            allow_residual_replay=self.config.hybrid_apc_allow_residual_replay,
        )
        self.hybrid_apc_slot_allocator = HybridAPCSlotAllocator(
            self.config.max_gdn_checkpoint_slots
        )
        self.hybrid_apc_bridge = HybridAPCSchedulerBridge(
            store=self.hybrid_apc_store,
            slot_allocator=self.hybrid_apc_slot_allocator,
            cache_salt=self.config.hybrid_apc_cache_salt,
            model_revision=self.config.hybrid_apc_model_revision,
            layout_version=self.config.hybrid_apc_layout_version,
            tp_rank=tp_rank,
            recurrent_dtype=self.config.hybrid_recurrent_cache_dtype,
            conv_dtype=self.config.hybrid_conv_cache_dtype,
            allow_local_hash_fallback=self.config.hybrid_apc_allow_local_hash_fallback,
            require_attention_block_refs=self.config.hybrid_apc_require_attention_block_refs,
            reject_unbacked_attention_hits=(
                self.config.hybrid_apc_reject_unbacked_attention_hits
            ),
        )

    def ensure_hybrid_apc_scheduler_bridge(self):
        if not _qwen36_config_flag(
            self.config,
            self.neuron_config,
            "use_hybrid_apc_manager",
        ):
            return None
        if getattr(self, "hybrid_apc_bridge", None) is None:
            self._init_hybrid_apc_scheduler_bridge()
        return self.hybrid_apc_bridge

    def on_attention_block_evicted(self, block_ref: int):
        if self.hybrid_apc_store is None:
            return []
        return self.hybrid_apc_store.on_attention_block_evicted(block_ref)

    def on_attention_blocks_evicted(self, block_refs):
        invalidated = []
        if self.hybrid_apc_store is None:
            return invalidated
        for block_ref in block_refs:
            invalidated.extend(
                self.hybrid_apc_store.on_attention_block_evicted(block_ref)
            )
        return invalidated

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
        # Qwen3.5-2B has tie_word_embeddings=True. HF only stores
        # embed_tokens.weight and derives lm_head from it. NxDI's lm_head is a
        # separate ColumnParallelLinear that needs its own weight tensor.
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
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

        return convert_qwen35_hf_to_neuron_state_dict(new_sd, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        disable_wlo = bool(
            getattr(self.config, "disable_token_generation_wlo", False)
        ) or os.environ.get("QWEN36_DISABLE_TOKEN_GENERATION_WLO") == "1"
        super().enable_token_generation(enable_wlt_optimization=not disable_wlo)

    def _copy_past_key_values(self, outputs):
        """Override to also copy DeltaNet state buffers on CPU."""
        super()._copy_past_key_values(outputs)
        if getattr(self.config, "use_hybrid_cache_manager", False):
            return

        num_output_from_trace = Qwen35DecoderModelInstance._num_trace_outputs_before_aliases(
            self.neuron_config
        )

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

        checkpoint_start = state_start + len(tkg_params)
        tkg_checkpoint_params = getattr(tkg_model, "_hybrid_gdn_checkpoint_params", [])
        cte_checkpoint_params = getattr(cte_model, "_hybrid_gdn_checkpoint_params", [])
        if (
            len(tkg_checkpoint_params) > 0
            and checkpoint_start + len(tkg_checkpoint_params) <= len(outputs)
        ):
            for i, (tkg_param, cte_param) in enumerate(
                zip(tkg_checkpoint_params, cte_checkpoint_params)
            ):
                new_state = outputs[checkpoint_start + i]
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
        hybrid_restore_slot_ids=None,
        hybrid_restore_mask=None,
        hybrid_restore_prefix_lens=None,
        hybrid_commit_slot_ids=None,
        hybrid_commit_mask=None,
    ):
        """Override to pass Qwen/vLLM positional args explicitly."""
        prefill_completion_state = getattr(
            self,
            "_qwen36_vllm_prefill_completion_state",
            None,
        )
        is_prefill = _qwen36_is_prefill_request(
            input_ids,
            position_ids,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
            prefill_completion_state=prefill_completion_state,
        )
        metadata_by_request_id = getattr(
            self,
            "_qwen36_vllm_hybrid_apc_metadata_by_request_id",
            None,
        )
        request_records = getattr(
            self,
            "_qwen36_vllm_hybrid_apc_request_records",
            None,
        )
        request_ids = _qwen36_request_ids_from_hybrid_apc_records(request_records)
        if request_ids is None:
            request_ids = _qwen36_select_vllm_hybrid_apc_request_ids_for_input(
                metadata_by_request_id,
                all_request_ids=getattr(self, "_qwen36_vllm_request_ids", None),
                new_request_ids=getattr(self, "_qwen36_vllm_new_request_ids", None),
                full_context_lens=full_context_lens,
                computed_context_lens=computed_context_lens,
                prefill_completion_state=prefill_completion_state,
            )
        if not is_prefill:
            (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                adapter_ids,
                slot_mapping,
            ) = _qwen36_unpack_packed_decode_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                adapter_ids=adapter_ids,
                slot_mapping=slot_mapping,
                full_context_lens=full_context_lens,
                computed_context_lens=computed_context_lens,
            )
        seq_ids = _qwen36_stable_seq_ids_for_request_ids(
            self,
            seq_ids,
            request_ids,
        )

        hybrid_apc_request_dict = None
        if (
            is_prefill
            and _qwen36_config_flag(
                self.config,
                self.neuron_config,
                "use_hybrid_apc_manager",
            )
            and getattr(self.neuron_config, "is_prefix_caching", False)
            and _qwen36_hybrid_apc_controls_need_prepare(
                hybrid_restore_mask,
                hybrid_commit_mask,
            )
        ):
            hybrid_apc_request_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "seq_ids": seq_ids,
                "sampling_params": sampling_params,
                "adapter_ids": adapter_ids,
                "slot_mapping": slot_mapping,
                "block_table": block_table,
                "full_context_lens": full_context_lens,
                "computed_context_lens": computed_context_lens,
            }
            if llava_args:
                hybrid_apc_request_dict["llava_args"] = llava_args
                if len(llava_args) >= 3:
                    hybrid_apc_request_dict["rotary_position_ids"] = llava_args[2]
            if request_records is not None:
                hybrid_apc_request_dict["hybrid_request_records"] = request_records
            if request_ids is not None:
                if isinstance(request_ids, list):
                    request_ids = tuple(request_ids)
                if isinstance(request_ids, tuple) and len(request_ids) == 1:
                    hybrid_apc_request_dict["hybrid_request_id"] = request_ids[0]
                else:
                    hybrid_apc_request_dict["hybrid_request_id"] = request_ids
            cached_request_ids = getattr(
                self,
                "_qwen36_vllm_cached_request_ids",
                None,
            )
            if cached_request_ids is not None:
                hybrid_apc_request_dict["hybrid_cached_request_ids"] = (
                    cached_request_ids
                )
            if prefill_completion_state is not None:
                hybrid_apc_request_dict[
                    "hybrid_prefill_completion_state"
                ] = prefill_completion_state
            _qwen36_add_vllm_hybrid_apc_metadata(
                hybrid_apc_request_dict,
                request_ids=request_ids,
                metadata_by_request_id=metadata_by_request_id,
            )
            prepared_inputs = prepare_hybrid_apc_request_for_execution(
                self,
                hybrid_apc_request_dict,
            )
            input_ids = prepared_inputs.get("input_ids", input_ids)
            attention_mask = prepared_inputs.get("attention_mask", attention_mask)
            position_ids = prepared_inputs.get("position_ids", position_ids)
            seq_ids = prepared_inputs.get("seq_ids", seq_ids)
            sampling_params = prepared_inputs.get("sampling_params", sampling_params)
            adapter_ids = prepared_inputs.get("adapter_ids", adapter_ids)
            slot_mapping = prepared_inputs.get("slot_mapping", slot_mapping)
            block_table = prepared_inputs.get("block_table", block_table)
            full_context_lens = prepared_inputs.get("full_context_lens", full_context_lens)
            computed_context_lens = prepared_inputs.get(
                "computed_context_lens",
                computed_context_lens,
            )
            num_queries = prepared_inputs.get("num_queries", num_queries)
            hybrid_restore_slot_ids = prepared_inputs.get("hybrid_restore_slot_ids")
            hybrid_restore_mask = prepared_inputs.get("hybrid_restore_mask")
            hybrid_restore_prefix_lens = prepared_inputs.get(
                "hybrid_restore_prefix_lens"
            )
            hybrid_commit_slot_ids = prepared_inputs.get("hybrid_commit_slot_ids")
            hybrid_commit_mask = prepared_inputs.get("hybrid_commit_mask")
            prepared_mrope_position_ids = prepared_inputs.get(
                "rotary_position_ids",
                prepared_inputs.get("rotary_position_id"),
            )
            if prepared_mrope_position_ids is not None and llava_args:
                llava_args = list(llava_args)
                if len(llava_args) >= 3:
                    llava_args[2] = prepared_mrope_position_ids
                elif len(llava_args) >= 2:
                    llava_args.append(prepared_mrope_position_ids)
            elif prepared_mrope_position_ids is not None:
                mrope_position_ids = prepared_mrope_position_ids
        else:
            prepared_mrope_position_ids = None

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
            if getattr(self.config, "use_text_only_cte_inputs", True):
                vision_embeddings = torch.zeros(
                    (0,), dtype=self.config.neuron_config.torch_dtype
                )
                vision_mask = torch.zeros((0,), dtype=torch.int32)
            else:
                vision_embeddings = torch.zeros(
                    (batch_size, seq_len, self.config.hidden_size),
                    dtype=self.config.neuron_config.torch_dtype,
                )
                vision_mask = torch.full(
                    (batch_size, seq_len, 1),
                    fill_value=seq_len - 1,
                    dtype=torch.int32,
                )
            mrope_position_ids = prepared_mrope_position_ids
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

        def _empty():
            return torch.empty(0)

        def _optional_tensor(value):
            return value if value is not None else _empty()

        def _length_matrix(value, default_value, batch=batch_size):
            if value is None or not hasattr(value, "numel") or value.numel() == 0:
                return torch.full((batch, 1), default_value, dtype=torch.int32)
            value = value.to(torch.int32)
            if value.ndim == 0:
                return value.reshape(1, 1)
            if value.ndim == 1:
                return value.reshape(-1, 1)
            return value

        def _slice_batch(value, start, end):
            if value is None or not hasattr(value, "numel") or value.numel() == 0:
                return _empty()
            if value.ndim > 0 and value.shape[0] >= end:
                return value[start:end]
            return value

        def _pad_batch(value, target_batch, fill_value=0):
            if value is None or not hasattr(value, "numel") or value.numel() == 0:
                return value
            if value.ndim == 0 or value.shape[0] >= target_batch:
                return value
            pad_shape = (target_batch - value.shape[0],) + tuple(value.shape[1:])
            pad = torch.full(pad_shape, fill_value, dtype=value.dtype)
            return torch.cat([value, pad], dim=0)

        def _pad_batch_repeat_first(value, target_batch):
            if value is None or not hasattr(value, "numel") or value.numel() == 0:
                return value
            if value.ndim == 0 or value.shape[0] >= target_batch:
                return value
            pad_n = target_batch - value.shape[0]
            return torch.cat([value, value[:1].expand(pad_n, *value.shape[1:])], dim=0)

        if self.neuron_config.is_prefix_caching:
            if is_prefill:
                computed_context_lens_arg = _length_matrix(computed_context_lens, 0)
                full_context_lens_arg = _length_matrix(full_context_lens, seq_len)
                num_queries_arg = (
                    full_context_lens_arg - computed_context_lens_arg
                ).to(torch.int32)
            else:
                if seq_len != 1:
                    raise ValueError(
                        "Qwen3.6 TKG expects active decode length 1, "
                        f"got input_ids.shape[-1]={seq_len}"
                    )
                num_queries_arg = torch.full(
                    (batch_size, 1), seq_len, dtype=torch.int32
                )
                if (
                    position_ids is not None
                    and hasattr(position_ids, "numel")
                    and position_ids.numel() > 0
                ):
                    computed_context_lens_arg = _length_matrix(position_ids, 0)
                elif full_context_lens is not None:
                    computed_context_lens_arg = _length_matrix(
                        full_context_lens, seq_len
                    )
                else:
                    computed_context_lens_arg = _length_matrix(
                        computed_context_lens,
                        attention_mask.shape[-1] if attention_mask is not None else 0,
                    )
            slot_mapping_arg = _optional_tensor(slot_mapping)
            slot_mapping_arg = _normalize_qwen36_slot_mapping(
                slot_mapping_arg,
                batch_size,
                seq_len,
            )
            block_table_arg = _optional_tensor(block_table)
        else:
            computed_context_lens_arg = _empty()
            num_queries_arg = _empty()
            slot_mapping_arg = _empty()
            block_table_arg = _empty()

        if hybrid_restore_slot_ids is None:
            hybrid_restore_slot_ids = torch.zeros((batch_size,), dtype=torch.int32)
        if hybrid_restore_mask is None:
            hybrid_restore_mask = torch.zeros((batch_size,), dtype=torch.int32)
        if hybrid_restore_prefix_lens is None:
            hybrid_restore_prefix_lens = torch.zeros((batch_size,), dtype=torch.int32)
        if hybrid_commit_slot_ids is None:
            hybrid_commit_slot_ids = torch.zeros((batch_size,), dtype=torch.int32)
        if hybrid_commit_mask is None:
            hybrid_commit_mask = torch.zeros((batch_size,), dtype=torch.int32)

        if is_prefill:
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
                chunk_slot_mapping = _slice_batch(slot_mapping_arg, cb, cb_end)
                chunk_block_table = _slice_batch(block_table_arg, cb, cb_end)
                chunk_num_queries = _slice_batch(num_queries_arg, cb, cb_end)
                chunk_computed_context_lens = _slice_batch(
                    computed_context_lens_arg, cb, cb_end
                )
                chunk_restore_slots = hybrid_restore_slot_ids[cb:cb_end]
                chunk_restore_mask = hybrid_restore_mask[cb:cb_end]
                chunk_restore_prefix = hybrid_restore_prefix_lens[cb:cb_end]
                chunk_commit_slots = hybrid_commit_slot_ids[cb:cb_end]
                chunk_commit_mask = hybrid_commit_mask[cb:cb_end]
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
                    pad_seq = torch.full(
                        (pad_n,), -1, dtype=chunk_seq_ids.dtype
                    )
                    chunk_seq_ids = torch.cat([chunk_seq_ids, pad_seq], dim=0)
                    chunk_sampling = torch.cat(
                        [chunk_sampling, chunk_sampling[:1].expand(pad_n, -1)], dim=0
                    )
                    chunk_slot_mapping = _pad_batch(chunk_slot_mapping, ctx_bs, -1)
                    chunk_block_table = _pad_batch_repeat_first(
                        chunk_block_table, ctx_bs
                    )
                    chunk_num_queries = _pad_batch_repeat_first(
                        chunk_num_queries, ctx_bs
                    )
                    chunk_computed_context_lens = _pad_batch(
                        chunk_computed_context_lens, ctx_bs, 0
                    )
                    # Dummy CTE rows repeat active token tensors to satisfy the
                    # compiled batch shape, but they must not advertise a
                    # prefix-cache restore. Their seq_ids are marked negative
                    # and the DeltaNet state update preserves negative rows, so
                    # recurrent state cannot leak into seq_ids later reused by
                    # real requests.
                    (
                        chunk_restore_slots,
                        chunk_restore_mask,
                        chunk_restore_prefix,
                    ) = _qwen36_pad_hybrid_restore_controls_for_dummy_cte_rows(
                        chunk_restore_slots,
                        chunk_restore_mask,
                        chunk_restore_prefix,
                        ctx_bs,
                    )
                    chunk_commit_slots = torch.cat(
                        [chunk_commit_slots, torch.zeros(pad_n, dtype=chunk_commit_slots.dtype)],
                        dim=0,
                    )
                    chunk_commit_mask = torch.cat(
                        [chunk_commit_mask, torch.zeros(pad_n, dtype=chunk_commit_mask.dtype)],
                        dim=0,
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

                if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
                    def _dbg_minmax(tensor):
                        if not hasattr(tensor, "numel") or tensor.numel() == 0:
                            return "empty"
                        flat = tensor.reshape(-1)
                        return f"{int(flat.min().item())}:{int(flat.max().item())}"

                    print(
                        "[hybrid_apc_debug] qwen-cte-call "
                        f"input_shape={tuple(chunk_input_ids.shape)} "
                        f"attention_shape={tuple(chunk_attn_mask.shape)} "
                        f"position_shape={tuple(chunk_pos_ids.shape)} "
                        f"position_minmax={_dbg_minmax(chunk_pos_ids)} "
                        f"seq_ids={chunk_seq_ids.reshape(-1).tolist() if hasattr(chunk_seq_ids, 'numel') and chunk_seq_ids.numel() else []} "
                        f"slot_shape={tuple(chunk_slot_mapping.shape)} "
                        f"slot_minmax={_dbg_minmax(chunk_slot_mapping)} "
                        f"block_shape={tuple(chunk_block_table.shape)} "
                        f"block_minmax={_dbg_minmax(chunk_block_table)} "
                        f"num_queries={chunk_num_queries.reshape(-1).tolist() if hasattr(chunk_num_queries, 'numel') and chunk_num_queries.numel() else []} "
                        f"computed={chunk_computed_context_lens.reshape(-1).tolist() if hasattr(chunk_computed_context_lens, 'numel') and chunk_computed_context_lens.numel() else []} "
                        f"restore_slots={chunk_restore_slots.reshape(-1).tolist()} "
                        f"restore_mask={chunk_restore_mask.reshape(-1).tolist()} "
                        f"restore_prefix={chunk_restore_prefix.reshape(-1).tolist()} "
                        f"commit_slots={chunk_commit_slots.reshape(-1).tolist()} "
                        f"commit_mask={chunk_commit_mask.reshape(-1).tolist()}",
                        flush=True,
                    )

                cte_prefix_args = [
                    chunk_input_ids,
                    chunk_attn_mask,
                    chunk_pos_ids,
                    chunk_seq_ids,
                    chunk_sampling,
                    chunk_prev_hidden,
                    chunk_adapter_ids,
                    _empty(),
                    _empty(),
                    _empty(),
                    _empty(),
                    chunk_slot_mapping,
                    chunk_block_table,
                    chunk_num_queries,
                    chunk_computed_context_lens,
                    _empty(),
                    _empty(),
                    _empty(),
                    _empty(),
                    _empty(),
                    _empty(),
                ]
                cte_args = build_cte_args(
                    self.config,
                    cte_prefix_args,
                    chunk_mrope,
                    chunk_vis_emb,
                    chunk_vis_mask,
                    hybrid_args=(
                        chunk_restore_slots,
                        chunk_restore_mask,
                        chunk_restore_prefix,
                        chunk_commit_slots,
                        chunk_commit_mask,
                    ),
                )
                _debug_qwen36_arg_contract(
                    "runtime",
                    CONTEXT_ENCODING_MODEL_TAG,
                    self.config,
                    cte_args,
                )
                _qwen36_prefill_timing = os.environ.get("QWEN36_PREFILL_TIMING") == "1"
                _qwen36_cte_start = time.perf_counter() if _qwen36_prefill_timing else None
                try:
                    chunk_out = self.context_encoding_model(*cte_args)
                except Exception:
                    if hybrid_apc_request_dict is not None:
                        cancel_hybrid_apc_request(hybrid_apc_request_dict)
                        hybrid_apc_request_dict = None
                    raise
                if _qwen36_prefill_timing and _qwen36_cte_start is not None:
                    print(
                        "[qwen36_perf] qwen_cte_call "
                        f"elapsed_ms={(time.perf_counter() - _qwen36_cte_start) * 1000.0:.3f} "
                        f"actual_chunk={actual_chunk} ctx_bs={ctx_bs} "
                        f"input_shape={tuple(chunk_input_ids.shape)} "
                        f"num_queries={chunk_num_queries.reshape(-1).tolist() if hasattr(chunk_num_queries, 'numel') and chunk_num_queries.numel() else []} "
                        f"computed={chunk_computed_context_lens.reshape(-1).tolist() if hasattr(chunk_computed_context_lens, 'numel') and chunk_computed_context_lens.numel() else []} "
                        f"restore_mask={chunk_restore_mask.reshape(-1).tolist()} "
                        f"commit_mask={chunk_commit_mask.reshape(-1).tolist()}",
                        flush=True,
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
            if hybrid_apc_request_dict is not None:
                finish_hybrid_apc_request(hybrid_apc_request_dict)
        else:
            _validate_qwen36_tkg_input_ids(
                input_ids,
                getattr(self.config, "vocab_size", None),
            )
            legacy_tkg_args = _use_legacy_tkg_args()
            if (
                os.environ.get("QWEN36_TKG_INPUT_DEBUG") == "1"
                or os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1"
            ):
                max_model_len = getattr(
                    self.neuron_config,
                    "max_length",
                    getattr(self.neuron_config, "seq_len", None),
                )
                print(
                    "[hybrid_apc_debug] qwen-tkg-call "
                    f"arg_mode={'prefix24_legacy' if legacy_tkg_args else 'hybrid29'} "
                    f"input_shape={_debug_tensor_shape(input_ids)} "
                    f"input_values={_debug_tensor_values(input_ids)} "
                    f"attention_shape={_debug_tensor_shape(attention_mask)} "
                    f"position_shape={_debug_tensor_shape(position_ids)} "
                    f"position_minmax={_debug_tensor_minmax(position_ids)} "
                    f"slot_shape={_debug_tensor_shape(slot_mapping_arg)} "
                    f"slot_minmax={_debug_tensor_minmax(slot_mapping_arg)} "
                    f"block_shape={_debug_tensor_shape(block_table_arg)} "
                    f"block_minmax={_debug_tensor_minmax(block_table_arg)} "
                    f"num_queries={_debug_tensor_values(num_queries_arg)} "
                    "computed_context_lens="
                    f"{_debug_tensor_values(computed_context_lens_arg)} "
                    f"pa_num_blocks={getattr(self.neuron_config, 'pa_num_blocks', None)} "
                    f"block_size={getattr(self.neuron_config, 'pa_block_size', None)} "
                    f"seq_len={seq_len} max_model_len={max_model_len}",
                    flush=True,
                )
            tkg_prefix_args = [
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                _empty(),
                _empty(),
                _empty(),
                _empty(),
                slot_mapping_arg,
                block_table_arg,
                num_queries_arg,
                computed_context_lens_arg,
                _empty(),
                _empty(),
                _empty(),
                _empty(),
                _empty(),
                _empty(),
            ]
            tkg_args = build_tkg_args(
                self.config,
                tkg_prefix_args,
                mrope_position_ids,
                vision_embeddings,
                vision_mask,
                hybrid_args=(
                    hybrid_restore_slot_ids,
                    hybrid_restore_mask,
                    hybrid_restore_prefix_lens,
                    hybrid_commit_slot_ids,
                    hybrid_commit_mask,
                ),
            )
            _debug_qwen36_arg_contract(
                "runtime",
                TOKEN_GENERATION_MODEL_TAG,
                self.config,
                tkg_args,
            )
            outputs = self.token_generation_model(*tkg_args)
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
