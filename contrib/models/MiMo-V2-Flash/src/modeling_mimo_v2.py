# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This implementation is based on the MiMo-V2-Flash model from Xiaomi.
# Reference: https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash

"""MiMo-V2-Flash model for NXD inference."""

import gc
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.utils.distributed import (
    split_along_dim,
    get_cp_rank,
)
from neuronx_distributed_inference.modules.attention.attention_process_groups import (
    get_context_parallel_attention_cp_group,
)

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)


def get_rmsnorm_cls():
    """Get appropriate RMSNorm class based on execution environment."""
    return MiMoV2RMSNorm if cpu_mode() else CustomRMSNorm


class MiMoV2RMSNorm(nn.Module):
    """RMSNorm implementation for CPU mode."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MiMoV2RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for MiMo-V2-Flash.

    Supports partial rotary embedding where only a fraction of dimensions
    use rotary position encoding.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 262144,
        base: float = 5000000.0,
        partial_rotary_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        # Calculate the actual dimension used for rotary embedding
        self.rope_dim = int(dim * partial_rotary_factor)
        # Ensure rope_dim is even
        self.rope_dim = self.rope_dim - (self.rope_dim % 2)

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_ids: Position indices of shape (batch_size, seq_len)

        Returns:
            Tuple of (cos, sin) tensors for rotary embedding
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MiMoV2InferenceConfig(InferenceConfig):
    """Configuration class for MiMo-V2-Flash inference on Neuron."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MoE configuration
        self.num_local_experts = self.n_routed_experts
        self.n_shared_experts = 0  # MiMo-V2-Flash has no shared experts

        # Set intermediate_size for MoE layers
        self.intermediate_size = self.moe_intermediate_size

        # Check and pad intermediate size if needed
        self.maybe_pad_intermediate()

        # Router configuration
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "sigmoid"  # MiMo uses sigmoid

        # Disable numeric CC token as workaround
        self.neuron_config.disable_numeric_cc_token = True

        # MiMo normalizes top-k affinities
        self.neuron_config.normalize_top_k_affinities = True

        # Parse hybrid layer pattern
        self._parse_hybrid_pattern()

    def _parse_hybrid_pattern(self):
        """Parse hybrid layer pattern to determine attention types."""
        if hasattr(self, 'hybrid_layer_pattern') and self.hybrid_layer_pattern:
            self.layer_attention_types = [
                "sliding_window" if p == 1 else "full"
                for p in self.hybrid_layer_pattern
            ]
        else:
            self.layer_attention_types = ["full"] * self.num_hidden_layers

        # Parse MoE layer frequency
        if hasattr(self, 'moe_layer_freq') and self.moe_layer_freq:
            self.layer_uses_moe = [bool(f) for f in self.moe_layer_freq]
        else:
            self.layer_uses_moe = [True] * self.num_hidden_layers

    def maybe_pad_intermediate(self):
        """Pad intermediate size if required for efficient computation."""
        from neuronx_distributed_inference.models.config import (
            SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
        )

        moe_tp_degree = self.neuron_config.moe_tp_degree
        I_TP = self.moe_intermediate_size // moe_tp_degree

        if getattr(
            self.neuron_config.blockwise_matmul_config,
            "use_shard_on_intermediate_dynamic_while",
            False,
        ):
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded_size = (
                    math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                    * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                    * moe_tp_degree
                )
                self.moe_intermediate_pad_size = max(
                    padded_size - self.moe_intermediate_size, 0
                )
                self.moe_intermediate_size = padded_size

    def get_required_attributes(self) -> List[str]:
        return [
            "attention_bias",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "hybrid_layer_pattern",
            "layernorm_epsilon",
            "max_position_embeddings",
            "moe_intermediate_size",
            "moe_layer_freq",
            "n_routed_experts",
            "norm_topk_prob",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "partial_rotary_factor",
            "rope_theta",
            "scoring_func",
            "sliding_window",
            "swa_head_dim",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "swa_rope_theta",
            "swa_v_head_dim",
            "tie_word_embeddings",
            "v_head_dim",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MoENeuronConfig]:
        return MoENeuronConfig


class NeuronMiMoV2Attention(NeuronAttentionBase):
    """MiMo-V2-Flash Attention implementation supporting hybrid attention patterns.

    Supports both full attention and sliding window attention with different
    head dimensions for Q/K vs V.
    """

    def __init__(
        self,
        config: MiMoV2InferenceConfig,
        layer_idx: int,
        is_sliding_window: bool = False,
    ):
        self.layer_idx = layer_idx
        self.is_sliding_window = is_sliding_window

        # Select parameters based on attention type.
        #
        # IMPORTANT: we intentionally force attn_v_head_dim == attn_head_dim.
        # The HF model has asymmetric head_dim (Q/K=192, V=128), but NxDI's
        # KV cache manager requires K and V to share the same last-dim size.
        # Runtime-padding V (was the previous approach) introduced subtle
        # numerical drift under TP=64 + long decode that compounded into
        # output collapse. Instead, the preprocess script pre-pads V's output
        # dim with zeros so V is physically [num_kv_heads, head_dim, hidden];
        # the forward() slices the attention output back to the real
        # v_head_dim before o_proj. See preprocess_mimo_v2_flash_fp8.py.
        if is_sliding_window:
            self.attn_head_dim = config.swa_head_dim
            self.attn_num_heads = config.swa_num_attention_heads
            self.attn_num_kv_heads = config.swa_num_key_value_heads
            self.attn_real_v_head_dim = config.swa_v_head_dim
            rope_theta = getattr(config, 'swa_rope_theta', 10000.0)
            self.sliding_window_size = config.sliding_window
        else:
            self.attn_head_dim = config.head_dim
            self.attn_num_heads = config.num_attention_heads
            self.attn_num_kv_heads = config.num_key_value_heads
            self.attn_real_v_head_dim = config.v_head_dim
            rope_theta = config.rope_theta
            self.sliding_window_size = None
        # Padded dim used for projection + KV cache: same as Q/K head_dim.
        self.attn_v_head_dim = self.attn_head_dim

        # Calculate partial rotary dimensions
        self.partial_rotary_factor = config.partial_rotary_factor
        self.rope_dim = int(self.attn_head_dim * self.partial_rotary_factor)
        self.rope_dim = self.rope_dim - (self.rope_dim % 2)  # Ensure even
        self.nope_dim = self.attn_head_dim - self.rope_dim

        # Create rotary embedding
        rotary_emb = MiMoV2RotaryEmbedding(
            dim=self.attn_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
        )

        # Initialize base attention
        # NOTE: We pass v_head_dim to base class, but MiMo uses asymmetric Q/K (192) vs V (128).
        # We override init_gqa_properties() to prevent the base class from creating
        # incompatible projection layers (which cause crashes when CP > 1).
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=self.attn_num_heads,
            num_key_value_heads=self.attn_num_kv_heads,
            head_dim=self.attn_v_head_dim,  # Use v_head_dim for base class
            rotary_emb=rotary_emb,
            rms_norm_eps=config.layernorm_epsilon,
            use_qk_norm=False,
        )

        # Initialize MiMo-specific projections with correct dimensions
        self._init_projections(config)

        # Scaling factor
        self.scaling = self.attn_head_dim ** -0.5
        # HF MiMoV2Attention (modeling_mimo_v2_flash.py) multiplies value_states
        # by config.attention_value_scale (0.707 for Flash) right after the V
        # projection, before attention softmax*V. Matching that here — applied
        # to value_states in forward() rather than to attn_output.
        self.value_scale = float(getattr(config, "attention_value_scale", 1.0))

        # Store cache KV heads for cache compatibility
        # With CONVERT_TO_MHA, all layers have num_attention_heads KV heads
        # Otherwise, use max of full and sliding window kv heads
        tp_degree = config.neuron_config.tp_degree
        if self.use_gqa_convert_to_mha:
            # CONVERT_TO_MHA: cache stores num_attention_heads (same as Q heads)
            self.cache_num_kv_heads = self.attn_num_heads
            self.local_cache_kv_heads = self.local_num_heads
        else:
            # Standard GQA: cache uses max of full and sliding window kv heads
            self.cache_num_kv_heads = max(
                config.num_key_value_heads,
                getattr(config, 'swa_num_key_value_heads', config.num_key_value_heads)
            )
            self.local_cache_kv_heads = max(1, self.cache_num_kv_heads // tp_degree)

    def init_gqa_properties(self):
        """Override base class to prevent creating incompatible QKV projections.

        MiMo-V2-Flash has asymmetric Q/K head_dim (192) vs V head_dim (128),
        which is incompatible with the base class's GroupQueryAttention_QKV.
        MiMo uses its own custom projections via _init_projections() instead.

        When CP > 1, the base class would create cte_qkv_proj/tkg_qkv_proj with
        wrong head_dim=128, causing compilation crashes. This no-op prevents that.
        """
        pass

    def _init_projections(self, config: MiMoV2InferenceConfig):
        """Initialize projection layers with correct dimensions.

        When CONVERT_TO_MHA is needed (tp_degree > num_kv_heads), K/V projections
        are sized for num_attention_heads (not original num_kv_heads). The checkpoint
        weights are replicated in preshard_hook before loading.
        """
        dtype = config.neuron_config.torch_dtype
        tp_degree = config.neuron_config.tp_degree

        # Check if we need GQA CONVERT_TO_MHA (when tp_degree > num_kv_heads)
        self.use_gqa_convert_to_mha = tp_degree > self.attn_num_kv_heads

        # Store source heads for preshard_hook
        self._src_num_kv_heads = self.attn_num_kv_heads
        self._kv_replication_factor = self.attn_num_heads // self.attn_num_kv_heads if self.use_gqa_convert_to_mha else 1

        if self.use_gqa_convert_to_mha:
            # CONVERT_TO_MHA: K and V use num_attention_heads for proper TP splitting
            k_num_heads = self.attn_num_heads
            v_num_heads = self.attn_num_heads
        else:
            k_num_heads = self.attn_num_kv_heads
            v_num_heads = self.attn_num_kv_heads

        # Q/K use head_dim. V is pre-padded to head_dim in the preprocess
        # script so K and V share one KV-cache shape; the attention output is
        # sliced back to the real v_head_dim before o_proj, so o_proj still
        # takes real_v_head_dim per head (matches the HF o_proj weight shape).
        q_hidden_size = self.attn_num_heads * self.attn_head_dim
        k_hidden_size = k_num_heads * self.attn_head_dim
        v_hidden_size = v_num_heads * self.attn_v_head_dim
        o_hidden_size = self.attn_num_heads * self.attn_real_v_head_dim

        if parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()

            # Q projection
            self.q_proj = ColumnParallelLinear(
                config.hidden_size,
                q_hidden_size,
                bias=config.attention_bias,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )

            # K projection
            self.k_proj = ColumnParallelLinear(
                config.hidden_size,
                k_hidden_size,
                bias=config.attention_bias,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )

            # V projection
            self.v_proj = ColumnParallelLinear(
                config.hidden_size,
                v_hidden_size,
                bias=config.attention_bias,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )

            # Output projection - with sequence parallel to scatter output
            self.o_proj = RowParallelLinear(
                o_hidden_size,
                config.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=1 if self.sequence_parallel_enabled else None,
            )

            # Calculate local dimensions after TP split
            self.local_num_heads = self.attn_num_heads // tp_degree
            if self.use_gqa_convert_to_mha:
                # With CONVERT_TO_MHA, local KV heads = local Q heads
                self.local_num_kv_heads = self.local_num_heads
            else:
                self.local_num_kv_heads = max(1, self.attn_num_kv_heads // tp_degree)
        else:
            self.q_proj = nn.Linear(config.hidden_size, q_hidden_size, bias=config.attention_bias)
            self.k_proj = nn.Linear(config.hidden_size, k_hidden_size, bias=config.attention_bias)
            self.v_proj = nn.Linear(config.hidden_size, v_hidden_size, bias=config.attention_bias)
            self.o_proj = nn.Linear(o_hidden_size, config.hidden_size, bias=False)

            self.local_num_heads = self.attn_num_heads
            self.local_num_kv_heads = k_num_heads

        # Override base class attributes that were computed with wrong head_dim
        # The base class init_gqa_properties() uses head_dim=v_head_dim which is wrong for Q/K
        # We need to override these to ensure correct computation
        self.num_heads = self.local_num_heads
        self.num_key_value_heads = self.local_num_kv_heads
        self.num_key_value_groups = self.local_num_heads // self.local_num_kv_heads
        self.head_dim = self.attn_head_dim  # Override to use actual Q/K head_dim (192)

        # Remove qkv_proj from base class if exists (we use separate q_proj, k_proj, v_proj)
        if hasattr(self, 'qkv_proj'):
            self.qkv_proj = None

        # Attention sink bias for attention layers (following HF implementation)
        # This is a learnable parameter that allows attention to "sink" to an extra position
        add_full_attention_sink_bias = getattr(config, 'add_full_attention_sink_bias', False)
        add_swa_attention_sink_bias = getattr(config, 'add_swa_attention_sink_bias', True)

        # Determine if this layer uses sink bias based on config
        self._use_sink_bias = (add_full_attention_sink_bias and not self.is_sliding_window) or \
                              (add_swa_attention_sink_bias and self.is_sliding_window)

        if self._use_sink_bias:
            # Shape: [num_attention_heads] - will be split across TP ranks
            # The weight is loaded from checkpoint with shape [num_attention_heads]
            # and will be sliced to [local_num_heads] during forward
            self.attention_sink_bias = nn.Parameter(
                torch.zeros(self.attn_num_heads, dtype=dtype), requires_grad=False
            )
        else:
            self.attention_sink_bias = None

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        """Pre-shard hook to replicate K/V weights for CONVERT_TO_MHA.

        NOTE: This method is NOT currently called because NeuronMiMoV2Attention
        is not a BaseGroupQueryAttention subclass. K/V weight replication is
        instead done in convert_mimo_v2_hf_to_neuron_state_dict().

        This method is kept for reference and potential future use.
        """
        # This hook is not called - see note above
        return False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for MiMo-V2-Flash attention with Context Parallelism support."""

        # Context Parallelism: only active during context encoding (no past_key_value)
        is_context_parallel = past_key_value is None and self.cp_degree > 1
        cp_rank = None

        if is_context_parallel:
            cp_rank = get_cp_rank(
                self.rank_util.get_rank(), self.tp_degree,
                self.cp_degree, self.neuron_config.switch_cc,
            )
            # Split attention_mask (dim=2 = Q rows) and position_ids (dim=1 = seq)
            attention_mask = split_along_dim(
                attention_mask, dim=2, rank=cp_rank, num_partitions=self.cp_degree
            )
            # Keep full position_ids for RoPE computation on full-length K/V
            local_position_ids = split_along_dim(
                position_ids, dim=1, rank=cp_rank, num_partitions=self.cp_degree
            )

        # Handle sequence parallel
        if self.sequence_parallel_enabled and parallel_state.model_parallel_is_initialized():
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=parallel_state.get_tensor_model_parallel_group(),
            )

        # Context Parallelism without sequence parallel: split hidden_states
        if is_context_parallel and not self.sequence_parallel_enabled:
            hidden_states = split_along_dim(
                hidden_states, dim=1, rank=cp_rank, num_partitions=self.cp_degree
            )

        bsz, q_len, _ = hidden_states.size()

        # Determine if this is token generation (past_key_value is not None)
        is_token_gen = past_key_value is not None

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # HF MiMoV2Attention scales V by attention_value_scale (0.707 for Flash)
        # right after v_proj, before the attention softmax*V. Earlier revisions
        # of this file applied it post-attention or not at all; both produce
        # gibberish for prompts longer than ~20 tokens.
        if self.value_scale != 1.0:
            value_states = value_states * self.value_scale

        # Reshape for multi-head attention: [bsz, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.local_num_heads, self.attn_head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.local_num_kv_heads, self.attn_head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.local_num_kv_heads, self.attn_v_head_dim).transpose(1, 2)

        # Split into rope and non-rope parts
        query_rope = query_states[..., :self.rope_dim]
        query_nope = query_states[..., self.rope_dim:]
        key_rope = key_states[..., :self.rope_dim]
        key_nope = key_states[..., self.rope_dim:]

        # Compute rotary embeddings
        # IMPORTANT: Always compute for this layer because different layer types
        # (full vs sliding window) use different rope_theta values.
        # Full attention: rope_theta = 5000000
        # Sliding window: rope_theta = 10000
        # We cannot reuse cached cos/sin from other layers!
        #
        # For CP with sequence_parallel: Q/K/V have full S, use full position_ids for RoPE.
        # For CP without sequence_parallel: Q/K/V have S/CP, use local_position_ids for RoPE
        # (local_position_ids contain the correct global positions for this CP rank).
        if is_context_parallel and not self.sequence_parallel_enabled:
            rope_position_ids = local_position_ids
        else:
            rope_position_ids = position_ids
        cos_cache, sin_cache = self.rotary_emb(value_states, rope_position_ids)

        # Apply rotary position embedding to rope parts only
        query_rope, key_rope = apply_rotary_pos_emb(
            query_rope, key_rope, cos_cache, sin_cache, rope_position_ids
        )

        # Concatenate rope and non-rope parts
        query_states = torch.cat([query_rope, query_nope], dim=-1)
        key_states = torch.cat([key_rope, key_nope], dim=-1)

        # Context Parallelism: split Q and save local KV for cache
        if is_context_parallel:
            if self.sequence_parallel_enabled:
                # Q/K/V have full S. Split Q to local portion, save local KV for cache.
                # Use split_along_dim (torch.index_select) instead of Python slicing
                # because XLA tracing doesn't support dynamic tensor indices in slice notation.
                query_states = split_along_dim(query_states, dim=2, rank=cp_rank, num_partitions=self.cp_degree)
                key_states_for_cache = split_along_dim(key_states, dim=2, rank=cp_rank, num_partitions=self.cp_degree)
                value_states_for_cache = split_along_dim(value_states, dim=2, rank=cp_rank, num_partitions=self.cp_degree)
                q_len = q_len // self.cp_degree
                # K/V stay at full S for attention computation
            else:
                # Q/K/V have S/CP. Save local KV for cache, then all-gather K/V.
                key_states_for_cache = key_states
                value_states_for_cache = value_states
                key_states = gather_from_tensor_model_parallel_region_with_dim(
                    key_states, gather_dim=2,
                    process_group=get_context_parallel_attention_cp_group(),
                )
                value_states = gather_from_tensor_model_parallel_region_with_dim(
                    value_states, gather_dim=2,
                    process_group=get_context_parallel_attention_cp_group(),
                )
                # Q stays at S/CP
        else:
            # Store key/value states BEFORE GQA repeat for KV cache
            key_states_for_cache = key_states
            value_states_for_cache = value_states

        # NOTE: No runtime V pad or KV head pad here. Previously this block
        # padded V's head_dim from 128 to 192 at runtime so K and V could share
        # the same KV cache shape, and separately repeated KV heads for layers
        # that have fewer kv heads than the cache expects. Both are now
        # handled in the preprocess script (V is physically padded to 192
        # with zero rows, and CONVERT_TO_MHA at TP=64 makes local_num_kv_heads
        # == local_cache_kv_heads, so head-pad is a no-op). Runtime pad/slice
        # on KV cache storage was the source of the long-decode output drift.

        # Repeat KV heads for GQA (only needed without CONVERT_TO_MHA)
        # With CONVERT_TO_MHA, K/V already have num_attention_heads
        num_key_value_groups = self.local_num_heads // self.local_num_kv_heads
        if num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

        if is_token_gen:
            # Token generation: use decomposed attention with prior (cached) and active (current) KV
            # past_key_value[0] = cached K, shape [bsz, cache_kv_heads, kv_seq_len, head_dim]
            # past_key_value[1] = cached V, shape [bsz, cache_kv_heads, kv_seq_len, head_dim]
            K_prior = past_key_value[0]
            V_prior = past_key_value[1]

            # Repeat cached KV for GQA (only needed without CONVERT_TO_MHA)
            # With CONVERT_TO_MHA, cached K/V already have num_attention_heads
            if num_key_value_groups > 1:
                K_prior = K_prior.repeat_interleave(num_key_value_groups, dim=1)
                V_prior = V_prior.repeat_interleave(num_key_value_groups, dim=1)

            # Compute attention on prior (cached) KV
            # K_prior shape: [bsz, num_heads, kv_seq_len, head_dim]
            prior_scores = torch.matmul(query_states, K_prior.transpose(-2, -1)) * self.scaling

            # Apply attention mask to prior scores
            if attention_mask is not None:
                # Convert boolean mask to additive mask if needed
                if attention_mask.dtype == torch.bool:
                    prior_scores = prior_scores.masked_fill(~attention_mask, float('-inf'))
                else:
                    prior_scores = prior_scores + attention_mask

            # Apply sliding window mask for SWA layers
            if self.is_sliding_window and self.sliding_window_size is not None and position_ids is not None:
                kv_seq_len = prior_scores.size(-1)
                current_pos = position_ids[0, 0]
                pos_indices = torch.arange(kv_seq_len, device=prior_scores.device)
                sliding_mask = pos_indices >= (current_pos - self.sliding_window_size + 1)
                sliding_mask = sliding_mask[None, None, None, :]
                prior_scores = prior_scores.masked_fill(~sliding_mask, float('-inf'))

            prior_scores = prior_scores.to(torch.float32)

            # Compute attention on active (current) KV
            active_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
            active_scores = active_scores.to(torch.float32)

            # Combined softmax over prior and active scores
            all_scores = torch.cat([prior_scores, active_scores], dim=-1)

            # Add attention sink bias (following HF implementation)
            # This must be applied to token generation as well!
            use_sink = self._use_sink_bias and self.attention_sink_bias is not None
            if use_sink:
                tp_rank = parallel_state.get_tensor_model_parallel_rank() if parallel_state.model_parallel_is_initialized() else 0
                local_sink = self.attention_sink_bias[tp_rank * self.local_num_heads:(tp_rank + 1) * self.local_num_heads]
                sink_bias = local_sink.reshape(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
                all_scores = torch.cat([all_scores, sink_bias], dim=-1)

            # Numerical stability: subtract max before softmax
            all_scores = all_scores - all_scores.max(dim=-1, keepdim=True).values
            attn_weights = F.softmax(all_scores, dim=-1, dtype=torch.float32)

            # Drop the sink column after softmax
            if use_sink:
                attn_weights = attn_weights[..., :-1]

            # Split attention weights back
            prior_weights = attn_weights[..., :-q_len].to(V_prior.dtype)
            active_weights = attn_weights[..., -q_len:].to(value_states.dtype)

            # Compute attention outputs
            attn_prior = torch.matmul(prior_weights, V_prior)
            attn_active = torch.matmul(active_weights, value_states)
            attn_output = attn_prior + attn_active
        else:
            # Context encoding: standard attention
            # With CP: Q is local [B, H, S/CP, D], K/V are full [B, H, S, D]
            # Without CP: Q/K/V all have same seq_len
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

            # Apply attention mask (additive mask: 0 = attend, -inf = mask out)
            # The framework creates boolean masks, so we need to convert them
            # With CP: attention_mask is already split to [B, 1, S/CP, S] (local Q rows, full K cols)
            if attention_mask is not None:
                # Convert boolean mask to additive mask if needed
                if attention_mask.dtype == torch.bool:
                    # True = attend (0), False = mask (-inf)
                    additive_mask = torch.zeros_like(attn_weights)
                    additive_mask = additive_mask.masked_fill(~attention_mask, float('-inf'))
                    attn_weights = attn_weights + additive_mask
                else:
                    # Already additive mask
                    attn_weights = attn_weights + attention_mask

            # Apply sliding window mask for SWA layers
            if self.is_sliding_window and self.sliding_window_size is not None:
                kv_seq_len = attn_weights.size(-1)
                if is_context_parallel:
                    # With CP: Q has local seq len, K has full seq len.
                    # Use local_position_ids for correct global Q positions.
                    row_idx = local_position_ids[0].unsqueeze(1).to(attn_weights.device)
                else:
                    row_idx = torch.arange(kv_seq_len, device=attn_weights.device).unsqueeze(1)
                col_idx = torch.arange(kv_seq_len, device=attn_weights.device).unsqueeze(0)
                # Causal: col <= row, and within window: col >= row - window_size + 1
                sliding_mask = (col_idx <= row_idx) & (col_idx >= row_idx - self.sliding_window_size + 1)
                sliding_mask = sliding_mask[None, None, :, :]
                # Convert to additive mask
                attn_weights = attn_weights.masked_fill(~sliding_mask, float('-inf'))

            # Add attention sink bias (following HF implementation)
            # This adds an extra "sink" column to attention weights
            use_sink = self._use_sink_bias and self.attention_sink_bias is not None
            if use_sink:
                # Get local portion of sink bias for this TP rank
                tp_rank = parallel_state.get_tensor_model_parallel_rank() if parallel_state.model_parallel_is_initialized() else 0
                local_sink = self.attention_sink_bias[tp_rank * self.local_num_heads:(tp_rank + 1) * self.local_num_heads]
                # Reshape and expand: [local_num_heads] -> [bsz, local_num_heads, q_len, 1]
                sink_bias = local_sink.reshape(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
                attn_weights = torch.cat([attn_weights, sink_bias], dim=-1)

            # Numerical stability: subtract max before softmax (like HF implementation)
            attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values

            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

            # Drop the sink column after softmax
            if use_sink:
                attn_weights = attn_weights[..., :-1]

            attn_weights = attn_weights.to(value_states.dtype)

            # Apply attention to values
            attn_output = torch.matmul(attn_weights, value_states)

        # Slice off the padded v_head_dim tail. V was physically padded from
        # real_v_head_dim (128) to head_dim (192) in the preprocess script so
        # it can share NxDI's KV cache shape with K; the padded dims hold
        # zero weights, so the attention output in those dims is always zero
        # and the slice is numerically exact.
        if self.attn_real_v_head_dim < self.attn_v_head_dim:
            attn_output = attn_output[..., :self.attn_real_v_head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.local_num_heads * self.attn_real_v_head_dim)

        # Context Parallelism: gather output across CP ranks BEFORE o_proj.
        # With SP enabled, o_proj scatters along seq dim. The input must have full S
        # (not S/CP), otherwise the SP-scattered output won't match the residual.
        # Without SP, gather after o_proj to restore full seq_len for residual.
        if is_context_parallel:
            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output, gather_dim=1,
                process_group=get_context_parallel_attention_cp_group(),
            )

        attn_output = self.o_proj(attn_output)

        # Prepare KV cache output - return as tuple for KV cache manager
        # Return LOCAL key/value states for cache (each CP rank stores its portion)
        new_key_value = (key_states_for_cache, value_states_for_cache)

        return attn_output, new_key_value, cos_cache, sin_cache


class MiMoV2MLP(nn.Module):
    """Standard MLP for non-MoE layers in MiMo-V2-Flash."""

    def __init__(self, config: MiMoV2InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Use the dense intermediate size for non-MoE layers
        self.intermediate_size = getattr(config, 'dense_intermediate_size', config.intermediate_size * 8)

        dtype = config.neuron_config.torch_dtype

        if parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()

            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NeuronMiMoV2DecoderLayer(nn.Module):
    """MiMo-V2-Flash Decoder Layer with hybrid attention and conditional MoE."""

    def __init__(self, config: MiMoV2InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Determine attention type for this layer
        is_sliding_window = config.layer_attention_types[layer_idx] == "sliding_window"
        self.attention_type = "sliding_window" if is_sliding_window else "full"

        # Create attention module
        self.self_attn = NeuronMiMoV2Attention(
            config=config,
            layer_idx=layer_idx,
            is_sliding_window=is_sliding_window,
        )

        # Determine if this layer uses MoE
        self.uses_moe = config.layer_uses_moe[layer_idx]

        # Create MLP/MoE module
        if self.uses_moe:
            self.mlp = initialize_moe_module(config=config)
        else:
            self.mlp = MiMoV2MLP(config)

        # Layer norms
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )

        # Config flags
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass for decoder layer."""

        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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

        # MLP/MoE with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.uses_moe:
            hidden_states = self.mlp(hidden_states, padding_mask)[0]
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronMiMoV2Model(NeuronBaseModel):
    """MiMo-V2-Flash Model for NXD inference."""

    def setup_attr_for_model(self, config: MiMoV2InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        # Check if we need GQA CONVERT_TO_MHA mode
        # When tp_degree > num_kv_heads, we replicate K/V to match num_attention_heads
        min_kv_heads = min(
            config.num_key_value_heads,
            getattr(config, 'swa_num_key_value_heads', config.num_key_value_heads)
        )
        self.use_gqa_convert_to_mha = self.tp_degree > min_kv_heads

        if self.use_gqa_convert_to_mha:
            # With CONVERT_TO_MHA, KV cache stores num_attention_heads (same as Q)
            self.num_key_value_heads = config.num_attention_heads
        else:
            # Standard GQA: use the maximum num_kv_heads for KV cache
            # (handles hybrid full/sliding window attention)
            self.num_key_value_heads = max(
                config.num_key_value_heads,
                getattr(config, 'swa_num_key_value_heads', config.num_key_value_heads)
            )

        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

        # MiMo has hybrid attention (full + sliding window)
        # NOTE: Do NOT set self.sliding_window here because it affects KV cache size globally.
        # MiMo handles sliding window per-layer in the attention module itself.
        # Setting has_mixed_attn = True enables proper mask creation without affecting cache size.
        self.has_mixed_attn = True

    def init_model(self, config: MiMoV2InferenceConfig):
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )

        self.layers = nn.ModuleList([
            NeuronMiMoV2DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
        )


def _replicate_kv_weights_for_convert_to_mha(
    tensor: torch.Tensor,
    source_heads: int,
    target_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Replicate K/V weights from source_heads to target_heads for CONVERT_TO_MHA.

    Args:
        tensor: Weight tensor of shape [source_heads * head_dim, hidden_size]
        source_heads: Number of source KV heads
        target_heads: Number of target heads (num_attention_heads)
        head_dim: Head dimension

    Returns:
        Replicated tensor of shape [target_heads * head_dim, hidden_size]
    """
    if tensor is None or source_heads >= target_heads:
        return tensor

    repeats = target_heads // source_heads

    # Reshape to [source_heads, head_dim, hidden_size]
    original_shape = tensor.shape
    tensor = tensor.view(source_heads, head_dim, -1)

    # Repeat along head dimension
    tensor = tensor.repeat_interleave(repeats, dim=0)

    # Reshape back to [num_heads * head_dim, hidden_size]
    tensor = tensor.view(-1, original_shape[-1])

    return tensor


def convert_mimo_v2_hf_to_neuron_state_dict(
    neuron_state_dict: Dict[str, Any],
    config: MiMoV2InferenceConfig,
) -> Dict[str, Any]:
    """Convert HuggingFace MiMo-V2-Flash weights to Neuron format.

    This handles:
    1. Router weight renaming
    2. Expert weight concatenation and transposition
    3. FP8 dequantization if needed
    4. K/V weight replication for CONVERT_TO_MHA mode
    """

    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # Dequantize layers if needed
    _maybe_dequantize_layer(neuron_state_dict, config)

    # Add rank utility tensors
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    # Determine if CONVERT_TO_MHA is needed
    tp_degree = config.neuron_config.tp_degree
    num_attention_heads = config.num_attention_heads

    # MiMo-V2-Flash has different KV heads for full and sliding window attention
    full_num_kv_heads = config.num_key_value_heads  # 4
    swa_num_kv_heads = config.swa_num_key_value_heads  # 8

    # Check if we need to replicate K/V weights
    full_use_convert_to_mha = tp_degree > full_num_kv_heads
    swa_use_convert_to_mha = tp_degree > swa_num_kv_heads

    for layer_idx in range(config.num_hidden_layers):
        # Add rank utility for attention
        neuron_state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # Determine attention type for this layer
        is_sliding_window = config.layer_attention_types[layer_idx] == "sliding_window"

        if is_sliding_window:
            src_num_kv_heads = swa_num_kv_heads
            use_convert_to_mha = swa_use_convert_to_mha
            head_dim = config.swa_head_dim  # 192
            v_head_dim = config.swa_v_head_dim  # 128
        else:
            src_num_kv_heads = full_num_kv_heads
            use_convert_to_mha = full_use_convert_to_mha
            head_dim = config.head_dim  # 192
            v_head_dim = config.v_head_dim  # 128

        # Replicate K/V weights if CONVERT_TO_MHA is needed
        if use_convert_to_mha:
            k_proj_key = f"layers.{layer_idx}.self_attn.k_proj.weight"
            v_proj_key = f"layers.{layer_idx}.self_attn.v_proj.weight"

            if k_proj_key in neuron_state_dict:
                neuron_state_dict[k_proj_key] = _replicate_kv_weights_for_convert_to_mha(
                    neuron_state_dict[k_proj_key],
                    src_num_kv_heads,
                    num_attention_heads,
                    head_dim,
                )

            if v_proj_key in neuron_state_dict:
                # V is pre-padded to head_dim in the preprocess script so K/V
                # can share KV cache shape; use head_dim (not v_head_dim) here.
                neuron_state_dict[v_proj_key] = _replicate_kv_weights_for_convert_to_mha(
                    neuron_state_dict[v_proj_key],
                    src_num_kv_heads,
                    num_attention_heads,
                    head_dim,
                )

            # FP8 path: replicate per-row scales ([src_heads*head_dim, 1]) in
            # lockstep with the weights. Without this the shard_weights step
            # rejects the scale shape mismatch (e.g. [12,1] vs expected [192,1]).
            # BF16 has no .scale key, so this loop is a no-op there. Both
            # k_proj and v_proj scales use head_dim since V is pre-padded.
            for proj, hd in (("k_proj", head_dim), ("v_proj", head_dim)):
                scale_key = f"layers.{layer_idx}.self_attn.{proj}.scale"
                if scale_key in neuron_state_dict:
                    neuron_state_dict[scale_key] = _replicate_kv_weights_for_convert_to_mha(
                        neuron_state_dict[scale_key],
                        src_num_kv_heads,
                        num_attention_heads,
                        hd,
                    )

        # Only convert MoE layers
        if not config.layer_uses_moe[layer_idx]:
            continue

        # Check if this layer has MoE weights
        gate_key = f"layers.{layer_idx}.mlp.gate.weight"
        if gate_key not in neuron_state_dict:
            continue

        # Rename router weights
        neuron_state_dict[f"layers.{layer_idx}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[gate_key].detach().clone()
        )
        del neuron_state_dict[gate_key]

        # Get dimensions from first expert
        expert_0_gate = f"layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
        if expert_0_gate not in neuron_state_dict:
            continue

        intermediate_size, hidden_size = neuron_state_dict[expert_0_gate].shape
        device = neuron_state_dict[expert_0_gate].device
        dtype = neuron_state_dict[expert_0_gate].dtype

        num_experts = config.n_routed_experts

        # Concatenate gate and up projections
        gate_up_proj = torch.empty(
            num_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )

        for e in range(num_experts):
            gate_proj_weights = neuron_state_dict[
                f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
            ].T.detach().clone()
            up_proj_weights = neuron_state_dict[
                f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"
            ].T.detach().clone()

            gate_up_proj[e, :, :intermediate_size] = gate_proj_weights
            gate_up_proj[e, :, intermediate_size:] = up_proj_weights

            del neuron_state_dict[f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"]
            del neuron_state_dict[f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"]

        # Pad if needed
        pad_size = getattr(config, "moe_intermediate_pad_size", 0)
        if pad_size > 0:
            gate_up_proj = gate_up_proj.reshape(num_experts, hidden_size, 2, -1)
            gate_up_proj = F.pad(gate_up_proj, (0, pad_size))
            gate_up_proj = gate_up_proj.reshape(num_experts, hidden_size, -1)

        neuron_state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # Convert down projections
        down_proj = torch.empty(
            num_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )

        for e in range(num_experts):
            down_proj_weights = neuron_state_dict[
                f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
            ].T.detach().clone()
            down_proj[e] = down_proj_weights
            del neuron_state_dict[f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"]

        # Pad if needed
        if pad_size > 0:
            down_proj = F.pad(down_proj, (0, 0, 0, pad_size))

        neuron_state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    # --- Expand MoE blockwise scales along the TP-partitioned dim (FP8 only). ---
    # NxDI's shard_checkpoint splits the scale on its partition dim into
    # `per_partition_size = dim_size / tp_degree`. At TP=64 both projections
    # have per-rank "intermediate" smaller than the 128-wide scale block, so
    # several ranks share one scale block — we need to replicate scale entries
    # along that dim. Adjacent ranks whose weight falls inside the same
    # 128-wide block genuinely share that block's scale. No-op when the
    # .scale keys are absent (BF16 path).
    if getattr(config.neuron_config, "quantized", False):
        tp = config.neuron_config.tp_degree
        for layer_idx in range(config.num_hidden_layers):
            if not config.layer_uses_moe[layer_idx]:
                continue

            # down_proj (RowParallel on intermediate dim). Scale: [E, I_blocks, H_blocks]
            dp_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.scale"
            if dp_key in neuron_state_dict:
                s = neuron_state_dict[dp_key]
                i_blocks = s.shape[1]
                h_blocks = s.shape[2]
                intermediate = i_blocks * 128
                i_per_rank = intermediate // tp  # 32 @ TP=64 for IM=2048
                if i_per_rank < 128:
                    ranks_per_block = 128 // i_per_rank  # 4 @ TP=64
                    s_exp = s.unsqueeze(2).expand(-1, -1, ranks_per_block, -1)
                    s_exp = s_exp.reshape(s.shape[0], i_blocks * ranks_per_block, h_blocks)
                    assert s_exp.shape[1] == tp, (
                        f"down_proj.scale expansion produced {s_exp.shape[1]} rows, "
                        f"expected TP={tp}"
                    )
                    neuron_state_dict[dp_key] = s_exp.contiguous()

            # gate_up_proj (ColumnParallel on 2*intermediate dim, gate|up fused
            # along last axis). Scale: [E, H_blocks, 2*I_blocks] stored as
            # [gate_half | up_half]. Module parameter has per-rank last-dim=1
            # (via _apply_blockwise_scale_stride_fix patch forcing
            # partition_stride=1), so the full scale must have last-dim=tp
            # with gate entries 0..tp/2 and up entries tp/2..tp. Expand each
            # half independently to preserve the gate/up boundary when NxD
            # does `split(per_partition=2*I/tp, dim=-1)`.
            gu_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.scale"
            if gu_key in neuron_state_dict:
                s = neuron_state_dict[gu_key]
                h_blocks = s.shape[1]
                two_i_blocks = s.shape[2]
                assert two_i_blocks % 2 == 0, (
                    f"gate_up_proj.scale last dim must be 2*i_blocks, got {two_i_blocks}"
                )
                i_blocks = two_i_blocks // 2
                intermediate = i_blocks * 128
                out_per_rank = (2 * intermediate) // tp  # 64 @ TP=64 for IM=2048
                if out_per_rank < 128:
                    assert tp % 2 == 0, f"TP={tp} must be even for gate/up scale split"
                    ranks_per_half = tp // 2  # 32 @ TP=64
                    assert ranks_per_half % i_blocks == 0, (
                        f"ranks_per_half={ranks_per_half} must be divisible by "
                        f"i_blocks={i_blocks}"
                    )
                    ranks_per_block = ranks_per_half // i_blocks  # 2 @ TP=64 (i_blocks=16)
                    gate_half = s[..., :i_blocks]   # [E, H_blocks, i_blocks]
                    up_half = s[..., i_blocks:]
                    gate_exp = (
                        gate_half.unsqueeze(-1)
                        .expand(-1, -1, -1, ranks_per_block)
                        .reshape(s.shape[0], h_blocks, ranks_per_half)
                    )
                    up_exp = (
                        up_half.unsqueeze(-1)
                        .expand(-1, -1, -1, ranks_per_block)
                        .reshape(s.shape[0], h_blocks, ranks_per_half)
                    )
                    s_exp = torch.cat([gate_exp, up_exp], dim=-1)
                    assert s_exp.shape[-1] == tp, (
                        f"gate_up_proj.scale expansion produced {s_exp.shape[-1]} "
                        f"entries, expected TP={tp}"
                    )
                    neuron_state_dict[gu_key] = s_exp.contiguous()

    return neuron_state_dict


def _maybe_dequantize_layer(
    neuron_state_dict: Dict[str, Any],
    config: MiMoV2InferenceConfig,
):
    """Dequantize FP8 layers if present."""
    scale_layers = []

    for layer_key in list(neuron_state_dict.keys()):
        if "_scale_inv" in layer_key:
            scales = neuron_state_dict[layer_key]
            scale_layers.append(layer_key)

            fp8_layer_name = layer_key.replace("_scale_inv", "")
            if fp8_layer_name not in neuron_state_dict:
                continue

            fp8_layer = neuron_state_dict[fp8_layer_name]

            # Get block size from config if available
            if hasattr(config, 'quantization_config') and config.quantization_config:
                block_size = config.quantization_config.get("weight_block_size", [128, 128])
            else:
                block_size = [128, 128]

            # Expand scales and dequantize
            scales_expanded = scales.repeat_interleave(block_size[0], dim=0)
            scales_expanded = scales_expanded.repeat_interleave(block_size[1], dim=1)

            # Ensure shapes match
            if scales_expanded.shape != fp8_layer.shape:
                scales_expanded = scales_expanded[:fp8_layer.shape[0], :fp8_layer.shape[1]]

            scaled_layer = fp8_layer.to(torch.float32) * scales_expanded.to(torch.float32)
            neuron_state_dict[fp8_layer_name] = scaled_layer.to(config.neuron_config.torch_dtype)

    # Remove scale layers
    for scale_layer in scale_layers:
        del neuron_state_dict[scale_layer]


class NeuronMiMoV2ForCausalLM(NeuronBaseForCausalLM):
    """MiMo-V2-Flash for Causal Language Modeling on Neuron."""

    _model_cls = NeuronMiMoV2Model

    def __init__(self, *args, **kwargs):
        # Install FP8 monkey-patches BEFORE super().__init__ so the patched
        # RouterTopK.__init__ and quantization layer classes are in effect
        # when NxDI builds the decoder (and instantiates routers). Harnesses
        # that drive us via model.compile()/model.load() (e.g. vllm-neuron)
        # call those methods AFTER construction, so patching from inside
        # compile()/load() is too late — RouterTopK instances would already
        # lack our e_score_correction_bias parameter, silently routing tokens
        # to wrong experts and producing gibberish output.
        #
        # _install_fp8_patches() reads self.neuron_config, which needs to
        # exist; grab it from the args or the config arg the same way the
        # base class does.
        ncfg = kwargs.get("config") or (args[1] if len(args) > 1 else None)
        if ncfg is not None and getattr(getattr(ncfg, "neuron_config", None), "quantized", False):
            self._apply_ep_scale_fix()
            self._apply_blockwise_scale_stride_fix()
            self._apply_2d_per_channel_fix()
            self._apply_router_noaux_tc_fix()
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_hf_model(model_path: str, **kwargs):
        """Load HuggingFace model.

        Note: MiMo-V2-Flash uses custom code, so we need trust_remote_code=True
        """
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls) -> Type[MiMoV2InferenceConfig]:
        return MiMoV2InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, Any],
        config: MiMoV2InferenceConfig,
    ) -> Dict[str, Any]:
        return convert_mimo_v2_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    # ------------------------------------------------------------------
    # FP8 quantized-inference monkey-patches (no-op unless quantized=True).
    #
    # Reconcile the preprocessed Neuron-FP8 checkpoint (blockwise-MoE +
    # per-row-attn) with NxDI's global blockwise_symmetric q_config. All
    # four are gated by self.neuron_config.quantized so the BF16 path is
    # completely untouched.
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_ep_scale_fix():
        """Skip per-channel `scale` params when marking expert-parallel
        weights; they have shape [1, 1, W] and cannot be EP-sharded."""
        from neuronx_distributed.modules.moe.moe_parallel_layers import (
            ExpertFusedLinear,
        )

        if getattr(ExpertFusedLinear, "_mimo_v2_ep_scale_patched", False):
            return

        def _patched_mark(
            self_inner,
            iterable=None,
            expert_parallel_group_size=None,
            is_prefill=True,
            expert_distribution=None,
        ):
            from neuronx_distributed.parallel_layers.parallel_state import (
                get_expert_model_parallel_size,
            )

            if expert_parallel_group_size is None:
                expert_parallel_group_size = get_expert_model_parallel_size()

            if expert_parallel_group_size > 1:
                if iterable is None:
                    params_to_mark = []
                    for name, p in self_inner.named_parameters():
                        if name == "scale" and p.shape[0] == 1:
                            continue
                        params_to_mark.append(p)
                    iterable = params_to_mark

                for p in iterable:
                    p.expert_model_parallel = True
                    if is_prefill:
                        p.is_prefill = True
                    p.expert_distribution = expert_distribution

        ExpertFusedLinear._mark_expert_parallel_weights = _patched_mark
        ExpertFusedLinear._mimo_v2_ep_scale_patched = True

    @staticmethod
    def _apply_blockwise_scale_stride_fix():
        """Force scale.partition_stride=1 for BLOCKWISE_SYMMETRIC quantization
        — stride>1 causes strided-splitting failures when per-rank weight size
        is smaller than a block."""
        from neuronx_distributed.quantization.quantization_config import (
            QuantizationType,
        )
        from neuronx_distributed.quantization.quantization_layers import (
            BaseQuantizeParallelLinear,
        )

        if getattr(BaseQuantizeParallelLinear, "_mimo_v2_blockwise_stride_patched", False):
            return

        _original_setup = BaseQuantizeParallelLinear._setup_for_scale

        def _patched_setup(self_inner, *args, **kwargs):
            _original_setup(self_inner, *args, **kwargs)
            if (
                hasattr(self_inner, "quantization_type")
                and self_inner.quantization_type == QuantizationType.BLOCKWISE_SYMMETRIC
                and hasattr(self_inner, "scale")
                and hasattr(self_inner.scale, "partition_stride")
                and self_inner.scale.partition_stride > 1
            ):
                self_inner.scale.partition_stride = 1

        BaseQuantizeParallelLinear._setup_for_scale = _patched_setup
        BaseQuantizeParallelLinear._mimo_v2_blockwise_stride_patched = True

    @staticmethod
    def _apply_2d_per_channel_fix():
        """Route 2D self_attn + layer-0 dense-MLP swaps through per_channel_symmetric.

        Flash's preprocess writes:
            - MoE experts: 3D weights with (E, out//128, in//128) blockwise scales.
            - self_attn q/k/v + layer-0 mlp gate/up/down: 2D weights with
              (out, 1) per-row scales.

        NxDI's q_config is global blockwise_symmetric (to satisfy the MoE).
        Feeding that into the 2D classes triggers
        `block axis cannot be < 0 or > 2, received 2` in _setup_for_scale
        (block axes [1, 2] exceed rank-2 weight_shape). This wraps the 2D
        classes' from_float to override q_config on the fly.
        """
        from neuronx_distributed.quantization.quantization_config import (
            QuantizationType,
        )
        from neuronx_distributed.quantization.quantization_layers import (
            QuantizedColumnParallel,
            QuantizedRowParallel,
        )

        def _wrap(cls):
            if getattr(cls, "_mimo_v2_2d_patched", False):
                return
            original_from_float = cls.from_float

            def _patched_from_float(klass, mod, q_config=None, _orig=original_from_float):
                if q_config is not None and q_config.get("quantization_type") == \
                        QuantizationType.BLOCKWISE_SYMMETRIC:
                    q_config = dict(q_config)
                    q_config["quantization_type"] = QuantizationType.PER_CHANNEL_SYMMETRIC
                    q_config["quantization_per_channel_axis"] = 0
                    q_config.pop("block_axis", None)
                    q_config.pop("block_size", None)
                if q_config is None:
                    return _orig(mod)
                return _orig(mod, q_config)

            cls.from_float = classmethod(_patched_from_float)
            cls._mimo_v2_2d_patched = True

        _wrap(QuantizedColumnParallel)
        _wrap(QuantizedRowParallel)

    @staticmethod
    def _apply_router_noaux_tc_fix():
        """Register e_score_correction_bias on NxD RouterTopK and fold it into
        top-k selection so Flash's noaux_tc routing matches HF reference.

        Flash's HF config uses topk_method='noaux_tc': each expert score is
        `sigmoid(logits) + e_score_correction_bias`, top-k indices are chosen
        from THAT biased score; the returned expert weights (affinities)
        come from the UNBIASED sigmoid(logits). NxD's stock RouterTopK is
        plain topk with no bias slot, so without this the bias is silently
        dropped and ~all tokens route to wrong experts.
        """
        from neuronx_distributed.modules.moe.routing import RouterTopK

        if getattr(RouterTopK, "_mimo_v2_noaux_tc_patched", False):
            return

        original_init = RouterTopK.__init__

        def _patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(self.num_experts, dtype=torch.float32),
                requires_grad=False,
            )

        def _patched_forward(self, hidden_states):
            router_logits = self.get_router_logits(hidden_states)
            expert_affinities = self.apply_activation_fn(router_logits)

            selection_scores = expert_affinities.to(torch.float32) + \
                self.e_score_correction_bias
            _, expert_index = torch.topk(selection_scores, self.top_k, dim=-1)

            expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
            expert_index = expert_index.detach().to(dtype=torch.long)
            return router_logits, expert_affinities, expert_index

        RouterTopK.__init__ = _patched_init
        RouterTopK.forward = _patched_forward
        RouterTopK._mimo_v2_noaux_tc_patched = True

    def _install_fp8_patches(self):
        """Install all FP8-specific runtime patches. No-op for BF16."""
        if not getattr(self.neuron_config, "quantized", False):
            return
        self._apply_ep_scale_fix()
        self._apply_blockwise_scale_stride_fix()
        self._apply_2d_per_channel_fix()
        self._apply_router_noaux_tc_fix()

    def compile(self, *args, **kwargs):
        # save_sharded_checkpoint=True serializes shards during compile() and
        # that code path reads scale.partition_stride — patches must be live.
        self._install_fp8_patches()
        return super().compile(*args, **kwargs)

    def load(self, *args, **kwargs):
        self._install_fp8_patches()
        return super().load(*args, **kwargs)

    @classmethod
    def save_quantized_state_dict(cls, model_path, config):
        """Flash ships pre-quantized FP8 safetensors via our preprocess script.
        The base implementation calls AutoModelForCausalLM.from_pretrained to
        re-quantize, which requires a CUDA GPU (finegrained_fp8 gate) and
        materializes an ~600 GB BF16 copy. Skip if the checkpoint directory
        already contains a Neuron-FP8 index produced by preprocess."""
        import os as _os
        qpath = (
            getattr(config.neuron_config, "quantized_checkpoints_path", None)
            or model_path
        )
        if qpath and _os.path.isdir(qpath):
            index = _os.path.join(qpath, "model.safetensors.index.json")
            if _os.path.isfile(index):
                return
        return super().save_quantized_state_dict(model_path, config)

    def get_compiler_args(self) -> str:
        """Get compiler arguments optimized for MiMo-V2-Flash."""
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"
        else:
            optimization_level = "-O1"

        compiler_args = (
            f"--enable-saturate-infinity "
            f"--enable-mixed-precision-accumulation "
            f"--model-type transformer "
            f"{optimization_level}"
        )

        # Add CC overlap optimization
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2'"
        )

        compiler_args += " --auto-cast=none"

        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"

        if self.neuron_config.scratchpad_page_size:
            compiler_args += f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}"

        return compiler_args
