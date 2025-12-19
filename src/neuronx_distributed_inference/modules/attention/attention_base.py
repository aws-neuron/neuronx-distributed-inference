import logging
import math
import warnings
from enum import Enum
from typing import Optional, Tuple, Callable
from dataclasses import dataclass, fields
from importlib import import_module

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager, KV_CACHE_PAD_FOR_SEQ_IDS_MASKING
from neuronx_distributed_inference.modules.attention.sink import LearnedSink

from neuronx_distributed_inference.modules.attention.attention_process_groups import (
    get_context_parallel_attention_cp_group,
    get_context_parallel_attention_tp_group,
    init_context_parallel_attention_process_groups,
    init_data_parallel_attention_process_groups,
    get_data_parallel_attention_dp_group,
    get_data_parallel_attention_tp_group,
)
from neuronx_distributed_inference.modules.chunked_prefill.flash_pa_with_schedule import (
    flash_paged_attention_with_schedule,
)
from neuronx_distributed_inference.modules.sliding_window.attention import (
    flash_fwd, FlashConfig, DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE, MIN_SLIDING_WINDOW_SEQ_TILE_SIZE
)
from neuronx_distributed_inference.models.config import InferenceConfig

from .utils import (
    apply_rotary_pos_emb,
    distributed_softmax,
    manual_softmax,
    move_heads_front,
    repeat_kv,
    get_context_parallel_reordered_tp_mapping,
    validate_tp_prefill_to_dp_decode,
    reshape_qkv_for_chunked_flash_attention_kernel,
    get_last_kv_chunk,
    get_last_kv_window,
    get_context_parallel_reordered_dp_mapping,
    order_strided_tensor,
    get_cp8_tp8_rank_ordering,
    apply_seq_id_mask,
)
from neuronx_distributed.modules.attention.utils import apply_rotary_polar_compatible, precompute_freqs_cis

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402
from neuronxcc.nki._private_kernels.prefix_caching_attention import prefix_caching_attention_fwd_isa_kernel

import neuronx_distributed as nxd
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import utils  # noqa: E402
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.parallel_state import get_kv_shared_group, get_tensor_model_parallel_group
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
)
from neuronx_distributed_inference.utils.distributed import (
    get_tp_group,
    split_along_dim,
    get_cp_rank,
    get_dp_rank,
    get_kv_head_group_number,
)

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402
from torch_neuronx.utils import get_platform_target
from neuronxcc.nki import jit
from neuronxcc.nki.compiler import skip_middle_end_transformations, enable_stack_allocator

from .gqa import GQA, GroupQueryAttention_O, GroupQueryAttention_QKV  # noqa: E402

from neuronx_distributed_inference.utils.decorator_peeling import peel_decorations


def import_nki_cte_attention_kernel():
    mod = import_module("neuronxcc.nki._pre_prod_kernels.attn_fwd")

    _has_new_kernel = False
    _has_new_pc_kernel = False
    _has_native_gqa_tp_support = False
    platform_target = get_platform_target()
    attention_nki_kernel_adapter = getattr(mod, "attention_nki_kernel_adapter", None)
    if attention_nki_kernel_adapter is None:
        return None, _has_new_kernel, _has_new_pc_kernel, _has_native_gqa_tp_support

    has_trn1_no_shared_constant_bugfix = getattr(mod, "TRN1_NO_SHARED_CONSTANT", None)
    if has_trn1_no_shared_constant_bugfix is not True:
        return None, _has_new_kernel, _has_new_pc_kernel, _has_native_gqa_tp_support

    _has_new_kernel = True
    _has_native_gqa_tp_support = getattr(mod, "NATIVE_GQA_TP_SUPPORT", False) and platform_target != "trn1"

    # attention_nki_kernel_adapter's decoration's in the compiler suffer from bug in detecting the correct target platform
    # This can be circumvented by utilizing mode='torchxla' instead of mode='trace'. Which is what the peeling of decorations and redecorations do here
    # This should not be done in the compiler side since it results in the loss of JAX compatibility and causes unit tests to fail
    # TODO: Remove peeling and redecorating hack once the bug is resolved
    undecorated_attention_kernel_adapter = peel_decorations(attention_nki_kernel_adapter)
    signature = undecorated_attention_kernel_adapter.sign
    if "k_prior" in signature.parameters and platform_target != "trn1":
        _has_new_pc_kernel = True
    decorated_attention_kernel_adapter = jit(undecorated_attention_kernel_adapter, mode='torchxla', platform_target=platform_target, show_compiler_tb=True, debug_kernel=True)
    decorated_attention_kernel_adapter = skip_middle_end_transformations(decorated_attention_kernel_adapter)
    decorated_attention_kernel_adapter = enable_stack_allocator(decorated_attention_kernel_adapter, log_level=logging.INFO)

    return decorated_attention_kernel_adapter, _has_new_kernel, _has_new_pc_kernel, _has_native_gqa_tp_support


_flash_fwd_call_nki, _has_new_kernel, _has_new_pc_kernel, _has_native_gqa_tp_support = import_nki_cte_attention_kernel()
_flash_fwd_call_bir = nki_jit()(attention_isa_kernel)
_flash_fwd_call_strided_context_parallel = nki_jit()(attention_isa_kernel)
_flash_fwd_pc_call_nki = _flash_fwd_call_nki
_flash_fwd_pc_call_bir = nki_jit()(prefix_caching_attention_fwd_isa_kernel)

logger = logging.getLogger("Neuron")

try:
    from neuronxcc.nki._private_kernels.attention import attention_tkg_fwd_isa_kernel
    _attn_builtin_token_gen_call = nki_jit()(attention_tkg_fwd_isa_kernel)
except ImportError:
    logger.warning(
        "Use a more recent neuron compiler version to enable builtin token-gen attention kernel"
    )
    _attn_builtin_token_gen_call = None

try:
    from neuronxcc.nki._pre_prod_kernels.attention_token_gen import attention_token_gen_kernel
except ImportError:
    logger.warning(
        "Use a more recent neuron compiler version to enable token-gen attention NKI kernel"
    )
    attention_token_gen_kernel = None

try:
    from neuronxcc.nki._pre_prod_kernels.attention_token_gen import llama3_nki_attention_block_token_gen_kernel
except ImportError:
    logger.warning(
        "Use a more recent neuron compiler version to enable token-gen attention block NKI kernel"
    )
    llama3_nki_attention_block_token_gen_kernel = None

try:
    from neuronxcc.nki._private_kernels.attention_cte import (
        llama3_nki_attention_block_cte_kernel,
    )
except ImportError:
    logger.warning(
        "Use a more recent neuron compiler version to enable context encoding attention block NKI kernel"
    )
    llama3_nki_attention_block_cte_kernel = None


class FlashAttentionStrategy(Enum):
    NONE = 0
    UNSHARDED_KERNEL = 1
    SHARDED_KERNEL = 2
    CONTEXT_PARALLEL_KERNEL = 3
    STRIDED_CONTEXT_PARALLEL_KERNEL = 4
    SLIDING_WINDOW_KERNEL = 5  # use flash_fwd NKI kernel for SWA


class QKNormPlacement(Enum):
    PRE_ROPE = 0
    POST_ROPE = 1


@dataclass(frozen=True)
class NeuronAttentionBaseOutput:
    hidden_states: torch.tensor
    present_key_value: torch.tensor
    cos_cache: Optional[torch.tensor] = None
    sin_cache: Optional[torch.tensor] = None
    residual: Optional[torch.tensor] = None
    attn_input_hidden_states: Optional[torch.tensor] = None

    # maintain old unpacking behavior
    def __iter__(self):
        return iter([self.hidden_states, self.present_key_value, self.cos_cache, self.sin_cache])

    # maintain old tuple indexing behavior
    def __getitem__(self, i):
        return getattr(self, fields(self)[i].name)


class NeuronAttentionBase(nn.Module):
    """
    This base attention class implements the core Neuron related adaptation including
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self,
                 config: InferenceConfig,
                 *,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 head_dim: int = None,
                 rotary_emb=None,
                 rope_theta: float = None,
                 use_scaled_rope: bool = False,
                 rms_norm_eps: float = None,
                 use_qk_norm: bool = False,
                 qk_norm_placement: QKNormPlacement = QKNormPlacement.PRE_ROPE,
                 q_layernorm: Callable = None,
                 k_layernorm: Callable = None,
                 clip_qkv: float = None,
                 qkv_bias: bool = False,
                 o_bias: bool = False,
                 num_cores_per_group: int = 1,
                 sequence_parallel_enabled: bool = None,
                 attention_chunk_size: int = None,
                 is_post_global_attn_layer: bool = False,
                 is_pre_global_attn_layer: bool = False,
                 sliding_window: int = None,
                 tensor_model_parallel_group: Optional[ProcessGroup] = None,
                 o_proj_layer_name: str = "o_proj",
                 learned_sinks_size: Optional[int] = None,
                 optimize_interleave_attn: bool = False):

        super().__init__()

        self.config = config
        self.neuron_config = config.neuron_config

        self.tensor_model_parallel_group = None
        self.rank_util = None

        if tensor_model_parallel_group is not None:
            self.tensor_model_parallel_group = tensor_model_parallel_group
        elif nxd.parallel_layers.parallel_state.model_parallel_is_initialized() and self.config.neuron_config.cp_degree > 1 and self.neuron_config.is_prefill_stage:
            init_context_parallel_attention_process_groups(config)
            self.tensor_model_parallel_group = get_context_parallel_attention_tp_group()
        elif nxd.parallel_layers.parallel_state.model_parallel_is_initialized() and self.config.neuron_config.attention_dp_degree > 1 and not self.neuron_config.is_prefill_stage:
            init_data_parallel_attention_process_groups(config)
            self.tensor_model_parallel_group = get_data_parallel_attention_tp_group()
        elif nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
            self.tensor_model_parallel_group = nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()

        if self.tensor_model_parallel_group is not None:
            self.rank_util = SPMDRank(config.neuron_config.tp_degree)

        self.tp_degree = self.neuron_config.tp_degree if self.tensor_model_parallel_group is None else self.tensor_model_parallel_group.size()
        self.cp_degree = self.neuron_config.cp_degree
        self.dp_degree = self.neuron_config.attention_dp_degree

        self.rpl_reduce_dtype = self.neuron_config.rpl_reduce_dtype
        self.torch_dtype = config.neuron_config.attention_dtype if config.neuron_config.attention_dtype is not None else config.neuron_config.torch_dtype
        self.fused_qkv = self.neuron_config.fused_qkv
        self.qkv_cte_nki_kernel_fuse_rope = self.neuron_config.qkv_cte_nki_kernel_fuse_rope

        # Accounts for cases where some sub-modules always have SP enabled / disabled
        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled if sequence_parallel_enabled is None else sequence_parallel_enabled
        self.qkv_proj_sp_enabled = self.o_proj_sp_enabled = self.sequence_parallel_enabled
        self.attention_chunk_size = attention_chunk_size
        self.optimize_interleave_attn = optimize_interleave_attn

        # Determine if chunked attn needs to be disabled
        if self.attention_chunk_size and self.attention_chunk_size >= self.config.neuron_config.seq_len:
            logger.warning(f"attention chunk size {self.attention_chunk_size} is greater than or equal to seq_len {self.config.neuron_config.seq_len}. Chunked attention is disabled")
            self.attention_chunk_size = None
        self.is_post_global_attn_layer = is_post_global_attn_layer
        self.is_pre_global_attn_layer = is_pre_global_attn_layer
        if self.is_post_global_attn_layer and self.is_pre_global_attn_layer:
            self.is_pre_global_attn_layer = False
            self.is_post_global_attn_layer = False
            self.optimize_interleave_attn = False  # turning off the optimal flow because we have to gather and regather again for the [chunked, global, chunked, global ...] attn pattern

        if self.attention_chunk_size and self.cp_degree > 1 and self.neuron_config.is_prefill_stage:
            self.qkv_proj_sp_enabled = False
            if self.optimize_interleave_attn and self.is_pre_global_attn_layer or not self.optimize_interleave_attn:
                self.o_proj_sp_enabled = False

        self.windowed_context_encoding_size = config.neuron_config.windowed_context_encoding_size
        self.sliding_window = sliding_window
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.padding_side = self.neuron_config.padding_side
        self.flash_decoding_enabled = self.neuron_config.flash_decoding_enabled
        self.mlp_kernel_enabled = self.neuron_config.mlp_kernel_enabled
        self.attn_kernel_enabled = self.neuron_config.attn_kernel_enabled
        self.strided_context_parallel_kernel_enabled = self.neuron_config.strided_context_parallel_kernel_enabled
        self.attn_tkg_builtin_kernel_enabled = self.neuron_config.attn_tkg_builtin_kernel_enabled
        self.attn_tkg_nki_kernel_enabled = self.neuron_config.attn_tkg_nki_kernel_enabled
        self.attn_block_tkg_nki_kernel_enabled = self.neuron_config.attn_block_tkg_nki_kernel_enabled
        self.attn_block_tkg_nki_kernel_cache_update = self.neuron_config.attn_block_tkg_nki_kernel_cache_update
        self.attn_block_cte_nki_kernel_enabled = self.neuron_config.attn_block_cte_nki_kernel_enabled
        self.k_cache_transposed = self.neuron_config.k_cache_transposed
        self.logical_nc_config = self.neuron_config.logical_nc_config
        self.qk_layernorm = self.neuron_config.qk_layernorm

        if self.sliding_window and self.neuron_config.enable_fused_speculation:
            assert self.attn_block_tkg_nki_kernel_enabled and self.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention, 'Currently we only support speculative decoding with sliding window attention when the cascaded tkg attention kernel is enabled'

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.rotary_emb = rotary_emb
        self.rope_theta = rope_theta
        self.use_scaled_rope = use_scaled_rope

        self.inv_freqs = None
        if self.attn_tkg_builtin_kernel_enabled:
            self.inv_freqs = rotary_emb.get_inv_freqs().unsqueeze(1)

        self.num_cores_per_group = num_cores_per_group
        self.qkv_bias = qkv_bias
        self.o_bias = o_bias
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm
        # applying norm before and after norm are equivalent operations, but precision accumulates differently
        self.qk_norm_placement = qk_norm_placement
        self.clip_qkv = clip_qkv
        self.o_proj_layer_name = o_proj_layer_name
        self.q_layernorm = q_layernorm
        self.k_layernorm = k_layernorm

        self.learned_sinks_size = learned_sinks_size
        self.is_eagle3_draft = self.neuron_config.is_eagle3 and self.neuron_config.is_eagle_draft
        self.init_gqa_properties()

        self.qk_norm = None
        if use_qk_norm:
            self.init_qk_norm()

    def init_tkg_cp_qkv_o_proj(self, process_group, rank_ordering=None):
        qkv_hidden_size = self.hidden_size * 2 if self.is_eagle3_draft else self.hidden_size
        self.tkg_qkv_proj = GroupQueryAttention_QKV(
            hidden_size=qkv_hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=process_group.size(),
            dtype=self.torch_dtype,
            bias=self.qkv_bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=process_group,
            rms_norm_eps=self.rms_norm_eps,
            qkv_kernel_enabled=self.neuron_config.qkv_kernel_enabled,
            qkv_nki_kernel_enabled=self.neuron_config.qkv_nki_kernel_enabled,
            fused_rmsnorm_skip_gamma=self.neuron_config.fused_rmsnorm_skip_gamma,
            logical_nc_config=self.neuron_config.logical_nc_config,
            qkv_kernel_nbsd_layout=self.neuron_config.qkv_kernel_nbsd_layout,
            on_cpu=self.neuron_config.on_cpu,
            rank_ordering=rank_ordering,
        )
        self.tkg_o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=process_group.size(),
            dtype=self.torch_dtype,
            bias=self.o_bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=process_group,
            rpl_reduce_dtype=self.rpl_reduce_dtype,
            # Mega kernels include QKV projection, RoPE, attention, and output projection
            # out_proj kernel will be enabled if mega kernel is enabled
            out_proj_kernel_enabled=self.attn_block_tkg_nki_kernel_enabled or self.neuron_config.out_proj_kernel_enabled,
            logical_nc_config=self.neuron_config.logical_nc_config,
            rank_ordering=rank_ordering,
        )
        if self.learned_sinks_size is not None:
            self.tkg_learned_sinks = LearnedSink(self.learned_sinks_size, self.num_attention_heads, self.torch_dtype, process_group.size(), rank_ordering)

    def init_gqa_properties(self):
        cte_rank_ordering = None
        if self.cp_degree == 1 and self.dp_degree > 1:
            validate_tp_prefill_to_dp_decode(self.num_key_value_heads, self.neuron_config.tp_degree, self.dp_degree)
            cte_rank_ordering = get_context_parallel_reordered_tp_mapping(self.neuron_config.tp_degree, self.dp_degree, self.num_key_value_heads)
        elif self.cp_degree == 8 and self.tp_degree == 8:
            cte_rank_ordering = get_cp8_tp8_rank_ordering(self.neuron_config.tp_degree, self.cp_degree, switch_cc=self.neuron_config.switch_cc)

        qkv_hidden_size = self.hidden_size * 2 if self.is_eagle3_draft else self.hidden_size
        qkv_proj = GroupQueryAttention_QKV(
            hidden_size=qkv_hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.qkv_bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.qkv_proj_sp_enabled,
            sequence_dimension=1 if self.qkv_proj_sp_enabled else None,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rms_norm_eps=self.rms_norm_eps,
            qkv_kernel_enabled=self.neuron_config.qkv_kernel_enabled,
            qkv_nki_kernel_enabled=self.neuron_config.qkv_nki_kernel_enabled,
            fused_rmsnorm_skip_gamma=self.neuron_config.fused_rmsnorm_skip_gamma,
            logical_nc_config=self.neuron_config.logical_nc_config,
            qkv_kernel_nbsd_layout=self.neuron_config.qkv_kernel_nbsd_layout,
            on_cpu=self.neuron_config.on_cpu,
            tiling_factor=self.neuron_config.cc_pipeline_tiling_factor if self.neuron_config.tile_cc else 1,
            seq_len_threshold_for_cc_tiling=self.neuron_config.seq_len_threshold_for_cc_tiling,
            rank_ordering=cte_rank_ordering,
        )
        o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.o_bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.o_proj_sp_enabled,
            sequence_dimension=1 if self.o_proj_sp_enabled else None,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rpl_reduce_dtype=self.rpl_reduce_dtype,
            # Mega kernels include QKV projection, RoPE, attention, and output projection
            # out_proj kernel will be enabled if mega kernel is enabled
            out_proj_kernel_enabled=self.attn_block_tkg_nki_kernel_enabled or self.attn_block_cte_nki_kernel_enabled or self.neuron_config.out_proj_kernel_enabled,
            logical_nc_config=self.neuron_config.logical_nc_config,
            tiling_factor=self.neuron_config.cc_pipeline_tiling_factor if self.neuron_config.tile_cc else 1,
            rank_ordering=cte_rank_ordering,
        )
        # Sink for CTE
        # Sink is initialized along side GQA, as sink sharding is tied to how GQA heads are sharded.
        self.learned_sinks = None
        if self.learned_sinks_size is not None:
            self.learned_sinks = LearnedSink(self.learned_sinks_size, self.num_attention_heads, self.torch_dtype, self.tp_degree, cte_rank_ordering)

        if self.dp_degree > 1 and nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
            init_data_parallel_attention_process_groups(self.config)

        # DP > 1 and CP == 1, CTE runs in full TP and Decode in TP + DP
        if self.dp_degree > 1 and self.cp_degree == 1:
            self.cte_qkv_proj = qkv_proj
            self.cte_o_proj = o_proj

            self.init_tkg_cp_qkv_o_proj(get_data_parallel_attention_tp_group())
        # CP > 1 and DP == 1, CTE runs in TP + CP and Decode in TP
        elif self.cp_degree > 1 and self.dp_degree == 1:
            self.cte_qkv_proj = qkv_proj
            self.cte_o_proj = o_proj

            rank_ordering = get_context_parallel_reordered_tp_mapping(self.neuron_config.tp_degree, self.neuron_config.cp_degree, self.num_key_value_heads, cte_rank_ordering=cte_rank_ordering)

            self.init_tkg_cp_qkv_o_proj(get_tensor_model_parallel_group(), rank_ordering=rank_ordering)
        # CP > 1 and DP > 1 and CP != DP, CTE runs in TP + CP and Decode in TP + DP
        elif self.cp_degree > 1 and self.dp_degree > 1 and self.cp_degree != self.dp_degree:
            self.cte_qkv_proj = qkv_proj
            self.cte_o_proj = o_proj

            rank_ordering = get_context_parallel_reordered_dp_mapping(self.neuron_config.tp_degree, self.cp_degree, self.dp_degree, self.num_key_value_heads, switch_cc=self.neuron_config.switch_cc, cte_rank_ordering=cte_rank_ordering)

            self.init_tkg_cp_qkv_o_proj(get_data_parallel_attention_tp_group(), rank_ordering=rank_ordering)
        # CP == DP OR no CP and DP enabled
        else:
            self.qkv_proj = qkv_proj
            self.o_proj = o_proj
            self.learned_sinks = self.learned_sinks

        self.num_heads = utils.divide(qkv_proj.get_num_attention_heads(), self.tp_degree)
        self._src_num_key_value_heads = self.num_key_value_heads
        self.num_key_value_heads = utils.divide(
            qkv_proj.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qkv_sharding_strategy = qkv_proj.sharding_strategy

        # q_layernorm and k_layernorm values passed to __init__ take precedence
        if (self.q_layernorm is None) != (self.k_layernorm is None):
            raise ValueError("q_layernorm and k_layernorm must both be set or both be None")
        if (self.q_layernorm is None) and (self.k_layernorm is None):
            self.q_layernorm = nn.LayerNorm(self.head_dim) if self.qk_layernorm else None
            self.k_layernorm = nn.LayerNorm(self.head_dim) if self.qk_layernorm else None

    def init_qk_norm(self):
        if self.use_qk_norm:
            if self.qk_norm is None:
                self.qk_norm = (
                    CustomRMSNorm()
                    if self.rms_norm_eps is None
                    else CustomRMSNorm(eps=self.rms_norm_eps)
                )

    def get_learned_sinks(self):
        if self.learned_sinks_size is None:
            return None
        if self.neuron_config.is_prefill_stage and self.cp_degree != self.dp_degree:
            return self.learned_sinks.sink
        elif not self.neuron_config.is_prefill_stage and self.cp_degree != self.dp_degree:
            return self.tkg_learned_sinks.sink
        else:
            return self.learned_sinks.sink

    def scaled_qk(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            QK = torch.where(attention_mask.to(torch.bool), QK, torch.finfo(QK.dtype).min)
        return QK

    def get_qkv_proj(self):
        if self.neuron_config.is_prefill_stage and self.cp_degree != self.dp_degree:
            return self.cte_qkv_proj
        elif not self.neuron_config.is_prefill_stage and self.cp_degree != self.dp_degree:
            return self.tkg_qkv_proj
        else:
            return self.qkv_proj

    def get_o_proj(self):
        if self.neuron_config.is_prefill_stage and self.cp_degree != self.dp_degree:
            return self.cte_o_proj
        elif not self.neuron_config.is_prefill_stage and self.cp_degree != self.dp_degree:
            return self.tkg_o_proj
        else:
            return self.o_proj

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        elif use_polar_compatible_rope:
            rotary_freqs = precompute_freqs_cis(self.head_dim,
                                                self.neuron_config.max_context_length * 2,
                                                self.rope_theta,
                                                self.use_scaled_rope,
                                                device=Q.device)
            rotary_freqs = rotary_freqs[position_ids]
            Q, K = apply_rotary_polar_compatible(Q.transpose(1, 2), K.transpose(1, 2), rotary_freqs)
            Q, K = Q.transpose(1, 2), K.transpose(1, 2)
        return Q, K, cos_cache, sin_cache

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        """take care of the shape, layout, group query, custom position encoding, etc.
           also return residual for MLP """
        is_qkv_cte_fuse_rope_nki_kernel_enabled = self.neuron_config.is_prefill_stage and self.qkv_cte_nki_kernel_fuse_rope
        assert not (is_qkv_cte_fuse_rope_nki_kernel_enabled and self.use_qk_norm and self.qk_norm_placement == QKNormPlacement.PRE_ROPE), "qkv cte nki kernel fuse rope is not compatible with pre rope qk norm"
        if is_qkv_cte_fuse_rope_nki_kernel_enabled:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(hidden_states, position_ids)
            Q, K, V, residual = self.get_qkv_proj()(
                hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual, cos_cache=cos_cache, sin_cache=sin_cache,
            )
        else:
            Q, K, V, residual = self.get_qkv_proj()(
                hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual
            )

            if self.use_qk_norm and self.qk_norm_placement == QKNormPlacement.PRE_ROPE:
                Q = self.qk_norm(Q)
                K = self.qk_norm(K)

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.qkv_proj_sp_enabled:
            q_len *= self.tensor_model_parallel_group.size()
        Q = move_heads_front(
            Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm
        )
        K = move_heads_front(
            K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=self.k_layernorm
        )
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        if not skip_rope and not is_qkv_cte_fuse_rope_nki_kernel_enabled:
            # Rotate Q and K
            Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(Q, K, V,
                                                                     position_ids,
                                                                     cos_cache,
                                                                     sin_cache,
                                                                     use_polar_compatible_rope)

        if self.use_qk_norm and self.qk_norm_placement == QKNormPlacement.POST_ROPE:
            Q = self.qk_norm(Q)
            K = self.qk_norm(K)

        # Gather KV to full S when CP is enabled, before this gather, each cp_rank will only have S/CP K, V
        # [2, B, H, S/CP, D] --> [2, B, H, S, D]
        if past_key_value is None and self.cp_degree > 1:
            stacked_kv = torch.stack([K, V], dim=0)

            # Gather along dim 3 (K and V's S dim)
            stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                stacked_kv,
                gather_dim=3,
                process_group=get_context_parallel_attention_cp_group(),
            )

            if self.get_flash_attention_strategy_cp(q_len * self.cp_degree) == FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL:
                stacked_kv = order_strided_tensor(stacked_kv, 3, self.cp_degree)

            K, V = torch.unbind(stacked_kv, dim=0)

        return Q, K, V, cos_cache, sin_cache, residual

    def context_parallel_flash_attention_kernel(self, Q, K_active, V_active, q_len, bsz, strategy):
        Q = Q.reshape(-1, q_len, self.head_dim)  # B * heads, S, d_head
        Q = Q / math.sqrt(self.head_dim)

        if strategy == FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL and _has_new_kernel and _has_native_gqa_tp_support:
            # We perform transpose inside kernel and do not replicate in this case
            K_active = K_active.reshape(
                -1, q_len * self.cp_degree, self.head_dim
            )  # B * heads, S, d_head
            tp_k_new_kernel = True
        else:
            K_active = repeat_kv(K_active, self.num_key_value_groups)
            V_active = repeat_kv(V_active, self.num_key_value_groups)
            K_active = K_active.reshape(
                -1, q_len * self.cp_degree, self.head_dim
            ).permute(0, 2, 1)  # B * heads, d_head, S
            tp_k_new_kernel = False

        V_active = V_active.reshape(
            -1, q_len * self.cp_degree, self.head_dim
        )  # B * heads, S, d_head

        attn_output = torch.zeros(
            bsz * self.num_heads, self.head_dim, q_len, dtype=Q.dtype, device=Q.device
        )

        grid = (nc(self.logical_nc_config),)

        # tp_degree when using CP is the reduced TP that Attention runs in
        cp_rank = get_cp_rank(self.rank_util.get_rank(), self.tp_degree, self.cp_degree, self.neuron_config.switch_cc)

        cp_fa_kernel_kwargs = {}
        if self.learned_sinks_size is not None:
            cp_fa_kernel_kwargs["sink"] = self.get_learned_sinks().unsqueeze(-1)
        if self.sliding_window:
            cp_fa_kernel_kwargs["sliding_window"] = self.sliding_window

        if strategy == FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL:
            # Context parallel is not enabled for trn1 version of NKI kernel
            if _has_new_kernel and get_platform_target() != "trn1":
                attn_output = _flash_fwd_call_nki[grid](
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    do_out_tp=True,
                    tp_q=True,
                    tp_k=tp_k_new_kernel,
                    use_dma_transpose=True,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                    global_cp_deg=self.cp_degree,
                    cp_offset=(cp_rank * q_len).reshape((1, 1)),
                    **cp_fa_kernel_kwargs,
                )
            else:
                _flash_fwd_call_bir[grid](
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    use_dma_transpose=True,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                    global_n_tiles=self.cp_degree,
                    tile_i=cp_rank * q_len,
                    **cp_fa_kernel_kwargs,
                )
        elif strategy == FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL:
            _flash_fwd_call_strided_context_parallel[grid](
                Q,
                K_active,
                V_active,
                1.0,
                attn_output,
                kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                use_dma_transpose=True,
                global_n_tiles=self.cp_degree,
                tile_i=cp_rank,
                strided_q_slicing=True,
                **cp_fa_kernel_kwargs,
            )

        else:
            raise ValueError(f"{strategy} is not supported with context parallel")

        attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))

        return attn_output

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        flash_attn_strategy = self.get_flash_attention_strategy(q_len, attention_mask is not None)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if flash_attn_strategy != FlashAttentionStrategy.NONE and self.cp_degree > 1:
            attn_output = self.context_parallel_flash_attention_kernel(Q, K, V, q_len, bsz, flash_attn_strategy)

        elif flash_attn_strategy != FlashAttentionStrategy.NONE:
            logger.debug(f"ATTN kernel: logical_nc_config={self.logical_nc_config}")
            # if we are using left padding, then the bzs needs be 1 (otherwise we get wrong result
            # because flash attention does not use attention_mask). In practice, we use right
            # padding so this is unlikely to cause issues
            assert self.padding_side == "right" or bsz == 1

            # original shape of q, k, v is BHSD, and expected output is also BHSD.
            logger.debug(f"Using flash_fwd for Q.shape={Q.shape}")
            # make sure to cast inputs to torch_dtype (this is needed because the downcast to bf16
            # might happen after the kernel hlo creation step). Also convert shapes as expected by the kernel.

            # original Q shape: batch, num_heads, seqlen, d_head
            if _has_new_kernel and _has_native_gqa_tp_support:
                # do Q and K transpose inside kernel, no need to replicate heads
                Q = (
                    Q.reshape((bsz * self.num_heads, q_len, self.head_dim))
                    .to(self.torch_dtype)
                )
                tp_q_new_kernel = True
                K_active = K
                V_active = V
                num_kv_heads = self.num_key_value_heads
                K_active = (
                    K_active.reshape((bsz * num_kv_heads, q_len, self.head_dim))
                    .to(self.torch_dtype)
                )
                tp_k_new_kernel = True
            else:
                Q = (
                    Q.permute(0, 1, 3, 2)  # after permute: batch, num_heads, d_head, seqlen
                    .reshape((bsz * self.num_heads, self.head_dim, q_len))
                    .to(self.torch_dtype)
                )
                tp_q_new_kernel = False
                K_active = repeat_kv(K, self.num_key_value_groups)
                V_active = repeat_kv(V, self.num_key_value_groups)
                num_kv_heads = self.num_heads
                K_active = (
                    K_active.permute(0, 1, 3, 2)
                    .reshape((bsz * num_kv_heads, self.head_dim, q_len))
                    .to(self.torch_dtype)
                )
                tp_k_new_kernel = False
            Q = Q / math.sqrt(self.head_dim)
            V_active = V_active.reshape((bsz * num_kv_heads, q_len, self.head_dim)).to(
                self.torch_dtype
            )
            # shape: (B*H)DS
            attn_output = torch.zeros(
                bsz * self.num_heads, self.head_dim, q_len, dtype=Q.dtype, device=Q.device
            )

            logger.debug("Input parameter shapes")
            logger.debug(f"Q input shape {Q.shape}")
            logger.debug(f"K input shape {K_active.shape}")
            logger.debug(f"V input shape {V_active.shape}")
            logger.debug(f"Attn output shape {attn_output.shape}")

            # Set use_dma_transpose to True to enable longer sequence lengths (otherwise descriptor blowup)
            use_dma_transpose = q_len <= self.neuron_config.seq_len_threshold_for_cc_tiling

            fa_kernel_kwargs = {}
            if self.learned_sinks_size is not None:
                fa_kernel_kwargs["sink"] = self.get_learned_sinks().unsqueeze(-1)
                assert get_platform_target() != "trn1", "sink argument is not supported by trn1"
            if self.sliding_window:
                fa_kernel_kwargs["sliding_window"] = self.sliding_window
                assert get_platform_target() != "trn1", "sliding_window argument is not supported by trn1"

            if flash_attn_strategy == FlashAttentionStrategy.SHARDED_KERNEL:
                grid = (nc(self.logical_nc_config),)

                if _has_new_kernel:
                    attn_output = _flash_fwd_call_nki[grid](
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        do_out_tp=True,
                        tp_q=tp_q_new_kernel,
                        tp_k=tp_k_new_kernel,
                        use_dma_transpose=use_dma_transpose,
                        kernel_name=(
                            "AttentionMMSoftmaxMMWithoutSwap"
                            if attention_mask is None
                            else "CausalAttentionMMSoftmaxMMWithoutSwap"
                        ),
                        **fa_kernel_kwargs,
                    )
                else:
                    _flash_fwd_call_bir[grid](
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        attn_output,
                        use_dma_transpose=use_dma_transpose,
                        kernel_name=(
                            "AttentionMMSoftmaxMMWithoutSwap"
                            if attention_mask is None
                            else "CausalAttentionMMSoftmaxMMWithoutSwap"
                        ),
                        **fa_kernel_kwargs,
                    )
            elif flash_attn_strategy == FlashAttentionStrategy.UNSHARDED_KERNEL:
                if _has_new_kernel:
                    attn_output = _flash_fwd_call_nki(
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        do_out_tp=True,
                        tp_q=tp_q_new_kernel,
                        tp_k=tp_k_new_kernel,
                        use_dma_transpose=use_dma_transpose,
                        kernel_name=(
                            "AttentionMMSoftmaxMMWithoutSwap"
                            if attention_mask is None
                            else "CausalAttentionMMSoftmaxMMWithoutSwap"
                        ),
                        **fa_kernel_kwargs,
                    )
                else:
                    _flash_fwd_call_bir(
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        attn_output,
                        use_dma_transpose=use_dma_transpose,
                        kernel_name=(
                            "AttentionMMSoftmaxMMWithoutSwap"
                            if attention_mask is None
                            else "CausalAttentionMMSoftmaxMMWithoutSwap"
                        ),
                        **fa_kernel_kwargs,
                    )
            else:
                raise ValueError(f"Invalid flash attention strategy: {flash_attn_strategy}")

            # shape: BHDS
            attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
            logger.debug(f"Attn output after reshape {attn_output.shape}")
        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
            # Replicate KV for non-kernel path
            K_active = repeat_kv(K, self.num_key_value_groups)
            V_active = repeat_kv(V, self.num_key_value_groups)
            active_scores = self.scaled_qk(Q, K_active, attention_mask)
            learned_sinks = self.get_learned_sinks()
            if learned_sinks is not None:
                # Validate the sink is of size one
                assert learned_sinks.ndim == 1 and learned_sinks.shape[0] == self.num_heads
                learned_sinks = learned_sinks.reshape(1, self.num_heads, 1, 1).expand(bsz, -1, q_len, -1)
                active_scores = torch.cat((active_scores, learned_sinks), dim=-1)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
                Q.dtype
            )
            if learned_sinks is not None:
                active_scores = active_scores[..., :-1]
            attn_output = torch.matmul(active_scores, V_active)
        return attn_output, flash_attn_strategy

    def perform_prefix_prefill(self, Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        prior_len = K_prior.shape[-2]

        flash_attn_strategy = self.get_flash_attention_strategy(q_len, has_attention_mask=False)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            logger.debug(f"ATTN kernel: logical_nc_config={self.logical_nc_config}")
            # if we are using left padding, then the bzs needs be 1 (otherwise we get wrong result
            # because flash attention does not use attention_mask). In practice, we use right
            # padding so this is unlikely to cause issues
            assert self.padding_side == "right" or bsz == 1

            # original shape of q, k, v is BHSD, and expected output is also BHSD.
            logger.debug(f"Using flash_fwd for Q.shape={Q.shape}")
            # make sure to cast inputs to torch_dtype (this is needed because the downcast to bf16
            # might happen after the kernel hlo creation step). Also convert shapes as expected by the kernel.

            # original Q shape: batch, num_heads, seqlen, d_head
            if _has_new_pc_kernel and _has_native_gqa_tp_support:
                # do Q transpose inside kernel, no need to replicate heads
                Q = (
                    Q.reshape(bsz * self.num_heads, q_len, self.head_dim)
                    .to(self.torch_dtype)
                )
                tp_q_new_kernel = True
                K_active = K
                V_active = V
                num_kv_heads = self.num_key_value_heads
            else:
                Q = (
                    Q.reshape(bsz * self.num_heads, q_len, self.head_dim)
                    .permute(0, 2, 1)
                    .to(self.torch_dtype)
                )
                tp_q_new_kernel = False
                K_active = repeat_kv(K, self.num_key_value_groups)
                V_active = repeat_kv(V, self.num_key_value_groups)
                K_prior = repeat_kv(K_prior, self.num_key_value_groups)
                V_prior = repeat_kv(V_prior, self.num_key_value_groups)
                num_kv_heads = self.num_heads

            Q = Q / math.sqrt(self.head_dim)
            K_active = K_active.reshape((bsz * num_kv_heads, q_len, self.head_dim)).to(
                self.torch_dtype
            )
            V_active = V_active.reshape((bsz * num_kv_heads, q_len, self.head_dim)).to(
                self.torch_dtype
            )
            K_prior = K_prior.reshape((bsz * num_kv_heads, prior_len, self.head_dim)).to(
                self.torch_dtype
            )
            V_prior = V_prior.reshape((bsz * num_kv_heads, prior_len, self.head_dim)).to(
                self.torch_dtype
            )
            # shape: (B*H)DS
            attn_output = torch.zeros(
                bsz * self.num_heads, self.head_dim, q_len, dtype=Q.dtype, device=Q.device
            )

            logger.debug("Input parameter shapes")
            logger.debug(f"Q input shape {Q.shape}")
            logger.debug(f"K input shape {K_active.shape}")
            logger.debug(f"V input shape {V_active.shape}")
            logger.debug(f"Attn output shape {attn_output.shape}")

            if flash_attn_strategy == FlashAttentionStrategy.SHARDED_KERNEL:
                grid = (nc(self.logical_nc_config),)

                # TODO for WCTE: this kernel requires `attention_mask` as a param, requiring us to
                #   fully materialize the mask in DRAM. This is highly inefficient for long contexts.
                #   Change it such that kernel doesn't require `attention_mask`.

                # Ensures backwards compatibility with V2 kernel
                if _has_new_pc_kernel:
                    prior_used_len = torch.sum(attention_mask, dim=1).to(torch.int32)
                    attn_output = _flash_fwd_pc_call_nki[grid](
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                        k_prior=K_prior,
                        v_prior=V_prior,
                        prior_used_len=prior_used_len,
                        do_out_tp=True,
                        tp_q=tp_q_new_kernel,
                        tp_k=True,
                    )
                else:
                    _flash_fwd_pc_call_bir[grid](
                        Q,
                        K_active,
                        V_active,
                        K_prior,
                        V_prior,
                        attention_mask,
                        1.0,
                        attn_output,
                        kernel_name="V2CausalAttentionMMSoftmaxMMWithoutSwap",
                    )
            elif flash_attn_strategy == FlashAttentionStrategy.UNSHARDED_KERNEL:
                # Ensures backwards compatibility with V2 kernel
                if _has_new_pc_kernel:
                    prior_used_len = torch.sum(attention_mask, dim=1).to(torch.int32)
                    attn_output = _flash_fwd_pc_call_nki(
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                        k_prior=K_prior,
                        v_prior=V_prior,
                        prior_used_len=prior_used_len,
                        do_out_tp=True,
                        tp_q=tp_q_new_kernel,
                        tp_k=True,
                    )
                else:
                    _flash_fwd_pc_call_bir(
                        Q,
                        K_active,
                        V_active,
                        K_prior,
                        V_prior,
                        attention_mask,
                        1.0,
                        attn_output,
                        kernel_name="V2CausalAttentionMMSoftmaxMMWithoutSwap",
                    )
            else:
                raise ValueError(f"Invalid flash attention strategy: {flash_attn_strategy}")

            # shape: BHDS
            attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
            logger.debug(f"Attn output after reshape {attn_output.shape}")
        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")

            # Replicate KV for non-kernel path
            K_active = repeat_kv(K, self.num_key_value_groups)
            V_active = repeat_kv(V, self.num_key_value_groups)
            K_prior = repeat_kv(K_prior, self.num_key_value_groups)
            V_prior = repeat_kv(V_prior, self.num_key_value_groups)

            # Attention computation: softmax((Q.K/√dkv) + mask).V
            # i. prior (cached) KV
            if not self.k_cache_transposed:
                K_prior = K_prior.transpose(2, 3)
            prior_scores = torch.matmul(Q, K_prior) / math.sqrt(self.head_dim)
            prior_scores = prior_scores.to(torch.float32)

            # ii. active (current/new) KV
            active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
            active_scores = active_scores.to(torch.float32)

            # iii. attention scores
            softmax_prior, softmax_active = manual_softmax(
                prior_scores, active_scores, True
            )
            softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
            attn_prior = torch.matmul(softmax_prior, V_prior)
            attn_active = torch.matmul(softmax_active, V_active)
            attn_output = attn_prior + attn_active

        return attn_output, flash_attn_strategy

    def perform_prefill_chunked_attn(self, Q, K, V, q_len, bsz, attention_mask, chunk_size) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        flash_attn_strategy = self.get_flash_attention_strategy(q_len, attention_mask is not None)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")
        n_chunks = math.ceil(q_len / chunk_size)
        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            logger.debug(f"ATTN kernel: logical_nc_config={self.logical_nc_config}")
            assert self.padding_side == "right", "To enable chunked attention, padding side has to be right"
            assert bsz == 1, "Chunked attention only works with bsz 1 for CTE."
            assert q_len % chunk_size == 0, "Chunked attention only works with cte q_len bucket divisible by chunk_size"
            logger.debug(f"Using flash_fwd for Q.shape={Q.shape}")
            Q, K_active, V_active = reshape_qkv_for_chunked_flash_attention_kernel(Q, K_active, V_active, chunk_size, self.torch_dtype)

            Q = Q / math.sqrt(self.head_dim)
            attn_output = torch.zeros(
                n_chunks * self.num_heads, self.head_dim, chunk_size, dtype=Q.dtype, device=Q.device
            )

            logger.debug("Input parameter shapes")
            logger.debug(f"Q input shape {Q.shape}")
            logger.debug(f"K input shape {K_active.shape}")
            logger.debug(f"V input shape {V_active.shape}")
            logger.debug(f"Attn output shape {attn_output.shape}")

            # Set use_dma_transpose to True to enable longer sequence lengths (otherwise descriptor blowup)
            use_dma_transpose = q_len <= self.neuron_config.seq_len_threshold_for_cc_tiling

            if flash_attn_strategy == FlashAttentionStrategy.SHARDED_KERNEL:
                grid = (nc(self.logical_nc_config),)

                # Ensures backwards compatibility with attention_isa_kernel
                if _has_new_kernel:
                    attn_output = _flash_fwd_call_nki[grid](
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        do_out_tp=True,
                        tp_q=False,
                        use_dma_transpose=use_dma_transpose,
                        kernel_name=(
                            "AttentionMMSoftmaxMMWithoutSwap"
                            if attention_mask is None
                            else "CausalAttentionMMSoftmaxMMWithoutSwap"
                        ),
                    )
                else:
                    _flash_fwd_call_bir[grid](
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        attn_output,
                        use_dma_transpose=use_dma_transpose,
                        kernel_name=(
                            "AttentionMMSoftmaxMMWithoutSwap"
                            if attention_mask is None
                            else "CausalAttentionMMSoftmaxMMWithoutSwap"
                        ),
                    )
            elif flash_attn_strategy == FlashAttentionStrategy.UNSHARDED_KERNEL:
                # Ensures backwards compatibility with attention_isa_kernel
                if _has_new_kernel:
                    attn_output = _flash_fwd_call_nki(
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        do_out_tp=True,
                        tp_q=False,
                        use_dma_transpose=use_dma_transpose,
                        kernel_name=(
                            "AttentionMMSoftmaxMMWithoutSwap"
                            if attention_mask is None
                            else "CausalAttentionMMSoftmaxMMWithoutSwap"
                        ),
                    )
                else:
                    _flash_fwd_call_bir(
                        Q,
                        K_active,
                        V_active,
                        1.0,
                        attn_output,
                        use_dma_transpose=use_dma_transpose,
                        kernel_name=(
                            "AttentionMMSoftmaxMMWithoutSwap"
                            if attention_mask is None
                            else "CausalAttentionMMSoftmaxMMWithoutSwap"
                        ),
                    )
            else:
                raise ValueError(f"Invalid flash attention strategy: {flash_attn_strategy}")

            attn_output = attn_output.reshape((n_chunks, self.num_heads, self.head_dim, chunk_size))
            logger.debug(f"Attn output after reshape {attn_output.shape}")

        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")

            if self.learned_sinks_size is not None:
                raise ValueError("Learned sinks not supported for native compiler chunked attention.")

            outputs = []
            for i in range(n_chunks):
                end_q_idx = min((i + 1) * chunk_size, q_len)
                local_attention_mask = attention_mask[:, :, chunk_size * i:end_q_idx, chunk_size * i:end_q_idx]
                current_chunk_q = Q[:, :, chunk_size * i:end_q_idx, :]
                current_chunk_k = K_active[:, :, chunk_size * i:end_q_idx, :]
                current_chunk_v = V_active[:, :, chunk_size * i:end_q_idx, :]

                active_scores = self.scaled_qk(
                    current_chunk_q,
                    current_chunk_k,
                    local_attention_mask
                )
                active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
                    Q.dtype
                )
                outputs.append(torch.matmul(
                    active_scores,
                    current_chunk_v
                ))
            attn_output = torch.cat(outputs, dim=2)
        return attn_output, flash_attn_strategy

    def perform_prefix_prefill_windowed_attn(self, Q, K, V, q_len, bsz, attention_mask, window_size, past_key_value, active_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)

        flash_attn_strategy = self.get_flash_attention_strategy(q_len, has_attention_mask=False)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            K = torch.concat((K_prior, K_active), dim=2)
            V = torch.concat((V_prior, V_active), dim=2)
            batch_size, n_head, seq_len, _ = Q.shape
            Q, K = Q.permute(0, 1, 3, 2), K.permute(0, 1, 3, 2)  # BHSD -> BHDS
            config = FlashConfig(windowed_context_encoding=True) if seq_len >= DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE else FlashConfig(windowed_context_encoding=True, seq_tile_size=MIN_SLIDING_WINDOW_SEQ_TILE_SIZE)
            attn_output = flash_fwd[batch_size, n_head](Q, K, V, window_size=(window_size - 1, -1), config=config)
        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")

            # Attention computation: softmax((Q.K/√dkv) + mask).V
            # i. prior (cached) KV
            if not self.k_cache_transposed:
                K_prior = K_prior.transpose(2, 3)
            prior_scores = torch.matmul(Q, K_prior) / math.sqrt(self.head_dim)
            prior_scores = torch.where(
                attention_mask[:, :, :, -2 * window_size : -window_size], prior_scores, torch.finfo(prior_scores.dtype).min
            )
            prior_scores = prior_scores.to(torch.float32)

            # ii. active (current/new) KV
            active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
            active_scores = active_scores.to(torch.float32)

            # iii. attention scores
            softmax_prior, softmax_active = manual_softmax(
                prior_scores, active_scores, True
            )
            softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
            attn_prior = torch.matmul(softmax_prior, V_prior)
            attn_active = torch.matmul(softmax_active, V_active)
            attn_output = attn_prior + attn_active

        return attn_output, flash_attn_strategy

    def get_flash_attention_strategy_cp(self, q_len):
        """
        Gets the flash attention strategy for context parallel use-cases.

        For LNC1, we currently do not support flash attention kernel with context parallel.

        For LNC2, flash attention is enabled when sequence_length // cp_degree > head_dim.
        """
        strategy = FlashAttentionStrategy.NONE

        # CP FA kernel determines S dim by inferring it as the largest dim, so q_len needs to be > d_head.
        if self.cp_degree > 1 and self.logical_nc_config >= 2 and q_len > self.head_dim:
            if not self.strided_context_parallel_kernel_enabled:
                strategy = FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL
            elif not self.sliding_window:
                strategy = FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL

        if self.strided_context_parallel_kernel_enabled and strategy != FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL:
            raise ValueError(
                "Strided context parallel kernel is enabled but cannot be used. "
                "Ensure cp_degree is > 1, LNC = 2, and (sequence_length // cp_degree) > head_dim, or disable the strided CP kernel. "
                f"Current values: cp_degree={self.cp_degree}, LNC={self.logical_nc_config}, "
                f"sequence_length // cp_degree={q_len} and head_dim={self.head_dim}."
            )

        return strategy

    def perform_prefill_windowed_attn(self, Q, K, V, q_len, bsz, attention_mask, window_size) -> Tensor:
        flash_attn_strategy = self.get_flash_attention_strategy(q_len, attention_mask is not None)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if (
            flash_attn_strategy != FlashAttentionStrategy.NONE
            and flash_attn_strategy != FlashAttentionStrategy.SLIDING_WINDOW_KERNEL
        ):
            # Use the BIR flash attention kernel.
            attn_output, _ = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
            assert attn_output.shape == (bsz, self.num_heads, self.head_dim, q_len)
            return attn_output, flash_attn_strategy

        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        if flash_attn_strategy == FlashAttentionStrategy.SLIDING_WINDOW_KERNEL:
            # Use the NKI flash attention kernel.
            batch_size, n_head, seq_len, _ = Q.shape
            Q, K_active = Q.permute(0, 1, 3, 2), K_active.permute(0, 1, 3, 2)  # BHSD -> BHDS
            config = FlashConfig() if seq_len >= DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE else FlashConfig(seq_tile_size=MIN_SLIDING_WINDOW_SEQ_TILE_SIZE)
            attn_output = flash_fwd[batch_size, n_head](Q, K_active, V_active, window_size=(window_size - 1, -1), config=config)
            return attn_output, flash_attn_strategy

        # Flat compiler implementation.
        logger.debug("Windowed ATTN: native compiler")
        active_scores = self.scaled_qk(Q, K_active, attention_mask)
        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            # Validate sink of size one
            assert learned_sinks.ndim == 1 and learned_sinks.shape[0] == self.num_heads
            learned_sinks = learned_sinks.reshape(1, self.num_heads, 1, 1).expand(bsz, -1, q_len, -1)
            active_scores = torch.cat((active_scores, learned_sinks), dim=-1)
        active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
        if learned_sinks is not None:
            active_scores = active_scores[..., :-1]
        attn_output = torch.matmul(active_scores, V_active)
        return attn_output, flash_attn_strategy

    def get_flash_attention_strategy(self, q_len, has_attention_mask) -> FlashAttentionStrategy:
        """
        Gets the flash attention strategy.

        For LNC1, use the unsharded kernel if context length is at least 4096 to get the best performance.
        The unsharded kernel requires a context length of at least 512.

        For LNC2, use the sharded kernel if context length is at least 1024 and is divisible by 512.
        Additionally, the sharded kernel supports context lengths under 1024 that are divisible by 256.
        Otherwise, use no kernel, because the unsharded kernel has worse performance than no kernel.

        These constraints may change later.

        TODO: Throw an exception instead of disabling flash attention if explicitly enabled but not eligible.
              This must consider bucketing to avoid throwing an exception for smaller buckets.
        """
        # There are three cases in the neuron_config.attn_kernel_enabled: True, False and None (default)
        # Here we disable the kernel only when it's set to False explicitly for the back-compatible reason
        if self.attn_kernel_enabled is False:
            return FlashAttentionStrategy.NONE

        if self.cp_degree > 1:  # Includes CP SWA case.
            return self.get_flash_attention_strategy_cp(q_len)

        # Use NKI CTE Attention kernel only when platform is not trn1
        if self.sliding_window and get_platform_target() == "trn1":
            if q_len >= MIN_SLIDING_WINDOW_SEQ_TILE_SIZE:
                return FlashAttentionStrategy.SLIDING_WINDOW_KERNEL
            return FlashAttentionStrategy.NONE

        if int(self.logical_nc_config) > 1:
            if has_attention_mask:
                if q_len >= 1024:
                    if q_len % 512 == 0:
                        return FlashAttentionStrategy.SHARDED_KERNEL
                else:
                    if q_len % 256 == 0:
                        return FlashAttentionStrategy.SHARDED_KERNEL

                warnings.warn(
                    "Flash attention disabled. For flash attn to be performant, LNC2 requires context_len >= 1024 "
                    "to be divisible by 512, or context_len < 1024 to be divisible by 256"
                )
                return FlashAttentionStrategy.NONE
            else:
                return FlashAttentionStrategy.SHARDED_KERNEL

        # If seq_len is at least 4096, enable flash attn automatically to improve performance.
        if q_len >= 4096:
            return FlashAttentionStrategy.UNSHARDED_KERNEL

        # At lower seq lens, enable only if explicitly enabled.
        if self.attn_kernel_enabled and q_len >= 512:
            return FlashAttentionStrategy.UNSHARDED_KERNEL

        return FlashAttentionStrategy.NONE

    def compute_for_flash_decoding(
        self, Q, K, V, past_key_value, attention_mask, active_mask
    ) -> Tensor:
        # TODO: refactor/decompose this to reduce duplication with compute_for_token_gen
        # active attention
        n_repeat = Q.shape[1]
        K_active = repeat_kv(K, n_repeat)
        V_active = repeat_kv(V, n_repeat)
        active_scores = (torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)).to(
            torch.float32
        )
        active_scores = torch.where(
            active_mask, active_scores, torch.finfo(active_scores.dtype).min
        )

        # prior attention
        K_prior = repeat_kv(past_key_value[0], n_repeat)
        V_prior = repeat_kv(past_key_value[1], n_repeat)
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / math.sqrt(self.head_dim)
        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)

        # attention scores
        softmax_prior, softmax_active = distributed_softmax(prior_scores, active_scores)
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active
        return attn_output

    def attention_tokengen_kernel_shared(
        self, Q, K, V, past_key_value, attention_mask, active_mask
    ):
        q_heads = self.num_heads
        kv_head = self.num_key_value_heads

        logger.debug(
            f"TKG Attn kernel: Q.shape = {Q.shape}, K.shape = {K.shape}, V.shape = {V.shape}"
        )

        # original Q shape: batch, num_heads, seqlen, d_head
        bsz, _, q_len, _ = Q.shape
        assert Q.shape == (bsz, q_heads, q_len, self.head_dim)
        assert K.shape == (bsz, kv_head, q_len, self.head_dim)
        assert V.shape == (bsz, kv_head, q_len, self.head_dim)

        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        s_prior = attention_mask.shape[3]
        s_prior_full = V_prior.shape[2]
        assert K_prior.shape[1] == kv_head
        assert V_prior.shape[1] == kv_head

        expected_k_cache_shape = (
            (bsz, kv_head, self.head_dim, s_prior_full)
            if self.k_cache_transposed
            else (bsz, kv_head, s_prior_full, self.head_dim)
        )
        assert (
            K_prior.shape == expected_k_cache_shape
        ), f"Expect K cache shape: {expected_k_cache_shape}, got {K_prior.shape}"

        logger.debug(f"TKG Attn kernel: K_cache_transposed = {self.k_cache_transposed}")

        if q_len == 1:
            active_mask = torch.ones((bsz, q_heads, q_len, q_len), dtype=Q.dtype, device=Q.device)
        else:
            assert active_mask.shape == (
                bsz,
                1,
                q_len,
                q_len,
            ), f"{active_mask.shape} != ({bsz}, 1, {q_len}, {q_len})"
            # duplicate the mask across q_heads
            active_mask = active_mask.expand(-1, q_heads, -1, -1)
        assert active_mask.shape == (
            bsz,
            q_heads,
            q_len,
            q_len,
        ), f"{active_mask.shape} != ({bsz}, {q_heads}, {q_len}, {q_len})"

        return (
            q_heads,
            kv_head,
            bsz,
            q_len,
            K_prior,
            V_prior,
            s_prior,
            s_prior_full,
            active_mask,
        )

    def attention_tokengen_kernel_nki(
        self,
        Q,
        K,
        V,
        past_key_value,
        attention_mask,
        active_mask,
    ) -> torch.Tensor:
        (
            q_heads,
            kv_head,
            bsz,
            q_len,
            K_prior,
            V_prior,
            s_prior,
            s_prior_full,
            active_mask,
        ) = self.attention_tokengen_kernel_shared(
            Q, K, V, past_key_value, attention_mask, active_mask
        )

        # Q shape: BNSd -> BdNS
        Q = Q.permute(0, 3, 1, 2)
        Q = Q / math.sqrt(self.head_dim)
        # K shape: BNSd -> BNdS
        K = K.permute(0, 1, 3, 2)
        # K shape: BNdS -> BdS (assume N == 1)
        K = K.reshape((bsz, self.head_dim, q_len))
        # V shape: BNSd -> BSd (assume N == 1)
        V = V.reshape((bsz, q_len, self.head_dim))
        # BNLd --> BLd (assume N == 1)
        # or w/transpose: BNdL --> BdL (assume N == 1)
        K_prior = torch.squeeze(K_prior, (1))
        V_prior = torch.squeeze(V_prior, (1))

        # duplicate the mask across q_heads
        attention_mask = attention_mask.expand(-1, q_heads, -1, -1)
        assert attention_mask.shape == (
            bsz,
            q_heads,
            q_len,
            s_prior,
        ), f"{attention_mask.shape} != ({bsz}, {q_heads}, {q_len}, {s_prior})"

        attn_output = torch.zeros(
            self.head_dim, bsz * q_heads * q_len, dtype=Q.dtype, device=Q.device
        )
        grid = (nc(self.logical_nc_config),)
        attn_output = attention_token_gen_kernel[grid](
            Q,
            K,
            V,
            K_prior,
            V_prior,
            attention_mask,
            active_mask,
            K_cache_transposed=self.k_cache_transposed,
        )

        # d(B*N*S) -> BNSd
        return attn_output.permute(1, 0).reshape((bsz, self.num_heads, q_len, self.head_dim))

    def attention_tokengen_kernel_builtin(
        self,
        Q,
        K,
        V,
        position_ids,
        past_key_value,
        attention_mask,
        active_mask,
        rotary_position_ids,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            q_heads,
            kv_head,
            bsz,
            q_len,
            K_prior,
            V_prior,
            s_prior,
            s_prior_full,
            active_mask,
        ) = self.attention_tokengen_kernel_shared(
            Q, K, V, past_key_value, attention_mask, active_mask
        )

        # active_mask expected shape is [q_len, bsz, q_heads, q_len]
        # also expects upper triangular matrix instead of lower
        active_mask = active_mask.permute(3, 0, 1, 2)

        # get the starting position of currently generating tokens for all batches.
        assert position_ids.shape == (bsz, q_len)
        pos_id = position_ids[:, 0].unsqueeze(-1)
        assert pos_id.shape == (bsz, 1), f"{pos_id.shape} != ({bsz}, 1)"

        attn_output = torch.zeros(
            bsz, q_heads, self.head_dim, q_len, dtype=Q.dtype, device=Q.device
        )
        k_output = torch.zeros(bsz, kv_head, self.head_dim, q_len, dtype=Q.dtype, device=Q.device)

        rope_pos_ids = rotary_position_ids.to(torch.float32)
        assert rope_pos_ids.shape == (bsz, q_len), f"rope_pos_ids.shape: {rope_pos_ids.shape}"
        assert rope_pos_ids.dtype == torch.float32

        assert self.inv_freqs.shape == (
            self.head_dim // 2,
            1,
        ), f"inv_freqs.shape: {self.inv_freqs.shape}"
        assert self.inv_freqs.dtype == torch.float32

        grid = (nc(self.logical_nc_config),)
        _attn_builtin_token_gen_call[grid](
            q=Q,
            k_active=K,
            v_active=V,
            k_prior=K_prior,
            v_prior=V_prior,
            pos_id=pos_id,
            active_mask=active_mask,
            inv_freqs=self.inv_freqs.to(Q.device),
            rope_pos_ids=rope_pos_ids,
            out=attn_output,
            k_out=k_output,
            kernel_name="AttentionTkgFwd",
            curr_sprior=s_prior,
            full_sprior=s_prior_full,
            tp_k_prior=not self.k_cache_transposed,
            use_pos_id=True,
            fuse_rope=True,
            strided_mm1=True,
            use_dma_tp=True,
        )

        # reshape: BNdS -> BNSd
        k_output = k_output.permute(0, 1, 3, 2)
        attn_output = attn_output.permute(0, 1, 3, 2)

        return attn_output, k_output

    def attention_block_tokengen_nki_kernel(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_kv_per_layer: bool = True,
        active_block_table: Optional[torch.Tensor] = None,
        use_polar_compatible_rope: bool = False,
    ):
        if self.sequence_parallel_enabled and self.tensor_model_parallel_group is not None:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )
        bsz, q_len, h = hidden_states.size()

        # Prepare cosine and sine coefficients.
        skip_rope = False

        # For llama3, always have rotary_emb to be not None
        # llama3 always go to this if branch
        if self.rotary_emb is not None:
            # compute cos_cache and sin_cache for the first time and then cache them
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(hidden_states, rotary_position_ids)

                # Take first half and reshape to [dim//2, batch_size, seq_len]
                cos_cache = cos_cache[..., : cos_cache.shape[-1] // 2].permute(2, 0, 1)
                sin_cache = sin_cache[..., : sin_cache.shape[-1] // 2].permute(2, 0, 1)
        # For llama4, use_polar_compatible_rope flags whether or not do RoPE computation
        # Note that llama3 and llama4 use different RoPE implementation
        # llama3 -> first-second-half implementation
        # llama4 -> even-odd implementation (aka polar_compatible)
        elif use_polar_compatible_rope:
            rotary_freqs = precompute_freqs_cis(self.head_dim,
                                                self.neuron_config.max_context_length * 2,
                                                self.rope_theta,
                                                self.use_scaled_rope,
                                                device=hidden_states.device)
            rotary_freqs = rotary_freqs[position_ids]  # (bz, seq_len, dim//2)
            cos_cache = rotary_freqs.cos().permute(2, 0, 1)
            sin_cache = rotary_freqs.sin().permute(2, 0, 1)
        # For llama4 NoPE layer, skip rope computation

        else:
            # pass in a pesudo cos and sin cache to satisfy the kernel assertions
            # they will not be used in the actual computation
            expected_rope_coeff_shape = (self.head_dim // 2, bsz, q_len)
            cos_cache = torch.zeros(expected_rope_coeff_shape).to(hidden_states)
            sin_cache = torch.zeros(expected_rope_coeff_shape).to(hidden_states)
            skip_rope = True

        # Check KV cache shapes.
        K_prior, V_prior = past_key_value[0:2]

        use_cascaded_attn = self.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention
        use_online_softmax = self.neuron_config.attn_block_tkg_nki_kernel_use_online_softmax

        q_heads = self.num_heads
        kv_heads = self.num_key_value_heads
        if not self.neuron_config.is_block_kv_layout:
            s_max_ctx = V_prior.shape[2]
            expected_k_cache_shape = (
                (bsz, kv_heads, self.head_dim, s_max_ctx)
                if self.k_cache_transposed
                else (bsz, kv_heads, s_max_ctx, self.head_dim)
            )
            assert (
                K_prior.shape == expected_k_cache_shape
            ), f"Expect K cache shape: {expected_k_cache_shape}, got {K_prior.shape}"
        else:
            total_blocks = K_prior.shape[0]  # Might be self.neuron_config.pa_num_blocks + 1
            expected_cache_shape = (total_blocks, self.neuron_config.pa_block_size, kv_heads, self.head_dim)
            assert K_prior.shape == expected_cache_shape, f'{K_prior.shape} vs {expected_cache_shape}'
            assert V_prior.shape == expected_cache_shape
            assert kv_heads == 1

        # Prepare causal masks.
        s_prior = attention_mask.shape[-1]  # Current bucket's context length.
        expected_cache_mask_shape = [(bsz, 1, q_len, s_prior), (bsz, q_heads, q_len, s_prior)]
        assert (
            attention_mask.shape in expected_cache_mask_shape
        ), f"{attention_mask.shape} not matching any of expected shapes of {expected_cache_mask_shape}"
        # Duplicate the mask across q_heads, no op if mask already has q_heads in dim-1.
        attention_mask = attention_mask.expand(-1, q_heads, -1, -1)

        the_dtype = hidden_states.dtype
        the_device = hidden_states.device
        expected_active_mask_shape = (bsz, 1, q_len, q_len)
        if q_len == 1:
            active_mask = torch.ones(expected_active_mask_shape, dtype=the_dtype, device=the_device)
        else:
            assert (
                active_mask.shape == expected_active_mask_shape
            ), f"{active_mask.shape} != {expected_active_mask_shape}"
        # Duplicate the mask across q_heads
        active_mask = active_mask.expand(-1, q_heads, -1, -1)

        if use_cascaded_attn:
            # Put active_mask to the end of attention_mask and transpose for cascaded-reduce layout.
            attention_mask[:, :, :, -q_len:] = active_mask
            attention_mask = attention_mask.permute(3, 0, 1, 2)

        attn_output = torch.zeros(
            self.head_dim, bsz, q_heads * q_len, dtype=the_dtype, device=the_device
        )

        W_qkv = self.get_qkv_proj().Wqkv.weight
        W_qkv_bias = self.get_qkv_proj().Wqkv.bias.unsqueeze(0) if self.qkv_bias else None
        fused_rmsnorm = rmsnorm is not None
        W_gamma = (
            rmsnorm.weight.unsqueeze(0) if fused_rmsnorm else torch.ones((1, h), device=the_device)
        )
        update_cache_in_kernel = update_kv_per_layer and self.attn_block_tkg_nki_kernel_cache_update

        if update_cache_in_kernel:
            K = K_prior
            V = V_prior
        else:
            K = torch.zeros(self.head_dim, bsz, q_len, dtype=the_dtype, device=the_device)
            V = torch.zeros(bsz, q_len, self.head_dim, dtype=the_dtype, device=the_device)

        W_out = self.get_o_proj().o_proj.weight
        h_out = h // 2 if self.is_eagle3_draft else h  # eagle3 draft w_o hidden size is half of qkv hidden size
        assert W_out.shape == (q_heads * self.head_dim, h_out), f"W_out.shape = {W_out.shape}"

        W_out_bias = self.get_o_proj().o_proj.bias.unsqueeze(0) if self.o_bias else None
        if W_out_bias is not None:
            assert W_out_bias.shape == (1, h), f"W_out_bias.shape = {W_out_bias.shape}"
            W_out_bias = W_out_bias / self.tp_degree

        grid = (nc(self.logical_nc_config),)

        attn_blk_kernel = llama3_nki_attention_block_token_gen_kernel
        if use_cascaded_attn:  # Skip compiler middle end transformation when using cascade-reduce attention.
            from neuronxcc.nki.compiler import skip_middle_end_transformations
            attn_blk_kernel = skip_middle_end_transformations(attn_blk_kernel)

        # TODO: deperecate this and pass the below args as None once the arguments are available.
        tkg_kernel_kwargs = {}
        if self.o_bias:
            tkg_kernel_kwargs["bias_out"] = W_out_bias
        if self.qkv_bias:
            tkg_kernel_kwargs["bias_qkv"] = W_qkv_bias
        if use_cascaded_attn:
            tkg_kernel_kwargs["use_cascaded_attn"] = True
        if self.learned_sinks_size is not None:
            tkg_kernel_kwargs["sink"] = self.get_learned_sinks().unsqueeze(-1)
        if not use_online_softmax:
            tkg_kernel_kwargs["use_online_softmax"] = False

        # applies to padded checkpoints
        if hasattr(self.config, "original_hidden_size"):
            tkg_kernel_kwargs["H_actual"] = self.config.original_hidden_size

        attn_output, K, V = attn_blk_kernel[grid](
            X=hidden_states,
            W_qkv=W_qkv,
            W_gamma=W_gamma,
            rmsnorm_eps=self.rms_norm_eps,
            cos=cos_cache,
            sin=sin_cache,
            W_out=W_out,
            K_cache=K_prior,
            V_cache=V_prior,
            mask_cache=attention_mask,
            mask_active=active_mask,
            position_ids=position_ids.to(torch.int32),
            update_cache=update_cache_in_kernel,
            active_blocks_table=active_block_table,
            K_cache_transposed=self.k_cache_transposed,
            fused_rmsnorm=fused_rmsnorm,
            pre_rope_rmsnorm=self.neuron_config.pre_rope_rmsnorm,
            skip_rope=skip_rope,
            rope_first_second_half_impl=not use_polar_compatible_rope,
            qk_norm=self.use_qk_norm,
            qk_norm_eps=self.rms_norm_eps,
            **tkg_kernel_kwargs,
        )

        # Did the output projection in kernel. We need to reduce across TP ranks here.
        attn_output = attn_output.reshape((bsz, q_len, h_out))
        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            attn_output = reduce_scatter_to_sequence_parallel_region(
                attn_output, 1, process_group=self.tensor_model_parallel_group
            )
        else:
            attn_output = reduce_from_tensor_model_parallel_region(
                attn_output, process_group=self.tensor_model_parallel_group
            )

        if not update_cache_in_kernel:
            # K in dBS, V in BSd, we want to output BNSd where N is 1.
            #   if k_cache_transposed, output k in BNdS
            K = K.permute(1, 0, 2) if self.k_cache_transposed else K.permute(1, 2, 0)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)

        return attn_output, (K, V), cos_cache, sin_cache

    def attention_block_cte_nki_kernel(
        self,
        hidden_states: torch.Tensor,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
    ):
        assert not self.sequence_parallel_enabled, "attention block cte nki kernel can only used for short seq where SP is disabled"
        assert (
            self.rotary_emb is not None
        ), "attn-block-cte-nki-kernel-enabled always implements RoPE so self.rotary_emb must be specified."

        bsz, q_len, h = hidden_states.size()

        # Prepare cosine and sine coefficients.
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(hidden_states, rotary_position_ids)
            assert cos_cache.shape == (
                bsz,
                q_len,
                self.head_dim,
            ), f"cos_cache.shape: {cos_cache.shape}"

            # Take first half and reshape to [dim//2, batch_size, seq_len]
            cos_cache = cos_cache[..., : cos_cache.shape[-1] // 2].permute(2, 0, 1)
            sin_cache = sin_cache[..., : sin_cache.shape[-1] // 2].permute(2, 0, 1)

        expected_rope_coeff_shape = (self.head_dim // 2, bsz, q_len)
        assert cos_cache.shape == expected_rope_coeff_shape, f"cos_cache.shape: {cos_cache.shape}"
        assert sin_cache.shape == expected_rope_coeff_shape, f"sin_cache.shape: {sin_cache.shape}"

        q_heads = self.num_heads

        dtype = hidden_states.dtype
        device = hidden_states.device

        # Check KV cache shapes.
        attn_output = torch.zeros(
            self.head_dim, bsz, q_heads * q_len, dtype=dtype, device=device
        )

        fused_rmsnorm = rmsnorm is not None
        assert fused_rmsnorm, "attn-block-cte-nki-kernel-enabled always fuse rmsnorm"

        W_gamma = (
            rmsnorm.weight.unsqueeze(0) if fused_rmsnorm else torch.ones((1, h), device=device)
        )

        # expect out projection weight are transposed in CPU
        W_out = self.o_proj.o_proj.weight
        assert W_out.shape == (q_heads * self.head_dim, h), f"W_out.shape == {W_out.shape} != ({q_heads} * {self.head_dim}, {h})"

        W_qkv = self.qkv_proj.Wqkv.weight

        grid = (nc(self.logical_nc_config),)
        if residual is None:
            residual = torch.zeros(
                hidden_states.shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        attn_output, K, V, residual = llama3_nki_attention_block_cte_kernel[grid](
            hidden_states,
            W_qkv,
            W_gamma,
            None,
            None,
            residual,  # residual from prev mlp
            sin_cache,
            cos_cache,
            W_out,
            d_head=self.head_dim,
            k_cache_transposed=self.k_cache_transposed,
        )

        attn_output = attn_output.reshape((bsz, q_len, h))

        # Did the output projection in kernel. We need to reduce across TP ranks here.
        attn_output = reduce_from_tensor_model_parallel_region(
            attn_output, process_group=get_tp_group(self.config)
        )

        # K in BSd, V in BSd, we want to output BNSd where N is 1.
        # if k_cache_transposed is true, then output k is in BNdS
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        return attn_output, (K, V), cos_cache, sin_cache, residual

    def compute_for_token_gen(
        self,
        Q,
        K,
        V,
        position_ids,
        past_key_value,
        attention_mask,
        active_mask,
        is_prefix_caching=False,
    ) -> Tensor:
        """
        Attention computation at token generation phase

        This implementation decomposes TKG attention into a prior part and an
        active part, to read the KV cache and compute matmul in parallel. More
        details are available in document lqdaAJbPvsfV.

        To correctly use this decomposed TKG attention, ensure that the
        attention_mask is a boolean mask of shape (batch_size, num_kv_heads,
        q_seq_len, kv_seq_len), and attention_mask[:, :, :, i] = True only
        when i < computed_context_len.
        """
        is_speculation = False if position_ids is None else position_ids.shape[-1] > 1
        if self.attention_chunk_size and is_speculation:
            raise NotImplementedError("Speculative decoding is not supported by chunked attention yet.")

        # Attention computation: softmax((Q.K/√dkv) + mask).V
        # i. prior (cached) KV

        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)
        if not self.k_cache_transposed:
            K_prior = K_prior.transpose(2, 3)
        prior_scores = torch.matmul(Q, K_prior) / math.sqrt(self.head_dim)

        # pad the attention mask if the KV cache is padded
        if prior_scores.shape[-1] > attention_mask.shape[-1] and self.neuron_config.apply_seq_ids_mask:
            attention_mask = F.pad(attention_mask, (0, prior_scores.shape[-1] - attention_mask.shape[-1]), "constant", 0)

        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)
        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)
        if is_speculation or is_prefix_caching:
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
        active_scores = active_scores.to(torch.float32)
        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            assert learned_sinks.ndim == 1 and learned_sinks.shape[0] == self.num_heads
            bsz, _, seqlen, _ = active_scores.shape
            sinks = learned_sinks.reshape(1, self.num_heads, 1, 1).expand(bsz, -1, seqlen, -1)

            # For token generation, concatenated learned sinks with prior scores instead of active scores
            # for compatibility with manual_softmax.
            prior_scores = torch.cat((prior_scores, sinks), dim=-1)

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(
            prior_scores, active_scores, is_speculation or is_prefix_caching
        )

        if learned_sinks is not None:
            softmax_prior = softmax_prior[..., :-1]

        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output

    def perform_contexted_prefill(self, Q, K, V, past_key_value, attention_mask, **kwargs):
        """
        Attention computation for chunked prefill

        For chunked prefill, all prompts are concatenated along the seq dim, so
        the batch size is always one.
        """
        batch_size, num_q_heads, q_len, head_dim = Q.size()
        dtype = Q.dtype

        K_cache = past_key_value[0]
        V_cache = past_key_value[1]
        num_kv_heads_per_rank = K_cache.size()[1]

        tile_q_indices = kwargs.get("tile_q_indices")
        tile_block_tables = kwargs.get("tile_block_tables")
        tile_masks = kwargs.get("tile_masks")

        active_mask = attention_mask[0, 0, :, :]

        # Q: BHSD -> (1, n_q_heads, d, seq_q)
        Q = Q.permute(0, 1, 3, 2)
        # K: BHSD -> (1, n_kv_heads, d, seq_k)
        K = K.permute(0, 1, 3, 2)
        # V: BHSD -> (1, n_kv_heads, seq_v, d)
        # K_cache: (num_blocks, n_kv_heads, block_size, d)
        # V_cache: (num_blocks, n_kv_heads, block_size, d)

        # attn_output is in BHSD layout
        attn_output = flash_paged_attention_with_schedule[batch_size, num_kv_heads_per_rank](
            Q,
            K,
            V,
            K_cache,
            V_cache,
            tile_q_indices,
            tile_block_tables,
            tile_masks,
            active_mask,
            softmax_scale=None,
            mixed_precision=True,
        )

        # Clear the ouput at the padding positions
        num_queries = kwargs.get("num_queries")
        output_mask = torch.arange(q_len, dtype=dtype, device=Q.device) < torch.sum(
            num_queries, dtype=dtype
        )
        output_mask = output_mask[None, None, :, None]
        attn_output *= output_mask
        return attn_output

    def attention_context_encode(self, Q, K, V, q_len, bsz, attention_mask, past_key_value=None, active_mask=None):
        if past_key_value is None:
            attn_output, flash_attn_strategy = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
        else:
            attn_output, flash_attn_strategy = self.perform_prefix_prefill(Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask)
        if self.flash_decoding_enabled:
            K, V = self._filter_kv_for_flash_decoding(K, V, q_len, Q)

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            # transpose BHDS -> BSHD
            # this layout avoids additional transposes between attention kernel and output projection
            attn_output = attn_output.permute(0, 3, 1, 2)
        elif self.attention_chunk_size:
            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2)
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, K, V

    def _filter_kv_for_flash_decoding(self, K, V, q_len, Q):
        assert not self.k_cache_transposed, 'Transposed K cache is not yet supported by flash decoding feature.'
        assert self.qkv_proj.sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE, (
            "Flash decoding lives in the context of GQA (grouped query attention) and traditional MHA "
            "multi-head attention) won't work!"
        )
        rank_id = self.rank_util.get_rank()
        rank_id_in_kv_group = torch.remainder(rank_id, self.num_cores_per_group).to(torch.int64)
        # shard KV by seq len and pick the values based on rank
        assert q_len == Q.shape[2], f"Q shape is {Q.shape}"
        # selecting positions (on S dim) that belongs to the current rank
        offset = torch.arange(
            0, q_len, self.num_cores_per_group, dtype=torch.int64, device=Q.device
        )
        selected_seq_pos = offset + rank_id_in_kv_group
        K = torch.index_select(input=K, dim=2, index=selected_seq_pos)
        V = torch.index_select(input=V, dim=2, index=selected_seq_pos)
        return K, V

    def attention_context_encode_chunked_attention(self, Q, K, V, q_len, bsz, attention_mask, chunk_size=None):
        attn_output, flash_attn_strategy = self.perform_prefill_chunked_attn(Q, K, V, q_len, bsz, attention_mask, chunk_size)
        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            n_chunks = q_len // chunk_size
            attn_output = attn_output.permute(0, 3, 1, 2)
            attn_output = attn_output.reshape(n_chunks * chunk_size, self.num_heads, self.head_dim).unsqueeze(0)
        else:
            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, K, V

    def attention_context_encode_windowed_attention(self, Q, K, V, q_len, bsz, attention_mask, window_size=None, past_key_value=None, active_mask=None):
        if past_key_value is None:
            attn_output, flash_attn_strategy = self.perform_prefill_windowed_attn(Q, K, V, q_len, bsz, attention_mask, window_size)
            if flash_attn_strategy not in [FlashAttentionStrategy.NONE, FlashAttentionStrategy.SLIDING_WINDOW_KERNEL]:
                attn_output = attn_output.permute(0, 3, 1, 2)  # transpose BHDS -> BSHD
            else:
                attn_output = attn_output.transpose(1, 2).contiguous()  # transpose BHSD -> BSHD
        else:
            attn_output, _ = self.perform_prefix_prefill_windowed_attn(Q, K, V, q_len, bsz, attention_mask, window_size, past_key_value, active_mask)
            attn_output = attn_output.transpose(1, 2).contiguous()  # transpose BHSD -> BSHD
        return attn_output, K, V

    def attention_tokengen(
        self,
        Q,
        K,
        V,
        attention_mask,
        position_ids,
        past_key_value,
        active_mask,
        **kwargs,
    ):

        if self.attn_tkg_nki_kernel_enabled:
            return self.attention_tokengen_kernel_nki(
                Q,
                K,
                V,
                past_key_value,
                attention_mask,
                active_mask,
            )

        if self.neuron_config.is_prefix_caching:
            return self.compute_for_token_gen(
                Q,
                K,
                V,
                position_ids,
                past_key_value,
                attention_mask,
                active_mask,
                is_prefix_caching=True,
            )

        if self.neuron_config.is_chunked_prefill:
            q_len = Q.shape[2]  # Q shape: BHSD
            # If a TKG model is enabled for chunked prefill, decoding-only
            # requests will be passed to the base TKG code
            # path self.compute_for_token_gen()
            if q_len > 1:
                # Can process both prefilling and decoding requests
                return self.perform_contexted_prefill(
                    Q, K, V, past_key_value, attention_mask, **kwargs
                )

        if self.flash_decoding_enabled:
            assert active_mask is not None, "Flash decoding requires active mask is not None!"
            # gather Q from all cores in its KV group
            groups = get_kv_shared_group(as_list=True)
            Q = xm.all_gather(Q, dim=1, groups=groups, pin_layout=False)

            attn_output = self.compute_for_flash_decoding(
                Q, K, V, past_key_value, attention_mask, active_mask
            )
            return xm.reduce_scatter(
                xm.REDUCE_SUM,
                attn_output,
                scale=1,
                scatter_dim=1,
                shard_count=len(groups[0]),
                groups=groups,
                pin_layout=False,
            )

        return self.compute_for_token_gen(
            Q,
            K,
            V,
            position_ids,
            past_key_value,
            attention_mask,
            active_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        # args for kv cache usage
        kv_mgr: Optional[KVCacheManager] = None,
        get_kv_per_layer: bool = False,
        update_kv_per_layer: bool = False,
        residual: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        if self.attention_chunk_size and self.cp_degree == 1:
            return self.chunked_attention_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                adapter_ids,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                kv_mgr,
                get_kv_per_layer,
                update_kv_per_layer,
                residual,
                **kwargs,
            )
        elif self.attention_chunk_size and self.cp_degree > 1:
            logger.debug("Running chunked attn")
            return self.chunked_attention_with_context_parallel_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                adapter_ids,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                kv_mgr,
                get_kv_per_layer,
                update_kv_per_layer,
                residual,
                **kwargs,
            )
        elif self.sliding_window:
            return self.windowed_attention_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                adapter_ids,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                kv_mgr,
                get_kv_per_layer,
                update_kv_per_layer,
                residual,
                windowed_context_encoding_window_idx,
                **kwargs,
            )
        else:
            logger.debug("Running global attn")
            return self.standard_causal_attention_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                adapter_ids,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                kv_mgr,
                get_kv_per_layer,
                update_kv_per_layer,
                residual,
                windowed_context_encoding_window_idx,
                **kwargs,
            )

    def standard_causal_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        # args for kv cache usage
        kv_mgr: Optional[KVCacheManager] = None,
        get_kv_per_layer: bool = False,
        update_kv_per_layer: bool = False,
        residual: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Implements each layer's forward pass for the attention block."""

        use_polar_compatible_rope = kwargs.get("use_polar_compatible_rope", False)

        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.torch_dtype)
        seq_ids = kwargs.get("seq_ids")
        is_context_parallel = past_key_value is None and self.cp_degree > 1
        is_data_parallel = past_key_value is not None and self.dp_degree > 1
        if is_context_parallel:
            # split all inputs into S/CP pieces based on the cp_rank, each specified dim is the 'S' dim

            attention_mask, hidden_states, position_ids, cos_cache, sin_cache = self._split_inputs_for_context_parallel(attention_mask, hidden_states, position_ids, cos_cache, sin_cache)

        if is_data_parallel:
            # split all inputs into B/DP pieces based on the dp_rank, each specified dim is the batch dim

            dp_rank = get_dp_rank(self.rank_util.get_rank(), self.tp_degree, self.dp_degree, self.neuron_config.switch_cc)

            hidden_states = split_along_dim(
                hidden_states, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            attention_mask = split_along_dim(
                attention_mask, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            position_ids = split_along_dim(
                position_ids, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )

        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        if rotary_position_ids is None:
            rotary_position_ids = position_ids

        if get_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.get_kv_by_layer_id(**kwargs)

        is_token_gen = past_key_value is not None

        if windowed_context_encoding_window_idx >= 0:
            is_token_gen = False

        if self.neuron_config.is_prefix_caching:
            # For prefix caching, we might still have past_key_value
            # corresponding to cached prefix during context encoding.
            # The smallest non zero prefix size supported is 128 which
            # is used to differentiate between token gen and smallest
            # prefix bucket during context encoding.
            is_token_gen = is_token_gen and q_len < 128

        if self.attn_block_tkg_nki_kernel_enabled and is_token_gen:
            if self.neuron_config.is_block_kv_layout:
                position_ids = kwargs['scatter_index']
            if self.attn_block_tkg_nki_kernel_cache_update and self.neuron_config.apply_seq_ids_mask:
                # In the KV cache manager, the S dimension of the KV cache was extended by 128
                # as a  create a "padding zone" for invalid seq id writes.
                # If we set position_ids to S + 128 - 1 for invalid seq_ids, we see an OOB error in the NKI kernel.
                # As a workaround, we set it to different values: S + [1..K] which lands in the "padding zone",
                # where K is the speculation length, or 1 during token gen.
                position_ids_invalid = (past_key_value[1].shape[2] - KV_CACHE_PAD_FOR_SEQ_IDS_MASKING) + \
                    torch.arange(position_ids.shape[-1], device=position_ids.device, dtype=position_ids.dtype).reshape(1, -1).broadcast_to(position_ids.shape)
                seq_ids_mask = torch.ge(seq_ids, torch.full_like(seq_ids, 0))
                seq_ids_mask = seq_ids_mask.reshape(-1, 1).broadcast_to(position_ids.shape)
                position_ids = torch.where(seq_ids_mask, position_ids, position_ids_invalid)

            if is_data_parallel:
                kv_cache = kv_mgr.get_kv_by_layer_id(idx=kwargs['idx'], kvcache_buffer=kwargs['kvcache_buffer'], seq_len=q_len, skip_slice=True)
            else:
                kv_cache = kv_mgr._fetch_cache(idx=kwargs['idx'], kvcache_buffer=kwargs['kvcache_buffer'])

            attn_output, KV, cos_cache, sin_cache = self.attention_block_tokengen_nki_kernel(
                hidden_states,
                attention_mask,
                position_ids,
                kv_cache,
                active_mask,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                update_kv_per_layer,
                kwargs['active_block_table'],
                use_polar_compatible_rope=use_polar_compatible_rope
            )
            if update_kv_per_layer and not self.attn_block_tkg_nki_kernel_cache_update:
                assert kv_mgr is not None
                KV = kv_mgr.update_kv_by_layer_id(
                    kv_per_layer=KV,
                    position_ids=position_ids,
                    **kwargs,
                )

            if is_data_parallel:
                attn_output = gather_from_tensor_model_parallel_region_with_dim(
                    attn_output, gather_dim=0, process_group=get_data_parallel_attention_dp_group()
                )

            return NeuronAttentionBaseOutput(attn_output, KV, cos_cache, sin_cache)

        if self.attn_block_cte_nki_kernel_enabled and not is_token_gen and not self.neuron_config.is_prefix_caching:
            attn_output, KV, cos_cache, sin_cache, residual = self.attention_block_cte_nki_kernel(
                hidden_states,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                residual
            )
            if update_kv_per_layer:
                assert kv_mgr is not None
                KV = kv_mgr.update_kv_by_layer_id(
                    kv_per_layer=KV,
                    position_ids=position_ids,
                    **kwargs,
                )
            return NeuronAttentionBaseOutput(attn_output, KV, cos_cache, sin_cache, residual)

        tkg_attn_kernel_fused_rope = is_token_gen and self.attn_tkg_builtin_kernel_enabled

        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            rotary_position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=tkg_attn_kernel_fused_rope,
            residual=residual,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )

        if is_token_gen:
            if tkg_attn_kernel_fused_rope:
                # also returns K cache
                attn_output, K = self.attention_tokengen_kernel_builtin(
                    Q,
                    K,
                    V,
                    position_ids,
                    past_key_value,
                    attention_mask,
                    active_mask,
                    rotary_position_ids,
                )
            else:
                attn_output = self.attention_tokengen(
                    Q, K, V, attention_mask, position_ids, past_key_value, active_mask, **kwargs
                )

            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output, K, V = self.attention_context_encode(Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask)

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

        if self.k_cache_transposed:
            # Output K in BNSd if not transposed, otherwise BNdS
            K = K.permute(0, 1, 3, 2)

        kv: Tuple[Tensor, Tensor] = (K, V)

        if update_kv_per_layer:
            assert kv_mgr is not None
            kv = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=kv,
                position_ids=position_ids,
                **kwargs,
            )

        if is_context_parallel and not self.sequence_parallel_enabled:
            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output, gather_dim=1, process_group=get_context_parallel_attention_cp_group()
            )

        if is_data_parallel:
            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output, gather_dim=0, process_group=get_data_parallel_attention_dp_group()
            )

        attn_output = attn_output.to(original_dtype)

        return NeuronAttentionBaseOutput(attn_output, kv, cos_cache, sin_cache, residual)

    def _should_split_hidden_states_for_context_parallel(self):
        if not self.sequence_parallel_enabled:
            return True
        if self.attention_chunk_size:
            if not self.optimize_interleave_attn or self.is_post_global_attn_layer:
                return True
            else:
                return False
        else:
            return False

    def _split_inputs_for_context_parallel(self, attention_mask, hidden_states, position_ids, cos, sin):
        cp_rank = get_cp_rank(self.rank_util.get_rank(), self.tp_degree, self.cp_degree, self.neuron_config.switch_cc)
        attention_mask = split_along_dim(
            attention_mask, dim=2, rank=cp_rank, num_partitions=self.cp_degree
        )
        position_ids = split_along_dim(
            position_ids, dim=1, rank=cp_rank, num_partitions=self.cp_degree
        )
        if self._should_split_hidden_states_for_context_parallel():
            hidden_states = split_along_dim(
                hidden_states, dim=1, rank=cp_rank, num_partitions=self.cp_degree
            )
        if self.attention_chunk_size:
            cos = split_along_dim(
                cos, dim=1, rank=cp_rank, num_partitions=self.cp_degree
            )
            sin = split_along_dim(
                sin, dim=1, rank=cp_rank, num_partitions=self.cp_degree
            )
        return attention_mask, hidden_states, position_ids, cos, sin

    def chunked_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        # args for kv cache usage
        kv_mgr: Optional[KVCacheManager] = None,
        get_kv_per_layer: bool = False,
        update_kv_per_layer: bool = False,
        residual: Optional[torch.Tensor] = None,
        use_polar_compatible_rope: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Implements each layer's forward pass for the attention block."""
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        if rotary_position_ids is None:
            rotary_position_ids = position_ids

        if get_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.get_kv_by_layer_id(**kwargs)

        is_token_gen = past_key_value is not None

        tkg_attn_kernel_fused_rope = is_token_gen and self.attn_tkg_builtin_kernel_enabled

        if self.attn_block_tkg_nki_kernel_enabled and is_token_gen:
            return self.attention_block_tokengen_nki_kernel_chunked_attn(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, kv_mgr=kv_mgr, active_mask=active_mask,
                cos_cache=cos_cache, sin_cache=sin_cache, rmsnorm=rmsnorm,
                rotary_position_ids=rotary_position_ids, update_kv_per_layer=update_kv_per_layer,
                **kwargs
            )
        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            rotary_position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=tkg_attn_kernel_fused_rope,
            residual=residual,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )

        if is_token_gen:
            attn_output = self.attention_tokengen(
                Q, K, V, attention_mask, position_ids, past_key_value, active_mask, **kwargs
            )

            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output, K, V = self.attention_context_encode_chunked_attention(Q, K, V, q_len, bsz, attention_mask, self.attention_chunk_size)
            K, V = get_last_kv_chunk(self.attention_chunk_size, position_ids, K, V)

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

        if self.k_cache_transposed:
            # Output K in BNSd if not transposed, otherwise BNdS
            K = K.permute(0, 1, 3, 2)

        kv: Tuple[Tensor, Tensor] = (K, V)

        if update_kv_per_layer:
            assert kv_mgr is not None
            kv = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=kv,
                position_ids=position_ids,
                **kwargs,
            )

        return NeuronAttentionBaseOutput(attn_output, kv, cos_cache, sin_cache, residual)

    def chunked_attention_with_context_parallel_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        # args for kv cache usage
        kv_mgr: Optional[KVCacheManager] = None,
        get_kv_per_layer: bool = False,
        update_kv_per_layer: bool = False,
        residual: Optional[torch.Tensor] = None,
        use_polar_compatible_rope: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Implements each layer's forward pass with the chunked attention mechanism.
        This forward is for cp > 1 and bs = 1 only. This allows us to have a separate flow well-suited for cp
        and also reshape hidden states from (B, S, H) to (n_chunks, chunk_size, H).
        We can then perform batched matmul for attention scores both in torch and FA kernel.
        """
        assert not self.qkv_proj_sp_enabled, (
            "In chunked attn with context parallel, sequence parallel for qkv proj has to be disabled."
        )
        original_dtype = hidden_states.dtype
        attn_input_hidden_states = None

        is_token_gen = past_key_value is not None

        if self.optimize_interleave_attn and not self.is_post_global_attn_layer and not self.is_pre_global_attn_layer and not is_token_gen:
            attn_input_hidden_states = hidden_states.clone()

        hidden_states = hidden_states.to(self.torch_dtype)

        # Gather hidden states and residuals from SP region
        if not is_token_gen and self.sequence_parallel_enabled:
            if self.is_post_global_attn_layer or not self.optimize_interleave_attn:
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states,
                    1,
                    process_group=get_tensor_model_parallel_group(),  # this is gather over world size. Shape after gather will be (B, S, H). We only do this for the first chunk layer.
                )
            else:
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states,
                    1,
                    process_group=self.tensor_model_parallel_group  # this is partial gather over tp group. Shape after gather will be (B, S/cp, H)
                )

        bsz, q_len, _ = hidden_states.size()

        # if the layer is not post global attn, the input that will be chunked has a (B, S/cp, H) shape
        chunk_slice_size = self.attention_chunk_size if (self.is_post_global_attn_layer or not self.optimize_interleave_attn) else self.attention_chunk_size // self.cp_degree

        n_chunks = math.ceil(q_len / chunk_slice_size)

        if get_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.get_kv_by_layer_id(**kwargs)
        if rotary_position_ids is None:
            rotary_position_ids = position_ids

        if not is_token_gen:
            if self.rotary_emb is not None and (cos_cache is None or sin_cache is None):
                cos_cache, sin_cache = self.rotary_emb(hidden_states, rotary_position_ids)  # calculate cos sin cache for CTE once before chunked attn flow
            attention_outputs = []
            input_hidden_states = []
            Ks = []
            Vs = []

            for chunk_idx in range(n_chunks):
                # split inputs into chunks
                chunked_attn_output, chunked_K, chunked_V, chunked_attn_input_hidden_states = self.compute_attention_per_chunk_with_context_parallel(
                    chunk_idx,
                    chunk_slice_size,
                    hidden_states,
                    rotary_position_ids,
                    attention_mask,
                    cos_cache,
                    sin_cache,
                    past_key_value,
                    adapter_ids,
                    rmsnorm,
                    residual,
                    active_mask,
                    bsz,
                    use_polar_compatible_rope
                )
                if self.optimize_interleave_attn and self.is_pre_global_attn_layer:
                    chunked_attn_output = self.get_o_proj()(chunked_attn_output, adapter_ids=adapter_ids)
                    chunked_attn_output = chunked_attn_output + chunked_attn_input_hidden_states
                    chunked_attn_output = gather_from_tensor_model_parallel_region_with_dim(
                        chunked_attn_output,
                        gather_dim=1,
                        process_group=get_context_parallel_attention_cp_group(),
                    )

                # append compute results to lists
                attention_outputs.append(chunked_attn_output)
                Ks.append(chunked_K)
                Vs.append(chunked_V)
                if self.optimize_interleave_attn and self.is_post_global_attn_layer:
                    input_hidden_states.append(chunked_attn_input_hidden_states)

            # concat outputs
            attn_output = torch.cat(attention_outputs, dim=1).contiguous()
            K = torch.cat(Ks, dim=2)
            V = torch.cat(Vs, dim=2)
            if len(input_hidden_states) > 0:
                attn_input_hidden_states = torch.cat(input_hidden_states, dim=1).contiguous()

            if not self.optimize_interleave_attn or not self.is_pre_global_attn_layer:
                attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

            # scatter again into SP region
            if (not self.optimize_interleave_attn or self.is_pre_global_attn_layer) and self.sequence_parallel_enabled:
                attn_output = split_along_dim(
                    attn_output,
                    dim=1,
                    rank=self.rank_util.get_rank(),
                    num_partitions=self.cp_degree * self.tp_degree,
                )
            if self.optimize_interleave_attn and self.is_post_global_attn_layer and self.sequence_parallel_enabled:
                attn_input_hidden_states = split_along_dim(
                    attn_input_hidden_states,
                    dim=1,
                    rank=get_kv_head_group_number(self.rank_util.get_rank(), self.tp_degree),
                    num_partitions=self.tp_degree,
                )
            # Get the latest chunk of KV to update KV cache
            K, V = get_last_kv_chunk(
                self.attention_chunk_size, position_ids, K, V
            )
        else:
            if self.attn_block_tkg_nki_kernel_enabled:
                return self.attention_block_tokengen_nki_kernel_chunked_attn(
                    hidden_states=hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, kv_mgr=kv_mgr, active_mask=active_mask,
                    cos_cache=cos_cache, sin_cache=sin_cache, rmsnorm=rmsnorm,
                    rotary_position_ids=rotary_position_ids, update_kv_per_layer=update_kv_per_layer,
                    use_polar_compatible_rope=use_polar_compatible_rope,
                    **kwargs
                )

            Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
                rotary_position_ids,
                hidden_states,
                past_key_value,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                rmsnorm=rmsnorm,
                skip_rope=False,
                residual=residual,
                use_polar_compatible_rope=use_polar_compatible_rope,
            )
            attn_output = self.attention_tokengen(
                Q,
                K,
                V,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                **kwargs,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            # merge multi head hidden
            attn_output = attn_output.reshape(
                bsz, q_len, self.num_heads * self.head_dim
            )
            # Z = Z.Wo
            attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)
            # transpose BHSD -> BSHD

        if self.k_cache_transposed:
            K = K.permute(0, 1, 3, 2)

        kv: Tuple[Tensor, Tensor] = (K, V)
        attn_output = attn_output.to(original_dtype)
        if update_kv_per_layer:
            assert kv_mgr is not None
            kv = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=kv,
                position_ids=position_ids,
                **kwargs,
            )
        return NeuronAttentionBaseOutput(
            attn_output, kv, cos_cache, sin_cache, residual, attn_input_hidden_states
        )

    def compute_attention_per_chunk_with_context_parallel(
        self,
        chunk_idx,
        chunk_slice_size,
        hidden_states,
        rotary_position_ids,
        attention_mask,
        cos_cache=None,
        sin_cache=None,
        past_key_value=None,
        adapter_ids=None,
        rmsnorm=None,
        residual=None,
        active_mask=None,
        bsz=1,
        use_polar_compatible_rope=False
    ):
        """
        Process a single chunk of the chunked attention mechanism with context parallelism.

        Returns:
            Tuple containing:
            - attn_output: Attention output for this chunk
            - chunked_cos: Cosine cache for this chunk
            - chunked_sin: Sine cache for this chunk
            - K: Key tensor
            - V: Value tensor
        """
        # split inputs into chunks
        # hidden states will be sliced with cp_chunk_start and cp_chunk_end because hidden states can have the (B, S/cp, H) shape
        # the other inputs will be sliced with normal chunk_start and chunk_end because they always have full S.
        cp_chunk_start = chunk_idx * chunk_slice_size
        cp_chunk_end = min((chunk_idx + 1) * chunk_slice_size, hidden_states.shape[1])
        chunk_start = chunk_idx * self.attention_chunk_size
        if self.is_post_global_attn_layer or not self.optimize_interleave_attn:
            chunk_end = min((chunk_idx + 1) * chunk_slice_size, hidden_states.shape[1])
        else:
            chunk_end = min((chunk_idx + 1) * self.attention_chunk_size, hidden_states.shape[1] * self.cp_degree)

        chunked_hidden_states = torch.ops.aten.slice(
            hidden_states, dim=1, start=cp_chunk_start, end=cp_chunk_end
        )
        chunked_position_ids = torch.ops.aten.slice(
            rotary_position_ids, dim=1, start=chunk_start, end=chunk_end
        )
        chunked_cos = (
            torch.ops.aten.slice(cos_cache, dim=1, start=chunk_start, end=chunk_end)
            if cos_cache is not None
            else None
        )
        chunked_sin = (
            torch.ops.aten.slice(sin_cache, dim=1, start=chunk_start, end=chunk_end)
            if sin_cache is not None
            else None
        )
        chunked_attention_mask = torch.ops.aten.slice(
            attention_mask, dim=2, start=chunk_start, end=chunk_end
        )
        chunked_attention_mask = torch.ops.aten.slice(
            chunked_attention_mask, dim=3, start=chunk_start, end=chunk_end
        )

        # slice inputs into cp_degree slices
        (
            chunked_attention_mask,
            chunked_hidden_states,
            chunked_position_ids,
            chunked_cos,
            chunked_sin,
        ) = self._split_inputs_for_context_parallel(
            chunked_attention_mask,
            chunked_hidden_states,
            chunked_position_ids,
            chunked_cos,
            chunked_sin,
        )
        chunked_attn_input_hidden_states = None
        if self.optimize_interleave_attn and (self.is_post_global_attn_layer or self.is_pre_global_attn_layer):
            chunked_attn_input_hidden_states = chunked_hidden_states.clone()

        Q, K, V, chunked_cos, chunked_sin, residual = self.prep_qkv_tensors(
            chunked_position_ids,
            chunked_hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=chunked_cos,
            sin_cache=chunked_sin,
            rmsnorm=rmsnorm,
            skip_rope=False,
            residual=residual,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )
        cp_chunk_len = Q.shape[2]
        attn_output, K, V = self.attention_context_encode(
            Q,
            K,
            V,
            cp_chunk_len,
            bsz,
            chunked_attention_mask,
            past_key_value,
            active_mask,
        )

        # merge multi head hidden
        attn_output = attn_output.reshape(
            bsz,
            cp_chunk_len,
            self.num_heads * self.head_dim,
        )
        # Gather from cp region (C/cp -> C) for attn_output and cos and sin cache
        if not self.optimize_interleave_attn:
            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output,
                gather_dim=1,
                process_group=get_context_parallel_attention_cp_group(),
            )

        return attn_output, K, V, chunked_attn_input_hidden_states

    def attention_block_tokengen_nki_kernel_chunked_attn(
            self, hidden_states,
            attention_mask, position_ids,
            kv_mgr, active_mask, cos_cache,
            sin_cache, rmsnorm,
            rotary_position_ids, update_kv_per_layer,
            use_polar_compatible_rope=False,
            **kwargs
    ):

        seq_ids = kwargs.get("seq_ids")
        if self.attn_block_tkg_nki_kernel_cache_update:
            if self.config.neuron_config.apply_seq_ids_mask:
                position_ids = apply_seq_id_mask(
                    position_ids, seq_ids,
                    self.attention_chunk_size + KV_CACHE_PAD_FOR_SEQ_IDS_MASKING - 1)
            else:
                position_ids = position_ids % self.attention_chunk_size

        attn_output, KV, cos_cache, sin_cache = self.attention_block_tokengen_nki_kernel(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_mgr._fetch_cache(idx=kwargs['idx'], kvcache_buffer=kwargs['kvcache_buffer']),
            active_mask=active_mask,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            rotary_position_ids=rotary_position_ids,
            update_kv_per_layer=update_kv_per_layer,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )

        if not self.attn_block_tkg_nki_kernel_cache_update and update_kv_per_layer:
            assert kv_mgr is not None
            KV = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=KV,
                position_ids=position_ids,
                **kwargs,
            )

        return NeuronAttentionBaseOutput(attn_output, KV, cos_cache, sin_cache)

    def attention_block_tokengen_nki_kernel_sliding_window_attn(
            self, hidden_states,
            attention_mask, position_ids,
            kv_mgr, active_mask, cos_cache,
            sin_cache, rmsnorm,
            rotary_position_ids, update_kv_per_layer,
            is_data_parallel,
            **kwargs
    ):

        if self.attn_block_tkg_nki_kernel_cache_update:
            if self.neuron_config.enable_fused_speculation:
                kv_cache_size = self.sliding_window - 2 + self.neuron_config.speculation_length
                position_ids = position_ids % kv_cache_size
            else:
                position_ids = position_ids % (self.sliding_window - 1)

        if is_data_parallel:
            kv_cache = kv_mgr.get_kv_by_layer_id(
                idx=kwargs['idx'],
                kvcache_buffer=kwargs['kvcache_buffer'],
                seq_len=self.sliding_window,
                skip_slice=True,
            )
        else:
            kv_cache = kv_mgr._fetch_cache(
                idx=kwargs['idx'],
                kvcache_buffer=kwargs['kvcache_buffer'],
            )

        attn_output, KV, cos_cache, sin_cache = self.attention_block_tokengen_nki_kernel(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache,
            active_mask=active_mask,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            rotary_position_ids=rotary_position_ids,
            update_kv_per_layer=update_kv_per_layer,
        )

        if not self.attn_block_tkg_nki_kernel_cache_update and update_kv_per_layer:
            assert kv_mgr is not None
            KV = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=KV,
                position_ids=position_ids,
                **kwargs,
            )

        if is_data_parallel:
            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output, gather_dim=0, process_group=get_data_parallel_attention_dp_group()
            )

        return NeuronAttentionBaseOutput(attn_output, KV, cos_cache, sin_cache)

    def windowed_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        # args for kv cache usage
        kv_mgr: Optional[KVCacheManager] = None,
        get_kv_per_layer: bool = False,
        update_kv_per_layer: bool = False,
        residual: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        is_context_parallel = past_key_value is None and self.cp_degree > 1
        is_data_parallel = past_key_value is not None and self.dp_degree > 1

        full_position_ids = position_ids.clone()

        if is_context_parallel:
            # split all inputs into S/CP pieces based on the cp_rank
            attention_mask, hidden_states, position_ids, cos_cache, sin_cache = self._split_inputs_for_context_parallel(attention_mask, hidden_states, position_ids, cos_cache, sin_cache)

        if is_data_parallel:
            # split all inputs into B/DP pieces based on the dp_rank, each specified dim is the batch dim

            dp_rank = get_dp_rank(self.rank_util.get_rank(), self.tp_degree, self.dp_degree, self.neuron_config.switch_cc)

            hidden_states = split_along_dim(
                hidden_states, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            attention_mask = split_along_dim(
                attention_mask, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            position_ids = split_along_dim(
                position_ids, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )

        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        if rotary_position_ids is None:
            rotary_position_ids = position_ids

        if get_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.get_kv_by_layer_id(**kwargs)

        is_token_gen = past_key_value is not None

        if windowed_context_encoding_window_idx >= 0:
            is_token_gen = False

        if self.attn_block_tkg_nki_kernel_enabled and is_token_gen:
            return self.attention_block_tokengen_nki_kernel_sliding_window_attn(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, kv_mgr=kv_mgr, active_mask=active_mask,
                cos_cache=cos_cache, sin_cache=sin_cache, rmsnorm=rmsnorm,
                rotary_position_ids=rotary_position_ids, update_kv_per_layer=update_kv_per_layer,
                is_data_parallel=is_data_parallel,
                **kwargs
            )

        tkg_attn_kernel_fused_rope = is_token_gen and self.attn_tkg_builtin_kernel_enabled

        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            rotary_position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=tkg_attn_kernel_fused_rope,
            residual=residual,
        )

        if is_token_gen:
            attn_output = self.attention_tokengen(
                Q, K, V, attention_mask, position_ids, past_key_value, active_mask, **kwargs
            )

            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output, K, V = self.attention_context_encode_windowed_attention(Q, K, V, q_len, bsz, attention_mask, self.sliding_window, past_key_value, active_mask)
            K, V = get_last_kv_window(self.sliding_window, full_position_ids, K, V, windowed_context_encoding_window_idx, self.neuron_config.speculation_length)

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

        if self.k_cache_transposed:
            # Output K in BNSd if not transposed, otherwise BNdS
            K = K.permute(0, 1, 3, 2)

        kv: Tuple[Tensor, Tensor] = (K, V)

        if update_kv_per_layer:
            assert kv_mgr is not None
            kv = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=kv,
                position_ids=position_ids,
                **kwargs,
            )

        if is_context_parallel and not self.sequence_parallel_enabled:
            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output, gather_dim=1, process_group=get_context_parallel_attention_cp_group()
            )

        return NeuronAttentionBaseOutput(attn_output, kv, cos_cache, sin_cache, residual)
