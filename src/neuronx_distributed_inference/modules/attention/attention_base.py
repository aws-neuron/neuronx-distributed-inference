import logging
import math
import warnings
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager

from neuronx_distributed_inference.modules.attention.attention_process_groups import (
    get_context_parallel_attention_cp_group,
    get_context_parallel_attention_tp_group,
    init_context_parallel_attention_process_groups,
)
from neuronx_distributed_inference.modules.chunked_prefill.flash_pa_with_schedule import (
    flash_paged_attention_with_schedule,
)
from neuronx_distributed_inference.models.config import InferenceConfig

from .utils import (
    apply_rotary_pos_emb,
    distributed_softmax,
    manual_softmax,
    move_heads_front,
    repeat_kv,
    get_context_parallel_reordered_tp_mapping,
)
import torch.nn.functional as F

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
from neuronx_distributed.parallel_layers.parallel_state import get_kv_shared_group, get_tensor_model_parallel_group, get_tensor_model_parallel_size
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
)
from neuronx_distributed_inference.utils.distributed import get_tp_group, split_along_dim, get_cp_rank

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

from .gqa import GQA, GroupQueryAttention_O, GroupQueryAttention_QKV  # noqa: E402

logger = logging.getLogger("Neuron")

_flash_fwd_call = nki_jit()(attention_isa_kernel)
_flash_fwd_pc_call = nki_jit()(prefix_caching_attention_fwd_isa_kernel)

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


class FlashAttentionStrategy(Enum):
    NONE = 0
    UNSHARDED_KERNEL = 1
    SHARDED_KERNEL = 2
    CONTEXT_PARALLEL_KERNEL = 3


@dataclass(frozen=True)
class NeuronAttentionBaseOutput:
    hidden_states: torch.tensor
    present_key_value: torch.tensor
    cos_cache: Optional[torch.tensor] = None
    sin_cache: Optional[torch.tensor] = None
    residual: Optional[torch.tensor] = None

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
                 rms_norm_eps: float = None,
                 use_qk_norm: bool = False,
                 clip_qkv: float = None,
                 qkv_bias: bool = False,
                 o_bias: bool = False,
                 num_cores_per_group: int = 1,
                 sequence_parallel_enabled: bool = None,
                 attention_chunk_size: int = None,
                 tensor_model_parallel_group: Optional[ProcessGroup] = None):

        super().__init__()

        self.config = config
        self.neuron_config = config.neuron_config

        self.tensor_model_parallel_group = None
        self.rank_util = None
        self.global_rank = None

        if tensor_model_parallel_group is not None:
            self.tensor_model_parallel_group = tensor_model_parallel_group
        elif nxd.parallel_layers.parallel_state.model_parallel_is_initialized() and self.config.neuron_config.cp_degree > 1 and self.neuron_config.is_prefill_stage:
            init_context_parallel_attention_process_groups(config)
            self.tensor_model_parallel_group = get_context_parallel_attention_tp_group()
        elif nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
            self.tensor_model_parallel_group = nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()

        if self.tensor_model_parallel_group is not None:
            self.rank_util = SPMDRank(world_size=self.tensor_model_parallel_group.size())
            self.global_rank = SPMDRank(world_size=config.neuron_config.world_size)

        self.tp_degree = self.neuron_config.tp_degree if self.tensor_model_parallel_group is None else self.tensor_model_parallel_group.size()
        self.cp_degree = self.neuron_config.cp_degree

        self.rpl_reduce_dtype = self.neuron_config.rpl_reduce_dtype
        self.torch_dtype = config.neuron_config.attention_dtype if config.neuron_config.attention_dtype is not None else config.neuron_config.torch_dtype
        self.fused_qkv = self.neuron_config.fused_qkv

        # Accounts for cases where some sub-modules always have SP enabled / disabled
        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled if sequence_parallel_enabled is None else sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.padding_side = self.neuron_config.padding_side
        self.flash_decoding_enabled = self.neuron_config.flash_decoding_enabled
        self.mlp_kernel_enabled = self.neuron_config.mlp_kernel_enabled
        self.attn_kernel_enabled = self.neuron_config.attn_kernel_enabled
        self.attn_tkg_builtin_kernel_enabled = self.neuron_config.attn_tkg_builtin_kernel_enabled
        self.attn_tkg_nki_kernel_enabled = self.neuron_config.attn_tkg_nki_kernel_enabled
        self.attn_block_tkg_nki_kernel_enabled = self.neuron_config.attn_block_tkg_nki_kernel_enabled
        self.attn_block_tkg_nki_kernel_cache_update = self.neuron_config.attn_block_tkg_nki_kernel_cache_update
        self.k_cache_transposed = self.neuron_config.k_cache_transposed
        self.logical_nc_config = self.neuron_config.logical_nc_config
        self.qk_layernorm = self.neuron_config.qk_layernorm

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.rotary_emb = rotary_emb

        self.inv_freqs = None
        if self.attn_tkg_builtin_kernel_enabled:
            self.inv_freqs = rotary_emb.get_inv_freqs().unsqueeze(1)

        self.num_cores_per_group = num_cores_per_group
        self.qkv_bias = qkv_bias
        self.o_bias = o_bias
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm
        self.clip_qkv = clip_qkv
        self.o_proj_layer_name = "o_proj"
        self.attention_chunk_size = attention_chunk_size

        self.init_gqa_properties()

        self.qk_norm = None
        if use_qk_norm:
            self.init_qk_norm()

    def init_tkg_cp_qkv_o_proj(self):
        rank_ordering = get_context_parallel_reordered_tp_mapping(self.neuron_config.tp_degree, self.neuron_config.cp_degree)

        self.tkg_qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=get_tensor_model_parallel_size(),
            dtype=self.torch_dtype,
            bias=self.qkv_bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=get_tensor_model_parallel_group(),
            rms_norm_eps=self.rms_norm_eps,
            qkv_kernel_enabled=self.neuron_config.qkv_kernel_enabled,
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
            tp_degree=get_tensor_model_parallel_size(),
            dtype=self.torch_dtype,
            bias=self.o_bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=get_tensor_model_parallel_group(),
            rpl_reduce_dtype=self.rpl_reduce_dtype,
            out_proj_kernel_enabled=self.attn_block_tkg_nki_kernel_enabled,
            logical_nc_config=self.neuron_config.logical_nc_config,
            rank_ordering=rank_ordering,
        )

    def init_gqa_properties(self):
        qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.qkv_bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rms_norm_eps=self.rms_norm_eps,
            qkv_kernel_enabled=self.neuron_config.qkv_kernel_enabled,
            fused_rmsnorm_skip_gamma=self.neuron_config.fused_rmsnorm_skip_gamma,
            logical_nc_config=self.neuron_config.logical_nc_config,
            qkv_kernel_nbsd_layout=self.neuron_config.qkv_kernel_nbsd_layout,
            on_cpu=self.neuron_config.on_cpu,
            tiling_factor=self.neuron_config.cc_pipeline_tiling_factor if self.neuron_config.tile_cc else 1,
            seq_len_threshold_for_cc_tiling=self.neuron_config.seq_len_threshold_for_cc_tiling,
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
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rpl_reduce_dtype=self.rpl_reduce_dtype,
            out_proj_kernel_enabled=self.attn_block_tkg_nki_kernel_enabled,
            logical_nc_config=self.neuron_config.logical_nc_config,
            tiling_factor=self.neuron_config.cc_pipeline_tiling_factor if self.neuron_config.tile_cc else 1,
        )

        if self.cp_degree > 1:
            self.cte_qkv_proj = qkv_proj
            self.cte_o_proj = o_proj
            self.init_tkg_cp_qkv_o_proj()
        else:
            self.qkv_proj = qkv_proj
            self.o_proj = o_proj

        self.num_heads = utils.divide(qkv_proj.get_num_attention_heads(), self.tp_degree)
        self._src_num_key_value_heads = self.num_key_value_heads
        self.num_key_value_heads = utils.divide(
            qkv_proj.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qkv_sharding_strategy = qkv_proj.sharding_strategy

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

    def scaled_qk(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            QK = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
        return QK

    def get_qkv_proj(self):
        if self.neuron_config.is_prefill_stage and self.cp_degree > 1:
            return self.cte_qkv_proj
        elif not self.neuron_config.is_prefill_stage and self.cp_degree > 1:
            return self.tkg_qkv_proj
        else:
            return self.qkv_proj

    def get_o_proj(self):
        if self.neuron_config.is_prefill_stage and self.cp_degree > 1:
            return self.cte_o_proj
        elif not self.neuron_config.is_prefill_stage and self.cp_degree > 1:
            return self.tkg_o_proj
        else:
            return self.o_proj

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
    ):
        """take care of the shape, layout, group query, custom position encoding, etc.
           also return residual for MLP """
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual
        )

        if self.use_qk_norm:
            Q = self.qk_norm(Q)
            K = self.qk_norm(K)

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q = move_heads_front(
            Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm
        )
        K = move_heads_front(
            K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=self.k_layernorm
        )
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Rotate Q and K
        if not skip_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

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

            K, V = torch.unbind(stacked_kv, dim=0)

        return Q, K, V, cos_cache, sin_cache, residual

    def context_parallel_flash_attention_kernel(self, Q, K_active, V_active, q_len, bsz):
        Q = Q.reshape(-1, q_len, self.head_dim)  # B * heads, S, d_head
        Q = Q / math.sqrt(self.head_dim)

        K_active = K_active.reshape(-1, q_len * self.cp_degree, self.head_dim).permute(
            0, 2, 1
        )  # B * heads, d_head, S
        V_active = V_active.reshape(
            -1, q_len * self.cp_degree, self.head_dim
        )  # B * heads, S, d_head

        attn_output = torch.zeros(
            bsz * self.num_heads, self.head_dim, q_len, dtype=Q.dtype, device=Q.device
        )

        grid = (nc(self.logical_nc_config),)

        # tp_degree when using CP is the reduced TP that Attention runs in
        cp_rank = get_cp_rank(self.global_rank.get_rank(), self.tp_degree)

        _flash_fwd_call[grid](
            Q,
            K_active,
            V_active,
            1.0,
            attn_output,
            kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
            use_dma_transpose=True,
            global_n_tiles=self.cp_degree,
            tile_i=cp_rank,
        )

        attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))

        return attn_output

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        flash_attn_strategy = self.get_flash_attention_strategy(q_len, attention_mask is not None)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if flash_attn_strategy == FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL:
            attn_output = self.context_parallel_flash_attention_kernel(Q, K_active, V_active, q_len, bsz)

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
            Q = (
                Q.permute(0, 1, 3, 2)  # after permute: batch, num_heads, d_head, seqlen
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.torch_dtype)
            )
            Q = Q / math.sqrt(self.head_dim)
            K_active = (
                K_active.permute(0, 1, 3, 2)
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.torch_dtype)
            )
            V_active = V_active.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(
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

            if flash_attn_strategy == FlashAttentionStrategy.SHARDED_KERNEL:
                grid = (nc(self.logical_nc_config),)

                _flash_fwd_call[grid](
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
                _flash_fwd_call(
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

            # shape: BHDS
            attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
            logger.debug(f"Attn output after reshape {attn_output.shape}")
        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
            active_scores = self.scaled_qk(Q, K_active, attention_mask)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
                Q.dtype
            )
            attn_output = torch.matmul(active_scores, V_active)
        return attn_output, flash_attn_strategy

    def perform_prefix_prefill(self, Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        prior_len = K_prior.shape[-2]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)

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
            Q = (
                Q.reshape(bsz * self.num_heads, q_len, self.head_dim)
                .permute(0, 2, 1)
                .to(self.torch_dtype)
            )
            Q = Q / math.sqrt(self.head_dim)
            K_active = K_active.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(
                self.torch_dtype
            )
            V_active = V_active.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(
                self.torch_dtype
            )
            K_prior = K_prior.reshape((bsz * self.num_heads, prior_len, self.head_dim)).to(
                self.torch_dtype
            )
            V_prior = V_prior.reshape((bsz * self.num_heads, prior_len, self.head_dim)).to(
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

                _flash_fwd_pc_call[grid](
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
                _flash_fwd_pc_call(
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
            raise NotImplementedError(f"Chunked attention not implemented for {flash_attn_strategy} yet")
        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")

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
            pad_len = q_len - attn_output.shape[2]
            F.pad(attn_output, (0, 0, 0, 0, 0, pad_len), 'constant', 0)
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
        if self.cp_degree > 1 and self.logical_nc_config < 2:
            return FlashAttentionStrategy.NONE
        if int(self.logical_nc_config) > 1:
            if has_attention_mask:
                # CP FA kernel determines S dim by inferring it as the largest dim
                if self.cp_degree > 1 and q_len >= self.head_dim:
                    return FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL

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
    ):
        if self.sequence_parallel_enabled and self.tensor_model_parallel_group is not None:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )
        bsz, q_len, h = hidden_states.size()

        # Prepare cosine and sine coefficients.
        assert (
            self.rotary_emb is not None
        ), "attn-block-tkg-nki-kernel-enabled always implements RoPE so self.rotary_emb must be specified."
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

        # Check KV cache shapes.
        K_prior, V_prior = past_key_value[0:2]

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

        attn_output = torch.zeros(
            self.head_dim, bsz, q_heads * q_len, dtype=the_dtype, device=the_device
        )

        W_qkv = self.qkv_proj.Wqkv.weight
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

        W_out = self.o_proj.o_proj.weight
        assert W_out.shape == (q_heads * self.head_dim, h), f"W_out.shape = {W_out.shape}"
        grid = (nc(self.logical_nc_config),)
        attn_output, K, V = llama3_nki_attention_block_token_gen_kernel[grid](
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
        )

        # Did the output projection in kernel. We need to reduce across TP ranks here.
        attn_output = attn_output.reshape((bsz, q_len, h))
        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            attn_output = reduce_scatter_to_sequence_parallel_region(
                attn_output, 1, process_group=get_tp_group(self.config)
            )
        else:
            attn_output = reduce_from_tensor_model_parallel_region(
                attn_output, process_group=get_tp_group(self.config)
            )

        if not update_cache_in_kernel:
            # K in dBS, V in BSd, we want to output BNSd where N is 1.
            #   if k_cache_transposed, output k in BNdS
            K = K.permute(1, 0, 2) if self.k_cache_transposed else K.permute(1, 2, 0)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)

        return attn_output, (K, V), cos_cache, sin_cache

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

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(
            prior_scores, active_scores, is_speculation or is_prefix_caching
        )
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
        else:
            # transpose BHSD -> BSHD
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
            raise NotImplementedError(f"Chunked attention not implemented for {flash_attn_strategy} yet")
        else:
            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()
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
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        if self.attention_chunk_size:
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
        else:
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
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Implements each layer's forward pass for the attention block."""
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.torch_dtype)
        seq_ids = kwargs.get("seq_ids")
        is_context_parallel = past_key_value is None and self.cp_degree > 1

        if is_context_parallel:
            # split all inputs into S/CP pieces based on the cp_rank, each specified dim is the 'S' dim

            cp_rank = get_cp_rank(self.global_rank.get_rank(), self.tp_degree)

            if not self.sequence_parallel_enabled:
                hidden_states = split_along_dim(
                    hidden_states, dim=1, rank=cp_rank, num_partitions=self.cp_degree
                )

            attention_mask = split_along_dim(
                attention_mask, dim=2, rank=cp_rank, num_partitions=self.cp_degree
            )
            position_ids = split_along_dim(
                position_ids, dim=1, rank=cp_rank, num_partitions=self.cp_degree
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
                position_ids_invalid = (past_key_value[1].shape[2] - 128) + torch.arange(position_ids.shape[-1], device=position_ids.device, dtype=position_ids.dtype).reshape(1, -1).broadcast_to(position_ids.shape)
                seq_ids_mask = torch.ge(seq_ids, torch.full_like(seq_ids, 0))
                seq_ids_mask = seq_ids_mask.reshape(-1, 1).broadcast_to(position_ids.shape)
                position_ids = torch.where(seq_ids_mask, position_ids, position_ids_invalid)
            attn_output, KV, cos_cache, sin_cache = self.attention_block_tokengen_nki_kernel(
                hidden_states,
                attention_mask,
                position_ids,
                kv_mgr._fetch_cache(idx=kwargs['idx'], kvcache_buffer=kwargs['kvcache_buffer']),
                active_mask,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                update_kv_per_layer,
                kwargs['active_block_table'],
            )
            if update_kv_per_layer and not self.attn_block_tkg_nki_kernel_cache_update:
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

        attn_output = attn_output.to(original_dtype)

        return NeuronAttentionBaseOutput(attn_output, kv, cos_cache, sin_cache, residual)

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
            attn_output, K, V = self.attention_context_encode_chunked_attention(Q, K, V, q_len, bsz, attention_mask, self.attention_chunk_size)

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output, adapter_ids=adapter_ids)

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
