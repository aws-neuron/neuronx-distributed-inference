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

from .utils import (
    apply_rotary_pos_emb,
    distributed_softmax,
    manual_softmax,
    move_heads_front,
    repeat_kv,
)

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402

import neuronx_distributed as nxd
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import utils  # noqa: E402
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.parallel_state import get_kv_shared_group
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from neuronx_distributed_inference.utils.distributed import get_tp_group

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

from neuronx_distributed_inference.modules.kvcache.utils import contexted_kv, contexted_kv_v2

from .gqa import GQA, GroupQueryAttention_O, GroupQueryAttention_QKV  # noqa: E402

logger = logging.getLogger("Neuron")

_flash_fwd_call = nki_jit()(attention_isa_kernel)

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

    def __init__(self, tensor_model_parallel_group: Optional[ProcessGroup] = None):
        super().__init__()

        if tensor_model_parallel_group is not None:
            self.tensor_model_parallel_group = tensor_model_parallel_group
            self.rank_util = SPMDRank(world_size=self.tensor_model_parallel_group.size())
        elif nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
            self.tensor_model_parallel_group = (
                nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()
            )
            self.rank_util = SPMDRank(world_size=self.tensor_model_parallel_group.size())
        else:
            # CPU flow doesn need rank_util and TP group now
            self.tensor_model_parallel_group = None
            self.rank_util = None

        self.is_causal = True
        self.num_key_value_groups = None
        self.num_key_value_heads = None
        self.num_heads = None
        self.rotary_emb = None
        self.o_proj = None
        self.qkv_proj = None
        self.bias = False
        self.k_layernorm = None
        self.q_layernorm = None
        self.qk_layernorm = False
        self.rms_norm_eps = None
        self.use_qk_norm = False
        self.qk_norm = None

        self.num_cores_per_group = 1
        self.flash_decoding_enabled = False
        self.sequence_parallel_enabled = False
        self.sequence_dimension = None
        self.rpl_reduce_dtype = None

        self.o_proj_layer_name = "o_proj"

        self.attn_kernel_enabled = False
        self.attn_tkg_builtin_kernel_enabled = False
        self.attn_tkg_nki_kernel_enabled = False
        self.attn_block_tkg_nki_kernel_enabled = False
        self.attn_block_tkg_nki_kernel_cache_update = False

        self.k_cache_transposed = False
        self.logical_nc_config = None

    def init_gqa_properties(self):
        self.attn_kernel_enabled = self.neuron_config.attn_kernel_enabled
        self.attn_tkg_nki_kernel_enabled = self.neuron_config.attn_tkg_nki_kernel_enabled
        self.attn_tkg_builtin_kernel_enabled = self.neuron_config.attn_tkg_builtin_kernel_enabled
        self.attn_block_tkg_nki_kernel_enabled = (
            self.neuron_config.attn_block_tkg_nki_kernel_enabled
        )
        self.attn_block_tkg_nki_kernel_cache_update = (
            self.neuron_config.attn_block_tkg_nki_kernel_cache_update
        )
        self.k_cache_transposed = self.neuron_config.k_cache_transposed
        self.logical_nc_config = self.neuron_config.logical_nc_config

        self.qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.bias,
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
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rpl_reduce_dtype=self.rpl_reduce_dtype,
            out_proj_kernel_enabled=self.attn_block_tkg_nki_kernel_enabled,
            logical_nc_config=self.neuron_config.logical_nc_config,
        )
        self.num_heads = utils.divide(self.qkv_proj.get_num_attention_heads(), self.tp_degree)
        self.num_key_value_heads = utils.divide(
            self.qkv_proj.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.head_dim)
            self.k_layernorm = nn.LayerNorm(self.head_dim)

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
        Q, K, V, residual = self.qkv_proj(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual
        )
        if self.use_qk_norm:
            self.init_qk_norm()  # TODO: when attentionbase can take config parameters in init, move this to init function
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

        return Q, K, V, cos_cache, sin_cache, residual

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        flash_attn_strategy = self.get_flash_attention_strategy(q_len)
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

            if flash_attn_strategy == FlashAttentionStrategy.SHARDED_KERNEL:
                grid = (nc(self.logical_nc_config),)

                _flash_fwd_call[grid](
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
            elif flash_attn_strategy == FlashAttentionStrategy.UNSHARDED_KERNEL:
                _flash_fwd_call(
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
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

    def get_flash_attention_strategy(self, q_len) -> FlashAttentionStrategy:
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
        if int(self.logical_nc_config) > 1:
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
        s_max_ctx = V_prior.shape[2]

        expected_k_cache_shape = (
            (bsz, kv_heads, self.head_dim, s_max_ctx)
            if self.k_cache_transposed
            else (bsz, kv_heads, s_max_ctx, self.head_dim)
        )
        assert (
            K_prior.shape == expected_k_cache_shape
        ), f"Expect K cache shape: {expected_k_cache_shape}, got {K_prior.shape}"

        # Prepare causal masks.
        s_prior = attention_mask.shape[-1]  # Current bucket's context length.
        assert attention_mask.shape == (
            bsz,
            1,
            q_len,
            s_prior,
        ), f"{attention_mask.shape} != ({bsz}, 1, {q_len}, {s_prior})"
        # duplicate the mask across q_heads
        attention_mask = attention_mask.expand(-1, q_heads, -1, -1)
        attention_mask = attention_mask.reshape((bsz, q_heads, q_len, s_prior))

        the_dtype = hidden_states.dtype
        the_device = hidden_states.device

        if q_len == 1:
            active_mask = torch.ones(
                (bsz, q_heads, q_len, q_len), dtype=the_dtype, device=the_device
            )
        else:
            assert active_mask.shape == (
                bsz,
                1,
                q_len,
                q_len,
            ), f"{active_mask.shape} != ({bsz}, 1, {q_len}, {q_len})"
            # duplicate the mask across q_heads
            active_mask = active_mask.expand(-1, q_heads, -1, -1)
        active_mask = active_mask.reshape((bsz, q_heads, q_len, q_len))

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
        is_block_kv=False,
    ) -> Tensor:
        """attention computation at token generation phase"""
        is_speculation = False if position_ids is None else position_ids.shape[-1] > 1

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
        if is_speculation or is_block_kv:
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
        active_scores = active_scores.to(torch.float32)

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(
            prior_scores, active_scores, is_speculation or is_block_kv
        )
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output

    def compute_for_block_kv(self, Q, K, V, past_key_value, attention_mask, **kwargs):
        """
        Attention computation for block kv layout for prefix caching

        The first step is to combine the KV from current requset with the KV cache
        using the cache_mask and current_reordered_idx.
        The second step is to do the attention calculation.

        Args:
            Q: hidden state of current queries in shape of (batch_size, num_heads, n_active_tokens, head_dim)
            K: hidden state of current keys in shape of (batch_size, num_heads, n_active_tokens, head_dim)
            V: hidden state of current values in shape of (batch_size, num_heads, n_active_tokens, head_dim)
            past_key_value: tuple of KV caches with each in shape of (batch_size, num_heads, max_seq_len, head_dim)
            cache_mask: the precomputed cache mask in shape of (batch_size, max_seq_len)
            current_reordered_idx: the scatter indices mapping the newly computed KV
                to full-length KV in shape of (batch_size, max_seq_len)

        Returns:
            attn_output: output hidden state in shape of (batch_size, num_heads, n_active_tokens, head_dim)
        """
        assert not self.k_cache_transposed, 'Transposed K cache is not yet supported by block KV feature.'
        K_active = K
        V_active = V

        K_prior = past_key_value[0]
        V_prior = past_key_value[1]

        args = (
            kwargs.get("cache_mask"),
            kwargs.get("current_reordered_idx"),
        )
        K_combined = contexted_kv_v2(K_prior, K_active, *args)
        V_combined = contexted_kv_v2(V_prior, V_active, *args)

        K_combined = repeat_kv(K_combined, self.num_key_value_groups)
        V_combined = repeat_kv(V_combined, self.num_key_value_groups)

        active_scores = self.scaled_qk(Q, K_combined, attention_mask)
        active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
            Q.dtype
        )
        attn_output = torch.matmul(active_scores, V_combined)
        return attn_output

    def perform_contexted_prefill(self, Q, K, V, past_key_value, attention_mask, **kwargs):
        """
        Attention computation for chunked prefill

        The Q here contains only logits from current request, but the K and V
        will contain logits from previous and current. This is similar to
        compute_for_token_gen, but the Q here is from concatenated prompt, so
        the num of queries for each seq can be a value that is not one.
        """
        assert not self.k_cache_transposed, 'Transposed K cache is not yet supported by contexted prefill feature.'
        K_active = K
        V_active = V

        K_prior = past_key_value[0]
        V_prior = past_key_value[1]

        args = (
            kwargs.get("cache_mask"),
            kwargs.get("cache_reordered_idx"),
            kwargs.get("current_reordered_idx"),
        )
        K_combined = contexted_kv(K_prior, K_active, *args)
        V_combined = contexted_kv(V_prior, V_active, *args)

        K_combined = repeat_kv(K_combined, self.num_key_value_groups)
        V_combined = repeat_kv(V_combined, self.num_key_value_groups)

        active_scores = self.scaled_qk(Q, K_combined, attention_mask)
        active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
            Q.dtype
        )
        attn_output = torch.matmul(active_scores, V_combined)
        return attn_output

    def attention_context_encode(self, Q, K, V, q_len, bsz, attention_mask):
        attn_output, flash_attn_strategy = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
        if self.flash_decoding_enabled:
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

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            # transpose BHDS -> BSHD
            # this layout avoids additional transposes between attention kernel and output projection
            attn_output = attn_output.permute(0, 3, 1, 2)
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
                is_block_kv=True,
            )

        if self.neuron_config.is_chunked_prefill:
            # Leverage contexted prefill to do attention for block kV layout.
            return self.perform_contexted_prefill(Q, K, V, past_key_value, attention_mask, **kwargs)

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

        if self.attn_block_tkg_nki_kernel_enabled and is_token_gen:
            attn_output, KV, cos_cache, sin_cache = self.attention_block_tokengen_nki_kernel(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                update_kv_per_layer,
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
            attn_output, K, V = self.attention_context_encode(Q, K, V, q_len, bsz, attention_mask)

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
