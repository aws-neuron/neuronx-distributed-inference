import logging
import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributed import ProcessGroup

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
from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

from .gqa import GQA, GroupQueryAttention_O, GroupQueryAttention_QKV  # noqa: E402

logger = logging.getLogger("Neuron")

_flash_fwd_call = nki_jit()(attention_isa_kernel)


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

        self.num_cores_per_group = 1
        self.flash_decoding_enabled = False
        self.sequence_parallel_enabled = False
        self.sequence_dimension = None
        self.rpl_reduce_dtype = None

        self.o_proj_layer_name = "o_proj"

    def init_gqa_properties(self):
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )

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
            rms_norm_eps=self.config.rms_norm_eps,
            qkv_kernel_enabled=self.neuron_config.qkv_kernel_enabled,
            logical_neuron_cores=self.neuron_config.logical_neuron_cores,
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
        )
        self.num_heads = utils.divide(self.qkv_proj.get_num_attention_heads(), self.tp_degree)
        self.num_key_value_heads = utils.divide(
            self.qkv_proj.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.head_dim)
            self.k_layernorm = nn.LayerNorm(self.head_dim)
        self.attn_kernel_enabled = self.neuron_config.attn_kernel_enabled
        self.logical_neuron_cores = self.neuron_config.logical_neuron_cores

    def scaled_qk(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
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
    ):
        """take care of the shape, layout, group query, custom position encoding, etc."""
        Q, K, V = self.qkv_proj(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids
        )

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
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        return Q, K, V, cos_cache, sin_cache

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        # use flash attention if
        # (i) attn_kernel_enabled is True in neuron_config
        # (ii) sequence length is large enough to get the best performance,
        # (iii) Q, K, and V have the same shape. Conditions can be changed in the future.
        flash_attention_eligible = (
            self.attn_kernel_enabled or q_len >= 4096
        ) and Q.shape == K_active.shape == V_active.shape

        if flash_attention_eligible:
            logger.debug(f"ATTN kernel: logical_neuron_cores={self.logical_neuron_cores}")
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

            if int(self.logical_neuron_cores) > 1:
                grid = (vnc(self.logical_neuron_cores),)

                _flash_fwd_call[grid](
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
            else:
                # attention kernel does not support passing in a grid for LNC=1
                _flash_fwd_call(
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
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
        return attn_output

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

    def compute_for_token_gen(
        self, Q, K, V, position_ids, past_key_value, attention_mask, active_mask
    ) -> Tensor:
        """attention computation at token generation phase"""
        is_speculation = position_ids.shape[-1] > 1

        # Attention computation: softmax((Q.K/√dkv) + mask).V
        # i. prior (cached) KV
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / math.sqrt(self.head_dim)
        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)

        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)
        if is_speculation:
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
        active_scores = active_scores.to(torch.float32)

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores, is_speculation)
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output

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
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Implements each layer's forward pass for the attention block."""
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q, K, V, cos_cache, sin_cache = self.prep_qkv_tensors(
            position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
        )

        if past_key_value is None:
            attn_output = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
            if self.flash_decoding_enabled:
                assert self.qkv_proj.sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE, (
                    "Flash decoding lives in the context of GQA (grouped query attention) and traditional MHA "
                    "multi-head attention) won't work!"
                )
                rank_id = self.rank_util.get_rank()
                rank_id_in_kv_group = torch.remainder(rank_id, self.num_cores_per_group).to(
                    torch.int64
                )
                # shard KV by seq len and pick the values based on rank
                assert q_len == Q.shape[2], f"Q shape is {Q.shape}"
                # selecting positions (on S dim) that belongs to the current rank
                selected_seq_pos = torch.arange(
                    rank_id_in_kv_group.item(),
                    q_len,
                    self.num_cores_per_group,
                    dtype=torch.int64,
                    device=Q.device,
                )
                K = torch.index_select(input=K, dim=2, index=selected_seq_pos)
                V = torch.index_select(input=V, dim=2, index=selected_seq_pos)
        else:
            if self.flash_decoding_enabled:
                assert active_mask is not None, "Flash decoding requires active mask is not None!"
                # gather Q from all cores in its KV group
                groups = get_kv_shared_group(as_list=True)
                Q = xm.all_gather(Q, dim=1, groups=groups, pin_layout=False)

                attn_output = self.compute_for_flash_decoding(
                    Q, K, V, past_key_value, attention_mask, active_mask
                )
                attn_output = xm.reduce_scatter(
                    xm.REDUCE_SUM,
                    attn_output,
                    scale=1,
                    scatter_dim=1,
                    shard_count=len(groups[0]),
                    groups=groups,
                    pin_layout=False,
                )
            else:
                attn_output = self.compute_for_token_gen(
                    Q, K, V, position_ids, past_key_value, attention_mask, active_mask
                )

        if self.attn_kernel_enabled and past_key_value is None:
            # transpose BHDS -> BSHD
            # this layout avoids additional transposes between attention kernel and output projection
            attn_output = attn_output.permute(0, 3, 1, 2)
        else:
            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output, adapter_ids=adapter_ids)

        past_key_value: Tuple[Tensor, Tensor] = (K, V)

        return attn_output, past_key_value, cos_cache, sin_cache
