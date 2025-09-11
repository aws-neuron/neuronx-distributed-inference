import math
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import (
    get_kv_shared_group,
    get_tensor_model_parallel_group,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from torch import Tensor, nn

from neuronx_distributed_inference.modules.custom_calls import neuron_cumsum
from neuronx_distributed_inference.modules.attention.attention_process_groups import tp_mesh_8_by_8, _fully_contiguous_tp_mesh, get_tp_cp_group_mesh

weight_cache = {}


def _get_weight_from_state_dict(prefix: str, state_dict: Dict[str, Any]) -> torch.Tensor:
    if prefix in weight_cache:
        return weight_cache[prefix]

    if (prefix + "weight") in state_dict:
        transposed_weight = state_dict[prefix + "weight"].t()
        weight_cache[prefix] = transposed_weight
        return transposed_weight

    else:
        raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")


def _set_weight_to_state_dict(
    prefix: str, tensor: torch.Tensor, state_dict: Dict[str, Any]
) -> None:
    if (prefix + "weight") in state_dict:
        state_dict[prefix + "weight"] = tensor.t()
    else:
        raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")


def transpose_parallel_linear_layer(parallel_layer):
    """
    This function clones and transposes a ColumnParallelLinear or RowParallelLinear
    The attributes are also cloned and partition_dim is updated
    """
    orig_attrs = vars(parallel_layer)
    new_layer = torch.nn.Parameter(parallel_layer.clone().T, requires_grad=False)
    new_layer.__dict__.update(orig_attrs)
    # flip the partition_dim from 0->1 or 1->0
    setattr(new_layer, "partition_dim", 1 - getattr(new_layer, "partition_dim"))
    setattr(new_layer, "get_tensor_from_state_dict", _get_weight_from_state_dict)
    setattr(new_layer, "set_tensor_to_state_dict", _set_weight_to_state_dict)
    return new_layer


def pad_to_128_multiple(x, dim):
    # Strided padding for unsharded weight, so after sharding
    # each rank will have dense padding at the end.
    # Eg orig shape = [16384, 53248], with dim = 1
    # We reshape to [16384, 128, 416] (TP_degree = 128)
    # Then pad to [16384, 128, 512].
    # Then collapse the original dim [16384, 65536].
    TP_DEGREE = get_tensor_model_parallel_group().size()
    orig_shape = x.shape
    new_shape = list(x.shape)
    new_shape[dim] = orig_shape[dim] // TP_DEGREE
    new_shape.insert(dim, TP_DEGREE)
    x = x.reshape(new_shape)
    dim += 1
    padding_length = get_padding_length(x.shape[dim], 128)
    dimlist = [0] * (len(x.shape) * 2)
    dimlist[dim * 2] = padding_length
    padded = torch.nn.functional.pad(x, tuple(dimlist[::-1]))
    new_padded_shape = list(orig_shape)
    new_padded_shape[dim - 1] = -1
    padded = padded.reshape(new_padded_shape)
    return padded


quantized_weight_cache = {}


def _get_weight_from_state_dict_quantized(prefix: str, state_dict: Dict[str, Any]) -> torch.Tensor:
    if prefix in quantized_weight_cache:
        return quantized_weight_cache[prefix]

    if (prefix + "weight") in state_dict:
        # Need to pad tensor to nearest multiple of 128 (after sharding), then transpose.
        # Padding not supported for fp8 so view as int8 then view back.
        quantized_tensor = state_dict[prefix + "weight"]
        assert (
            quantized_tensor.dtype == torch.float8_e4m3fn
        ), "Expected weight type to be float8_e4m3fn"
        dim = 0 if "down_proj" in prefix else 1
        quantized_tensor = pad_to_128_multiple(quantized_tensor.view(torch.int8).t(), dim)
        quantized_tensor = quantized_tensor.view(torch.float8_e4m3fn)
        quantized_tensor = quantized_tensor.contiguous()
        quantized_weight_cache[prefix] = quantized_tensor
        return quantized_tensor
    else:
        raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")


quantized_scale_cache = {}


def _get_scale_from_state_dict_quantized(prefix: str, state_dict: Dict[str, Any]) -> torch.Tensor:
    if prefix in quantized_scale_cache:
        return quantized_scale_cache[prefix]

    if (prefix + "scale") in state_dict:
        # Transformations for fp8 kernel scale inputs

        # Original shape in checkpoint
        # gate/up:  [I, 1]
        # down:     [H, 1]

        # New shape needed (gate/up)
        # pad I to be multiple of 128 after sharding --> [I_padded, 1]
        # transpose --> [1, I_padded]
        # broadcast --> [128, I_padded]

        # New shape needed (down)
        # transpose --> [1, H]
        # broadcast --> [128, H]
        scale = state_dict[prefix + "scale"]
        if "down_proj" not in prefix:
            scale = pad_to_128_multiple(scale, 0)
        scale = scale.t()
        scale = torch.broadcast_to(scale, (128, scale.shape[1]))
        scale = scale.contiguous()
        quantized_scale_cache[prefix] = scale
        return scale
    else:
        raise RuntimeError(f"Cannot find {(prefix + 'scale')} in the state_dict")


def preprocess_quantized_linear_weight(layer):
    orig_weight_attrs = vars(layer.weight)
    layer.weight = torch.nn.Parameter(layer.weight.clone().T, requires_grad=False)

    # Add methods for loading from checkpoint
    layer.weight.__dict__.update(orig_weight_attrs)
    setattr(layer.weight, "partition_dim", 1 - getattr(layer.weight, "partition_dim"))
    setattr(layer.weight, "get_tensor_from_state_dict", _get_weight_from_state_dict_quantized)
    # setattr(layer.weight, "set_tensor_to_state_dict", _set_weight_to_state_dict) # TODO: Is this needed?


def preprocess_quantized_linear_scale(layer):
    orig_scale_attrs = vars(layer.scale)

    # Transpose scale
    scale = layer.scale.clone().T
    del layer.scale

    # Broadcast scale
    scale = torch.broadcast_to(scale, (128, scale.shape[1]))
    # In the checkpoint the attr is scale, so patch here.
    setattr(layer, "scale", torch.nn.Parameter(scale, requires_grad=False))

    # Add methods for loading from checkpoint
    layer.scale.__dict__.update(orig_scale_attrs)
    setattr(layer.scale, "partition_dim", 1 - getattr(layer.scale, "partition_dim"))
    setattr(layer.scale, "get_tensor_from_state_dict", _get_scale_from_state_dict_quantized)
    # setattr(layer.weight, "set_tensor_to_state_dict", _set_weight_to_state_dict) # TODO: Is this needed?


def preprocess_quantized_linear_layer(layer):
    preprocess_quantized_linear_weight(layer)
    preprocess_quantized_linear_scale(layer)


def move_heads_front(
    tensor: Tensor, bsz: int, seq_len: int, num_head: int, head_dim: int, layernorm=None
) -> Tensor:
    """Reshape input tensor: BSHD -> BHSD, and apply layer normalization if layernorm is specified"""
    tensor = tensor.view(bsz, seq_len, num_head, head_dim)
    if layernorm:
        tensor = layernorm(tensor)
    return tensor.transpose(1, 2).contiguous()


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _rotate_half(x) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
) -> Tuple[Tensor, Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def manual_softmax(prior_scores, active_scores, is_speculation) -> Tuple[Tensor, Tensor]:
    """
    simple softmax computation: denominator is the sum of exp over all vocab and only need compute numerator (exp)
    """
    max_score = torch.max(prior_scores, dim=-1, keepdim=True)[0]
    max_active_score = torch.max(active_scores, dim=-1, keepdim=True)[0]
    max_score = (
        torch.maximum(max_score, max_active_score)
        if is_speculation
        else torch.maximum(max_score, active_scores)
    )

    exp_prior = torch.exp(prior_scores - max_score)
    exp_active = torch.exp(active_scores - max_score)
    denominator = exp_prior.sum(dim=-1, keepdim=True) + exp_active.sum(dim=-1, keepdim=True)

    softmax_prior = exp_prior / denominator
    softmax_active = exp_active / denominator
    return softmax_prior, softmax_active


def distributed_softmax(prior_scores, active_scores) -> Tuple[Tensor, Tensor]:
    """
    compute partial softmax and then gather and correct final softmax.
    """
    # find local max
    max_score = torch.max(prior_scores, dim=-1, keepdim=True)[0]
    max_active_score = torch.max(active_scores, dim=-1, keepdim=True)[0]
    local_max_score = torch.maximum(max_score, max_active_score)

    exp_prior = torch.exp(prior_scores - local_max_score)
    exp_active = torch.exp(active_scores - local_max_score)
    denominator = exp_prior.sum(dim=-1, keepdim=True) + exp_active.sum(dim=-1, keepdim=True)

    # collect for global max and exp sum (denominator)
    groups = get_kv_shared_group(as_list=True)
    gather_payload = torch.cat((local_max_score, denominator), dim=0)
    gathered_res = xm.all_gather(gather_payload, dim=-1, groups=groups, pin_layout=False)
    gathered_max, gathered_denom = torch.chunk(gathered_res, 2, dim=0)
    global_max = torch.max(gathered_max, dim=-1, keepdim=True)[0]

    # softmax correction
    scaling_factor = torch.exp(gathered_max - global_max.expand(gathered_max.shape))
    corrected_denominator = torch.multiply(scaling_factor, gathered_denom)
    corrected_denominator = torch.sum(corrected_denominator, dim=-1, keepdim=True)

    corrected_exp_prior = torch.exp(prior_scores - global_max)
    corrected_exp_active = torch.exp(active_scores - global_max)

    softmax_prior = corrected_exp_prior / corrected_denominator
    softmax_active = corrected_exp_active / corrected_denominator
    return softmax_prior, softmax_active


class RotaryEmbedding(nn.Module):
    """
    Adapted from Llama 4.0 impl https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models
    /llama/modeling_llama.py#L96-L145
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
        return 1.0 / (self.base ** (freq_indices / self.dim))

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = self.get_inv_freqs(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Utility functions to create attention mask
def create_block_diagonal_attn_mask(
    query_lens: torch.Tensor,
    key_lens: torch.Tensor,
    max_query_len: torch.Tensor,
    max_key_len: torch.Tensor,
    is_prior: bool = False,
):
    """
    Return a block diagonal atttention mask which can be used by chunked
    prefill.

    This function is written in a way that it can be traced, so it can
    be used inside the NeuronBaseModel class.

    Example:
        query_lens = [2,3,1,0]
        key_lens = [4,5,4,0]
        max_query_len = 8
        max_key_len = 16

        mask = [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 1st sequence
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 4 attend to 1st sequence
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 2nd sequence
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # At position 4 attend to 2nd sequence
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # At position 5 attend to 2nd sequence
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], # At position 3 attend to 3rd sequence
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
        ]
    Args:
        query_lens: a list of query lengths for each sequence
        key_lens: a list of key lengths for each sequence
        max_query_len: the max value of the sum of query lengths
        max_key_len: the max value of the sum of key lengths

    Return:
        mask: the causal attention mask for chunked prefill
    """
    batch_size = query_lens.shape[0]
    dtype = query_lens.dtype
    device = query_lens.device

    row_idx = torch.arange(max_query_len, dtype=dtype, device=device).reshape(-1, 1)
    col_idx = torch.arange(max_key_len, dtype=dtype, device=device).reshape(1, -1)

    q_cumsum = neuron_cumsum(query_lens.reshape(1, -1).float()).reshape(-1).int()
    q_cumsum = F.pad(q_cumsum, pad=[1, 0])
    k_cumsum = neuron_cumsum(key_lens.reshape(1, -1).float()).reshape(-1).int()
    k_cumsum = F.pad(k_cumsum, pad=[1, 0])

    mask = torch.zeros(max_query_len, max_key_len, dtype=torch.bool, device=device)
    for seq_id in range(batch_size):
        ri = q_cumsum[seq_id]  # row index
        ci = k_cumsum[seq_id]  # column index
        nr = query_lens[seq_id]  # number of rows
        nc = key_lens[seq_id]  # number of columns

        offset = ci + nc - ri - nr
        # upper right triangle is set to false
        diagonal_mask = (row_idx - col_idx + offset) >= 0

        left_mask = col_idx >= ci
        top_mask = row_idx >= ri
        bottom_mask = row_idx < ri + nr

        if is_prior:
            right_mask = col_idx < ci + nc - nr
            mask_per_seq = diagonal_mask & left_mask & top_mask & bottom_mask & right_mask
        else:
            mask_per_seq = diagonal_mask & left_mask & top_mask & bottom_mask

        mask = mask | mask_per_seq

    return mask


def neuron_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    # Python-level implementation for torch.nn.functional.scaled_dot_product_attention

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def get_context_parallel_reordered_tp_mapping(world_size, cp_degree, num_kv_heads, device=torch.device("cpu"), cte_rank_ordering=None):
    # world_size: world size
    # cp_degree: the cp degree CTE attention is running in

    # Flattens the CP mesh which contains the TP ordering with the contigous KV heads
    # This is done to enable running full TP decode after doing context parallel CTE

    # Returns a list where each index, i, is the original rank and list[i] is the new rank assuming TP decode
    # This ordering aligns the KV heads written by CTE with how we shard weights in TKG

    assert world_size >= num_kv_heads, "CP is with full TP decode is currently not supported with num_kv_heads > world_size"

    if cte_rank_ordering is None:
        cte_rank_ordering = list(range(0, world_size))

    tp_degree = world_size // cp_degree
    cp_interleave_factor = max(tp_degree // num_kv_heads, 1)

    heads_in_cp = torch.stack(torch.arange(num_kv_heads, device=device, dtype=torch.int32).repeat_interleave(cp_interleave_factor).tensor_split(tp_degree)).repeat(cp_degree, 1)
    heads_in_cp = torch.index_select(heads_in_cp, dim=0, index=torch.tensor(cte_rank_ordering, dtype=torch.int32, device=device))
    heads_in_tp = torch.arange(num_kv_heads, device=device, dtype=torch.int32).repeat_interleave(world_size // num_kv_heads)
    heads_in_tp = heads_in_tp.view(-1, 1)

    output = []
    used_indices = set()

    for cp_heads_per_rank in heads_in_cp:
        for idx, tp_head_per_rank in enumerate(heads_in_tp):
            if idx not in used_indices and tp_head_per_rank[0] in cp_heads_per_rank:
                output.append(idx)
                used_indices.add(idx)
                break

    return output


def get_kv_head_indices_context_parallel_full_tp_decode(num_kv_heads, world_size, cp_degree, device, cte_rank_ordering=None):
    # world_size: world_size
    # cp_degree: the cp degree CTE attention is running in

    # Returns the index of the first KV head per rank wrt the context parallel KV heads per rank
    # Example: TP = 4, KV = 4, CP = 2
    # CP Heads: [[(R0) KV0 KV1, (R1) KV2 KV3], [(R2) KV0 KV1, (R3) KV2 KV3]]
    # TP Heads: [(R0) KV0, (R2) KV1, (R1) KV2, (R3) KV3]
    # Output: [0, 1, 0, 1]

    tp_ordering = get_context_parallel_reordered_tp_mapping(world_size, cp_degree, num_kv_heads, device, cte_rank_ordering)
    tp_degree = world_size // cp_degree

    assert world_size >= num_kv_heads, "CP is with full TP decode is currently not supported with num_kv_heads > world_size"

    # If TP < num_kv_heads or TP == num_kv_heads, no need to interleave for padding
    cp_interleave_factor = max(tp_degree // num_kv_heads, 1)

    heads_in_cp = torch.stack(torch.arange(num_kv_heads, device=device, dtype=torch.int32).repeat_interleave(cp_interleave_factor).tensor_split(tp_degree)).repeat(cp_degree, 1)
    heads_in_tp = torch.arange(num_kv_heads, device=device, dtype=torch.int32).repeat_interleave(world_size // num_kv_heads)
    heads_in_tp = torch.index_select(heads_in_tp, dim=0, index=torch.tensor(tp_ordering, dtype=torch.int32, device=device))
    heads_in_tp = heads_in_tp.view(-1, 1)
    mask = (heads_in_cp == heads_in_tp)
    indices = mask.int().argmax(dim=1)

    return indices


def get_context_parallel_reordered_dp_mapping(world_size: int, cp_degree: int, dp_degree: int, num_kv_heads: int, device: torch.device = torch.device("cpu"), cte_rank_ordering: List = None):
    # world_size: world size
    # cp_degree: the cp degree CTE attention is running in
    # dp_degree: the dp degree TKG attention is running in
    # cte_rank_ordering: ordering of ranks used during CTE

    # Determines the rank ordering for the first TP group in DP i.e. ranks [0, world_size / dp] and
    # offsets the ranks dp number of times to get the entire rank ordering

    # Returns a list where each index, i, is the original rank and list[i] is the new rank assuming TP + DP decode

    mapping = []

    dp_tp_size = world_size // dp_degree

    # We don't support CP < DP degree, the reason is due to the KV cache writes, when CP < DP
    # The TP for CP is greater than the TP for DP, in prefill we only write the cache based on whether
    # those ranks operate on that batch in DP, when the TP for CP is greater, there is no guarantee that all KV
    # heads are written. When the TP for CP is lower, we know all heads will be written and can be reordered given we have enough copies.

    if cte_rank_ordering is not None:
        cte_rank_ordering = cte_rank_ordering[0:dp_tp_size]

    # Treat DP like we run full TP in smaller TP blocks to ensure we only reorder ranks within the TP groups
    tp_mapping = get_context_parallel_reordered_tp_mapping(dp_tp_size, cp_degree // dp_degree, num_kv_heads, device, cte_rank_ordering)

    # Offset all the tp mapping by the dp ranks, the above tp mapping only maps the first [0 - world_size / dp] ranks
    for dp_rank in range(dp_degree):
        offset_tp_mapping = tp_mapping.copy()
        offset_tp_mapping = [rank + (dp_tp_size * dp_rank) for rank in offset_tp_mapping]
        mapping += offset_tp_mapping

    # The above ordering assumes a continuous ordering in decode, when we have 8x8 that's not the case.
    if dp_degree == 8 and dp_tp_size == 8:
        shuffle_accounted_ordering = [-1] * world_size
        true_ordering = sum(get_tp_cp_group_mesh(world_size, dp_degree), [])

        for rank in range(0, world_size):
            shuffle_accounted_ordering[rank] = mapping[true_ordering.index(rank)]

        return shuffle_accounted_ordering

    return mapping


def get_kv_head_indices_context_parallel_dp_decode(num_kv_heads: int, world_size: int, cp_degree: int, dp_degree: int, device: torch.device, cte_rank_ordering: List = None, decode_rank_ordering: List = None):
    # world_size: world_size
    # cp_degree: the cp degree CTE attention is running in
    # dp_degree: the dp degree TKG attention is running in
    # decode_rank_ordering: ordering of ranks during decode

    # Returns the index of the first KV head per rank wrt the context parallel KV heads per rank assuming DP decode

    cp_tp_degree = world_size // cp_degree
    dp_tp_degree = world_size // dp_degree

    # TODO: support this case by writing a slice of the KV in kv_manager rather than a single head
    assert dp_tp_degree >= num_kv_heads, "CP with DP decode when CP != DP is currently not supported with num_kv_heads > (world_size / dp_degree)"

    # If TP < num_kv_heads or TP == num_kv_heads, no need to interleave for padding
    cp_interleave_factor = max(cp_tp_degree // num_kv_heads, 1)
    dp_interleave_factor = max(dp_tp_degree // num_kv_heads, 1)

    required_dp_kv_copies = dp_interleave_factor * dp_degree
    existing_cp_kv_copies = cp_interleave_factor * cp_degree

    assert existing_cp_kv_copies >= required_dp_kv_copies, f"CP{cp_degree} with DP{dp_degree} and {num_kv_heads} KV Heads is not a supported configuration"

    tp_ordering = get_context_parallel_reordered_dp_mapping(world_size, cp_degree, dp_degree, num_kv_heads, device, cte_rank_ordering)

    heads_in_cp = torch.stack(torch.arange(num_kv_heads, device=device, dtype=torch.int32).repeat_interleave(cp_interleave_factor).tensor_split(cp_tp_degree)).repeat(cp_degree, 1)
    heads_in_dp = torch.stack(torch.arange(num_kv_heads, device=device, dtype=torch.int32).repeat_interleave(dp_interleave_factor).tensor_split(dp_tp_degree)).repeat(dp_degree, 1)
    heads_in_dp = torch.index_select(heads_in_dp, dim=0, index=torch.tensor(tp_ordering, dtype=torch.int32, device=device))

    if decode_rank_ordering:
        heads_in_dp = torch.index_select(heads_in_dp, dim=0, index=torch.tensor(decode_rank_ordering, dtype=torch.int32, device=device))

    heads_in_dp = heads_in_dp.view(-1, 1)
    mask = (heads_in_cp == heads_in_dp)
    indices = mask.int().argmax(dim=1)

    return indices


def validate_tp_prefill_to_dp_decode(num_kv_heads, world_size, dp_degree):
    # num_kv_heads: total number of kv heads
    # world_size: world_size
    # dp_degree: the dp degree decode attention is running in

    # validates whether it's possible to run prefill in full TP and decode in TP + DP

    # This is the reduced tp_degree that decode attention will be running in,
    # we make sure the reduced tp decode and full tp prefill have the same number of heads
    tp_degree = world_size // dp_degree

    dp_heads_per_rank = max(num_kv_heads // tp_degree, 1)
    tp_heads_per_rank = max(num_kv_heads // world_size, 1)

    # If we have more heads in DP decode, we need a collective to write all the expected heads during CTE
    # this is currently not implemented due to not being able to do this with a reasonable perf

    # TODO: if dp_heads_per_rank < tp_heads_per_rank we can support this case by slicing the KV writes
    assert tp_heads_per_rank == dp_heads_per_rank, "DP configuration is not supported"


def reshape_qkv_for_chunked_flash_attention_kernel(Q, K, V, chunk_size, torch_dtype):
    # Since bsz is always 1, we can bring n_chunks dimension into batch dimension
    # Starting shape for Q, K, V: BHSD. Since S = n_chunks * chunk_size, we can reshape as follows
    bsz, num_heads, q_len, head_dim = Q.shape
    n_chunks = math.ceil(q_len / chunk_size)
    Q = Q.reshape(bsz, num_heads, n_chunks, chunk_size, head_dim).squeeze(0)
    Q = (
        Q.permute(1, 0, 3, 2)
        .reshape((n_chunks * num_heads, head_dim, chunk_size))
        .to(torch_dtype)
    )
    K = K.reshape(bsz, num_heads, n_chunks, chunk_size, head_dim).squeeze(0)
    K = (
        K.permute(1, 0, 3, 2)
        .reshape((n_chunks * num_heads, head_dim, chunk_size))
        .to(torch_dtype)
    )
    V = V.reshape(bsz, num_heads, n_chunks, chunk_size, head_dim).squeeze(0)
    V = V.reshape((n_chunks * num_heads, chunk_size, head_dim)).to(
        torch_dtype
    )
    return Q, K, V


def get_last_kv_chunk(attention_chunk_size, position_ids, latest_k, latest_v):
    """
    For chunked attention, first determine the latest chunk we are in based on position id.
    Then we only gather that chunk of KV into KV cache.
    In the edge case where seq_len bucket is not divisible by chunk size, our latest chunk will not have the full chunk size.
    In this case, we will have to pad latest kv to chunk size.
    """
    latest_position_ids = torch.amax(position_ids, dim=1)
    chunk_idx = latest_position_ids // attention_chunk_size
    # Create gather indices for the chunk
    batch_size, num_heads, seq_len, head_dim = latest_k.shape
    max_len = position_ids.shape[1]

    if max_len > attention_chunk_size and max_len % attention_chunk_size != 0:
        chunk_pad_len = attention_chunk_size - max_len % attention_chunk_size
    else:
        chunk_pad_len = 0
    latest_k = F.pad(latest_k, (0, 0, 0, chunk_pad_len), mode='constant', value=0)
    latest_v = F.pad(latest_v, (0, 0, 0, chunk_pad_len), mode='constant', value=0)
    chunk_idx = chunk_idx[:, None].expand(batch_size, attention_chunk_size)
    gather_indices = torch.arange(attention_chunk_size)[None, :].expand(batch_size, attention_chunk_size) + chunk_idx * attention_chunk_size
    gather_indices = gather_indices[:, None, :, None].expand(batch_size, num_heads, attention_chunk_size, head_dim).to(device=latest_k.device)
    # Gather the chunks
    latest_k = torch.gather(latest_k, dim=2, index=gather_indices)
    latest_v = torch.gather(latest_v, dim=2, index=gather_indices)
    return latest_k, latest_v


def get_last_kv_window(window_size, position_ids, latest_k, latest_v):
    batch_size, num_head, _, head_dim = latest_k.shape
    latest_pos = torch.amax(position_ids, dim=1)
    end_idx = (latest_pos + 1).clamp(min=window_size)
    start_idx = (end_idx - window_size).clamp(min=0)
    orig_indices = start_idx[:, None] + torch.arange(window_size)

    # Calculate per-batch left shifts
    left_shifts = (window_size - (end_idx % window_size)) % window_size
    base = torch.arange(window_size).expand(batch_size, window_size)
    shifted_idx = (base + left_shifts[:, None]) % window_size

    # Determine per-batch shifted gather indices
    gather_idx = torch.gather(orig_indices, dim=1, index=shifted_idx)
    gather_idx = gather_idx[:, None, :, None].expand(batch_size, num_head, window_size, head_dim).to(device=latest_k.device)

    # Gather to create non-physically contiguous KV cache
    latest_k = torch.gather(latest_k, dim=2, index=gather_idx)
    latest_v = torch.gather(latest_v, dim=2, index=gather_idx)
    return latest_k, latest_v


def stride_tensor(tensor: torch.tensor, dim: int, stride: int):
    """
    Reorders elements in a tensor along the specified dimension using stride pattern.

    Takes a tensor and reorders its elements along the specified dimension
    by grouping elements that are `step_size` apart. For example, with step_size=2,
    elements at positions [0,2,4,...] are grouped together, followed by elements
    at positions [1,3,5,...].

    tensor: Tensor to stride
    dim: Dimension to stride on
    step_size: The stride length

    Example:
        tensor = torch.tensor([0, 1, 2, 3, 4, 5])
        stride_tensor(tensor, 0, 2)

        tensor([0, 2, 4, 1, 3, 5])
    """
    dim_size = tensor.shape[dim]
    assert dim_size % stride == 0, f"Dimension size {dim_size} must be divisible by stride {stride}"

    block_size = dim_size // stride
    indices = (torch.arange(dim_size, device=tensor.device) % block_size) * stride + (torch.arange(dim_size, device=tensor.device) // block_size)

    idx = [slice(None)] * tensor.dim()
    idx[dim] = indices

    return tensor[tuple(idx)]


def order_strided_tensor(tensor: torch.tensor, dim: int, stride: int):
    """
    Restores the original order of a tensor that was previously reordered by stride_tensor.

    This function is the inverse operation of `stride_tensor`. It takes a tensor whose
    elements have been reordered along a specified dimension using a stride pattern,
    and restores them to their original sequential order.

    tensor: Tensor to stride
    dim: Dimension to stride on
    step_size: The stride length

    Example:
        tensor = torch.tensor([0, 2, 4, 1, 3, 5])
        stride_tensor(tensor, 0, 2)

        tensor([0, 1, 2, 3, 4, 5])
    """
    dim_size = tensor.shape[dim]
    assert dim_size % stride == 0, f"Dimension size {dim_size} must be divisible by stride {stride}"

    block_size = dim_size // stride
    inverse_indices = block_size * (torch.arange(dim_size, device=tensor.device) % stride) + (torch.arange(dim_size, device=tensor.device) // stride)

    idx = [slice(None)] * tensor.dim()
    idx[dim] = inverse_indices

    return tensor[tuple(idx)]


def get_cp8_tp8_rank_ordering(world_size, cp_degree, device=torch.device("cpu")):
    """
    When the 8x8 mesh is being used, the TP group ranks are discontiguous. This function returns the rank ordering
    needed to correct for the sharding such that the discontiguous ranks get the right weights.
    """
    non_contiguous_mesh = tp_mesh_8_by_8()
    non_contiguous_mesh = sum(non_contiguous_mesh, [])

    contiguous_mesh = _fully_contiguous_tp_mesh(world_size, cp_degree)
    contiguous_mesh = sum(contiguous_mesh, [])

    combined = dict(zip(non_contiguous_mesh, contiguous_mesh))

    cte_rank_ordering = []
    for i in range(0, world_size):
        cte_rank_ordering.append(combined[i])

    return torch.tensor(cte_rank_ordering, device=device)


def chunk_and_reorder_tensor(tensor: torch.tensor, order: List, dim: int):
    """
    Split a tensor into chunks along a specified dimension and reorder them.
    The number of chunks is defined by the length of the order specified.

    Example:
        tensor = [0, 1, 2, 3]
        order = [1, 0]

        output = [2, 3, 0, 1]

    tensor (torch.Tensor): The input tensor of any dimension
    order (list): List specifying the new ordering of chunks
    dim (int): The dimension along which to reorder chunks (default: 0)
    """
    n_chunks = len(order)
    dim_size = tensor.shape[dim]

    chunk_starts = torch.tensor([int(dim_size * i / n_chunks) for i in range(n_chunks)], device=tensor.device)
    chunk_ends = torch.tensor([int(dim_size * (i + 1) / n_chunks) for i in range(n_chunks)], device=tensor.device)

    # Create indices for each chunk in the specified order
    indices = torch.cat([torch.arange(chunk_starts[i], chunk_ends[i], device=tensor.device) for i in order])

    return torch.index_select(tensor, dim, indices)
