from typing import Any, Dict, Tuple

import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import (
    get_kv_shared_group,
    get_tensor_model_parallel_group,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from torch import Tensor, nn

torch.manual_seed(0)

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

    if (prefix + "weight_scale") in state_dict:
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
        scale = state_dict[prefix + "weight_scale"]
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
    # Broadcast scale
    scale = torch.broadcast_to(scale, (128, scale.shape[1]))
    # In the checkpoint the attr is weight_scale, so patch here.
    setattr(layer, "weight_scale", torch.nn.Parameter(scale))

    # Add methods for loading from checkpoint
    layer.weight_scale.__dict__.update(orig_scale_attrs)
    setattr(layer.weight_scale, "partition_dim", 1 - getattr(layer.weight_scale, "partition_dim"))
    setattr(layer.weight_scale, "get_tensor_from_state_dict", _get_scale_from_state_dict_quantized)

    del layer.scale
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

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
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

    row_idx = torch.arange(max_query_len, dtype=torch.int).reshape(-1, 1)
    col_idx = torch.arange(max_key_len, dtype=torch.int).reshape(1, -1)

    q_cumsum = torch.cumsum(query_lens, dim=0)
    q_cumsum = torch.cat([torch.tensor(0).reshape(1), q_cumsum])
    k_cumsum = torch.cumsum(key_lens, dim=0)
    k_cumsum = torch.cat([torch.tensor(0).reshape(1), k_cumsum])

    mask = torch.zeros(max_query_len, max_key_len, dtype=torch.bool)
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

        mask_per_seq = diagonal_mask & left_mask & top_mask & bottom_mask
        mask = mask | mask_per_seq

    return mask
