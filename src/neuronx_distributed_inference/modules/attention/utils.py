from typing import Tuple

import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import get_kv_shared_group
from torch import Tensor, nn

torch.manual_seed(0)


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
