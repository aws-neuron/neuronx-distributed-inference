import logging
from typing import Tuple

import torch 
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)06d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


class Cohere2RotaryEmbedding(nn.Module):
    """Rotational position embedding (RoPE) implementation adapted from [the HuggingFace Transformers implementation](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/cohere/modular_cohere.py#L74) 
    of Cohere RoPE.
    """

    def __init__(self, 
                 head_dim: int, 
                 rope_theta: float,
                 ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = rope_theta
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.FloatTensor, position_ids: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Creates RoPE using interleaved frequencies (different from Llama).

        Args:
            x (torch.FloatTensor): Activation tensor of shape `[batch_size, num_attention_heads, seq_len, head_dim]`
            position_ids (torch.LongTensor): Position IDs tensor of shape `[batch_size, seq_len]` and type Int32 or Int64.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Cos & Sin halves of RoPE, each of shape `[batch_size, seq_len, head_dim]`
        """
        if self.inv_freq is None:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).to(device=x.device, dtype=torch.int64) / self.head_dim))
            self.inv_freq = inv_freq.to(dtype=torch.float32)
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].to(dtype=torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _rotate_half(x: torch.FloatTensor) -> torch.FloatTensor:
    """Permute input tensor coordinates so it can be multiplied with the sin-half of pre-computed RoPE assuming cos & sin 
    RoPE tensors have been generated with interleaved frequencies.

    Args:
        x (torch.FloatTensor): Activation tensor of shape `[batch_size, num_attention_heads, seq_len, head_dim]`

    Returns:
        torch.FloatTensor: Rotated activation tensor of shape `[batch_size, num_attention_heads, seq_len, head_dim]`
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rotary_position_embedding(
        q: torch.FloatTensor, 
        k: torch.FloatTensor, 
        cos: torch.FloatTensor, 
        sin: torch.FloatTensor, 
        unsqueeze_dim: int=1
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Apply RoPE to input query and key tensors using pre-computed cos & sin RoPE tensors.

    Args:
        q (torch.FloatTensor): Query activations tensor of shape `[batch_size, num_attention_heads, seq_len, head_dim]`
        k (torch.FloatTensor): Key activations tensor of shape `[batch_size, num_attention_heads, seq_len, head_dim]`
        cos (torch.FloatTensor): RoPE cos half tensor of shape `[batch_size, seq_len, head_dim]`
        sin (torch.FloatTensor): RoPE cos half tensor of shape `[batch_size, seq_len, head_dim]`
        unsqueeze_dim (int, optional): Position of the `num_attention_heads` dimension in input query & key tensors. Defaults to 1.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Rotated query & key activations tensors, each of shape `[batch_size, num_attention_heads, seq_len, head_dim]`
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
