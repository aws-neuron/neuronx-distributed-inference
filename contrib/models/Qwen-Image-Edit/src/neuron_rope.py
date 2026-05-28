"""
Neuron-compatible RoPE (Rotary Position Embedding) implementation for QwenImage.

This module provides RoPE implementations that don't use complex numbers,
which are not supported by AWS Neuron.

The original QwenImage uses torch.polar() to create complex frequencies,
but Neuron doesn't support C64 (complex64) datatypes. This implementation
uses (cos, sin) pairs instead.
"""

import torch
from torch import nn
from typing import List, Tuple, Optional, Union
import functools


class NeuronQwenEmbedRope(nn.Module):
    """
    Neuron-compatible RoPE for QwenImage that doesn't use complex numbers.

    Instead of storing complex frequencies, we store (cos, sin) pairs.
    The original implementation uses:
        freqs = torch.polar(torch.ones_like(freqs), freqs)  # complex
    We use:
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
    """
    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        # Precompute position indices (same as original)
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        # Compute frequencies as (cos, sin) instead of complex
        # Original: torch.polar(ones, freqs) -> complex exp(i*freqs)
        # We store: cos(freqs), sin(freqs) separately
        self.pos_freqs_cos, self.pos_freqs_sin = self._compute_all_freqs(pos_index)
        self.neg_freqs_cos, self.neg_freqs_sin = self._compute_all_freqs(neg_index)

    def _rope_params_real(self, index: torch.Tensor, dim: int, theta: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE frequencies as (cos, sin) instead of complex.

        Original: freqs = torch.polar(torch.ones_like(freqs), freqs)
        This returns complex tensor of shape [len(index), dim//2]

        We return (cos, sin) each of shape [len(index), dim//2]
        """
        assert dim % 2 == 0
        # Compute angles: outer product of positions and frequency bases
        freqs = torch.outer(
            index.float(),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).float() / dim)
        )
        # Return cos and sin instead of complex polar
        return torch.cos(freqs), torch.sin(freqs)

    def _compute_all_freqs(self, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute frequencies for all axes and concatenate."""
        freqs = []
        for dim in self.axes_dim:
            cos_f, sin_f = self._rope_params_real(index, dim, self.theta)
            freqs.append((cos_f, sin_f))

        # Concatenate along dimension axis
        # Each has shape [4096, axes_dim[i]//2]
        cos_all = torch.cat([f[0] for f in freqs], dim=1)
        sin_all = torch.cat([f[1] for f in freqs], dim=1)

        return cos_all, sin_all

    def forward(
        self,
        video_fhw: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        txt_seq_lens: Optional[List[int]] = None,
        device: torch.device = None,
        max_txt_seq_len: Optional[Union[int, torch.Tensor]] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute RoPE frequencies for video and text.

        Handles multiple img_shapes formats:
        - (T, H, W): single tuple for one video
        - [(T, H, W)]: list with single tuple
        - [(T1, H, W), (T2, H, W)]: list of tuples (multiple images)
        - [[(T1, H, W), (T2, H, W)]]: nested list (batch of multiple images)

        For multiple images, frames are summed to get total patch count.

        Returns:
            Tuple of (vid_freqs, txt_freqs), each being (cos, sin) tuple
        """
        # Handle deprecated txt_seq_lens parameter
        if txt_seq_lens is not None and max_txt_seq_len is None:
            max_txt_seq_len = max(txt_seq_lens) if isinstance(txt_seq_lens, list) else txt_seq_lens

        if max_txt_seq_len is None:
            raise ValueError("Either max_txt_seq_len or txt_seq_lens must be provided.")

        # Parse video_fhw into (total_frames, height, width)
        # Need to handle different formats correctly:
        # 1. (T, H, W) - single tuple
        # 2. [(T, H, W)] - list with single tuple
        # 3. [(T1, H, W), (T2, H, W)] - list of tuples for multiple images
        # 4. [[(T1, H, W), (T2, H, W)]] - nested list for batch

        if isinstance(video_fhw, tuple) and len(video_fhw) == 3 and isinstance(video_fhw[0], int):
            # Format 1: (T, H, W) - single tuple
            frame, height, width = video_fhw
        elif isinstance(video_fhw, list) and len(video_fhw) > 0:
            first_elem = video_fhw[0]
            if isinstance(first_elem, tuple) and len(first_elem) == 3 and isinstance(first_elem[0], int):
                # Format 2 or 3: [(T, H, W)] or [(T1, H, W), (T2, H, W), ...]
                # Sum frames from all tuples, assume same H, W
                frame = sum(t[0] for t in video_fhw)
                height, width = first_elem[1], first_elem[2]
            elif isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                # Format 4: [[(T1, H, W), (T2, H, W), ...]] - nested list
                # Take first batch item, sum frames from all images
                shapes = first_elem
                if isinstance(shapes[0], tuple) and len(shapes[0]) == 3:
                    frame = sum(t[0] for t in shapes)
                    height, width = shapes[0][1], shapes[0][2]
                else:
                    raise ValueError(f"Unsupported nested video_fhw format: {video_fhw}")
            else:
                raise ValueError(f"Unsupported video_fhw format: {video_fhw}")
        else:
            raise ValueError(f"Unsupported video_fhw format: {video_fhw}")

        # Compute video frequencies
        vid_cos, vid_sin = self._compute_video_freqs(frame, height, width, device)

        # Compute text frequencies
        max_txt_seq_len_int = int(max_txt_seq_len)
        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)

        txt_cos = self.pos_freqs_cos.to(device)[max_vid_index:max_vid_index + max_txt_seq_len_int]
        txt_sin = self.pos_freqs_sin.to(device)[max_vid_index:max_vid_index + max_txt_seq_len_int]

        return (vid_cos, vid_sin), (txt_cos, txt_sin)

    def _compute_video_freqs(
        self, frame: int, height: int, width: int, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute video frequencies for given dimensions."""
        seq_lens = frame * height * width

        pos_cos = self.pos_freqs_cos.to(device) if device is not None else self.pos_freqs_cos
        pos_sin = self.pos_freqs_sin.to(device) if device is not None else self.pos_freqs_sin
        neg_cos = self.neg_freqs_cos.to(device) if device is not None else self.neg_freqs_cos
        neg_sin = self.neg_freqs_sin.to(device) if device is not None else self.neg_freqs_sin

        # Split by axes dimensions (each is dim//2 because we computed with dim//2 freqs)
        split_dims = [x // 2 for x in self.axes_dim]

        pos_cos_split = pos_cos.split(split_dims, dim=1)
        pos_sin_split = pos_sin.split(split_dims, dim=1)
        neg_cos_split = neg_cos.split(split_dims, dim=1)
        neg_sin_split = neg_sin.split(split_dims, dim=1)

        # Frame frequencies (always from positive)
        freqs_frame_cos = pos_cos_split[0][:frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        freqs_frame_sin = pos_sin_split[0][:frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)

        if self.scale_rope:
            # Height: combine negative and positive
            h_neg_len = height - height // 2
            freqs_height_cos = torch.cat([neg_cos_split[1][-h_neg_len:], pos_cos_split[1][:height // 2]], dim=0)
            freqs_height_sin = torch.cat([neg_sin_split[1][-h_neg_len:], pos_sin_split[1][:height // 2]], dim=0)
            freqs_height_cos = freqs_height_cos.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_height_sin = freqs_height_sin.view(1, height, 1, -1).expand(frame, height, width, -1)

            # Width: combine negative and positive
            w_neg_len = width - width // 2
            freqs_width_cos = torch.cat([neg_cos_split[2][-w_neg_len:], pos_cos_split[2][:width // 2]], dim=0)
            freqs_width_sin = torch.cat([neg_sin_split[2][-w_neg_len:], pos_sin_split[2][:width // 2]], dim=0)
            freqs_width_cos = freqs_width_cos.view(1, 1, width, -1).expand(frame, height, width, -1)
            freqs_width_sin = freqs_width_sin.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height_cos = pos_cos_split[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_height_sin = pos_sin_split[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width_cos = pos_cos_split[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            freqs_width_sin = pos_sin_split[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        # Concatenate all axes
        freqs_cos = torch.cat([freqs_frame_cos, freqs_height_cos, freqs_width_cos], dim=-1).reshape(seq_lens, -1)
        freqs_sin = torch.cat([freqs_frame_sin, freqs_height_sin, freqs_width_sin], dim=-1).reshape(seq_lens, -1)

        return freqs_cos.clone().contiguous(), freqs_sin.clone().contiguous()


def apply_rotary_emb_neuron(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """
    Apply rotary embeddings without using complex numbers.

    This is a drop-in replacement for apply_rotary_emb_qwen that uses
    (cos, sin) tuples instead of complex tensors.

    The rotation is applied as:
        out[2k] = x[2k] * cos[k] - x[2k+1] * sin[k]
        out[2k+1] = x[2k] * sin[k] + x[2k+1] * cos[k]

    This is equivalent to complex multiplication:
        (x_real + i*x_imag) * (cos + i*sin) = (x_real*cos - x_imag*sin) + i*(x_real*sin + x_imag*cos)

    Args:
        x: Input tensor [B, S, H, D]
        freqs_cis: Tuple of (cos, sin) tensors, each [S, D//2]
        use_real: Always True for Neuron (we don't use complex)
        use_real_unbind_dim: Dimension for unbinding (-1 or -2)

    Returns:
        Tensor with rotary embeddings applied
    """
    cos, sin = freqs_cis

    # cos/sin have shape [S, D//2] where D is the head_dim
    # x has shape [B, S, H, D]

    # Expand cos/sin to match x's D dimension by interleaving
    # [c0, c1, ..., c31] -> [c0, c0, c1, c1, ..., c31, c31]
    # This uses repeat_interleave which is more compiler-friendly than stack+flatten
    cos = cos.repeat_interleave(2, dim=-1)  # [S, D]
    sin = sin.repeat_interleave(2, dim=-1)  # [S, D]

    # Expand dims for broadcasting: [S, D] -> [1, S, 1, D]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Move to same device as x
    cos = cos.to(x.device)
    sin = sin.to(x.device)

    # For use_real_unbind_dim == -1 (default for QwenImage)
    # x is stored as [x0_real, x0_imag, x1_real, x1_imag, ...]
    # x_rotated should be [-x0_imag, x0_real, -x1_imag, x1_real, ...]
    if use_real_unbind_dim == -1:
        # Reshape to separate real/imag pairs, then create rotated version
        # Use view instead of reshape for better tracing
        orig_shape = x.shape
        x_reshape = x.view(orig_shape[0], orig_shape[1], orig_shape[2], -1, 2)  # [B, S, H, D//2, 2]
        # Create rotated: [-imag, real] for each pair
        x_rotated = torch.cat([-x_reshape[..., 1:2], x_reshape[..., 0:1]], dim=-1)  # [B, S, H, D//2, 2]
        x_rotated = x_rotated.view(orig_shape)  # [B, S, H, D]

    elif use_real_unbind_dim == -2:
        # x is stored as [x0_real, x1_real, ..., x0_imag, x1_imag, ...]
        half_d = x.shape[-1] // 2
        x_real = x[..., :half_d]
        x_imag = x[..., half_d:]
        x_rotated = torch.cat([-x_imag, x_real], dim=-1)
    else:
        raise ValueError(f"use_real_unbind_dim={use_real_unbind_dim} but should be -1 or -2.")

    # Apply rotation: out = x * cos + x_rotated * sin
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out


def patch_qwenimage_rope(transformer):
    """
    Patch the QwenImage transformer to use Neuron-compatible RoPE.

    This replaces the complex-number based RoPE with sin/cos based implementation.
    """
    # Get original config
    orig_rope = transformer.pos_embed
    theta = orig_rope.theta
    axes_dim = orig_rope.axes_dim
    scale_rope = orig_rope.scale_rope

    print(f"  Original RoPE: theta={theta}, axes_dim={axes_dim}, scale_rope={scale_rope}")

    # Replace with Neuron-compatible version
    transformer.pos_embed = NeuronQwenEmbedRope(
        theta=theta,
        axes_dim=axes_dim,
        scale_rope=scale_rope
    )

    # Patch the apply_rotary_emb_qwen function to use our version
    import diffusers.models.transformers.transformer_qwenimage as qwen_module

    # Store original function
    if not hasattr(qwen_module, '_orig_apply_rotary_emb_qwen'):
        qwen_module._orig_apply_rotary_emb_qwen = qwen_module.apply_rotary_emb_qwen

    # Replace with neuron-compatible version
    qwen_module.apply_rotary_emb_qwen = apply_rotary_emb_neuron

    print("  Patched QwenImage transformer with Neuron-compatible RoPE (no complex numbers)")
    return transformer
