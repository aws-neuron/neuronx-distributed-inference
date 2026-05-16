"""
RoPE pre-computation for LongCat-Image-Edit (FLUX-style 3-axis RoPE).

LongCat uses FLUX-style RoPE which is ALREADY real-valued (cos, sin) --
no complex number workaround is needed (unlike Qwen reference).

3-axis decomposition:
  - modality: 16 dims (text=0, target=1, source=2)
  - row: 56 dims (spatial height position)
  - col: 56 dims (spatial width position)
  Total: 128 dims = head_dim

IMPORTANT: The pipeline does 2x2 spatial packing (_pack_latents), so the
effective patch grid is (latent_h//2) x (latent_w//2). Image row/col
positions are offset by text_seq_len (matching prepare_pos_ids).

This module pre-computes RoPE tensors at compile time and saves them as
rope_cache.pt for loading at inference time.
"""

import torch
import math
from typing import Tuple


def compute_rope_from_model(
    pipe,
    height: int,
    width: int,
    text_seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute RoPE using the model's own pos_embed, with the exact same position
    IDs that the pipeline creates at runtime.

    This is the preferred method as it guarantees exact match with inference.

    Args:
        pipe: LongCatImageEditPipeline instance
        height: Image height in pixels
        width: Image width in pixels
        text_seq_len: Text sequence length (prompt_embeds_length)
        dtype: Output dtype

    Returns:
        (text_cos, text_sin, img_cos, img_sin) each [S, 128]
    """
    from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import prepare_pos_ids

    vae_scale_factor = 8
    # Match pipeline's height/width calculation
    latent_h = 2 * (height // (vae_scale_factor * 2))  # = height // 8
    latent_w = 2 * (width // (vae_scale_factor * 2))    # = width // 8
    patch_h = latent_h // 2  # After 2x2 packing
    patch_w = latent_w // 2

    print(f"  RoPE computation (from model):")
    print(f"    Image: {height}x{width}, Latent: {latent_h}x{latent_w}")
    print(f"    Patch grid: {patch_h}x{patch_w} = {patch_h * patch_w} per image")
    print(f"    Text seq len: {text_seq_len}")

    # Create the same position IDs as the pipeline
    text_ids = prepare_pos_ids(
        modality_id=0, type="text", num_token=text_seq_len
    )

    target_ids = prepare_pos_ids(
        modality_id=1, type="image",
        start=(text_seq_len, text_seq_len),
        height=patch_h, width=patch_w,
    )

    source_ids = prepare_pos_ids(
        modality_id=2, type="image",
        start=(text_seq_len, text_seq_len),
        height=patch_h, width=patch_w,
    )

    # Combine image IDs (target + source)
    img_ids = torch.cat([target_ids, source_ids], dim=0)

    # Compute RoPE using model's pos_embed
    pos_embed = pipe.transformer.pos_embed

    # Text RoPE
    txt_cos, txt_sin = pos_embed(text_ids)
    print(f"    txt_cos: {txt_cos.shape}, txt_sin: {txt_sin.shape}")

    # Image RoPE
    img_cos, img_sin = pos_embed(img_ids)
    print(f"    img_cos: {img_cos.shape}, img_sin: {img_sin.shape}")

    return (
        txt_cos.to(dtype),
        txt_sin.to(dtype),
        img_cos.to(dtype),
        img_sin.to(dtype),
    )


def precompute_rope_for_longcat(
    height: int,
    width: int,
    text_seq_len: int,
    theta: int = 10000,
    axes_dim: Tuple[int, ...] = (16, 56, 56),
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pre-compute RoPE (cos, sin) tensors for LongCat transformer.
    Manual fallback when pipeline is not available.

    Matches LongCatImagePosEmbed.forward() + prepare_pos_ids exactly:
    - Uses get_1d_rotary_pos_embed with repeat_interleave_real=True, use_real=True
    - Output cos/sin are [S, head_dim] (128), NOT [S, head_dim//2]
    - Each axis contributes its full dim to the output (16+56+56=128)
    - Image positions are OFFSET by text_seq_len (matching prepare_pos_ids)
    - Patch grid uses 2x2 packing: (latent_h//2) x (latent_w//2)

    For compilation, we separate into txt and img RoPE and concatenate at runtime.
    """
    vae_scale_factor = 8
    latent_h = 2 * (height // (vae_scale_factor * 2))
    latent_w = 2 * (width // (vae_scale_factor * 2))
    # After 2x2 FLUX packing
    patch_h = latent_h // 2
    patch_w = latent_w // 2
    num_patches = patch_h * patch_w

    # Create position grids for image patches (OFFSET by text_seq_len)
    rows = torch.arange(patch_h).float() + text_seq_len
    cols = torch.arange(patch_w).float() + text_seq_len
    grid_h, grid_w = torch.meshgrid(rows, cols, indexing="ij")
    grid_h = grid_h.reshape(-1)
    grid_w = grid_w.reshape(-1)

    def get_1d_rope(positions, dim, repeat_interleave=True):
        """Match diffusers' get_1d_rotary_pos_embed with repeat_interleave_real=True."""
        # Use float64 for frequency computation (matches diffusers)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).double() / dim))
        angles = torch.outer(positions.double(), freqs)  # [S, dim//2]
        cos = torch.cos(angles).float()
        sin = torch.sin(angles).float()
        if repeat_interleave:
            # repeat_interleave_real=True: [c0,c0,c1,c1,...] -> [S, dim]
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        return cos, sin  # [S, dim]

    # ---- Text RoPE: modality=0, row=[0..txt_len-1], col=[0..txt_len-1] ----
    text_positions = torch.arange(text_seq_len).float()
    text_modality = torch.zeros(text_seq_len).float()

    t_mod_cos, t_mod_sin = get_1d_rope(text_modality, axes_dim[0])
    t_row_cos, t_row_sin = get_1d_rope(text_positions, axes_dim[1])
    t_col_cos, t_col_sin = get_1d_rope(text_positions, axes_dim[2])

    text_cos = torch.cat([t_mod_cos, t_row_cos, t_col_cos], dim=-1)  # [txt_seq, 128]
    text_sin = torch.cat([t_mod_sin, t_row_sin, t_col_sin], dim=-1)

    # ---- Target image RoPE: modality=1, positions offset by text_seq_len ----
    tgt_modality = torch.ones(num_patches).float()
    tgt_mod_cos, tgt_mod_sin = get_1d_rope(tgt_modality, axes_dim[0])
    tgt_row_cos, tgt_row_sin = get_1d_rope(grid_h, axes_dim[1])
    tgt_col_cos, tgt_col_sin = get_1d_rope(grid_w, axes_dim[2])

    tgt_cos = torch.cat([tgt_mod_cos, tgt_row_cos, tgt_col_cos], dim=-1)  # [patches, 128]
    tgt_sin = torch.cat([tgt_mod_sin, tgt_row_sin, tgt_col_sin], dim=-1)

    # ---- Source image RoPE: modality=2, same positions as target ----
    src_modality = torch.full((num_patches,), 2.0)
    src_mod_cos, src_mod_sin = get_1d_rope(src_modality, axes_dim[0])
    src_row_cos, src_row_sin = get_1d_rope(grid_h, axes_dim[1])
    src_col_cos, src_col_sin = get_1d_rope(grid_w, axes_dim[2])

    src_cos = torch.cat([src_mod_cos, src_row_cos, src_col_cos], dim=-1)
    src_sin = torch.cat([src_mod_sin, src_row_sin, src_col_sin], dim=-1)

    # Image = target + source
    img_cos = torch.cat([tgt_cos, src_cos], dim=0)  # [2*patches, 128]
    img_sin = torch.cat([tgt_sin, src_sin], dim=0)

    return (
        text_cos.to(dtype),
        text_sin.to(dtype),
        img_cos.to(dtype),
        img_sin.to(dtype),
    )
