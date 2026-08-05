# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cosmos3 diffusion pipeline utilities: patchify/unpatchify, position IDs,
# tokenization helpers, and the denoising loop.

import json
import logging
import time
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Patchify / Unpatchify
# =============================================================================


def patchify(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    Convert VAE latents to patch tokens for the transformer.

    Cosmos3 uses patch_spatial=2: each 2x2 spatial block of the latent is
    flattened into a single token with spatial-first, channels-last ordering.

    Matches the reference einsum: "cthpwq->thwpqc"

    Args:
        latents: [B, C, T, H, W] - VAE latent space (C=48, T=1 for images)
        patch_size: spatial patch size (default 2)

    Returns:
        patches: [B, N_patches, patch_channels]
                 N_patches = T * (H // patch_size) * (W // patch_size)
                 patch_channels = patch_size * patch_size * C
    """
    B, C, T, H, W = latents.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"Latent spatial dims ({H}, {W}) must be divisible by patch_size={patch_size}"
    )

    pH = H // patch_size
    pW = W // patch_size

    # Reshape: [B, C, T, H, W] -> [B, C, T, pH, ps, pW, ps]
    latents = latents.reshape(B, C, T, pH, patch_size, pW, patch_size)
    # Permute to match reference "cthpwq->thwpqc": [B, T, pH, pW, ps_h, ps_w, C]
    latents = latents.permute(0, 2, 3, 5, 4, 6, 1)
    # Flatten patches: [B, T*pH*pW, ps*ps*C]
    patches = latents.reshape(B, T * pH * pW, C * patch_size * patch_size)

    return patches


def unpatchify(
    patches: torch.Tensor,
    T: int,
    H: int,
    W: int,
    channels: int = 48,
    patch_size: int = 2,
) -> torch.Tensor:
    """
    Convert patch tokens back to VAE latent space.

    Matches the reference einsum: "thwpqc->cthpwq"

    Args:
        patches: [B, N_patches, patch_channels] - velocity prediction
        T: temporal dimension of latent
        H: height of latent (before patching)
        W: width of latent (before patching)
        channels: number of latent channels (48)
        patch_size: spatial patch size (2)

    Returns:
        latents: [B, C, T, H, W]
    """
    B = patches.shape[0]
    pH = H // patch_size
    pW = W // patch_size

    # Reshape: [B, T*pH*pW, ps*ps*C] -> [B, T, pH, pW, ps_h, ps_w, C]
    patches = patches.reshape(B, T, pH, pW, patch_size, patch_size, channels)
    # Permute back (inverse of "cthpwq->thwpqc"): [B, C, T, pH, ps_h, pW, ps_w]
    latents = patches.permute(0, 6, 1, 2, 4, 3, 5)
    # Flatten spatial: [B, C, T, H, W]
    latents = latents.reshape(B, channels, T, H, W)

    return latents


# =============================================================================
# Position IDs for M-RoPE
# =============================================================================


def build_position_ids(
    max_text_len: int,
    actual_text_len: int,
    T: int,
    pH: int,
    pW: int,
    temporal_margin: int = 15000,
) -> torch.Tensor:
    """
    Build M-RoPE 3D position IDs for text + vision tokens.

    Text tokens: all 3 axes (T, H, W) share the same incrementing IDs [0..max_text_len-1].
    Vision tokens: temporal = actual_text_len + temporal_margin, H=[0..pH-1], W=[0..pW-1].

    The temporal_margin (15000) separates text and vision position spaces, matching
    the reference `unified_3d_mrope_temporal_modality_margin` parameter.

    Args:
        max_text_len: padded text sequence length (model input dimension)
        actual_text_len: actual number of meaningful text tokens (for temporal offset)
        T: temporal patches (1 for images)
        pH: spatial height patches
        pW: spatial width patches
        temporal_margin: gap between text and vision temporal positions (default 15000)

    Returns:
        position_ids: [max_text_len + T*pH*pW, 3] - (t, h, w) per token
    """
    vision_t_offset = actual_text_len + temporal_margin

    # Text: all 3 axes share same incrementing IDs
    text_ids = torch.arange(max_text_len, dtype=torch.long)
    text_pos = text_ids.unsqueeze(1).expand(-1, 3)  # [max_text_len, 3]

    # Vision: 3D grid via meshgrid
    t_coords = torch.arange(T, dtype=torch.long) + vision_t_offset
    h_coords = torch.arange(pH, dtype=torch.long)
    w_coords = torch.arange(pW, dtype=torch.long)
    grid_t, grid_h, grid_w = torch.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    vis_pos = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=1)

    return torch.cat([text_pos, vis_pos], dim=0)


def generate_position_ids(text_len: int, T: int, pH: int, pW: int) -> torch.Tensor:
    """Simple position ID generation (text_len used as both max and actual)."""
    return build_position_ids(text_len, text_len, T, pH, pW)


# =============================================================================
# Tokenization (matching reference Cosmos3OmniPipeline.tokenize_prompt)
# =============================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate images from a give prompt."
)
RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."
NEGATIVE_RESOLUTION_TEMPLATE = "This image is not of {height}x{width} resolution."


def tokenize_prompt(
    tokenizer,
    prompt: str,
    height: int = 512,
    width: int = 512,
    max_len: int = 256,
    negative: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Tokenize a prompt following the Cosmos3 reference pipeline format.

    The reference pipeline uses:
    1. System prompt: "You are a helpful assistant who will generate images..."
    2. User content: prompt + resolution template (or inverse for negative)
    3. apply_chat_template with add_generation_prompt=True
    4. Append eos_token_id + <|vision_start|> token

    Args:
        tokenizer: Qwen2 tokenizer (from the model)
        prompt: text prompt (or empty for negative)
        height: image height in pixels
        width: image width in pixels
        max_len: maximum token length (padded)
        negative: if True, use inverse resolution template

    Returns:
        (padded_ids, actual_len): padded token tensor [1, max_len] and actual length
    """
    if negative:
        user_text = NEGATIVE_RESOLUTION_TEMPLATE.format(height=height, width=width)
    else:
        user_text = (
            prompt.rstrip(".")
            + ". "
            + RESOLUTION_TEMPLATE.format(height=height, width=width)
        )

    conversations = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    result = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=True,
    )

    eos_id = tokenizer.eos_token_id
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    input_ids = list(result["input_ids"]) + [eos_id, vision_start_id]
    actual_len = len(input_ids)

    # Pad
    pad_id = tokenizer.pad_token_id or 0
    if actual_len > max_len:
        input_ids = input_ids[:max_len]
        actual_len = max_len
    else:
        input_ids = input_ids + [pad_id] * (max_len - actual_len)

    return torch.tensor([input_ids], dtype=torch.long), actual_len


# =============================================================================
# Denoising Loop
# =============================================================================


def denoise(
    backbone,
    cond_ids: torch.Tensor,
    uncond_ids: torch.Tensor,
    cond_pos: torch.Tensor,
    uncond_pos: torch.Tensor,
    scheduler,
    latents: torch.Tensor,
    num_steps: int = 35,
    cfg_scale: float = 6.0,
    latent_channels: int = 48,
    patch_size: int = 2,
    cfg_parallel: bool = False,
) -> torch.Tensor:
    """
    Run the denoising loop with CFG.

    This is the optimized inner loop that:
    - Pre-computes timestep tensors
    - Uses return_dict=False for scheduler
    - Keeps CFG math in bf16
    - Optionally uses CFG-parallel (batch=2 single call)

    Args:
        backbone: compiled NeuronCosmos3BackboneApplication
        cond_ids: [1, max_text_len] - conditional token IDs
        uncond_ids: [1, max_text_len] - unconditional token IDs
        cond_pos: [total_seq, 3] - conditional position IDs
        uncond_pos: [total_seq, 3] - unconditional position IDs
        scheduler: UniPCMultistepScheduler (already configured)
        latents: [1, C, T, H, W] - initial noisy latents (float32)
        num_steps: denoising steps
        cfg_scale: classifier-free guidance scale
        latent_channels: number of VAE latent channels (48)
        patch_size: spatial patch size (2)
        cfg_parallel: if True, pack cond+uncond into batch=2 single call

    Returns:
        latents: [1, C, T, H, W] - denoised latents (float32)
    """
    _, _, T, H_latent, W_latent = latents.shape

    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    latents = latents * scheduler.init_noise_sigma

    # Pre-compute timestep tensors
    if cfg_parallel:
        ts_tensors = [
            torch.tensor([t.item() * 0.001, t.item() * 0.001], dtype=torch.bfloat16)
            for t in timesteps
        ]
        # Pack text IDs into batch=2
        text_ids_batch = torch.cat([cond_ids, uncond_ids], dim=0)  # [2, max_text_len]
    else:
        ts_tensors = [
            torch.tensor([t.item() * 0.001], dtype=torch.bfloat16) for t in timesteps
        ]

    start = time.time()
    for i, t_val in enumerate(timesteps):
        vis_patches = patchify(latents.to(torch.bfloat16), patch_size=patch_size)

        if cfg_parallel:
            # Single call with batch=2: [cond_patches, uncond_patches]
            vis_batch = vis_patches.expand(2, -1, -1).contiguous()
            output = backbone(text_ids_batch, vis_batch, ts_tensors[i], cond_pos)
            v_cond = output[0:1]
            v_uncond = output[1:2]
        else:
            v_cond = backbone(cond_ids, vis_patches, ts_tensors[i], cond_pos)
            v_uncond = backbone(uncond_ids, vis_patches, ts_tensors[i], uncond_pos)

        velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
        vel_latent = unpatchify(
            velocity.float(),
            T,
            H_latent,
            W_latent,
            channels=latent_channels,
            patch_size=patch_size,
        )

        latents = scheduler.step(vel_latent, t_val, latents, return_dict=False)[0]

        if i < 2 or i == num_steps - 1 or (i + 1) % 10 == 0:
            logger.info(
                f"  Step {i + 1}/{num_steps}: v_norm={vel_latent.norm():.1f}, "
                f"lat_norm={latents.norm():.1f}"
            )

    elapsed = time.time() - start
    mode_str = "CFG-parallel" if cfg_parallel else "sequential"
    logger.info(
        f"  Denoising ({mode_str}): {elapsed:.2f}s ({elapsed / num_steps * 1000:.1f}ms/step)"
    )

    return latents


def denormalize_latents(latents: torch.Tensor, vae_config_path: str) -> torch.Tensor:
    """
    Denormalize latents from scheduler space to VAE space.

    The denoising operates in a normalized space (zero mean, unit variance).
    The VAE expects native latent space. Transform: latents = latents * std + mean.

    Args:
        latents: [1, 48, T, H, W] - denoised latents
        vae_config_path: path to vae/config.json

    Returns:
        denormalized latents ready for VAE decoding
    """
    with open(vae_config_path) as f:
        vae_cfg = json.load(f)

    lat_mean = torch.tensor(vae_cfg["latents_mean"]).view(1, 48, 1, 1, 1).float()
    lat_std = torch.tensor(vae_cfg["latents_std"]).view(1, 48, 1, 1, 1).float()

    return latents.float() * lat_std + lat_mean
