# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TCDecoder (TAEHV) for FlashVSR on AWS Trainium.

The TCDecoder converts latent video representations to RGB frames.
It uses temporal recurrence via MemBlock layers for inter-frame coherence.

Two execution modes:
  - Sequential (production): Process one latent frame at a time with explicit
    MemBlock state carried between calls. Deep temporal context, best quality.
  - Multi-frame: Process two latent frames per call. 1.24x faster decode.

Compiled via torch_neuronx.trace() (not NxDI ModelBuilder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import namedtuple
from typing import Optional, List
from einops import rearrange


DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


# ===================================================================
# Core TCDecoder layers
# ===================================================================


class IdentityConv2d(nn.Conv2d):
    """Conv2d initialized to identity (Dirac delta)."""

    def __init__(self, C, kernel_size=3, bias=False):
        pad = kernel_size // 2
        super().__init__(C, C, kernel_size, padding=pad, bias=bias)
        with torch.no_grad():
            init.dirac_(self.weight)
            if self.bias is not None:
                self.bias.zero_()


def conv2d_3x3(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Temporal memory block -- concatenates current frame with previous frame."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv2d_3x3(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            conv2d_3x3(n_out, n_out),
            nn.ReLU(inplace=True),
            conv2d_3x3(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    """Temporal pooling (reduces temporal dimension by stride)."""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    """Temporal growth (increases temporal dimension by stride)."""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


class TCPixelShuffle3d(nn.Module):
    """PixelShuffle3d variant for TCDecoder (pads temporal dim if needed)."""

    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x):
        B, C, F, H, W = x.shape
        if F % self.ff != 0:
            first_frame = x[:, :, 0:1, :, :].repeat(1, 1, self.ff - F % self.ff, 1, 1)
            x = torch.cat([first_frame, x], dim=2)
        return rearrange(
            x,
            "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
            ff=self.ff,
            hh=self.hh,
            ww=self.ww,
        ).transpose(1, 2)


# ===================================================================
# TAEHV decoder model
# ===================================================================


class TAEHV(nn.Module):
    """Temporal Autoencoder with Hierarchical Video decoder."""

    image_channels = 3

    def __init__(
        self,
        checkpoint_path=None,
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
        channels=[256, 128, 64, 64],
        latent_channels=16,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        n_f = channels
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1

        base_decoder = nn.Sequential(
            Clamp(),
            conv2d_3x3(self.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv2d_3x3(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv2d_3x3(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv2d_3x3(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            conv2d_3x3(n_f[3], TAEHV.image_channels),
        )
        self.decoder = self._apply_identity_deepen(base_decoder, how_many_each=1, k=3)
        self.pixel_shuffle = TCPixelShuffle3d(4, 8, 8)

        if checkpoint_path is not None:
            self.load_state_dict(
                self.patch_tgrow_layers(
                    torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                ),
                strict=False,
            )
        self.mem = [None] * len(self.decoder)

    @staticmethod
    def _apply_identity_deepen(decoder, how_many_each=1, k=3):
        new_layers = []
        for b in decoder:
            new_layers.append(b)
            if isinstance(b, nn.ReLU):
                C = None
                if len(new_layers) >= 2 and isinstance(new_layers[-2], nn.Conv2d):
                    C = new_layers[-2].out_channels
                elif len(new_layers) >= 2 and isinstance(new_layers[-2], MemBlock):
                    C = new_layers[-2].conv[-1].out_channels
                if C is not None:
                    for _ in range(how_many_each):
                        new_layers.append(IdentityConv2d(C, kernel_size=k, bias=False))
                        new_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*new_layers)

    def patch_tgrow_layers(self, sd):
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    def clean_mem(self):
        self.mem = [None] * len(self.decoder)


def build_tcdecoder(
    new_channels=[512, 256, 128, 128],
    device="cpu",
    dtype=torch.bfloat16,
    new_latent_channels=None,
):
    """Build a TAEHV TCDecoder instance."""
    latent_ch = new_latent_channels if new_latent_channels is not None else 16
    big = (
        TAEHV(
            checkpoint_path=None,
            channels=new_channels,
            latent_channels=latent_ch,
        )
        .to(device)
        .to(dtype)
        .train()
    )
    big.clean_mem()
    return big


# ===================================================================
# Neuron-traceable sequential TCDecoder wrapper
# ===================================================================


def patch_inplace_relu(module: nn.Module):
    """Recursively replace ReLU(inplace=True) with ReLU(inplace=False)."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            patch_inplace_relu(child)
    if isinstance(module, nn.Sequential):
        for i, layer in enumerate(module):
            if isinstance(layer, nn.ReLU) and layer.inplace:
                module[i] = nn.ReLU(inplace=False)
            else:
                patch_inplace_relu(layer)


class NeuronTCDecoderSequential(nn.Module):
    """Neuron-traceable wrapper for sequential TCDecoder execution.

    Processes one latent frame per call with explicit MemBlock state I/O.
    Each forward: (x, mem_0..mem_8) -> (4_rgb_frames, new_mem_0..new_mem_8)
    """

    def __init__(self, taehv: TAEHV):
        super().__init__()
        self.decoder = taehv.decoder
        patch_inplace_relu(self.decoder)

        # Pre-analyze layer types
        self.layer_types = []
        for layer in self.decoder:
            if isinstance(layer, MemBlock):
                self.layer_types.append("memblock")
            elif isinstance(layer, TGrow):
                self.layer_types.append("tgrow")
            elif isinstance(layer, TPool):
                self.layer_types.append("tpool")
            else:
                self.layer_types.append("standard")

    def forward(self, x: torch.Tensor, *mem_states) -> tuple:
        """Process one latent frame sequentially.

        Args:
            x: (1, C, H, W) single latent frame
            *mem_states: 9 MemBlock state tensors

        Returns:
            Tuple of (output_frames, new_mem_0, ..., new_mem_8)
            output_frames: (4, 3, H_out, W_out) RGB frames
        """
        mem_list = list(mem_states)
        mem_idx = 0

        for i, layer in enumerate(self.decoder):
            lt = self.layer_types[i]
            if lt == "memblock":
                past = mem_list[mem_idx]
                mem_list[mem_idx] = x.clone()
                x = layer(x, past)
                mem_idx += 1
            elif lt == "tgrow":
                x = layer(x)
                # TGrow with stride=2: output is (2, C, H, W) from (1, C, H, W)
            else:
                x = layer(x)

        return (x, *mem_list)


# ===================================================================
# Sequential decode function
# ===================================================================


def neuron_decode_video_sequential(
    traced_tcdecoder,
    latents: torch.Tensor,
    cond: torch.Tensor,
    pixel_shuffle_fn,
    frames_to_trim: int = 3,
) -> torch.Tensor:
    """Decode video using sequential-mode traced Neuron TCDecoder.

    Processes one latent frame at a time with explicit MemBlock state.

    Args:
        traced_tcdecoder: torch.jit.ScriptModule
        latents: (N, T, C, H, W) latent tensor
        cond: (N, C_cond, T_cond, H_cond, W_cond) LQ conditioning
        pixel_shuffle_fn: TCPixelShuffle3d module
        frames_to_trim: frames to trim from start (default 3)

    Returns:
        (N, 3, T_out, H_out, W_out) decoded RGB video
    """
    N = latents.shape[0]
    assert N == 1, "TCDecoder only supports batch_size=1"

    # Pixel shuffle conditioning and concatenate
    cond_shuffled = pixel_shuffle_fn(cond)
    x = torch.cat([cond_shuffled, latents], dim=2)

    T_total = x.shape[1]
    H_lat = x.shape[3]
    W_lat = x.shape[4]
    x_4d = x.reshape(N * T_total, x.shape[2], H_lat, W_lat)

    # Initialize MemBlock states
    state_dtype = x_4d.dtype
    mem_states = [
        torch.zeros(1, 512, H_lat, W_lat, dtype=state_dtype),
        torch.zeros(1, 512, H_lat, W_lat, dtype=state_dtype),
        torch.zeros(1, 512, H_lat, W_lat, dtype=state_dtype),
        torch.zeros(1, 256, H_lat * 2, W_lat * 2, dtype=state_dtype),
        torch.zeros(1, 256, H_lat * 2, W_lat * 2, dtype=state_dtype),
        torch.zeros(1, 256, H_lat * 2, W_lat * 2, dtype=state_dtype),
        torch.zeros(1, 128, H_lat * 4, W_lat * 4, dtype=state_dtype),
        torch.zeros(1, 128, H_lat * 4, W_lat * 4, dtype=state_dtype),
        torch.zeros(1, 128, H_lat * 4, W_lat * 4, dtype=state_dtype),
    ]

    # Process each frame sequentially
    outputs = []
    for t in range(T_total):
        xt = x_4d[t : t + 1]
        with torch.no_grad():
            result = traced_tcdecoder(xt, *mem_states)
        frames_t = result[0]
        mem_states = list(result[1:])
        outputs.append(frames_t)

    # Concatenate and reshape
    all_frames = torch.cat(outputs, dim=0)
    T_out = all_frames.shape[0]
    result = all_frames.reshape(N, T_out, *all_frames.shape[1:])
    result = result[:, frames_to_trim:]
    result = result.transpose(1, 2)  # NTCHW -> NCTHW

    return result
