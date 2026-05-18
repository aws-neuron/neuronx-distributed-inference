# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LQ Projection module for FlashVSR on AWS Trainium.

The LQ Projection (Causal_LQ4x_Proj) processes low-quality video frames into
per-token conditioning residuals for the DiT. These residuals guide the
denoising process to preserve content from the LQ input.

Architecture:
  PixelShuffle3d(1,16,16) -> CausalConv3d(768->2048, stride=2) -> RMSNorm -> SiLU
  -> CausalConv3d(2048->3072, stride=2) -> RMSNorm -> SiLU -> Linear(3072->1536)

Compiled via torch_neuronx.trace() (single-pass, processes all frames at once).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# Supporting layers
# ===================================================================


class RMS_norm(nn.Module):
    """RMSNorm for convolutional features."""

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution (left-padded in temporal dimension)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding, mode="replicate")
        return super().forward(x)


class PixelShuffle3d(nn.Module):
    """3D pixel shuffle (spatial only, ff=1)."""

    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x):
        from einops import rearrange

        return rearrange(
            x,
            "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
            ff=self.ff,
            hh=self.hh,
            ww=self.ww,
        )


# ===================================================================
# Causal_LQ4x_Proj (CPU model for weight loading)
# ===================================================================


class Causal_LQ4x_Proj(nn.Module):
    """CPU-side LQ projection model (for weight loading and reference)."""

    def __init__(self, in_dim=3, out_dim=1536, layer_num=1):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num
        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)
        self.conv1 = CausalConv3d(
            in_dim * self.ff * self.hh * self.ww,
            self.hidden_dim1,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()
        self.conv2 = CausalConv3d(
            self.hidden_dim1,
            self.hidden_dim2,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)]
        )


# ===================================================================
# Neuron-traceable LQ Projection wrapper
# ===================================================================


class NeuronLQProj(nn.Module):
    """Neuron-traceable wrapper for Causal_LQ4x_Proj.

    Processes the full LQ video in a single forward pass (no streaming/cache).
    Compiled via torch_neuronx.trace() for fixed input shapes.
    """

    def __init__(self, lq_proj: Causal_LQ4x_Proj):
        """Wrap an existing Causal_LQ4x_Proj with loaded weights.

        Args:
            lq_proj: Loaded Causal_LQ4x_Proj instance
        """
        super().__init__()
        self.conv1 = lq_proj.conv1
        self.norm1 = lq_proj.norm1
        self.act1 = lq_proj.act1
        self.conv2 = lq_proj.conv2
        self.norm2 = lq_proj.norm2
        self.act2 = lq_proj.act2
        self.linear = lq_proj.linear_layers[0]  # layer_num=1 for FlashVSR-v1.1
        self.ps_hh = lq_proj.hh
        self.ps_ww = lq_proj.ww

    def pixel_shuffle_3d(self, x):
        """PixelShuffle3d without einops. ff=1, hh=16, ww=16."""
        B, C, F, H, W = x.shape
        hh, ww = self.ps_hh, self.ps_ww
        x = x.reshape(B, C, F, H // hh, hh, W // ww, ww)
        x = x.permute(0, 1, 4, 6, 2, 3, 5)
        x = x.reshape(B, C * hh * ww, F, H // hh, W // ww)
        return x

    def causal_conv3d_no_cache(self, conv, x):
        """Run CausalConv3d without cache -- full temporal padding."""
        padding = conv._padding
        x = F.pad(x, padding, mode="replicate")
        return F.conv3d(
            x,
            conv.weight,
            conv.bias,
            conv.stride,
            (0, 0, 0),
            conv.dilation,
            conv.groups,
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Process full LQ video in one pass.

        Args:
            video: (B, 3, T, H, W) full LQ video tensor, bf16

        Returns:
            (B, S, 1536) LQ conditioning tokens for all DiT chunks
        """
        # Prepend 3 copies of first frame (causal warmup)
        first_frame = video[:, :, :1, :, :].expand(-1, -1, 3, -1, -1)
        x = torch.cat([first_frame, video], dim=2)

        # PixelShuffle3d (spatial pixel unshuffle)
        x = self.pixel_shuffle_3d(x)

        # CausalConv3d block 1
        x = self.causal_conv3d_no_cache(self.conv1, x)
        x = self.norm1(x)
        x = self.act1(x)

        # CausalConv3d block 2
        x = self.causal_conv3d_no_cache(self.conv2, x)
        x = self.norm2(x)
        x = self.act2(x)

        # Skip first temporal frame (warmup equivalent)
        B, C, F_out, H_out, W_out = x.shape
        x = x[:, :, 1:, :, :]
        F_out = F_out - 1

        # Flatten spatial and project to model dim
        x = x.permute(0, 2, 3, 4, 1)  # (B, F, H, W, C)
        x = x.reshape(B, F_out * H_out * W_out, C)
        out = self.linear(x)

        return out
