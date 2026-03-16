"""
NxDI LTX-2.3 VAE Decoder — Tensor Parallel Model
==================================================
Tensor-parallel VAE decoder for the LTX-2.3 video diffusion model (ltx-core).

Adapted from the LTX-2 (Diffusers) TP VAE decoder. Key differences from LTX-2:
  - ltx-core uses a flat `up_blocks` list (interleaved UNetMidBlock3D + DepthToSpaceUpsample)
    instead of Diffusers' hierarchical (mid_block + up_blocks with sub-upsamplers)
  - LTX-2.3 has timestep conditioning: AdaLN in each ResnetBlock3D, time embedders
    in each UNetMidBlock3D, and a final AdaLN after norm_out
  - Normalization is PixelNorm (same math as PerChannelRMSNorm)
  - CausalConv3d (ltx-core) ≡ LTX2VideoCausalConv3d (Diffusers)

Channel progression: 128 -> 1024 -> 512 -> 256 -> 128 -> 48 -> 3

Compilation boundary (same as LTX-2):
  - H_latent × W_latent ≤ 64 elements (SRAM limit)
  - 4×16 = 64 ✓ (optimal for wide outputs like 1024×1536)
  - 8×8 = 64 ✓

Pre-processing (done on CPU, OUTSIDE the compiled graph):
  - Noise injection: sample = noise * 0.025 + sample * 0.975
  - Denormalization: per_channel_statistics.un_normalize(sample)
  - Timestep embedding computation

The compiled graph takes:
  - latent_tile: [1, 128, T, H_tile, W_tile] (denormalized)
  - timestep_embed_tile: [1, C_embed, 1, 1, 1] (precomputed on CPU)
  - last_ada_values: [1, 2, C_final, 1, 1, 1] (precomputed on CPU)
"""

import os
from functools import partial

import torch
import torch.nn as nn

os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")

COMPILER_FLAGS = (
    "--model-type=unet-inference -O1 --auto-cast none "
    "--enable-fast-loading-neuron-binaries"
)


def get_sharded_data(data, dim):
    """Get shard for current TP rank along given dimension."""
    from neuronx_distributed.parallel_layers import parallel_state

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()
    shard_size = data.shape[dim] // tp_size
    if dim == 0:
        return data[shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()
    else:
        raise ValueError(f"Unsupported dim: {dim}")


# ── Temporal padding helper ─────────────────────────────────────────


def make_noncausal_pad_fn(conv_module):
    """Create non-causal temporal padding function for CausalConv3d.

    ltx-core CausalConv3d stores kernel_size as tuple; non-causal mode pads
    (k-1)//2 frames of first/last frame on each side.
    """
    # CausalConv3d stores time_kernel_size attribute
    kernel_t = getattr(conv_module, "time_kernel_size", 3)
    if hasattr(conv_module, "kernel_size") and isinstance(
        conv_module.kernel_size, tuple
    ):
        kernel_t = conv_module.kernel_size[0]
    pad_t = (kernel_t - 1) // 2  # 1 for kernel=3

    def pad_fn(x):
        if pad_t > 0:
            pad_left = x[:, :, :1].repeat(1, 1, pad_t, 1, 1)
            pad_right = x[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
            x = torch.cat([pad_left, x, pad_right], dim=2)
        return x

    return pad_fn


def make_noncausal_pad_fn_from_kernel(kernel_t):
    """Create non-causal temporal padding from explicit kernel size."""
    pad_t = (kernel_t - 1) // 2

    def pad_fn(x):
        if pad_t > 0:
            pad_left = x[:, :, :1].repeat(1, 1, pad_t, 1, 1)
            pad_right = x[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
            x = torch.cat([pad_left, x, pad_right], dim=2)
        return x

    return pad_fn


# ── Parallel Conv3d layers ───────────────────────────────────────────


class ColumnParallelConv3d(nn.Module):
    """Conv3d with output channels sharded across TP ranks.
    Input: full channels. Output: sharded channels."""

    def __init__(self, inner_conv, tp_degree):
        super().__init__()
        self.sharded_out = inner_conv.out_channels // tp_degree
        self.conv = nn.Conv3d(
            inner_conv.in_channels,
            self.sharded_out,
            kernel_size=inner_conv.kernel_size,
            stride=inner_conv.stride,
            padding=inner_conv.padding,
            padding_mode=inner_conv.padding_mode,
            bias=inner_conv.bias is not None,
        )
        self.conv.weight.data = get_sharded_data(inner_conv.weight.data, 0)
        if inner_conv.bias is not None:
            self.conv.bias.data = get_sharded_data(inner_conv.bias.data, 0)

    def forward(self, x):
        return self.conv(x)


class RowParallelConv3d(nn.Module):
    """Conv3d with input channels sharded. Output is all-reduced (full channels).
    Input: sharded channels. Output: full channels."""

    def __init__(self, inner_conv, tp_degree):
        super().__init__()
        from neuronx_distributed.parallel_layers.mappings import (
            reduce_from_tensor_model_parallel_region,
        )

        self.reduce = reduce_from_tensor_model_parallel_region
        self.sharded_in = inner_conv.in_channels // tp_degree
        self.conv = nn.Conv3d(
            self.sharded_in,
            inner_conv.out_channels,
            kernel_size=inner_conv.kernel_size,
            stride=inner_conv.stride,
            padding=inner_conv.padding,
            padding_mode=inner_conv.padding_mode,
            bias=inner_conv.bias is not None,
        )
        self.conv.weight.data = get_sharded_data(inner_conv.weight.data, 1)
        if inner_conv.bias is not None:
            self.conv.bias.data = inner_conv.bias.data.clone() / tp_degree

    def forward(self, x):
        return self.reduce(self.conv(x))


class ColumnRowParallelConv3d(nn.Module):
    """Conv3d with sharded input -> sharded output.

    All-gathers input channels, then applies column-parallel conv
    (full in -> sharded out). 1 all-gather per forward.
    """

    def __init__(self, inner_conv, tp_degree):
        super().__init__()
        from neuronx_distributed.parallel_layers.mappings import (
            gather_from_tensor_model_parallel_region_with_dim,
        )

        self.gather_channels = gather_from_tensor_model_parallel_region_with_dim
        self.sharded_out = inner_conv.out_channels // tp_degree
        self.conv = nn.Conv3d(
            inner_conv.in_channels,  # full input
            self.sharded_out,  # sharded output
            kernel_size=inner_conv.kernel_size,
            stride=inner_conv.stride,
            padding=inner_conv.padding,
            padding_mode=inner_conv.padding_mode,
            bias=inner_conv.bias is not None,
        )
        self.conv.weight.data = get_sharded_data(inner_conv.weight.data, 0)
        if inner_conv.bias is not None:
            self.conv.bias.data = get_sharded_data(inner_conv.bias.data, 0)

    def forward(self, x):
        x_full = self.gather_channels(x, gather_dim=1)
        return self.conv(x_full)


# ── Sharded RMSNorm (PixelNorm) ────────────────────────────────────


class ShardedPixelNorm(nn.Module):
    """PixelNorm that works with sharded channel dimension.

    PixelNorm computes x / sqrt(mean(x^2, dim=1) + eps).
    With sharded channels, we need global mean across all ranks.

    Strategy:
      1. Compute local sum of x^2 over local channels
      2. All-reduce (sum) across ranks for global sum
      3. Divide by total channels for global mean
      4. x / sqrt(global_mean + eps)
    """

    def __init__(self, original_norm, tp_degree):
        super().__init__()
        from neuronx_distributed.parallel_layers.mappings import (
            reduce_from_tensor_model_parallel_region,
        )

        self.reduce = reduce_from_tensor_model_parallel_region
        self.tp_degree = tp_degree
        self.eps = getattr(original_norm, "eps", 1e-8)

    def forward(self, x):
        local_sq_sum = (x**2).sum(dim=1, keepdim=True)
        global_sq_sum = self.reduce(local_sq_sum)
        n_channels = x.shape[1] * self.tp_degree
        rms = torch.sqrt(global_sq_sum / n_channels + self.eps)
        return x / rms


# ── Sharded ResNet Block ────────────────────────────────────────────


class ShardedResnetBlock3D(nn.Module):
    """LTX-2.3 ResnetBlock3D with sharded channels and timestep conditioning.

    Forward order (from ltx-core source):
      x = norm1(x)
      if timestep_conditioning: x = x * (1 + scale1) + shift1
      x = SiLU(x) -> conv1
      x = norm2(x)
      if timestep_conditioning: x = x * (1 + scale2) + shift2
      x = SiLU(x) -> dropout -> conv2
      residual = norm3(input) -> conv_shortcut
      return x + residual

    When in_channels == out_channels: conv_shortcut = Identity, norm3 = Identity.
    """

    def __init__(self, original_block, tp_degree):
        super().__init__()
        self.nonlinearity = nn.SiLU()
        self.timestep_conditioning = original_block.timestep_conditioning

        # Norms — sharded PixelNorm
        self.norm1 = ShardedPixelNorm(original_block.norm1, tp_degree)
        self.norm2 = ShardedPixelNorm(original_block.norm2, tp_degree)

        # Extract inner Conv3d from CausalConv3d wrappers
        conv1_inner = original_block.conv1.conv
        conv2_inner = original_block.conv2.conv

        # Both convs: sharded in -> sharded out
        self.conv1 = ColumnRowParallelConv3d(conv1_inner, tp_degree)
        self.conv2 = ColumnRowParallelConv3d(conv2_inner, tp_degree)

        # Temporal padding (non-causal mode)
        self._pad1 = make_noncausal_pad_fn_from_kernel(
            original_block.conv1.time_kernel_size
        )
        self._pad2 = make_noncausal_pad_fn_from_kernel(
            original_block.conv2.time_kernel_size
        )

        # Timestep conditioning: scale_shift_table is [4, C] — NOT sharded
        # The shift/scale values are broadcast over spatial dims; they must be
        # sliced to match local channel count
        if self.timestep_conditioning:
            # scale_shift_table: [4, C_full] -> we need [4, C_shard]
            shard_size = original_block.scale_shift_table.shape[1] // tp_degree
            from neuronx_distributed.parallel_layers import parallel_state

            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            start = tp_rank * shard_size
            end = start + shard_size
            self.scale_shift_table = nn.Parameter(
                original_block.scale_shift_table.data[:, start:end].clone()
            )

        # Shortcut (identity when in_channels == out_channels)
        self.is_identity_shortcut = isinstance(
            original_block.conv_shortcut, nn.Identity
        )
        if not self.is_identity_shortcut:
            # norm3 is GroupNorm(1, in_channels) — sharded
            # conv_shortcut is a 1x1 linear — ColumnRowParallel
            self.norm3 = nn.GroupNorm(
                num_groups=1,
                num_channels=original_block.norm3.num_channels // tp_degree,
                eps=original_block.norm3.eps,
                affine=True,
            )
            # Shard GroupNorm weight/bias
            self.norm3.weight.data = get_sharded_data(
                original_block.norm3.weight.data, 0
            )
            self.norm3.bias.data = get_sharded_data(original_block.norm3.bias.data, 0)
            # conv_shortcut is make_linear_nd -> Conv3d(1,1,1)
            shortcut_conv = original_block.conv_shortcut
            if hasattr(shortcut_conv, "conv"):
                shortcut_conv = shortcut_conv.conv
            self.conv_shortcut = ColumnRowParallelConv3d(shortcut_conv, tp_degree)
        else:
            self.norm3 = nn.Identity()
            self.conv_shortcut = nn.Identity()

    def forward(self, x, timestep_embed=None):
        residual = x

        x = self.norm1(x)

        if self.timestep_conditioning and timestep_embed is not None:
            # timestep_embed: [B, 4*C_shard, 1, 1, 1] from UNetMidBlock3D
            batch_size = x.shape[0]
            ada_values = self.scale_shift_table[None, ..., None, None, None].to(
                device=x.device, dtype=x.dtype
            ) + timestep_embed.reshape(
                batch_size,
                4,
                -1,
                timestep_embed.shape[-3],
                timestep_embed.shape[-2],
                timestep_embed.shape[-1],
            )
            shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)
            x = x * (1 + scale1) + shift1

        x = self.nonlinearity(x)
        x = self._pad1(x)
        x = self.conv1(x)

        x = self.norm2(x)

        if self.timestep_conditioning and timestep_embed is not None:
            x = x * (1 + scale2) + shift2

        x = self.nonlinearity(x)
        x = self._pad2(x)
        x = self.conv2(x)

        if not self.is_identity_shortcut:
            residual = self.norm3(residual)
            residual = self.conv_shortcut(residual)

        return x + residual


# ── Sharded UNetMidBlock3D ──────────────────────────────────────────


class ShardedUNetMidBlock3D(nn.Module):
    """Sharded UNetMidBlock3D with timestep conditioning.

    Contains a time_embedder (small MLP, not sharded) and N ResnetBlock3D instances.
    The time_embedder produces [B, C*4, 1, 1, 1] which is sliced per-rank for
    the 4 shift/scale values in each ResnetBlock3D.
    """

    def __init__(self, original_block, tp_degree):
        super().__init__()
        self.timestep_conditioning = original_block.timestep_conditioning

        if self.timestep_conditioning:
            # time_embedder: small MLP producing [B, C*4] — keep on all ranks (broadcast)
            # But we need to shard the output since ResnetBlock3D expects sharded shift/scale
            self.time_embedder = original_block.time_embedder
            self.tp_degree = tp_degree

        self.res_blocks = nn.ModuleList()
        for resnet in original_block.res_blocks:
            self.res_blocks.append(ShardedResnetBlock3D(resnet, tp_degree))

    def forward(self, hidden_states, scaled_timestep=None):
        timestep_embed = None
        if self.timestep_conditioning and scaled_timestep is not None:
            batch_size = hidden_states.shape[0]
            # time_embedder output: [B, C*4] where C is full channel count
            embed_full = self.time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=hidden_states.dtype,
            )
            # Reshape to [B, C*4, 1, 1, 1]
            timestep_embed = embed_full.view(batch_size, embed_full.shape[-1], 1, 1, 1)
            # Shard: [B, 4*C_full, 1, 1, 1] -> [B, 4*C_shard, 1, 1, 1]
            # The 4*C layout is interleaved: [shift1_0..shift1_C, scale1_0..scale1_C, ...]
            # Actually the ResnetBlock3D reshapes it as (B, 4, C, 1, 1, 1) then unbinds
            # So the layout is [4, C] contiguous in the embed dim
            # We need to shard C within each of the 4 groups
            full_c = embed_full.shape[-1] // 4
            shard_c = full_c // self.tp_degree
            from neuronx_distributed.parallel_layers import parallel_state

            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            start = tp_rank * shard_c
            end = start + shard_c
            # Reshape to [B, 4, C, 1, 1, 1], shard C, reshape back
            embed_4d = timestep_embed.view(batch_size, 4, full_c, 1, 1, 1)
            embed_sharded = embed_4d[:, :, start:end, :, :, :].contiguous()
            timestep_embed = embed_sharded.view(batch_size, 4 * shard_c, 1, 1, 1)

        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states, timestep_embed=timestep_embed)

        return hidden_states


# ── Sharded DepthToSpaceUpsample ────────────────────────────────────


class ShardedDepthToSpaceUpsample(nn.Module):
    """LTX-2.3 DepthToSpaceUpsample (sub-pixel shuffle) with TP support.

    The sub-pixel shuffle is a local reshape+permute operation. At any TP
    degree, per-rank channel counts are divisible by stride_prod, so
    the shuffle works independently per rank.

    ltx-core's DepthToSpaceUpsample uses einops rearrange for the shuffle.
    We replicate the exact same logic with explicit reshape+permute.
    """

    def __init__(self, original_upsampler, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        # stride is a tuple like (2, 2, 2)
        self.stride = (
            original_upsampler.stride
            if hasattr(original_upsampler, "stride")
            else (2, 2, 2)
        )
        self.residual = getattr(original_upsampler, "residual", False)

        # The conv inside DepthToSpaceUpsample
        conv_module = original_upsampler.conv
        # CausalConv3d wraps nn.Conv3d
        inner_conv = conv_module.conv if hasattr(conv_module, "conv") else conv_module
        self.conv = ColumnRowParallelConv3d(inner_conv, tp_degree)

        # Temporal padding
        kernel_t = getattr(conv_module, "time_kernel_size", 3)
        self._pad = make_noncausal_pad_fn_from_kernel(kernel_t)

        # Compute per-rank out_channels for the shuffle
        # Original: out_channels = prod(stride) * in_channels // reduction_factor
        # After TP sharding, each rank has out_channels // tp_degree
        stride_prod = self.stride[0] * self.stride[1] * self.stride[2]
        self.stride_prod = stride_prod

        # Store out_channels_reduction_factor if present
        self.out_channels_reduction_factor = getattr(
            original_upsampler, "out_channels_reduction_factor", 1
        )

    def forward(self, x):
        batch_size, num_channels, num_frames, height, width = x.shape
        s_t, s_h, s_w = self.stride

        residual = None
        if self.residual:
            # Reshape for skip connection: depth-to-space on input
            residual = x.reshape(
                batch_size, -1, s_t, s_h, s_w, num_frames, height, width
            )
            residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4)
            residual = residual.flatten(6, 7).flatten(4, 5).flatten(2, 3)
            # Repeat channels for reduction factor
            repeats = self.stride_prod // self.out_channels_reduction_factor
            if repeats > 1:
                residual = residual.repeat(1, repeats, 1, 1, 1)
            # Remove first frame (temporal stride)
            residual = residual[:, :, s_t - 1 :]

        x = self._pad(x)
        x = self.conv(x)

        # Sub-pixel shuffle
        b2, c2, f2, h2, w2 = x.shape
        x = x.reshape(b2, -1, s_t, s_h, s_w, f2, h2, w2)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        # Remove first frame
        x = x[:, :, s_t - 1 :]

        if residual is not None:
            x = x + residual

        return x


# ── Full Sharded LTX-2.3 Decoder ───────────────────────────────────


class ShardedLTX23Decoder(nn.Module):
    """Tensor Parallel LTX-2.3 VAE Decoder.

    Wraps the ltx-core VideoDecoder's flat up_blocks list.

    The compiled graph expects pre-processed input:
      - Noise already injected (if timestep_conditioning)
      - Latent already denormalized via per_channel_statistics
      - Scaled timestep computed on CPU

    Channel flow:
      conv_in:    128 full  -> 1024 sharded  (ColumnParallel)
      up_blocks:  flat list of UNetMidBlock3D + DepthToSpaceUpsample
        [0] res_x@1024: 5 sharded resnets
        [1] compress_all: 1024 sharded -> upsampler -> 512 sharded
        [2] res_x@512: 5 sharded resnets
        [3] compress_all: 512 sharded -> upsampler -> 256 sharded
        [4] res_x@256: 5 sharded resnets
        [5] compress_all: 256 sharded -> upsampler -> 128 sharded
        [6] res_x@128: 5 sharded resnets
      norm_out:   ShardedPixelNorm (128 sharded)
      [AdaLN]:    last_scale_shift_table applied (if timestep_conditioning)
      conv_act:   SiLU
      conv_out:   128 sharded -> 48 full (RowParallel)
      unpatchify: 48 -> 3 (spatial rearrangement, patch_size=4)
    """

    def __init__(self, video_decoder, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.timestep_conditioning = video_decoder.timestep_conditioning
        self.patch_size = video_decoder.patch_size

        # conv_in: 128 full -> 1024 sharded
        conv_in_module = video_decoder.conv_in
        inner_conv_in = (
            conv_in_module.conv if hasattr(conv_in_module, "conv") else conv_in_module
        )
        self._conv_in_pad = make_noncausal_pad_fn_from_kernel(
            getattr(conv_in_module, "time_kernel_size", 3)
        )
        self.conv_in = ColumnParallelConv3d(inner_conv_in, tp_degree)

        # Timestep conditioning: compute scaled_timestep on CPU, pass in
        if self.timestep_conditioning:
            self.timestep_scale_multiplier = video_decoder.timestep_scale_multiplier

        # up_blocks: walk the flat list
        self.sharded_up_blocks = nn.ModuleList()
        self.up_block_types = []  # "mid" or "upsample"

        for block in video_decoder.up_blocks:
            block_type = type(block).__name__
            if "UNetMidBlock3D" in block_type:
                self.sharded_up_blocks.append(ShardedUNetMidBlock3D(block, tp_degree))
                self.up_block_types.append("mid")
            elif "DepthToSpace" in block_type:
                self.sharded_up_blocks.append(
                    ShardedDepthToSpaceUpsample(block, tp_degree)
                )
                self.up_block_types.append("upsample")
            elif "ResnetBlock3D" in block_type:
                self.sharded_up_blocks.append(ShardedResnetBlock3D(block, tp_degree))
                self.up_block_types.append("resnet")
            else:
                raise ValueError(f"Unknown up_block type: {block_type}")

        # norm_out: sharded PixelNorm
        self.norm_out = ShardedPixelNorm(video_decoder.conv_norm_out, tp_degree)

        # Final AdaLN (timestep conditioning)
        if self.timestep_conditioning:
            # last_time_embedder: small MLP, keep on all ranks
            self.last_time_embedder = video_decoder.last_time_embedder
            # last_scale_shift_table: [2, C_full] — shard C
            self.last_scale_shift_table_full = video_decoder.last_scale_shift_table

        # conv_act: SiLU
        self.conv_act = nn.SiLU()

        # conv_out: 128 sharded -> 48 full (RowParallel)
        conv_out_module = video_decoder.conv_out
        inner_conv_out = (
            conv_out_module.conv
            if hasattr(conv_out_module, "conv")
            else conv_out_module
        )
        self._conv_out_pad = make_noncausal_pad_fn_from_kernel(
            getattr(conv_out_module, "time_kernel_size", 3)
        )
        self.conv_out = RowParallelConv3d(inner_conv_out, tp_degree)

    def forward(self, latent, scaled_timestep=None):
        """Forward pass on denormalized, noise-injected latent.

        Args:
            latent: [B, 128, T, H_tile, W_tile] — denormalized latent tile
            scaled_timestep: [B] — timestep * scale_multiplier (or None)

        Returns:
            [B, 3, T_out, H_out, W_out] — decoded video tile
        """
        # conv_in
        x = self._conv_in_pad(latent)
        x = self.conv_in(x)

        # up_blocks
        for block, btype in zip(self.sharded_up_blocks, self.up_block_types):
            if btype == "mid":
                x = block(x, scaled_timestep=scaled_timestep)
            elif btype == "upsample":
                x = block(x)
            elif btype == "resnet":
                x = block(x)

        # norm_out
        x = self.norm_out(x)

        # Final AdaLN
        if self.timestep_conditioning and scaled_timestep is not None:
            batch_size = x.shape[0]
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=x.dtype,
            )
            embedded_timestep = embedded_timestep.view(
                batch_size, embedded_timestep.shape[-1], 1, 1, 1
            )
            ada_values = self.last_scale_shift_table_full[
                None, ..., None, None, None
            ].to(device=x.device, dtype=x.dtype) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            # Shard shift/scale to match local channel count
            from neuronx_distributed.parallel_layers import parallel_state

            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            shard_c = x.shape[1]
            start = tp_rank * shard_c
            end = start + shard_c
            shift = shift[:, start:end]
            scale = scale[:, start:end]
            x = x * (1 + scale) + shift

        # conv_act + conv_out (gathers channels)
        x = self.conv_act(x)
        x = self._conv_out_pad(x)
        x = self.conv_out(x)  # [B, 48, F, H, W] — gathered (full channels)

        # Unpatchify: 48 channels -> 3 RGB channels
        p = self.patch_size  # 4
        batch_size, num_channels, num_frames, height, width = x.shape
        # From ltx-core: unpatchify(sample, patch_size_hw=4, patch_size_t=1)
        # Equivalent to rearrange "b (c p1 p2) f h w -> b c f (h p1) (w p2)"
        # where p1=p2=4, c=3
        x = x.reshape(batch_size, -1, p, p, num_frames, height, width)
        x = x.permute(0, 1, 4, 5, 2, 6, 3)  # [B, 3, F, H, p, W, p]
        x = x.reshape(batch_size, -1, num_frames, height * p, width * p)

        return x


class DecoderWrapperTP(nn.Module):
    """Wrapper for parallel_model_trace."""

    def __init__(self, video_decoder, tp_degree):
        super().__init__()
        self.decoder = ShardedLTX23Decoder(video_decoder, tp_degree)

    def forward(self, latent, scaled_timestep=None):
        return self.decoder(latent, scaled_timestep)


def get_decoder_model(tp_degree, model_path, config):
    """Factory function for parallel_model_trace.

    Loads the ltx-core VideoDecoder from safetensors and wraps it.

    Returns:
        (model, empty_dict): The DecoderWrapperTP model and empty state dict
    """
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.loader.sd_ops import SDOps

    vd_ops = (
        SDOps("v")
        .with_matching(prefix="vae.")
        .with_replacement("vae.decoder.", "")
        .with_replacement("vae.", "")
    )
    vd_builder = SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path=model_path,
        model_sd_ops=vd_ops,
    )
    video_decoder = vd_builder.build(device=torch.device("cpu"), dtype=torch.float32)
    video_decoder.eval()

    wrapper = DecoderWrapperTP(video_decoder, tp_degree)
    wrapper.eval()
    return wrapper, {}


def compile_vae_decoder(
    tp_degree=4,
    tile_height=128,
    tile_width=512,
    num_frames=121,
    output_dir="/home/ubuntu/ltx23_vae_tp4",
    compiler_workdir="/home/ubuntu/compiler_workdir_vae23",
    model_path="/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors",
    config=None,
    timestep_conditioning=True,
):
    """Compile the TP VAE decoder for Neuron.

    Args:
        tp_degree: Tensor parallel degree (default 4)
        tile_height: Tile height in pixels (default 128 for 4-latent)
        tile_width: Tile width in pixels (default 512 for 16-latent)
        num_frames: Number of video frames (default 121)
        output_dir: Directory to save compiled model
        compiler_workdir: Directory for compiler intermediate files
        model_path: Path to LTX-2.3 safetensors
        config: Parsed config dict (loaded from safetensors metadata if None)
        timestep_conditioning: Whether the decoder uses timestep conditioning

    Returns:
        compiled: The compiled parallel model
    """
    import json
    import time

    import neuronx_distributed

    if config is None:
        from safetensors import safe_open

        with safe_open(model_path, framework="pt") as f:
            metadata = f.metadata()
        config = json.loads(metadata["config"])

    latent_f = (num_frames - 1) // 8 + 1
    latent_h = tile_height // 32
    latent_w = tile_width // 32

    os.environ["LOCAL_WORLD_SIZE"] = str(tp_degree)
    os.environ["NEURON_CC_FLAGS"] = (
        os.environ.get("NEURON_CC_FLAGS", "") + f" {COMPILER_FLAGS}"
    )

    print("=" * 70)
    print("LTX-2.3 VAE Decoder — Tensor Parallel Compilation")
    print("=" * 70)
    print(f"  Tile size:   {tile_height}x{tile_width} pixels")
    print(f"  Latent tile: [1, 128, {latent_f}, {latent_h}, {latent_w}]")
    print(f"  TP degree:   {tp_degree}")
    print(f"  Timestep conditioning: {timestep_conditioning}")
    print(f"  Output dir:  {output_dir}")

    latent_input = torch.randn(
        1, 128, latent_f, latent_h, latent_w, dtype=torch.float32
    )

    # Build trace inputs
    if timestep_conditioning:
        # scaled_timestep is a scalar tensor [B=1]
        scaled_timestep = torch.tensor([0.05 * 1000.0], dtype=torch.float32)  # 50.0
        trace_inputs = (latent_input, scaled_timestep)
    else:
        trace_inputs = (latent_input,)

    print(f"\n  Compiling (this may take 10-30 minutes)...")
    t0 = time.time()

    get_model_fn = partial(get_decoder_model, tp_degree, model_path, config)

    compiled = neuronx_distributed.trace.parallel_model_trace(
        get_model_fn,
        trace_inputs,
        compiler_workdir=compiler_workdir,
        compiler_args=COMPILER_FLAGS,
        tp_degree=tp_degree,
        inline_weights_to_neff=False,
    )

    compile_time = time.time() - t0
    print(f"  Compiled in {compile_time:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    neuronx_distributed.trace.parallel_model_save(compiled, output_dir)
    print(f"  Saved to {output_dir}")

    # Quick validation
    print("\n  Running validation...")
    with torch.no_grad():
        neuron_out = compiled(*trace_inputs)
    print(f"  Output shape: {list(neuron_out.shape)}")
    print(f"  Output range: [{neuron_out.min():.3f}, {neuron_out.max():.3f}]")
    T_out = (num_frames - 1) + 1 if latent_f == 1 else num_frames
    print(f"  Expected:    [1, 3, ~{T_out}, {tile_height}, {tile_width}]")

    return compiled
