"""
NxDI LTX-2 VAE Decoder — Tensor Parallel Model
================================================
Tensor-parallel VAE decoder for the LTX-2 video diffusion model.

Compiles the LTX-2 VAE decoder with tensor parallelism to overcome the
single-core compilation limit (max 64 latent spatial elements at 128 channels).

The optimal tile shape is 4×16 latent (128×512 pixels), which maximizes the
spatial area within the 64-element SRAM budget. Rectangular tiles reduce the
tile count by 43-69% compared to 8×8 tiles at 1024×1536 resolution while
being 12.5% faster per-tile (1257ms vs 1440ms).

Architecture (verified from diffusers source):
  conv_in (128 -> 1024):                   ColumnParallel (full in, sharded out)
  mid_block (5 resnets @ 1024ch):          ColumnRowParallel (sharded throughout)
  up_block_0: upsampler (1024->512) + 5 resnets @ 512ch  — all sharded
  up_block_1: upsampler (512->256) + 5 resnets @ 256ch   — all sharded
  up_block_2: upsampler (256->128) + 5 resnets @ 128ch   — all sharded
  norm_out + conv_out (128 -> 48):         RowParallel on conv_out (sharded in, gathered out)

CRITICAL: Up-block order is UPSAMPLER FIRST, then resnets (verified from source).

The sub-pixel shuffle in the upsampler rearranges channels into spatial dims.
Channel counts are all divisible by stride_t*stride_h*stride_w=8 at every
TP degree (1024/8=128, 512/8=64, 256/8=32), so shuffle works per-rank.

Compilation boundary:
  - H_latent × W_latent ≤ 64 elements (SRAM limit, NCC_IGCA030 above this)
  - 8×8 = 64 ✓ (1440ms/tile)
  - 4×16 = 64 ✓ (1257ms/tile — 12.5% faster, optimal for wide outputs)
  - 4×20 = 80 ✗ (NCC_IGCA030)
  - 9×9 = 81 ✗ (NCC_IGCA030)

Usage:
  See compile_vae.py for the standalone compilation script, or
  tiled_vae_decode.py for the tiled runtime decoder.
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


def make_noncausal_pad_fn(kernel_size_t):
    """Create non-causal temporal padding function for LTX2VideoCausalConv3d.

    Replicates the non-causal branch of LTX2VideoCausalConv3d.forward():
      pad_left = first frame repeated (kernel_t - 1) // 2 times
      pad_right = last frame repeated (kernel_t - 1) // 2 times
    """
    pad_t = (kernel_size_t - 1) // 2  # 1 for kernel=3

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

    def __init__(self, original_conv, tp_degree):
        super().__init__()
        self.sharded_out = original_conv.out_channels // tp_degree
        self.conv = nn.Conv3d(
            original_conv.in_channels,
            self.sharded_out,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            padding_mode=original_conv.padding_mode,
            bias=original_conv.bias is not None,
        )
        self.conv.weight.data = get_sharded_data(original_conv.weight.data, 0)
        if original_conv.bias is not None:
            self.conv.bias.data = get_sharded_data(original_conv.bias.data, 0)

    def forward(self, x):
        return self.conv(x)


class RowParallelConv3d(nn.Module):
    """Conv3d with input channels sharded. Output is all-reduced (full channels).
    Input: sharded channels. Output: full channels."""

    def __init__(self, original_conv, tp_degree):
        super().__init__()
        from neuronx_distributed.parallel_layers.mappings import (
            reduce_from_tensor_model_parallel_region,
        )

        self.reduce = reduce_from_tensor_model_parallel_region
        self.sharded_in = original_conv.in_channels // tp_degree
        self.conv = nn.Conv3d(
            self.sharded_in,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            padding_mode=original_conv.padding_mode,
            bias=original_conv.bias is not None,
        )
        self.conv.weight.data = get_sharded_data(original_conv.weight.data, 1)
        if original_conv.bias is not None:
            # Bias is added after reduce (sum), so divide by tp_degree to compensate
            self.conv.bias.data = original_conv.bias.data.clone() / tp_degree

    def forward(self, x):
        return self.reduce(self.conv(x))


class ColumnRowParallelConv3d(nn.Module):
    """Conv3d with sharded input -> sharded output via all-gather + column parallel.

    Correct implementation: all-gather input along channel dim (dim=1) across
    TP ranks to reconstruct full input channels, then apply ColumnParallel conv
    (full in -> sharded out).

    This replaces the previous incorrect "diagonal-only" implementation that
    skipped cross-rank channel interactions.

    Communication: 1 all-gather per forward pass.
    """

    def __init__(self, original_conv, tp_degree):
        super().__init__()
        from neuronx_distributed.parallel_layers.mappings import (
            gather_from_tensor_model_parallel_region_with_dim,
        )

        self.gather_channels = gather_from_tensor_model_parallel_region_with_dim
        self.sharded_out = original_conv.out_channels // tp_degree
        # Conv takes FULL input channels, produces SHARDED output channels
        self.conv = nn.Conv3d(
            original_conv.in_channels,  # full input
            self.sharded_out,  # sharded output
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            padding_mode=original_conv.padding_mode,
            bias=original_conv.bias is not None,
        )
        # Shard weights along output dim only (input stays full)
        self.conv.weight.data = get_sharded_data(original_conv.weight.data, 0)
        if original_conv.bias is not None:
            self.conv.bias.data = get_sharded_data(original_conv.bias.data, 0)

    def forward(self, x):
        # x: [B, C_shard, F, H, W] -> all-gather along dim=1 -> [B, C_full, F, H, W]
        x_full = self.gather_channels(x, gather_dim=1)
        return self.conv(x_full)


# ── Sharded RMSNorm ─────────────────────────────────────────────────


class ShardedPerChannelRMSNorm(nn.Module):
    """PerChannelRMSNorm that works with sharded channel dimension.

    When channels are sharded across TP ranks, each rank only has C/tp channels.
    The local mean(x^2) over local channels is wrong — we need the global mean.

    Strategy:
      1. Compute local sum of x^2 over local channels: [B, 1, F, H, W]
      2. All-reduce (sum) across ranks to get global sum of x^2
      3. Divide by total channel count for global mean
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
        # x shape: [B, C_shard, F, H, W]
        local_sq_sum = (x**2).sum(dim=1, keepdim=True)  # [B, 1, F, H, W]
        global_sq_sum = self.reduce(local_sq_sum)  # sum across TP ranks
        n_channels = x.shape[1] * self.tp_degree  # total channels
        rms = torch.sqrt(global_sq_sum / n_channels + self.eps)
        return x / rms


# ── Sharded ResNet Block ────────────────────────────────────────────


class ShardedLTX2ResnetBlock(nn.Module):
    """LTX-2 ResNet block with sharded channels.

    Verified forward order (from diffusers source):
      residual = inputs
      x = norm1(x) -> SiLU -> conv1 -> norm2 -> SiLU -> conv2
      return x + residual
    """

    def __init__(self, original_block, tp_degree):
        super().__init__()
        self.nonlinearity = nn.SiLU()

        # Norms — sharded
        self.norm1 = ShardedPerChannelRMSNorm(original_block.norm1, tp_degree)
        self.norm2 = ShardedPerChannelRMSNorm(original_block.norm2, tp_degree)

        # Extract inner Conv3d from LTX2VideoCausalConv3d wrappers
        conv1_inner = original_block.conv1.conv
        conv2_inner = original_block.conv2.conv

        # Both convs are ColumnRowParallel (sharded in, sharded out)
        self.conv1 = ColumnRowParallelConv3d(conv1_inner, tp_degree)
        self.conv2 = ColumnRowParallelConv3d(conv2_inner, tp_degree)

        # Temporal padding functions (non-causal mode)
        self._pad1 = make_noncausal_pad_fn(original_block.conv1.kernel_size[0])
        self._pad2 = make_noncausal_pad_fn(original_block.conv2.kernel_size[0])

    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self._pad1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self._pad2(x)
        x = self.conv2(x)

        return x + residual


# ── Sharded Upsampler ───────────────────────────────────────────────


class ShardedLTX2Upsampler(nn.Module):
    """LTX-2 sub-pixel shuffle upsampler with TP support.

    The sub-pixel shuffle is a local reshape+permute operation. At any TP
    degree, the per-rank channel counts are divisible by stride_prod=8, so
    the shuffle works independently per rank.
    """

    def __init__(self, original_upsampler, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.stride = original_upsampler.stride
        self.upscale_factor = original_upsampler.upscale_factor
        self.residual = original_upsampler.residual

        conv_inner = original_upsampler.conv.conv
        self.conv = ColumnRowParallelConv3d(conv_inner, tp_degree)
        self._pad = make_noncausal_pad_fn(original_upsampler.conv.kernel_size[0])

    def forward(self, x):
        batch_size, num_channels, num_frames, height, width = x.shape

        residual = None
        if self.residual:
            residual = x.reshape(
                batch_size,
                -1,
                self.stride[0],
                self.stride[1],
                self.stride[2],
                num_frames,
                height,
                width,
            )
            residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4)
            residual = residual.flatten(6, 7).flatten(4, 5).flatten(2, 3)
            repeats = (
                self.stride[0] * self.stride[1] * self.stride[2]
            ) // self.upscale_factor
            residual = residual.repeat(1, repeats, 1, 1, 1)
            residual = residual[:, :, self.stride[0] - 1 :]

        x = self._pad(x)
        x = self.conv(x)

        batch_size2, num_channels2, num_frames2, height2, width2 = x.shape
        x = x.reshape(
            batch_size2,
            -1,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            num_frames2,
            height2,
            width2,
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        x = x[:, :, self.stride[0] - 1 :]

        if residual is not None:
            x = x + residual

        return x


# ── Full Sharded Decoder ────────────────────────────────────────────


class ShardedLTX2Decoder(nn.Module):
    """Tensor Parallel LTX-2 VAE Decoder.

    All layers stay sharded until the very end. Channel flow:

      conv_in:    128 full  -> 1024 sharded  (ColumnParallel)
      mid_block:  1024 sharded (5 resnets, ColumnRowParallel)
      up_block_0: 1024 sharded -> upsampler -> 512 sharded -> 5 resnets @ 512
      up_block_1: 512 sharded  -> upsampler -> 256 sharded -> 5 resnets @ 256
      up_block_2: 256 sharded  -> upsampler -> 128 sharded -> 5 resnets @ 128
      norm_out:   128 sharded -> ShardedRMSNorm -> 128 sharded
      conv_act:   SiLU (element-wise, sharded OK)
      conv_out:   128 sharded -> 48 full (RowParallel, gathers via all-reduce)
      unpatchify: reshape 48 channels -> 3 RGB channels via patch_size=4

    CRITICAL order: upsampler runs FIRST in each up_block, then resnets.
    """

    def __init__(self, vae, tp_degree):
        super().__init__()
        decoder = vae.decoder
        self.tp_degree = tp_degree

        # conv_in: 128 full -> 1024 sharded (ColumnParallel)
        self._conv_in_pad = make_noncausal_pad_fn(decoder.conv_in.kernel_size[0])
        self.conv_in = ColumnParallelConv3d(decoder.conv_in.conv, tp_degree)

        # mid_block: 5 resnets at 1024 channels (all sharded)
        self.mid_resnets = nn.ModuleList()
        for resnet in decoder.mid_block.resnets:
            self.mid_resnets.append(ShardedLTX2ResnetBlock(resnet, tp_degree))

        # up_blocks: upsampler FIRST, then resnets (verified from source)
        self.up_upsamplers = nn.ModuleList()
        self.up_resnets = nn.ModuleList()

        for up_block in decoder.up_blocks:
            self.up_upsamplers.append(
                ShardedLTX2Upsampler(up_block.upsamplers[0], tp_degree)
            )
            block_resnets = nn.ModuleList()
            for resnet in up_block.resnets:
                block_resnets.append(ShardedLTX2ResnetBlock(resnet, tp_degree))
            self.up_resnets.append(block_resnets)

        # norm_out: sharded PerChannelRMSNorm
        self.norm_out = ShardedPerChannelRMSNorm(decoder.norm_out, tp_degree)

        # conv_act: SiLU (element-wise, works on sharded data)
        self.conv_act = nn.SiLU()

        # conv_out: 128 sharded -> 48 full (RowParallel gathers)
        self._conv_out_pad = make_noncausal_pad_fn(decoder.conv_out.kernel_size[0])
        self.conv_out = RowParallelConv3d(decoder.conv_out.conv, tp_degree)

        # Unpatchify config (from VAE config)
        self.patch_size = decoder.patch_size  # 4
        self.patch_size_t = decoder.patch_size_t  # 1

    def forward(self, latent):
        # conv_in
        x = self._conv_in_pad(latent)
        x = self.conv_in(x)

        # mid_block
        for resnet in self.mid_resnets:
            x = resnet(x)

        # up_blocks: upsampler FIRST, then resnets
        for upsampler, resnets in zip(self.up_upsamplers, self.up_resnets):
            x = upsampler(x)
            for resnet in resnets:
                x = resnet(x)

        # Output path (norm -> act -> conv_out gathers channels)
        x = self.norm_out(x)
        x = self.conv_act(x)
        x = self._conv_out_pad(x)
        x = self.conv_out(x)  # [B, 48, F, H, W] — gathered (full channels)

        # Unpatchify: 48 channels -> 3 RGB channels
        p = self.patch_size  # 4
        p_t = self.patch_size_t  # 1

        batch_size, num_channels, num_frames, height, width = x.shape
        x = x.reshape(batch_size, -1, p_t, p, p, num_frames, height, width)
        x = x.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return x


class DecoderWrapperTP(nn.Module):
    """Wrapper for parallel_model_trace."""

    def __init__(self, vae, tp_degree):
        super().__init__()
        self.decoder = ShardedLTX2Decoder(vae, tp_degree)

    def forward(self, latent):
        return self.decoder(latent)


def get_decoder_model(tp_degree, vae_path="Lightricks/LTX-2"):
    """Factory function for parallel_model_trace.

    Returns:
        (model, empty_dict): The DecoderWrapperTP model and empty state dict
    """
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        vae_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.eval()

    wrapper = DecoderWrapperTP(vae, tp_degree)
    wrapper.eval()
    return wrapper, {}


def compile_vae_decoder(
    tp_degree=4,
    tile_height=128,
    tile_width=512,
    num_frames=121,
    output_dir="/home/ubuntu/ltx2_vae_tp4",
    compiler_workdir="/home/ubuntu/compiler_workdir_vae",
    vae_path="Lightricks/LTX-2",
):
    """Compile the TP VAE decoder for Neuron.

    Args:
        tp_degree: Tensor parallel degree (default 4)
        tile_height: Tile height in pixels (default 128 for 4-latent)
        tile_width: Tile width in pixels (default 512 for 16-latent)
        num_frames: Number of video frames (default 121)
        output_dir: Directory to save compiled model
        compiler_workdir: Directory for compiler intermediate files
        vae_path: HuggingFace model ID for the VAE

    Returns:
        compiled: The compiled parallel model
    """
    import time

    import neuronx_distributed

    latent_f = (num_frames - 1) // 8 + 1
    latent_h = tile_height // 32
    latent_w = tile_width // 32

    os.environ["LOCAL_WORLD_SIZE"] = str(tp_degree)
    os.environ["NEURON_CC_FLAGS"] = (
        os.environ.get("NEURON_CC_FLAGS", "") + f" {COMPILER_FLAGS}"
    )

    print("=" * 70)
    print("LTX-2 VAE Decoder — Tensor Parallel Compilation")
    print("=" * 70)
    print(f"  Tile size:   {tile_height}x{tile_width} pixels")
    print(f"  Latent tile: [1, 128, {latent_f}, {latent_h}, {latent_w}]")
    print(f"  TP degree:   {tp_degree}")
    print(f"  Output dir:  {output_dir}")

    latent_input = torch.randn(
        1, 128, latent_f, latent_h, latent_w, dtype=torch.float32
    )

    print("\n  Compiling (this may take 10-30 minutes)...")
    t0 = time.time()

    get_model_fn = partial(get_decoder_model, tp_degree, vae_path)

    compiled = neuronx_distributed.trace.parallel_model_trace(
        get_model_fn,
        (latent_input,),
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
        neuron_out = compiled(latent_input)
    print(f"  Output shape: {list(neuron_out.shape)}")
    print(f"  Output range: [{neuron_out.min():.3f}, {neuron_out.max():.3f}]")
    print(f"  Expected:    [1, 3, {num_frames}, {tile_height}, {tile_width}]")

    return compiled
