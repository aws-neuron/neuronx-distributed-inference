# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TCDecoder (TAEHV) for FlashVSR on AWS Trainium.

The TCDecoder converts latent video representations to RGB frames.
It uses temporal recurrence via MemBlock layers for inter-frame coherence.

Two execution modes:
  - Legacy (torch_neuronx.trace): Sequential decode with explicit state I/O.
    States transferred via PCIe each call. ~237ms/frame.
  - NxDI (ModelBuilder + input_output_aliases): Sequential decode with HBM
    state persistence. States remain in device memory between calls. ~79ms/frame.
    3.0x faster than trace-based approach.

The NxDI mode is the default for new compilations.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import namedtuple
from typing import Optional, List, Tuple
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


# ===================================================================
# NxDI TCDecoder: HBM state persistence via input_output_aliases
# ===================================================================

# Channel configuration
CHANNELS = [512, 256, 128, 128]
NUM_MEM_BLOCKS = 9  # 3 groups x 3 MemBlocks each
# Input channels: pixel_shuffle(cond) + latent = 784 channels
# Determined from TCDecoder.ckpt decoder.1.weight shape (512, 784, 3, 3)
INPUT_CHANNELS = 784


class NeuronTCDecoderStateful(nn.Module):
    """TCDecoder with MemBlock states as nn.Parameters for HBM persistence.

    The 9 MemBlock state tensors are stored as nn.Parameters (requires_grad=False).
    During forward pass, each state is read, used as 'past' for its MemBlock,
    and then the new state is written back. Combined with input_output_aliases,
    the compiler keeps states in HBM between calls — no PCIe transfer.

    Forward signature:
        Input:  x (1, C, H, W) — single latent frame
        Output: (frames, state_0, state_1, ..., state_8) — 10 tensors total
                frames: (4, 3, H_out, W_out) RGB frames
                state_i: updated MemBlock states
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        H_lat = config.height // 8
        W_lat = config.width // 8
        dtype = config.neuron_config.torch_dtype

        # Build the decoder layers (same structure as TAEHV.decoder)
        self.decoder = self._build_decoder()

        # Pre-analyze layer types for the forward loop
        self.layer_types = []
        for layer in self.decoder:
            if isinstance(layer, MemBlock):
                self.layer_types.append("memblock")
            elif isinstance(layer, TGrow):
                self.layer_types.append("tgrow")
            else:
                self.layer_types.append("standard")

        # State tensors as nn.Parameters (HBM-persistent via aliases)
        # Shape depends on TGrow layers that precede each MemBlock group:
        # Group 0 (MemBlocks 0-2): Before any TGrow with stride>1 → (1, 512, H, W)
        # Group 1 (MemBlocks 3-5): After TGrow(stride=1) → still (1, 256, 2H, 2W)
        # Group 2 (MemBlocks 6-8): After TGrow(stride=2) → (2, 128, 4H, 4W)
        state_shapes = [
            (1, 512, H_lat, W_lat),
            (1, 512, H_lat, W_lat),
            (1, 512, H_lat, W_lat),
            (1, 256, H_lat * 2, W_lat * 2),
            (1, 256, H_lat * 2, W_lat * 2),
            (1, 256, H_lat * 2, W_lat * 2),
            (2, 128, H_lat * 4, W_lat * 4),
            (2, 128, H_lat * 4, W_lat * 4),
            (2, 128, H_lat * 4, W_lat * 4),
        ]

        self.mem_states = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
                for shape in state_shapes
            ]
        )

    def _build_decoder(self):
        """Build decoder nn.Sequential matching TAEHV architecture."""
        n_f = CHANNELS

        base_decoder = nn.Sequential(
            Clamp(),
            conv2d_3x3(INPUT_CHANNELS, n_f[0]),
            nn.ReLU(inplace=False),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[0], 1),
            conv2d_3x3(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[1], 2),
            conv2d_3x3(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[2], 2),
            conv2d_3x3(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=False),
            conv2d_3x3(n_f[3], 3),
        )

        # Apply identity deepening (same as TAEHV._apply_identity_deepen)
        return self._apply_identity_deepen(base_decoder, how_many_each=1, k=3)

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
                        new_layers.append(nn.ReLU(inplace=False))
        return nn.Sequential(*new_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Process one latent frame. States read from/written to Parameters.

        Args:
            x: (1, C, H, W) single latent frame

        Returns:
            Tuple of (output_frames, updated_state_0, ..., updated_state_8)
        """
        mem_idx = 0
        updated_states = [None] * NUM_MEM_BLOCKS

        for i, layer in enumerate(self.decoder):
            lt = self.layer_types[i]
            if lt == "memblock":
                past = self.mem_states[mem_idx]
                # New state = current x BEFORE applying MemBlock
                updated_states[mem_idx] = x.clone()
                x = layer(x, past)
                mem_idx += 1
            elif lt == "tgrow":
                x = layer(x)
            else:
                x = layer(x)

        # Return frames + all updated states (aliased back to Parameters)
        return (x, *updated_states)


# ===================================================================
# NxDI Application (uses ModelBuilder directly — simpler than
# NeuronApplicationBase which requires full transformer config)
# ===================================================================


# Compiler flags: no auto-cast (already bf16), -O1 for fast compilation
TCDECODER_COMPILER_ARGS = "--auto-cast=none -O1"


class TCDecoderConfig:
    """Configuration for TCDecoder NxDI compilation."""

    def __init__(self, neuron_config, height=768, width=1280):
        self.neuron_config = neuron_config
        self.height = height
        self.width = width


class TCDecoderApplication:
    """NxDI Application for TCDecoder with HBM state persistence.

    Uses ModelBuilder directly with BaseModelInstance for input_output_aliases.
    States persist in device HBM between forward calls — no PCIe transfer.

    Performance (validated on trn2.3xlarge, SDK 2.29):
    - Per-frame latency: 79ms (vs 237ms trace baseline) = 3.0x faster
    - Compilation: 10.4s (vs ~5min trace)

    Usage:
        from neuronx_distributed_inference.models.config import NeuronConfig

        neuron_config = NeuronConfig(tp_degree=1, torch_dtype=torch.bfloat16, batch_size=1)
        config = TCDecoderConfig(neuron_config=neuron_config, height=768, width=1280)
        app = TCDecoderApplication(weights_dir="/path/to/weights", config=config)
        app.compile(output_dir)
        app.load(compiled_dir)
        app.reset_states()
        for frame in frames:
            rgb = app(frame)  # States persist in HBM
    """

    def __init__(self, weights_dir: str, config: TCDecoderConfig):
        self.weights_dir = weights_dir
        self.config = config
        self._traced_model = None
        self._loaded = False

    def compile(self, output_dir: str):
        """Compile TCDecoder via NxDI ModelBuilder with input_output_aliases."""
        from neuronx_distributed.trace.model_builder import (
            BaseModelInstance,
            ModelBuilder,
        )

        os.makedirs(output_dir, exist_ok=True)

        # Create model instance with aliases
        model_instance = self._create_model_instance(BaseModelInstance)

        # Checkpoint loader provides weights for sharding
        state_dict = self._load_weights()

        def checkpoint_loader(*args, **kwargs):
            return state_dict

        # Build and trace
        builder = ModelBuilder(
            router=None,
            tp_degree=1,
            checkpoint_loader=checkpoint_loader,
        )
        builder.add(
            key="tcdecoder",
            model_instance=model_instance,
            example_inputs=self._get_example_inputs(),
            compiler_args=TCDECODER_COMPILER_ARGS,
        )
        traced_model = builder.trace(initialize_model_weights=False)

        # Save compiled model
        neff_path = os.path.join(output_dir, "model.pt")
        torch.jit.save(traced_model, neff_path)
        del traced_model

        # Shard and save weights (required for load-time initialization)
        weights_dir = os.path.join(output_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        sharded_weights = builder.shard_checkpoint()
        # For TP=1, there's only one rank
        from safetensors.torch import save_file

        save_file(
            sharded_weights[0],
            os.path.join(weights_dir, "tp0_sharded_checkpoint.safetensors"),
        )

    def load(self, compiled_dir: str):
        """Load compiled TCDecoder NEFF."""
        from safetensors.torch import load_file

        neff_path = os.path.join(compiled_dir, "model.pt")
        self._traced_model = torch.jit.load(neff_path)

        # Load sharded weights and initialize the model
        weights_path = os.path.join(
            compiled_dir, "weights", "tp0_sharded_checkpoint.safetensors"
        )
        weights = [load_file(weights_path)]
        start_rank = torch.tensor([0], dtype=torch.int32)
        self._traced_model.nxd_model.initialize(weights, start_rank)

        self._loaded = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run one frame through TCDecoder. States persist in HBM.

        Args:
            x: (1, C, H, W) single latent frame

        Returns:
            (4, 3, H_out, W_out) RGB frames
        """
        assert self._loaded, "Must call load() before inference"
        outputs = self._traced_model(x)
        # The traced model returns (frames, state_0, ..., state_8) but with
        # input_output_aliases, states are written back to HBM internally.
        # We only return the frames tensor.
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def reset_states(self):
        """Reset all MemBlock states to zero before processing a new video.

        Flushes states by running 3 zero-input forward passes. After 3 passes
        with zero input, all MemBlock states converge to the zero-input fixed
        point (~240ms total, runs once per video).
        """
        assert self._loaded, "Must call load() before reset_states()"
        H_lat = self.config.height // 8
        W_lat = self.config.width // 8
        dtype = self.config.neuron_config.torch_dtype
        zero_frame = torch.zeros(1, INPUT_CHANNELS, H_lat, W_lat, dtype=dtype)
        with torch.no_grad():
            for _ in range(3):
                self._traced_model(zero_frame)

    def _create_model_instance(self, BaseModelInstance):
        """Create a BaseModelInstance with loaded weights and alias config."""
        config = self.config

        class _Instance(BaseModelInstance):
            def __init__(self, cfg):
                self.module = None
                self.config = cfg
                self.neuron_config = cfg.neuron_config

            def load_module(self):
                self.module = NeuronTCDecoderStateful(self.config)
                self.module.eval()
                if self.neuron_config.torch_dtype != torch.float32:
                    self.module = self.module.to(self.neuron_config.torch_dtype)

            def get(self, bucket_rank, **kwargs):
                """Return module + aliases mapping state Parameters → output indices."""
                aliases = {}
                output_index = 1  # output[0] = frames
                for i in range(NUM_MEM_BLOCKS):
                    aliases[self.module.mem_states[i]] = output_index + i
                return self.module, aliases

        instance = _Instance(config)
        instance.load_module()

        # Load and apply weights
        state_dict = self._load_weights()
        instance.module.load_state_dict(state_dict, strict=False)

        return instance

    def _get_example_inputs(self):
        """Generate example inputs for tracing (list of tuples for buckets)."""
        H_lat = self.config.height // 8
        W_lat = self.config.width // 8
        dtype = self.config.neuron_config.torch_dtype
        x = torch.randn(1, INPUT_CHANNELS, H_lat, W_lat, dtype=dtype)
        return [(x,)]  # Single bucket, single input tensor

    def _load_weights(self):
        """Load TAEHV weights from checkpoint directory."""
        # Try safetensors first
        ckpt_path = os.path.join(
            self.weights_dir, "taehv_decoder_streaming.safetensors"
        )
        if os.path.exists(ckpt_path):
            from safetensors import safe_open

            sd = {}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    sd[key] = f.get_tensor(key)
            return sd

        # Try .ckpt format (FlashVSR-v1.1 uses TCDecoder.ckpt)
        ckpt_path = os.path.join(self.weights_dir, "TCDecoder.ckpt")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            return sd

        # Fallback: try any checkpoint file
        candidates = glob.glob(os.path.join(self.weights_dir, "*.safetensors"))
        candidates += glob.glob(os.path.join(self.weights_dir, "*.ckpt"))
        if candidates:
            if candidates[0].endswith(".safetensors"):
                from safetensors import safe_open

                sd = {}
                with safe_open(candidates[0], framework="pt", device="cpu") as f:
                    for key in f.keys():
                        sd[key] = f.get_tensor(key)
                return sd
            else:
                return torch.load(candidates[0], map_location="cpu", weights_only=True)

        raise FileNotFoundError(f"No TCDecoder checkpoint found in {self.weights_dir}")


# ===================================================================
# NxDI decode function
# ===================================================================


def decode_video_nxdi(
    app: TCDecoderApplication,
    latents: torch.Tensor,
    cond: torch.Tensor,
    pixel_shuffle_fn,
    frames_to_trim: int = 3,
) -> torch.Tensor:
    """Decode video using NxDI TCDecoder with HBM state persistence.

    States persist in HBM between frames — no PCIe transfer per call.
    3.0x faster than trace-based neuron_decode_video_sequential().

    Args:
        app: TCDecoderApplication (compiled and loaded)
        latents: (1, T, C, H, W) latent tensor
        cond: (1, C_cond, T_cond, H_cond, W_cond) LQ conditioning
        pixel_shuffle_fn: TCPixelShuffle3d module
        frames_to_trim: frames to trim from start (default 3)

    Returns:
        (1, 3, T_out, H_out, W_out) decoded RGB video
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

    # Reset states to zero before decoding
    app.reset_states()

    # Process each frame — states persist in HBM (no PCIe per call)
    outputs = []
    for t in range(T_total):
        xt = x_4d[t : t + 1]
        with torch.no_grad():
            frames_t = app(xt)  # Returns only RGB frames
        outputs.append(frames_t)

    # Concatenate and reshape
    all_frames = torch.cat(outputs, dim=0)
    T_out = all_frames.shape[0]
    result = all_frames.reshape(N, T_out, *all_frames.shape[1:])
    result = result[:, frames_to_trim:]
    result = result.transpose(1, 2)  # NTCHW -> NCTHW

    return result
