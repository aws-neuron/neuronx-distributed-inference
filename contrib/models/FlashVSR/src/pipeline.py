# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FlashVSR inference pipeline for AWS Trainium.

Orchestrates the full FlashVSR video super-resolution pipeline:
  1. LQ Projection (torch_neuronx.trace) -- generates per-token conditioning
  2. DiT denoising (NxDI ModelBuilder, TP=4) -- streaming chunks (first f=6, then f=2)
  3. TCDecoder (torch_neuronx.trace, sequential) -- latent to RGB conversion
  4. Color correction (CPU) -- wavelet/adain alignment with LQ reference

Usage:
    from src.pipeline import FlashVSRPipeline, compile_pipeline, load_pipeline, run_inference

    # Compile all components (run once per resolution):
    compile_pipeline(weights_dir="/path/to/FlashVSR-v1.1", output_dir="/path/to/compiled")

    # Load compiled models:
    pipeline = load_pipeline(compiled_dir="/path/to/compiled", weights_dir="/path/to/FlashVSR-v1.1")

    # Run inference:
    result = run_inference(pipeline, input_video="/path/to/input.mp4", output_dir="/path/to/output")
"""

import os
import gc
import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from .modeling_flashvsr import (
    FlashVSRDiTConfig,
    NeuronFlashVSRDiT,
    precompute_freqs_cis_3d,
    build_rope_for_grid,
    HEAD_DIM,
    DIM,
    NUM_HEADS,
    PATCH_T,
    PATCH_H,
    PATCH_W,
    LCSA_WIN,
    IN_CHANNELS,
)


# ===================================================================
# Pipeline configuration
# ===================================================================


@dataclass
class FlashVSRPipelineConfig:
    """Configuration for the FlashVSR pipeline."""

    # Model paths
    weights_dir: str = ""
    compiled_dit_first: str = ""
    compiled_dit_stream: str = ""
    compiled_lq_proj: str = ""
    compiled_tcdecoder: str = ""
    prompt_path: str = ""

    # Resolution
    height: int = 768
    width: int = 1280
    scale: int = 4

    # Hardware
    tp_degree: int = 4

    # Pipeline options
    color_correction: str = "adain"  # "adain", "wavelet", or "none"
    max_chunks: int = 0  # 0 = process all chunks


# ===================================================================
# Input preparation utilities
# ===================================================================


def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def compute_scaled_and_target_dims(w0, h0, scale=4.0, multiple=128):
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple
    if tW == 0 or tH == 0:
        raise ValueError(f"Scaled size too small ({sW}x{sH}) for multiple={multiple}")
    return sW, sH, tW, tH


def upscale_then_center_crop(img, scale, tW, tH):
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def pil_to_tensor_neg1_1(img, dtype=torch.bfloat16, device="cpu"):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(
        device=device, dtype=torch.float32
    )
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def prepare_input_tensor(path, scale=4, dtype=torch.bfloat16, device="cpu"):
    """Load video and prepare bicubic-upscaled LQ input tensor.

    Returns:
        vid: (1, C, F, H, W) tensor in [-1, 1]
        tH, tW: target height/width
        F_count: number of frames (8n+1 format)
        fps: frames per second
    """
    import imageio

    rdr = imageio.get_reader(path)
    first = Image.fromarray(rdr.get_data(0)).convert("RGB")
    w0, h0 = first.size
    meta = {}
    try:
        meta = rdr.get_meta_data()
    except Exception:
        pass
    fps_val = meta.get("fps", 30)
    fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

    total = 0
    try:
        nf = meta.get("nframes", None)
        if isinstance(nf, int) and nf > 0:
            total = nf
    except Exception:
        pass
    if total <= 0:
        try:
            total = rdr.count_frames()
        except Exception:
            n = 0
            try:
                while True:
                    rdr.get_data(n)
                    n += 1
            except Exception:
                total = n

    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)

    idx = list(range(total)) + [total - 1] * 4
    F_count = largest_8n1_leq(len(idx))
    if F_count == 0:
        rdr.close()
        raise RuntimeError(f"Not enough frames: {len(idx)}")
    idx = idx[:F_count]

    frames = []
    try:
        for i in idx:
            img = Image.fromarray(rdr.get_data(i)).convert("RGB")
            img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
    finally:
        try:
            rdr.close()
        except Exception:
            pass

    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    return vid, tH, tW, F_count, fps


# ===================================================================
# Color correction
# ===================================================================


def _make_gaussian3x3_kernel(dtype, device):
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x, radius):
    N, C, H, W = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    pad = radius
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(
        x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=C
    )


def _wavelet_decompose(x, levels=5):
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        radius = 2**i
        blurred = _wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _calc_mean_std(feat, eps=1e-5):
    N, C = feat.shape[:2]
    var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def _adain(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def color_correct_wavelet(hq, lq, method="adain", levels=5, chunk_size=16):
    """Color correct HQ output using LQ reference. Both (B, C, f, H, W)."""
    B, C, f, H, W = hq.shape
    outs = []
    for start in range(0, f, chunk_size):
        end = min(start + chunk_size, f)
        hq_chunk = hq[:, :, start:end]
        lq_chunk = lq[:, :, start:end]
        bf = hq_chunk.shape[2]
        hq4 = hq_chunk.permute(0, 2, 1, 3, 4).reshape(B * bf, C, H, W)
        lq4 = lq_chunk.permute(0, 2, 1, 3, 4).reshape(B * bf, C, H, W)
        if method == "wavelet":
            from .pipeline import _wavelet_decompose

            c_high, _ = _wavelet_decompose(hq4, levels=levels)
            _, s_low = _wavelet_decompose(lq4, levels=levels)
            out4 = c_high + s_low
        elif method == "adain":
            out4 = _adain(hq4, lq4)
        else:
            raise ValueError(f"Unknown method: {method}")
        out4 = torch.clamp(out4, -1, 1)
        out_chunk = out4.reshape(B, bf, C, H, W).permute(0, 2, 1, 3, 4)
        outs.append(out_chunk)
    return torch.cat(outs, dim=2)


# ===================================================================
# Output utilities
# ===================================================================


def tensor2video(frames):
    """Convert (C, T, H, W) tensor in [-1,1] to list of PIL Images."""
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return [Image.fromarray(frame) for frame in frames]


def save_video(frames, save_path, fps=30, quality=5):
    """Save list of PIL images as video."""
    import imageio

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in frames:
        w.append_data(np.array(f))
    w.close()


# ===================================================================
# Neuron DiT forward wrapper
# ===================================================================


def neuron_dit_forward(
    app,
    base_freqs,
    cur_latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    height: int,
    width: int,
    cur_process_idx: int,
    lq_residual_0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run compiled Neuron DiT forward pass for one chunk.

    Args:
        app: FlashVSRApplication with loaded model
        base_freqs: precomputed RoPE frequencies
        cur_latents: (1, 16, f, H_lat, W_lat)
        encoder_hidden_states: (1, 512, 4096)
        height, width: target resolution
        cur_process_idx: chunk index (0 = first, >0 = stream)
        lq_residual_0: (1, S, 1536) or None

    Returns:
        noise_pred: (1, 16, f, H_lat, W_lat)
    """
    lat_h = height // 8
    lat_w = width // 8
    f = cur_latents.shape[2]

    post_f = f // PATCH_T
    post_h = lat_h // PATCH_H
    post_w = lat_w // PATCH_W
    seq_len = post_f * post_h * post_w

    # Temporal offset for RoPE
    temporal_offset = 0 if cur_process_idx == 0 else (4 + cur_process_idx * 2)

    rope_cos, rope_sin = build_rope_for_grid(
        *base_freqs,
        post_f,
        post_h,
        post_w,
        temporal_offset=temporal_offset,
    )

    # Block mask (all zeros = dense attention, Phase 1)
    num_q_blocks = (
        (post_f // LCSA_WIN[0]) * (post_h // LCSA_WIN[1]) * (post_w // LCSA_WIN[2])
    )
    attn_mask = torch.zeros(
        1, NUM_HEADS, num_q_blocks, num_q_blocks, dtype=torch.bfloat16
    )

    timestep = torch.tensor([1000.0], dtype=torch.bfloat16)

    # LQ residual
    if lq_residual_0 is not None:
        lq_input = lq_residual_0.to(dtype=torch.bfloat16)
    else:
        lq_input = torch.zeros(1, seq_len, DIM, dtype=torch.bfloat16)

    inputs = (
        cur_latents,
        timestep,
        encoder_hidden_states,
        rope_cos,
        rope_sin,
        attn_mask,
        lq_input,
    )

    with torch.no_grad():
        outputs = app(*inputs)

    return outputs[0]


# ===================================================================
# Pipeline class
# ===================================================================


@dataclass
class FlashVSRPipeline:
    """Loaded FlashVSR pipeline with compiled models ready for inference."""

    config: FlashVSRPipelineConfig
    dit_first_app: object = None
    dit_stream_app: object = None
    lq_proj_model: object = None
    tcdecoder_model: object = None
    tc_pixel_shuffle: object = None
    base_freqs: tuple = None
    prompt_emb: Optional[torch.Tensor] = None


# ===================================================================
# Compile pipeline
# ===================================================================


def compile_pipeline(
    weights_dir: str,
    output_dir: str,
    height: int = 768,
    width: int = 1280,
    tp_degree: int = 4,
):
    """Compile all FlashVSR pipeline components for Neuron.

    This function compiles:
    1. DiT (first chunk, f=6) via NxDI ModelBuilder
    2. DiT (stream chunk, f=2) via NxDI ModelBuilder
    3. LQ Projection via torch_neuronx.trace
    4. TCDecoder (sequential) via torch_neuronx.trace

    Args:
        weights_dir: Path to FlashVSR-v1.1 weights directory
        output_dir: Path to store compiled NEFFs
        height, width: Target output resolution
        tp_degree: Tensor parallel degree (default 4 for trn2.3xlarge)
    """
    from .modeling_flashvsr import (
        FlashVSRApplication,
        FlashVSRInferenceConfig,
    )
    from neuronx_distributed_inference.models.config import NeuronConfig

    os.makedirs(output_dir, exist_ok=True)

    # Compile DiT (first)
    dit_first_dir = os.path.join(output_dir, "dit_first")
    if not os.path.exists(dit_first_dir):
        os.makedirs(dit_first_dir, exist_ok=True)
        neuron_config = NeuronConfig(
            tp_degree=tp_degree,
            torch_dtype=torch.bfloat16,
            batch_size=1,
            save_sharded_checkpoint=True,
        )
        config = FlashVSRInferenceConfig(
            neuron_config=neuron_config,
            attn_mode="first",
            height=height,
            width=width,
        )
        app = FlashVSRApplication(model_path=weights_dir, config=config)
        app.compile(dit_first_dir)
        app.shard_weights(dit_first_dir)

    # Compile DiT (stream)
    dit_stream_dir = os.path.join(output_dir, "dit_stream")
    if not os.path.exists(dit_stream_dir):
        os.makedirs(dit_stream_dir, exist_ok=True)
        neuron_config = NeuronConfig(
            tp_degree=tp_degree,
            torch_dtype=torch.bfloat16,
            batch_size=1,
            save_sharded_checkpoint=True,
        )
        config = FlashVSRInferenceConfig(
            neuron_config=neuron_config,
            attn_mode="stream",
            height=height,
            width=width,
        )
        app = FlashVSRApplication(model_path=weights_dir, config=config)
        app.compile(dit_stream_dir)
        app.shard_weights(dit_stream_dir)


# ===================================================================
# Load pipeline
# ===================================================================


def load_pipeline(
    compiled_dir: str,
    weights_dir: str,
    prompt_path: str,
    tp_degree: int = 4,
    height: int = 768,
    width: int = 1280,
    tcdecoder_path: Optional[str] = None,
    lq_proj_path: Optional[str] = None,
) -> FlashVSRPipeline:
    """Load all compiled FlashVSR pipeline components.

    Args:
        compiled_dir: Path containing compiled NEFFs (dit_first/, dit_stream/)
        weights_dir: Path to FlashVSR-v1.1 weights directory
        prompt_path: Path to pre-computed text embedding (.pth)
        tp_degree: Tensor parallel degree
        height, width: Target output resolution
        tcdecoder_path: Path to compiled TCDecoder NEFF (.pt)
        lq_proj_path: Path to compiled LQ Projection NEFF (.pt)

    Returns:
        FlashVSRPipeline with all components loaded
    """
    import concurrent.futures
    from .modeling_flashvsr import FlashVSRApplication, FlashVSRInferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig

    config = FlashVSRPipelineConfig(
        weights_dir=weights_dir,
        compiled_dit_first=os.path.join(compiled_dir, "dit_first"),
        compiled_dit_stream=os.path.join(compiled_dir, "dit_stream"),
        compiled_lq_proj=lq_proj_path or "",
        compiled_tcdecoder=tcdecoder_path or "",
        prompt_path=prompt_path,
        height=height,
        width=width,
        tp_degree=tp_degree,
    )

    pipeline = FlashVSRPipeline(config=config)

    # Patch ThreadPoolExecutor for NxDI load
    original_init = concurrent.futures.ThreadPoolExecutor.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["max_workers"] = 1
        original_init(self, *args, **kwargs)

    concurrent.futures.ThreadPoolExecutor.__init__ = patched_init

    try:
        # Load LQ Projection (if available)
        if lq_proj_path and os.path.exists(lq_proj_path):
            import torch_neuronx  # noqa: F401

            pipeline.lq_proj_model = torch.jit.load(lq_proj_path)

        # Load DiT (first chunk)
        neuron_config = NeuronConfig(
            tp_degree=tp_degree,
            torch_dtype=torch.bfloat16,
            batch_size=1,
            save_sharded_checkpoint=True,
        )
        dit_first_config = FlashVSRInferenceConfig(
            neuron_config=neuron_config,
            attn_mode="first",
            height=height,
            width=width,
        )
        dit_first_app = FlashVSRApplication(
            model_path=weights_dir, config=dit_first_config
        )
        dit_first_app.load(config.compiled_dit_first)
        pipeline.dit_first_app = dit_first_app

        # Load DiT (stream)
        dit_stream_config = FlashVSRInferenceConfig(
            neuron_config=neuron_config,
            attn_mode="stream",
            height=height,
            width=width,
        )
        dit_stream_app = FlashVSRApplication(
            model_path=weights_dir, config=dit_stream_config
        )
        dit_stream_app.load(config.compiled_dit_stream)
        pipeline.dit_stream_app = dit_stream_app

        # Load TCDecoder (if available)
        if tcdecoder_path and os.path.exists(tcdecoder_path):
            import torch_neuronx  # noqa: F401

            pipeline.tcdecoder_model = torch.jit.load(tcdecoder_path)
            # Create pixel_shuffle for TCDecoder conditioning
            from .tcdecoder import TCPixelShuffle3d

            pipeline.tc_pixel_shuffle = TCPixelShuffle3d(4, 8, 8)

        # Precompute RoPE frequencies
        pipeline.base_freqs = precompute_freqs_cis_3d(HEAD_DIM)

        # Load prompt embedding
        prompt_emb = torch.load(prompt_path, map_location="cpu")
        if prompt_emb.dim() == 2:
            prompt_emb = prompt_emb.unsqueeze(0)
        pipeline.prompt_emb = prompt_emb.to(dtype=torch.bfloat16)

    finally:
        concurrent.futures.ThreadPoolExecutor.__init__ = original_init

    return pipeline


# ===================================================================
# Run inference
# ===================================================================


def run_inference(
    pipeline: FlashVSRPipeline,
    input_video: str,
    output_dir: str,
    scale: int = 4,
    max_chunks: int = 0,
    color_correction: str = "adain",
    save_mp4: bool = True,
) -> str:
    """Run FlashVSR inference on a video.

    Args:
        pipeline: Loaded FlashVSRPipeline
        input_video: Path to input video
        output_dir: Directory to save output
        scale: Upscaling factor (default 4)
        max_chunks: Maximum chunks to process (0 = all)
        color_correction: "adain", "wavelet", or "none"
        save_mp4: Whether to save as MP4

    Returns:
        Path to output video file
    """
    os.makedirs(output_dir, exist_ok=True)
    dtype = torch.bfloat16
    device = "cpu"

    # Step 1: Prepare input
    LQ_video, th, tw, num_frames, fps = prepare_input_tensor(
        input_video,
        scale=scale,
        dtype=dtype,
        device=device,
    )

    # Step 2: Run LQ Projection (pre-compute all tokens)
    all_lq_tokens = None
    tokens_per_frame = (th // 16) * (tw // 16)
    first_chunk_tokens = 6 * tokens_per_frame
    stream_chunk_tokens = 2 * tokens_per_frame

    if pipeline.lq_proj_model is not None:
        lq_input = LQ_video.to(dtype=torch.bfloat16)
        with torch.no_grad():
            _ = pipeline.lq_proj_model(lq_input)  # Warmup
            all_lq_tokens = pipeline.lq_proj_model(lq_input)
        # Free LQ NEFF from HBM
        del pipeline.lq_proj_model
        pipeline.lq_proj_model = None
        gc.collect()

    # Step 3: Streaming DiT inference
    process_total_num = (num_frames - 1) // 8 - 2
    if max_chunks > 0:
        process_total_num = min(process_total_num, max_chunks)

    noise = torch.randn(
        1, 16, (num_frames - 1) // 4, th // 8, tw // 8, dtype=dtype, device=device
    )
    latents = noise
    latents_total = []

    with torch.no_grad():
        for cur_process_idx in range(process_total_num):
            # Select current chunk latents
            if cur_process_idx == 0:
                cur_latents = latents[:, :, :6, :, :]
            else:
                cur_latents = latents[
                    :, :, 4 + cur_process_idx * 2 : 6 + cur_process_idx * 2, :, :
                ]

            # Get LQ residual for this chunk
            lq_residual = None
            if all_lq_tokens is not None:
                if cur_process_idx == 0:
                    lq_residual = all_lq_tokens[:, :first_chunk_tokens, :]
                else:
                    offset = (
                        first_chunk_tokens + (cur_process_idx - 1) * stream_chunk_tokens
                    )
                    lq_residual = all_lq_tokens[
                        :, offset : offset + stream_chunk_tokens, :
                    ]

            # Select DiT model
            active_app = (
                pipeline.dit_first_app
                if cur_process_idx == 0
                else pipeline.dit_stream_app
            )

            # Forward pass
            noise_pred = neuron_dit_forward(
                active_app,
                pipeline.base_freqs,
                cur_latents,
                pipeline.prompt_emb,
                th,
                tw,
                cur_process_idx,
                lq_residual_0=lq_residual,
            )

            # One-step denoising
            cur_latents = cur_latents - noise_pred
            latents_total.append(cur_latents)

    latents_out = torch.cat(latents_total, dim=2)

    # Step 4: TCDecoder
    if pipeline.tcdecoder_model is not None and pipeline.tc_pixel_shuffle is not None:
        from .tcdecoder import neuron_decode_video_sequential

        LQ_cur_idx = process_total_num * 8 + 21 if process_total_num > 0 else 21
        frames = neuron_decode_video_sequential(
            pipeline.tcdecoder_model,
            latents_out.transpose(1, 2),  # NCTHW -> NTCHW
            LQ_video[:, :, :LQ_cur_idx, :, :],
            pipeline.tc_pixel_shuffle,
            frames_to_trim=3,
        )
    else:
        raise RuntimeError("TCDecoder not loaded -- required for full pipeline")

    # Step 5: Color correction
    if color_correction != "none":
        lq_resized = F.interpolate(
            LQ_video[:, :, : frames.shape[2], :, :].reshape(-1, 3, th, tw),
            size=(frames.shape[3], frames.shape[4]),
            mode="bilinear",
            align_corners=False,
        ).reshape(1, 3, frames.shape[2], frames.shape[3], frames.shape[4])
        frames = color_correct_wavelet(frames, lq_resized, method=color_correction)

    # Step 6: Save output
    output_path = os.path.join(output_dir, "output.mp4")
    if save_mp4:
        pil_frames = tensor2video(frames[0])
        save_video(pil_frames, output_path, fps=fps)

    return output_path
