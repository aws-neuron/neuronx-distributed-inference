# coding=utf-8
# Copyright 2025 Google LLC. Ported to standalone PyTorch.
#
# Licensed under the Apache License, Version 2.0

"""
Gemma-4 Audio Encoder (Conformer) + Feature Extraction + Embedder.

Architecture:
    Audio waveform (16 kHz)
    → Mel spectrogram (128 bins)
    → Subsampling Conv (4x time reduction → [T/4, 1024])  [CPU]
    → 12 Conformer layers (FFN1 → ChunkedAttn → LightConv → FFN2)  [Neuron]
    → Output projection ([T/4, 1024] → [T/4, 1536])  [Neuron]
    → Multimodal Embedder (RMSNorm + Linear → [T/4, text_hidden_size])  [Neuron]

Split architecture for Neuron:
    - CPU: mel extraction, Conv2d subsampling, attention mask, position embeddings
    - Neuron: 12 Conformer layers + output_proj + embedder (compiled via torch_neuronx.trace)
"""

import math
import os
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class Gemma4AudioConfig:
    """Audio encoder configuration from config.json['audio_config']."""
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"
    output_proj_dims: int = 1536
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1e9
    rms_norm_eps: float = 1e-6
    subsampling_conv_channels: tuple = (128, 32)
    use_clipped_linears: bool = True
    gradient_clipping: float = 1e10

    @classmethod
    def from_dict(cls, d: dict) -> "Gemma4AudioConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ===========================================================================
# Feature Extraction (Mel Spectrogram)
# ===========================================================================


def _mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    """HTK-scale mel filter bank. Returns [num_frequency_bins, num_mel_filters]."""

    def _hz_to_mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = _hz_to_mel(min_frequency)
    mel_max = _hz_to_mel(max_frequency)
    mel_points = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    hz_points = _mel_to_hz(mel_points)

    freq_bins = np.arange(num_frequency_bins) * sampling_rate / ((num_frequency_bins - 1) * 2)

    filter_bank = np.zeros((num_frequency_bins, num_mel_filters), dtype=np.float64)
    for i in range(num_mel_filters):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]

        for j in range(num_frequency_bins):
            if lower <= freq_bins[j] <= center and center > lower:
                filter_bank[j, i] = (freq_bins[j] - lower) / (center - lower)
            elif center < freq_bins[j] <= upper and upper > center:
                filter_bank[j, i] = (upper - freq_bins[j]) / (upper - center)

    return filter_bank


def _window_function(window_length: int) -> np.ndarray:
    """Periodic Hann window."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(window_length) / window_length)


def _unfold(array: np.ndarray, size: int, step: int) -> np.ndarray:
    """Unfold last dimension of [B, T] array into [B, num_frames, size]."""
    batch_size, original_length = array.shape
    num_frames = (original_length - size) // step + 1
    if num_frames <= 0:
        return np.zeros((batch_size, 0, size), dtype=array.dtype)
    output_shape = (batch_size, num_frames, size)
    output_strides = (array.strides[0], array.strides[1] * step, array.strides[1])
    return np.lib.stride_tricks.as_strided(array, shape=output_shape, strides=output_strides)


def extract_mel_features(
    raw_audio: np.ndarray,
    sampling_rate: int = 16000,
    feature_size: int = 128,
    frame_length_ms: float = 20.0,
    hop_length_ms: float = 10.0,
    min_frequency: float = 0.0,
    max_frequency: float = 8000.0,
    mel_floor: float = 1e-3,
    max_length: int = 480000,
    pad_to_multiple_of: int = 128,
) -> tuple:
    """
    Extract log-mel spectrogram features from raw audio.

    Args:
        raw_audio: 1D numpy array of audio samples (float32, 16kHz)
    Returns:
        (input_features, input_features_mask) as numpy arrays
        input_features: [num_frames, 128] log-mel spectrogram
        input_features_mask: [num_frames] boolean mask (True = valid)
    """
    # Truncate if too long
    if len(raw_audio) > max_length:
        raw_audio = raw_audio[:max_length]

    frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))  # 320
    hop_length = int(round(sampling_rate * hop_length_ms / 1000.0))  # 160
    fft_length = 2 ** math.ceil(math.log2(frame_length))  # 512
    window = _window_function(frame_length).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mel_filters = _mel_filter_bank(
            num_frequency_bins=fft_length // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            sampling_rate=sampling_rate,
        )

    # Create attention mask for raw audio (1 = real, 0 = padding)
    num_samples = len(raw_audio)
    # Pad to multiple
    padded_length = ((num_samples + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    padded_audio = np.zeros(padded_length, dtype=np.float32)
    padded_audio[:num_samples] = raw_audio
    attention_mask = np.zeros(padded_length, dtype=np.float32)
    attention_mask[:num_samples] = 1.0

    # Semicausal padding: prepend frame_length // 2 zeros
    pad_left = frame_length // 2
    waveform = np.pad(padded_audio[np.newaxis, :], ((0, 0), (pad_left, 0)), mode="constant")
    attn_mask = np.pad(attention_mask, (pad_left, 0), mode="constant", constant_values=0)

    frame_size_for_unfold = frame_length + 1

    # Unfold into frames
    frames_to_process = _unfold(waveform, size=frame_size_for_unfold, step=hop_length)

    # No preemphasis (default 0.0 for Gemma4)
    frames = frames_to_process[..., :-1]

    # Window + FFT
    frames = frames * window
    stft = np.fft.rfft(frames, n=fft_length, axis=-1)
    magnitude_spec = np.abs(stft)

    # Mel spectrogram
    mel_spec = np.matmul(magnitude_spec, mel_filters)
    log_mel_spec = np.log(mel_spec + mel_floor)

    mel_spectrogram = log_mel_spec.squeeze(0)  # [num_frames, 128]
    num_mel_frames = mel_spectrogram.shape[0]

    # Frame-aware mask
    frame_end_indices = np.arange(num_mel_frames) * hop_length + frame_size_for_unfold - 1
    valid_indices = frame_end_indices < len(attn_mask)
    mask = np.zeros(num_mel_frames, dtype=bool)
    mask[valid_indices] = attn_mask[frame_end_indices[valid_indices]].astype(bool)

    # Apply mask to spectrogram (zero out padding frames)
    mel_spectrogram = mel_spectrogram * mask[:, np.newaxis].astype(np.float32)

    return mel_spectrogram.astype(np.float32), mask


def compute_audio_num_tokens(num_samples: int, sampling_rate: int = 16000) -> int:
    """Compute number of audio tokens after subsampling (matching HF processor)."""
    frame_length = int(round(sampling_rate * 20.0 / 1000.0))  # 320
    hop_length = int(round(sampling_rate * 10.0 / 1000.0))  # 160
    frame_size = frame_length + 1  # 321
    pad_left = frame_length // 2  # 160

    padded = num_samples + pad_left
    num_mel_frames = max(0, (padded - frame_size) // hop_length + 1)

    # Two Conv2d layers (kernel=3, stride=2, padding=1)
    t = num_mel_frames
    for _ in range(2):
        t = (t + 2 * 1 - 3) // 2 + 1
    return t


# ===========================================================================
# Audio Encoder Components
# ===========================================================================


class Gemma4RMSNorm(nn.Module):
    """RMSNorm matching HF Gemma4 style."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


class ClippableLinear(nn.Module):
    """Linear with optional input/output clamping (for clipped linears)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 use_clipping: bool = True):
        super().__init__()
        self.use_clipping = use_clipping
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_clipping:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clipping:
            x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipping:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x


class AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal relative position encoding for chunked attention."""

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.context_size = (
            config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        )
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales) * -log_timescale_increment
        )
        self.register_buffer("inv_timescales", inv_timescales.unsqueeze(0).unsqueeze(0), persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Position ids: 12, 11, 10, ..., 1, 0 (descending)
        position_ids = torch.arange(12, -1, -1, device=hidden_states.device)
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales.to(device=hidden_states.device)
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.to(dtype=hidden_states.dtype)


class AudioAttention(nn.Module):
    """Chunked local attention with relative position bias and softcapping."""

    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon
        self.attention_invalid_logits_value = config.attention_invalid_logits_value

        self.q_scale = (self.head_dim ** -0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)

        self.q_proj = ClippableLinear(config.hidden_size, config.hidden_size, use_clipping=config.use_clipped_linears)
        self.k_proj = ClippableLinear(config.hidden_size, config.hidden_size, use_clipping=config.use_clipped_linears)
        self.v_proj = ClippableLinear(config.hidden_size, config.hidden_size, use_clipping=config.use_clipped_linears)
        self.post = ClippableLinear(config.hidden_size, config.hidden_size, use_clipping=config.use_clipped_linears)

        self.relative_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))
        self.register_buffer("softcap", torch.tensor(config.attention_logit_cap), persistent=False)

    def _convert_to_block(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, H, D] → [B, num_blocks, chunk_size, H, D]"""
        B, S, H, D = x.shape
        num_blocks = (S + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - S
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x.reshape(B, num_blocks, self.chunk_size, H, D).contiguous()

    def _extract_block_context(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, H, D] → [B, num_blocks, context_size, H, D]

        Replaces unfold with gather for Neuron compatibility.
        """
        B, S, H, D = x.shape
        x = F.pad(x, (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1))
        padded_len = x.shape[1]
        num_blocks = (S + self.chunk_size - 1) // self.chunk_size

        # Build gather indices: [num_blocks, context_size]
        block_starts = torch.arange(num_blocks, device=x.device) * self.chunk_size
        offsets = torch.arange(self.context_size, device=x.device)
        indices = block_starts[:, None] + offsets[None, :]  # [num_blocks, context_size]

        # Gather: x_padded[B, indices, H, D] → [B, num_blocks, context_size, H, D]
        idx = indices.view(1, -1, 1, 1).expand(B, -1, H, D)  # [B, num_blocks*ctx, H, D]
        gathered = x.gather(1, idx)  # [B, num_blocks*ctx, H, D]
        return gathered.view(B, num_blocks, self.context_size, H, D).contiguous()

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Relative position shift for blocked attention."""
        B, H, num_blocks, block_size, pos_len = x.shape
        ctx = self.context_size
        x = F.pad(x, (0, ctx + 1 - pos_len))
        x = x.view(B, H, num_blocks, block_size * (ctx + 1))
        x = x[..., :block_size * ctx]
        return x.view(B, H, num_blocks, block_size, ctx)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        B, S, _ = hidden_states.shape
        shape = (B, S, self.num_heads, self.head_dim)

        q = self.q_proj(hidden_states).float().view(shape)
        k = self.k_proj(hidden_states).float().view(shape)
        v = self.v_proj(hidden_states).float().view(shape)

        q = q * self.q_scale * F.softplus(self.per_dim_scale)
        k = k * self.k_scale

        q_blocked = self._convert_to_block(q)
        k_ctx = self._extract_block_context(k)
        v_ctx = self._extract_block_context(v)
        num_blocks = q_blocked.shape[1]

        rel_k = self.relative_k_proj(position_embeddings)
        rel_k = rel_k.view(-1, self.num_heads, self.head_dim).to(dtype=q.dtype)

        # Content attention: Q @ K^T
        queries = q_blocked.permute(0, 3, 1, 2, 4)  # [B, H, blocks, chunk, D]
        matrix_ac = queries @ k_ctx.permute(0, 3, 1, 4, 2)  # [B, H, blocks, chunk, ctx]

        # Relative position attention
        queries_flat = queries.reshape(B, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ rel_k.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(B, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd

        # Softcapping
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.logical_not(), self.attention_invalid_logits_value)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v_ctx.dtype)
        attn_output = attn_weights @ v_ctx.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(B, num_blocks * self.chunk_size, -1)
        attn_output = attn_output[:, :S].contiguous()
        attn_output = self.post(attn_output.to(dtype=self.post.linear.weight.dtype))
        return attn_output


class AudioFeedForward(nn.Module):
    """Conformer FFN with pre/post norms and residual scaling."""

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.ffw_layer_1 = ClippableLinear(config.hidden_size, config.hidden_size * 4,
                                           use_clipping=config.use_clipped_linears)
        self.ffw_layer_2 = ClippableLinear(config.hidden_size * 4, config.hidden_size,
                                           use_clipping=config.use_clipped_linears)
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.act_fn = nn.SiLU()
        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gc = min(self.gradient_clipping, torch.finfo(self.ffw_layer_1.linear.weight.dtype).max)
        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)
        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states *= self.post_layer_scale
        hidden_states += residual
        return hidden_states


class AudioCausalConv1d(nn.Module):
    """Causal depthwise 1D convolution with left-padding.

    Uses manual shifted-multiply implementation instead of nn.Conv1d
    for Neuron compatibility (NCC has a bug with depthwise Conv1d).
    Weight layout matches nn.Conv1d(groups=channels): [C, 1, K].
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        assert groups == in_channels == out_channels, "Only depthwise supported"
        assert not bias, "bias not supported"
        self.in_channels = in_channels
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.stride = (1,)
        self.dilation = (1,)
        self.groups = groups
        # Weight shape: [out_channels, 1, kernel_size] (same as nn.Conv1d depthwise)
        self.weight = nn.Parameter(torch.empty(out_channels, 1, self.kernel_size[0]))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @cached_property
    def left_pad(self):
        effective_kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = F.pad(x, (self.left_pad, 0))
        # Manual depthwise conv: sum of shifted element-wise multiplications
        K = self.kernel_size[0]
        T = x.shape[2] - K + 1
        w = self.weight.squeeze(1)  # [C, K]
        out = torch.zeros(x.shape[0], self.in_channels, T, dtype=x.dtype, device=x.device)
        for k in range(K):
            out = out + x[:, :, k:k + T] * w[:, k].unsqueeze(0).unsqueeze(2)
        return out


class AudioLightConv1d(nn.Module):
    """Lightweight 1D convolution block with GLU gating."""

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.linear_start = ClippableLinear(config.hidden_size, config.hidden_size * 2,
                                            use_clipping=config.use_clipped_linears)
        self.linear_end = ClippableLinear(config.hidden_size, config.hidden_size,
                                          use_clipping=config.use_clipped_linears)
        self.depthwise_conv1d = AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.act_fn = nn.SiLU()
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = F.glu(hidden_states, dim=-1)
        hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)
        gc = min(self.gradient_clipping, torch.finfo(self.linear_start.linear.weight.dtype).max)
        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states += residual
        return hidden_states


class AudioSubSampleConvLayer(nn.Module):
    """Single 2D subsampling conv layer (stride=2, kernel=3x3)."""

    def __init__(self, in_channels: int, out_channels: int, norm_eps: float):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps, elementwise_affine=True, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            mask = mask.to(device=x.device)
            x = x * mask[:, None, :, None]
        x = self.conv(x.to(self.conv.weight.dtype))
        x = self.act(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        if mask is not None:
            mask = mask[:, ::2]
        return x, mask


class AudioSubSampleConvProjection(nn.Module):
    """Two-stage subsampling: 2x Conv2d (4x time reduction) + linear projection."""

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        channels = config.subsampling_conv_channels  # [128, 32]
        self.layer0 = AudioSubSampleConvLayer(1, channels[0], config.rms_norm_eps)
        self.layer1 = AudioSubSampleConvLayer(channels[0], channels[1], config.rms_norm_eps)
        proj_input_dim = (channels[0] // 4) * channels[1]  # (128//4)*32 = 1024
        self.input_proj_linear = nn.Linear(proj_input_dim, config.hidden_size, bias=False)

    def forward(self, input_features: torch.Tensor,
                input_features_mask: Optional[torch.Tensor] = None):
        x = input_features.unsqueeze(1)  # [B, 1, T, 128]
        x, mask = self.layer0(x, input_features_mask)
        x, mask = self.layer1(x, mask)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, T, -1)
        return self.input_proj_linear(x), mask


class AudioConformerLayer(nn.Module):
    """Conformer layer: FFN1 → Attention → LightConv1D → FFN2 → Norm."""

    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.feed_forward1 = AudioFeedForward(config)
        self.feed_forward2 = AudioFeedForward(config)
        self.self_attn = AudioAttention(config, layer_idx)
        self.lconv1d = AudioLightConv1d(config)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_out = Gemma4RMSNorm(config.hidden_size)
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor],
                position_embeddings: torch.Tensor) -> torch.Tensor:
        gc = min(self.gradient_clipping, torch.finfo(self.norm_pre_attn.weight.dtype).max)

        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = self.norm_pre_attn(hidden_states)

        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)

        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states += residual

        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states


# ===========================================================================
# Top-level Audio Encoder
# ===========================================================================


class Gemma4AudioEncoder(nn.Module):
    """
    Complete Gemma4 audio encoder (Conformer).

    Input: mel spectrogram [B, T, 128]
    Output: audio features [B, T/4, output_proj_dims] and validity mask [B, T/4]
    """

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config
        self.subsample_conv_projection = AudioSubSampleConvProjection(config)
        self.rel_pos_enc = AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [AudioConformerLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.output_proj = nn.Linear(config.hidden_size, config.output_proj_dims, bias=True)

    def _build_attention_mask(self, hidden_states: torch.Tensor,
                              output_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Build 5D blocked attention mask for chunked local attention."""
        B, S, _ = hidden_states.shape
        chunk_size = self.config.attention_chunk_size
        max_past = self.config.attention_context_left - 1
        max_future = self.config.attention_context_right
        device = hidden_states.device

        if output_mask is None:
            return None

        # Build 4D bidirectional mask [B, 1, S, S]
        mask_1d = output_mask.float()  # [B, S]
        mask_4d = mask_1d.unsqueeze(1).unsqueeze(2) * mask_1d.unsqueeze(1).unsqueeze(3)  # [B, 1, S, S]

        # Add sliding window constraint
        row_idx = torch.arange(S, device=device).unsqueeze(1)
        col_idx = torch.arange(S, device=device).unsqueeze(0)
        window_mask = (col_idx > row_idx - max_past) & (col_idx <= row_idx + max_future)
        mask_4d = mask_4d * window_mask.float().unsqueeze(0).unsqueeze(0)
        mask_4d = mask_4d.bool()

        # Convert to 5D blocked format
        num_blocks = (S + chunk_size - 1) // chunk_size
        padded_S = num_blocks * chunk_size
        pad = padded_S - S

        mask_4d = F.pad(mask_4d, (0, pad, 0, pad), value=False)
        mask_5d = mask_4d.reshape(B, 1, num_blocks, chunk_size, padded_S)
        mask_5d = F.pad(mask_5d, (max_past, max_future), value=False)

        block_starts = torch.arange(num_blocks, device=device) * chunk_size
        offsets = torch.arange(chunk_size + max_past + max_future, device=device)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(B, 1, -1, chunk_size, -1)

        return mask_5d.gather(-1, kv_indices)

    @torch.no_grad()
    def forward(self, input_features: torch.Tensor,
                input_features_mask: Optional[torch.Tensor] = None):
        """
        Args:
            input_features: [B, T, 128] mel spectrogram
            input_features_mask: [B, T] boolean mask (True = valid frame)
        Returns:
            (hidden_states, output_mask):
              hidden_states: [B, T/4, output_proj_dims]
              output_mask: [B, T/4] boolean mask
        """
        hidden_states, output_mask = self.subsample_conv_projection(input_features, input_features_mask)
        position_embeddings = self.rel_pos_enc(hidden_states)
        attention_mask = self._build_attention_mask(hidden_states, output_mask)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_embeddings)

        hidden_states = self.output_proj(hidden_states)
        return hidden_states, output_mask


class Gemma4AudioEmbedder(nn.Module):
    """Projects audio encoder output into text embedding space."""

    def __init__(self, audio_hidden_size: int, text_hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_pre_projection_norm = Gemma4RMSNorm(audio_hidden_size, eps=eps, with_scale=False)
        self.embedding_projection = nn.Linear(audio_hidden_size, text_hidden_size, bias=False)
        self.embedding_post_projection_norm = Gemma4RMSNorm(text_hidden_size, eps=eps, with_scale=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_pre_projection_norm(x)
        x = self.embedding_projection(x)
        return self.embedding_post_projection_norm(x)


# ===========================================================================
# Neuron-compiled Audio Encoder
# ===========================================================================


class ConformerForNeuron(nn.Module):
    """
    Conformer layers + output_proj + embedder for Neuron tracing.

    Subsampling, attention mask, and position embeddings are pre-computed on CPU.
    This module only contains the heavy computation (12 Conformer layers + projections).
    """

    def __init__(self, encoder: "Gemma4AudioEncoder", embedder: "Gemma4AudioEmbedder"):
        super().__init__()
        self.layers = encoder.layers
        self.output_proj = encoder.output_proj
        self.embedder_norm = embedder.embedding_pre_projection_norm
        self.embedder_proj = embedder.embedding_projection
        self.embedder_post_norm = embedder.embedding_post_projection_norm

    def forward(self, hidden_states, attention_mask, position_embeddings):
        """
        Args:
            hidden_states: [B, S, 1024] subsampled features (from CPU)
            attention_mask: [B, 1, num_blocks, chunk_size, context_size] pre-computed
            position_embeddings: [B, 13, 1024] pre-computed
        Returns:
            audio_embeds: [B, S, text_hidden_size]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_embeddings)

        hidden_states = self.output_proj(hidden_states)
        hidden_states = self.embedder_norm(hidden_states)
        hidden_states = self.embedder_proj(hidden_states)
        hidden_states = self.embedder_post_norm(hidden_states)

        return hidden_states


class NeuronAudioEncoder:
    """
    Hybrid CPU+Neuron audio encoder.

    CPU: mel extraction → Conv2d subsampling → attention mask → position embeddings
    Neuron: 12 Conformer layers + output_proj + embedder

    Usage:
        neuron_encoder = NeuronAudioEncoder.from_pretrained(model_path, config, text_hidden_size)
        neuron_encoder.compile(save_path)  # or neuron_encoder.load(save_path)
        audio_embeddings, valid_mask = neuron_encoder(mel_features, mel_mask)
    """

    def __init__(self, encoder: Gemma4AudioEncoder, embedder: Gemma4AudioEmbedder,
                 config: Gemma4AudioConfig, target_mel_len: int = 2048):
        self.encoder = encoder  # CPU (for subsampling + mask + pos)
        self.embedder = embedder
        self.config = config
        self.target_mel_len = target_mel_len
        self.conformer = ConformerForNeuron(encoder, embedder)
        self.conformer = self.conformer.to(torch.bfloat16).eval()
        self.traced_model = None

    @classmethod
    def from_pretrained(cls, model_path: str, config: Gemma4AudioConfig,
                        text_hidden_size: int, target_mel_len: int = 2048):
        """Load weights and create hybrid encoder."""
        encoder, embedder = load_audio_encoder(model_path, config, text_hidden_size)
        return cls(encoder, embedder, config, target_mel_len)

    def _pad_mel(self, mel_features: np.ndarray, mel_mask: np.ndarray):
        """Pad/truncate mel to fixed length for Neuron."""
        target = self.target_mel_len
        if mel_features.shape[0] < target:
            pad_len = target - mel_features.shape[0]
            mel_features = np.pad(mel_features, ((0, pad_len), (0, 0)), mode='constant')
            mel_mask = np.pad(mel_mask, (0, pad_len), mode='constant', constant_values=False)
        elif mel_features.shape[0] > target:
            mel_features = mel_features[:target]
            mel_mask = mel_mask[:target]
        return mel_features, mel_mask

    def _cpu_preprocess(self, mel_features: np.ndarray, mel_mask: np.ndarray):
        """Run CPU part: pad, subsample, compute mask and pos embeddings."""
        mel_features, mel_mask = self._pad_mel(mel_features, mel_mask)

        mel_tensor = torch.from_numpy(mel_features).unsqueeze(0).to(torch.bfloat16)
        mask_tensor = torch.from_numpy(mel_mask).unsqueeze(0)

        with torch.no_grad():
            sub_hidden, output_mask = self.encoder.subsample_conv_projection(mel_tensor, mask_tensor)
            position_embeddings = self.encoder.rel_pos_enc(sub_hidden)
            attention_mask = self.encoder._build_attention_mask(sub_hidden, output_mask)

        return sub_hidden, attention_mask, position_embeddings, output_mask

    def compile(self, save_path: str):
        """Compile Conformer for Neuron and save."""
        import torch_neuronx

        # Create example inputs from a dummy mel
        dummy_mel = np.zeros((self.target_mel_len, 128), dtype=np.float32)
        dummy_mask = np.ones(self.target_mel_len, dtype=bool)
        sub_hidden, attn_mask, pos_emb, _ = self._cpu_preprocess(dummy_mel, dummy_mask)

        print(f"  Tracing ConformerForNeuron: hidden={sub_hidden.shape}, "
              f"attn={attn_mask.shape}, pos={pos_emb.shape}")

        self.traced_model = torch_neuronx.trace(
            self.conformer,
            (sub_hidden, attn_mask, pos_emb),
            compiler_args=["--target", "trn2"],
        )

        os.makedirs(save_path, exist_ok=True)
        self.traced_model.save(os.path.join(save_path, "model.pt"))
        print(f"  Saved compiled audio encoder to {save_path}")

    def load(self, save_path: str):
        """Load pre-compiled Neuron model."""
        import torch_neuronx
        self.traced_model = torch.jit.load(os.path.join(save_path, "model.pt"))

    def __call__(self, mel_features: np.ndarray, mel_mask: np.ndarray):
        """
        Run full audio encoding: CPU preprocess → Neuron conformer.

        Args:
            mel_features: [T, 128] mel spectrogram (numpy)
            mel_mask: [T] boolean mask (numpy)
        Returns:
            (audio_embeddings, valid_mask):
                audio_embeddings: [1, S, text_hidden_size] tensor
                valid_mask: [S] boolean tensor (True = valid audio token)
        """
        if self.traced_model is None:
            raise RuntimeError("Call compile() or load() before inference")

        sub_hidden, attn_mask, pos_emb, output_mask = self._cpu_preprocess(mel_features, mel_mask)

        with torch.no_grad():
            audio_embeddings = self.traced_model(sub_hidden, attn_mask, pos_emb)

        valid_mask = output_mask[0] if output_mask is not None else None
        return audio_embeddings, valid_mask


# ===========================================================================
# Weight Loading
# ===========================================================================


def load_audio_encoder(model_path: str, config: Gemma4AudioConfig,
                       text_hidden_size: int, dtype=torch.bfloat16):
    """
    Load audio encoder + embedder from HF checkpoint.

    Args:
        model_path: path to HF model (e.g., ~/models/gemma-4-E2B-it)
        config: Gemma4AudioConfig
        text_hidden_size: text decoder hidden size (for embedder)
        dtype: weight dtype

    Returns:
        (encoder, embedder) tuple
    """
    import glob
    from safetensors import safe_open

    encoder = Gemma4AudioEncoder(config)
    embedder = Gemma4AudioEmbedder(config.output_proj_dims, text_hidden_size, config.rms_norm_eps)

    # Collect all audio weights from safetensors
    audio_weights = {}
    files = sorted(glob.glob(f"{model_path}/model*.safetensors"))
    for f in files:
        with safe_open(f, framework="pt") as st:
            for key in st.keys():
                if "audio_tower" in key or "embed_audio" in key:
                    audio_weights[key] = st.get_tensor(key)

    # Map HF keys to our module structure
    # HF: model.audio_tower.{module_path} → encoder.{module_path}
    encoder_sd = {}
    embedder_sd = {}

    for hf_key, tensor in audio_weights.items():
        if "model.audio_tower." in hf_key:
            local_key = hf_key.replace("model.audio_tower.", "")
            encoder_sd[local_key] = tensor
        elif "model.embed_audio." in hf_key:
            local_key = hf_key.replace("model.embed_audio.", "")
            embedder_sd[local_key] = tensor

    # Load encoder weights (handle ClippableLinear → linear.weight mapping)
    missing_enc, unexpected_enc = encoder.load_state_dict(encoder_sd, strict=False)
    if missing_enc:
        print(f"  Audio encoder missing keys ({len(missing_enc)}): {missing_enc[:5]}...")
    if unexpected_enc:
        print(f"  Audio encoder unexpected keys ({len(unexpected_enc)}): {unexpected_enc[:5]}...")

    # Load embedder weights
    missing_emb, unexpected_emb = embedder.load_state_dict(embedder_sd, strict=False)
    if missing_emb:
        print(f"  Audio embedder missing keys ({len(missing_emb)}): {missing_emb[:5]}...")
    if unexpected_emb:
        print(f"  Audio embedder unexpected keys ({len(unexpected_emb)}): {unexpected_emb[:5]}...")

    encoder = encoder.to(dtype=dtype).eval()
    embedder = embedder.to(dtype=dtype).eval()

    return encoder, embedder
