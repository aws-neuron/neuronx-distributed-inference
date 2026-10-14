# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shrutam-2: Multilingual Indic ASR on AWS Neuron (Trainium2).

Three-stage pipeline for automatic speech recognition across 12 Indian languages:
  Stage 1: Conformer encoder (607.7M params, FP32 -> BF16 via auto-cast)
           CPU mel preprocessing + Neuron-traced Conformer + downsampler
  Stage 2: SMEAR-MoE projector (50.4M params, FP32 -> BF16 via auto-cast)
           Neuron-traced utterance-level soft MoE (8 experts)
  Stage 3: LLM decoder (1.2B params, BF16) via NxD Inference
           LlamaForCausalLM with audio embedding scatter via ImageToTextModelWrapper

Architecture:
  Audio -> MelSpectrogram (CPU) -> Conformer(24 layers) -> Downsampler(Conv1d)
  -> SMEAR-MoE(8 experts, soft routing) -> [B, T, 2048] audio embeddings
  -> scatter into LLM input -> Llama(16 layers, 2048 hidden, GQA 4:1) -> text

Reference: https://arxiv.org/abs/2601.19451
Weights: https://huggingface.co/bharatgenai/Shrutam-2
License: BharatGen Non-Commercial License
"""

import json
import logging
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Audio config
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 160
ENCODER_DIM = 1024
LLM_DIM = 2048
AUDIO_PLACEHOLDER_ID = 0  # pad_token_id used as audio placeholder

PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


# ============================================================
# Audio Preprocessing (CPU)
# ============================================================


class MelSpectrogramPreprocessor(nn.Module):
    """Log-mel spectrogram feature extractor.

    Matches the preprocessing from Shrutam-2's SpeechEncoder:
    NeMo-style mel spectrogram with preemphasis and per-feature normalization.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        preemph=0.97,
        normalize="per_feature",
        log_zero_guard=2**-24,
    ):
        super().__init__()
        self.preemph = preemph
        self.normalize_type = normalize
        self.log_zero_guard = log_zero_guard
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.register_buffer("window", torch.hann_window(win_length))

        n_freqs = n_fft // 2 + 1
        f_max = sample_rate / 2.0
        mel_low = 2595.0 * math.log10(1.0 + 0.0 / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + f_max / 700.0)
        mel_points = torch.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        freq_bins = torch.linspace(0.0, f_max, n_freqs)

        fb = torch.zeros(n_freqs, n_mels)
        for i in range(n_mels):
            lo, mid, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            fb[:, i] = torch.clamp(
                torch.minimum(
                    (freq_bins - lo) / (mid - lo + 1e-10),
                    (hi - freq_bins) / (hi - mid + 1e-10),
                ),
                min=0.0,
            )
        self.register_buffer("fb", fb)

    def forward(self, input_signal, length):
        x = input_signal
        if self.preemph and self.preemph > 0.0:
            x = torch.cat([x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]], dim=1)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window.size(0),
            window=self.window,
            center=True,
            return_complex=True,
        )

        power = spec.abs().pow(2)
        mel = torch.matmul(power.transpose(1, 2), self.fb).transpose(1, 2)
        log_mel = torch.log(mel + self.log_zero_guard)

        out_length = torch.div(length, self.hop_length, rounding_mode="floor") + 1

        if self.normalize_type == "per_feature":
            mean = log_mel.mean(dim=-1, keepdim=True)
            std = log_mel.std(dim=-1, keepdim=True)
            log_mel = (log_mel - mean) / (std + 1e-5)

        return log_mel, out_length


# ============================================================
# Conformer Encoder Components
# ============================================================


class ConvSubsampling(nn.Module):
    """Convolutional subsampling: 8x time reduction, mel -> d_model."""

    def __init__(self, n_mel=80, d_model=1024, channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
        )
        self.out = nn.Linear(channels * (n_mel // 8), d_model)

    def forward(self, x, lengths):
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv(x)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        x = self.out(x)
        for _ in range(3):
            lengths = (lengths + 1) // 2
        return x, lengths


class RelativePositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding."""

    def __init__(self, d_model=1024, max_len=5000):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(2 * max_len - 1, d_model)
        positions = torch.arange(0, 2 * max_len - 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len):
        start = self.max_len - seq_len
        end = self.max_len + seq_len - 1
        return self.pe[:, start:end]


class MultiHeadRelPositionAttention(nn.Module):
    """Multi-head attention with Transformer-XL relative positional encoding."""

    def __init__(self, d_model=1024, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model)

        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_head))

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def _rel_shift(self, x):
        B, H, T, P = x.size()
        x = F.pad(x, (1, 0))
        x = x.view(B, H, P + 1, T)
        x = x[:, :, 1:, :].view(B, H, T, P)
        return x[:, :, :, :T]

    def forward(self, x, pos_emb, mask=None):
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        p = (
            self.linear_pos(pos_emb)
            .view(1, -1, self.n_heads, self.d_head)
            .transpose(1, 2)
        )

        q_with_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))

        q_with_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        pos_score = torch.matmul(q_with_v, p.transpose(-2, -1))
        pos_score = self._rel_shift(pos_score)

        scores = (content_score + pos_score) * self.scale

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.linear_out(attn_out)


class FeedForwardModule(nn.Module):
    """Position-wise feed-forward with SiLU activation."""

    def __init__(self, d_model=1024, ff_dim=4096, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.dropout(F.silu(self.linear1(x)))))


class ConvolutionModule(nn.Module):
    """Conformer convolution: pointwise -> GLU -> depthwise -> act -> pointwise."""

    def __init__(self, d_model=1024, kernel_size=9, dropout=0.1):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if mask is not None:
            x = x * mask.unsqueeze(1).float()
        x = self.depthwise_conv(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return self.dropout(x.transpose(1, 2))


class ConformerBlock(nn.Module):
    """Conformer block: FF -> Attn -> Conv -> FF (Macaron-style)."""

    def __init__(
        self, d_model=1024, n_heads=8, ff_dim=4096, conv_kernel_size=9, dropout=0.1
    ):
        super().__init__()
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1 = FeedForwardModule(d_model, ff_dim, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadRelPositionAttention(d_model, n_heads, dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2 = FeedForwardModule(d_model, ff_dim, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, mask=None):
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        x = x + self.dropout(self.attn(self.norm_attn(x), pos_emb, mask))
        x = x + self.conv(self.norm_conv(x), mask)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        return self.norm_out(x)


class ConformerEncoder(nn.Module):
    """Conformer speech encoder (24 layers, 607.7M params)."""

    def __init__(
        self,
        d_model=1024,
        n_heads=8,
        ff_dim=4096,
        conv_kernel_size=9,
        n_layers=24,
        n_mel=80,
        sub_channels=256,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.subsampling = ConvSubsampling(n_mel, d_model, sub_channels)
        self.pos_enc = RelativePositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(d_model, n_heads, ff_dim, conv_kernel_size, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, audio_signal, lengths):
        x, out_lengths = self.subsampling(audio_signal, lengths)
        x = x * self.scale
        T = x.size(1)
        pos_emb = self.pos_enc(T)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < out_lengths.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, pos_emb, mask)
        return x.transpose(1, 2), out_lengths


# ============================================================
# Trace-Friendly Wrappers
# ============================================================


class ConformerEncoderForTrace(nn.Module):
    """Trace-friendly Conformer wrapper for torch_neuronx.trace().

    Takes fixed-length mel spectrogram (zero-padded) with actual mel_lengths
    for proper attention masking. Includes the Conv1d downsampler.
    Returns [B, T', d_model].
    """

    def __init__(self, encoder: ConformerEncoder, downsampler: nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.downsampler = downsampler

    def forward(
        self, audio_signal: torch.Tensor, mel_lengths: torch.Tensor
    ) -> torch.Tensor:
        x, out_lengths = self.encoder.subsampling(audio_signal, mel_lengths)
        x = x * self.encoder.scale
        T = x.size(1)
        pos_emb = self.encoder.pos_enc(T)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < out_lengths.unsqueeze(1)
        for layer in self.encoder.layers:
            x = layer(x, pos_emb, mask=mask)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x


class EncoderDownsamplerConv1d(nn.Module):
    """Conv1d-based frame rate reduction (from asr_model.py)."""

    def __init__(self, encoder_dim=1024, ds_rate=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            encoder_dim, encoder_dim, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            encoder_dim, encoder_dim, kernel_size=ds_rate, stride=ds_rate, padding=0
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        return x


# ============================================================
# SMEAR-MoE Projector
# ============================================================


class EncoderProjectorLinear(nn.Module):
    """Single SMEAR expert: 2-layer MLP (1024 -> 2048 -> 2048)."""

    def __init__(self, encoder_dim=1024, llm_dim=2048):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(encoder_dim, 2048)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(2048, llm_dim)

    def forward(self, x):
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class MoELayer_SMEAR(nn.Module):
    """SMEAR: utterance-level soft MoE routing with weight merging.

    Routes via utterance-mean router probabilities merged into expert weights
    using einsum. 8 experts with soft routing (no top-k gating).
    """

    def __init__(self, experts: nn.ModuleList, input_dim: int):
        super().__init__()
        self.experts = experts
        self.num_experts = len(experts)
        self.input_dim = input_dim
        self.router = nn.Linear(input_dim, self.num_experts)

    def forward(self, x, mask=None):
        B, L, D_in = x.shape
        E = self.num_experts

        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        if mask is not None:
            mask = mask.to(device=x.device, dtype=x.dtype)
            router_probs = router_probs * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            denom = float(L)

        if mask is not None:
            utterance_probs = router_probs.sum(dim=1) / denom
        else:
            utterance_probs = router_probs.mean(dim=1)

        W1 = torch.stack([e.linear1.weight for e in self.experts], dim=0)
        b1 = torch.stack([e.linear1.bias for e in self.experts], dim=0)
        W2 = torch.stack([e.linear2.weight for e in self.experts], dim=0)
        b2 = torch.stack([e.linear2.bias for e in self.experts], dim=0)

        merged_W1 = torch.einsum("be,ehi->bhi", utterance_probs, W1)
        merged_b1 = torch.einsum("be,eh->bh", utterance_probs, b1)
        merged_W2 = torch.einsum("be,eoi->boi", utterance_probs, W2)
        merged_b2 = torch.einsum("be,eo->bo", utterance_probs, b2)

        merged_W1_T = merged_W1.transpose(1, 2)
        hidden = torch.einsum("bld,bdh->blh", x, merged_W1_T) + merged_b1.unsqueeze(1)
        hidden = F.relu(hidden)

        merged_W2_T = merged_W2.transpose(1, 2)
        out = torch.einsum("blh,bho->blo", hidden, merged_W2_T) + merged_b2.unsqueeze(1)

        return out, None, None


class SMEARForTrace(nn.Module):
    """Trace-friendly SMEAR wrapper: returns only output tensor."""

    def __init__(self, smear: MoELayer_SMEAR):
        super().__init__()
        self.smear = smear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.smear(x, mask=None)
        return out


# ============================================================
# Weight Loading
# ============================================================


def load_encoder_weights(encoder_weights_path: str, device: str = "cpu"):
    """Load Conformer encoder from encoder.pt checkpoint."""
    encoder = ConformerEncoder()
    state_dict = torch.load(
        encoder_weights_path, map_location=device, weights_only=True
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def load_downsampler_weights(
    model_pt_path: str, encoder_dim: int = 1024, device: str = "cpu"
):
    """Load downsampler from model.pt checkpoint."""
    downsampler = EncoderDownsamplerConv1d(encoder_dim=encoder_dim, ds_rate=1)
    if model_pt_path and os.path.exists(model_pt_path):
        ckpt = torch.load(model_pt_path, map_location=device, weights_only=True)
        ds_state = {}
        for k, v in ckpt.items():
            if k.startswith("down_sampler."):
                ds_state[k.replace("down_sampler.", "")] = v
        if ds_state:
            downsampler.load_state_dict(ds_state)
    downsampler.eval()
    return downsampler


def build_smear(encoder_dim=1024, llm_dim=2048, num_experts=8):
    """Build SMEAR module."""
    experts = nn.ModuleList(
        [EncoderProjectorLinear(encoder_dim, llm_dim) for _ in range(num_experts)]
    )
    return MoELayer_SMEAR(experts, input_dim=encoder_dim)


def load_smear_weights(smear, model_pt_path: str, device: str = "cpu"):
    """Load SMEAR weights from model.pt checkpoint."""
    ckpt = torch.load(model_pt_path, map_location=device, weights_only=True)
    smear_state = {}
    for k, v in ckpt.items():
        if k.startswith("MoELayer_routing."):
            smear_state[k.replace("MoELayer_routing.", "")] = v
    if smear_state:
        smear.load_state_dict(smear_state, strict=False)
    smear.eval()
    return smear


# ============================================================
# NxDI LLM Decoder
# ============================================================


def build_llm_model(
    llm_path: str,
    tp_degree: int = 1,
    batch_size: int = 1,
    seq_len: int = 2048,
    n_positions: int = 4096,
    lnc: int = 2,
    on_device_sampling: bool = False,
):
    """Build the Shrutam-2 LLM decoder using NxDI.

    Returns:
        (model, config) tuple. Model must be compiled or loaded separately.
    """
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.models.llama.modeling_llama import (
        NeuronLlamaModel,
        NeuronLlamaForCausalLM,
    )
    from neuronx_distributed_inference.models.image_to_text_model_base import (
        NeuronBaseForImageToText,
    )
    from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
        ImageToTextModelWrapper,
    )
    from neuronx_distributed_inference.models.pixtral.modeling_pixtral import (
        PixtralInferenceConfig,
    )
    from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
        scatter_by_index_put,
    )

    with open(os.path.join(llm_path, "config.json")) as f:
        llm_cfg = json.load(f)

    on_device_sampling_config = None
    if on_device_sampling:
        from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig

        on_device_sampling_config = OnDeviceSamplingConfig(
            do_sample=False,
            top_k=1,
            dynamic=False,
        )

    text_neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        seq_len=seq_len,
        n_positions=n_positions,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=on_device_sampling_config,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        fused_qkv=True,
    )

    vision_neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        seq_len=1500,
        torch_dtype=torch.bfloat16,
        enable_bucketing=False,
        on_device_sampling_config=None,
    )

    def load_config(config_obj):
        config_obj.text_config = SimpleNamespace(
            hidden_size=llm_cfg["hidden_size"],
            num_attention_heads=llm_cfg["num_attention_heads"],
            num_hidden_layers=llm_cfg["num_hidden_layers"],
            num_key_value_heads=llm_cfg["num_key_value_heads"],
            vocab_size=llm_cfg["vocab_size"],
            max_position_embeddings=llm_cfg.get("max_position_embeddings", 4096),
            rope_theta=llm_cfg.get("rope_theta", 10000),
            rms_norm_eps=llm_cfg.get("rms_norm_eps", 1e-5),
            hidden_act=llm_cfg.get("hidden_act", "silu"),
            intermediate_size=llm_cfg.get("intermediate_size", 8192),
            head_dim=llm_cfg.get("head_dim", 64),
            sliding_window=llm_cfg.get("sliding_window", None),
            tie_word_embeddings=llm_cfg.get("tie_word_embeddings", True),
            pad_token_id=llm_cfg.get("pad_token_id", 0),
            bos_token_id=llm_cfg.get("bos_token_id", 2),
            eos_token_id=llm_cfg.get("eos_token_id", 128001),
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        config_obj.vision_config = SimpleNamespace(
            hidden_size=2048,
            image_size=1024,
            patch_size=16,
            num_hidden_layers=1,
            num_channels=3,
            num_attention_heads=8,
            rope_theta=10000.0,
            head_dim=64,
            intermediate_size=4096,
            hidden_act="silu",
        )
        config_obj._name_or_path = llm_path
        config_obj.multimodal_projector_bias = False
        config_obj.projector_hidden_act = "gelu"
        config_obj.vision_feature_layer = -1
        config_obj.output_attentions = False
        config_obj.output_hidden_states = False
        config_obj.return_dict = True
        config_obj.tie_word_embeddings = llm_cfg.get("tie_word_embeddings", True)
        config_obj.image_token_index = AUDIO_PLACEHOLDER_ID

    pixtral_config = PixtralInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_config,
    )

    # Capture lnc for compiler args
    _lnc = lnc

    class ShrutamTextModel(NeuronLlamaModel):
        def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
            return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    class ShrutamForCausalLM(NeuronBaseForImageToText):
        text_model_cls = ShrutamTextModel
        text_model_wrapper = ImageToTextModelWrapper
        vision_model_cls = None
        vision_model_wrapper = None

        @staticmethod
        def convert_hf_to_neuron_state_dict(state_dict, config):
            if (
                "lm_head.weight" not in state_dict
                and "embed_tokens.weight" in state_dict
            ):
                state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
            return NeuronLlamaForCausalLM.convert_hf_to_neuron_state_dict(
                state_dict, config.text_config
            )

        @staticmethod
        def update_state_dict_for_tied_weights(state_dict):
            if (
                "lm_head.weight" not in state_dict
                and "embed_tokens.weight" in state_dict
            ):
                state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

        def __init__(self, model_path, inference_config, *args, **kwargs):
            super().__init__(
                self.text_model_cls,
                self.vision_model_cls,
                self.text_model_wrapper,
                self.vision_model_wrapper,
                model_path,
                inference_config,
                *args,
                **kwargs,
            )

        def enable_vision_encoder(self, **kwargs):
            pass

        def compile(self, compiled_model_path, debug=False, dry_run=False):
            logger = logging.getLogger("Neuron")
            self.config.save(compiled_model_path)
            text_path = os.path.join(compiled_model_path, "text_model") + "/"
            os.makedirs(text_path, exist_ok=True)
            log.info("Tracing text model (CTE + TKG)...")
            t0 = time.time()
            text_traced_model = self.get_text_builder(debug).trace(
                initialize_model_weights=False, dry_run=dry_run
            )
            log.info(f"Trace completed in {time.time() - t0:.1f}s")
            if not dry_run:
                torch.jit.save(text_traced_model, text_path + "model.pt")
                del text_traced_model
                logger.info("Finished compiling text model!")
            self._save_configs_to_compiler_workdir()
            if dry_run:
                return
            self.shard_text_weights(text_path, debug)
            logger.info("Finished sharding text weights!")
            self.is_compiled = True

        def load(
            self,
            compiled_model_path,
            start_rank_id=None,
            local_ranks_size=None,
            skip_warmup=False,
        ):
            logger = logging.getLogger("Neuron")
            text_path = os.path.join(compiled_model_path, "text_model") + "/"
            self.text_traced_model = torch.jit.load(text_path + "model.pt")
            if start_rank_id is None:
                start_rank_id = self.neuron_config.start_rank_id
            logger.info("Sharding weights on load...")
            text_weights = self.get_text_builder().shard_checkpoint()
            start_rank_tensor = torch.tensor(
                [start_rank_id], dtype=torch.int32, device="cpu"
            )
            self.text_traced_model.nxd_model.initialize(text_weights, start_rank_tensor)
            logger.info("Finished text weights loading")
            for model_wrapper in self.text_models:
                model_wrapper.model = self.text_traced_model
            self.is_loaded_to_neuron = True
            if not self.neuron_config.skip_warmup and not skip_warmup:
                self.warmup()
            else:
                logger.info("Skipping model warmup")

        @classmethod
        def get_config_cls(cls):
            return PixtralInferenceConfig

        def get_compiler_args(self):
            return (
                "--auto-cast=none --model-type=transformer "
                "--tensorizer-options='--enable-ccop-compute-overlap "
                "--cc-pipeline-tiling-factor=2 --vectorize-strided-dma ' "
                f"--lnc={_lnc} -O1"
            )

        def _get_model_outputs(
            self,
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            prev_hidden,
            adapter_ids,
            vision_embeddings,
            vision_mask,
            deepstack_vision_embeds,
            medusa_args,
            llava_args,
            slot_mapping=None,
            block_table=None,
            full_context_lens=None,
            computed_context_lens=None,
            rotary_position_ids=None,
        ):
            if rotary_position_ids is None:
                rotary_position_ids = torch.empty(0)
            if vision_embeddings is None:
                vision_embeddings = torch.zeros(
                    (
                        self.config.text_config.neuron_config.batch_size,
                        self.config.text_config.neuron_config.seq_len,
                        self.config.text_config.hidden_size,
                    ),
                    dtype=self.config.text_config.neuron_config.torch_dtype,
                )
            if vision_mask is None:
                vision_mask = torch.full(
                    (
                        self.config.text_config.neuron_config.batch_size,
                        self.config.text_config.neuron_config.seq_len,
                        1,
                    ),
                    fill_value=self.config.text_config.neuron_config.seq_len - 1,
                    dtype=torch.int32,
                )

            if self._is_prefill(position_ids):
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    *[torch.empty(0) for _ in range(16)],
                    rotary_position_ids,
                    vision_embeddings,
                    vision_mask,
                )
                self.kv_cache_populated = True
                is_run_on_neuron = self.context_encoding_model.is_neuron()
            else:
                outputs = self.token_generation_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    *[torch.empty(0) for _ in range(16)],
                    rotary_position_ids,
                    torch.empty(0, dtype=torch.bfloat16),
                    torch.empty(0, dtype=torch.bool),
                )
                is_run_on_neuron = self.token_generation_model.is_neuron()
            return outputs, is_run_on_neuron

        def get_required_kwargs(self):
            return ["vision_embeddings", "vision_mask"]

    model = ShrutamForCausalLM(llm_path, pixtral_config)
    return model, pixtral_config


# ============================================================
# Shrutam-2 Pipeline
# ============================================================


class Shrutam2Pipeline:
    """End-to-end Shrutam-2 ASR pipeline on Neuron.

    Three-stage pipeline:
    1. Conformer encoder (traced NEFF) -- speech to encoder features
    2. SMEAR-MoE projector (traced NEFF) -- encoder features to LLM embeddings
    3. LLM decoder (NxDI Llama) -- audio embeddings + prompt -> text

    Example:
        pipeline = Shrutam2Pipeline(
            encoder_neff_path="/mnt/models/encoder_neuron.pt",
            smear_neff_path="/mnt/models/smear_neuron.pt",
            llm_compiled_path="/mnt/models/compiled/shrutam2_decoder_tp1",
            llm_path="/mnt/models/Shrutam-2-hf/llm",
        )
        result = pipeline.transcribe("audio.wav", prompt="Transcribe speech to Hindi text.")
    """

    def __init__(
        self,
        encoder_neff_path: str,
        smear_neff_path: str,
        llm_compiled_path: str,
        llm_path: str,
        tp_degree: int = 1,
        batch_size: int = 1,
        seq_len: int = 2048,
        n_positions: int = 4096,
        lnc: int = 2,
        on_device_sampling: bool = False,
        audio_seconds: float = 10.0,
    ):
        """Initialize the Shrutam-2 pipeline.

        Args:
            encoder_neff_path: Path to traced Conformer encoder NEFF
            smear_neff_path: Path to traced SMEAR-MoE NEFF
            llm_compiled_path: Path to compiled NxDI LLM directory
            llm_path: Path to LLM weights and tokenizer
            tp_degree: Tensor parallelism degree (default: 1)
            batch_size: Batch size (default: 1)
            seq_len: CTE sequence length (default: 2048)
            n_positions: KV cache positions (default: 4096)
            lnc: Logical NeuronCore config (default: 2)
            on_device_sampling: Enable on-device sampling (default: False)
            audio_seconds: Fixed audio duration for encoder trace (default: 10.0)
        """
        import torch_neuronx
        from transformers import AutoTokenizer
        from neuronx_distributed_inference.utils.hf_adapter import (
            HuggingFaceGenerationAdapter,
        )
        from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
            generate_positions_from_mask,
            pad_positions,
            pad_vision_embeddings,
        )

        self.seq_len = seq_len
        self.audio_seconds = audio_seconds
        self.traced_T_mel = int(audio_seconds * SAMPLE_RATE) // HOP_LENGTH + 1
        self._generate_positions_from_mask = generate_positions_from_mask
        self._pad_positions = pad_positions
        self._pad_vision_embeddings = pad_vision_embeddings

        # Audio preprocessor (CPU)
        self.preprocessor = MelSpectrogramPreprocessor()
        self.preprocessor.eval()

        # Conformer encoder (Neuron)
        log.info(f"Loading Conformer encoder from {encoder_neff_path}...")
        t0 = time.time()
        self.encoder_neuron = torch.jit.load(encoder_neff_path)
        torch_neuronx.async_load(self.encoder_neuron)
        log.info(f"Encoder loaded in {time.time() - t0:.1f}s")

        # SMEAR-MoE projector (Neuron)
        log.info(f"Loading SMEAR-MoE from {smear_neff_path}...")
        t0 = time.time()
        self.smear_neuron = torch.jit.load(smear_neff_path)
        torch_neuronx.async_load(self.smear_neuron)
        log.info(f"SMEAR loaded in {time.time() - t0:.1f}s")

        # Traced SMEAR sequence length (ceil(traced_T_mel / 8) for 8x subsampling)
        self.traced_smear_len = math.ceil(self.traced_T_mel / 8)

        # LLM decoder (NxDI)
        log.info(f"Loading LLM decoder from {llm_compiled_path}...")
        t0 = time.time()
        self.llm_model, _ = build_llm_model(
            llm_path,
            tp_degree,
            batch_size,
            seq_len,
            n_positions,
            lnc,
            on_device_sampling,
        )
        self.llm_model.load(llm_compiled_path)
        log.info(f"LLM loaded in {time.time() - t0:.1f}s")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # HF adapter for generation
        self.adapter = HuggingFaceGenerationAdapter(self.llm_model)

    def preprocess_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to mel spectrogram.

        Args:
            wav: [num_samples] or [1, num_samples] raw audio at 16kHz

        Returns:
            mel: [1, 80, T_mel] log-mel spectrogram
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav[:1]
        length = torch.tensor([wav.shape[-1]])
        with torch.no_grad():
            mel, _ = self.preprocessor(wav, length)
        return mel

    def transcribe(
        self,
        audio_path: str,
        prompt: str = "Transcribe speech to Hindi text.",
        max_new_tokens: int = 200,
        repetition_penalty: float = 1.3,
    ) -> dict:
        """Transcribe an audio file.

        Args:
            audio_path: Path to WAV file (16kHz mono or will be resampled)
            prompt: Text prompt for the decoder
            max_new_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty (1.3 recommended for greedy)

        Returns:
            Dict with: text, num_tokens, encoder_ms, smear_ms, gen_time_s, total_time_s
        """
        import soundfile as sf

        t_start = time.time()

        # Load audio
        wav_np, sr = sf.read(audio_path)
        wav_np = wav_np.astype(np.float32)
        if wav_np.ndim > 1:
            wav_np = wav_np[:, 0]
        if sr != SAMPLE_RATE:
            from scipy.signal import resample as scipy_resample

            num_samples = int(len(wav_np) * SAMPLE_RATE / sr)
            wav_np = scipy_resample(wav_np, num_samples).astype(np.float32)
        wav = torch.from_numpy(wav_np).unsqueeze(0)

        return self.transcribe_tensor(
            wav, prompt, max_new_tokens, repetition_penalty, t_start
        )

    def transcribe_tensor(
        self,
        wav: torch.Tensor,
        prompt: str = "Transcribe speech to Hindi text.",
        max_new_tokens: int = 200,
        repetition_penalty: float = 1.3,
        t_start: Optional[float] = None,
    ) -> dict:
        """Transcribe from a raw waveform tensor.

        Args:
            wav: [1, num_samples] raw audio at 16kHz
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty
            t_start: Optional start time for total timing

        Returns:
            Dict with transcription results
        """
        if t_start is None:
            t_start = time.time()

        # Step 1: Mel spectrogram
        mel = self.preprocess_audio(wav)

        # Step 2: Pad/truncate mel to traced shape
        actual_T_mel = mel.shape[2]
        if actual_T_mel < self.traced_T_mel:
            mel_padded = F.pad(mel, (0, self.traced_T_mel - actual_T_mel), value=0.0)
        elif actual_T_mel > self.traced_T_mel:
            mel_padded = mel[:, :, : self.traced_T_mel]
            actual_T_mel = self.traced_T_mel
        else:
            mel_padded = mel

        mel_lengths = torch.tensor([actual_T_mel], dtype=torch.long)

        # Step 3: Conformer encoder
        t_enc = time.time()
        with torch.no_grad():
            encoder_out = self.encoder_neuron(mel_padded, mel_lengths)
        encoder_ms = (time.time() - t_enc) * 1000

        # Trim to actual audio length
        actual_T_out = math.ceil(actual_T_mel / 8)
        if actual_T_out < encoder_out.shape[1]:
            encoder_out = encoder_out[:, :actual_T_out, :]

        # Step 4: SMEAR-MoE projector
        actual_enc_len = encoder_out.shape[1]
        if actual_enc_len < self.traced_smear_len:
            smear_input = F.pad(
                encoder_out,
                (0, 0, 0, self.traced_smear_len - actual_enc_len),
                value=0.0,
            )
        elif actual_enc_len > self.traced_smear_len:
            smear_input = encoder_out[:, : self.traced_smear_len, :]
        else:
            smear_input = encoder_out

        t_smear = time.time()
        with torch.no_grad():
            audio_embeds = self.smear_neuron(smear_input)
        smear_ms = (time.time() - t_smear) * 1000

        # Trim SMEAR output
        if actual_enc_len < self.traced_smear_len:
            audio_embeds = audio_embeds[:, :actual_enc_len, :]

        num_audio_tokens = audio_embeds.shape[1]

        # Step 5: Build input sequence
        prompt_text = PROMPT_TEMPLATE.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        audio_placeholder_ids = [AUDIO_PLACEHOLDER_ID] * num_audio_tokens
        full_ids = audio_placeholder_ids + prompt_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        total_input_len = input_ids.shape[1]

        # Step 6: Build vision scatter args
        audio_embeds_bf16 = audio_embeds.to(torch.bfloat16)
        modality_mask_1d = torch.zeros(total_input_len, dtype=torch.bool)
        modality_mask_1d[:num_audio_tokens] = True

        vision_mask = self._generate_positions_from_mask(modality_mask_1d)
        vision_mask_padded = self._pad_positions(
            vision_mask, self.seq_len, self.seq_len - 1
        )
        audio_embeds_padded = self._pad_vision_embeddings(
            audio_embeds_bf16, self.seq_len
        )

        # Step 7: Generate
        t_gen = time.time()
        output_ids = self.adapter.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            vision_embeddings=audio_embeds_padded,
            vision_mask=vision_mask_padded,
        )
        gen_time = time.time() - t_gen

        new_tokens = output_ids[0, total_input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        non_pad = (new_tokens != self.tokenizer.pad_token_id).sum().item()
        total_time = time.time() - t_start

        return {
            "text": text,
            "num_tokens": non_pad,
            "encoder_ms": encoder_ms,
            "smear_ms": smear_ms,
            "gen_time_s": gen_time,
            "total_time_s": total_time,
            "tok_per_s": non_pad / gen_time if gen_time > 0 else 0,
            "audio_tokens": num_audio_tokens,
        }


# ============================================================
# Tracing Utilities
# ============================================================


def trace_encoder(
    encoder_weights_path: str,
    model_pt_path: str,
    output_path: str,
    batch_size: int = 1,
    audio_seconds: float = 10.0,
    lnc: int = 2,
):
    """Trace the Conformer encoder + downsampler for Neuron.

    Args:
        encoder_weights_path: Path to encoder.pt
        model_pt_path: Path to model.pt (for downsampler weights)
        output_path: Path to save traced NEFF
        batch_size: Batch size
        audio_seconds: Audio duration in seconds
        lnc: LNC config

    Returns:
        (traced_model, trace_time_s)
    """
    import torch_neuronx

    encoder = load_encoder_weights(encoder_weights_path)
    downsampler = load_downsampler_weights(model_pt_path)
    model = ConformerEncoderForTrace(encoder, downsampler)
    model.eval()

    n_samples = int(audio_seconds * SAMPLE_RATE)
    T_mel = n_samples // HOP_LENGTH + 1
    example_mel = torch.randn(batch_size, N_MELS, T_mel)
    example_lengths = torch.full((batch_size,), T_mel, dtype=torch.long)

    compiler_args = [
        "--model-type",
        "transformer",
        "--auto-cast",
        "matmult",
        "--auto-cast-type",
        "bf16",
        "--tensorizer-options",
        "--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2",
        "--lnc",
        str(lnc),
    ]

    t0 = time.time()
    traced = torch_neuronx.trace(
        model,
        (example_mel, example_lengths),
        compiler_args=compiler_args,
        inline_weights_to_neff=True,
    )
    trace_time = time.time() - t0

    torch.jit.save(traced, output_path)
    log.info(f"Encoder traced in {trace_time:.1f}s, saved to {output_path}")
    return traced, trace_time


def trace_smear(
    model_pt_path: str,
    output_path: str,
    batch_size: int = 1,
    seq_len: int = 126,
    lnc: int = 2,
):
    """Trace the SMEAR-MoE projector for Neuron.

    Args:
        model_pt_path: Path to model.pt
        output_path: Path to save traced NEFF
        batch_size: Batch size
        seq_len: Encoder output sequence length
        lnc: LNC config

    Returns:
        (traced_model, trace_time_s)
    """
    import torch_neuronx

    smear = build_smear()
    smear = load_smear_weights(smear, model_pt_path)
    wrapper = SMEARForTrace(smear)
    wrapper.eval()

    example_input = torch.randn(batch_size, seq_len, ENCODER_DIM)

    compiler_args = [
        "--model-type",
        "transformer",
        "--auto-cast",
        "matmult",
        "--auto-cast-type",
        "bf16",
        "--tensorizer-options",
        "--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2",
        "--lnc",
        str(lnc),
    ]

    t0 = time.time()
    traced = torch_neuronx.trace(
        wrapper,
        (example_input,),
        compiler_args=compiler_args,
        inline_weights_to_neff=True,
    )
    trace_time = time.time() - t0

    torch.jit.save(traced, output_path)
    log.info(f"SMEAR traced in {trace_time:.1f}s, saved to {output_path}")
    return traced, trace_time
