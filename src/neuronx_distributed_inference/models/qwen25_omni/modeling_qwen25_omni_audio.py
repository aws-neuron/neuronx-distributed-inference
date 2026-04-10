# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni Audio Encoder for NXD inference.
#
# Whisper-style audio encoder with:
#   - Conv1d frontend (128 mel bins -> 1280 dim)
#   - Sinusoidal positional embeddings
#   - 32 transformer layers with LayerNorm, GELU MLP, standard multi-head attention
#   - Asymmetric attention bias (q/v have bias, k has NO bias)
#   - Chunked attention (n_window=100) for long audio
#   - AvgPool1d downsampling + Linear projection to text model dim (3584)
#
# Runs on CPU: chunked variable-length attention doesn't fit Neuron's
# fixed-shape compilation model. The encoder is small (~500MB) and fast
# on CPU for typical audio lengths (<60s).
#
# Key dims: d_model=1280, heads=20, head_dim=64, ffn=5120, output=3584

"""Qwen2.5-Omni Audio Encoder for NXD inference."""

import logging
import math
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.config import InferenceConfig

logger = logging.getLogger(__name__)


class SinusoidsPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings (same as Whisper/HF Qwen2.5-Omni)."""

    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class AudioAttention(nn.Module):
    """Multi-head attention for audio encoder.

    Asymmetric bias: q_proj and v_proj have bias, k_proj has NO bias.
    Uses nn.Linear (not TP-parallel) since 20 heads / 32 TP doesn't divide.
    """

    def __init__(self, d_model, num_heads, dtype=torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=True, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=True, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None):
        seq_len = hidden_states.shape[0]

        q = self.q_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)

        # (1, num_heads, seq_len, head_dim)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.squeeze(0)
            .transpose(0, 1)
            .reshape(seq_len, -1)
            .contiguous()
        )
        return self.out_proj(attn_output)


class AudioEncoderLayer(nn.Module):
    """Audio encoder transformer layer.

    Pre-norm architecture: LayerNorm -> attention -> residual -> LayerNorm -> MLP -> residual.
    Uses standard GELU activation (not SwiGLU).
    """

    def __init__(self, d_model, num_heads, ffn_dim, dtype=torch.bfloat16):
        super().__init__()
        self.self_attn = AudioAttention(d_model, num_heads, dtype)
        # LayerNorm stays in float32 (input is cast to float32 for normalization)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim, dtype=dtype)
        self.fc2 = nn.Linear(ffn_dim, d_model, dtype=dtype)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states.float()).to(
            residual.dtype
        )
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states.float()).to(
            residual.dtype
        )
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # Clamp for float16 stability (matches HF)
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        return hidden_states


class NeuronQwen25OmniAudioEncoder(nn.Module):
    """Qwen2.5-Omni Audio Encoder.

    Whisper-style encoder that processes mel spectrograms into audio
    embeddings compatible with the text model. Runs on CPU.

    Architecture:
      mel (n_mels, mel_len)
        -> Conv1d(n_mels, d_model) + GELU
        -> Conv1d(d_model, d_model, stride=2) + GELU
        -> + sinusoidal positional embeddings
        -> 32 transformer layers (chunked attention via n_window)
        -> AvgPool1d(2, stride=2)
        -> LayerNorm
        -> Linear(d_model, output_dim)
      = audio embeddings (num_tokens, output_dim=3584)

    Audio token count: ((mel_len - 1) // 2 + 1 - 2) // 2 + 1
    """

    def __init__(self, audio_config, dtype=torch.bfloat16):
        super().__init__()
        # Accept both dict and namespace
        if isinstance(audio_config, dict):
            audio_config = SimpleNamespace(**audio_config)
        self.audio_config = audio_config

        d_model = audio_config.d_model  # 1280
        num_mel_bins = audio_config.num_mel_bins  # 128
        max_source_positions = audio_config.max_source_positions  # 1500
        encoder_layers = audio_config.encoder_layers  # 32
        encoder_attention_heads = audio_config.encoder_attention_heads  # 20
        encoder_ffn_dim = audio_config.encoder_ffn_dim  # 5120
        output_dim = audio_config.output_dim  # 3584
        self.n_window = audio_config.n_window  # 100
        self.d_model = d_model

        # Conv frontend
        self.conv1 = nn.Conv1d(
            num_mel_bins, d_model, kernel_size=3, padding=1, dtype=dtype
        )
        self.conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=2, padding=1, dtype=dtype
        )

        # Positional embeddings
        self.positional_embedding = SinusoidsPositionEmbedding(
            max_source_positions, d_model
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                AudioEncoderLayer(
                    d_model, encoder_attention_heads, encoder_ffn_dim, dtype
                )
                for _ in range(encoder_layers)
            ]
        )

        # Post-processing (LayerNorm stays float32)
        self.ln_post = nn.LayerNorm(d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(d_model, output_dim, dtype=dtype)

        # Audio BOS/EOS token embeddings
        self.audio_bos_eos_token = nn.Embedding(2, output_dim)

    def _get_feat_extract_output_lengths(self, input_lengths):
        """Compute output lengths after conv and avg_pool."""
        input_lengths = (input_lengths - 1) // 2 + 1  # after conv stride-2
        output_lengths = (input_lengths - 2) // 2 + 1  # after avg_pool stride-2
        return input_lengths, output_lengths

    def _prepare_attention_mask(self, inputs_tensor, cu_seqlens):
        """Create block-diagonal attention mask from cu_seqlens.

        Each chunk only attends to itself. Positions outside a chunk
        are masked with large negative values.
        """
        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            s, e = cu_seqlens[i - 1], cu_seqlens[i]
            attention_mask[..., s:e, s:e] = 0
        return attention_mask

    def _padded_and_mask_function(self, chunk_list, chunk_lengths):
        """Pad chunks to same length and create masks.

        Args:
            chunk_list: list of (n_mels, chunk_len) tensors
            chunk_lengths: (num_chunks,) actual lengths

        Returns:
            padded_feature: (num_chunks, n_mels, max_chunk_len)
            padded_mask: (num_chunks, 1, max_chunk_len) for conv masking
            padded_mask_after_cnn: (num_chunks, max_len_after_cnn) bool
        """
        max_len = chunk_lengths.max().item()
        dim = chunk_list[0].shape[0]  # n_mels

        padded_tensor = torch.zeros(
            len(chunk_list),
            dim,
            max_len,
            dtype=chunk_list[0].dtype,
            device=chunk_list[0].device,
        )
        batch_mask = torch.zeros(
            len(chunk_lengths),
            max_len,
            dtype=torch.long,
            device=chunk_list[0].device,
        )

        for i, (chunk, length) in enumerate(zip(chunk_list, chunk_lengths)):
            length = length.item()
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :chunk.shape[1]] = chunk

        feature_lens_after_cnn = (chunk_lengths - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max().item()
        batch_mask_after_cnn = torch.zeros(
            len(chunk_lengths),
            max_len_after_cnn,
            dtype=torch.bool,
            device=chunk_list[0].device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = True

        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn,
        )

    def forward(self, input_features, feature_lens, aftercnn_lens=None):
        """Process mel spectrogram through audio encoder.

        Args:
            input_features: (n_mels, total_mel_len) mel spectrogram
                (concatenated across batch, padding already removed)
            feature_lens: (num_audios,) mel length for each audio
            aftercnn_lens: (num_audios,) length after conv, optional
                (computed from feature_lens if not provided)

        Returns:
            audio_embeddings: (total_audio_tokens, output_dim) tensor
        """
        if aftercnn_lens is None:
            aftercnn_lens, _ = self._get_feat_extract_output_lengths(feature_lens)

        # Split into chunks of n_window * 2 mel frames
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum().item(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(
            chunk_lengths == 0, self.n_window * 2, chunk_lengths
        )

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)

        # Pad chunks and create masks
        padded_feature, padded_mask, padded_mask_after_cnn = (
            self._padded_and_mask_function(chunk_list, chunk_lengths)
        )

        # Conv frontend
        padded_embed = F.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = F.gelu(self.conv2(padded_embed)).transpose(1, 2)

        # Add positional embeddings
        padded_embed = padded_embed + self.positional_embedding(
            padded_embed.shape[1]
        ).unsqueeze(0).to(padded_embed.dtype)

        # Flatten valid tokens across all chunks
        hidden_states = padded_embed[padded_mask_after_cnn]

        # Prepare chunk-boundary attention mask
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, device=feature_lens.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0).to(torch.int32),
            ]
        )
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Split back per-audio and post-process
        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(
                each_audio_states.transpose(0, 1)
            ).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states.float()).to(
                each_audio_states.dtype
            )
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)

        return torch.cat(token_audio_list, dim=0)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, dtype=torch.bfloat16
    ) -> dict:
        """Convert HF state dict to NxDI audio encoder format.

        Simply strips the 'thinker.audio_tower.' or 'audio_tower.' prefix.
        Module naming matches HF exactly, so no key renaming needed.

        LayerNorm weights stay in float32 (they operate in float32).
        All other weights are cast to the target dtype.
        """
        new_state_dict = {}

        # LayerNorm keys that should remain float32
        ln_suffixes = (
            "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
            "final_layer_norm.weight", "final_layer_norm.bias",
            "ln_post.weight", "ln_post.bias",
        )

        for key, value in state_dict.items():
            if key.startswith("thinker.audio_tower."):
                new_key = key[len("thinker.audio_tower."):]
            elif key.startswith("audio_tower."):
                new_key = key[len("audio_tower."):]
            else:
                # Pass through non-audio keys unchanged
                new_state_dict[key] = value
                continue

            # Keep LayerNorm weights in float32
            if any(new_key.endswith(s) for s in ln_suffixes):
                new_state_dict[new_key] = (
                    value.clone().detach().contiguous().to(torch.float32)
                )
            else:
                new_state_dict[new_key] = (
                    value.clone().detach().contiguous().to(dtype)
                )

        return new_state_dict

    @staticmethod
    def from_pretrained_state_dict(audio_config, state_dict, dtype=torch.bfloat16):
        """Create audio encoder and load weights from converted state dict.

        Args:
            audio_config: Audio encoder config (dict or namespace)
            state_dict: Already-converted state dict (audio keys only)
            dtype: Target dtype

        Returns:
            Initialized NeuronQwen25OmniAudioEncoder
        """
        encoder = NeuronQwen25OmniAudioEncoder(audio_config, dtype=dtype)

        # Filter to only audio encoder keys
        audio_keys = {}
        for key, value in state_dict.items():
            # Skip any non-audio keys that might be in the dict
            if any(
                key.startswith(p)
                for p in ["model.", "lm_head.", "visual.", "layers."]
            ):
                continue
            audio_keys[key] = value

        missing, unexpected = encoder.load_state_dict(audio_keys, strict=False)
        if missing:
            logger.warning(
                "Audio encoder missing keys: %s", missing[:10]
            )
        if unexpected:
            logger.warning(
                "Audio encoder unexpected keys: %s", unexpected[:10]
            )
        logger.info(
            "Loaded %d weights into audio encoder", len(audio_keys)
        )

        return encoder
