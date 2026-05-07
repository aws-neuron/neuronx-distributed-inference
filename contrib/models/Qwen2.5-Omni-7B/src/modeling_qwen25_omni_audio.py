# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni Audio Encoder for NXD inference.
#
# Whisper-style audio encoder with TP=4 support:
#   - Conv1d frontend + sinusoidal positional embeddings (CPU preprocessing)
#   - 32 transformer layers with TP-parallel attention and MLP (Neuron)
#   - AvgPool1d + LayerNorm + Linear projection (CPU postprocessing)
#
# Architecture:
#   d_model=1280, heads=20 (5 per rank at TP=4), head_dim=64
#   ffn=5120 (1280 per rank at TP=4), output_dim=3584
#   Asymmetric attention bias: q/v have bias, k has NO bias
#
# The transformer layers are compiled on Neuron via ModelWrapper.
# Conv1d frontend and postprocessing run on CPU since they involve
# variable-length processing (chunking, per-audio AvgPool).

"""Qwen2.5-Omni Audio Encoder for NXD inference."""

import logging
import math
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import (
    EncoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.padding import pad_tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU components (not compiled on Neuron)
# ---------------------------------------------------------------------------

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


class AudioCPUFrontend(nn.Module):
    """Conv1d frontend + positional embeddings + chunking (CPU).

    Processes raw mel spectrograms into token sequences ready for the
    Neuron transformer. Also handles audio chunking for long audio.
    """

    def __init__(self, audio_config, dtype=torch.bfloat16):
        super().__init__()
        if isinstance(audio_config, dict):
            audio_config = SimpleNamespace(**audio_config)

        d_model = audio_config.d_model  # 1280
        num_mel_bins = audio_config.num_mel_bins  # 128
        max_source_positions = audio_config.max_source_positions  # 1500
        self.n_window = audio_config.n_window  # 100
        self.d_model = d_model

        self.conv1 = nn.Conv1d(
            num_mel_bins, d_model, kernel_size=3, padding=1, dtype=dtype
        )
        self.conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=2, padding=1, dtype=dtype
        )
        self.positional_embedding = SinusoidsPositionEmbedding(
            max_source_positions, d_model
        )

    def _padded_and_mask_function(self, chunk_list, chunk_lengths):
        """Pad chunks to same length and create masks."""
        max_len = chunk_lengths.max().item()
        dim = chunk_list[0].shape[0]

        padded_tensor = torch.zeros(
            len(chunk_list), dim, max_len,
            dtype=chunk_list[0].dtype, device=chunk_list[0].device,
        )
        batch_mask = torch.zeros(
            len(chunk_lengths), max_len,
            dtype=torch.long, device=chunk_list[0].device,
        )
        for i, (chunk, length) in enumerate(zip(chunk_list, chunk_lengths)):
            length = length.item()
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :chunk.shape[1]] = chunk

        feature_lens_after_cnn = (chunk_lengths - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max().item()
        batch_mask_after_cnn = torch.zeros(
            len(chunk_lengths), max_len_after_cnn,
            dtype=torch.bool, device=chunk_list[0].device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = True

        return padded_tensor, batch_mask.unsqueeze(1), batch_mask_after_cnn

    def forward(self, input_features, feature_lens):
        """Process mel spectrogram through conv frontend.

        Args:
            input_features: (n_mels, total_mel_len) mel spectrogram
            feature_lens: (num_audios,) mel length for each audio

        Returns:
            hidden_states: (total_valid_tokens, d_model) token embeddings
            aftercnn_lens: (num_audios,) valid token count per audio
            cu_seqlens: cumulative sequence lengths for attention masking
        """
        aftercnn_lens = (feature_lens - 1) // 2 + 1

        # Split into chunks of n_window * 2 mel frames
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum().item(),
            dtype=torch.long, device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(
            chunk_lengths == 0, self.n_window * 2, chunk_lengths
        )

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
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

        # Flatten valid tokens
        hidden_states = padded_embed[padded_mask_after_cnn]

        # Compute cu_seqlens for attention mask
        cu_seqlens = torch.cat([
            torch.zeros(1, device=feature_lens.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0).to(torch.int32),
        ])

        return hidden_states, aftercnn_lens, cu_seqlens


class AudioCPUPostprocessor(nn.Module):
    """AvgPool + LayerNorm + projection (CPU).

    Post-processes transformer output into final audio embeddings.
    """

    def __init__(self, audio_config, dtype=torch.bfloat16):
        super().__init__()
        if isinstance(audio_config, dict):
            audio_config = SimpleNamespace(**audio_config)

        d_model = audio_config.d_model  # 1280
        output_dim = audio_config.output_dim  # 3584

        self.ln_post = nn.LayerNorm(d_model)  # stays float32
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(d_model, output_dim, dtype=dtype)
        self.audio_bos_eos_token = nn.Embedding(2, output_dim)

    def forward(self, hidden_states, aftercnn_lens):
        """Post-process transformer output.

        Args:
            hidden_states: (total_tokens, d_model) transformer output
            aftercnn_lens: (num_audios,) token count per audio

        Returns:
            audio_embeddings: (total_output_tokens, output_dim)
        """
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


# ---------------------------------------------------------------------------
# Neuron-compiled transformer components (TP=4)
# ---------------------------------------------------------------------------

class NeuronAudioAttention(nn.Module):
    """TP-parallel self-attention for audio encoder.

    Asymmetric bias: q_proj and v_proj have bias, k_proj has NO bias.
    Uses ColumnParallelLinear for Q/K/V and RowParallelLinear for output.
    """

    def __init__(self, d_model, num_heads, tp_degree, dtype=torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_heads_per_rank = num_heads // tp_degree
        self.scaling = self.head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(
            d_model, d_model, bias=True, gather_output=False, dtype=dtype,
        )
        self.k_proj = ColumnParallelLinear(
            d_model, d_model, bias=False, gather_output=False, dtype=dtype,
        )
        self.v_proj = ColumnParallelLinear(
            d_model, d_model, bias=True, gather_output=False, dtype=dtype,
        )
        self.out_proj = RowParallelLinear(
            d_model, d_model, bias=True, input_is_parallel=True, dtype=dtype,
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, 1, seq_len, seq_len) with 0 for valid, -inf for masked
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project Q/K/V (ColumnParallelLinear outputs d_model/tp per rank)
        q = self.q_proj(hidden_states)  # (bsz, seq, d_model/tp)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (bsz, num_heads_per_rank, seq, head_dim)
        q = q.view(bsz, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output (RowParallelLinear allreduces across ranks)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


class NeuronAudioEncoderLayer(nn.Module):
    """Audio encoder transformer layer (TP-parallel).

    Pre-norm: LayerNorm -> attention -> residual -> LayerNorm -> MLP -> residual.
    Uses GELU activation (not SwiGLU). LayerNorm operates in float32.
    """

    def __init__(self, d_model, num_heads, ffn_dim, tp_degree, dtype=torch.bfloat16):
        super().__init__()
        self.self_attn = NeuronAudioAttention(d_model, num_heads, tp_degree, dtype)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = ColumnParallelLinear(
            d_model, ffn_dim, bias=True, gather_output=False, dtype=dtype,
        )
        self.fc2 = RowParallelLinear(
            ffn_dim, d_model, bias=True, input_is_parallel=True, dtype=dtype,
        )
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states.float()).to(residual.dtype)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states.float()).to(residual.dtype)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NeuronAudioTransformerModel(nn.Module):
    """Audio encoder transformer (32 layers, compiled on Neuron).

    Takes pre-processed hidden states (after conv + positional embedding)
    and returns transformer output. The attention mask handles block-diagonal
    masking for chunked audio.

    Input:  (1, padded_seq_len, d_model) + (1, 1, padded_seq_len, padded_seq_len)
    Output: (1, padded_seq_len, d_model)
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        audio_config = config.audio_config
        if isinstance(audio_config, dict):
            audio_config = SimpleNamespace(**audio_config)

        tp_degree = config.neuron_config.tp_degree
        dtype = config.neuron_config.torch_dtype

        d_model = audio_config.d_model
        num_heads = audio_config.encoder_attention_heads
        ffn_dim = audio_config.encoder_ffn_dim
        num_layers = audio_config.encoder_layers

        self.layers = nn.ModuleList([
            NeuronAudioEncoderLayer(d_model, num_heads, ffn_dim, tp_degree, dtype)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, attention_mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# ---------------------------------------------------------------------------
# ModelWrapper and Application classes
# ---------------------------------------------------------------------------

class AudioTransformerModelWrapper(ModelWrapper):
    """Handles bucketing on sequence length, padding, and compilation."""

    def __init__(self, config, model_cls, tag="", compiler_args=None,
                 priority_model_idx=None, pipeline_execution=True,
                 return_ranked_to_cpu=False, model_init_kwargs={}):
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx,
            pipeline_execution, return_ranked_to_cpu, model_init_kwargs,
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """Generate sample inputs for each sequence length bucket."""
        inputs = []
        dtype = self.config.neuron_config.torch_dtype
        d_model = self.config.audio_config.d_model
        if isinstance(d_model, dict):
            d_model = d_model.get("d_model", 1280)

        for bucket in self.config.neuron_config.buckets:
            hidden_states = torch.ones([1, bucket, d_model], dtype=dtype)
            attention_mask = torch.zeros(
                [1, 1, bucket, bucket], dtype=dtype,
            )
            inputs.append((hidden_states, attention_mask))
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def get_target_bucket(self, seq_len):
        """Find the smallest bucket that fits the sequence length."""
        for bucket in self.config.neuron_config.buckets:
            if bucket >= seq_len:
                return bucket
        raise ValueError(
            f"No bucket found for seq_len={seq_len}. "
            f"Buckets: {self.config.neuron_config.buckets}"
        )

    def forward(self, hidden_states, attention_mask):
        """Pad to bucket size and run compiled model."""
        if self.model is None:
            raise RuntimeError("Forward called before load.")

        seq_len = hidden_states.shape[1]
        bucket = self.get_target_bucket(seq_len)

        # Pad sequence to bucket size
        if seq_len < bucket:
            pad_len = bucket - seq_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            # Extend attention mask: new positions are masked out
            dtype = attention_mask.dtype
            mask_pad = torch.full(
                (1, 1, bucket, bucket),
                torch.finfo(dtype).min,
                dtype=dtype,
            )
            mask_pad[:, :, :seq_len, :seq_len] = attention_mask
            attention_mask = mask_pad

        output = self._forward(hidden_states, attention_mask)
        # Trim back to original length
        return output[:, :seq_len, :]


class NeuronQwen25OmniForAudioEncoding(NeuronApplicationBase):
    """Neuron application for audio encoder transformer layers.

    Handles compilation, loading, and inference of the 32 transformer layers.
    The conv frontend and postprocessing are handled separately on CPU.
    """

    _model_cls = NeuronAudioTransformerModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AudioTransformerModelWrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)

    def forward(self, hidden_states, attention_mask):
        return self.models[0](hidden_states, attention_mask)

    def get_compiler_args(self):
        return (
            "--auto-cast=none --model-type=transformer "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 ' -O1 "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, inference_config):
        """Convert HF state dict to Neuron format for audio transformer layers.

        Extracts audio_tower.layers.* keys and strips prefix.
        Only returns transformer layer keys (not conv/postprocessing).
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("thinker.audio_tower.layers."):
                new_key = key[len("thinker.audio_tower."):]
            elif key.startswith("audio_tower.layers."):
                new_key = key[len("audio_tower."):]
            else:
                new_state_dict[key] = value
                continue
            new_state_dict[new_key] = value
        return new_state_dict

    @classmethod
    def get_config_cls(cls):
        return AudioEncoderInferenceConfig


class AudioEncoderInferenceConfig(InferenceConfig):
    """Config for audio encoder transformer compilation."""

    def __init__(self, neuron_config, audio_config, **kwargs):
        # Store audio_config before calling super
        self.audio_config = audio_config
        if isinstance(audio_config, dict):
            self.audio_config = SimpleNamespace(**audio_config)
        super().__init__(neuron_config=neuron_config, **kwargs)

    def add_derived_config(self):
        pass

    def get_required_attributes(self):
        return []


# ---------------------------------------------------------------------------
# Full Audio Encoder (combines CPU frontend + Neuron transformer + CPU postprocessing)
# ---------------------------------------------------------------------------

class NeuronQwen25OmniAudioEncoder(nn.Module):
    """Qwen2.5-Omni Audio Encoder with Neuron acceleration.

    Architecture:
      CPU: mel → Conv1d frontend → positional embeddings → chunking
      Neuron (TP=4): 32 transformer layers with block-diagonal attention
      CPU: AvgPool → LayerNorm → Linear projection → audio embeddings

    Usage:
      1. Create with audio_config
      2. Call compile_transformer() to compile on Neuron
      3. Call load_transformer() to load compiled model
      4. Call forward() to encode audio
    """

    def __init__(self, audio_config, neuron_config=None, dtype=torch.bfloat16):
        super().__init__()
        if isinstance(audio_config, dict):
            audio_config = SimpleNamespace(**audio_config)
        self.audio_config = audio_config
        self.dtype = dtype
        self.n_window = audio_config.n_window

        # CPU components
        self.frontend = AudioCPUFrontend(audio_config, dtype=dtype)
        self.postprocessor = AudioCPUPostprocessor(audio_config, dtype=dtype)

        # Neuron transformer (initialized via compile/load cycle)
        self.transformer = None
        self._neuron_config = neuron_config

    def _get_feat_extract_output_lengths(self, input_lengths):
        """Compute output lengths after conv and avg_pool."""
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def _prepare_attention_mask(self, seq_length, cu_seqlens, dtype):
        """Create block-diagonal attention mask from cu_seqlens."""
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(dtype).min,
            dtype=dtype,
        )
        for i in range(1, len(cu_seqlens)):
            s, e = cu_seqlens[i - 1], cu_seqlens[i]
            attention_mask[..., s:e, s:e] = 0
        return attention_mask

    def forward(self, input_features, feature_lens, aftercnn_lens=None):
        """Process mel spectrogram through audio encoder.

        Args:
            input_features: (n_mels, total_mel_len) mel spectrogram
            feature_lens: (num_audios,) mel length for each audio
            aftercnn_lens: optional pre-computed lengths after conv

        Returns:
            audio_embeddings: (total_audio_tokens, output_dim) tensor
        """
        if aftercnn_lens is None:
            aftercnn_lens, _ = self._get_feat_extract_output_lengths(feature_lens)

        # CPU: Conv frontend + chunking
        hidden_states, aftercnn_lens_actual, cu_seqlens = self.frontend(
            input_features, feature_lens
        )

        if self.transformer is not None:
            # Neuron: transformer layers
            seq_len = hidden_states.shape[0]
            attention_mask = self._prepare_attention_mask(
                seq_len, cu_seqlens, self.dtype
            )
            # Add batch dimension for Neuron model
            hidden_states = hidden_states.unsqueeze(0)
            hidden_states = self.transformer(hidden_states, attention_mask)
            hidden_states = hidden_states.squeeze(0)
        else:
            # Fallback: CPU transformer (for testing without Neuron)
            logger.warning(
                "Audio transformer not compiled. Running transformer on CPU "
                "(this is slow and should only be used for testing)."
            )
            attention_mask = self._prepare_attention_mask(
                hidden_states.shape[0], cu_seqlens, self.dtype
            )
            # CPU fallback would require loading transformer weights separately
            raise RuntimeError(
                "Audio transformer must be compiled and loaded before inference. "
                "Call compile_transformer() and load_transformer() first."
            )

        # CPU: Postprocessing
        return self.postprocessor(hidden_states, aftercnn_lens_actual)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, dtype=torch.bfloat16):
        """Convert HF state dict to audio encoder format.

        Splits keys into three groups:
        - frontend.*: conv1, conv2, positional_embedding (CPU)
        - layers.*: transformer layers (Neuron)
        - postprocessor.*: ln_post, avg_pooler, proj, audio_bos_eos_token (CPU)

        Returns dict with prefixed keys for the split architecture.
        """
        new_state_dict = {}

        # LayerNorm keys that should remain float32
        ln_suffixes = (
            "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
            "final_layer_norm.weight", "final_layer_norm.bias",
            "ln_post.weight", "ln_post.bias",
        )

        # CPU frontend keys
        frontend_prefixes = ("conv1.", "conv2.", "positional_embedding.")
        # CPU postprocessor keys
        postprocessor_prefixes = ("ln_post.", "proj.", "avg_pooler.", "audio_bos_eos_token.")

        for key, value in state_dict.items():
            # Strip audio_tower prefix
            if key.startswith("thinker.audio_tower."):
                clean_key = key[len("thinker.audio_tower."):]
            elif key.startswith("audio_tower."):
                clean_key = key[len("audio_tower."):]
            else:
                new_state_dict[key] = value
                continue

            # Determine target dtype
            if any(clean_key.endswith(s) for s in ln_suffixes):
                target_dtype = torch.float32
            else:
                target_dtype = dtype

            # Route to correct sub-module
            if any(clean_key.startswith(p) for p in frontend_prefixes):
                new_state_dict["frontend." + clean_key] = (
                    value.clone().detach().contiguous().to(target_dtype)
                )
            elif any(clean_key.startswith(p) for p in postprocessor_prefixes):
                new_state_dict["postprocessor." + clean_key] = (
                    value.clone().detach().contiguous().to(target_dtype)
                )
            elif clean_key.startswith("layers."):
                # Transformer layers (will be loaded into Neuron model)
                new_state_dict["transformer." + clean_key] = (
                    value.clone().detach().contiguous().to(target_dtype)
                )
            else:
                logger.warning("Unknown audio key: %s", clean_key)

        return new_state_dict

    @staticmethod
    def from_pretrained_state_dict(audio_config, state_dict, dtype=torch.bfloat16):
        """Create audio encoder and load CPU weights from converted state dict.

        Note: Transformer weights need to be loaded separately via
        compile_transformer() + load_transformer() for Neuron execution.
        """
        encoder = NeuronQwen25OmniAudioEncoder(audio_config, dtype=dtype)

        # Load only frontend and postprocessor weights
        cpu_keys = {}
        for key, value in state_dict.items():
            if key.startswith("frontend.") or key.startswith("postprocessor."):
                cpu_keys[key] = value

        if cpu_keys:
            missing, unexpected = encoder.load_state_dict(cpu_keys, strict=False)
            # Filter out transformer keys from missing (expected)
            missing = [k for k in missing if not k.startswith("transformer.")]
            if missing:
                logger.warning("Audio encoder CPU missing keys: %s", missing[:10])
            logger.info("Loaded %d CPU weights into audio encoder", len(cpu_keys))

        return encoder
