"""Qwen3-Omni Audio Encoder for NxD Inference.

Conv2d frontend (3 layers, downsample_hidden_size=480) +
32 transformer layers (d_model=1280, heads=20, ffn=5120) on Neuron +
proj1 + GELU + proj2 postprocessor on CPU.

Key differences from Qwen2.5-Omni audio encoder:
  - Conv2d frontend (2D mel processing) instead of Conv1d
  - All attention projections have bias=True (k_proj included)
  - proj1 + GELU + proj2 output instead of AvgPool + single proj
  - No audio_bos_eos_token
  - Different output length calculation (3-stage Conv2d downsampling)
"""

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

logger = logging.getLogger(__name__)


def _get_feat_extract_output_lengths(input_lengths):
    """Compute output lengths after 3x Conv2d (stride=2 each) + chunk-aware calculation.

    Matches HF Qwen3OmniMoeAudioEncoder._get_feat_extract_output_lengths.
    Each Conv2d with stride=2 halves the time dimension: (L-1)//2 + 1.
    The chunking introduces a correction factor of 13 per full window.
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


# ---------------------------------------------------------------------------
# CPU components
# ---------------------------------------------------------------------------

class SinusoidsPositionEmbedding(nn.Module):
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
    """Conv2d frontend + positional embeddings + chunking (CPU).

    3x Conv2d layers with stride=2 each: mel (1, 128, T) -> (480, freq_reduced, T/8).
    Then linear projection to d_model and sinusoidal positional embeddings.
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
        downsample_hidden_size = getattr(audio_config, "downsample_hidden_size", 480)

        self.conv2d1 = nn.Conv2d(1, downsample_hidden_size, 3, 2, padding=1, dtype=dtype)
        self.conv2d2 = nn.Conv2d(
            downsample_hidden_size, downsample_hidden_size, 3, 2, padding=1, dtype=dtype
        )
        self.conv2d3 = nn.Conv2d(
            downsample_hidden_size, downsample_hidden_size, 3, 2, padding=1, dtype=dtype
        )
        # After 3x stride-2 convs on freq axis: ((128+1)//2+1)//2+1)//2 = 16 (actually 17, let's compute)
        # freq=128: after conv1 (s=2,p=1): (128+2*1-3)//2+1 = 64
        # after conv2: (64+2*1-3)//2+1 = 32
        # after conv3: (32+2*1-3)//2+1 = 16
        # HF formula: ((((num_mel_bins+1)//2+1)//2+1)//2 which is the same
        freq_reduced = (((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            downsample_hidden_size * freq_reduced, d_model, bias=False, dtype=dtype
        )
        self.positional_embedding = SinusoidsPositionEmbedding(
            max_source_positions, d_model
        )

    def forward(self, input_features, feature_lens):
        """Process mel spectrogram through Conv2d frontend.

        Args:
            input_features: (n_mels, total_mel_len) mel spectrogram
            feature_lens: (num_audios,) mel length for each audio

        Returns:
            hidden_states: (total_valid_tokens, d_model)
            aftercnn_lens: (num_audios,) valid tokens per audio
            cu_seqlens: cumulative sequence lengths for attention masking
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)

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

        # Split mel into chunks, pad to max chunk length
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)  # (num_chunks, mel_bins, max_chunk_time)

        # Compute per-chunk output lengths after 3x Conv2d
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device)
             for length in feature_lens_after_cnn],
            batch_first=True,
        )

        # Conv2d frontend: (num_chunks, 1, mel_bins, time) -> (num_chunks, 480, freq, time)
        padded_feature = padded_feature.unsqueeze(1)
        padded_embed = F.gelu(self.conv2d1(padded_feature))
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        padded_embed = F.gelu(self.conv2d3(padded_embed))

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq) -> linear -> (batch, time, d_model)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        # Add positional embeddings
        positional_embedding = (
            self.positional_embedding.positional_embedding[:padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding

        # Flatten valid tokens
        hidden_states = padded_embed[padded_mask_after_cnn]

        # Compute cu_seqlens for block-diagonal attention
        cu_seqlens = torch.cat([
            torch.zeros(1, device=feature_lens.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0).to(torch.int32),
        ])

        return hidden_states, aftercnn_lens, cu_seqlens, padded_mask_after_cnn


class AudioCPUPostprocessor(nn.Module):
    """LayerNorm + proj1 + GELU + proj2 (CPU).

    No AvgPool (unlike Qwen2.5-Omni). No audio_bos_eos_token.
    """

    def __init__(self, audio_config, dtype=torch.bfloat16):
        super().__init__()
        if isinstance(audio_config, dict):
            audio_config = SimpleNamespace(**audio_config)

        d_model = audio_config.d_model  # 1280
        output_dim = audio_config.output_dim  # 3584

        self.ln_post = nn.LayerNorm(d_model)  # stays float32
        self.proj1 = nn.Linear(d_model, d_model, dtype=dtype)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_model, output_dim, dtype=dtype)

    def forward(self, hidden_states):
        """Post-process transformer output.

        Args:
            hidden_states: (total_tokens, d_model) transformer output

        Returns:
            audio_embeddings: (total_tokens, output_dim)
        """
        hidden_states = self.ln_post(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Neuron-compiled transformer components (TP-parallel)
# ---------------------------------------------------------------------------

class NeuronAudioAttention(nn.Module):
    """TP-parallel self-attention for audio encoder.

    All projections have bias=True (unlike Qwen2.5-Omni where k_proj has no bias).

    To support tp_degree values that do not evenly divide num_heads (the Qwen3-Omni
    audio tower has 20 heads, which does not divide the text model's TP=8), we pad
    the head count up to the next multiple of tp_degree and zero-fill the added
    heads' QKV/output weights. The padding heads produce zeros and are discarded
    by the output projection.
    """

    def __init__(self, d_model, num_heads, tp_degree, dtype=torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.num_heads_orig = num_heads
        self.head_dim = d_model // num_heads
        # Round num_heads up to the next multiple of tp_degree.
        self.num_heads = ((num_heads + tp_degree - 1) // tp_degree) * tp_degree
        self.num_heads_per_rank = self.num_heads // tp_degree
        # Hidden size used internally (may be padded beyond the upstream d_model).
        self.padded_hidden = self.num_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # q/k/v are ColumnParallel: output is split by tp. Use padded_hidden as the
        # output size so each rank gets exactly num_heads_per_rank heads.
        self.q_proj = ColumnParallelLinear(
            d_model, self.padded_hidden, bias=True, gather_output=False, dtype=dtype,
        )
        self.k_proj = ColumnParallelLinear(
            d_model, self.padded_hidden, bias=True, gather_output=False, dtype=dtype,
        )
        self.v_proj = ColumnParallelLinear(
            d_model, self.padded_hidden, bias=True, gather_output=False, dtype=dtype,
        )
        # out_proj is RowParallel: input is split by tp, output is d_model.
        self.out_proj = RowParallelLinear(
            self.padded_hidden, d_model, bias=True, input_is_parallel=True, dtype=dtype,
        )

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


class NeuronAudioEncoderLayer(nn.Module):
    """Pre-norm transformer layer with TP parallelism."""

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
    """Handles bucketing, padding, and Neuron compilation for audio transformer."""

    def __init__(self, config, model_cls, tag="", compiler_args=None,
                 priority_model_idx=None, pipeline_execution=True,
                 return_ranked_to_cpu=False, model_init_kwargs={}):
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx,
            pipeline_execution, return_ranked_to_cpu, model_init_kwargs,
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
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
        for bucket in self.config.neuron_config.buckets:
            if bucket >= seq_len:
                return bucket
        raise ValueError(
            f"No bucket found for seq_len={seq_len}. "
            f"Buckets: {self.config.neuron_config.buckets}"
        )

    def forward(self, hidden_states, attention_mask):
        if self.model is None:
            raise RuntimeError("Forward called before load.")

        seq_len = hidden_states.shape[1]
        bucket = self.get_target_bucket(seq_len)

        if seq_len < bucket:
            pad_len = bucket - seq_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            dtype = attention_mask.dtype
            mask_pad = torch.full(
                (1, 1, bucket, bucket),
                torch.finfo(dtype).min,
                dtype=dtype,
            )
            mask_pad[:, :, :seq_len, :seq_len] = attention_mask
            attention_mask = mask_pad

        output = self._forward(hidden_states, attention_mask)
        # Handle ranked output from Neuron pipeline execution
        if isinstance(output, list):
            output = output[0][0]
        if output.device.type != "cpu":
            output = output.to("cpu")
        return output[:, :seq_len, :]


class NeuronQwen3OmniForAudioEncoding(NeuronApplicationBase):
    """Neuron application for audio encoder transformer layers."""

    _model_cls = NeuronAudioTransformerModel

    def __init__(self, model_path, config=None, neuron_config=None, transformer_state_dict=None):
        # NeuronApplicationBase signature: (model_path, config, neuron_config)
        super().__init__(model_path=model_path, config=config, neuron_config=neuron_config)
        self._transformer_state_dict = transformer_state_dict
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

    def checkpoint_loader_fn(self, mmap: bool = False):
        """Return pre-converted transformer-only state dict.

        Bypass NeuronApplicationBase's HF-loading path since we only have
        audio_tower weights, already extracted by the caller.
        """
        if self._transformer_state_dict is None:
            raise RuntimeError(
                "transformer_state_dict must be provided to NeuronQwen3OmniForAudioEncoding "
                "for compilation/sharding."
            )
        # Only keep transformer.layers.* weights (drop frontend/postprocessor).
        sd = {}
        for k, v in self._transformer_state_dict.items():
            if k.startswith("transformer.layers."):
                new_key = k[len("transformer."):]
                sd[new_key] = v
        return sd

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        raise NotImplementedError(
            "NeuronQwen3OmniForAudioEncoding loads weights via checkpoint_loader_fn, "
            "not load_hf_model."
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, inference_config):
        """Extract audio_tower.layers.* keys and strip prefix for Neuron."""
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
        self.audio_config = audio_config
        if isinstance(audio_config, dict):
            self.audio_config = SimpleNamespace(**audio_config)
        super().__init__(neuron_config=neuron_config, **kwargs)
        self.num_cores_per_group = 1

    def add_derived_config(self):
        pass

    def get_required_attributes(self):
        return []


# ---------------------------------------------------------------------------
# Full Audio Encoder
# ---------------------------------------------------------------------------

class NeuronQwen3OmniAudioEncoder(nn.Module):
    """Qwen3-Omni Audio Encoder with Neuron acceleration.

    CPU: mel -> Conv2d frontend -> positional embeddings -> chunking
    Neuron (TP): 32 transformer layers with block-diagonal attention
    CPU: LayerNorm -> proj1 -> GELU -> proj2 -> audio embeddings
    """

    def __init__(self, audio_config, neuron_config=None, dtype=torch.bfloat16):
        super().__init__()
        if isinstance(audio_config, dict):
            audio_config = SimpleNamespace(**audio_config)
        self.audio_config = audio_config
        self.dtype = dtype
        self.n_window = audio_config.n_window
        self.n_window_infer = getattr(audio_config, "n_window_infer", 400)

        self.frontend = AudioCPUFrontend(audio_config, dtype=dtype)
        self.postprocessor = AudioCPUPostprocessor(audio_config, dtype=dtype)
        self.transformer = None
        self._neuron_config = neuron_config

    def _prepare_attention_mask(self, seq_length, cu_seqlens, dtype):
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(dtype).min,
            dtype=dtype,
        )
        for i in range(1, len(cu_seqlens)):
            s, e = cu_seqlens[i - 1], cu_seqlens[i]
            attention_mask[..., s:e, s:e] = 0
        return attention_mask

    def _compute_inference_cu_seqlens(self, aftercnn_lens, padded_mask_after_cnn):
        """Compute cu_seqlens using n_window_infer chunking (matches HF forward).

        The HF encoder uses n_window_infer to sub-chunk the attention windows
        for inference efficiency: each audio is split into sub-windows of size
        window_aftercnn = mask_time_dim * (n_window_infer / (n_window * 2)).
        """
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            self.n_window_infer // (self.n_window * 2)
        )
        cu_chunk_lens = [0]
        for cnn_len in aftercnn_lens:
            cnn_len = cnn_len.item()
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(
            cu_chunk_lens, device=aftercnn_lens.device
        ).cumsum(-1, dtype=torch.int32)
        return cu_seqlens

    def forward(self, input_features, feature_lens, aftercnn_lens=None):
        """Process mel spectrogram through audio encoder.

        Args:
            input_features: (n_mels, total_mel_len) mel spectrogram
            feature_lens: (num_audios,) mel length for each audio

        Returns:
            audio_embeddings: (total_audio_tokens, output_dim)
        """
        # CPU: Conv2d frontend + chunking
        hidden_states, aftercnn_lens_actual, _, padded_mask_after_cnn = self.frontend(
            input_features, feature_lens
        )

        if self.transformer is None:
            raise RuntimeError(
                "Audio transformer must be compiled and loaded before inference."
            )

        # Neuron: transformer layers
        seq_len = hidden_states.shape[0]
        # HF uses n_window_infer-based sub-window chunking (window_aftercnn tokens
        # per block) rather than one block per n_window chunk. Short audios
        # produce the same block-diagonal structure either way, but for long
        # audios the basic (per-n_window) grouping creates too-narrow attention
        # windows (13 tokens) and the encoder output degrades into noise.
        cu_seqlens = self._compute_inference_cu_seqlens(
            aftercnn_lens_actual, padded_mask_after_cnn
        )
        attention_mask = self._prepare_attention_mask(
            seq_len, cu_seqlens, self.dtype
        )
        hidden_states = hidden_states.unsqueeze(0)
        hidden_states = self.transformer(hidden_states, attention_mask)
        hidden_states = hidden_states.squeeze(0)

        # CPU: Postprocessing (ln_post + proj1 + GELU + proj2)
        return self.postprocessor(hidden_states)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, dtype=torch.bfloat16,
                                        tp_degree=1, num_heads=20, head_dim=64):
        """Convert HF state dict to split architecture format.

        Prefixes keys for three groups:
        - frontend.*: conv2d1, conv2d2, conv2d3, conv_out, positional_embedding (CPU)
        - layers.*: transformer layers (Neuron)
        - postprocessor.*: ln_post, proj1, proj2 (CPU)

        When tp_degree does not divide num_heads, pad q/k/v/out_proj weights along
        the head dimension with zeros so each TP rank gets an integer number of
        heads. The added heads produce zero output; the out_proj's zero columns
        ensure they don't affect the residual stream.
        """
        new_state_dict = {}

        ln_suffixes = (
            "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
            "final_layer_norm.weight", "final_layer_norm.bias",
            "ln_post.weight", "ln_post.bias",
        )

        frontend_prefixes = (
            "conv2d1.", "conv2d2.", "conv2d3.", "conv_out.", "positional_embedding.",
        )
        postprocessor_prefixes = ("ln_post.", "proj1.", "proj2.")

        orig_hidden = num_heads * head_dim
        padded_heads = ((num_heads + tp_degree - 1) // tp_degree) * tp_degree
        padded_hidden = padded_heads * head_dim
        pad_out = padded_hidden - orig_hidden  # rows to add to q/k/v

        def _pad_attn(clean_key, tensor):
            """Zero-pad q/k/v/out_proj tensors along the head dim."""
            if pad_out == 0:
                return tensor
            if (clean_key.endswith(".self_attn.q_proj.weight")
                    or clean_key.endswith(".self_attn.k_proj.weight")
                    or clean_key.endswith(".self_attn.v_proj.weight")):
                # shape: (orig_hidden, d_model) -> (padded_hidden, d_model)
                return F.pad(tensor, (0, 0, 0, pad_out))
            if (clean_key.endswith(".self_attn.q_proj.bias")
                    or clean_key.endswith(".self_attn.k_proj.bias")
                    or clean_key.endswith(".self_attn.v_proj.bias")):
                # shape: (orig_hidden,) -> (padded_hidden,)
                return F.pad(tensor, (0, pad_out))
            if clean_key.endswith(".self_attn.out_proj.weight"):
                # shape: (d_model, orig_hidden) -> (d_model, padded_hidden)
                return F.pad(tensor, (0, pad_out))
            return tensor

        for key, value in state_dict.items():
            if key.startswith("thinker.audio_tower."):
                clean_key = key[len("thinker.audio_tower."):]
            elif key.startswith("audio_tower."):
                clean_key = key[len("audio_tower."):]
            else:
                new_state_dict[key] = value
                continue

            if any(clean_key.endswith(s) for s in ln_suffixes):
                target_dtype = torch.float32
            else:
                target_dtype = dtype

            if any(clean_key.startswith(p) for p in frontend_prefixes):
                new_state_dict["frontend." + clean_key] = (
                    value.clone().detach().contiguous().to(target_dtype)
                )
            elif any(clean_key.startswith(p) for p in postprocessor_prefixes):
                new_state_dict["postprocessor." + clean_key] = (
                    value.clone().detach().contiguous().to(target_dtype)
                )
            elif clean_key.startswith("layers."):
                padded = _pad_attn(clean_key, value)
                new_state_dict["transformer." + clean_key] = (
                    padded.clone().detach().contiguous().to(target_dtype)
                )
            else:
                logger.warning("Unknown audio key: %s", clean_key)

        return new_state_dict

    @staticmethod
    def from_pretrained_state_dict(audio_config, state_dict, dtype=torch.bfloat16):
        """Create audio encoder and load CPU weights from converted state dict."""
        encoder = NeuronQwen3OmniAudioEncoder(audio_config, dtype=dtype)

        cpu_keys = {}
        for key, value in state_dict.items():
            if key.startswith("frontend.") or key.startswith("postprocessor."):
                cpu_keys[key] = value

        if cpu_keys:
            missing, unexpected = encoder.load_state_dict(cpu_keys, strict=False)
            missing = [k for k in missing if not k.startswith("transformer.")]
            if missing:
                logger.warning("Audio encoder CPU missing keys: %s", missing[:10])
            logger.info("Loaded %d CPU weights into audio encoder", len(cpu_keys))

        return encoder
