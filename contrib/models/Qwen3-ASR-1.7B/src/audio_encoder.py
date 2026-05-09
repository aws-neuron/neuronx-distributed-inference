# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-ASR Audio Encoder for Neuron tracing.

The encoder uses a Whisper-like architecture with:
- Conv2D frontend (3 layers, stride 2, channel 480)
- 24 transformer encoder layers (bidirectional, d_model=1024)
- Output projector (1024 -> 2048 to match text decoder hidden_size)

This module provides StaticQwen3ASREncoder which rewrites the encoder
to be trace-friendly (no dynamic shapes, no cu_seqlens, static attention mask).

Bucket tracing strategy:
- Trace 3 encoder NEFFs for different audio durations: 5s/10s/30s
- At inference, select the smallest bucket that fits the input
- Pad input mel to bucket size, trim output to actual token count
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_encoder_output_length(T_mel: int) -> int:
    """Compute number of encoder output tokens from mel frame count."""
    input_lengths_leave = T_mel % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (T_mel // 100) * 13
    return output_lengths


def select_bucket(T_mel: int, buckets: List[int] = [500, 1000, 3000]) -> int:
    """Select smallest bucket that fits the mel frame count."""
    for b in buckets:
        if T_mel <= b:
            return b
    return buckets[-1]


class StaticQwen3ASREncoder(nn.Module):
    """Static (trace-friendly) wrapper for Qwen3-ASR audio encoder.

    Rewrites the encoder forward pass to use:
    - Fixed-size reshape instead of dynamic split/pad_sequence
    - Pre-computed block-diagonal attention mask (no cu_seqlens)
    - Eager attention instead of Flash Attention 2

    The encoder processes audio in chunks of 100 mel frames (1 second).
    Input must be padded to the nearest 100-frame boundary.

    Args:
        audio_tower: The HuggingFace Qwen3ASRAudioEncoder module
        fixed_T: Fixed mel frame count (must be multiple of 100)
    """

    def __init__(self, audio_tower, fixed_T: int):
        super().__init__()
        assert fixed_T % 100 == 0, f"fixed_T must be multiple of 100, got {fixed_T}"

        self.fixed_T = fixed_T
        self.n_chunks = fixed_T // 100

        # Copy convolution layers
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out

        # Positional embedding
        self.positional_embedding = audio_tower.positional_embedding

        # Transformer layers
        self.layers = audio_tower.layers

        # Post-processing
        self.ln_post = audio_tower.ln_post
        self.proj1 = audio_tower.proj1
        self.act = audio_tower.act
        self.proj2 = audio_tower.proj2

        # Pre-compute attention mask (block-diagonal)
        # Each chunk of 13 tokens attends only to itself
        tokens_per_chunk = 13
        total_tokens = self.n_chunks * tokens_per_chunk
        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)
        for i in range(self.n_chunks):
            start = i * tokens_per_chunk
            end = start + tokens_per_chunk
            mask[start:end, start:end] = True
        # Convert to float mask: 0 for attend, -inf for no attend
        self.register_buffer(
            "attention_mask", (~mask).float() * torch.finfo(torch.float32).min
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the static encoder.

        Args:
            input_features: Mel spectrogram [128, fixed_T]

        Returns:
            Audio embeddings [total_output_tokens, 2048]
        """
        # input_features: [128, T] -> add batch and channel: [1, 1, 128, T]
        x = input_features.unsqueeze(0).unsqueeze(0)

        # Conv2D frontend: [1, 1, 128, T] -> [1, 480, 16, T/8]
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        # Reshape to chunks: each chunk has T_chunk/8 time steps
        # x shape: [1, 480, 16, T/8]
        B, C, F_dim, T_dim = x.shape  # 1, 480, 16, fixed_T/8

        # Reshape: [n_chunks, 480, 16, chunk_time] where chunk_time = 100/8 = 12.5 -> 13
        # Actually, after 3x stride-2 conv: 100 -> 50 -> 25 -> 13 time steps per chunk
        tokens_per_chunk = T_dim // self.n_chunks  # Should be 13 for 100-frame chunks

        # Reshape [1, 480, 16, T/8] -> [n_chunks, 480, 16, tokens_per_chunk]
        x = x.squeeze(0)  # [480, 16, T/8]
        x = x.view(
            C, F_dim, self.n_chunks, tokens_per_chunk
        )  # [480, 16, n_chunks, tpc]
        x = x.permute(2, 0, 1, 3)  # [n_chunks, 480, 16, tpc]

        # Flatten freq and channel: [n_chunks, tpc, 480*16=7680]
        x = x.permute(0, 3, 1, 2)  # [n_chunks, tpc, 480, 16]
        x = x.reshape(
            self.n_chunks, tokens_per_chunk, C * F_dim
        )  # [n_chunks, tpc, 7680]

        # Linear projection: [n_chunks, tpc, 7680] -> [n_chunks, tpc, 1024]
        x = self.conv_out(x)

        # Add positional embedding
        x = x + self.positional_embedding.weight[:tokens_per_chunk]

        # Flatten to single sequence: [total_tokens, 1024]
        total_tokens = self.n_chunks * tokens_per_chunk
        x = x.reshape(total_tokens, -1)  # [total_tokens, 1024]

        # Transformer layers with block-diagonal attention
        for layer in self.layers:
            # Pre-norm
            residual = x
            x_norm = layer.self_attn_layer_norm(x)

            # Self-attention with block-diagonal mask
            # Q, K, V projections
            q = layer.self_attn.q_proj(x_norm)
            k = layer.self_attn.k_proj(x_norm)
            v = layer.self_attn.v_proj(x_norm)

            # Multi-head attention
            head_dim = q.shape[-1] // layer.self_attn.num_heads
            q = q.view(total_tokens, layer.self_attn.num_heads, head_dim).transpose(
                0, 1
            )
            k = k.view(total_tokens, layer.self_attn.num_heads, head_dim).transpose(
                0, 1
            )
            v = v.view(total_tokens, layer.self_attn.num_heads, head_dim).transpose(
                0, 1
            )

            # Scaled dot-product attention with mask
            scale = 1.0 / math.sqrt(head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = attn_weights + self.attention_mask.unsqueeze(0)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

            # Merge heads and project
            attn_output = attn_output.transpose(0, 1).reshape(total_tokens, -1)
            attn_output = layer.self_attn.out_proj(attn_output)

            x = residual + attn_output

            # FFN
            residual = x
            x_norm = layer.final_layer_norm(x)
            x = residual + layer.fc2(F.gelu(layer.fc1(x_norm)))

        # Post-processing: layernorm + projector
        x = self.ln_post(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)  # [total_tokens, 2048]

        return x


def trace_encoder(
    model_path: str,
    output_dir: str,
    buckets: List[int] = [500, 1000, 3000],
    compiler_args: Optional[List[str]] = None,
) -> Dict[int, str]:
    """Trace the audio encoder for multiple bucket sizes.

    Args:
        model_path: Path to HuggingFace Qwen3-ASR model
        output_dir: Directory to save traced encoder NEFFs
        buckets: List of mel frame counts to trace (default: 5s, 10s, 30s)
        compiler_args: Additional neuron compiler arguments

    Returns:
        Dict mapping bucket T -> saved NEFF path
    """
    import torch_neuronx
    from transformers import AutoConfig

    if compiler_args is None:
        compiler_args = [
            "--auto-cast",
            "matmult",
            "--auto-cast-type",
            "bf16",
            "--model-type",
            "transformer",
        ]

    # Load model
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Import model class
    import sys

    sys.path.insert(0, model_path)
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
    import torch

    hf_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float32
    )
    audio_tower = hf_model.thinker.audio_tower
    audio_tower.eval()

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    for T in buckets:
        print(f"  Tracing encoder for T={T} ({T // 100}s audio)...")
        encoder = StaticQwen3ASREncoder(audio_tower, T)
        encoder.eval()

        example_input = torch.randn(128, T)

        traced = torch_neuronx.trace(
            encoder,
            (example_input,),
            compiler_args=compiler_args,
            # DO NOT use inline_weights_to_neff=True (causes accuracy regression)
        )

        save_path = os.path.join(output_dir, f"encoder_T{T}.pt")
        traced.save(save_path)
        saved_paths[T] = save_path
        print(f"    Saved to {save_path}")

    return saved_paths


def load_encoders(
    encoder_dir: str,
    buckets: List[int] = [500, 1000, 3000],
    device: int = 0,
    warmup: bool = True,
) -> Dict[int, torch.jit.ScriptModule]:
    """Load traced encoder NEFFs and move to device.

    Args:
        encoder_dir: Directory containing encoder_T{bucket}.pt files
        buckets: Bucket sizes to load
        device: Neuron device ID
        warmup: Whether to run a warmup inference

    Returns:
        Dict mapping bucket T -> loaded traced model
    """
    import torch_neuronx

    encoders = {}
    for T in buckets:
        path = os.path.join(encoder_dir, f"encoder_T{T}.pt")
        enc = torch.jit.load(path)
        torch_neuronx.move_trace_to_device(enc, device)
        if warmup:
            _ = enc(torch.randn(128, T))
        encoders[T] = enc

    return encoders
