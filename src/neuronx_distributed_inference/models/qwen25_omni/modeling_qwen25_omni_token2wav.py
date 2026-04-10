# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni Token2Wav for NXD inference.
#
# CPU-based wrapper around HF's Qwen2_5OmniToken2WavModel.
# Converts codec tokens from the Talker into audio waveforms.
#
# Architecture:
#   - DiT (Diffusion Transformer): 22 blocks, dim=1024, 16 heads
#     - ECAPA-TDNN speaker encoder for speaker conditioning
#     - Codec embedding + RoPE + AdaLayerNorm
#     - ODE sampling (Runge-Kutta 4) for mel spectrogram generation
#   - BigVGAN vocoder: mel spectrogram -> waveform
#     - conv_pre(80->1536) + 6 upsample stages + AMPBlock residuals
#     - Snake activation, conv_post(24->1)
#
# Runs on CPU in float32 (required for ODE solver precision).
# Token2Wav has ~809 state dict keys total.

"""Qwen2.5-Omni Token2Wav model for NXD inference."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class NeuronQwen25OmniToken2Wav:
    """Wrapper around HF's Qwen2_5OmniToken2WavModel.

    Token2Wav converts codec tokens into audio waveforms through:
      1. DiT model: codec tokens + speaker embedding -> mel spectrogram
         (via ODE sampling with classifier-free guidance)
      2. BigVGAN vocoder: mel spectrogram -> waveform

    Speaker conditioning requires a speaker dict (spk_dict.pt) containing
    per-speaker 'cond' (conditioning) and 'ref_mel' (reference mel) tensors,
    plus 'bos_token' for the Talker.

    This wrapper:
      1. Instantiates the HF Token2Wav from config
      2. Loads weights from converted state dict
      3. Exposes waveform generation API
    """

    def __init__(self, token2wav_config):
        """Initialize Token2Wav.

        Args:
            token2wav_config: Token2Wav config (dict or HF config object).
                Must contain dit_config and bigvgan_config sub-configs.
        """
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
            Qwen2_5OmniToken2WavConfig,
        )
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniToken2WavModel,
        )

        if isinstance(token2wav_config, dict):
            token2wav_config = Qwen2_5OmniToken2WavConfig(**token2wav_config)

        self.model = Qwen2_5OmniToken2WavModel(token2wav_config)
        # Token2Wav must run in float32 for ODE solver precision
        self.model.float()
        self.model.eval()
        self.config = token2wav_config

    def load_state_dict(self, state_dict, strict=True):
        """Load converted state dict into the HF Token2Wav model."""
        return self.model.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def __call__(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        **kwargs,
    ):
        """Generate waveform from codec tokens.

        Args:
            code: (batch, seq_len) codec token IDs from the Talker
            conditioning: (batch, mel_len, enc_dim) speaker conditioning
                (from spk_dict.pt 'cond' key)
            reference_mel: (batch, mel_len, mel_dim) reference mel spectrogram
                (from spk_dict.pt 'ref_mel' key)
            num_steps: Number of ODE solver steps (default 10)
            guidance_scale: Classifier-free guidance scale (default 0.5)
            sway_coefficient: Time schedule sway (default -1.0)
            **kwargs: Additional kwargs passed to Token2Wav

        Returns:
            waveform: (samples,) audio waveform tensor on CPU
        """
        return self.model(
            code=code,
            conditioning=conditioning,
            reference_mel=reference_mel,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
            **kwargs,
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict) -> dict:
        """Convert HF state dict to Token2Wav format.

        Strips 'token2wav.' prefix from keys. Non-token2wav keys are passed through.

        Args:
            state_dict: Full or partial state dict with token2wav.* keys.

        Returns:
            State dict with token2wav prefix stripped.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("token2wav."):
                new_state_dict[key[len("token2wav."):]] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    @staticmethod
    def load_speaker_dict(speaker_dict_path):
        """Load speaker dictionary from spk_dict.pt.

        Args:
            speaker_dict_path: Path to spk_dict.pt

        Returns:
            dict: Speaker name -> {cond, ref_mel, bos_token}
        """
        return torch.load(speaker_dict_path, weights_only=True)

    @classmethod
    def from_pretrained_state_dict(cls, token2wav_config, state_dict):
        """Create Token2Wav and load weights from converted state dict.

        Args:
            token2wav_config: Token2Wav config (dict or HF config object)
            state_dict: Already-converted state dict (token2wav keys only)

        Returns:
            Initialized NeuronQwen25OmniToken2Wav
        """
        token2wav = cls(token2wav_config)

        # Filter to only token2wav keys (skip non-token2wav prefixes)
        t2w_keys = {}
        for key, value in state_dict.items():
            if any(
                key.startswith(p)
                for p in [
                    "lm_head.", "visual.", "audio_tower.",
                    "thinker.", "talker.", "token2wav.",
                ]
            ):
                continue
            t2w_keys[key] = value

        missing, unexpected = token2wav.load_state_dict(t2w_keys, strict=False)
        if missing:
            logger.warning("Token2Wav missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("Token2Wav unexpected keys: %s", unexpected[:10])
        logger.info("Loaded %d weights into Token2Wav", len(t2w_keys))

        return token2wav
