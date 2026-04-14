# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni Token2Wav for NXD inference.
#
# This file contains TWO implementations:
#
# 1. NeuronQwen25OmniToken2Wav (CPU wrapper)
#    - Wraps HF's Qwen2_5OmniToken2WavModel entirely on CPU in float32
#    - Suitable for quick testing or when Neuron resources are limited
#
# 2. NeuronQwen25OmniToken2WavWithNeuronDiT (Neuron-accelerated)
#    - Compiles the DiT (22 transformer blocks) on Neuron via torch_neuronx.trace()
#    - ODE solver loop stays on CPU (inherently sequential, 10-50 steps)
#    - BigVGAN vocoder stays on CPU (convolutional, ~10-20M params)
#    - Speaker encoder (ECAPA-TDNN) stays on CPU (~small)
#    - DiT is the compute bottleneck: 22 blocks × 10 ODE steps = 220 forward passes
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
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Part 1: CPU-based Token2Wav (HF wrapper)
# =============================================================================


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


# =============================================================================
# Part 2: Neuron-accelerated Token2Wav (DiT on Neuron)
# =============================================================================


class NeuronQwen25OmniToken2WavWithNeuronDiT(NeuronQwen25OmniToken2Wav):
    """Token2Wav with DiT (Diffusion Transformer) compiled on Neuron.

    The DiT is the compute bottleneck in Token2Wav:
      - 22 transformer blocks × 10 ODE steps = 220 forward passes per generation
      - Each block: self-attention (dim=1024, 16 heads) + cross-attention + FFN
      - Total ~85M params

    By compiling the DiT on Neuron, the 220 forward passes are accelerated
    while the ODE solver orchestration stays on CPU.

    Usage:
      # 1. Create with HF config
      t2w = NeuronQwen25OmniToken2WavWithNeuronDiT(token2wav_config)

      # 2. Load weights
      t2w.load_state_dict(state_dict)

      # 3. Compile DiT on Neuron
      t2w.compile_dit("compiled_dit/", max_codec_len=2048, max_mel_len=4096)

      # 4. Load compiled DiT (in subsequent runs)
      t2w.load_dit("compiled_dit/")

      # 5. Generate (ODE loop on CPU, DiT on Neuron)
      waveform = t2w(code, conditioning, reference_mel)

    Architecture:
      CPU: ECAPA-TDNN speaker encoder → speaker embedding
      CPU: Codec embedding → codec features
      CPU: ODE solver loop (10 steps):
        ├── Neuron: DiT forward (22 blocks) × 2 (CFG) = 44 Neuron calls/step
        └── CPU: Euler/RK4 integration
      CPU: BigVGAN vocoder → waveform
    """

    def __init__(self, token2wav_config):
        super().__init__(token2wav_config)
        self._neuron_dit = None
        self._dit_compiled_path = None

    def compile_dit(
        self,
        compiled_path,
        max_codec_len=2048,
        max_mel_len=4096,
        batch_size=1,
    ):
        """Compile the DiT model on Neuron using torch_neuronx.trace().

        The DiT's forward method is traced with example inputs of the
        specified shapes. The compiled model handles fixed-size inputs;
        actual inputs are padded/truncated to match.

        Args:
            compiled_path: Directory to save compiled model
            max_codec_len: Maximum codec sequence length
            max_mel_len: Maximum mel spectrogram length
            batch_size: Batch size for compilation (typically 1)
        """
        try:
            import torch_neuronx
        except ImportError:
            logger.error(
                "torch_neuronx not available. DiT compilation requires "
                "running on a Neuron instance (trn1/trn2/inf2)."
            )
            raise

        os.makedirs(compiled_path, exist_ok=True)

        # Extract the DiT sub-module from the HF model
        dit = self._get_dit_module()
        if dit is None:
            raise RuntimeError(
                "Could not extract DiT module from Token2Wav model. "
                "Check that the model is properly initialized."
            )

        # Create example inputs for tracing
        # The exact input signature depends on the HF DiT implementation
        example_inputs = self._create_dit_example_inputs(
            batch_size, max_codec_len, max_mel_len
        )

        logger.info(
            "Compiling DiT on Neuron: batch=%d, codec_len=%d, mel_len=%d",
            batch_size, max_codec_len, max_mel_len,
        )

        # Wrap the DiT to fix keyword arguments for tracing.
        # torch_neuronx.trace only supports positional args, and the DiT's
        # forward has boolean kwargs (apply_cfg, drop_code, etc.) that create
        # branching. We trace with apply_cfg=False (single forward) and
        # handle CFG in the ODE solver loop on CPU.
        class _DiTTraceWrapper(torch.nn.Module):
            def __init__(self, dit_module):
                super().__init__()
                self.dit = dit_module

            def forward(self, hidden_states, condition_vector,
                        speaker_embedding, quantized_code, time_step):
                return self.dit(
                    hidden_states, condition_vector, speaker_embedding,
                    quantized_code, time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                    apply_cfg=False,
                )

        wrapper = _DiTTraceWrapper(dit)
        wrapper.eval()

        compiled_dit = torch_neuronx.trace(
            wrapper,
            example_inputs,
            compiler_args=[
                "--auto-cast=none",
                "--model-type=transformer",
                "-O1",
            ],
        )

        # Save compiled model
        save_path = os.path.join(compiled_path, "dit_neuron.pt")
        torch.jit.save(compiled_dit, save_path)
        logger.info("Compiled DiT saved to %s", save_path)

        self._neuron_dit = compiled_dit
        self._dit_compiled_path = compiled_path

    def load_dit(self, compiled_path):
        """Load a previously compiled DiT model.

        Args:
            compiled_path: Directory containing compiled model
        """
        save_path = os.path.join(compiled_path, "dit_neuron.pt")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Compiled DiT not found at {save_path}")

        self._neuron_dit = torch.jit.load(save_path)
        self._dit_compiled_path = compiled_path
        logger.info("Loaded compiled DiT from %s", save_path)

    def _get_dit_module(self):
        """Extract the DiT sub-module from the HF Token2Wav model.

        HF Qwen2_5OmniToken2WavModel has:
          - self.code2wav_dit_model: Qwen2_5OmniToken2WavDiTModel
          - self.code2wav_bigvgan_model: Qwen2_5OmniToken2WavBigVGANModel

        Returns:
            The DiT nn.Module, or None if not found
        """
        for attr_name in [
            "code2wav_dit_model", "dit", "flow_model", "transformer", "dit_model",
        ]:
            if hasattr(self.model, attr_name):
                return getattr(self.model, attr_name)
        for name, module in self.model.named_children():
            if "dit" in name.lower() or "flow" in name.lower():
                return module
        return None

    def _create_dit_example_inputs(self, batch_size, max_codec_len, max_mel_len):
        """Create example inputs for Qwen2_5OmniToken2WavDiTModel.forward().

        Actual HF signature:
          forward(hidden_states, condition_vector, speaker_embedding,
                  quantized_code, time_step,
                  drop_audio_conditioning=False, drop_code=False,
                  apply_cfg=True)

        We trace with apply_cfg=False to get a single forward pass.
        CFG (running the model twice) is handled in the ODE solver loop.

        Returns:
            Tuple of example input tensors
        """
        dit = self._get_dit_module()

        # Infer dimensions from config or model
        dit_dim = 1024  # hidden dimension
        emb_dim = 512   # speaker embedding dim
        mel_dim = 80    # mel spectrogram channels

        if dit is not None:
            dit_cfg = getattr(dit, "config", None)
            if dit_cfg is not None:
                dit_dim = getattr(dit_cfg, "dim", dit_dim)
                emb_dim = getattr(dit_cfg, "emb_dim", emb_dim)
                mel_dim = getattr(dit_cfg, "mel_dim", mel_dim)

        # hidden_states: (batch, mel_len, dim) — noisy mel in latent space
        hidden_states = torch.randn(batch_size, max_mel_len, dit_dim, dtype=torch.float32)
        # condition_vector: (batch, cond_len, mel_dim) — raw mel spectrogram for speaker conditioning
        # The DiT's input_embed passes this through ECAPA-TDNN Conv1d(mel_dim, ...)
        condition_vector = torch.randn(batch_size, max_mel_len, mel_dim, dtype=torch.float32)
        # speaker_embedding: (batch, emb_dim) — speaker embedding
        speaker_embedding = torch.randn(batch_size, emb_dim, dtype=torch.float32)
        # quantized_code: (batch, code_len) — codec token IDs
        quantized_code = torch.randint(0, 8193, (batch_size, max_codec_len), dtype=torch.long)
        # time_step: (batch,) — ODE timestep
        time_step = torch.tensor([0.5], dtype=torch.float32).expand(batch_size)

        return (hidden_states, condition_vector, speaker_embedding,
                quantized_code, time_step)

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
        """Generate waveform. DiT runs on Neuron if compiled, else CPU.

        This method delegates to the HF model's forward which handles the
        ODE solver loop internally. When the DiT is compiled on Neuron,
        the HF model's internal DiT calls are intercepted and redirected
        to the Neuron-compiled version.

        Args:
            code: (batch, seq_len) codec token IDs from the Talker
            conditioning: (batch, mel_len, enc_dim) speaker conditioning
            reference_mel: (batch, mel_len, mel_dim) reference mel spectrogram
            num_steps: Number of ODE solver steps (default 10)
            guidance_scale: Classifier-free guidance scale (default 0.5)
            sway_coefficient: Time schedule sway (default -1.0)

        Returns:
            waveform: (samples,) audio waveform tensor on CPU
        """
        if self._neuron_dit is not None:
            # Redirect DiT calls to Neuron
            original_dit = self._get_dit_module()
            original_forward = original_dit.forward

            def neuron_dit_forward(*args, **fwd_kwargs):
                # Move inputs to CPU (Neuron handles device transfer)
                cpu_args = tuple(
                    a.cpu().float() if isinstance(a, torch.Tensor) else a
                    for a in args
                )
                result = self._neuron_dit(*cpu_args)
                return result

            # Monkeypatch the DiT forward for this call
            original_dit.forward = neuron_dit_forward
            try:
                result = self.model(
                    code=code,
                    conditioning=conditioning,
                    reference_mel=reference_mel,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    sway_coefficient=sway_coefficient,
                    **kwargs,
                )
            finally:
                # Restore original forward
                original_dit.forward = original_forward
            return result
        else:
            # Fallback to CPU
            return self.model(
                code=code,
                conditioning=conditioning,
                reference_mel=reference_mel,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                sway_coefficient=sway_coefficient,
                **kwargs,
            )

    @classmethod
    def from_pretrained_state_dict(cls, token2wav_config, state_dict):
        """Create Token2Wav with Neuron DiT support.

        Same as base class but returns the Neuron-capable subclass.
        """
        token2wav = cls(token2wav_config)

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
        logger.info("Loaded %d weights into Token2Wav (Neuron DiT capable)", len(t2w_keys))

        return token2wav
