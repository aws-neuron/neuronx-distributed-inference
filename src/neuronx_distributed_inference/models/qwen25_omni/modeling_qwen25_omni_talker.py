# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni Talker for NXD inference.
#
# CPU-based wrapper around HF's Qwen2_5OmniTalkerForConditionalGeneration.
# The Talker is a small Qwen2 decoder that converts Thinker hidden states
# into codec tokens for speech synthesis.
#
# Architecture:
#   - embed_tokens: Embedding(8448, 3584) — codec vocab in Thinker's dim space
#   - thinker_to_talker_proj: Linear(3584 -> 896)
#   - 24 Qwen2 decoder layers (GQA: 12 heads, 4 kv_heads, head_dim=128)
#   - MLP: SiLU gate_proj/up_proj(896->18944), down_proj(18944->896)
#   - RMSNorm(896)
#   - codec_head: Linear(896 -> 8448, no bias)
#
# Runs on CPU for the following reasons:
#   1. Non-standard head_dim: hidden_size=896 with 12 heads gives a fractional
#      head_dim (74.67). The actual head_dim=128 means the attention's internal
#      dimension (12×128=1536) differs from hidden_size (896), which is
#      incompatible with NxDI's NeuronAttentionBase (computes head_dim as
#      hidden_size // num_attention_heads).
#   2. 3D mRoPE: The Talker uses multimodal rotary position embeddings with
#      position_ids of shape (3, batch, seq). The initial positions depend on
#      the Thinker's text output via get_rope_index(), requiring access to
#      input_text_ids, image/video/audio grid info at each step.
#   3. Custom input pipeline: Every autoregressive step combines codec token
#      embeddings (3584-d) with Thinker hidden states, then projects to 896-d.
#      This per-step thinker-state injection doesn't fit NeuronBaseForCausalLM.
#   4. Small model: ~690M params in 24 layers — not a performance bottleneck
#      compared to the 7B Thinker running on Neuron.

"""Qwen2.5-Omni Talker model for NXD inference."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class NeuronQwen25OmniTalker:
    """Wrapper around HF's Qwen2_5OmniTalkerForConditionalGeneration.

    The Talker takes Thinker hidden states + codec embeddings as input,
    projects them from embedding_size (3584) to hidden_size (896), runs
    through 24 Qwen2 decoder layers, and outputs codec tokens via a
    codec_head linear layer.

    This wrapper:
      1. Instantiates the HF Talker from config
      2. Loads weights from converted state dict
      3. Exposes generation API for codec token synthesis
    """

    def __init__(self, talker_config, dtype=torch.bfloat16):
        """Initialize the Talker.

        Args:
            talker_config: Talker config (dict or HF config object).
                Must contain: vocab_size, embedding_size, hidden_size,
                num_hidden_layers, num_attention_heads, num_key_value_heads,
                intermediate_size, etc.
            dtype: Model dtype (default bfloat16).
        """
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
            Qwen2_5OmniTalkerConfig,
        )
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniTalkerForConditionalGeneration,
        )

        if isinstance(talker_config, dict):
            talker_config = Qwen2_5OmniTalkerConfig(**talker_config)

        self.model = Qwen2_5OmniTalkerForConditionalGeneration(talker_config)
        self.model.to(dtype)
        self.model.eval()
        self.config = talker_config
        self.dtype = dtype

        # Expose codec token IDs for orchestration
        self.codec_bos_token = talker_config.tts_codec_start_token_id
        self.codec_eos_token = talker_config.tts_codec_end_token_id
        self.codec_pad_token = talker_config.tts_codec_pad_token_id
        self.codec_mask_token = talker_config.tts_codec_mask_token_id
        self.text_bos_token = talker_config.tts_text_start_token_id
        self.text_eos_token = talker_config.tts_text_end_token_id
        self.text_pad_token = talker_config.tts_text_pad_token_id

    def load_state_dict(self, state_dict, strict=True):
        """Load converted state dict into the HF Talker model."""
        return self.model.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        input_text_ids,
        thinker_reply_part,
        inputs_embeds,
        attention_mask=None,
        max_new_tokens=4096,
        do_sample=True,
        top_k=40,
        top_p=0.8,
        temperature=0.9,
        eos_token_id=None,
        repetition_penalty=1.05,
        suppress_tokens=None,
        **kwargs,
    ):
        """Generate codec tokens from Thinker hidden states.

        Args:
            input_ids: (batch, seq_len) codec input IDs
            input_text_ids: (batch, seq_len) text input IDs (for position calc)
            thinker_reply_part: (batch, reply_len, 3584) Thinker hidden states
            inputs_embeds: (batch, seq_len, 3584) input embeddings
            attention_mask: (batch, seq_len) optional attention mask
            max_new_tokens: Maximum codec tokens to generate
            do_sample: Whether to sample (vs greedy)
            top_k: Top-k sampling
            top_p: Nucleus sampling probability
            temperature: Sampling temperature
            eos_token_id: EOS token(s) for stopping
            repetition_penalty: Repetition penalty
            suppress_tokens: Token IDs to suppress during generation
            **kwargs: Additional generation kwargs

        Returns:
            (batch, total_len) generated codec token IDs
        """
        if eos_token_id is None:
            eos_token_id = [self.codec_eos_token, self.codec_pad_token]
        if suppress_tokens is None:
            suppress_tokens = [self.codec_bos_token]

        return self.model.generate(
            input_ids=input_ids,
            input_text_ids=input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            suppress_tokens=suppress_tokens,
            **kwargs,
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict) -> dict:
        """Convert HF state dict to Talker format.

        Strips 'talker.' prefix from keys. Non-talker keys are passed through.

        Args:
            state_dict: Full or partial state dict with talker.* keys.

        Returns:
            State dict with talker prefix stripped.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("talker."):
                new_state_dict[key[len("talker."):]] = value
            else:
                # Pass through non-talker keys
                new_state_dict[key] = value
        return new_state_dict

    @classmethod
    def from_pretrained_state_dict(cls, talker_config, state_dict, dtype=torch.bfloat16):
        """Create Talker and load weights from converted state dict.

        Args:
            talker_config: Talker config (dict or HF config object)
            state_dict: Already-converted state dict (talker keys only)
            dtype: Target dtype

        Returns:
            Initialized NeuronQwen25OmniTalker
        """
        talker = cls(talker_config, dtype=dtype)

        # Filter to only talker keys (skip non-talker prefixes)
        talker_keys = {}
        for key, value in state_dict.items():
            if any(
                key.startswith(p)
                for p in [
                    "lm_head.", "visual.", "audio_tower.",
                    "thinker.", "token2wav.", "talker.",
                ]
            ):
                continue
            talker_keys[key] = value

        missing, unexpected = talker.load_state_dict(talker_keys, strict=False)
        if missing:
            logger.warning("Talker missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("Talker unexpected keys: %s", unexpected[:10])
        logger.info("Loaded %d weights into Talker", len(talker_keys))

        return talker
