# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni Talker for NXD inference.
#
# This file contains TWO implementations:
#
# 1. NeuronQwen25OmniTalker (CPU wrapper)
#    - Wraps HF's Qwen2_5OmniTalkerForConditionalGeneration
#    - Runs on CPU — suitable for quick testing or when Neuron resources
#      are reserved for the 7B Thinker
#
# 2. NeuronQwen25OmniTalkerForCausalLM (Neuron-compiled)
#    - Compiles the 24-layer transformer on Neuron with KV cache
#    - Uses fused embedding (8448→3584→896 collapsed to 8448→896)
#    - Supports mRoPE (3D position_ids) and explicit head_dim=128
#    - Recommended TP=4 (3 Q heads/rank, 1 KV head/rank)
#    - Uses ImageToTextModelWrapper for thinker state injection during
#      context encoding (projected thinker states passed as vision_embeddings)
#
# Talker Architecture:
#   - embed_tokens: Embedding(8448, 3584) — codec vocab in Thinker's dim space
#   - thinker_to_talker_proj: Linear(3584 -> 896)
#   - 24 Qwen2 decoder layers (GQA: 12 heads, 4 kv_heads, head_dim=128)
#   - MLP: SiLU gate_proj/up_proj(896->18944), down_proj(18944->896)
#   - RMSNorm(896)
#   - codec_head: Linear(896 -> 8448, no bias)
#
# Total Parameters: ~690M

"""Qwen2.5-Omni Talker model for NXD inference."""

import gc
import logging
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

logger = logging.getLogger(__name__)


# =============================================================================
# Part 1: CPU-based Talker (HF wrapper)
# =============================================================================


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


# =============================================================================
# Part 2: Neuron-compiled Talker
# =============================================================================

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import _rotate_half
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def _get_rmsnorm_cls():
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


def _apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Apply multimodal RoPE (reused from Qwen2-VL)."""
    mrope_section = mrope_section * 2
    split_indices = [sum(mrope_section[:i + 1]) for i in range(len(mrope_section) - 1)]
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(torch.tensor_split(cos, split_indices, dim=-1))],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(torch.tensor_split(sin, split_indices, dim=-1))],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _apply_standard_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply standard 1D RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TalkerNeuronConfig(NeuronConfig):
    """NeuronConfig subclass for Talker.

    Sets the default attention class to NeuronTalkerAttention.
    Recommended TP=4 for the Talker (12 Q heads / 4 = 3 per rank).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronTalkerAttention


class TalkerInferenceConfig(InferenceConfig):
    """InferenceConfig for the Neuron-compiled Talker.

    Talker-specific attributes:
      - head_dim = 128 (explicit, NOT hidden_size // num_attention_heads)
      - qkv_bias = True, o_bias = False (Qwen2 pattern)
      - rope_scaling with mrope_section for 3D mRoPE
      - thinker_hidden_size = 3584 (for projection during context encoding)
    """

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.qkv_bias = True
        self.o_bias = False
        # Head dim is EXPLICIT for the Talker
        # hidden_size=896, num_attention_heads=12 → 896/12=74.67 (fractional!)
        # Actual head_dim=128, so attention internal dim = 12×128 = 1536
        if not hasattr(self, "head_dim") or self.head_dim is None:
            self.head_dim = 128
        # mRoPE config (default matching Qwen2.5-Omni)
        if not hasattr(self, "rope_scaling") or self.rope_scaling is None:
            self.rope_scaling = {"type": "mrope", "mrope_section": [16, 24, 24]}
        # Store thinker hidden size for projection
        if not hasattr(self, "thinker_hidden_size"):
            self.thinker_hidden_size = 3584

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",          # 896
            "num_attention_heads",  # 12
            "num_hidden_layers",    # 24
            "num_key_value_heads",  # 4
            "vocab_size",           # 8448
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",           # silu
            "intermediate_size",    # 18944
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[TalkerNeuronConfig]:
        return TalkerNeuronConfig


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TalkerRotaryEmbedding(nn.Module):
    """Rotary position embedding for the Talker.

    Uses head_dim (128) as the RoPE dimension, NOT hidden_size // num_heads.
    Supports both standard 1D and multimodal 3D position_ids.
    """

    def __init__(self, config: TalkerInferenceConfig, device=None):
        super().__init__()
        self.dim = config.head_dim  # 128
        self.base = getattr(config, "rope_theta", 1000000.0)
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", None, persistent=False)
        self.inv_freq = self._compute_inv_freq(device)

    def _compute_inv_freq(self, device=None):
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.base ** (freq_indices / self.dim))

    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            # Expand 2D (batch, seq) → 3D (3, batch, seq) for mRoPE
            # Same approach as Qwen3-VL: replicate across temporal/height/width
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # 3D mRoPE: position_ids shape (3, batch, seq)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(-2, -1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class NeuronTalkerAttention(NeuronAttentionBase):
    """Talker self-attention with explicit head_dim=128 and mRoPE.

    Key difference from standard Qwen2 attention:
      - head_dim=128 ≠ hidden_size(896) / num_attention_heads(12)
      - Internal attention dimension = 12 × 128 = 1536 ≠ hidden_size
      - Q projection: (896, 1536), K/V projection: (896, 512)
      - Output projection: (1536, 896)
    """

    def __init__(self, config: TalkerInferenceConfig):
        rotary_emb = TalkerRotaryEmbedding(config)
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,           # 896
            num_attention_heads=config.num_attention_heads,  # 12
            num_key_value_heads=config.num_key_value_heads,  # 4
            head_dim=config.head_dim,                   # 128
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            rotary_emb=rotary_emb,
        )
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.use_mrope = (
            self.rope_scaling is not None
            and "mrope_section" in self.rope_scaling
        )
        if self.use_mrope:
            self.mrope_section = self.rope_scaling["mrope_section"]

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            if self.use_mrope:
                Q, K = _apply_multimodal_rotary_pos_emb(
                    Q, K, cos_cache, sin_cache, self.mrope_section
                )
            else:
                Q, K = _apply_standard_rotary_pos_emb(Q, K, cos_cache, sin_cache)
        return Q, K, cos_cache, sin_cache


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class NeuronTalkerDecoderLayer(nn.Module):
    """Talker decoder layer: pre-norm attention + SwiGLU MLP."""

    def __init__(self, config: TalkerInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronTalkerAttention(config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = _get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = _get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, adapter_ids=adapter_ids)[0]
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Model (NeuronBaseModel)
# ---------------------------------------------------------------------------

class NeuronTalkerModel(NeuronBaseModel):
    """Neuron-compiled Talker transformer.

    Uses fused embedding: original embed_tokens(8448, 3584) + proj(3584, 896)
    are collapsed into embed_tokens(8448, 896) during state dict conversion.

    For context encoding with thinker hidden states, the projected states
    (batch, reply_len, 896) are passed as vision_embeddings and substituted
    via encode_vision_to_input().
    """

    # Enable vision_embeddings usage during token generation (for per-step
    # thinker state injection). When True, model_base.get_model_output will
    # call encode_vision_to_input during both context encoding AND token
    # generation phases.
    apply_vision_during_token_gen = True

    def setup_attr_for_model(self, config: TalkerInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: TalkerInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Fused embedding: 8448 → 896 (original 8448→3584→896 collapsed)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,     # 8448
            config.hidden_size,    # 896
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        self.layers = nn.ModuleList(
            [NeuronTalkerDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = _get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        # codec_head: 896 → 8448
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,    # 896
            config.vocab_size,     # 8448
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
        )

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        """Inject projected thinker states into token embeddings.

        Operates in two modes:
        - Context encoding (seq > 1): REPLACE placeholder embeddings with
          projected thinker states. All positions get thinker states.
        - Token generation (seq == 1): ADD per-step thinker reply state to
          the codec token embedding. This provides text guidance at each
          autoregressive step, matching HF's per-step injection behavior.

        Args:
            inputs_embeds: (batch, seq, 896) from embed_tokens
            vision_embeddings: (batch, seq, 896) projected thinker states
            vision_mask: (batch, seq, 1) int32 mask

        Returns:
            (batch, seq, 896) with thinker states injected
        """
        if inputs_embeds.shape[1] > 1:
            # Context encoding: REPLACE embeddings with thinker states
            vision_mask_bool = vision_mask.bool()
            if vision_mask_bool.dim() == 3:
                vision_mask_bool = vision_mask_bool.squeeze(-1)
            mask_expanded = vision_mask_bool.unsqueeze(-1).expand_as(inputs_embeds)
            return torch.where(mask_expanded, vision_embeddings, inputs_embeds)
        else:
            # Token generation: ADD thinker state to codec token embedding
            # This matches HF's behavior where thinker_reply_part[step] is
            # added to embed_tokens(codec_token) at each generation step.
            return inputs_embeds + vision_embeddings


# ---------------------------------------------------------------------------
# Model Wrapper (tracing with per-step vision_embeddings)
# ---------------------------------------------------------------------------


class TalkerModelWrapper:
    """Mixin that overrides get_dummy_vision_inputs for per-step injection.

    Unlike the default ImageToTextModelWrapper which uses empty
    vision_embeddings during token generation tracing, this provides
    (batch, 1, hidden_size) tensors so the compiled NEFF includes the
    ADD operation for thinker state injection at each generation step.
    """

    @staticmethod
    def get_dummy_vision_inputs(config, input_ids, n_active_tokens, fill_value):
        input_batch_size, input_sequence_len = input_ids.shape[0], input_ids.shape[-1]
        if input_sequence_len > 1:
            # Context encoding: full-sequence vision embeddings
            vision_embeddings = torch.zeros(
                input_batch_size,
                n_active_tokens,
                config.hidden_size,
                dtype=config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                size=(input_batch_size, n_active_tokens, 1),
                fill_value=fill_value,
                dtype=torch.int32,
            )
        else:
            # Token generation: single-step vision embeddings for per-step
            # thinker state injection (ADD to codec token embedding)
            vision_embeddings = torch.zeros(
                input_batch_size,
                1,
                config.hidden_size,
                dtype=config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                size=(input_batch_size, 1, 1),
                fill_value=fill_value,
                dtype=torch.int32,
            )
        return vision_embeddings, vision_mask


# ---------------------------------------------------------------------------
# Application (NeuronBaseForCausalLM)
# ---------------------------------------------------------------------------

class NeuronQwen25OmniTalkerForCausalLM(NeuronBaseForCausalLM):
    """Neuron-compiled Talker for autoregressive codec token generation.

    Compilation:
      - Recommended TP=4 (12 Q heads / 4 = 3 per rank)
      - Uses ImageToTextModelWrapper for vision_embeddings support
      - Context encoding: thinker states injected as vision_embeddings
      - Token generation: standard autoregressive with fused embedding

    State dict conversion:
      - Fuses embed_tokens(8448, 3584) + thinker_to_talker_proj(3584, 896)
        into embed_tokens(8448, 896)
      - Maps codec_head → lm_head
      - Supports fused QKV

    Usage:
      # 1. Create config
      talker_config = TalkerInferenceConfig(neuron_config, load_config=hf_config)

      # 2. Create application
      app = NeuronQwen25OmniTalkerForCausalLM(model_path, config=talker_config)

      # 3. Compile
      app.compile(compiled_model_path)

      # 4. Load
      app.load(compiled_model_path)

      # 5. Generate (token generation on Neuron)
      logits = app.forward(input_ids, attention_mask, position_ids, seq_ids, ...)
    """

    _model_cls = NeuronTalkerModel

    def get_model_wrapper_cls(self):
        from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
            ImageToTextModelWrapper,
        )

        # Dynamically create a wrapper with per-step vision_embeddings support.
        # staticmethod() preserves the descriptor when assigning to class attr.
        class _TalkerImageToTextModelWrapper(ImageToTextModelWrapper):
            get_dummy_vision_inputs = staticmethod(
                TalkerModelWrapper.get_dummy_vision_inputs
            )

        return _TalkerImageToTextModelWrapper

    @classmethod
    def get_config_cls(cls):
        return TalkerInferenceConfig

    def set_vision_embeddings(self, vision_embeddings, vision_mask,
                              thinker_reply_embeds=None):
        """Store vision embeddings for the next generate() call.

        During context encoding, projected thinker states (896-dim) are
        injected as vision_embeddings. During token generation, per-step
        thinker reply states are ADDED to codec token embeddings.

        Vision embeddings are padded to max_context_length to match the
        compiled bucket shapes (the compiled model expects fixed-size
        vision_embeddings matching the bucket, while input_ids and
        attention_mask are padded by preprocess_inputs).

        Args:
            vision_embeddings: (batch, seq, 896) projected thinker states
                for context encoding
            vision_mask: (batch, seq, 1) int32 mask (all positions active)
            thinker_reply_embeds: (batch, n_reply, 896) optional per-step
                thinker reply states for token generation. If provided,
                reply_embeds[:, step, :] is added to the codec token
                embedding at each generation step.
        """
        # Pad vision_embeddings and vision_mask to max_context_length so they
        # match the compiled NEFF bucket shapes.
        max_ctx = self.neuron_config.max_context_length
        batch, seq, dim = vision_embeddings.shape
        if seq < max_ctx:
            pad_ve = torch.zeros(
                batch, max_ctx - seq, dim, dtype=vision_embeddings.dtype
            )
            vision_embeddings = torch.cat([vision_embeddings, pad_ve], dim=1)
            pad_vm = torch.zeros(
                batch, max_ctx - seq, 1, dtype=vision_mask.dtype
            )
            vision_mask = torch.cat([vision_mask, pad_vm], dim=1)

        self._vision_embeddings = vision_embeddings
        self._vision_mask = vision_mask
        self._thinker_reply_embeds = thinker_reply_embeds
        self._vision_dtype = vision_embeddings.dtype
        self._tkg_step = 0

    def _get_model_outputs(
        self, input_ids, attention_mask, position_ids, seq_ids,
        sampling_params, prev_hidden, adapter_ids,
        medusa_args=None, llava_args=None, **kwargs
    ):
        """Override to pass vision_embeddings to ImageToTextModelWrapper.

        Context encoding: passes full projected thinker states as
        vision_embeddings (REPLACE mode in encode_vision_to_input).

        Token generation: passes per-step thinker reply state as
        vision_embeddings (ADD mode in encode_vision_to_input).

        ImageToTextModelWrapper traces with 24 positional args:
          0-4: input_ids, attention_mask, position_ids, seq_ids, sampling_params
          5-20: empty placeholders (prev_hidden, adapter_ids, medusa/block args)
          21: rotary_position_ids
          22: vision_embeddings
          23: vision_mask
        """
        vision_embeddings = getattr(self, '_vision_embeddings', torch.empty(0))
        vision_mask = getattr(self, '_vision_mask', torch.empty(0))

        if self._is_prefill(position_ids):
            outputs = self.context_encoding_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),  # prev_hidden
                torch.empty(0),  # adapter_ids
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                torch.empty(0),  # slot_mapping
                torch.empty(0),  # active_block_table
                torch.empty(0),  # num_queries
                torch.empty(0),  # computed_context_lens
                torch.empty(0),  # tile_q_indices
                torch.empty(0),  # tile_block_tables
                torch.empty(0),  # tile_masks
                torch.empty(0),  # inputs_embeds
                torch.empty(0),  # kv_cache
                torch.empty(0),  # active_mask
                torch.empty(0),  # rotary_position_ids
                vision_embeddings,
                vision_mask,
            )
            self.kv_cache_populated = True
            # Clear context vision (no longer needed), keep reply embeds
            self._vision_embeddings = torch.empty(0)
            self._vision_mask = torch.empty(0)
            self._tkg_step = 0
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        else:
            # Get per-step thinker reply state for this generation step
            reply_embeds = getattr(self, '_thinker_reply_embeds', None)
            dtype = getattr(self, '_vision_dtype', torch.bfloat16)
            batch_size = input_ids.shape[0]
            hidden_size = self.config.hidden_size

            if reply_embeds is not None and self._tkg_step < reply_embeds.shape[1]:
                step_ve = reply_embeds[:, self._tkg_step:self._tkg_step + 1, :]
                step_vm = torch.ones(batch_size, 1, 1, dtype=torch.int32)
                self._tkg_step += 1
            elif reply_embeds is not None and reply_embeds.shape[1] > 0:
                # Repeat the last reply state (matches HF behavior where
                # thinker_reply_part stays at the last element when exhausted)
                step_ve = reply_embeds[:, -1:, :]
                step_vm = torch.ones(batch_size, 1, 1, dtype=torch.int32)
            else:
                step_ve = torch.zeros(
                    batch_size, 1, hidden_size, dtype=dtype
                )
                step_vm = torch.ones(batch_size, 1, 1, dtype=torch.int32)

            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),  # prev_hidden
                torch.empty(0),  # adapter_ids
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                torch.empty(0),  # slot_mapping
                torch.empty(0),  # active_block_table
                torch.empty(0),  # num_queries
                torch.empty(0),  # computed_context_lens
                torch.empty(0),  # tile_q_indices
                torch.empty(0),  # tile_block_tables
                torch.empty(0),  # tile_masks
                torch.empty(0),  # inputs_embeds
                torch.empty(0),  # kv_cache
                torch.empty(0),  # active_mask
                torch.empty(0),  # rotary_position_ids
                step_ve,         # vision_embeddings (per-step thinker state)
                step_vm,         # vision_mask
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: TalkerInferenceConfig
    ) -> dict:
        """Convert HF Talker state dict to Neuron format.

        Key transformations:
          1. Strip 'talker.' and 'model.' prefixes
          2. Fuse embed_tokens(8448, 3584) + thinker_to_talker_proj(3584, 896)
             → embed_tokens(8448, 896)
          3. Map codec_head → lm_head
          4. Add rank utilities for distributed inference
          5. Optionally fuse QKV projections

        The original thinker_to_talker_proj weights are stored as
        '_thinker_proj_weight' and '_thinker_proj_bias' in the returned
        dict for use during context encoding (CPU-side projection).
        """
        neuron_config = config.neuron_config
        new_state_dict = {}
        proj_weight = None
        proj_bias = None

        for key, value in state_dict.items():
            # Strip talker. prefix
            if key.startswith("talker."):
                key = key[len("talker."):]

            # Route keys
            if key.startswith("model."):
                new_key = key[len("model."):]
            elif key == "codec_head.weight":
                new_key = "lm_head.weight"
            elif key == "thinker_to_talker_proj.weight":
                proj_weight = value
                continue
            elif key == "thinker_to_talker_proj.bias":
                proj_bias = value
                continue
            else:
                # Skip keys from other components
                if any(key.startswith(p) for p in [
                    "lm_head.", "visual.", "audio_tower.",
                    "thinker.", "token2wav.",
                ]):
                    continue
                new_key = key
            new_state_dict[new_key] = value

        # Fuse embedding: embed(8448, 3584) @ proj.T(3584, 896) → (8448, 896)
        # NOTE: projection bias is NOT included in the fused embedding.
        # During token generation, the bias is already included once in the
        # projected thinker reply states (proj(reply) = W @ reply + bias).
        # Including it here would cause double-bias: fused(token) + proj(reply)
        # = (W @ E + bias) + (W @ reply + bias) = W @ (E+reply) + 2*bias,
        # whereas HF computes proj(E + reply) = W @ (E+reply) + bias.
        if "embed_tokens.weight" in new_state_dict and proj_weight is not None:
            embed_weight = new_state_dict["embed_tokens.weight"]  # (8448, 3584)
            fused_embed = embed_weight.float() @ proj_weight.float().T  # (8448, 896)
            new_state_dict["embed_tokens.weight"] = fused_embed.to(
                neuron_config.torch_dtype
            )
            logger.info(
                "Fused embed_tokens (%s) + proj (%s) → (%s) (bias excluded)",
                list(embed_weight.shape), list(proj_weight.shape),
                list(new_state_dict["embed_tokens.weight"].shape),
            )

        # Save projection weights for CPU-side context encoding
        if proj_weight is not None:
            new_state_dict["_thinker_proj_weight"] = proj_weight
        if proj_bias is not None:
            new_state_dict["_thinker_proj_bias"] = proj_bias

        # Add rank utilities
        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        tp_degree = neuron_config.tp_degree
        for i in range(config.num_hidden_layers):
            new_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Fuse QKV if enabled
        if neuron_config.fused_qkv:
            new_state_dict = _fuse_talker_qkv(new_state_dict, config)

        new_state_dict["rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )

        gc.collect()
        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        # Talker does not tie embed_tokens and lm_head weights
        pass

    def get_compiler_args(self):
        return (
            "--enable-saturate-infinity --enable-mixed-precision-accumulation "
            "--auto-cast=none --model-type transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 --vectorize-strided-dma' "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )


def _fuse_talker_qkv(state_dict: dict, config: InferenceConfig) -> dict:
    """Fuse Q/K/V weight and bias tensors into Wqkv for the Talker."""
    for layer_idx in range(config.num_hidden_layers):
        for attr in ["weight", "bias"]:
            q_key = f"layers.{layer_idx}.self_attn.q_proj.{attr}"
            k_key = f"layers.{layer_idx}.self_attn.k_proj.{attr}"
            v_key = f"layers.{layer_idx}.self_attn.v_proj.{attr}"
            if all(k in state_dict for k in [q_key, k_key, v_key]):
                state_dict[f"layers.{layer_idx}.self_attn.Wqkv.{attr}"] = torch.cat([
                    state_dict.pop(q_key),
                    state_dict.pop(k_key),
                    state_dict.pop(v_key),
                ])
    gc.collect()
    return state_dict


# ---------------------------------------------------------------------------
# Helper: CPU-side thinker state projection
# ---------------------------------------------------------------------------

class ThinkerToTalkerProjection(nn.Module):
    """CPU-side projection from Thinker hidden space to Talker hidden space.

    During context encoding, thinker hidden states (3584-d) need to be
    projected to 896-d before being injected into the Neuron model as
    vision_embeddings.

    This module is loaded from the original thinker_to_talker_proj weights
    that are extracted during state dict conversion.
    """

    def __init__(self, thinker_hidden_size=3584, talker_hidden_size=896):
        super().__init__()
        self.proj = nn.Linear(thinker_hidden_size, talker_hidden_size)

    @torch.no_grad()
    def forward(self, thinker_hidden_states):
        """Project thinker states for Neuron context encoding.

        Args:
            thinker_hidden_states: (batch, seq_len, 3584)

        Returns:
            (batch, seq_len, 896) projected states ready for vision_embeddings
        """
        return self.proj(thinker_hidden_states)

    @classmethod
    def from_state_dict(cls, state_dict, dtype=torch.bfloat16):
        """Create projection from extracted state dict.

        Args:
            state_dict: Must contain '_thinker_proj_weight' and
                optionally '_thinker_proj_bias'.

        Returns:
            ThinkerToTalkerProjection on CPU in specified dtype
        """
        proj_weight = state_dict.get("_thinker_proj_weight")
        proj_bias = state_dict.get("_thinker_proj_bias")
        if proj_weight is None:
            raise ValueError(
                "State dict missing '_thinker_proj_weight'. "
                "Run convert_hf_to_neuron_state_dict first."
            )
        in_features = proj_weight.shape[1]
        out_features = proj_weight.shape[0]
        module = cls(in_features, out_features)
        module.proj.weight.data = proj_weight.to(dtype)
        if proj_bias is not None:
            module.proj.bias.data = proj_bias.to(dtype)
        else:
            module.proj.bias = None
        module.to(dtype)
        module.eval()
        return module
