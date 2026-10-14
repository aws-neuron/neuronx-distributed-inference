# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gemma 3 1B IT support for NeuronX Distributed Inference.

The official ``models/gemma3/`` implementation targets the 4B/12B/27B variants
which all have head_dim=128.  The 1B variant has several unusual architecture
parameters that require additional handling:

  * head_dim=256  – exceeds the NKI kernel limit of 128, and triggers a Neuron
    compiler DGE out-of-bounds issue for CTE buckets < 512.
  * vocab_size=262144 – the 4B+ variants use 262208; the upstream config class
    hardcodes the larger value.
  * GQA with num_kv_heads=1 – interacts with k_cache_transposed + SWA to
    produce a layout mismatch in repeat_kv.

This module subclasses the official Gemma 3 NxDI classes and adds only the
minimal overrides required for the 1B variant:

  1. **Chunked attention** – Q@K^T and scores@V matmuls are split into
     128-wide chunks along head_dim to stay within hardware addressing limits.
  2. **vocab_size from HF config** – reads the actual value instead of
     hardcoding 262208.
  3. **Auto-disable NKI kernel** – when head_dim > 128.
  4. **k_cache_transposed fix** – restores the config value for SWA layers
     and transposes K around repeat_kv so GQA expansion works correctly.
  5. **query_pre_attn_scalar weight fusion** – fuses the Gemma 3 attention
     scaling correction into Q/K weight matrices at load time (following the
     pattern from Pierre Lienhart's gemma3-vision contrib) so NxDI's default
     sqrt(head_dim) scaling produces the correct result with zero runtime cost.

Required configuration knobs (via vLLM --additional-config):

    context_encoding_buckets: [512]   # MUST be >= 512 (compiler issue)
    attn_kernel_enabled: false        # NKI kernel asserts head_dim <= 128
    k_cache_transposed: true          # required for the repeat_kv fix

See the README for full usage instructions.
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import copy

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import (
    Gemma3InferenceConfig as _UpstreamGemma3InferenceConfig,
    Gemma3NeuronConfig as _UpstreamGemma3NeuronConfig,
    NeuronGemma3Attention as _UpstreamNeuronGemma3Attention,
    NeuronGemma3DecoderLayer as _UpstreamNeuronGemma3DecoderLayer,
    NeuronGemma3ForCausalLM as _UpstreamNeuronGemma3ForCausalLM,
    NeuronGemma3TextModel as _UpstreamNeuronGemma3TextModel,
    NeuronGemma3RMSNorm,
    get_rmsnorm_cls,
    get_updated_configs,
)
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.utils import repeat_kv

logger = logging.getLogger(__name__)

# Maximum head dimension that the Neuron compiler can handle without DGE
# out-of-bounds errors in the standard matmul paths.
_MAX_UNCHUNKED_HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Chunked attention helpers
# ---------------------------------------------------------------------------


def _chunked_qk(
    Q: Tensor,
    K: Tensor,
    scale: float,
    chunk_size: int = _MAX_UNCHUNKED_HEAD_DIM,
) -> Tensor:
    """Q @ K^T / scale, chunked along head_dim to avoid DGE OOB.

    Args:
        Q: (B, H, S_q, D)
        K: (B, H, S_k, D) – NOT transposed
        scale: divisor (typically sqrt(head_dim))
        chunk_size: max inner-dim width per matmul
    """
    head_dim = Q.shape[-1]
    if head_dim <= chunk_size:
        return torch.matmul(Q, K.transpose(2, 3)) / scale

    QK = torch.matmul(Q[..., :chunk_size], K[..., :chunk_size].transpose(2, 3))
    for start in range(chunk_size, head_dim, chunk_size):
        end = min(start + chunk_size, head_dim)
        QK = QK + torch.matmul(Q[..., start:end], K[..., start:end].transpose(2, 3))
    return QK / scale


def _chunked_qk_transposed(
    Q: Tensor,
    K_t: Tensor,
    scale: float,
    chunk_size: int = _MAX_UNCHUNKED_HEAD_DIM,
) -> Tensor:
    """Q @ K_t / scale where K_t is already (B, H, D, S_k)."""
    head_dim = Q.shape[-1]
    if head_dim <= chunk_size:
        return torch.matmul(Q, K_t) / scale

    QK = torch.matmul(Q[..., :chunk_size], K_t[..., :chunk_size, :])
    for start in range(chunk_size, head_dim, chunk_size):
        end = min(start + chunk_size, head_dim)
        QK = QK + torch.matmul(Q[..., start:end], K_t[..., start:end, :])
    return QK / scale


def _chunked_v_matmul(
    scores: Tensor,
    V: Tensor,
    chunk_size: int = _MAX_UNCHUNKED_HEAD_DIM,
) -> Tensor:
    """scores @ V, chunked along V's head_dim."""
    head_dim = V.shape[-1]
    if head_dim <= chunk_size:
        return torch.matmul(scores, V)

    chunks = []
    for start in range(0, head_dim, chunk_size):
        end = min(start + chunk_size, head_dim)
        chunks.append(torch.matmul(scores, V[..., start:end]))
    return torch.cat(chunks, dim=-1)


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------


class Gemma3_1B_NeuronConfig(_UpstreamGemma3NeuronConfig):
    """NeuronConfig that points to our 1B-specific attention class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGemma3_1B_Attention


class Gemma3_1B_InferenceConfig(_UpstreamGemma3InferenceConfig):
    """InferenceConfig fixes for the 1B variant.

    Changes vs upstream:
      - Reads vocab_size from HF config (262144 for 1B) instead of hardcoding
        262208.
      - Auto-disables NKI attention kernel when head_dim > 128.
    """

    def __init__(self, neuron_config, fused_spec_config=None, load_config=None):
        # Let the parent set everything up (including load_config which
        # populates vocab_size from HF).
        #
        # The parent unconditionally sets vocab_size=262208 *after* load_config.
        # We need to capture the HF value before that happens.
        self._hf_vocab_size = None

        # Intercept load_config to capture vocab_size before parent overwrites it.
        if load_config is not None:
            original_load_config = load_config

            def _capturing_load_config(self_inner):
                original_load_config(self_inner)
                # Capture the HF vocab_size before parent overwrites it.
                self._hf_vocab_size = getattr(self_inner, "vocab_size", None)

            load_config = _capturing_load_config

        super().__init__(neuron_config, fused_spec_config, load_config)

        # Restore the correct vocab_size.
        if self._hf_vocab_size is not None:
            self.vocab_size = self._hf_vocab_size

        # Auto-disable NKI kernel when head_dim > 128.
        head_dim = getattr(
            self, "head_dim", self.hidden_size // self.num_attention_heads
        )
        if (
            head_dim > _MAX_UNCHUNKED_HEAD_DIM
            and self.neuron_config.attn_kernel_enabled is not False
        ):
            logger.warning(
                "Gemma3-1B: head_dim=%d > %d, auto-disabling NKI attention kernel",
                head_dim,
                _MAX_UNCHUNKED_HEAD_DIM,
            )
            self.neuron_config.attn_kernel_enabled = False

    @classmethod
    def get_neuron_config_cls(cls):
        return Gemma3_1B_NeuronConfig


# ---------------------------------------------------------------------------
# Attention override
# ---------------------------------------------------------------------------


class NeuronGemma3_1B_Attention(_UpstreamNeuronGemma3Attention):
    """Attention for Gemma 3 1B (head_dim=256).

    Adds:
      - Chunked Q@K^T and scores@V for head_dim > 128.
      - Restores k_cache_transposed for SWA layers (base class forces False).
      - Transposes K around repeat_kv so GQA expansion works for BHDS layout.
    """

    def __init__(self, config):
        super().__init__(config)
        self._needs_chunked_attn = self.head_dim > _MAX_UNCHUNKED_HEAD_DIM

        # The base class forces k_cache_transposed=False for SWA layers
        # (attention_base.py line 316), but the KV cache manager uses the
        # NeuronConfig value globally.  Restore the config value so that
        # SWA layers interpret the cache layout correctly.
        self.k_cache_transposed = config.neuron_config.k_cache_transposed

    # -- CTE overrides (prefill) ----------------------------------------

    def scaled_qk(self, Q, K, attention_mask):
        """Override: chunk Q@K^T for large head_dim."""
        if self._needs_chunked_attn:
            QK = _chunked_qk(Q, K, scale=math.sqrt(self.head_dim))
        else:
            QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            QK = torch.where(
                attention_mask.to(torch.bool), QK, torch.finfo(QK.dtype).min
            )
        return QK

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """Override: use chunked V matmul for the flat-compiler CTE path."""
        from neuronx_distributed_inference.modules.attention.attention_base import (
            FlashAttentionStrategy,
        )

        flash_attn_strategy = self.get_flash_attention_strategy(
            q_len, attention_mask is not None
        )
        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            return super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)

        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = self.scaled_qk(Q, K_active, attention_mask)

        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            learned_sinks = learned_sinks.reshape(1, self.num_heads, 1, 1).expand(
                bsz, -1, q_len, -1
            )
            active_scores = torch.cat((active_scores, learned_sinks), dim=-1)

        active_scores = nn.functional.softmax(
            active_scores, dim=-1, dtype=torch.float32
        ).to(Q.dtype)

        if learned_sinks is not None:
            active_scores = active_scores[..., :-1]

        attn_output = (
            _chunked_v_matmul(active_scores, V_active)
            if self._needs_chunked_attn
            else torch.matmul(active_scores, V_active)
        )
        return attn_output, flash_attn_strategy

    def perform_prefill_windowed_attn(
        self, Q, K, V, q_len, bsz, attention_mask, window_size
    ) -> Tensor:
        """Override: use chunked matmuls for windowed (SWA) CTE path."""
        from neuronx_distributed_inference.modules.attention.attention_base import (
            FlashAttentionStrategy,
        )

        flash_attn_strategy = self.get_flash_attention_strategy(
            q_len, attention_mask is not None
        )
        if flash_attn_strategy not in (
            FlashAttentionStrategy.NONE,
            FlashAttentionStrategy.SLIDING_WINDOW_KERNEL,
        ):
            attn_output, _ = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
            return attn_output, flash_attn_strategy

        if flash_attn_strategy == FlashAttentionStrategy.SLIDING_WINDOW_KERNEL:
            return super().perform_prefill_windowed_attn(
                Q, K, V, q_len, bsz, attention_mask, window_size
            )

        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = self.scaled_qk(Q, K_active, attention_mask)

        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            learned_sinks = learned_sinks.reshape(1, self.num_heads, 1, 1).expand(
                bsz, -1, q_len, -1
            )
            active_scores = torch.cat((active_scores, learned_sinks), dim=-1)

        active_scores = nn.functional.softmax(
            active_scores, dim=-1, dtype=torch.float32
        ).to(Q.dtype)

        if learned_sinks is not None:
            active_scores = active_scores[..., :-1]

        attn_output = (
            _chunked_v_matmul(active_scores, V_active)
            if self._needs_chunked_attn
            else torch.matmul(active_scores, V_active)
        )
        return attn_output, flash_attn_strategy

    # -- TKG override (token generation) ---------------------------------

    def compute_for_token_gen(
        self,
        Q,
        K,
        V,
        position_ids,
        past_key_value,
        attention_mask,
        active_mask,
        is_prefix_caching=False,
    ) -> Tensor:
        """Override: chunked matmuls + k_cache_transposed repeat_kv fix."""
        if not self._needs_chunked_attn:
            return super().compute_for_token_gen(
                Q,
                K,
                V,
                position_ids,
                past_key_value,
                attention_mask,
                active_mask,
                is_prefix_caching,
            )

        from neuronx_distributed_inference.modules.attention.attention_base import (
            manual_softmax,
        )

        is_speculation = False if position_ids is None else position_ids.shape[-1] > 1
        if self.attention_chunk_size and is_speculation:
            raise NotImplementedError(
                "Speculative decoding not supported with chunked attention."
            )

        K_prior = past_key_value[0]
        V_prior = past_key_value[1]

        # Handle k_cache_transposed: K_prior is BHDS, repeat_kv expects BHSD.
        if self.k_cache_transposed:
            K_prior = K_prior.transpose(2, 3)  # BHDS -> BHSD
            K_prior = repeat_kv(K_prior, self.num_key_value_groups)
            K_prior = K_prior.transpose(2, 3)  # BHSD -> BHDS
            V_prior = repeat_kv(V_prior, self.num_key_value_groups)
            prior_scores = _chunked_qk_transposed(
                Q, K_prior, scale=math.sqrt(self.head_dim)
            )
        else:
            K_prior = repeat_kv(K_prior, self.num_key_value_groups)
            V_prior = repeat_kv(V_prior, self.num_key_value_groups)
            prior_scores = _chunked_qk(Q, K_prior, scale=math.sqrt(self.head_dim))

        # Pad attention mask if KV cache is padded.
        if (
            prior_scores.shape[-1] > attention_mask.shape[-1]
            and self.neuron_config.apply_seq_ids_mask
        ):
            attention_mask = F.pad(
                attention_mask,
                (0, prior_scores.shape[-1] - attention_mask.shape[-1]),
                "constant",
                0,
            )

        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)

        # Active (current) KV.
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = _chunked_qk(Q, K_active, scale=math.sqrt(self.head_dim))
        if is_speculation or is_prefix_caching:
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
        active_scores = active_scores.to(torch.float32)

        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            bsz, _, seqlen, _ = active_scores.shape
            sinks = learned_sinks.reshape(1, self.num_heads, 1, 1).expand(
                bsz, -1, seqlen, -1
            )
            prior_scores = torch.cat((prior_scores, sinks), dim=-1)

        softmax_prior, softmax_active = manual_softmax(
            prior_scores, active_scores, is_speculation or is_prefix_caching
        )

        if learned_sinks is not None:
            softmax_prior = softmax_prior[..., :-1]

        softmax_prior, softmax_active = (
            softmax_prior.to(Q.dtype),
            softmax_active.to(Q.dtype),
        )
        attn_prior = _chunked_v_matmul(softmax_prior, V_prior)
        attn_active = _chunked_v_matmul(softmax_active, V_active)

        return attn_prior + attn_active


# ---------------------------------------------------------------------------
# Decoder layer + text model overrides
# ---------------------------------------------------------------------------


class NeuronGemma3_1B_DecoderLayer(_UpstreamNeuronGemma3DecoderLayer):
    """Decoder layer that uses our 1B-specific attention class.

    The upstream decoder hardcodes ``NeuronGemma3Attention(config)`` instead
    of using ``config.neuron_config.attn_cls``.  We override ``__init__``
    to swap in ``NeuronGemma3_1B_Attention``.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace the attention module with our 1B-specific version.
        self.self_attn = NeuronGemma3_1B_Attention(config)


class NeuronGemma3_1B_TextModel(_UpstreamNeuronGemma3TextModel):
    """Text model that uses our 1B decoder layers."""

    def init_model(self, config):
        super().init_model(config)
        # Replace layers with our 1B-specific decoder layers.
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [
                NeuronGemma3_1B_DecoderLayer(conf, idx)
                for idx, conf in enumerate(updated_configs)
            ]
        )


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------


class NeuronGemma3_1B_ForCausalLM(_UpstreamNeuronGemma3ForCausalLM):
    """Gemma 3 1B causal LM with query_pre_attn_scalar weight fusion.

    Overrides convert_hf_to_neuron_state_dict to fuse the attention scaling
    correction (query_pre_attn_scalar vs head_dim) into Q/K weight matrices.
    This avoids any runtime change while producing mathematically identical
    attention scores.

    Pattern credit: Pierre Lienhart (gemma3-vision contrib).
    """

    _model_cls = NeuronGemma3_1B_TextModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        # Run the upstream conversion first (renames q_norm -> q_layernorm, etc).
        state_dict = _UpstreamNeuronGemma3ForCausalLM.convert_hf_to_neuron_state_dict(
            state_dict, config
        )

        # Fuse query_pre_attn_scalar into Q and K weights.
        #
        # NxDI's attention base uses  QK^T / sqrt(head_dim).
        # Gemma 3 specifies         QK^T / sqrt(query_pre_attn_scalar).
        #
        # By scaling Q and K weights by gamma, we get:
        #   (Q*gamma)(K*gamma)^T / sqrt(head_dim)
        #   = Q K^T * gamma^2 / sqrt(head_dim)
        #   = Q K^T / sqrt(query_pre_attn_scalar)
        #
        # gamma = sqrt( (1/sqrt(head_dim)) * sqrt(query_pre_attn_scalar) )
        #       = (query_pre_attn_scalar / head_dim) ** 0.25
        query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", None)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        if query_pre_attn_scalar is not None and query_pre_attn_scalar != head_dim:
            default_qk_scaling_factor_inv = math.sqrt(float(query_pre_attn_scalar))
            gemma_qk_scaling_factor = 1.0 / math.sqrt(float(head_dim))
            gamma = math.sqrt(gemma_qk_scaling_factor * default_qk_scaling_factor_inv)

            logger.info(
                "Fusing query_pre_attn_scalar=%s into Q/K weights (gamma=%.6f)",
                query_pre_attn_scalar,
                gamma,
            )

            for key in list(state_dict.keys()):
                if key.endswith(
                    (
                        ".q_proj.weight",
                        ".k_proj.weight",
                        ".qkv_proj.q_proj.weight",
                        ".qkv_proj.k_proj.weight",
                    )
                ):
                    orig_dtype = state_dict[key].dtype
                    state_dict[key] = (state_dict[key].to(torch.float32) * gamma).to(
                        orig_dtype
                    )

        return state_dict

    @classmethod
    def get_config_cls(cls):
        return Gemma3_1B_InferenceConfig
