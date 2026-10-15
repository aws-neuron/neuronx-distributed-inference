# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InternVL3-8B-Instruct: Text backbone (Qwen2.5-7B) for NxDI.

This module contains the text decoder model, attention, decoder layers,
text model wrapper (ImageToTextModelWrapper), and weight conversion.

Text backbone: Qwen2.5-7B
- 28 layers, hidden_size=3584, 28 Q heads, 4 KV heads (GQA 7:1)
- Standard RoPE (rope_theta=1e6), RMSNorm, SiLU gated MLP
- QKV bias=True, O bias=False
- vocab_size=151674, tie_word_embeddings=False

Weight key mapping (HF -> NxDI):
  language_model.model.layers.{i}.* -> layers.{i}.*
  language_model.model.embed_tokens.weight -> embed_tokens.weight
  language_model.model.norm.weight -> norm.weight
  language_model.lm_head.weight -> lm_head.weight
"""

import torch
from torch import nn

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
    ImageToTextModelWrapper,
)
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    scatter_by_index_put,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    """Get RMSNorm implementation: HF for CPU, CustomRMSNorm for Neuron."""
    return Qwen2RMSNorm if cpu_mode() else CustomRMSNorm


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class NeuronInternVL3Attention(NeuronAttentionBase):
    """
    InternVL3 text attention: GQA with standard RoPE and QKV bias.

    - 28 Q heads, 4 KV heads (7:1 GQA ratio)
    - head_dim = 128
    - Q/K/V have bias, O does not
    - Standard RoPE (not M-RoPE)
    - No Q-K normalization (unlike Qwen3)
    """

    def __init__(self, config):
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            qkv_bias=True,
            o_bias=False,
            rms_norm_eps=config.rms_norm_eps,
        )


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class NeuronInternVL3DecoderLayer(nn.Module):
    """
    InternVL3 text decoder layer: pre-norm RMSNorm + GQA attention + SwiGLU MLP.

    Supports NKI kernel fused RMSNorm when qkv_kernel_enabled or mlp_kernel_enabled.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronInternVL3Attention(config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # NKI kernel flags — fuse RMSNorm into kernels when enabled
        neuron_config = config.neuron_config
        self.qkv_kernel_enabled = neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = neuron_config.mlp_kernel_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        residual = hidden_states

        # When QKV kernel is enabled, pass the RMSNorm module to be fused
        # into the NKI kernel instead of applying it separately
        if self.qkv_kernel_enabled:
            qkv_fused_rmsnorm = self.input_layernorm
        else:
            hidden_states = self.input_layernorm(hidden_states)
            qkv_fused_rmsnorm = None

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        # When MLP kernel is enabled, pass the RMSNorm module to be fused
        if self.mlp_kernel_enabled:
            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
            )
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)[0]

        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Text model wrapper (ImageToTextModelWrapper)
# ---------------------------------------------------------------------------


class InternVL3TextModelWrapper(ImageToTextModelWrapper):
    """
    Text model wrapper for InternVL3 that includes vision embedding inputs
    in the compiled NEFF trace signature.

    Inherits ImageToTextModelWrapper which generates 24-argument input tuples
    with vision_embeddings (arg 22) and vision_mask (arg 23).
    """

    def __init__(
        self,
        config,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        pipeline_execution=True,
        return_ranked_to_cpu=True,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )

    @staticmethod
    def get_dummy_vision_inputs(config, input_ids, n_active_tokens, fill_value):
        """
        Create dummy vision tensors for tracing and text-only / token-gen passes.

        For context encoding (seq_len > 1):
          - vision_embeddings: [batch, seq_len, hidden_size] zeros
          - vision_mask: [batch, n_active_tokens, 1] filled with fill_value (int32 positions)
        For token generation (seq_len == 1):
          - Both are empty tensors
        """
        input_batch_size, input_sequence_len = input_ids.shape[0], input_ids.shape[-1]
        if input_sequence_len > 1:
            vision_embeddings = torch.zeros(
                input_batch_size,
                config.neuron_config.seq_len,
                config.hidden_size,
                dtype=config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                size=(input_batch_size, n_active_tokens, 1),
                fill_value=fill_value,
                dtype=torch.int32,
            )
        else:
            vision_embeddings = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
            vision_mask = torch.zeros((0), dtype=torch.bool)
        return vision_embeddings, vision_mask


# ---------------------------------------------------------------------------
# Text model (traced on Neuron)
# ---------------------------------------------------------------------------


class NeuronInternVL3TextModel(NeuronBaseModel):
    """
    InternVL3 text model (Qwen2.5-7B backbone) for NxDI.

    Components:
    - ParallelEmbedding (vocab=151674, hidden=3584)
    - 28x NeuronInternVL3DecoderLayer
    - RMSNorm
    - ColumnParallelLinear lm_head

    Implements encode_vision_to_input() for merging vision embeddings
    into text embeddings during context encoding.
    """

    def setup_attr_for_model(self, config):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        self.layers = nn.ModuleList(
            [
                NeuronInternVL3DecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

    def encode_vision_to_input(
        self,
        inputs_embeds: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scatter vision embeddings into text embeddings at image token positions.

        Called by NeuronBaseModel.get_model_output() during context encoding only.
        Runs ON-DEVICE inside the compiled NEFF.

        Args:
            inputs_embeds: [batch, seq_len, hidden_size] -- text token embeddings
            vision_embeddings: [batch, seq_len, hidden_size] -- padded vision embeddings
            vision_mask: [batch, seq_len, 1] -- int32 position indices

        Returns:
            inputs_embeds with vision positions replaced by vision embeddings
        """
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)


# ---------------------------------------------------------------------------
# Weight conversion helper (text-only, used by top-level model)
# ---------------------------------------------------------------------------


class NeuronInternVL3TextForCausalLM(NeuronBaseForCausalLM):
    """
    Helper class for text weight conversion only.
    Not used directly for inference -- the top-level NeuronBaseForImageToText
    class handles that via NeuronInternVL3TextModel.
    """

    _model_cls = NeuronInternVL3TextModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return None

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """
        Convert InternVL3 text weights from HuggingFace to Neuron format.

        HF layout:
          language_model.model.embed_tokens.weight
          language_model.model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
          language_model.model.layers.{i}.self_attn.o_proj.weight
          language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight
          language_model.model.layers.{i}.{input,post_attention}_layernorm.weight
          language_model.model.norm.weight
          language_model.lm_head.weight

        NxDI layout (fused_qkv=False):
          embed_tokens.weight
          layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
          layers.{i}.self_attn.o_proj.weight
          layers.{i}.mlp.{gate,up,down}_proj.weight
          layers.{i}.{input,post_attention}_layernorm.weight
          norm.weight
          lm_head.weight

        NxDI layout (fused_qkv=True):
          layers.{i}.self_attn.qkv_proj.Wqkv.{weight,bias}
          (q/k/v fused into single Wqkv tensor)

        When fused_qkv=False, separate q/k/v_proj are kept and the GQA
        preshard_hook fuses them during weight sharding.

        When fused_qkv=True (required for NKI QKV kernel), we pre-fuse
        q/k/v into Wqkv here because the preshard_hook expects it.
        """
        neuron_config = config.neuron_config
        neuron_state_dict = {}

        # Add rank tensors for tensor parallelism
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        for key, value in state_dict.items():
            # Only process text weights (language_model.*)
            if not key.startswith("language_model."):
                continue

            # Strip the language_model.model. prefix
            new_key = key
            if key.startswith("language_model.model."):
                new_key = key[len("language_model.model.") :]
            elif key.startswith("language_model."):
                new_key = key[len("language_model.") :]

            neuron_state_dict[new_key] = value.detach().clone()

        # When fused_qkv=True, fuse separate q/k/v weights into Wqkv
        if neuron_config.fused_qkv:
            for i in range(config.num_hidden_layers):
                prefix = f"layers.{i}.self_attn"
                # Fuse weights: [q_size, hidden] + [kv_size, hidden] + [kv_size, hidden]
                q_w = neuron_state_dict.pop(f"{prefix}.q_proj.weight")
                k_w = neuron_state_dict.pop(f"{prefix}.k_proj.weight")
                v_w = neuron_state_dict.pop(f"{prefix}.v_proj.weight")
                neuron_state_dict[f"{prefix}.qkv_proj.Wqkv.weight"] = torch.cat(
                    [q_w, k_w, v_w], dim=0
                )
                # Fuse biases (Q/K/V all have bias in InternVL3)
                q_b = neuron_state_dict.pop(f"{prefix}.q_proj.bias", None)
                k_b = neuron_state_dict.pop(f"{prefix}.k_proj.bias", None)
                v_b = neuron_state_dict.pop(f"{prefix}.v_proj.bias", None)
                if q_b is not None and k_b is not None and v_b is not None:
                    neuron_state_dict[f"{prefix}.qkv_proj.Wqkv.bias"] = torch.cat(
                        [q_b, k_b, v_b], dim=0
                    )

        # Add per-layer rank tensors for attention TP sharding
        tp_degree = neuron_config.tp_degree
        for i in range(config.num_hidden_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Add base model rank tensor
        neuron_state_dict["rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )

        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied embeddings (InternVL3: tie_word_embeddings=False, but support both)."""
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        # This helper class doesn't need its own config -- used through top-level
        from neuronx_distributed_inference.models.config import InferenceConfig

        return InferenceConfig
