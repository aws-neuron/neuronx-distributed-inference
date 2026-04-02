# coding=utf-8
# Copyright 2025 The Qwen Team. All rights reserved.
# Adapted for Qwen2.5-VL NxDI implementation.
#
# Text backbone is identical to Qwen2-VL: same GQA, QKV bias, M-RoPE, SwiGLU MLP.
# This file is adapted from NxDI qwen2_vl/modeling_qwen2_vl_text.py.

import gc
import logging
from typing import Optional, Tuple

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.models.config import InferenceConfig
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
from neuronx_distributed_inference.modules.attention.utils import _rotate_half
from transformers.models.llama.modeling_llama import LlamaRMSNorm

logger = logging.getLogger("Neuron")


# M-RoPE: Multimodal Rotary Position Embedding
# Identical to Qwen2-VL -- splits head_dim into [temporal, height, width] sections
# using mrope_section = [16, 24, 24] (doubled for cos/sin pairs)
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    split_indices = [sum(mrope_section[: i + 1]) for i in range(len(mrope_section) - 1)]
    cos = torch.cat(
        [
            m[i % 3]
            for i, m in enumerate(torch.tensor_split(cos, split_indices, dim=-1))
        ],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [
            m[i % 3]
            for i, m in enumerate(torch.tensor_split(sin, split_indices, dim=-1))
        ],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def get_rmsnorm_cls():
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class NeuronQwen2_5_VLRotaryEmbedding(nn.Module):
    """3D Rotary embedding for M-RoPE (temporal, height, width).
    Identical to Qwen2-VL rotary embedding."""

    def __init__(self, config: InferenceConfig, device=None):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.base = getattr(config, "rope_theta", 1000000.0)
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", None, persistent=False)
        self.inv_freq = self.get_inv_freqs(device)

    def get_inv_freqs(self, device=None):
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.base ** (freq_indices / self.dim))

    def forward(self, x, position_ids):
        # position_ids shape: (3, batch_size, seq_len) for [temporal, height, width]
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(
            3, position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[None, :, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronQwen2_5_VLAttention(NeuronAttentionBase):
    """Qwen2.5-VL text attention: GQA with QKV bias and M-RoPE.
    Identical to Qwen2-VL text attention."""

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=True,
            o_bias=False,
            rotary_emb=NeuronQwen2_5_VLRotaryEmbedding(config),
            rms_norm_eps=config.rms_norm_eps,
            attention_chunk_size=getattr(config, "attention_chunk_size", None),
            sliding_window=getattr(config, "sliding_window", None),
        )
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.mrope_section = config.rope_scaling["mrope_section"]
        self.padding_side = "right"

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_multimodal_rotary_pos_emb(
                Q, K, cos_cache, sin_cache, self.mrope_section
            )
        return Q, K, cos_cache, sin_cache


class NeuronQwen2_5_VLDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer with GQA + SwiGLU MLP."""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen2_5_VLAttention(config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class Qwen2_5_VLTextModelWrapper(ImageToTextModelWrapper):
    """Wrapper for text model that provides dummy vision inputs."""

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
    ):
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


class NeuronQwen2_5_VLTextModel(NeuronBaseModel):
    """Qwen2.5-VL text decoder model.
    Identical architecture to Qwen2-VL text model."""

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size, self.hidden_size, self.padding_idx
            )
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.layers = nn.ModuleList(
            [
                NeuronQwen2_5_VLDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


class NeuronQwen2_5_VLTextForCausalLM(NeuronBaseForCausalLM):
    """CausalLM wrapper for compilation and state dict conversion."""

    _model_cls = NeuronQwen2_5_VLTextModel

    @staticmethod
    def load_hf_model(model_path):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        """Convert HF state dict for text model.
        Qwen2.5-VL uses 'model.' prefix (same as Qwen2-VL, not 'language_model.' like Qwen3-VL)."""
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
        }
        new_state_dict = {}
        for dict_key in state_dict:
            if "model." in dict_key:
                new_key = dict_key.replace("model.", "")
                if not inference_config.neuron_config.fused_qkv:
                    for atten_key in attention_keys:
                        if atten_key in new_key:
                            new_key = new_key.replace(
                                atten_key, attention_keys[atten_key]
                            )
                new_state_dict[new_key] = state_dict[dict_key]
            else:
                new_state_dict[dict_key] = state_dict[dict_key]

        if inference_config.neuron_config.fused_qkv:
            new_state_dict = _convert_state_dict_to_fused_qkv(
                new_state_dict, inference_config
            )

        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return InferenceConfig


# --- Fused QKV helpers ---


def _helper_concat_and_delete_qkv(state_dict, layer_num, attr):
    state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ]
    )
    del state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def _convert_state_dict_to_fused_qkv(state_dict, cfg: InferenceConfig):
    mods_to_not_conv = getattr(cfg.neuron_config, "modules_to_not_convert", None) or []
    for layer in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(state_dict, layer, "weight")
        _helper_concat_and_delete_qkv(state_dict, layer, "bias")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled
            or cfg.neuron_config.quantized
        ) and f"layers.{layer}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(state_dict, layer, "scale")
    gc.collect()
    return state_dict
