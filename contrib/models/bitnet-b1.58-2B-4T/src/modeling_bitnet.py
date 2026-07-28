# coding=utf-8
# Copyright 2025 The BitNet Team and The HuggingFace Inc. team.
"""
PyTorch BitNet b1.58 2B 4T model for NeuronX Distributed Inference.

BitNet differences from Llama:
  1. attn_sub_norm: RMSNorm before o_proj (gamma fused into o_proj weights)
  2. ffn_sub_norm: RMSNorm before down_proj (gamma fused into down_proj weights)
  3. relu2 (ReLU squared) activation
  4. Ternary quantized weights (unpacked during loading)
  5. Tied word embeddings

The sub_norm gamma is fused into the following linear layer's weights during
convert_hf_to_neuron_state_dict. At runtime, a unit RMSNorm (gamma=1) with
TP-aware all-reduce for correct RMS computation is applied before the linear.
"""

import gc

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed.utils import cpu_mode
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group


def get_rmsnorm_cls():
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


def unpack_bitnet_weights(packed_weight, weight_scale, target_dtype=torch.bfloat16):
    """Unpack BitNet ternary weights from uint8 packed format.

    The packing format matches HuggingFace's BitNet implementation:
    - Packed along dim 0: each packed row of shape [packed_rows, in_features]
      contains 4 values. The unpacking order is:
      unpacked[0:packed_rows] = bits[0:1] of each byte
      unpacked[packed_rows:2*packed_rows] = bits[2:3] of each byte
      etc.
    - Values: 0->-1, 1->0, 2->+1
    """
    packed_rows = packed_weight.shape[0]
    in_features = packed_weight.shape[1]
    out_features = packed_rows * 4

    unpacked = torch.zeros(out_features, in_features, dtype=torch.uint8, device=packed_weight.device)

    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        mask = 3 << (2 * i)
        unpacked[start:end] = (packed_weight & mask) >> (2 * i)

    return (unpacked.to(target_dtype) - 1.0) * weight_scale.to(target_dtype)


class _TPAwareUnitRMSNorm(nn.Module):
    """RMSNorm without learnable gamma that computes RMS correctly across TP ranks.

    When tensor parallelism shards the hidden dimension, each rank only has a portion.
    To compute the correct RMS (which requires the full variance), we all-reduce
    the sum of squares across ranks before taking the mean.

    For non-TP execution, this is just standard RMSNorm with gamma=1.
    """
    def __init__(self, sharded_dim, full_dim, eps=1e-6, tp_group=None):
        super().__init__()
        self.variance_epsilon = eps
        self.full_dim = full_dim
        self.sharded_dim = sharded_dim
        self.tp_group = tp_group
        self.use_tp = (full_dim != sharded_dim)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if self.use_tp and self.tp_group is not None:
            local_sq_sum = hidden_states.pow(2).sum(-1, keepdim=True)
            global_sq_sum = reduce_from_tensor_model_parallel_region(
                local_sq_sum, process_group=self.tp_group
            )
            variance = global_sq_sum / self.full_dim
        else:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)


class BitNetInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'is_encoder_decoder'):
            self.is_encoder_decoder = False

    def get_required_attributes(self):
        return [
            "hidden_size", "num_attention_heads", "num_hidden_layers",
            "num_key_value_heads", "pad_token_id", "vocab_size",
            "max_position_embeddings", "rope_theta", "rms_norm_eps", "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path, neuron_config=None, **kwargs):
        import json, os
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            cfg_dict = json.load(f)
        for key in ['quantization_config', 'auto_map', 'architectures', 'model_type',
                     'torch_dtype', '_name_or_path', 'transformers_version']:
            cfg_dict.pop(key, None)
        if cfg_dict.get('pad_token_id') is None:
            cfg_dict['pad_token_id'] = cfg_dict.get('eos_token_id', 0)
        if neuron_config is None:
            neuron_config = NeuronConfig()
        return cls(neuron_config=neuron_config, **cfg_dict, **kwargs)


class _NormedOProj(nn.Module):
    """Wrapper: applies TP-aware unit RMSNorm before o_proj."""
    def __init__(self, inner_o_proj, unit_norm):
        super().__init__()
        self.inner = inner_o_proj
        self.unit_norm = unit_norm

    def forward(self, hidden_states, adapter_ids=None):
        hidden_states = self.unit_norm(hidden_states)
        if adapter_ids is not None:
            return self.inner(hidden_states, adapter_ids=adapter_ids)
        return self.inner(hidden_states)

    def preshard_hook(self, model_state_dict, prefix):
        return True


class NeuronBitNetAttention(NeuronAttentionBase):
    def __init__(self, config):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
        )

        # Disable default preshard_hook on QKV and all o_proj variants
        def _noop_preshard(model_state_dict, prefix):
            return True
        for module_name in ['qkv_proj', 'o_proj', 'cte_o_proj', 'tkg_o_proj']:
            module = getattr(self, module_name, None)
            if module is not None and hasattr(module, 'preshard_hook'):
                module.preshard_hook = _noop_preshard

        # Compute dimensions
        tp_degree = config.neuron_config.tp_degree
        from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads
        extra_heads = get_number_of_extra_heads(config.num_attention_heads, tp_degree)
        total_heads = config.num_attention_heads + extra_heads
        sharded_dim = (total_heads * head_dim) // tp_degree
        full_dim = config.hidden_size

        # Get TP group
        tp_group = None
        if parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()

        # Create TP-aware unit RMSNorm for attn_sub_norm
        attn_norm = _TPAwareUnitRMSNorm(sharded_dim, full_dim, config.rms_norm_eps, tp_group)

        # Wrap o_proj with unit norm
        for attr in ('o_proj', 'cte_o_proj', 'tkg_o_proj'):
            proj = getattr(self, attr, None)
            if proj is not None:
                setattr(self, attr, _NormedOProj(proj, attn_norm))


class NeuronBitNetMLP(nn.Module):
    """BitNet MLP with relu2. ffn_sub_norm gamma is fused into down_proj weights."""

    def __init__(self, config):
        super().__init__()
        self.act_fn = ACT2FN[config.hidden_act]

        tp_degree = config.neuron_config.tp_degree
        sharded_intermediate = config.intermediate_size // tp_degree
        tp_group = None
        if parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()

        # TP-aware unit RMSNorm (gamma fused into down_proj weights)
        self.unit_norm = _TPAwareUnitRMSNorm(
            sharded_intermediate, config.intermediate_size, config.rms_norm_eps, tp_group
        )

        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                config.hidden_size, config.intermediate_size, bias=False,
                gather_output=False, dtype=config.neuron_config.torch_dtype,
                pad=True, tensor_model_parallel_group=get_tp_group(config),
            )
            self.up_proj = ColumnParallelLinear(
                config.hidden_size, config.intermediate_size, bias=False,
                gather_output=False, dtype=config.neuron_config.torch_dtype,
                pad=True, tensor_model_parallel_group=get_tp_group(config),
            )
            self.down_proj = RowParallelLinear(
                config.intermediate_size, config.hidden_size, bias=False,
                input_is_parallel=True, dtype=config.neuron_config.torch_dtype,
                pad=True, tensor_model_parallel_group=get_tp_group(config),
                reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            )
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate = gate_output * up_output
        intermediate = self.unit_norm(intermediate)
        return (self.down_proj(intermediate), None)


class NeuronBitNetDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = NeuronBitNetAttention(config)
        self.mlp = NeuronBitNetMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value, **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronBitNetModel(NeuronBaseModel):
    def setup_attr_for_model(self, config):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
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
            config.vocab_size, config.hidden_size, self.padding_idx,
            dtype=config.neuron_config.torch_dtype, shard_across_embedding=True, pad=True,
        )
        self.layers = nn.ModuleList(
            [NeuronBitNetDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, pad=True,
            gather_output=not self.on_device_sampling,
        )


class NeuronBitNetForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronBitNetModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        num_layers = config.num_hidden_layers
        target_dtype = neuron_config.torch_dtype
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim_val = getattr(config, "head_dim", config.hidden_size // num_heads)

        # Step 1: Unpack ternary weights
        scale_keys = [k for k in state_dict if k.endswith('.scale')]
        for key in scale_keys:
            weight_key = key.replace('.scale', '.weight')
            if weight_key in state_dict and state_dict[weight_key].dtype == torch.uint8:
                state_dict[weight_key] = unpack_bitnet_weights(
                    state_dict[weight_key], state_dict[key], target_dtype
                )
                del state_dict[key]

        # Step 2: Per-layer transforms
        need_kv_replicate = (num_kv_heads % tp_degree != 0)
        kv_repeats = num_heads // num_kv_heads if need_kv_replicate else 1

        for i in range(num_layers):
            # Fuse attn_sub_norm gamma into o_proj weights
            sub_norm_key = f"layers.{i}.self_attn.attn_sub_norm.weight"
            o_proj_key = f"layers.{i}.self_attn.o_proj.weight"
            if sub_norm_key in state_dict and o_proj_key in state_dict:
                gamma = state_dict.pop(sub_norm_key).to(target_dtype)
                state_dict[o_proj_key] = state_dict[o_proj_key].to(target_dtype) * gamma.unsqueeze(0)

            # Fuse ffn_sub_norm gamma into down_proj weights
            ffn_norm_key = f"layers.{i}.mlp.ffn_sub_norm.weight"
            down_proj_key = f"layers.{i}.mlp.down_proj.weight"
            if ffn_norm_key in state_dict and down_proj_key in state_dict:
                gamma = state_dict.pop(ffn_norm_key).to(target_dtype)
                state_dict[down_proj_key] = state_dict[down_proj_key].to(target_dtype) * gamma.unsqueeze(0)

            # Rename Q/K/V keys and replicate KV for CONVERT_TO_MHA
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                old_key = f"layers.{i}.self_attn.{proj}.weight"
                new_key = f"layers.{i}.self_attn.qkv_proj.{proj}.weight"
                if old_key in state_dict:
                    weight = state_dict.pop(old_key)
                    if need_kv_replicate and proj in ['k_proj', 'v_proj']:
                        weight = weight.reshape(num_kv_heads, head_dim_val, -1)
                        weight = weight.repeat_interleave(kv_repeats, dim=0)
                        weight = weight.reshape(num_heads * head_dim_val, -1)
                    state_dict[new_key] = weight

            # o_proj: Rename for _NormedOProj wrapper
            old_key = f"layers.{i}.self_attn.o_proj.weight"
            new_key = f"layers.{i}.self_attn.o_proj.inner.o_proj.weight"
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32)

        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        gc.collect()
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return BitNetInferenceConfig

    def get_compiler_args(self):
        return (
            "--enable-saturate-infinity --enable-mixed-precision-accumulation "
            "--auto-cast=none --model-type transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
