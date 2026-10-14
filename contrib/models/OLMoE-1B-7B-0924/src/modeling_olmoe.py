# coding=utf-8
# Copyright 2025 OLMoE NeuronX Porting
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
OLMoE-1B-7B-0924 model ported to NeuronX Distributed Inference.

This implementation is based on the HuggingFace OLMoE model and adapted
for AWS Trainium using NeuronX Distributed Inference framework.

Key architectural features:
- MoE (Mixture of Experts) with 64 experts, top-8 routing
- Q/K normalization (similar to Qwen3 and OLMo)
- Standard RMSNorm for layer normalization
- RoPE (Rotary Position Embeddings)
- SwiGLU activation in MLP experts
- GQA (Grouped Query Attention) with 16 heads (num_heads == num_kv_heads)
"""

import gc
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch import nn

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Try except for compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode
from torch_neuronx.xla_impl.ops import nki_jit

from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding, move_heads_front
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


def get_rmsnorm_cls():
    """
    Return the appropriate RMSNorm implementation.
    - Use CustomRMSNorm for NeuronX (optimized for hardware)
    - Use HuggingFace RMSNorm for CPU mode
    """
    if cpu_mode():
        # Create a simple RMSNorm for CPU mode
        class CPURMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return (self.weight * hidden_states).to(input_dtype)
        return CPURMSNorm
    else:
        return CustomRMSNorm


def get_modules_to_not_convert(neuron_config: MoENeuronConfig):
    """Get the modules_to_not_convert from the neuron configs"""
    return getattr(neuron_config, "modules_to_not_convert", None)


def _helper_concat_and_delete_qkv(state_dict: Dict[str, Any], layer_num: int, attr: str):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).

    Args:
        state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(state_dict: Dict[str, Any], cfg: InferenceConfig):
    """
    This function concatenates the qkv weights and scales to a Wqkv weight and scale
    for fusedqkv, and deletes the qkv weights.
    """
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(state_dict, l, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(state_dict, l, "scale")

    gc.collect()

    return state_dict


def convert_olmoe_hf_to_neuron_state_dict(neuron_state_dict, config):
    """
    Convert HuggingFace OLMoE checkpoint to NeuronX state dictionary format.

    This handles:
    - Rank utilities for tensor parallelism
    - Q/K normalization layer naming (q_norm -> q_layernorm, k_norm -> k_layernorm)
    - Router weights (gate -> linear_router)
    - MoE expert weights (concatenate gate_proj/up_proj, copy down_proj)
    - Optional fused QKV conversion

    Based on convert_flexolmo_hf_to_neuron_state_dict and convert_qwen3_moe_hf_to_neuron_state_dict.
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    num_experts = config.num_experts

    for l in range(config.num_hidden_layers):
        # 1. Rename Q/K normalization layers (q_norm -> q_layernorm, k_norm -> k_layernorm)
        # OLMoE applies RMSNorm on full [hidden_size] Q/K BEFORE head reshape.
        # Keep full [hidden_size] weight — our prep_qkv_tensors override handles the pre-reshape norm.
        if f"layers.{l}.self_attn.q_norm.weight" in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = neuron_state_dict.pop(
                f"layers.{l}.self_attn.q_norm.weight"
            )

        if f"layers.{l}.self_attn.k_norm.weight" in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = neuron_state_dict.pop(
                f"layers.{l}.self_attn.k_norm.weight"
            )

        # 2. Rename router weights (gate -> linear_router)
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = neuron_state_dict[
            f"layers.{l}.mlp.gate.weight"
        ]
        del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

        # 3. Get expert weight dimensions and device info
        intermediate_size, hidden_size = neuron_state_dict[
            f"layers.{l}.mlp.experts.0.gate_proj.weight"
        ].shape
        device = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].device
        dtype = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].dtype

        # 4. Fuse ALL expert gate_proj + up_proj into single tensor
        # Do NOT pre-shard by TP — NXDI handles TP sharding during weight sharding
        gate_up_proj = torch.empty(
            num_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )

        for expert_idx in range(num_experts):
            gate_key = f"layers.{l}.mlp.experts.{expert_idx}.gate_proj.weight"
            up_key = f"layers.{l}.mlp.experts.{expert_idx}.up_proj.weight"
            gate_up_proj[expert_idx, :, :intermediate_size] = neuron_state_dict[gate_key].T
            gate_up_proj[expert_idx, :, intermediate_size:] = neuron_state_dict[up_key].T
            del neuron_state_dict[gate_key]
            del neuron_state_dict[up_key]

        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # 5. Fuse ALL expert down_proj into single tensor
        down_proj = torch.empty(
            num_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )

        for expert_idx in range(num_experts):
            down_key = f"layers.{l}.mlp.experts.{expert_idx}.down_proj.weight"
            down_proj[expert_idx, :, :] = neuron_state_dict[down_key].T
            del neuron_state_dict[down_key]

        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    # 7. Optional: Convert to fused QKV
    if config.neuron_config.fused_qkv:
        neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

    return neuron_state_dict


class OlmoeInferenceConfig(InferenceConfig):
    """
    Configuration class for OLMoE model inference on NeuronX.

    OLMoE is a MoE model with 64 experts and top-8 routing.
    It uses Q/K normalization similar to Qwen3 and OLMo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # OLMoE uses num_experts (not num_local_experts)
        # Add num_local_experts as expected by initialize_moe_module
        self.num_local_experts = self.num_experts

        # OLMoE has no shared experts
        self.n_shared_experts = 0

        # ExpertMLPsV2 reads moe_intermediate from config.intermediate_size
        # For OLMoE, all experts use the same intermediate_size
        self.moe_intermediate_size = self.intermediate_size

        # Required neuron_config flags (DirectModelCompiler doesn't set these)
        self.neuron_config.fused_qkv = True
        self.neuron_config.glu_mlp = True
        self.neuron_config.router_use_fp32 = True

        # Router config: use FP32 softmax for routing accuracy
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"

        # CRITICAL: OLMoE has norm_topk_prob=false — do NOT normalize top-k routing weights.
        # NXDI defaults to True, which rescales expert contributions by ~8x with top-8 routing.
        if not getattr(self, 'norm_topk_prob', False):
            self.neuron_config.normalize_top_k_affinities = False

        # Required by _setup_func_config in model_base.py
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_cache"):
            self.use_cache = False

    @staticmethod
    def get_required_attributes():
        """
        List of required configuration attributes for OLMoE.
        """
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "hidden_act",
            "intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "norm_topk_prob",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load OLMoE configuration from HuggingFace model directory.

        Args:
            model_path: Path to HuggingFace model directory
            **kwargs: Additional config parameters (including neuron_config)

        Returns:
            OlmoeInferenceConfig instance
        """
        import json
        import os

        neuron_config = kwargs.pop("neuron_config", None)
        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            params = json.load(f)

        # Extract OLMoE config parameters
        config_dict = {
            "hidden_size": params["hidden_size"],
            "num_attention_heads": params["num_attention_heads"],
            "num_hidden_layers": params["num_hidden_layers"],
            "num_key_value_heads": params["num_key_value_heads"],
            "hidden_act": params["hidden_act"],
            "intermediate_size": params["intermediate_size"],
            "max_position_embeddings": params["max_position_embeddings"],
            "vocab_size": params["vocab_size"],
            "rope_theta": params["rope_theta"],
            "rms_norm_eps": params.get("rms_norm_eps", 1e-5),
            "num_experts": params["num_experts"],
            "num_experts_per_tok": params["num_experts_per_tok"],
            "norm_topk_prob": params["norm_topk_prob"],
            "attention_bias": params.get("attention_bias", False),
            "clip_qkv": params.get("clip_qkv", None),
            "pad_token_id": params.get("pad_token_id", 1),
            "eos_token_id": params.get("eos_token_id", 50279),
        }

        # Add neuron_config if provided
        if neuron_config is not None:
            config_dict["neuron_config"] = neuron_config

        # Merge with any additional kwargs
        config_dict.update(kwargs)

        return cls(**config_dict)


class NeuronOlmoeAttention(NeuronAttentionBase):
    """
    OLMoE attention with Q/K normalization.

    This is similar to Qwen3/OLMo attention but adapted for OLMoE.
    Uses RMSNorm on query and key projections before RoPE.
    """

    def __init__(self, config: OlmoeInferenceConfig):
        # Calculate head_dim
        head_dim = config.hidden_size // config.num_attention_heads

        # Create RoPE embeddings
        rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Initialize base class with required parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,  # We'll override the norm layers
        )

        # OLMoE applies RMSNorm on full Q/K [B,S,hidden_size] BEFORE head reshape,
        # unlike NXDI default which applies per-head AFTER reshape.
        # Use hidden_size for Q norm and num_kv_heads*head_dim for K norm.
        self.q_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.k_layernorm = get_rmsnorm_cls()(
            config.num_key_value_heads * head_dim,
            eps=config.rms_norm_eps,
        )

        if not parallel_state.model_parallel_is_initialized():
            raise RuntimeError(
                "NeuronOlmoeAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )

    def prep_qkv_tensors(self, position_ids, hidden_states, past_key_value, **kwargs):
        """Override to apply Q/K norm BEFORE head reshape (OLMoE convention)."""
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states,
            rmsnorm=kwargs.get("rmsnorm"),
            adapter_ids=kwargs.get("adapter_ids"),
            residual=kwargs.get("residual"),
        )

        # Apply RMSNorm on full Q [B,S,H*D] and K [B,S,Hkv*D] BEFORE reshape
        Q = self.q_layernorm(Q)
        K = self.k_layernorm(K)

        bsz, q_len, _ = hidden_states.size()
        # Reshape without layernorm (already applied above)
        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=None)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Apply RoPE
        Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(
            Q, K, V, position_ids,
            kwargs.get("cos_cache"),
            kwargs.get("sin_cache"),
            kwargs.get("use_polar_compatible_rope", False),
        )

        return Q, K, V, cos_cache, sin_cache, residual


class NeuronOlmoeDecoderLayer(nn.Module):
    """
    OLMoE decoder layer with MoE and Q/K normalized attention.
    """

    def __init__(self, config: OlmoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronOlmoeAttention(config=config)

        # MoE (Mixture of Experts) module
        self.mlp = initialize_moe_module(config=config)

        # Layer normalization (pre-layer norm architecture)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for OLMoE decoder layer.

        Args:
            hidden_states: input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: attention mask
            position_ids: position indices
            past_key_value: cached key/value states for autoregressive generation

        Returns:
            Tuple of (hidden_states, present_key_value)
        """
        residual = hidden_states

        # Self-attention with pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]  # MoE returns (hidden_states, router_logits)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronOlmoeModel(NeuronBaseModel):
    """
    OLMoE transformer model for NeuronX.
    """

    def setup_attr_for_model(self, config: OlmoeInferenceConfig):
        """Setup model attributes from config."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: OlmoeInferenceConfig):
        """Initialize model layers."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [NeuronOlmoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronOlmoeForCausalLM(NeuronBaseForCausalLM):
    """
    OLMoE model for causal language modeling on NeuronX.
    """

    _model_cls = NeuronOlmoeModel

    def __init__(self, model_path: str = None, config: OlmoeInferenceConfig = None, **kwargs):
        # NeuronApplicationBase expects (model_path, config) signature
        # Don't create self.model here - the base class will instantiate _model_cls at the right time
        super().__init__(model_path=model_path, config=config, **kwargs)

    def get_compiler_args(self):
        """
        Get compiler arguments for OLMoE model.
        Returns appropriate optimization level based on compile stage and EP configuration.
        """
        # compile_tag might not be set during initialization, use getattr with default
        compile_tag = getattr(self, 'compile_tag', CONTEXT_ENCODING_MODEL_TAG)

        if compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif compile_tag == TOKEN_GENERATION_MODEL_TAG:
            # Disable Modular flow for TKG graph with EP enabled as it causes perf degradation
            optimization_level = "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"
        else:
            optimization_level = "-O1"  # Default fallback

        # Base compiler arguments
        compiler_args = f"--model-type=transformer --enable-saturate-infinity --verbose=35 {optimization_level}"

        return compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: OlmoeInferenceConfig) -> dict:
        """
        Convert HuggingFace OLMoE state dict to NeuronX format.

        Called by the base class get_state_dict after loading and prefix stripping.
        Handles Q/K norm renaming, router weight renaming, and MoE expert conversion.
        """
        return convert_olmoe_hf_to_neuron_state_dict(state_dict, config)
