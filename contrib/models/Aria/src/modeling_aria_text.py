"""
NeuronX Distributed Inference implementation of Aria Text Model.

The Aria text model is a Mixture-of-Experts (MoE) architecture based on LLaMA:
- 64 routed experts with top-6 routing
- 2 shared experts that process all tokens
- LLaMA-style attention with RoPE (20 Q heads, 20 KV heads)
- 28 decoder layers

HuggingFace model: rhymes-ai/Aria
"""
import gc
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module


# ==================== Configuration ====================


class AriaTextInferenceConfig(InferenceConfig):
    """
    Configuration for Aria Text model inference on NeuronX.

    Aria's HF config stores text model params under a 'text_config' key.
    This config class maps those to the standard attributes expected by
    initialize_moe_module (num_local_experts, num_experts_per_tok, n_shared_experts, etc).
    """

    # Map Aria-specific HF config names to NXDI-standard names
    attribute_map: Dict[str, str] = {
        "moe_num_experts": "num_local_experts",
        "moe_topk": "num_experts_per_tok",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Aria uses 2 shared experts
        self.n_shared_experts = getattr(self, "moe_num_shared_experts", 2)
        # Ensure num_local_experts is set (may come from attribute_map)
        if not hasattr(self, "num_local_experts"):
            self.num_local_experts = 64
        if not hasattr(self, "num_experts_per_tok"):
            self.num_experts_per_tok = 6
        # Aria uses SiLU/SwiGLU
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"
        # HF-compatible attributes needed by NeuronBaseForCausalLM
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_return_dict"):
            self.use_return_dict = True
        # Router dtype should be FP32 for accuracy
        self.neuron_config.router_config.dtype = torch.float32
        # Normalize top-k affinities
        self.neuron_config.normalize_top_k_affinities = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "num_local_experts",
            "num_experts_per_tok",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig

    def get_text_config(self):
        """Override to return self since Aria text model IS the text config."""
        return self

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from a pretrained Aria model directory.

        Handles Aria's nested config format where text model params are under 'text_config'.
        """
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            full_config = json.load(f)

        # Aria stores text model config under 'text_config'
        text_config = full_config.get("text_config", full_config)

        config_dict = {
            "hidden_size": text_config.get("hidden_size", 2560),
            "intermediate_size": text_config.get("intermediate_size", 1664),
            "num_hidden_layers": text_config.get("num_hidden_layers", 28),
            "num_attention_heads": text_config.get("num_attention_heads", 20),
            "num_key_value_heads": text_config.get("num_key_value_heads", 20),
            "vocab_size": text_config.get("vocab_size", 100352),
            "max_position_embeddings": text_config.get("max_position_embeddings", 65536),
            "rope_theta": float(text_config.get("rope_theta", 5000000)),
            "rms_norm_eps": text_config.get("rms_norm_eps", 1e-6),
            "hidden_act": text_config.get("hidden_act", "silu"),
            # MoE parameters - map to standard names
            "num_local_experts": text_config.get("moe_num_experts", 64),
            "num_experts_per_tok": text_config.get("moe_topk", text_config.get("num_experts_per_tok", 6)),
            "moe_num_shared_experts": text_config.get("moe_num_shared_experts", 2),
        }

        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


# ==================== Model Components ====================


class NeuronAriaTextAttention(NeuronAttentionBase):
    """Standard LLaMA-style attention with RoPE for Aria."""

    def __init__(self, config: AriaTextInferenceConfig):
        head_dim = config.hidden_size // config.num_attention_heads
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
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
        )


class NeuronAriaTextDecoderLayer(nn.Module):
    """
    Aria decoder layer: attention + MoE (with shared experts).

    Uses initialize_moe_module from moe_v2 which handles both routed
    and shared experts via the standard NXDI MoE infrastructure.
    """

    def __init__(self, config: AriaTextInferenceConfig, layer_idx: int):
        super().__init__()
        self.self_attn = NeuronAriaTextAttention(config=config)
        self.mlp = initialize_moe_module(config=config)
        self.input_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple:
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

        # MoE with shared experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ==================== Main Model Classes ====================


class NeuronAriaTextModel(NeuronBaseModel):
    """Aria Text base model (embeddings + decoder layers + final norm)."""

    def setup_attr_for_model(self, config: AriaTextInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: AriaTextInferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList([
            NeuronAriaTextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
        )


class NeuronAriaTextForCausalLM(NeuronBaseForCausalLM):
    """Aria Text model with language modeling head for NeuronX inference."""

    _model_cls = NeuronAriaTextModel

    @classmethod
    def get_config_cls(cls):
        return AriaTextInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: AriaTextInferenceConfig) -> dict:
        """
        Convert HuggingFace Aria weights to NeuronX format.

        Key transformations:
        1. Strip 'language_model.model.' prefix (and 'language_model.' for lm_head)
        2. Skip vision_tower and multi_modal_projector weights (text-only)
        3. Map router weights: mlp.router.weight -> mlp.router.linear_router.weight
        4. Fuse expert fc1 (already gate+up fused) into gate_up_proj format
        5. Map shared expert weights to standard MoE shared expert naming
        """
        neuron_state_dict = {}

        # Add rank utility tensors for MoE
        neuron_state_dict["rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        for key, value in state_dict.items():
            # Skip non-text-model weights
            if not key.startswith("language_model."):
                continue

            # Strip the language_model prefix
            if key.startswith("language_model.model."):
                new_key = key[len("language_model.model."):]
            elif key.startswith("language_model."):
                new_key = key[len("language_model."):]
            else:
                continue

            neuron_state_dict[new_key] = value

        # Now transform per-layer keys to match NXDI MoE module structure
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"layers.{layer_idx}"

            # Add rank utility for attention
            neuron_state_dict[f"{prefix}.self_attn.rank_util.rank"] = torch.arange(
                0, config.neuron_config.tp_degree, dtype=torch.int32
            )

            # Router: mlp.router.weight -> mlp.router.linear_router.weight
            router_key = f"{prefix}.mlp.router.weight"
            if router_key in neuron_state_dict:
                neuron_state_dict[f"{prefix}.mlp.router.linear_router.weight"] = (
                    neuron_state_dict.pop(router_key)
                )

            # Expert weights: fc1 and fc2 -> gate_up_proj and down_proj
            # fc1: [num_experts, hidden_size, intermediate_size*2] (already gate+up fused)
            # fc2: [num_experts, intermediate_size, hidden_size]
            fc1_key = f"{prefix}.mlp.experts.fc1.weight"
            fc2_key = f"{prefix}.mlp.experts.fc2.weight"

            if fc1_key in neuron_state_dict:
                neuron_state_dict[f"{prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = (
                    neuron_state_dict.pop(fc1_key)
                )

            if fc2_key in neuron_state_dict:
                neuron_state_dict[f"{prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"] = (
                    neuron_state_dict.pop(fc2_key)
                )

            # Shared experts: rename to match SharedExperts module naming
            # gate_proj -> shared_experts.gate_proj
            # up_proj -> shared_experts.up_proj
            # down_proj -> shared_experts.down_proj
            # These should already match if the SharedExperts module uses the same naming.
            # SharedExperts in NxD uses: gate_proj, up_proj, down_proj (same as HF)
            # So mlp.shared_experts.{gate,up,down}_proj.weight should map correctly.

            gc.collect()

        return neuron_state_dict

    def get_compiler_args(self):
        compiler_args = (
            "--enable-saturate-infinity --enable-mixed-precision-accumulation "
            "--model-type transformer -O1"
        )
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2'"
        )
        compiler_args += " --auto-cast=none"
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args
