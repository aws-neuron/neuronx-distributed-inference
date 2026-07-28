# coding=utf-8
# Copyright 2026 Sarvam AI and NeuronX Porting
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
Sarvam-30B MoE model ported to NeuronX Distributed Inference.

This implementation is based on the HuggingFace Sarvam MoE model and adapted
for AWS Trainium using NeuronX Distributed Inference framework.

Key architectural features:
- MoE (Mixture of Experts) with 128 experts + 1 shared expert, top-6 routing
- Sigmoid routing with routed_scaling_factor=2.5 (NOT softmax)
- Q/K normalization (RMSNorm on Q/K projections after split, head_dim=64)
- Fused QKV projection with GQA (64 heads, 4 KV heads, head_dim=64)
- Standard RMSNorm for layer normalization
- RoPE (Rotary Position Embeddings, theta=8000000)
- SwiGLU activation in MLP experts
- first_k_dense_replace=1 (layer 0 is dense, layers 1+ are MoE)

Routing: sigmoid scores → top-6 → normalize → scale by routed_scaling_factor (2.5).
Shared expert is handled separately from MoE module to support the scaling factor.
"""

import gc
import json
import os
from typing import Optional, Tuple, Dict, Any

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module, initialize_moe_process_group
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig

# Import HuggingFace model for weight loading
from transformers import AutoModelForCausalLM


def get_rmsnorm_cls():
    """
    Return the appropriate RMSNorm implementation.
    - Use CustomRMSNorm for NeuronX (optimized for hardware)
    - Use standard RMSNorm for CPU mode
    """
    if cpu_mode():
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


class SarvamRouterTopK(RouterTopK):
    """
    RouterTopK with post-sigmoid expert_bias (e_score_correction_bias).

    Sarvam applies expert_bias AFTER sigmoid but BEFORE topk selection:
        scores = sigmoid(W*x) + expert_bias
        topk_indices = topk(scores)

    This cannot be approximated as a pre-sigmoid gate bias because:
    1. topk(sigmoid(W*x) + b) ≠ topk(W*x + Δ) for any fixed Δ
    2. The affinities (used as routing weights) must include the bias

    We use nn.Parameter for expert_bias so it gets a NEFF input slot
    and can be loaded from the checkpoint at runtime.
    """

    def __init__(self, num_experts, **kwargs):
        # No pre-sigmoid bias on the linear gate
        kwargs['bias'] = False
        super().__init__(num_experts=num_experts, **kwargs)
        # Post-sigmoid expert_bias: loaded from HF gate.expert_bias
        self.expert_bias = torch.nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states):
        # Get router logits: (T, E)
        router_logits = self.get_router_logits(hidden_states)

        # Sigmoid in float64 for numerical precision (same as parent apply_activation_fn)
        unbiased_scores = torch.sigmoid(router_logits.to(dtype=torch.float64))

        # Add post-sigmoid expert_bias for ROUTING SELECTION ONLY
        biased_scores = unbiased_scores + self.expert_bias.to(dtype=torch.float64)

        # Topk on BIASED scores (expert_bias affects which experts are selected)
        _, expert_index = torch.topk(biased_scores, self.top_k, dim=-1)

        # Return UNBIASED scores as affinities — HF gathers unbiased sigmoid
        # scores for the routing weights, NOT the biased scores. expert_bias only
        # affects expert selection, not the actual routing weight magnitudes.
        expert_affinities = unbiased_scores.to(dtype=hidden_states.dtype)
        expert_index = expert_index.detach().to(dtype=torch.long)

        return router_logits, expert_affinities, expert_index


def initialize_sarvam_moe_module(config: InferenceConfig):
    """
    Initialize MoE module for Sarvam.

    Uses SarvamRouterTopK which adds expert_bias (e_score_correction_bias) after
    sigmoid activation but before topk selection, exactly matching HF behavior.
    """
    moe_tkg_tp_group, moe_tkg_ep_group, moe_cte_tp_group, moe_cte_ep_group = \
        initialize_moe_process_group(config, enabled_hybrid_sharding=False)

    router = SarvamRouterTopK(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        dtype=config.neuron_config.router_config.dtype,
        act_fn=config.neuron_config.router_config.act_fn,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
    )
    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            top_k=config.num_experts_per_tok,
            hidden_act=config.hidden_act,
            bias=False,
            glu_mlp=config.neuron_config.glu_mlp,
            glu_type=config.neuron_config.glu_type,
            hidden_act_scaling_factor=config.neuron_config.hidden_act_scaling_factor,
            hidden_act_bias=config.neuron_config.hidden_act_bias,
            use_index_calc_kernel=config.neuron_config.use_index_calc_kernel,
            gate_clamp_upper_limit=config.neuron_config.gate_clamp_upper_limit,
            gate_clamp_lower_limit=config.neuron_config.gate_clamp_lower_limit,
            up_clamp_upper_limit=config.neuron_config.up_clamp_upper_limit,
            up_clamp_lower_limit=config.neuron_config.up_clamp_lower_limit,
            early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation,
            normalize_top_k_affinities=config.neuron_config.normalize_top_k_affinities,
            enable_spmd_rank=config.neuron_config.blockwise_matmul_config.parallelize_token_to_block_mapping,
        ),
        blockwise_matmul_config=config.neuron_config.blockwise_matmul_config,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        dtype=config.neuron_config.torch_dtype,
        is_prefill=config.neuron_config.is_prefill_stage,
        enabled_hybrid_sharding=False,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        expert_model_parallel_group=parallel_state.get_expert_model_parallel_group(),
        cte_tensor_model_parallel_group=moe_cte_tp_group,
        cte_expert_model_parallel_group=moe_cte_ep_group,
        tkg_tensor_model_parallel_group=moe_tkg_tp_group,
        tkg_expert_model_parallel_group=moe_tkg_ep_group,
    )
    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=None,  # We handle shared experts separately
        rmsnorm=None,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        return_router_logits=config.neuron_config.return_router_logits,
        sequence_dimension=1,
    )
    moe.eval()
    return moe


def _helper_concat_and_delete_qkv(state_dict: Dict[str, Any], layer_num: int, attr: str):
    """
    Helper function to rename fused QKV for NXDI.

    Sarvam already has fused QKV as query_key_value, we just rename it to Wqkv.
    """
    if f"layers.{layer_num}.self_attn.query_key_value.{attr}" in state_dict:
        state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = state_dict[
            f"layers.{layer_num}.self_attn.query_key_value.{attr}"
        ]
        del state_dict[f"layers.{layer_num}.self_attn.query_key_value.{attr}"]


def convert_state_dict_to_fused_qkv(state_dict: Dict[str, Any], cfg: InferenceConfig):
    """
    Convert Sarvam fused QKV (query_key_value) to NXDI format (Wqkv).
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


def convert_sarvam_hf_to_neuron_state_dict(neuron_state_dict, config):
    """
    Convert HuggingFace Sarvam MoE checkpoint to NeuronX state dictionary format.

    Handles:
    - HF key renaming (attention.* -> self_attn.*, word_embeddings -> embed_tokens)
    - Fused QKV renaming (query_key_value -> Wqkv)
    - Q/K normalization layers (query_layernorm -> q_layernorm, key_layernorm -> k_layernorm)
    - Output projection renaming (dense -> o_proj)
    - Router weights (gate.weight -> router.linear_router.weight)
    - MoE expert weights (concatenate gate_proj/up_proj, copy down_proj)
    - Shared expert weights
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # Step 0: Rename HF-specific key prefixes to NXDI conventions
    # Sarvam uses 'attention.*' instead of 'self_attn.*', 'word_embeddings' instead of 'embed_tokens'
    rename_map = {
        ".attention.query_key_value.": ".self_attn.query_key_value.",
        ".attention.dense.": ".self_attn.o_proj.",
        ".attention.query_layernorm.": ".self_attn.q_layernorm.",
        ".attention.key_layernorm.": ".self_attn.k_layernorm.",
    }
    for key in list(neuron_state_dict.keys()):
        new_key = key
        for old_prefix, new_prefix in rename_map.items():
            if old_prefix in new_key:
                new_key = new_key.replace(old_prefix, new_prefix)
        # word_embeddings -> embed_tokens
        if new_key.startswith("word_embeddings."):
            new_key = new_key.replace("word_embeddings.", "embed_tokens.", 1)
        if new_key != key:
            neuron_state_dict[new_key] = neuron_state_dict.pop(key)

    num_experts = config.num_experts

    for l in range(config.num_hidden_layers):
        # 1. Q/K normalization layers (already correctly named in HF checkpoint)
        # query_layernorm, key_layernorm - no changes needed

        # 2. Check if this is a MoE layer (skip first_k_dense_replace layers)
        first_k_dense = getattr(config, "first_k_dense_replace", 1)
        is_moe_layer = l >= first_k_dense

        if is_moe_layer:
            # 3. Rename router weights (gate.weight -> router.linear_router.weight)
            if f"layers.{l}.mlp.gate.weight" in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = neuron_state_dict[
                    f"layers.{l}.mlp.gate.weight"
                ]
                del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

            # 4. Map expert_bias to router.expert_bias (nn.Parameter in SarvamRouterTopK)
            # HF: scores = sigmoid(logits) + expert_bias (post-sigmoid, pre-topk)
            # Our SarvamRouterTopK applies this exactly in its forward()
            if f"layers.{l}.mlp.gate.expert_bias" in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.mlp.router.expert_bias"] = (
                    neuron_state_dict.pop(f"layers.{l}.mlp.gate.expert_bias")
                )

            # 5. Get expert weight dimensions
            ref_weight = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"]
            intermediate_size, hidden_size = ref_weight.shape
            device = ref_weight.device
            dtype = ref_weight.dtype

            # 6. Fuse expert weights: gate_proj+up_proj and down_proj in a single loop
            gate_up_proj = torch.empty(
                num_experts, hidden_size, 2 * intermediate_size, dtype=dtype, device=device,
            )
            down_proj = torch.empty(
                num_experts, intermediate_size, hidden_size, dtype=dtype, device=device,
            )

            for expert_idx in range(num_experts):
                prefix = f"layers.{l}.mlp.experts.{expert_idx}"
                gate_up_proj[expert_idx, :, :intermediate_size] = (
                    neuron_state_dict.pop(f"{prefix}.gate_proj.weight").T.clone()
                )
                gate_up_proj[expert_idx, :, intermediate_size:] = (
                    neuron_state_dict.pop(f"{prefix}.up_proj.weight").T.clone()
                )
                down_proj[expert_idx, :, :] = (
                    neuron_state_dict.pop(f"{prefix}.down_proj.weight").T.clone()
                )

            neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
            neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

            # 7. Move shared expert weights from MoE namespace to decoder layer namespace.
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                old_key = f"layers.{l}.mlp.shared_experts.{proj}.weight"
                new_key = f"layers.{l}.shared_experts.{proj}.weight"
                if old_key in neuron_state_dict:
                    neuron_state_dict[new_key] = neuron_state_dict.pop(old_key)

    gc.collect()

    # 10. Convert to fused QKV
    if config.neuron_config.fused_qkv:
        neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

    return neuron_state_dict


class SarvamInferenceConfig(InferenceConfig):
    """
    Configuration class for Sarvam-30B MoE model inference on NeuronX.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Sarvam uses num_experts (not num_local_experts)
        self.num_local_experts = self.num_experts

        # Shared expert is handled SEPARATELY from MoE module
        # to support routed_scaling_factor. Tell MoE module: no shared experts.
        self.n_shared_experts = 0
        self.num_shared_experts_actual = getattr(self, "num_shared_experts", 1)

        # ExpertMLPsV2 reads moe_intermediate from config
        self.moe_intermediate_size = getattr(self, "moe_intermediate_size", 1024)

        # Shared expert intermediate size (from HF config)
        self.shared_expert_intermediate_size = getattr(
            self, "moe_shared_expert_intermediate_size",
            self.moe_intermediate_size * self.num_shared_experts_actual,
        )

        # Sarvam router config
        # CRITICAL: Sarvam uses sigmoid scoring, NOT softmax
        self.neuron_config.router_config.act_fn = "sigmoid"
        # norm_topk_prob=True in HF config → normalize routing weights
        self.neuron_config.normalize_top_k_affinities = True
        # routed_scaling_factor: multiply normalized routing weights by this
        self.routed_scaling_factor = getattr(self, "routed_scaling_factor", 2.5)

        # Required neuron_config flags (DirectModelCompiler doesn't set these)
        self.neuron_config.fused_qkv = True
        self.neuron_config.glu_mlp = True
        self.neuron_config.router_use_fp32 = True
        # Router dtype must be explicitly set on router_config (not just router_use_fp32)
        # This matches the pattern used by working Qwen3 MoE port
        self.neuron_config.router_config.dtype = torch.float32

        # Required by _setup_func_config in model_base.py
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_cache"):
            self.use_cache = False

        # Save original intermediate_size for dense layers
        # MoE layers use moe_intermediate_size via ExpertMLPsV2
        self.dense_intermediate_size = self.intermediate_size
        # ExpertMLPsV2 reads intermediate_size from config for expert MLP size
        self.intermediate_size = self.moe_intermediate_size

    @staticmethod
    def get_required_attributes():
        """
        Return required attributes for Sarvam model configuration.
        """
        return [
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "max_position_embeddings",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        """Return the neuron config class"""
        return MoENeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load Sarvam configuration from HuggingFace model directory.
        """
        neuron_config = kwargs.pop("neuron_config", None)
        config_path = os.path.join(model_path, "config.json")

        with open(config_path, "r") as f:
            params = json.load(f)

        # Extract Sarvam config parameters
        config_dict = {
            "hidden_size": params["hidden_size"],
            "num_attention_heads": params["num_attention_heads"],
            "num_hidden_layers": params["num_hidden_layers"],
            "num_key_value_heads": params["num_key_value_heads"],
            "head_dim": params["head_dim"],
            "hidden_act": params.get("hidden_act", "silu"),
            "intermediate_size": params["intermediate_size"],
            "max_position_embeddings": params["max_position_embeddings"],
            "vocab_size": params["vocab_size"],
            "rope_theta": params["rope_theta"],
            "rms_norm_eps": params.get("rms_norm_eps", 1e-6),
            "num_experts": params["num_experts"],
            "num_shared_experts": params.get("num_shared_experts", 1),
            "num_experts_per_tok": params["num_experts_per_tok"],
            "moe_intermediate_size": params.get("moe_intermediate_size", 1024),
            "first_k_dense_replace": params.get("first_k_dense_replace", 1),
            "use_qk_norm": params.get("use_qk_norm", True),
            "routed_scaling_factor": params.get("routed_scaling_factor", 2.5),
            "moe_shared_expert_intermediate_size": params.get(
                "moe_shared_expert_intermediate_size",
                params.get("moe_intermediate_size", 1024) * params.get("num_shared_experts", 1),
            ),
            "pad_token_id": params.get("pad_token_id", 0),
            "eos_token_id": params.get("eos_token_id", 1),
        }

        # Add neuron_config if provided
        if neuron_config is not None:
            config_dict["neuron_config"] = neuron_config

        # Merge with any additional kwargs
        config_dict.update(kwargs)

        return cls(**config_dict)


class NeuronSarvamAttention(NeuronAttentionBase):
    """
    Sarvam multi-headed attention with GQA and Q/K normalization.

    Features:
    - Fused QKV projection (handled by base class)
    - GQA: 64 attention heads, 4 KV heads, head_dim=64
    - Q/K normalization: RMSNorm applied per-head on head_dim after split
    - RoPE embeddings with theta=8000000
    """

    def __init__(self, config: SarvamInferenceConfig):
        # Initialize rotary embeddings
        rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Initialize base class with use_qk_norm=False (we override the norm layers)
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,  # We override q/k layernorm below
        )

        # Override Q/K normalization layers with RMSNorm on head_dim
        self.q_layernorm = get_rmsnorm_cls()(config.head_dim, config.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(config.head_dim, config.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronSarvamAttention has to be initialized in a distributed env. "
                "Please use neuronx_distributed module to initialize a distributed env."
            )


class NeuronSarvamDenseMLP(nn.Module):
    """
    Dense SwiGLU MLP for first layer(s).
    Uses separate gate/up projections to avoid TP stride issues.
    """

    def __init__(self, config: SarvamInferenceConfig):
        super().__init__()
        from neuronx_distributed.parallel_layers.layers import RowParallelLinear

        hidden_size = config.hidden_size
        # Dense layers use the original intermediate_size (not moe_intermediate_size)
        intermediate_size = getattr(config, "dense_intermediate_size", config.intermediate_size)

        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            gather_output=False, dtype=config.neuron_config.torch_dtype,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            gather_output=False, dtype=config.neuron_config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            input_is_parallel=True, dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class NeuronSarvamSharedExpertMLP(nn.Module):
    """
    Separate shared expert MLP for Sarvam MoE layers.

    This is kept separate from the NXDI MoE module to support
    routed_scaling_factor (2.5x scaling on routed expert output
    before adding shared expert output).
    """

    def __init__(self, config: SarvamInferenceConfig):
        super().__init__()
        from neuronx_distributed.parallel_layers.layers import RowParallelLinear

        hidden_size = config.hidden_size
        intermediate_size = config.shared_expert_intermediate_size

        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            gather_output=False, dtype=config.neuron_config.torch_dtype,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            gather_output=False, dtype=config.neuron_config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            input_is_parallel=True, dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class NeuronSarvamDecoderLayer(nn.Module):
    """
    Sarvam decoder layer with attention and MoE/MLP.
    """

    def __init__(self, config: SarvamInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = NeuronSarvamAttention(config=config)

        # MoE or dense MLP
        # First layer(s) use dense MLP as per first_k_dense_replace
        first_k_dense = getattr(config, "first_k_dense_replace", 1)
        self.is_moe_layer = layer_idx >= first_k_dense

        if self.is_moe_layer:
            # MoE layer — routed experts only (n_shared_experts=0)
            # Shared expert handled separately to support routed_scaling_factor
            # SarvamRouterTopK: exact post-sigmoid expert_bias (e_score_correction_bias)
            self.mlp = initialize_sarvam_moe_module(config=config)

            # Separate shared expert MLP
            self.shared_experts = NeuronSarvamSharedExpertMLP(config)

            # Scaling factor for routed expert output
            self.routed_scaling_factor = config.routed_scaling_factor
        else:
            # Dense MLP for first few layers
            self.mlp = NeuronSarvamDenseMLP(config)

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
        Forward pass for Sarvam decoder layer.
        """
        residual = hidden_states

        # Pre-attention norm
        hidden_states = self.input_layernorm(hidden_states)

        # Attention (NeuronAttentionBase returns 4 values)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        # Pre-MLP norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP (MoE or dense)
        if self.is_moe_layer:
            # Routed experts (MoE module with n_shared_experts=0)
            # NXDI normalizes routing weights internally (normalize_top_k_affinities=True)
            moe_output = self.mlp(hidden_states)[0]

            # Apply routed_scaling_factor (Sarvam scales normalized weights by 2.5)
            moe_output = moe_output * self.routed_scaling_factor

            # Shared expert gets the same normed input as routed experts.
            # In HF: identity = hidden_states (already post-norm when passed to MoE block)
            shared_output = self.shared_experts(hidden_states)

            hidden_states = moe_output + shared_output
        else:
            hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronSarvamModel(NeuronBaseModel):
    """
    Sarvam base model (without LM head).
    """

    def setup_attr_for_model(self, config: SarvamInferenceConfig):
        """Setup model attributes required by NeuronBaseModel."""
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: SarvamInferenceConfig):
        """Initialize model components."""
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.vocab_size

        # Embeddings — must match NXDI's standard pattern (see Llama model).
        # Default: shard_across_embedding=True (each core has full vocab, partial hidden dim)
        # With vocab_parallel=True: shard across vocab + use_spmd_rank for per-core masking.
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=not config.neuron_config.vocab_parallel,
            pad=True,
            tensor_model_parallel_group=get_tp_group(config),
            use_spmd_rank=config.neuron_config.vocab_parallel,
        )

        # Embedding dropout
        self.embedding_dropout = nn.Dropout(
            getattr(config, "embedding_dropout", 0.0)
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                NeuronSarvamDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final norm
        self.norm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # LM head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
            tensor_model_parallel_group=get_tp_group(config),
        )

    def __init__(self, config: SarvamInferenceConfig):
        super().__init__(config)


class NeuronSarvamForCausalLM(NeuronBaseForCausalLM):
    """
    Sarvam model with causal LM head for NeuronX inference.
    """

    _model_cls = NeuronSarvamModel
    _config_cls = SarvamInferenceConfig

    def __init__(self, model_path: str = None, config: SarvamInferenceConfig = None, **kwargs):
        # NeuronApplicationBase expects (model_path, config) signature
        # Don't create self.model here - the base class will instantiate _model_cls at the right time
        super().__init__(model_path=model_path, config=config, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: SarvamInferenceConfig) -> dict:
        """
        Convert HuggingFace Sarvam state dict to NeuronX format.

        Called by the base class get_state_dict after loading and prefix stripping.
        Handles QKV fusion, router weight renaming, and MoE expert conversion.
        """
        return convert_sarvam_hf_to_neuron_state_dict(state_dict, config)

    @staticmethod
    def load_hf_model(model_path: str):
        """Load HuggingFace model with trust_remote_code=True."""
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
