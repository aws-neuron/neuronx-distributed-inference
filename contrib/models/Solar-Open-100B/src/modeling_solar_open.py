#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
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
NeuronX Distributed Inference implementation for the Solar Open 100B model
(upstage/Solar-Open-100B).

Architecture:
- SolarOpenForCausalLM: 102.6B MoE, 12B active per token
- 48 layers, ALL MoE (no dense layers)
- 128 routed + 1 shared expert, top-8 sigmoid routing with e_score_correction_bias
- GQA: 64 attention heads / 8 KV heads, head_dim=128
- hidden_size=4096 (128-aligned, no padding needed)
- YaRN RoPE scaling (factor=2.0, original_max=65536)
- BF16 native weights
- Built-in HuggingFace transformers model (4.57+), no trust_remote_code needed

Key differences from GPT-OSS (closest NxDI model):
- No hidden_size padding (4096 is already 128-aligned)
- No MXFP4 -- BF16 native
- No learned attention sinks
- No sliding window / mixed attention
- No EAGLE speculative decoding
- Simpler MoE: no clamping, no scaling/bias on hidden acts
- Shared expert uses separate gate/up/down Linear (not fused)

Target: trn2.48xlarge, tp=64 (128 experts / 64 shards = 2 per shard)
"""

import copy
import gc
import json
import logging
import math
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
    to_dict,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.gqa import (
    BaseGroupQueryAttention,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import (
    calculate_num_cores_per_group,
)
from neuronx_distributed_inference.modules.generation.sampling import create_sampler
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KVCacheManager,
)

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.utils.distributed import get_tp_group

# MoE v2 module (required for MoE layers)
try:
    from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

    MOE_V2_AVAILABLE = True
except ImportError:
    MOE_V2_AVAILABLE = False

logger = logging.getLogger("Neuron")


# ---------------------------------------------------------------------------
# Sigmoid routing patch for fused TKG kernel
# ---------------------------------------------------------------------------
# The SDK 2.28 fused MoE TKG NKI kernel's router only supports softmax.
# Solar Open uses sigmoid routing. We patch to force ISA router fallback.
# Same pattern as Trinity contrib.


def _patch_fused_tkg_for_sigmoid():
    """Patch MoEFusedTKG kernel to use ISA router fallback for sigmoid routing."""
    try:
        import neuronx_distributed.modules.moe.moe_fused_tkg as fused_tkg_mod

        original_kernel = fused_tkg_mod._moe_token_gen_selective_load_kernel_nki_call
        if original_kernel is None:
            logger.warning(
                "Fused TKG selective load kernel not available, skipping patch"
            )
            return

        class _PatchedKernelCall:
            def __init__(self, original):
                self._original = original

            def __getitem__(self, grid):
                original_grid_call = self._original[grid]

                def patched_call(*args, **kwargs):
                    kwargs["use_router_topk_nki_kernel"] = False
                    return original_grid_call(*args, **kwargs)

                return patched_call

        fused_tkg_mod._moe_token_gen_selective_load_kernel_nki_call = (
            _PatchedKernelCall(original_kernel)
        )

        original_all = fused_tkg_mod._moe_tkg_forward_all_experts_nki_call
        if original_all is not None:
            fused_tkg_mod._moe_tkg_forward_all_experts_nki_call = _PatchedKernelCall(
                original_all
            )

        logger.warning("Patched MoEFusedTKG for sigmoid routing (ISA fallback)")
    except ImportError:
        logger.info("moe_fused_tkg module not available (SDK < 2.28), skipping patch")
    except Exception as e:
        logger.warning("Failed to patch MoEFusedTKG for sigmoid: %s", e)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def convert_gate_up_proj(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert gate_up_proj from Solar Open HF format to NxDI format.

    Solar Open HF stores experts.gate_up_proj as [E, 2*I, H] with chunked
    gate and up projections (first I rows = gate, second I rows = up).
    The HF forward does: gate, up = linear(x, W).chunk(2, dim=-1)

    NxDI expects [E, H, 2*I] with the same chunked layout.
    We just need to transpose dims 1 and 2.

    Args:
        tensor: [E, 2*I, H] chunked gate/up weights

    Returns:
        [E, H, 2*I] chunked gate/up weights
    """
    return tensor.transpose(1, 2).contiguous()


def get_lm_head_pad_config(
    vocab_size: int,
    tp_degree: int,
    lm_head_pad_alignment_size: int = 1,
    skip_lm_head_pad: bool = False,
):
    """Check if lm_head padding is necessary for proper sharding."""
    if vocab_size % (tp_degree * lm_head_pad_alignment_size) == 0 or skip_lm_head_pad:
        return False, 1
    return True, lm_head_pad_alignment_size


def preshard_hook_fn(
    module: torch.nn.Module, model_state_dict: dict, prefix: str
) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)
    return False


# ---------------------------------------------------------------------------
# YaRN Rotary Embedding
# ---------------------------------------------------------------------------


class SolarOpenYaRNRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) rotary embedding for Solar Open.

    Solar Open uses YaRN with factor=2.0, original_max_position_embeddings=65536,
    extending to 128K context. This implements the NTK-by-parts interpolation
    from the YaRN paper (arXiv:2309.00071).

    The GPT-OSS model has a similar but different parametrization (ntk_alpha/ntk_beta).
    Solar Open uses the standard HuggingFace YaRN convention from rope_scaling config.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 1000000.0,
        factor: float = 2.0,
        original_max_position_embeddings: int = 65536,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.register_buffer("inv_freq", None, persistent=False)
        self.attention_scaling = None

    def _compute_inv_freq_and_scaling(self, device=None):
        """Compute YaRN inv_freq using NTK-by-parts interpolation."""
        # Base inv_freq
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
        inv_freq = 1.0 / (self.base ** (freq_indices / self.dim))

        if self.factor <= 1.0:
            self.attention_scaling = 1.0
            return inv_freq

        # YaRN concentration / attention scaling
        # From the YaRN paper: t' = 0.1*ln(s) + 1.0
        self.attention_scaling = 0.1 * math.log(self.factor) + 1.0

        # NTK-by-parts: compute low/high frequency boundaries
        # Match HF's find_correction_range(beta_fast, beta_slow, dim, base, orig_max):
        #   low = find_correction_dim(beta_fast) -> lower index (high freq boundary)
        #   high = find_correction_dim(beta_slow) -> higher index (low freq boundary)
        d_half = self.dim / 2

        low = (
            d_half
            * math.log(
                self.original_max_position_embeddings / (self.beta_fast * 2 * math.pi)
            )
            / math.log(self.base)
        )
        high = (
            d_half
            * math.log(
                self.original_max_position_embeddings / (self.beta_slow * 2 * math.pi)
            )
            / math.log(self.base)
        )

        # Truncate to integer boundaries (matches HF's default truncate=True)
        low = max(math.floor(low), 0)
        high = min(math.ceil(high), self.dim - 1)

        # Interpolation (scaled down by factor) and extrapolation (unchanged base)
        # HF: inv_freq_interpolation = 1.0 / (factor * pos_freqs) = inv_freq / factor
        # HF: inv_freq_extrapolation = 1.0 / pos_freqs = inv_freq (no change)
        inv_freq_interpolation = inv_freq / self.factor
        inv_freq_extrapolation = inv_freq

        # Ramp: linear from 0 at low boundary to 1 at high boundary
        # extrapolation_factor = 1 - ramp: 1.0 for indices <= low (keep original), 0.0 for indices >= high (interpolate)
        ramp = (torch.arange(d_half, dtype=torch.float32, device=device) - low) / (
            high - low
        )
        inv_freq_extrapolation_factor = 1 - ramp.clamp(0, 1)

        # Mix: extrapolation where factor=1, interpolation where factor=0
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )

        return inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: [bs, num_attention_heads, seq_len, head_size]
            position_ids: [bs, seq_len]

        Returns:
            cos, sin: both [bs, seq_len, head_dim]
        """
        if self.inv_freq is None:
            self.inv_freq = self._compute_inv_freq_and_scaling(x.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------


class SolarOpenInferenceConfig(InferenceConfig):
    """
    Inference config for Solar Open 100B.

    Maps Solar Open HF config fields to what NxDI expects.
    No hidden_size padding needed (4096 is 128-aligned).
    """

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return MoENeuronConfig

    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            self.num_cores_per_group = calculate_num_cores_per_group(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.neuron_config.tp_degree,
            )

        # Map Solar Open config field names to NxDI expected names
        if not hasattr(self, "num_local_experts"):
            if hasattr(self, "n_routed_experts"):
                self.num_local_experts = self.n_routed_experts
            elif hasattr(self, "num_experts"):
                self.num_local_experts = self.num_experts

        if not hasattr(self, "num_experts_per_tok"):
            if hasattr(self, "num_experts_per_tok"):
                pass  # already set
            elif hasattr(self, "experts_per_token"):
                self.num_experts_per_tok = self.experts_per_token

    def get_required_attributes(self) -> List[str]:
        return [
            "num_hidden_layers",
            "num_local_experts",
            "num_experts_per_tok",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "rope_theta",
            "pad_token_id",
        ]

    def validate_config(self):
        missing_attributes = [
            x for x in self.get_required_attributes() if not hasattr(self, x)
        ]
        assert len(missing_attributes) == 0, f"Config must define {missing_attributes}"

    def to_json_string(self):
        config_copy = copy.deepcopy(self)
        config_dict = to_dict(config_copy)
        return json.dumps(config_dict, indent=2, sort_keys=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solar Open config
        self.rms_norm_eps = getattr(self, "rms_norm_eps", 1e-05)
        # hidden_act = "silu" for the SwiGLU MLP activation (NOT the router).
        # The router activation is configured separately via router_config.act_fn.
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"

        # MoE config: Solar Open stores moe_intermediate_size separately
        # NxDI's initialize_moe_module reads config.intermediate_size for expert MLP size
        # We need to swap: store original dense intermediate, set intermediate_size to moe size
        moe_intermediate = getattr(self, "moe_intermediate_size", 1280)
        dense_intermediate = getattr(self, "intermediate_size", 10240)
        self.dense_intermediate_size = dense_intermediate
        self.intermediate_size = moe_intermediate
        self.moe_intermediate_size = moe_intermediate

        # Shared experts: disable NxDI's built-in shared expert handling
        # We handle shared experts as a separate module
        # Store the actual count before overriding to 0
        self.num_shared_experts_actual = getattr(self, "n_shared_experts", 1)
        self.n_shared_experts = 0

        # Set glu_type and router config for MoE initialization
        # Solar Open uses SiLU activation with gate/up split: silu(gate) * up
        # In NxDI's Experts._activation():
        #   GLU:    activation_fn(gate) * up  →  silu(gate) * up = gate*sigmoid(gate)*up  ✓ CORRECT
        #   SWIGLU: gate * activation_fn(gate) * up  →  gate * silu(gate) * up = gate^2*sigmoid(gate)*up  ✗ WRONG
        # Must use GLU, not SWIGLU, when hidden_act="silu".
        self.neuron_config.glu_mlp = True
        self.neuron_config.glu_type = "glu"
        self.neuron_config.router_config.act_fn = "sigmoid"
        self.neuron_config.router_config.dtype = torch.bfloat16

        # Solar Open has no clamping, no scaling, no bias on hidden activations
        self.neuron_config.hidden_act_scaling_factor = 1.0
        self.neuron_config.hidden_act_bias = 0
        self.neuron_config.gate_clamp_upper_limit = None
        self.neuron_config.gate_clamp_lower_limit = None
        self.neuron_config.up_clamp_upper_limit = None
        self.neuron_config.up_clamp_lower_limit = None
        self.neuron_config.normalize_top_k_affinities = True  # norm_topk_prob=True
        self.neuron_config.transpose_shared_experts_weights = False
        self.neuron_config.early_expert_affinity_modulation = False

        # YaRN RoPE parameters (extracted from rope_scaling dict)
        rope_scaling = getattr(self, "rope_scaling", None)
        if rope_scaling is not None:
            self.yarn_factor = rope_scaling.get("factor", 2.0)
            self.yarn_original_max = rope_scaling.get(
                "original_max_position_embeddings", 65536
            )
            self.yarn_beta_fast = rope_scaling.get("beta_fast", 32.0)
            self.yarn_beta_slow = rope_scaling.get("beta_slow", 1.0)
        else:
            self.yarn_factor = 1.0
            self.yarn_original_max = getattr(self, "max_position_embeddings", 131072)
            self.yarn_beta_fast = 32.0
            self.yarn_beta_slow = 1.0

        # Standard HF config attributes expected by NeuronBaseModel.forward()
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_cache"):
            self.use_cache = True
        if not hasattr(self, "return_dict"):
            self.return_dict = True


# ---------------------------------------------------------------------------
# Shared Expert Module
# ---------------------------------------------------------------------------


class NeuronSolarOpenSharedExpert(nn.Module):
    """
    Standalone shared expert for Solar Open.

    Solar Open has 1 shared expert with intermediate_size = moe_intermediate_size * n_shared_experts.
    For the default config: 1280 * 1 = 1280 (NOT the dense intermediate_size of 10240).
    Uses separate gate_proj, up_proj, down_proj Linear layers with SwiGLU.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        hidden_size = config.hidden_size
        # Shared expert intermediate = moe_intermediate_size * actual_n_shared_experts
        num_shared = getattr(config, "num_shared_experts_actual", 1)
        intermediate_size = config.moe_intermediate_size * num_shared

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.up_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=tp_group,
            )
            from neuronx_distributed.parallel_layers.layers import RowParallelLinear

            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class NeuronSolarOpenAttention(NeuronAttentionBase):
    """
    Solar Open attention: standard GQA with YaRN RoPE.

    No learned sinks, no sliding window, no QK normalization.
    Much simpler than GPT-OSS attention.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            rotary_emb=self._get_rope(config),
            num_cores_per_group=config.num_cores_per_group,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            use_scaled_rope=None,
            qkv_bias=False,  # Solar Open has attention_bias=False
            o_bias=False,
        )

    @staticmethod
    def _get_rope(config: SolarOpenInferenceConfig):
        return SolarOpenYaRNRotaryEmbedding(
            dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            max_position_embeddings=getattr(config, "max_position_embeddings", 131072),
            base=config.rope_theta,
            factor=config.yarn_factor,
            original_max_position_embeddings=config.yarn_original_max,
            beta_fast=config.yarn_beta_fast,
            beta_slow=config.yarn_beta_slow,
        )


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class NeuronSolarOpenMoE(nn.Module):
    """
    Solar Open MoE module wrapping NxDI's initialize_moe_module.

    Key settings:
    - Sigmoid routing with e_score_correction_bias applied POST-sigmoid
    - The HF routing: sigmoid(W@x), then add bias for top-k selection,
      but use the unbiased sigmoid values as expert weights
    - NxDI's RouterTopK adds bias PRE-sigmoid (inside the Linear), which is wrong
    - Fix: set router_bias=False, store bias separately, patch router forward

    - experts_bias=False (Solar Open experts have no bias)
    - apply_act_fn_over_topk=False
    - No clamping on gate/up projections
    """

    def __init__(self, config: InferenceConfig, rmsnorm: Optional[nn.Module] = None):
        super().__init__()

        assert MOE_V2_AVAILABLE, "MoE v2 module required for Solar Open"

        self.moe = initialize_moe_module(
            config=config,
            rmsnorm=rmsnorm,
            init_tkg_module=not config.neuron_config.on_cpu,
            router_bias=False,  # NO bias in linear — we handle it post-sigmoid
            experts_bias=False,  # Solar Open experts have no bias
            apply_act_fn_over_topk=False,
        )

        # Store e_score_correction_bias as a separate buffer.
        # It will be loaded during weight conversion and applied post-sigmoid in the router.
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_local_experts, dtype=torch.float32),
        )

        # Patch the router forward to apply bias post-sigmoid for selection
        self._patch_router()

    def _patch_router(self):
        """Patch the MoE router to match HF Solar Open routing logic.

        HF logic:
        1. router_logits = W @ x (no bias)
        2. affinities = sigmoid(router_logits)
        3. selection_scores = affinities + e_score_correction_bias
        4. top_k on selection_scores
        5. weights = affinities gathered at top_k indices (NO bias in weights)
        6. normalize weights
        """
        router = self.moe.router
        original_forward = router.forward
        moe_module = self  # Capture reference to access e_score_correction_bias

        def patched_router_forward(hidden_states):
            # Step 1: Get raw logits (no bias)
            router_logits = router.get_router_logits(hidden_states)

            # Step 2: Apply sigmoid to get affinities
            expert_affinities = torch.sigmoid(router_logits)

            # Step 3: Add e_score_correction_bias for selection
            selection_scores = (
                expert_affinities
                + moe_module.e_score_correction_bias.to(expert_affinities.dtype)
            )

            # Step 4: Top-k on selection_scores
            _, expert_index = torch.topk(selection_scores, router.top_k)

            # Cast to required dtype
            expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
            expert_index = expert_index.detach().to(dtype=torch.long)

            return router_logits, expert_affinities, expert_index

        router.forward = patched_router_forward

    def forward(self, hidden_states, is_speculative_decoding=False, residual=None):
        result = self.moe(
            hidden_states,
            is_speculative_decoding=is_speculative_decoding,
            residual=residual,
        )
        hidden_states = result[0]
        router_logits = result[1] if self.moe.return_router_logits else None
        expert_index = (
            result[-2]
            if (self.moe.return_expert_index and residual is not None)
            else (result[-1] if self.moe.return_expert_index else None)
        )
        residual_out = result[-1] if residual is not None else None

        return tuple(
            x
            for x in (hidden_states, router_logits, expert_index, residual_out)
            if x is not None
        )


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class NeuronSolarOpenDecoderLayer(nn.Module):
    """
    Solar Open decoder layer.

    All layers are MoE (no dense layers). Standard pre-norm with RMSNorm.
    No sliding window, no MXFP4 shuffling, no EAGLE.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        # Track actual shared expert count (before we override to 0 for NxDI)
        self.num_shared_experts = getattr(config, "num_shared_experts_actual", 1)

        # Attention with pre-norm
        self.self_attn = NeuronSolarOpenAttention(config=config)

        if cpu_mode():
            self.input_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.input_layernorm = CustomRMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = CustomRMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )

        # MoE feed-forward with post-attention layernorm fused
        self.feed_forward = NeuronSolarOpenMoE(
            config, rmsnorm=self.post_attention_layernorm
        )

        # Shared expert (separate from MoE module)
        if self.num_shared_experts > 0:
            self.shared_expert = NeuronSolarOpenSharedExpert(config)
        else:
            self.shared_expert = None

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        cos_cache = kwargs.pop("cos_cache", None)
        sin_cache = kwargs.pop("sin_cache", None)

        # Residual connection
        residual = hidden_states.clone()

        # Pre-norm (fused with QKV kernel when SP is disabled)
        if not self.qkv_kernel_enabled or self.sequence_parallel_enabled:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            **kwargs,
        )

        # MoE with fused residual:
        # Inside MoE: input = attn_output + residual, then route to experts.
        # Returns (routed_output, ..., fused_residual) where fused_residual = attn_output + residual.
        is_speculative_decoding = (
            self.config.neuron_config.enable_fused_speculation
            and not self.config.neuron_config.is_prefill_stage
        )
        moe_result = self.feed_forward(hidden_states, is_speculative_decoding, residual)
        moe_hidden_states = moe_result[0]
        # fused_residual = original_hidden_states + attn_output
        fused_residual = (
            moe_result[-1] if len(moe_result) > 1 else (residual + hidden_states)
        )

        # Shared expert: takes same post-norm input as routed experts.
        # In the HF reference: shared_experts(residuals) where residuals = pre-norm hidden_states.
        # In our flow: fused_residual = residual + attn_output (same as HF's residuals before norm).
        # The post_attention_layernorm is already applied inside MoE for routed experts.
        # We apply it here for the shared expert.
        if self.shared_expert is not None:
            shared_input = self.post_attention_layernorm(fused_residual)
            shared_output = self.shared_expert(shared_input)
            moe_hidden_states = moe_hidden_states + shared_output

        # Final: fused_residual + routed_output + shared_output
        hidden_states = fused_residual + moe_hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuronSolarOpenModel(NeuronBaseModel):
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
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )

            should_pad_lm_head, lm_head_pad_alignment_size = get_lm_head_pad_config(
                vocab_size=config.vocab_size,
                tp_degree=config.neuron_config.tp_degree,
                lm_head_pad_alignment_size=(
                    config.neuron_config.lm_head_pad_alignment_size
                    * config.neuron_config.logical_nc_config
                ),
                skip_lm_head_pad=not config.neuron_config.lm_head_pad,
            )

            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=should_pad_lm_head,
                pad=True,
                pad_alignment_size_per_rank=lm_head_pad_alignment_size,
                keep_padded_output=should_pad_lm_head,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # All layers are identical MoE layers (no dense/sliding window alternation)
        self.layers = nn.ModuleList(
            [
                NeuronSolarOpenDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        if cpu_mode():
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = CustomRMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )

        # Patch fused MoE TKG kernel for sigmoid routing (must happen before compile).
        # Solar Open uses sigmoid routing but the fused NKI kernel's router only
        # supports softmax. This forces the ISA router fallback.
        if getattr(config.neuron_config, "moe_fused_nki_kernel_enabled", False):
            _patch_fused_tkg_for_sigmoid()

    def init_inference_optimization(self, config: InferenceConfig):
        if self.on_device_sampling:
            lm_head_tp_degree = None
            if hasattr(self, "lm_head") and hasattr(
                self.lm_head, "tensor_parallel_group"
            ):
                lm_head_tp_degree = self.lm_head.tensor_parallel_group.size()
            self.sampler = create_sampler(config.neuron_config, lm_head_tp_degree)

        # Standard KV cache manager (no sliding window)
        self.kv_mgr = KVCacheManager(
            config, num_kv_head=self.num_key_value_heads, global_rank=self.rank_util
        )


# ---------------------------------------------------------------------------
# ForCausalLM (top-level entry point)
# ---------------------------------------------------------------------------


class NeuronSolarOpenForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronSolarOpenModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """
        Convert Solar Open HuggingFace state dict to NxDI format.

        Handles two HF weight formats:
        A. Per-expert safetensors (standard HF):
           layers.{i}.mlp.experts.{e}.gate_proj.weight [I, H]
           layers.{i}.mlp.experts.{e}.up_proj.weight   [I, H]
           layers.{i}.mlp.experts.{e}.down_proj.weight  [H, I]
        B. Fused format (from SolarOpenNaiveMoe module):
           layers.{i}.mlp.experts.gate_up_proj [E, 2*I, H]
           layers.{i}.mlp.experts.down_proj    [E, H, I]

        Both produce NxDI format:
           layers.{i}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.weight [E, H, 2*I]
           layers.{i}.feed_forward.moe.expert_mlps.mlp_op.down_proj.weight    [E, I, H]

        Note: 'model.' prefix is already stripped by NeuronBaseForCausalLM.get_state_dict().
        """
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        target_dtype = neuron_config.torch_dtype

        # Note: 'model.' prefix already stripped by NeuronBaseForCausalLM.get_state_dict()

        for layer in range(num_layers):
            prefix = f"layers.{layer}"

            # NOTE: Attention weight key renaming (q_proj -> qkv_proj.q_proj, etc.)
            # is handled automatically by NeuronAttentionBase's preshard_hook.
            # Do NOT rename attention keys here.

            # --- Router ---
            router_weight_key = f"{prefix}.mlp.gate.weight"
            if router_weight_key in state_dict:
                state_dict[f"{prefix}.feed_forward.moe.router.linear_router.weight"] = (
                    state_dict.pop(router_weight_key).to(target_dtype)
                )

            router_bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
            if router_bias_key in state_dict:
                # Store as separate buffer on NeuronSolarOpenMoE (NOT on the linear router)
                state_dict[f"{prefix}.feed_forward.e_score_correction_bias"] = (
                    state_dict.pop(router_bias_key).to(torch.float32)
                )

            # --- Expert weights ---
            # HF stores per-expert separate tensors:
            #   layers.{i}.mlp.experts.{e}.gate_proj.weight  [I, H]
            #   layers.{i}.mlp.experts.{e}.up_proj.weight    [I, H]
            #   layers.{i}.mlp.experts.{e}.down_proj.weight  [H, I]
            # NxDI expects fused stacked tensors:
            #   layers.{i}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.weight  [E, H, 2*I]
            #   layers.{i}.feed_forward.moe.expert_mlps.mlp_op.down_proj.weight     [E, I, H]

            # Check if per-expert format (HF safetensors) or fused format
            first_expert_key = f"{prefix}.mlp.experts.0.gate_proj.weight"
            fused_gate_up_key = f"{prefix}.mlp.experts.gate_up_proj"

            if first_expert_key in state_dict:
                # Per-expert format: stack into fused tensors
                num_experts = config.num_local_experts
                # Get dimensions from first expert
                gate_w = state_dict[f"{prefix}.mlp.experts.0.gate_proj.weight"]
                intermediate_size, hidden_size = gate_w.shape  # [I, H]

                gate_up_proj = torch.empty(
                    num_experts,
                    hidden_size,
                    2 * intermediate_size,
                    dtype=target_dtype,
                    device="cpu",
                )
                down_proj = torch.empty(
                    num_experts,
                    intermediate_size,
                    hidden_size,
                    dtype=target_dtype,
                    device="cpu",
                )

                for e in range(num_experts):
                    g_key = f"{prefix}.mlp.experts.{e}.gate_proj.weight"
                    u_key = f"{prefix}.mlp.experts.{e}.up_proj.weight"
                    d_key = f"{prefix}.mlp.experts.{e}.down_proj.weight"

                    gate_w = state_dict.pop(g_key).to(target_dtype)  # [I, H]
                    up_w = state_dict.pop(u_key).to(target_dtype)  # [I, H]
                    down_w = state_dict.pop(d_key).to(target_dtype)  # [H, I]

                    # gate_up: cat [I, H] + [I, H] -> [2I, H], transpose -> [H, 2I]
                    gate_up_proj[e] = torch.cat([gate_w, up_w], dim=0).T
                    # down: [H, I] -> transpose -> [I, H]
                    down_proj[e] = down_w.T

                state_dict[
                    f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.weight"
                ] = gate_up_proj
                state_dict[
                    f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.down_proj.weight"
                ] = down_proj

            elif fused_gate_up_key in state_dict:
                # Fused format: [E, 2*I, H] chunked -> [E, H, 2*I]
                state_dict[
                    f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.weight"
                ] = convert_gate_up_proj(state_dict.pop(fused_gate_up_key)).to(
                    target_dtype
                )

                down_proj_key = f"{prefix}.mlp.experts.down_proj"
                if down_proj_key in state_dict:
                    # [E, H, I] -> [E, I, H]
                    state_dict[
                        f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.down_proj.weight"
                    ] = state_dict.pop(down_proj_key).transpose(1, 2).to(target_dtype)

            # --- Shared expert ---
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                shared_key = f"{prefix}.mlp.shared_experts.{proj_name}.weight"
                if shared_key in state_dict:
                    state_dict[f"{prefix}.shared_expert.{proj_name}.weight"] = (
                        state_dict.pop(shared_key).to(target_dtype)
                    )

            # --- Fused MoE TKG: duplicate RMSNorm + transpose router weight ---
            if neuron_config.moe_fused_nki_kernel_enabled:
                post_norm_key = f"{prefix}.post_attention_layernorm.weight"
                if post_norm_key in state_dict:
                    state_dict[
                        f"{prefix}.feed_forward.moe.moe_fused_tkg.post_attention_layernorm.weight"
                    ] = state_dict[post_norm_key].clone()

                router_w_key = f"{prefix}.feed_forward.moe.router.linear_router.weight"
                if router_w_key in state_dict:
                    state_dict[f"{prefix}.feed_forward.moe.router.weight_T"] = (
                        state_dict[router_w_key].T.contiguous()
                    )

        # --- LM Head padding ---
        should_pad_lm_head, _ = get_lm_head_pad_config(
            vocab_size=config.vocab_size,
            tp_degree=neuron_config.tp_degree,
            lm_head_pad_alignment_size=(
                neuron_config.lm_head_pad_alignment_size
                * neuron_config.logical_nc_config
            ),
            skip_lm_head_pad=not neuron_config.lm_head_pad,
        )
        if should_pad_lm_head:
            state_dict["lm_head.bias"] = torch.zeros(
                state_dict["lm_head.weight"].shape[0], dtype=torch.float32
            )

        # --- Fused QKV ---
        if neuron_config.fused_qkv:
            for layer in range(num_layers):
                prefix = f"layers.{layer}"
                qkv_weight = torch.cat(
                    [
                        state_dict.pop(f"{prefix}.self_attn.q_proj.weight"),
                        state_dict.pop(f"{prefix}.self_attn.k_proj.weight"),
                        state_dict.pop(f"{prefix}.self_attn.v_proj.weight"),
                    ],
                    dim=0,
                )
                state_dict[f"{prefix}.self_attn.Wqkv.weight"] = qkv_weight

        # --- Vocab parallel rank utility ---
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # --- Rank utilities for attention and base model ---
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        gc.collect()
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @classmethod
    def get_config_cls(cls):
        return SolarOpenInferenceConfig

    @staticmethod
    def get_compiler_args() -> str:
        return "--model-type=transformer -O1 --auto-cast=matmult"
