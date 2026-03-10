# coding=utf-8
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
"""Granite 4.0-H-Small (GraniteMoeHybridForCausalLM) model for NxD Inference.

IBM Granite 4.0-H-Small is a hybrid Mamba2/Attention architecture with MoE:
- 40 layers: 36 Mamba2 + 4 Attention (at indices 5, 15, 25, 35)
- 72 experts with top-10 routing + shared expert per layer
- hidden_size=4096, no position embeddings ("nope")
- tie_word_embeddings=True, embedding_multiplier=12, logits_scaling=16

Key implementation details:
- Mamba state persistence via nn.ParameterList + input_output_aliases
  (same mechanism as KV cache, following MLlama vision_key_values pattern)
- Manual depthwise conv1d to avoid TEN404 NKI kernel bug on seq_len=1
- Full-sequence parallel scan for prefill, O(1) recurrence for decode
- GraniteRMSNormGated: gate applied BEFORE norm (norm_before_gate=False)
- ScaledEmbedding/ScaledLMHead wrappers for Granite's multiplier/scaling
"""

import gc
import logging
import warnings
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    ModelWrapper,
    DecoderModelInstance,
)
from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs
from neuronx_distributed.utils import cpu_mode

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


class GraniteInferenceConfig(InferenceConfig):
    """Configuration class for Granite 4.0-H-Small model inference."""

    output_attentions = False
    output_hidden_states = False
    use_return_dict = True

    def get_required_attributes(self) -> List[str]:
        return [
            "attention_bias",
            "attention_dropout",
            "attention_multiplier",
            "embedding_multiplier",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "layer_types",
            "logits_scaling",
            "mamba_chunk_size",
            "mamba_conv_bias",
            "mamba_d_conv",
            "mamba_d_head",
            "mamba_d_state",
            "mamba_expand",
            "mamba_n_groups",
            "mamba_n_heads",
            "mamba_proj_bias",
            "max_position_embeddings",
            "normalization_function",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_local_experts",
            "position_embedding_type",
            "residual_multiplier",
            "rms_norm_eps",
            "shared_intermediate_size",
            "tie_word_embeddings",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


# ==============================================================================
# Model Wrapper (Mamba state aliasing)
# ==============================================================================


class GraniteDecoderModelInstance(DecoderModelInstance):
    """
    Extends DecoderModelInstance to alias Mamba state parameters.

    After calling super().get() which aliases the KV cache parameters,
    we add Mamba state parameters (conv_state, ssm_state for each Mamba layer)
    to the input_output_aliases dict. This tells the XLA compiler to persist
    these tensors across graph executions via in-place updates.

    The output indices must match the order in which forward() returns tensors:
        [res, K0, V0, K1, V1, ..., conv_state_0, ssm_state_0, conv_state_1, ssm_state_1, ...]
    """

    def get(self, bucket_rank, **kwargs):
        self.module, self.input_output_aliases = super().get(bucket_rank, **kwargs)

        past_key_values = self.module.kv_mgr.past_key_values
        mamba_states = self.module.mamba_states

        # Count where Mamba state outputs start in the output list
        num_output_from_trace = 1  # logits/tokens
        if getattr(self.module, "neuron_config", None) and getattr(
            self.module.neuron_config, "output_logits", False
        ):
            num_output_from_trace = 2
        num_output_from_trace += len(past_key_values)

        for i in range(len(mamba_states)):
            self.input_output_aliases[mamba_states[i]] = num_output_from_trace + i

        logger.info(
            f"GraniteDecoderModelInstance: aliased {len(past_key_values)} KV cache entries "
            f"and {len(mamba_states)} Mamba state entries "
            f"(Mamba starts at output index {num_output_from_trace})"
        )

        return self.module, self.input_output_aliases


class GraniteModelWrapper(ModelWrapper):
    """Custom ModelWrapper that returns GraniteDecoderModelInstance."""

    def get_model_instance(self):
        return GraniteDecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )


# ==============================================================================
# Mamba2 Layer
# ==============================================================================


class GraniteRMSNormGated(nn.Module):
    """
    Gated RMSNorm matching HF GraniteMoeHybrid exactly.
    Gate is applied BEFORE norm (norm_before_gate=False in Mamba2 terminology).
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


class NeuronMamba2Layer(nn.Module):
    """
    Mamba2 layer with external state passing for XLA graph persistence.

    Architecture:
    - in_proj: hidden_size -> projection_size (gather_output=True)
    - Split into: gate (z), xBC_input, dt
    - Manual depthwise conv1d on xBC (avoids TEN404 NKI bug)
    - SiLU activation
    - Split conv output into: x, B, C
    - SSM computation (parallel scan for prefill, recurrence for decode)
    - Gated norm: norm(y) * silu(gate)
    - out_proj: intermediate_size -> hidden_size (gather_output=True)

    State shapes (Granite 4.0-H-Small defaults):
    - conv_state: [batch, 8448, 3]
    - ssm_state: [batch, 128, 64, 128]
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.mamba_n_heads
        self.head_dim = config.mamba_d_head
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.n_groups = config.mamba_n_groups
        self.chunk_size = config.mamba_chunk_size
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias

        mamba_expand = config.mamba_expand
        self.intermediate_size = mamba_expand * self.hidden_size
        self.groups_time_state_size = self.n_groups * self.ssm_state_size
        self.conv_dim = self.intermediate_size + 2 * self.groups_time_state_size
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads

        # Input/output projections with gather_output=True (avoids manual TP)
        if parallel_state.model_parallel_is_initialized():
            self.in_proj = ColumnParallelLinear(
                self.hidden_size,
                projection_size,
                bias=self.use_bias,
                gather_output=True,
            )
            self.out_proj = ColumnParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                gather_output=True,
            )
        else:
            self.in_proj = nn.Linear(
                self.hidden_size, projection_size, bias=self.use_bias
            )
            self.out_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=False
            )

        # Manual depthwise Conv1d — the NKI auto-inserted kernel crashes with
        # TEN404 on seq_len=1. We store weights as plain parameters.
        self.conv_weight = nn.Parameter(
            torch.randn(self.conv_dim, self.conv_kernel_size)
        )
        if self.use_conv_bias:
            self.conv_bias = nn.Parameter(torch.zeros(self.conv_dim))
        else:
            self.conv_bias = None

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.num_heads))

        # Gated RMSNorm (gate before norm, matching HF)
        self.norm = GraniteRMSNormGated(
            self.intermediate_size,
            eps=config.rms_norm_eps,
        )

        self.time_step_limit = (0.0, float("inf"))

    @staticmethod
    def get_state_shapes(config, batch_size=1):
        """Return the shapes of conv_state and ssm_state for buffer allocation."""
        mamba_expand = config.mamba_expand
        intermediate_size = mamba_expand * config.hidden_size
        groups_time_state_size = config.mamba_n_groups * config.mamba_d_state
        conv_dim = intermediate_size + 2 * groups_time_state_size
        conv_shape = (batch_size, conv_dim, config.mamba_d_conv - 1)
        ssm_shape = (
            batch_size,
            config.mamba_n_heads,
            config.mamba_d_head,
            config.mamba_d_state,
        )
        return conv_shape, ssm_shape

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        mamba_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Forward pass with external state.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            mamba_state: (conv_state, ssm_state) from persistence buffers

        Returns:
            output: (batch, seq_len, hidden_size)
            present_key_value: dummy (K, V) tuple for KV cache compatibility
            updated_mamba_state: (conv_state, ssm_state) for persistence
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        if mamba_state is not None:
            conv_state, ssm_state = mamba_state
        else:
            conv_state = torch.zeros(
                batch_size,
                self.conv_dim,
                self.conv_kernel_size - 1,
                device=hidden_states.device,
                dtype=dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )

        # Extract 2D padding mask (attention_mask is 4D causal, useless for Mamba)
        padding_mask = kwargs.get("padding_mask", None)
        if padding_mask is None and position_ids is not None and seq_len > 1:
            indices = torch.arange(seq_len, device=position_ids.device).unsqueeze(0)
            padding_mask = ((position_ids > 0) | (indices == 0)).float()

        # Zero padding positions before in_proj
        if padding_mask is not None and seq_len > 1:
            hidden_states = (hidden_states * padding_mask[:, :, None]).to(dtype)

        # Project input
        projected_states = self.in_proj(hidden_states)

        # Explicit slicing (NOT split()) — Neuron XLA compatibility
        gate = projected_states[..., : self.intermediate_size]
        hidden_states_B_C = projected_states[
            ..., self.intermediate_size : self.intermediate_size + self.conv_dim
        ]
        dt = projected_states[..., -self.num_heads :]

        if seq_len > 1:
            # Prefill path
            output, conv_state_new, ssm_state_new = self._forward_prefill(
                hidden_states_B_C,
                gate,
                dt,
                batch_size,
                seq_len,
                dtype,
                padding_mask=padding_mask,
            )
            # Keep input state params in XLA graph (prevents pruning during tracing)
            conv_state_new = conv_state_new + conv_state * 0
            ssm_state_new = ssm_state_new + ssm_state * 0
        else:
            # Decode path (seq_len == 1)
            output, conv_state_new, ssm_state_new = self._forward_decode(
                hidden_states_B_C,
                gate,
                dt,
                batch_size,
                dtype,
                conv_state,
                ssm_state,
            )

        # Dummy KV cache for compatibility with attention-based generation loop
        dummy_k = torch.zeros(1, 1, 1, 1, dtype=output.dtype, device=output.device)
        dummy_v = torch.zeros(1, 1, 1, 1, dtype=output.dtype, device=output.device)

        return (output, (dummy_k, dummy_v), (conv_state_new, ssm_state_new))

    def _forward_prefill(
        self,
        hidden_states_B_C,
        gate,
        dt,
        batch_size,
        seq_len,
        dtype,
        padding_mask=None,
    ):
        """Prefill path: process full sequence with parallel scan."""
        # Manual depthwise conv1d with causal padding
        padded = F.pad(
            hidden_states_B_C, (0, 0, self.conv_kernel_size - 1, 0), value=0.0
        )

        hidden_states_conv = torch.zeros_like(hidden_states_B_C)
        for k in range(self.conv_kernel_size):
            hidden_states_conv = hidden_states_conv + (
                padded[:, k : k + seq_len, :]
                * self.conv_weight[:, k].unsqueeze(0).unsqueeze(0)
            )

        hidden_states_B_C_conv = hidden_states_conv
        if self.conv_bias is not None:
            hidden_states_B_C_conv = hidden_states_B_C_conv + self.conv_bias.unsqueeze(
                0
            ).unsqueeze(0)

        # Save conv_state from last K-1 real token positions
        if padding_mask is not None and seq_len >= self.conv_kernel_size - 1:
            real_len = padding_mask[:, :seq_len].sum(dim=1, keepdim=True).long()
            K_minus_1 = self.conv_kernel_size - 1
            offsets = torch.arange(
                K_minus_1, device=hidden_states_B_C.device
            ).unsqueeze(0)
            gather_idx = (real_len - K_minus_1 + offsets).clamp(min=0)
            gather_idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, self.conv_dim)
            conv_state_seq = torch.gather(hidden_states_B_C, 1, gather_idx_expanded)
            conv_state_new = conv_state_seq.transpose(1, 2).contiguous()
        elif seq_len >= self.conv_kernel_size - 1:
            conv_state_new = (
                hidden_states_B_C[:, -(self.conv_kernel_size - 1) :, :]
                .transpose(1, 2)
                .contiguous()
            )
        else:
            pad_len = self.conv_kernel_size - 1 - seq_len
            conv_state_new = F.pad(
                hidden_states_B_C.transpose(1, 2), (pad_len, 0), value=0.0
            ).contiguous()

        hidden_states_B_C_conv = F.silu(hidden_states_B_C_conv)

        # Zero out padding positions after conv1d+silu
        if padding_mask is not None:
            hidden_states_B_C_conv = hidden_states_B_C_conv * padding_mask[
                :, :seq_len, None
            ].to(hidden_states_B_C_conv.dtype)

        # Split conv output
        hidden_states_ssm = hidden_states_B_C_conv[..., : self.intermediate_size]
        B = hidden_states_B_C_conv[
            ...,
            self.intermediate_size : self.intermediate_size
            + self.groups_time_state_size,
        ]
        C = hidden_states_B_C_conv[..., -self.groups_time_state_size :]

        # Vectorized SSM (parallel scan)
        A = -torch.exp(self.A_log.float())

        dt_processed = F.softplus(dt + self.dt_bias)
        dt_processed = torch.clamp(dt_processed, self.time_step_limit[0], 1e6)

        if padding_mask is not None:
            dt_processed = dt_processed * padding_mask[:, :seq_len, None].to(
                dt_processed.dtype
            )

        hidden_states_ssm = hidden_states_ssm.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).float()
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2)

        dA_log = dt_processed * A.view(1, 1, -1)
        dB = dt_processed.unsqueeze(-1) * B
        dBx = dB.unsqueeze(3) * hidden_states_ssm.unsqueeze(-1)

        log_dA_cumsum = torch.cumsum(dA_log, dim=1)

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=hidden_states_ssm.device,
                dtype=hidden_states_ssm.dtype,
            )
        )

        log_diff = log_dA_cumsum.unsqueeze(2) - log_dA_cumsum.unsqueeze(1)
        log_diff = log_diff.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(-1) == 0, -1e9
        )
        weights = torch.exp(log_diff)

        states = torch.einsum("btih,bihds->bthds", weights, dBx)

        # Save final SSM state from last real token position
        if padding_mask is not None:
            real_len = padding_mask[:, :seq_len].sum(dim=1, keepdim=True).long()
            last_real_idx = (real_len - 1).clamp(min=0)
            gather_idx = last_real_idx.view(batch_size, 1, 1, 1, 1).expand(
                -1, -1, self.num_heads, self.head_dim, self.ssm_state_size
            )
            ssm_state_new = torch.gather(states, 1, gather_idx).squeeze(1).contiguous()
        else:
            ssm_state_new = states[:, -1, :, :, :].contiguous()

        y = torch.einsum("blhs,blhds->blhd", C, states)
        y = y + self.D.view(1, 1, -1, 1) * hidden_states_ssm
        y = y.reshape(batch_size, seq_len, -1)

        scan_output = self.norm(y, gate)
        output = self.out_proj(scan_output.to(dtype))
        return output, conv_state_new, ssm_state_new.to(dtype)

    def _forward_decode(
        self, hidden_states_B_C, gate, dt, batch_size, dtype, conv_state, ssm_state
    ):
        """Decode path: single token, O(1) SSM update."""
        xBC_new = hidden_states_B_C.squeeze(1)

        # Conv1d with state
        xBC_new_t = xBC_new.unsqueeze(2)
        conv_input = torch.cat([conv_state, xBC_new_t], dim=2)

        conv_out = (conv_input * self.conv_weight.unsqueeze(0)).sum(dim=2)
        if self.conv_bias is not None:
            conv_out = conv_out + self.conv_bias

        conv_state_new = conv_input[:, :, 1:].contiguous()
        conv_out = F.silu(conv_out)

        x = conv_out[..., : self.intermediate_size]
        B = conv_out[
            ...,
            self.intermediate_size : self.intermediate_size
            + self.groups_time_state_size,
        ]
        C = conv_out[..., -self.groups_time_state_size :]

        # SSM recurrence
        A = -torch.exp(self.A_log.float())
        dt_processed = F.softplus(dt.squeeze(1) + self.dt_bias)
        dt_processed = torch.clamp(dt_processed, self.time_step_limit[0], 1e6)

        x = x.reshape(batch_size, self.num_heads, self.head_dim).float()
        B = B.reshape(batch_size, self.n_groups, self.ssm_state_size).float()
        C = C.reshape(batch_size, self.n_groups, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=1)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=1)

        dA = torch.exp(dt_processed * A.view(1, -1))
        dB = dt_processed.unsqueeze(-1) * B
        dBx = dB.unsqueeze(2) * x.unsqueeze(-1)

        ssm_state_new = dA.unsqueeze(-1).unsqueeze(-1) * ssm_state.float() + dBx

        y = torch.einsum("bhds,bhs->bhd", ssm_state_new, C)
        y = y + self.D.view(1, -1, 1) * x
        y = y.reshape(batch_size, -1)

        gate_squeezed = gate.squeeze(1)
        scan_output = self.norm(y, gate_squeezed)

        if len(scan_output.shape) == 2:
            scan_output = scan_output.unsqueeze(1)
        output = self.out_proj(scan_output.to(dtype))
        return output, conv_state_new, ssm_state_new.to(dtype)


# ==============================================================================
# Utility functions
# ==============================================================================


def get_rmsnorm_cls():
    """Return appropriate RMSNorm implementation (CPU or Neuron)."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


def convert_hf_to_neuron_mamba_weights(
    hf_state_dict: Dict[str, torch.Tensor], tp_degree: int = 4
) -> Dict[str, torch.Tensor]:
    """Convert HF Granite Mamba conv1d weight keys to our parameter names."""
    converted = {}
    for key, tensor in hf_state_dict.items():
        if "conv1d.weight" in key:
            new_key = key.replace("conv1d.weight", "conv_weight")
            converted[new_key] = tensor.squeeze(1)
        elif "conv1d.bias" in key:
            new_key = key.replace("conv1d.bias", "conv_bias")
            converted[new_key] = tensor
        else:
            converted[key] = tensor
    return converted


# ==============================================================================
# Attention
# ==============================================================================


class NeuronGraniteAttention(NeuronAttentionBase):
    """Granite attention layer. Uses no position embeddings ("nope")."""

    def __init__(self, config: GraniteInferenceConfig, layer_idx: int):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=None,  # Granite uses "nope" (no position embeddings)
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,
        )

        self.layer_idx = layer_idx
        self.attention_multiplier = config.attention_multiplier

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronGraniteAttention must be initialized in a distributed env."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        """Forward pass with attention multiplier scaling."""
        hidden_states, present_key_value, cos_cache, sin_cache = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        if self.attention_multiplier != 1.0:
            hidden_states = hidden_states * self.attention_multiplier

        return (hidden_states, present_key_value, cos_cache, sin_cache)


# ==============================================================================
# Decoder Layer
# ==============================================================================


class NeuronGraniteDecoderLayer(nn.Module):
    """Granite decoder layer — either attention or Mamba2, with MoE MLP."""

    def __init__(self, config: GraniteInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "attention":
            self.self_attn = NeuronGraniteAttention(config=config, layer_idx=layer_idx)
        elif self.layer_type == "mamba":
            self.mamba = NeuronMamba2Layer(config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        # MoE MLP with shared experts
        router = RouterTopK(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            sequence_dimension=1,
        )
        expert_mlps = ExpertMLPs(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            capacity_factor=config.neuron_config.capacity_factor,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            glu_mlp=config.neuron_config.glu_mlp,
            normalize_top_k_affinities=True,
            is_prefill=config.neuron_config.is_prefill_stage,
        )
        shared_expert = SharedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_intermediate_size,
            num_shared_experts=1,
            hidden_act=config.hidden_act,
            dtype=config.neuron_config.torch_dtype,
        )
        self.mlp = MoE(
            router=router,
            expert_mlps=expert_mlps,
            shared_experts=shared_expert,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            sequence_dimension=1,
        )
        self.mlp.eval()

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        mamba_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated. Use `attention_mask` instead."
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        updated_mamba_state = None
        if self.layer_type == "attention":
            hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
        else:
            hidden_states, present_key_value, updated_mamba_state = self.mamba(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                mamba_state=mamba_state,
                **kwargs,
            )
            cos_cache, sin_cache = None, None

        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states * self.residual_multiplier

        return (
            hidden_states,
            present_key_value,
            cos_cache,
            sin_cache,
            None,
            updated_mamba_state,
        )


# ==============================================================================
# Embedding/LM Head Wrappers
# ==============================================================================


class ScaledEmbedding(nn.Module):
    """Applies embedding_multiplier after embedding lookup."""

    def __init__(self, embedding, multiplier):
        super().__init__()
        self.embedding = embedding
        self.multiplier = multiplier

    def forward(self, input_ids):
        return self.embedding(input_ids) * self.multiplier


class ScaledLMHead(nn.Module):
    """Applies logits_scaling (division) after lm_head projection.

    Uses __getattr__ delegation for weight/bias access so framework code
    can find them without registering duplicate parameters.
    """

    def __init__(self, lm_head, scaling):
        super().__init__()
        self.lm_head = lm_head
        self.scaling = scaling
        if hasattr(lm_head, "gather_output"):
            self.gather_output = lm_head.gather_output
        if hasattr(lm_head, "tensor_parallel_group"):
            self.tensor_parallel_group = lm_head.tensor_parallel_group
        if hasattr(lm_head, "pad_size"):
            self.pad_size = lm_head.pad_size

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.lm_head, name)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states) / self.scaling


# ==============================================================================
# Model Body
# ==============================================================================


class NeuronGraniteModel(NeuronBaseModel):
    """
    NeuronGraniteModel — traced model body for Granite 4.0-H-Small.

    Overrides forward() and get_model_output() to handle Mamba state persistence
    alongside the standard KV cache. Mamba state is stored in nn.ParameterList
    (self.mamba_states) aliased via input_output_aliases in GraniteDecoderModelInstance.
    """

    def setup_attr_for_model(self, config: GraniteInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.embedding_multiplier = config.embedding_multiplier
        self.logits_scaling = config.logits_scaling

    def init_model(self, config: GraniteInferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        raw_embed = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.embed_tokens = ScaledEmbedding(raw_embed, self.embedding_multiplier)

        self.layers = nn.ModuleList(
            [
                NeuronGraniteDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)

        raw_lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )
        self.lm_head = ScaledLMHead(raw_lm_head, self.logits_scaling)

        # Mamba state persistence buffers
        self._mamba_layer_indices = [
            i
            for i in range(config.num_hidden_layers)
            if config.layer_types[i] == "mamba"
        ]
        batch_size = config.neuron_config.batch_size
        conv_shape, ssm_shape = NeuronMamba2Layer.get_state_shapes(config, batch_size)
        dtype = config.neuron_config.torch_dtype

        self.mamba_states = nn.ParameterList()
        for _ in self._mamba_layer_indices:
            self.mamba_states.append(
                nn.Parameter(torch.zeros(conv_shape, dtype=dtype), requires_grad=False)
            )
            self.mamba_states.append(
                nn.Parameter(torch.zeros(ssm_shape, dtype=dtype), requires_grad=False)
            )

        logger.info(
            f"Initialized Mamba state persistence: {len(self._mamba_layer_indices)} layers, "
            f"{len(self.mamba_states)} buffers"
        )

    def _get_mamba_states(self):
        """Get Mamba states as list of (conv_state, ssm_state) tuples."""
        states = []
        for i in range(0, len(self.mamba_states), 2):
            states.append((self.mamba_states[i], self.mamba_states[i + 1]))
        return states

    def _build_mamba_state_map(self):
        """Map layer_idx -> mamba_idx for Mamba layers."""
        return {
            layer_idx: mamba_idx
            for mamba_idx, layer_idx in enumerate(self._mamba_layer_indices)
        }

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Thread Mamba state through decoder layers alongside KV cache.
        Returns: (hidden_states, next_decoder_cache, updated_mamba_state_list)
        """
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        hidden_states = inputs_embeds

        if self.sequence_parallel_enabled:
            self.validate_sequence_parallel(seq_length)
        hidden_states = self.process_sequence_parallel_hidden_states(
            inputs_embeds, seq_length, kwargs.get("active_block_table", None)
        )

        next_decoder_cache = ()
        cos_cache = None
        sin_cache = None
        cache_size = self.n_positions

        if not is_for_context_encoding or windowed_context_encoding_window_idx >= 1:
            past_key_values = self.kv_mgr.get_cache(
                seq_ids=seq_ids,
                seq_len=cache_size,
                is_for_context_encoding=is_for_context_encoding,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        mamba_states = self._get_mamba_states()
        mamba_state_map = self._build_mamba_state_map()
        updated_mamba_states = [None] * len(self._mamba_layer_indices)

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            mamba_state = None
            if idx in mamba_state_map:
                mamba_state = mamba_states[mamba_state_map[idx]]

            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                kv_mgr=self.kv_mgr,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                local_mask=local_attn_mask,
                padding_mask=padding_mask,
                mamba_state=mamba_state,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            cos_cache, sin_cache = layer_outputs[2:4]
            layer_mamba_state = layer_outputs[5]

            if idx in mamba_state_map and layer_mamba_state is not None:
                updated_mamba_states[mamba_state_map[idx]] = layer_mamba_state

        if update_cache:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return (hidden_states, next_decoder_cache, updated_mamba_states)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
    ):
        """
        Traced forward — appends Mamba state tensors to output list.
        Output: [res, K0, V0, ..., conv_state_0, ssm_state_0, ...]
        """
        prev_hidden = self.set_none_if_empty(prev_hidden)
        adapter_ids = self.set_none_if_empty(adapter_ids)
        accepted_indices = self.set_none_if_empty(accepted_indices)
        current_length = self.set_none_if_empty(current_length)
        medusa_mask = self.set_none_if_empty(medusa_mask)
        scatter_index = self.set_none_if_empty(scatter_index)
        slot_mapping = self.set_none_if_empty(slot_mapping)
        active_block_table = self.set_none_if_empty(active_block_table)
        num_queries = self.set_none_if_empty(num_queries)
        computed_context_lens = self.set_none_if_empty(computed_context_lens)
        tile_q_indices = self.set_none_if_empty(tile_q_indices)
        tile_block_tables = self.set_none_if_empty(tile_block_tables)
        tile_masks = self.set_none_if_empty(tile_masks)
        inputs_embeds = self.set_none_if_empty(inputs_embeds)
        kv_cache = self.set_none_if_empty(kv_cache)
        active_mask = self.set_none_if_empty(active_mask)
        rotary_position_id = self.set_none_if_empty(rotary_position_id)
        vision_embeddings = self.set_none_if_empty(vision_embeddings)
        vision_mask = self.set_none_if_empty(vision_mask)

        is_for_context_encoding = self._is_context_encoding(input_ids)

        attn_mask = self.create_attn_mask(
            attention_mask,
            is_for_context_encoding,
            False,
            position_ids=position_ids,
        )
        padding_mask = self.create_padding_mask(position_ids)

        hidden_states, updated_kv_cache, updated_mamba_states = self.get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            prev_hidden=prev_hidden,
            is_for_context_encoding=is_for_context_encoding,
            scatter_index=slot_mapping
            if getattr(self, "is_block_kv_layout", False)
            else scatter_index,
            kvcache_buffer=kv_cache,
            update_cache=True,
            padding_mask=padding_mask,
        )

        # Slice to last token for context encoding
        batch_size = input_ids.shape[0]
        if not self.sliced_hidden:
            if not (
                position_ids.shape[-1] == getattr(self, "speculation_length", 0)
                or position_ids.shape[-1] == 1
            ):
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if hasattr(self.lm_head, "pad_size"):
            if self.lm_head.gather_output:
                rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
                world_size = 1
            else:
                rank_id = self.rank_util.get_rank()
                world_size = torch.distributed.get_world_size(
                    group=self.lm_head.tensor_parallel_group
                )
            from neuronx_distributed_inference.models.model_base import (
                mask_padded_logits,
            )

            logits = mask_padded_logits(
                logits, rank_id, world_size, pad_size=self.lm_head.pad_size
            )

        if self.on_device_sampling:
            res = self._sample_on_device(
                logits, sampling_params, False, is_for_context_encoding
            )
        else:
            res = logits

        outputs = [res]
        if getattr(self.neuron_config, "output_logits", False):
            from neuronx_distributed_inference.models.model_base import (
                _gather_along_dim,
                get_tp_group,
            )

            gathered_logits = _gather_along_dim(
                logits,
                partition_dim=2,
                process_group=get_tp_group(self.config),
            )
            outputs += [gathered_logits]
        outputs += updated_kv_cache

        # Append Mamba states for aliasing
        for conv_state, ssm_state in updated_mamba_states:
            outputs.append(conv_state)
            outputs.append(ssm_state)

        return outputs


# ==============================================================================
# CausalLM Wrapper
# ==============================================================================


class NeuronGraniteForCausalLM(NeuronBaseForCausalLM):
    """Top-level causal LM class for Granite 4.0-H-Small."""

    _model_cls = NeuronGraniteModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return GraniteInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: GraniteInferenceConfig
    ) -> dict:
        return _convert_granite_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        return (
            "--enable-saturate-infinity --enable-mixed-precision-accumulation "
            "--model-type transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
            "--auto-cast=none"
        )

    def get_model_wrapper_cls(self):
        return GraniteModelWrapper

    def _copy_past_key_values(self, outputs):
        """Also copy Mamba states for CPU debugging path."""
        n_mamba_entries = len(self.context_encoding_model.model.mamba_states)

        if n_mamba_entries > 0:
            super()._copy_past_key_values(outputs[:-n_mamba_entries])

            mamba_outputs = outputs[-n_mamba_entries:]
            for i, state_tensor in enumerate(mamba_outputs):
                self.token_generation_model.model.mamba_states[i].data = state_tensor
                self.context_encoding_model.model.mamba_states[i].data = state_tensor
        else:
            super()._copy_past_key_values(outputs)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: dict):
        """Handle tied embeddings and ScaledEmbedding/ScaledLMHead wrapper paths."""
        if "embed_tokens.weight" in state_dict:
            state_dict["embed_tokens.embedding.weight"] = state_dict.pop(
                "embed_tokens.weight"
            )

        if (
            "embed_tokens.embedding.weight" in state_dict
            and "lm_head.lm_head.weight" not in state_dict
        ):
            state_dict["lm_head.lm_head.weight"] = state_dict[
                "embed_tokens.embedding.weight"
            ]

        if "lm_head.weight" in state_dict:
            state_dict["lm_head.lm_head.weight"] = state_dict.pop("lm_head.weight")

        return state_dict


# ==============================================================================
# State Dict Conversion
# ==============================================================================


def _convert_granite_hf_to_neuron_state_dict(
    neuron_state_dict: Dict[str, Any], config: GraniteInferenceConfig
):
    """Convert HF checkpoints to Neuron-compatible state dict."""
    neuron_state_dict = convert_hf_to_neuron_mamba_weights(
        neuron_state_dict, config.neuron_config.tp_degree
    )

    # Remove "model." prefix
    new_state_dict = {}
    for key, value in neuron_state_dict.items():
        new_key = key[6:] if key.startswith("model.") else key
        new_state_dict[new_key] = value
    neuron_state_dict = new_state_dict

    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for layer_idx in range(config.num_hidden_layers):
        layer_type = config.layer_types[layer_idx]

        if layer_type == "attention":
            if config.neuron_config.fused_qkv:
                _helper_concat_and_delete_qkv(neuron_state_dict, layer_idx, "weight")
                if (
                    config.neuron_config.quantized_mlp_kernel_enabled
                    or config.neuron_config.quantized
                ):
                    _helper_concat_and_delete_qkv(neuron_state_dict, layer_idx, "scale")

        _convert_granite_moe_weights(neuron_state_dict, config, layer_idx)

    gc.collect()
    return neuron_state_dict


def _helper_concat_and_delete_qkv(
    state_dict: Dict[str, Any], layer_num: int, attr: str
):
    """Concatenate QKV weights for fused attention."""
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


def _convert_granite_moe_weights(
    state_dict: Dict[str, Any], config: GraniteInferenceConfig, layer_idx: int
):
    """Convert Granite MoE weights to neuronx_distributed format."""
    router_key = f"layers.{layer_idx}.block_sparse_moe.router.layer.weight"
    if router_key not in state_dict:
        return

    # Router weights
    neuron_router_key = f"layers.{layer_idx}.mlp.router.linear_router.weight"
    state_dict[neuron_router_key] = state_dict[router_key].detach().clone()
    del state_dict[router_key]

    # Expert weights (transpose for NxD format)
    input_linear_key = f"layers.{layer_idx}.block_sparse_moe.input_linear.weight"
    output_linear_key = f"layers.{layer_idx}.block_sparse_moe.output_linear.weight"

    if input_linear_key in state_dict and output_linear_key in state_dict:
        state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = (
            state_dict[input_linear_key].transpose(1, 2)
        )
        del state_dict[input_linear_key]

        state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"] = (
            state_dict[output_linear_key].transpose(1, 2)
        )
        del state_dict[output_linear_key]

    # Shared expert weights
    shared_input_key = f"layers.{layer_idx}.shared_mlp.input_linear.weight"
    shared_output_key = f"layers.{layer_idx}.shared_mlp.output_linear.weight"

    if shared_input_key in state_dict:
        input_linear = state_dict[shared_input_key]
        shared_intermediate_size = input_linear.shape[0] // 2
        state_dict[f"layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = (
            input_linear[:shared_intermediate_size, :]
        )
        state_dict[f"layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = (
            input_linear[shared_intermediate_size:, :]
        )
        del state_dict[shared_input_key]

    if shared_output_key in state_dict:
        state_dict[f"layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = (
            state_dict[shared_output_key].detach().clone()
        )
        del state_dict[shared_output_key]
