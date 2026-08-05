# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and The HuggingFace Inc. team.
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
Qwen3-Omni-MoE text model (Thinker) for NxD Inference.

Supports both text-only CausalLM mode and multimodal (vision+audio+text) mode.

The thinker text model is architecturally identical to Qwen3-MoE with mRoPE
(multimodal rotary position embeddings) for 3D position encoding (time, height, width).
This implementation reuses the NxD MoE modules and Qwen3-VL's mRoPE.

Key features:
  1. Config navigation: thinker_config -> text_config
  2. State dict prefix stripping: "thinker.model." / "thinker.lm_head."
  3. mRoPE with interleaved layout (same as Qwen3-VL)
  4. Vision embedding scatter for multimodal fusion (deepstack)
"""
import gc
import json
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple, Type, Union

import math
import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
    MOE_TKG_MK_INTERMEDIATE_PER_TP,
)
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.image_to_text_model_wrapper import ImageToTextModelWrapper
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import scatter_by_index_put
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed.parallel_layers.mappings import (
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
)
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

logger = logging.getLogger(__name__)


def get_rmsnorm_cls():
    return Qwen3MoeRMSNorm if cpu_mode() else CustomRMSNorm


# ---------------------------------------------------------------------------
# State dict conversion helpers (same logic as qwen3_moe)
# ---------------------------------------------------------------------------

def _strip_thinker_prefix(state_dict: dict) -> dict:
    """
    Strip the thinker prefix from HF Qwen3-Omni state dict keys.

    HF keys look like:
      thinker.model.embed_tokens.weight
      thinker.model.layers.0.self_attn.q_proj.weight
      thinker.lm_head.weight

    We map them to:
      embed_tokens.weight
      layers.0.self_attn.q_proj.weight
      lm_head.weight
    """
    prefixes = ["model.thinker.model.", "thinker.model."]
    lm_head_prefixes = ["model.thinker.lm_head.", "thinker.lm_head."]

    # detect prefix
    model_prefix = ""
    for p in prefixes:
        if any(k.startswith(p) for k in state_dict):
            model_prefix = p
            break

    lm_head_prefix = ""
    for p in lm_head_prefixes:
        if any(k.startswith(p) for k in state_dict):
            lm_head_prefix = p
            break

    stripped = {}
    for key, value in state_dict.items():
        if model_prefix and key.startswith(model_prefix):
            new_key = key[len(model_prefix):]
            stripped[new_key] = value
        elif lm_head_prefix and key.startswith(lm_head_prefix):
            new_key = "lm_head." + key[len(lm_head_prefix):]
            stripped[new_key] = value
        elif not model_prefix:
            # no prefix detected — keys are already bare
            stripped[key] = value

    logger.info(
        "Stripped thinker prefix: %d HF keys -> %d thinker text keys (prefix='%s')",
        len(state_dict), len(stripped), model_prefix,
    )
    return stripped


def _helper_concat_and_delete_qkv(sd: dict, layer: int, attr: str):
    sd[f"layers.{layer}.self_attn.Wqkv.{attr}"] = torch.cat([
        sd[f"layers.{layer}.self_attn.q_proj.{attr}"],
        sd[f"layers.{layer}.self_attn.k_proj.{attr}"],
        sd[f"layers.{layer}.self_attn.v_proj.{attr}"],
    ])
    del sd[f"layers.{layer}.self_attn.q_proj.{attr}"]
    del sd[f"layers.{layer}.self_attn.k_proj.{attr}"]
    del sd[f"layers.{layer}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(sd: dict, cfg: InferenceConfig) -> dict:
    mods_to_skip = getattr(cfg.neuron_config, "modules_to_not_convert", None) or []
    for l in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(sd, l, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_skip:
            _helper_concat_and_delete_qkv(sd, l, "scale")
    gc.collect()
    return sd


def convert_qwen3_omni_moe_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
    """
    Convert HF Qwen3-Omni-MoE thinker text state dict to NxD MoE format.

    Steps:
      1. Strip thinker.model.* prefix
      2. Add rank_util tensors for TP
      3. Rename q_norm/k_norm -> q_layernorm/k_layernorm
      4. Rename router: mlp.gate -> mlp.router.linear_router
      5. Reorganize expert weights into stacked 3D tensors
      6. Optionally fuse QKV
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    neuron_state_dict = _strip_thinker_prefix(state_dict)

    # rank utilities
    tp = config.neuron_config.tp_degree
    neuron_state_dict["rank_util.rank"] = torch.arange(0, tp, dtype=torch.int32)

    for l in range(config.num_hidden_layers):
        neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(0, tp, dtype=torch.int32)

        # rename qk norm
        for proj in ("q", "k"):
            old = f"layers.{l}.self_attn.{proj}_norm.weight"
            new = f"layers.{l}.self_attn.{proj}_layernorm.weight"
            if old in neuron_state_dict:
                neuron_state_dict[new] = neuron_state_dict.pop(old).detach().clone()

        # rename router
        gate_key = f"layers.{l}.mlp.gate.weight"
        if gate_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
                neuron_state_dict.pop(gate_key).detach().clone()
            )

        # reorganize expert weights
        sample_key = f"layers.{l}.mlp.experts.0.gate_proj.weight"
        if sample_key not in neuron_state_dict:
            continue
        intermediate_size, hidden_size = neuron_state_dict[sample_key].shape
        device = neuron_state_dict[sample_key].device
        dtype = neuron_state_dict[sample_key].dtype

        gate_up_proj = torch.empty(config.num_experts, hidden_size, 2 * intermediate_size, dtype=dtype, device=device)
        down_proj = torch.empty(config.num_experts, intermediate_size, hidden_size, dtype=dtype, device=device)

        for e in range(config.num_experts):
            gp = neuron_state_dict.pop(f"layers.{l}.mlp.experts.{e}.gate_proj.weight").T.detach().clone()
            up = neuron_state_dict.pop(f"layers.{l}.mlp.experts.{e}.up_proj.weight").T.detach().clone()
            dp = neuron_state_dict.pop(f"layers.{l}.mlp.experts.{e}.down_proj.weight").T.detach().clone()

            gate_up_proj[e, :, :intermediate_size] = gp
            gate_up_proj[e, :, intermediate_size:] = up
            down_proj[e] = dp

        pad_size = getattr(config, "moe_intermediate_pad_size", 0)
        if pad_size > 0:
            gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, 2, -1)
            gate_up_proj = torch.nn.functional.pad(gate_up_proj, (0, pad_size))
            gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, -1)
            down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))

        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj
        gc.collect()

    if config.neuron_config.fused_qkv:
        neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

    return neuron_state_dict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Qwen3OmniMoeInferenceConfig(InferenceConfig):
    """
    Inference config for Qwen3-Omni-MoE thinker text model.

    Navigates the nested Omni config (thinker_config.text_config) and sets up
    MoE parameters identically to Qwen3MoeInferenceConfig.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_local_experts = self.num_experts
        self.n_shared_experts = 0

        self.maybe_pad_intermediate()
        self.enable_moe_fused_nki_kernel()

        self.intermediate_size = self.moe_intermediate_size

        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"
        self.neuron_config.disable_numeric_cc_token = True
        self.neuron_config.normalize_top_k_affinities = True

    def maybe_pad_intermediate(self):
        moe_tp = self.neuron_config.moe_tp_degree
        i_tp = self.moe_intermediate_size // moe_tp
        if getattr(self.neuron_config.blockwise_matmul_config, "use_shard_on_intermediate_dynamic_while", False):
            if i_tp % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded = math.ceil(i_tp / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP) * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP * moe_tp
                self.moe_intermediate_pad_size = max(padded - self.moe_intermediate_size, 0)
                self.moe_intermediate_size = padded

    def enable_moe_fused_nki_kernel(self):
        i_tp = self.moe_intermediate_size // self.neuron_config.moe_tp_degree
        if getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", False) and i_tp % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0:
            self.moe_fused_nki_kernel_enabled = True

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "norm_topk_prob",
            "num_attention_heads",
            "num_experts",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_scaling",
            "rope_theta",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


def load_qwen3_omni_thinker_text_config(model_path: str):
    """
    Return a load_config hook that extracts thinker.text_config from the
    Qwen3-Omni config.json and applies it to the InferenceConfig.
    """
    def load_config(self: InferenceConfig):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            full = json.load(f)

        thinker = full.get("thinker_config", {})
        text = thinker.get("text_config", {})
        if not text:
            raise ValueError(
                f"Could not find thinker_config.text_config in {config_path}"
            )

        # torch_dtype handling
        hf_dtype = text.pop("torch_dtype", text.pop("dtype", None))
        if hf_dtype and self.neuron_config and not self.neuron_config.overrides_torch_dtype:
            from neuronx_distributed_inference.models.config import to_torch_dtype
            if isinstance(hf_dtype, str):
                hf_dtype = to_torch_dtype(hf_dtype)
            self.neuron_config.torch_dtype = hf_dtype

        # Remove keys that conflict with InferenceConfig internals
        for skip in ("model_type", "transformers_version", "architectures", "_name_or_path"):
            text.pop(skip, None)

        self.__dict__.update(text)

    return load_config


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class NeuronQwen3OmniMoERotaryEmbedding(nn.Module):
    """mRoPE for Qwen3-Omni — identical to Qwen3-VL's interleaved layout."""
    inv_freq: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self.rope_type = rope_scaling.get("rope_type", "default")
        assert self.rope_type == "default", f"Only 'default' rope_type supported, got {self.rope_type}"

        base = config.rope_theta
        dim = config.head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = NeuronQwen3OmniMoERotaryEmbedding.neuron_compute_freqs_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def neuron_compute_freqs_mrope(freqs: torch.Tensor, mrope_section: list) -> torch.Tensor:
        """XLA-friendly interleaved mRoPE frequency computation."""
        last_dim = freqs.shape[-1]
        indices = torch.arange(last_dim, device=freqs.device, dtype=torch.int64)
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            mask = (indices % 3 == offset) & (indices < length)
            freqs_t = torch.where(mask, freqs[dim], freqs_t)
        return freqs_t


class NeuronQwen3OmniMoEAttention(NeuronAttentionBase):
    def __init__(self, config: Qwen3OmniMoeInferenceConfig):
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling and rope_scaling.get("mrope_section"):
            rotary_emb = NeuronQwen3OmniMoERotaryEmbedding(config)
        else:
            rotary_emb = RotaryEmbedding(
                config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,
        )
        self.q_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwen3OmniMoEAttention must be initialized in a distributed env."
            )


class NeuronQwen3OmniMoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3OmniMoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3OmniMoEAttention(config=config)
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", False)

        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        if self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True,
            )
        else:
            self.mlp = initialize_moe_module(config=config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated. Use `attention_mask` instead."
            )

        residual = hidden_states

        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        qkv_fused_rmsnorm = None
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

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
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        is_spec = (
            self.config.neuron_config.enable_fused_speculation
            and not self.config.neuron_config.is_prefill_stage
        )
        hidden_states = self.mlp(hidden_states, padding_mask, is_speculative_decoding=is_spec)[0]
        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronQwen3OmniMoeModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: Qwen3OmniMoeInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen3OmniMoeInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList([
            NeuronQwen3OmniMoeDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
        )

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask) -> torch.Tensor:
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    def deepstack_process_xla(
        self,
        hidden_states: torch.Tensor,
        visual_embeds: torch.Tensor,
        vision_mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.sequence_parallel_enabled:
            from neuronx_distributed_inference.utils.distributed import get_tp_group
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=get_tp_group(self.config),
            )

        assert hidden_states.shape == visual_embeds.shape, (
            f"Shape mismatch: hidden_states.shape={hidden_states.shape}, "
            f"visual_embeds.shape={visual_embeds.shape}"
        )

        expanded_visual_embeds = torch.zeros_like(hidden_states)
        expanded_visual_embeds = scatter_by_index_put(
            expanded_visual_embeds, visual_embeds, vision_mask_positions
        )
        hidden_states = hidden_states + expanded_visual_embeds

        if self.sequence_parallel_enabled:
            from neuronx_distributed_inference.utils.distributed import get_tp_group
            hidden_states = _reduce_scatter_along_dim(
                hidden_states,
                self.sequence_dimension,
                xm.REDUCE_MAX,
                process_group=get_tp_group(self.config),
            )

        return hidden_states


# ---------------------------------------------------------------------------
# Top-level CausalLM
# ---------------------------------------------------------------------------

class NeuronQwen3OmniMoeForCausalLM(NeuronBaseForCausalLM):
    """
    Causal LM wrapper for the Qwen3-Omni-MoE thinker text model on Neuron.

    Usage:
        from modeling_qwen3_omni_moe import (
            NeuronQwen3OmniMoeForCausalLM,
            Qwen3OmniMoeInferenceConfig,
            load_qwen3_omni_thinker_text_config,
        )
        from neuronx_distributed_inference.models.config import MoENeuronConfig

        neuron_config = MoENeuronConfig(tp_degree=8, batch_size=1, seq_len=512, ...)
        config = Qwen3OmniMoeInferenceConfig(
            neuron_config,
            load_config=load_qwen3_omni_thinker_text_config(model_path),
        )
        model = NeuronQwen3OmniMoeForCausalLM(model_path, config)
        model.compile(compiled_model_path)
        model.load(compiled_model_path)
    """

    _model_cls = NeuronQwen3OmniMoeModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Qwen3OmniMoeInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        return convert_qwen3_omni_moe_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            opt = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            opt = "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"
        else:
            opt = "-O1"

        args = (
            f"--enable-saturate-infinity --enable-mixed-precision-accumulation "
            f"--model-type transformer {opt}"
        )
        args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        args += " --auto-cast=none"
        args += " --internal-enable-dge-levels vector_dynamic_offsets"
        args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"

        if self.neuron_config.scratchpad_page_size:
            args += f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size} "

        if self.neuron_config.attn_block_tkg_nki_kernel_enabled:
            assert self.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention, (
                "attn_block_tkg_nki_kernel_enabled requires attn_block_tkg_nki_kernel_cascaded_attention"
            )
            self.neuron_config.pre_rope_rmsnorm = True
            args += " --internal-max-instruction-limit=15000000"

        return args


# ---------------------------------------------------------------------------
# Text model wrapper for multimodal (ImageToText) mode
# ---------------------------------------------------------------------------

class NeuronQwen3OmniMoeTextModelWrapper(ImageToTextModelWrapper):
    """Wraps the MoE text model for multimodal inference with mRoPE position IDs."""

    _ROTARY_POSITION_IDS_INDEX = 21

    def _forward_with_pad(self, *args):
        """Fix rotary_position_ids after parent's incorrect dim-0 batch slice."""
        args = list(args)
        rpi = args[self._ROTARY_POSITION_IDS_INDEX]

        if rpi.dim() == 3 and rpi.shape[0] != 3:
            rpi = rpi[:1].expand(3, -1, -1)

        if rpi.dim() == 3 and rpi.shape[1] < self.neuron_config.batch_size:
            pad_size = self.neuron_config.batch_size - rpi.shape[1]
            padding = rpi[:, :1, :].expand(-1, pad_size, -1)
            rpi = torch.cat([rpi, padding], dim=1)

        args[self._ROTARY_POSITION_IDS_INDEX] = rpi
        return super()._forward_with_pad(*args)

    @staticmethod
    def get_dummy_vision_inputs(config, input_ids, n_active_tokens, fill_value):
        input_batch_size, input_sequence_len = input_ids.shape[0], input_ids.shape[-1]
        if input_sequence_len > 1:  # prefill
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
            deepstack_vision_embeds = [
                torch.zeros(
                    input_batch_size,
                    config.neuron_config.seq_len,
                    config.hidden_size,
                    dtype=config.neuron_config.torch_dtype,
                )
                for _ in getattr(config, "deepstack_visual_indexes", [])
            ]
            if len(deepstack_vision_embeds) > 0:
                deepstack_vision_embeds = torch.stack(deepstack_vision_embeds)
            else:
                deepstack_vision_embeds = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
        else:  # decode
            vision_embeddings = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
            vision_mask = torch.zeros((0), dtype=torch.bool)
            deepstack_vision_embeds = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
        return vision_embeddings, vision_mask, deepstack_vision_embeds

    def input_generator(self):
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket
                if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )

            input_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            attention_mask = torch.zeros(
                (self.neuron_config.batch_size, bucket), dtype=torch.int32
            )
            position_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)

            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros(
                (self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32
            )

            vision_embeddings, vision_mask, deepstack_vision_embeds = (
                self.get_dummy_vision_inputs(
                    config=self.config,
                    input_ids=input_ids,
                    n_active_tokens=n_active_tokens,
                    fill_value=0,
                )
            )

            rotary_position_ids = torch.zeros(
                (3, self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )

            if self.tag == CONTEXT_ENCODING_MODEL_TAG or self.tag == TOKEN_GENERATION_MODEL_TAG:
                inputs.append(
                    (
                        input_ids,                  # 0
                        attention_mask,             # 1
                        position_ids,               # 2
                        seq_ids,                    # 3
                        sampling_params,            # 4
                        torch.empty(0),             # 5  prev_hidden
                        torch.empty(0),             # 6  adapter_ids
                        torch.empty(0),             # 7  accepted_indices
                        torch.empty(0),             # 8  current_length
                        torch.empty(0),             # 9  medusa_mask
                        torch.empty(0),             # 10 scatter_index
                        torch.empty(0),             # 11 slot_mapping
                        torch.empty(0),             # 12 active_block_table
                        torch.empty(0),             # 13 num_queries
                        torch.empty(0),             # 14 computed_context_lens
                        torch.empty(0),             # 15 tile_q_indices
                        torch.empty(0),             # 16 tile_block_tables
                        torch.empty(0),             # 17 tile_masks
                        torch.empty(0),             # 18 inputs_embeds
                        torch.empty(0),             # 19 kv_cache
                        torch.empty(0),             # 20 active_mask
                        rotary_position_ids,        # 21
                        vision_embeddings,          # 22
                        vision_mask,                # 23
                        deepstack_vision_embeds,    # 24
                    )
                )
            else:
                raise ValueError(f"Unsupported model tag '{self.tag}'")

        return inputs
