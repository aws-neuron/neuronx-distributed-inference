"""Qwen3-Omni MoE text model for NxD Inference.

Combines Qwen3-VL's multimodal attention (MRoPE, deepstack, vision scatter)
with Qwen3-MoE's sparse mixture-of-experts FFN layers.
"""

import gc
import math
import warnings
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
    MOE_TKG_MK_INTERMEDIATE_PER_TP,
)
from neuronx_distributed_inference.models.image_to_text_model_base import NeuronBaseForImageToText
from neuronx_distributed_inference.models.image_to_text_model_wrapper import ImageToTextModelWrapper
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

# Reuse Qwen3-VL components (identical attention + MRoPE + vision integration)
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
    NeuronQwen3VLAttention,
    NeuronQwen3VLRotaryEmbedding,
    NeuronQwen3VLTextModel,
    get_rmsnorm_cls,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import scatter_by_index_put


class NeuronQwen3OmniMoEDecoderLayer(nn.Module):
    """Decoder layer: Qwen3-VL attention (MRoPE) + Qwen3-MoE sparse FFN."""

    def __init__(self, config: InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3VLAttention(config)
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", False)

        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        if self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True
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
        rotary_position_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated. Use `attention_mask` instead."
            )

        residual = hidden_states

        qkv_fused_rmsnorm = None
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
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
            rotary_position_ids=rotary_position_ids,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        is_speculative_decoding = (
            self.config.neuron_config.enable_fused_speculation
            and not self.config.neuron_config.is_prefill_stage
        )
        hidden_states = self.mlp(hidden_states, padding_mask, is_speculative_decoding=is_speculative_decoding)[0]
        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronQwen3OmniTextModel(NeuronQwen3VLTextModel):
    """MoE text model with deepstack and vision scatter from Qwen3-VL."""

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
                gather_output=not self.on_device_sampling,
                bias=False,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size, self.hidden_size, self.padding_idx,
            )
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.layers = nn.ModuleList(
            [NeuronQwen3OmniMoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


class NeuronQwen3OmniTextModelWrapper(ImageToTextModelWrapper):
    """Wrapper with MRoPE input generator and deepstack dummy inputs.

    Identical to NeuronQwen3VLTextModelWrapper.
    """

    _ROTARY_POSITION_IDS_INDEX = 21

    def _forward_with_pad(self, *args):
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
        if input_sequence_len > 1:
            vision_embeddings = torch.zeros(
                input_batch_size, config.neuron_config.seq_len, config.hidden_size,
                dtype=config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                size=(input_batch_size, n_active_tokens, 1),
                fill_value=fill_value,
                dtype=torch.int32,
            )
            deepstack_vision_embeds = [
                torch.zeros(
                    input_batch_size, config.neuron_config.seq_len, config.hidden_size,
                    dtype=config.neuron_config.torch_dtype,
                )
                for _ in config.deepstack_visual_indexes
            ]
            if len(deepstack_vision_embeds) > 0:
                deepstack_vision_embeds = torch.stack(deepstack_vision_embeds)
            else:
                deepstack_vision_embeds = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
        else:
            vision_embeddings = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
            vision_mask = torch.zeros((0), dtype=torch.bool)
            deepstack_vision_embeds = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
        return vision_embeddings, vision_mask, deepstack_vision_embeds

    def input_generator(self):
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )
            input_ids = torch.zeros((self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32)
            attention_mask = torch.zeros((self.neuron_config.batch_size, bucket), dtype=torch.int32)
            position_ids = torch.zeros((self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32)
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros((self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32)
            vision_embeddings, vision_mask, deepstack_vision_embeds = self.get_dummy_vision_inputs(
                config=self.config, input_ids=input_ids,
                n_active_tokens=n_active_tokens, fill_value=0,
            )
            rotary_position_ids = torch.zeros(
                (3, self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            if self.tag in (CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG):
                inputs.append((
                    input_ids,              # 0
                    attention_mask,         # 1
                    position_ids,           # 2
                    seq_ids,                # 3
                    sampling_params,        # 4
                    torch.empty(0),         # 5  prev_hidden
                    torch.empty(0),         # 6  adapter_ids
                    torch.empty(0),         # 7  accepted_indices
                    torch.empty(0),         # 8  current_length
                    torch.empty(0),         # 9  medusa_mask
                    torch.empty(0),         # 10 scatter_index
                    torch.empty(0),         # 11 slot_mapping
                    torch.empty(0),         # 12 active_block_table
                    torch.empty(0),         # 13 num_queries
                    torch.empty(0),         # 14 computed_context_lens
                    torch.empty(0),         # 15 tile_q_indices
                    torch.empty(0),         # 16 tile_block_tables
                    torch.empty(0),         # 17 tile_masks
                    torch.empty(0),         # 18 inputs_embeds
                    torch.empty(0),         # 19 kv_cache
                    torch.empty(0),         # 20 active_mask
                    rotary_position_ids,    # 21
                    vision_embeddings,      # 22
                    vision_mask,            # 23
                    deepstack_vision_embeds,  # 24
                ))
            else:
                raise ValueError(f"Unsupported model tag '{self.tag}'")
        return inputs


def convert_qwen3_omni_text_hf_to_neuron(state_dict: dict, config: InferenceConfig) -> dict:
    """Convert HF Qwen3-Omni thinker text weights to Neuron format.

    Handles both MRoPE attention key remapping (Qwen3-VL style) and
    MoE expert weight stacking (Qwen3-MoE style).
    """
    assert config.neuron_config.glu_mlp is True

    new_sd: Dict[str, Any] = {}

    # Step 1: Strip thinker prefix from text-model keys; preserve already-converted
    # vision (blocks.*) and audio (frontend.*, transformer.*, postprocessor.*) keys.
    for k, v in state_dict.items():
        if k.startswith("thinker.model."):
            new_key = k[len("thinker.model."):]
            new_sd[new_key] = v
        elif k.startswith("thinker.lm_head."):
            new_key = k[len("thinker."):]
            new_sd[new_key] = v
        elif k.startswith("thinker."):
            # Drop any other thinker.* keys (e.g., thinker.audio_tower leftovers)
            continue
        else:
            new_sd[k] = v

    state_dict = new_sd

    # Step 2: Attention key remapping (Qwen3-VL style)
    attention_renames = {
        ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
        ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
        ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
        ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
    }
    renamed_sd: Dict[str, Any] = {}
    for k, v in state_dict.items():
        new_key = k
        if not config.neuron_config.fused_qkv:
            for old, new in attention_renames.items():
                if old in new_key:
                    new_key = new_key.replace(old, new)
                    break
        if ".q_norm." in new_key:
            new_key = new_key.replace(".q_norm.", ".q_layernorm.")
        if ".k_norm." in new_key:
            new_key = new_key.replace(".k_norm.", ".k_layernorm.")
        renamed_sd[new_key] = v
    state_dict = renamed_sd

    # Step 3: rank_util tensors
    state_dict["rank_util.rank"] = torch.arange(0, config.neuron_config.tp_degree, dtype=torch.int32)

    # Step 4: MoE weight conversion (Qwen3-MoE style)
    for l in range(config.num_hidden_layers):
        state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # Router: gate -> router.linear_router
        gate_key = f"layers.{l}.mlp.gate.weight"
        if gate_key in state_dict:
            state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = state_dict.pop(gate_key)

        # Stack expert weights
        expert_key_0 = f"layers.{l}.mlp.experts.0.gate_proj.weight"
        if expert_key_0 not in state_dict:
            continue

        intermediate_size, hidden_size = state_dict[expert_key_0].shape
        device = state_dict[expert_key_0].device
        dtype = state_dict[expert_key_0].dtype

        gate_up_proj = torch.empty(config.num_experts, hidden_size, 2 * intermediate_size, dtype=dtype, device=device)
        for e in range(config.num_experts):
            gw = state_dict.pop(f"layers.{l}.mlp.experts.{e}.gate_proj.weight")
            uw = state_dict.pop(f"layers.{l}.mlp.experts.{e}.up_proj.weight")
            # copy_() writes into the preallocated buffer and releases gw/uw after the copy;
            # avoids holding a second transposed materialization in RAM.
            gate_up_proj[e, :, :intermediate_size].copy_(gw.T)
            gate_up_proj[e, :, intermediate_size:].copy_(uw.T)
            del gw, uw

        pad_size = getattr(config, "moe_intermediate_pad_size", 0)
        if pad_size > 0:
            gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, 2, -1)
            gate_up_proj = torch.nn.functional.pad(gate_up_proj, (0, pad_size))
            gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, -1)
        state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        down_proj = torch.empty(config.num_experts, intermediate_size, hidden_size, dtype=dtype, device=device)
        for e in range(config.num_experts):
            dw = state_dict.pop(f"layers.{l}.mlp.experts.{e}.down_proj.weight")
            down_proj[e].copy_(dw.T)
            del dw
        if pad_size > 0:
            down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))
        state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return state_dict
