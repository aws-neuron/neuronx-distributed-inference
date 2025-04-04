# Meta Llama 3 is licensed under the Meta Llama 3 Community License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE in the
# # current directory, mllama/.

"""PyTorch LLaMA Multimodal model for NXD inference."""
import copy
import inspect
import json
import logging
import math
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.mappings import (
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
)
from torch import Tensor, nn
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from neuronx_distributed_inference.models.llama.modeling_llama import Llama3RotaryEmbedding
from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    apply_rotary_polar_compatible,
    move_heads_front,
    precompute_freqs_cis,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm  # noqa: F401
from neuronx_distributed_inference.modules.kvcache.multimodal_kv_cache_manager import (
    MultimodalKVCacheManager,
)
from neuronx_distributed_inference.utils.distributed import get_tp_group

from .model_wrapper_mllama import ModelWrapperMllama
from .utils import HF_CHECKPOINT, META_CHECKPOINT

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import (  # noqa: E402
    InferenceConfig,
    MultimodalVisionNeuronConfig,
    to_dict,
)
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
    get_cache_size,
    mask_util,
    turn_2d_mask_to_4d,
)

from .modeling_mllama_vision import NeuronMllamaVisionModel  # noqa: E402

DEBUG = False


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF's LlamaRMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.get_tensor_model_parallel_size() > 1 else LlamaRMSNorm


class MllamaInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "checkpoint"):
            self.checkpoint = kwargs.get("checkpoint", HF_CHECKPOINT)

        assert self.checkpoint in [
            HF_CHECKPOINT,
            META_CHECKPOINT,
        ], f"Uknown checkpoint: {self.checkpoint}"

        if hasattr(self, "text_config"):
            if isinstance(self.text_config, SimpleNamespace):
                self.text_config = vars(self.text_config)
            # replicating what's done in hf_adapter's load_config()
            self.text_config.pop("torch_dtype", None)
            self.text_config = InferenceConfig(self.neuron_config, **self.text_config)
            if not hasattr(self.text_config, "checkpoint"):
                setattr(self.text_config, "checkpoint", self.checkpoint)

        if hasattr(self, "vision_config"):
            if isinstance(self.vision_config, SimpleNamespace):
                self.vision_config = vars(self.vision_config)
            # replicating what's done in hf_adapter's load_config()
            self.vision_config.pop("torch_dtype", None)
            self.vision_config = InferenceConfig(self.neuron_config, **self.vision_config)
            if not hasattr(self.vision_config, "checkpoint"):
                setattr(self.vision_config, "checkpoint", self.checkpoint)

    def get_required_attributes(self) -> List[str]:
        # To validate if the config.json include all the configs we need in model.
        # Need to manually add what's required in below list
        return [
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.pad_token_id",
            "text_config.vocab_size",
            "text_config.max_position_embeddings",
            "text_config.rope_theta",
            "text_config.rms_norm_eps",
            "text_config.hidden_act",
            "text_config.cross_attention_layers",
            "vision_config.max_num_tiles",
            "vision_config.image_size",
            "vision_config.patch_size",
            "vision_config.num_hidden_layers",
            "vision_config.num_global_layers",
            "vision_config.num_channels",
            "vision_config.hidden_size",
            "vision_config.attention_heads",
            "vision_config.intermediate_layers_indices",
        ]

    def validate_config(self):
        """
        Validates that the config has all required attributes.
        """

        def hasattr_nested(obj, attr_chain):
            attrs = attr_chain.split(".")
            for attr in attrs:
                if isinstance(obj, dict):
                    if attr not in obj:
                        return False
                    obj = obj[attr]
                else:
                    if not hasattr(obj, attr):
                        return False
                    obj = getattr(obj, attr)
            return True

        missing_attributes = [
            x for x in self.get_required_attributes() if not hasattr_nested(self, x)
        ]
        assert len(missing_attributes) == 0, f"Config must define {missing_attributes}"

        assert (
            self.neuron_config.is_medusa is False and self.neuron_config.speculation_length == 0
        ), f"Speculative Decoding is not yet supported in this Model. \
                is_medusa was set to {self.neuron_config.is_medusa}. \
                speculation_length was set to {self.neuron_config.speculation_length}"
        assert (
            int(self.neuron_config.logical_nc_config) == 1
        ), "This model currently only support logical_nc_config=1"

    def to_json_string(self):
        config_copy = copy.deepcopy(self)
        config_dict = to_dict(config_copy)
        config_dict["text_config"].pop("neuron_config", None)
        config_dict["vision_config"].pop("neuron_config", None)
        return json.dumps(config_dict, indent=2, sort_keys=True)

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MultimodalVisionNeuronConfig]:
        return MultimodalVisionNeuronConfig


class NeuronLlamaAttention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()

        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = self.neuron_config.padding_side
        self.torch_dtype = self.neuron_config.torch_dtype
        self.is_medusa = self.neuron_config.is_medusa

        self.attn_kernel_enabled = True

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            self.tp_degree = 1
        self.fused_qkv = getattr(self.neuron_config, "fused_qkv", False)
        self.clip_qkv = None

        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None

        self.init_gqa_properties()

        self.init_rope()

    def init_rope(self):
        if not hasattr(self.config, "rope_scaling") or self.config.rope_scaling is None:
            # TODO(yihsian): Check if we can just use our own implementation
            if self.is_medusa:
                self.rotary_emb = LlamaRotaryEmbedding(self.config)
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
        else:
            rope_type = self.config.rope_scaling.get(
                "rope_type", self.config.rope_scaling.get("type", None)
            )
            if rope_type == "llama3":
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    factor=self.config.rope_scaling["factor"],
                    low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                    high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                # Warning: The HF implementation may have precision issues when run on Neuron.
                # We include it here for compatibility with other scaling types.
                self.rotary_emb = LlamaRotaryEmbedding(self.config)

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        rotary_freqs=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
    ):
        """
        Override NeuronAttentionBase.prep_qkv_tensors() to use apply_rotary_polar_compatible
        to match Llama3.2 MM Pytorch implementation
        """
        cos_cache, sin_cache = None, None
        if self.config.checkpoint == HF_CHECKPOINT:
            return super().prep_qkv_tensors(
                position_ids,
                hidden_states,
                past_key_value,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                rmsnorm=rmsnorm,
            )

        Q, K, V = self.qkv_proj(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids
        )

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q = move_heads_front(
            Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm
        )
        K = move_heads_front(
            K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=self.k_layernorm
        )
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Rotate Q and K
        if self.rotary_emb is not None:
            assert rotary_freqs is not None, "rotary_emb is initilazed but rotary_freqs is None"
            Q, K = apply_rotary_polar_compatible(Q.transpose(1, 2), K.transpose(1, 2), rotary_freqs)
            Q, K = Q.transpose(1, 2), K.transpose(1, 2)
        return Q, K, V, cos_cache, sin_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        rotary_freqs: Optional[torch.LongTensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Override NeuronAttentionBase.forward() to use apply_rotary_polar_compatible
        to match Llama3.2 MM Pytorch implementation
        """
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q, K, V, cos_cache, sin_cache = self.prep_qkv_tensors(
            position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            rotary_freqs=rotary_freqs,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
        )

        flash_attn_strategy = FlashAttentionStrategy.NONE
        if past_key_value is None:
            attn_output, flash_attn_strategy = self.perform_prefill(
                Q, K, V, q_len, bsz, attention_mask
            )
        else:
            attn_output = self.compute_for_token_gen(
                Q, K, V, position_ids, past_key_value, attention_mask, active_mask
            )

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            # transpose BHDS -> BSHD
            # this layout avoids additional transposes between attention kernel and output projection
            attn_output = attn_output.permute(0, 3, 1, 2)
        else:
            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output, adapter_ids=adapter_ids)

        past_key_value: Tuple[Tensor, Tensor] = (K, V)

        return attn_output, past_key_value, cos_cache, sin_cache


class DummyCrossAttention:
    """Dummy cross-attention transformer block that follows self-attention tranformer block in non-cross-attn layers."""

    def __call__(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states, (None, None)


class NeuronLlamaCrossAttention(torch.nn.Module):
    def __init__(self, config: InferenceConfig, vision_config):
        """Cross attention layer with model-parallel attention layers."""
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.vision_config = vision_config
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.rms_norm_eps = config.rms_norm_eps
        self.max_num_chunks = self.vision_config.max_num_tiles
        self.torch_dtype = config.neuron_config.torch_dtype
        # cross-attention heads are model parallel similar to
        # self-attention, and we also use the identical KV head
        # combination to ensure parity with the corresponding
        # trunk LLM (i.e., group query attention) -- @dubeya
        # local heads
        if parallel_state.model_parallel_is_initialized():
            self.model_parallel_size = parallel_state.get_tensor_model_parallel_size()
        else:
            self.model_parallel_size = 1

        self.replication_factor = 1
        if self.model_parallel_size > 8:
            self.replication_factor = self.model_parallel_size // 8
        self.org_n_kv_heads = self.n_kv_heads
        self.n_kv_heads *= self.replication_factor
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_heads % self.model_parallel_size == 0
        assert self.n_kv_heads % self.model_parallel_size == 0

        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None

        self.n_local_heads = self.n_heads // self.model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // self.model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.wq = ColumnParallelLinear(
            self.hidden_size,
            self.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=None,
        )
        self.wk = ColumnParallelLinear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=None,
        )
        self.wv = ColumnParallelLinear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=None,
        )
        self.wo = RowParallelLinear(
            self.n_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=None,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
        )
        self.q_norm = get_rmsnorm_cls()(
            self.head_dim,
            eps=self.rms_norm_eps,
        )
        self.k_norm = get_rmsnorm_cls()(
            self.head_dim,
            eps=self.rms_norm_eps,
        )

    def _replicate_kv(self, tensor, source_heads: int, repeats: int, head_dim=0):
        if tensor is None:
            return tensor
        shape = (
            tensor.shape[:head_dim]
            + (source_heads, tensor.shape[head_dim] // source_heads)
            + tensor.shape[head_dim + 1 :]
        )
        tensor = tensor.view(shape)
        tensor = torch.repeat_interleave(tensor, repeats=repeats, dim=head_dim)
        shape = (
            tensor.shape[:head_dim]
            + (tensor.shape[head_dim] * tensor.shape[head_dim + 1],)
            + tensor.shape[head_dim + 2 :]
        )
        return tensor.view(shape)

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        layer_prefix = ".".join(prefix.split(".")[:-1])
        model_state_dict[layer_prefix + ".wk.weight"] = self._replicate_kv(
            model_state_dict[layer_prefix + ".wk.weight"],
            self.org_n_kv_heads,
            self.replication_factor,
            0,
        )
        model_state_dict[layer_prefix + ".wv.weight"] = self._replicate_kv(
            model_state_dict[layer_prefix + ".wv.weight"],
            self.org_n_kv_heads,
            self.replication_factor,
            0,
        )

    def _get_full_row_masked_out_mask(
        self,
        attn_bias,
        negative_inf_value,
    ):
        """
        attn_bias should be a 4D tensor of shape [B, H, S1, S2]
        where B is the batch size, H is the number of heads,
        and S1/S2 are the sequence lengths. This returns
        a 4D tensor of shape [B, H, S1, 1] which stores boolean
        values which are 0 if the a full row in the last dimension
        contains negative infinity values, otherwise it's 1.
        """
        return (attn_bias != negative_inf_value).any(dim=-1).type_as(attn_bias)[..., None]

    def _pad_masks(
        self,
        all_masks: List[List[List[int]]],
        all_num_chunks: List[List[int]],
        total_len: int,
        max_num_chunks: int,
        device,
    ) -> torch.Tensor:
        inf_value = torch.finfo(self.torch_dtype).min

        bsz = len(all_masks)
        max_num_media = max([len(m) for m in all_masks])

        out_masks = torch.full(
            (bsz, total_len, max_num_media, max_num_chunks), inf_value, device=device
        )
        for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
            for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
                out_mask_a = out_masks[idx, :, mask_idx, :]
                ax = torch.arange(out_masks.shape[1], device=all_masks.device)
                out_mask_b = out_mask_a * ((ax < mask_elem[0]) + (ax > total_len)).reshape(-1, 1)
                ax2 = torch.arange(out_masks.shape[3], device=all_masks.device)
                out_mask_c = out_mask_a * (ax2 >= mask_num_chunks).reshape(1, -1)
                out_masks[idx, :, mask_idx, :] = torch.minimum(out_mask_b, out_mask_c)
        return out_masks

    def _get_xattn_mask(self, total_len, vision_tokens, vision_masks, num_chunks, has_image):
        cross_attention_masks = self._pad_masks(
            vision_masks,
            num_chunks,
            total_len,
            self.max_num_chunks,
            device=vision_tokens.device,
        )

        num_tokens = total_len
        assert vision_tokens is not None, "Vision tokens must be provided"
        vision_seqlen = vision_tokens.shape[3]
        assert (
            vision_tokens.shape[1] == cross_attention_masks.shape[2]
        ), f"Mismatch in number of images given and number of masks given {vision_tokens.shape} {cross_attention_masks.shape}"
        assert (
            vision_tokens.shape[2] == cross_attention_masks.shape[3]
        ), f"Vision tokens shape {vision_tokens.shape} mismatch with xattn shape {cross_attention_masks.shape}"
        assert (
            num_tokens == cross_attention_masks.shape[1]
        ), f"Mismatch in text sequence length and cross attention mask sequence length {num_tokens} {cross_attention_masks.shape}"
        _, _, _, num_image_tokens, image_token_dim = tuple(vision_tokens.shape)
        bsz, ntext, nimg, nchunks = cross_attention_masks.shape
        cross_attention_masks = (
            cross_attention_masks.repeat_interleave(vision_seqlen, dim=3)
            .view(bsz, ntext, -1)
            .unsqueeze(1)
        )
        full_text_row_masked_out_mask = (
            self._get_full_row_masked_out_mask(
                cross_attention_masks,
                torch.finfo(self.torch_dtype).min,
            )
            * has_image
        )

        cross_attention_masks *= full_text_row_masked_out_mask

        return (
            cross_attention_masks.to(self.torch_dtype),
            full_text_row_masked_out_mask.to(self.torch_dtype),
        )

    def _compute_xattn_kv(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        bsz = xattn_tokens.shape[0]
        xk = self.wk(xattn_tokens)
        xv = self.wv(xattn_tokens)
        _, seqlen_y, _ = xk.shape

        xk = xk.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)

        xk, xv = [tensor.transpose(1, 2) for tensor in (xk, xv)]

        xk = self.k_norm(xk)

        return xk.to(self.torch_dtype), xv.to(self.torch_dtype)

    def forward(
        self,
        x: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_key_value: Optional[Tuple[torch.Tensor]] = None,
        vision_mask: Optional[Tuple[torch.Tensor]] = None,
        num_chunks: Optional[Tuple[torch.Tensor]] = None,
        has_image: Optional[Tuple[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(x, self.sequence_dimension)

        xq = self.wq(x)
        bsz, seqlen, _ = x.shape

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = self.q_norm(xq)
        xq = xq.transpose(1, 2)
        if vision_key_value is None:
            bsz, nimg, nchunk, ntok, image_token_dim = tuple(vision_tokens.shape)

            xk, xv = self._compute_xattn_kv(vision_tokens.view(bsz, -1, image_token_dim))

        else:
            xk, xv = vision_key_value[0], vision_key_value[1]

        vision_key_value: Tuple[torch.Tensor, torch.Tensor] = (xk, xv)

        # repeat k/v heads if n_kv_heads < n_heads
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        if seqlen > 1:  # context encoding
            xattn_mask, full_text_row_masked_out_mask = self._get_xattn_mask(
                seqlen, vision_tokens, vision_mask, num_chunks, has_image.view(-1, 1, 1, 1)
            )
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=xattn_mask, dropout_p=0.0)
        else:
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=0.0)
            full_text_row_masked_out_mask = has_image.view(-1, 1, 1, 1)
        output = output * full_text_row_masked_out_mask
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
        out = self.wo(output)
        return out, full_text_row_masked_out_mask, vision_key_value


class NeuronLlamaCrossAttentionBlock(torch.nn.Module):
    def __init__(self, config: InferenceConfig, vision_config):
        """Cross attention layer with model-parallel attention layers."""
        super().__init__()
        self.config = config
        self.vision_config = vision_config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.rms_norm_eps = config.rms_norm_eps
        self.max_num_chunks = self.vision_config.max_num_tiles
        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1
        if parallel_state.model_parallel_is_initialized():
            self.model_parallel_size = parallel_state.get_tensor_model_parallel_size()
        else:
            self.model_parallel_size = 1

        self.replication_factor = 1
        if self.model_parallel_size > 8:
            self.replication_factor = self.model_parallel_size // 8

        self.n_kv_heads *= self.replication_factor

        assert self.n_heads % self.n_kv_heads == 0

        self.xatten = NeuronLlamaCrossAttention(config, self.vision_config)

        # cross attention transformer block layers
        self.attention_norm = get_rmsnorm_cls()(
            self.hidden_size,
            eps=self.rms_norm_eps,
        )
        self.gate_attn = torch.nn.Parameter(torch.zeros(1))

        self.feed_forward = NeuronLlamaMLP(config)
        self.ffn_norm = get_rmsnorm_cls()(
            self.hidden_size,
            eps=self.rms_norm_eps,
        )
        self.gate_ffwd = torch.nn.Parameter(torch.zeros(1))
        self.no_ffn = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_key_value: Optional[Tuple[torch.Tensor]] = None,
        vision_mask: Optional[Tuple[torch.Tensor]] = None,
        num_chunks: Optional[Tuple[torch.Tensor]] = None,
        has_image: Optional[Tuple[torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = hidden_states

        bsz, seqlen, _ = x.shape
        _attn_out, full_text_row_masked_out_mask, vision_key_value = self.xatten(
            x=self.attention_norm(x),
            vision_tokens=vision_tokens,
            vision_key_value=vision_key_value,
            vision_mask=vision_mask,
            num_chunks=num_chunks,
            has_image=has_image,
        )

        h = x + self.gate_attn.tanh() * _attn_out * has_image.view(-1, 1, 1)
        _ffn, _ = self.feed_forward(self.ffn_norm(h))

        if seqlen > 1:
            if self.sequence_parallel_enabled:
                full_text_row_masked_out_mask = _reduce_scatter_along_dim(
                    full_text_row_masked_out_mask, 2, xm.REDUCE_MAX
                )
            _ffn = full_text_row_masked_out_mask[:, 0] * _ffn
        h = h + self.gate_ffwd.tanh() * _ffn * float(not self.no_ffn) * has_image.view(-1, 1, 1)
        return h, vision_key_value


class NeuronLlamaAttentionBlock(torch.nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.self_attn = NeuronLlamaAttention(config=config)

        self.feed_forward = NeuronLlamaMLP(config)
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache)

        return outputs


class NeuronMllamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig, vision_config, is_xatten_layer: bool):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vision_config = vision_config
        self.xatten = (
            NeuronLlamaCrossAttentionBlock(config, self.vision_config)
            if (is_xatten_layer and not config.neuron_config.skip_vision)
            else DummyCrossAttention()
        )
        self.self_attn = NeuronLlamaAttentionBlock(config)
        self.is_xatten_layer = is_xatten_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_key_value: Optional[Tuple[torch.Tensor]] = None,
        vision_mask: Optional[Tuple[torch.Tensor]] = None,
        num_chunks: Optional[Tuple[torch.Tensor]] = None,
        has_image: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # apply cross attention
        hidden_states, new_vision_key_value = self.xatten(
            hidden_states, vision_tokens, vision_key_value, vision_mask, num_chunks, has_image
        )

        # apply regular attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        outputs = (hidden_states, present_key_value, new_vision_key_value, cos_cache, sin_cache)
        return outputs


class PartiallyTrainable_Embedding(torch.nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.replication_factor = self.tp_degree // 8
        self.tok_embeddings = ParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        self.learnable_embedding = nn.Embedding(
            8,
            self.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        self.num_frozen_embeddings = self.tok_embeddings.num_embeddings
        self._thresh = self.num_frozen_embeddings - 1

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        xz = torch.zeros_like(hidden_states, device=hidden_states.device)
        oz = torch.ones_like(hidden_states, device=hidden_states.device)
        x_orig = torch.minimum(
            hidden_states, torch.tensor(self._thresh, device=hidden_states.device)
        )
        x_new = (
            torch.maximum(
                hidden_states, torch.tensor(self._thresh + 1, device=hidden_states.device)
            )
            - self.num_frozen_embeddings
        )

        mask_orig = torch.where(hidden_states >= self.num_frozen_embeddings, xz, oz).unsqueeze(-1)
        mask_new = torch.where(hidden_states < self.num_frozen_embeddings, xz, oz).unsqueeze(-1)
        x_orig = self.tok_embeddings(x_orig)
        x_new = self.learnable_embedding(x_new).type_as(x_orig)
        return x_orig * mask_orig.type_as(x_orig) + x_new * mask_new.type_as(x_new)


class NeuronMllamaTextModel(NeuronBaseModel):
    """
    The neuron version of language model of the Llama Multimodal Model
    """

    def __init__(self, config: InferenceConfig, vision_config):
        self.vision_config = vision_config
        super().__init__(config)

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.num_cross_attention_layers = len(config.cross_attention_layers)
        if self.config.checkpoint == META_CHECKPOINT:
            self.num_hidden_layers = config.num_hidden_layers
        elif self.config.checkpoint == HF_CHECKPOINT:
            # neuron modeling includes cross attention in the same layer as self attention whereas
            # hf modeling uses different layer index for cross attention
            self.num_hidden_layers = config.num_hidden_layers - self.num_cross_attention_layers

    def _init_fusion_schedule(
        self,
        num_layers: int,
        num_xatten_layers: int,
    ):
        num_layers = list(range(num_layers))

        # uniformly spread the layers
        k = math.ceil(len(num_layers) / num_xatten_layers)
        return num_layers[::-1][::k][:num_xatten_layers][::-1]

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = PartiallyTrainable_Embedding(config)
            self.lm_head = ColumnParallelLinear(
                self.hidden_size,
                self.vocab_size,
                gather_output=False if self.on_device_sampling else True,
                bias=False,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                self.hidden_size,
                self.vocab_size,
                bias=False,
            )

        # calculate the layer ids that should have cross attention
        self.fusion_schedule = self._init_fusion_schedule(
            self.num_hidden_layers,
            self.num_cross_attention_layers,
        )

        if self.config.checkpoint == META_CHECKPOINT:
            # precompute rope scaling frequences to use in each NeuronLlamaAttention layer.
            self.rotary_freqs = precompute_freqs_cis(
                config.hidden_size // config.num_attention_heads,
                config.neuron_config.max_context_length * 2,
                config.rope_theta,
                True,
            ).to("xla")
        else:
            self.rotary_freqs = None

        self.layers = nn.ModuleList(
            [
                NeuronMllamaDecoderLayer(config, self.vision_config, (i in self.fusion_schedule))
                for i in range(self.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def init_inference_optimization(self, config: InferenceConfig):
        super().init_inference_optimization(config)
        # Use the MultimodalKVCacheManager that stores both text and vision kv cache for self and cross attn
        self.kv_mgr = MultimodalKVCacheManager(
            config, self.vision_config, self.num_hidden_layers, num_kv_head=self.num_key_value_heads
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        vision_tokens=None,
        vision_mask=None,
        num_chunks=None,
        has_image=None,
    ):
        """Override NeuronBaseModel.forward() to add vision_tokens input, and vision_key_values, updated_vision_cache"""
        is_for_context_encoding = 1 < input_ids.shape[-1] != self.speculation_length
        is_for_speculation = input_ids.shape[-1] == self.speculation_length

        cache_size = (
            get_cache_size(self.n_positions, self.num_cores_per_group)
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values, vision_key_values = None, None
        else:
            past_key_values, vision_key_values = self.kv_mgr.get_cache(self.n_positions)

        # Prepare attention mask(s)
        attention_mask = self.create_attn_mask(
            attention_mask,
            is_for_context_encoding,
            is_for_speculation,
        )
        active_mask = None

        # FD masks
        active_mask_2d = None
        if self.neuron_config.flash_decoding_enabled and not is_for_context_encoding:
            rank_id = self.rank_util.get_rank()
            active_mask_2d, attention_mask_2d = mask_util(
                pos_ids=position_ids,
                rank_id=rank_id,
                num_cores_per_group=self.num_cores_per_group,
                cache_size=cache_size,
            )
            active_mask = turn_2d_mask_to_4d(
                active_mask_2d, n_positions=1, batch_size=self.batch_size
            )
            attention_mask = turn_2d_mask_to_4d(
                attention_mask_2d, n_positions=cache_size, batch_size=self.batch_size
            )

        hidden_states, past_key_values, vision_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
            vision_tokens=vision_tokens,
            vision_key_values=vision_key_values,
            vision_mask=vision_mask,
            num_chunks=num_chunks,
            has_image=has_image,
        )

        updated_kv_cache = self.kv_mgr.update_cache(
            is_for_context_encoding=is_for_context_encoding,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=past_key_values,
            seq_len=cache_size,
            active_mask=active_mask_2d,
        )
        updated_vision_cache = self.kv_mgr.update_vision_cache(
            is_for_context_encoding, seq_ids, position_ids, vision_key_values, self.n_positions
        )
        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            # simple token generation
            index = torch.max(position_ids, dim=1, keepdim=True).indices
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        res = logits
        if self.on_device_sampling:
            # perform sampling on Neuron to get tokens
            res = self.sampler(logits[:, -1, :], sampling_params, rank_id=self.rank_util.get_rank())

            res = res.to(torch.int32)

        return [res] + updated_kv_cache + updated_vision_cache

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_key_values: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        num_chunks: Optional[torch.Tensor] = None,
        has_image: Optional[torch.Tensor] = None,
    ):
        """
        Override NeuronBaseModel.get_model_output() to add vision_tokens, vision_key_values inputs,
        replace cos_cache and sin_cache with vision_decoder_cache and rotary_freqs to use polar_compatible_rope
        """
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device  # noqa
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if self.sequence_parallel_enabled:
            # TODO: Replace this with rankid + scatter call once supported
            hidden_states = _reduce_scatter_along_dim(
                inputs_embeds,
                self.sequence_dimension,
                xm.REDUCE_MAX,
                process_group=get_tp_group(self.config),
            )
        else:
            hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = ()
        vision_decoder_cache = ()
        cos_cache = None
        sin_cache = None
        xtten_layer_idx = 0
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            vision_key_value = None
            if vision_key_values is not None and decoder_layer.is_xatten_layer:
                vision_key_value = vision_key_values[xtten_layer_idx]
                xtten_layer_idx += 1
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                vision_tokens=vision_tokens,
                vision_mask=vision_mask,
                vision_key_value=vision_key_value,
                num_chunks=num_chunks,
                has_image=has_image,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                rotary_freqs=(
                    self.rotary_freqs[position_ids] if self.rotary_freqs is not None else None
                ),
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            vision_decoder_cache += (layer_outputs[2],)
            cos_cache, sin_cache = layer_outputs[3:]

        if self.sequence_parallel_enabled:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        hidden_states = self.norm(hidden_states)

        return (hidden_states, next_decoder_cache, vision_decoder_cache)


class NeuronMllamaModel(NeuronBaseModel):
    def __init__(self, config: InferenceConfig):
        self.vision_config = config.vision_config
        self.text_config = config.text_config
        super().__init__(self.text_config)

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

        self.neuron_config = config.neuron_config

    def init_model(self, config: InferenceConfig):
        self.vision_model = NeuronMllamaVisionModel(self.text_config, self.vision_config)
        self.text_model = NeuronMllamaTextModel(self.text_config, self.vision_config)

    def init_inference_optimization(self, config: InferenceConfig):
        super().init_inference_optimization(config)
        # only need one kv cache mgr
        self.kv_mgr = self.text_model.kv_mgr

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        pixel_values,
        aspect_ratios,
        vision_mask,
        num_chunks,
        has_image,
    ):
        is_for_context_encoding = input_ids.shape[-1] > 1

        if is_for_context_encoding:
            # We mask the output with `has_image`
            # So vision_tokens becomes all zeros if input has no image,
            if self.neuron_config.skip_vision:
                vision_tokens = pixel_values
            else:
                # We mask the output with `has_image`
                # So vision_tokens becomes all zeros if input has no image,
                vision_tokens = self.vision_model(pixel_values, aspect_ratios) * has_image.view(
                    -1, 1, 1, 1, 1
                )
        else:
            vision_tokens = torch.tensor([0] * self.neuron_config.batch_size)

        self.text_model.n_positions = self.n_positions
        outputs = self.text_model(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            vision_tokens,
            vision_mask,
            num_chunks,
            has_image,
        )

        return outputs


class NeuronMllamaForCausalLM(NeuronBaseForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronMllamaModel

    @classmethod
    def get_config_cls(cls):
        return MllamaInferenceConfig

    @classmethod
    def get_neuron_config_cls(cls):
        return MultimodalVisionNeuronConfig

    @staticmethod
    def load_hf_model(model_path):
        raise Exception("HuggingFace checkpoint is not supported yet")

    def get_compiler_args(self) -> str:
        return "--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor=2 --vectorize-strided-dma' -O1 \
                --hbm-scratchpad-page-size=1024"

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        if inference_config.checkpoint == HF_CHECKPOINT:
            from .hf_state_dict_conversion import convert_hf_state_dict_to_neuron_state_dict

            return convert_hf_state_dict_to_neuron_state_dict(state_dict, inference_config)
        elif inference_config.checkpoint == META_CHECKPOINT:
            from .meta_state_dict_conversion import convert_meta_state_dict_to_neuron_state_dict

            return convert_meta_state_dict_to_neuron_state_dict(state_dict, inference_config)

    def get_model_wrapper_cls(self):
        return ModelWrapperMllama

    def _convert_input_dict_to_ordered_tuple(self, input_dict: Dict[str, Any]):
        """
        Utility function to convert input dictionary to ordered tuple
        based on input signature of _get_model_outputs
        """
        args = []
        ordered_keys = inspect.getfullargspec(self._get_model_outputs).args

        for key in ordered_keys:
            if key == "self":
                continue
            elif (key == "medusa_args" or key == "llava_args") and input_dict[key]:
                for custom_arg in input_dict[key]:
                    args.append(custom_arg)
            else:
                args.append(input_dict[key])

        return tuple(args)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adapter_ids: Optional[torch.FloatTensor] = None,
        medusa_args=None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        aspect_ratios: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        num_chunks: Optional[torch.Tensor] = None,
        has_image: Optional[torch.Tensor] = None,
        vision_key_values: Optional[List[torch.FloatTensor]] = None,
        llava_args: Optional[List] = [],
        input_capture_hook: Optional[Callable] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        if self.async_mode:
            # derive future cpu inputs from current cpu inputs
            if position_ids.shape[1] == input_ids.shape[1]:
                next_position_ids = torch.amax(position_ids, 1, keepdim=True)
            else:
                next_position_ids = position_ids

            next_position_ids = next_position_ids + 1
            next_attention_mask = self._infer_attention_mask(next_position_ids)
            self.next_cpu_inputs = {
                "attention_mask": next_attention_mask,
                "position_ids": next_position_ids,
            }

        sampling_params = (
            self.default_sampling_params if sampling_params is None else sampling_params
        )
        self.sampling_params = sampling_params

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(
            output_attentions, output_hidden_states, return_dict
        )

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(position_ids)

        self._log_input(input_ids, attention_mask, position_ids, seq_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        if self.async_mode:
            bs, _ = input_ids.shape
            outputs, is_run_on_neuron = self._get_model_outputs_async(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                adapter_ids=adapter_ids,
                medusa_args=medusa_args,
                llava_args=llava_args,
                pixel_values=pixel_values if input_ids.shape[-1] > 1 else torch.tensor([0] * bs),
                aspect_ratios=aspect_ratios,
                vision_mask=vision_mask,
                num_chunks=num_chunks,
                has_image=has_image,
                prev_hidden=None,
            )
        else:
            outputs, is_run_on_neuron = self._get_model_outputs(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                pixel_values,
                aspect_ratios,
                vision_mask,
                num_chunks,
                has_image,
            )

        generation_model = self.get_generation_model()
        if not generation_model.is_neuron():
            self._copy_past_key_values(outputs)

        if is_run_on_neuron:
            # When run on neuron, KV cache remains on device
            logits_or_next_tokens = outputs
        else:
            # When run on cpu, KV cache is returned which has to be ignored
            logits_or_next_tokens, *_ = outputs

        logging.debug("---output---")
        logging.debug(
            f"{'tokens' if self.on_device_sampling else 'logits'} = %s, ",
            logits_or_next_tokens,
        )

        return self._construct_output(logits_or_next_tokens)

    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        pixel_values,
        aspect_ratios,
        vision_mask,
        num_chunks,
        has_image,
    ):
        bs, _ = input_ids.shape
        if input_ids.shape[-1] > 1:  # context encoding
            outputs = self.context_encoding_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params[seq_ids],
                pixel_values,
                aspect_ratios,
                vision_mask,
                num_chunks,
                has_image,
            )
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()

        else:  # token generation
            if (
                self.next_cpu_inputs is not None and self.prior_outputs is not None
            ):  # this is never not None and not in async mode
                _input_ids = self.prior_outputs
                _attention_mask = self.next_cpu_inputs["attention_mask"]
                _position_ids = self.next_cpu_inputs["position_ids"]
            else:
                _input_ids = input_ids
                _attention_mask = attention_mask
                _position_ids = position_ids

            outputs = self.token_generation_model(
                _input_ids,
                _attention_mask,
                _position_ids,
                seq_ids,
                sampling_params,
                # Llama-MM specific
                torch.tensor([0] * bs),  # dummy pixel_values
                aspect_ratios,
                vision_mask,
                num_chunks,
                has_image,
            )

            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    def _copy_past_key_values(self, outputs):
        """
        Override NeuronBaseForCausalLM._copy_past_key_values() to add vision_tokens
        """
        super()._copy_past_key_values(outputs[:-1])
        self.token_generation_model.model.vision_tokens.data = outputs[-1]

    def get_required_kwargs(self) -> List[str]:
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return [
            "pixel_values",
            "aspect_ratios",
            "vision_key_values",
            "vision_mask",
            "num_chunks",
            "has_image",
        ]

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass
