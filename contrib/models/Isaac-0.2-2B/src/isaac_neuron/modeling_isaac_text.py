# Copyright 2025 © Amazon.com and Affiliates
"""Isaac text model for NxDI: Qwen3 decoder layers adapted for VLM.

Isaac's text backbone is a standard Qwen3 model (28 layers, 2048 hidden, GQA 16/8 heads).
This module wraps Qwen3 decoder layers in the NeuronBaseModel VLM pattern, supporting:
- Vision embedding injection via scatter_by_index_put
- Standard NxDI KV cache management
- On-device sampling
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.parallel_layers.mappings import _gather_along_dim
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
    QKNormPlacement,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import (
    get_cache_size,
    mask_util,
    turn_2d_mask_to_4d,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    Sampler,
    mask_padded_logits,
)
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KVCacheManager,
)
from neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager import (
    generate_tokengen_slot_mapping,
)
from neuronx_distributed_inference.modules.custom_calls import neuron_cumsum
from neuronx_distributed_inference.utils.distributed import get_tp_group

# Use HF Qwen3RMSNorm for CPU, CustomRMSNorm for Neuron
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

logger = logging.getLogger("Neuron")


def get_rmsnorm_cls():
    """Return appropriate RMSNorm class based on execution mode."""
    return Qwen3RMSNorm if cpu_mode() else CustomRMSNorm


class NeuronIsaacAttention(NeuronAttentionBase):
    """Isaac attention: standard Qwen3 GQA with QK normalization.

    Qwen3 applies QK norm BEFORE RoPE (pre-rope), same as NxDI built-in Qwen3.
    Config: 16 attention heads, 8 KV heads, head_dim=128, rope_theta=1M
    """

    def __init__(self, config: InferenceConfig):
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rotary_emb = RotaryEmbedding(
            dim=head_dim,
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
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            rms_norm_eps=config.rms_norm_eps,
            qk_norm_placement=QKNormPlacement.PRE_ROPE,
            q_layernorm=get_rmsnorm_cls()(
                hidden_size=head_dim, eps=config.rms_norm_eps
            ),
            k_layernorm=get_rmsnorm_cls()(
                hidden_size=head_dim, eps=config.rms_norm_eps
            ),
        )


class NeuronIsaacDecoderLayer(nn.Module):
    """Isaac decoder layer: Qwen3 architecture (RMSNorm -> Attn -> RMSNorm -> MLP).

    Identical to NeuronQwen3DecoderLayer from NxDI built-in, but adapted
    for the VLM text model pattern.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size

        self.self_attn = NeuronIsaacAttention(config)
        self.mlp = NeuronLlamaMLP(config)  # Qwen3 MLP is compatible with LlamaMLP

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Kernel enablement flags
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.quantized_mlp_kernel_enabled = (
            config.neuron_config.quantized_mlp_kernel_enabled
        )
        self.rmsnorm_quantize_kernel_enabled = (
            config.neuron_config.rmsnorm_quantize_kernel_enabled
        )
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled

        # Fused rmsnorm only when sequence parallelism is disabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.mlp_kernel_fused_rmsnorm = not self.sequence_parallel_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states

        # QKV kernel fusion with RMSNorm
        if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
            qkv_fused_rmsnorm = self.input_layernorm
        else:
            hidden_states = self.input_layernorm(hidden_states)
            qkv_fused_rmsnorm = None

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = attn_output.hidden_states

        # First residual
        hidden_states = residual + hidden_states
        residual = hidden_states

        # MLP kernel fusion with RMSNorm
        if self.mlp_kernel_enabled and self.mlp_kernel_fused_rmsnorm:
            mlp_fused_rmsnorm = self.post_attention_layernorm
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_fused_rmsnorm = None

        hidden_states, _ = self.mlp(
            hidden_states,
            rmsnorm=mlp_fused_rmsnorm,
            adapter_ids=adapter_ids,
        )

        # Second residual
        hidden_states = residual + hidden_states

        return (
            hidden_states,
            attn_output.present_key_value,
            attn_output.cos_cache,
            attn_output.sin_cache,
            None,  # residual (not used for Qwen3)
        )


class NeuronIsaacTextModel(NeuronBaseModel):
    """Isaac text model for VLM: Qwen3 decoder with vision embedding injection.

    Follows the same pattern as NeuronGemma3TextModel:
    - Inherits from NeuronBaseModel
    - Uses scatter_by_index_put for vision token injection
    - Manages KV cache and on-device sampling
    """

    def scatter_by_index_put(self, h_image, encoded_patches_proj, positions):
        """Scatter vision embeddings into the input embedding sequence.

        Args:
            h_image: (B, max_positions, hidden_dim) - text input embeddings
            encoded_patches_proj: (num_patches, patch_size, hidden_dim) - vision embeddings
            positions: (B, num_positions, 1) - scatter positions

        Returns:
            Updated h_image with vision embeddings scattered in.
        """
        B, max_positions, embedding_dim = h_image.shape
        h_image_new = h_image.clone()
        encoded_patches_flat = encoded_patches_proj.view(-1, embedding_dim)
        positions = positions.view(-1)

        num_updates_per_batch = positions.shape[0] // B
        batch_idx = torch.arange(B, device=h_image.device, dtype=positions.dtype)
        batch_idx = batch_idx.repeat_interleave(num_updates_per_batch)

        h_image_new.index_put_(
            (batch_idx.long(), positions.long()),
            encoded_patches_flat,
            accumulate=False,
        )
        return h_image_new

    def encode_vision_to_input(
        self, inputs_embeds, vision_embeddings, vision_mask
    ) -> torch.Tensor:
        """Inject vision embeddings into text input embeddings."""
        return self.scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    def setup_attr_for_model(self, config: InferenceConfig):
        """Set up model attributes needed for inference."""
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.is_chunked_prefill = config.neuron_config.is_chunked_prefill

    def init_model(self, config: InferenceConfig):
        """Initialize the Qwen3 text model components."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding layer
        if parallel_state_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=get_tp_group(config),
            )

            lm_head_pad = config.neuron_config.lm_head_pad
            lnc = config.neuron_config.logical_nc_config
            lm_head_pad_alignment_size = (
                config.neuron_config.lm_head_pad_alignment_size * lnc
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=lm_head_pad,
                pad=True,
                pad_alignment_size_per_rank=lm_head_pad_alignment_size
                if lm_head_pad
                else 1,
                keep_padded_output=lm_head_pad,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            from transformers.models.qwen3.modeling_qwen3 import (
                Qwen3RMSNorm as HFQwen3RMSNorm,
            )

            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronIsaacDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final norm
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def init_inference_optimization(self, config: InferenceConfig):
        """Initialize KV cache and sampling for inference."""
        super().init_inference_optimization(config)

        if self.on_device_sampling:
            self.sampler = Sampler(config.neuron_config)

        self.kv_mgr = KVCacheManager(
            config,
            num_kv_head=self.num_key_value_heads,
            global_rank=self.rank_util,
        )

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
        """Forward pass for Isaac text model with vision support.

        This follows NeuronBaseModel.forward() pattern with vision embedding injection.
        The 25 positional arguments match ImageToTextModelWrapper's expected interface.
        """
        # Handle optional empty tensors
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

        is_for_token_gen = attention_mask.dim() == 4
        is_for_context_encoding = self._is_context_encoding(input_ids)
        is_for_speculation = self._is_for_speculation(input_ids)

        # For non-speculative prefix caching, generate the slot mapping
        if (
            not is_for_context_encoding
            and not self.neuron_config.enable_fused_speculation
            and not self.neuron_config.enable_eagle_speculation
            and self.is_prefix_caching
            and active_block_table is not None
        ):
            block_size = torch.tensor(
                self.neuron_config.pa_block_size,
                device=position_ids.device,
                dtype=torch.int32,
            )
            slot_mapping = generate_tokengen_slot_mapping(
                position_ids, slot_mapping, active_block_table, block_size
            )

        cache_size = (
            get_cache_size(
                self.n_positions, self.num_cores_per_group, is_for_context_encoding
            )
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )

        # Prepare attention mask
        if self.is_chunked_prefill:
            attn_mask = self.create_attn_mask(
                attention_mask,
                is_for_context_encoding,
                is_for_speculation,
                query_lens=num_queries,
                key_lens=num_queries + computed_context_lens,
            )
        else:
            attn_mask = self.create_attn_mask(
                attention_mask,
                is_for_context_encoding,
                is_for_speculation,
                position_ids=position_ids,
            )

        active_mask = None
        if self.is_prefix_caching:
            active_length = (
                self.speculation_length if is_for_speculation else self.n_active_tokens
            )
            active_mask = torch.full(
                (active_length, active_length),
                True,
                device=attention_mask.device,
            ).tril(diagonal=0)
            active_mask = active_mask[None, None, :, :].expand(
                self.batch_size, 1, active_length, active_length
            )
        if is_for_speculation:
            active_mask = torch.full(
                (self.speculation_length, self.speculation_length),
                True,
                device=attention_mask.device,
            ).tril(diagonal=0)
            active_mask = active_mask[None, None, :, :].expand(
                self.batch_size, 1, self.speculation_length, self.speculation_length
            )

        # FlashDecoding masks
        active_mask_2d = None
        if self.neuron_config.flash_decoding_enabled and not is_for_context_encoding:
            rank_id = self.rank_util.get_rank()
            active_mask_tmp, attention_mask_tmp = mask_util(
                pos_ids=position_ids,
                rank_id=rank_id,
                num_cores_per_group=self.num_cores_per_group,
                cache_size=cache_size,
            )
            if is_for_speculation:
                active_mask = active_mask_tmp[:, None, :, :].expand(
                    self.batch_size, 1, -1, -1
                )
                attn_mask = attention_mask_tmp[:, None, :, :].expand(
                    self.batch_size, 1, -1, -1
                )
                active_mask_2d = active_mask_tmp.sum(dim=-2, keepdims=False).to(
                    torch.bool
                )
            else:
                active_mask = turn_2d_mask_to_4d(
                    active_mask_tmp, n_positions=1, batch_size=self.batch_size
                )
                attn_mask = turn_2d_mask_to_4d(
                    attention_mask_tmp,
                    n_positions=cache_size,
                    batch_size=self.batch_size,
                )
                active_mask_2d = active_mask_tmp

        # Context encoding or token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            past_key_values = self.kv_mgr.get_cache(self.n_positions)

        hidden_states, updated_kv_cache = self.get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            prev_hidden=prev_hidden,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            num_queries=num_queries,
            is_for_context_encoding=is_for_context_encoding,
            scatter_index=slot_mapping if self.is_block_kv_layout else scatter_index,
            kvcache_buffer=kv_cache,
            is_for_speculation=is_for_speculation,
            active_block_table=active_block_table,
            kv_active_mask=active_mask_2d,
            update_cache=True,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

        batch_size = input_ids.shape[0]
        if not self.sliced_hidden:
            if self.padding_side == "left":
                index = torch.tensor(
                    [hidden_states.shape[1] - 1], device=hidden_states.device
                )
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)
            elif self.is_chunked_prefill:
                if is_for_context_encoding:
                    index = neuron_cumsum(num_queries.reshape(1, -1).float()).int() - 1
                    index = index.reshape(1, -1, 1)
                    index = index.expand(batch_size, -1, self.hidden_size)
                    hidden_states = torch.gather(hidden_states, dim=1, index=index)
            else:
                if not (
                    position_ids.shape[-1] == self.speculation_length
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
            logits = mask_padded_logits(
                logits, rank_id, world_size, pad_size=self.lm_head.pad_size
            )

        if self.on_device_sampling:
            res = self._sample_on_device(
                logits, sampling_params, is_for_speculation, is_for_context_encoding
            )
        else:
            res = logits

        # Ensure active_block_table and attention_mask not optimized away for prefix caching
        if self.is_prefix_caching:
            if active_block_table is not None and len(active_block_table.shape) == 1:
                res = res + active_block_table[0] * 0
            if attention_mask is not None and self.prefix_size == 0:
                res = res + attention_mask[0] * 0

        outputs = [res]
        if self.neuron_config.output_logits:
            logits = _gather_along_dim(
                logits,
                partition_dim=2,
                process_group=get_tp_group(self.config),
            )
            outputs += [logits]
        outputs += updated_kv_cache

        return outputs


def parallel_state_initialized():
    """Check if parallel state is initialized."""
    from neuronx_distributed.parallel_layers import parallel_state

    return parallel_state.model_parallel_is_initialized()
