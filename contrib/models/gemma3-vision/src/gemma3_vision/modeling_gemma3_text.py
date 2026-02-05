import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch_neuronx.xla_impl.ops import RmsNorm
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding, Gemma3RMSNorm

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.parallel_layers.mappings import _gather_along_dim
from neuronx_distributed.quantization import dequantize
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.attention_process_groups import (
    get_flattened_inverted_tp_cp_group_mesh
)
from neuronx_distributed_inference.modules.attention.utils import (
    chunk_and_reorder_tensor,
    RotaryEmbedding,
    stride_tensor,
)
from neuronx_distributed_inference.modules.custom_calls import neuron_cumsum
from neuronx_distributed_inference.modules.flashdecode.utils import (
    get_cache_size,
    mask_util,
    turn_2d_mask_to_4d,
)
from neuronx_distributed_inference.modules.generation.sampling import Sampler, mask_padded_logits
from neuronx_distributed_inference.modules.kvcache.utils import get_layer_to_kv_cache_size_mapping_for_mixed_attn
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager, _slice_kv_cacheline
from neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager import generate_tokengen_slot_mapping
from neuronx_distributed_inference.utils.distributed import get_tp_group

logger = logging.getLogger("Neuron")


class HybridAttnKVCacheManager(KVCacheManager):

    def get_kv_by_layer_id(
        self,
        idx,
        seq_len: int,
        skip_slice=False,
        medusa_metadata=None,
        kvcache_buffer=None,
        seq_ids=None,
        is_for_speculation: bool = False,
        **kwargs,
    ):
        """
        Override KVCacheManager's get_kv_by_layer_id() to handle hybrid attention patterns.

        Changes:
        1. Removed the following lines:
            ```
            if hasattr(self, "v_shapes"):
                seq_len = self.v_shapes[idx][2]
            ```

        Without this override, get_kv_by_layer_id() would return caches with shape
        [batch_size, num_head_per_rank, max_seq_len, head_dim] instead of the expected
        [batch_size, num_head_per_rank, n_positions (bucket length), head_dim].
        """
        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)
        if (
            self.neuron_config.batch_size != self.neuron_config.max_batch_size
            and is_for_speculation
        ):
            assert seq_ids is not None
            updated_seq_ids = self.get_cache_update_index_for_seq_ids(seq_ids)
            k_cache = k_cache[updated_seq_ids]
            v_cache = v_cache[updated_seq_ids]
        elif self.kv_cache_padding_size > 0:
            k_cache = k_cache[: -self.kv_cache_padding_size]
            v_cache = v_cache[: -self.kv_cache_padding_size]
        if self.is_medusa:
            slice_index, gather_index = self.configure_medusa_gather_slice_idx(medusa_metadata)
            accepted_k_cache = torch.gather(input=k_cache, dim=3 if self.k_cache_transposed else 2, index=gather_index)
            accepted_v_cache = torch.gather(input=v_cache, dim=2, index=gather_index)
            k_cache = torch.scatter(input=k_cache, dim=3 if self.k_cache_transposed else 2, index=slice_index, src=accepted_k_cache)
            v_cache = torch.scatter(input=v_cache, dim=2, index=slice_index, src=accepted_v_cache)

        attn_kernel_enabled = (
            self.neuron_config.attn_tkg_builtin_kernel_enabled
            or self.neuron_config.attn_tkg_nki_kernel_enabled
            or self.neuron_config.attn_block_tkg_nki_kernel_enabled
        )
        if attn_kernel_enabled:  # Attention TKG Kernels do not need slicing.
            skip_slice = True

        # slice for partial view
        if not skip_slice:
            k_cache = _slice_kv_cacheline(self.padding_side, seq_len, k_cache, self.k_cache_transposed)
            v_cache = _slice_kv_cacheline(self.padding_side, seq_len, v_cache, False)
        if self.quant:
            k_cache = dequantize.direct_cast_dequantize(k_cache, self.dequant_dtype)
            v_cache = dequantize.direct_cast_dequantize(v_cache, self.dequant_dtype)
        return k_cache, v_cache


class NeuronGemma3RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states, original_dtype = hidden_states.to(torch.float32), hidden_states.dtype
        gamma = (1.0 + self.weight).to(torch.float32)
        y = RmsNorm.apply(hidden_states, gamma, self.eps, hidden_states.dim() - 1)
        return y.to(original_dtype)


def get_rmsnorm_cls():
    return Gemma3RMSNorm if cpu_mode() else NeuronGemma3RMSNorm


class NeuronGemma3TextScaledWordEmbedding(ParallelEmbedding):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int,
                 embed_scale: float = 1.0,
                 **kwargs) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, **kwargs)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class NeuronGemma3MLP(NeuronLlamaMLP):
    pass


class NeuronGemma3RotaryEmbedding(RotaryEmbedding):

    def __init__(self,
                 dim: int,
                 max_position_embeddings: int,
                 base: float,
                 scaling_type: str = "default",
                 scaling_factor: float = 1.0,
                 ) -> None:
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base
        )

        self.scaling_type = scaling_type
        if self.scaling_type == "default":
            self.scaling_factor = 1.0
        elif self.scaling_type == "linear":
            self.scaling_factor = scaling_factor
        else:
            raise ValueError(
                f"Unsupported RoPE scaling type '{scaling_type}'. Gemma3 RoPE only supports 'default' or 'linear'."
            )

    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        inv_freq = super().get_inv_freqs(device=device)
        if self.scaling_type == "linear":
            return inv_freq / self.scaling_factor
        return inv_freq


class NeuronGemma3Attention(NeuronAttentionBase):

    @staticmethod
    def get_rope(config: InferenceConfig, is_swa_layer: bool) -> NeuronGemma3RotaryEmbedding:
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(config.head_dim * partial_rotary_factor)
        max_position_embeddings = config.max_position_embeddings
        if is_swa_layer:
            # RoPE for SWA layers
            return NeuronGemma3RotaryEmbedding(
                dim=dim,
                max_position_embeddings=max_position_embeddings,
                base=config.rope_local_base_freq,
            )
        else:
            # RoPE for global attention layers
            if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                scaling_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
                scaling_factor = config.rope_scaling.get("factor", 1.0)
            else:
                scaling_type = "default"
                scaling_factor = 1.0
            return NeuronGemma3RotaryEmbedding(
                dim=dim,
                max_position_embeddings=max_position_embeddings,
                base=config.rope_theta,
                scaling_type=scaling_type,
                scaling_factor=scaling_factor,
            )


class NeuronGemma3DecoderLayer(nn.Module):

    def __init__(self, config: InferenceConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        config_sliding_window = getattr(config, "sliding_window", None)
        self.is_swa_layer = False if config_sliding_window is None else bool((layer_idx + 1) % config._sliding_window_pattern)
        self.sliding_window = config_sliding_window if self.is_swa_layer else None

        rms_norm_cls = get_rmsnorm_cls()
        rms_norm_eps = getattr(config, "rms_norm_eps", None)
        q_norm = rms_norm_cls(config.head_dim, rms_norm_eps) if rms_norm_eps else rms_norm_cls(config.head_dim)
        k_norm = rms_norm_cls(config.head_dim, rms_norm_eps) if rms_norm_eps else rms_norm_cls(config.head_dim)

        self.self_attn = NeuronGemma3Attention(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rotary_emb=NeuronGemma3Attention.get_rope(config=config, is_swa_layer=self.is_swa_layer),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            num_cores_per_group=config.num_cores_per_group,
            tensor_model_parallel_group=get_tp_group(config),
            sliding_window=self.sliding_window,
            use_qk_norm=False,
            q_layernorm=q_norm,
            k_layernorm=k_norm
        )

        self.mlp = NeuronGemma3MLP(config)
        self.input_layernorm = None
        if (
            not config.neuron_config.is_eagle_draft
            or config.neuron_config.enable_eagle_draft_input_norm
        ):
            self.input_layernorm = rms_norm_cls(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        self.post_attention_layernorm = rms_norm_cls(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = rms_norm_cls(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = rms_norm_cls(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.quantized_mlp_kernel_enabled = config.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = config.neuron_config.rmsnorm_quantize_kernel_enabled
        self.mlp_kernel_fuse_residual_add = config.neuron_config.mlp_kernel_fuse_residual_add
        self.qkv_kernel_fuse_residual_add = config.neuron_config.qkv_kernel_fuse_residual_add
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.is_prefill_stage = config.neuron_config.is_prefill_stage

        if self.is_prefill_stage and self.config.neuron_config.is_mlp_quantized():
            # for CTE, quantized MLP kernel does not support fused rmsnorm
            self.mlp_kernel_fused_rmsnorm = False
        else:
            self.mlp_kernel_fused_rmsnorm = not self.sequence_parallel_enabled

        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        local_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        adapter_ids=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        residual: Optional[torch.FloatTensor] = None,  # residual from previous layer if QKV kernel with fused residual is enabled
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], Optional[torch.FloatTensor], Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        # Adapted from NeuronLlamaDecoderLayer
        is_token_gen = past_key_value is not None
        entry_hidden_states = hidden_states

        # Hybrid SWA/global attention layers are specific to Gemma3
        if self.is_swa_layer:
            attention_mask = local_mask

        if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
            attn_fused_rmsnorm = self.input_layernorm
        else:
            hidden_states = self.input_layernorm(hidden_states)
            attn_fused_rmsnorm = None

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=attn_fused_rmsnorm,
            rotary_position_ids=rotary_position_ids,
            residual=residual,
            **kwargs,
        )

        # Post-attention RMS norm is specific to Gemma3
        hidden_states = self.post_attention_layernorm(attn_output.hidden_states)

        if attn_output.residual is not None:
        # In the case the QKV kernel is enabled (attn_output.residual is not None), the input hidden
        # states actually do not correspond to the attention layer's inputs. They are computed within
        # the layer (by the fused QKV kernel) and returned as "residual" output.
            assert self.qkv_kernel_fuse_residual_add, \
                "residual add before qkv should be computed in the previous layer, \
                unless qkv_kernel_fuse_residual_add is specified"
            assert (
                not self.sequence_parallel_enabled
            ), "qkv_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            assert (
                self.qkv_kernel_enabled
            ), "qkv_kernel_fuse_residual_add should be used with qkv_kernel_enabled"
            assert (
                not is_token_gen
            ), "cannot fuse residual add for tokengen"
            residual = attn_output.residual
        else:
            residual = entry_hidden_states  # attention layer inputs to be used for residuals addition

        if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
            assert (
                not self.sequence_parallel_enabled
            ), "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            hidden_states, residual = self.mlp(
                hidden_states,
                rmsnorm=self.pre_feedforward_layernorm,
                residual=residual,
                adapter_ids=adapter_ids,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states

            if self.mlp_kernel_enabled and self.mlp_kernel_fused_rmsnorm:
                mlp_fused_rmsnorm = self.pre_feedforward_layernorm
            else:
                hidden_states = self.pre_feedforward_layernorm(hidden_states)
                mlp_fused_rmsnorm = None

            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=mlp_fused_rmsnorm,
                adapter_ids=adapter_ids,
            )

        # Post-feed-forward RMS norm is specific to Gemma3
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        # If the QKV kernel with fused residual addition is not enabled, we perform the residual addition here,
        # otherwise, we return the residual so the fused kernel in the next block can perform the addition
        if not self.qkv_kernel_fuse_residual_add or is_token_gen:
            hidden_states = residual + hidden_states
            residual = None

        return  (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, residual)


class NeuronGemma3TextModel(NeuronBaseModel):

    def scatter_by_index_put(self, h_image, encoded_patches_proj, positions):
        """
        Scatter encoded patches into an image tensor.
        Compared to neuronx_distributed_inference/models/llama4/utils/encoder_utils.py's scatter_by_index_put(),
        this function supports Batch Size >= 1.

        Args:
        h_image (torch.Tensor): The target image tensor of shape (B, max_positions, embedding_dim)
        encoded_patches_proj (torch.Tensor): The encoded patches to be scattered, of shape (num_patches, patch_size, embedding_dim)
        positions (torch.Tensor): The positions where patches should be scattered, of shape (B, num_positions, 1)

        Returns:
        torch.Tensor: The updated image tensor with scattered patches
        """
        B, max_positions, embedding_dim = h_image.shape

        # Create a new tensor instead of modifying h_image in-place
        h_image_new = h_image.clone()

        # Flatten encoded_patches_proj
        encoded_patches_flat = encoded_patches_proj.view(-1, embedding_dim)

        # Flatten positions
        positions = positions.view(-1)

        # Create Batch Indices
        # We need to tell PyTorch: "This update belongs to batch 0, that one to batch 1"
        # If positions is (B, N), we need batch_idx to look like [0,0..0, 1,1..1, ...]
        num_updates_per_batch = positions.shape[0] // B

        batch_idx = torch.arange(B, device=h_image.device, dtype=positions.dtype)
        batch_idx = batch_idx.repeat_interleave(num_updates_per_batch)

        # Use index_put_ to scatter the embeddings
        h_image_new.index_put_(
            (batch_idx.long(), positions.long()),
            encoded_patches_flat,
            accumulate=False
        )

        return h_image_new

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask) -> torch.Tensor:
        # Concat vision and text embeddings during context encoding
        # Both inputs_embeds and vision_embeddings should be of the same shape: [BS, Total tokens (image + text), Hidden]
        # And vision_mask should of the shape [BS, Total tokens (image + text), 1]
        # Entries in vision_mask with value `True` represent vision tokens and with value `False` represent text tokens
        # For text-only inputs, vision_mask should be all `False`
        return self.scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        """
        Modified init_model of NeuronLlama4TextModel:
        1. add self.sliding_window. This will allow creating local attention masks in forward()
        2. replace embedding modules with 'scaled' embeddings"""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.sliding_window = config.sliding_window

        if self.sliding_window and config.neuron_config.seq_len < self.sliding_window:
            # When the model context (seq_len) is shorter than the window, the sliding window
            # effectively covers the entire sequence (full attention). Update to match.
            config.sliding_window = config.neuron_config.seq_len
            self.sliding_window = config.sliding_window

        if self.sliding_window:
            is_layer_locals = [layer_idx % config._sliding_window_pattern != config._sliding_window_pattern - 1 for layer_idx in range(config.num_hidden_layers)]
            self.layer_to_cache_size_mapping = get_layer_to_kv_cache_size_mapping_for_mixed_attn(config.sliding_window, config.neuron_config.seq_len, is_layer_locals)
            logger.info("layer_to_cache_size_mapping initialized")

        self.has_mixed_attn = True

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = NeuronGemma3TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                config.hidden_size**0.5, # embed_scale
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )

            lm_head_pad = config.neuron_config.lm_head_pad
            lnc = config.neuron_config.logical_nc_config
            lm_head_pad_alignment_size = config.neuron_config.lm_head_pad_alignment_size * lnc
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=lm_head_pad,
                pad=True,
                pad_alignment_size_per_rank=lm_head_pad_alignment_size if lm_head_pad else 1,
                keep_padded_output=lm_head_pad,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = Gemma3TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                config.hidden_size**0.5 # embed_scale
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        self.layers = nn.ModuleList(
            [NeuronGemma3DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )

        if not config.neuron_config.is_eagle_draft:
            self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            self.fc = ColumnParallelLinear(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True
            )

        # TODO: medusa needed?
        # self.is_medusa = config.neuron_config.is_medusa
        # self.num_medusa_heads = config.neuron_config.num_medusa_heads
        # self.medusa_speculation_length = config.neuron_config.medusa_speculation_length

        # if self.is_medusa:
        #     if parallel_state.model_parallel_is_initialized():
        #         medusa_head_cls = ColumnParallelLinear
        #     else:
        #         medusa_head_cls = nn.Linear
        #     for i in range(self.num_medusa_heads):
        #         medusa_head = nn.Sequential(
        #             *([ResBlock(config.hidden_size)] * 1),
        #             medusa_head_cls(
        #                 config.hidden_size,
        #                 config.vocab_size,
        #                 gather_output=not self.on_device_sampling,
        #                 bias=False,
        #             ),
        #         )
        #         setattr(self, f"medusa_head_{i}", medusa_head)

    def init_inference_optimization(self, config: InferenceConfig):
        """
        Compared to neuronx_distributed_inference/models/model_base.py's init_inference_optimization(),
        use HybridAttnKVCacheManager instead of KVCacheManager
        """
        super().init_inference_optimization(config)

        if self.on_device_sampling:
            self.sampler = Sampler(config.neuron_config)

        self.kv_mgr = HybridAttnKVCacheManager(
            config,
            num_kv_head=self.num_key_value_heads,
            global_rank=self.rank_util,
            sliding_window=self.sliding_window,
            layer_to_cache_size_mapping=self.layer_to_cache_size_mapping)

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
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
    ):
        """
        Compared to NxDI NeuronBaseModel.forward(),
        1. pass 'past_key_values' to get_model_output
        2. always create local attention mask (for sliding window attn layers)
        """
        # Optional argument cannot be set to None in NXDI now as NxD does not support
        # kwargs. Now we are working around by passing an empty tensor.
        #
        # But empty tensors break the logic like
        #     if input_embeds is None:
        #         input_embeds = embed()
        #
        # We are forced to pass in a value for optional params
        # Passing in none does not work as it breaks torchscripting.
        # Once kwargs support is in, we can remove this workaround.
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
        local_attn_mask = None

        if self.neuron_config.is_medusa:
            return self._medusa_forward(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                adapter_ids,
                accepted_indices,
                current_length,
                medusa_mask,
                scatter_index,
            )

        is_for_token_gen = attention_mask.dim() == 4

        if (
            is_for_token_gen
            and self.neuron_config.enable_token_tree
            and self.neuron_config.enable_eagle_speculation
        ):
            logging.warning("entering _eagle_token_tree_forward")
            return self._eagle_token_tree_forward(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                scatter_index=scatter_index,
                inputs_embeds=inputs_embeds,
                kv_cache=kv_cache,
                active_mask=active_mask,
                rotary_position_id=rotary_position_id,
            )
        # TODO: This will not work for a context encoding model with bucket size
        # equal to the speculation length
        is_for_context_encoding = self._is_context_encoding(input_ids)
        is_for_speculation = self._is_for_speculation(input_ids)

        # For non-speculative prefix caching, generate the slot mapping within the traced model.
        # This is necessary for async mode, as the active_block_table is up-to-date but the slot mapping
        # passed into the traced model may be from a prior iteration.
        if (
            not is_for_context_encoding
            and not self.neuron_config.enable_fused_speculation
            and not self.neuron_config.enable_eagle_speculation
            and self.is_prefix_caching
            and active_block_table is not None
        ):
            block_size = torch.tensor(self.neuron_config.pa_block_size, device=position_ids.device, dtype=torch.int32)
            slot_mapping = generate_tokengen_slot_mapping(position_ids, slot_mapping, active_block_table, block_size)

        cache_size = (
            get_cache_size(self.n_positions, self.num_cores_per_group, is_for_context_encoding)
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )

        # Prepare attention mask(s)
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
            if self.attention_chunk_size:
                if is_for_context_encoding:
                    local_attn_mask = self._create_chunked_attn_mask_cte(attention_mask, self.attention_chunk_size)
                else:
                    local_attn_mask = self._create_chunked_attn_mask_tkg(attention_mask, self.attention_chunk_size, position_ids)
            elif self.sliding_window:
                if is_for_context_encoding:
                    local_attn_mask = self._create_windowed_attn_mask_cte(attention_mask, self.sliding_window)
                else:
                    local_attn_mask = self._create_windowed_attn_mask_tkg(attention_mask, self.sliding_window, position_ids)

        active_mask = None
        if self.is_prefix_caching:
            active_length = self.speculation_length if is_for_speculation else self.n_active_tokens
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

        # FlashDecoding masks, for KV cache updates
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
                active_mask = active_mask_tmp[:, None, :, :].expand(self.batch_size, 1, -1, -1)
                attn_mask = attention_mask_tmp[:, None, :, :].expand(self.batch_size, 1, -1, -1)
                # only for cache udpate
                active_mask_2d = active_mask_tmp.sum(dim=-2, keepdims=False).to(torch.bool)
            else:
                active_mask = turn_2d_mask_to_4d(
                    active_mask_tmp, n_positions=1, batch_size=self.batch_size
                )
                attn_mask = turn_2d_mask_to_4d(
                    attention_mask_tmp, n_positions=cache_size, batch_size=self.batch_size
                )
                active_mask_2d = active_mask_tmp

        if self.neuron_config.strided_context_parallel_kernel_enabled and is_for_context_encoding:
            logging.debug("strided_context_parallel_kernel_enabled enabled, shuffling inputs")

            # The strided CP FA kernel expected inputs to be strided, due to SP happening in model_base
            # stride here rather than in attention to order it before we move the inputs to SP region
            input_ids = stride_tensor(input_ids, 1, self.neuron_config.cp_degree)
            position_ids = stride_tensor(position_ids, 1, self.neuron_config.cp_degree)

        # When using SP with 8x8 CP, the mesh is non-contiguous, so we reorder the input to have a non-contiguous SP split
        # When we AG in attention using 8x8, the resulting sequence is contiguous
        if is_for_context_encoding and self.neuron_config.cp_degree > 1 and self.neuron_config.cp_degree == 8 and (self.neuron_config.tp_degree // self.neuron_config.cp_degree) == 8 and self.sequence_parallel_enabled:
            ordering = get_flattened_inverted_tp_cp_group_mesh(self.neuron_config.tp_degree, self.neuron_config.cp_degree)

            logging.debug("CP8 and SP enabled, reordering the input on S", ordering)
            input_ids = chunk_and_reorder_tensor(input_ids, ordering, 1)

        # It is either for context encoding or for token generation
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
            local_attn_mask=local_attn_mask,
        )

        batch_size = input_ids.shape[0]
        if not self.sliced_hidden:
            if self.padding_side == "left":
                index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)
            elif self.is_chunked_prefill:
                if is_for_context_encoding:
                    # chunked prefill will return cp_config.max_num_seqs, not
                    # just the last one
                    index = neuron_cumsum(num_queries.reshape(1, -1).float()).int() - 1
                    index = index.reshape(1, -1, 1)
                    index = index.expand(batch_size, -1, self.hidden_size)
                    hidden_states = torch.gather(hidden_states, dim=1, index=index)
            else:
                if not (
                    position_ids.shape[-1] == self.speculation_length or position_ids.shape[-1] == 1
                ):
                    # context encoding
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
            logits = mask_padded_logits(logits, rank_id, world_size, pad_size=self.lm_head.pad_size)

        if self.on_device_sampling:
            res = self._sample_on_device(
                logits, sampling_params, is_for_speculation, is_for_context_encoding
            )
        else:
            res = logits

        # A hack to ensure active_block_table and attention_mask is not optimized away
        # if not None for prefix caching flow.
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

        if self.neuron_config.enable_eagle_speculation:
            if is_for_context_encoding:
                outputs = outputs + [hidden_states] + [self.full_hidden_states]
            else:
                outputs = outputs + [self.full_hidden_states]

        return outputs
