import logging
from typing import List, Optional, Tuple

import torch
from neuronx_distributed.parallel_layers import parallel_state, utils
from neuronx_distributed.quantization import dequantize, quantize
from torch import Tensor, nn
from torch_neuronx.xla_impl.ops import ConcatenateOp

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402; noqa: E402; noqa: E402
    determine_sharding_strategy,
    get_shardable_head_counts,
)
from neuronx_distributed_inference.modules.flashdecode.utils import get_cache_size
from neuronx_distributed_inference.modules.kvcache.utils import dynamic_update_slice, update_cache_const_indices, fill_prefix, get_kv_shapes

from neuronx_distributed_inference.modules.attention.utils import get_kv_head_indices_context_parallel_full_tp_decode, get_kv_head_indices_context_parallel_dp_decode, get_cp8_tp8_rank_ordering
from neuronx_distributed_inference.modules.attention.attention_process_groups import get_tp_cp_group_mesh
from neuronx_distributed_inference.utils.distributed import split_along_dim, get_dp_rank


def untile_cache(cache: Tensor, transposed: bool):
    """
    If transposed flag is True, K-tensor is stored in BHD(128-tiled)S format and we untile it into BHDS format.
    Otherwise tensor is untiled into BHSD. `transposed` flag is False for V tensor.
    """
    if transposed:
        batch_size, head_dim, dim_per_head, tile_size, seq_len = cache.shape
        desired_shape = (
            batch_size,
            head_dim,
            dim_per_head,
            tile_size * seq_len,
        )
    else:
        batch_size, head_dim, tile_size, seq_len, dim_per_head = cache.shape
        desired_shape = (
            batch_size,
            head_dim,
            tile_size * seq_len,
            dim_per_head,
        )

    cache = cache.reshape(desired_shape)
    return cache


def tile_cache(cache: Tensor, transposed: bool):
    """
    If the transposed flag is true, this indicates that the K tensor is stored in BHDS.
    The tiling is done on the S dimension. So if transposed=true, we tile it as BHD(128 tiled)S.
    `transposed` flag is False for V tensor.
    """
    if transposed:
        batch_size, head_dim, dim_per_head, seq_len = cache.shape
        desired_shape = (
            batch_size,
            head_dim,
            dim_per_head,
            128,
            seq_len // 128,
        )
    else:
        batch_size, head_dim, seq_len, dim_per_head = cache.shape
        desired_shape = (
            batch_size,
            head_dim,
            128,
            seq_len // 128,
            dim_per_head,
        )
    cache = cache.view(desired_shape)
    return cache


def _slice_kv_cacheline(padding_side: str, seq_len: int, cache: Tensor, transposed: bool):
    seqlen_dim = 3 if transposed else 2
    if padding_side == "right":
        return torch.ops.aten.slice(cache, dim=seqlen_dim, start=0, end=seq_len)
    max_idx = cache.shape[seqlen_dim]
    return torch.ops.aten.slice(cache, dim=seqlen_dim, start=max_idx - seq_len, end=max_idx)


def _gather_slice_into_kv_cacheline(cache, padding_side, seq_len: int, bucket_slice: Tensor, transposed: bool):
    seqlen_dim = 3 if transposed else 2
    max_idx = cache.shape[seqlen_dim]
    if padding_side == "right":
        remaining = torch.ops.aten.slice(cache, dim=seqlen_dim, start=seq_len, end=max_idx)
        if remaining.dtype == torch.float8_e4m3fn:
            return ConcatenateOp.apply(bucket_slice, remaining, dim=seqlen_dim)
        return torch.cat([bucket_slice, remaining], dim=seqlen_dim)
    else:
        remaining = torch.ops.aten.slice(cache, dim=seqlen_dim, start=0, end=max_idx - seq_len)
        if remaining.dtype == torch.float8_e4m3fn:
            return ConcatenateOp.apply(bucket_slice, remaining, dim=seqlen_dim)
        return torch.cat([remaining, bucket_slice], dim=seqlen_dim)


class KVCacheManager(nn.Module):
    """
    Key Value Cache Management.
    It stores KV cache as a parameter list of the shape (batch_sz, num_kv_head_per_rank, max_len, head_dim),
    and vends out read and write operations.
    """

    def __init__(self, config: InferenceConfig, num_kv_head, global_rank=None, attention_chunk_size=None, sliding_window=None, layer_to_cache_size_mapping=None, **kwargs):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.is_medusa = config.neuron_config.is_medusa
        self.num_medusa_heads = config.neuron_config.num_medusa_heads
        self.padding_side = config.neuron_config.padding_side
        self.is_continuous_batching = config.neuron_config.is_continuous_batching
        self.flash_decoding_enabled = config.neuron_config.flash_decoding_enabled
        self.num_cores_per_group = config.num_cores_per_group
        self.num_kv_head = num_kv_head
        self.kv_cache_batch_size = config.neuron_config.kv_cache_batch_size
        self.kv_cache_padding_size = config.neuron_config.kv_cache_padding_size
        self.batch_size = config.neuron_config.batch_size
        self.padding_side = config.neuron_config.padding_side
        self.k_cache_transposed = config.neuron_config.k_cache_transposed
        self.global_rank = global_rank
        self.attention_chunk_size = attention_chunk_size
        self.sliding_window = sliding_window

        # NOTE: Tiling the sequence dimension of the KV cache enables specific compiler optimizations like cascaded reductions
        self.is_kv_cache_tiled = config.neuron_config.kv_cache_tiling
        self._init_kv_shape(config, layer_to_cache_size_mapping)
        self.quant = config.neuron_config.kv_cache_quant

        num_layer = config.num_hidden_layers
        dtype = config.neuron_config.attention_dtype if config.neuron_config.attention_dtype is not None else config.neuron_config.torch_dtype
        if self.quant:
            self.quant_dtype = torch.float8_e4m3fn
            self.dequant_dtype = dtype

        if layer_to_cache_size_mapping:
            self.past_key_values = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(k_or_v_shape, dtype=dtype), requires_grad=False)
                    for layer_idx in range(num_layer) for k_or_v_shape in [self.k_shapes[layer_idx], self.v_shapes[layer_idx]]
                ]
            )
        else:
            self.past_key_values = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(k_or_v_shape, dtype=dtype), requires_grad=False)
                    for _ in range(num_layer) for k_or_v_shape in [self.k_shape, self.v_shape]
                ]
            )
        if self.quant:
            self.past_key_values = self.past_key_values.to(self.quant_dtype)

    def _get_num_kv_heads_per_rank(self, config: InferenceConfig):
        tp_degree = config.neuron_config.tp_degree
        dp_degree = config.neuron_config.attention_dp_degree

        if dp_degree > 1:
            tp_degree = tp_degree // dp_degree

        num_kv_head = self.num_kv_head
        num_atten_head = config.num_attention_heads

        gqa_sharding_strategy = determine_sharding_strategy(tp_degree, num_kv_head)
        _, num_key_value_heads = get_shardable_head_counts(
            tp_degree, num_atten_head, num_kv_head, gqa_sharding_strategy
        )

        if parallel_state.model_parallel_is_initialized():
            num_kv_heads_per_rank = utils.divide(num_key_value_heads, tp_degree)
        else:
            num_kv_heads_per_rank = num_key_value_heads

        return num_kv_heads_per_rank

    def _get_hidden_dim_per_head(self, config: InferenceConfig):
        hidden_size = config.hidden_size
        num_atten_head = config.num_attention_heads
        hidden_dim_per_head = getattr(config, "head_dim", hidden_size // num_atten_head)
        return hidden_dim_per_head

    def _init_kv_shape(self, config: InferenceConfig, layer_to_cache_size_mapping: Optional[List[int]] = None):
        max_batch_size = (
            config.neuron_config.kv_cache_batch_size + config.neuron_config.kv_cache_padding_size
        )
        max_len = config.neuron_config.max_length
        if self.attention_chunk_size and self.attention_chunk_size < max_len and not layer_to_cache_size_mapping:
            logging.warning('initializing chunk-size kv cache for all layers')
            max_len = self.attention_chunk_size
        elif self.sliding_window:
            max_len = self.sliding_window
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)
        hidden_dim_per_head = self._get_hidden_dim_per_head(config)

        if self.flash_decoding_enabled:
            padded_max_len = max_len
            if max_len % self.num_cores_per_group != 0:
                padded_max_len += self.num_cores_per_group - max_len % self.num_cores_per_group
                logging.warning(
                    f"Max length needs to be multiples of num_cores_per_group {self.num_cores_per_group}"
                    f" but got {max_len}. Padding it to {padded_max_len} meet the requirement."
                )
            max_len = get_cache_size(padded_max_len, self.num_cores_per_group)

        self.max_len = max_len
        if layer_to_cache_size_mapping:
            self.k_shapes = []
            self.v_shapes = []
            for cache_len in layer_to_cache_size_mapping:
                k_shape, v_shape = get_kv_shapes(cache_len, max_batch_size,
                                                 num_kv_heads_per_rank, hidden_dim_per_head,
                                                 self.k_cache_transposed, self.is_kv_cache_tiled)
                self.k_shapes.append(k_shape)
                self.v_shapes.append(v_shape)
        else:
            k_shape, v_shape = get_kv_shapes(max_len, max_batch_size,
                                             num_kv_heads_per_rank, hidden_dim_per_head,
                                             self.k_cache_transposed, self.is_kv_cache_tiled)
            self.k_shape = k_shape
            self.v_shape = v_shape

    def _fetch_cache(self, idx: int, kvcache_buffer=None):
        if kvcache_buffer is not None:
            if (
                len(kvcache_buffer) == len(self.past_key_values) // 2
                and len(kvcache_buffer[0]) == 2
            ):
                k_cache = kvcache_buffer[idx][0]
                v_cache = kvcache_buffer[idx][1]
            elif len(kvcache_buffer) == len(self.past_key_values):
                k_cache = kvcache_buffer[2 * idx]
                v_cache = kvcache_buffer[2 * idx + 1]
            else:
                raise ValueError(
                    f"Received kvcache_buffer has length {len(kvcache_buffer)}"
                    f"kvcache_buffer must be a list of 2 element tuples of length {len(self.past_key_values) // 2}"
                    f"or a flat list of length {len(self.past_key_values)}"
                )
        else:
            k_cache = self.past_key_values[2 * idx]
            v_cache = self.past_key_values[2 * idx + 1]

        if self.is_kv_cache_tiled:
            k_cache = untile_cache(cache=k_cache, transposed=self.k_cache_transposed)
            v_cache = untile_cache(cache=v_cache, transposed=False)

        return k_cache, v_cache

    def configure_medusa_gather_slice_idx(self, metadata):
        assert not self.k_cache_transposed, 'Transposed K cache not yet implemented for medusa.'
        assert (
            "current_length" in metadata and "accepted_indices" in metadata
        ), "current_length and accepted_indices should be specified for medusa decoding!"

        current_length = metadata["current_length"]
        accepted_indices = metadata["accepted_indices"]
        slice_index = current_length.view(-1, 1, current_length.shape[-1], 1).expand_as(
            self.past_key_values[0][:, :, 0 : self.num_medusa_heads + 1, :]
        )
        gather_index = accepted_indices.view(-1, 1, accepted_indices.shape[-1], 1).expand_as(
            self.past_key_values[0][:, :, 0 : self.num_medusa_heads + 1, :]
        )
        return slice_index, gather_index

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

        if hasattr(self, "v_shapes"):
            seq_len = self.v_shapes[idx][2]
        # slice for partial view
        if not skip_slice:
            k_cache = _slice_kv_cacheline(self.padding_side, seq_len, k_cache, self.k_cache_transposed)
            v_cache = _slice_kv_cacheline(self.padding_side, seq_len, v_cache, False)
        if self.quant:
            k_cache = dequantize.direct_cast_dequantize(k_cache, self.dequant_dtype)
            v_cache = dequantize.direct_cast_dequantize(v_cache, self.dequant_dtype)
        return k_cache, v_cache

    def get_cache(
        self, seq_len: int, skip_slice=False, kvcache_buffer=None, seq_ids=None, **kwargs
    ):
        """
        Return network (all layers)'s previously cached K and V, up to seq_len.

        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
        :param skip_slice: whether to skip slicing the KV cache to the seq_len
        :return: list of tuple of (K, V)
        """
        past_key_values = []
        for idx in range(len(self.past_key_values) // 2):
            # get kv per layer
            k_cache, v_cache = self.get_kv_by_layer_id(
                idx=idx,
                skip_slice=skip_slice,
                seq_len=seq_len,
                kvcache_buffer=kvcache_buffer,
                seq_ids=seq_ids,
                **kwargs,
            )
            past_key_values.append([k_cache, v_cache])
        return past_key_values

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        new_key_values: List[Tensor],
        seq_len: int,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        **kwargs,
    ):
        """
        Given the passed-in new_key_values, update the cache

        :param is_for_context_encoding: bool
        :param seq_ids: tensor of size (batch_sz)
        :param position_ids: tensor of size (batch_sz, bucket_sz)
        :param new_key_values: list of tuple, the latest kv obtained at the end of the network from forward pass
        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
        :param scatter_index: tensor representing index to update
        :param active_mask: tensor representing index to update
        :param kvcache_buffer: if passed key states are updates to this buffer.
               kvcache_buffer is 2D list where, 1st dim for layer and the second denotes K and V.
               For example,
                    kvcache_buffer[1][0] is the K cache of the 1st layer
                    kvcache_buffer[4][1] is the V cache of the 4th layer
        :return: list of tuple of (K, V)
        """

        updated_kv_cache = []

        for idx, kv_per_layer in enumerate(new_key_values):
            k_cache, v_cache = self.update_kv_by_layer_id(
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                kv_per_layer=kv_per_layer,
                seq_len=seq_len,
                scatter_index=scatter_index,
                kv_active_mask=kv_active_mask,
                kvcache_buffer=kvcache_buffer
            )

            # If is_kv_cache_tiled=True, we store the KV cache in a sequence tiled layout in the HBM.
            # This tiling functions as a hint for the compiler. The torch level logic is not dependent on the layout,
            # so we keep just the storage in tiled layout and the compute is performed in the non tiled layout.
            # Here, before we update the cache which is in non-tiled layout, we tile it along sequence
            # so we can write it back to the tiled buffer.
            if self.is_kv_cache_tiled:
                k_cache = tile_cache(k_cache, self.k_cache_transposed)
                v_cache = tile_cache(v_cache, False)

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        # return updated kv cache to NxD runtime
        return updated_kv_cache

    def update_kv_by_layer_id(
        self,
        idx,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        kv_per_layer: Tuple[Tensor, Tensor],
        seq_len: int,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        **kwargs,
    ):
        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]
        if self.quant:
            latest_k = quantize.direct_cast_quantize(latest_k, self.quant_dtype)
            latest_v = quantize.direct_cast_quantize(latest_v, self.quant_dtype)

        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)

        cte_rank_ordering = None
        if self.neuron_config.cp_degree == 8 and self.neuron_config.tp_degree // self.neuron_config.cp_degree == 8:
            cte_rank_ordering = get_cp8_tp8_rank_ordering(self.neuron_config.tp_degree, self.neuron_config.cp_degree)

        if not is_for_context_encoding and self.neuron_config.attention_dp_degree > 1:
            dp_rank = get_dp_rank(self.get_rank(device=seq_ids.device), self.neuron_config.tp_degree // self.neuron_config.attention_dp_degree, self.neuron_config.attention_dp_degree)
            seq_ids = split_along_dim(seq_ids, dim=0, rank=dp_rank, num_partitions=self.neuron_config.attention_dp_degree)
            position_ids = split_along_dim(position_ids, dim=0, rank=dp_rank, num_partitions=self.neuron_config.attention_dp_degree)

        if is_for_context_encoding:
            if self.neuron_config.cp_degree > 1 and self.neuron_config.cp_degree != self.neuron_config.attention_dp_degree:
                # When we run CP without DP, decode will run in full TP, selectively write the heads that are used in decode
                rank = self.get_rank(device=seq_ids.device)
                if self.neuron_config.attention_dp_degree == 1:
                    kv_head_indices = get_kv_head_indices_context_parallel_full_tp_decode(self.num_kv_head, self.neuron_config.tp_degree, self.neuron_config.cp_degree, k_cache.device, cte_rank_ordering)
                else:
                    decode_ordering = None
                    if self.neuron_config.attention_dp_degree == 8 and self.neuron_config.tp_degree // self.neuron_config.attention_dp_degree == 8:
                        decode_ordering = sum(get_tp_cp_group_mesh(self.neuron_config.tp_degree, self.neuron_config.attention_dp_degree), [])

                    kv_head_indices = get_kv_head_indices_context_parallel_dp_decode(self.num_kv_head, self.neuron_config.tp_degree,
                                                                                     self.neuron_config.cp_degree, self.neuron_config.attention_dp_degree,
                                                                                     k_cache.device, cte_rank_ordering=cte_rank_ordering,
                                                                                     decode_rank_ordering=decode_ordering)
                head_idx = torch.index_select(kv_head_indices, dim=0, index=rank)
                latest_k = torch.index_select(latest_k, dim=1, index=head_idx)
                latest_v = torch.index_select(latest_v, dim=1, index=head_idx)

            if self.is_continuous_batching:
                assert seq_ids.dim() == 1 and seq_ids.shape[0] == 1, "only supports single seq_id"
                if self.neuron_config.k_cache_transposed or self.neuron_config.attention_dp_degree > 1:
                    cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
                    indices = [cache_idx] + [torch.zeros(1, device=seq_ids.device) for _ in range(k_cache.dim() - 1)]
                    indices = [t.squeeze().to(torch.int32) for t in indices]
                    k_cache = dynamic_update_slice(k_cache, latest_k, indices)
                    v_cache = dynamic_update_slice(v_cache, latest_v, indices)
                else:
                    k_cache = update_cache_const_indices(k_cache, latest_k, seq_ids)
                    v_cache = update_cache_const_indices(v_cache, latest_v, seq_ids)
            else:
                k_cache = fill_prefix(k_cache, latest_k)
                v_cache = fill_prefix(v_cache, latest_v)
        else:
            if self.padding_side == "left":
                assert not self.k_cache_transposed, 'Transposed K cache not yet implemented for left padding_side'
                k_cache = k_cache[:, :, 1:, :]
                v_cache = v_cache[:, :, 1:, :]
                k_cache = torch.cat([k_cache, latest_k], dim=2)
                v_cache = torch.cat([v_cache, latest_v], dim=2)
            else:
                # copy the tensor of the new position into kv cache
                if self.flash_decoding_enabled:
                    assert (
                        not self.k_cache_transposed
                    ), "Transposed K cache not yet implemented for flash decoding."
                    assert (
                        kv_active_mask is not None
                    ), "active_mask should be specified for flash decoding!"
                    garbage_pos = seq_len - 1  # treat last pos as garbage
                    updated_pos_ids = position_ids // self.num_cores_per_group
                    scatter_index = torch.where(kv_active_mask == 1, updated_pos_ids, garbage_pos)
                    scatter_index_new_k = scatter_index.view(
                        -1, 1, scatter_index.shape[-1], 1
                    ).expand_as(latest_k)
                    scatter_index_new_v = scatter_index_new_k
                ###############################################################################
                # Handles the case where the batch size is smaller than the KV cache batch size.
                ###############################################################################
                elif self.batch_size < self.kv_cache_batch_size:
                    assert not self.k_cache_transposed, 'Transposed K cache not yet implemented for batch_size < kv_cache_batch_size'
                    garbage_pos = seq_len - 1
                    updated_latest_kv_shape = k_cache.shape[:1] + latest_k.shape[1:]
                    cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
                    scatter_index = torch.full(
                        (
                            self.kv_cache_batch_size + self.kv_cache_padding_size,
                            position_ids.shape[-1],
                        ),
                        garbage_pos,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
                    scatter_index[cache_idx] = position_ids
                    scatter_index_new_k = (
                        scatter_index.view(-1, 1, scatter_index.shape[-1], 1)
                        .expand(updated_latest_kv_shape)
                        .to(torch.long)
                    )
                    scatter_index_new_v = scatter_index_new_k
                    # Update latest_k and latest_v with dummy values for non-active sequences.
                    updated_latest_k = torch.zeros(updated_latest_kv_shape).to(
                        dtype=latest_k.dtype, device=latest_k.device
                    )
                    updated_latest_v = torch.zeros(updated_latest_kv_shape).to(
                        dtype=latest_v.dtype, device=latest_v.device
                    )
                    updated_latest_k[cache_idx], updated_latest_v[cache_idx] = (
                        latest_k,
                        latest_v,
                    )
                    latest_k, latest_v = updated_latest_k, updated_latest_v
                else:
                    if self.config.neuron_config.apply_seq_ids_mask:
                        seq_ids_mask = torch.ge(seq_ids, torch.full_like(seq_ids, 0))
                        seq_ids_mask = seq_ids_mask.reshape(-1, 1).broadcast_to(position_ids.shape)
                        padded_pos_id = torch.full_like(position_ids, self.max_len - 1)
                        position_ids = torch.where(seq_ids_mask, position_ids, padded_pos_id)

                    scatter_index_new_k = self._get_index_to_update_new_position(
                        scatter_index, position_ids, latest_k, self.k_cache_transposed, idx
                    )
                    scatter_index_new_v = self._get_index_to_update_new_position(
                        scatter_index, position_ids, latest_v, False, idx
                    )
                k_cache = torch.scatter(
                    input=k_cache,
                    dim=(2 if not self.k_cache_transposed else 3),
                    index=scatter_index_new_k,
                    src=latest_k,
                )
                v_cache = torch.scatter(
                    input=v_cache, dim=2, index=scatter_index_new_v, src=latest_v
                )
        return k_cache, v_cache

    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int):
        if self.attention_chunk_size:
            if hasattr(self, "v_shapes"):
                cache_len = self.v_shapes[layer_idx][2]
            else:
                cache_len = self.attention_chunk_size
            position_ids = position_ids % cache_len
        elif self.sliding_window:
            position_ids = position_ids % self.sliding_window
        index = scatter_index if self.is_medusa else position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)

    def get_cache_update_index_for_seq_ids(self, seq_ids):
        """
        Override this method to map seq_id to cache index.

        By default, seq_ids map directly to cache_idx in batch dimension
        """
        if self.kv_cache_padding_size > 0:
            # handle out-of-bound seq_ids
            garbage_pos = self.kv_cache_batch_size + self.kv_cache_padding_size - 1  # last position
            seq_ids = torch.where(seq_ids < self.kv_cache_batch_size, seq_ids, garbage_pos)
        return seq_ids

    def get_rank(self, device=torch.device("cpu")):
        rank = self.global_rank.get_rank()
        if self.neuron_config.attention_dp_degree == 8 and self.neuron_config.tp_degree // self.neuron_config.attention_dp_degree == 8:
            rank_ordering = get_cp8_tp8_rank_ordering(self.neuron_config.tp_degree, self.neuron_config.attention_dp_degree, device=device)
            return torch.index_select(rank_ordering, dim=0, index=rank)

        return rank
