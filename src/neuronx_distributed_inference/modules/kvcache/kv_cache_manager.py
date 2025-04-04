import logging
from typing import List

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
from neuronx_distributed_inference.modules.kvcache.utils import dynamic_update_slice, fill_prefix


def _reshape_tiled_cache(cache: Tensor):
    # We merge the tiles BHS(128 tiled)D -> BHSD
    cache_shape = cache.shape
    desired_shape = (
        cache_shape[0],
        cache_shape[1],
        cache_shape[2] * cache_shape[3],
        cache_shape[4],
    )
    cache = cache.reshape(desired_shape)
    return cache


def _slice_kv_cacheline(padding_side: str, seq_len: int, cache: Tensor):
    if padding_side == "right":
        return torch.ops.aten.slice(cache, dim=2, start=0, end=seq_len)
    max_idx = cache.shape[2]
    return torch.ops.aten.slice(cache, dim=2, start=max_idx - seq_len, end=max_idx)


def _gather_slice_into_kv_cacheline(cache, padding_side, seq_len: int, bucket_slice: Tensor):
    max_idx = cache.shape[2]
    if padding_side == "right":
        remaining = torch.ops.aten.slice(cache, dim=2, start=seq_len, end=max_idx)
        if remaining.dtype == torch.float8_e4m3fn:
            return ConcatenateOp.apply(bucket_slice, remaining, dim=2)
        return torch.cat([bucket_slice, remaining], dim=2)
    else:
        remaining = torch.ops.aten.slice(cache, dim=2, start=0, end=max_idx - seq_len)
        if remaining.dtype == torch.float8_e4m3fn:
            return ConcatenateOp.apply(bucket_slice, remaining, dim=2)
        return torch.cat([remaining, bucket_slice], dim=2)


class KVCacheManager(nn.Module):
    """
    Key Value Cache Management.
    It stores KV cache as a parameter list of the shape (batch_sz, num_kv_head_per_rank, max_len, head_dim),
    and vends out read and write operations.
    """

    def __init__(self, config: InferenceConfig, **kwargs):
        super().__init__()
        self.is_medusa = config.neuron_config.is_medusa
        self.num_medusa_heads = config.neuron_config.num_medusa_heads
        self.padding_side = config.neuron_config.padding_side
        self.is_continuous_batching = config.neuron_config.is_continuous_batching
        self.flash_decoding_enabled = config.neuron_config.flash_decoding_enabled
        self.num_cores_per_group = config.num_cores_per_group
        self.num_kv_head = kwargs["num_kv_head"]
        self.kv_cache_batch_size = config.neuron_config.kv_cache_batch_size
        self.kv_cache_padding_size = config.neuron_config.kv_cache_padding_size

        # NOTE: Tiling the sequence dimension of the KV cache enables specific compiler optimizations like cascaded reductions
        self.is_kv_cache_tiled = config.neuron_config.kv_cache_tiling
        self._init_kv_shape(config)
        self.quant = config.neuron_config.kv_cache_quant

        num_layer = config.num_hidden_layers
        dtype = config.neuron_config.torch_dtype
        if self.quant:
            self.quant_dtype = torch.float8_e4m3fn
            self.dequant_dtype = dtype
        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.kv_shape, dtype=dtype), requires_grad=False)
                for _ in range(num_layer * 2)
            ]
        )
        if self.quant:
            self.past_key_values = self.past_key_values.to(self.quant_dtype)

    def _get_num_kv_heads_per_rank(self, config: InferenceConfig):
        tp_degree = config.neuron_config.tp_degree
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

    def _init_kv_shape(self, config: InferenceConfig):
        max_batch_size = (
            config.neuron_config.kv_cache_batch_size + config.neuron_config.kv_cache_padding_size
        )
        max_len = config.neuron_config.max_length
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

        if self.is_kv_cache_tiled:
            num_tiles = int(max_len / 128)
            # KV cache layout : BHS(128 tiled)D
            self.kv_shape = (
                max_batch_size,
                num_kv_heads_per_rank,
                128,  # Sequence dim is tiled
                num_tiles,  # max_len = 128 * num_tiles
                hidden_dim_per_head,
            )
        else:
            # KV cache layout : BHSD
            self.kv_shape = (
                max_batch_size,
                num_kv_heads_per_rank,
                max_len,
                hidden_dim_per_head,
            )

    def _fetch_cache(self, idx: int, kvcache_buffer=None):
        if kvcache_buffer is not None:
            return kvcache_buffer[idx][0], kvcache_buffer[idx][1]
        k_cache, v_cache = self.past_key_values[idx * 2], self.past_key_values[idx * 2 + 1]
        if self.is_kv_cache_tiled:
            return _reshape_tiled_cache(k_cache), _reshape_tiled_cache(v_cache)
        return k_cache, v_cache

    def configure_medusa_gather_slice_idx(self, metadata):
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

    def get_kv_by_layer_id(self, key_layer_idx, gather_index=None, slice_index=None):
        k_cache = self.past_key_values[key_layer_idx]
        v_cache = self.past_key_values[key_layer_idx + 1]
        if self.kv_cache_padding_size > 0:
            k_cache = k_cache[: -self.kv_cache_padding_size]
            v_cache = v_cache[: -self.kv_cache_padding_size]
        if self.is_medusa:
            accepted_k_cache = torch.gather(input=k_cache, dim=2, index=gather_index)
            accepted_v_cache = torch.gather(input=v_cache, dim=2, index=gather_index)
            k_cache = torch.scatter(input=k_cache, dim=2, index=slice_index, src=accepted_k_cache)
            v_cache = torch.scatter(input=v_cache, dim=2, index=slice_index, src=accepted_v_cache)
        return k_cache, v_cache

    def get_cache(self, seq_len: int, skip_slice=False, **kwargs):
        """
        Return network (all layers)'s previously cached K and V, up to seq_len.

        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
        :param skip_slice: whether to skip slicing the KV cache to the seq_len
        :return: list of tuple of (K, V)
        """
        slice_index, gather_index = None, None
        if self.is_medusa:
            assert (
                "medusa_metadata" in kwargs
            ), "medusa_metadata should be specified for medusa decoding!"
            medusa_metadata = kwargs["medusa_metadata"]
            slice_index, gather_index = self.configure_medusa_gather_slice_idx(medusa_metadata)
        past_key_values = []
        for key_layer_idx in range(0, len(self.past_key_values), 2):
            # get kv per layer
            k_cache, v_cache = self.get_kv_by_layer_id(
                key_layer_idx, gather_index=gather_index, slice_index=slice_index
            )

            if self.is_kv_cache_tiled:
                k_cache = _reshape_tiled_cache(k_cache)
                v_cache = _reshape_tiled_cache(v_cache)

            # slice for partial view
            if not skip_slice:
                k_cache = _slice_kv_cacheline(self.padding_side, seq_len, k_cache)
                v_cache = _slice_kv_cacheline(self.padding_side, seq_len, v_cache)

            if self.quant:
                k_cache = dequantize.direct_cast_dequantize(k_cache, self.dequant_dtype)
                v_cache = dequantize.direct_cast_dequantize(v_cache, self.dequant_dtype)
            past_key_values.append([k_cache, v_cache])
        return past_key_values

    def _get_latest_kv(self, kv_per_layer):
        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]

        if self.quant:
            latest_k = quantize.direct_cast_quantize(latest_k, self.quant_dtype)
            latest_v = quantize.direct_cast_quantize(latest_v, self.quant_dtype)
        return latest_k, latest_v

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        new_key_values: List[Tensor],
        seq_len: int,
        scatter_index=None,
        active_mask=None,
        kvcache_buffer=None,
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
            latest_k, latest_v = self._get_latest_kv(kv_per_layer)
            k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)

            if is_for_context_encoding:
                if self.is_continuous_batching:
                    assert (
                        seq_ids.dim() == 1 and seq_ids.shape[0] == 1
                    ), "only supports single seq_id"

                    cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)

                    indices = torch.zeros(k_cache.dim(), dtype=seq_ids.dtype, device=seq_ids.device)
                    indices = indices.scatter(
                        dim=0,
                        index=torch.tensor([0], dtype=torch.int64, device=k_cache.device),
                        src=cache_idx,
                    ).to(torch.int32)

                    indices = indices.split(1)
                    indices = [t.squeeze() for t in indices]
                    k_cache = dynamic_update_slice(k_cache, latest_k, indices)
                    v_cache = dynamic_update_slice(v_cache, latest_v, indices)
                else:
                    k_cache = fill_prefix(k_cache, latest_k)
                    v_cache = fill_prefix(v_cache, latest_v)
            else:
                if self.padding_side == "left":
                    k_cache = k_cache[:, :, 1:, :]
                    v_cache = v_cache[:, :, 1:, :]
                    k_cache = torch.cat([k_cache, latest_k], dim=2)
                    v_cache = torch.cat([v_cache, latest_v], dim=2)
                else:
                    # copy the tensor of the new position into kv cache
                    if self.flash_decoding_enabled:
                        assert (
                            active_mask is not None
                        ), "active_mask should be specified for flash decoding!"
                        garbage_pos = seq_len - 1  # treat last pos as garbage
                        updated_pos_ids = position_ids // self.num_cores_per_group
                        scatter_index = torch.where(active_mask == 1, updated_pos_ids, garbage_pos)
                        scatter_index_new = scatter_index.view(
                            -1, 1, scatter_index.shape[-1], 1
                        ).expand_as(latest_k)
                    else:
                        scatter_index_new = self._get_index_to_update_new_position(
                            scatter_index, position_ids, latest_k
                        )
                    k_cache = torch.scatter(
                        input=k_cache, dim=2, index=scatter_index_new, src=latest_k
                    )
                    v_cache = torch.scatter(
                        input=v_cache, dim=2, index=scatter_index_new, src=latest_v
                    )

            # Retiling
            # TODO once compiler fixes CR 158191111 we can turn back output tiling on
            # k_cache = k_cache.view(cache_shape)
            # v_cache = v_cache.view(cache_shape)

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        # return updated kv cache to NxD runtime
        return updated_kv_cache

    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k):
        if self.is_medusa:
            scatter_index = scatter_index.view(-1, 1, scatter_index.shape[-1], 1).expand_as(full_k)
        else:
            scatter_index = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(full_k)
        return scatter_index

    def get_cache_update_index_for_seq_ids(self, seq_ids):
        """
        Override this method to map seq_id to cache index.

        By default, seq_ids map directly to cache_idx in batch dimension
        """
        if self.kv_cache_padding_size > 0:
            # handle out-of-bound seq_ids
            garbase_pos = self.kv_cache_batch_size - 1  # last position
            seq_ids = torch.where(seq_ids < self.kv_cache_batch_size, seq_ids, garbase_pos)
        return seq_ids
