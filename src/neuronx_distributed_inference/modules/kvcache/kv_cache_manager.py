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
from neuronx_distributed_inference.modules.autobucketing import slice_lhs, slice_rhs


def _slice_kv_cacheline(padding_side, seq_len: int, cache: Tensor):
    if padding_side == "right":
        return slice_lhs(cache, seq_len, 2)
    else:
        max_idx = cache.shape[2]
        return slice_rhs(cache, seq_len, max_idx, 2)


def _gather_slice_into_kv_cacheline(cache, padding_side, seq_len: int, bucket_slice: Tensor):
    max_idx = cache.shape[2]
    if padding_side == "right":
        remaining = slice_rhs(cache, max_idx - seq_len, max_idx, dim=2)
        if remaining.dtype == torch.float8_e4m3fn:
            return ConcatenateOp.apply(bucket_slice, remaining, dim=2)
        return torch.cat([bucket_slice, remaining], dim=2)
    else:
        remaining = slice_lhs(cache, max_idx - seq_len, dim=2)
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
        self._init_kv_shape(config)
        self.quant = config.neuron_config.kv_cache_quant

        num_layer = config.num_hidden_layers
        dtype = config.neuron_config.torch_dtype
        if self.quant:
            self.quant_dtype = torch.float8_e4m3fn
            self.dequant_dtype = torch.bfloat16
        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.kv_shape, dtype=dtype), requires_grad=False)
                for _ in range(num_layer * 2)
            ]
        )
        if self.quant:
            self.past_key_values=self.past_key_values.to(self.quant_dtype)

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
        hidden_dim_per_head = hidden_size // num_atten_head
        return hidden_dim_per_head

    def _init_kv_shape(self, config: InferenceConfig):
        max_batch_size = config.neuron_config.max_batch_size
        max_len = config.neuron_config.max_length
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)
        hidden_dim_per_head = self._get_hidden_dim_per_head(config)

        if self.flash_decoding_enabled:
            max_len = utils.divide(max_len, self.num_cores_per_group)

        # BHSD KV cache layout for classic attention
        self.kv_shape = (
            max_batch_size,
            num_kv_heads_per_rank,
            max_len,
            hidden_dim_per_head,
        )

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
        if self.is_medusa:
            accepted_k_cache = torch.gather(input=k_cache, dim=2, index=gather_index)
            accepted_v_cache = torch.gather(input=v_cache, dim=2, index=gather_index)
            k_cache = torch.scatter(input=k_cache, dim=2, index=slice_index, src=accepted_k_cache)
            v_cache = torch.scatter(input=v_cache, dim=2, index=slice_index, src=accepted_v_cache)
        return k_cache, v_cache

    def get_cache(self, seq_len: int, **kwargs):
        """
        Return network (all layers)'s previously cached K and V, up to seq_len.

        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
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
            # slice for partial view
            key_state = _slice_kv_cacheline(self.padding_side, seq_len, k_cache)
            value_state = _slice_kv_cacheline(self.padding_side, seq_len, v_cache)
            if self.quant:
                key_state = dequantize.direct_cast_dequantize(key_state, self.dequant_dtype)
                value_state = dequantize.direct_cast_dequantize(value_state, self.dequant_dtype)
            past_key_values.append([key_state, value_state])
        return past_key_values

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        past_key_values: List[Tensor],
        seq_len: int,
        scatter_index=None,
        active_mask=None,
    ):
        """
        Given the passed-in past_key_values, update the cache

        :param scatter_index: tensor representing index to update
        :param is_for_context_encoding: bool
        :param seq_ids: tensor of size (batch_sz)
        :param position_ids: tensor of size (batch_sz, bucket_sz)
        :param past_key_values: list of tuple, the latest kv obtained at the end of the network from forward pass
        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
        :param scatter_index: tensor representing index to update
        :param active_mask: tensor representing index to update
        :return: list of tuple of (K, V)
        """
        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(past_key_values):
            latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]

            if self.quant:
                latest_k = quantize.direct_cast_quantize(latest_k, self.quant_dtype)
                latest_v = quantize.direct_cast_quantize(latest_v, self.quant_dtype)

            if self.is_medusa:
                k_cache = self.past_key_values[idx * 2]
                v_cache = self.past_key_values[idx * 2 + 1]
            else:
                # read differently based on padding side and bucket sz
                k_cache = _slice_kv_cacheline(
                    self.padding_side, seq_len, self.past_key_values[idx * 2]
                )
                v_cache = _slice_kv_cacheline(
                    self.padding_side, seq_len, self.past_key_values[idx * 2 + 1]
                )

            if is_for_context_encoding:
                if self.is_continuous_batching:
                    # scatter back to the desired seq_ids
                    seq_id_index_shape = seq_ids.shape[:1] + k_cache.shape[1:]
                    seq_id_index = seq_ids.view(-1, 1, 1, 1).expand(seq_id_index_shape)
                    k_cache = torch.scatter(input=k_cache, dim=0, index=seq_id_index, src=latest_k)
                    v_cache = torch.scatter(input=v_cache, dim=0, index=seq_id_index, src=latest_v)
                else:
                    # assign back to full kv_cacheline
                    k_cache = latest_k
                    v_cache = latest_v
                    if self.is_medusa:
                        k_cache = _gather_slice_into_kv_cacheline(
                            self.past_key_values[idx * 2], self.padding_side, seq_len, k_cache
                        )
                        v_cache = _gather_slice_into_kv_cacheline(
                            self.past_key_values[idx * 2 + 1], self.padding_side, seq_len, v_cache
                        )
            else:
                if self.padding_side == "left":
                    # TODO: fix it with scatter after right padding
                    k_cache = k_cache[:, :, 1:, :]
                    v_cache = v_cache[:, :, 1:, :]
                    k_cache = torch.cat([k_cache, latest_k], dim=2)
                    v_cache = torch.cat([v_cache, latest_v], dim=2)
                else:
                    # copy the tensor of the new position into kv cache
                    if self.flash_decoding_enabled:
                        assert active_mask is not None, 'active_mask should be specified for flash decoding!'
                        garbage_pos = seq_len-1  # treat last pos as garbage
                        updated_pos_ids = position_ids // self.num_cores_per_group
                        scatter_index = torch.where(active_mask == 1, updated_pos_ids, garbage_pos)
                        scatter_index_new = scatter_index.view(-1, 1, scatter_index.shape[-1], 1).expand_as(latest_k)
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

            # update
            if not self.is_medusa:
                k_cache = _gather_slice_into_kv_cacheline(
                    self.past_key_values[idx * 2], self.padding_side, seq_len, k_cache
                )
                v_cache = _gather_slice_into_kv_cacheline(
                    self.past_key_values[idx * 2 + 1], self.padding_side, seq_len, v_cache
                )
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
