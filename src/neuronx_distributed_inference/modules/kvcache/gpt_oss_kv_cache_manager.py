from typing import List, Tuple

import torch
from neuronx_distributed.parallel_layers import parallel_state, utils
from torch import Tensor, nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402; noqa: E402; noqa: E402
    determine_sharding_strategy,
    get_shardable_head_counts,
)
from neuronx_distributed_inference.modules.kvcache.utils import dynamic_update_slice, update_cache_const_indices, fill_prefix

from neuronx_distributed_inference.modules.attention.utils import get_kernel_cache_size_bucket, get_kv_head_indices_context_parallel_full_tp_decode, get_kv_head_indices_context_parallel_dp_decode, get_cp8_tp8_rank_ordering
from neuronx_distributed_inference.modules.attention.attention_process_groups import get_tp_cp_group_mesh
from neuronx_distributed_inference.utils.distributed import split_along_dim, get_dp_rank


def _slice_kv_cacheline(padding_side: str, seq_len: int, cache: Tensor, transposed: bool):
    seqlen_dim = 3 if transposed else 2
    if padding_side == "right":
        return torch.ops.aten.slice(cache, dim=seqlen_dim, start=0, end=seq_len)
    max_idx = cache.shape[seqlen_dim]
    return torch.ops.aten.slice(cache, dim=seqlen_dim, start=max_idx - seq_len, end=max_idx)

# TODO: KV Cache needs to be layer dependent in NxDI, we currently initialize it like this in GptOss due
# to a variable B and S cache size across layers.


class GptOssKVCacheManager(nn.Module):
    """
    Key Value Cache Management.
    It stores KV cache as a parameter list of the shape (batch_sz, num_kv_head_per_rank, max_len, head_dim),
    and vends out read and write operations.
    """

    def __init__(self, config: InferenceConfig, num_kv_head, sliding_window, global_rank=None):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.padding_side = config.neuron_config.padding_side
        self.is_continuous_batching = config.neuron_config.is_continuous_batching
        self.num_kv_head = num_kv_head
        self.batch_size = config.neuron_config.max_batch_size
        self.padding_side = config.neuron_config.padding_side
        self.k_cache_transposed = config.neuron_config.k_cache_transposed
        self.global_rank = global_rank
        self.sliding_window = sliding_window - 1 if not self.neuron_config.enable_fused_speculation else sliding_window + self.neuron_config.speculation_length - 2
        self.num_layers = config.num_hidden_layers
        self.dp_degree = config.neuron_config.attention_dp_degree
        self.swa_dp_degree = config.neuron_config.sliding_window_attention_dp_degree
        self.cp_degree = config.neuron_config.cp_degree
        self.world_size = config.neuron_config.tp_degree
        self.dtype = config.neuron_config.attention_dtype if config.neuron_config.attention_dtype is not None else config.neuron_config.torch_dtype
        self.num_attention_heads = config.num_attention_heads

        self._init_kv_shape()

        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(k_or_v_shape, dtype=self.dtype), requires_grad=False)
                for layer_idx in range(self.num_layers) for k_or_v_shape in [self.k_shapes[layer_idx], self.v_shapes[layer_idx]]
            ]
        )

    def _get_num_kv_heads_per_rank(self, tp_degree):
        num_kv_head = self.num_kv_head
        num_atten_head = self.num_attention_heads

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

    def _init_kv_shape(self):
        self.k_shapes = []
        self.v_shapes = []

        self.dp_cache_bs = (self.batch_size // self.dp_degree) + 1  # +1 for garbage position

        for layer in range(0, self.num_layers):
            is_swa_layer = layer % 2 == 0
            tp_degree = self.world_size if is_swa_layer else self.world_size // self.dp_degree
            dp_degree = self.swa_dp_degree if is_swa_layer else self.dp_degree

            batch_size = self.batch_size
            if dp_degree > 1:
                batch_size = self.dp_cache_bs

            max_len = self.config.neuron_config.max_length if not is_swa_layer else get_kernel_cache_size_bucket(self.sliding_window)
            self.max_len = max_len

            num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(tp_degree)
            hidden_dim_per_head = self._get_hidden_dim_per_head(self.config)

            k_shape = v_shape = (batch_size, num_kv_heads_per_rank, max_len, hidden_dim_per_head)

            self.k_shapes.append(k_shape)
            self.v_shapes.append(v_shape)

            print(f"For layer {layer}, using K shape {k_shape}")

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

        return k_cache, v_cache

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
        is_swa_layer = idx % 2 == 0
        dp_degree = self.swa_dp_degree if is_swa_layer else self.dp_degree
        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)

        if dp_degree > 1:
            k_cache = k_cache[: -1]  # remove garbage position
            v_cache = v_cache[: -1]

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
        is_swa_layer = idx % 2 == 0
        dp_degree = self.swa_dp_degree if is_swa_layer else self.dp_degree

        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]

        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)

        cte_rank_ordering = None
        if self.cp_degree == 8 and self.world_size // self.cp_degree == 8:
            cte_rank_ordering = get_cp8_tp8_rank_ordering(self.neuron_config.tp_degree, self.neuron_config.cp_degree, switch_cc=self.neuron_config.switch_cc)

        if not is_for_context_encoding and dp_degree > 1:
            dp_rank = get_dp_rank(self.get_rank(dp_degree, seq_ids.device), self.neuron_config.tp_degree // dp_degree, dp_degree, self.neuron_config.switch_cc)
            seq_ids = split_along_dim(seq_ids, dim=0, rank=dp_rank, num_partitions=dp_degree)
            position_ids = split_along_dim(position_ids, dim=0, rank=dp_rank, num_partitions=dp_degree)

        if is_for_context_encoding:
            if self.cp_degree > 1 and self.cp_degree != dp_degree:
                # When we run CP without DP, decode will run in full TP, selectively write the heads that are used in decode
                rank = self.get_rank(dp_degree, seq_ids.device)

                # TP update
                if dp_degree == 1:
                    kv_head_indices = get_kv_head_indices_context_parallel_full_tp_decode(self.num_kv_head, self.neuron_config.tp_degree, self.neuron_config.cp_degree, k_cache.device, cte_rank_ordering)

                # TP + DP update
                else:
                    decode_ordering = None
                    if dp_degree == 8 and self.neuron_config.tp_degree // dp_degree == 8:
                        decode_ordering = sum(get_tp_cp_group_mesh(self.neuron_config.tp_degree, dp_degree, self.neuron_config.switch_cc), [])

                    kv_head_indices = get_kv_head_indices_context_parallel_dp_decode(self.num_kv_head, self.neuron_config.tp_degree,
                                                                                     self.neuron_config.cp_degree, dp_degree,
                                                                                     k_cache.device, cte_rank_ordering=cte_rank_ordering,
                                                                                     decode_rank_ordering=decode_ordering,
                                                                                     switch_cc=self.neuron_config.switch_cc)
                head_idx = torch.index_select(kv_head_indices, dim=0, index=rank)
                latest_k = torch.index_select(latest_k, dim=1, index=head_idx)
                latest_v = torch.index_select(latest_v, dim=1, index=head_idx)

            if self.is_continuous_batching:
                assert seq_ids.dim() == 1 and seq_ids.shape[0] == 1, "only supports single seq_id"
                if self.neuron_config.k_cache_transposed or dp_degree > 1:
                    cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids, dp_degree)
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
        is_swa_layer = layer_idx % 2 == 0
        if is_swa_layer:
            position_ids = position_ids % (self.sliding_window)
        index = position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)

    def get_cache_update_index_for_seq_ids(self, seq_ids, dp_degree):
        # handle out-of-bound seq_ids
        garbage_pos = self.dp_cache_bs - 1
        true_batch_size = self.dp_cache_bs - 1

        dp_rank = torch.div(
            self.get_rank(dp_degree, seq_ids.device),
            self.world_size // dp_degree,
            rounding_mode="floor",
        ).to(torch.int32)

        kv_range_start = dp_rank * true_batch_size
        kv_range_end = kv_range_start + true_batch_size
        valid_mask = torch.logical_and(
            seq_ids >= kv_range_start,
            seq_ids < kv_range_end,
        )

        seq_ids = torch.where(
            valid_mask, seq_ids - kv_range_start, garbage_pos
        )
        return seq_ids

    def get_rank(self, dp_degree, device=torch.device("cpu")):
        rank = self.global_rank.get_rank()
        if dp_degree == 8 and self.neuron_config.tp_degree // dp_degree == 8:
            rank_ordering = get_cp8_tp8_rank_ordering(self.neuron_config.tp_degree, dp_degree, switch_cc=self.neuron_config.switch_cc, device=device)
            return torch.index_select(rank_ordering, dim=0, index=rank)

        return rank
