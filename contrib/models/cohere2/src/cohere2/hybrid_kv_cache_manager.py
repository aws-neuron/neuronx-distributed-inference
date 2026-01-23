import logging
from typing import List, Tuple

import torch

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.flashdecode.utils import get_cache_size
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
from neuronx_distributed_inference.modules.kvcache.utils import (
    dynamic_update_slice, 
    fill_prefix, 
    update_cache_const_indices
    )
from neuronx_distributed.quantization import dequantize, quantize


class HybridKVCacheManager(KVCacheManager):

    def __init__(self, config: InferenceConfig, **kwargs) -> None:
        self.sliding_window_size = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.k_shape = self.v_shape = (1, 1, 1, 1) # Necessary to call super().__init__
        super().__init__(config=config, **kwargs)

        dtype = config.neuron_config.torch_dtype

        self.past_key_values = torch.nn.ParameterList()
        for layer_idx in range(config.num_hidden_layers):
            if self._is_sliding_window_enabled(layer_idx=layer_idx):
                self.past_key_values.extend([
                    torch.nn.Parameter(torch.zeros(self.sliding_kv_shape, dtype=dtype), requires_grad=False),
                    torch.nn.Parameter(torch.zeros(self.sliding_kv_shape, dtype=dtype), requires_grad=False)
                ])
            else:
                self.past_key_values.extend([
                    torch.nn.Parameter(torch.zeros(self.global_kv_shape, dtype=dtype), requires_grad=False),
                    torch.nn.Parameter(torch.zeros(self.global_kv_shape, dtype=dtype), requires_grad=False)
                ])

        if self.quant:
            self.past_key_values = self.past_key_values.to(self.quant_dtype)

    def _is_sliding_window_enabled(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.sliding_window_pattern != 0

    def _init_kv_shape(self, config: InferenceConfig, layer_to_cache_size_mapping=None) -> None:
        max_batch_size = config.neuron_config.max_batch_size
        max_total_sequence_length = config.neuron_config.max_length
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)
        hidden_dim_per_head = self._get_hidden_dim_per_head(config)

        if self.flash_decoding_enabled:
            # 1. We ensure that max_seq_length can be divided by num_cores_per_group by padding it if necessary
            padded_max_total_sequence_length = max_total_sequence_length
            if max_total_sequence_length % self.num_cores_per_group != 0:
                padded_max_len += self.num_cores_per_group - max_total_sequence_length % self.num_cores_per_group
                logging.warning(
                    f"Max length needs to be multiples of num_cores_per_group {self.num_cores_per_group}"
                    f" but got {max_total_sequence_length}. Padding it to {padded_max_total_sequence_length} meet the requirement."
                )
            # 2. Local maximum sequence length is max_seq_length // num_cores_per_group + garbage tile size
            max_total_sequence_length = get_cache_size(
                seq_len=padded_max_total_sequence_length, 
                num_cores_per_group=self.num_cores_per_group,
                is_ctx=False
                )

        # Flash Decoding: Only global attention layers are sharded across the sequence dimension
        if self.is_kv_cache_tiled:
            num_tiles_global = int(max_total_sequence_length / 128)
            num_tiles_sliding = int(self.sliding_window_size / 128)
            # KV cache layout : BHS(128 tiled)D
            self.global_kv_shape = (
                max_batch_size,
                num_kv_heads_per_rank,
                128,  # Sequence dim is tiled
                num_tiles_global,  # max_len = 128 * num_tiles
                hidden_dim_per_head,
            )
            self.sliding_kv_shape = (
                max_batch_size,
                num_kv_heads_per_rank,
                128,  # Sequence dim is tiled
                num_tiles_sliding,  # max_len = 128 * num_tiles
                hidden_dim_per_head,
            )
        else:
            # KV cache layout : BHSD
            self.global_kv_shape = (
                max_batch_size,
                num_kv_heads_per_rank,
                max_total_sequence_length,
                hidden_dim_per_head,
            )
            self.sliding_kv_shape = (
                max_batch_size,
                num_kv_heads_per_rank,
                self.sliding_window_size,
                hidden_dim_per_head,
            )

    def get_cache(self, seq_len: int, skip_slice=False, kvcache_buffer=None, seq_ids=None, **kwargs):
        """
        Return network (all layers)'s previously cached K and V, up to seq_len.

        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
        :param skip_slice: whether to skip slicing the KV cache to the seq_len
        :return: list of tuple of (K, V)
        """
        past_key_values = []
        if not skip_slice:
            if self.flash_decoding_enabled:
                global_layer_slice_length = get_cache_size(seq_len=seq_len, num_cores_per_group=self.num_cores_per_group, is_ctx=False)
            else: 
                global_layer_slice_length = seq_len
            swa_layer_slice_length = min(seq_len, self.sliding_window_size) 

        for idx in range(len(self.past_key_values) // 2):
            is_swa_layer = self._is_sliding_window_enabled(layer_idx=idx)
            slice_length = swa_layer_slice_length if is_swa_layer else global_layer_slice_length

            k_cache, v_cache = self.get_kv_by_layer_id(
                idx=idx,
                skip_slice=skip_slice,
                seq_len=seq_len,
                kvcache_buffer=kvcache_buffer,
                seq_ids=slice_length,
                **kwargs,
            )

            past_key_values.append([k_cache, v_cache])
            
        return past_key_values
    
    def update_kv_by_layer_id(
        self,
        idx: int,
        is_for_context_encoding: bool,
        seq_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        kv_per_layer: Tuple[torch.FloatTensor],
        seq_len: int,
        scatter_index = None,
        kv_active_mask: torch.BoolTensor = None,
        kvcache_buffer: List[Tuple[torch.FloatTensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        bucket_size = seq_len
        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]

        is_swa_layer = self._is_sliding_window_enabled(layer_idx=idx)
        # If bucket_size<window_size, we stick to global attention behavior since equivalent
        swa_enabled = (is_swa_layer and bucket_size > self.sliding_window_size)

        if self.quant:
            latest_k = quantize.direct_cast_quantize(latest_k, self.quant_dtype)
            latest_v = quantize.direct_cast_quantize(latest_v, self.quant_dtype)

        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)

        if is_for_context_encoding:
            if swa_enabled:
                # If SWA layer & bucket larger than window size -> gather
                latest_k, latest_v = self._apply_sliding_window(
                    position_ids=position_ids,
                    latest_k=latest_k,
                    latest_v=latest_v,
                )
            if self.is_continuous_batching:
                # ctx_batch_size=1<max_batch size, i.e. max_batch_size>1
                assert (seq_ids.dim() == 1 and seq_ids.shape[0] == 1), \
                "Continuous batching only supports single seq_id (ctx_batch_size=1)"
                if self.neuron_config.k_cache_transposed:
                    cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
                    indices = [cache_idx] + [torch.zeros(1, device=seq_ids.device) for _ in range(k_cache.dim() - 1)]
                    indices = [t.squeeze().to(torch.int32) for t in indices]
                    k_cache = dynamic_update_slice(k_cache, latest_k, indices)
                    v_cache = dynamic_update_slice(v_cache, latest_v, indices)
                else:
                    k_cache = update_cache_const_indices(k_cache, latest_k, seq_ids)
                    v_cache = update_cache_const_indices(v_cache, latest_v, seq_ids)
            else:
                # ctx_batch_size=max_batch_size, therefore latest_k and k_cache have the same size along dim0
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
                if not is_swa_layer and self.flash_decoding_enabled:
                    assert (kv_active_mask is not None), "active_mask should be specified for flash decoding!"
                    global_layer_slice_length = get_cache_size(seq_len=bucket_size, num_cores_per_group=self.num_cores_per_group, is_ctx=False)
                    garbage_pos = global_layer_slice_length - 1
                    updated_pos_ids = position_ids // self.num_cores_per_group
                    scatter_index = torch.where(kv_active_mask == 1, updated_pos_ids, garbage_pos)
                    update_index = scatter_index.view(-1, 1, scatter_index.shape[-1], 1).expand_as(latest_k)
                else:
                    if swa_enabled:
                        k_cache, v_cache = self._roll_cache(
                                position_ids=position_ids,
                                k_cache=k_cache,
                                v_cache=v_cache
                            )
                            
                    update_index = self._get_index_to_update_new_position(
                        scatter_index=scatter_index,
                        position_ids=position_ids,
                        update_shape=latest_k.shape,
                        swa_enabled=swa_enabled,
                    )

                    k_cache = torch.scatter(
                        input=k_cache, dim=2, index=update_index, src=latest_k
                    )
                    v_cache = torch.scatter(
                        input=v_cache, dim=2, index=update_index, src=latest_v
                    )
        return k_cache, v_cache

    def _get_index_to_update_new_position(self, 
                                          scatter_index: torch.LongTensor, 
                                          position_ids: torch.LongTensor, 
                                          update_shape: Tuple[int],
                                          swa_enabled: bool,
                                          ) -> torch.LongTensor:
        batch_size, num_kv_heads, _, head_dim = update_shape
        if self.is_medusa:
            raise NotImplementedError("Speculative decoding is currently not supported for hybrid KV cache")
        if self.padding_side == "left":
            position_ids = torch.max(position_ids)[None, None].expand(batch_size, -1)
        if swa_enabled:
            position_ids = torch.clamp(position_ids, min=0, max=self.sliding_window_size - 1)        
        update_index = position_ids.view(-1, 1, 1, 1).expand(-1, num_kv_heads, 1, head_dim)
        return update_index

    def _apply_sliding_window(self, 
                              position_ids: torch.LongTensor, 
                              latest_k: torch.FloatTensor, 
                              latest_v: torch.FloatTensor
                              ) -> Tuple[torch.FloatTensor]:
        batch_size, num_kv_heads, _, head_dim = latest_k.shape
        if self.padding_side == "left":
            #max_pos_ids = torch.amax(position_ids, keepdim=True).expand(batch_size, -1)
            max_position_ids = torch.max(position_ids)[None, None].expand(batch_size, -1)
        else:
            max_position_ids = torch.amax(position_ids, dim=1, keepdim=True)
        offset = torch.clamp(max_position_ids - self.sliding_window_size + 1, min=0)
        index = torch.arange(self.sliding_window_size, device=latest_k.device)[None, :] + offset
        index = index[:, None, :, None].expand(-1, num_kv_heads, -1, head_dim)
        latest_k = torch.gather(latest_k, dim=2, index=index) 
        latest_v = torch.gather(latest_v, dim=2, index=index)
        return latest_k, latest_v

    def _roll_cache(self, 
                    position_ids: torch.LongTensor, 
                    k_cache: torch.FloatTensor, 
                    v_cache: torch.FloatTensor,
                    in_place: bool = False,
                    return_view: bool = False,
                    ) -> Tuple[torch.FloatTensor]:
        if in_place:
            assert return_view, "In-place update returns a view by design"
            k_cache, v_cache = self._roll_cache_in_place(
                position_ids=position_ids,
                k_cache=k_cache,
                v_cache=v_cache
            )
            return k_cache, v_cache
        else:
            rolled_k_cache, rolled_v_cache = self._roll_cache_out_of_place(
                position_ids=position_ids,
                k_cache=k_cache,
                v_cache=v_cache
            )
            if return_view:
                k_cache = fill_prefix(k_cache, rolled_k_cache)
                v_cache = fill_prefix(v_cache, rolled_v_cache)
                return k_cache, v_cache
            else: 
                return rolled_k_cache, rolled_v_cache
                
    def _roll_cache_out_of_place(self, 
                    position_ids: torch.LongTensor, 
                    k_cache: torch.FloatTensor, 
                    v_cache: torch.FloatTensor
                    ) -> Tuple[torch.FloatTensor]:
        # binary_offset -> 1 if roll cache, else 0
        if self.padding_side == "left":
            batch_size, _ = position_ids.shape
            position_ids = torch.max(position_ids)[None, None].expand(batch_size, -1)
        binary_offset = torch.clamp(position_ids - (self.sliding_window_size - 1), min=0, max=1)
        roll_index = torch.arange(0, self.sliding_window_size, device=k_cache.device)[None, :]
        roll_index = torch.remainder(roll_index + binary_offset, self.sliding_window_size)\
            .view(-1, 1, self.sliding_window_size, 1)\
            .expand_as(k_cache)
        k_cache = torch.gather(k_cache, dim=2, index=roll_index)
        v_cache = torch.gather(v_cache, dim=2, index=roll_index)
        return k_cache, v_cache

    def _roll_cache_in_place(self, 
                    position_ids: torch.LongTensor, 
                    k_cache: torch.FloatTensor, 
                    v_cache: torch.FloatTensor
                    ) -> Tuple[torch.FloatTensor]:
        if self.padding_side == "left":
            binary_offset = torch.ones_like(position_ids)
        else:
            binary_offset = torch.clamp(position_ids - self.sliding_window_size + 1, min=0, max=1)
        roll_index = torch.arange(0, self.sliding_window_size, device=k_cache.device)[None, :]
        roll_index = torch.remainder(roll_index - binary_offset, self.sliding_window_size)
        roll_index = roll_index.view(-1, 1, self.sliding_window_size, 1).expand_as(k_cache)
                
        k_cache.scatter_(dim=2, index=roll_index, src=k_cache.clone())
        v_cache.scatter_(dim=2, index=roll_index, src=v_cache.clone())
        
        return k_cache, v_cache
