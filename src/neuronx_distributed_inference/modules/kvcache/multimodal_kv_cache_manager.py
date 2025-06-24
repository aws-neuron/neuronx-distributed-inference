from typing import List

import torch
from neuronx_distributed.parallel_layers import utils
from torch import Tensor, nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager


class MultimodalKVCacheManager(KVCacheManager):
    """
    Multimodal Key Value Cache Management.
    It inherents the text KV cache from `KVCacheManager` which is used in the self-attention layers.
    In addition, it adds a new set of vision KV cache that is used in the cross-attention layers of Llama3.2 Multimodal.
    The full vision KV cache is populated in context encoding, and does not update during token generation.
    """

    def __init__(
        self,
        config: InferenceConfig,
        vision_config: InferenceConfig,
        num_self_attention_layers,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        dtype = config.neuron_config.attention_dtype if config.neuron_config.attention_dtype is not None else config.neuron_config.torch_dtype

        # override past_key_values from base class with correct num_hidden_layers passed as arg
        # as num_hidden_layers are interpreted differently in the various config's
        assert (
            self.k_shape == self.v_shape
        ), 'K and V cache shapes must match in MultimodalKVCacheManager'
        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.k_shape, dtype=dtype), requires_grad=False)
                for _ in range(num_self_attention_layers * 2)
            ]
        )

        num_xatten_layers = len(config.cross_attention_layers)
        self._init_vision_kv_shape(config, vision_config)
        self.vision_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.vision_kv_shape, dtype=dtype), requires_grad=False)
                for _ in range(num_xatten_layers * 2)
            ]
        )

    def get_cache(self, seq_len: int, **kwargs):
        """
        Return network (all layers)'s previously cached K and V, up to seq_len.

        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
        :return: Tuple( List[Tuple(textK, textV)], List[Tuple(visionK, visionV)] )
        """
        past_key_values = super().get_cache(seq_len, **kwargs)

        vision_key_values = []
        for key_layer_idx in range(0, len(self.vision_key_values), 2):
            # get kv per layer
            k_cache, v_cache = (
                self.vision_key_values[key_layer_idx],
                self.vision_key_values[key_layer_idx + 1],
            )
            vision_key_values.append([k_cache, v_cache])

        return past_key_values, vision_key_values

    def update_vision_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        vision_key_values: List[Tensor],
        seq_len: int,
        scatter_index=None,
    ):
        updated_kv_cache = []
        xatten_idx = 0
        for idx, kv_per_layer in enumerate(vision_key_values):
            if kv_per_layer[0] is None:
                # dummy xattention layer, skipping
                continue

            k_cache = self.vision_key_values[xatten_idx * 2]
            v_cache = self.vision_key_values[xatten_idx * 2 + 1]

            if is_for_context_encoding:
                if self.is_continuous_batching:
                    # update certain batch-line(s) that matches seq_id(s)
                    latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]

                    seq_id_index_shape = seq_ids.shape[:1] + k_cache.shape[1:]
                    seq_id_index = seq_ids.view(-1, 1, 1, 1).expand(seq_id_index_shape)

                    k_cache = torch.scatter(input=k_cache, dim=0, index=seq_id_index, src=latest_k)
                    v_cache = torch.scatter(input=v_cache, dim=0, index=seq_id_index, src=latest_v)
                else:
                    # update the entire cache
                    # Why (k_cache * 0)?
                    # write-only buffers are removed by torch XLA, so we do a dummy read from k_cache
                    # and v_cache so XLA doesn't remove the buffer during CTX encoding lowering
                    k_cache = kv_per_layer[0] + (k_cache * 0)
                    v_cache = kv_per_layer[1] + (v_cache * 0)

            else:
                # during token_generation, vision tokens don't grow.
                # no need to update the cache and we return KV cache as is
                pass

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)
            xatten_idx += 1

        return updated_kv_cache

    def _init_vision_kv_shape(self, config: InferenceConfig, vision_config: InferenceConfig):
        if config.neuron_config.tp_degree > config.num_key_value_heads:
            num_kv_heads_per_rank = 1
        else:
            num_kv_heads_per_rank = utils.divide(
                config.num_key_value_heads, config.neuron_config.tp_degree
            )
        kv_dim_2 = vision_config.max_num_tiles * (
            (vision_config.image_size // vision_config.patch_size) ** 2 + 1
        )
        head_dim = self._get_hidden_dim_per_head(config)
        self.vision_kv_shape = [
            config.neuron_config.max_batch_size,
            num_kv_heads_per_rank,
            kv_dim_2,
            head_dim,
        ]
