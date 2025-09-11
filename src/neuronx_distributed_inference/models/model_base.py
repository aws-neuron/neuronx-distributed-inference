import copy
import inspect
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import neuronx_distributed as nxd
import torch
import torch_xla.core.xla_model as xm
import neuronx_distributed_inference.modules.autobucketing as autobucketing

from neuronx_distributed.operators.argmax import argmax as nxd_argmax
from neuronx_distributed.operators.topk import topk as nxd_topk
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.mappings import (
    _gather_along_dim,
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
    _traced_spmd_tiled_rs,
    _traced_tiled_rs,
)
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group
from neuronxcc.nki.compiler.backends.neuron.dimensions import CCPipeline   # noqa: N813
from neuronx_distributed.quantization.quantization_utils import convert_qint8_to_int8_state_dict
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.models.model_wrapper import (  # noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,
    FUSED_SPECULATION_MODEL_TAG,
    MEDUSA_MODEL_TAG,
    SPECULATION_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.async_execution import causal_lm_async_execution
from neuronx_distributed_inference.modules.attention import utils as attn_utils
from neuronx_distributed_inference.modules.chunked_prefill.scheduler import GridTileScheduler
from neuronx_distributed_inference.modules.custom_calls import neuron_cumsum
from neuronx_distributed_inference.modules.eagle.hidden_state import HiddenStateRollingBuffer
from neuronx_distributed_inference.modules.eagle.token_tree import TokenTree
from neuronx_distributed_inference.modules.flashdecode.utils import (
    get_cache_size,
    mask_util,
    turn_2d_mask_to_4d,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    Sampler,
    mask_padded_logits,
    prepare_sampling_params,
    rand_like,
    validate_sampling_params,
    infer_sampling_params,
)
from neuronx_distributed_inference.modules.generation.seq_parallel_logits_slice import (
    seq_parallel_slice_last_token,
)
from neuronx_distributed_inference.modules.kvcache import utils as kvcache_utils
from neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager import BlockKVCacheManager, generate_tokengen_slot_mapping, generate_fusedspec_slot_mapping
from neuronx_distributed_inference.modules.kvcache.data_parallel_kv_cache_manager import DataParallelKVCacheManager
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
from neuronx_distributed_inference.modules.lora_serving import LoraCheckpoint, wrap_model_with_lora
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.modules.attention.utils import stride_tensor, chunk_and_reorder_tensor
from neuronx_distributed_inference.modules.attention.attention_process_groups import get_flattened_inverted_tp_cp_group_mesh
from neuronxcc.nki.language import nc
try:
    from neuronxcc.nki._pre_prod_kernels.attention_token_gen import (
        gen_cache_mask_for_attention_tkg_kernel,
    )
except ImportError:
    logging.warning(
        "Use a more recent neuron compiler version to enable gen_cache_mask_for_attention_tkg_kernel"
    )
    gen_cache_mask_for_attention_tkg_kernel = None


class NeuronBaseModel(nn.Module):
    """
    Base model that NeuronXXXModel classes inherit from.

    The forward() function will be traced and compiled by NxD.
    """

    def __init__(self, config: InferenceConfig, optimize_inference=True):
        super().__init__()

        self.config = config
        self.sampler = None
        self.kv_mgr = None
        self.neuron_config = config.neuron_config
        self.batch_size = config.neuron_config.batch_size
        self.n_positions = config.neuron_config.n_positions
        self.prefix_size = 0
        self.n_active_tokens = config.neuron_config.n_active_tokens
        self.vocab_size = config.vocab_size
        self.speculation_length = config.neuron_config.speculation_length
        self.padding_side = config.neuron_config.padding_side
        self.max_length = config.neuron_config.max_length
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1
        self.rank_util = SPMDRank(world_size=self.config.neuron_config.tp_degree)
        self.num_cores_per_group = config.num_cores_per_group
        self.is_block_kv_layout = config.neuron_config.is_block_kv_layout
        self.is_chunked_prefill = config.neuron_config.is_chunked_prefill
        self.is_prefix_caching = config.neuron_config.is_prefix_caching
        self.attention_chunk_size = None
        self.sliding_window = None
        self.layer_to_cache_size_mapping = None
        self.has_mixed_attn = False

        self.setup_attr_for_model(config)
        self.init_model(config)
        if self.attention_chunk_size and self.attention_chunk_size >= self.config.neuron_config.seq_len:
            logging.warning(f"attention chunk size {self.attention_chunk_size} is greater than or equal to seq_len {self.config.neuron_config.seq_len}. Chunked attention is disabled")
            self.attention_chunk_size = None
            self.has_mixed_attn = False

        if optimize_inference:
            self.init_inference_optimization(config)

        lora_config = self.neuron_config.lora_config
        if lora_config is not None:
            wrap_model_with_lora(self, lora_config)
            self.lora_checkpoint = LoraCheckpoint(lora_config)
        self.sliced_hidden = False

    def setup_attr_for_model(self, config: InferenceConfig):
        """
        Please provide model-specific definition for the following attributes
            self.on_device_sampling
            self.tp_degree
            self.hidden_size
            self.num_attention_heads
            self.num_key_value_heads
            self.max_batch_size
            self.buckets
        """
        raise NotImplementedError("setup_attr_for_model() is not implemented")

    def init_model(self, config: InferenceConfig):
        """
        Please provide definition for the following components:
            self.embed_tokens
            self.layers
            self.norm
            self.lm_head
        """
        raise NotImplementedError("init_model() is not implemented")

    def initialize_process_group(self, seed: int = 0):
        if not torch.dist.is_initialized():
            torch.dist.init_process_group(backend="xla")
        else:
            logging.warning("torch.distributed was already initialized, skipping...")

        if not nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
            nxd.parallel_layers.initialize_model_parallel(
                tensor_model_parallel_size=self.neuron_config.tp_degree,
                pipeline_model_parallel_size=self.neuron_config.pp_degree,
                expert_model_parallel_size=self.neuron_config.ep_degree,
            )
        else:
            logging.warning("NxD was already initialized, skipping...")

        # set seed
        set_random_seed(seed)

    def init_inference_optimization(self, config: InferenceConfig):
        if self.on_device_sampling:
            self.sampler = Sampler(config.neuron_config)

        if config.neuron_config.attention_dp_degree > 1:
            self.kv_mgr = DataParallelKVCacheManager(config, num_kv_head=self.num_key_value_heads, global_rank=self.rank_util)
        elif config.neuron_config.is_block_kv_layout:
            self.kv_mgr = BlockKVCacheManager(config, num_kv_head=self.num_key_value_heads)
        else:
            self.kv_mgr = KVCacheManager(config,
                                         num_kv_head=self.num_key_value_heads,
                                         global_rank=self.rank_util,
                                         attention_chunk_size=self.attention_chunk_size,
                                         sliding_window=self.sliding_window,
                                         layer_to_cache_size_mapping=self.layer_to_cache_size_mapping)

    def _is_context_encoding(self, input_ids: torch.Tensor):
        return input_ids.shape[-1] > 1 and input_ids.shape[-1] != self.speculation_length

    def _is_for_speculation(self, input_ids: torch.Tensor):
        return input_ids.shape[-1] == self.speculation_length

    def _create_token_tree_attn_mask(
        self, attention_mask, is_for_context_encoding, is_for_token, **kwargs
    ):
        if is_for_token:
            return attention_mask
        if is_for_context_encoding:
            return self._create_context_attn_mask(attention_mask, **kwargs)
        else:
            return self._create_simple_attn_mask(attention_mask)

    def _create_context_attn_mask(self, attention_mask):
        if self.is_prefix_caching and self.prefix_size != 0:
            # Mask as per prefix caching kernel flow.
            return attention_mask
        # Lower triangle causal mask for classic attention
        mask = torch.full(
            (self.n_positions, self.n_positions), True, device=attention_mask.device
        ).tril(diagonal=0)
        mask = mask[None, None, :, :].expand(self.batch_size, 1, self.n_positions, self.n_positions)

        if self.padding_side == "right":
            return mask
        else:
            expanded_mask = (
                attention_mask[:, None, None, :]
                .expand(self.batch_size, 1, self.n_positions, self.n_positions)
                .to(torch.bool)
            )
            return torch.logical_and(mask, expanded_mask)

    def _create_chunked_attn_mask_cte(self, attention_mask, chunk_size):
        # Create blockwise attention mask. For example, n_positions = 4, chunk_size = 2
        # mask will be
        # [[1 0 0 0]
        #  [1 1 0 0]
        #  [0 0 1 0]
        #  [0 0 1 1]]
        block_pos = torch.abs(
            (torch.arange(self.n_positions).unsqueeze(0) // chunk_size)
            - (torch.arange(self.n_positions).unsqueeze(1) // chunk_size)
        )
        token_pos = torch.arange(self.n_positions).unsqueeze(0) - torch.arange(self.n_positions).unsqueeze(1)
        mask = (block_pos == 0) & (token_pos <= 0)
        mask = mask[None, None, :, :].expand(self.batch_size, 1, self.n_positions, self.n_positions).to(device=attention_mask.device)
        return mask

    def _create_windowed_attn_mask_cte(self, attention_mask, window_size) -> torch.Tensor:
        # Create a causal, window attention mask. E.g. n = 5, window_size = 2, mask is:
        #  [[1 0 0 0 0]
        #   [1 1 0 0 0]
        #   [0 1 1 0 0]
        #   [0 0 1 1 0]
        #   [0 0 0 1 1]]
        i = torch.arange(self.n_positions, device=attention_mask.device).unsqueeze(1)
        j = torch.arange(self.n_positions, device=attention_mask.device).unsqueeze(0)
        mask = (j <= i) & (j >= (i - window_size + 1))  # Create mask: causal and within window
        mask = mask[None, None, :, :].expand(self.batch_size, 1, self.n_positions, self.n_positions)
        return mask

    def _create_chunked_prefill_attn_mask(
        self,
        attention_mask: torch.Tensor,
        is_for_context_encoding: bool,
        query_lens: torch.Tensor,
        key_lens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if is_for_context_encoding:
            # CTE usecase, and it only needs to generate a mask for the active
            # part, because the mask for the prior part will be generated by
            # the chunked prefill scheduler
            max_query_len = attention_mask.shape[-1]
            causal_mask = attn_utils.create_block_diagonal_attn_mask(
                query_lens, query_lens, max_query_len, max_query_len, is_prior=False,
            )
            num_query, num_key = causal_mask.shape
            return causal_mask.reshape(1, 1, num_query, num_key)
        else:
            # TKG usecase
            batch_size = query_lens.shape[0]
            computed_context_lens = key_lens - query_lens

            kv_cache_len = attention_mask.shape[1]  # (batch_size, kv_cache_len)
            arange_mask = torch.arange(kv_cache_len, device=attention_mask.device)
            arange_mask = arange_mask.expand(batch_size, kv_cache_len)
            causal_mask = arange_mask < computed_context_lens
            causal_mask = causal_mask[:, None, None, :]
            return causal_mask

    def _create_spec_attn_mask(self, attention_mask):
        return (
            attention_mask[:, None, None, :]
            .expand(self.batch_size, 1, self.speculation_length, self.n_positions)
            .to(torch.bool)
        )

    def _create_simple_attn_mask(self, attention_mask):
        batch_size = attention_mask.shape[0]

        return (
            attention_mask[:, None, None, :].expand(batch_size, 1, 1, self.n_positions).to(torch.bool)
        )

    def _create_chunked_attn_mask_tkg(
        self, attention_mask, chunk_size, position_ids
    ):
        # Create tkg chunk-size attention mask for prior tokens based on the current position ids
        # For example with chunk_size = 4, seq_len 8:
        # position_id = 4: we are in the second chunk -> mask = [0 0 0 0]: kv cache is reset because we have entered a new chunk
        # position_id = 3: we are in the first chunk -> mask =  [1 1 1 0]: attend to all prior tokens in chunk 1. The active token is the last token in chunk 1.
        # position_id = 5: we are in the second chunk -> mask = [1 0 0 0]: attend to first token in chunk 2
        # TODO: For TKG, we will compute for only current chunk of cache and mask will have chunk-size length
        batch_size = attention_mask.shape[0]
        local_position_ids_in_chunk = position_ids % chunk_size
        temp_array = torch.arange(chunk_size)[None, :].expand(batch_size, chunk_size).to(attention_mask.device)
        attention_mask = temp_array < local_position_ids_in_chunk
        return attention_mask[:, None, None, :]

    def _create_windowed_attn_mask_tkg(
        self, attention_mask, window_size, position_ids
    ):
        # Create tkg mask for sliding window. E.g.:
        # position = 3, window_size = 4 -> mask = [1,1,1,0]
        # position = 5, window_size = 4 -> mask = [1,0,1,1]
        batch_size, _ = attention_mask.shape
        pos = position_ids[:, 0]
        idx = torch.arange(window_size, device=attention_mask.device).unsqueeze(0)
        base_mask = idx < pos.unsqueeze(1)  # for input_len <= window_size

        full_mask = torch.ones((batch_size, window_size), dtype=torch.bool, device=attention_mask.device)
        zero_pos = pos % window_size
        zero_mask = idx == zero_pos.unsqueeze(1)
        full_mask = torch.where(zero_mask, False, full_mask)  # for input_len > window_size

        seq_less_than_window = pos < window_size
        final_mask = torch.where(seq_less_than_window.unsqueeze(1), base_mask, full_mask)
        return final_mask[:, None, None, :]

    def create_attn_mask(
        self,
        attention_mask,
        is_for_context_encoding,
        is_for_speculation,
        position_ids=None,
        **kwargs,
    ):
        if self.is_chunked_prefill:
            return self._create_chunked_prefill_attn_mask(
                attention_mask, is_for_context_encoding, **kwargs
            )

        # When we have mixed attention, we pass models both a global and local mask.
        # This function always generates a global mask.
        if is_for_context_encoding:
            if self.sliding_window and not self.has_mixed_attn:
                return self._create_windowed_attn_mask_cte(attention_mask, self.sliding_window)
            else:
                return self._create_context_attn_mask(attention_mask)
        else:
            if self.sliding_window and not self.has_mixed_attn:
                return self._create_windowed_attn_mask_tkg(attention_mask, self.sliding_window, position_ids)

        if self.neuron_config.attn_block_tkg_nki_kernel_enabled and self.is_prefix_caching:
            assert not is_for_context_encoding
            assert position_ids is not None, (
                "position_ids is required by gen_cache_mask_for_attention_tkg_kernel "
                "to infer current cache length."
            )
            grid = (nc(self.neuron_config.logical_nc_config),)
            cache_len = position_ids
            # Eagle draft's context/cache length is 1 shorter than target,
            # we have already deducted the position_ids for draft at this point.
            # But for Block KV, the first token of the first cache block is dummy for Eagle draft,
            # so the cache length is still the same between draft and target.  So add back 1 here.
            if self.neuron_config.is_block_kv_layout and self.neuron_config.is_eagle_draft:
                cache_len += 1
            tkg_kernel_mask = gen_cache_mask_for_attention_tkg_kernel[grid](
                cache_len=cache_len,
                num_heads=(self.num_attention_heads + self.tp_degree - 1) // self.tp_degree,
                S_tkg=self.speculation_length if is_for_speculation else 1,
                S_ctx=self.n_positions,
                blk_len=(
                    self.neuron_config.pa_block_size if self.neuron_config.is_block_kv_layout else 0
                ),
            )
            if self.neuron_config.is_eagle_draft:
                # First eagle head position masked out.
                tkg_kernel_mask[:, :, :, 0] = False
            return tkg_kernel_mask
        if is_for_speculation:
            return self._create_spec_attn_mask(attention_mask)
        return self._create_simple_attn_mask(attention_mask)

    def _medusa_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
    ):
        # TODO: This will not work for a context encoding model with bucket size
        # equal to the medusa speculation length
        is_for_context_encoding = (
            input_ids.shape[-1] > 1 and input_ids.shape[-1] != self.medusa_speculation_length
        )
        is_for_medusa_speculation = input_ids.shape[-1] == self.medusa_speculation_length

        medusa_metadata = {
            "current_length": current_length,
            "accepted_indices": accepted_indices,
        }

        # Prepare attention mask(s)
        attention_mask = self.create_attn_mask(
            attention_mask,
            is_for_context_encoding,
            False,
        )
        active_mask = None
        if is_for_medusa_speculation:
            medusa_mask = medusa_mask[0].bool()
            active_mask = medusa_mask[None, None, :, :].expand(
                self.batch_size, 1, self.medusa_speculation_length, self.medusa_speculation_length
            )

        hidden_states, updated_kv_cache = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            active_mask=active_mask,
            adapter_ids=adapter_ids,
            update_cache=True,
            is_for_context_encoding=is_for_context_encoding,
            medusa_metadata=medusa_metadata,
            seq_ids=seq_ids,
            scatter_index=scatter_index,
        )

        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            if position_ids.shape[-1] == self.medusa_speculation_length:
                index = torch.min(position_ids)
                index = torch.arange(
                    index, index + self.medusa_speculation_length, device=hidden_states.device
                )
                index = index[None, :, None].expand(
                    self.batch_size, self.medusa_speculation_length, self.hidden_size
                )
                hidden_states = torch.gather(hidden_states, dim=1, index=index)
            else:
                # simple token generation
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
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

        medusa_logits = [logits] + [
            head(hidden_states).float()
            for head in [
                getattr(self, f"medusa_head_{i}")
                for i in range(self.neuron_config.num_medusa_heads)
            ]
        ]
        stacked_logits = torch.stack(medusa_logits, dim=0)

        if is_for_context_encoding:
            result = [
                self.sampler(
                    stacked_logits[i : i + 1, -1, :].squeeze(0),
                    sampling_params,
                    rank_id=self.rank_util.get_rank(),
                )
                for i in range(self.neuron_config.num_medusa_heads + 1)
            ]
            res = torch.stack(result, dim=0)  # 5, 1, 10
        else:
            result = [
                self.sampler(
                    stacked_logits[i : i + 1].squeeze(0),
                    sampling_params,
                    rank_id=self.rank_util.get_rank(),
                )
                for i in range(self.neuron_config.num_medusa_heads + 1)
            ]
            res = torch.stack(result, dim=0)  # 5, 1, 64, 10

        return [res] + updated_kv_cache

    def _eagle_token_tree_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        scatter_index=None,
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        active_mask=None,
        rotary_position_id=None,
    ):
        # TODO: This will not work for a context encoding model with bucket size
        # equal to the speculation length

        is_for_context_encoding = (
            input_ids.shape[-1] > 1
            and input_ids.shape[-1] != self.speculation_length
            and attention_mask.dim() != 4
        )

        is_for_token = attention_mask.dim() == 4

        assert is_for_token != is_for_context_encoding
        if is_for_token:
            is_for_context_encoding = False
            token_tree_input_len = input_ids.shape[-1]

        # Prepare attention mask(s)
        attention_mask = self._create_token_tree_attn_mask(
            attention_mask, is_for_context_encoding, is_for_token
        )

        if active_mask is not None:
            if active_mask.shape[-1] == self.speculation_length:
                active_mask = active_mask.reshape(
                    self.batch_size, 1, self.speculation_length, self.speculation_length
                ).to(device=attention_mask.device, dtype=torch.bool)
        else:
            if is_for_token:
                active_mask = torch.eye(
                    token_tree_input_len,
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                active_mask = active_mask[None, None, :, :].expand(
                    self.batch_size, 1, token_tree_input_len, token_tree_input_len
                )

        # FD masks
        active_mask_2d = None

        rotary_position_ids = None
        if rotary_position_id is not None:
            rotary_position_ids = rotary_position_id

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            prev_hidden=prev_hidden,
            rotary_position_ids=rotary_position_ids,
            is_for_context_encoding=is_for_context_encoding,
            seq_ids=seq_ids,
            scatter_index=scatter_index,
            kv_active_mask=active_mask_2d,
            kvcache_buffer=kv_cache,
            # KV cache is update here only when it is drafting stage
            update_cache=self.neuron_config.is_eagle_draft,
        )

        full_hidden_states = hidden_states

        if self.neuron_config.is_eagle_draft:
            updated_kv_cache = past_key_values
        else:
            updated_kv_cache = None

        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            index = torch.min(position_ids, dim=1, keepdim=True).indices
            index = torch.arange(
                index[0, 0], index[0, 0] + position_ids.shape[-1], device=hidden_states.device
            )
            index = index.unsqueeze(1).expand(
                self.batch_size, position_ids.shape[-1], self.hidden_size
            )
            hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        res = logits
        if self.on_device_sampling:
            # perform sampling on Neuron to get tokens
            # FIXME, logits[:, -1, :] is not correct for speculation model, this is a tempory fix.
            if is_for_token and not self.neuron_config.on_device_sampling_config.do_sample:
                # res = nxd_argmax(tensor=logits, dim=2, gather_dim=2, keepdim=False)
                # res = res.to(torch.int32)
                # sampling done in ...
                pass
            elif (
                is_for_context_encoding
                or not self.neuron_config.enable_eagle_speculation
                or not self.neuron_config.on_device_sampling_config.do_sample
            ):
                res = self.sampler(logits[:, -1, :], sampling_params)
                res = res.to(torch.int32)

        outputs = [res]
        if self.neuron_config.output_logits:
            logits = _gather_along_dim(
                logits,
                partition_dim=2,
                process_group=get_tp_group(self.config),
            )
            outputs += [logits]
        if updated_kv_cache is not None:
            outputs += updated_kv_cache
        else:
            outputs += [None]

        outputs = outputs + [full_hidden_states] + [past_key_values]

        return outputs

    def set_none_if_empty(self, param):
        if param is None:
            return param
        if torch.is_tensor(param) and param.numel() == 0:
            # We use empty tensors instead of None as None
            # is not yet a supported input type.
            # Will be removed once kwargs support is in.
            return None
        return param

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

        hidden_states, updated_kv_cache = self.get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
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

    def _find_first_captured_tensor_for_module(self, module_captured_tensors_dict, module_name, tensor_type, device):
        """
        Find the first tensor for a given module and tensor type (inputs/outputs).

        Modules inputs/outputs registration can create nested tensor structures with various
        suffixes (e.g., .0, .1, .hidden_states, .kwargs.arg_name, etc.).

        Since module_captured_tensors_dict is an OrderedDict and registration happens
        deterministically, we preserve the natural registration order without sorting.
        The module hierarchy is properly separated by dots, so simple prefix matching
        is sufficient to find the exact module's tensors.

        Args:
            module_captured_tensors_dict: OrderedDict of all captured module tensors
            module_name: Name of the module to search for
            tensor_type: Either "inputs" or "outputs"
            device: Device to create fallback tensor on if no tensor is found

        Returns:
            torch.Tensor: The first tensor found for the module/type, or an empty tensor fallback
        """
        exact_prefix = f"{module_name}.{tensor_type}"

        # Return the first tensor that matches our exact prefix
        # Since we're using OrderedDict, iteration preserves registration order
        for key, tensor in module_captured_tensors_dict.items():
            if key.startswith(exact_prefix):
                return tensor

        # Fallback: create empty tensor if no tensor found
        # This ensures consistent output structure for tracing
        return torch.zeros(1, dtype=torch.bfloat16, device=device)

    def _get_captured_tensors(
        self,
        device
    ):
        """
        Returns tensors captured during model execution based on tensor_capture_config settings.

        This function handles two types of tensor capture:
        1. Module tensors: Captured from specific modules defined in modules_to_capture
        2. Manual tensors: Manually registered tensors during execution

        Args:
            device: Device to create padding tensors on (typically the same as model outputs)

        Returns:
            List of captured tensors. For manual tensors, padding is added to ensure
            the list has exactly max_intermediate_tensors elements for input/output aliasing.
        """
        registry = TensorRegistry.get_instance()
        captured_tensors = []

        # 1. Process module tensors - capture outputs and optionally inputs from specified modules
        if self.neuron_config.tensor_capture_config.modules_to_capture:
            module_captured_tensors_dict = registry.get_module_tensors()
            for module_name in self.neuron_config.tensor_capture_config.modules_to_capture:
                # Get first output tensor for this module
                output_tensor = self._find_first_captured_tensor_for_module(
                    module_captured_tensors_dict, module_name, "outputs", device
                )
                captured_tensors += [output_tensor]

                # Get first input tensor if capture_inputs is enabled
                if self.neuron_config.tensor_capture_config.capture_inputs:
                    input_tensor = self._find_first_captured_tensor_for_module(
                        module_captured_tensors_dict, module_name, "inputs", device
                    )
                    captured_tensors += [input_tensor]

        # Gather the manually registered tensors
        manual_captured_tensors = None
        if self.neuron_config.tensor_capture_config.max_intermediate_tensors:
            manual_captured_tensors_dict = registry.get_manual_tensors()
            manual_captured_tensors = []

            for key, tensor in manual_captured_tensors_dict.items():
                manual_captured_tensors.append(tensor)

            # Slice the manually captured tensors to max_intermediate_tensors
            # This is necessary to maintain a fixed-size output structure for input/output aliasing
            if len(manual_captured_tensors) > self.neuron_config.tensor_capture_config.max_intermediate_tensors:
                discarded_count = len(manual_captured_tensors) - self.neuron_config.tensor_capture_config.max_intermediate_tensors
                logging.warning(
                    f"Number of manually captured tensors ({len(manual_captured_tensors)}) exceeds max_intermediate_tensors "
                    f"({self.neuron_config.tensor_capture_config.max_intermediate_tensors}). "
                    f"Discarding {discarded_count} tensors."
                )
                manual_captured_tensors = manual_captured_tensors[:self.neuron_config.tensor_capture_config.max_intermediate_tensors]

            # Add padding tensors if we have fewer than max_intermediate_tensors
            # This is necessary to maintain a fixed-size output structure for input/output aliasing
            if len(manual_captured_tensors) < self.neuron_config.tensor_capture_config.max_intermediate_tensors:
                padding_count = self.neuron_config.tensor_capture_config.max_intermediate_tensors - len(manual_captured_tensors)
                for _ in range(padding_count):
                    # Use a nan value to easily identify padding tensors
                    padding_tensor = torch.full((1,), float('nan'), dtype=torch.bfloat16, device=device)
                    manual_captured_tensors.append(padding_tensor)

        # Add manual tensors to the output list
        if manual_captured_tensors:
            captured_tensors += manual_captured_tensors

        # Clear the registry for the next model execution
        registry.clear()
        return captured_tensors

    def _sample_on_device(
        self,
        logits,
        sampling_params,
        is_for_speculation,
        is_for_context_encoding,
    ):
        assert self.on_device_sampling

        res = logits
        # perform sampling on Neuron to get tokens
        # FIXME, logits[:, -1, :] is not correct for speculation model, this is a tempory fix.
        if is_for_speculation and not self.neuron_config.on_device_sampling_config.do_sample:
            res = nxd_argmax(tensor=logits, dim=2, gather_dim=2, keepdim=False)
            res = res.to(torch.int32)
        elif (
            is_for_context_encoding
            or not self.neuron_config.enable_eagle_speculation
            or not self.neuron_config.on_device_sampling_config.do_sample
        ):
            if is_for_context_encoding and self.is_chunked_prefill:
                sampling_inputs = logits[0, :, :]
            else:
                sampling_inputs = logits[:, -1, :]
            res = self.sampler(
                sampling_inputs, sampling_params, rank_id=self.rank_util.get_rank()
            )
            res = res.to(torch.int32)
        # Otherwise we return the full logits for multinomial sampling in spec decoding
        return res

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask) -> torch.Tensor:
        """
        If vision_embeddings & vision_mask is supplied in the forward, this hook
        is invoked. This function takes in a input embedding and lets you encode
        vision embedding into it.
        """
        raise NotImplementedError("encode_vision_to_input is not implemented. "
                                  "Implement it in the child of NeuronBaseModel")

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape[:2]
        if self.config.neuron_config.layer_boundary_markers:
            input_ids = ModuleMarkerStartWrapper()(input_ids)

        past_key_values_length = 0
        if past_key_values is not None:
            # Check V cache's seqlen because K cache may be transposed.
            past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = (
                self.embed_tokens(input_ids)
                if not is_lora_module(self.embed_tokens)
                else self.embed_tokens(input_ids, adapter_ids=adapter_ids)
            )

        if (vision_embeddings is not None) and (vision_mask is not None):

            if vision_embeddings.dtype != self.config.neuron_config.torch_dtype:
                vision_embeddings = vision_embeddings.to(self.config.neuron_config.torch_dtype)
            if is_for_context_encoding:
                inputs_embeds = self.encode_vision_to_input(inputs_embeds, vision_embeddings, vision_mask)

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

        # NeuronLlamaModel class manages the KV cache. So the attention_mask will be generated and passed
        # through to LlamaModel. We override the HF's code that generates attention mask because HF does
        # not support left aligned RHS padding. This enables Neuron to achieve higher performance and
        # extensibility.
        #
        # 4d mask is passed through the layers
        # attention_mask = _prepare_4d_causal_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )

        if self.sequence_parallel_enabled:
            self.validate_sequence_parallel(seq_length)

        hidden_states = self.process_sequence_parallel_hidden_states(
            inputs_embeds, seq_length, kwargs.get("active_block_table", None)
        )

        if self.neuron_config.is_eagle_draft:
            hidden_states = self.process_eagle_draft_hidden_states(
                hidden_states, prev_hidden, seq_length, kwargs.get("active_block_table", None)
            )

        if self.config.neuron_config.layer_boundary_markers:
            hidden_states, position_ids = ModuleMarkerEndWrapper()(hidden_states, position_ids)

        update_kv_per_layer = update_cache and (
            self.neuron_config.layer_boundary_markers
            or (
                self.neuron_config.attn_block_tkg_nki_kernel_cache_update
                and not is_for_context_encoding
            )
        )

        # decoder layers
        next_decoder_cache = [] if update_kv_per_layer else ()
        cos_cache = None
        sin_cache = None

        cache_size = (
            get_cache_size(self.n_positions, self.num_cores_per_group, is_for_context_encoding)
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )
        if self.attention_chunk_size:
            cache_size = self.attention_chunk_size
        get_kv_per_layer = False
        active_block_table = None if 'active_block_table' not in kwargs else kwargs['active_block_table']
        empty_active_block_table = True if active_block_table is None else len(active_block_table.shape) == 1
        may_have_prefix = self.is_prefix_caching and is_for_context_encoding and not empty_active_block_table
        is_for_chunked_prefill = self.is_block_kv_layout and self.neuron_config.is_chunked_prefill
        if may_have_prefix or is_for_chunked_prefill or not is_for_context_encoding:
            if not self.config.neuron_config.layer_boundary_markers:
                past_key_values = self.kv_mgr.get_cache(
                    seq_ids=seq_ids,
                    seq_len=cache_size,
                    is_for_context_encoding=is_for_context_encoding,
                    **kwargs,
                )
            else:
                get_kv_per_layer = True

        residual = None
        if self.attention_chunk_size and not self.has_mixed_attn:
            attention_mask = local_attn_mask
            local_attn_mask = None
        for idx, decoder_layer in enumerate(self.layers):
            if self.config.neuron_config.layer_boundary_markers:
                hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                rotary_position_ids=rotary_position_ids,
                kv_mgr=self.kv_mgr,
                get_kv_per_layer=get_kv_per_layer,
                update_kv_per_layer=update_kv_per_layer,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                residual=residual,
                local_mask=local_attn_mask,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            kv = layer_outputs[1]
            if update_kv_per_layer:
                next_decoder_cache += kv
            else:
                next_decoder_cache += (kv,)
            cos_cache, sin_cache = layer_outputs[2:4]
            residual = layer_outputs[4]
            if self.config.neuron_config.layer_boundary_markers:
                k, v = kv
                hidden_states, k, v = ModuleMarkerEndWrapper()(hidden_states, k, v)
                kv = (k, v)
                cos_cache, sin_cache = None, None

        if update_cache and not update_kv_per_layer:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                **kwargs,
            )

        if self.config.neuron_config.layer_boundary_markers:
            hidden_states = ModuleMarkerStartWrapper()(hidden_states)

        if self.neuron_config.qkv_kernel_fuse_residual_add and is_for_context_encoding:
            # residual add for the last layer which cannot be fused with qkv
            # TODO: allow fusing residual add for tokengen
            hidden_states = residual + hidden_states

        if not self.neuron_config.is_eagle_draft:
            hidden_states = self.norm(hidden_states)

        self.full_hidden_states = None
        if self.neuron_config.enable_eagle_speculation:
            self.full_hidden_states = hidden_states  # (B, S/TP, H)

        if self.sequence_parallel_enabled:
            if is_for_context_encoding and self.neuron_config.cp_degree > 1 and self.neuron_config.cp_degree == 8 and (self.neuron_config.tp_degree // self.neuron_config.cp_degree) == 8:
                ordering = get_flattened_inverted_tp_cp_group_mesh(self.neuron_config.tp_degree, self.neuron_config.cp_degree)
                position_ids = chunk_and_reorder_tensor(position_ids, ordering, 1)

            physical_position_ids = position_ids
            active_block_table = kwargs.get("active_block_table", None)
            has_prefix = active_block_table is not None and len(active_block_table.shape) > 1
            if has_prefix and self.neuron_config.is_eagle_draft:
                physical_position_ids += 1

            hidden_states = seq_parallel_slice_last_token(
                hidden_states,
                physical_position_ids,
                self.sequence_dimension,
                self.batch_size,
                self.hidden_size,
                kwargs["num_queries"],
                self.neuron_config,
                self.config,
            )  # (B, 1, H)
            self.sliced_hidden = True

        if self.config.neuron_config.layer_boundary_markers:
            hidden_states = ModuleMarkerEndWrapper()(hidden_states)
            return (hidden_states, next_decoder_cache)

        return (hidden_states, next_decoder_cache)

    def validate_sequence_parallel(self, seq_length):
        # SP is enabled only for context encoding.
        tp_group = get_tp_group(self.config)
        if tp_group is None:
            tp_group = get_tensor_model_parallel_group()
        tp_group_size = tp_group.size()
        assert seq_length % tp_group_size == 0, (
            f"When sequence parallel is enabled, context length ({seq_length}) "
            f"must be divisible by TP group size ({tp_group_size})"
        )

    def process_sequence_parallel_hidden_states(
        self,
        inputs_embeds: torch.FloatTensor,
        seq_length: int,
        active_block_table: torch.IntTensor = None,
    ) -> torch.Tensor:
        """
        Process input embeddings with sequence parallelism for transformer models.

        This method handles the sharding of hidden states across neuron cores when
        sequence parallelism is enabled, with special handling for different configurations
        including Eagle Draft mode and custom pipeline tiling factors.

        Notes:
            - For cc_pipeline_tiling_factor == 1, uses simple reduce scatter
            - For other tiling factors, implements custom tiled reduction operations
            - Handles both vocab_parallel and non-vocab_parallel cases
        """
        if self.sequence_parallel_enabled:
            if self.neuron_config.tile_cc:
                tp_size = self.neuron_config.tp_degree
                shape = list(inputs_embeds.shape)
                partition_dim = self.sequence_dimension
                assert shape[partition_dim] % tp_size == 0
                shape[partition_dim] //= tp_size
                if self.neuron_config.vocab_parallel:
                    tiled_cc_op, compute_op, cc_factor = (_traced_tiled_rs, xm.REDUCE_SUM, 1)
                else:
                    tiled_cc_op, compute_op, cc_factor = (_traced_spmd_tiled_rs, xm.REDUCE_MAX, self.neuron_config.cc_pipeline_tiling_factor)
                # Create in the shape used by consumer QKV/MLP kernel.
                hidden_states = torch.empty(shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                tiled_cc_op[(CCPipeline(cc_factor),)](
                    inputs_embeds, hidden_states, cc_dim=partition_dim,
                    tp_rank=tp_size, op=compute_op)
            else:
                shift_target_hidden = self.is_prefix_caching and len(active_block_table.shape) > 1
                if self.neuron_config.is_eagle_draft and (
                    seq_length < self.neuron_config.weight_gather_seq_len_threshold
                    or shift_target_hidden
                ):
                    hidden_states = inputs_embeds
                else:
                    # TODO: Replace this with rankid + scatter call once supported
                    hidden_states = _reduce_scatter_along_dim(
                        inputs_embeds,
                        self.sequence_dimension,
                        xm.REDUCE_SUM if self.neuron_config.vocab_parallel else xm.REDUCE_MAX,
                        process_group=get_tp_group(self.config),
                    )
        else:
            hidden_states = inputs_embeds
        return hidden_states

    def process_eagle_draft_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        prev_hidden: torch.FloatTensor,
        seq_length: int,
        active_block_table: torch.IntTensor = None,
    ) -> torch.Tensor:
        """
        Process hidden states specifically for Eagle Draft mode, handling sequence parallelism
        and weight gathering based on sequence length thresholds.

        This method manages the combination of current and previous hidden states, applying
        weight gathering optimization when appropriate. It includes special handling for
        sequence parallel configurations and different pipeline tiling factors.

        Notes:
            - Weight gathering is enabled for sequences longer than the configured threshold
            - When sequence parallelism is enabled but weight gathering is not used i.e., for smaller buckets,
            the method gathers from sequence parallel regions since the eagle fc layer is ColumnParallel
            - Applies reduce scatter operation across sequence dimension for smaller buckets
            to shard the hidden states again so that the shapes are compatible in GQA and G_outproj modules
        """
        weight_gather = False
        shift_target_hidden = self.is_prefix_caching and len(active_block_table.shape) > 1
        if self.sequence_parallel_enabled:
            if seq_length >= self.neuron_config.weight_gather_seq_len_threshold and not shift_target_hidden:
                weight_gather = True
            else:
                if self.neuron_config.cc_pipeline_tiling_factor != 1:
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states,
                        self.sequence_dimension,
                        process_group=get_tp_group(self.config),
                    )

                prev_hidden = gather_from_sequence_parallel_region(
                    prev_hidden,
                    self.sequence_dimension,
                    process_group=get_tp_group(self.config),
                )
        if shift_target_hidden:
            prev_hidden = torch.cat(
                (
                    prev_hidden[:, self.neuron_config.pa_block_size - 1:, :],
                    prev_hidden[:, :self.neuron_config.pa_block_size - 1, :]
                ), dim=1)
        concat_states = torch.cat((hidden_states, prev_hidden), dim=2)
        hidden_states = self.fc.forward_wg(concat_states, weight_gather)
        if not weight_gather and self.sequence_parallel_enabled:
            hidden_states = _reduce_scatter_along_dim(
                hidden_states,
                self.sequence_dimension,
                xm.REDUCE_MAX,
                process_group=get_tp_group(self.config),
            )
        return hidden_states

    def update_weights_for_lora(self, model_sd):
        return self.lora_checkpoint.update_weights_for_lora(self, model_sd)


class NeuronFusedSpecModel(nn.Module):
    """
    Class to handle fused speculation flow
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.draft_neuron_config = config.fused_spec_config.draft_config.neuron_config
        self.worker_cls = config.fused_spec_config.worker_cls
        self.n_positions = config.neuron_config.n_positions
        self.batch_size = config.neuron_config.batch_size
        self.hidden_size = config.hidden_size
        if config.neuron_config.enable_eagle_speculation:
            self.hidden_state_rolling_buffer = HiddenStateRollingBuffer(
                config.neuron_config.max_batch_size,
                config.neuron_config.speculation_length * 2,
                self.hidden_size,
                dtype=config.neuron_config.torch_dtype,
                apply_seq_ids_mask=self.neuron_config.apply_seq_ids_mask,
            )

        config.fused_spec_config.draft_config.neuron_config.use_draft_group = True
        config.fused_spec_config.draft_config.neuron_config.quantized_mlp_kernel_enabled = False

        self.draft_model = self.worker_cls(config.fused_spec_config.draft_config)
        self.target_model = self.worker_cls(config)

        # currently we enforce draft to be greedy
        draft_config = copy.deepcopy(config.fused_spec_config.draft_config.neuron_config)
        draft_config.on_device_sampling_config.do_sample = False
        draft_config.on_device_sampling_config.dynamic = False
        self.draft_sampler = Sampler(draft_config)
        self.target_sampler = Sampler(config.neuron_config)
        self.greedy = not config.neuron_config.on_device_sampling_config.do_sample

        # async related
        self.async_mode = self.neuron_config.async_mode
        self.acceptance_padding_token = 0

        # Prefix caching check
        self.is_prefix_caching = config.neuron_config.is_prefix_caching

        if self.config.neuron_config.enable_token_tree:
            assert self.config.neuron_config.token_tree_config
            self.token_tree = TokenTree(self.neuron_config.token_tree_config)

    def _select_from(self, to_indices, from_indices, from_values):
        if to_indices.ndim > from_indices.ndim:
            from_indices = from_indices[:, :, None].expand(to_indices.shape)
            from_values = from_values[:, :, None].expand(to_indices.shape)
            eq = torch.eq(to_indices, from_indices).to(from_values.dtype)
            to_values = from_values * eq
        elif to_indices.ndim < from_indices.ndim:
            to_indices = to_indices[:, :, None].expand(from_indices.shape)
            eq = torch.eq(to_indices, from_indices)
            to_values = torch.where(eq, from_values, 0)
            to_values = torch.sum(to_values, dim=2)

        return to_values

    def _adjust_target_probs(self, draft_probs, draft_indices, target_probs, target_indices, k):
        sliced_target_indices = target_indices[:, :k, :]
        sliced_target_probs = target_probs[:, :k, :]
        last_target_probs = target_probs[:, k : k + 1, :]

        adjusted_draft_probs = self._select_from(sliced_target_indices, draft_indices, draft_probs)
        adjusted_target_probs = sliced_target_probs - adjusted_draft_probs
        adjusted_target_probs = torch.clamp(adjusted_target_probs, min=0)

        adjusted_sum = torch.sum(adjusted_target_probs, dim=2, keepdim=True)
        # TODO: need to fix this!!
        is_zero = torch.lt(adjusted_sum, 1e-30)
        adjusted_sum = torch.where(is_zero, 1.0, adjusted_sum)
        adjusted_target_probs = torch.div(adjusted_target_probs, adjusted_sum)
        adjusted_target_probs = torch.where(is_zero, 1.0, adjusted_target_probs)
        adjusted_target_probs = torch.cat([adjusted_target_probs, last_target_probs], dim=1)

        return adjusted_target_probs

    def _speculative_mask(
        self, draft_ids, draft_probs_indices, draft_probs, target_probs_indices, target_probs
    ):
        target_probs = self._select_from(draft_ids, target_probs_indices, target_probs)
        # we don't need this for greedy draft
        # draft_probs = self.select_from(draft_ids, draft_probs_indices, draft_probs)

        ratio = torch.div(target_probs, draft_probs)
        ratio = torch.clamp(ratio, max=1.0).to(torch.float32)
        random = rand_like(ratio)
        accepted_mask = torch.lt(random, ratio).to(torch.int)
        accepted_cumsum = torch.cumsum(accepted_mask, dim=1)

        batch_size, k = ratio.shape

        positions = torch.range(1, k, dtype=accepted_cumsum.dtype, device=ratio.device)[
            None, :
        ].expand(ratio.shape)
        accepted_mask = torch.eq(accepted_cumsum, positions)
        accepted_mask = torch.nn.functional.pad(accepted_mask, (0, 1), value=False)
        return accepted_mask

    def _speculative_token_selection(
        self,
        draft_ids,
        target_ids,
        draft_probs_indices,
        draft_probs,
        target_probs_indices,
        target_probs,
    ):
        accepted_mask = self._speculative_mask(
            draft_ids,
            draft_probs_indices,
            draft_probs,
            target_probs_indices,
            target_probs,
        )

        draft_ids = torch.nn.functional.pad(draft_ids, (0, 1), value=0)
        tokens = torch.where(accepted_mask, draft_ids.to(torch.int64), target_ids.to(torch.int64))

        pad_token_id = self.config.pad_token_id

        positions = torch.range(0, tokens.shape[1] - 1, device=tokens.device, dtype=tokens.dtype)[
            None, :
        ].expand(tokens.shape)
        index = torch.sum(accepted_mask.to(torch.int), dim=1, keepdim=True)
        mask = torch.ge(index, positions)
        tokens = torch.where(mask, tokens, pad_token_id)

        return tokens, index

    def _context_encoding_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
    ):
        self.draft_model.n_positions = self.n_positions
        self.target_model.n_positions = self.n_positions

        assert self.neuron_config.on_device_sampling_config

        target_outputs = self.target_model(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            slot_mapping,
            active_block_table,
            num_queries,
            computed_context_lens,
        )
        draft_outputs = self.draft_model(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            slot_mapping,
            active_block_table,
            num_queries,
            computed_context_lens,
        )
        if self.neuron_config.output_logits:
            return (
                [draft_outputs[0]]
                + [target_outputs[0]]
                + [draft_outputs[1]]
                + [target_outputs[1]]
                + draft_outputs[1:]
                + target_outputs[1:]
            )
        return [draft_outputs[0]] + [target_outputs[0]] + draft_outputs[1:] + target_outputs[1:]

    def _token_gen_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
    ):
        spec_len = self.neuron_config.speculation_length
        bs = input_ids.shape[0]

        assert self.neuron_config.on_device_sampling_config

        draft_position_ids = position_ids.expand(bs, spec_len)  # [1, 5]
        candidate_input_ids = input_ids
        target_position_ids = position_ids
        draft_attention_mask = copy.deepcopy(attention_mask)

        draft_cache = None
        num_outputs = 1 if not self.neuron_config.output_logits else 2
        draft_logits_list = []
        # 1. "k" iterations of the draft model. We use only first "k-1" tokens.
        # Extra run is for populating the kv cache
        for i in range(spec_len):
            draft_position_id = draft_position_ids[:, i : i + 1] + i
            draft_input_ids = candidate_input_ids[:, -1:]

            target_position_id = draft_position_ids[:, i : i + 1] + i + 1
            target_position_ids = torch.cat([target_position_ids, target_position_id], dim=1)

            if self.is_prefix_caching:
                current_slot_mapping = slot_mapping[:, i : i + 1]
                current_num_queries = num_queries
                current_computed_context_lens = computed_context_lens + i
                model_output = self.draft_model(
                    draft_input_ids,
                    draft_attention_mask,
                    draft_position_id,
                    seq_ids,
                    sampling_params,
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    current_slot_mapping,
                    active_block_table,
                    current_num_queries,
                    current_computed_context_lens,
                    kv_cache=draft_cache,
                )
            else:
                model_output = self.draft_model(
                    draft_input_ids,
                    draft_attention_mask,
                    draft_position_id,
                    seq_ids,
                    sampling_params,
                    kv_cache=draft_cache,
                )

            draft_outputs = model_output[0]
            draft_cache = model_output[num_outputs:]
            if self.neuron_config.output_logits:
                draft_logits_list.append(model_output[1])

            draft_attention_mask.index_fill_(
                1, draft_position_id.to(torch.int64).squeeze(), 1
            ).view(bs, -1)
            new_draft_token = draft_outputs.view(bs, -1)

            candidate_input_ids = torch.cat((candidate_input_ids, new_draft_token), dim=-1)

        # Retile the cache
        flat_draft_cache = []
        for idx in range(len(draft_cache)):
            # TODO once compiler fixes CR 158191111 we can turn back output tiling on
            # flat_draft_cache.append(draft_cache[idx].view(self.draft_model.kv_mgr.kv_shape))
            flat_draft_cache.append(draft_cache[idx])

        # 2. Run target model on the draft produced tokens
        outputs = self.target_model(
            candidate_input_ids[:, :-1],
            attention_mask,
            target_position_ids[:, :-1],
            seq_ids,
            sampling_params,
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            slot_mapping=slot_mapping,
            active_block_table=active_block_table,
            num_queries=num_queries + spec_len - 1 if self.is_prefix_caching else None,
            computed_context_lens=computed_context_lens,
        )
        target_tokens = outputs[0]
        target_cache = outputs[num_outputs:]

        if self.neuron_config.output_logits:
            draft_logits = torch.cat(draft_logits_list, dim=1)
            target_logits = outputs[1]
            return (
                [candidate_input_ids[:, 1:]]
                + [target_tokens]
                + [draft_logits]
                + [target_logits]
                + flat_draft_cache
                + target_cache
            )
        return [candidate_input_ids[:, 1:]] + [target_tokens] + flat_draft_cache + target_cache

    def _eagle_context_encoding_forward_with_prefix_caching(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        target_input_ids=None,
        target_attention_mask=None,
        target_position_ids=None,
        target_slot_mapping=None,
        target_active_block_table=None,
    ):
        """
        Eagle with block kv cache layout requires extra block computation for the target model to
        generate the required hidden states which is sent as input to the draft during context encoding.
        """
        self.draft_model.n_positions = self.n_positions
        self.target_model.n_positions = self.n_positions
        assert self.neuron_config.on_device_sampling_config

        target_outputs = self.target_model(
            target_input_ids,
            target_attention_mask,
            target_position_ids,
            seq_ids,
            sampling_params,
            torch.empty(0),  # prev_hidden
            torch.empty(0),  # adapter_ids
            torch.empty(0),  # accepted_indices
            torch.empty(0),  # current_length
            torch.empty(0),  # medusa_mask
            torch.empty(0),  # scatter_index
            target_slot_mapping,
            target_active_block_table,
            num_queries,
            computed_context_lens,
        )
        last_hidden_state = target_outputs[-2]
        draft_input_ids = copy.deepcopy(input_ids)
        draft_position_ids = copy.deepcopy(position_ids)
        draft_slot_mapping = copy.deepcopy(slot_mapping)

        if len(active_block_table.shape) == 1:
            # No Cache hit during context encoding
            hidden_state = target_outputs[-1]
            gather_index = torch.arange(0, input_ids.shape[1], device=input_ids.device) + 1
            gather_index[-1] = 0
            gather_index = gather_index.expand(input_ids.shape)
            draft_input_ids = torch.gather(input_ids, 1, gather_index)
            scatter_index = num_queries + computed_context_lens - 1
            zeros = torch.zeros(scatter_index.shape, dtype=attention_mask.dtype, device=attention_mask.device)
            draft_position_ids = torch.scatter(draft_position_ids, 1, scatter_index, zeros)
            draft_slot_mapping = torch.gather(slot_mapping, 1, gather_index)
        else:
            # Target's hidden_state will be shifted in process_eagle_draft_hidden_states
            hidden_state = target_outputs[-1]
            draft_position_ids = draft_position_ids - 1

        draft_outputs = self.draft_model(
            draft_input_ids,
            attention_mask,
            draft_position_ids,
            seq_ids,
            sampling_params,
            hidden_state,
            torch.empty(0),  # adapter_ids
            torch.empty(0),  # accepted_indices
            torch.empty(0),  # current_length
            torch.empty(0),  # medusa_mask
            torch.empty(0),  # scatter_index
            draft_slot_mapping,
            active_block_table,
            num_queries,
            computed_context_lens,
        )
        num_outputs = 1 if not self.neuron_config.output_logits else 2
        draft_cache = draft_outputs[num_outputs:-2]
        target_cache = target_outputs[num_outputs:-2]

        if self.neuron_config.output_logits:
            return (
                [draft_outputs[0]]
                + [target_outputs[0]]
                + [draft_outputs[1]]
                + [target_outputs[1]]
                + draft_cache
                + target_cache
                + [last_hidden_state]
            )
        return (
            [draft_outputs[0]]
            + [target_outputs[0]]
            + draft_cache
            + target_cache
            + [last_hidden_state]
        )

    def _eagle_context_encoding_forward(
        self, input_ids, attention_mask, position_ids, seq_ids, sampling_params
    ):
        self.draft_model.n_positions = self.n_positions
        self.target_model.n_positions = self.n_positions

        assert self.neuron_config.on_device_sampling_config

        target_outputs = self.target_model(
            input_ids, attention_mask, position_ids, seq_ids, sampling_params
        )
        last_hidden_state = target_outputs[-2]
        hidden_state = target_outputs[-1]
        # Create draft args from target args
        # Draft is always running 1 position behind the target
        # So if target input is ABCDE, draft input will be BCDE

        draft_input_ids = copy.deepcopy(input_ids)
        gather_index = torch.arange(0, input_ids.shape[1], device=input_ids.device) + 1
        gather_index[-1] = 0
        gather_index = gather_index.expand(input_ids.shape)
        draft_input_ids = torch.gather(input_ids, 1, gather_index)

        draft_position_ids = copy.deepcopy(position_ids)
        scatter_index = torch.sum(attention_mask, dim=1, keepdim=True) - 1
        zeros = torch.zeros(
            scatter_index.shape, dtype=attention_mask.dtype, device=attention_mask.device
        )
        draft_position_ids = torch.scatter(draft_position_ids, 1, scatter_index, zeros)
        draft_outputs = self.draft_model(
            draft_input_ids,
            attention_mask,
            draft_position_ids,
            seq_ids,
            sampling_params,
            hidden_state,
        )
        num_outputs = 1 if not self.neuron_config.output_logits else 2
        draft_cache = draft_outputs[num_outputs:-2]
        target_cache = target_outputs[num_outputs:-2]

        if self.neuron_config.output_logits:
            return (
                [draft_outputs[0]]
                + [target_outputs[0]]
                + [draft_outputs[1]]
                + [target_outputs[1]]
                + draft_cache
                + target_cache
                + [last_hidden_state]
            )
        return (
            [draft_outputs[0]]
            + [target_outputs[0]]
            + draft_cache
            + target_cache
            + [last_hidden_state]
        )

    def _eagle_tree_token_gen_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
    ):
        """
        This is the forward pass of Token Tree based Eagle Speculative Decoding.
        The inputs for this forward pass is similar to Sequence based Eagle SD

        Token Tree based Eagle Speculative Decoding supports all valid tree structure
        for token generation. The token tree will be specified in json format.

        For example:
        {
            "0": ["1", "2"],
            "1": ["3", "4"],
            "2": ["5", "6"],
        }

        The above json specify a perfect binary tree with depth = 3.
        Leaf node can be specified for clarity but can also be ommited for simplicity.

        Currently, only greedy sampling is supported.
        """

        assert self.neuron_config.on_device_sampling_config
        assert self.token_tree.tree_config

        # Currently Only Support Greedy Sampling
        self.greedy = True

        spec_len = self.neuron_config.speculation_length
        bs = input_ids.shape[0]

        # Position ID is different for each level of the token tree, the offset is precomputed
        draft_position_offset = torch.tensor(
            self.token_tree.position_id_offset, device=position_ids.device, dtype=torch.int32
        )

        new_position_ids = position_ids[:, 0].view(bs, 1, 1) - 1 + draft_position_offset.unsqueeze(0).expand(bs, -1, -1)

        hidden_state = self.hidden_state_rolling_buffer.get_state(
            seq_ids, new_position_ids[:, 1, 0]
        )

        tree_depth = self.token_tree.depth

        draft_position_ids = new_position_ids
        candidate_input_ids = input_ids.clone()

        # Initialize attention mask for different stage in Token Tree flow
        draft_attention_mask = copy.deepcopy(attention_mask)
        target_attention_mask = copy.deepcopy(attention_mask)
        draft_update_attention_mask = copy.deepcopy(attention_mask)

        # Initialized for Target Cache update stage
        cache_scatter_indices = torch.tensor(
            self.token_tree.cache_scatter_indices, dtype=torch.int32, device=attention_mask.device
        )

        # The index to update attention mask for drafting stage
        draft_next_scatter_index = (
            torch.sum(draft_attention_mask, dim=1, keepdim=True) - 1
        ).expand(bs, spec_len)

        draft_next_scatter_index = draft_next_scatter_index + torch.arange(0, spec_len).expand(
            bs, spec_len
        )

        # Compute target Position_ids before verification stage
        target_position_ids = torch.sum(draft_attention_mask, dim=1, keepdim=True).expand(
            bs, spec_len
        ) + torch.arange(0, spec_len).expand(bs, spec_len)

        target_position_ids = target_position_ids.to(
            dtype=candidate_input_ids.dtype, device=candidate_input_ids.device
        ).expand(bs, -1)

        target_attention_mask = (
            target_attention_mask[:, None, None, :]
            .expand(bs, 1, spec_len, self.n_positions)
            .to(torch.bool)
        )
        draft_update_attention_mask = (
            draft_update_attention_mask[:, None, None, :]
            .expand(bs, 1, tree_depth, self.n_positions)
            .to(torch.bool)
        )

        draft_tree_attention_masks = self.token_tree.draft_tree_attn_mask
        full_tree_attention_mask = self.token_tree.full_tree_attn_mask

        full_tree_attention_mask = full_tree_attention_mask.to(
            device=target_attention_mask.device, dtype=torch.bool
        )

        level_node_count = self.token_tree.level_node_count

        orig_hidden = hidden_state
        draft_cache = None
        draft_logits_list = []
        num_outputs = 1 if not self.neuron_config.output_logits else 2

        drafted_num = 0
        drafted_nums = self.token_tree.drafted_nums
        topk_permute_indice = self.token_tree.topk_permute_index
        draft_hidden_gather_indice = self.token_tree.draft_hidden_gather_index

        for i in range(tree_depth - 1):
            drafted_num = drafted_nums[i]

            draft_position_id = draft_position_ids[:, i, : level_node_count[i]]
            draft_input_ids = candidate_input_ids[:, drafted_num:]

            draft_attention_mask = copy.deepcopy(attention_mask)
            draft_attention_mask = (
                draft_attention_mask[:, None, None, :]
                .expand(bs, 1, level_node_count[i], self.n_positions)
                .to(torch.bool)
            )

            # Pick draft attention mask for this tree level from precomputed list
            draft_tree_attention_mask = draft_tree_attention_masks[i].to(
                device=draft_attention_mask.device, dtype=torch.bool
            )

            draft_next_scatter_index_ = draft_next_scatter_index.unsqueeze(1).unsqueeze(2).expand(bs, 1, level_node_count[i], spec_len)
            draft_tree_attention_mask_ = draft_tree_attention_mask.unsqueeze(0).unsqueeze(1).expand(bs, 1, level_node_count[i], spec_len)

            draft_attention_mask = torch.scatter(
                draft_attention_mask,
                3,
                draft_next_scatter_index_,
                draft_tree_attention_mask_,
            )

            # Compute rotary position id for rotary embedding in Llama
            rotary_offset = i + new_position_ids[:, 0, 0].view(bs, 1)
            draft_rotary_position_ids = (
                torch.zeros_like(draft_position_id, device=draft_position_id.device) + rotary_offset
            )
            draft_rotary_position_ids = draft_rotary_position_ids.view(
                -1, draft_position_id.shape[1]
            ).long()

            logging.warning("draft_model drafting")

            model_output = self.draft_model(
                draft_input_ids,
                draft_attention_mask,
                draft_position_id,
                seq_ids,
                sampling_params,
                prev_hidden=hidden_state,
                kv_cache=draft_cache,
                rotary_position_id=draft_rotary_position_ids,
            )
            if not self.greedy:
                pass

            else:
                output_logits = model_output[0]

                # Pick largest topK value from the level and apply to all the node
                node_topks = self.token_tree.level_child[i]
                max_topk = max(node_topks)
                output_logits, output_index = nxd_topk(
                    tensor=output_logits,
                    k=max_topk,
                    dim=2,
                    gather_dim=2,
                    process_group=get_tp_group(self.draft_model.config),
                )

                # Based on the actual topK of each node, select corresponding output tokens
                selected = output_index.reshape(bs, -1)

                topk_permute_index = torch.tensor(topk_permute_indice[i], device=position_ids.device, dtype=torch.int32)
                topk_permute_index = topk_permute_index.unsqueeze(0).expand(bs, -1)
                selected = torch.gather(selected, dim=1, index=topk_permute_index)

                draft_outputs = selected[:, :level_node_count[i + 1]].to(torch.int32)

                new_draft_token = draft_outputs.reshape(bs, -1)
                candidate_input_ids = torch.cat([candidate_input_ids, new_draft_token], dim=1)

            draft_cache = model_output[num_outputs:-2]
            hidden_state = model_output[-2]

            # Prepare hidden state for next drafting forward pass, the drafted token will use parent's hidden in next forward pass
            if i != tree_depth - 2:
                draft_hidden_gather_index = torch.tensor(draft_hidden_gather_indice[i], device=position_ids.device, dtype=torch.int32)
                draft_hidden_gather_index = draft_hidden_gather_index.unsqueeze(0).unsqueeze(-1).expand(bs, draft_hidden_gather_index.shape[0], hidden_state.shape[-1])
                hidden_state = torch.gather(hidden_state, dim=1, index=draft_hidden_gather_index)

            if self.neuron_config.output_logits:
                draft_logits_list.append(model_output[1])

        if not self.greedy:
            pass

        # 2. Run target model on the draft produced tokens

        logging.warning("target_model verification")

        # Compute rotary position id for target verfication
        rotary_position_id_offset = self.token_tree.rotary_position_id_offset.to(
            device=target_position_ids.device
        )

        target_rotary_position_id = (rotary_position_id_offset + target_position_ids[:, 0].view(bs, 1)).expand(
            bs, spec_len
        )

        active_mask = full_tree_attention_mask.to(
            device=target_attention_mask.device, dtype=torch.bool
        )
        active_mask = active_mask.unsqueeze(0).unsqueeze(1).expand(bs, 1, spec_len, spec_len)

        outputs = self.target_model(
            candidate_input_ids,
            target_attention_mask,
            target_position_ids,
            seq_ids,
            sampling_params,
            active_mask=active_mask,
            rotary_position_id=target_rotary_position_id,
        )
        if not self.greedy:
            pass
        else:
            target_output_logits = outputs[0]
            target_tokens = nxd_argmax(
                tensor=target_output_logits, dim=2, gather_dim=2, keepdim=False
            )
            target_tokens = target_tokens.to(torch.int32)

        target_cache = outputs[num_outputs:-2]
        hidden_state = outputs[-2]

        # target past key values is stored for target cache update stage
        target_past_key_values = outputs[-1]
        prev_hidden = torch.cat([orig_hidden, hidden_state[:, : spec_len - 1, :]], dim=1)
        reshaped_cache = []

        for i in range(0, len(draft_cache), 2):
            reshaped_cache.append([draft_cache[i], draft_cache[i + 1]])
        draft_cache = reshaped_cache

        if not self.greedy:
            pass
        else:

            # Based on all possible paths, create tensors to compare matched token nums
            # For example:
            # target tokens: [a, b, c, d, e, f, g]
            # candidate input ids: [A, B, C, D, E, F, G]
            # possible path: [[0, 1, 3], [0, 1, 4], [0, 2, 5], [0, 2, 6]]
            # target tokens for compare: [[a, b, d], [a, b, e], [a, c, f], [a, c, g]]
            # Candidate Input Ids for compare: [[A, B, D], [A, B, E], [A, C, F], [A, C, G]]
            paths = self.token_tree.path.to(device=candidate_input_ids.device, dtype=torch.int32)
            parent_paths = self.token_tree.parent_path.to(
                device=candidate_input_ids.device, dtype=torch.int32
            )

            candidate_input_ids_comp = candidate_input_ids[:, paths]
            target_tokens_comp = target_tokens[:, parent_paths]

            index = (
                (~(candidate_input_ids_comp[:, :, 1:] == target_tokens_comp[:, :, :-1])).cumsum(
                    dim=-1
                )
                < 1
            ).sum(dim=-1)

            # Select path that has max token matched
            dest_idx = index.argmax(dim=1)
            dest_idx = dest_idx.unsqueeze(-1)

            dest_len = torch.gather(index, dim=1, index=dest_idx)

            # Output hidden state will be the hidden state of the last token in the selected path
            last_hidden_pos = dest_len
            last_hidden_index = last_hidden_pos.view(bs, 1, 1).expand(bs, 1, self.hidden_size)

        # Get permutation masks with correct device and dtype
        permute_masks = self.token_tree.path_permute_mask.to(
            device=target_tokens.device, dtype=torch.int32
        )
        parent_permute_masks = self.token_tree.parent_path_permute_mask.to(
            device=target_tokens.device, dtype=torch.int32
        )

        # Select permute mask based on accepted path
        permute_mask_gather_idx = dest_idx.reshape(bs, 1).expand(bs, permute_masks.shape[1])
        permute_mask = torch.gather(permute_masks, dim=0, index=permute_mask_gather_idx)

        cache_scatter_index = torch.gather(
            cache_scatter_indices,
            dim=0,
            index=dest_idx.view(bs, 1).expand(bs, cache_scatter_indices.shape[1]),
        )

        parent_permute_mask = torch.gather(
            parent_permute_masks,
            dim=0,
            index=dest_idx.view(bs, 1).expand(bs, parent_permute_masks.shape[1]),
        )

        gather_index = permute_mask.expand(bs, -1)
        parent_gather_index = parent_permute_mask.expand(bs, -1)
        prev_hidden_gather_index = (
            permute_mask.unsqueeze(-1).expand(bs, spec_len, self.hidden_size)
        )

        target_token_gather_index = parent_gather_index
        candidate_input_gather_index = gather_index
        target_hidden_gather_idx = (
            parent_permute_mask
            .view(bs, spec_len, 1)
            .expand(bs, spec_len, self.hidden_size)
        )

        # Permute target tokens based on accepted path, so the target token will start with
        # target tokens from accepted path
        target_tokens = torch.gather(target_tokens, dim=1, index=target_token_gather_index)
        candidate_input_ids = torch.gather(
            candidate_input_ids, dim=1, index=candidate_input_gather_index
        )

        # Prepare hidden state from target output for draft cache update
        draft_update_prev_hidden = torch.gather(prev_hidden, dim=1, index=prev_hidden_gather_index)

        # Prepare hidden state for output
        # Hack fix for if index has same shape as the src tensor, will introduce random small number
        # TODO: Raise compiler ticket for the above problem
        if tree_depth == hidden_state.shape[1]:
            hidden_state = torch.cat([hidden_state.clone(), hidden_state.clone()], dim=1)

        hidden_state = torch.gather(
            hidden_state, dim=1, index=target_hidden_gather_idx[:, :tree_depth, :]
        )
        hidden_state = torch.gather(hidden_state, dim=1, index=last_hidden_index)

        # 3 Final draft run to update KV cache. This is done after the target run since we need to send
        # the hidden states from the target output as input to the final draft run.

        logging.warning("draft_model updating")

        draft_update_position_id = target_position_ids - 1
        draft_update_active_mask = torch.full(
            (tree_depth, tree_depth),
            True,
            device=attention_mask.device,
        ).tril(diagonal=0)

        draft_update_active_mask = draft_update_active_mask[None, None, :, :].expand(
            bs, 1, tree_depth, tree_depth
        )

        model_output = self.draft_model(
            candidate_input_ids[:, :tree_depth],
            draft_update_attention_mask,
            draft_update_position_id[:, :tree_depth],
            seq_ids,
            sampling_params,
            prev_hidden=draft_update_prev_hidden[:, :tree_depth, :],
            kv_cache=draft_cache,
            active_mask=draft_update_active_mask,
            rotary_position_id=draft_update_position_id[:, :tree_depth],
        )

        draft_cache = model_output[num_outputs:-2]

        # Retile the cache
        flat_draft_cache = []
        for idx in range(len(draft_cache)):
            flat_draft_cache.append(draft_cache[idx])

        # Permute position id based on accepted path to update corresponding kv cache position
        cache_scatter_index = cache_scatter_index + new_position_ids[:, 0, 0].view(bs, 1) + 1
        target_updated_kv_cache = self.target_model.kv_mgr.update_cache(
            is_for_context_encoding=False,
            seq_ids=seq_ids,
            position_ids=cache_scatter_index,
            new_key_values=target_past_key_values,
            seq_len=self.n_positions,
            scatter_index=None,
            active_mask=None,
        )

        target_cache = target_updated_kv_cache
        if self.neuron_config.output_logits:
            draft_logits = torch.cat(draft_logits_list, dim=1)
            target_logits = outputs[1]

            logits_gather_index = permute_mask.unsqueeze(0).unsqueeze(-1).expand_as(target_logits)
            target_logits = torch.gather(target_logits, dim=1, index=logits_gather_index)

            return (
                [candidate_input_ids[:, :tree_depth]]
                + [target_tokens[:, :tree_depth]]
                + [draft_logits]
                + [target_logits[:, :tree_depth, :]]
                + flat_draft_cache
                + target_cache
                + [hidden_state]
            )
        return (
            [candidate_input_ids[:, :tree_depth]]
            + [target_tokens[:, :tree_depth]]
            + flat_draft_cache
            + target_cache
            + [hidden_state]
        )

    def _eagle_token_gen_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
    ):
        spec_len = self.neuron_config.speculation_length
        bs = input_ids.shape[0]
        hidden_state = self.hidden_state_rolling_buffer.get_state(seq_ids, position_ids)

        assert self.neuron_config.on_device_sampling_config

        # 1. Generate k-1 candidate tokens
        draft_position_ids = position_ids.expand(bs, spec_len) - 1  # [1, 5]
        candidate_input_ids = input_ids
        target_position_ids = position_ids
        draft_attention_mask = copy.deepcopy(attention_mask)
        scatter_index = torch.sum(draft_attention_mask, dim=1, keepdim=True) - 1
        zeros = torch.zeros(
            scatter_index.shape,
            dtype=draft_attention_mask.dtype,
            device=draft_attention_mask.device,
        )
        draft_attention_mask = torch.scatter(draft_attention_mask, 1, scatter_index, zeros)

        orig_hidden = hidden_state
        draft_cache = None
        draft_probs = []
        draft_logits_list = []
        num_outputs = 1 if not self.neuron_config.output_logits else 2
        for i in range(spec_len - 1):
            draft_position_id = draft_position_ids[:, i : i + 1] + i
            draft_input_ids = candidate_input_ids[:, -1:]

            target_position_id = draft_position_ids[:, i : i + 1] + i + 2
            target_position_ids = torch.cat([target_position_ids, target_position_id], dim=1)
            if self.is_prefix_caching:
                current_slot_mapping = slot_mapping[:, i : i + 1]
                model_output = self.draft_model(
                    draft_input_ids,
                    draft_attention_mask,
                    draft_position_id,
                    seq_ids,
                    sampling_params,
                    hidden_state,
                    torch.empty(0),  # adapter_ids
                    torch.empty(0),  # accepted_indices
                    torch.empty(0),  # current_length
                    torch.empty(0),  # medusa_mask
                    torch.empty(0),  # scatter_index
                    current_slot_mapping,
                    active_block_table,
                    num_queries,
                    computed_context_lens,
                    kv_cache=draft_cache,
                )
            else:
                model_output = self.draft_model(
                    draft_input_ids,
                    draft_attention_mask,
                    draft_position_id,
                    seq_ids,
                    sampling_params,
                    prev_hidden=hidden_state,
                    kv_cache=draft_cache,
                )
            if not self.greedy:
                draft_outputs, single_draft_probs = self.draft_sampler(
                    model_output[0],
                    sampling_params,
                    return_values=True,
                )
                draft_probs.append(single_draft_probs)
            else:
                draft_outputs = model_output[0]
            draft_cache = model_output[num_outputs:-1]
            hidden_state = model_output[-1]
            if self.neuron_config.output_logits:
                draft_logits_list.append(model_output[1])

            ones = torch.ones(
                draft_position_id.shape,
                dtype=draft_attention_mask.dtype,
                device=draft_attention_mask.device,
            )
            draft_attention_mask = torch.scatter(draft_attention_mask, 1, draft_position_id, ones)

            new_draft_token = draft_outputs.view(bs, -1)

            candidate_input_ids = torch.cat((candidate_input_ids, new_draft_token), dim=-1)

        if not self.greedy:
            draft_probs = torch.cat(draft_probs, dim=1)

        # 2. Run target model on the draft produced tokens
        if self.is_prefix_caching:
            outputs = self.target_model(
                candidate_input_ids,
                attention_mask,
                target_position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),  # prev_hidden
                torch.empty(0),  # adapter_ids
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                slot_mapping,
                active_block_table,
                num_queries,
                computed_context_lens,
            )
        else:
            outputs = self.target_model(
                candidate_input_ids,
                attention_mask,
                target_position_ids,
                seq_ids,
                sampling_params,
            )
        if not self.greedy:
            target_tokens, target_probs = self.target_sampler(
                outputs[0],
                sampling_params,
                return_values=True,
                rank_id=self.target_model.rank_util.get_rank(),
            )
        else:
            target_tokens = outputs[0]
        target_cache = outputs[num_outputs:-1]
        hidden_state = outputs[-1]
        prev_hidden = torch.cat([orig_hidden, hidden_state[:, : spec_len - 1, :]], dim=1)

        if not self.is_prefix_caching:
            reshaped_cache = []
            for i in range(0, len(draft_cache), 2):
                reshaped_cache.append([draft_cache[i], draft_cache[i + 1]])
            draft_cache = reshaped_cache

        # 3 Final draft run to update KV cache. This is done after the target run since we need to send
        # the hidden states from the target output as input to the final draft run.
        if self.is_prefix_caching:
            model_output = self.draft_model(
                candidate_input_ids,
                attention_mask,
                target_position_ids - 1,
                seq_ids,
                sampling_params,
                prev_hidden,
                torch.empty(0),  # adapter_ids
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                slot_mapping,
                active_block_table,
                num_queries,
                computed_context_lens,
                kv_cache=draft_cache,
            )
        else:
            model_output = self.draft_model(
                candidate_input_ids,
                attention_mask,
                target_position_ids - 1,
                seq_ids,
                sampling_params,
                prev_hidden=prev_hidden,
                kv_cache=draft_cache,
            )
        draft_cache = model_output[num_outputs:-1]

        # Retile the cache
        flat_draft_cache = []
        for idx in range(len(draft_cache)):
            # TODO once compiler fixes CR 158191111 we can turn back output tiling on
            # flat_draft_cache.append(draft_cache[idx].view(self.draft_model.kv_mgr.kv_shape))
            flat_draft_cache.append(draft_cache[idx])

        if not self.greedy:
            adjusted_target_probs = self._adjust_target_probs(
                draft_probs, candidate_input_ids[:, 1:], target_probs, target_tokens, spec_len - 1
            )
            target_ids = self.target_sampler._multinomial(adjusted_target_probs, 2)
            target_ids = torch.gather(target_tokens, 2, target_ids)
            target_ids = torch.squeeze(target_ids, 2)
            draft_ids = candidate_input_ids[:, 1:]
            sliced_target_indices = target_tokens[:, : spec_len - 1, :]
            sliced_target_probs = target_probs[:, : spec_len - 1, :]

            tokens, index = self._speculative_token_selection(
                draft_ids,
                target_ids,
                draft_ids,
                draft_probs,
                sliced_target_indices,
                sliced_target_probs,
            )
            target_tokens = tokens
            index = index[:, :, None]

        else:
            index = (
                ((~(candidate_input_ids[:, 1:] == target_tokens[:, :-1])).cumsum(dim=-1) < 1)
                .sum(dim=-1, keepdim=True, dtype=torch.int32)
                .view(self.batch_size, -1)
            )

        index = index.reshape(self.batch_size, -1, 1).expand(self.batch_size, 1, self.hidden_size)
        hidden_state = torch.gather(hidden_state, dim=1, index=index)

        if self.neuron_config.output_logits:
            draft_logits = torch.cat(draft_logits_list, dim=1)
            target_logits = outputs[1]
            return (
                [candidate_input_ids]
                + [target_tokens]
                + [draft_logits]
                + [target_logits]
                + flat_draft_cache
                + target_cache
                + [hidden_state]
            )

        return (
            [candidate_input_ids]
            + [target_tokens]
            + flat_draft_cache
            + target_cache
            + [hidden_state]
        )

    def _cte_postprocessor(
        self, context_outs, input_ids, attention_mask, position_ids, speculation_length,
        num_queries=None, computed_context_lens=None
    ):
        batch_size = input_ids.shape[0]
        if not self.is_prefix_caching:
            cur_len = torch.sum(attention_mask, dim=1).to(torch.int32)
        else:
            cur_len = num_queries + computed_context_lens
            cur_len = cur_len.to(torch.int32)

        selected_output = context_outs[1]
        selected_output = selected_output.reshape(batch_size, 1)
        padded_output = torch.cat(
            [
                selected_output,
                torch.full(
                    (batch_size, speculation_length - 1),
                    fill_value=self.acceptance_padding_token,
                    dtype=selected_output.dtype,
                    device=selected_output.device,
                ),
            ],
            dim=1,
        ).to(torch.int32)

        next_pos_ids = torch.reshape(cur_len, (batch_size, 1)).to(torch.int32)

        batch_size, _ = position_ids.shape
        sequence_length = self.neuron_config.seq_len
        position_ids_to_compare = next_pos_ids.expand(batch_size, sequence_length) - 1
        mask = (
            torch.arange(sequence_length)
            .view(1, -1)
            .expand(batch_size, sequence_length)
            .to(position_ids.device)
        )
        next_attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)

        next_input_ids = padded_output[:, :1]

        return [padded_output, next_input_ids, next_attention_mask, next_pos_ids]

    def _tkg_postprocessor(
        self,
        token_gen_outs,
        attention_mask,
        position_ids,
    ):
        candidate_tokens = token_gen_outs[0]
        target_tokens = token_gen_outs[1]
        if self.config.neuron_config.enable_eagle_speculation:
            candidate_new_tokens = candidate_tokens[:, 1:]
        else:
            candidate_new_tokens = candidate_tokens[:, :-1]

        selected_tokens = target_tokens[:, :-1]

        # this is to get contiguous matches, instead of straight matches
        n_matches = ((candidate_new_tokens != selected_tokens).cumsum(dim=-1) < 1).sum(
            dim=-1, keepdim=True
        )
        n_matches = n_matches.reshape(self.batch_size, 1)
        n_matches += 1

        # logic to select accepted tokens with padding
        accepted_tokens_mask = (
            torch.arange(target_tokens.shape[1])
            .expand(target_tokens.shape)
            .to(target_tokens.device)
            < n_matches
        )
        pad_tokens = torch.full_like(target_tokens, fill_value=self.acceptance_padding_token)
        accepted_tokens = torch.where(accepted_tokens_mask, target_tokens, pad_tokens).to(
            torch.int32
        )

        next_pos_ids = (position_ids + n_matches).to(torch.int32)

        batch_size, sequence_length = attention_mask.shape
        position_ids_to_compare = next_pos_ids.expand(batch_size, sequence_length) - 1
        mask = (
            torch.arange(sequence_length)
            .view(1, -1)
            .expand(batch_size, sequence_length)
            .to(position_ids.device)
        )
        next_attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)

        speculation_length = self.neuron_config.speculation_length
        first_pad_token = torch.sum(
            (accepted_tokens != self.acceptance_padding_token).to(torch.int), dim=1
        )
        # if no pad token is found, we must take the last index, otherwise take the previous token
        input_ids_idx = (first_pad_token + (speculation_length - 1)) % speculation_length
        next_input_ids = (accepted_tokens[torch.arange(batch_size), input_ids_idx]).reshape(
            batch_size, 1
        )

        return [accepted_tokens, next_input_ids, next_attention_mask, next_pos_ids]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        target_input_ids=None,
        target_attention_mask=None,
        target_position_ids=None,
        target_slot_mapping=None,
        target_active_block_table=None,
        llava_args: Optional[List] = [],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        speculation_length = self.neuron_config.speculation_length
        if self.config.neuron_config.enable_eagle_speculation:
            if (
                input_ids.shape[-1] > 1
                and input_ids.shape[-1] != self.neuron_config.speculation_length
                and input_ids.shape[-1] != self.neuron_config.medusa_speculation_length
            ):
                if self.is_prefix_caching:
                    context_outs = self._eagle_context_encoding_forward_with_prefix_caching(
                        input_ids, attention_mask, position_ids, seq_ids, sampling_params,
                        slot_mapping=slot_mapping,
                        active_block_table=active_block_table,
                        num_queries=num_queries,
                        computed_context_lens=computed_context_lens,
                        target_input_ids=target_input_ids,
                        target_attention_mask=target_attention_mask,
                        target_position_ids=target_position_ids,
                        target_slot_mapping=target_slot_mapping,
                        target_active_block_table=target_active_block_table,
                    )
                else:
                    context_outs = self._eagle_context_encoding_forward(
                        input_ids, attention_mask, position_ids, seq_ids, sampling_params
                    )
                outputs = self._cte_postprocessor(
                    context_outs, input_ids, attention_mask, position_ids, speculation_length,
                    num_queries=num_queries, computed_context_lens=computed_context_lens
                )

                # assign hidden state to next position ids
                next_pos_ids = outputs[-1]
                hidden_state = context_outs[-1]
                hidden_state_full = self.hidden_state_rolling_buffer.set_state(
                    seq_ids, next_pos_ids, hidden_state
                )
                context_outs[-1] = hidden_state_full

                # NOTE: index 2 onwards could be logits + kv cache or just kv cache
                return outputs + context_outs[2:]
            else:
                # Perform position ID clipping to prevent out-of-bounds in speculative token generation.
                generation_length = self.neuron_config.speculation_length
                bucket_size = attention_mask.shape[-1]
                position_ids = torch.clamp(position_ids, min=0, max=bucket_size - generation_length)

                # For prefix caching with EAGLE speculation, generate the slot mapping within the traced model.
                # This is necessary for async mode, as the active_block_table is up-to-date but the slot mapping
                # passed into the traced model may be from a prior iteration. This differs from the non-speculative
                # case because the slot mapping requires extrapolation of the slot mapping by speculation_length.
                if self.neuron_config.is_prefix_caching:
                    block_size = torch.tensor(self.neuron_config.pa_block_size, device=position_ids.device, dtype=torch.int32)
                    slot_mapping = generate_fusedspec_slot_mapping(position_ids, slot_mapping, active_block_table, block_size)

                    B = input_ids.shape[0]
                    num_queries = torch.ones((B, 1), dtype=torch.int32, device=position_ids.device)
                    computed_context_lens = position_ids.clone()

                # verify how many tokens here
                if (
                    self.config.neuron_config.enable_token_tree
                    and self.config.neuron_config.token_tree_config
                ):
                    token_gen_outs = self._eagle_tree_token_gen_forward(
                        input_ids,
                        attention_mask,
                        position_ids,
                        seq_ids,
                        sampling_params,
                    )
                    outputs = self._tkg_postprocessor(
                        token_gen_outs,
                        attention_mask,
                        position_ids,
                    )
                else:
                    token_gen_outs = self._eagle_token_gen_forward(
                        input_ids, attention_mask, position_ids, seq_ids, sampling_params,
                        slot_mapping=slot_mapping,
                        active_block_table=active_block_table,
                        num_queries=num_queries,
                        computed_context_lens=computed_context_lens,
                    )
                    outputs = self._tkg_postprocessor(
                        token_gen_outs,
                        attention_mask,
                        position_ids,
                    )

                # assign hidden state to next position ids
                hidden_state = token_gen_outs[-1]
                next_pos_ids = outputs[-1]
                hidden_state_full = self.hidden_state_rolling_buffer.set_state(
                    seq_ids,
                    next_pos_ids,
                    hidden_state,
                )
                token_gen_outs[-1] = hidden_state_full

                # NOTE: index 2 onwards could be logits + kv cache or just kv cache
                return outputs + token_gen_outs[2:]
        else:
            if (
                input_ids.shape[-1] > 1
                and input_ids.shape[-1] != self.neuron_config.speculation_length
                and input_ids.shape[-1] != self.neuron_config.medusa_speculation_length
            ):
                context_outs = self._context_encoding_forward(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    slot_mapping=slot_mapping,
                    active_block_table=active_block_table,
                    num_queries=num_queries,
                    computed_context_lens=computed_context_lens,
                )
                outputs = self._cte_postprocessor(
                    context_outs, input_ids, attention_mask, position_ids, speculation_length,
                    num_queries=num_queries, computed_context_lens=computed_context_lens
                )

                # NOTE: index 2 onwards could be logits + kv cache or just kv cache
                return outputs + context_outs[2:]
            else:
                # TODO - Determine if position ID clipping is necessary for fused speculation.
                token_gen_outs = self._token_gen_forward(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    slot_mapping=slot_mapping,
                    active_block_table=active_block_table,
                    num_queries=num_queries,
                    computed_context_lens=computed_context_lens,
                )
                outputs = self._tkg_postprocessor(
                    token_gen_outs,
                    attention_mask,
                    position_ids,
                )

                # NOTE: index 2 onwards could be logits + kv cache or just kv cache
                return outputs + token_gen_outs[2:]


class NeuronBaseForCausalLM(NeuronApplicationBase):
    _model_cls = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.text_config = self.config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.padding_side = self.neuron_config.padding_side
        self.kv_cache_populated = False

        # async related
        self.async_mode = self.neuron_config.async_mode
        self.next_cpu_inputs = None
        self.prior_outputs = None
        self.async_should_stop = False
        self.prior_seq_ids = None
        self.unequal_batching = (
            self.neuron_config.ctx_batch_size != self.neuron_config.tkg_batch_size
        )
        if self.async_mode:
            os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "2"
            os.environ["NEURON_RT_IO_RING_CACHE_SIZE"] = "2"

        self.sampler = None
        self.default_sampling_params = prepare_sampling_params(
            batch_size=self.neuron_config.batch_size, top_k=[1], top_p=[1.0], temperature=[1.0]
        )
        self.model_wrapper = self.get_model_wrapper_cls()

        if self.neuron_config.enable_fused_speculation:
            self.__class__._model_cls = NeuronFusedSpecModel
            self.enable_context_encoding()
            self.enable_fused_spec()
        else:
            self.enable_context_encoding()
            if self.neuron_config.speculation_length > 0:
                self.enable_speculation()
            elif self.neuron_config.medusa_speculation_length > 0:
                self.enable_medusa_speculation()
            elif self._should_enable_tkg():
                self.enable_token_generation()

        for model in self.models:
            assert (
                model.neuron_config.is_prefill_stage is not None
            ), f"{model.tag} doesn't indicate whether it is part of the prefill or generation step."

    def _should_enable_tkg(self):
        """
        Check if a token generation model should be enabled or not

        There are two cases:
            1. chunked prefill disabled: a token generation model should be
                enabled in this case
            2. chunked prefill enabled: check chunked prefill config to see
                if it should be enabled or not
        """
        if not self.neuron_config.is_chunked_prefill:
            return True

        chunked_prefill_config = self.neuron_config.chunked_prefill_config
        return chunked_prefill_config.tkg_model_enabled

    def get_model_wrapper_cls(self):
        return ModelWrapper

    def enable_fused_spec(self):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.tkg_batch_size
        if self.neuron_config.enable_fused_speculation:
            new_config.fused_spec_config.draft_config.neuron_config.batch_size = (
                self.neuron_config.tkg_batch_size
            )
            new_config.fused_spec_config.draft_config.neuron_config.n_active_tokens = 1
        new_config.neuron_config.n_active_tokens = 1
        new_config.neuron_config.bucket_n_active_tokens = False
        new_config.neuron_config.is_prefill_stage = False

        new_config.neuron_config.buckets = autobucketing.generate_buckets_for_fused_spec(new_config)

        # Explicitly turn off sequence parallel for token generation
        new_config.neuron_config.sequence_parallel_enabled = False
        new_config.fused_spec_config.draft_config.neuron_config.sequence_parallel_enabled = False

        self.fused_spec_model = self.model_wrapper(
            config=new_config,
            model_cls=self._model_cls,
            # call
            tag=FUSED_SPECULATION_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,  # to turn on weight layout optimization
        )
        self.models.append(self.fused_spec_model)

    def enable_context_encoding(self, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.ctx_batch_size
        if self.neuron_config.enable_fused_speculation:
            new_config.fused_spec_config.draft_config.neuron_config.batch_size = (
                self.neuron_config.ctx_batch_size
            )
        new_config.neuron_config.n_active_tokens = self.neuron_config.max_context_length
        new_config.neuron_config.bucket_n_active_tokens = True
        new_config.neuron_config.is_prefill_stage = True

        new_config.neuron_config.buckets = autobucketing.generate_buckets_for_cte(new_config)

        # Check if it should perform weight layout optimization based on CTE
        # By default, it should be based on TKG for best ITL.
        wlo_based_on_cte = False
        if self.neuron_config.is_chunked_prefill:
            # With chunked prefill, there could be cases where TKG is not
            # enabled.
            wlo_based_on_cte = not self.neuron_config.chunked_prefill_config.tkg_model_enabled

        self.context_encoding_model = self.model_wrapper(
            config=new_config,
            model_cls=self._model_cls,
            tag=CONTEXT_ENCODING_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=0 if wlo_based_on_cte else None,
        )
        self.models.append(self.context_encoding_model)

    def enable_token_generation(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        if self.neuron_config.is_chunked_prefill:
            tkg_batch_size = self.neuron_config.chunked_prefill_config.max_num_seqs
        else:
            tkg_batch_size = self.neuron_config.tkg_batch_size
        new_config.neuron_config.batch_size = tkg_batch_size
        new_config.neuron_config.n_active_tokens = 1
        new_config.neuron_config.bucket_n_active_tokens = False
        new_config.neuron_config.sequence_parallel_enabled = False
        new_config.neuron_config.is_prefill_stage = False

        new_config.neuron_config.buckets = autobucketing.generate_buckets_for_tkg(new_config)

        # shouldn't be used in token gen models
        new_config.neuron_config.sequence_parallel_enabled = False

        self.token_generation_model = self.model_wrapper(
            config=new_config,
            model_cls=self._model_cls,
            tag=TOKEN_GENERATION_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=(
                0 if enable_wlt_optimization else None
            ),  # to turn on weight layout optimization
            model_init_kwargs=model_init_kwargs,
        )
        self.models.append(self.token_generation_model)

    def enable_speculation(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.spec_batch_size
        new_config.neuron_config.n_active_tokens = self.neuron_config.speculation_length
        new_config.neuron_config.bucket_n_active_tokens = False

        new_config.neuron_config.sequence_parallel_enabled = False
        new_config.neuron_config.is_prefill_stage = False

        new_config.neuron_config.buckets = autobucketing.generate_buckets_for_speculation(
            new_config
        )

        self.speculation_model = self.model_wrapper(
            config=new_config,
            model_cls=self._model_cls,
            tag=SPECULATION_MODEL_TAG,
            model_init_kwargs=model_init_kwargs,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=(
                0 if enable_wlt_optimization else None
            ),  # to turn on weight layout optimization
        )

        self.models.append(self.speculation_model)

    def enable_medusa_speculation(self):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.spec_batch_size
        new_config.neuron_config.n_active_tokens = self.neuron_config.medusa_speculation_length
        self.medusa_speculation_model = self.model_wrapper(
            config=new_config, model_cls=self._model_cls, tag=MEDUSA_MODEL_TAG
        )
        new_config.neuron_config.is_prefill_stage = False

        self.models.append(self.medusa_speculation_model)

    @classmethod
    def prepare_quantized_state_dict(cls, hf_model_quant):
        model_quant_sd = hf_model_quant.model.state_dict()
        lm_head_quant_sd = hf_model_quant.lm_head.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        convert_qint8_to_int8_state_dict(lm_head_quant_sd)
        for key in lm_head_quant_sd.keys():
            model_quant_sd[f"lm_head.{key}"] = lm_head_quant_sd[key]

        return model_quant_sd

    def get_generation_model(self) -> ModelWrapper:
        if self.neuron_config.enable_fused_speculation:
            return self.fused_spec_model
        elif self.neuron_config.medusa_speculation_length > 0:
            return self.medusa_speculation_model
        elif self.neuron_config.speculation_length > 0:
            return self.speculation_model
        elif self.neuron_config.is_chunked_prefill:
            return self.context_encoding_model
        else:
            return self.token_generation_model

    def preprocess_inputs(self,
                          input_ids: torch.LongTensor = None,
                          seq_ids: Optional[torch.LongTensor] = None,
                          attention_mask: Optional[torch.Tensor] = None,
                          position_ids: Optional[torch.LongTensor] = None,
                          past_key_values: Optional[List[torch.FloatTensor]] = None,
                          inputs_embeds: Optional[torch.FloatTensor] = None,
                          sampling_params: Optional[torch.FloatTensor] = None,
                          prev_hidden: Optional[torch.FloatTensor] = None,
                          labels: Optional[torch.LongTensor] = None,
                          use_cache: Optional[bool] = None,
                          output_attentions: Optional[bool] = None,
                          output_hidden_states: Optional[bool] = None,
                          adapter_ids: Optional[torch.LongTensor] = None,
                          medusa_args=None,
                          return_dict: Optional[bool] = None,
                          llava_args: Optional[List] = [],
                          input_capture_hook: Optional[Callable] = None,
                          slot_mapping: Optional[torch.LongTensor] = None,
                          block_table: Optional[torch.LongTensor] = None,
                          full_context_lens: Optional[torch.LongTensor] = None,
                          computed_context_lens: Optional[torch.LongTensor] = None,):

        if self.async_mode and not self.neuron_config.enable_fused_speculation:
            # derive future cpu inputs from current cpu inputs
            if position_ids.shape[1] == input_ids.shape[1]:
                next_position_ids = torch.amax(position_ids, 1, keepdim=True)
            else:
                next_position_ids = position_ids

            next_position_ids = next_position_ids + 1
            next_attention_mask = self._infer_attention_mask(
                next_position_ids, full_context_lens, computed_context_lens
            )
            self.next_cpu_inputs = {
                "attention_mask": next_attention_mask,
                "position_ids": next_position_ids,
            }

            if self.neuron_config.is_prefix_caching:
                prefix_caching_next_inputs = self._prepare_prefix_caching_next_inputs(block_table,
                                                                                      slot_mapping,
                                                                                      full_context_lens,
                                                                                      computed_context_lens)
                self.next_cpu_inputs.update(**prefix_caching_next_inputs)
        elif self.neuron_config.is_prefix_caching and self.async_mode and (self.neuron_config.enable_eagle_speculation or self.neuron_config.enable_fused_speculation) and block_table is not None:
            # Maintain the most recently received block table for async block KV fused speculation.
            next_block_table = block_table.clone()
            self.next_cpu_inputs = {
                "block_table": next_block_table,
            }

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(
                position_ids, full_context_lens, computed_context_lens,
            )

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        sampling_params = (
            self.default_sampling_params if sampling_params is None else sampling_params
        )
        self.validate_sampling_params(sampling_params)
        sampling_params = infer_sampling_params(sampling_params)
        self.sampling_params = sampling_params

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(
            output_attentions, output_hidden_states, return_dict
        )

        if logging.root.isEnabledFor(logging.DEBUG):
            self._log_input(input_ids, attention_mask, position_ids, seq_ids, adapter_ids)

        if input_capture_hook is not None and not self.kv_cache_populated:
            self.initial_input_size = len(input_ids[0])

        if input_capture_hook is not None:
            input_capture_hook(
                self, [input_ids, attention_mask, position_ids, seq_ids, sampling_params]
            )

        # self.prior_seq_ids should never be None
        if self.prior_seq_ids is None:
            self.prior_seq_ids = seq_ids

        return input_ids, attention_mask, position_ids, seq_ids, sampling_params

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        medusa_args=None,
        return_dict: Optional[bool] = None,
        llava_args: Optional[List] = [],
        input_capture_hook: Optional[Callable] = None,
        slot_mapping: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        full_context_lens: Optional[torch.LongTensor] = None,
        computed_context_lens: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        input_ids, attention_mask, position_ids, seq_ids, sampling_params = self.preprocess_inputs(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            sampling_params=sampling_params,
            prev_hidden=prev_hidden,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            adapter_ids=adapter_ids,
            medusa_args=medusa_args,
            return_dict=return_dict,
            llava_args=llava_args,
            input_capture_hook=input_capture_hook,
            slot_mapping=slot_mapping,
            block_table=block_table,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens)

        if self.async_mode:
            if not self.neuron_config.is_block_kv_layout:
                outputs, is_run_on_neuron = self._get_model_outputs_async(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    prev_hidden=prev_hidden,
                    adapter_ids=adapter_ids,
                    medusa_args=medusa_args,
                    llava_args=llava_args,
                )
            else:
                outputs, is_run_on_neuron = self._get_model_outputs_async(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    prev_hidden=prev_hidden,
                    adapter_ids=adapter_ids,
                    medusa_args=medusa_args,
                    llava_args=llava_args,
                    slot_mapping=slot_mapping,
                    block_table=block_table,
                    full_context_lens=full_context_lens,
                    computed_context_lens=computed_context_lens,
                )
        elif self.neuron_config.is_block_kv_layout:
            outputs, is_run_on_neuron = self._get_model_outputs(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                medusa_args,
                llava_args,
                slot_mapping,
                block_table,
                full_context_lens,
                computed_context_lens,
            )
        else:
            outputs, is_run_on_neuron = self._get_model_outputs(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                medusa_args,
                llava_args,
            )

        generation_model = self.get_generation_model()
        if not generation_model.is_neuron():
            self._copy_past_key_values(outputs)

        # process outputs
        if self.on_device_sampling and self.neuron_config.output_logits and not \
                (self.neuron_config.enable_fused_speculation or self.neuron_config.is_medusa):
            logits_or_next_tokens = outputs[:2]
            constructed_outputs = self._construct_output_with_tokens_and_logits(next_tokens=logits_or_next_tokens[0], logits=logits_or_next_tokens[1])
        else:
            if is_run_on_neuron:
                # When run on neuron, KV cache remains on device
                logits_or_next_tokens = outputs
            else:
                # When run on cpu, KV cache is returned which has to be ignored
                logits_or_next_tokens, *_ = outputs
            constructed_outputs = self._construct_output(logits_or_next_tokens)

        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug("---output---")
            logging.debug(
                f"{'tokens' if self.on_device_sampling else 'logits'} = %s, ",
                logits_or_next_tokens,
            )

        return constructed_outputs

    def validate_sampling_params(self, params: torch.Tensor) -> None:
        if self.on_device_sampling:
            # Call validate_sampling_params from the Sampler.
            validate_sampling_params(params, self.neuron_config.on_device_sampling_config)

    def _setup_func_config(self, output_attentions, output_hidden_states, return_dict):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.text_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.text_config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "use_return_dict", None)
        )
        return output_attentions, output_hidden_states, return_dict

    def _infer_attention_mask(
        self,
        position_ids,
        full_context_lens: Optional[torch.LongTensor] = None,
        computed_context_lens: Optional[torch.LongTensor] = None,
    ):
        assert (
            position_ids is not None
        ), "need to call forward with position_ids if attention_mask is not provided"
        if self.neuron_config.is_chunked_prefill:
            return self._infer_attention_mask_for_chunked_prefill(
                full_context_lens, computed_context_lens
            )

        batch_size, seq_len = position_ids.shape
        if position_ids.shape[-1] != 1:
            seq_len = position_ids.shape[-1]
            position_ids_to_compare = position_ids
        else:
            seq_len = torch.max(position_ids)
            position_ids_to_compare = position_ids.expand(batch_size, seq_len) - 1
        mask = torch.arange(seq_len).view(1, -1).expand(batch_size, seq_len)
        attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)
        return attention_mask

    def _infer_attention_mask_for_chunked_prefill(
        self,
        full_context_lens: Optional[torch.LongTensor] = None,
        computed_context_lens: Optional[torch.LongTensor] = None,
    ):
        """
        Compute a positional attention mask based on full_context_lens and
        computed_context_lens, and this is for chunked prefill only.

        Args:
            full_context_lens: length of computed context and new input ids,
                with a shape of (batch_size, )
            computed_context_lens: length of computed context, with a
                shape of (batch_size, )

        Return:
            attention_mask: positional attention mask. If it is for CTE,
                it will have a shape of (1, len_input_ids); if it is for TKG,
                it will have a shape of (batch_size, max_seq_len_in_batch)
        """
        assert self.neuron_config.is_chunked_prefill
        cp_config = self.neuron_config.chunked_prefill_config

        query_lens = full_context_lens - computed_context_lens
        dtype = query_lens.dtype

        use_cte = torch.any(query_lens > 1) or not cp_config.tkg_model_enabled
        if use_cte:
            # batch size is 1 because it concatenates all seqs along seq dim
            attention_mask = torch.ones((1, query_lens.sum()), dtype=dtype)
        else:
            # similar logics as for TKG, but it relies on computed_context_lens
            batch_size = computed_context_lens.shape[0]
            max_seq_len = torch.max(computed_context_lens)
            pos_ids_to_compare = computed_context_lens.reshape(batch_size, 1)
            pos_ids_to_compare = pos_ids_to_compare.expand(batch_size, max_seq_len) - 1
            mask = torch.arange(max_seq_len).view(1, -1).expand(batch_size, max_seq_len)
            attention_mask = (pos_ids_to_compare >= mask).to(dtype)
        return attention_mask

    def _log_input(
        self, input_ids, attention_mask, position_ids, seq_ids, adapter_ids=None, **kwargs
    ):
        logging.debug("---input---")
        logging.debug("input_ids shape = %s type=%s", input_ids.shape, input_ids.type())
        logging.debug(
            "attention_mask shape = %s type=%s", attention_mask.shape, attention_mask.type()
        )
        logging.debug("position_ids shape = %s type=%s", position_ids.shape, position_ids.type())
        logging.debug("input_ids =%s", input_ids)
        logging.debug("attention_mask =%s", attention_mask)
        logging.debug("position_ids =%s", position_ids)
        logging.debug(f"seq_ids: {seq_ids}")
        logging.debug(f"adapter_ids: {adapter_ids}")

        generation_model = self.get_generation_model()
        if not generation_model.is_neuron():
            logging.debug(
                f"first layer kv_cache: {generation_model.model.kv_mgr.past_key_values[0][:, 0, :, 0]}"
            )

    def _convert_input_dict_to_ordered_tuple(self, input_dict: Dict[str, Any]):
        """
        Utility function to convert input dictionary to ordered tuple
        based on input signature of _get_model_outputs
        """
        args = []
        ordered_keys = inspect.getfullargspec(NeuronBaseForCausalLM._get_model_outputs).args

        for key in ordered_keys:
            if key == "self":
                continue
            elif (key == "medusa_args" or key == "llava_args") and input_dict[key]:
                for custom_arg in input_dict[key]:
                    args.append(custom_arg)
            elif key in input_dict:
                args.append(input_dict[key])

        return tuple(args)

    def _is_prefill(self, position_ids):
        return position_ids.min().item() == 0

    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden,
        adapter_ids,
        medusa_args,
        llava_args,
        slot_mapping=None,
        block_table=None,
        full_context_lens=None,
        computed_context_lens=None,
    ):
        if self.neuron_config.is_prefix_caching:
            num_queries = full_context_lens - computed_context_lens
            # Expect active and ordered block table for each seq after this step
            batch_size, _ = num_queries.shape

            is_context_encoding = input_ids.shape[-1] > 1 and not position_ids.min().item()
            generation_model = self.fused_spec_model if self.neuron_config.enable_fused_speculation else self.token_generation_model

            self.base_model = (
                self.context_encoding_model if is_context_encoding else generation_model
            )
            if self.neuron_config.enable_eagle_speculation:
                outputs = self.base_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    torch.empty(0),
                    torch.empty(0),
                    slot_mapping,
                    block_table,
                    num_queries,
                    computed_context_lens,
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    *llava_args,
                )
            elif self.neuron_config.enable_fused_speculation:
                outputs = self.base_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    torch.empty(0),
                    torch.empty(0),
                    slot_mapping,
                    block_table,
                    num_queries,
                    computed_context_lens,
                    *llava_args,
                )
            else:
                outputs = self.base_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty(0),
                    slot_mapping,
                    block_table,
                    num_queries,
                    computed_context_lens,
                    *llava_args,
                )
            is_run_on_neuron = self.base_model.is_neuron()
        elif self.neuron_config.is_chunked_prefill:
            outputs, is_run_on_neuron = self._get_model_outputs_for_chunked_prefill(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                slot_mapping,
                block_table,
                full_context_lens,
                computed_context_lens,
            )
        elif self._is_prefill(position_ids):
            if self.neuron_config.is_medusa:
                medusa_args = self._prepare_inputs()
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    prev_hidden,
                    adapter_ids,
                    *medusa_args,
                )
            else:
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    prev_hidden,
                    adapter_ids,
                    *llava_args,
                )

            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        elif self.neuron_config.enable_fused_speculation:
            outputs = self.fused_spec_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
            )
            is_run_on_neuron = self.fused_spec_model.is_neuron()
        elif input_ids.shape[-1] == self.neuron_config.speculation_length:
            outputs = self.speculation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
            )
            is_run_on_neuron = self.speculation_model.is_neuron()
        elif input_ids.shape[-1] == self.neuron_config.medusa_speculation_length:
            outputs = self.medusa_speculation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                *medusa_args,
            )
            is_run_on_neuron = self.medusa_speculation_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                *llava_args,
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    def _get_model_outputs_for_chunked_prefill(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        slot_mapping,
        block_table,
        full_context_lens,
        computed_context_lens,
    ):
        cp_config = self.neuron_config.chunked_prefill_config
        query_lens = full_context_lens - computed_context_lens
        actual_num_seqs = query_lens.shape[0]

        use_cte = torch.any(query_lens > 1) or not cp_config.tkg_model_enabled
        if use_cte:
            model = self.context_encoding_model

            # Expect active and ordered block table for each seq after this step
            active_block_table = kvcache_utils.get_active_block_table(
                block_table=block_table,
                context_lens=computed_context_lens,
                block_size=self.neuron_config.pa_block_size,
            )

            pa_scheduler = GridTileScheduler(
                query_lens,
                computed_context_lens,
                tile_size_q=cp_config.kernel_q_tile_size,
                tile_size_kv=cp_config.kernel_kv_tile_size,
                block_size=self.neuron_config.pa_block_size,
            )
            schedule = pa_scheduler.compute_schedule()
            tile_q_indices = schedule.get_tile_q_indices()
            tile_block_tables = schedule.build_tile_block_tables(
                active_block_table,
                skip_value=self.neuron_config.pa_num_blocks * 1000,
            )
            tile_masks = schedule.build_tile_masks()

            # Once tile_block_tables is generated, CTE doesn't need
            # active_block_table because it loads all the kv cache blocks and
            # selects them based on tile_block_tables
            active_block_table = torch.empty(0)
        else:
            model = self.token_generation_model
            max_num_seqs = cp_config.max_num_seqs

            # Reshape inputs to batching format (batch_size, seq_len)
            input_ids = input_ids.reshape(-1, 1)
            # skip attention_mask as it is already in batching format
            position_ids = position_ids.reshape(-1, 1)
            seq_ids = seq_ids.expand(max_num_seqs)
            sampling_params = sampling_params.expand(max_num_seqs, 3)
            slot_mapping = slot_mapping.reshape(-1, 1)
            active_block_table = block_table
            query_lens = query_lens.reshape(-1, 1)
            computed_context_lens = computed_context_lens.reshape(-1, 1)

            # Need to ensure CTE and TKG has same number of inputs due to
            # tracing limitation.
            tile_q_indices = torch.empty(0)
            tile_block_tables = torch.empty(0)
            tile_masks = torch.empty(0)

        outputs = model(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            slot_mapping,
            active_block_table,
            query_lens,
            computed_context_lens,
            tile_q_indices,
            tile_block_tables,
            tile_masks,
        )
        is_run_on_neuron = model.is_neuron()

        if not use_cte and not self.on_device_sampling:
            # Reshape back to concatenated format
            if is_run_on_neuron:
                outputs = outputs.transpose(0, 1)
            else:
                logits_or_tokens, kv_cache = outputs
                logits_or_tokens = logits_or_tokens.transpose(0, 1)
                outputs = logits_or_tokens, kv_cache
            return outputs, is_run_on_neuron

        if self.on_device_sampling:
            # Remove padding from on-device-sampling
            if is_run_on_neuron:
                outputs = outputs[:actual_num_seqs]
            else:
                logits_or_tokens, kv_cache = outputs
                logits_or_tokens = logits_or_tokens[:actual_num_seqs]
                outputs = logits_or_tokens, kv_cache

        return outputs, is_run_on_neuron

    def _get_model_outputs_async(self, **input_dict):
        """
        Handles the async execution of cte+tkg flow or the fused spec flow

        We do the below:
            for cte + tkg flow:
                prefill step: <cte>
                first generation step: <tkg current inputs -> <tkg next step>> <block on tkg current inputs> <tkg next step -> prior_outputs>
                all next generation steps: <tkg -> next_outputs> <block on prior_outputs> <next_outputs -> prior_outputs>
        """
        outputs, is_run_on_neuron = causal_lm_async_execution(
            self, input_dict, is_fused_speculation=self.neuron_config.enable_fused_speculation
        )
        if self._is_prefill(input_dict["position_ids"]):
            self.kv_cache_populated = True
        return outputs, is_run_on_neuron

    def _copy_kv_cache(self, source_model, target_model):
        for source, target in zip(source_model.model.models, target_model.model.models):
            encoder_kv_cache_line = source.states
            token_gen_kv_cache_line = target.states
            for name, _ in token_gen_kv_cache_line._parameters.items():
                token_gen_kv_cache_line._parameters[name] = encoder_kv_cache_line._parameters[name]

    def _copy_past_key_values(self, outputs):
        if self.neuron_config.enable_fused_speculation:
            draft_model_layers = len(self.fused_spec_model.draft_model.layers)
            new_draft_past_key_values = outputs[2 : draft_model_layers * 2]
            new_target_past_key_values = outputs[2 + draft_model_layers * 2 :]

            for i, new_draft_past_key_value in enumerate(new_draft_past_key_values):
                self.fused_spec_model.draft_model.past_key_values[i].data = new_draft_past_key_value
                self.context_encoding_model.draft_model.past_key_values[i].data = (
                    new_draft_past_key_value
                )

            for i, new_target_past_key_value in enumerate(new_target_past_key_values):
                self.fused_spec_model.target_model.past_key_values[i].data = (
                    new_target_past_key_value
                )
                self.context_encoding_model.target_model.past_key_values[i].data = (
                    new_target_past_key_value
                )
        else:
            if self.neuron_config.output_logits and self.neuron_config.on_device_sampling_config:
                # preserve tokens and logits (first 2 tensors)
                new_past_key_values = outputs[2:]
            else:
                new_past_key_values = outputs[1:]

            for i, new_past_key_value in enumerate(new_past_key_values):
                self.token_generation_model.model.kv_mgr.past_key_values[i].data = (
                    new_past_key_value
                )
                self.context_encoding_model.model.kv_mgr.past_key_values[i].data = (
                    new_past_key_value
                )

    def _construct_output_with_tokens_and_logits(self, next_tokens, logits, hidden_states=[]):
        OutputParams = CausalLMOutputWithPast(
            logits=logits,
            hidden_states=hidden_states,
            attentions=None,
        )
        OutputParams.tokens = next_tokens
        return OutputParams

    def _construct_output(self, logits_or_next_tokens):
        if self.neuron_config.is_medusa:
            next_tokens = logits_or_next_tokens[:1, :, :]
        else:
            if (
                self.async_mode
                and not self.neuron_config.enable_fused_speculation
                and isinstance(logits_or_next_tokens, list)
            ):
                logits_or_next_tokens = logits_or_next_tokens[0]
            next_tokens = logits_or_next_tokens

        OutputParams = CausalLMOutputWithPast(
            logits=None if self.on_device_sampling else logits_or_next_tokens,
            hidden_states=logits_or_next_tokens,
            attentions=None,
        )

        if self.neuron_config.is_medusa:
            OutputParams.tokens = next_tokens[:1, :, :]
            OutputParams.medusa_tokens = next_tokens[1:, :, :]
        elif self.neuron_config.enable_fused_speculation:
            OutputParams.fused_outputs = next_tokens
            OutputParams.async_should_stop = self.async_should_stop
        else:
            OutputParams.tokens = next_tokens

        return OutputParams

    def _prepare_inputs(self):
        accepted_indices = torch.zeros(
            (self.neuron_config.batch_size, self.neuron_config.num_medusa_heads + 1),
            dtype=torch.int32,
        )
        current_length = torch.zeros(
            (self.neuron_config.batch_size, self.neuron_config.num_medusa_heads + 1),
            dtype=torch.int32,
        )
        medusa_mask = torch.zeros(
            (
                self.neuron_config.batch_size,
                self.neuron_config.medusa_speculation_length,
                self.neuron_config.medusa_speculation_length,
            ),
            dtype=torch.int32,
        )
        scatter_index = torch.zeros(
            (self.neuron_config.batch_size, self.neuron_config.medusa_speculation_length),
            dtype=torch.int32,
        )
        return accepted_indices, current_length, medusa_mask, scatter_index

    def _prepare_prefix_caching_next_inputs(self,
                                            block_table,
                                            slot_mapping,
                                            full_context_lens,
                                            computed_context_lens):
        next_cpu_inputs = {}

        # Full context lens
        next_full_context_lens = full_context_lens + 1

        # Computed context lens
        if slot_mapping.shape[-1] != 1:  # Context encoding
            next_computed_context_lens = full_context_lens.clone()
        else:  # Token generation
            next_computed_context_lens = computed_context_lens + 1

        if block_table is not None:
            next_block_table = block_table.clone()
            next_cpu_inputs["block_table"] = next_block_table

        if full_context_lens is not None and computed_context_lens is not None:
            next_num_queries = next_full_context_lens - next_computed_context_lens
            next_cpu_inputs["num_queries"] = next_num_queries
            next_cpu_inputs["computed_context_lens"] = next_computed_context_lens

        return next_cpu_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def reset(self):
        # We need to reset the KV cache flag for a new batch of inference.
        # When the flag is reset, the subsequent run will invoke the
        # context encoding model.
        self.kv_cache_populated = False

    def get_required_kwargs(self) -> List[str]:
        """The list of required kwargs to the model's forward"""
        return []

    def reset_kv_cache(self):
        # Zero out kv cache for debug.
        # For new batch inference, use reset() instead
        if not self.context_encoding_model.is_neuron():
            for i, kv_tensor in enumerate(self.context_encoding_model.model.past_key_values):
                self.context_encoding_model.model.past_key_values[i] = torch.zeros_like(kv_tensor)

        if not self.token_generation_model.is_neuron():
            for i, kv_tensor in enumerate(self.token_generation_model.model.past_key_values):
                self.token_generation_model.model.past_key_values[i] = torch.zeros_like(kv_tensor)
