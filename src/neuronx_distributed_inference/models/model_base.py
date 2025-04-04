import copy
import inspect
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import neuronx_distributed as nxd
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.operators.argmax import argmax as nxd_argmax
from neuronx_distributed.operators.topk import topk as nxd_topk
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.mappings import (
    _gather_along_dim,
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
)
from neuronx_distributed.quantization.quantization_utils import convert_qint8_to_int8_state_dict
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,
    FUSED_SPECULATION_MODEL_TAG,
    MEDUSA_MODEL_TAG,
    SPECULATION_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.async_execution import causal_lm_async_execution
from neuronx_distributed_inference.modules.attention import utils as attn_utils
from neuronx_distributed_inference.modules.autobucketing import generate_buckets
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
)
from neuronx_distributed_inference.modules.kvcache import utils as kvcache_utils
from neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager import BlockKVCacheManager
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KVCacheManager,
    _slice_kv_cacheline,
)
from neuronx_distributed_inference.modules.lora_serving import LoraCheckpoint, wrap_model_with_lora
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed_inference.utils.random import set_random_seed


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
        self.vocab_size = config.vocab_size
        self.speculation_length = config.neuron_config.speculation_length
        self.padding_side = config.neuron_config.padding_side
        self.max_length = config.neuron_config.max_length
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rank_util = SPMDRank(world_size=self.config.neuron_config.tp_degree)
        self.num_cores_per_group = config.num_cores_per_group
        self.is_block_kv_layout = config.neuron_config.is_block_kv_layout
        self.is_chunked_prefill = config.neuron_config.is_chunked_prefill
        self.is_prefix_caching = config.neuron_config.is_prefix_caching

        self.setup_attr_for_model(config)
        self.init_model(config)
        if optimize_inference:
            self.init_inference_optimization(config)

        lora_config = self.neuron_config.lora_config
        if lora_config is not None:
            wrap_model_with_lora(self, lora_config)
            self.lora_checkpoint = LoraCheckpoint(lora_config)

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
        if config.neuron_config.is_block_kv_layout:
            self.kv_mgr = BlockKVCacheManager(config, num_kv_head=self.num_key_value_heads)
        else:
            self.kv_mgr = KVCacheManager(config, num_kv_head=self.num_key_value_heads)

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

    def _create_chunked_prefill_attn_mask(
        self,
        query_lens: torch.Tensor,
        key_lens: torch.Tensor,
        max_query_len: int,
        max_key_len: int,
        **kwargs,
    ) -> torch.Tensor:
        causal_mask = attn_utils.create_block_diagonal_attn_mask(
            query_lens, key_lens, max_query_len, max_key_len
        )
        num_query, num_key = causal_mask.shape
        return causal_mask.reshape(1, 1, num_query, num_key)

    def _create_block_kv_attn_mask(
        self,
        query_lens: torch.Tensor,
        key_lens: torch.Tensor,
        max_query_len: int,
        max_key_len: int,
        is_prior: bool,
    ) -> torch.Tensor:
        batch_size = query_lens.shape[0]
        causal_masks = []
        for i in range(batch_size):
            if is_prior:
                causal_mask = attn_utils.create_block_diagonal_attn_mask(
                    query_lens[i, :], key_lens[i, :], max_query_len, max_key_len, is_prior=is_prior
                )
            else:
                # Use causal mask instead.
                causal_mask = torch.full(
                    (max_query_len, max_query_len), True, device=query_lens.device
                ).tril(diagonal=0)
            causal_masks.append(causal_mask)
        causal_masks = torch.stack(causal_masks, dim=0).unsqueeze(1)
        return causal_masks

    def _create_spec_attn_mask(self, attention_mask):
        return (
            attention_mask[:, None, None, :]
            .expand(self.batch_size, 1, self.speculation_length, self.n_positions)
            .to(torch.bool)
        )

    def _create_simple_attn_mask(self, attention_mask):
        batch_size = attention_mask.shape[0]
        return (
            attention_mask[:, None, None, :]
            .expand(batch_size, 1, 1, self.n_positions)
            .to(torch.bool)
        )

    def create_attn_mask(
        self, attention_mask, is_for_context_encoding, is_for_speculation, **kwargs
    ):
        if self.is_prefix_caching:
            return self._create_block_kv_attn_mask(**kwargs)
        elif self.is_chunked_prefill:
            return self._create_chunked_prefill_attn_mask(**kwargs)
        elif is_for_context_encoding:
            return self._create_context_attn_mask(attention_mask)
        elif is_for_speculation:
            return self._create_spec_attn_mask(attention_mask)
        else:
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

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            medusa_metadata = {
                "current_length": current_length,
                "accepted_indices": accepted_indices,
            }
            past_key_values = self.kv_mgr.get_cache(
                self.n_positions, medusa_metadata=medusa_metadata
            )

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

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
            adapter_ids=adapter_ids,
        )

        updated_kv_cache = self.kv_mgr.update_cache(
            is_for_context_encoding,
            seq_ids,
            position_ids,
            past_key_values,
            self.n_positions,
            scatter_index,
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

        cache_size = (
            get_cache_size(self.n_positions, self.num_cores_per_group, is_for_context_encoding)
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            if kv_cache is None:
                past_key_values = self.kv_mgr.get_cache(cache_size)
            else:
                past_key_values = self._slice_kv_cache(kv_cache, cache_size)

        # Prepare attention mask(s)
        attention_mask = self._create_token_tree_attn_mask(
            attention_mask, is_for_context_encoding, is_for_token
        )

        if active_mask is not None:
            if active_mask.shape[-1] == self.speculation_length:
                active_mask = active_mask.reshape(
                    1, 1, self.speculation_length, self.speculation_length
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
            past_key_values=past_key_values,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            prev_hidden=prev_hidden,
            rotary_position_ids=rotary_position_ids,
        )

        full_hidden_states = hidden_states

        # KV cache is update here only when it is drafting stage
        if self.neuron_config.is_eagle_draft:
            if kv_cache is None:
                updated_kv_cache = self.kv_mgr.update_cache(
                    is_for_context_encoding=is_for_context_encoding,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    new_key_values=past_key_values,
                    seq_len=cache_size,
                    scatter_index=scatter_index,
                    active_mask=active_mask_2d,
                )
            else:
                updated_kv_cache = self.kv_mgr.update_cache(
                    is_for_context_encoding=is_for_context_encoding,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    new_key_values=past_key_values,
                    seq_len=cache_size,
                    scatter_index=scatter_index,
                    active_mask=active_mask_2d,
                    kvcache_buffer=kv_cache,
                )
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

    def _slice_kv_cache(self, kv_cache, n_positions):
        past_key_values = []
        for idx in range(len(kv_cache)):
            k_cache = _slice_kv_cacheline(
                self.config.neuron_config.padding_side, n_positions, kv_cache[idx][0]
            )
            v_cache = _slice_kv_cacheline(
                self.config.neuron_config.padding_side, n_positions, kv_cache[idx][1]
            )
            past_key_values.append([k_cache, v_cache])
        return past_key_values

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
        cache_mask=None,
        current_reordered_idx=None,
        cache_reordered_idx=None,
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        active_mask=None,
        rotary_position_id=None,
    ):
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

        cache_size = (
            get_cache_size(self.n_positions, self.num_cores_per_group, is_for_context_encoding)
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )

        # It is either for context encoding or for token generation
        if self.is_block_kv_layout:
            past_key_values = self.kv_mgr.get_cache(active_block_table=active_block_table)
        elif is_for_context_encoding:
            past_key_values = None
        else:
            if kv_cache is None:
                past_key_values = self.kv_mgr.get_cache(seq_len=cache_size)
            else:
                past_key_values = self._slice_kv_cache(kv_cache, cache_size)

        # Prepare attention mask(s)
        if self.is_prefix_caching:
            attention_mask = self.create_attn_mask(
                attention_mask,
                is_for_context_encoding,
                is_for_speculation,
                query_lens=num_queries,
                key_lens=num_queries + computed_context_lens,
                max_query_len=self.neuron_config.n_active_tokens,
                max_key_len=self.neuron_config.max_context_length,
                is_prior=True,
            )
        elif self.is_chunked_prefill:
            max_total_len = (
                self.neuron_config.cp_num_active_blocks * self.neuron_config.pa_block_size
                + self.neuron_config.max_context_length
            )
            attention_mask = self.create_attn_mask(
                attention_mask,
                is_for_context_encoding,
                is_for_speculation,
                query_lens=num_queries,
                key_lens=num_queries + computed_context_lens,
                max_query_len=self.neuron_config.max_context_length,
                max_key_len=max_total_len,
            )
        else:
            attention_mask = self.create_attn_mask(
                attention_mask,
                is_for_context_encoding,
                is_for_speculation,
            )

        active_mask = None
        if self.is_prefix_caching:
            active_mask = self._create_block_kv_attn_mask(
                query_lens=num_queries,
                key_lens=num_queries,
                max_query_len=self.neuron_config.n_active_tokens,
                max_key_len=self.neuron_config.n_active_tokens,
                is_prior=False,
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

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            prev_hidden=prev_hidden,
            cache_mask=cache_mask,
            current_reordered_idx=current_reordered_idx,
            cache_reordered_idx=cache_reordered_idx,
        )

        if self.neuron_config.enable_eagle_speculation:
            full_hidden_states = hidden_states

        updated_kv_cache = self.kv_mgr.update_cache(
            is_for_context_encoding=is_for_context_encoding,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=past_key_values,
            seq_len=cache_size,
            scatter_index=slot_mapping if self.is_block_kv_layout else scatter_index,
            active_mask=active_mask_2d,
            kvcache_buffer=kv_cache,
        )

        batch_size = input_ids.shape[0]
        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        elif self.is_chunked_prefill:
            # chunked prefill will return cp_max_num_seqs, not just the last one
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

        res = logits
        if self.on_device_sampling:
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
                res = self.sampler(
                    logits[:, -1, :], sampling_params, rank_id=self.rank_util.get_rank()
                )
                res = res.to(torch.int32)
            # Otherwise we return the full logits for multinomial sampling in spec decoding

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
            outputs = outputs + [full_hidden_states]

        return outputs

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if inputs_embeds is None:
            inputs_embeds = (
                self.embed_tokens(input_ids)
                if not is_lora_module(self.embed_tokens)
                else self.embed_tokens(input_ids, adapter_ids=adapter_ids)
            )

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

        if self.neuron_config.is_eagle_draft:
            concat_states = torch.cat((hidden_states, prev_hidden), dim=2)
            hidden_states = self.fc(concat_states)

        # decoder layers
        next_decoder_cache = ()
        cos_cache = None
        sin_cache = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                rotary_position_ids=rotary_position_ids,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            cos_cache, sin_cache = layer_outputs[2:]

        if not self.neuron_config.is_eagle_draft:
            hidden_states = self.norm(hidden_states)

        if self.sequence_parallel_enabled:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        return (hidden_states, next_decoder_cache)

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
        self, input_ids, attention_mask, position_ids, seq_ids, sampling_params
    ):
        self.draft_model.n_positions = self.n_positions
        self.target_model.n_positions = self.n_positions

        assert self.neuron_config.on_device_sampling_config

        target_outputs = self.target_model(
            input_ids, attention_mask, position_ids, seq_ids, sampling_params
        )
        draft_outputs = self.draft_model(
            input_ids, attention_mask, position_ids, seq_ids, sampling_params
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

    def _token_gen_forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params):
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

            if draft_cache is None:
                draft_cache = self.draft_model.kv_mgr.get_cache(self.n_positions, skip_slice=True)
            else:
                # draft cache returned from the model is flattened. We reshape it to match the expected input schema.
                # kvcache_buffer is 2D list where, 1st dim for layer and the second denotes K and V.
                # For example,
                #     kvcache_buffer[1][0] is the K cache of the 1st layer
                #     kvcache_buffer[4][1] is the V cache of the 4th layer
                reshaped_cache = []
                for i in range(0, len(draft_cache), 2):
                    reshaped_cache.append([draft_cache[i], draft_cache[i + 1]])
                draft_cache = reshaped_cache

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

    def _eagle_context_encoding_forward(
        self, input_ids, attention_mask, position_ids, seq_ids, sampling_params
    ):
        self.draft_model.n_positions = self.n_positions
        self.target_model.n_positions = self.n_positions

        assert self.neuron_config.on_device_sampling_config

        target_outputs = self.target_model(
            input_ids, attention_mask, position_ids, seq_ids, sampling_params
        )
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
        draft_cache = draft_outputs[num_outputs:-1]
        target_cache = target_outputs[num_outputs:-1]
        index = torch.max(position_ids, dim=1, keepdim=True).indices
        index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
        hidden_state = torch.gather(hidden_state, dim=1, index=index)

        if self.neuron_config.output_logits:
            return (
                [draft_outputs[0]]
                + [target_outputs[0]]
                + [draft_outputs[1]]
                + [target_outputs[1]]
                + draft_cache
                + target_cache
                + [hidden_state]
            )
        return (
            [draft_outputs[0]] + [target_outputs[0]] + draft_cache + target_cache + [hidden_state]
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
        new_position_ids = position_ids[0, 0] - 1 + draft_position_offset
        new_position_ids = new_position_ids.unsqueeze(0)
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
        draft_next_scatter_index = draft_next_scatter_index + torch.arange(0, spec_len).reshape(
            bs, spec_len
        )

        # Compute target Position_ids before verification stage
        target_position_ids = torch.sum(draft_attention_mask, dim=1, keepdim=True).expand(
            bs, spec_len
        ) + torch.arange(0, spec_len).reshape(bs, spec_len)
        target_position_ids = target_position_ids.to(
            dtype=candidate_input_ids.dtype, device=candidate_input_ids.device
        ).expand(bs, -1)

        target_attention_mask = (
            target_attention_mask[:, None, None, :]
            .expand(self.batch_size, 1, spec_len, self.n_positions)
            .to(torch.bool)
        )
        draft_update_attention_mask = (
            draft_update_attention_mask[:, None, None, :]
            .expand(self.batch_size, 1, tree_depth, self.n_positions)
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

            if draft_cache is None:
                draft_cache = self.draft_model.kv_mgr.get_cache(self.n_positions, skip_slice=True)
            else:
                # draft cache returned from the model is flattened. We reshape it to match the expected input schema.
                # kvcache_buffer is 2D list where, 1st dim for layer and the second denotes K and V.
                reshaped_cache = []
                for j in range(0, len(draft_cache), 2):
                    reshaped_cache.append([draft_cache[j], draft_cache[j + 1]])
                draft_cache = reshaped_cache

            draft_attention_mask = copy.deepcopy(attention_mask)
            draft_attention_mask = (
                draft_attention_mask[:, None, None, :]
                .expand(self.batch_size, 1, level_node_count[i], self.n_positions)
                .to(torch.bool)
            )

            # Pick draft attention mask for this tree level from precomputed list
            draft_tree_attention_mask = draft_tree_attention_masks[i].to(
                device=draft_attention_mask.device, dtype=torch.bool
            )

            draft_attention_mask = torch.scatter(
                draft_attention_mask,
                3,
                draft_next_scatter_index.unsqueeze(0)
                .unsqueeze(1)
                .expand(bs, 1, level_node_count[i], spec_len),
                draft_tree_attention_mask.unsqueeze(0)
                .unsqueeze(1)
                .expand(bs, 1, level_node_count[i], spec_len),
            )

            # Compute rotary position id for rotary embedding in Llama
            rotary_offset = i + new_position_ids[0, 0, 0]
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
                selected = output_index.reshape(1, -1)
                topk_permute_index = torch.tensor(
                    topk_permute_indice[i], device=position_ids.device, dtype=torch.int32
                )
                topk_permute_index = topk_permute_index.expand(bs, -1)
                selected = torch.gather(selected, dim=1, index=topk_permute_index)
                draft_outputs = selected[:, : level_node_count[i + 1]].to(torch.int32)

                new_draft_token = draft_outputs[0].reshape(bs, -1)
                candidate_input_ids = torch.cat([candidate_input_ids, new_draft_token], dim=1)

            draft_cache = model_output[num_outputs:-2]
            hidden_state = model_output[-2]

            # Prepare hidden state for next drafting forward pass, the drafted token will use parent's hidden in next forward pass
            if i != tree_depth - 2:
                draft_hidden_gather_index = torch.tensor(
                    draft_hidden_gather_indice[i], device=position_ids.device, dtype=torch.int32
                )
                draft_hidden_gather_index = (
                    draft_hidden_gather_index.unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(1, draft_hidden_gather_index.shape[0], hidden_state.shape[-1])
                )
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
        target_rotary_position_id = (rotary_position_id_offset + target_position_ids[0, 0]).expand(
            bs, spec_len
        )
        active_mask = full_tree_attention_mask.to(
            device=target_attention_mask.device, dtype=torch.bool
        )

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
            dest_idx = dest_idx.unsqueeze(0)
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
        permute_mask_gather_idx = dest_idx.reshape(1, 1).expand(1, permute_masks.shape[1])
        permute_mask = torch.gather(permute_masks, dim=0, index=permute_mask_gather_idx).squeeze(0)

        cache_scatter_index = torch.gather(
            cache_scatter_indices,
            dim=0,
            index=dest_idx.view(1, 1).expand(1, cache_scatter_indices.shape[1]),
        ).squeeze(0)

        parent_permute_mask = torch.gather(
            parent_permute_masks,
            dim=0,
            index=dest_idx.view(1, 1).expand(1, parent_permute_masks.shape[1]),
        ).squeeze(0)

        gather_index = permute_mask.unsqueeze(0).expand(target_tokens.shape[0], -1)
        parent_gather_index = parent_permute_mask.unsqueeze(0).expand(target_tokens.shape[0], -1)
        prev_hidden_gather_index = (
            permute_mask.unsqueeze(0).unsqueeze(-1).expand(bs, spec_len, self.hidden_size)
        )

        target_token_gather_index = parent_gather_index
        candidate_input_gather_index = gather_index
        target_hidden_gather_idx = (
            parent_permute_mask.unsqueeze(0)
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
        cache_scatter_index = cache_scatter_index + new_position_ids[0, 0, 0] + 1
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
        self, input_ids, attention_mask, position_ids, seq_ids, sampling_params
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

            if draft_cache is None:
                draft_cache = self.draft_model.kv_mgr.get_cache(self.n_positions, skip_slice=True)
            else:
                # draft cache returned from the model is flattened. We reshape it to match the expected input schema.
                # kvcache_buffer is 2D list where, 1st dim for layer and the second denotes K and V.
                # For example,
                #     kvcache_buffer[1][0] is the K cache of the 1st layer
                #     kvcache_buffer[4][1] is the V cache of the 4th layer
                reshaped_cache = []
                for i in range(0, len(draft_cache), 2):
                    reshaped_cache.append([draft_cache[i], draft_cache[i + 1]])
                draft_cache = reshaped_cache

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
                    model_output[0], sampling_params, return_values=True,
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

        reshaped_cache = []
        for i in range(0, len(draft_cache), 2):
            reshaped_cache.append([draft_cache[i], draft_cache[i + 1]])
        draft_cache = reshaped_cache

        # 3 Final draft run to update KV cache. This is done after the target run since we need to send
        # the hidden states from the target output as input to the final draft run.
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
        self, context_outs, input_ids, attention_mask, position_ids, speculation_length
    ):
        batch_size = input_ids.shape[0]
        cur_len = torch.sum(attention_mask, dim=1).to(torch.int32)

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
        llava_args: Optional[List] = [],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        speculation_length = self.neuron_config.speculation_length
        if self.config.neuron_config.enable_eagle_speculation:
            if (
                input_ids.shape[-1] > 1
                and input_ids.shape[-1] != self.neuron_config.speculation_length
                and input_ids.shape[-1] != self.neuron_config.medusa_speculation_length
            ):
                context_outs = self._eagle_context_encoding_forward(
                    input_ids, attention_mask, position_ids, seq_ids, sampling_params
                )
                outputs = self._cte_postprocessor(
                    context_outs, input_ids, attention_mask, position_ids, speculation_length
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
                        input_ids, attention_mask, position_ids, seq_ids, sampling_params
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
                    input_ids, attention_mask, position_ids, seq_ids, sampling_params
                )
                outputs = self._cte_postprocessor(
                    context_outs, input_ids, attention_mask, position_ids, speculation_length
                )

                # NOTE: index 2 onwards could be logits + kv cache or just kv cache
                return outputs + context_outs[2:]
            else:
                # TODO - Determine if position ID clipping is necessary for fused speculation.
                token_gen_outs = self._token_gen_forward(
                    input_ids, attention_mask, position_ids, seq_ids, sampling_params
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
        self.unequal_batching = (
            self.neuron_config.ctx_batch_size != self.neuron_config.tkg_batch_size
        )
        if self.async_mode:
            os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "2"

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
            elif not self.neuron_config.is_chunked_prefill:
                # Chunked prefill runs both prefilling and decoding inside
                # CTE, so there is no need for TKG.
                self.enable_token_generation()

        for model in self.models:
            assert (
                model.neuron_config.is_prefill_stage is not None
            ), f"{model.tag} doesn't indicate whether it is part of the prefill or generation step."

        self.set_async_mode(self.neuron_config.async_mode)

    def get_model_wrapper_cls(self):
        return ModelWrapper

    def enable_fused_spec(self):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.tkg_batch_size
        if self.neuron_config.enable_fused_speculation:
            new_config.fused_spec_config.draft_config.neuron_config.batch_size = (
                self.neuron_config.tkg_batch_size
            )
        new_config.neuron_config.n_active_tokens = 1
        new_config.neuron_config.bucket_n_active_tokens = False
        new_config.neuron_config.is_prefill_stage = False

        if not new_config.neuron_config.enable_bucketing:
            new_config.neuron_config.buckets = generate_buckets(
                self.neuron_config.max_length, self.neuron_config.max_length
            )
        else:
            if new_config.neuron_config.token_generation_buckets is not None:
                new_config.neuron_config.buckets = new_config.neuron_config.token_generation_buckets
            else:
                new_config.neuron_config.buckets = generate_buckets(
                    128, self.neuron_config.max_length
                )

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

        if not new_config.neuron_config.enable_bucketing:
            new_config.neuron_config.buckets = generate_buckets(
                new_config.neuron_config.max_context_length,
                new_config.neuron_config.max_context_length,
            )
        else:
            if new_config.neuron_config.context_encoding_buckets is not None:
                new_config.neuron_config.buckets = new_config.neuron_config.context_encoding_buckets
            else:
                new_config.neuron_config.buckets = generate_buckets(
                    128, new_config.neuron_config.max_context_length
                )

        self.context_encoding_model = self.model_wrapper(
            config=new_config,
            model_cls=self._model_cls,
            tag=CONTEXT_ENCODING_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            # to turn on weight layout optimization
            priority_model_idx=0 if self.config.neuron_config.is_chunked_prefill else None,
        )
        self.models.append(self.context_encoding_model)

    def enable_token_generation(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.tkg_batch_size
        new_config.neuron_config.n_active_tokens = 1
        new_config.neuron_config.bucket_n_active_tokens = False
        new_config.neuron_config.sequence_parallel_enabled = False
        new_config.neuron_config.is_prefill_stage = False

        if not new_config.neuron_config.enable_bucketing:
            new_config.neuron_config.buckets = generate_buckets(
                self.neuron_config.max_length, self.neuron_config.max_length
            )
        else:
            if new_config.neuron_config.token_generation_buckets is not None:
                new_config.neuron_config.buckets = new_config.neuron_config.token_generation_buckets
            else:
                new_config.neuron_config.buckets = generate_buckets(
                    128, self.neuron_config.max_length
                )

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

    def enable_speculation(self):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.spec_batch_size
        new_config.neuron_config.n_active_tokens = self.neuron_config.speculation_length
        new_config.neuron_config.bucket_n_active_tokens = False

        new_config.neuron_config.sequence_parallel_enabled = False
        new_config.neuron_config.is_prefill_stage = False

        if not new_config.neuron_config.enable_bucketing:
            new_config.neuron_config.buckets = generate_buckets(
                self.neuron_config.max_length, self.neuron_config.max_length
            )
        else:
            if new_config.neuron_config.token_generation_buckets is not None:
                new_config.neuron_config.buckets = new_config.neuron_config.token_generation_buckets
            else:
                new_config.neuron_config.buckets = generate_buckets(
                    128, self.neuron_config.max_length
                )

        self.speculation_model = self.model_wrapper(
            config=new_config,
            model_cls=self._model_cls,
            tag=SPECULATION_MODEL_TAG,
            priority_model_idx=0,  # to turn on weight layout optimization
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

    def set_async_mode(self, async_mode: bool):
        if async_mode and not self.on_device_sampling:
            raise ValueError("Cannot set async mode when on device sampling is not enabled.")

        # reset async mode state
        self.async_mode = async_mode
        self.next_cpu_inputs = None
        self.prior_outputs = None
        self.async_should_stop = False
        self.prior_seq_ids = None

        # set/unset nrt env vars for async mode
        NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS = "NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"
        NEURON_RT_IO_RING_CACHE_SIZE = "NEURON_RT_IO_RING_CACHE_SIZE"
        if self.async_mode:
            os.environ[NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS] = "2"
            os.environ[NEURON_RT_IO_RING_CACHE_SIZE] = "2"
        elif NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS in os.environ:
            os.unsetenv(NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS)
            os.unsetenv(NEURON_RT_IO_RING_CACHE_SIZE)

        # set async mode in model wrapper models
        for model in self.models:
            model.async_mode = async_mode

            # refresh internal async state
            if self.is_loaded_to_neuron:
                for spmd_model in self._get_spmd_model_objects(model.tag):
                    spmd_model.refresh_async_state()

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
        if self.async_mode and not self.neuron_config.enable_fused_speculation:
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
        self.validate_sampling_params(sampling_params)
        self.sampling_params = sampling_params

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(
            output_attentions, output_hidden_states, return_dict
        )

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(position_ids)

        if logging.root.isEnabledFor(logging.DEBUG):
            self._log_input(input_ids, attention_mask, position_ids, seq_ids, adapter_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        if input_capture_hook is not None and not self.kv_cache_populated:
            self.initial_input_size = len(input_ids[0])

        if input_capture_hook is not None:
            input_capture_hook(
                self, [input_ids, attention_mask, position_ids, seq_ids, sampling_params]
            )

        # self.prior_seq_ids should never be None
        if self.prior_seq_ids is None:
            self.prior_seq_ids = seq_ids

        if self.async_mode:
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

        if is_run_on_neuron:
            # When run on neuron, KV cache remains on device
            logits_or_next_tokens = outputs
        else:
            # When run on cpu, KV cache is returned which has to be ignored
            logits_or_next_tokens, *_ = outputs

        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug("---output---")
            logging.debug(
                f"{'tokens' if self.on_device_sampling else 'logits'} = %s, ",
                logits_or_next_tokens,
            )

        return self._construct_output(logits_or_next_tokens)

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

    def _infer_attention_mask(self, position_ids):
        assert (
            position_ids is not None
        ), "need to call forward with position_ids if attention_mask is not provided"
        batch_size, seq_len = position_ids.shape
        if position_ids.shape[-1] != 1 or self.neuron_config.is_chunked_prefill:
            seq_len = position_ids.shape[-1]
            position_ids_to_compare = position_ids
        else:
            seq_len = torch.max(position_ids)
            position_ids_to_compare = position_ids.expand(batch_size, seq_len) - 1
        mask = torch.arange(seq_len).view(1, -1).expand(batch_size, seq_len)
        attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)
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
            self.base_model = (
                self.context_encoding_model if is_context_encoding else self.token_generation_model
            )

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
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        elif self.neuron_config.is_chunked_prefill:
            num_queries = full_context_lens - computed_context_lens
            # Expect active and ordered block table for each seq after this step
            active_block_table = kvcache_utils.get_active_block_table(
                block_table=block_table,
                context_lens=computed_context_lens,
                block_size=self.neuron_config.pa_block_size,
            )
            cache_mask, cache_reordered_idx, current_reordered_idx = (
                kvcache_utils.contexted_kv_indexing_dynamic(
                    q_lens=num_queries,
                    k_lens=full_context_lens,
                    block_size=self.neuron_config.pa_block_size,
                )
            )

            outputs = self.context_encoding_model(
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
                cache_mask,
                current_reordered_idx,
                cache_reordered_idx,
                *llava_args,
            )
            is_run_on_neuron = self.context_encoding_model.is_neuron()
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
            new_past_key_values = outputs[1:]
            for i, new_past_key_value in enumerate(new_past_key_values):
                self.token_generation_model.model.kv_mgr.past_key_values[i].data = (
                    new_past_key_value
                )
                self.context_encoding_model.model.kv_mgr.past_key_values[i].data = (
                    new_past_key_value
                )

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
