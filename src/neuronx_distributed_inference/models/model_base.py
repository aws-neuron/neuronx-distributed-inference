import copy
import logging
from typing import List, Optional, Tuple, Union

import neuronx_distributed as nxd
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.operators.argmax import argmax as nxd_argmax
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.mappings import (
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
from neuronx_distributed_inference.modules.attention import utils as attn_utils
from neuronx_distributed_inference.modules.autobucketing import generate_buckets
from neuronx_distributed_inference.modules.flashdecode.utils import (
    get_cache_size,
    mask_util,
    turn_2d_mask_to_4d,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    Sampler,
    prepare_sampling_params,
)
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KVCacheManager,
    _slice_kv_cacheline,
)
from neuronx_distributed_inference.modules.lora_serving import (
    update_weights_for_lora,
    wrap_model_with_lora,
)
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

        self.setup_attr_for_model(config)
        self.init_model(config)
        if optimize_inference:
            self.init_inference_optimization(config)

        if self.neuron_config.lora_config is not None:
            wrap_model_with_lora(self, self.neuron_config.lora_config)

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
        self.kv_mgr = KVCacheManager(config, num_kv_head=self.num_key_value_heads)

    def _create_context_attn_mask(self, attention_mask, **kwargs):
        # Block diagonal causal mask for chunked prefill
        if self.neuron_config.is_chunked_prefill:
            return self._create_chunked_prefill_attn_mask(**kwargs)

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
        return attn_utils.create_block_diagonal_attn_mask(
            query_lens, key_lens, max_query_len, max_key_len, **kwargs
        )

    def _create_spec_attn_mask(self, attention_mask):
        return (
            attention_mask[:, None, None, :]
            .expand(self.batch_size, 1, self.speculation_length, self.n_positions)
            .to(torch.bool)
        )

    def _create_simple_attn_mask(self, attention_mask):
        return (
            attention_mask[:, None, None, :]
            .expand(self.batch_size, 1, 1, self.n_positions)
            .to(torch.bool)
        )

    def create_attn_mask(
        self, attention_mask, is_for_context_encoding, is_for_speculation, **kwargs
    ):
        if is_for_context_encoding:
            return self._create_context_attn_mask(attention_mask, **kwargs)
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
        is_for_context_encoding = 1 < input_ids.shape[-1] != self.medusa_speculation_length
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
            sampling_params=sampling_params,
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

        medusa_logits = [logits] + [
            head(hidden_states).float()
            for head in [getattr(self, f"medusa_head_{i}") for i in range(self.num_medusa_heads)]
        ]
        stacked_logits = torch.stack(medusa_logits, dim=0)

        res = logits
        if is_for_context_encoding:
            result = [
                self.sampler(stacked_logits[i : i + 1, -1, :].squeeze(0), sampling_params)
                for i in range(self.neuron_config.num_medusa_heads + 1)
            ]
            res = torch.stack(result, dim=0)  # 5, 1, 10
        else:
            results = []
            for i in range(stacked_logits.shape[1]):
                result = [
                    self.sampler(stacked_logits[j : j + 1, i, :].squeeze(0), sampling_params)
                    for j in range(self.neuron_config.num_medusa_heads + 1)
                ]
                res = torch.stack(result, dim=0)
                results.append(res)

        return [res] + updated_kv_cache

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
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
    ):
        if self.neuron_config.is_medusa:
            return self._medusa_forward(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                adapter_ids,
                accepted_indices,
                current_length,
                medusa_mask,
                scatter_index,
            )

        is_for_context_encoding = 1 < input_ids.shape[-1] != self.speculation_length
        is_for_speculation = input_ids.shape[-1] == self.speculation_length

        cache_size = (
            get_cache_size(self.n_positions, self.num_cores_per_group)
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            if kv_cache is None:
                past_key_values = self.kv_mgr.get_cache(self.n_positions)
            else:
                past_key_values = self._slice_kv_cache(kv_cache, self.n_positions)

        # Prepare attention mask(s)
        attention_mask = self.create_attn_mask(
            attention_mask, is_for_context_encoding, is_for_speculation
        )
        active_mask = None
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
        )

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

        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            # speculative decoding case; only batch_size=1
            # will need to extend the logic to support multi-batch later
            # maybe just use position_ids for index?
            if position_ids.shape[-1] == self.speculation_length:
                index = torch.min(position_ids)
                index = torch.arange(
                    index, index + self.speculation_length, device=hidden_states.device
                )
                index = (
                    index.unsqueeze(0)
                    .unsqueeze(2)
                    .expand(self.batch_size, self.speculation_length, self.hidden_size)
                )
                hidden_states = torch.gather(hidden_states, dim=1, index=index)
            else:
                # simple token generation
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        res = logits
        if self.on_device_sampling:
            # perform sampling on Neuron to get tokens
            # FIXME, logits[:, -1, :] is not correct for speculation model, this is a tempory fix.
            if is_for_speculation:
                res = nxd_argmax(tensor=logits, dim=2, gather_dim=2, keepdim=False)
            else:
                res = self.sampler(logits[:, -1, :], sampling_params)

        return [res] + updated_kv_cache

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
        adapter_ids=None,
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
            inputs_embeds = self.embed_tokens(input_ids)

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
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            cos_cache, sin_cache = layer_outputs[2:]

        if self.sequence_parallel_enabled:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        hidden_states = self.norm(hidden_states)

        return (hidden_states, next_decoder_cache)

    def update_weights_for_lora(self, model_sd):
        return update_weights_for_lora(self, model_sd)


class NeuronFusedSpecModel(nn.Module):
    """
    Class to handle fused speculation flow
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.draft_neuron_config = config.fused_spec_config.draft_neuron_config
        self.worker_cls = config.fused_spec_config.worker_cls
        self.n_positions = config.neuron_config.n_positions

        config.fused_spec_config.draft_config.neuron_config.use_draft_group = True

        self.draft_model = self.worker_cls(config.fused_spec_config.draft_config)
        self.target_model = self.worker_cls(config)

    def _context_encoding_forward(
        self, input_ids, attention_mask, position_ids, seq_ids, sampling_params
    ):
        self.draft_model.n_positions = self.n_positions
        self.target_model.n_positions = self.n_positions

        assert self.neuron_config.on_device_sampling_config

        draft_outputs = self.draft_model(
            input_ids, attention_mask, position_ids, seq_ids, sampling_params
        )
        target_outputs = self.target_model(
            input_ids, attention_mask, position_ids, seq_ids, sampling_params
        )

        return [draft_outputs[0]] + [target_outputs[0]] + draft_outputs[1:] + target_outputs[1:]

    def _token_gen_forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params):
        spec_len = self.neuron_config.speculation_length
        bs = input_ids.shape[0]

        assert self.neuron_config.on_device_sampling_config

        # 1. Generate k-1 candidate tokens
        draft_position_ids = position_ids.expand(bs, spec_len)  # [1, 5]
        candidate_input_ids = input_ids
        target_position_ids = position_ids
        draft_attention_mask = copy.deepcopy(attention_mask)

        draft_cache = None
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
            draft_cache = model_output[1:]

            draft_attention_mask.index_fill_(1, draft_position_id.squeeze(), 1).view(bs, -1)
            new_draft_token = draft_outputs[0].view(bs, -1)

            candidate_input_ids = torch.cat((candidate_input_ids, new_draft_token), dim=-1)

        # Retile the cache
        flat_draft_cache = []
        for idx in range(len(draft_cache)):
            flat_draft_cache.append(draft_cache[idx].view(self.draft_model.kv_mgr.kv_shape))

        # # 2. verify candidates
        outputs = self.target_model(
            candidate_input_ids[:, :-1],
            attention_mask,
            target_position_ids[:, :-1],
            seq_ids,
            sampling_params,
        )
        target_tokens = outputs[0]

        return [candidate_input_ids[:, 1:]] + [target_tokens] + flat_draft_cache + outputs[1:]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        adapter_ids=None,
        llava_args: Optional[List] = [],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if (
            input_ids.shape[-1] > 1
            and input_ids.shape[-1] != self.neuron_config.speculation_length
            and input_ids.shape[-1] != self.neuron_config.medusa_speculation_length
        ):
            return self._context_encoding_forward(
                input_ids, attention_mask, position_ids, seq_ids, sampling_params
            )
        else:
            # verify how many tokens here
            return self._token_gen_forward(
                input_ids, attention_mask, position_ids, seq_ids, sampling_params
            )


class NeuronBaseForCausalLM(NeuronApplicationBase):
    _model_cls = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vocab_size = self.config.vocab_size
        self.padding_side = self.neuron_config.padding_side
        self.kv_cache_populated = False

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
            if self.neuron_config.trace_tokengen_model:
                self.enable_token_generation()
            if self.neuron_config.speculation_length > 0:
                self.enable_speculation()
            if self.neuron_config.medusa_speculation_length > 0:
                self.enable_medusa_speculation()

    def get_model_wrapper_cls(self):
        return ModelWrapper

    def enable_fused_spec(self):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.tkg_batch_size
        new_config.neuron_config.n_active_tokens = 1
        new_config.neuron_config.bucket_n_active_tokens = False
        new_config.fused_spec_config.draft_config.neuron_config.sequence_parallel_enabled = False

        if not new_config.neuron_config.enable_bucketing:
            new_config.neuron_config.buckets = generate_buckets(
                self.neuron_config.max_length, self.neuron_config.max_length
            )
        else:
            new_config.neuron_config.buckets = generate_buckets(128, self.neuron_config.max_length)

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
        new_config.neuron_config.n_active_tokens = self.neuron_config.max_context_length
        new_config.neuron_config.bucket_n_active_tokens = True

        if not new_config.neuron_config.enable_bucketing:
            new_config.neuron_config.buckets = generate_buckets(
                new_config.neuron_config.max_context_length,
                new_config.neuron_config.max_context_length,
            )
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
        )
        self.models.append(self.context_encoding_model)

    def enable_token_generation(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.tkg_batch_size
        new_config.neuron_config.n_active_tokens = 1
        new_config.neuron_config.bucket_n_active_tokens = False
        new_config.neuron_config.sequence_parallel_enabled = False

        if not new_config.neuron_config.enable_bucketing:
            new_config.neuron_config.buckets = generate_buckets(
                self.neuron_config.max_length, self.neuron_config.max_length
            )
        else:
            new_config.neuron_config.buckets = generate_buckets(128, self.neuron_config.max_length)

        # shouldn't be used in token gen models
        new_config.neuron_config.sequence_parallel_enabled = False

        self.token_generation_model = self.model_wrapper(
            config=new_config,
            model_cls=self._model_cls,
            tag=TOKEN_GENERATION_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0
            if enable_wlt_optimization
            else None,  # to turn on weight layout optimization
            model_init_kwargs=model_init_kwargs,
        )
        self.models.append(self.token_generation_model)

    def enable_speculation(self):
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.spec_batch_size
        new_config.neuron_config.n_active_tokens = self.neuron_config.speculation_length
        new_config.neuron_config.bucket_n_active_tokens = False

        if not new_config.neuron_config.enable_bucketing:
            new_config.neuron_config.buckets = generate_buckets(
                self.neuron_config.max_length, self.neuron_config.max_length
            )
        else:
            new_config.neuron_config.buckets = generate_buckets(128, self.neuron_config.max_length)

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

        self.models.append(self.medusa_speculation_model)

    @classmethod
    def prepare_quantized_state_dict(cls, hf_model_quant):
        model_quant_sd = hf_model_quant.model.state_dict()
        lm_head_quant_sd = hf_model_quant.lm_head.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        convert_qint8_to_int8_state_dict(lm_head_quant_sd)

        model_quant_sd["lm_head.weight"] = lm_head_quant_sd["weight"]
        model_quant_sd["lm_head.scale"] = lm_head_quant_sd["scale"]

        return model_quant_sd

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adapter_ids: Optional[torch.FloatTensor] = None,
        medusa_args=None,
        return_dict: Optional[bool] = None,
        llava_args: Optional[List] = [],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        sampling_params = (
            self.default_sampling_params if sampling_params is None else sampling_params
        )
        self.sampling_params = sampling_params

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(
            output_attentions, output_hidden_states, return_dict
        )

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(position_ids)

        self._log_input(input_ids, attention_mask, position_ids, seq_ids, adapter_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        outputs, is_run_on_neuron = self._get_model_outputs(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            adapter_ids,
            medusa_args,
            llava_args,
        )

        if self.neuron_config.trace_tokengen_model and not self.token_generation_model.is_neuron():
            self._copy_past_key_values(outputs)

        if is_run_on_neuron:
            # When run on neuron, KV cache remains on device
            logits_or_next_tokens = outputs
        else:
            # When run on cpu, KV cache is returned which has to be ignored
            logits_or_next_tokens, *_ = outputs

        logging.debug("---output---")
        logging.debug(
            f"{'tokens' if self.on_device_sampling else 'logits'} = %s, ",
            logits_or_next_tokens,
        )

        return self._construct_output(logits_or_next_tokens)

    def _setup_func_config(self, output_attentions, output_hidden_states, return_dict):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
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
        if position_ids.shape[-1] == 1:
            seq_len = self.neuron_config.n_positions
            position_ids_to_compare = position_ids.expand(batch_size, seq_len) - 1
        else:
            seq_len = position_ids.shape[-1]
            position_ids_to_compare = position_ids
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

        if self.neuron_config.trace_tokengen_model and not self.token_generation_model.is_neuron():
            logging.debug(
                f"first layer kv_cache: {self.token_generation_model.model.past_key_values[0][:, 0, :, 0]}"
            )

    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        adapter_ids,
        medusa_args,
        llava_args,
    ):
        if (
            input_ids.shape[-1] > 1
            and input_ids.shape[-1] != self.neuron_config.speculation_length
            and input_ids.shape[-1] != self.neuron_config.medusa_speculation_length
        ):
            if self.neuron_config.is_medusa:
                medusa_args = self._prepare_inputs()
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
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
                    adapter_ids,
                    *llava_args,
                )

            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        elif self.neuron_config.enable_fused_speculation:
            outputs = self.fused_spec_model(
                input_ids, attention_mask, position_ids, seq_ids, sampling_params, adapter_ids
            )

            is_run_on_neuron = self.fused_spec_model.is_neuron()
        elif input_ids.shape[-1] == self.neuron_config.speculation_length:
            outputs = self.speculation_model(
                input_ids, attention_mask, position_ids, seq_ids, sampling_params, adapter_ids
            )
            is_run_on_neuron = self.speculation_model.is_neuron()
        elif input_ids.shape[-1] == self.neuron_config.medusa_speculation_length:
            outputs = self.medusa_speculation_model(
                input_ids, attention_mask, position_ids, seq_ids, sampling_params, *medusa_args
            )
            is_run_on_neuron = self.medusa_speculation_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                adapter_ids,
                *llava_args,
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

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
                self.context_encoding_model.draft_model.past_key_values[
                    i
                ].data = new_draft_past_key_value

            for i, new_target_past_key_value in enumerate(new_target_past_key_values):
                self.fused_spec_model.target_model.past_key_values[
                    i
                ].data = new_target_past_key_value
                self.context_encoding_model.target_model.past_key_values[
                    i
                ].data = new_target_past_key_value
        else:
            new_past_key_values = outputs[1:]
            for i, new_past_key_value in enumerate(new_past_key_values):
                self.token_generation_model.model.past_key_values[i].data = new_past_key_value
                self.context_encoding_model.model.past_key_values[i].data = new_past_key_value

    def _construct_output(self, logits_or_next_tokens):
        if self.neuron_config.is_medusa:
            next_tokens = logits_or_next_tokens[:1, :, :]
        else:
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
        else:
            OutputParams.tokens = next_tokens

        return OutputParams

    def _prepare_inputs(self):
        accepted_indices = torch.zeros(
            (self.neuron_config.batch_size, self.neuron_config.num_medusa_heads + 1),
            dtype=torch.int64,
        )
        current_length = torch.zeros(
            (self.neuron_config.batch_size, self.neuron_config.num_medusa_heads + 1),
            dtype=torch.int64,
        )
        medusa_mask = torch.zeros(
            (
                self.neuron_config.batch_size,
                self.neuron_config.medusa_speculation_length,
                self.neuron_config.medusa_speculation_length,
            ),
            dtype=torch.int64,
        )
        scatter_index = torch.zeros(
            (self.neuron_config.batch_size, self.neuron_config.medusa_speculation_length),
            dtype=torch.int64,
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

    def reset_kv_cache(self):
        # Zero out kv cache for debug.
        # For new batch inference, use reset() instead
        if not self.context_encoding_model.is_neuron():
            for i, kv_tensor in enumerate(self.context_encoding_model.model.past_key_values):
                self.context_encoding_model.model.past_key_values[i] = torch.zeros_like(kv_tensor)

        if not self.token_generation_model.is_neuron():
            for i, kv_tensor in enumerate(self.token_generation_model.model.past_key_values):
                self.token_generation_model.model.past_key_values[i] = torch.zeros_like(kv_tensor)
