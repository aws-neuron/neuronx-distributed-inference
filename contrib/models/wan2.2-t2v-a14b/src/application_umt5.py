"""NxDI application for UMT5-XXL text encoder.

UMT5 differs from standard T5 in one critical way: each layer has its own
relative attention bias weights (not shared from block 0). This module
patches the NxDI T5 implementation to support per-layer bias.
"""
import torch
from typing import List, Tuple

from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import (
    NeuronT5Application, NeuronT5EncoderModel, NeuronT5Stack, NeuronT5Block,
    ModelWrapperT5, T5InferenceConfig,
)
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance


class UMT5Stack(NeuronT5Stack):
    """NeuronT5Stack where every block has its own relative attention bias."""

    def __init__(self, config, embed_tokens=None):
        # Skip NeuronT5Stack.__init__ and call nn.Module.__init__ directly,
        # then recreate blocks with has_relative_attention_bias=True for ALL
        super(NeuronT5Stack, self).__init__()
        from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import (
            T5LayerNorm,
        )
        import torch.nn as nn

        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [NeuronT5Block(config, has_relative_attention_bias=True)
             for _ in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, *args, **kwargs):
        # The parent forward passes position_bias between blocks.
        # For UMT5, we need each block to compute its own bias.
        # We achieve this by always passing position_bias=None to each block,
        # forcing it to recompute from its own relative_attention_bias weights.
        #
        # Override the loop portion by monkey-patching position_bias to None
        # after each block. We do this by wrapping the parent forward.
        #
        # Actually, the simplest approach: just set position_bias=None before
        # each block call. Let's override forward entirely with minimal changes.
        return self._umt5_forward(*args, **kwargs)

    def _umt5_forward(
        self, input_ids=None, attention_mask=None, encoder_hidden_states=None,
        encoder_attention_mask=None, inputs_embeds=None, head_mask=None,
        cross_attn_head_mask=None, past_key_values=None, use_cache=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else True

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        # Convert from [0,1] to additive format: 0 for attend, large negative for mask
        extended_attention_mask = (1.0 - extended_attention_mask.float()) * -1e9

        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        head_mask = [None] * self.config.num_layers
        cross_attn_head_mask = [None] * self.config.num_layers

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            # UMT5: always pass position_bias=None so each block computes its own
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=None,  # Force each block to use its own bias
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                encoder_decoder_position_bias=None,
                layer_head_mask=head_mask[i],
                cross_attn_layer_head_mask=cross_attn_head_mask[i],
                past_key_value=past_key_value,
                use_cache=False,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=hidden_states)


class UMT5EncoderModel(NeuronT5EncoderModel):
    """NeuronT5EncoderModel with UMT5Stack (per-layer bias) and attention mask."""

    def __init__(self, config):
        # NeuronT5EncoderModel.__init__ creates a NeuronT5Stack.
        # We call it, then replace the encoder with our UMT5Stack.
        super().__init__(config)
        self.encoder = UMT5Stack(config, self.shared)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)


class UMT5ModelWrapper(ModelWrapperT5):
    """ModelWrapper with attention_mask input and UMT5EncoderModel."""

    def input_generator(self) -> List[Tuple[torch.Tensor, ...]]:
        return [(
            torch.zeros(1, self.config.max_length, dtype=torch.long),
            torch.ones(1, self.config.max_length, dtype=torch.long),
        )]

    def get_model_instance(self):
        config = self.config
        def _create():
            m = UMT5EncoderModel(config)
            m.eval()
            return m
        return BaseModelInstance(module_cls=_create, input_output_aliases={})


class NeuronUMT5Application(NeuronT5Application):
    """NeuronT5Application adapted for UMT5 (per-layer bias + attention mask)."""

    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__(model_path=model_path, config=config, *args, **kwargs)
        self.models.clear()
        self.model = UMT5ModelWrapper(
            config=self.config, model_cls=UMT5EncoderModel,
            tag="NeuronT5EncoderModel",
            compiler_args=self.get_compiler_args(), priority_model_idx=0,
        )
        self.models.append(self.model)

    def forward(self, input_ids, attention_mask):
        result = self.models[0](input_ids, attention_mask)
        if hasattr(result, 'last_hidden_state'):
            return result.last_hidden_state
        if isinstance(result, dict):
            return result.get('last_hidden_state', list(result.values())[0])
        if isinstance(result, (list, tuple)):
            return result[0]
        return result
