import gc
import logging
from typing import Optional, Tuple

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.image_to_text_model_wrapper import ImageToTextModelWrapper
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import scatter_by_index_put
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

logger = logging.getLogger("Neuron")


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class NeuronQwen3VLRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_type = self.config.rope_scaling["rope_type"]
        rope_init_fn = self.compute_default_rope_parameters

        assert (
            self.rope_type == "default"
        ), "Only 'default' rope_type is supported for Qwen3VLTextRotaryEmbedding"

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    def compute_default_rope_parameters(
        self,
        config=None,
        device=None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        config = self.config
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                / dim
            )
        )
        return inv_freq, attention_factor

    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            # happens if we compile model with text only inputs in NeuronQwen3VLTextForCausalLM
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # If we compile NeuronQwen3VLForCausalLM and expect either vision+text or text-only input
        # actually rotary_position_ids is passed into this forward
        # so the shape should already be 3D (3, bs, seq_len)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = NeuronQwen3VLRotaryEmbedding.neuron_compute_freqs_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def neuron_compute_freqs_mrope(freqs: torch.Tensor, mrope_section: list) -> torch.Tensor:
        """XLA-friendly multimodal RoPE frequency computation - generalized."""
        last_dim = freqs.shape[-1]
        indices = torch.arange(last_dim, device=freqs.device, dtype=torch.int64)

        # Start with T (dimension 0)
        freqs_t = freqs[0].clone()  # explicit copy, no in-place mutation

        # Process H (dim=1, offset=1) and W (dim=2, offset=2)
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            mask = (indices % 3 == offset) & (indices < length)
            freqs_t = torch.where(mask, freqs[dim], freqs_t)
        return freqs_t


class NeuronQwen3VLAttention(NeuronAttentionBase):
    """Self-attention similar to *NeuronLlamaAttention* but with qkv_bias=True and RoPE sharing
    semantics compatible with Qwen3-VL (3-way multimodal RoPE).
    """

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=False,
            o_bias=False,
            rotary_emb=NeuronQwen3VLRotaryEmbedding(config),
            rms_norm_eps=config.rms_norm_eps,
            attention_chunk_size=getattr(config, "attention_chunk_size", None),
            sliding_window=getattr(config, "sliding_window", None),
            q_layernorm=get_rmsnorm_cls()(head_dim, config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(head_dim, config.rms_norm_eps),
        )
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.mrope_section = config.rope_scaling["mrope_section"]


class NeuronQwen3VLDecoderLayer(nn.Module):
    """Combines self‑attention, optional cross‑attention (vision) and MLP."""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3VLAttention(config)

        self.mlp = NeuronLlamaMLP(config)  # can reuse LlamaMLP module

        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        layer_norm_hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        self_attention_hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=layer_norm_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rotary_position_ids=rotary_position_ids,
            **kwargs,
        )
        hidden_states = residual + self_attention_hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        # Return the final hidden_states and the residual before MLP for potential fused ops
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronQwen3VLTextModel(NeuronBaseModel):

    @staticmethod
    def deepstack_process_xla(
        hidden_states: torch.Tensor,
        visual_embeds: torch.Tensor,
        vision_mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrates visual embeddings into hidden states at specified positions.

        This implementation is optimized for XLA compilation by avoiding
        dynamic indexing operations.

        Args:
            hidden_states: Language model hidden states.
                Shape: [batch_size, seq_len, hidden_dim]
            visual_embeds: Visual embeddings to be inserted.
                Shape: [batch_size, seq_len, hidden_dim]
            vision_mask_positions: Indices indicating where visual embeddings
                should be placed within the sequence.
                Shape: [1, seq_len, 1]

        Note: seq_len contains both text and vision tokens. Hence, visual_embeds must be padded to match hidden_states shape.

        Returns:
            torch.Tensor: Updated hidden states with visual embeddings added
                at the specified mask positions.
                Shape: [batch_size, seq_len, hidden_dim]
        """
        expanded_visual_embeds = torch.zeros_like(hidden_states)
        expanded_visual_embeds = scatter_by_index_put(
            expanded_visual_embeds, visual_embeds, vision_mask_positions
        )
        return hidden_states + expanded_visual_embeds

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask) -> torch.Tensor:
        # Concat vision and text embeddings during context encoding
        # Both inputs_embeds and vision_embeddings should be of the same shape: [BS, Total tokens (image + text), Hidden]
        # And vision_mask should be of the shape [BS, Total tokens (image + text), 1]
        # Entries in vision_mask with value `True` represent vision tokens and with value `False` represent text tokens
        # For text-only inputs, vision_mask should be all `False`
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=False,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                self.hidden_size,
                self.vocab_size,
                bias=False,
            )

        self.layers = nn.ModuleList(
            [NeuronQwen3VLDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


class NeuronQwen3VLTextForCausalLM(NeuronBaseForCausalLM):

    _model_cls = NeuronQwen3VLTextModel

    @staticmethod
    def load_hf_model(model_path):
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path)

        return model

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        # text model state dict convertion
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
        }
        new_state_dict = {}
        for dict_key in state_dict:
            if "language_model." in dict_key:
                new_key = dict_key.replace("language_model.", "")
                if not inference_config.neuron_config.fused_qkv:
                    for atten_key in attention_keys:
                        if atten_key in new_key:
                            replacement_atten_key = attention_keys[atten_key]
                            new_key = new_key.replace(atten_key, replacement_atten_key)
                if ".q_norm." in dict_key:
                    new_key = new_key.replace(".q_norm.", ".q_layernorm.")
                if ".k_norm." in dict_key:
                    new_key = new_key.replace(".k_norm.", ".k_layernorm.")
                new_state_dict[new_key] = state_dict[dict_key]
            else:
                new_state_dict[dict_key] = state_dict[dict_key]

        if inference_config.neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(new_state_dict, inference_config)

        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return InferenceConfig


def _helper_concat_and_delete_qkv(qwen_state_dict, layer_num, attr):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    Args:
        qwen_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    qwen_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            qwen_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            qwen_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            qwen_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del qwen_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del qwen_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del qwen_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(qwen_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv, and deletes the qkv weights.
    """
    mods_to_not_conv = getattr(cfg.neuron_config, "modules_to_not_convert", None)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for layer in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(qwen_state_dict, layer, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{layer}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(qwen_state_dict, layer, "scale")

    gc.collect()

    return qwen_state_dict


class NeuronQwen3VLTextModelWrapper(ImageToTextModelWrapper):

    @staticmethod
    def get_dummy_vision_inputs(config, input_ids, n_active_tokens, fill_value):
        input_batch_size, input_sequence_len = input_ids.shape[0], input_ids.shape[-1]
        if input_sequence_len > 1:  # prefill
            vision_embeddings = torch.zeros(
                input_batch_size,
                n_active_tokens,
                config.hidden_size,
                dtype=config.neuron_config.torch_dtype,
            )
            # we trace not actual vision mask, but positions, for best performance.
            vision_mask = torch.full(
                size=(input_batch_size, n_active_tokens, 1),
                fill_value=fill_value,
                dtype=torch.int32,
            )
            deepstack_vision_embeds = [
                torch.zeros(
                    input_batch_size,
                    n_active_tokens,
                    config.hidden_size,
                    dtype=config.neuron_config.torch_dtype,
                )
                for _ in config.deepstack_visual_indexes
            ]
            if len(deepstack_vision_embeds) > 0:  # vision+text
                deepstack_vision_embeds = torch.stack(deepstack_vision_embeds)
            else:  # text only
                deepstack_vision_embeds = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
        else:  # decode
            vision_embeddings = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
            vision_mask = torch.zeros((0), dtype=torch.bool)
            deepstack_vision_embeds = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
        return vision_embeddings, vision_mask, deepstack_vision_embeds

    def input_generator(
        self,
    ):
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket
                if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )

            input_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            attention_mask = torch.zeros((self.neuron_config.batch_size, bucket), dtype=torch.int32)
            position_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)

            # Get the count of sampling params currently supported.
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros(
                (self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32
            )
            # During model tracing, we fill vision embeddings and vision_mask with zeros
            vision_embeddings, vision_mask, deepstack_vision_embeds = self.get_dummy_vision_inputs(
                config=self.config,
                input_ids=input_ids,
                n_active_tokens=n_active_tokens,
                fill_value=0,
            )

            # Use this for multimodal rotary positional embedding calculation
            rotary_position_ids = torch.zeros(
                (3, self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )  # hardcoded `3` is time, height, width. This is unique in Qwen3 VL
            # which uses an enhanced MRope with interleaved layout for better spatial-temporal modeling

            if self.tag == CONTEXT_ENCODING_MODEL_TAG or self.tag == TOKEN_GENERATION_MODEL_TAG:
                inputs.append(
                    (
                        input_ids,                  # 0
                        attention_mask,             # 1
                        position_ids,               # 2
                        seq_ids,                    # 3
                        sampling_params,            # 4
                        torch.empty(0),             # 5  prev_hidden
                        torch.empty(0),             # 6  adapter_ids
                        torch.empty(0),             # 7  accepted_indices
                        torch.empty(0),             # 8  current_length
                        torch.empty(0),             # 9  medusa_mask
                        torch.empty(0),             # 10 scatter_index
                        torch.empty(0),             # 11 slot_mapping=None,
                        torch.empty(0),             # 12 active_block_table=None,
                        torch.empty(0),             # 13 num_queries=None,
                        torch.empty(0),             # 14 computed_context_lens=None,
                        torch.empty(0),             # 15 tile_q_indices=None,
                        torch.empty(0),             # 16 tile_block_tables=None,
                        torch.empty(0),             # 17 tile_masks=None,
                        torch.empty(0),             # 18 inputs_embeds: Optional[torch.FloatTensor] = None,
                        torch.empty(0),             # 19 kv_cache: Optional[torch.Tensor] = None,
                        torch.empty(0),             # 20 active_mask=None,
                        rotary_position_ids,        # 21 (new in Qwen3 VL to calculate enhance MRope with 3D position_ids)
                        vision_embeddings,          # 22
                        vision_mask,                # 23
                        deepstack_vision_embeds,    # 24 (new in Qwen3 VL)
                    )
                )
            else:
                raise ValueError(f"Unsupported model tag '{self.tag}' for ImageToText models")

        return inputs
