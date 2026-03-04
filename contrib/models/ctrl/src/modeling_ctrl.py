"""
CTRL Model Implementation for NeuronX

Follows the proven GPT2 porting pattern from NeuroborosFoundations:
- ParallelEmbedding for position embeddings (required for HLO compilation)
- Override forward() to inject position embeddings via inputs_embeds
- Custom NeuronConfig with attn_cls (required for token generation tracing)

Bugs fixed (vs original):
1. Attention scaling: use default 1/sqrt(head_dim=80) matching HF reference
   (was incorrectly overridden to 1/sqrt(d_model=1280))
2. Position encoding layout: use HF's [sines|cosines] concatenated layout
   (was using interleaved [sin,cos,sin,cos,...])
3. lm_head bias: include bias=True and load lm_head.bias from HF checkpoint
4. Position embeddings: use ParallelEmbedding (not nn.Embedding/nn.Parameter)
5. Model forward: override forward() not get_model_output() for embedding injection
"""

import json
import logging
import math
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    NeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)

logger = logging.getLogger("Neuron")


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    """Create sinusoidal position encodings matching HF CTRL's layout.

    HF CTRL uses concatenated layout: [sin(f0),...,sin(fN), cos(f0),...,cos(fN)]
    NOT the standard interleaved layout: [sin(f0),cos(f0),sin(f1),cos(f1),...]
    """
    position = torch.arange(num_pos, dtype=torch.float).unsqueeze(1)
    dim_indices = torch.arange(dim, dtype=torch.float).unsqueeze(0)
    angle_rates = 1.0 / torch.pow(10000.0, (2.0 * (dim_indices // 2)) / dim)
    angle_rads = position * angle_rates

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])
    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding


class CTRLNeuronConfig(NeuronConfig):
    """Custom NeuronConfig for CTRL - required for token generation tracing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = None  # Set after class definition


class CTRLInferenceConfig(InferenceConfig):
    def __init__(
        self,
        vocab_size=246534,
        n_positions=50000,
        n_embd=1280,
        dff=8192,
        n_layer=48,
        n_head=16,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        neuron_config: Optional[NeuronConfig] = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.hidden_size = n_embd
        self.max_position_embeddings = n_positions
        self.dff = dff
        self.intermediate_size = dff
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_head
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        super().__init__(neuron_config=neuron_config, **kwargs)

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.sliding_window = None
        if not hasattr(self, 'head_dim'):
            self.head_dim = self.hidden_size // self.num_attention_heads

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return CTRLNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "CTRLInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)
        config_path = os.path.join(model_path, "config.json")

        with open(config_path, "r") as f:
            params = json.load(f)

        config_dict = {
            "vocab_size": params.get("vocab_size", 246534),
            "n_positions": params.get("n_positions", 50000),
            "n_embd": params.get("n_embd", 1280),
            "dff": params.get("dff", 8192),
            "n_layer": params.get("n_layer", 48),
            "n_head": params.get("n_head", 16),
            "resid_pdrop": params.get("resid_pdrop", 0.1),
            "embd_pdrop": params.get("embd_pdrop", 0.1),
            "layer_norm_epsilon": params.get("layer_norm_epsilon", 1e-6),
            "initializer_range": params.get("initializer_range", 0.02),
            "use_cache": params.get("use_cache", True),
        }

        config_dict.update(kwargs)

        if neuron_config is None:
            neuron_config = cls.get_neuron_config_cls()()

        return cls(neuron_config=neuron_config, **config_dict)

    @classmethod
    def load(cls, model_path: str, **kwargs) -> "CTRLInferenceConfig":
        return cls.from_pretrained(model_path, **kwargs)


class NeuronCTRLAttention(NeuronAttentionBase):
    """CTRL attention - uses default 1/sqrt(head_dim) scaling from base class.

    HF CTRL uses dk = k.shape[-1] = head_dim for scaling, which matches the
    NeuronAttentionBase default. No scaled_qk override needed.
    """

    def __init__(self, config: CTRLInferenceConfig):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=None,
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=True,
            o_bias=True,
            sliding_window=None,
        )


class NeuronCTRLMLP(nn.Module):
    def __init__(self, config: CTRLInferenceConfig):
        super().__init__()
        self.fc_in = ColumnParallelLinear(
            config.hidden_size,
            config.dff,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act_fn = nn.ReLU()
        self.fc_out = RowParallelLinear(
            config.dff,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, x):
        x = self.fc_in(x)
        x = self.act_fn(x)
        x = self.fc_out(x)
        return x, None


class NeuronCTRLDecoderLayer(nn.Module):
    def __init__(self, config: CTRLInferenceConfig):
        super().__init__()
        self.self_attn = NeuronCTRLAttention(config)
        self.mlp = NeuronCTRLMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output, _ = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronCTRLModel(NeuronBaseModel):
    """CTRL model following the GPT2 porting pattern.

    Uses ParallelEmbedding for position embeddings and overrides forward()
    to inject position embeddings via inputs_embeds before calling super().
    """

    def setup_attr_for_model(self, config: CTRLInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: CTRLInferenceConfig):
        self.padding_idx = None
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )

        # Position embeddings - ParallelEmbedding (same pattern as GPT2)
        self.wpe = ParallelEmbedding(
            config.n_positions,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )

        # Embedding scale factor: sqrt(d_model)
        self.embed_scale = math.sqrt(config.hidden_size)

        # LM head with bias (CTRL has separate lm_head.bias in HF checkpoint)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronCTRLDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False

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
        inputs_embeds=None,
        kv_cache=None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
        **kwargs,
    ):
        """Override forward to add position embeddings before calling base class."""
        if inputs_embeds is None and input_ids is not None:
            batch_size, seq_length = input_ids.shape

            # Token embeddings scaled by sqrt(d_model)
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds * self.embed_scale

            # Position embeddings
            if position_ids is None:
                device = input_ids.device
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

            position_embeds = self.wpe(position_ids)
            inputs_embeds = inputs_embeds + position_embeds

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            prev_hidden=prev_hidden,
            adapter_ids=adapter_ids,
            accepted_indices=accepted_indices,
            current_length=current_length,
            medusa_mask=medusa_mask,
            scatter_index=scatter_index,
            slot_mapping=slot_mapping,
            active_block_table=active_block_table,
            num_queries=num_queries,
            computed_context_lens=computed_context_lens,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
            active_mask=active_mask,
            rotary_position_id=rotary_position_id,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            **kwargs,
        )


class NeuronCTRLForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronCTRLModel

    def __init__(self, model_path: str, config: CTRLInferenceConfig = None, neuron_config: NeuronConfig = None):
        super().__init__(model_path=model_path, config=config, neuron_config=neuron_config)

    @classmethod
    def from_config(cls, config: CTRLInferenceConfig):
        model_path = getattr(config, 'model_path', '.')
        return cls(model_path=model_path, config=config)

    @classmethod
    def get_config_cls(cls):
        return CTRLInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        neuron_state_dict = {}
        tp_degree = config.neuron_config.tp_degree

        # Token embeddings
        if "transformer.w.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["transformer.w.weight"]

        # LM head weight and bias (separate from embedding in HF checkpoint)
        if "lm_head.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
        elif "transformer.w.weight" in state_dict:
            # Tied weights - clone to avoid shared memory during save
            neuron_state_dict["lm_head.weight"] = state_dict["transformer.w.weight"].clone()
        if "lm_head.bias" in state_dict:
            neuron_state_dict["lm_head.bias"] = state_dict["lm_head.bias"]

        # Position embeddings - generate sinusoidal matching HF layout
        # HF CTRL computes these dynamically, they're not in the checkpoint
        pos_enc = create_sinusoidal_positions(config.n_positions, config.hidden_size)
        neuron_state_dict["wpe.weight"] = pos_enc

        # Final layer norm
        if "transformer.layernorm.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["transformer.layernorm.weight"]
        if "transformer.layernorm.bias" in state_dict:
            neuron_state_dict["norm.bias"] = state_dict["transformer.layernorm.bias"]

        # Decoder layers
        for i in range(config.num_hidden_layers):
            layer_prefix = f"transformer.h.{i}"
            neuron_prefix = f"layers.{i}"

            mappings = [
                (f"{layer_prefix}.multi_head_attention.Wq.weight", f"{neuron_prefix}.self_attn.qkv_proj.q_proj.weight"),
                (f"{layer_prefix}.multi_head_attention.Wq.bias", f"{neuron_prefix}.self_attn.qkv_proj.q_proj.bias"),
                (f"{layer_prefix}.multi_head_attention.Wk.weight", f"{neuron_prefix}.self_attn.qkv_proj.k_proj.weight"),
                (f"{layer_prefix}.multi_head_attention.Wk.bias", f"{neuron_prefix}.self_attn.qkv_proj.k_proj.bias"),
                (f"{layer_prefix}.multi_head_attention.Wv.weight", f"{neuron_prefix}.self_attn.qkv_proj.v_proj.weight"),
                (f"{layer_prefix}.multi_head_attention.Wv.bias", f"{neuron_prefix}.self_attn.qkv_proj.v_proj.bias"),
                (f"{layer_prefix}.multi_head_attention.dense.weight", f"{neuron_prefix}.self_attn.o_proj.weight"),
                (f"{layer_prefix}.multi_head_attention.dense.bias", f"{neuron_prefix}.self_attn.o_proj.bias"),
                (f"{layer_prefix}.ffn.0.weight", f"{neuron_prefix}.mlp.fc_in.weight"),
                (f"{layer_prefix}.ffn.0.bias", f"{neuron_prefix}.mlp.fc_in.bias"),
                (f"{layer_prefix}.ffn.2.weight", f"{neuron_prefix}.mlp.fc_out.weight"),
                (f"{layer_prefix}.ffn.2.bias", f"{neuron_prefix}.mlp.fc_out.bias"),
                (f"{layer_prefix}.layernorm1.weight", f"{neuron_prefix}.input_layernorm.weight"),
                (f"{layer_prefix}.layernorm1.bias", f"{neuron_prefix}.input_layernorm.bias"),
                (f"{layer_prefix}.layernorm2.weight", f"{neuron_prefix}.post_attention_layernorm.weight"),
                (f"{layer_prefix}.layernorm2.bias", f"{neuron_prefix}.post_attention_layernorm.bias"),
            ]

            for src, dst in mappings:
                if src in state_dict:
                    neuron_state_dict[dst] = state_dict[src]

            neuron_state_dict[f"{neuron_prefix}.self_attn.rank_util.rank"] = \
                torch.arange(0, tp_degree, dtype=torch.int32)

        if config.neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = \
                torch.arange(0, config.neuron_config.local_ranks_size, dtype=torch.int32)

        return neuron_state_dict


# Set attention class after all classes are defined (same pattern as GPT2)
def _init_ctrl_neuron_config(self, **kwargs):
    super(CTRLNeuronConfig, self).__init__(**kwargs)
    self.attn_cls = NeuronCTRLAttention

CTRLNeuronConfig.__init__ = _init_ctrl_neuron_config


__all__ = [
    "CTRLInferenceConfig",
    "CTRLNeuronConfig",
    "NeuronCTRLAttention",
    "NeuronCTRLMLP",
    "NeuronCTRLDecoderLayer",
    "NeuronCTRLModel",
    "NeuronCTRLForCausalLM",
]
