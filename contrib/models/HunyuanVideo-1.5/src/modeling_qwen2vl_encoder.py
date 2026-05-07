"""
Qwen2.5-VL Text Encoder for HunyuanVideo-1.5 on NxDI

Encoder-only model that extracts hidden_states[-3] from the 28-layer Qwen2.5-VL.
Uses TP=2 with ColumnParallelLinear/RowParallelLinear. No KV cache, no generation.

Usage:
    from modeling_qwen2vl_encoder import (
        Qwen2VLEncoderConfig, NeuronQwen2VLEncoder
    )
    config = Qwen2VLEncoderConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = NeuronQwen2VLEncoder("Qwen/Qwen2.5-VL-7B-Instruct", config=config)
    model.compile(save_path)
    model.load(save_path)
    hidden_states = model.forward(input_ids, attention_mask)  # [B, 1108, 3584]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear, RowParallelLinear, ParallelEmbedding,
)
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance, ModelWrapper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MLLM_SEQ_LEN = 1108  # 103 system tokens + 1000 user tokens + 5 formatting

class Qwen2VLEncoderConfig(InferenceConfig):
    """Configuration for Qwen2.5-VL encoder-only model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_required_attributes(self):
        return ["hidden_size", "num_hidden_layers", "num_attention_heads"]

    @classmethod
    def from_pretrained(cls, model_path: str, tp_degree: int = 2, **kwargs):
        neuron_config = NeuronConfig(
            tp_degree=tp_degree, torch_dtype=torch.bfloat16, batch_size=1,
            seq_len=MLLM_SEQ_LEN, save_sharded_checkpoint=True,
        )
        defaults = dict(
            hidden_size=3584, num_hidden_layers=28, num_attention_heads=28,
            num_key_value_heads=4, intermediate_size=18944, rms_norm_eps=1e-6,
            rope_theta=1000000.0, max_position_embeddings=32768,
            vocab_size=152064, pad_token_id=151643,
        )
        defaults.update(kwargs)
        return cls(neuron_config=neuron_config, **defaults)


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class Qwen2VLAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        tp = config.neuron_config.tp_degree
        self.num_heads_per_tp = config.num_attention_heads // tp
        self.num_kv_heads_per_tp = max(1, config.num_key_value_heads // tp)
        self.q_proj = ColumnParallelLinear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True, gather_output=False)
        self.k_proj = ColumnParallelLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True, gather_output=False)
        self.v_proj = ColumnParallelLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True, gather_output=False)
        self.o_proj = RowParallelLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False, input_is_parallel=True)

    def forward(self, hs, cos, sin, attn_mask):
        B, S, _ = hs.shape
        q = self.q_proj(hs).view(B, S, self.num_heads_per_tp, self.head_dim)
        k = self.k_proj(hs).view(B, S, self.num_kv_heads_per_tp, self.head_dim)
        v = self.v_proj(hs).view(B, S, self.num_kv_heads_per_tp, self.head_dim)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        if self.num_kv_heads_per_tp < self.num_heads_per_tp:
            rep = self.num_heads_per_tp // self.num_kv_heads_per_tp
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class Qwen2VLMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, gather_output=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False, input_is_parallel=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen2VLAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen2VLMLP(config)

    def forward(self, hs, cos, sin, attn_mask):
        residual = hs
        hs = self.self_attn(self.input_layernorm(hs), cos, sin, attn_mask)
        hs = residual + hs
        residual = hs
        hs = self.mlp(self.post_attention_layernorm(hs))
        return residual + hs


# ---------------------------------------------------------------------------
# Full Encoder Model
# ---------------------------------------------------------------------------
class Qwen2VLEncoderModel(nn.Module):
    """
    Qwen2.5-VL encoder-only model.

    Extracts hidden_states[-3] (layer 25 of 28) for use as text embeddings.
    Pre-computes RoPE as buffer (mrope with 3 identical sections for text-only).

    Inputs:
        input_ids: [B, seq_len] — tokenized input
        attention_mask: [B, seq_len] — 1 for valid, 0 for padding

    Output:
        [B, seq_len, 3584] — hidden states from layer 25
    """

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size, config.hidden_size, config.pad_token_id,
            dtype=config.neuron_config.torch_dtype, shard_across_embedding=True, pad=True,
        )
        self.layers = nn.ModuleList([
            Qwen2VLDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.target_layer = config.num_hidden_layers - 3  # hidden_states[-3]
        self._init_rope(config)

    def _init_rope(self, config):
        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (config.rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        t = torch.arange(config.neuron_config.seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # mrope: 3 identical sections for text-only, take first head_dim
        self.register_buffer("rope_cos", torch.cat([cos, cos, cos], dim=-1)[:, :head_dim])
        self.register_buffer("rope_sin", torch.cat([sin, sin, sin], dim=-1)[:, :head_dim])

    def forward(self, input_ids, attention_mask):
        B, S = input_ids.shape
        hs = self.embed_tokens(input_ids)
        causal = torch.triu(torch.full((S, S), -1e9, dtype=hs.dtype, device=hs.device), diagonal=1)
        pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
        mask = (causal.unsqueeze(0) + pad_mask).to(hs.dtype)
        cos = self.rope_cos[:S].unsqueeze(0).unsqueeze(2).to(hs.dtype)
        sin = self.rope_sin[:S].unsqueeze(0).unsqueeze(2).to(hs.dtype)
        extracted = hs
        for i, layer in enumerate(self.layers):
            hs = layer(hs, cos, sin, mask)
            if i == self.target_layer:
                extracted = hs
        return extracted


# ---------------------------------------------------------------------------
# NxDI Wrapper and Application
# ---------------------------------------------------------------------------
class Qwen2VLEncoderWrapper(ModelWrapper):
    def __init__(self, config, model_cls, **kwargs):
        super().__init__(config, model_cls, **kwargs)
        self.bucket_config = None

    def input_generator(self):
        seq_len = self.config.neuron_config.seq_len
        return [(
            torch.zeros(1, seq_len, dtype=torch.long),
            torch.ones(1, seq_len, dtype=torch.long),
        )]

    def get_model_instance(self):
        def _create():
            m = self.model_cls(self.config)
            m.to(self.config.neuron_config.torch_dtype).eval()
            return m
        return BaseModelInstance(module_cls=_create, input_output_aliases={})

    def forward(self, *args):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._forward(*args)


class NeuronQwen2VLEncoder(NeuronApplicationBase):
    """
    Qwen2.5-VL Encoder for NxDI.

    Usage:
        config = Qwen2VLEncoderConfig.from_pretrained(model_path)
        model = NeuronQwen2VLEncoder(model_path, config=config)
        model.compile(save_path)
        model.load(save_path)
        output = model.forward(input_ids, attention_mask)
    """

    _model_cls = Qwen2VLEncoderModel

    def __init__(self, model_path, *args, **kwargs):
        super().__init__(model_path, *args, **kwargs)
        self.model = Qwen2VLEncoderWrapper(
            config=self.config, model_cls=self._model_cls,
            tag="Qwen2VLEncoder", compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)

    def forward(self, *args):
        return self.models[0](*args)

    def get_compiler_args(self):
        return "--model-type=transformer -O1 --auto-cast=none"

    @staticmethod
    def load_hf_model(model_path):
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path.rstrip("/"), torch_dtype=torch.bfloat16,
        )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """
        Convert HuggingFace Qwen2_5_VLForConditionalGeneration state dict.

        Key mapping:
            model.language_model.* -> * (strip both prefixes)
            Skip: lm_head, visual, multi_modal_projector
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[6:]
            if new_key.startswith("language_model."):
                new_key = new_key[15:]
            if any(new_key.startswith(p) for p in ["lm_head", "visual", "multi_modal_projector"]):
                continue
            new_state_dict[new_key] = value
        return new_state_dict
