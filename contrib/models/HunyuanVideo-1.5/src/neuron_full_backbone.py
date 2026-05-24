"""
Full 54-block transformer backbone for Neuron with TP support.
Uses ColumnParallelLinear/RowParallelLinear for tensor parallelism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


class FullBackboneConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for attr, default in [
            ("num_layers", 54), ("hidden_size", 2048), ("num_heads", 16),
            ("head_dim", 128), ("mlp_ratio", 4.0), ("n_latent", 400), ("n_encoder", 1985),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default)

    def get_required_attributes(self) -> List[str]:
        return ["num_layers", "hidden_size", "num_heads", "head_dim"]


# ── Submodules with TP ──

class TPAdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ColumnParallelLinear(dim, 6 * dim, gather_output=True)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift, scale, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale)[:, None] + shift[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class TPJointAttention(nn.Module):
    def __init__(self, dim=2048, num_heads=16, head_dim=128):
        super().__init__()
        self.head_dim = head_dim
        tp = get_tensor_model_parallel_size()
        self.num_heads = num_heads // tp

        self.to_q = ColumnParallelLinear(dim, num_heads * head_dim, bias=True, gather_output=False)
        self.to_k = ColumnParallelLinear(dim, num_heads * head_dim, bias=True, gather_output=False)
        self.to_v = ColumnParallelLinear(dim, num_heads * head_dim, bias=True, gather_output=False)
        self.to_out = RowParallelLinear(num_heads * head_dim, dim, bias=True, input_is_parallel=True)
        self.norm_q = CustomRMSNorm(head_dim, eps=1e-6)
        self.norm_k = CustomRMSNorm(head_dim, eps=1e-6)

        self.add_q_proj = ColumnParallelLinear(dim, num_heads * head_dim, bias=True, gather_output=False)
        self.add_k_proj = ColumnParallelLinear(dim, num_heads * head_dim, bias=True, gather_output=False)
        self.add_v_proj = ColumnParallelLinear(dim, num_heads * head_dim, bias=True, gather_output=False)
        self.to_add_out = RowParallelLinear(num_heads * head_dim, dim, bias=True, input_is_parallel=True)
        self.norm_added_q = CustomRMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = CustomRMSNorm(head_dim, eps=1e-6)

    def forward(self, hs, enc_hs, attn_mask, freqs_cos, freqs_sin):
        bsz, n_lat, _ = hs.shape
        q = self.to_q(hs).view(bsz, n_lat, self.num_heads, self.head_dim)
        k = self.to_k(hs).view(bsz, n_lat, self.num_heads, self.head_dim)
        v = self.to_v(hs).view(bsz, n_lat, self.num_heads, self.head_dim)
        q, k = self.norm_q(q), self.norm_k(k)
        q, k = self._apply_rope(q, k, freqs_cos, freqs_sin)

        n_enc = enc_hs.shape[1]
        eq = self.add_q_proj(enc_hs).view(bsz, n_enc, self.num_heads, self.head_dim)
        ek = self.add_k_proj(enc_hs).view(bsz, n_enc, self.num_heads, self.head_dim)
        ev = self.add_v_proj(enc_hs).view(bsz, n_enc, self.num_heads, self.head_dim)
        eq, ek = self.norm_added_q(eq), self.norm_added_k(ek)

        q_cat = torch.cat([q, eq], dim=1).transpose(1, 2)
        k_cat = torch.cat([k, ek], dim=1).transpose(1, 2)
        v_cat = torch.cat([v, ev], dim=1).transpose(1, 2)

        out = F.scaled_dot_product_attention(q_cat, k_cat, v_cat, attn_mask=attn_mask)
        out = out.transpose(1, 2).flatten(2)

        return self.to_out(out[:, :n_lat]), self.to_add_out(out[:, n_lat:])

    def _apply_rope(self, q, k, cos, sin):
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        qr, qi = q.reshape(*q.shape[:-1], -1, 2).unbind(-1)
        kr, ki = k.reshape(*k.shape[:-1], -1, 2).unbind(-1)
        q_rot = torch.stack([-qi, qr], dim=-1).flatten(-2)
        k_rot = torch.stack([-ki, kr], dim=-1).flatten(-2)
        return (q.float() * cos + q_rot.float() * sin).to(q.dtype), \
               (k.float() * cos + k_rot.float() * sin).to(k.dtype)


class TPFFN(nn.Module):
    def __init__(self, dim=2048, mult=4.0):
        super().__init__()
        inner = int(dim * mult)
        self.proj = ColumnParallelLinear(dim, inner, bias=True, gather_output=False)
        self.act = nn.GELU(approximate="tanh")
        self.out = RowParallelLinear(inner, dim, bias=True, input_is_parallel=True)

    def forward(self, x):
        return self.out(self.act(self.proj(x)))


class TPTransformerBlock(nn.Module):
    def __init__(self, dim=2048, num_heads=16, head_dim=128, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = TPAdaLayerNormZero(dim)
        self.norm1_context = TPAdaLayerNormZero(dim)
        self.attn = TPJointAttention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = TPFFN(dim, mult=mlp_ratio)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = TPFFN(dim, mult=mlp_ratio)

    def forward(self, hs, enc_hs, temb, attn_mask, freqs_cos, freqs_sin):
        norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hs, temb)
        norm_enc, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(enc_hs, temb)

        attn_out, ctx_out = self.attn(norm_hs, norm_enc, attn_mask, freqs_cos, freqs_sin)

        hs = hs + attn_out * gate_msa.unsqueeze(1)
        enc_hs = enc_hs + ctx_out * c_gate_msa.unsqueeze(1)

        hs = hs + gate_mlp.unsqueeze(1) * self.ff(self.norm2(hs) * (1 + scale_mlp[:, None]) + shift_mlp[:, None])
        enc_hs = enc_hs + c_gate_mlp.unsqueeze(1) * self.ff_context(self.norm2_context(enc_hs) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None])

        return hs, enc_hs


class NeuronFullBackbone(nn.Module):
    """Full 54-block transformer backbone with TP."""
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            TPTransformerBlock(config.hidden_size, config.num_heads, config.head_dim, config.mlp_ratio)
            for _ in range(config.num_layers)
        ])

    def forward(self, hidden_states, encoder_hidden_states, temb, attn_mask, freqs_cos, freqs_sin):
        for block in self.blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attn_mask, freqs_cos, freqs_sin
            )
        return hidden_states


# ── NxDI Wrapper ──

class BackboneModelWrapper(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None,
                 priority_model_idx=None, model_init_kwargs={}):
        super().__init__(config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs)
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        dtype = self.config.neuron_config.torch_dtype
        n_lat = self.config.n_latent
        n_enc = self.config.n_encoder
        dim = self.config.hidden_size
        total = n_lat + n_enc
        return [(
            torch.randn(1, n_lat, dim, dtype=dtype),
            torch.randn(1, n_enc, dim, dtype=dtype),
            torch.randn(1, dim, dtype=dtype),
            torch.zeros(1, 1, total, total, dtype=dtype),
            torch.randn(n_lat, self.config.head_dim, dtype=dtype),
            torch.randn(n_lat, self.config.head_dim, dtype=dtype),
        )]

    def get_model_instance(self):
        def _create():
            m = self.model_cls(self.config)
            m.to(self.config.neuron_config.torch_dtype).eval()
            return m
        return BaseModelInstance(module_cls=_create, input_output_aliases={})

    def forward(self, *args):
        if self.model is None:
            raise RuntimeError("Forward called before load.")
        return self._forward(*args)


class NeuronFullBackboneApplication(NeuronApplicationBase):
    _model_cls = NeuronFullBackbone

    def __init__(self, model_path, *args, **kwargs):
        super().__init__(model_path, *args, **kwargs)
        self.model = BackboneModelWrapper(
            config=self.config, model_cls=self._model_cls,
            tag="NeuronFullBackbone", compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)

    def forward(self, *args):
        return self.models[0](*args)

    def get_compiler_args(self):
        return "--model-type=transformer -O1 --auto-cast=none"

    @staticmethod
    def load_hf_model(model_path):
        from diffusers import HunyuanVideo15Transformer3DModel
        return HunyuanVideo15Transformer3DModel.from_pretrained(model_path.rstrip("/"), subfolder="transformer")

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        new_sd = {}
        for k, v in state_dict.items():
            if not k.startswith("transformer_blocks."):
                continue
            nk = k
            nk = nk.replace("attn.to_out.0.", "attn.to_out.")
            nk = nk.replace(".ff.net.0.proj.", ".ff.proj.")
            nk = nk.replace(".ff.net.2.", ".ff.out.")
            nk = nk.replace(".ff_context.net.0.proj.", ".ff_context.proj.")
            nk = nk.replace(".ff_context.net.2.", ".ff_context.out.")
            # transformer_blocks.X → blocks.X
            nk = nk.replace("transformer_blocks.", "blocks.")
            new_sd[nk] = v.clone().detach().contiguous()
        return new_sd
