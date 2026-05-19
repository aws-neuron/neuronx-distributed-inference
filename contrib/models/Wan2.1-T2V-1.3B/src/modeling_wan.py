"""
NeuronX implementation of WAN 2.1 T2V 1.3B — standalone model.

Mirrors the HF diffusers WanTransformer3DModel architecture with:
- XLA-compatible operations (no negative indexing, no nearest-exact)
- Pre-computed RoPE as explicit inputs (avoids XLA unused-input bug)
- Same state dict keys as HF for direct weight loading

Usage:
    model = NeuronWanTransformer3DModel.from_config(hf_config)
    model.load_hf_weights(hf_state_dict)
    output = model(hidden_states, timestep, encoder_hidden_states, rope_cos, rope_sin)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Embeddings ──────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class Timesteps(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TextProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class WanConditionEmbedder(nn.Module):
    """Timestep + text conditioning. Produces temb and projects text embeddings."""

    def __init__(self, dim, freq_dim, text_dim):
        super().__init__()
        self.timesteps_proj = Timesteps(freq_dim)
        self.time_embedder = TimestepEmbedding(freq_dim, dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, dim * 6)
        self.text_embedder = TextProjection(text_dim, dim)

    def forward(self, timestep, encoder_hidden_states, **kwargs):
        temb = self.time_embedder(self.timesteps_proj(timestep).to(dtype=self.time_embedder.linear_1.weight.dtype))
        temb = temb.type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


# ─── Attention ───────────────────────────────────────────────────────────────

class WanAttention(nn.Module):
    """Multi-head attention with across-heads RMSNorm on Q/K."""

    def __init__(self, dim, num_heads, dim_head, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=True), nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(inner_dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(inner_dim, eps=eps, elementwise_affine=True)

    def forward(self, hidden_states, encoder_hidden_states=None, rotary_emb=None):
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = self.norm_q(self.to_q(hidden_states))
        key = self.norm_k(self.to_k(kv_input))
        value = self.to_v(kv_input)

        query = query.unflatten(2, (self.num_heads, self.dim_head))
        key = key.unflatten(2, (self.num_heads, self.dim_head))
        value = value.unflatten(2, (self.num_heads, self.dim_head))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        ).transpose(1, 2).flatten(2, 3)

        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


def _apply_rotary_emb(x, freqs_cos, freqs_sin):
    """Apply rotary position embedding. x: [B, S, H, D], freqs: [B, S, 1, D]."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


# ─── Transformer Block ───────────────────────────────────────────────────────

class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, 
                           self.weight.float() if self.weight is not None else None,
                           self.bias.float() if self.bias is not None else None,
                           self.eps).type_as(x)


class WanTransformerBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6, cross_attn_norm=True):
        super().__init__()
        dim_head = dim // num_heads
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(dim, num_heads, dim_head, eps)
        self.attn2 = WanAttention(dim, num_heads, dim_head, eps)
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = WanFeedForward(dim, ffn_dim)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        norm_hs = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_out = self.attn1(norm_hs, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_out * gate_msa).type_as(hidden_states)

        norm_hs = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_out = self.attn2(norm_hs, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_out

        norm_hs = (self.norm3(hidden_states.float()) * (1 + c_scale) + c_shift).type_as(hidden_states)
        ff_out = self.ffn(norm_hs)
        hidden_states = (hidden_states.float() + ff_out.float() * c_gate).type_as(hidden_states)
        return hidden_states


class WanFeedForward(nn.Module):
    """GELU feed-forward: Linear -> GELU -> Dropout -> Linear."""
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.net = nn.ModuleList([
            GELU(dim, ffn_dim),
            nn.Dropout(0.0),
            nn.Linear(ffn_dim, dim, bias=True),
        ])

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class GELU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.gelu(self.proj(x))


# ─── Full Model ──────────────────────────────────────────────────────────────

class NeuronWanTransformer3DModel(nn.Module):
    """Standalone WAN 2.1 backbone for Neuron.

    Takes pre-computed RoPE as inputs to avoid XLA unused-input bug.
    State dict keys match HF WanTransformer3DModel for direct weight loading.
    """

    def __init__(self, num_heads=12, dim_head=128, num_layers=30, ffn_dim=8960,
                 in_channels=16, out_channels=16, text_dim=4096, freq_dim=256,
                 patch_size=(1, 2, 2), eps=1e-6, cross_attn_norm=True):
        super().__init__()
        dim = num_heads * dim_head
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.condition_embedder = WanConditionEmbedder(dim, freq_dim, text_dim)
        self.blocks = nn.ModuleList([
            WanTransformerBlock(dim, ffn_dim, num_heads, eps, cross_attn_norm)
            for _ in range(num_layers)
        ])
        self.norm_out = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, hidden_states, timestep, encoder_hidden_states, rope_cos, rope_sin):
        b, c, f, h, w = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = f // p_t, h // p_h, w // p_w

        rotary_emb = (rope_cos, rope_sin)

        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep.expand(hidden_states.shape[0]), encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(b, ppf, pph, ppw, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    @classmethod
    def from_pretrained(cls, path, torch_dtype=torch.bfloat16):
        """Load from HF WanTransformer3DModel pretrained weights."""
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel as HF
        hf = HF.from_pretrained(path, torch_dtype=torch_dtype)
        cfg = hf.config
        model = cls(
            num_heads=cfg.num_attention_heads, dim_head=cfg.attention_head_dim,
            num_layers=cfg.num_layers, ffn_dim=cfg.ffn_dim,
            in_channels=cfg.in_channels, out_channels=cfg.out_channels,
            text_dim=cfg.text_dim, freq_dim=cfg.freq_dim,
            patch_size=tuple(cfg.patch_size), eps=cfg.eps,
            cross_attn_norm=cfg.cross_attn_norm,
        ).to(torch_dtype)
        model.load_state_dict(hf.state_dict(), strict=False)
        # Compute RoPE helper
        model._rope = hf.rope
        return model

    def compute_rope(self, dummy_latent):
        """Pre-compute RoPE on CPU. Call once, pass results to forward()."""
        return self._rope(dummy_latent)


# ─── VAE Decoder XLA Compatibility ───────────────────────────────────────────

def make_decoder_xla_compatible(decoder):
    """Monkey-patch a WAN VAE decoder to fix XLA incompatibilities.

    1. Replace x[:,:,-N:,:,:] with safe indexing (XLA negative index bug)
    2. Replace mode='nearest-exact' with mode='nearest' (XLA custom-call bug)

    Call once after loading VAE, before tracing. No diffusers source changes.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanResample, WanResidualBlock, WanDecoder3d, CACHE_T,
    )
    _patch_resample_forward(WanResample, CACHE_T)
    _patch_residual_block_forward(WanResidualBlock, CACHE_T)
    _patch_decoder_forward(WanDecoder3d, CACHE_T)
    for m in decoder.modules():
        if isinstance(m, nn.Upsample) and m.mode == "nearest-exact":
            m.mode = "nearest"


def _patch_resample_forward(cls, CACHE_T):
    def safe_forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, max(0, x.shape[2] - CACHE_T):, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat([feat_cache[idx][:, :, max(0, feat_cache[idx].shape[2] - 1), :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, max(0, x.shape[2] - 1):, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
            return x
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        return x
    cls.forward = safe_forward


def _patch_residual_block_forward(cls, CACHE_T):
    def safe_forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.conv_shortcut(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, max(0, x.shape[2] - CACHE_T):, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, max(0, feat_cache[idx].shape[2] - 1), :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, max(0, x.shape[2] - CACHE_T):, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, max(0, feat_cache[idx].shape[2] - 1), :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)
        return x + h
    cls.forward = safe_forward


def _patch_decoder_forward(cls, CACHE_T):
    def safe_forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, max(0, x.shape[2] - CACHE_T):, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, max(0, feat_cache[idx].shape[2] - 1), :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
        x = self.nonlinearity(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, max(0, x.shape[2] - CACHE_T):, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, max(0, feat_cache[idx].shape[2] - 1), :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x
    cls.forward = safe_forward


# ─── VAE Block Wrappers (for Neuron tracing) ─────────────────────────────────

VAE_BLOCK_ORDER = [
    ("conv_in", 0, 1), ("mid_block", 1, 4),
    ("up0_resnet0", 5, 2), ("up0_resnet1", 7, 2), ("up0_resnet2", 9, 2), ("up0_upsample", 11, 1),
    ("up1_resnet0", 12, 2), ("up1_resnet1", 14, 2), ("up1_resnet2", 16, 2), ("up1_upsample", 18, 1),
    ("up2_resnet0", 19, 2), ("up2_resnet1", 21, 2), ("up2_resnet2", 23, 2), ("up2_upsample", 25, 0),
    ("up3_resnet0", 25, 2), ("up3_resnet1", 27, 2), ("up3_resnet2", 29, 2), ("norm_conv_out", 31, 1),
]


class ConvInCached(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
    def forward(self, x, cache):
        out = self.conv(x, cache_x=cache)
        new_cache = torch.cat([cache, x], dim=2)[:, :, max(0, cache.shape[2] + x.shape[2] - 2):, :, :]
        return out, new_cache


class BlockCached(nn.Module):
    def __init__(self, block, cache_start, cache_count):
        super().__init__()
        self.block = block
        self.cs = cache_start
        self.cc = cache_count
    def forward(self, x, *caches):
        fc = [None] * 33
        for i, c in enumerate(caches):
            fc[self.cs + i] = c
        idx = [self.cs]
        out = self.block(x, feat_cache=fc, feat_idx=idx)
        return (out,) + tuple(fc[self.cs + i] for i in range(self.cc))


class NormConvOutCached(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.norm = decoder.norm_out
        self.act = decoder.nonlinearity
        self.conv = decoder.conv_out
    def forward(self, x, cache):
        h = self.act(self.norm(x))
        out = self.conv(h, cache_x=cache)
        new_cache = torch.cat([cache, h], dim=2)[:, :, max(0, cache.shape[2] + h.shape[2] - 2):, :, :]
        return out, new_cache


class NoCacheWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, x):
        return self.block(x)


class UpsampleSpatialOnly(nn.Module):
    """Frame-0 upsampler: spatial only, no time_conv."""
    def __init__(self, block):
        super().__init__()
        self.resample = block.resample
    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        return x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)


class UpsampleFirstChunk(nn.Module):
    """Frame-1 upsampler: time_conv without cache (Rep path)."""
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, x):
        b, c, t, h, w = x.size()
        x = self.block.time_conv(x)
        x = x.reshape(b, 2, c, t, h, w)
        x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
        x = x.reshape(b, c, t * 2, h, w)
        t2 = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t2, c, h, w)
        x = self.block.resample(x)
        return x.view(b, t2, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)


# ─── T5 Wrapper ──────────────────────────────────────────────────────────────

class T5Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
