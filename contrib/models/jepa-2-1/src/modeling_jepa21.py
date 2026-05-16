# V-JEPA 2.1 encoder for Neuron inference
# Self-contained port from https://github.com/facebookresearch/vjepa2
# Original license: MIT
#
# This file contains the V-JEPA 2.1 Vision Transformer encoder and predictor,
# adapted for inference on AWS Trainium via torch_neuronx.trace().
# All upstream imports have been replaced with inline implementations.

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# NKI flash attention kernel — only available on Neuron devices
_nki_flash_attn = None
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
    from torch_neuronx.xla_impl.ops import nki_jit
    _nki_flash_attn = nki_jit()(attention_isa_kernel)
except ImportError:
    pass

# Modular compilation markers — only available with NxDI
_ModuleMarkerStart = None
_ModuleMarkerEnd = None
try:
    from neuronx_distributed_inference.models.layer_boundary_marker import (
        ModuleMarkerStartWrapper as _ModuleMarkerStart,
        ModuleMarkerEndWrapper as _ModuleMarkerEnd,
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Utility: truncated normal init (replaces src.utils.tensors.trunc_normal_)
# ---------------------------------------------------------------------------
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    return tensor


# ---------------------------------------------------------------------------
# Utility: drop_path (replaces timm.models.layers.drop_path)
# At eval, this is identity. Included for completeness.
# ---------------------------------------------------------------------------
def drop_path_fn(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor_(random_tensor + keep_prob)
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_fn(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Patch Embeddings
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Image to Patch Embedding (2D)."""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding (3D tubelets)."""

    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        # x: (B, C, T, H, W)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# 3D Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------
def rotate_queries_or_keys_v21(x, pos, n_registers=0, has_cls_first=False):
    """V-JEPA 2.1 RoPE rotation (uses repeat_interleave, not repeat)."""
    B, num_heads, N, D = x.size()
    assert D % 2 == 0

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)

    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)

    parts = []
    if n_cls:
        parts.append(x_cls)
    parts.append(out_ctx)
    if n_registers:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------
class RoPEAttention(nn.Module):
    """Multi-head attention with 3D RoPE for V-JEPA 2.1."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        use_nki_flash=False,
        grid_size=14,
        is_causal=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.use_nki_flash = use_nki_flash and (_nki_flash_attn is not None)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.pretrained_patch_size = patch_size
        if patch_size == 14:
            self.pretrained_grid_size = int(252 / patch_size)
        elif patch_size == 16:
            self.pretrained_grid_size = int(256 / patch_size)
        else:
            self.pretrained_grid_size = grid_size

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
        B, N, C = x.size()

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                N_ctx = N - self.n_registers
                grid_depth = int(N_ctx // (self.grid_size * self.grid_size))
                mask = torch.arange(
                    int(grid_depth * self.grid_size * self.grid_size), device=x.device
                )
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        if self.interpolate_rope:
            if H_patches is None:
                H_patches = int(self.grid_size)
            if W_patches is None:
                W_patches = int(self.grid_size)
            h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)

        s = 0
        qd = rotate_queries_or_keys_v21(q[..., s:s + self.d_dim], pos=d_mask,
                                         n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        kd = rotate_queries_or_keys_v21(k[..., s:s + self.d_dim], pos=d_mask,
                                         n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        s += self.d_dim
        qh = rotate_queries_or_keys_v21(q[..., s:s + self.h_dim], pos=h_mask,
                                         n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        kh = rotate_queries_or_keys_v21(k[..., s:s + self.h_dim], pos=h_mask,
                                         n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        s += self.h_dim
        qw = rotate_queries_or_keys_v21(q[..., s:s + self.w_dim], pos=w_mask,
                                         n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        kw = rotate_queries_or_keys_v21(k[..., s:s + self.w_dim], pos=w_mask,
                                         n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        s += self.w_dim

        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if self.use_nki_flash:
            # NKI ISA kernel layout: q/k=(B*H, d, seqlen), v=(B*H, seqlen, d), out=(B*H, seqlen, d)
            q_nki = q.reshape(B * self.num_heads, N, self.head_dim).permute(0, 2, 1).contiguous()
            k_nki = k.reshape(B * self.num_heads, N, self.head_dim).permute(0, 2, 1).contiguous()
            v_nki = v.reshape(B * self.num_heads, N, self.head_dim).contiguous()
            attn_output = torch.zeros(B * self.num_heads, N, self.head_dim, dtype=q.dtype, device=q.device)
            _nki_flash_attn(q_nki, k_nki, v_nki, self.scale, attn_output,
                            kernel_name="AttentionMMSoftmaxMMWithoutSwap")
            x = attn_output.reshape(B, self.num_heads, N, self.head_dim)
            attn = None
        elif self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal
                )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1).to(v.dtype)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x, None


class Attention(nn.Module):
    """Standard multi-head attention (no RoPE)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, use_sdpa=True, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal
                )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1).to(v.dtype)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# MLP Modules
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.SiLU, drop=0.0, wide_silu=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        swiglu_hidden_features = hidden_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
        drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, wide_silu=True,
        norm_layer=nn.LayerNorm, use_sdpa=True, use_nki_flash=False, is_causal=False,
        grid_size=16, use_rope=False, n_registers=0, has_cls_first=False,
        interpolate_rope=False, patch_size=16, **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_rope = use_rope
        if use_rope:
            self.attn = RoPEAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, use_sdpa=use_sdpa, use_nki_flash=use_nki_flash,
                is_causal=is_causal, grid_size=grid_size, proj_drop=drop,
                n_registers=n_registers, has_cls_first=has_cls_first,
                interpolate_rope=interpolate_rope, patch_size=patch_size,
            )
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, use_sdpa=use_sdpa, is_causal=is_causal, proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim,
                act_layer=act_layer, wide_silu=wide_silu, drop=drop,
            )
        else:
            self.mlp = MLP(
                in_features=dim, hidden_features=mlp_hidden_dim,
                act_layer=act_layer, drop=drop,
            )

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None,
                return_attn=False, mode="video"):
        if self.use_rope:
            y, attn = self.attn(
                self.norm1(x), mask=mask, T=T, H_patches=H_patches,
                W_patches=W_patches, return_attn=return_attn,
            )
        else:
            y = self.attn(self.norm1(x))
            attn = None
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        return x, None


# ---------------------------------------------------------------------------
# V-JEPA 2.1 Vision Transformer Encoder
# ---------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    """V-JEPA 2.1 Vision Transformer encoder for inference."""

    def __init__(
        self,
        img_size=(384, 384),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_nki_flash=False,
        modular_compilation_group_size=0,
        use_activation_checkpointing=False,
        is_causal=False,
        use_rope=True,
        handle_nonsquare_inputs=True,
        img_temporal_dim_size=None,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        n_output_distillation=4,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.use_activation_checkpointing = use_activation_checkpointing
        self.modular_compilation_group_size = modular_compilation_group_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size, tubelet_size=tubelet_size,
                in_chans=in_chans, embed_dim=embed_dim,
            )
            self.num_patches = (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # Separate image patch embed for img_temporal_dim_size
        if self.img_temporal_dim_size is not None:
            self.patch_embed_img = PatchEmbed3D(
                patch_size=patch_size, tubelet_size=1,
                in_chans=in_chans, embed_dim=embed_dim,
            )
        else:
            self.patch_embed_img = None

        self.uniform_power = uniform_power
        self.use_rope = use_rope

        self.blocks = nn.ModuleList([
            Block(
                use_rope=use_rope, grid_size=img_size[0] // patch_size,
                grid_depth=num_frames // tubelet_size, dim=embed_dim,
                num_heads=num_heads, mlp_ratio=mlp_ratio, use_sdpa=use_sdpa,
                use_nki_flash=use_nki_flash, is_causal=is_causal,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, act_layer=nn.SiLU if use_silu else nn.GELU,
                wide_silu=wide_silu, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, n_registers=n_registers,
                has_cls_first=has_cls_first, interpolate_rope=interpolate_rope,
                patch_size=patch_size,
            )
            for i in range(depth)
        ])

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Hierarchical layer indices
        _layer_map = {
            12: [2, 5, 8, 11],
            24: [5, 11, 17, 23],
            40: [9, 19, 29, 39],
            48: [11, 23, 37, 47],
        }
        self.hierarchical_layers = _layer_map.get(depth, [depth - 1])

        if n_output_distillation == 4:
            self.out_layers_distillation = self.hierarchical_layers[:]
        elif n_output_distillation == 1:
            self.out_layers_distillation = [self.hierarchical_layers[-1]]
        else:
            self.out_layers_distillation = self.hierarchical_layers[-n_output_distillation:]

        self.norms_block = nn.ModuleList([
            norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))
        ])

        self.cls_token = None
        self.return_hierarchical = False

        # Modality embeddings
        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def check_temporal_dim(self, shape) -> bool:
        if self.img_temporal_dim_size is not None:
            if shape[2] == self.img_temporal_dim_size:
                return True
        return False

    def forward(self, x, masks=None, training=False):
        """
        Args:
            x: input tensor. Image: (B, C, H, W) or (B, C, 1, H, W). Video: (B, C, T, H, W).
            masks: optional mask indices (training only).
            training: if True, return hierarchical concatenated features.
        Returns:
            Tensor of shape (B, N, D) where N is the number of patch tokens.
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        elif x.ndim == 5:
            _, _, T_raw, H, W = x.shape
            if self.check_temporal_dim(x.shape):
                T = T_raw // 1
            else:
                T = T_raw // self.tubelet_size

        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        # Patch embedding
        if x.ndim == 5 and self.check_temporal_dim(x.shape):
            assert self.patch_embed_img is not None
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x = x + self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x = x + self.video_mod_embed.repeat(x.shape[0], 1, 1)

        # Masking (training only)
        if masks is not None:
            from src.masks.utils import apply_masks
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        # Forward through blocks
        gs = self.modular_compilation_group_size
        use_markers = gs > 0 and _ModuleMarkerStart is not None
        hier = []
        for i, blk in enumerate(self.blocks):
            if use_markers and i % gs == 0:
                x = _ModuleMarkerStart()(x)
            x, _attn = blk(
                x, mask=masks, T=T, H_patches=H_patches, W_patches=W_patches,
                return_attn=False, mode=mode,
            )
            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))
            if use_markers and (i % gs == gs - 1 or i == len(self.blocks) - 1):
                x = _ModuleMarkerEnd()(x)

        if training or self.return_hierarchical:
            return torch.cat(hier, dim=2)
        else:
            # Return last hierarchical layer's normed output
            return self.norms_block[-1](x)


# ---------------------------------------------------------------------------
# V-JEPA 2.1 Predictor (for completeness — not needed for basic inference)
# ---------------------------------------------------------------------------
class VisionTransformerPredictor(nn.Module):
    """V-JEPA 2.1 predictor for mask prediction / action anticipation."""

    def __init__(
        self,
        img_size=(384, 384),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=None,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_rope=True,
        n_output_distillation=4,
        teacher_embed_dim=None,
        return_all_tokens=False,
        modality_embedding=True,
        img_temporal_dim_size=None,
        interpolate_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.grid_depth = num_frames // tubelet_size

        if self.is_video:
            self.num_patches = self.grid_depth * self.grid_height * self.grid_width
        else:
            self.num_patches = self.grid_height * self.grid_width

        # Hierarchical layers
        _layer_map = {4: [0,1,2,3], 8: [1,3,5,7], 12: [2,5,8,11], 20: [4,9,14,19], 24: [4,11,17,23], 40: [9,19,29,39]}
        all_hier = _layer_map.get(depth, list(range(depth)))
        self.hierarchical_layers = all_hier[-n_output_distillation:]

        act_layer_mlp = nn.SiLU if use_silu else nn.GELU
        if len(self.hierarchical_layers) == 1:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        else:
            self.predictor_embed = nn.Sequential(
                nn.Linear(embed_dim * len(self.hierarchical_layers), embed_dim, bias=True),
                act_layer_mlp(),
                nn.Linear(embed_dim, predictor_embed_dim, bias=True),
            )

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for _ in range(num_mask_tokens)
            ])

        # Modality embeddings
        self.modality_embedding = False
        if img_temporal_dim_size is not None and modality_embedding:
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            self.modality_embedding = True

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList([
            Block(
                use_rope=use_rope, grid_size=self.grid_height, grid_depth=self.grid_depth,
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                act_layer=nn.SiLU if use_silu else nn.GELU, wide_silu=wide_silu,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                interpolate_rope=interpolate_rope, patch_size=patch_size,
            )
            for i in range(depth)
        ])

        if out_embed_dim is None:
            if teacher_embed_dim is not None:
                out_embed_dim = teacher_embed_dim // len(self.hierarchical_layers)
            else:
                out_embed_dim = embed_dim

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim, len(self.hierarchical_layers) * out_embed_dim, bias=True
        )
        if self.return_all_tokens:
            self.predictor_proj_context = nn.Linear(
                predictor_embed_dim, out_embed_dim * len(self.hierarchical_layers), bias=True
            )

        self.init_std = init_std
        if use_mask_tokens and not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.predictor_blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward(self, x, masks_x=None, masks_y=None, mod="video", mask_index=1):
        """Forward pass. For inference without masks, just pass features through."""
        if masks_x is None or masks_y is None:
            # Simple forward without masking (inference mode)
            x = self.predictor_embed(x)
            for blk in self.predictor_blocks:
                x, _ = blk(x)
            x = self.predictor_norm(x)
            x = self.predictor_proj(x)
            return x, None

        # Full masked prediction (training mode) — not ported for Neuron
        raise NotImplementedError("Masked prediction forward not implemented for Neuron inference")


# ---------------------------------------------------------------------------
# Builder Functions
# ---------------------------------------------------------------------------
_ARCH_CONFIGS = {
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0),
    "vit_large": dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0),
    "vit_giant": dict(embed_dim=1408, depth=40, num_heads=22, mlp_ratio=48 / 11),
    "vit_gigantic": dict(embed_dim=1664, depth=48, num_heads=26, mlp_ratio=64 / 13),
}


def build_vjepa21_encoder(
    arch: str = "vit_large",
    img_size: int = 384,
    num_frames: int = 64,
    patch_size: int = 16,
    tubelet_size: int = 2,
    use_sdpa: bool = True,
    use_nki_flash: bool = False,
    modular_compilation_group_size: int = 0,
    use_rope: bool = True,
    interpolate_rope: bool = True,
    img_temporal_dim_size: int = 1,
    modality_embedding: bool = True,
    n_output_distillation: int = 4,
    pretrained: bool = False,
    **kwargs,
) -> VisionTransformer:
    """Build a V-JEPA 2.1 encoder.

    Args:
        arch: one of 'vit_base', 'vit_large', 'vit_giant', 'vit_gigantic'
        img_size: spatial resolution (square)
        num_frames: number of video frames
        pretrained: if True, load pretrained weights (requires network access)
    """
    if arch not in _ARCH_CONFIGS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(_ARCH_CONFIGS.keys())}")

    cfg = _ARCH_CONFIGS[arch]
    encoder = VisionTransformer(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=use_sdpa,
        use_nki_flash=use_nki_flash,
        modular_compilation_group_size=modular_compilation_group_size,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=use_rope,
        interpolate_rope=interpolate_rope,
        img_temporal_dim_size=img_temporal_dim_size,
        modality_embedding=modality_embedding,
        n_output_distillation=n_output_distillation,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qkv_bias=True,
        **cfg,
        **kwargs,
    )

    if pretrained:
        _load_pretrained_weights(encoder, arch)

    return encoder


def _load_pretrained_weights(encoder, arch):
    """Load pretrained V-JEPA 2.1 weights from Meta's servers."""
    VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
    _CKPT_MAP = {
        "vit_base": "vjepa2_1_vitb_dist_vitG_384",
        "vit_large": "vjepa2_1_vitl_dist_vitG_384",
        "vit_giant": "vjepa2_1_vitg_384",
        "vit_gigantic": "vjepa2_1_vitG_384",
    }
    model_file = _CKPT_MAP[arch]
    url = f"{VJEPA_BASE_URL}/{model_file}.pt"

    checkpoint_key = "ema_encoder" if arch in ("vit_base", "vit_large") else "target_encoder"

    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    encoder_sd = state_dict[checkpoint_key]

    # Clean keys: remove 'module.' and 'backbone.' prefixes
    cleaned = {}
    for k, v in encoder_sd.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v

    encoder.load_state_dict(cleaned, strict=True)
    print(f"Loaded pretrained weights from {url} (key={checkpoint_key})")
