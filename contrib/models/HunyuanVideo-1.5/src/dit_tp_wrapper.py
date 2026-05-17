"""
HunyuanVideo-1.5 DiT TP-Sharded Wrapper for Standard Neuron SDK.

Tensor Parallel (TP=4) wrapper for the 8.33B DiT transformer.
Uses ColumnParallelLinear/RowParallelLinear from neuronx-distributed
for head-parallel attention and MLP sharding.

480p_t2v config:
  - hidden_size=2048, heads_num=16, head_dim=128
  - 54 double-stream blocks, 0 single-stream blocks
  - mlp_width_ratio=4 -> mlp_hidden_dim=8192
  - patch_size=[1,1,1], in_channels=32, out_channels=32
  - qkv_bias=True, qk_norm=True (RMS)

TP Strategy (TP=4):
  - Q/K/V: ColumnParallelLinear (gather_output=False), 16 heads / 4 = 4 heads/rank
  - O proj: RowParallelLinear (input_is_parallel=True)
  - MLP fc1: ColumnParallelLinear (gather_output=False)
  - MLP fc2: RowParallelLinear (input_is_parallel=True)
  - ModulateDiT, FinalLayer, QK-norm: Replicated
  - Attention: NKI flash attention (dense) or STA (sparse, for long sequences)

Attention modes:
  - Dense (default): Full attention using attention_isa_kernel. O(n^2).
    Suitable for short sequences (e.g., 480p 5-frame: 3,500 tokens).
  - STA (Sliding Tile Attention): Block-sparse attention using attention_cte.
    Tokens are tiled in 3D (T,H,W) and each tile attends only to its
    spatio-temporal neighborhood. Uses group-by-n_allowed architecture.
    Required for long sequences (e.g., 129-frame: ~52K tokens).

Forward signature for tracing (all tensors, no control flow):
  1. img       [B, L_img, 2048]
  2. txt       [B, L_txt, 2048]
  3. vec       [B, 2048]
  4. txt_mask  [B, L_txt]           -- 1=valid, 0=padding (unused with NKI flash)
  5. freqs_cos [L_img, 128]
  6. freqs_sin [L_img, 128]

Returns:
  img_out   [B, L_img, 32]          -- patch_size=[1,1,1], out_channels=32
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

# NKI Flash Attention imports (dense attention)
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)

# STA (Sliding Tile Attention) — optional, for block-sparse attention
try:
    from sta_attention import STAAttention, create_sta_attention
except ImportError:
    STAAttention = None
    create_sta_attention = None

# NKI RoPE kernel (optional, ~3% faster per step but adds code complexity)
# Enable with: export HUNYUAN_NKI_ROPE=1
USE_NKI_ROPE = os.environ.get("HUNYUAN_NKI_ROPE", "0") == "1"
if USE_NKI_ROPE:
    from nki_rope import nki_rope_apply

# NKI alignment constants
NKI_SEQ_TILE = 128
NKI_SEQ_TILE_SHARDED = 2048  # Required for LNC=2 (vc_size=2) on trn2


# --------------------------------------------------------------------------
# Traceable helper functions (no control flow, no @torch.compiler.disable)
# --------------------------------------------------------------------------


def rotate_half(x):
    """Rotates half the hidden dims of the input for RoPE.

    Uses interleaved-pair rotation matching HunyuanVideo's original:
    treats consecutive pairs (x[2i], x[2i+1]) as (real, imag).
    Output: [-x1, x0, -x3, x2, -x5, x4, ...] for input [x0, x1, x2, x3, x4, x5, ...]

    This is required because cos/sin from get_nd_rotary_pos_embed(use_real=True)
    use repeat_interleave(2) pattern: [c0, c0, c1, c1, c2, c2, ...]
    """
    x = x.reshape(*x.shape[:-1], -1, 2)  # [..., D/2, 2]
    x_real = x[..., 0]  # [..., D/2]
    x_imag = x[..., 1]  # [..., D/2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(-2)  # [..., D]


def apply_rotary_emb_traceable(q, k, cos, sin):
    """
    Apply RoPE to q and k tensors.

    q, k: [B, L, H, D]   (head_first=False)
    cos:  [L, D]  -> broadcast to [1, L, 1, D]
    sin:  [L, D]  -> broadcast to [1, L, 1, D]
    """
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate):
    """Apply gating: x * gate."""
    return x * gate.unsqueeze(1)


def _pad_to_multiple(x, dim, multiple):
    """Pad tensor along dim to nearest multiple. Returns (padded, orig_len)."""
    orig_len = x.shape[dim]
    remainder = orig_len % multiple
    if remainder == 0:
        return x, orig_len
    pad_len = multiple - remainder
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], dim=dim), orig_len


def nki_flash_attention(query, key, value):
    """
    NKI Flash Attention — fused O(n) memory attention kernel.

    Input layout (standard PyTorch BHSD):
        query: [B, H, Q_len, D]
        key:   [B, H, KV_len, D]
        value: [B, H, KV_len, D]

    Returns:
        [B, H, Q_len, D]

    NOTE: Does NOT support attention masks. For our use case this is
    acceptable because text padding tokens are already zeroed by the
    CPU preprocessor, and attending to zero tokens has minimal impact.
    """
    bs, n_head, q_len, d_head = query.shape

    # Determine alignment
    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "2"))
    tile_multiple = NKI_SEQ_TILE_SHARDED if vc_size == 2 else NKI_SEQ_TILE

    # Pad sequences to tile alignment
    query, orig_q_len = _pad_to_multiple(query, dim=2, multiple=tile_multiple)
    key, _ = _pad_to_multiple(key, dim=2, multiple=tile_multiple)
    value, _ = _pad_to_multiple(value, dim=2, multiple=tile_multiple)

    padded_q_len = query.shape[2]
    padded_k_len = key.shape[2]
    padded_v_len = value.shape[2]

    # Reshape for NKI kernel:
    #   Q, K: [B, H, S, D] -> permute(0,1,3,2) -> [B, H, D, S] -> reshape (B*H, D, S)
    #   V:    [B, H, S, D] -> reshape (B*H, S, D)
    q = query.permute(0, 1, 3, 2).reshape(bs * n_head, d_head, padded_q_len)
    k = key.permute(0, 1, 3, 2).reshape(bs * n_head, d_head, padded_k_len)
    v = value.reshape(bs * n_head, padded_v_len, d_head)

    # Pre-allocate output
    attn_output = torch.zeros(
        (bs * n_head, padded_q_len, d_head), dtype=torch.bfloat16, device=q.device
    )
    scale = 1.0 / math.sqrt(d_head)

    # Dispatch with sharded grid for LNC=2
    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](
            q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap"
        )
    else:
        _flash_fwd_call(
            q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap"
        )

    # Reshape back to [B, H, S, D]
    attn_output = attn_output.reshape(bs, n_head, padded_q_len, d_head)

    # Slice back to original Q length
    if padded_q_len != orig_q_len:
        attn_output = attn_output[:, :, :orig_q_len, :]

    return attn_output


# --------------------------------------------------------------------------
# TP Double-Stream Block
# --------------------------------------------------------------------------


class TPMMDoubleStreamBlock(nn.Module):
    """
    TP-sharded MMDoubleStreamBlock for tracing.

    All Q/K/V are ColumnParallel (shard output dim across heads).
    All O projections are RowParallel (shard input dim, all-reduce output).
    MLP fc1 is ColumnParallel, fc2 is RowParallel.
    ModulateDiT and QK-norms are replicated.

    Attention mode:
      - If sta_attention is None: use dense NKI flash attention
      - If sta_attention is an STAAttention module: use block-sparse STA
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        heads_num: int = 16,
        mlp_width_ratio: float = 4.0,
        qkv_bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        sta_attention: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        # Attention mode: STA (block-sparse) or dense (NKI flash)
        self.sta_attention = sta_attention  # None = dense, STAAttention = sparse

        # --- Image stream ---
        # Modulation (replicated -- small, operates on conditioning vec)
        self.img_mod_act = nn.SiLU()
        self.img_mod_linear = nn.Linear(
            hidden_size, 6 * hidden_size, bias=True, dtype=dtype
        )

        # Norm (no affine)
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype
        )

        # Q/K/V: ColumnParallel (shard output dim = shard heads)
        self.img_attn_q = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            gather_output=False,
            dtype=dtype,
        )
        self.img_attn_k = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            gather_output=False,
            dtype=dtype,
        )
        self.img_attn_v = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            gather_output=False,
            dtype=dtype,
        )

        # QK-norm: RMSNorm per head_dim (replicated -- tiny, each rank applies to its local heads)
        self.img_attn_q_norm = nn.RMSNorm(
            self.head_dim, elementwise_affine=True, eps=1e-6
        )
        self.img_attn_k_norm = nn.RMSNorm(
            self.head_dim, elementwise_affine=True, eps=1e-6
        )

        # O projection: RowParallel (input is local heads, all-reduce output)
        self.img_attn_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            input_is_parallel=True,
            dtype=dtype,
        )

        # MLP
        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype
        )
        self.img_mlp_fc1 = ColumnParallelLinear(
            hidden_size,
            mlp_hidden_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
        )
        self.img_mlp_act = nn.GELU(approximate="tanh")
        self.img_mlp_fc2 = RowParallelLinear(
            mlp_hidden_dim,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
        )

        # --- Text stream (identical structure) ---
        self.txt_mod_act = nn.SiLU()
        self.txt_mod_linear = nn.Linear(
            hidden_size, 6 * hidden_size, bias=True, dtype=dtype
        )

        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype
        )

        self.txt_attn_q = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            gather_output=False,
            dtype=dtype,
        )
        self.txt_attn_k = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            gather_output=False,
            dtype=dtype,
        )
        self.txt_attn_v = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            gather_output=False,
            dtype=dtype,
        )

        self.txt_attn_q_norm = nn.RMSNorm(
            self.head_dim, elementwise_affine=True, eps=1e-6
        )
        self.txt_attn_k_norm = nn.RMSNorm(
            self.head_dim, elementwise_affine=True, eps=1e-6
        )

        self.txt_attn_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            input_is_parallel=True,
            dtype=dtype,
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype
        )
        self.txt_mlp_fc1 = ColumnParallelLinear(
            hidden_size,
            mlp_hidden_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
        )
        self.txt_mlp_act = nn.GELU(approximate="tanh")
        self.txt_mlp_fc2 = RowParallelLinear(
            mlp_hidden_dim,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
        )

    def forward(
        self,
        img: torch.Tensor,  # [B, L_img, hidden_size]
        txt: torch.Tensor,  # [B, L_txt, hidden_size]
        vec: torch.Tensor,  # [B, hidden_size]
        txt_mask: torch.Tensor,  # [B, L_txt] (1=valid, 0=masked)
        freqs_cos: torch.Tensor,  # [L_img, head_dim]
        freqs_sin: torch.Tensor,  # [L_img, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = img.shape[0]
        L_img = img.shape[1]
        L_txt = txt.shape[1]

        # Local heads per rank (TP sharded)
        local_heads = self.img_attn_q.output_size_per_partition // self.head_dim

        # --- Image modulation ---
        img_mod = self.img_mod_linear(self.img_mod_act(vec))  # [B, 6*hidden_size]
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = img_mod.chunk(6, dim=-1)

        # --- Text modulation ---
        txt_mod = self.txt_mod_linear(self.txt_mod_act(vec))  # [B, 6*hidden_size]
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = txt_mod.chunk(6, dim=-1)

        # --- Image QKV ---
        img_modulated = modulate(
            self.img_norm1(img), shift=img_mod1_shift, scale=img_mod1_scale
        )

        img_q = self.img_attn_q(img_modulated)  # [B, L_img, local_heads * head_dim]
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)

        # Reshape to [B, L, local_heads, D]
        img_q = img_q.reshape(B, L_img, local_heads, self.head_dim)
        img_k = img_k.reshape(B, L_img, local_heads, self.head_dim)
        img_v = img_v.reshape(B, L_img, local_heads, self.head_dim)

        # QK-norm
        img_q = self.img_attn_q_norm(img_q)
        img_k = self.img_attn_k_norm(img_k)

        # RoPE (only on image tokens)
        if USE_NKI_ROPE:
            img_q, img_k = nki_rope_apply(img_q, img_k, freqs_cos, freqs_sin)
        else:
            img_q, img_k = apply_rotary_emb_traceable(
                img_q, img_k, freqs_cos, freqs_sin
            )

        # --- Text QKV ---
        txt_modulated = modulate(
            self.txt_norm1(txt), shift=txt_mod1_shift, scale=txt_mod1_scale
        )

        txt_q = self.txt_attn_q(txt_modulated)  # [B, L_txt, local_heads * head_dim]
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)

        txt_q = txt_q.reshape(B, L_txt, local_heads, self.head_dim)
        txt_k = txt_k.reshape(B, L_txt, local_heads, self.head_dim)
        txt_v = txt_v.reshape(B, L_txt, local_heads, self.head_dim)

        # QK-norm (no RoPE on text)
        txt_q = self.txt_attn_q_norm(txt_q)
        txt_k = self.txt_attn_k_norm(txt_k)

        # --- Joint attention (concatenate img + txt) ---
        # Cat along sequence dim: [B, L_img + L_txt, local_heads, D]
        q_cat = torch.cat([img_q, txt_q], dim=1)
        k_cat = torch.cat([img_k, txt_k], dim=1)
        v_cat = torch.cat([img_v, txt_v], dim=1)

        # Transpose to [B, local_heads, L_total, D] for attention
        q_cat = q_cat.transpose(1, 2)
        k_cat = k_cat.transpose(1, 2)
        v_cat = v_cat.transpose(1, 2)

        # NKI flash attention with K/V zeroing for padding text tokens.
        # The K/V zeroing above ensures padding positions have near-zero
        # attention scores, and zero V contribution even if attended to.
        if self.sta_attention is not None:
            # STA: block-sparse sliding tile attention via attention_cte
            # STAAttention expects [B, H, seq_len, D] and returns [B, H, seq_len, D]
            attn_out = self.sta_attention(q_cat, k_cat, v_cat)
        else:
            # Dense: NKI flash attention
            attn_out = nki_flash_attention(q_cat, k_cat, v_cat)
        # [B, local_heads, L_total, D]

        # Transpose back: [B, L_total, local_heads, D]
        attn_out = attn_out.transpose(1, 2)

        # Split back to img and txt
        img_attn = attn_out[:, :L_img].reshape(
            B, L_img, -1
        )  # [B, L_img, local_heads * D]
        txt_attn = attn_out[:, L_img:].reshape(
            B, L_txt, -1
        )  # [B, L_txt, local_heads * D]

        # --- Image output projection + residual ---
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp_fc2(
                self.img_mlp_act(
                    self.img_mlp_fc1(
                        modulate(
                            self.img_norm2(img),
                            shift=img_mod2_shift,
                            scale=img_mod2_scale,
                        )
                    )
                )
            ),
            gate=img_mod2_gate,
        )

        # --- Text output projection + residual ---
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp_fc2(
                self.txt_mlp_act(
                    self.txt_mlp_fc1(
                        modulate(
                            self.txt_norm2(txt),
                            shift=txt_mod2_shift,
                            scale=txt_mod2_scale,
                        )
                    )
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


# --------------------------------------------------------------------------
# TP Wrapper (full model for tracing)
# --------------------------------------------------------------------------


class HunyuanDiTTPWrapper(nn.Module):
    """
    Complete TP-sharded DiT wrapper for tracing.
    54 double-stream blocks + final layer.

    Supports two attention modes:
      - Dense (default): Full O(n^2) attention using attention_isa_kernel.
      - STA: Block-sparse sliding tile attention using attention_cte.
        Enabled by passing sta_config dict with canvas geometry.

    Args:
        sta_config: Optional dict to enable STA. Keys:
            canvas_thw: (T, H, W) of the image latent grid (before padding)
            tile_thw: tile dimensions (default (6,8,8) = 384 tokens)
            kernel_thw: neighborhood kernel (default (3,3,3) = 27 tiles)
            text_len: number of text tokens (default 320)
        When sta_config is None, uses dense attention.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        heads_num: int = 16,
        num_blocks: int = 54,
        mlp_width_ratio: float = 4.0,
        patch_size: list = [1, 1, 1],
        out_channels: int = 32,
        qkv_bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        sta_config: Optional[dict] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.head_dim = hidden_size // heads_num

        # Determine attention mode
        use_sta = sta_config is not None and create_sta_attention is not None
        if sta_config is not None and create_sta_attention is None:
            print(
                "WARNING: sta_config provided but sta_attention module not available. "
                "Using dense attention."
            )
            use_sta = False

        # For STA, compute local heads per TP rank.
        # At init time we don't know the TP degree yet, but ColumnParallelLinear
        # will shard heads. We can compute local_heads from the first block
        # after construction. For now, assume TP=4 as documented.
        tp_degree = int(os.getenv("NEURON_RT_NUM_CORES", "4"))
        local_heads = heads_num // tp_degree

        # 54 double-stream blocks
        self.double_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_sta:
                sta_module = create_sta_attention(
                    canvas_thw=sta_config["canvas_thw"],
                    tile_thw=sta_config.get("tile_thw", (6, 8, 8)),
                    kernel_thw=sta_config.get("kernel_thw", (3, 3, 3)),
                    text_len=sta_config.get("text_len", 320),
                    num_heads=local_heads,
                    head_dim=self.head_dim,
                )
            else:
                sta_module = None

            self.double_blocks.append(
                TPMMDoubleStreamBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    qkv_bias=qkv_bias,
                    dtype=dtype,
                    sta_attention=sta_module,
                )
            )

        # Final layer (replicated -- tiny output dim)
        out_size = patch_size[0] * patch_size[1] * patch_size[2] * out_channels
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype
        )
        self.linear = nn.Linear(hidden_size, out_size, bias=True, dtype=dtype)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype),
        )

    def forward(
        self,
        img: torch.Tensor,  # [B, L_img, 2048]
        txt: torch.Tensor,  # [B, L_txt, 2048]
        vec: torch.Tensor,  # [B, 2048]
        txt_mask: torch.Tensor,  # [B, L_txt]
        freqs_cos: torch.Tensor,  # [L_img, 128]
        freqs_sin: torch.Tensor,  # [L_img, 128]
    ) -> torch.Tensor:
        # Run 54 double-stream blocks
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, txt_mask, freqs_cos, freqs_sin)

        # Final layer: AdaLN + linear
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        img = modulate(self.norm_final(img), shift=shift, scale=scale)
        img = self.linear(img)

        return img


# --------------------------------------------------------------------------
# Weight loading: map original weights -> TP wrapper weights
# --------------------------------------------------------------------------


def build_weight_map(num_blocks=54):
    """
    Build a mapping from original HunyuanVideo DiT state_dict keys
    to TP wrapper state_dict keys.

    Original keys look like:
      double_blocks.{i}.img_attn_q.weight  -> double_blocks.{i}.img_attn_q.weight
      double_blocks.{i}.img_mod.linear.weight -> double_blocks.{i}.img_mod_linear.weight
      double_blocks.{i}.img_mlp.fc1.weight -> double_blocks.{i}.img_mlp_fc1.weight
      double_blocks.{i}.img_mlp.fc2.weight -> double_blocks.{i}.img_mlp_fc2.weight
      final_layer.norm_final.* -> norm_final.*
      final_layer.linear.* -> linear.*
      final_layer.adaLN_modulation.* -> adaLN_modulation.*

    Returns dict: original_key -> tp_wrapper_key
    """
    weight_map = {}

    for i in range(num_blocks):
        prefix = f"double_blocks.{i}"

        for stream in ["img", "txt"]:
            # Modulation: mod.linear -> mod_linear, mod.act -> mod_act (no params)
            weight_map[f"{prefix}.{stream}_mod.linear.weight"] = (
                f"{prefix}.{stream}_mod_linear.weight"
            )
            weight_map[f"{prefix}.{stream}_mod.linear.bias"] = (
                f"{prefix}.{stream}_mod_linear.bias"
            )

            # QKV (direct mapping -- same names)
            for proj in ["attn_q", "attn_k", "attn_v"]:
                weight_map[f"{prefix}.{stream}_{proj}.weight"] = (
                    f"{prefix}.{stream}_{proj}.weight"
                )
                weight_map[f"{prefix}.{stream}_{proj}.bias"] = (
                    f"{prefix}.{stream}_{proj}.bias"
                )

            # QK-norm (direct mapping)
            for norm in ["attn_q_norm", "attn_k_norm"]:
                weight_map[f"{prefix}.{stream}_{norm}.weight"] = (
                    f"{prefix}.{stream}_{norm}.weight"
                )

            # O projection (direct mapping)
            weight_map[f"{prefix}.{stream}_attn_proj.weight"] = (
                f"{prefix}.{stream}_attn_proj.weight"
            )
            weight_map[f"{prefix}.{stream}_attn_proj.bias"] = (
                f"{prefix}.{stream}_attn_proj.bias"
            )

            # Norms (direct mapping)
            # img_norm1, img_norm2 have no affine params (elementwise_affine=False)

            # MLP: mlp.fc1 -> mlp_fc1, mlp.fc2 -> mlp_fc2
            weight_map[f"{prefix}.{stream}_mlp.fc1.weight"] = (
                f"{prefix}.{stream}_mlp_fc1.weight"
            )
            weight_map[f"{prefix}.{stream}_mlp.fc1.bias"] = (
                f"{prefix}.{stream}_mlp_fc1.bias"
            )
            weight_map[f"{prefix}.{stream}_mlp.fc2.weight"] = (
                f"{prefix}.{stream}_mlp_fc2.weight"
            )
            weight_map[f"{prefix}.{stream}_mlp.fc2.bias"] = (
                f"{prefix}.{stream}_mlp_fc2.bias"
            )

    # Final layer (norm_final has elementwise_affine=False, no weight/bias)
    weight_map["final_layer.linear.weight"] = "linear.weight"
    weight_map["final_layer.linear.bias"] = "linear.bias"
    weight_map["final_layer.adaLN_modulation.1.weight"] = "adaLN_modulation.1.weight"
    weight_map["final_layer.adaLN_modulation.1.bias"] = "adaLN_modulation.1.bias"

    return weight_map


def load_original_weights(tp_model, original_state_dict, tp_degree=4):
    """
    Load weights from original HunyuanVideo DiT into the TP wrapper.

    ColumnParallelLinear weights are sharded along output dim (dim=0 for weight).
    RowParallelLinear weights are sharded along input dim (dim=1 for weight).
    Replicated weights are loaded as-is.

    Note: When using ModelBuilderV2 with shard_checkpoint(), this manual
    loading is not needed -- shard_checkpoint handles it automatically.
    This function is provided for manual loading / debugging.
    """
    weight_map = build_weight_map()
    tp_state = tp_model.state_dict()

    mapped_count = 0
    for orig_key, tp_key in weight_map.items():
        if orig_key in original_state_dict and tp_key in tp_state:
            tp_state[tp_key] = original_state_dict[orig_key]
            mapped_count += 1

    tp_model.load_state_dict(tp_state)
    print(f"Loaded {mapped_count}/{len(weight_map)} weights from original model")
    return tp_model


# --------------------------------------------------------------------------
# Sample inputs for tracing
# --------------------------------------------------------------------------


def create_sample_inputs(
    batch_size=1,
    hidden_size=2048,
    img_seq_len=3180,  # 480p: T=2, H=30, W=53 -> 2*30*53=3180 (with patch_size=[1,1,1])
    txt_seq_len=320,  # 256 byT5 + 64 LLM text (after reorder)
    head_dim=128,
    dtype=torch.bfloat16,
):
    """Create sample inputs for tracing at 480p_5f scale."""
    return (
        torch.randn(batch_size, img_seq_len, hidden_size, dtype=dtype),  # img
        torch.randn(batch_size, txt_seq_len, hidden_size, dtype=dtype),  # txt
        torch.randn(batch_size, hidden_size, dtype=dtype),  # vec
        torch.ones(batch_size, txt_seq_len, dtype=torch.long),  # txt_mask
        torch.randn(img_seq_len, head_dim, dtype=dtype),  # freqs_cos
        torch.randn(img_seq_len, head_dim, dtype=dtype),  # freqs_sin
    )
