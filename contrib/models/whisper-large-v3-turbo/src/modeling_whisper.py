# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/openai/whisper/blob/main/whisper/model.py
#
# Whisper (openai/whisper-large-v3-turbo) for NxD Inference.
#
# Optimizations included:
#   1. Cross-attention K/V cache: skip redundant K/V projections during decode (~2.5x decode speedup)
#   2. Fused QKV projections: 3 matmuls → 1 for self-attention
#   3. NKI flash attention (encoder): bidirectional flash attention for all 32 encoder layers
#   4. NKI fused Conv1D+GELU (encoder): fused conv1d kernel for encoder frontend
#   5. LNC flag: compiler args pass --lnc= for LNC=1 support on trn2
#   6. Batch size >1: batched decode with per-sample positional embedding and logit extraction

import math
import os
from typing import Optional, Iterable, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import (
    BaseModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase

from neuronx_distributed_inference.experimental.functional.attention.causal_attention_functions import (
    scaled_dot_product_attention_kernel,
)

from utils.config import get_dims_from_config
from utils.decoding import decode as decode_function
from utils.state_dict import convert_hf_state_dict_to_neuron, expand_state_dict

# NKI fused Conv1D+GELU kernel (optional — falls back to PyTorch if nkilib not available)
try:
    from nkilib.experimental.conv.conv1d import conv1d as nki_conv1d
    from nkilib.core.utils.common_types import ActFnType

    _HAS_NKI_CONV1D = True
except ImportError:
    _HAS_NKI_CONV1D = False

# NKI encoder megakernel (optional, experimental — gated behind WHISPER_USE_MEGAKERNEL=1).
# This fuses an entire Whisper encoder layer into a single @nki.jit kernel.
# STATUS: Negative result in benchmarks (16.9x slower than compiler-generated code due to
# weight transfer overhead and hand-written matmuls being ~13x slower than compiler-optimized).
# Preserved here as a reference implementation and starting point for future NKI kernel work.
try:
    from whisper_encoder_megakernel import (
        whisper_encoder_layer_fwd,
        P_MAX,
        D_MODEL as MK_D_MODEL,
        SEQ_PAD as MK_SEQ_PAD,
        N_HEADS as MK_N_HEADS,
        HEAD_DIM as MK_HEAD_DIM,
        MLP_DIM as MK_MLP_DIM,
    )

    _HAS_MEGAKERNEL = True
except ImportError:
    _HAS_MEGAKERNEL = False
    # Dummy constants so module-level _MK_WEIGHT_SHAPES can be defined
    # (only used when megakernel is active, which requires _HAS_MEGAKERNEL=True)
    P_MAX = 128
    MK_D_MODEL = 1280
    MK_SEQ_PAD = 1536
    MK_N_HEADS = 20
    MK_HEAD_DIM = 64
    MK_MLP_DIM = 5120

from transformers import WhisperModel
from transformers.models.whisper.modeling_whisper import sinusoids
from whisper import Whisper


def ceil_div(a: int, b: int) -> int:
    """Integer division with ceiling."""
    return -(-a // b)


def _tile_ln_weight(w: Tensor, p_max: int = 128) -> Tensor:
    """Tile a LayerNorm weight [F] to [P_MAX, F] by repeating each row.

    The NKI megakernel expects LN weight/bias pre-tiled so that element-wise
    multiply/add works on [P_MAX, F] tiles without broadcasting.
    """
    return w.unsqueeze(0).expand(p_max, -1).contiguous()


def _prepare_megakernel_weights(block):
    """Extract and reshape weights from a NeuronResidualAttentionBlock for the megakernel.

    Returns a dict of tensors ready to pass to whisper_encoder_layer_fwd.
    All weights are bf16 and contiguous.

    The megakernel expects:
    - LayerNorm weights tiled to [P_MAX, D_MODEL]
    - QKV weight as [3*D_MODEL, D_MODEL] (already fused by NxDI state_dict conversion)
    - Biases pre-tiled to [P_MAX, dim] (NKI tensor_tensor requires same shapes)
    - FC1/FC2 weights in their original [out, in] layout
    """
    dtype = torch.bfloat16

    # Pre-attention LayerNorm (tiled)
    attn_ln_w = _tile_ln_weight(block.attn_ln.weight.to(dtype))
    attn_ln_b = _tile_ln_weight(block.attn_ln.bias.to(dtype))

    # Fused QKV: ColumnParallelLinear stores weight as [out_features, in_features]
    # After TP sharding, this is [3*n_heads_per_tp*head_dim, n_state].
    # For TP=1: [3*1280, 1280] = [3840, 1280]
    qkv_w = block.attn.qkv_proj.weight.to(dtype).contiguous()
    qkv_b = _tile_ln_weight(block.attn.qkv_proj.bias.to(dtype))  # [P_MAX, 3840]

    # Output projection: RowParallelLinear, weight [n_state, n_heads_per_tp*head_dim]
    # For TP=1: [1280, 1280]
    out_w = block.attn.out.weight.to(dtype).contiguous()
    out_b = _tile_ln_weight(block.attn.out.bias.to(dtype))  # [P_MAX, 1280]

    # Pre-MLP LayerNorm (tiled)
    mlp_ln_w = _tile_ln_weight(block.mlp_ln.weight.to(dtype))
    mlp_ln_b = _tile_ln_weight(block.mlp_ln.bias.to(dtype))

    # MLP FC1 (up_proj): ColumnParallelLinear [MLP_DIM/TP, D_MODEL]
    # For TP=1: [5120, 1280]
    fc1_w = block.mlp.up_proj.weight.to(dtype).contiguous()
    fc1_b = _tile_ln_weight(block.mlp.up_proj.bias.to(dtype))  # [P_MAX, 5120]

    # MLP FC2 (down_proj): RowParallelLinear [D_MODEL, MLP_DIM/TP]
    # For TP=1: [1280, 5120]
    fc2_w = block.mlp.down_proj.weight.to(dtype).contiguous()
    fc2_b = _tile_ln_weight(block.mlp.down_proj.bias.to(dtype))  # [P_MAX, 1280]

    return {
        "attn_ln_w": attn_ln_w,
        "attn_ln_b": attn_ln_b,
        "qkv_w": qkv_w,
        "qkv_b": qkv_b,
        "out_w": out_w,
        "out_b": out_b,
        "mlp_ln_w": mlp_ln_w,
        "mlp_ln_b": mlp_ln_b,
        "fc1_w": fc1_w,
        "fc1_b": fc1_b,
        "fc2_w": fc2_w,
        "fc2_b": fc2_b,
    }


# Canonical order of weight keys for packing/unpacking
_MK_WEIGHT_KEYS = [
    "attn_ln_w",
    "attn_ln_b",
    "qkv_w",
    "qkv_b",
    "out_w",
    "out_b",
    "mlp_ln_w",
    "mlp_ln_b",
    "fc1_w",
    "fc1_b",
    "fc2_w",
    "fc2_b",
]

# Pre-computed shapes for each weight (TP=1, Whisper large-v3-turbo)
_MK_WEIGHT_SHAPES = {
    "attn_ln_w": (P_MAX, MK_D_MODEL),  # [128, 1280]
    "attn_ln_b": (P_MAX, MK_D_MODEL),  # [128, 1280]
    "qkv_w": (3 * MK_D_MODEL, MK_D_MODEL),  # [3840, 1280]
    "qkv_b": (P_MAX, 3 * MK_D_MODEL),  # [128, 3840]
    "out_w": (MK_D_MODEL, MK_D_MODEL),  # [1280, 1280]
    "out_b": (P_MAX, MK_D_MODEL),  # [128, 1280]
    "mlp_ln_w": (P_MAX, MK_D_MODEL),  # [128, 1280]
    "mlp_ln_b": (P_MAX, MK_D_MODEL),  # [128, 1280]
    "fc1_w": (MK_MLP_DIM, MK_D_MODEL),  # [5120, 1280]
    "fc1_b": (P_MAX, MK_MLP_DIM),  # [128, 5120]
    "fc2_w": (MK_D_MODEL, MK_MLP_DIM),  # [1280, 5120]
    "fc2_b": (P_MAX, MK_D_MODEL),  # [128, 1280]
}

# Total elements per layer (sum of all weight numel)
_MK_ELEMENTS_PER_LAYER = sum(s[0] * s[1] for s in _MK_WEIGHT_SHAPES.values())


def _pack_all_layer_weights(blocks) -> Tensor:
    """Pack all megakernel weights for all layers into a single 1D bf16 tensor.

    Args:
        blocks: nn.ModuleList of NeuronResidualAttentionBlock (encoder layers)

    Returns:
        packed: 1D bf16 tensor of shape [n_layers * elements_per_layer]
    """
    n_layers = len(blocks)
    packed = torch.empty(n_layers * _MK_ELEMENTS_PER_LAYER, dtype=torch.bfloat16)
    offset = 0
    for block in blocks:
        weights = _prepare_megakernel_weights(block)
        for key in _MK_WEIGHT_KEYS:
            w = weights[key].contiguous().view(-1)
            packed[offset : offset + w.numel()] = w
            offset += w.numel()
    assert offset == packed.numel(), f"Packing mismatch: {offset} != {packed.numel()}"
    return packed


def _unpack_layer_weights(packed: Tensor, layer_idx: int):
    """Unpack the 12 weight tensors for a single layer from the packed tensor.

    Args:
        packed: 1D bf16 tensor from _pack_all_layer_weights()
        layer_idx: which layer (0-indexed)

    Returns:
        tuple of 12 tensors in canonical order (attn_ln_w, attn_ln_b, qkv_w, ...)
    """
    base = layer_idx * _MK_ELEMENTS_PER_LAYER
    offset = base
    tensors = []
    for key in _MK_WEIGHT_KEYS:
        shape = _MK_WEIGHT_SHAPES[key]
        numel = shape[0] * shape[1]
        t = packed[offset : offset + numel].view(shape)
        tensors.append(t)
        offset += numel
    return tuple(tensors)


class WhisperInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = get_dims_from_config(self)


class LayerNorm(nn.LayerNorm):
    """
    Converts input to float32 before applying LayerNorm to avoid precision issues.
    """

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class NeuronMLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert parallel_state.model_parallel_is_initialized(), (
            "Model parallel not initialized"
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=True, gather_output=False, dtype=dtype
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
        )

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class NeuronAttention(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype = torch.float32,
        kvcache=True,
    ):
        super().__init__()

        assert n_state % n_head == 0, (
            f"n_state ({n_state}) must be divisible by n_head ({n_head})"
        )
        self.head_dim = n_state // n_head

        assert parallel_state.model_parallel_is_initialized(), (
            "Model parallel not initialized"
        )
        tp_degree = parallel_state.get_tensor_model_parallel_group().size()

        # head per core
        self.n_heads = ceil_div(n_head, tp_degree)
        self.n_kv_heads = self.n_heads  # Whisper doesn't use GQA

        # Fused QKV projection: single matmul instead of 3 separate ones.
        # Bias is included for all 3 (K portion is zeroed in state dict conversion).
        self.qkv_proj = ColumnParallelLinear(
            n_state,
            3 * self.n_heads * tp_degree * self.head_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
        )
        self.out = RowParallelLinear(
            self.n_heads * tp_degree * self.head_dim,
            n_state,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
        )

        self.cache_k = (
            nn.Parameter(
                torch.zeros(
                    (batch_size, self.n_kv_heads, seq_len, self.head_dim), dtype=dtype
                ),
                requires_grad=False,
            )
            if kvcache
            else None
        )
        self.cache_v = (
            nn.Parameter(
                torch.zeros(
                    (batch_size, self.n_kv_heads, seq_len, self.head_dim), dtype=dtype
                ),
                requires_grad=False,
            )
            if kvcache
            else None
        )

    def forward(
        self,
        x: Tensor,
        last_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        bsz, seq_len, hidden_dim = x.shape

        # Fused QKV: single matmul, then split into Q, K, V (contiguous layout)
        qkv = self.qkv_proj(x)
        n_state_per_tp = self.n_heads * self.head_dim
        q, k, v = torch.tensor_split(qkv, (n_state_per_tp, 2 * n_state_per_tp), dim=2)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.cache_k is not None and self.cache_v is not None:
            if seq_len > 1:  # prefill: save all to cache
                indices = torch.arange(
                    start=0, end=seq_len, dtype=torch.int64, device=q.device
                )
                indices = indices.view(1, 1, seq_len, 1)
                indices = indices.expand(bsz, self.n_kv_heads, seq_len, self.head_dim)
            else:  # decode: save only the last token [last_pos] to cache
                indices = last_pos.view(bsz, 1, 1, 1).expand_as(k).to(torch.int64)

            updated_kcache = torch.scatter(self.cache_k, 2, indices, k)
            updated_vcache = torch.scatter(self.cache_v, 2, indices, v)

            k = updated_kcache
            v = updated_vcache

        if self.cache_k is None:
            # Encoder path: use NKI flash attention kernel (avoids materializing
            # the full 1500x1500 score matrix across all 32 encoder layers).
            # Q, K, V are already in (B, H, S, d) layout from lines above.
            output = scaled_dot_product_attention_kernel(
                q, k, v, is_causal=False, scale=1.0 / math.sqrt(self.head_dim)
            )
            # Output is (B, H, S, d) -- transpose to (B, S, H*d)
            output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        else:
            # Decoder path: standard matmul attention (KV cache changes seq dims)
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = torch.where(mask, scores, torch.finfo(scores.dtype).min)
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = torch.matmul(scores, v)
            output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        if self.cache_k is not None and self.cache_v is not None:
            return self.out(output), updated_kcache, updated_vcache
        else:
            return self.out(output)


class NeuronCrossAttention(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        batch_size: int,
        kv_seq_len: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        assert n_state % n_head == 0, (
            f"n_state ({n_state}) must be divisible by n_head ({n_head})"
        )
        self.head_dim = n_state // n_head

        assert parallel_state.model_parallel_is_initialized(), (
            "Model parallel not initialized"
        )
        tp_degree = parallel_state.get_tensor_model_parallel_group().size()

        # head per core
        self.n_heads = ceil_div(n_head, tp_degree)
        self.n_kv_heads = self.n_heads  # Whisper doesn't use GQA

        self.query = ColumnParallelLinear(
            n_state,
            self.n_heads * tp_degree * self.head_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
        )
        self.key = ColumnParallelLinear(
            n_state,
            self.n_kv_heads * tp_degree * self.head_dim,
            bias=False,  # No bias for key projection
            gather_output=False,
            dtype=dtype,
        )
        self.value = ColumnParallelLinear(
            n_state,
            self.n_kv_heads * tp_degree * self.head_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
        )
        self.out = RowParallelLinear(
            self.n_heads * tp_degree * self.head_dim,
            n_state,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
        )

        self.cache_k = nn.Parameter(
            torch.zeros(
                (batch_size, self.n_kv_heads, kv_seq_len, self.head_dim), dtype=dtype
            ),
            requires_grad=False,
        )
        self.cache_v = nn.Parameter(
            torch.zeros(
                (batch_size, self.n_kv_heads, kv_seq_len, self.head_dim), dtype=dtype
            ),
            requires_grad=False,
        )

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        is_prefill: bool = True,
    ):
        bsz, seq_len, hidden_dim = x.shape

        # Q projection (always needed for both prefill and decode)
        q = (
            self.query(x)
            .view(bsz, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        if is_prefill:
            # Prefill: compute K/V from encoder output and populate cache
            kv_seq_len = xa.shape[1]
            k = (
                self.key(xa)
                .view(bsz, kv_seq_len, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.value(xa)
                .view(bsz, kv_seq_len, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
            )

            indices = torch.arange(
                start=0, end=kv_seq_len, dtype=torch.int64, device=q.device
            )
            indices = indices.view(1, 1, kv_seq_len, 1)
            indices = indices.expand(bsz, self.n_kv_heads, kv_seq_len, self.head_dim)

            updated_kcache = torch.scatter(self.cache_k, 2, indices, k)
            updated_vcache = torch.scatter(self.cache_v, 2, indices, v)
        else:
            # Decode: use cached K/V directly (no K/V projection needed, xa is unused)
            updated_kcache = self.cache_k
            updated_vcache = self.cache_v

        k = updated_kcache
        v = updated_vcache

        # Q.K^T/√d
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out(output), updated_kcache, updated_vcache


class NeuronResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        batch_size: int,
        seq_len: int,
        cross_attention: bool = False,
        cross_attn_seq_len: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.attn = NeuronAttention(
            n_state, n_head, batch_size, seq_len, dtype=dtype, kvcache=cross_attention
        )
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            NeuronCrossAttention(
                n_state, n_head, batch_size, cross_attn_seq_len, dtype=dtype
            )
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = NeuronMLP(n_state, n_mlp, dtype=dtype)
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,  # "a" for audio
        last_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        if self.cross_attn:
            h, self_attn_cache_k, self_attn_cache_v = self.attn(
                self.attn_ln(x), last_pos=last_pos, mask=mask
            )
        else:
            h = self.attn(self.attn_ln(x), last_pos=last_pos, mask=mask)
        x = x + h
        if self.cross_attn:
            h, cross_attn_cache_k, cross_attn_cache_v = self.cross_attn(
                self.cross_attn_ln(x), xa, is_prefill=x.shape[1] > 1
            )
            x = x + h
        x = x + self.mlp(self.mlp_ln(x))

        if self.cross_attn:
            return (
                x,
                self_attn_cache_k,
                self_attn_cache_v,
                cross_attn_cache_k,
                cross_attn_cache_v,
            )
        else:
            return x


class NeuronAudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        seq_len = n_ctx
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1, dtype=dtype)
        self.conv2 = nn.Conv1d(
            n_state, n_state, kernel_size=3, stride=2, padding=1, dtype=dtype
        )
        self.positional_embedding = nn.Parameter(
            sinusoids(n_ctx, n_state), requires_grad=False
        )

        self.blocks: Iterable[NeuronResidualAttentionBlock] = nn.ModuleList(
            [
                NeuronResidualAttentionBlock(
                    n_state, n_head, batch_size, seq_len, dtype=dtype
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, packed_weights: Optional[Tensor] = None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        packed_weights : torch.Tensor, optional, shape = (n_layers * elements_per_layer,)
            Packed megakernel weights (1D bf16 tensor). Required when megakernel is active.
            Created by _pack_all_layer_weights(). Passed as a forward() argument (not
            nn.Parameter) because NKI kernel tracing resolves forward args but not
            nn.Parameter attributes.
        """
        if _HAS_NKI_CONV1D:
            # NKI fused Conv1D+GELU: single kernel call per layer instead of
            # separate Conv1D + GELU ops. Weights transposed from PyTorch
            # (C_out, C_in, K) to NKI (K, C_in, C_out) layout.
            x = nki_conv1d(
                x,
                self.conv1.weight.permute(2, 1, 0),
                self.conv1.bias,
                stride=1,
                padding=(1, 1),
                activation_fn=ActFnType.GELU,
            )
            x = nki_conv1d(
                x,
                self.conv2.weight.permute(2, 1, 0),
                self.conv2.bias,
                stride=2,
                padding=(1, 1),
                activation_fn=ActFnType.GELU,
            )
        else:
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        use_megakernel = (
            _HAS_MEGAKERNEL
            and os.environ.get("WHISPER_USE_MEGAKERNEL", "0") == "1"
            and packed_weights is not None
        )

        if use_megakernel:
            # Megakernel path: fuse entire encoder layer into single NKI kernel.
            # Requires TP=1 and matching dimensions.
            # NOTE: This is experimental and currently 16.9x slower than the default
            # path due to weight transfer overhead and hand-written matmul inefficiency.
            n_layers = len(self.blocks)
            bsz, seq_len, hidden = x.shape
            # Pad sequence from 1500 to 1536 (12 * 128) for tile alignment
            if seq_len < MK_SEQ_PAD:
                x = F.pad(x, (0, 0, 0, MK_SEQ_PAD - seq_len))  # pad seq dim

            # Process each batch item (megakernel is single-batch)
            outputs = []
            for b in range(bsz):
                x_b = x[b]  # [S, D_MODEL]
                for layer_idx in range(n_layers):
                    (
                        attn_ln_w,
                        attn_ln_b,
                        qkv_w,
                        qkv_b,
                        out_w,
                        out_b,
                        mlp_ln_w,
                        mlp_ln_b,
                        fc1_w,
                        fc1_b,
                        fc2_w,
                        fc2_b,
                    ) = _unpack_layer_weights(packed_weights, layer_idx)
                    x_b = whisper_encoder_layer_fwd[1](
                        x_b,
                        attn_ln_w,
                        attn_ln_b,
                        qkv_w,
                        qkv_b,
                        out_w,
                        out_b,
                        mlp_ln_w,
                        mlp_ln_b,
                        fc1_w,
                        fc1_b,
                        fc2_w,
                        fc2_b,
                    )
                outputs.append(x_b)
            x = torch.stack(outputs, dim=0)  # [bsz, S, D_MODEL]

            # Strip padding back to original seq_len
            if seq_len < MK_SEQ_PAD:
                x = x[:, :seq_len, :]
        else:
            for block in self.blocks:
                x = block(x)

        x = self.ln_post(x)
        return x


class NeuronTextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_text_ctx: int,
        n_audio_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = n_text_ctx
        self.vocab_size = n_vocab

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Embedding(n_text_ctx, n_state)

        self.blocks: Iterable[NeuronResidualAttentionBlock] = nn.ModuleList(
            [
                NeuronResidualAttentionBlock(
                    n_state,
                    n_head,
                    self.batch_size,
                    self.seq_len,
                    cross_attention=True,
                    cross_attn_seq_len=n_audio_ctx,
                    dtype=dtype,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

    def forward(
        self, x: Tensor, xa: Tensor, last_pos: torch.Tensor, pad_mask: torch.Tensor
    ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        last_pos : torch.Tensor, shape = (batch_size,)
            indices of the last valid token position for each sequence in the batch
        pad_mask : torch.Tensor, shape = (batch_size, n_ctx)
            boolean mask indicating valid positions (True) vs padded positions (False)
        """
        assert x.shape[1] == 1 or x.shape[1] == self.seq_len, (
            f"Input sequence length {x.shape[1]} must be 1 (decode) or {self.seq_len} (prefill)"
        )

        is_prefill = x.shape[1] > 1
        if is_prefill:
            pe = self.positional_embedding.weight
        else:
            # last_pos shape: (batch_size,) — index PE per sample, unsqueeze
            # to (batch_size, 1, n_state) for broadcast with token embedding
            pe = self.positional_embedding(last_pos).unsqueeze(1)
        x = self.token_embedding(x) + pe
        x = x.to(xa.dtype)

        mask = None
        if is_prefill:
            mask = torch.full(
                (self.seq_len, self.seq_len), True, device=pad_mask.device
            ).tril(diagonal=0)
            input_mask = (
                pad_mask[:, None, None, :]
                .expand(self.batch_size, 1, self.seq_len, self.seq_len)
                .to(torch.bool)
            )
            mask = torch.logical_and(mask, input_mask)
        else:
            mask = (
                pad_mask[:, None, None, :]
                .expand(self.batch_size, 1, 1, self.seq_len)
                .to(torch.bool)
            )

        self_attn_k_caches = []
        self_attn_v_caches = []
        cross_attn_k_caches = []
        cross_attn_v_caches = []

        for block in self.blocks:
            x, sk, sv, ck, cv = block(x, xa, last_pos=last_pos, mask=mask)
            self_attn_k_caches.append(sk)
            self_attn_v_caches.append(sv)
            cross_attn_k_caches.append(ck)
            cross_attn_v_caches.append(cv)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return (
            logits,
            *self_attn_k_caches,
            *self_attn_v_caches,
            *cross_attn_k_caches,
            *cross_attn_v_caches,
        )


class WhisperModelEncoderInstance(BaseModelInstance):
    def __init__(self, config):
        self.module = None
        self.config = config
        self.neuron_config = config.neuron_config

    def load_module(self):
        dims = self.config.dims
        self.module = NeuronAudioEncoder(
            dims.n_mels,
            dims.n_audio_ctx,
            dims.n_audio_state,
            dims.n_audio_head,
            dims.n_audio_layer,
            batch_size=self.neuron_config.batch_size,
            dtype=self.neuron_config.torch_dtype,
        )

    def get(self, bucket_rank, **kwargs):
        aliases = {}
        return self.module, aliases


class WhisperModelDecoderInstance(BaseModelInstance):
    def __init__(self, config):
        self.module = None
        self.config = config
        self.neuron_config = config.neuron_config

    def load_module(self):
        dims = self.config.dims
        self.module = NeuronTextDecoder(
            dims.n_vocab,
            dims.n_text_ctx,
            dims.n_audio_ctx,
            dims.n_text_state,
            dims.n_text_head,
            dims.n_text_layer,
            batch_size=self.neuron_config.batch_size,
            dtype=self.neuron_config.torch_dtype,
        )

    def get(self, bucket_rank, **kwargs):
        aliases = {}
        output_index = 1
        for i, layer in enumerate(self.module.blocks):
            aliases[layer.attn.cache_k] = output_index
            output_index = output_index + 1
        for i, layer in enumerate(self.module.blocks):
            aliases[layer.attn.cache_v] = output_index
            output_index = output_index + 1
        for i, layer in enumerate(self.module.blocks):
            aliases[layer.cross_attn.cache_k] = output_index
            output_index = output_index + 1
        for i, layer in enumerate(self.module.blocks):
            aliases[layer.cross_attn.cache_v] = output_index
            output_index = output_index + 1
        return self.module, aliases


class ModelWrapperWhisperEncoder(ModelWrapper):
    def __init__(
        self,
        config,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        model_init_kwargs={},
    ):
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs
        )
        self.bucket_config = None  # Set to None if no bucketing needed
        self._use_megakernel = (
            _HAS_MEGAKERNEL and os.environ.get("WHISPER_USE_MEGAKERNEL", "0") == "1"
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        # Generate example inputs for tracing
        audio = torch.randn(
            self.neuron_config.batch_size,
            self.config.dims.n_mels,
            self.config.dims.n_audio_ctx * 2,
            dtype=self.neuron_config.torch_dtype,
        )
        if self._use_megakernel:
            # Packed weights tensor: 1D bf16 dummy with correct total size
            n_layers = self.config.dims.n_audio_layer
            packed_weights = torch.zeros(
                n_layers * _MK_ELEMENTS_PER_LAYER, dtype=torch.bfloat16
            )
            inputs = [(audio, packed_weights)]
        else:
            inputs = [(audio,)]
        return inputs

    def get_model_instance(self):
        return WhisperModelEncoderInstance(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ModelWrapperWhisperDecoderPrefill(ModelWrapper):
    def __init__(
        self,
        config,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        model_init_kwargs={},
    ):
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs
        )
        self.bucket_config = None  # Set to None if no bucketing needed

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        # Generate example inputs for tracing
        audio_embed = torch.randn(
            self.neuron_config.batch_size,
            self.config.dims.n_audio_ctx,
            self.config.dims.n_audio_state,
            dtype=self.neuron_config.torch_dtype,
        )
        padded_tokens = torch.zeros(
            (self.neuron_config.batch_size, self.config.dims.n_text_ctx),
            dtype=torch.int32,
        )
        last_pos = torch.zeros(self.neuron_config.batch_size, dtype=torch.int32)
        pad_mask = torch.zeros(
            (self.neuron_config.batch_size, self.config.dims.n_text_ctx),
            dtype=torch.int32,
        )
        inputs = [
            (padded_tokens, audio_embed, last_pos, pad_mask),
        ]
        return inputs

    def get_model_instance(self):
        return WhisperModelDecoderInstance(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ModelWrapperWhisperDecoderDecode(ModelWrapper):
    def __init__(
        self,
        config,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        model_init_kwargs={},
    ):
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs
        )
        self.bucket_config = None  # Set to None if no bucketing needed

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        # Generate example inputs for tracing.
        # Use minimal dummy xa (1 token instead of n_audio_ctx) since decode reads
        # cross-attention K/V from cache, not from xa. The xa tensor must be present
        # for forward signature compatibility but is unused in the decode graph.
        audio_embed = torch.randn(
            self.neuron_config.batch_size,
            1,
            self.config.dims.n_audio_state,
            dtype=self.neuron_config.torch_dtype,
        )
        padded_tokens = torch.zeros(
            (self.neuron_config.batch_size, 1), dtype=torch.int32
        )
        last_pos = torch.zeros(self.neuron_config.batch_size, dtype=torch.int32)
        pad_mask = torch.zeros(
            (self.neuron_config.batch_size, self.config.dims.n_text_ctx),
            dtype=torch.int32,
        )
        inputs = [
            (padded_tokens, audio_embed, last_pos, pad_mask),
        ]
        return inputs

    def get_model_instance(self):
        return WhisperModelDecoderInstance(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class NeuronApplicationWhisperEncoder(NeuronApplicationBase):
    _model_cls = NeuronAudioEncoder

    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__(model_path, config, *args, **kwargs)
        self.dims = config.dims
        self._use_megakernel = (
            _HAS_MEGAKERNEL and os.environ.get("WHISPER_USE_MEGAKERNEL", "0") == "1"
        )
        self.encoder_model = ModelWrapperWhisperEncoder(
            config=self.config,
            model_cls=self._model_cls,
            tag="Encoder",
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.encoder_model)

        # Packed megakernel weights (populated after weight loading)
        self._packed_weights = None

        # workaround for whisper PyTorchInference init, dummy blocks
        self.blocks = []

    def get_compiler_args(self):
        compiler_args = "--model-type=transformer"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        if self.config.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        compiler_args += f" --lnc={self.config.neuron_config.logical_nc_config}"
        return compiler_args

    @staticmethod
    def load_hf_model(model_path):
        return WhisperModel.from_pretrained(model_path)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: WhisperInferenceConfig
    ) -> dict:
        state_dict = convert_hf_state_dict_to_neuron(state_dict, type="encoder")
        state_dict = expand_state_dict(
            state_dict, config.dims, config.neuron_config.tp_degree
        )
        return state_dict

    def _build_packed_weights(self):
        """Build packed megakernel weights from the HF checkpoint.

        Called after load() to create the packed weight tensor that will be
        passed as a forward() input at inference time. Extracts weights directly
        from the neuron state dict (avoiding instantiation of parallel layers)
        and packs them into a single 1D bf16 tensor.
        """
        if not self._use_megakernel:
            return

        import logging

        logger = logging.getLogger(__name__)
        logger.info("Building packed megakernel weights from checkpoint...")

        # Load and convert the state dict (same path as normal weight loading)
        model_sd = self.checkpoint_loader_fn()

        # Extract and pack weights directly from the state dict.
        dtype = torch.bfloat16
        n_layers = self.config.dims.n_audio_layer
        packed = torch.empty(n_layers * _MK_ELEMENTS_PER_LAYER, dtype=dtype)
        offset = 0

        for i in range(n_layers):
            prefix = f"blocks.{i}"

            # Helper to get, cast, and optionally tile a weight
            def _get(key):
                return model_sd[f"{prefix}.{key}"].to(dtype).contiguous()

            def _get_tiled(key):
                return _tile_ln_weight(_get(key))

            # Pack in canonical order: attn_ln_w, attn_ln_b, qkv_w, qkv_b, ...
            weights_in_order = [
                _get_tiled("attn_ln.weight"),  # attn_ln_w [P_MAX, D_MODEL]
                _get_tiled("attn_ln.bias"),  # attn_ln_b [P_MAX, D_MODEL]
                _get("attn.qkv_proj.weight"),  # qkv_w [3*D_MODEL, D_MODEL]
                _get_tiled("attn.qkv_proj.bias"),  # qkv_b [P_MAX, 3*D_MODEL]
                _get("attn.out.weight"),  # out_w [D_MODEL, D_MODEL]
                _get_tiled("attn.out.bias"),  # out_b [P_MAX, D_MODEL]
                _get_tiled("mlp_ln.weight"),  # mlp_ln_w [P_MAX, D_MODEL]
                _get_tiled("mlp_ln.bias"),  # mlp_ln_b [P_MAX, D_MODEL]
                _get("mlp.up_proj.weight"),  # fc1_w [MLP_DIM, D_MODEL]
                _get_tiled("mlp.up_proj.bias"),  # fc1_b [P_MAX, MLP_DIM]
                _get("mlp.down_proj.weight"),  # fc2_w [D_MODEL, MLP_DIM]
                _get_tiled("mlp.down_proj.bias"),  # fc2_b [P_MAX, D_MODEL]
            ]

            for w in weights_in_order:
                flat = w.contiguous().view(-1)
                packed[offset : offset + flat.numel()] = flat
                offset += flat.numel()

        assert offset == packed.numel(), (
            f"Packing mismatch: {offset} != {packed.numel()}"
        )
        self._packed_weights = packed
        logger.info(
            f"Packed megakernel weights: {self._packed_weights.shape} "
            f"({self._packed_weights.numel() * 2 / 1024 / 1024:.1f} MB)"
        )

    def load(self, compiled_model_path, *args, **kwargs):
        """Override load to also build packed megakernel weights."""
        super().load(compiled_model_path, *args, **kwargs)
        self._build_packed_weights()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Whisper encoder.
        :param audio: Tensor of shape (batch_size, n_mels, n_audio_ctx)
        :return: Encoded audio features
        """
        audio_typed = audio.to(self.config.neuron_config.torch_dtype)
        if self._use_megakernel and self._packed_weights is not None:
            return self.traced_model(audio_typed, self._packed_weights).to(audio.dtype)
        else:
            return self.traced_model(audio_typed).to(audio.dtype)


class NeuronApplicationWhisperDecoder(NeuronApplicationBase):
    _model_cls = NeuronTextDecoder

    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__(model_path, config, *args, **kwargs)
        self.dims = config.dims
        self.decoder_prefill_model = ModelWrapperWhisperDecoderPrefill(
            config=self.config,
            model_cls=self._model_cls,
            tag="DecoderPrefill",
            compiler_args=self.get_compiler_args(),
        )
        self.decoder_decode_model = ModelWrapperWhisperDecoderDecode(
            config=self.config,
            model_cls=self._model_cls,
            tag="DecoderDecode",
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.decoder_prefill_model)
        self.models.append(self.decoder_decode_model)

        # workaround for whisper PyTorchInference init, dummy blocks
        self.blocks = []

    def get_compiler_args(self):
        compiler_args = "--model-type=transformer"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        if self.config.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        compiler_args += f" --lnc={self.config.neuron_config.logical_nc_config}"
        return compiler_args

    @staticmethod
    def load_hf_model(model_path):
        return WhisperModel.from_pretrained(model_path)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: WhisperInferenceConfig
    ) -> dict:
        state_dict = convert_hf_state_dict_to_neuron(state_dict, type="decoder")
        state_dict = expand_state_dict(
            state_dict, config.dims, config.neuron_config.tp_degree
        )
        return state_dict

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        last_pos: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Whisper decoder.
        :param text: Tensor of shape (batch_size, <= n_text_ctx)
        :param audio: Encoded audio features of shape (batch_size, n_audio_ctx, n_audio_state)
        :param last_pos: Tensor of shape (batch_size,) indicating the last valid token position per sample
        :param pad_mask: Tensor of shape (batch_size, n_text_ctx) indicating valid positions
        :return: Logits for the next token prediction
        """
        return self.traced_model(text, audio, last_pos, pad_mask)


class NeuronApplicationWhisper(Whisper):
    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__(config.dims)
        self.config = config
        self.dims = config.dims
        self.encoder_path_suffix = "encoder"
        self.decoder_path_suffix = "decoder"
        self.encoder = NeuronApplicationWhisperEncoder(
            model_path=os.path.join(model_path, self.encoder_path_suffix),
            config=config,
            *args,
            **kwargs,
        )
        self.decoder = NeuronApplicationWhisperDecoder(
            model_path=os.path.join(model_path, self.decoder_path_suffix),
            config=config,
            *args,
            **kwargs,
        )

    def compile(self, compiled_model_path, *args, **kwargs):
        self.encoder.compile(
            os.path.join(compiled_model_path, self.encoder_path_suffix), *args, **kwargs
        )
        self.decoder.compile(
            os.path.join(compiled_model_path, self.decoder_path_suffix), *args, **kwargs
        )

    def load(self, compiled_model_path, *args, **kwargs):
        self.encoder.load(
            os.path.join(compiled_model_path, self.encoder_path_suffix), *args, **kwargs
        )
        self.decoder.load(
            os.path.join(compiled_model_path, self.decoder_path_suffix), *args, **kwargs
        )

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        tokens = tokens.to(torch.int32)
        padded_tokens, last_pos, pad_mask = self._prepare_decoder_inputs(tokens)
        is_prefill = padded_tokens.shape[1] > 1
        if is_prefill:
            xa = audio_features.to(self.config.neuron_config.torch_dtype)
        else:
            # Decode: pass minimal dummy xa since cross-attention K/V caches
            # were populated during prefill. xa is unused in the decode graph.
            xa = torch.zeros(
                audio_features.shape[0],
                1,
                audio_features.shape[2],
                dtype=self.config.neuron_config.torch_dtype,
            )
        logits = self.decoder(padded_tokens, xa, last_pos, pad_mask)
        if is_prefill:
            # Gather logits at each sample's last valid position.
            # last_pos shape: (batch_size,) — each value is the index of the
            # last real token for that sample in the padded sequence.
            idx = (
                last_pos.to(torch.int64).view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
            )
            logits = torch.gather(logits, 1, idx)
        return logits

    def _prepare_decoder_inputs(self, tokens: torch.Tensor):
        pad_token = -1
        last_pos = torch.tensor(
            [len(prompt) - 1 for prompt in tokens], dtype=torch.int32
        )
        padded_tokens = F.pad(
            tokens, (0, self.dims.n_text_ctx - tokens.shape[1]), value=pad_token
        )
        pad_mask = torch.where(padded_tokens != pad_token, 1, 0).to(torch.int32)
        padded_tokens = torch.where(padded_tokens == pad_token, 0, padded_tokens)
        return padded_tokens, last_pos, pad_mask

    @property
    def device(self):
        return torch.device("cpu")

    decode = decode_function
