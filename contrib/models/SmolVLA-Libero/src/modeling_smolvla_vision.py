"""
SmolVLA vision subgraph
=======================

Compiled subgraph #1 of three:

    Input:  pixel_values      [B, 3, 512, 512]    BF16
    Output: vision_features   [B, 64, 960]        BF16

Pipeline (all on Neuron):

    SigLIPVisionTransformer (12 layers, hidden=768)
      → patch_embedding (Conv2d 16x16, stride 16) → [B, 1024, 768]
      → + position_embedding[1024, 768]
      → 12x SigLIP encoder layer (post-LN architecture, GELU MLP, eager attn)
      → post_layernorm

    SmolVLMConnector
      → pixel_shuffle x4    [B, 1024, 768]  → [B, 64, 12288]
      → modality_projection.proj  Linear(12288, 960, bias=False)

The Neuron compiler accepts Conv2d in eval mode for the patch embedding (it
unfolds internally), so we keep the Conv2d as in the HF source rather than
pre-unfolding to keep weight mapping trivial.

DEVIATIONS from "everything on Neuron":
  - None for this subgraph. Image preprocessing (PIL → resize → normalize)
    runs on CPU, but that's data-loading, not model compute.

Per-camera vs all-cameras:
  This subgraph runs once per camera (B=1, single image at a time). The
  caller stacks the three outputs to get [B, 192, 960] before passing into
  the prefix subgraph. This avoids a 3x increase in patch-embedding tile size
  at compile time.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.model_wrapper import ModelWrapper

import config_constants as C
from neuron_action_head_base import NeuronDenoisingConfig


# ---------------------------------------------------------------------------
# Parallel-linear helpers (TP=1 on this instance, see config_constants.py)
# ---------------------------------------------------------------------------

def _col(in_f: int, out_f: int, bias: bool = True) -> nn.Module:
    if parallel_state.model_parallel_is_initialized():
        return ColumnParallelLinear(
            in_f, out_f,
            bias=bias,
            gather_output=False,
            dtype=torch.bfloat16,
            tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        )
    return nn.Linear(in_f, out_f, bias=bias)


def _row(in_f: int, out_f: int, bias: bool = True) -> nn.Module:
    if parallel_state.model_parallel_is_initialized():
        return RowParallelLinear(
            in_f, out_f,
            bias=bias,
            input_is_parallel=True,
            dtype=torch.bfloat16,
            tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        )
    return nn.Linear(in_f, out_f, bias=bias)


# ---------------------------------------------------------------------------
# SigLIP encoder
# ---------------------------------------------------------------------------

class SigLIPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        H, D = C.VISION_HIDDEN, C.VISION_HEAD_DIM
        self.num_heads = C.VISION_NUM_HEADS
        self.head_dim = D
        self.scale = D ** -0.5
        # SigLIP attention has bias on q/k/v and out_proj
        self.q_proj = _col(H, H, bias=True)
        self.k_proj = _col(H, H, bias=True)
        self.v_proj = _col(H, H, bias=True)
        self.out_proj = _row(H, H, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)   # [B, H, N, D]
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(attn, v)                           # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, H * D)
        return self.out_proj(out)


class SigLIPMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = _col(C.VISION_HIDDEN, C.VISION_INTERMEDIATE, bias=True)
        self.fc2 = _row(C.VISION_INTERMEDIATE, C.VISION_HIDDEN, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SigLIPEncoderLayer(nn.Module):
    """Pre-norm transformer block (matches HF SiglipEncoderLayer)."""
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(C.VISION_HIDDEN, eps=C.VISION_LAYER_NORM_EPS)
        self.self_attn = SigLIPAttention()
        self.layer_norm2 = nn.LayerNorm(C.VISION_HIDDEN, eps=C.VISION_LAYER_NORM_EPS)
        self.mlp = SigLIPMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SmolVLMVisionModel(nn.Module):
    """SigLIP vision tower + post-layernorm."""
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=C.VISION_HIDDEN,
            kernel_size=C.VISION_PATCH_SIZE,
            stride=C.VISION_PATCH_SIZE,
            padding="valid",
        )
        self.position_embedding = nn.Embedding(
            C.VISION_NUM_PATCHES, C.VISION_HIDDEN
        )
        self.register_buffer(
            "position_ids",
            torch.arange(C.VISION_NUM_PATCHES).unsqueeze(0),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            [SigLIPEncoderLayer() for _ in range(C.VISION_NUM_LAYERS)]
        )
        self.post_layernorm = nn.LayerNorm(C.VISION_HIDDEN, eps=C.VISION_LAYER_NORM_EPS)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [B, 3, 512, 512] -> [B, 768, 32, 32] -> [B, 1024, 768]
        x = self.patch_embedding(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embedding(self.position_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        return x   # [B, 1024, 768]


class SmolVLMConnector(nn.Module):
    """Pixel-shuffle 4x then linear projection to VLM hidden size."""
    def __init__(self):
        super().__init__()
        self.scale_factor = C.PIXEL_SHUFFLE_SCALE
        # HF naming: connector.modality_projection.proj
        self.modality_projection_proj = _col(
            C.CONNECTOR_INPUT_DIM, C.VLM_HIDDEN, bias=False
        )

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1024, 768]  ->  [B, 64, 12288]
        B, N, D = x.shape
        H = W = int(N ** 0.5)             # 32
        s = self.scale_factor             # 4
        x = x.view(B, H, W, D)
        x = x.view(B, H, W // s, D * s)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, W // s, H // s, D * s * s)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, (H // s) * (W // s), D * s * s)
        return x

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.pixel_shuffle(image_hidden_states)
        return self.modality_projection_proj(x)   # [B, 64, 960]


# ---------------------------------------------------------------------------
# Combined vision encoder model — the thing that gets compiled to a NEFF
# ---------------------------------------------------------------------------

class SmolVLAVisionEncoder(nn.Module):
    """
    pixel_values [B, 3, 512, 512] BF16  ->  vision_features [B, 64, 960] BF16

    The SigLIP image embeddings are scaled by sqrt(hidden_dim) AFTER the
    connector to match `embed_prefix` in the HF source (see modeling_smolvla.py
    line 654-659). The scale is folded into this subgraph so the prefix
    subgraph receives ready-to-concat embeddings.
    """
    def __init__(self):
        super().__init__()
        self.vision_model = SmolVLMVisionModel()
        self.connector = SmolVLMConnector()
        # img_emb * sqrt(VLM_HIDDEN) — pre-compute scalar
        self.register_buffer(
            "vlm_hidden_sqrt",
            torch.tensor(C.VLM_HIDDEN ** 0.5, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.vision_model(pixel_values)        # [B, 1024, 768]
        x = self.connector(x)                       # [B, 64, 960]
        x = x * self.vlm_hidden_sqrt
        return x


# ---------------------------------------------------------------------------
# Vision wrapper for ModelBuilder compilation
# ---------------------------------------------------------------------------

class SmolVLAVisionWrapper(ModelWrapper):
    """
    Wraps SmolVLAVisionEncoder for compilation via NxDI's ModelBuilder.

    Takes the same minimal NeuronDenoisingConfig used by the action head — its
    fields are a strict superset of what ModelWrapper needs.
    """

    tag = "vision_encoder"

    def __init__(self, config: NeuronDenoisingConfig):
        nn.Module.__init__(self)
        super().__init__(config=config, model_cls=type(self))
        self.config = config
        self.model = None  # constructed in load_module() — see nxdi_background.md

    def load_module(self):
        # parallel_state is active here under ModelBuilder
        self.model = SmolVLAVisionEncoder()
        self.model = self.model.bfloat16().eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)

    def input_generator(self) -> List[Tuple[torch.Tensor, ...]]:
        return [(
            torch.zeros(
                self.config.neuron_config.batch_size, 3,
                C.VISION_IMAGE_SIZE, C.VISION_IMAGE_SIZE,
                dtype=torch.bfloat16,
            ),
        )]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return super().load_state_dict(state_dict, strict=strict, **kwargs)
