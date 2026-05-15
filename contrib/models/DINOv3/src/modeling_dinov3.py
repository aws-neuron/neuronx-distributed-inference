"""
DINOv3 vision foundation models on AWS Neuron.

Supports two compilation paths:
1. torch_neuronx.trace() -- for ViT-S/B/L/H+ and ConvNeXt-T/S/B/L (single NeuronCore)
2. neuronx-distributed ModelBuilder with TP -- for ViT-7B (6.7B params, requires TP=4)

All models are encoder-only with static input shapes -- ideal for torch_neuronx.trace().
FP32 by default; --auto-cast=matmult is critical for performance.

Architecture:
  - ViT: patch size 16, CLS + register/storage tokens, 2D axial RoPE, SwiGLU FFN, LayerScale
  - ConvNeXt: hierarchical conv backbone (Conv2d, GroupNorm, GELU, LayerScale)
  - Input: 224x224 RGB images
  - Output: dense feature embeddings (CLS token for ViT, pooled features for ConvNeXt)

Reference: https://github.com/facebookresearch/dinov3
License: DINOv3 License (not Apache/MIT)
"""

import math
import os
import sys
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_neuronx
from torch import Tensor, nn

# Default compiler args for trace-based models
COMPILER_ARGS_VIT = [
    "--auto-cast",
    "matmult",
    "--auto-cast-type",
    "bf16",
    "--model-type",
    "transformer",
]

COMPILER_ARGS_CONVNEXT = [
    "--auto-cast",
    "matmult",
    "--auto-cast-type",
    "bf16",
]

# Model registry: hub function name -> configuration
MODEL_REGISTRY = {
    "vit_s": {
        "hub_name": "dinov3_vits16",
        "arch": "vit",
        "embed_dim": 384,
        "params_M": 21.6,
    },
    "vit_b": {
        "hub_name": "dinov3_vitb16",
        "arch": "vit",
        "embed_dim": 768,
        "params_M": 85.7,
    },
    "vit_l": {
        "hub_name": "dinov3_vitl16",
        "arch": "vit",
        "embed_dim": 1024,
        "params_M": 303.2,
    },
    "vit_h_plus": {
        "hub_name": "dinov3_vith16plus",
        "arch": "vit",
        "embed_dim": 1280,
        "params_M": 840.6,
    },
    "convnext_tiny": {
        "hub_name": "dinov3_convnext_tiny",
        "arch": "convnext",
        "embed_dim": 768,
        "params_M": 27.8,
    },
    "convnext_base": {
        "hub_name": "dinov3_convnext_base",
        "arch": "convnext",
        "embed_dim": 1024,
        "params_M": 87.6,
    },
}

IMG_SIZE = 224
BATCH_SIZE = 1


def load_dinov3_model(hub_name: str, repo_dir: str = "/mnt/models/dinov3"):
    """Load a DINOv3 model from the cloned repository.

    Args:
        hub_name: Model function name (e.g., 'dinov3_vits16', 'dinov3_convnext_tiny')
        repo_dir: Path to cloned dinov3 repository

    Returns:
        PyTorch model in eval mode with random weights (pretrained=False)
    """
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from dinov3.hub.backbones import (
        dinov3_vits16,
        dinov3_vitb16,
        dinov3_vitl16,
        dinov3_vith16plus,
        dinov3_vit7b16,
        dinov3_convnext_tiny,
        dinov3_convnext_base,
    )

    hub_fns = {
        "dinov3_vits16": dinov3_vits16,
        "dinov3_vitb16": dinov3_vitb16,
        "dinov3_vitl16": dinov3_vitl16,
        "dinov3_vith16plus": dinov3_vith16plus,
        "dinov3_vit7b16": dinov3_vit7b16,
        "dinov3_convnext_tiny": dinov3_convnext_tiny,
        "dinov3_convnext_base": dinov3_convnext_base,
    }

    if hub_name not in hub_fns:
        raise ValueError(
            f"Unknown model: {hub_name}. Available: {list(hub_fns.keys())}"
        )

    model = hub_fns[hub_name](pretrained=False)
    model.eval()
    return model


def trace_dinov3(
    model: nn.Module,
    is_convnext: bool = False,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    save_path: Optional[str] = None,
    inline_weights: bool = True,
) -> torch.jit.ScriptModule:
    """Trace a DINOv3 model for Neuron via torch_neuronx.trace().

    Args:
        model: DINOv3 model (ViT or ConvNeXt) in eval mode
        is_convnext: True for ConvNeXt models (uses different compiler args)
        img_size: Input image size (default: 224)
        batch_size: Batch size for tracing (default: 1)
        save_path: Optional path to save compiled model
        inline_weights: Whether to inline weights into NEFF (default: True)

    Returns:
        Compiled Neuron model
    """
    compiler_args = COMPILER_ARGS_CONVNEXT if is_convnext else COMPILER_ARGS_VIT
    example_input = torch.randn(batch_size, 3, img_size, img_size)

    print(f"  Tracing with compiler_args={compiler_args}")
    t0 = time.time()
    model_neuron = torch_neuronx.trace(
        model,
        example_input,
        compiler_args=compiler_args,
        inline_weights_to_neff=inline_weights,
    )
    compile_time = time.time() - t0
    print(f"  Compilation time: {compile_time:.1f}s")

    if save_path:
        torch.jit.save(model_neuron, save_path)
        neff_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  Saved to {save_path} ({neff_size_mb:.1f} MB)")

    return model_neuron


def validate_accuracy(
    model_cpu: nn.Module,
    model_neuron: torch.jit.ScriptModule,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, float]:
    """Compare CPU vs Neuron outputs for accuracy validation.

    Args:
        model_cpu: CPU reference model
        model_neuron: Compiled Neuron model
        img_size: Input image size
        batch_size: Batch size

    Returns:
        Dict with cosine_sim, max_diff, l2_rel_error
    """
    example_input = torch.randn(batch_size, 3, img_size, img_size)

    with torch.no_grad():
        cpu_out = model_cpu(example_input)
        neuron_out = model_neuron(example_input)

    cpu_flat = cpu_out.flatten().float()
    neuron_flat = neuron_out.flatten().float()

    cosine_sim = F.cosine_similarity(
        cpu_flat.unsqueeze(0), neuron_flat.unsqueeze(0)
    ).item()

    max_diff = (cpu_flat - neuron_flat).abs().max().item()
    l2_rel = (torch.norm(cpu_flat - neuron_flat) / torch.norm(cpu_flat)).item()

    return {
        "cosine_sim": cosine_sim,
        "max_diff": max_diff,
        "l2_rel_error": l2_rel,
    }


def benchmark_model(
    model_neuron,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    warmup_iters: int = 5,
    bench_iters: int = 50,
) -> Dict[str, float]:
    """Benchmark latency and throughput of a compiled Neuron model.

    Args:
        model_neuron: Compiled Neuron model (trace or TP)
        img_size: Input image size
        batch_size: Batch size per inference call
        warmup_iters: Number of warmup iterations
        bench_iters: Number of timed iterations

    Returns:
        Dict with mean_latency_ms, median_latency_ms, p99_latency_ms, throughput_img_s
    """
    example_input = torch.randn(batch_size, 3, img_size, img_size)

    # For TP models that expect bfloat16 input
    if hasattr(model_neuron, "_is_tp_model"):
        example_input = example_input.bfloat16()

    for _ in range(warmup_iters):
        model_neuron(example_input)

    latencies = []
    for _ in range(bench_iters):
        t0 = time.time()
        model_neuron(example_input)
        latencies.append((time.time() - t0) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_latency_ms": latencies.mean(),
        "median_latency_ms": float(np.median(latencies)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_img_s": 1000.0 / latencies.mean() * batch_size,
    }


def benchmark_dataparallel(
    model_neuron,
    num_cores: int = 4,
    img_size: int = IMG_SIZE,
    batch_sizes: Optional[List[int]] = None,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> Dict[int, Dict[str, float]]:
    """Benchmark DataParallel throughput across multiple batch sizes.

    Args:
        model_neuron: Compiled Neuron model (single-core)
        num_cores: Number of NeuronCores for DataParallel
        img_size: Input image size
        batch_sizes: List of batch sizes to test (default: [cores, 2*cores, 4*cores])
        warmup_iters: Number of warmup iterations
        bench_iters: Number of timed iterations

    Returns:
        Dict mapping batch_size -> benchmark metrics
    """
    if batch_sizes is None:
        batch_sizes = [num_cores, num_cores * 2, num_cores * 4]

    model_dp = torch_neuronx.DataParallel(
        model_neuron,
        device_ids=list(range(num_cores)),
        dim=0,
    )

    results = {}
    for bs in batch_sizes:
        dp_input = torch.randn(bs, 3, img_size, img_size)

        for _ in range(warmup_iters):
            model_dp(dp_input)

        latencies = []
        for _ in range(bench_iters):
            t0 = time.time()
            model_dp(dp_input)
            latencies.append((time.time() - t0) * 1000)

        latencies = np.array(latencies)
        results[bs] = {
            "mean_latency_ms": latencies.mean(),
            "median_latency_ms": float(np.median(latencies)),
            "throughput_img_s": 1000.0 / latencies.mean() * bs,
        }

    return results


# ============================================================================
# TP ViT-7B Model Definition
# ============================================================================
# The following classes implement a standalone tensor-parallel ViT-7B/16 model
# using neuronx-distributed parallel layers. This is required because the 6.7B
# parameter model produces a 20.1 GB NEFF that does not fit in single-core HBM.
#
# This is the first encoder-only vision model to use TP on Neuron.
# ============================================================================


def _rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (_rope_rotate_half(x) * sin)


class RopePositionEmbedding(nn.Module):
    """2D axial RoPE for vision patches. Replicated across TP ranks."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        base: float = 100.0,
        normalize_coords: str = "separate",
        rescale_coords: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        D_head = embed_dim // num_heads
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        periods = base ** (2 * torch.arange(D_head // 4, dtype=dtype) / (D_head // 2))
        self.register_buffer("periods", periods, persistent=True)

    def forward(self, H: int, W: int) -> Tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unsupported normalize_coords: {self.normalize_coords}")

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return (sin, cos)


class PatchEmbed(nn.Module):
    """Patch embedding via Conv2d. Replicated across TP ranks (small layer)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # [B, C, H, W]
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = x.reshape(-1, H, W, x.shape[-1])  # [B, H, W, C]
        return x

    def reset_parameters(self):
        k = 1 / (self.proj.in_channels * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class LayerScale(nn.Module):
    """Per-dimension scaling. Replicated across TP ranks."""

    def __init__(
        self, dim: int, init_values: float = 1e-5, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        self.gamma.data *= 1e-5


class TPSelfAttention(nn.Module):
    """Self-attention with tensor-parallel Q/K/V/O projections.

    TP strategy:
      - Q, K, V: ColumnParallelLinear (gather_output=True) -- each rank gets full tensor
      - O: RowParallelLinear (input_is_parallel=False) -- handles all-reduce
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        mask_k_bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from neuronx_distributed.parallel_layers.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        self.q_proj = ColumnParallelLinear(
            dim, dim, bias=qkv_bias, gather_output=True, dtype=dtype, pad=True
        )
        self.k_proj = ColumnParallelLinear(
            dim, dim, bias=qkv_bias, gather_output=True, dtype=dtype, pad=True
        )
        self.v_proj = ColumnParallelLinear(
            dim, dim, bias=qkv_bias, gather_output=True, dtype=dtype, pad=True
        )
        self.proj = RowParallelLinear(
            dim, dim, bias=proj_bias, input_is_parallel=False, dtype=dtype
        )
        self.mask_k_bias = mask_k_bias

    def apply_rope(self, q, k, rope):
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q, k = q.to(dtype=rope_dtype), k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = _rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :prefix, :]
        k = _rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    def forward(self, x: Tensor, rope=None) -> Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.dim)
        return self.proj(attn_out)


class TPSwiGLUFFN(nn.Module):
    """SwiGLU FFN with tensor-parallel w1/w2 (gate/up) and w3 (down).

    TP strategy:
      - w1, w2: ColumnParallelLinear (gather_output=False) -- keep sharded
      - w3: RowParallelLinear (input_is_parallel=True) -- reduce
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        align_to: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from neuronx_distributed.parallel_layers.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)

        self.w1 = ColumnParallelLinear(
            in_features,
            swiglu_hidden_features,
            bias=bias,
            gather_output=False,
            dtype=dtype,
            pad=True,
        )
        self.w2 = ColumnParallelLinear(
            in_features,
            swiglu_hidden_features,
            bias=bias,
            gather_output=False,
            dtype=dtype,
            pad=True,
        )
        self.w3 = RowParallelLinear(
            swiglu_hidden_features,
            out_features,
            bias=bias,
            input_is_parallel=True,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class TPSelfAttentionBlock(nn.Module):
    """Transformer block: LayerNorm -> Attention -> LayerScale -> LayerNorm -> FFN -> LayerScale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 3.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        layerscale_init: Optional[float] = 1e-5,
        mask_k_bias: bool = False,
        align_to: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5, dtype=dtype)
        self.attn = TPSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mask_k_bias=mask_k_bias,
            dtype=dtype,
        )
        self.ls1 = (
            LayerScale(dim, init_values=layerscale_init, dtype=dtype)
            if layerscale_init
            else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-5, dtype=dtype)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = TPSwiGLUFFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            bias=ffn_bias,
            align_to=align_to,
            dtype=dtype,
        )
        self.ls2 = (
            LayerScale(dim, init_values=layerscale_init, dtype=dtype)
            if layerscale_init
            else nn.Identity()
        )

    def forward(self, x: Tensor, rope=None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class TPDinoViT7B(nn.Module):
    """DINOv3 ViT-7B/16 with tensor parallelism via neuronx-distributed.

    Config (from dinov3 hub/backbones.py dinov3_vit7b16):
      embed_dim=4096, depth=40, num_heads=32, head_dim=128
      ffn_layer=swiglu64 (SwiGLU with align_to=64), ffn_ratio=3
      qkv_bias=False, proj_bias=True, ffn_bias=True
      n_storage_tokens=4, layerscale_init=1e-5
      RoPE: base=100, normalize_coords=separate, rescale_coords=2
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 4096,
        depth: int = 40,
        num_heads: int = 32,
        ffn_ratio: float = 3.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        n_storage_tokens: int = 4,
        layerscale_init: float = 1e-5,
        mask_k_bias: bool = True,
        align_to: int = 64,
        rope_base: float = 100.0,
        rope_normalize_coords: str = "separate",
        rope_rescale_coords: Optional[float] = 2.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_storage_tokens = n_storage_tokens
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            dtype=dtype,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, dtype=dtype))
        self.storage_tokens = nn.Parameter(
            torch.empty(1, n_storage_tokens, embed_dim, dtype=dtype)
        )
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, dtype=dtype))
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=rope_base,
            normalize_coords=rope_normalize_coords,
            rescale_coords=rope_rescale_coords,
            dtype=torch.float32,
        )

        self.blocks = nn.ModuleList(
            [
                TPSelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    layerscale_init=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    align_to=align_to,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-5, dtype=dtype)
        self.head = nn.Identity()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        self.patch_embed.reset_parameters()
        for block in self.blocks:
            if hasattr(block, "ls1") and isinstance(block.ls1, LayerScale):
                block.ls1.reset_parameters()
            if hasattr(block, "ls2") and isinstance(block.ls2, LayerScale):
                block.ls2.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)  # [B, HW, C]

        cls_token = self.cls_token + 0 * self.mask_token  # tie mask_token gradient
        x = torch.cat(
            [cls_token.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1), x],
            dim=1,
        )

        rope = self.rope_embed(H=H, W=W)

        for blk in self.blocks:
            x = blk(x, rope=rope)

        x = self.norm(x)
        return self.head(x[:, 0])  # [B, embed_dim]


def create_vit7b_tp(dtype: torch.dtype = torch.bfloat16) -> TPDinoViT7B:
    """Create TP ViT-7B model with initialized weights.

    Must be called inside a NxDParallelState context for TP to work.

    Returns:
        TPDinoViT7B model in eval mode
    """
    model = TPDinoViT7B(dtype=dtype)
    model.init_weights()
    model.eval()
    return model


def compile_vit7b_tp(
    tp_degree: int = 4,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    save_dir: str = "/mnt/models/compiled/dinov3_vit7b_tp4",
) -> "NxDModel":
    """Compile ViT-7B with tensor parallelism via ModelBuilder.

    Must be run with NEURON_RT_NUM_CORES >= tp_degree.

    Args:
        tp_degree: Number of NeuronCores for TP (default: 4)
        img_size: Input image size (default: 224)
        batch_size: Batch size (default: 1)
        save_dir: Directory for compiler workdir and artifacts

    Returns:
        Compiled NxD model ready for inference
    """
    from neuronx_distributed.trace.parallel_context import NxDParallelState
    from neuronx_distributed import ModelBuilder

    os.makedirs(save_dir, exist_ok=True)
    os.environ["NEURON_RT_NUM_CORES"] = str(tp_degree)

    example_input = torch.randn(batch_size, 3, img_size, img_size).bfloat16()

    compiler_args = (
        "--auto-cast=matmult "
        "--auto-cast-type=bf16 "
        "--model-type=transformer "
        "--enable-saturate-infinity "
        "-O1"
    )

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        model = create_vit7b_tp(dtype=torch.bfloat16)
        model_state = model.state_dict()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  TP ViT-7B parameters (per-rank): {n_params:.1f}M")

        builder = ModelBuilder(model)

        t0 = time.time()
        builder.trace(args=example_input, tag="vit7b")
        trace_time = time.time() - t0
        print(f"  Trace time: {trace_time:.1f}s")

        t0 = time.time()
        nxd_model = builder.compile(
            compiler_workdir=os.path.join(save_dir, "compiler_workdir"),
            compiler_args=compiler_args,
        )
        compile_time = time.time() - t0
        print(f"  Compile time: {compile_time:.1f}s")

    # Outside context: set weights and load on Neuron
    sharded_checkpoint = [model_state for _ in range(tp_degree)]
    nxd_model.set_weights(sharded_checkpoint)
    nxd_model.to_neuron()

    # Tag for benchmark function
    nxd_model._is_tp_model = True

    return nxd_model
