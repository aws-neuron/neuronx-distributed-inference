"""Evo2-7B Neuron acceleration module.

Provides wrapper classes and compilation utilities for running Arc Institute's
Evo 2 (7B, StripedHyena 2 architecture) DNA language model on AWS Trainium 2
hardware. Uses vanilla torch_neuronx.trace() compilation with block-by-block
tracing for prefill and a batched mega-NEFF for decode.

Evo2 is a hybrid SSM+Attention model with 32 layers in a repeating pattern:
  HCS (FIR k=7) -> HCM (FIR k=128) -> HCL (IIR/SSM) -> ATT (SDPA)

Key challenges solved:
  1. torch.fft.* unsupported on Neuron XLA -> NeuronFFT (Stockham FFT)
  2. HBM state persistence for KV-cache + SSM state -> input_output_aliases
  3. bf16 parameter mutation from shared sub-modules -> deep-copy prefill state
  4. Neuron compiler ICE with reshape-based interleave -> stride slicing
  5. Block 24 nondeterministic NaN -> warmup inference passes

Two compilation strategies:
  - Prefill (seq_len up to 2048): 32 independent block NEFFs
  - Decode (single-token, batched): Single mega-NEFF with all 32 blocks,
    batch dimension, and input_output_aliases for HBM state

DP=4 parallelism via process-per-core (NEURON_RT_VISIBLE_CORES).

Architecture constants (from evo2-7b-8k.yml):
  hidden_size = 4096
  num_attention_heads = 32
  head_dim = 128
  vocab_size = 512 (single-nucleotide: A, C, G, T + special)
  num_layers = 32
  state_size = 16 (IIR/SSM modal state dimension)
"""

import gc
import logging
import math
import os
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Architecture Constants
# ============================================================================

HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128
VOCAB_SIZE = 512
NUM_LAYERS = 32
STATE_SIZE = 16
SHORT_FILTER_LENGTH = 3
HCM_FILTER_LENGTH = 128
HCS_FILTER_LENGTH = 7

# Block type assignments (from evo2-7b-8k.yml)
BLOCK_TYPE_MAP = {
    0: "HCS",
    4: "HCS",
    7: "HCS",
    11: "HCS",
    14: "HCS",
    18: "HCS",
    21: "HCS",
    25: "HCS",
    28: "HCS",
    1: "HCM",
    5: "HCM",
    8: "HCM",
    12: "HCM",
    15: "HCM",
    19: "HCM",
    22: "HCM",
    26: "HCM",
    29: "HCM",
    2: "HCL",
    6: "HCL",
    9: "HCL",
    13: "HCL",
    16: "HCL",
    20: "HCL",
    23: "HCL",
    27: "HCL",
    30: "HCL",
    3: "ATT",
    10: "ATT",
    17: "ATT",
    24: "ATT",
    31: "ATT",
}

# First block of each type (for weight-replacement reference compilation)
REPRESENTATIVE_BLOCKS = {"HCS": 0, "HCM": 1, "HCL": 2, "ATT": 3}


# ============================================================================
# NeuronFFT: Pure-PyTorch Stockham FFT for Neuron Compilation
# ============================================================================


def _precompute_stockham_twiddles(n):
    """Precompute twiddle factors for Stockham FFT.

    For each stage s (0..log2(n)-1), the twiddle factor at position k
    within a half-group of size 2^s is:
        W = exp(-2*pi*j * k / 2^(s+1))

    Returns:
        List of (tw_real, tw_imag) tensors, each shape [half_size_for_stage]
    """
    num_stages = int(math.log2(n))
    twiddles = []
    for s in range(num_stages):
        m = 1 << (s + 1)
        half_m = m // 2
        k = np.arange(half_m, dtype=np.float64)
        angles = -2.0 * np.pi * k / m
        tw_real = torch.tensor(np.cos(angles), dtype=torch.float32)
        tw_imag = torch.tensor(np.sin(angles), dtype=torch.float32)
        twiddles.append((tw_real, tw_imag))
    return twiddles


class NeuronFFT(nn.Module):
    """Graph-efficient FFT for Neuron using Stockham algorithm.

    Uses reshape operations instead of indexing for butterfly stages.
    Produces a compact XLA graph suitable for torch_neuronx.trace().

    CRITICAL DESIGN: Twiddle factors and bit-reversal indices are stored
    as plain Python attributes (NOT registered buffers). This keeps them
    out of torch_neuronx's NeuronModule weight ParameterDict, avoiding
    dtype mismatch errors when the model has bf16 parameters but FFT
    needs fp32 twiddle factors and int64 indices.

    During torch_neuronx.trace(), the forward pass runs on CPU to build
    the XLA graph. Plain tensor attributes used in computation are captured
    as constants in the compiled NEFF.

    Args:
        n: FFT size (must be power of 2)
    """

    def __init__(self, n):
        super().__init__()
        assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
        self.n = n
        self.num_stages = int(math.log2(n))
        self.bit_rev = self._bit_reverse_indices(n)
        twiddles = _precompute_stockham_twiddles(n)
        for s, (tw_r, tw_i) in enumerate(twiddles):
            setattr(self, f"tw_r_{s}", tw_r)
            setattr(self, f"tw_i_{s}", tw_i)

    @staticmethod
    def _bit_reverse_indices(n):
        bits = int(math.log2(n))
        indices = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            rev = 0
            val = i
            for _ in range(bits):
                rev = (rev << 1) | (val & 1)
                val >>= 1
            indices[i] = rev
        return indices

    def forward(self, x_real, x_imag):
        """Forward FFT on last dimension.

        Args:
            x_real, x_imag: shape [*, n], float32

        Returns:
            (y_real, y_imag): shape [*, n], float32
        """
        n = self.n
        batch_shape = x_real.shape[:-1]
        batch_size = 1
        for d in batch_shape:
            batch_size *= d

        y_r = x_real.reshape(batch_size, n)
        y_i = x_imag.reshape(batch_size, n)
        y_r = y_r[:, self.bit_rev]
        y_i = y_i[:, self.bit_rev]

        for s in range(self.num_stages):
            half_m = 1 << s
            m = half_m << 1
            num_groups = n // m
            tw_r = getattr(self, f"tw_r_{s}")
            tw_i = getattr(self, f"tw_i_{s}")

            y_r = y_r.reshape(batch_size, num_groups, m)
            y_i = y_i.reshape(batch_size, num_groups, m)

            e_r = y_r[:, :, :half_m]
            e_i = y_i[:, :, :half_m]
            o_r = y_r[:, :, half_m:]
            o_i = y_i[:, :, half_m:]

            tw_r_b = tw_r.reshape(1, 1, half_m)
            tw_i_b = tw_i.reshape(1, 1, half_m)

            to_r = tw_r_b * o_r - tw_i_b * o_i
            to_i = tw_r_b * o_i + tw_i_b * o_r

            top_r = e_r + to_r
            top_i = e_i + to_i
            bot_r = e_r - to_r
            bot_i = e_i - to_i

            y_r = torch.cat([top_r, bot_r], dim=-1)
            y_i = torch.cat([top_i, bot_i], dim=-1)

            y_r = y_r.reshape(batch_size, n)
            y_i = y_i.reshape(batch_size, n)

        y_r = y_r.reshape(*batch_shape, n)
        y_i = y_i.reshape(*batch_shape, n)
        return y_r, y_i


# ============================================================================
# NeuronFFT High-Level Wrappers (torch.fft.* drop-in replacements)
# ============================================================================


def neuron_rfft(x, n, fft_module):
    """Drop-in replacement for torch.fft.rfft().

    Args:
        x: Real input tensor, shape [*, L], float32
        n: FFT size (power of 2)
        fft_module: NeuronFFT instance for size n

    Returns:
        (y_real, y_imag): shape [*, n//2+1], positive-frequency half
    """
    L = x.shape[-1]
    x_real = x.to(torch.float32)
    if L < n:
        x_real = F.pad(x_real, (0, n - L))
    elif L > n:
        x_real = x_real[..., :n]
    x_imag = torch.zeros_like(x_real)
    y_real, y_imag = fft_module(x_real, x_imag)
    half = n // 2 + 1
    return y_real[..., :half], y_imag[..., :half]


def neuron_fft_real(x, n, fft_module):
    """FFT of a real-valued input (full spectrum)."""
    L = x.shape[-1]
    x_real = x.to(torch.float32)
    if L < n:
        x_real = F.pad(x_real, (0, n - L))
    elif L > n:
        x_real = x_real[..., :n]
    x_imag = torch.zeros_like(x_real)
    return fft_module(x_real, x_imag)


def neuron_irfft(y_real, y_imag, n, fft_module, norm=None):
    """Drop-in replacement for torch.fft.irfft().

    Args:
        y_real, y_imag: Half-spectrum, shape [*, n//2+1]
        n: Output size (power of 2)
        fft_module: NeuronFFT instance for size n
        norm: "forward" means no 1/N scaling

    Returns:
        result: Real tensor, shape [*, n]
    """
    interior_real = torch.flip(y_real[..., 1:-1], dims=[-1])
    interior_imag = -torch.flip(y_imag[..., 1:-1], dims=[-1])
    full_real = torch.cat([y_real, interior_real], dim=-1)
    full_imag = torch.cat([y_imag, interior_imag], dim=-1)
    fft_real, fft_imag = fft_module(full_real, -full_imag)
    result = fft_real / n
    if norm == "forward":
        result = result * n
    return result


def neuron_ifft(y_real, y_imag, n, fft_module):
    """Drop-in replacement for torch.fft.ifft()."""
    fft_real, fft_imag = fft_module(y_real, -y_imag)
    return fft_real / n, -fft_imag / n


def _complex_mul(a_r, a_i, b_r, b_i):
    """Complex multiply using real arithmetic."""
    return a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r


# ============================================================================
# Pre-Import Mocks (must run before importing Vortex/Evo2)
# ============================================================================


def _install_cuda_mocks():
    """Install mock modules to prevent CUDA import failures on Neuron.

    Must be called before importing evo2 or vortex packages.
    """
    if "flash_attn_2_cuda" not in sys.modules:
        mock = types.ModuleType("flash_attn_2_cuda")
        mock.fwd = None
        mock.bwd = None
        mock.varlen_fwd = None
        mock.varlen_bwd = None
        sys.modules["flash_attn_2_cuda"] = mock

    if "causal_conv1d_cuda" not in sys.modules:
        sys.modules["causal_conv1d_cuda"] = types.ModuleType("causal_conv1d_cuda")

    _original_cuda_device_ctx = torch.cuda.device

    class _SafeCudaDevice:
        def __init__(self, device):
            self._is_cuda = False
            if isinstance(device, int) and device >= 0:
                self._is_cuda = True
            elif isinstance(device, str) and device.startswith("cuda"):
                self._is_cuda = True
            elif isinstance(device, torch.device) and device.type == "cuda":
                self._is_cuda = True
            self._ctx = (
                _original_cuda_device_ctx(device) if self._is_cuda else nullcontext()
            )

        def __enter__(self):
            return self._ctx.__enter__()

        def __exit__(self, *args):
            return self._ctx.__exit__(*args)

    torch.cuda.device = _SafeCudaDevice
    torch.cuda.empty_cache = lambda: None


# ============================================================================
# Vortex Patches for Neuron
# ============================================================================


def patch_vortex_for_neuron():
    """Apply all Vortex monkey-patches required for Neuron compilation.

    Patches applied:
      1. apply_rotary -> PyTorch fallback (no Triton kernel)
      2. Flash Attention -> disabled (uses F.scaled_dot_product_attention)
      3. parallel_fir -> force F.conv1d for all filter lengths
    """
    import vortex.model.rotary as rotary_module
    import vortex.ops.embedding.rotary as rotary_ops_module

    _apply_rotary_emb_torch = rotary_module.apply_rotary_emb_torch

    def _patched_apply_rotary(
        x,
        cos,
        sin,
        seqlen_offsets=0,
        cu_seqlens=None,
        max_seqlen=None,
        interleaved=False,
        inplace=False,
        conjugate=False,
    ):
        if isinstance(seqlen_offsets, int) and seqlen_offsets > 0:
            cos = cos[seqlen_offsets:]
            sin = sin[seqlen_offsets:]
        seqlen = x.shape[1]
        if cos.shape[0] > seqlen:
            cos = cos[:seqlen]
            sin = sin[:seqlen]
        result = _apply_rotary_emb_torch(x, cos, sin, interleaved=interleaved)
        if inplace:
            x.copy_(result)
            return x
        return result

    rotary_ops_module.apply_rotary = _patched_apply_rotary
    rotary_module.apply_rotary = _patched_apply_rotary
    logger.info("[PATCH] apply_rotary -> PyTorch fallback")

    import vortex.model.attention as attn_module

    attn_module.local_flash_attn_qkvpacked_func = None
    attn_module.local_flash_attn_kvpacked_func = None
    attn_module.local_flash_attn_with_kvcache = None
    attn_module.local_flash_attn_varlen_qkvpacked_func = None
    attn_module.local_flash_attn_varlen_kvpacked_func = None
    logger.info("[PATCH] Flash Attention -> disabled")

    import vortex.model.engine as engine_module

    def patched_parallel_fir(
        self,
        fir_fn,
        u,
        weight,
        bias,
        L,
        dims,
        groups=None,
        gated_bias=False,
        column_split_hyena=False,
        dim_last=True,
        fir_length=3,
        gate=False,
        inference_params=None,
        prefill_mode=None,
        padding_mask=None,
    ):
        L = u.shape[1] if dim_last else u.shape[2]
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
        if gate:
            from vortex.model.utils import column_split

            if column_split_hyena:
                x2, x1, v = column_split(
                    u, num_attention_heads, hidden_size_per_attention_head
                )
            else:
                x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
            if self.hyena_flip_x1x2:
                x1, x2 = x2, x1
            u = x1 * v

        if fir_fn != F.conv1d:
            if dim_last:
                u = u.permute(0, 2, 1)
            z = fir_fn(u)[:, :L]
        else:
            if dim_last:
                u = u.permute(0, 2, 1)
            filter_w = weight.to(torch.float32)
            u_f32 = u.to(torch.float32)
            if filter_w.shape[0] != u_f32.shape[1]:
                repeat_factor = u_f32.shape[1] // filter_w.shape[0]
                filter_w = filter_w.repeat(repeat_factor, 1, 1)
            z = F.conv1d(
                u_f32,
                filter_w,
                bias=None,
                stride=1,
                padding=fir_length - 1,
                groups=u_f32.shape[1],
            )[..., :L]
            z = z.to(u.dtype)
            if bias is not None:
                if gated_bias:
                    z = z + bias[None, :, None] * u
                else:
                    z = z + bias[None, :, None]

        if type(padding_mask) == torch.Tensor:
            z = z * padding_mask[:, None]
        if gate:
            z = x2 * z

        fir_state = u[..., -fir_length + 1 :] if inference_params is not None else None
        return z, fir_state

    engine_module.HyenaInferenceEngine.parallel_fir = patched_parallel_fir
    logger.info("[PATCH] parallel_fir -> force F.conv1d")


def install_neuron_fft_patch(model, seq_len):
    """Install NeuronFFT v2 patch for HCL parallel_iir.

    Registers NeuronFFT modules as submodules of HCL blocks and patches
    the parallel_iir method to use them instead of torch.fft.*.

    Args:
        model: StripedHyena model instance
        seq_len: Prefill sequence length (FFT size = 2 * seq_len)
    """
    import vortex.model.engine as engine_module
    from vortex.model.model import HyenaCascade

    fft_size = 2 * seq_len
    logger.info(f"[FFT] Installing NeuronFFT v2 (fft_size={fft_size})")

    hcl_count = 0
    for name, module in model.named_modules():
        if isinstance(module, HyenaCascade) and module.h is None:
            module.register_module("neuron_fft", NeuronFFT(fft_size))
            hcl_count += 1

    logger.info(f"[FFT] Registered NeuronFFT on {hcl_count} HCL blocks")

    # Build layer_idx -> NeuronFFT mapping
    _layer_fft_map = {}
    for name, module in model.named_modules():
        if isinstance(module, HyenaCascade) and module.h is None:
            _layer_fft_map[module.layer_idx] = module.neuron_fft

    def _get_fft_module_for_layer(layer_idx):
        return _layer_fft_map[layer_idx]

    def patched_parallel_iir(
        self,
        z_pre,
        h,
        D,
        L,
        poles,
        residues,
        t,
        dims,
        layer_idx,
        inference_params=None,
        prefill_style="fft",
        fftconv_fn=None,
        padding_mask=None,
        use_flashfft=False,
        column_split_hyena=False,
        long_fir_threshold=None,
    ):
        """Neuron-compatible parallel_iir using NeuronFFT v2."""
        fft_size = 2 * L
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims

        if column_split_hyena:
            z = z_pre.reshape(
                z_pre.shape[0],
                num_attention_heads,
                3 * hidden_size_per_attention_head,
                z_pre.shape[2],
            )
            x2 = z[:, :, :hidden_size_per_attention_head]
            x1 = z[
                :,
                :,
                hidden_size_per_attention_head : 2 * hidden_size_per_attention_head,
            ]
            v = z[:, :, 2 * hidden_size_per_attention_head :]
            x2 = x2.reshape(x2.shape[0], -1, x2.shape[-1])
            x1 = x1.reshape(x1.shape[0], -1, x1.shape[-1])
            v = v.reshape(v.shape[0], -1, v.shape[-1])
        else:
            x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1
        x1v = x1 * v

        if inference_params is not None and prefill_style == "recurrence":
            y = self.prefill_via_direct_recurrence(
                inference_params=inference_params,
                x1v=x1v,
                L=L,
                poles=poles,
                residues=residues,
            )
        else:
            if long_fir_threshold is None:
                fft_mod = _get_fft_module_for_layer(layer_idx)
                H_r, H_i = neuron_rfft(h.to(torch.float32), fft_size, fft_mod)
                H_r = H_r / fft_size
                H_i = H_i / fft_size
                X_s_r, X_s_i = neuron_fft_real(x1v.to(torch.float32), fft_size, fft_mod)
                half = H_r.shape[-1]
                X_r = X_s_r[..., :half]
                X_i = X_s_i[..., :half]
                if len(z_pre.shape) > 3:
                    H_r = H_r.unsqueeze(1)
                    H_i = H_i.unsqueeze(1)
                prod_r, prod_i = _complex_mul(X_r, X_i, H_r, H_i)
                y = neuron_irfft(prod_r, prod_i, fft_size, fft_mod, norm="forward")[
                    ..., :L
                ]
            else:
                assert h.shape[0] == 1
                h_conv = h[0][:, None][..., :long_fir_threshold]
                y = F.conv1d(
                    x1v,
                    h_conv.to(dtype=x1v.dtype),
                    stride=1,
                    groups=x1v.shape[1],
                    padding=h_conv.shape[-1] - 1,
                )[..., :L]

        y = y.to(dtype=x1v.dtype)
        y = (y + x1v * D.unsqueeze(-1)) * x2

        if inference_params is not None:
            if prefill_style == "fft":
                logger.warning("prefill_via_modal_fft not yet implemented for Neuron")
        return y.permute(0, 2, 1)

    engine_module.HyenaInferenceEngine.parallel_iir = patched_parallel_iir
    logger.info("[FFT] parallel_iir patched with NeuronFFT v2")


# ============================================================================
# Model Loading
# ============================================================================


def load_config():
    """Load and patch evo2-7b-8k config for Neuron.

    Returns:
        dict: Model configuration with Neuron-safe settings.
    """
    import pkgutil
    import yaml

    config_data = pkgutil.get_data("evo2", "configs/evo2-7b-8k.yml")
    config = yaml.safe_load(config_data)
    config["use_flash_attn"] = False
    config["use_flash_rmsnorm"] = False
    config["use_flash_depthwise"] = False
    config["use_flashfft"] = False
    config["use_laughing_hyena"] = False
    config["use_fp8_input_projections"] = False
    config["inference_mode"] = True
    config["prefill_style"] = "fft"
    return config


def build_model(config, weights_dir=None):
    """Build StripedHyena model with optional weight loading.

    Args:
        config: Model configuration dict (from load_config())
        weights_dir: Path to weights file (e.g., "evo2_7b_base.pt").
            If None, model is initialized with random weights.

    Returns:
        (model, config): StripedHyena model and dotdict config
    """
    from vortex.model.utils import dotdict, load_checkpoint
    from vortex.model.model import StripedHyena

    config = dotdict(config)
    original_cuda_available = torch.cuda.is_available
    original_device_count = torch.cuda.device_count
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

    try:
        model = StripedHyena(config)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model: {param_count:,} params ({param_count * 2 / 1e9:.1f} GB bf16)"
        )
        if weights_dir:
            load_checkpoint(model, weights_dir)
            logger.info(f"Weights loaded from {weights_dir}")
    finally:
        torch.cuda.is_available = original_cuda_available
        torch.cuda.device_count = original_device_count

    model.eval()
    with torch.no_grad():
        for p in model.parameters():
            p.requires_grad_(False)
    return model, config


# ============================================================================
# Prefill Block Wrapper and Pipeline
# ============================================================================


class BlockWrapper(nn.Module):
    """Wrapper for a single block to trace independently for prefill.

    Args:
        block: A block from model.blocks[i]
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        out, _ = self.block(x, inference_params=None, padding_mask=None)
        return out


class Evo2PrefillPipeline:
    """Block-by-block Neuron prefill pipeline.

    Compiles and runs each block sequentially to avoid loading all 32 NEFFs
    simultaneously (which can exceed HBM on a single NeuronCore at LNC=2).

    Args:
        model: StripedHyena model
        config: Model config (dotdict)
        seq_len: Prefill sequence length
        compiler_args: Neuron compiler arguments
    """

    def __init__(self, model, config, seq_len=2048, compiler_args=None):
        self.model = model
        self.config = config
        self.seq_len = seq_len
        self.compiler_args = compiler_args or [
            "--auto-cast",
            "matmult",
            "--model-type=transformer",
        ]
        self._block_wrappers = []
        self._compiled = False

    def compile(self):
        """Prepare all 32 prefill block wrappers (bf16 conversion).

        Actual Neuron compilation happens lazily in forward() to avoid
        loading all 32 NEFFs into HBM simultaneously.

        Returns:
            True if all blocks prepared successfully.
        """
        self._block_wrappers = []
        for block_idx in range(NUM_LAYERS):
            wrapper = BlockWrapper(self.model.blocks[block_idx])
            wrapper.eval()

            with torch.no_grad():
                for param in wrapper.parameters():
                    if param.is_floating_point() and param.dtype != torch.bfloat16:
                        param.data = param.data.to(torch.bfloat16).contiguous()
                    elif not param.data.is_contiguous():
                        param.data = param.data.contiguous()

            self._block_wrappers.append(wrapper)

        self._compiled = True
        return True

    def forward(self, input_ids):
        """Run prefill inference, compiling and running each block sequentially.

        Each block is compiled, executed, and then its traced NEFF is released
        before moving to the next block. This keeps HBM usage bounded to a
        single block's weights + scratch at any time.

        Args:
            input_ids: (1, seq_len) int64

        Returns:
            logits: (1, seq_len, vocab_size) float32
        """
        import torch_neuronx

        assert self._compiled, "Call compile() first"

        x = self.model.embedding_layer(input_ids)
        x = x.to(dtype=torch.bfloat16)

        total_start = time.time()
        for block_idx in range(NUM_LAYERS):
            block_type = BLOCK_TYPE_MAP[block_idx]
            wrapper = self._block_wrappers[block_idx]

            logger.info(
                f"Compiling+running prefill block {block_idx} ({block_type})..."
            )
            start = time.time()

            traced = torch_neuronx.trace(
                wrapper,
                x,
                compiler_args=self.compiler_args,
            )
            x = traced(x)

            # Release traced NEFF to free HBM before compiling next block
            del traced

            elapsed = time.time() - start
            logger.info(f"  Block {block_idx}: {elapsed:.1f}s")

        total = time.time() - total_start
        logger.info(f"All {NUM_LAYERS} prefill blocks done in {total:.0f}s")

        x = self.model.norm(x)
        x = self.model.unembed(x)
        return x


# ============================================================================
# Decode Block Wrappers (Single-Sequence)
# ============================================================================


class ATTDecodeBlock(nn.Module):
    """Attention decode block with KV-cache.

    Implements single-token decode with manual RoPE, KV-cache scatter,
    and causal masked attention.

    Args:
        block: A Vortex block (from model.blocks[i]) of type ATT
        layer_idx: Block index (0-31)
        max_seqlen: Maximum sequence length for KV-cache
        num_heads: Number of attention heads
        head_dim: Dimension per head
    """

    def __init__(
        self, block, layer_idx, max_seqlen, num_heads=NUM_HEADS, head_dim=HEAD_DIM
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.max_seqlen = max_seqlen
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim

        self.pre_norm = block.pre_norm
        self.post_norm = block.post_norm
        self.mha = block.inner_mha_cls
        self.mlp = block.mlp

        rotary = self.mha.rotary_emb
        rotary._update_cos_sin_cache(
            max_seqlen, device=torch.device("cpu"), dtype=torch.bfloat16
        )
        self._rope_cos = rotary._cos_cached.clone()
        self._rope_sin = rotary._sin_cached.clone()
        self._rotary_half_dim = rotary._cos_cached.shape[-1]

    def forward(self, x, position, k_cache, v_cache):
        """Single-token decode.

        Args:
            x: (B, 1, D) current hidden state
            position: (1,) int64 position for scatter/RoPE
            k_cache: (B, max_seqlen, H, head_dim)
            v_cache: (B, max_seqlen, H, head_dim)

        Returns:
            (x_out, k_cache_updated, v_cache_updated)
        """
        B = x.shape[0]
        x_normed = self.pre_norm(x)
        qkv = self.mha.Wqkv(x_normed).reshape(B, 1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # RoPE
        pos_idx = position[0]
        cos_half = self._rope_cos[pos_idx : pos_idx + 1]
        sin_half = self._rope_sin[pos_idx : pos_idx + 1]
        cos_full = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(0)
        sin_full = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(0)

        def rotate_half(t):
            t1, t2 = t[..., : t.shape[-1] // 2], t[..., t.shape[-1] // 2 :]
            return torch.cat((-t2, t1), dim=-1)

        ro_dim = self._rotary_half_dim * 2
        q = q[..., :ro_dim] * cos_full + rotate_half(q[..., :ro_dim]) * sin_full
        k = k[..., :ro_dim] * cos_full + rotate_half(k[..., :ro_dim]) * sin_full

        # KV-cache scatter
        scatter_idx = position.view(1, 1, 1, 1).expand(
            B, 1, self.num_heads, self.head_dim
        )
        k_cache = torch.scatter(k_cache, 1, scatter_idx, k.squeeze(1).unsqueeze(1))
        v_cache = torch.scatter(v_cache, 1, scatter_idx, v.squeeze(1).unsqueeze(1))

        # Attention
        q_attn = q.permute(0, 2, 1, 3)
        k_attn = k_cache.permute(0, 2, 1, 3)
        v_attn = v_cache.permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale

        positions = torch.arange(self.max_seqlen, device=scores.device)
        causal_mask = (positions > position.unsqueeze(-1)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask, -10000.0)

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v_attn)
        context = context.permute(0, 2, 1, 3).reshape(B, 1, self.hidden_size)

        attn_out = self.mha.out_proj(context)
        x = attn_out + x
        x = self.mlp(self.post_norm(x)) + x
        return x, k_cache, v_cache


def _step_fir(u, fir_state, weight, bias, flip_filter=False, gated_bias=False):
    """Functional single-step FIR convolution.

    Uses cat-based shift register (no torch.roll or in-place ops, which
    cause Neuron compiler ICE).
    """
    weight = weight.squeeze()
    cache_size = fir_state.shape[-1]
    if flip_filter:
        weight = weight.flip(-1)
        weight = weight[..., -cache_size - 1 :].unsqueeze(0)
    else:
        weight = weight[..., : cache_size + 1].unsqueeze(0)

    u_f32 = u.float()
    weight_f32 = weight.float()
    fir_f32 = fir_state.float()
    bias_f32 = bias.float() if bias is not None else None

    h0 = weight_f32[..., -1]
    h = weight_f32[..., :-1]
    y = h0 * u_f32 + torch.sum(fir_f32 * h, dim=-1)

    if bias_f32 is not None:
        if gated_bias:
            y = y + bias_f32 * u_f32
        else:
            y = y + bias_f32

    new_fir_state = torch.cat([fir_f32[..., 1:], u_f32.unsqueeze(-1)], dim=-1)
    return y.to(u.dtype), new_fir_state.to(u.dtype)


def _interleave_split(
    z,
    do_interleave,
    column_split_hyena,
    num_attention_heads,
    hidden_size_per_attention_head,
    hidden_size,
):
    """Fused interleave + split using stride slicing.

    Avoids reshape->cat->reshape chain that causes Neuron compiler ICE
    (INTERNAL_ERROR: MemcpyElimination).
    """
    if do_interleave:
        x2 = z[:, 0::3]
        x1 = z[:, 1::3]
        v = z[:, 2::3]
    elif column_split_hyena:
        z_r = z.reshape(
            z.shape[0],
            num_attention_heads,
            3 * hidden_size_per_attention_head,
        )
        x2 = z_r[:, :, :hidden_size_per_attention_head].reshape(z.shape[0], -1)
        x1 = z_r[
            :, :, hidden_size_per_attention_head : 2 * hidden_size_per_attention_head
        ].reshape(z.shape[0], -1)
        v = z_r[:, :, 2 * hidden_size_per_attention_head :].reshape(z.shape[0], -1)
    else:
        x2, x1, v = z.split([hidden_size, hidden_size, hidden_size], dim=1)
    return x2, x1, v


class HCLDecodeBlock(nn.Module):
    """HCL (IIR/SSM) decode block with FIR + IIR state.

    State:
      fir_state: (B, 3*D, short_filter_length-1)
      iir_state: (B, D, state_size)

    Args:
        block: Vortex block of type HCL
        layer_idx: Block index
        hidden_size: Model hidden size
        state_size: IIR modal state dimension
        short_filter_length: Short FIR filter length
    """

    def __init__(
        self,
        block,
        layer_idx,
        hidden_size=HIDDEN_SIZE,
        state_size=STATE_SIZE,
        short_filter_length=SHORT_FILTER_LENGTH,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.state_size = state_size

        self.pre_norm = block.pre_norm
        self.post_norm = block.post_norm
        self.projections = block.projections
        self.out_filter_dense = block.out_filter_dense
        self.mlp = block.mlp

        cascade = block.filter
        self.short_filter_weight = cascade.short_filter_weight
        self.short_filter_bias = cascade.short_filter_bias
        self.D = cascade.D
        self.residues = cascade.residues
        self.log_poles = cascade.log_poles

        self.column_split_hyena = getattr(cascade, "column_split_hyena", True)
        self.hyena_flip_x1x2 = getattr(cascade, "hyena_flip_x1x2", False)
        self.num_attention_heads = getattr(cascade, "num_attention_heads", NUM_HEADS)
        self.hidden_size_per_attention_head = hidden_size // self.num_attention_heads
        self.do_interleave = getattr(cascade.config, "interleave", False)

    def forward(self, x, fir_state, iir_state):
        x_normed = self.pre_norm(x)
        z = self.projections(x_normed)
        if isinstance(z, tuple):
            z = z[0]
        u = z[:, -1]

        z_pre, fir_state = _step_fir(
            u,
            fir_state,
            self.short_filter_weight,
            self.short_filter_bias,
        )
        x2, x1, v = _interleave_split(
            z_pre,
            self.do_interleave,
            self.column_split_hyena,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.hidden_size,
        )
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        # IIR recurrence
        x1v = x1 * v
        poles = torch.exp(self.log_poles)[..., 0][None]
        residues = self.residues[None]
        iir_state = poles.float() * iir_state.float() + x1v.float()[..., None]
        res_state = torch.sum(residues.float() * iir_state, dim=-1)
        y = x2.float() * (res_state + self.D.float() * x1v.float())
        y = y.to(x2.dtype)
        iir_state = iir_state.to(x2.dtype)

        y = y[:, None].to(dtype=x.dtype)
        x = self.out_filter_dense(y) + x
        x = self.mlp(self.post_norm(x)) + x
        return x, fir_state, iir_state


class FIRDecodeBlock(nn.Module):
    """HCM/HCS decode block with outer FIR + inner FIR state.

    State:
      fir_state: (B, 3*D, short_filter_length-1)
      fir_inner_state: (B, D, inner_filter_length-1)

    Args:
        block: Vortex block of type HCM or HCS
        layer_idx: Block index
        hidden_size: Model hidden size
        short_filter_length: Short FIR filter length
        inner_filter_length: Inner FIR filter length (128 for HCM, 7 for HCS)
    """

    def __init__(
        self,
        block,
        layer_idx,
        hidden_size=HIDDEN_SIZE,
        short_filter_length=SHORT_FILTER_LENGTH,
        inner_filter_length=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.inner_filter_length = inner_filter_length

        self.pre_norm = block.pre_norm
        self.post_norm = block.post_norm
        self.projections = block.projections
        self.out_filter_dense = block.out_filter_dense
        self.mlp = block.mlp

        cascade = block.filter
        self.short_filter_weight = cascade.short_filter_weight
        self.short_filter_bias = cascade.short_filter_bias
        self.h = cascade.h
        self.D = cascade.D
        self.hyena_filter_groups = getattr(cascade, "hyena_filter_groups", 1)

        self.column_split_hyena = getattr(cascade, "column_split_hyena", True)
        self.hyena_flip_x1x2 = getattr(cascade, "hyena_flip_x1x2", False)
        self.num_attention_heads = getattr(cascade, "num_attention_heads", NUM_HEADS)
        self.hidden_size_per_attention_head = hidden_size // self.num_attention_heads
        self.do_interleave = getattr(cascade.config, "interleave", False)

        self.flip_filter = inner_filter_length >= 128 if inner_filter_length else False
        self.gated_bias = inner_filter_length >= 128 if inner_filter_length else False

    def forward(self, x, fir_state, fir_inner_state):
        x_normed = self.pre_norm(x)
        z = self.projections(x_normed)
        if isinstance(z, tuple):
            z = z[0]
        u = z[:, -1]

        z_pre, fir_state = _step_fir(
            u,
            fir_state,
            self.short_filter_weight,
            self.short_filter_bias,
        )
        x2, x1, v = _interleave_split(
            z_pre,
            self.do_interleave,
            self.column_split_hyena,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.hidden_size,
        )
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        h = self.h
        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)

        y, fir_inner_state = _step_fir(
            x1 * v,
            fir_inner_state,
            h,
            self.D,
            flip_filter=self.flip_filter,
            gated_bias=self.gated_bias,
        )
        y = y * x2

        y = y[:, None].to(dtype=x.dtype)
        x = self.out_filter_dense(y) + x
        x = self.mlp(self.post_norm(x)) + x
        return x, fir_state, fir_inner_state


class Evo2DecodeBlock(nn.Module):
    """Unified decode block wrapper with HBM-resident state parameters.

    Wraps ATTDecodeBlock, HCLDecodeBlock, or FIRDecodeBlock with
    nn.Parameter state tensors for input_output_aliases.

    Args:
        block: Vortex block
        block_idx: Block index (0-31)
        block_type: "ATT", "HCL", "HCM", or "HCS"
        max_seqlen: Maximum sequence length
        batch_size: Batch size for state allocation
    """

    def __init__(
        self,
        block,
        block_idx,
        block_type,
        max_seqlen,
        batch_size=1,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        state_size=STATE_SIZE,
        short_filter_length=SHORT_FILTER_LENGTH,
        inner_filter_length=None,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.block_type = block_type

        if block_type == "ATT":
            self.decode_block = ATTDecodeBlock(
                block,
                block_idx,
                max_seqlen,
                num_heads,
                head_dim,
            )
            self.k_cache = nn.Parameter(
                torch.zeros(
                    batch_size, max_seqlen, num_heads, head_dim, dtype=torch.bfloat16
                ),
                requires_grad=False,
            )
            self.v_cache = nn.Parameter(
                torch.zeros(
                    batch_size, max_seqlen, num_heads, head_dim, dtype=torch.bfloat16
                ),
                requires_grad=False,
            )
        elif block_type == "HCL":
            self.decode_block = HCLDecodeBlock(
                block,
                block_idx,
                hidden_size,
                state_size,
                short_filter_length,
            )
            self.fir_state = nn.Parameter(
                torch.zeros(
                    batch_size,
                    3 * hidden_size,
                    short_filter_length - 1,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            self.iir_state = nn.Parameter(
                torch.zeros(batch_size, hidden_size, state_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
        elif block_type in ("HCM", "HCS"):
            self.decode_block = FIRDecodeBlock(
                block,
                block_idx,
                hidden_size,
                short_filter_length,
                inner_filter_length,
            )
            self.fir_state = nn.Parameter(
                torch.zeros(
                    batch_size,
                    3 * hidden_size,
                    short_filter_length - 1,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            self.fir_inner_state = nn.Parameter(
                torch.zeros(
                    batch_size,
                    hidden_size,
                    inner_filter_length - 1,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )

    def forward(self, *args):
        if self.block_type == "ATT":
            x, position = args
            return self.decode_block(x, position, self.k_cache, self.v_cache)
        elif self.block_type == "HCL":
            (x,) = args
            return self.decode_block(x, self.fir_state, self.iir_state)
        else:
            (x,) = args
            return self.decode_block(x, self.fir_state, self.fir_inner_state)

    def get_trace_inputs(self):
        """Return example inputs for torch_neuronx.trace()."""
        test_x = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
        if self.block_type == "ATT":
            return (test_x, torch.tensor([0], dtype=torch.int64))
        return (test_x,)

    def get_state_params(self):
        """Return list of state parameters."""
        if self.block_type == "ATT":
            return [self.k_cache, self.v_cache]
        elif self.block_type == "HCL":
            return [self.fir_state, self.iir_state]
        return [self.fir_state, self.fir_inner_state]

    def get_input_output_aliases(self):
        """Build input_output_aliases dict for torch_neuronx.trace()."""
        params = self.get_state_params()
        return {params[0]: 1, params[1]: 2}


# ============================================================================
# Batched Decode Blocks (for throughput-optimized decode)
# ============================================================================


class BatchATTDecode(nn.Module):
    """Batched attention decode (lockstep position across batch)."""

    def __init__(
        self, block, layer_idx, max_seqlen, num_heads=NUM_HEADS, head_dim=HEAD_DIM
    ):
        super().__init__()
        self.max_seqlen = max_seqlen
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim

        self.pre_norm = block.pre_norm
        self.post_norm = block.post_norm
        self.mha = block.inner_mha_cls
        self.mlp = block.mlp

        rotary = self.mha.rotary_emb
        rotary._update_cos_sin_cache(
            max_seqlen,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self._rope_cos = rotary._cos_cached.clone()
        self._rope_sin = rotary._sin_cached.clone()
        self._rotary_half_dim = rotary._cos_cached.shape[-1]

    def forward(self, x, position, k_cache, v_cache):
        B = x.shape[0]
        x_normed = self.pre_norm(x)
        qkv = self.mha.Wqkv(x_normed).reshape(B, 1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        pos_idx = position[0]
        cos_half = self._rope_cos[pos_idx : pos_idx + 1]
        sin_half = self._rope_sin[pos_idx : pos_idx + 1]
        cos_full = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(0)
        sin_full = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(0)

        def rotate_half(t):
            t1, t2 = t[..., : t.shape[-1] // 2], t[..., t.shape[-1] // 2 :]
            return torch.cat((-t2, t1), dim=-1)

        ro_dim = self._rotary_half_dim * 2
        q = q[..., :ro_dim] * cos_full + rotate_half(q[..., :ro_dim]) * sin_full
        k = k[..., :ro_dim] * cos_full + rotate_half(k[..., :ro_dim]) * sin_full

        scatter_idx = position.view(1, 1, 1, 1).expand(
            B, 1, self.num_heads, self.head_dim
        )
        k_cache = torch.scatter(k_cache, 1, scatter_idx, k.squeeze(1).unsqueeze(1))
        v_cache = torch.scatter(v_cache, 1, scatter_idx, v.squeeze(1).unsqueeze(1))

        q_attn = q.permute(0, 2, 1, 3)
        k_attn = k_cache.permute(0, 2, 1, 3)
        v_attn = v_cache.permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale

        positions = torch.arange(self.max_seqlen, device=scores.device)
        causal_mask = (positions > position.unsqueeze(-1)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask, -10000.0)

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v_attn)
        context = context.permute(0, 2, 1, 3).reshape(B, 1, self.hidden_size)

        attn_out = self.mha.out_proj(context)
        x = attn_out + x
        x = self.mlp(self.post_norm(x)) + x
        return x, k_cache, v_cache


class BatchHCLDecode(nn.Module):
    """Batched HCL (IIR/SSM) decode."""

    def __init__(
        self,
        block,
        layer_idx,
        hidden_size=HIDDEN_SIZE,
        state_size=STATE_SIZE,
        short_filter_length=SHORT_FILTER_LENGTH,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.pre_norm = block.pre_norm
        self.post_norm = block.post_norm
        self.projections = block.projections
        self.out_filter_dense = block.out_filter_dense
        self.mlp = block.mlp

        cascade = block.filter
        self.short_filter_weight = cascade.short_filter_weight
        self.short_filter_bias = cascade.short_filter_bias
        self.D = cascade.D
        self.residues = cascade.residues
        self.log_poles = cascade.log_poles

        self.column_split_hyena = getattr(cascade, "column_split_hyena", True)
        self.hyena_flip_x1x2 = getattr(cascade, "hyena_flip_x1x2", False)
        self.num_attention_heads = getattr(cascade, "num_attention_heads", NUM_HEADS)
        self.hidden_size_per_attention_head = hidden_size // self.num_attention_heads
        self.do_interleave = getattr(cascade.config, "interleave", False)

    def forward(self, x, fir_state, iir_state):
        x_normed = self.pre_norm(x)
        z = self.projections(x_normed)
        if isinstance(z, tuple):
            z = z[0]
        u = z[:, -1]

        z_pre, fir_state = _step_fir(
            u,
            fir_state,
            self.short_filter_weight,
            self.short_filter_bias,
        )
        x2, x1, v = _interleave_split(
            z_pre,
            self.do_interleave,
            self.column_split_hyena,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.hidden_size,
        )
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        x1v = x1 * v
        poles = torch.exp(self.log_poles)[..., 0][None]
        residues = self.residues[None]
        iir_state = poles.float() * iir_state.float() + x1v.float()[..., None]
        res_state = torch.sum(residues.float() * iir_state, dim=-1)
        y = x2.float() * (res_state + self.D.float() * x1v.float())

        y = y[:, None].to(dtype=x.dtype)
        iir_state = iir_state.to(x.dtype)
        x = self.out_filter_dense(y) + x
        x = self.mlp(self.post_norm(x)) + x
        return x, fir_state, iir_state


class BatchFIRDecode(nn.Module):
    """Batched HCM/HCS decode."""

    def __init__(
        self,
        block,
        layer_idx,
        hidden_size=HIDDEN_SIZE,
        short_filter_length=SHORT_FILTER_LENGTH,
        inner_filter_length=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.inner_filter_length = inner_filter_length

        self.pre_norm = block.pre_norm
        self.post_norm = block.post_norm
        self.projections = block.projections
        self.out_filter_dense = block.out_filter_dense
        self.mlp = block.mlp

        cascade = block.filter
        self.short_filter_weight = cascade.short_filter_weight
        self.short_filter_bias = cascade.short_filter_bias
        self.h = cascade.h
        self.D = cascade.D
        self.hyena_filter_groups = getattr(cascade, "hyena_filter_groups", 1)

        self.column_split_hyena = getattr(cascade, "column_split_hyena", True)
        self.hyena_flip_x1x2 = getattr(cascade, "hyena_flip_x1x2", False)
        self.num_attention_heads = getattr(cascade, "num_attention_heads", NUM_HEADS)
        self.hidden_size_per_attention_head = hidden_size // self.num_attention_heads
        self.do_interleave = getattr(cascade.config, "interleave", False)

        self.flip_filter = inner_filter_length >= 128 if inner_filter_length else False
        self.gated_bias = inner_filter_length >= 128 if inner_filter_length else False

    def forward(self, x, fir_state, fir_inner_state):
        x_normed = self.pre_norm(x)
        z = self.projections(x_normed)
        if isinstance(z, tuple):
            z = z[0]
        u = z[:, -1]

        z_pre, fir_state = _step_fir(
            u,
            fir_state,
            self.short_filter_weight,
            self.short_filter_bias,
        )
        x2, x1, v = _interleave_split(
            z_pre,
            self.do_interleave,
            self.column_split_hyena,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.hidden_size,
        )
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        h = self.h
        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)

        y, fir_inner_state = _step_fir(
            x1 * v,
            fir_inner_state,
            h,
            self.D,
            flip_filter=self.flip_filter,
            gated_bias=self.gated_bias,
        )
        y = y * x2

        y = y[:, None].to(dtype=x.dtype)
        x = self.out_filter_dense(y) + x
        x = self.mlp(self.post_norm(x)) + x
        return x, fir_state, fir_inner_state


class BatchMegaDecode(nn.Module):
    """All 32 decode blocks in one module with batch dimension.

    Single mega-NEFF for throughput-optimized batched decode. All
    sequences in batch share the same position (lockstep decode).
    State tensors are registered as nn.Parameters for input_output_aliases.

    Args:
        blocks: model.blocks (nn.ModuleList of 32 blocks)
        max_seqlen: Maximum sequence length for KV-cache
        batch_size: Decode batch size
    """

    def __init__(self, blocks, max_seqlen, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.decode_blocks = nn.ModuleList()
        self.block_types = []
        self.state_param_names = []

        for i in range(NUM_LAYERS):
            block_type = BLOCK_TYPE_MAP[i]
            self.block_types.append(block_type)
            block = blocks[i]

            if block_type == "ATT":
                db = BatchATTDecode(block, i, max_seqlen, NUM_HEADS, HEAD_DIM)
                k_name = f"k_cache_{i}"
                v_name = f"v_cache_{i}"
                self.register_parameter(
                    k_name,
                    nn.Parameter(
                        torch.zeros(
                            batch_size,
                            max_seqlen,
                            NUM_HEADS,
                            HEAD_DIM,
                            dtype=torch.bfloat16,
                        ),
                        requires_grad=False,
                    ),
                )
                self.register_parameter(
                    v_name,
                    nn.Parameter(
                        torch.zeros(
                            batch_size,
                            max_seqlen,
                            NUM_HEADS,
                            HEAD_DIM,
                            dtype=torch.bfloat16,
                        ),
                        requires_grad=False,
                    ),
                )
                self.state_param_names.extend([k_name, v_name])

            elif block_type == "HCL":
                db = BatchHCLDecode(
                    block, i, HIDDEN_SIZE, STATE_SIZE, SHORT_FILTER_LENGTH
                )
                fir_name = f"fir_state_{i}"
                iir_name = f"iir_state_{i}"
                self.register_parameter(
                    fir_name,
                    nn.Parameter(
                        torch.zeros(
                            batch_size,
                            3 * HIDDEN_SIZE,
                            SHORT_FILTER_LENGTH - 1,
                            dtype=torch.bfloat16,
                        ),
                        requires_grad=False,
                    ),
                )
                self.register_parameter(
                    iir_name,
                    nn.Parameter(
                        torch.zeros(
                            batch_size, HIDDEN_SIZE, STATE_SIZE, dtype=torch.bfloat16
                        ),
                        requires_grad=False,
                    ),
                )
                self.state_param_names.extend([fir_name, iir_name])

            elif block_type in ("HCM", "HCS"):
                inner_fl = (
                    HCM_FILTER_LENGTH if block_type == "HCM" else HCS_FILTER_LENGTH
                )
                db = BatchFIRDecode(
                    block, i, HIDDEN_SIZE, SHORT_FILTER_LENGTH, inner_fl
                )
                fir_name = f"fir_state_{i}"
                inner_name = f"fir_inner_{i}"
                self.register_parameter(
                    fir_name,
                    nn.Parameter(
                        torch.zeros(
                            batch_size,
                            3 * HIDDEN_SIZE,
                            SHORT_FILTER_LENGTH - 1,
                            dtype=torch.bfloat16,
                        ),
                        requires_grad=False,
                    ),
                )
                self.register_parameter(
                    inner_name,
                    nn.Parameter(
                        torch.zeros(
                            batch_size, HIDDEN_SIZE, inner_fl - 1, dtype=torch.bfloat16
                        ),
                        requires_grad=False,
                    ),
                )
                self.state_param_names.extend([fir_name, inner_name])

            self.decode_blocks.append(db)

    def forward(self, x, position):
        """Forward pass through all 32 decode blocks.

        Args:
            x: (B, 1, D) bf16
            position: (1,) int64

        Returns:
            (x_out, *updated_states) where updated_states has 64 tensors
            (2 per block).
        """
        updated_states = []
        for i in range(NUM_LAYERS):
            block_type = self.block_types[i]
            db = self.decode_blocks[i]

            if block_type == "ATT":
                k = getattr(self, f"k_cache_{i}")
                v = getattr(self, f"v_cache_{i}")
                x, k_new, v_new = db(x, position, k, v)
                updated_states.extend([k_new, v_new])
            elif block_type == "HCL":
                fir = getattr(self, f"fir_state_{i}")
                iir = getattr(self, f"iir_state_{i}")
                x, fir_new, iir_new = db(x, fir, iir)
                updated_states.extend([fir_new, iir_new])
            else:
                fir = getattr(self, f"fir_state_{i}")
                inner = getattr(self, f"fir_inner_{i}")
                x, fir_new, inner_new = db(x, fir, inner)
                updated_states.extend([fir_new, inner_new])

        return (x, *updated_states)

    def get_input_output_aliases(self):
        """Build input_output_aliases for all 64 state parameters."""
        aliases = {}
        for idx, name in enumerate(self.state_param_names):
            param = getattr(self, name)
            aliases[param] = idx + 1  # output index 0 is x_out
        return aliases


# ============================================================================
# High-Level Pipeline
# ============================================================================


class Evo2NeuronPipeline:
    """End-to-end pipeline for Evo2 inference on Neuron.

    Handles model loading, source patching, compilation of prefill and
    decode NEFFs, state management, and generation.

    Two decode strategies:
      - Per-block: 32 independent decode NEFFs (low throughput, simple)
      - Batched mega-NEFF: Single NEFF with all 32 blocks + batch dim
        (high throughput, recommended)

    Usage::

        from modeling_evo2 import Evo2NeuronPipeline

        pipeline = Evo2NeuronPipeline(
            evo2_src_path="/home/ubuntu/evo2-repo",
            vortex_src_path="/home/ubuntu/vortex-repo",
            weights_path="~/.cache/.../evo2_7b_base.pt",
        )
        pipeline.load_model()
        pipeline.compile_prefill()
        pipeline.compile_decode(batch_size=32)
        tokens = pipeline.generate(input_ids, n_tokens=100)

    Args:
        evo2_src_path: Path to evo2 repository
        vortex_src_path: Path to vortex repository
        weights_path: Path to model weights file
        prefill_seq_len: Prefill sequence length (max 2048)
        max_gen_len: Maximum generation length
        compiler_args: Neuron compiler arguments
    """

    def __init__(
        self,
        evo2_src_path: str,
        vortex_src_path: str,
        weights_path: Optional[str] = None,
        prefill_seq_len: int = 2048,
        max_gen_len: int = 256,
        compiler_args: Optional[List[str]] = None,
    ):
        self.evo2_src_path = evo2_src_path
        self.vortex_src_path = vortex_src_path
        self.weights_path = weights_path
        self.prefill_seq_len = prefill_seq_len
        self.max_seqlen = prefill_seq_len + max_gen_len
        self.compiler_args = compiler_args or [
            "--auto-cast",
            "matmult",
            "--model-type=transformer",
        ]

        self.model = None
        self.config = None
        self.prefill_pipeline = None

        # Decode state
        self.mega_decode = None
        self.traced_mega = None
        self.decode_batch_size = None

        self.compile_times: Dict[str, float] = {}

    def load_model(self) -> None:
        """Load and configure Evo2 model.

        Installs CUDA mocks, adds repos to path, applies patches,
        loads config and weights.
        """
        _install_cuda_mocks()

        # Add repos to path
        if self.evo2_src_path not in sys.path:
            sys.path.insert(0, self.evo2_src_path)
        if self.vortex_src_path not in sys.path:
            sys.path.insert(0, self.vortex_src_path)

        patch_vortex_for_neuron()
        config = load_config()
        self.model, self.config = build_model(config, self.weights_path)
        install_neuron_fft_patch(self.model, self.prefill_seq_len)
        logger.info("Model loaded and patched for Neuron")

    def compile_prefill(self) -> bool:
        """Compile 32 prefill block NEFFs.

        Returns:
            True if all blocks compiled successfully.
        """
        assert self.model is not None, "Call load_model() first"
        self.prefill_pipeline = Evo2PrefillPipeline(
            self.model,
            self.config,
            self.prefill_seq_len,
            self.compiler_args,
        )
        t0 = time.time()
        ok = self.prefill_pipeline.compile()
        self.compile_times["prefill"] = time.time() - t0
        return ok

    def compile_decode(self, batch_size: int = 32) -> bool:
        """Compile batched decode mega-NEFF.

        IMPORTANT: Run compile_prefill() first. The prefill blocks share
        sub-modules with the model, and bf16 conversion during decode
        compilation mutates them. Always prefill before building decode.

        Args:
            batch_size: Number of sequences per decode step.

        Returns:
            True if compilation succeeded.
        """
        import torch_neuronx

        assert self.model is not None, "Call load_model() first"
        self.decode_batch_size = batch_size

        logger.info(f"Building BatchMegaDecode (BS={batch_size})...")
        self.mega_decode = BatchMegaDecode(
            self.model.blocks,
            self.max_seqlen,
            batch_size,
        )
        self.mega_decode.eval()

        # Convert all parameters to bf16 + contiguous
        with torch.no_grad():
            for param in self.mega_decode.parameters():
                if param.is_floating_point() and param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16).contiguous()
                elif not param.data.is_contiguous():
                    param.data = param.data.contiguous()

        aliases = self.mega_decode.get_input_output_aliases()
        test_x = torch.randn(batch_size, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
        test_pos = torch.tensor([0], dtype=torch.int64)

        logger.info(f"Compiling mega-NEFF ({len(aliases)} state aliases)...")
        t0 = time.time()
        try:
            self.traced_mega = torch_neuronx.trace(
                self.mega_decode,
                (test_x, test_pos),
                compiler_args=self.compiler_args,
                input_output_aliases=aliases,
                inline_weights_to_neff=False,
            )
            elapsed = time.time() - t0
            self.compile_times["decode"] = elapsed
            logger.info(f"Decode mega-NEFF compiled in {elapsed:.0f}s")
            return True
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"Decode compilation failed after {elapsed:.0f}s: {e}")
            return False

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run prefill and return logits.

        Args:
            input_ids: (1, prefill_seq_len) int64

        Returns:
            logits: (1, prefill_seq_len, vocab_size)
        """
        assert self.prefill_pipeline is not None, "Call compile_prefill() first"
        return self.prefill_pipeline.forward(input_ids)

    def initialize_decode_state(self, input_ids: torch.Tensor) -> Dict:
        """Initialize decode state by running CPU prefill with inference_params.

        Populates KV-caches and SSM states from the prefill sequence.
        Returns the saved state dict for copying into decode blocks.

        Args:
            input_ids: (1, prefill_seq_len) int64

        Returns:
            Dict mapping block_idx to state tensors (deep-copied).
        """
        assert self.model is not None, "Call load_model() first"

        inference_params_dict = self.model.initialize_inference_params(
            max_seqlen=self.max_seqlen,
        )

        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            x = self.model.embedding_layer(input_ids)
            for block_idx, block in enumerate(self.model.blocks):
                block_name = self.model.block_idx_to_name(block_idx)
                params = inference_params_dict[block_name]
                x, _ = block(x, inference_params=params)

        # Deep-copy state to prevent mutation from bf16 conversion
        import copy

        saved_state = {}
        for block_idx in range(NUM_LAYERS):
            block_type = BLOCK_TYPE_MAP[block_idx]
            if block_type == "ATT":
                kv = inference_params_dict["mha"].key_value_memory_dict[block_idx]
                sl = self.prefill_seq_len
                saved_state[block_idx] = {
                    "k": kv[:, :sl, 0, :, :].clone().to(torch.bfloat16),
                    "v": kv[:, :sl, 1, :, :].clone().to(torch.bfloat16),
                }
            elif block_type == "HCL":
                saved_state[block_idx] = {
                    "fir": inference_params_dict["hcl"]
                    .fir_state_dict[block_idx]
                    .clone()
                    .to(torch.bfloat16),
                    "iir": inference_params_dict["hcl"]
                    .state_dict[block_idx]
                    .clone()
                    .to(torch.bfloat16),
                }
            elif block_type == "HCM":
                saved_state[block_idx] = {
                    "fir": inference_params_dict["hcm"]
                    .fir_state_dict[block_idx]
                    .clone()
                    .to(torch.bfloat16),
                    "inner": inference_params_dict["hcm"]
                    .fir_inner_state_dict[block_idx]
                    .clone()
                    .to(torch.bfloat16),
                }
            elif block_type == "HCS":
                saved_state[block_idx] = {
                    "fir": inference_params_dict["hcs"]
                    .fir_state_dict[block_idx]
                    .clone()
                    .to(torch.bfloat16),
                    "inner": inference_params_dict["hcs"]
                    .fir_inner_state_dict[block_idx]
                    .clone()
                    .to(torch.bfloat16),
                }

        logger.info("Decode state initialized from CPU prefill")
        return saved_state

    def copy_state_to_mega_decode(
        self,
        saved_state: Dict,
        batch_size: Optional[int] = None,
    ) -> None:
        """Copy prefill state into BatchMegaDecode parameters.

        Replicates single-sequence state across all batch elements.

        Args:
            saved_state: From initialize_decode_state()
            batch_size: Override batch size (default: self.decode_batch_size)
        """
        import torch_neuronx

        assert self.mega_decode is not None, "Call compile_decode() first"
        bs = batch_size or self.decode_batch_size

        with torch.no_grad():
            for block_idx in range(NUM_LAYERS):
                block_type = BLOCK_TYPE_MAP[block_idx]
                state = saved_state[block_idx]

                if block_type == "ATT":
                    k_param = getattr(self.mega_decode, f"k_cache_{block_idx}")
                    v_param = getattr(self.mega_decode, f"v_cache_{block_idx}")
                    sl = state["k"].shape[1]
                    k_param.data.zero_()
                    v_param.data.zero_()
                    k_param.data[:, :sl] = state["k"].expand(bs, -1, -1, -1)
                    v_param.data[:, :sl] = state["v"].expand(bs, -1, -1, -1)
                elif block_type == "HCL":
                    fir_param = getattr(self.mega_decode, f"fir_state_{block_idx}")
                    iir_param = getattr(self.mega_decode, f"iir_state_{block_idx}")
                    fir_param.data[:] = state["fir"].expand(bs, -1, -1)
                    iir_param.data[:] = state["iir"].expand(bs, -1, -1)
                else:
                    fir_param = getattr(self.mega_decode, f"fir_state_{block_idx}")
                    inner_param = getattr(self.mega_decode, f"fir_inner_{block_idx}")
                    fir_param.data[:] = state["fir"].expand(bs, -1, -1)
                    inner_param.data[:] = state["inner"].expand(bs, -1, -1)

        # Push state to HBM
        torch_neuronx.replace_weights(self.traced_mega, self.mega_decode)
        logger.info(f"State copied to mega-NEFF (BS={bs})")

    def decode_step(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """Run one batched decode step.

        Args:
            x: (B, 1, D) bf16 input hidden states
            position: Current sequence position

        Returns:
            x_out: (B, 1, D) bf16 output hidden states
        """
        import torch_neuronx

        pos = torch.tensor([position], dtype=torch.int64)
        outputs = self.traced_mega(x, pos)
        x_out = outputs[0]

        # Copy updated state back and push to HBM
        for idx, name in enumerate(self.mega_decode.state_param_names):
            getattr(self.mega_decode, name).data.copy_(outputs[idx + 1])
        torch_neuronx.replace_weights(self.traced_mega, self.mega_decode)

        return x_out

    def warmup_decode(self, n_passes: int = 3) -> bool:
        """Run warmup passes to fully initialize HBM allocations.

        Prevents nondeterministic NaN (especially block 24 ATT).

        Args:
            n_passes: Number of warmup passes

        Returns:
            True if no NaN detected after warmup.
        """
        import torch_neuronx

        bs = self.decode_batch_size
        for i in range(n_passes):
            test_x = torch.randn(bs, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
            test_pos = torch.tensor([0], dtype=torch.int64)
            outputs = self.traced_mega(test_x, test_pos)
            has_nan = torch.isnan(outputs[0]).any().item()
            logger.info(f"  Warmup {i}: NaN={has_nan}")

        # Verify pass
        test_x = torch.randn(bs, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
        outputs = self.traced_mega(test_x, torch.tensor([0], dtype=torch.int64))
        final_nan = torch.isnan(outputs[0]).any().item()
        if final_nan:
            logger.warning("NaN detected after warmup - retrying")
        return not final_nan

    def generate(
        self,
        input_ids: torch.Tensor,
        n_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 1,
    ) -> List[int]:
        """Full generation: prefill + decode.

        IMPORTANT: This method uses the batched decode path. All batch
        elements produce identical output (replicated from same prefill).
        Only batch element 0 is used for token selection.

        Args:
            input_ids: (1, prefill_seq_len) int64
            n_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 for greedy with top_k=1)
            top_k: Top-k sampling (1 = greedy)

        Returns:
            List of generated token IDs
        """
        # Prefill
        logits = self.prefill(input_ids)
        saved_state = self.initialize_decode_state(input_ids)
        self.copy_state_to_mega_decode(saved_state)
        self.warmup_decode()
        self.copy_state_to_mega_decode(saved_state)  # Restore after warmup

        # First token
        next_logits = logits[:, -1, :]
        if temperature > 0:
            next_logits = next_logits / temperature
        next_token = (
            next_logits.argmax(-1).item()
            if top_k == 1
            else self._sample_topk(next_logits, top_k)
        )

        generated = [next_token]
        position = input_ids.shape[1]
        bs = self.decode_batch_size

        # Decode loop
        gen_start = time.time()
        for step in range(n_tokens - 1):
            token_ids = torch.tensor([[next_token]], dtype=torch.long)
            x = self.model.embedding_layer(token_ids).to(torch.bfloat16)
            x = x.expand(bs, -1, -1)  # Replicate across batch

            x_out = self.decode_step(x, position)

            # Unembed from batch element 0
            logits_step = self.model.unembed(self.model.norm(x_out[0:1]))
            next_logits = logits_step[:, -1, :]
            if temperature > 0:
                next_logits = next_logits / temperature
            next_token = (
                next_logits.argmax(-1).item()
                if top_k == 1
                else self._sample_topk(next_logits, top_k)
            )

            generated.append(next_token)
            position += 1

        gen_time = time.time() - gen_start
        tok_s = len(generated) / gen_time if gen_time > 0 else 0
        logger.info(
            f"Generated {len(generated)} tokens in {gen_time:.1f}s ({tok_s:.1f} tok/s)"
        )
        return generated

    @staticmethod
    def _sample_topk(logits, top_k):
        topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        return topk_indices[0, idx].item()
