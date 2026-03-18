"""
NxDI LTX-2 Transformer Model
=============================
Neuron-optimized implementation of the LTX-2 19B DiT audio-video
diffusion transformer. Follows the NxDI Flux pattern for diffusion models.

Architecture:
  - 48 LTX-2 joint transformer blocks (video + audio dual-stream)
  - TP-sharded attention (ColumnParallelLinear/RowParallelLinear)
  - DistributedRMSNorm for QK-norm (all-reduce across TP ranks)
  - SPMDRank-based RoPE slicing (critical fix for TP>1)
  - NKI flash attention for video self-attn (seq >= 512), BMM fallback otherwise
  - Batched CFG (BS=2): unconditional + conditional passes in a single forward call

The model takes 22 preprocessed tensor inputs (proj_in, time_embed,
RoPE, caption_projection, attention masks all computed on CPU) and
returns (video_output, audio_output). All inputs have batch_size=2
when CFG is active.

Usage:
  See application.py for the high-level NeuronLTX2Application class.
"""

import logging
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# These imports are only available on Neuron instances
try:
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
        SPMDRank,
    )
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_size,
    )
    from neuronx_distributed.parallel_layers.utils import (
        set_tensor_model_parallel_attributes,
    )
    import neuronx_distributed.trace.trace as _nxd_trace

    from neuronx_distributed_inference.models.application_base import (
        NeuronApplicationBase,
    )
    from neuronx_distributed_inference.models.config import (
        InferenceConfig,
        NeuronConfig,
    )
    from neuronx_distributed_inference.models.model_wrapper import (
        BaseModelInstance,
        ModelWrapper,
    )

    NEURON_AVAILABLE = True
except ImportError:
    NEURON_AVAILABLE = False


# ── NKI Flash Attention + BMM-based SDPA replacement ────────────────────────
_sdpa_replaced = False
_sdpa_original = None

# Minimum sequence length for NKI flash attention kernel eligibility
_NKI_FLASH_MIN_SEQ = 512


def _try_load_nki_cte():
    """Try to load the nkilib attention_cte kernel and build a 4D wrapper.

    The attention_cte kernel is a pure-NKI flash attention implementation with
    3-deep software pipelining and LNC2-aware sharding. It operates on 3D
    tensors (B*H, seq, d) but we wrap it to accept 4D (B, H, S, D) for
    drop-in compatibility with the ISA kernel interface.

    Integration uses the nki 0.2.0 public API:
      1. peel_decorations() strips the @nki.jit(mode='auto') decorator
      2. nki.jit(mode='torchxla') redecorates for PyTorch XLA tensor support
      3. Integer grid (2,) for LNC=2 sharding (nki 0.2.0 uses int grids)

    Returns a callable matching scaled_dot_product_attention_kernel's 4D interface,
    or None if nkilib is not available.
    """
    try:
        import nki
        from nkilib.core.attention.attention_cte import attention_cte
        from neuronx_distributed_inference.utils.decorator_peeling import (
            peel_decorations,
        )

        # Peel @nki.jit(mode='auto') and redecorate with mode='torchxla'
        # for PyTorch tensor compatibility during torch_neuronx.trace().
        raw_func = peel_decorations(attention_cte)
        decorated_cte = nki.jit(
            raw_func,
            mode="torchxla",
            platform_target="trn2",
        )

        logger.info("nkilib attention_cte kernel loaded (nki 0.2.0 torchxla)")

        def attention_cte_4d(Q, K, V, is_causal=False, scale=None):
            """4D wrapper for attention_cte: (B, H, S, D) -> (B, H, S, D).

            Reshapes 4D tensors to 3D (B*H, seq, d) with tp_q=True layout,
            calls the nkilib CTE kernel, and reshapes back.
            """
            bsz, num_heads, q_len, head_dim = Q.shape
            k_len = K.shape[2]

            if scale is None:
                scale = 1.0 / math.sqrt(head_dim)

            # Reshape to 3D: (B*H, seq, d) — tp_q=True, tp_k=True layout
            Q_3d = Q.reshape(bsz * num_heads, q_len, head_dim).contiguous()
            K_3d = K.reshape(bsz * num_heads, k_len, head_dim).contiguous()
            V_3d = V.reshape(bsz * num_heads, k_len, head_dim).contiguous()

            # Integer grid (2,) for LNC=2 sharding (nki 0.2.0 API)
            out = decorated_cte[(2,)](
                q=Q_3d,
                k=K_3d,
                v=V_3d,
                scale=scale,
                causal_mask=is_causal,
                tp_q=True,
                tp_k=True,
                tp_out=False,
            )

            return out.reshape(bsz, num_heads, q_len, head_dim)

        return attention_cte_4d

    except ImportError:
        logger.info("nkilib not available, will try ISA kernel fallback")
        return None
    except Exception as e:
        logger.info(f"nkilib attention_cte load failed: {e}")
        return None


def _try_load_nki_flash():
    """Try to load the NKI flash attention kernel from the SDK.

    Returns the kernel function or None if unavailable.
    """
    try:
        from neuronx_distributed_inference.experimental.functional.attention.causal_attention_functions import (
            scaled_dot_product_attention_kernel,
        )

        return scaled_dot_product_attention_kernel
    except ImportError:
        logger.warning("NKI flash attention kernel not available, using BMM fallback")
        return None


# ── NKI Cross-Attention Kernel (masked, for attn2/audio_attn2) ──────────────
def _try_load_nki_cross_attn():
    """Load the NKI cross-attention kernel from nki_cross_attention_kernel.py.

    The kernel is in a separate file so that nki.language and nki.isa are
    module-level imports visible to the NKI AST tracer (which inspects
    the kernel function's __globals__). Defining the kernel inline as a
    closure causes 'nl.ndarray not found' during compilation.

    Returns the @nki.jit kernel callable or None if NKI is unavailable.
    """
    try:
        from nki_cross_attention_kernel import cross_attn_kernel

        logger.info("NKI cross-attention kernel loaded (ISA-only, mode=torchxla)")
        return cross_attn_kernel
    except ImportError:
        logger.info("NKI not available for cross-attention kernel")
        return None
    except Exception as e:
        logger.info(f"NKI cross-attention kernel load failed: {e}")
        return None


def replace_sdpa_with_bmm():
    """Replace F.scaled_dot_product_attention with NKI flash + BMM hybrid.

    Kernel priority (auto mode):
      1. NxDI ISA kernel (AttentionMMSoftmaxMMWithoutSwap) — fastest compile
      2. nkilib attention_cte (pure NKI, 3-deep pipelined, LNC2-aware)
      3. BMM fallback (explicit batched matmul + softmax)

    For attention with no mask, Q.seq >= 512, and 4D input on Neuron:
      Uses the best available NKI flash attention kernel.
      This covers:
        - Video self-attention (attn1): Q.seq == K.seq == 6144
        - Audio-to-video cross-modal attention (a2v): Q.seq=6144, K.seq=121
      The kernel supports Q.seq != K.seq for non-causal attention.

    For attention with masks or Q.seq < 512:
      Falls back to explicit BMM + softmax.

    Falls back to original SDPA on CPU.
    """
    global _sdpa_replaced, _sdpa_original
    if _sdpa_replaced:
        return _sdpa_original
    _sdpa_original = torch.nn.functional.scaled_dot_product_attention

    # Try ISA kernel first (fastest compile), then nkilib CTE.
    # Set LTX2_FLASH_KERNEL=isa to force ISA, or LTX2_FLASH_KERNEL=cte to force nkilib CTE.
    kernel_pref = os.environ.get("LTX2_FLASH_KERNEL", "auto").lower()

    _nki_flash_kernel = None
    if kernel_pref in ("auto", "isa"):
        _nki_flash_kernel = _try_load_nki_flash()
        if _nki_flash_kernel is not None:
            logger.info("Using NxDI ISA kernel for flash attention")
    if _nki_flash_kernel is None and kernel_pref in ("auto", "cte"):
        _nki_flash_kernel = _try_load_nki_cte()
        if _nki_flash_kernel is not None:
            logger.info("Using nkilib attention_cte for flash attention")
    if _nki_flash_kernel is None:
        logger.warning("No NKI flash kernel available, all attention uses BMM")

    # Try to load NKI cross-attention kernel for masked attention (attn2).
    # Set LTX2_CROSS_ATTN_KERNEL=off to disable.
    _nki_cross_attn = None
    cross_attn_pref = os.environ.get("LTX2_CROSS_ATTN_KERNEL", "auto").lower()
    if cross_attn_pref != "off":
        _nki_cross_attn = _try_load_nki_cross_attn()
        if _nki_cross_attn is not None:
            logger.info("Using NKI cross-attention kernel for masked attention")
        else:
            logger.info(
                "NKI cross-attention kernel not available, masked attn uses BMM"
            )

    def neuron_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        # CPU fallback (for text encoder, preprocessing)
        if query.device.type == "cpu":
            return _sdpa_original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        # ── NKI Flash Attention path ────────────────────────────────────
        # Eligible when: 4D input, no mask, Q.seq >= 512
        # The kernel supports Q.seq != K.seq for non-causal attention.
        # This covers:
        #   - Video self-attention (attn1): Q=6144, K=6144 — majority of attn compute
        #   - Audio-to-video cross-modal (a2v): Q=6144, K=121
        # Cross-attention with masks (attn2, audio_attn2) uses BMM fallback.
        # Audio attention with Q.seq < 512 (audio_attn1, v2a) uses BMM fallback.
        if (
            _nki_flash_kernel is not None
            and attn_mask is None
            and len(query.shape) == 4
            and query.shape[2] >= _NKI_FLASH_MIN_SEQ
        ):
            if scale is None:
                scale = 1.0 / math.sqrt(query.shape[-1])
            return _nki_flash_kernel(
                query, key, value, is_causal=False, scale=float(scale)
            )

        # ── NKI Cross-Attention path (masked, K_seq=1024) ──────────────
        # Eligible when: 4D input, has mask, K_seq=1024, d=128, Q_seq >= 512
        # This covers: attn2 (text cross-attention), audio_attn2
        if (
            _nki_cross_attn is not None
            and attn_mask is not None
            and len(query.shape) == 4
            and query.shape[-1] == 128
            and key.shape[2] == 1024
            and query.shape[2] >= _NKI_FLASH_MIN_SEQ
            and query.shape[2] % 128 == 0
        ):
            b, h, sq, d_head = query.shape
            k_seq = key.shape[2]
            if scale is None:
                scale = 1.0 / math.sqrt(d_head)
            # Pre-scale Q, reshape to 3D
            Q_3d = (query * scale).reshape(b * h, sq, d_head).contiguous()
            K_3d = key.reshape(b * h, k_seq, d_head).contiguous()
            V_3d = value.reshape(b * h, k_seq, d_head).contiguous()
            # Reshape mask to [B*H, K_seq] (2D flat for NKI kernel)
            if attn_mask.ndim == 4:
                mask_3d = attn_mask.reshape(
                    b * h, attn_mask.shape[-2], attn_mask.shape[-1]
                )
            else:
                mask_3d = attn_mask
            mask_3d = mask_3d.float().contiguous()
            # NKI kernel expects mask as [B*H, 1024] (2D flat)
            # Original mask is [B*H, 1, K_seq=1024] -> squeeze to [B*H, 1024]
            mask_2d = mask_3d.reshape(b * h, k_seq).contiguous()
            out_3d = _nki_cross_attn(Q_3d, K_3d, V_3d, mask_2d)
            # Force XLA to keep ALL input tensors in the compiled graph.
            # XLA's dead code elimination can drop operands of opaque NKI
            # custom ops because it can't see the data flow inside them.
            # We create tiny dependencies on each input by extracting a
            # scalar value that rounds to 0 in bf16, preventing DCE while
            # not affecting numerical output.
            mask_dep = (mask_2d[:, :1].unsqueeze(-1) / 1e20).to(out_3d.dtype)
            k_dep = (K_3d[:, :1, :1] / 1e20).to(out_3d.dtype)
            v_dep = (V_3d[:, :1, :1] / 1e20).to(out_3d.dtype)
            out_3d = out_3d + mask_dep + k_dep + v_dep
            return out_3d.reshape(b, h, sq, d_head)

        # ── BMM fallback path ───────────────────────────────────────────
        d = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)
        orig_shape = None
        if len(query.shape) == 4:
            orig_shape = query.shape
            b, h, sq, d_head = query.shape
            query = query.reshape(b * h, sq, d_head)
            key = key.reshape(b * h, -1, d_head)
            value = value.reshape(b * h, -1, d_head)
            if attn_mask is not None and attn_mask.ndim == 4:
                attn_mask = attn_mask.reshape(
                    b * h, attn_mask.shape[-2], attn_mask.shape[-1]
                )
            elif attn_mask is not None and attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask is not None and attn_mask.ndim == 3:
                if attn_mask.shape[0] == orig_shape[0]:
                    attn_mask = (
                        attn_mask.unsqueeze(1)
                        .expand(orig_shape[0], orig_shape[1], -1, -1)
                        .reshape(
                            orig_shape[0] * orig_shape[1],
                            attn_mask.shape[-2],
                            attn_mask.shape[-1],
                        )
                    )
        scores = torch.bmm(query, key.transpose(-1, -2)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        probs = scores.softmax(dim=-1)
        out = torch.bmm(probs, value)
        if orig_shape is not None:
            out = out.reshape(orig_shape[0], orig_shape[1], -1, orig_shape[3])
        return out

    torch.nn.functional.scaled_dot_product_attention = neuron_sdpa
    _sdpa_replaced = True
    return _sdpa_original


# ── DistributedRMSNorm ──────────────────────────────────────────────────────
class DistributedRMSNorm(nn.Module):
    """RMSNorm with all-reduce for global variance computation across TP ranks.

    Standard RMSNorm on a TP-sharded hidden dimension only sees the local shard.
    This version computes sum-of-squares locally, all-reduces across ranks, then
    normalizes with the global RMS. Essential for QK-norm accuracy in TP>1.

    The all-reduce in this norm is NOT redundant — removing it (LocalRMSNorm
    experiment) made quality significantly worse.
    """

    def __init__(self, normalized_shape, eps=1e-5, tp_size=4, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        self.eps = eps
        self.tp_size = tp_size
        self.local_dim = normalized_shape

        if NEURON_AVAILABLE:
            set_tensor_model_parallel_attributes(
                self.weight, is_parallel=True, dim=0, stride=1, num_partitions=tp_size
            )

    def forward(self, hidden_states):
        hidden_states_f32 = hidden_states.to(torch.float32)
        local_sum_sq = hidden_states_f32.pow(2).sum(dim=-1, keepdim=True)
        import torch_xla.core.xla_model as xm

        global_sum_sq = xm.all_reduce(xm.REDUCE_SUM, local_sum_sq)
        global_dim = self.local_dim * self.tp_size
        rms = torch.rsqrt(global_sum_sq / global_dim + self.eps)
        hidden_states = hidden_states_f32 * rms
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


# Register DistributedRMSNorm as a supported sharded module for NxD tracing
if NEURON_AVAILABLE:
    _nxd_trace.__SUPPORTED_SHARDED_MODULES = (
        *_nxd_trace.__SUPPORTED_SHARDED_MODULES,
        DistributedRMSNorm,
    )


# ── NeuronLTX2TransformerBackbone ───────────────────────────────────────────
class NeuronLTX2TransformerBackbone(nn.Module):
    """The core LTX-2 DiT transformer backbone for Neuron.

    Contains the 48 transformer blocks, output normalization, and projection layers.
    Takes 22 preprocessed tensor inputs (all preprocessing done on CPU) and returns
    (video_output, audio_output).

    This corresponds to the FullTransformerWrapper from the standalone scripts,
    wrapped in NxDI conventions.

    Key features:
    - SPMDRank for correct per-rank RoPE slicing in TP>1
    - DistributedRMSNorm for QK-norm (all-reduce across ranks)
    - BMM-based SDPA with NKI flash attention for video self-attn (replaces torch SDPA for Neuron)
    """

    def __init__(self, config):
        """Initialize from InferenceConfig.

        If config has an `hf_config_dict` attribute (set during config creation),
        the transformer is automatically built from the diffusers model with TP sharding.
        Otherwise creates an empty backbone (for use with from_diffusers()).
        """
        super().__init__()
        self.config = config
        self.tp_degree = config.neuron_config.tp_degree

        # Check if we have the HF config dict to auto-build
        hf_config_dict = getattr(config, "hf_config_dict", None)
        if hf_config_dict is not None:
            self._build_from_diffusers(hf_config_dict)
        else:
            # Empty backbone — will be populated by from_diffusers()
            self.transformer_blocks = None
            self.norm_out = None
            self.proj_out = None
            self.scale_shift_table = None
            self.audio_norm_out = None
            self.audio_proj_out = None
            self.audio_scale_shift_table = None
            self.spmd_rank = None

    def _build_from_diffusers(self, hf_config_dict):
        """Build the TP-sharded transformer from a HuggingFace config dict."""
        replace_sdpa_with_bmm()

        from diffusers.models.transformers.transformer_ltx2 import (
            LTX2VideoTransformer3DModel,
        )

        hf_model = LTX2VideoTransformer3DModel.from_config(hf_config_dict)
        hf_model = hf_model.to(dtype=self.config.neuron_config.torch_dtype)
        hf_model.eval()

        if self.tp_degree > 1:
            _shard_ltx2_transformer(hf_model, self.tp_degree)

        self.transformer_blocks = hf_model.transformer_blocks
        self.norm_out = hf_model.norm_out
        self.proj_out = hf_model.proj_out
        self.scale_shift_table = hf_model.scale_shift_table
        self.audio_norm_out = hf_model.audio_norm_out
        self.audio_proj_out = hf_model.audio_proj_out
        self.audio_scale_shift_table = hf_model.audio_scale_shift_table

        if self.tp_degree > 1 and NEURON_AVAILABLE:
            self.spmd_rank = SPMDRank(self.tp_degree)
        else:
            self.spmd_rank = None

    @classmethod
    def from_diffusers(cls, config, hf_config_dict):
        """Build the Neuron transformer from a HuggingFace config dict.

        Args:
            config: NxDI InferenceConfig
            hf_config_dict: Dict from transformer/config.json (HuggingFace)

        Returns:
            NeuronLTX2TransformerBackbone instance with TP-sharded layers
        """
        # Store hf_config_dict on config so __init__ auto-builds
        config.hf_config_dict = hf_config_dict
        return cls(config)

    def _slice_rope(self, cos, sin):
        """Slice RoPE embeddings to the heads owned by this TP rank.

        Uses SPMDRank (a learnable parameter sharded per-rank) instead of a
        Python int to avoid baking rank=0 as a constant during XLA tracing.
        """
        if self.tp_degree <= 1 or self.spmd_rank is None:
            return (cos, sin)
        h_per_rank = cos.shape[1] // self.tp_degree
        rank = self.spmd_rank.get_rank()  # shape (1,), int32 — auto-sharded per-rank
        start = (rank[0] * h_per_rank).to(torch.long)
        indices = start + torch.arange(h_per_rank, device=cos.device, dtype=torch.long)
        cos_sliced = torch.index_select(cos, 1, indices)
        sin_sliced = torch.index_select(sin, 1, indices)
        return (cos_sliced, sin_sliced)

    def forward(
        self,
        hidden_states,  # (B, video_seq, inner_dim) — after proj_in
        audio_hidden_states,  # (B, audio_seq, audio_inner_dim) — after audio_proj_in
        encoder_hidden_states,  # (B, text_seq, inner_dim) — after caption_projection
        audio_encoder_hidden_states,  # (B, text_seq, audio_inner_dim)
        temb,  # (B, 1, 6*inner_dim) — packed time embedding (6 mod params)
        temb_audio,  # (B, 1, 6*audio_inner_dim) — packed audio time embedding
        embedded_timestep,  # (B, 1, inner_dim) — for output scaling
        audio_embedded_timestep,  # (B, 1, audio_inner_dim)
        temb_ca_ss,  # (B, 1, 4*inner_dim) — cross-attn video scale/shift (4 mod params)
        temb_ca_audio_ss,  # (B, 1, 4*audio_inner_dim) — cross-attn audio scale/shift
        temb_ca_gate,  # (B, 1, inner_dim) — cross-attn video gate (1 mod param)
        temb_ca_audio_gate,  # (B, 1, audio_inner_dim) — cross-attn audio gate
        video_rot_cos,  # (B, num_heads, video_seq, rope_dim)  rope_dim=inner_dim/heads/2
        video_rot_sin,  # (B, num_heads, video_seq, rope_dim)
        audio_rot_cos,  # (B, audio_num_heads, audio_seq, audio_rope_dim)
        audio_rot_sin,  # (B, audio_num_heads, audio_seq, audio_rope_dim)
        ca_video_rot_cos,  # (B, num_heads, video_seq, ca_rope_dim)  ca uses audio_ca_dim
        ca_video_rot_sin,
        ca_audio_rot_cos,  # (B, audio_num_heads, audio_seq, ca_rope_dim)
        ca_audio_rot_sin,
        encoder_attention_mask,  # (B, 1, text_seq) — additive bias
        audio_encoder_attention_mask,  # (B, 1, text_seq) — additive bias
    ):
        # Slice RoPE to local TP shard
        video_rotary_emb = self._slice_rope(video_rot_cos, video_rot_sin)
        audio_rotary_emb = self._slice_rope(audio_rot_cos, audio_rot_sin)
        ca_video_rotary_emb = self._slice_rope(ca_video_rot_cos, ca_video_rot_sin)
        ca_audio_rotary_emb = self._slice_rope(ca_audio_rot_cos, ca_audio_rot_sin)

        # Run 48 transformer blocks
        for block in self.transformer_blocks:
            hidden_states, audio_hidden_states = block(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=temb_ca_ss,
                temb_ca_audio_scale_shift=temb_ca_audio_ss,
                temb_ca_gate=temb_ca_gate,
                temb_ca_audio_gate=temb_ca_audio_gate,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=ca_video_rotary_emb,
                ca_audio_rotary_emb=ca_audio_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
            )

        # Video output projection
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        # Audio output projection
        audio_ssv = (
            self.audio_scale_shift_table[None, None]
            + audio_embedded_timestep[:, :, None]
        )
        audio_shift, audio_scale = audio_ssv[:, :, 0], audio_ssv[:, :, 1]
        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(audio_hidden_states)

        return output, audio_output


# ── TP Sharding ─────────────────────────────────────────────────────────────
def _shard_ltx2_transformer(transformer, tp_degree):
    """Apply tensor parallelism sharding to the LTX-2 transformer.

    Shards:
    - Attention Q/K/V projections → ColumnParallelLinear (shard output dim)
    - Attention output projections → RowParallelLinear (shard input dim)
    - FFN gate/up projections → ColumnParallelLinear
    - FFN down projections → RowParallelLinear
    - QK-norm → DistributedRMSNorm (all-reduce for global variance)
    - Attention heads count divided by tp_degree

    Each of the 48 blocks has 7 attention modules and 2 FFNs:
    - attn1, attn2 (video self-attn, video cross-attn)
    - audio_attn1, audio_attn2 (audio self-attn, audio cross-attn)
    - audio_to_video_attn, video_to_audio_attn (cross-modal)
    - ff, audio_ff (feed-forward)
    """
    from neuronx_distributed.parallel_layers import parallel_state

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()

    def get_shard(data, dim):
        s = data.shape[dim] // tp_size
        if dim == 0:
            return data[s * tp_rank : s * (tp_rank + 1)].clone()
        return data[:, s * tp_rank : s * (tp_rank + 1)].clone()

    def shard_attention(attn):
        for proj_name in ["to_q", "to_k", "to_v"]:
            proj = getattr(attn, proj_name)
            col = ColumnParallelLinear(
                proj.in_features,
                proj.out_features,
                bias=proj.bias is not None,
                gather_output=False,
                dtype=proj.weight.dtype,
            )
            col.weight.data = get_shard(proj.weight.data, 0)
            if proj.bias is not None:
                col.bias.data = get_shard(proj.bias.data, 0)
            setattr(attn, proj_name, col)

        out_linear = attn.to_out[0]
        row = RowParallelLinear(
            out_linear.in_features,
            out_linear.out_features,
            bias=out_linear.bias is not None,
            input_is_parallel=True,
            dtype=out_linear.weight.dtype,
        )
        row.weight.data = get_shard(out_linear.weight.data, 1)
        if out_linear.bias is not None:
            row.bias.data = out_linear.bias.data.clone()
        attn.to_out[0] = row

        for norm_name in ["norm_q", "norm_k"]:
            norm = getattr(attn, norm_name, None)
            if norm is not None and hasattr(norm, "weight") and norm.weight is not None:
                full_dim = norm.weight.shape[0]
                local_dim = full_dim // tp_size
                dist_norm = DistributedRMSNorm(
                    local_dim,
                    eps=norm.eps,
                    tp_size=tp_size,
                    dtype=norm.weight.dtype,
                )
                dist_norm.weight.data = get_shard(norm.weight.data, 0)
                setattr(attn, norm_name, dist_norm)

        attn.heads = attn.heads // tp_size
        attn.inner_dim = attn.inner_dim // tp_size
        if hasattr(attn, "inner_kv_dim"):
            attn.inner_kv_dim = attn.inner_kv_dim // tp_size

    def shard_ffn(ff):
        net = ff.net
        gate = net[0]
        if hasattr(gate, "proj"):
            proj = gate.proj
            col = ColumnParallelLinear(
                proj.in_features,
                proj.out_features,
                bias=proj.bias is not None,
                gather_output=False,
                dtype=proj.weight.dtype,
            )
            col.weight.data = get_shard(proj.weight.data, 0)
            if proj.bias is not None:
                col.bias.data = get_shard(proj.bias.data, 0)
            gate.proj = col
        elif isinstance(gate, nn.Linear):
            col = ColumnParallelLinear(
                gate.in_features,
                gate.out_features,
                bias=gate.bias is not None,
                gather_output=False,
                dtype=gate.weight.dtype,
            )
            col.weight.data = get_shard(gate.weight.data, 0)
            if gate.bias is not None:
                col.bias.data = get_shard(gate.bias.data, 0)
            net[0] = col

        down = net[-1]
        if isinstance(down, nn.Linear):
            row = RowParallelLinear(
                down.in_features,
                down.out_features,
                bias=down.bias is not None,
                input_is_parallel=True,
                dtype=down.weight.dtype,
            )
            row.weight.data = get_shard(down.weight.data, 1)
            if down.bias is not None:
                row.bias.data = down.bias.data.clone()
            net[len(net) - 1] = row

    for block in transformer.transformer_blocks:
        shard_attention(block.attn1)
        shard_attention(block.attn2)
        shard_attention(block.audio_to_video_attn)
        shard_ffn(block.ff)
        shard_attention(block.audio_attn1)
        shard_attention(block.audio_attn2)
        shard_attention(block.video_to_audio_attn)
        shard_ffn(block.audio_ff)


# ── NxDI Config ─────────────────────────────────────────────────────────────
class LTX2BackboneInferenceConfig(InferenceConfig if NEURON_AVAILABLE else object):
    """InferenceConfig for the LTX-2 transformer backbone."""

    def __init__(self, *args, **kwargs):
        if NEURON_AVAILABLE:
            super().__init__(*args, **kwargs)

    def get_required_attributes(self):
        return [
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "inner_dim",
            "audio_num_attention_heads",
            "audio_attention_head_dim",
            "audio_inner_dim",
            "audio_cross_attention_dim",
            "caption_channels",
            "video_seq",
            "audio_seq",
            "text_seq",
            "height",
            "width",
            "num_frames",
        ]


# ── NxDI ModelWrapper ───────────────────────────────────────────────────────
class ModelWrapperLTX2Backbone(ModelWrapper if NEURON_AVAILABLE else object):
    """ModelWrapper for the LTX-2 DiT transformer backbone."""

    def __init__(
        self,
        config,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        model_init_kwargs={},
    ):
        if NEURON_AVAILABLE:
            super().__init__(
                config,
                model_cls,
                tag,
                compiler_args,
                priority_model_idx,
                model_init_kwargs,
            )
        self.bucket_config = None

    @staticmethod
    def _make_attention_mask(bs, text_seq, dtype, valid_tokens=84):
        """Create a realistic additive attention mask for compilation.

        Returns (bs, 1, text_seq) with additive bias format:
          - First `valid_tokens` positions: 0.0 (attend)
          - Remaining positions: -10000.0 (mask out padding)

        Using all-zeros causes XLA to constant-fold the mask, dropping it
        from the compiled graph. This results in the NKI cross-attention
        kernel (and even the BMM fallback) receiving no mask data at runtime.
        """
        mask = torch.full((bs, 1, text_seq), -10000.0, dtype=dtype)
        mask[:, :, :valid_tokens] = 0.0
        return mask

    def input_generator(self):
        """Generate example inputs for Neuron compilation.

        Returns list of (tuple_of_tensors,) matching the 22-input forward() signature.

        Compiled with batch_size=2 to support CFG (classifier-free guidance) natively.
        With CFG, the Diffusers pipeline doubles the batch (uncond + cond) and the
        Neuron backbone processes both in a single forward pass, avoiding the overhead
        of two sequential BS=1 calls.

        Shapes are derived from the actual LTX-2 preprocessing pipeline:
        - time_embed produces (B, 6*inner_dim) for video, (B, 6*audio_inner_dim) for audio
        - cross-attn scale/shift has num_mod_params=4, gate has num_mod_params=1
        - Audio streams use audio_inner_dim (32*64=2048), not inner_dim (32*128=4096)
        - RoPE: video uses num_heads/head_dim, audio uses audio_num_heads/audio_head_dim
        See compile_full_tp4.py:precompute_inputs() for the authoritative reference.
        """
        dtype = self.config.neuron_config.torch_dtype
        inner_dim = self.config.inner_dim
        audio_inner_dim = self.config.audio_inner_dim
        audio_ca_dim = self.config.audio_cross_attention_dim
        video_seq = self.config.video_seq
        audio_seq = self.config.audio_seq
        text_seq = self.config.text_seq
        num_heads = self.config.num_attention_heads
        head_dim = self.config.attention_head_dim
        audio_num_heads = self.config.audio_num_attention_heads
        audio_head_dim = self.config.audio_attention_head_dim

        # Batch size: 2 for CFG (unconditional + conditional in one pass),
        # 1 for Stage 2 / non-CFG workloads. Configurable via config.batch_size.
        bs = getattr(self.config, "batch_size", 2)

        # RoPE rotation dim per head:
        # self-attn video: rope.dim=inner_dim → inner_dim / num_heads / 2
        # self-attn audio: audio_rope.dim=audio_inner_dim → audio_inner_dim / audio_num_heads / 2
        # cross-attn video: cross_attn_rope.dim=audio_cross_attention_dim → audio_ca_dim / num_heads / 2
        # cross-attn audio: cross_attn_audio_rope.dim=audio_cross_attention_dim → audio_ca_dim / audio_num_heads / 2
        video_rope_dim = inner_dim // num_heads // 2  # 4096/32/2 = 64
        audio_rope_dim = audio_inner_dim // audio_num_heads // 2  # 2048/32/2 = 32
        ca_video_rope_dim = audio_ca_dim // num_heads // 2  # 2048/32/2 = 32
        ca_audio_rope_dim = audio_ca_dim // audio_num_heads // 2  # 2048/32/2 = 32

        model_inputs = (
            # Projected hidden states (after proj_in / audio_proj_in on CPU)
            torch.randn(bs, video_seq, inner_dim, dtype=dtype),  # hidden_states
            torch.randn(
                bs, audio_seq, audio_inner_dim, dtype=dtype
            ),  # audio_hidden_states
            # Projected encoder hidden states (after caption_projection on CPU)
            torch.randn(bs, text_seq, inner_dim, dtype=dtype),  # encoder_hidden_states
            torch.randn(
                bs, text_seq, audio_inner_dim, dtype=dtype
            ),  # audio_encoder_hidden_states
            # Time embeddings: LTX2AdaLayerNormSingle with num_mod_params=6
            # time_embed.linear: (embedding_dim) → (6 * embedding_dim)
            torch.randn(bs, 1, 6 * inner_dim, dtype=dtype),  # temb
            torch.randn(bs, 1, 6 * audio_inner_dim, dtype=dtype),  # temb_audio
            # Embedded timestep (second return from time_embed, shape=embedding_dim)
            torch.randn(bs, 1, inner_dim, dtype=dtype),  # embedded_timestep
            torch.randn(bs, 1, audio_inner_dim, dtype=dtype),  # audio_embedded_timestep
            # Cross-attn scale/shift: num_mod_params=4
            torch.randn(bs, 1, 4 * inner_dim, dtype=dtype),  # temb_ca_ss
            torch.randn(bs, 1, 4 * audio_inner_dim, dtype=dtype),  # temb_ca_audio_ss
            # Cross-attn gate: num_mod_params=1
            torch.randn(bs, 1, 1 * inner_dim, dtype=dtype),  # temb_ca_gate
            torch.randn(bs, 1, 1 * audio_inner_dim, dtype=dtype),  # temb_ca_audio_gate
            # Video RoPE cos/sin: (B, num_heads, video_seq, video_rope_dim)
            torch.randn(
                bs, num_heads, video_seq, video_rope_dim, dtype=dtype
            ),  # video_rot_cos
            torch.randn(
                bs, num_heads, video_seq, video_rope_dim, dtype=dtype
            ),  # video_rot_sin
            # Audio RoPE cos/sin: (B, audio_num_heads, audio_seq, audio_rope_dim)
            torch.randn(
                bs, audio_num_heads, audio_seq, audio_rope_dim, dtype=dtype
            ),  # audio_rot_cos
            torch.randn(
                bs, audio_num_heads, audio_seq, audio_rope_dim, dtype=dtype
            ),  # audio_rot_sin
            # Cross-attn RoPE: same seq_len, but rope dim = audio_ca_dim / heads / 2
            # cross_attn_rope has dim=audio_cross_attention_dim, not inner_dim
            torch.randn(
                bs, num_heads, video_seq, ca_video_rope_dim, dtype=dtype
            ),  # ca_video_rot_cos
            torch.randn(
                bs, num_heads, video_seq, ca_video_rope_dim, dtype=dtype
            ),  # ca_video_rot_sin
            torch.randn(
                bs, audio_num_heads, audio_seq, ca_audio_rope_dim, dtype=dtype
            ),  # ca_audio_rot_cos
            torch.randn(
                bs, audio_num_heads, audio_seq, ca_audio_rope_dim, dtype=dtype
            ),  # ca_audio_rot_sin
            # Attention masks: additive bias (B, 1, text_seq)
            # CRITICAL: Must use realistic non-zero mask values, NOT all-zeros.
            # All-zeros causes XLA to constant-fold the mask tensor, dropping it
            # from the compiled graph. Real masks have ~84 valid (0.0) and ~940
            # padding (-10000.0) positions in additive bias format.
            self._make_attention_mask(bs, text_seq, dtype),  # encoder_attention_mask
            self._make_attention_mask(
                bs, text_seq, dtype
            ),  # audio_encoder_attention_mask
        )

        return [model_inputs]

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, *args, **kwargs):
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() first."
            )
        output = self._forward(*args)
        return output


# ── NxDI Application ────────────────────────────────────────────────────────
class NeuronLTX2BackboneApplication(
    NeuronApplicationBase if NEURON_AVAILABLE else object
):
    """NxDI Application wrapping the LTX-2 DiT transformer backbone.

    Handles compilation, weight sharding, loading, and inference.
    Follows the same pattern as NeuronFluxBackboneApplication.
    """

    _model_cls = NeuronLTX2TransformerBackbone

    def __init__(self, *args, **kwargs):
        if NEURON_AVAILABLE:
            super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    def get_model_wrapper_cls(self):
        return ModelWrapperLTX2Backbone

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self):
        """Compiler args for the LTX-2 transformer.

        Uses --auto-cast matmult (the spelling with two t's) and --lnc 2 for trn2.

        Extra flags can be appended via LTX2_EXTRA_COMPILER_FLAGS env var for A/B testing.
        Example: LTX2_EXTRA_COMPILER_FLAGS="--enable-saturate-infinity --internal-hoist-allgather"
        """
        compiler_args = "--model-type=transformer -O2"
        compiler_args += " --auto-cast matmult --lnc 2"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap'"

        # Allow extra compiler flags for A/B testing
        extra = os.environ.get("LTX2_EXTRA_COMPILER_FLAGS", "")
        if extra:
            compiler_args += " " + extra
            logger.info(f"Extra compiler flags: {extra}")

        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        os.environ["NEURON_FUSE_SOFTMAX"] = "1"
        os.environ["NEURON_CUSTOM_SILU"] = "1"
        os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"

        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def checkpoint_loader_fn(self, mmap: bool = False):
        """Load the LTX-2 transformer weights from HuggingFace or local path.

        Handles HuggingFace hub IDs (e.g., "Lightricks/LTX-2/transformer") by
        downloading the transformer via diffusers. Also handles local directories
        containing safetensors files.

        The state dict returned contains ALL HuggingFace transformer weights.
        Keys that don't match the backbone model (proj_in, time_embed, etc.) are
        ignored during load_state_dict(strict=False).
        """
        model_path = self.model_path
        logger.info("Loading LTX-2 transformer weights from %s", model_path)

        # Try local path first
        if os.path.isdir(model_path):
            from safetensors.torch import load_file
            import glob as _glob

            safetensors_files = sorted(
                _glob.glob(os.path.join(model_path, "*.safetensors"))
            )
            if safetensors_files:
                model_sd = {}
                for sf in safetensors_files:
                    model_sd.update(load_file(sf))
                logger.info(
                    "Loaded %d tensors from %d safetensors files",
                    len(model_sd),
                    len(safetensors_files),
                )
            else:
                # Fallback to loading from torch checkpoint
                import glob as _glob

                pt_files = _glob.glob(os.path.join(model_path, "*.bin")) + _glob.glob(
                    os.path.join(model_path, "*.pt")
                )
                if pt_files:
                    model_sd = {}
                    for pf in pt_files:
                        model_sd.update(torch.load(pf, map_location="cpu"))
                else:
                    raise FileNotFoundError(
                        f"No safetensors or pt files in {model_path}"
                    )
        else:
            # HuggingFace hub ID — download via diffusers
            from diffusers.models.transformers.transformer_ltx2 import (
                LTX2VideoTransformer3DModel,
            )

            logger.info("Downloading transformer from HuggingFace: %s", model_path)
            hf_model = LTX2VideoTransformer3DModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
            model_sd = hf_model.state_dict()
            del hf_model
            import gc

            gc.collect()
            logger.info("Loaded %d tensors from HuggingFace", len(model_sd))

        model_sd = self.convert_hf_to_neuron_state_dict(model_sd, self.config)
        return model_sd

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Convert HuggingFace state dict to Neuron format.

        Key transformations:
        1. Adds the SPMDRank arange tensor for per-rank RoPE slicing (critical for TP>1)
        2. Filters to only keep keys matching the backbone model structure
           (transformer_blocks.*, norm_out.*, proj_out.*, scale_shift_table, audio_*)
        3. Removes CPU preprocessing layers (proj_in, time_embed, rope, caption_projection, etc.)
        """
        # Add SPMDRank tensor for per-rank sharding
        state_dict["spmd_rank.rank"] = torch.arange(
            0, config.neuron_config.world_size, dtype=torch.int32
        )

        # Filter to keys the backbone model expects
        backbone_prefixes = (
            "transformer_blocks.",
            "norm_out.",
            "proj_out.",
            "scale_shift_table",
            "audio_norm_out.",
            "audio_proj_out.",
            "audio_scale_shift_table",
            "spmd_rank.",
        )
        filtered_sd = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k.startswith(backbone_prefixes):
                filtered_sd[k] = v.clone().detach().contiguous()
            else:
                skipped_keys.append(k)

        if skipped_keys:
            logger.info(
                "Filtered out %d CPU-only keys (proj_in, time_embed, rope, etc.)",
                len(skipped_keys),
            )

        return filtered_sd
