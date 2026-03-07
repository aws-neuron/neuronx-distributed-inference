"""
NxDI LTX-2.3 Transformer Model
===============================
Neuron-optimized implementation of the LTX-2.3 22B DiT audio-video
diffusion transformer. Uses the native ltx-core model architecture
with TransformerArgs dataclasses for block communication.

Architecture:
  - Native ltx-core LTXModel with BasicAVTransformerBlock
  - 48 transformer blocks, 32 heads, 4096 video dim, 2048 audio dim
  - Gated attention (to_gate_logits per attention head)
  - Cross-attention AdaLN (prompt_scale_shift_table)
  - QK-norm uses q_norm/k_norm (not norm_q/norm_k as in Diffusers)
  - RoPE type: split (not interleaved)
  - Caption projection is in text encoder connectors (not in transformer)

The backbone takes 22 flat tensor inputs (for XLA tracing), constructs
TransformerArgs dataclasses internally, and calls native block forwards.
All preprocessing (patchify, adaln, rope, connector) done on CPU.

Usage:
  See application.py for the high-level NeuronLTX23Application class.
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


# -- BMM-based SDPA replacement -----------------------------------------------
_sdpa_replaced = False
_sdpa_original = None


def replace_sdpa_with_bmm():
    """Replace F.scaled_dot_product_attention with BMM-based implementation.

    SDPA is not supported on Neuron XLA. This replacement uses explicit
    BMM + softmax which compiles cleanly. Handles 3D and 4D inputs,
    optional attention masks, and falls back to original SDPA on CPU.
    """
    global _sdpa_replaced, _sdpa_original
    if _sdpa_replaced:
        return _sdpa_original
    _sdpa_original = torch.nn.functional.scaled_dot_product_attention

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
                # Expand broadcastable dims to (b, h, ...) before flattening to 3D.
                # PytorchAttention may pass masks like (1, 1, 1, 256) where b=1 and h=1
                # are broadcastable over the actual (b, h) from query. A direct reshape
                # would fail because total elements differ. Expand first, then reshape.
                attn_mask = attn_mask.expand(b, h, -1, -1).reshape(
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


# -- DistributedRMSNorm -------------------------------------------------------
class DistributedRMSNorm(nn.Module):
    """RMSNorm with all-reduce for global variance computation across TP ranks.

    Standard RMSNorm on a TP-sharded hidden dimension only sees the local shard.
    This version computes sum-of-squares locally, all-reduces across ranks, then
    normalizes with the global RMS. Essential for QK-norm accuracy in TP>1.

    The all-reduce in this norm is NOT redundant -- removing it (LocalRMSNorm
    experiment in LTX-2) made quality significantly worse.
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


# -- NeuronLTX23TransformerBackbone -------------------------------------------
class NeuronLTX23TransformerBackbone(nn.Module):
    """The core LTX-2.3 DiT transformer backbone for Neuron.

    Contains the 48 transformer blocks, output normalization, and projection layers.
    Takes 22 preprocessed tensor inputs (all preprocessing done on CPU) and returns
    (video_output, audio_output).

    Key differences from LTX-2:
    - Uses native ltx-core LTXModel (not Diffusers LTX2VideoTransformer3DModel)
    - Caption projection is NOT in the transformer (moved to text encoder in 2.3)
    - Attention QK-norm uses q_norm/k_norm (vs norm_q/norm_k in Diffusers)
    - Gated attention may be present (to_gate_logits per attention module)
    """

    def __init__(self, config):
        """Initialize from InferenceConfig.

        If config has an `ltx_config_dict` attribute (set during config creation),
        the transformer is automatically built from the ltx-core model with TP sharding.
        """
        super().__init__()
        self.config = config
        self.tp_degree = config.neuron_config.tp_degree

        ltx_config_dict = getattr(config, "ltx_config_dict", None)
        if ltx_config_dict is not None:
            self._build_from_ltx_core(ltx_config_dict)
        else:
            self.transformer_blocks = None
            self.norm_out = None
            self.proj_out = None
            self.scale_shift_table = None
            self.audio_norm_out = None
            self.audio_proj_out = None
            self.audio_scale_shift_table = None
            self.spmd_rank = None

    def _build_from_ltx_core(self, ltx_config_dict):
        """Build the TP-sharded transformer from ltx-core config dict.

        The ltx_config_dict should be the full model config (as loaded from
        the safetensors metadata or provided manually) with 'transformer' key
        containing the transformer architecture parameters.
        """
        replace_sdpa_with_bmm()

        # Import and build the native ltx-core model
        from ltx_core.model.transformer.model_configurator import LTXModelConfigurator

        ltx_model = LTXModelConfigurator.from_config(ltx_config_dict)
        ltx_model = ltx_model.to(dtype=self.config.neuron_config.torch_dtype)
        ltx_model.eval()

        if self.tp_degree > 1:
            _shard_ltx23_transformer(ltx_model, self.tp_degree)

        # Copy references to the backbone layers
        self.transformer_blocks = ltx_model.transformer_blocks
        self.norm_out = ltx_model.norm_out
        self.proj_out = ltx_model.proj_out
        self.scale_shift_table = ltx_model.scale_shift_table
        self.audio_norm_out = ltx_model.audio_norm_out
        self.audio_proj_out = ltx_model.audio_proj_out
        self.audio_scale_shift_table = ltx_model.audio_scale_shift_table

        if self.tp_degree > 1 and NEURON_AVAILABLE:
            self.spmd_rank = SPMDRank(self.tp_degree)
        else:
            self.spmd_rank = None

    def _slice_rope(self, cos, sin):
        """Slice RoPE embeddings to the heads owned by this TP rank.

        Uses SPMDRank (a learnable parameter sharded per-rank) instead of a
        Python int to avoid baking rank=0 as a constant during XLA tracing.
        """
        if self.tp_degree <= 1 or self.spmd_rank is None:
            return (cos, sin)
        h_per_rank = cos.shape[1] // self.tp_degree
        rank = self.spmd_rank.get_rank()  # shape (1,), int32
        start = (rank[0] * h_per_rank).to(torch.long)
        indices = start + torch.arange(h_per_rank, device=cos.device, dtype=torch.long)
        cos_sliced = torch.index_select(cos, 1, indices)
        sin_sliced = torch.index_select(sin, 1, indices)
        return (cos_sliced, sin_sliced)

    def forward(
        self,
        hidden_states,  # (B, video_seq, inner_dim) -- after patchify_proj
        audio_hidden_states,  # (B, audio_seq, audio_inner_dim)
        encoder_hidden_states,  # (B, text_seq, inner_dim) -- already projected
        audio_encoder_hidden_states,  # (B, text_seq, audio_inner_dim)
        temb,  # (B, video_seq, 9*inner_dim) -- per-token time embedding
        temb_audio,  # (B, audio_seq, 9*audio_inner_dim)
        embedded_timestep,  # (B, video_seq, inner_dim) -- per-token output scaling
        audio_embedded_timestep,  # (B, audio_seq, audio_inner_dim)
        video_ca_ss,  # (B, 1, 4*inner_dim) -- cross-modal scale/shift
        audio_ca_ss,  # (B, 1, 4*audio_inner_dim)
        video_ca_gate,  # (B, 1, inner_dim) -- cross-modal a2v gate
        audio_ca_gate,  # (B, 1, audio_inner_dim) -- cross-modal v2a gate
        video_rot_cos,  # (B, H, video_seq, rope_dim) -- self-attn RoPE
        video_rot_sin,
        audio_rot_cos,  # (B, H_audio, audio_seq, rope_dim)
        audio_rot_sin,
        ca_video_rot_cos,  # (B, H, video_seq, ca_rope_dim) -- cross-modal RoPE
        ca_video_rot_sin,
        ca_audio_rot_cos,  # (B, H_audio, audio_seq, ca_rope_dim)
        ca_audio_rot_sin,
        encoder_attention_mask,  # (B, 1, 1, text_seq) -- additive bias mask
        audio_encoder_attention_mask,  # (B, 1, 1, text_seq)
        prompt_timestep,  # (B, 1, 2*inner_dim) -- cross-attn AdaLN prompt
        audio_prompt_timestep,  # (B, 1, 2*audio_inner_dim)
    ):
        """Forward pass: construct TransformerArgs from flat tensors, run native blocks.

        Takes 24 flat tensor inputs for XLA tracing compatibility.
        Constructs TransformerArgs dataclasses internally and calls the native
        BasicAVTransformerBlock.forward() which uses the ltx-core interface.
        """
        from ltx_core.model.transformer.transformer_args import TransformerArgs
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig

        batch_size = hidden_states.shape[0]

        # Slice RoPE to local TP shard
        video_pe = self._slice_rope(video_rot_cos, video_rot_sin)
        audio_pe = self._slice_rope(audio_rot_cos, audio_rot_sin)
        ca_video_pe = self._slice_rope(ca_video_rot_cos, ca_video_rot_sin)
        ca_audio_pe = self._slice_rope(ca_audio_rot_cos, ca_audio_rot_sin)

        # Construct TransformerArgs for video and audio
        video_args = TransformerArgs(
            x=hidden_states,
            context=encoder_hidden_states,
            context_mask=encoder_attention_mask,
            timesteps=temb,
            embedded_timestep=embedded_timestep,
            positional_embeddings=video_pe,  # (cos, sin) tuple
            cross_positional_embeddings=ca_video_pe,
            cross_scale_shift_timestep=video_ca_ss,
            cross_gate_timestep=video_ca_gate,
            enabled=True,
            prompt_timestep=prompt_timestep,
            self_attention_mask=None,
        )
        audio_args = TransformerArgs(
            x=audio_hidden_states,
            context=audio_encoder_hidden_states,
            context_mask=audio_encoder_attention_mask,
            timesteps=temb_audio,
            embedded_timestep=audio_embedded_timestep,
            positional_embeddings=audio_pe,
            cross_positional_embeddings=ca_audio_pe,
            cross_scale_shift_timestep=audio_ca_ss,
            cross_gate_timestep=audio_ca_gate,
            enabled=True,
            prompt_timestep=audio_prompt_timestep,
            self_attention_mask=None,
        )

        # Empty perturbations for deterministic tracing (no skip branches)
        perturbations = BatchedPerturbationConfig.empty(batch_size)

        # Run 48 transformer blocks using native ltx-core interface
        for block in self.transformer_blocks:
            video_args, audio_args = block(
                video=video_args,
                audio=audio_args,
                perturbations=perturbations,
            )

        # Video output projection (matches LTXModel._process_output)
        vx = video_args.x
        scale_shift_values = (
            self.scale_shift_table[None, None]
            + video_args.embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        vx = self.norm_out(vx)
        vx = vx * (1 + scale) + shift
        output = self.proj_out(vx)

        # Audio output projection
        ax = audio_args.x
        audio_ssv = (
            self.audio_scale_shift_table[None, None]
            + audio_args.embedded_timestep[:, :, None]
        )
        audio_shift, audio_scale = audio_ssv[:, :, 0], audio_ssv[:, :, 1]
        ax = self.audio_norm_out(ax)
        ax = ax * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(ax)

        return output, audio_output


# -- TP Sharding --------------------------------------------------------------
def _shard_ltx23_transformer(ltx_model, tp_degree):
    """Apply tensor parallelism sharding to the LTX-2.3 transformer.

    Adapted from LTX-2's _shard_ltx2_transformer for the native ltx-core
    model structure. Key differences:
    - QK-norm attribute names: q_norm/k_norm (not norm_q/norm_k)
    - Gated attention: to_gate_logits may be present (Column-sharded)
    - FFN structure: FeedForward with .net containing GEGLU gate + Linear down

    Each of the 48 blocks has 7 attention modules and 2 FFNs:
    - attn1, attn2 (video self-attn, video text cross-attn)
    - audio_attn1, audio_attn2 (audio self-attn, audio text cross-attn)
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
        """Shard a single Attention module for TP."""
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

        # Output projection: RowParallelLinear
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
            row.bias.data = out_linear.bias.data.clone()  # bias not sharded
        attn.to_out[0] = row

        # QK-norm -> DistributedRMSNorm
        # LTX-2.3 native model uses q_norm/k_norm (torch.nn.RMSNorm)
        # LTX-2 Diffusers used norm_q/norm_k
        for norm_name in ["q_norm", "k_norm"]:
            norm = getattr(attn, norm_name, None)
            if norm is not None and hasattr(norm, "weight") and norm.weight is not None:
                full_dim = norm.weight.shape[0]
                local_dim = full_dim // tp_size
                dist_norm = DistributedRMSNorm(
                    local_dim,
                    eps=getattr(norm, "eps", 1e-6),
                    tp_size=tp_size,
                    dtype=norm.weight.dtype,
                )
                dist_norm.weight.data = get_shard(norm.weight.data, 0)
                setattr(attn, norm_name, dist_norm)

        # Gated attention: to_gate_logits (Linear: query_dim -> heads)
        gate_logits = getattr(attn, "to_gate_logits", None)
        if gate_logits is not None and isinstance(gate_logits, nn.Linear):
            col = ColumnParallelLinear(
                gate_logits.in_features,
                gate_logits.out_features,
                bias=gate_logits.bias is not None,
                gather_output=False,
                dtype=gate_logits.weight.dtype,
            )
            col.weight.data = get_shard(gate_logits.weight.data, 0)
            if gate_logits.bias is not None:
                col.bias.data = get_shard(gate_logits.bias.data, 0)
            attn.to_gate_logits = col

        # Update head count for TP
        attn.heads = attn.heads // tp_size
        # The native model doesn't have inner_dim/inner_kv_dim attributes
        # but the dim_head stays the same. heads is divided.

    def shard_ffn(ff):
        """Shard a FeedForward module for TP.

        LTX-2.3 FeedForward structure:
          ff.net = [GEGLU(proj=Linear), Identity(), Linear]
          or ff.net = [Linear, activation, Linear]
        """
        net = ff.net
        gate = net[0]
        if hasattr(gate, "proj"):
            # GEGLU: gate.proj is the Linear
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

        # Down projection: last Linear in net
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

    for block in ltx_model.transformer_blocks:
        # Video attention modules
        shard_attention(block.attn1)
        shard_attention(block.attn2)
        shard_ffn(block.ff)
        # Audio attention modules
        shard_attention(block.audio_attn1)
        shard_attention(block.audio_attn2)
        shard_ffn(block.audio_ff)
        # Cross-modal attention
        shard_attention(block.audio_to_video_attn)
        shard_attention(block.video_to_audio_attn)


# -- NxDI Config --------------------------------------------------------------
class LTX23BackboneInferenceConfig(InferenceConfig if NEURON_AVAILABLE else object):
    """InferenceConfig for the LTX-2.3 transformer backbone."""

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
            "video_seq",
            "audio_seq",
            "text_seq",
            "height",
            "width",
            "num_frames",
        ]


# -- NxDI ModelWrapper ---------------------------------------------------------
class ModelWrapperLTX23Backbone(ModelWrapper if NEURON_AVAILABLE else object):
    """ModelWrapper for the LTX-2.3 DiT transformer backbone."""

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

    def input_generator(self):
        """Generate example inputs for Neuron compilation.

        Returns list of (tuple_of_tensors,) matching the 24-input forward() signature.
        The native ltx-core model uses 9 AdaLN mod params (not 6 as LTX-2 Diffusers):
        3 self-attn (shift, scale, gate) + 3 FFN (shift, scale, gate) +
        3 cross-attn AdaLN (shift_q, scale_q, gate_q).
        """
        dtype = self.config.neuron_config.torch_dtype
        inner_dim = self.config.inner_dim
        audio_inner_dim = self.config.audio_inner_dim
        audio_ca_dim = self.config.audio_cross_attention_dim
        video_seq = self.config.video_seq
        audio_seq = self.config.audio_seq
        text_seq = self.config.text_seq
        num_heads = self.config.num_attention_heads
        audio_num_heads = self.config.audio_num_attention_heads

        # RoPE rotation dim per head (split RoPE: dim // 2 per head)
        video_rope_dim = inner_dim // num_heads // 2  # 4096/32/2 = 64
        audio_rope_dim = audio_inner_dim // audio_num_heads // 2  # 2048/32/2 = 32
        ca_video_rope_dim = audio_ca_dim // num_heads // 2  # 2048/32/2 = 32
        ca_audio_rope_dim = audio_ca_dim // audio_num_heads // 2  # 2048/32/2 = 32

        model_inputs = (
            # Projected hidden states
            torch.randn(1, video_seq, inner_dim, dtype=dtype),
            torch.randn(1, audio_seq, audio_inner_dim, dtype=dtype),
            # Encoder hidden states (already projected by text encoder connectors)
            torch.randn(1, text_seq, inner_dim, dtype=dtype),
            torch.randn(1, text_seq, audio_inner_dim, dtype=dtype),
            # Time embeddings (9 mod params from adaln_single, per-token)
            torch.randn(1, video_seq, 9 * inner_dim, dtype=dtype),
            torch.randn(1, audio_seq, 9 * audio_inner_dim, dtype=dtype),
            # Embedded timestep (per-token, for output scaling)
            torch.randn(1, video_seq, inner_dim, dtype=dtype),
            torch.randn(1, audio_seq, audio_inner_dim, dtype=dtype),
            # Cross-modal scale/shift (4 mod params from av_ca_*_scale_shift, per-batch)
            torch.randn(1, 1, 4 * inner_dim, dtype=dtype),
            torch.randn(1, 1, 4 * audio_inner_dim, dtype=dtype),
            # Cross-modal gate (1 mod param from av_ca_*_gate, per-batch)
            torch.randn(1, 1, 1 * inner_dim, dtype=dtype),
            torch.randn(1, 1, 1 * audio_inner_dim, dtype=dtype),
            # Video self-attn RoPE cos/sin (split format)
            torch.randn(1, num_heads, video_seq, video_rope_dim, dtype=dtype),
            torch.randn(1, num_heads, video_seq, video_rope_dim, dtype=dtype),
            # Audio self-attn RoPE cos/sin
            torch.randn(1, audio_num_heads, audio_seq, audio_rope_dim, dtype=dtype),
            torch.randn(1, audio_num_heads, audio_seq, audio_rope_dim, dtype=dtype),
            # Cross-modal RoPE
            torch.randn(1, num_heads, video_seq, ca_video_rope_dim, dtype=dtype),
            torch.randn(1, num_heads, video_seq, ca_video_rope_dim, dtype=dtype),
            torch.randn(1, audio_num_heads, audio_seq, ca_audio_rope_dim, dtype=dtype),
            torch.randn(1, audio_num_heads, audio_seq, ca_audio_rope_dim, dtype=dtype),
            # Attention masks (additive bias, shape B x 1 x 1 x text_seq)
            # The native preprocessor converts binary masks to 4D additive bias:
            #   (B, text_seq) -> (B, 1, 1, text_seq) with 0 = attend, -max = ignore
            torch.zeros(1, 1, 1, text_seq, dtype=dtype),
            torch.zeros(1, 1, 1, text_seq, dtype=dtype),
            # Prompt timestep for cross-attn AdaLN (2 mod params, per-batch)
            torch.randn(1, 1, 2 * inner_dim, dtype=dtype),
            torch.randn(1, 1, 2 * audio_inner_dim, dtype=dtype),
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


# -- NxDI Application ---------------------------------------------------------
class NeuronLTX23BackboneApplication(
    NeuronApplicationBase if NEURON_AVAILABLE else object
):
    """NxDI Application wrapping the LTX-2.3 DiT transformer backbone.

    Handles compilation, weight sharding, loading, and inference.
    Follows the same pattern as NeuronFluxBackboneApplication / NeuronLTX2BackboneApplication.
    """

    _model_cls = NeuronLTX23TransformerBackbone

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
        return ModelWrapperLTX23Backbone

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self):
        """Compiler args for the LTX-2.3 transformer.

        Same as LTX-2: --auto-cast matmult (two t's) and --lnc 2 for trn2.
        """
        compiler_args = "--model-type=transformer -O1"
        compiler_args += " --auto-cast matmult --lnc 2"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap'"

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
        """Load the LTX-2.3 transformer weights from safetensors.

        LTX-2.3 weights are stored in a single safetensors file (not a
        HuggingFace model repo with config.json). Weight keys use the
        native ltx-core naming convention.
        """
        model_path = self.model_path
        logger.info("Loading LTX-2.3 transformer weights from %s", model_path)

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
                raise FileNotFoundError(f"No safetensors files in {model_path}")
        elif os.path.isfile(model_path) and model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            model_sd = load_file(model_path)
            logger.info("Loaded %d tensors from %s", len(model_sd), model_path)
        else:
            raise FileNotFoundError(f"Cannot load weights from {model_path}")

        model_sd = self.convert_to_neuron_state_dict(model_sd, self.config)
        return model_sd

    @staticmethod
    def convert_to_neuron_state_dict(state_dict, config):
        """Convert safetensors state dict to Neuron backbone format.

        Key transformations:
        1. Strips 'model.diffusion_model.' prefix (ComfyUI convention in safetensors)
        2. Adds the SPMDRank arange tensor for per-rank RoPE slicing
        3. Filters to only keep keys matching the backbone model structure
           (transformer_blocks.*, norm_out.*, proj_out.*, scale_shift_table, audio_*)
        4. Removes preprocessing layers (patchify_proj, adaln, rope, connectors, etc.)
        """
        # Strip ComfyUI prefix if present
        prefix = "model.diffusion_model."
        stripped_sd = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                stripped_sd[k[len(prefix) :]] = v
            else:
                stripped_sd[k] = v

        # Add SPMDRank tensor for per-rank sharding
        stripped_sd["spmd_rank.rank"] = torch.arange(
            0, config.neuron_config.world_size, dtype=torch.int32
        )

        # Filter to keys the backbone model expects
        # These match the native ltx-core LTXModel weight keys
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
        for k, v in stripped_sd.items():
            if k.startswith(backbone_prefixes):
                filtered_sd[k] = v.clone().detach().contiguous()
            else:
                skipped_keys.append(k)

        if skipped_keys:
            logger.info(
                "Filtered out %d preprocessing keys (patchify_proj, adaln, connectors, etc.): %s",
                len(skipped_keys),
                ", ".join(skipped_keys[:10])
                + ("..." if len(skipped_keys) > 10 else ""),
            )

        return filtered_sd
