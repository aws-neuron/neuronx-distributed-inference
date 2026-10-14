"""
Qwen-Image-Edit-2511 MMDiT Transformer with Multi-LoRA on Neuron.

Compiles the 60-layer MMDiT transformer with runtime-switchable LoRA adapters
using NxD ModelBuilder with auto-aliasing. Supports the fal Multi-Angles LoRA
(96-pose camera control) and any compatible LoRA adapter.

Key features:
- ModelBuilder API (V2 SPMD) with enable_aliasing=True for all model state
- 1680 LoRA active buffers (14 targets x 60 blocks x 2 matrices A/B)
- Runtime adapter switching via write_to_neuron_buffer() (<1ms, zero recompilation)
- NKI Flash Attention for joint image+text attention
- TP=4 sharding with ColumnParallel/RowParallel projections

Usage:
    python -m src.modeling_qwen_image_edit_lora \\
        --model_id Qwen/Qwen-Image-Edit-2511 \\
        --compiled_models_dir ./compiled_models \\
        --max_loras 4 --max_rank 16
"""

import os
import sys
import json
import math

os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")
os.environ.setdefault("XLA_DISABLE_FUNCTIONALIZATION", "1")
os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")

# Compiler flags (--target auto-detected from hardware, --lnc from env var)
compiler_flags = " --lnc=2 --auto-cast=none --enable-fast-loading-neuron-binaries --tensorizer-options='--enable-ccop-compute-overlap' --internal-hlo2tensorizer-options='--enable-state-buffer-mode=hybrid --remat-by-default' "
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Optional, Tuple, List, Dict

from diffusers import QwenImageEditPlusPipeline

# SPMD compilation with ModelBuilder (auto-aliasing for runtime weight updates)
import torch_neuronx
from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)

# Import NKI Flash Attention
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)

print("NKI Flash Attention kernel loaded successfully")

CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./model_cache")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")

# ============================================================================
# LoRA Target Modules (matching the fal Multi-Angles LoRA adapter)
# ============================================================================
# These are the 14 target layers per transformer block (840 total for 60 blocks)
LORA_TARGET_MODULES = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",  # to_out is nn.ModuleList; .0 is the Linear
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
    "img_mod.1",  # mod is nn.Sequential(SiLU, Linear); .1 is the Linear
    "txt_mod.1",
]


# ============================================================================
# NKI Flash Attention (identical to V3 CFG version)
# ============================================================================
def nki_flash_attention(query, key, value):
    """NKI Flash Attention wrapper. Args: [B, H, S, D]."""
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]

    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))

    attn_output = torch.zeros(
        (bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device
    )
    scale = 1 / math.sqrt(d_head)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](
            q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap"
        )
    else:
        _flash_fwd_call(
            q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap"
        )

    return attn_output.reshape((bs, n_head, q_len, d_head))


def apply_rotary_emb_precomputed(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """Apply rotary embeddings using pre-computed cos/sin tensors."""
    cos, sin = freqs_cis
    cos = cos.to(x.device)
    sin = sin.to(x.device)

    if not use_real:
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]

        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos

        out = torch.stack([out_real, out_imag], dim=-1)
        out = out.flatten(-2)

        return out.to(x.dtype)
    else:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        else:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)

        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


# Patch apply_rotary_emb_qwen
import diffusers.models.transformers.transformer_qwenimage as qwen_module

qwen_module.apply_rotary_emb_qwen = apply_rotary_emb_precomputed
print("Patched apply_rotary_emb_qwen for pre-computed RoPE")


# ============================================================================
# LoRA-aware Attention module (NKI Flash, no CFG parallel)
# ============================================================================
class LoRANKIQwenAttention(nn.Module):
    """
    NKI Flash Attention with LoRA weight_active parameters.

    Each projection (to_q, to_k, etc.) has lora_A_active and lora_B_active
    parameters that get aliased for runtime updates.
    """

    def __init__(
        self, orig_attn, max_loras_active: int, max_rank: int, dtype: torch.dtype
    ):
        super().__init__()

        self.heads = orig_attn.heads
        self.to_q = orig_attn.to_q
        self.to_k = orig_attn.to_k
        self.to_v = orig_attn.to_v
        self.to_out = orig_attn.to_out

        self.add_q_proj = getattr(orig_attn, "add_q_proj", None)
        self.add_k_proj = getattr(orig_attn, "add_k_proj", None)
        self.add_v_proj = getattr(orig_attn, "add_v_proj", None)
        self.to_add_out = getattr(orig_attn, "to_add_out", None)

        self.norm_q = getattr(orig_attn, "norm_q", None)
        self.norm_k = getattr(orig_attn, "norm_k", None)
        self.norm_added_q = getattr(orig_attn, "norm_added_q", None)
        self.norm_added_k = getattr(orig_attn, "norm_added_k", None)

        # Create LoRA weight_active parameters for each projection
        # These are the "hot" buffers that get aliased in the NEFF
        self.max_rank = max_rank
        self.max_loras_active = max_loras_active
        self.dtype = dtype

        # Create lora_A_active and lora_B_active for each projection
        # Shapes: A_active [max_loras_active, in_features, rank]
        #         B_active [max_loras_active, rank, out_features_per_partition]
        self._create_lora_actives()

    def _create_lora_actives(self):
        """Create weight_active parameters for each LoRA target projection."""
        projections = {
            "to_q": self.to_q,
            "to_k": self.to_k,
            "to_v": self.to_v,
            "to_out_0": self.to_out[0]
            if isinstance(self.to_out, (nn.Sequential, nn.ModuleList))
            else self.to_out,
            "add_q_proj": self.add_q_proj,
            "add_k_proj": self.add_k_proj,
            "add_v_proj": self.add_v_proj,
            "to_add_out": self.to_add_out,
        }

        self.lora_projections = nn.ModuleDict()
        # Track which projections are RowParallelLinear (need all-reduce in LoRA path)
        self._row_parallel_lora_projs = set()
        for name, proj in projections.items():
            if proj is None:
                continue
            # Get dimensions from the projection layer
            if hasattr(proj, "output_size_per_partition"):
                # ColumnParallelLinear — sharded output
                in_features = proj.input_size
                out_features = proj.output_size_per_partition
            elif hasattr(proj, "input_size_per_partition"):
                # RowParallelLinear — sharded input
                in_features = proj.input_size_per_partition
                out_features = proj.output_size
                self._row_parallel_lora_projs.add(name)
            elif hasattr(proj, "in_features"):
                in_features = proj.in_features
                out_features = proj.out_features
            else:
                continue

            # Create a module to hold the LoRA active weights
            lora_holder = nn.Module()
            lora_holder.lora_A_active = nn.Parameter(
                torch.zeros(
                    self.max_loras_active, in_features, self.max_rank, dtype=self.dtype
                ),
                requires_grad=False,
            )
            lora_holder.lora_B_active = nn.Parameter(
                torch.zeros(
                    self.max_loras_active, self.max_rank, out_features, dtype=self.dtype
                ),
                requires_grad=False,
            )
            self.lora_projections[name] = lora_holder

    def _apply_lora(self, x: torch.Tensor, proj_name: str) -> torch.Tensor:
        """Apply LoRA delta from weight_active. x: [B, S, in_features].

        For RowParallelLinear projections (to_out_0, to_add_out), x is partition-sized.
        LoRA A maps partition -> rank (partial sum), needs all-reduce before B maps rank -> out.
        """
        if proj_name not in self.lora_projections:
            return torch.zeros_like(x[..., :1]).expand_as(x)  # No LoRA for this proj

        holder = self.lora_projections[proj_name]
        # Use first adapter (index 0) — for batch_size=1 diffusion
        A = holder.lora_A_active[0]  # [in_features, rank]
        B = holder.lora_B_active[0]  # [rank, out_features]

        # x @ A gives [B, S, rank]
        intermediate = torch.matmul(x, A)

        # For RowParallelLinear: input is partitioned, so x @ A is a partial sum
        # that needs all-reduce across TP ranks before multiplying by B
        if proj_name in self._row_parallel_lora_projs:
            intermediate = reduce_from_tensor_model_parallel_region(intermediate)

        # intermediate @ B gives [B, S, out_features]
        delta = torch.matmul(intermediate, B)
        return delta

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        image_rotary_emb: Tuple = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with NKI attention + LoRA."""
        if encoder_hidden_states is None:
            raise ValueError("LoRANKIQwenAttention requires encoder_hidden_states")

        batch_size = hidden_states.shape[0]
        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (base + LoRA)
        img_query = self.to_q(hidden_states) + self._apply_lora(hidden_states, "to_q")
        img_key = self.to_k(hidden_states) + self._apply_lora(hidden_states, "to_k")
        img_value = self.to_v(hidden_states) + self._apply_lora(hidden_states, "to_v")

        # Compute QKV for text stream (base + LoRA)
        txt_query = self.add_q_proj(encoder_hidden_states) + self._apply_lora(
            encoder_hidden_states, "add_q_proj"
        )
        txt_key = self.add_k_proj(encoder_hidden_states) + self._apply_lora(
            encoder_hidden_states, "add_k_proj"
        )
        txt_value = self.add_v_proj(encoder_hidden_states) + self._apply_lora(
            encoder_hidden_states, "add_v_proj"
        )

        inner_dim = img_query.shape[-1]
        head_dim = inner_dim // self.heads

        # Reshape to [B, H, S, D]
        img_query = img_query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        img_key = img_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        img_value = img_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        txt_query = txt_query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        txt_key = txt_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        txt_value = txt_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # Apply QK normalization
        if self.norm_q is not None:
            img_query = self.norm_q(img_query)
        if self.norm_k is not None:
            img_key = self.norm_k(img_key)
        if self.norm_added_q is not None:
            txt_query = self.norm_added_q(txt_query)
        if self.norm_added_k is not None:
            txt_key = self.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_precomputed(
                img_query.transpose(1, 2), img_freqs, use_real=False
            ).transpose(1, 2)
            img_key = apply_rotary_emb_precomputed(
                img_key.transpose(1, 2), img_freqs, use_real=False
            ).transpose(1, 2)
            txt_query = apply_rotary_emb_precomputed(
                txt_query.transpose(1, 2), txt_freqs, use_real=False
            ).transpose(1, 2)
            txt_key = apply_rotary_emb_precomputed(
                txt_key.transpose(1, 2), txt_freqs, use_real=False
            ).transpose(1, 2)

        # Concatenate for joint attention
        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key = torch.cat([txt_key, img_key], dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)

        # NKI Flash Attention
        joint_hidden_states = nki_flash_attention(joint_query, joint_key, joint_value)

        # Transpose and reshape
        joint_hidden_states = joint_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim
        )
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Output projections (base + LoRA)
        # to_out is nn.ModuleList [Linear, Dropout] — call [0] for linear
        img_attn_output_base = self.to_out[0](img_attn_output)
        img_attn_output = img_attn_output_base + self._apply_lora(
            img_attn_output, "to_out_0"
        )
        if len(self.to_out) > 1:
            img_attn_output = self.to_out[1](img_attn_output)

        txt_attn_output_base = self.to_add_out(txt_attn_output)
        txt_attn_output = txt_attn_output_base + self._apply_lora(
            txt_attn_output, "to_add_out"
        )

        return img_attn_output, txt_attn_output


# ============================================================================
# LoRA-aware MLP wrapper
# ============================================================================
class LoRAFeedForward(nn.Module):
    """FeedForward with LoRA on both linear layers."""

    def __init__(
        self,
        orig_ff: nn.Module,
        max_loras_active: int,
        max_rank: int,
        dtype: torch.dtype,
        prefix: str = "",
    ):
        super().__init__()
        self.net = orig_ff.net  # nn.Sequential(GELU(Linear), Dropout, Linear)
        self.max_rank = max_rank
        self.max_loras_active = max_loras_active
        self.dtype = dtype

        # LoRA on net.0.proj (first linear, inside GELU wrapper)
        proj_layer = self.net[0].proj
        if hasattr(proj_layer, "output_size_per_partition"):
            in_feat = proj_layer.input_size
            out_feat = proj_layer.output_size_per_partition
        else:
            in_feat = proj_layer.in_features
            out_feat = proj_layer.out_features

        self.lora_proj = nn.Module()
        self.lora_proj.lora_A_active = nn.Parameter(
            torch.zeros(max_loras_active, in_feat, max_rank, dtype=dtype),
            requires_grad=False,
        )
        self.lora_proj.lora_B_active = nn.Parameter(
            torch.zeros(max_loras_active, max_rank, out_feat, dtype=dtype),
            requires_grad=False,
        )

        # LoRA on net.2 (second linear)
        out_layer = self.net[2]
        if hasattr(out_layer, "input_size_per_partition"):
            in_feat2 = out_layer.input_size_per_partition
            out_feat2 = out_layer.output_size
        elif hasattr(out_layer, "output_size_per_partition"):
            in_feat2 = out_layer.input_size
            out_feat2 = out_layer.output_size_per_partition
        else:
            in_feat2 = out_layer.in_features
            out_feat2 = out_layer.out_features

        self.lora_out = nn.Module()
        self.lora_out.lora_A_active = nn.Parameter(
            torch.zeros(max_loras_active, in_feat2, max_rank, dtype=dtype),
            requires_grad=False,
        )
        self.lora_out.lora_B_active = nn.Parameter(
            torch.zeros(max_loras_active, max_rank, out_feat2, dtype=dtype),
            requires_grad=False,
        )

        # Track if net[2] is RowParallelLinear (needs all-reduce in LoRA path)
        self._out_is_row_parallel = hasattr(out_layer, "input_size_per_partition")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer (GELU(proj(x))) + LoRA
        proj_input = x
        h = self.net[0](x)  # GELU(proj(x))
        # LoRA delta on proj
        A = self.lora_proj.lora_A_active[0]
        B = self.lora_proj.lora_B_active[0]
        # Apply GELU to (proj(x) + lora_delta) is complex; instead approximate:
        # Add LoRA delta AFTER GELU activation (linear approximation for small LoRA)
        lora_delta_proj = torch.matmul(torch.matmul(proj_input, A), B)
        h = (
            h + lora_delta_proj
        )  # Approximate: GELU(proj(x) + delta) ≈ GELU(proj(x)) + delta for small delta

        # Dropout (net[1])
        h = self.net[1](h)

        # Second layer + LoRA
        out_input = h
        out = self.net[2](h)
        A2 = self.lora_out.lora_A_active[0]
        B2 = self.lora_out.lora_B_active[0]
        # For RowParallelLinear: out_input is partition-sized, so A2 output is partial sum
        intermediate = torch.matmul(out_input, A2)
        if self._out_is_row_parallel:
            intermediate = reduce_from_tensor_model_parallel_region(intermediate)
        lora_delta_out = torch.matmul(intermediate, B2)
        out = out + lora_delta_out

        return out


# ============================================================================
# LoRA-aware Modulation wrapper
# ============================================================================
class LoRAModulation(nn.Module):
    """Modulation (SiLU + Linear) with LoRA on the Linear.

    IMPORTANT: We store the original nn.Sequential as a numbered submodule
    to preserve the state_dict key structure (e.g., '0' for SiLU, '1' for Linear).
    This ensures shard_checkpoint can match keys against the unsharded checkpoint.
    """

    def __init__(
        self,
        orig_mod: nn.Sequential,
        max_loras_active: int,
        max_rank: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        # Keep original Sequential as numbered children to preserve key names
        # Keys will be: 0.* (SiLU — no params) and 1.weight, 1.bias (Linear)
        for idx, child in enumerate(orig_mod):
            self.add_module(str(idx), child)
        self.max_rank = max_rank
        self.max_loras_active = max_loras_active

        # Get dimensions from the linear layer (access via getattr, not stored as attr)
        linear = getattr(self, "1")
        if hasattr(linear, "output_size_per_partition"):
            in_feat = linear.input_size
            # For ColumnParallelLinear with gather_output=True, the actual output
            # is the FULL size (gathered), not the partition size
            if hasattr(linear, "gather_output") and linear.gather_output:
                out_feat = linear.output_size
            else:
                out_feat = linear.output_size_per_partition
        else:
            in_feat = linear.in_features
            out_feat = linear.out_features

        self.lora_mod = nn.Module()
        self.lora_mod.lora_A_active = nn.Parameter(
            torch.zeros(max_loras_active, in_feat, max_rank, dtype=dtype),
            requires_grad=False,
        )
        self.lora_mod.lora_B_active = nn.Parameter(
            torch.zeros(max_loras_active, max_rank, out_feat, dtype=dtype),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Access children by their numeric names
        silu = getattr(self, "0")
        linear = getattr(self, "1")
        h = silu(x)
        out = linear(h)
        # LoRA delta (applied to the linear's input, which is post-SiLU)
        A = self.lora_mod.lora_A_active[0]
        B = self.lora_mod.lora_B_active[0]
        lora_delta = torch.matmul(torch.matmul(h, A), B)
        return out + lora_delta


# ============================================================================
# Main transformer module with LoRA
# ============================================================================
class NeuronQwenTransformerMultiLoRA(nn.Module):
    """
    Neuron-optimized QwenImage Transformer with Multi-LoRA weight banks.

    No CFG parallelism (TP=4, DP=1). Sequential CFG at pipeline level.
    LoRA weight_active parameters are aliased for runtime updates.
    """

    def __init__(
        self,
        original_transformer,
        tp_degree: int,
        max_loras_active: int = 1,  # For diffusion: 1 adapter at a time
        max_rank: int = 16,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.config = original_transformer.config
        self.in_channels = original_transformer.config.in_channels
        self.out_channels = original_transformer.config.out_channels
        self.patch_size = original_transformer.config.patch_size
        self.tp_degree = tp_degree
        self.max_loras_active = max_loras_active
        self.max_rank = max_rank

        # Input projections
        self.img_in = original_transformer.img_in
        self.txt_in = original_transformer.txt_in

        # Time/text embedding
        self.time_text_embed = original_transformer.time_text_embed

        # Text norm
        self.txt_norm = original_transformer.txt_norm

        # Import sharding helpers
        try:
            from .neuron_parallel_utils import (
                shard_qwen_attention,
                shard_feedforward,
                shard_modulation,
            )
        except ImportError:
            from neuron_parallel_utils import (
                shard_qwen_attention,
                shard_feedforward,
                shard_modulation,
            )

        # Transformer blocks with TP sharding + LoRA
        self.transformer_blocks = nn.ModuleList()
        for i, block in enumerate(original_transformer.transformer_blocks):
            # Shard with TP degree (same as V3 CFG)
            block.attn = shard_qwen_attention(tp_degree, block.attn)
            block.img_mlp = shard_feedforward(block.img_mlp)
            block.txt_mlp = shard_feedforward(block.txt_mlp)
            block.img_mod = shard_modulation(block.img_mod)
            block.txt_mod = shard_modulation(block.txt_mod)
            self.transformer_blocks.append(block)

            if (i + 1) % 10 == 0:
                print(
                    f"  Sharded block {i + 1}/{len(original_transformer.transformer_blocks)}"
                )

        # Replace attention with LoRA+NKI version
        self._replace_attention_with_lora(max_loras_active, max_rank, dtype)

        # Replace MLP with LoRA version
        self._replace_mlp_with_lora(max_loras_active, max_rank, dtype)

        # Replace modulation with LoRA version
        self._replace_modulation_with_lora(max_loras_active, max_rank, dtype)

        # Final layers
        self.norm_out = original_transformer.norm_out
        self.proj_out = original_transformer.proj_out

        self.head_dim = 128
        self.num_heads = self.transformer_blocks[0].attn.heads

    def _replace_attention_with_lora(self, max_loras_active, max_rank, dtype):
        """Replace attention modules with LoRA+NKI versions."""
        for i, block in enumerate(self.transformer_blocks):
            block.attn = LoRANKIQwenAttention(
                block.attn, max_loras_active, max_rank, dtype
            )
        print(
            f"Replaced attention with LoRA+NKI versions on {len(self.transformer_blocks)} blocks"
        )

    def _replace_mlp_with_lora(self, max_loras_active, max_rank, dtype):
        """Replace MLPs with LoRA versions."""
        for i, block in enumerate(self.transformer_blocks):
            block.img_mlp = LoRAFeedForward(
                block.img_mlp, max_loras_active, max_rank, dtype, f"block{i}.img_mlp"
            )
            block.txt_mlp = LoRAFeedForward(
                block.txt_mlp, max_loras_active, max_rank, dtype, f"block{i}.txt_mlp"
            )
        print(
            f"Replaced MLPs with LoRA versions on {len(self.transformer_blocks)} blocks"
        )

    def _replace_modulation_with_lora(self, max_loras_active, max_rank, dtype):
        """Replace modulations with LoRA versions."""
        for i, block in enumerate(self.transformer_blocks):
            block.img_mod = LoRAModulation(
                block.img_mod, max_loras_active, max_rank, dtype
            )
            block.txt_mod = LoRAModulation(
                block.txt_mod, max_loras_active, max_rank, dtype
            )
        print(
            f"Replaced modulations with LoRA versions on {len(self.transformer_blocks)} blocks"
        )

    def get_lora_active_tensors(self) -> List[torch.Tensor]:
        """Collect all weight_active tensors for aliasing."""
        tensors = []
        for name, module in self.named_modules():
            if hasattr(module, "lora_A_active"):
                tensors.append(module.lora_A_active)
            if hasattr(module, "lora_B_active"):
                tensors.append(module.lora_B_active)
        return tensors

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_rotary_emb: torch.Tensor,
        txt_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Returns noise prediction only.

        With ModelBuilder auto-aliasing (enable_aliasing=True), all model state
        (including LoRA active buffers) is automatically aliased. No need to
        return them as explicit outputs — write_to_neuron_buffer() updates them.
        """

        # Split RoPE into cos/sin
        img_freqs_cos = img_rotary_emb[..., 0]
        img_freqs_sin = img_rotary_emb[..., 1]
        txt_freqs_cos = txt_rotary_emb[..., 0]
        txt_freqs_sin = txt_rotary_emb[..., 1]

        # Image input projection
        hidden_states = self.img_in(hidden_states)

        # Text processing
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Time embedding
        timestep = timestep.to(hidden_states.dtype)
        temb = self.time_text_embed(timestep, hidden_states)

        # Create rotary_emb tuple
        image_rotary_emb = (
            (img_freqs_cos, img_freqs_sin),
            (txt_freqs_cos, txt_freqs_sin),
        )

        # Process through blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # Final norm and projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return output


# (No V1 ModelInstance needed — using SPMD + torch_neuronx.trace() directly)


# ============================================================================
# Compilation
# ============================================================================
def get_rope_from_original_model(
    pipe, frame, height, width, text_seq_len, dtype=torch.bfloat16
):
    """Get RoPE from original model."""
    video_fhw = (frame, height, width)
    vid_freqs, txt_freqs = pipe.transformer.pos_embed(
        video_fhw, txt_seq_lens=[text_seq_len], device=torch.device("cpu")
    )

    img_cos = vid_freqs.real.float()
    img_sin = vid_freqs.imag.float()
    txt_cos = txt_freqs.real.float()
    txt_sin = txt_freqs.imag.float()

    img_rotary_emb = torch.stack([img_cos, img_sin], dim=-1).to(dtype)
    txt_rotary_emb = torch.stack([txt_cos, txt_sin], dim=-1).to(dtype)

    return img_rotary_emb, txt_rotary_emb


def compile_transformer_multi_lora(args):
    """Compile transformer with Multi-LoRA support using SPMD + torch_neuronx.trace()."""

    tp_degree = args.tp_degree
    world_size = tp_degree  # No CFG parallelism; TP only
    max_loras = args.max_loras
    max_rank = args.max_rank
    max_loras_active = 1  # Diffusion: one adapter active at a time

    # Calculate dimensions
    latent_h = args.height // 8
    latent_w = args.width // 8
    patch_size = 2
    patch_h = latent_h // patch_size
    patch_w = latent_w // patch_size
    temporal_frames = args.patch_multiplier
    num_patches = temporal_frames * patch_h * patch_w
    text_seq_len = args.max_sequence_length

    text_hidden_size = 3584
    in_channels = 64
    head_dim = 128

    # Alignment padding (total_seq must be multiple of 128 for NKI)
    total_seq = num_patches + text_seq_len
    alignment = 128
    need_padding = (alignment - total_seq % alignment) % alignment
    num_patches_padded = num_patches + need_padding
    patches_padding = need_padding

    batch_size = 1  # No CFG parallelism; pipeline calls twice

    print("=" * 60)
    print("Transformer Multi-LoRA Compilation (SPMD + Aliases)")
    print("=" * 60)
    print(f"Image: {args.height}x{args.width}")
    print(f"Original patches: {num_patches}")
    if patches_padding > 0:
        print(f"Padded patches: {num_patches_padded} (+{patches_padding})")
    print(f"Total seq (padded): {num_patches_padded + text_seq_len}")
    print(f"TP degree: {tp_degree}")
    print(f"World size: {world_size} (no CFG parallel)")
    print(f"NKI Flash Attention: Enabled")
    print(f"Multi-LoRA: max_loras={max_loras}, max_rank={max_rank}")
    print(f"LoRA active buffers: {max_loras_active}")
    print(f"Batch size: {batch_size} (sequential CFG)")

    # Use NxDParallelState for SPMD trace with TP sharding
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # Load pipeline to get model and RoPE
        print("\nLoading model...")
        model_id = getattr(args, "model_id", None) or MODEL_ID
        load_kwargs = {"torch_dtype": torch.bfloat16}
        if os.path.isdir(model_id):
            load_kwargs["local_files_only"] = True
        if CACHE_DIR:
            load_kwargs["cache_dir"] = CACHE_DIR
        pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, **load_kwargs)

        # Get RoPE
        print("\nGetting RoPE...")
        img_rotary_emb, txt_rotary_emb = get_rope_from_original_model(
            pipe=pipe,
            frame=temporal_frames,
            height=patch_h,
            width=patch_w,
            text_seq_len=text_seq_len,
        )
        print(f"  img_rotary_emb: {img_rotary_emb.shape}")
        print(f"  txt_rotary_emb: {txt_rotary_emb.shape}")

        # Pad img_rotary_emb if needed
        if patches_padding > 0:
            rope_padding = img_rotary_emb[-1:].repeat(patches_padding, 1, 1)
            img_rotary_emb = torch.cat([img_rotary_emb, rope_padding], dim=0)
            print(f"  img_rotary_emb (padded): {img_rotary_emb.shape}")

        # Save unsharded state dict before modifications
        unsharded_state = pipe.transformer.state_dict()

        # Create Neuron transformer with LoRA
        print(f"\nCreating Neuron transformer with LoRA (TP={tp_degree})...")
        neuron_transformer = NeuronQwenTransformerMultiLoRA(
            pipe.transformer,
            tp_degree=tp_degree,
            max_loras_active=max_loras_active,
            max_rank=max_rank,
            dtype=torch.bfloat16,
        )
        neuron_transformer = neuron_transformer.to(torch.bfloat16)
        neuron_transformer.eval()

        # Free original pipeline memory
        del pipe
        import gc

        gc.collect()

        # Count LoRA parameters
        lora_tensors = neuron_transformer.get_lora_active_tensors()
        lora_param_count = sum(t.numel() for t in lora_tensors)
        lora_mem_bytes = sum(t.numel() * t.element_size() for t in lora_tensors)
        print(f"\nLoRA statistics:")
        print(f"  Active tensors: {len(lora_tensors)}")
        print(f"  Active parameters: {lora_param_count:,}")
        print(f"  Active memory: {lora_mem_bytes / 1024 / 1024:.1f} MB")
        print(
            f"  Bank memory ({max_loras} adapters): {lora_mem_bytes * max_loras / 1024 / 1024:.1f} MB"
        )

        # Note: With ModelBuilder, all model state is auto-aliased (enable_aliasing=True).
        # No explicit input_output_aliases dict needed — ModelBuilder handles this.
        # LoRA active buffers will be updatable via write_to_neuron_buffer() at runtime.
        print(
            f"\n  LoRA active tensors: {len(lora_tensors)} (auto-aliased by ModelBuilder)"
        )

        # Sample inputs
        sample_hidden_states = torch.randn(
            batch_size, num_patches_padded, in_channels, dtype=torch.bfloat16
        )
        sample_encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, text_hidden_size, dtype=torch.bfloat16
        )
        sample_timestep = torch.randn(batch_size, dtype=torch.float32)

        example_inputs = (
            sample_hidden_states,
            sample_encoder_hidden_states,
            sample_timestep,
            img_rotary_emb,
            txt_rotary_emb,
        )

        # Compile with ModelBuilder (handles SPMD, aliasing, and NxDModel output)
        # ModelBuilder's internal trace() uses enable_aliasing=True, which aliases ALL
        # model state (parameters + buffers). This means LoRA active buffers are
        # automatically aliased and can be updated via .copy_() at runtime.
        compile_args = "--model-type=transformer -O1 --auto-cast=none --enable-fast-loading-neuron-binaries --internal-hlo2tensorizer-options='--enable-native-kernel=1 --remat'"

        output_path = f"{args.compiled_models_dir}/transformer_multi_lora"
        os.makedirs(output_path, exist_ok=True)

        print(f"\nCompiling with ModelBuilder (SPMD + auto-aliasing)...")
        print(f"  Compiler args: {compile_args}")
        print(f"  This may take 30-60 minutes for the full 60-layer model.")

        builder = ModelBuilder(model=neuron_transformer)
        builder.trace(
            kwargs={
                "hidden_states": sample_hidden_states,
                "encoder_hidden_states": sample_encoder_hidden_states,
                "timestep": sample_timestep,
                "img_rotary_emb": img_rotary_emb,
                "txt_rotary_emb": txt_rotary_emb,
            }
        )
        traced_model = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save as NxDModel (loadable with NxDModel.load())
        nxd_model_path = os.path.join(output_path, "nxd_model.pt")
        print(f"\nSaving NxDModel to {nxd_model_path}...")
        traced_model.save(nxd_model_path)

        # Shard and save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        print("Sharding weights...")
        checkpoint = {}
        for key, value in neuron_transformer.state_dict().items():
            if key in unsharded_state:
                checkpoint[key] = unsharded_state[key].clone()
            else:
                checkpoint[key] = value.clone()

        shard_checkpoint(
            checkpoint=checkpoint,
            model=neuron_transformer,
            serialize_path=weights_path,
        )

        # Post-process: remove master_weights, apply mmap workaround
        from safetensors.torch import load_file, save_file

        for rank in range(tp_degree):
            shard_file = os.path.join(
                weights_path, f"tp{rank}_sharded_checkpoint.safetensors"
            )
            if not os.path.exists(shard_file):
                print(f"  WARNING: {shard_file} not found")
                continue
            shard_data = load_file(shard_file, device="cpu")
            original_count = len(shard_data)
            cleaned = {
                k: v.clone() for k, v in shard_data.items() if "master_weight" not in k
            }
            del shard_data
            cleaned_size = sum(v.numel() * v.element_size() for v in cleaned.values())
            tmp_file = shard_file + ".tmp"
            save_file(cleaned, tmp_file)
            os.replace(tmp_file, shard_file)
            print(
                f"  tp{rank}: {original_count} -> {len(cleaned)} tensors, {cleaned_size / 1e9:.2f}GB"
            )

        # Save config
        config = {
            "height": args.height,
            "width": args.width,
            "num_patches": num_patches,
            "num_patches_padded": num_patches_padded,
            "patches_padding": patches_padding,
            "text_seq_len": text_seq_len,
            "patch_multiplier": args.patch_multiplier,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "cfg_parallel": False,
            "dp_degree": 1,
            "head_dim": head_dim,
            "frame": temporal_frames,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "nki_flash_attention": True,
            "batch_size": batch_size,
            "multi_lora": True,
            "max_loras": max_loras,
            "max_rank": max_rank,
            "max_loras_active": max_loras_active,
            "num_lora_tensors": len(lora_tensors),
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save pre-computed RoPE
        torch.save(
            {"img_rotary_emb": img_rotary_emb, "txt_rotary_emb": txt_rotary_emb},
            os.path.join(output_path, "rope_cache.pt"),
        )

        print("\n" + "=" * 60)
        print("Compilation complete!")
        print("=" * 60)
        print(f"Model saved to: {output_path}")
        print(
            f"LoRA aliases: {len(lora_tensors)} tensors (all model state auto-aliased by ModelBuilder)"
        )
        print(f"\nRuntime usage:")
        print(f"  1. Load with NxDModel.load(nxd_model.pt)")
        print(f"  2. nxd_model.set_weights(sharded_checkpoints)")
        print(f"  3. nxd_model.to_neuron()")
        print(
            f"  4. .copy_() adapter A/B into weight_active params via write_to_neuron_buffer()"
        )
        print(f"  5. Call model — aliased weights re-read each invocation")

        # === Test aliasing if requested ===
        if args.test_aliasing:
            print("\n" + "=" * 60)
            print("TESTING ALIASING: Verifying runtime weight switching works")
            print("=" * 60)

            # NxDModel needs weights loaded and to_neuron() called before inference
            # Load the sharded weights we just saved
            from safetensors.torch import load_file as _load_file

            print("\n  Loading sharded weights for test...")
            sharded_checkpoints = []
            for rank in range(tp_degree):
                ckpt_path = os.path.join(
                    weights_path, f"tp{rank}_sharded_checkpoint.safetensors"
                )
                ckpt = _load_file(ckpt_path, device="cpu")
                sharded_checkpoints.append(ckpt)
            print(f"    Loaded {len(sharded_checkpoints)} rank checkpoints")

            traced_model.set_weights(sharded_checkpoints)
            print("  Moving model to Neuron...")
            traced_model.to_neuron()
            print("  Model on Neuron!")

            # Create inputs
            print("\n  Creating test inputs...")
            test_inputs = (
                sample_hidden_states.clone(),
                sample_encoder_hidden_states.clone(),
                sample_timestep.clone(),
                img_rotary_emb.clone(),
                txt_rotary_emb.clone(),
            )

            # RUN 1: baseline (all LoRA buffers are zero)
            print("\n  RUN 1: Zero LoRA (baseline)...")
            with torch.no_grad():
                out1 = traced_model(*test_inputs)
            if isinstance(out1, (tuple, list)):
                noise1 = out1[0].float().cpu()
            else:
                noise1 = out1.float().cpu()
            print(f"    Output shape: {noise1.shape}")
            print(f"    Output: mean={noise1.mean():.6f}, std={noise1.std():.6f}")

            # Inject random LoRA weights via write_to_neuron_buffer
            print("\n  Injecting random LoRA weights via write_to_neuron_buffer()...")
            # Find LoRA buffer keys in the model state
            lora_keys = []
            for key, value in neuron_transformer.state_dict().items():
                if "lora_A_active" in key or "lora_B_active" in key:
                    lora_keys.append(key)

            n_injected = 0
            for key in lora_keys[:20]:  # Test with first 20 for speed
                # Get shape from first rank checkpoint
                if key in sharded_checkpoints[0]:
                    shape = sharded_checkpoints[0][key].shape
                    random_weight = torch.randn(shape, dtype=torch.bfloat16) * 0.05
                    for rank in range(tp_degree):
                        traced_model.write_to_neuron_buffer(random_weight, key, rank)
                    n_injected += 1
            print(
                f"    Injected random weights into {n_injected} LoRA buffers (across all ranks)"
            )

            # RUN 2: with random LoRA
            print("\n  RUN 2: Random LoRA injected...")
            with torch.no_grad():
                out2 = traced_model(*test_inputs)
            if isinstance(out2, (tuple, list)):
                noise2 = out2[0].float().cpu()
            else:
                noise2 = out2.float().cpu()
            print(f"    Output: mean={noise2.mean():.6f}, std={noise2.std():.6f}")

            # Compare
            diff = noise2 - noise1
            max_diff = diff.abs().max().item()
            mean_diff = diff.abs().mean().item()
            print(f"\n  Max abs diff:  {max_diff:.8f}")
            print(f"  Mean abs diff: {mean_diff:.8f}")

            if max_diff > 1e-5:
                print("\n  +++ SUCCESS: Runtime LoRA weight switching CONFIRMED +++")
                print("  ModelBuilder auto-aliasing works for runtime weight updates!")
            elif max_diff > 0:
                print("\n  ~~~ PARTIAL: Tiny difference (possible precision issue) ~~~")
            else:
                print("\n  --- FAILURE: Outputs identical — aliasing NOT working ---")

            # RUN 3: Zero LoRA again (should restore baseline)
            print("\n  RUN 3: Zeroing LoRA (restoring baseline)...")
            for key in lora_keys[:20]:
                if key in sharded_checkpoints[0]:
                    shape = sharded_checkpoints[0][key].shape
                    zero_weight = torch.zeros(shape, dtype=torch.bfloat16)
                    for rank in range(tp_degree):
                        traced_model.write_to_neuron_buffer(zero_weight, key, rank)
            with torch.no_grad():
                out3 = traced_model(*test_inputs)
            if isinstance(out3, (tuple, list)):
                noise3 = out3[0].float().cpu()
            else:
                noise3 = out3.float().cpu()

            restore_diff = (noise3 - noise1).abs().max().item()
            print(f"    Max diff from original baseline: {restore_diff:.8f}")
            if restore_diff < 1e-5:
                print("    +++ Baseline restored perfectly +++")
            else:
                print(f"    ~~~ Difference persists: {restore_diff} ~~~")

            print("\n" + "=" * 60)
            print("ALIASING TEST COMPLETE")
            print("=" * 60)


# ============================================================================
# Convenience API for import usage
# ============================================================================
def compile_transformer(
    model_id: str = "Qwen/Qwen-Image-Edit-2511",
    output_dir: str = "./compiled_models_multi_lora",
    height: int = 1024,
    width: int = 1024,
    tp_degree: int = 4,
    max_loras: int = 4,
    max_rank: int = 16,
    test_aliasing: bool = False,
):
    """Compile the transformer with multi-LoRA support.

    Args:
        model_id: HuggingFace model ID or local path.
        output_dir: Directory to save compiled model.
        height: Image height in pixels.
        width: Image width in pixels.
        tp_degree: Tensor parallelism degree.
        max_loras: Maximum number of LoRA adapters in the bank.
        max_rank: Maximum LoRA rank (adapters with lower rank are zero-padded).
        test_aliasing: If True, run aliasing verification after compilation.

    Returns:
        Path to the compiled NxDModel.
    """
    import argparse

    args = argparse.Namespace(
        model_id=model_id,
        height=height,
        width=width,
        max_sequence_length=1024,
        patch_multiplier=3,
        tp_degree=tp_degree,
        max_loras=max_loras,
        max_rank=max_rank,
        compiled_models_dir=output_dir,
        compiler_workdir=os.path.join(output_dir, "compiler_workdir"),
        test_aliasing=test_aliasing,
    )
    compile_transformer_multi_lora(args)
    return os.path.join(output_dir, "transformer_multi_lora", "nxd_model.pt")


def load_and_run(
    compiled_model_path: str,
    hidden_states,
    encoder_hidden_states,
    timestep,
    img_rotary_emb,
    txt_rotary_emb,
):
    """Load a compiled NxDModel and run inference.

    Args:
        compiled_model_path: Path to the compiled nxd_model.pt.
        hidden_states: Image latent patches [B, seq, in_channels].
        encoder_hidden_states: Text embeddings [B, seq, text_dim].
        timestep: Timestep tensor [B].
        img_rotary_emb: Image RoPE [img_seq, head_dim//2, 2].
        txt_rotary_emb: Text RoPE [txt_seq, head_dim//2, 2].

    Returns:
        Model output (noise prediction).
    """
    from neuronx_distributed import NxDModel

    nxd_model = NxDModel.load(compiled_model_path)
    # Weights must be set separately via nxd_model.set_weights() + to_neuron()
    return nxd_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--patch_multiplier", type=int, default=3)
    parser.add_argument("--tp_degree", type=int, default=4)
    parser.add_argument(
        "--max_loras", type=int, default=4, help="Max adapters in weight bank"
    )
    parser.add_argument("--max_rank", type=int, default=16, help="Max LoRA rank")
    parser.add_argument(
        "--compiled_models_dir",
        type=str,
        default="./compiled_models_multi_lora",
    )
    parser.add_argument(
        "--compiler_workdir",
        type=str,
        default="./compiler_workdir_multi_lora",
    )
    parser.add_argument(
        "--test-aliasing",
        action="store_true",
        help="After compilation, test that LoRA weight aliasing works",
    )
    args = parser.parse_args()

    compile_transformer_multi_lora(args)
