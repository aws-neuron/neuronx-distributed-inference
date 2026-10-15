"""OpenFold3 Neuron acceleration module.

Provides wrapper classes and compilation utilities for running OpenFold3
(AlphaFold3 reproduction, ~330M params) on AWS Trainium 2 hardware.
Uses vanilla torch_neuronx.trace() compilation (no NKI kernels) with
the replace_weights pattern for multi-layer stacks.

Two compilation strategies are supported:

  1. **Monolithic** (N <= 256): Each PairFormerBlock is traced as a
     single unit. Fast compilation, low overhead, 12.1x speedup per layer.

   2. **Decomposed** (N > 256, up to 2048): PairFormerBlock is split into
      sub-operations, each traced independently. Strategy auto-selected by N:
        - N=257-384: Fused TriMulOut+In + merged TriAttn/APB (5 calls/layer)
        - N=385-512: Proj+merged BMM/Output + merged TriAttn/APB (9 calls/layer)
        - N=513-1024: 3-seg TriMul + 2-seg TriAttn/APB (14 calls/layer)
        - N>1024: Same + chunked TriAttn MHA (14+2*ceil(N/128) calls/layer)

Five model components are compiled:
  1. PairFormerBlock (48 layers) - main trunk
  2. MSA block type A (3 blocks) - full MSA blocks 0-2
  3. MSA block type B (1 block) - last MSA block (different structure)
  4. TemplatePairBlock (2 blocks) - template embedder
  5. DiffusionConditioning._forward() (1 block, shared weights) - conditioning

Architecture constants (from OpenFold3):
  C_S = 384 (single representation)
  C_Z = 128 (pair representation)
  C_M = 64 (MSA representation)
  C_TOKEN = 768 (token/atom representation)
"""

import os
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# Architecture Constants
# ============================================================================

C_S = 384
C_Z = 128
C_M = 64
C_TOKEN = 768

# Chunked attention threshold and chunk size
CHUNKED_ATTN_THRESHOLD = 1024  # Use chunked MHA for N > this value
CHUNKED_ATTN_CHUNK_SIZE = 128  # Number of rows per MHA chunk

# Merged segment thresholds (validated on SDK 2.28)
# Below these N values, merged wrappers compile successfully and are faster.
# Above these, compiler crashes (NCC_ITEN404/NCC_INLA001) require decomposition.
FUSED_TRIMUL_OUT_IN_MAX_N = (
    384  # Fused TriMulOut+TriMulIn compiles at N<=384 (~30% faster)
)
MERGED_TRIMUL_MAX_N = 384  # Full TriMul compiles at N<=384 (1.57x vs 3-seg)
MERGED_TRIMUL_BMM_OUTPUT_MAX_N = 512  # BMM+Output merge compiles at N<=512 (1.12-1.13x)
MERGED_ATTN_MAX_N = 512  # Merged TriAttn/APB compile at N<=512 (1.68-1.79x)


# ============================================================================
# Monolithic Wrapper Modules (N <= 256)
# ============================================================================


class PairFormerBlockWrapper(nn.Module):
    """Wrapper for a single OpenFold3 PairFormerBlock.

    The PairFormerBlock contains triangle multiplicative updates,
    triangle attention, pair transition, and single attention.
    This wrapper exposes (s, z, single_mask, pair_mask) -> (s, z)
    for tracing, fixing boolean kwargs to evaluation defaults.

    Args:
        block: A PairFormerBlock from model.pairformer_stack.blocks[i]
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one pairformer layer.

        Args:
            s: [1, N, C_S] single representation
            z: [1, N, N, C_Z] pair representation
            single_mask: [1, N] mask
            pair_mask: [1, N, N] mask

        Returns:
            (s, z): updated representations
        """
        s_out, z_out = self.block(
            s=s,
            z=z,
            single_mask=single_mask,
            pair_mask=pair_mask,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
        )
        return s_out, z_out


class MSABlockWrapper(nn.Module):
    """Wrapper for a single OpenFold3 MSA block.

    OpenFold3 has two MSA block types:
      - Type A (blocks 0-2): Full blocks with msa_att_row, msa_transition,
        outer_product_mean, and pair_stack (60 params each).
      - Type B (block 3): Reduced block with only outer_product_mean and
        pair_stack (47 params). No msa_att_row or msa_transition.

    Both types accept the same interface: (m, z, msa_mask, pair_mask) -> (m, z).

    Args:
        block: An MSA block from model.msa_module.blocks[i]
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one MSA block.

        Args:
            m: [1, N_msa, N, C_M] MSA representation
            z: [1, N, N, C_Z] pair representation
            msa_mask: [1, N_msa, N] mask
            pair_mask: [1, N, N] mask

        Returns:
            (m, z): updated representations
        """
        m_out, z_out = self.block(
            m=m,
            z=z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
        )
        return m_out, z_out


class TemplatePairBlockWrapper(nn.Module):
    """Wrapper for a single OpenFold3 TemplatePairBlock.

    The template embedder contains 2 pairformer-style blocks that
    operate on template pair representations with c_t=64 (smaller
    than the main pairformer's c_z=128).

    Args:
        block: A TemplatePairBlock from
               model.template_embedder.template_pair_stack.blocks[i]
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run one template pairformer layer.

        Args:
            t: [1, N_templ, N, N, 64] template pair representation
            mask: [1, N_templ, N, N] template mask

        Returns:
            t: updated template pair representation
        """
        t_out = self.block(
            t=t,
            mask=mask,
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=False,
            use_lma=False,
            inplace_safe=False,
        )
        return t_out


class DiffCondForwardWrapper(nn.Module):
    """Wrapper for DiffusionConditioning._forward().

    The outer DiffusionConditioning forward() uses batch dict inputs
    (not traceable), but the inner _forward() is pure tensor math:
    transition layers applied to (si, zij, token_mask). This wrapper
    extracts just those layers for tracing.

    Args:
        diff_cond: The DiffusionConditioning module from
                   model.diffusion_module.diffusion_conditioning
    """

    def __init__(self, diff_cond):
        super().__init__()
        self.transition_z = diff_cond.transition_z
        self.transition_s = diff_cond.transition_s

    def forward(
        self,
        si: torch.Tensor,
        zij: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run diffusion conditioning transitions.

        Args:
            si: [1, N, C_S] single conditioning
            zij: [1, N, N, C_Z] pair conditioning
            token_mask: [1, N] mask

        Returns:
            (si, zij): updated conditioning tensors
        """
        pair_token_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_token_mask)
        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask)
        return si, zij


# ============================================================================
# Decomposed Sub-Op Wrappers (N > 256)
# ============================================================================


class TriMulProjectionWrapper(nn.Module):
    """TriMul Segment A: LayerNorm + projections + gating + permute.

    Splits the projection phase from the matmul to avoid SBUF overflow
    and NCC_ITEN404 compiler issues at large N.

    Input: z [1, N, N, C_Z]
    Output: (a [1, D, N, N], b [1, D, N, N], z_normed [1, N, N, C_Z])
    """

    def __init__(self, tri_mul):
        super().__init__()
        self.layer_norm_in = tri_mul.layer_norm_in
        self.linear_a_g = tri_mul.linear_a_g
        self.linear_a_p = tri_mul.linear_a_p
        self.linear_b_g = tri_mul.linear_b_g
        self.linear_b_p = tri_mul.linear_b_p
        self.sigmoid = nn.Sigmoid()
        self._outgoing = tri_mul._outgoing

    def forward(self, z):
        z_norm = self.layer_norm_in(z)
        a = self.sigmoid(self.linear_a_g(z_norm)) * self.linear_a_p(z_norm)
        b = self.sigmoid(self.linear_b_g(z_norm)) * self.linear_b_p(z_norm)
        if self._outgoing:
            a = a.permute(0, 3, 1, 2).contiguous()
            b = b.permute(0, 3, 2, 1).contiguous()
        else:
            a = a.permute(0, 3, 2, 1).contiguous()
            b = b.permute(0, 3, 1, 2).contiguous()
        return a, b, z_norm


class TriMulBmmWrapper(nn.Module):
    """TriMul Segment B: Bare batched matmul.

    Input: a [1, D, N, N], b [1, D, N, N]
    Output: p [1, D, N, N]
    """

    def forward(self, a, b):
        return torch.einsum("...ij,...jk->...ik", a, b)


class TriMulOutputWrapper(nn.Module):
    """TriMul Segment C: permute_back + LayerNorm + linear + gate.

    Input: p [1, D, N, N], z_norm [1, N, N, C_Z]
    Output: result [1, N, N, C_Z]
    """

    def __init__(self, tri_mul):
        super().__init__()
        self.layer_norm_out = tri_mul.layer_norm_out
        self.linear_z = tri_mul.linear_z
        self.linear_g = tri_mul.linear_g
        self.sigmoid = nn.Sigmoid()

    def forward(self, p, z_norm):
        p = p.permute(0, 2, 3, 1).contiguous()
        p = self.layer_norm_out(p)
        p = self.linear_z(p)
        g = self.sigmoid(self.linear_g(z_norm))
        return p * g


class TriAttnBiasWrapper(nn.Module):
    """TriAttn Segment A: LayerNorm + bias computation.

    Separating the bias computation from MHA avoids NCC_ITEN404 at N > 512.

    Input: x [1, N, N, C_Z]
    Output: (x_normed [1, N, N, C_Z], triangle_bias [1, 1, H, N, N])
    """

    def __init__(self, tri_attn):
        super().__init__()
        self.layer_norm = tri_attn.layer_norm
        self.linear_z = tri_attn.linear_z

    def forward(self, x):
        x_normed = self.layer_norm(x)
        triangle_bias = self.linear_z(x_normed)
        triangle_bias = triangle_bias.permute(0, 3, 1, 2)
        triangle_bias = triangle_bias.unsqueeze(1)
        return x_normed, triangle_bias


class TriAttnMHAWrapper(nn.Module):
    """TriAttn Segment B: Full MHA (non-chunked, for N <= 1024).

    Input: (x_normed [1, N, N, C_Z], triangle_bias [1, 1, H, N, N])
    Output: attn_output [1, N, N, C_Z]
    """

    def __init__(self, tri_attn):
        super().__init__()
        self.mha = tri_attn.mha

    def forward(self, x_normed, triangle_bias):
        return self.mha(
            q_x=x_normed,
            kv_x=x_normed,
            biases=[triangle_bias],
            use_deepspeed_evo_attention=False,
            use_lma=False,
            use_cueq_triangle_kernels=False,
        )


class TriAttnMHAChunkedWrapper(nn.Module):
    """TriAttn Segment B (chunked): MHA for a chunk of rows.

    For N > 1024, the full attention scores tensor [1, N, H, N, N] exceeds
    24 GB HBM. Chunking over rows (each row's attention is independent)
    is mathematically exact and fits in memory.

    Input: (x_chunk [1, chunk_size, N, C_Z], triangle_bias [1, 1, H, N, N])
    Output: attn_output [1, chunk_size, N, C_Z]
    """

    def __init__(self, tri_attn):
        super().__init__()
        self.mha = tri_attn.mha

    def forward(self, x_chunk, triangle_bias):
        return self.mha(
            q_x=x_chunk,
            kv_x=x_chunk,
            biases=[triangle_bias],
            use_deepspeed_evo_attention=False,
            use_lma=False,
            use_cueq_triangle_kernels=False,
        )


class AttnPairBiasBiasWrapper(nn.Module):
    """AttnPairBias Segment A: LayerNorm + bias computation from z.

    Input: (a [1, N, C_S], z [1, N, N, C_Z])
    Output: (a_normed [1, N, C_S], pair_bias [1, H, N, N])
    """

    def __init__(self, apb):
        super().__init__()
        self.layer_norm_a = apb.layer_norm_a
        self.layer_norm_z = apb.layer_norm_z
        self.linear_z = apb.linear_z

    def forward(self, a, z):
        a_normed = self.layer_norm_a(a)
        z_normed = self.layer_norm_z(z)
        pair_bias = self.linear_z(z_normed)
        pair_bias = pair_bias.permute(0, 3, 1, 2)
        return a_normed, pair_bias


class AttnPairBiasMHAWrapper(nn.Module):
    """AttnPairBias Segment B: MHA with pre-computed pair bias.

    Input: (a_normed [1, N, C_S], pair_bias [1, H, N, N])
    Output: attn_output [1, N, C_S]
    """

    def __init__(self, apb):
        super().__init__()
        self.mha = apb.mha

    def forward(self, a_normed, pair_bias):
        return self.mha(
            q_x=a_normed,
            kv_x=a_normed,
            biases=[pair_bias],
            use_deepspeed_evo_attention=False,
            use_lma=False,
            use_cueq_triangle_kernels=False,
        )


class PairTransitionWrapper(nn.Module):
    """Monolithic PairTransition wrapper (compiles at all N sizes).

    Includes the residual add to match the PairFormerBlock behavior.

    Input: z [1, N, N, C_Z]
    Output: z + pair_transition(z)
    """

    def __init__(self, pt):
        super().__init__()
        self.pair_transition = pt

    def forward(self, z):
        return z + self.pair_transition(z, mask=None)


class SingleTransitionWrapper(nn.Module):
    """Monolithic SingleTransition wrapper (compiles at all N sizes).

    Input: s [1, N, C_S]
    Output: s + single_transition(s)
    """

    def __init__(self, st):
        super().__init__()
        self.single_transition = st

    def forward(self, s):
        return s + self.single_transition(s, mask=None)


# ============================================================================
# Merged Sub-Op Wrappers (N <= MERGED_*_MAX_N)
# ============================================================================
# These wrappers merge multiple sub-op segments into single traces,
# eliminating HBM materialization of intermediates. They compile at
# N <= 512 on SDK 2.28 but crash the compiler at larger N.


class TriMulFullWrapper(nn.Module):
    """Full TriMul in one trace: LayerNorm + projections + gate + einsum + output.

    Eliminates all intermediate HBM materializations. Compiles at N <= 384
    and is 1.57x faster than 3-segment at N=384. At N=512, compiles but
    is 0.69x slower than 3-segment due to compiler graph complexity.

    Input: z [1, N, N, C_Z]
    Output: result [1, N, N, C_Z]
    """

    def __init__(self, tri_mul):
        super().__init__()
        self.layer_norm_in = tri_mul.layer_norm_in
        self.linear_a_g = tri_mul.linear_a_g
        self.linear_a_p = tri_mul.linear_a_p
        self.linear_b_g = tri_mul.linear_b_g
        self.linear_b_p = tri_mul.linear_b_p
        self.layer_norm_out = tri_mul.layer_norm_out
        self.linear_z = tri_mul.linear_z
        self.linear_g = tri_mul.linear_g
        self.sigmoid = nn.Sigmoid()
        self._outgoing = tri_mul._outgoing

    def forward(self, z):
        z_norm = self.layer_norm_in(z)
        a = self.sigmoid(self.linear_a_g(z_norm)) * self.linear_a_p(z_norm)
        b = self.sigmoid(self.linear_b_g(z_norm)) * self.linear_b_p(z_norm)
        if self._outgoing:
            a = a.permute(0, 3, 1, 2).contiguous()
            b = b.permute(0, 3, 2, 1).contiguous()
        else:
            a = a.permute(0, 3, 2, 1).contiguous()
            b = b.permute(0, 3, 1, 2).contiguous()
        p = torch.einsum("...ij,...jk->...ik", a, b)
        p = p.permute(0, 2, 3, 1).contiguous()
        p = self.layer_norm_out(p)
        p = self.linear_z(p)
        g = self.sigmoid(self.linear_g(z_norm))
        return p * g


class FusedTriMulOutInWrapper(nn.Module):
    """Fused TriMulOut + TriMulIn in one trace: both triangle multiplications
    applied sequentially with residual connections.

    Eliminates 1 call/layer by combining both TriMul operations (outgoing
    and incoming) into a single traced model. Compiles at N <= 384 on
    SDK 2.28 (SBUF overflow at N=512). ~30% faster than two separate
    TriMulFullWrapper calls at N=384.

    Input: z [1, N, N, C_Z]
    Output: z_updated [1, N, N, C_Z] (after both TriMulOut and TriMulIn residuals)
    """

    def __init__(self, tri_mul_out, tri_mul_in):
        super().__init__()
        # TriMulOut components
        self.out_layer_norm_in = tri_mul_out.layer_norm_in
        self.out_linear_a_g = tri_mul_out.linear_a_g
        self.out_linear_a_p = tri_mul_out.linear_a_p
        self.out_linear_b_g = tri_mul_out.linear_b_g
        self.out_linear_b_p = tri_mul_out.linear_b_p
        self.out_layer_norm_out = tri_mul_out.layer_norm_out
        self.out_linear_z = tri_mul_out.linear_z
        self.out_linear_g = tri_mul_out.linear_g
        self.out_outgoing = tri_mul_out._outgoing

        # TriMulIn components
        self.in_layer_norm_in = tri_mul_in.layer_norm_in
        self.in_linear_a_g = tri_mul_in.linear_a_g
        self.in_linear_a_p = tri_mul_in.linear_a_p
        self.in_linear_b_g = tri_mul_in.linear_b_g
        self.in_linear_b_p = tri_mul_in.linear_b_p
        self.in_layer_norm_out = tri_mul_in.layer_norm_out
        self.in_linear_z = tri_mul_in.linear_z
        self.in_linear_g = tri_mul_in.linear_g
        self.in_outgoing = tri_mul_in._outgoing

        self.sigmoid = nn.Sigmoid()

    def _run_trimul(self, z, ln_in, a_g, a_p, b_g, b_p, ln_out, lz, lg, outgoing):
        """Run one TriMul operation (shared logic for out and in)."""
        z_norm = ln_in(z)
        a = self.sigmoid(a_g(z_norm)) * a_p(z_norm)
        b = self.sigmoid(b_g(z_norm)) * b_p(z_norm)
        if outgoing:
            a = a.permute(0, 3, 1, 2).contiguous()
            b = b.permute(0, 3, 2, 1).contiguous()
        else:
            a = a.permute(0, 3, 2, 1).contiguous()
            b = b.permute(0, 3, 1, 2).contiguous()
        p = torch.einsum("...ij,...jk->...ik", a, b)
        p = p.permute(0, 2, 3, 1).contiguous()
        p = ln_out(p)
        p = lz(p)
        g = self.sigmoid(lg(z_norm))
        return p * g

    def forward(self, z):
        # TriMulOut + residual
        z = z + self._run_trimul(
            z,
            self.out_layer_norm_in,
            self.out_linear_a_g,
            self.out_linear_a_p,
            self.out_linear_b_g,
            self.out_linear_b_p,
            self.out_layer_norm_out,
            self.out_linear_z,
            self.out_linear_g,
            self.out_outgoing,
        )
        # TriMulIn + residual
        z = z + self._run_trimul(
            z,
            self.in_layer_norm_in,
            self.in_linear_a_g,
            self.in_linear_a_p,
            self.in_linear_b_g,
            self.in_linear_b_p,
            self.in_layer_norm_out,
            self.in_linear_z,
            self.in_linear_g,
            self.in_outgoing,
        )
        return z


class TriMulBmmOutputWrapper(nn.Module):
    """Merged TriMul BMM + Output: einsum -> permute -> LayerNorm -> linear -> gate.

    Eliminates the HBM materialization of the p=[1,D,N,N] intermediate
    between the BMM and Output stages. Compiles at N <= 512 (1.12-1.13x
    faster than separate BMM + Output at N=384-512).

    Input: a [1, D, N, N], b [1, D, N, N], z_norm [1, N, N, C_Z]
    Output: result [1, N, N, C_Z]
    """

    def __init__(self, tri_mul):
        super().__init__()
        self.layer_norm_out = tri_mul.layer_norm_out
        self.linear_z = tri_mul.linear_z
        self.linear_g = tri_mul.linear_g
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, b, z_norm):
        p = torch.einsum("...ij,...jk->...ik", a, b)
        p = p.permute(0, 2, 3, 1).contiguous()
        p = self.layer_norm_out(p)
        p = self.linear_z(p)
        g = self.sigmoid(self.linear_g(z_norm))
        return p * g


class TriAttnFullWrapper(nn.Module):
    """Merged TriAttn: LayerNorm + bias + MHA in one trace.

    Eliminates the HBM materialization of x_normed and triangle_bias
    between the Bias and MHA stages. Compiles at N <= 512 (1.68-1.79x
    faster than 2-segment at N=384-512).

    Input: x [1, N, N, C_Z]
    Output: attn_output [1, N, N, C_Z]
    """

    def __init__(self, tri_attn):
        super().__init__()
        self.layer_norm = tri_attn.layer_norm
        self.linear_z = tri_attn.linear_z
        self.mha = tri_attn.mha

    def forward(self, x):
        x_normed = self.layer_norm(x)
        triangle_bias = self.linear_z(x_normed)
        triangle_bias = triangle_bias.permute(0, 3, 1, 2).unsqueeze(1)
        return self.mha(
            q_x=x_normed,
            kv_x=x_normed,
            biases=[triangle_bias],
            use_deepspeed_evo_attention=False,
            use_lma=False,
            use_cueq_triangle_kernels=False,
        )


class AttnPairBiasFullWrapper(nn.Module):
    """Merged AttnPairBias: LayerNorm + bias + MHA in one trace.

    Eliminates the HBM materialization of a_normed and pair_bias
    between the Bias and MHA stages. Compiles at N <= 512 (1.17-1.31x
    faster than 2-segment at N=384-512).

    Input: a [1, N, C_S], z [1, N, N, C_Z]
    Output: attn_output [1, N, C_S]
    """

    def __init__(self, apb):
        super().__init__()
        self.layer_norm_a = apb.layer_norm_a
        self.layer_norm_z = apb.layer_norm_z
        self.linear_z = apb.linear_z
        self.mha = apb.mha

    def forward(self, a, z):
        a_normed = self.layer_norm_a(a)
        z_normed = self.layer_norm_z(z)
        pair_bias = self.linear_z(z_normed)
        pair_bias = pair_bias.permute(0, 3, 1, 2)
        return self.mha(
            q_x=a_normed,
            kv_x=a_normed,
            biases=[pair_bias],
            use_deepspeed_evo_attention=False,
            use_lma=False,
            use_cueq_triangle_kernels=False,
        )


# ============================================================================
# Decomposed PairFormer Compiler
# ============================================================================


def _run_chunked_triattn(bias_neff, mha_neff, z_in, chunk_size):
    """Run TriAttn with chunked MHA over the row dimension.

    For N > 1024, the attention scores tensor exceeds HBM capacity.
    Since each row's attention is independent, we process chunks of
    rows and concatenate -- mathematically exact.

    Args:
        bias_neff: Traced TriAttnBiasWrapper
        mha_neff: Traced TriAttnMHAChunkedWrapper
        z_in: [1, N, N, C_Z] input tensor
        chunk_size: Number of rows per chunk

    Returns:
        attn_output: [1, N, N, C_Z]
    """
    x_normed, tri_bias = bias_neff(z_in)
    N = x_normed.shape[1]
    chunks_out = []
    for i in range(0, N, chunk_size):
        x_chunk = x_normed[:, i : i + chunk_size, :, :].contiguous()
        out_chunk = mha_neff(x_chunk, tri_bias)
        chunks_out.append(out_chunk)
    return torch.cat(chunks_out, dim=1)


class DecomposedPairFormerCompiler:
    """Compiles PairFormer sub-operations for N > 256.

    This compiler traces each PairFormer sub-operation independently,
    enabling compilation at sequence lengths up to N=2048+. The
    decomposition strategy is selected automatically based on N:

      N=257-384:  Fused TriMulOut+In (1 call) + merged TriAttn/APB (1 call each)
      N=385-512:  Proj + merged BMM+Output (2 calls) + merged TriAttn/APB
      N=513-1024: 3-segment TriMul + 2-segment TriAttn/APB
      N>1024:     Same as above + chunked TriAttn MHA (chunk_size=128)

    Calls per layer by N range:

      | N Range   | TriMul | TriAttn | APB | PTrans | STrans | Total |
      |-----------|--------|---------|-----|--------|--------|-------|
      | 257-384   | 1      | 1+1=2   | 1   | 1      | 1      | **5** |
      | 385-512   | 2+2=4  | 1+1=2   | 1   | 1      | 1      | 9     |
      | 513-1024  | 3+3=6  | 2+2=4   | 2   | 1      | 1      | 14    |
      | >1024     | 3+3=6  | 2+2=4*  | 2   | 1      | 1      | 14+C  |
      * TriAttn MHA is chunked: C = 2 * ceil(N / chunk_size) extra calls

    Usage::

        compiler = DecomposedPairFormerCompiler(model, n_token=2048)
        compile_time = compiler.compile_all()
        z, s = compiler.run_layer(z, s, layer_idx=0)

    Args:
        model: Loaded OpenFold3 model
        n_token: Sequence length
        compiler_args: Neuron compiler arguments
        chunk_size: Rows per TriAttn MHA chunk (default: 128)
    """

    def __init__(
        self,
        model,
        n_token: int,
        compiler_args: Optional[List[str]] = None,
        chunk_size: int = CHUNKED_ATTN_CHUNK_SIZE,
    ):
        self.model = model
        self.n_token = n_token
        self.compiler_args = compiler_args or ["--target", "trn2"]
        self.chunk_size = chunk_size
        self.use_chunked = n_token > CHUNKED_ATTN_THRESHOLD

        # Strategy flags based on N range
        self.use_fused_trimul_out_in = n_token <= FUSED_TRIMUL_OUT_IN_MAX_N
        self.use_full_trimul = (
            not self.use_fused_trimul_out_in and n_token <= MERGED_TRIMUL_MAX_N
        )
        self.use_merged_bmm_output = (
            not self.use_fused_trimul_out_in
            and not self.use_full_trimul
            and n_token <= MERGED_TRIMUL_BMM_OUTPUT_MAX_N
        )
        self.use_merged_attn = n_token <= MERGED_ATTN_MAX_N

        # Traced sub-ops (populated by compile_all)
        self.sub_ops: Dict[str, Any] = {}
        self.compile_times: Dict[str, float] = {}

    def _determine_strategy_label(self) -> str:
        """Return a human-readable label for the current strategy."""
        N = self.n_token
        if self.use_fused_trimul_out_in:
            return f"fused-trimul-out+in + merged-attn (N={N}, 5 calls/layer)"
        elif self.use_full_trimul:
            return f"full-trimul + merged-attn (N={N}, 7 calls/layer)"
        elif self.use_merged_bmm_output:
            return f"proj+merged-bmmout + merged-attn (N={N}, 9 calls/layer)"
        elif not self.use_chunked:
            return f"3-seg-trimul + 2-seg-attn (N={N}, 14 calls/layer)"
        else:
            n_chunks = (N + self.chunk_size - 1) // self.chunk_size
            total = 14 + 2 * n_chunks
            return (
                f"3-seg-trimul + 2-seg-attn + chunked-mha "
                f"(N={N}, chunk={self.chunk_size}, {total} calls/layer)"
            )

    def compile_all(self) -> Dict[str, float]:
        """Compile all decomposed sub-operations.

        Returns:
            dict mapping sub-op names to compilation times in seconds
        """
        import torch_neuronx

        N = self.n_token
        trace_kwargs = dict(
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )

        strategy = self._determine_strategy_label()
        print(f"  Strategy: {strategy}")

        block0 = self.model.pairformer_stack.blocks[0]

        z_dummy = torch.randn(1, N, N, C_Z)
        z_t_dummy = z_dummy.transpose(-2, -3).contiguous()
        s_dummy = torch.randn(1, N, C_S)

        # ================================================================
        # TriMul compilation (strategy depends on N)
        # ================================================================

        if self.use_fused_trimul_out_in:
            # N <= 384: Fused TriMulOut+TriMulIn in single trace (~30% faster)
            print("  Compiling TriMulOut+TriMulIn (fused)...", end=" ", flush=True)
            t0 = time.time()
            w = FusedTriMulOutInWrapper(
                block0.pair_stack.tri_mul_out,
                block0.pair_stack.tri_mul_in,
            )
            w.eval()
            self.sub_ops["trimul_fused"] = torch_neuronx.trace(
                w, (z_dummy,), **trace_kwargs
            )
            self.compile_times["trimul_fused"] = time.time() - t0
            print(f"OK ({self.compile_times['trimul_fused']:.1f}s)")

        elif self.use_full_trimul:
            # N <= 384: Full TriMul in single trace (1.57x faster than 3-seg)
            for name, tri_mul in [
                ("tmout", block0.pair_stack.tri_mul_out),
                ("tmin", block0.pair_stack.tri_mul_in),
            ]:
                label = "TriMulOut" if name == "tmout" else "TriMulIn"
                print(f"  Compiling {label} (full)...", end=" ", flush=True)
                t0 = time.time()
                w = TriMulFullWrapper(tri_mul)
                w.eval()
                self.sub_ops[f"{name}_full"] = torch_neuronx.trace(
                    w, (z_dummy,), **trace_kwargs
                )
                self.compile_times[f"{name}_full"] = time.time() - t0
                print(f"OK ({self.compile_times[f'{name}_full']:.1f}s)")

        elif self.use_merged_bmm_output:
            # N = 385-512: Projection + merged BMM+Output (1.12-1.13x)
            for name, tri_mul in [
                ("tmout", block0.pair_stack.tri_mul_out),
                ("tmin", block0.pair_stack.tri_mul_in),
            ]:
                label = "TriMulOut" if name == "tmout" else "TriMulIn"

                print(f"  Compiling {label} Projection...", end=" ", flush=True)
                t0 = time.time()
                w = TriMulProjectionWrapper(tri_mul)
                w.eval()
                self.sub_ops[f"{name}_proj"] = torch_neuronx.trace(
                    w, (z_dummy,), **trace_kwargs
                )
                a, b, zn = self.sub_ops[f"{name}_proj"](z_dummy)
                self.compile_times[f"{name}_proj"] = time.time() - t0
                print(f"OK ({self.compile_times[f'{name}_proj']:.1f}s)")

                print(
                    f"  Compiling {label} BMM+Output (merged)...",
                    end=" ",
                    flush=True,
                )
                t0 = time.time()
                w = TriMulBmmOutputWrapper(tri_mul)
                w.eval()
                self.sub_ops[f"{name}_bmmout"] = torch_neuronx.trace(
                    w,
                    (torch.randn_like(a), torch.randn_like(b), torch.randn_like(zn)),
                    **trace_kwargs,
                )
                self.compile_times[f"{name}_bmmout"] = time.time() - t0
                print(f"OK ({self.compile_times[f'{name}_bmmout']:.1f}s)")

        else:
            # N > 512: 3-segment decomposition (Projection, BMM, Output)
            for name, tri_mul in [
                ("tmout", block0.pair_stack.tri_mul_out),
                ("tmin", block0.pair_stack.tri_mul_in),
            ]:
                label = "TriMulOut" if name == "tmout" else "TriMulIn"

                print(f"  Compiling {label} Projection...", end=" ", flush=True)
                t0 = time.time()
                w = TriMulProjectionWrapper(tri_mul)
                w.eval()
                self.sub_ops[f"{name}_proj"] = torch_neuronx.trace(
                    w, (z_dummy,), **trace_kwargs
                )
                a, b, zn = self.sub_ops[f"{name}_proj"](z_dummy)
                self.compile_times[f"{name}_proj"] = time.time() - t0
                print(f"OK ({self.compile_times[f'{name}_proj']:.1f}s)")

                print(f"  Compiling {label} BMM...", end=" ", flush=True)
                t0 = time.time()
                w = TriMulBmmWrapper()
                w.eval()
                self.sub_ops[f"{name}_bmm"] = torch_neuronx.trace(
                    w,
                    (torch.randn_like(a), torch.randn_like(b)),
                    **trace_kwargs,
                )
                p = self.sub_ops[f"{name}_bmm"](a, b)
                self.compile_times[f"{name}_bmm"] = time.time() - t0
                print(f"OK ({self.compile_times[f'{name}_bmm']:.1f}s)")

                print(f"  Compiling {label} Output...", end=" ", flush=True)
                t0 = time.time()
                w = TriMulOutputWrapper(tri_mul)
                w.eval()
                self.sub_ops[f"{name}_out"] = torch_neuronx.trace(
                    w,
                    (torch.randn_like(p), torch.randn_like(zn)),
                    **trace_kwargs,
                )
                self.compile_times[f"{name}_out"] = time.time() - t0
                print(f"OK ({self.compile_times[f'{name}_out']:.1f}s)")

        # ================================================================
        # TriAttn compilation (merged or 2-segment)
        # ================================================================

        if self.use_merged_attn:
            # N <= 512: Merged TriAttn (1.68-1.79x faster)
            for name, tri_attn, dummy in [
                ("tas", block0.pair_stack.tri_att_start, z_dummy),
                ("tae", block0.pair_stack.tri_att_end, z_t_dummy),
            ]:
                label = "TriAttnStart" if name == "tas" else "TriAttnEnd"
                print(f"  Compiling {label} (merged)...", end=" ", flush=True)
                t0 = time.time()
                w = TriAttnFullWrapper(tri_attn)
                w.eval()
                self.sub_ops[f"{name}_full"] = torch_neuronx.trace(
                    w, (dummy,), **trace_kwargs
                )
                self.compile_times[f"{name}_full"] = time.time() - t0
                print(f"OK ({self.compile_times[f'{name}_full']:.1f}s)")

        else:
            # N > 512: 2-segment (Bias + MHA), with optional chunking
            for name, tri_attn, dummy in [
                ("tas", block0.pair_stack.tri_att_start, z_dummy),
                ("tae", block0.pair_stack.tri_att_end, z_t_dummy),
            ]:
                label = "TriAttnStart" if name == "tas" else "TriAttnEnd"

                print(f"  Compiling {label} Bias...", end=" ", flush=True)
                t0 = time.time()
                w = TriAttnBiasWrapper(tri_attn)
                w.eval()
                self.sub_ops[f"{name}_bias"] = torch_neuronx.trace(
                    w, (dummy,), **trace_kwargs
                )
                xn, tb = self.sub_ops[f"{name}_bias"](dummy)
                self.compile_times[f"{name}_bias"] = time.time() - t0
                print(f"OK ({self.compile_times[f'{name}_bias']:.1f}s)")

                if self.use_chunked:
                    print(
                        f"  Compiling {label} MHA "
                        f"(chunked, chunk_size={self.chunk_size})...",
                        end=" ",
                        flush=True,
                    )
                    t0 = time.time()
                    x_chunk_dummy = torch.randn(1, self.chunk_size, N, C_Z)
                    tb_dummy = torch.randn_like(tb)
                    w = TriAttnMHAChunkedWrapper(tri_attn)
                    w.eval()
                    self.sub_ops[f"{name}_mha"] = torch_neuronx.trace(
                        w, (x_chunk_dummy, tb_dummy), **trace_kwargs
                    )
                else:
                    print(f"  Compiling {label} MHA (full)...", end=" ", flush=True)
                    t0 = time.time()
                    w = TriAttnMHAWrapper(tri_attn)
                    w.eval()
                    self.sub_ops[f"{name}_mha"] = torch_neuronx.trace(
                        w,
                        (torch.randn_like(xn), torch.randn_like(tb)),
                        **trace_kwargs,
                    )
                self.compile_times[f"{name}_mha"] = time.time() - t0
                print(f"OK ({self.compile_times[f'{name}_mha']:.1f}s)")

        # ================================================================
        # AttnPairBias compilation (merged or 2-segment)
        # ================================================================

        if self.use_merged_attn:
            # N <= 512: Merged AttnPairBias (1.17-1.31x faster)
            print("  Compiling AttnPairBias (merged)...", end=" ", flush=True)
            t0 = time.time()
            w = AttnPairBiasFullWrapper(block0.attn_pair_bias)
            w.eval()
            self.sub_ops["apb_full"] = torch_neuronx.trace(
                w, (s_dummy, z_dummy), **trace_kwargs
            )
            self.compile_times["apb_full"] = time.time() - t0
            print(f"OK ({self.compile_times['apb_full']:.1f}s)")

        else:
            # N > 512: 2-segment (Bias + MHA)
            print("  Compiling AttnPairBias Bias...", end=" ", flush=True)
            t0 = time.time()
            w = AttnPairBiasBiasWrapper(block0.attn_pair_bias)
            w.eval()
            self.sub_ops["apb_bias"] = torch_neuronx.trace(
                w, (s_dummy, z_dummy), **trace_kwargs
            )
            an, pb = self.sub_ops["apb_bias"](s_dummy, z_dummy)
            self.compile_times["apb_bias"] = time.time() - t0
            print(f"OK ({self.compile_times['apb_bias']:.1f}s)")

            print("  Compiling AttnPairBias MHA...", end=" ", flush=True)
            t0 = time.time()
            w = AttnPairBiasMHAWrapper(block0.attn_pair_bias)
            w.eval()
            self.sub_ops["apb_mha"] = torch_neuronx.trace(
                w, (torch.randn_like(an), torch.randn_like(pb)), **trace_kwargs
            )
            self.compile_times["apb_mha"] = time.time() - t0
            print(f"OK ({self.compile_times['apb_mha']:.1f}s)")

        # ================================================================
        # PairTransition + SingleTransition (always monolithic)
        # ================================================================

        print("  Compiling PairTransition...", end=" ", flush=True)
        t0 = time.time()
        w = PairTransitionWrapper(block0.pair_stack.pair_transition)
        w.eval()
        self.sub_ops["ptrans"] = torch_neuronx.trace(w, (z_dummy,), **trace_kwargs)
        self.compile_times["ptrans"] = time.time() - t0
        print(f"OK ({self.compile_times['ptrans']:.1f}s)")

        print("  Compiling SingleTransition...", end=" ", flush=True)
        t0 = time.time()
        w = SingleTransitionWrapper(block0.single_transition)
        w.eval()
        self.sub_ops["strans"] = torch_neuronx.trace(w, (s_dummy,), **trace_kwargs)
        self.compile_times["strans"] = time.time() - t0
        print(f"OK ({self.compile_times['strans']:.1f}s)")

        total = sum(self.compile_times.values())
        print(f"  All {len(self.sub_ops)} sub-ops compiled in {total:.1f}s total")
        return self.compile_times

    def _replace_weights_for_layer(self, block):
        """Replace weights in all traced sub-ops for a given PairFormer block.

        Args:
            block: A PairFormerBlock from model.pairformer_stack.blocks[i]
        """
        import torch_neuronx

        ps = block.pair_stack

        # TriMul weight replacement (strategy-dependent)
        if self.use_fused_trimul_out_in:
            w = FusedTriMulOutInWrapper(ps.tri_mul_out, ps.tri_mul_in)
            torch_neuronx.replace_weights(self.sub_ops["trimul_fused"], w.state_dict())

        elif self.use_full_trimul:
            for name, tri_mul in [
                ("tmout", ps.tri_mul_out),
                ("tmin", ps.tri_mul_in),
            ]:
                w = TriMulFullWrapper(tri_mul)
                torch_neuronx.replace_weights(
                    self.sub_ops[f"{name}_full"], w.state_dict()
                )

        elif self.use_merged_bmm_output:
            for name, tri_mul in [
                ("tmout", ps.tri_mul_out),
                ("tmin", ps.tri_mul_in),
            ]:
                w = TriMulProjectionWrapper(tri_mul)
                torch_neuronx.replace_weights(
                    self.sub_ops[f"{name}_proj"], w.state_dict()
                )
                w = TriMulBmmOutputWrapper(tri_mul)
                torch_neuronx.replace_weights(
                    self.sub_ops[f"{name}_bmmout"], w.state_dict()
                )

        else:
            for name, tri_mul in [
                ("tmout", ps.tri_mul_out),
                ("tmin", ps.tri_mul_in),
            ]:
                w = TriMulProjectionWrapper(tri_mul)
                torch_neuronx.replace_weights(
                    self.sub_ops[f"{name}_proj"], w.state_dict()
                )
                w = TriMulOutputWrapper(tri_mul)
                torch_neuronx.replace_weights(
                    self.sub_ops[f"{name}_out"], w.state_dict()
                )
                # Note: BMM wrapper has no weights, no replacement needed

        # TriAttn weight replacement (strategy-dependent)
        if self.use_merged_attn:
            for name, tri_attn in [
                ("tas", ps.tri_att_start),
                ("tae", ps.tri_att_end),
            ]:
                w = TriAttnFullWrapper(tri_attn)
                torch_neuronx.replace_weights(
                    self.sub_ops[f"{name}_full"], w.state_dict()
                )
        else:
            for name, tri_attn in [
                ("tas", ps.tri_att_start),
                ("tae", ps.tri_att_end),
            ]:
                w = TriAttnBiasWrapper(tri_attn)
                torch_neuronx.replace_weights(
                    self.sub_ops[f"{name}_bias"], w.state_dict()
                )
                if self.use_chunked:
                    w = TriAttnMHAChunkedWrapper(tri_attn)
                else:
                    w = TriAttnMHAWrapper(tri_attn)
                torch_neuronx.replace_weights(
                    self.sub_ops[f"{name}_mha"], w.state_dict()
                )

        # AttnPairBias weight replacement (strategy-dependent)
        if self.use_merged_attn:
            w = AttnPairBiasFullWrapper(block.attn_pair_bias)
            torch_neuronx.replace_weights(self.sub_ops["apb_full"], w.state_dict())
        else:
            w = AttnPairBiasBiasWrapper(block.attn_pair_bias)
            torch_neuronx.replace_weights(self.sub_ops["apb_bias"], w.state_dict())
            w = AttnPairBiasMHAWrapper(block.attn_pair_bias)
            torch_neuronx.replace_weights(self.sub_ops["apb_mha"], w.state_dict())

        # PairTransition + SingleTransition (always same)
        w = PairTransitionWrapper(ps.pair_transition)
        torch_neuronx.replace_weights(self.sub_ops["ptrans"], w.state_dict())

        w = SingleTransitionWrapper(block.single_transition)
        torch_neuronx.replace_weights(self.sub_ops["strans"], w.state_dict())

    def _run_trimul(self, name, z):
        """Run a TriMul sub-op (out or in) with appropriate strategy."""
        if self.use_full_trimul:
            return self.sub_ops[f"{name}_full"](z)
        elif self.use_merged_bmm_output:
            a, b, zn = self.sub_ops[f"{name}_proj"](z)
            return self.sub_ops[f"{name}_bmmout"](a, b, zn)
        else:
            a, b, zn = self.sub_ops[f"{name}_proj"](z)
            p = self.sub_ops[f"{name}_bmm"](a, b)
            return self.sub_ops[f"{name}_out"](p, zn)

    def _run_triattn(self, name, z_in):
        """Run a TriAttn sub-op (start or end) with appropriate strategy."""
        if self.use_merged_attn:
            return self.sub_ops[f"{name}_full"](z_in)
        elif self.use_chunked:
            return _run_chunked_triattn(
                self.sub_ops[f"{name}_bias"],
                self.sub_ops[f"{name}_mha"],
                z_in,
                self.chunk_size,
            )
        else:
            xn, tb = self.sub_ops[f"{name}_bias"](z_in)
            return self.sub_ops[f"{name}_mha"](xn, tb)

    def _run_apb(self, s, z):
        """Run AttnPairBias with appropriate strategy."""
        if self.use_merged_attn:
            return self.sub_ops["apb_full"](s, z)
        else:
            an, pb = self.sub_ops["apb_bias"](s, z)
            return self.sub_ops["apb_mha"](an, pb)

    def run_layer(self, z, s, layer_idx):
        """Run a single PairFormer layer using decomposed sub-ops.

        Replaces weights for the specified layer, then executes all
        sub-operations in the correct order with residual connections.

        Args:
            z: [1, N, N, C_Z] pair representation
            s: [1, N, C_S] single representation
            layer_idx: Index of the PairFormer layer (0-47)

        Returns:
            (z, s): Updated representations
        """
        block = self.model.pairformer_stack.blocks[layer_idx]
        self._replace_weights_for_layer(block)

        # TriMulOut + TriMulIn (fused or separate)
        if self.use_fused_trimul_out_in:
            # Fused: single call does both TriMulOut and TriMulIn with residuals
            z = self.sub_ops["trimul_fused"](z)
        else:
            # Separate: two calls with residuals
            z = z + self._run_trimul("tmout", z)
            z = z + self._run_trimul("tmin", z)

        # TriAttnStart
        z = z + self._run_triattn("tas", z)

        # TriAttnEnd (operates on transposed z)
        z_t = z.transpose(-2, -3).contiguous()
        z_t = z_t + self._run_triattn("tae", z_t)
        z = z_t.transpose(-2, -3).contiguous()

        # PairTransition (includes residual add in wrapper)
        z = self.sub_ops["ptrans"](z)

        # AttnPairBias
        s = s + self._run_apb(s, z)

        # SingleTransition (includes residual add in wrapper)
        s = self.sub_ops["strans"](s)

        return z, s


# ============================================================================
# Source Code Patches
# ============================================================================


def patch_openfold3_source(openfold3_path: str) -> List[str]:
    """Apply Neuron compatibility patches to OpenFold3 source code.

    OpenFold3 contains CUDA-specific code that must be replaced for
    Neuron compatibility. This function patches the source files
    in-place. It is idempotent -- running it multiple times is safe.

    Patches applied:
      1. autocast("cuda") -> autocast("cpu") in 5 files (13 occurrences)
      2. device_type="cuda" -> device_type="cpu" in 3 files
      3. torch.cuda.empty_cache() -> pass in 6 files
      4. torch.cuda.synchronize() -> pass in callbacks.py
      5. torch.cuda.manual_seed_all() -> pass in callbacks.py
      6. use_deepspeed_evo_attention: True -> False in model_config.py

    Args:
        openfold3_path: Path to the openfold3 package directory
            (e.g., '/home/ubuntu/openfold-3/openfold3')

    Returns:
        List of patch descriptions applied
    """
    patches = []
    base = openfold3_path

    # Patch 1: autocast("cuda") -> autocast("cpu")
    autocast_files = [
        "core/model/primitives/attention.py",
        "core/model/primitives/linear.py",
        "core/model/primitives/normalization.py",
        "core/utils/geometry/kabsch_alignment.py",
        "core/loss/diffusion.py",
    ]
    for f in autocast_files:
        path = os.path.join(base, f)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            content = fh.read()
        original = content
        content = content.replace(
            'torch.amp.autocast("cuda"', 'torch.amp.autocast("cpu"'
        )
        if content != original:
            with open(path, "w") as fh:
                fh.write(content)
            count = original.count('torch.amp.autocast("cuda"')
            patches.append(f"{f}: replaced {count} autocast('cuda') -> autocast('cpu')")

    # Patch 2: device_type="cuda" -> device_type="cpu"
    device_type_files = [
        "projects/of3_all_atom/model.py",
        "core/model/heads/prediction_heads.py",
        "core/model/feature_embedders/input_embedders.py",
    ]
    for f in device_type_files:
        path = os.path.join(base, f)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            content = fh.read()
        original = content
        content = content.replace('device_type="cuda"', 'device_type="cpu"')
        if content != original:
            with open(path, "w") as fh:
                fh.write(content)
            count = original.count('device_type="cuda"')
            patches.append(
                f"{f}: replaced {count} device_type='cuda' -> device_type='cpu'"
            )

    # Patch 3: torch.cuda.empty_cache() -> pass
    empty_cache_files = [
        "projects/of3_all_atom/runner.py",
        "projects/of3_all_atom/model.py",
        "core/model/latent/base_stacks.py",
        "core/model/latent/pairformer.py",
        "core/model/latent/msa_module.py",
        "core/model/latent/evoformer.py",
    ]
    for f in empty_cache_files:
        path = os.path.join(base, f)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            content = fh.read()
        original = content
        content = content.replace(
            "torch.cuda.empty_cache()", "pass  # empty_cache removed for Neuron"
        )
        if content != original:
            with open(path, "w") as fh:
                fh.write(content)
            count = original.count("torch.cuda.empty_cache()")
            patches.append(f"{f}: replaced {count} empty_cache() -> pass")

    # Patch 4: callbacks.py
    callbacks_path = os.path.join(base, "core/utils/callbacks.py")
    if os.path.exists(callbacks_path):
        with open(callbacks_path) as fh:
            content = fh.read()
        original = content
        content = content.replace(
            "torch.cuda.synchronize()", "pass  # synchronize removed for Neuron"
        )
        content = content.replace(
            "torch.cuda.manual_seed_all(rank_specific_seed)",
            "pass  # manual_seed_all removed for Neuron",
        )
        if content != original:
            with open(callbacks_path, "w") as fh:
                fh.write(content)
            patches.append(
                "core/utils/callbacks.py: replaced synchronize() and manual_seed_all()"
            )

    # Patch 5: model_config.py -- disable deepspeed evo for eval
    config_path = os.path.join(base, "projects/of3_all_atom/config/model_config.py")
    if os.path.exists(config_path):
        with open(config_path) as fh:
            lines = fh.readlines()
        modified = False
        for i, line in enumerate(lines):
            if '"use_deepspeed_evo_attention": True,' in line:
                lines[i] = line.replace(
                    '"use_deepspeed_evo_attention": True,',
                    '"use_deepspeed_evo_attention": False,  # Neuron: disabled',
                )
                modified = True
        if modified:
            with open(config_path, "w") as fh:
                fh.writelines(lines)
            patches.append("model_config.py: set use_deepspeed_evo_attention=False")

    return patches


# ============================================================================
# Dummy Batch Creation
# ============================================================================


def create_dummy_batch(
    n_token: int = 256,
    n_atom: int = 256,
    n_msa: int = 4,
    n_templ: int = 1,
    seed: int = 42,
) -> dict:
    """Create a dummy input batch for OpenFold3 inference.

    The batch format matches what OpenFold3.forward() expects. Uses
    is_protein=0, is_atomized=1 for a 1-atom-per-token setup to avoid
    CB atom index OOB errors.

    Args:
        n_token: Number of tokens
        n_atom: Number of atoms (set equal to n_token for 1:1 mapping)
        n_msa: Number of MSA sequences
        n_templ: Number of templates
        seed: Random seed for reproducibility

    Returns:
        dict: Input batch dictionary with all required keys
    """
    torch.manual_seed(seed)

    batch = {
        # Token-level
        "residue_index": torch.arange(n_token).long().unsqueeze(0),
        "token_index": torch.arange(n_token).long().unsqueeze(0),
        "asym_id": torch.zeros(1, n_token, dtype=torch.long),
        "entity_id": torch.zeros(1, n_token, dtype=torch.long),
        "sym_id": torch.zeros(1, n_token, dtype=torch.long),
        "restype": torch.nn.functional.one_hot(
            torch.zeros(1, n_token, dtype=torch.long), 32
        ).float(),
        "is_protein": torch.zeros(1, n_token),
        "is_rna": torch.zeros(1, n_token),
        "is_dna": torch.zeros(1, n_token),
        "is_ligand": torch.zeros(1, n_token),
        "is_atomized": torch.ones(1, n_token),
        "token_bonds": torch.zeros(1, n_token, n_token),
        "token_mask": torch.ones(1, n_token),
        "num_atoms_per_token": torch.ones(1, n_token, dtype=torch.long),
        "start_atom_index": torch.arange(n_token).long().unsqueeze(0),
        "profile": torch.zeros(1, n_token, 32),
        "deletion_mean": torch.zeros(1, n_token),
        # Atom-level
        "ref_pos": torch.randn(1, n_atom, 3),
        "ref_mask": torch.ones(1, n_atom),
        "ref_element": torch.zeros(1, n_atom, 119),  # 119 element types
        "ref_charge": torch.zeros(1, n_atom),
        "ref_atom_name_chars": torch.zeros(1, n_atom, 4, 64),
        "ref_space_uid": torch.arange(n_atom).long().unsqueeze(0),
        "atom_mask": torch.ones(1, n_atom),
        "atom_to_token_index": torch.arange(n_atom).long().unsqueeze(0),
        # MSA
        "msa": torch.zeros(1, n_msa, n_token, 32),
        "has_deletion": torch.zeros(1, n_msa, n_token),
        "deletion_value": torch.zeros(1, n_msa, n_token),
        "msa_mask": torch.ones(1, n_msa, n_token),
        "num_paired_seqs": torch.tensor([0], dtype=torch.long),
        # Template
        "template_restype": torch.zeros(1, n_templ, n_token, 32),
        "template_pseudo_beta_mask": torch.zeros(1, n_templ, n_token),
        "template_backbone_frame_mask": torch.zeros(1, n_templ, n_token),
        "template_distogram": torch.zeros(1, n_templ, n_token, n_token, 39),
        "template_unit_vector": torch.zeros(1, n_templ, n_token, n_token, 3),
    }
    return batch


# ============================================================================
# OpenFold3 Neuron Pipeline
# ============================================================================


class OpenFold3NeuronPipeline:
    """End-to-end pipeline for OpenFold3 inference on Neuron.

    Handles model loading, source patching, compilation of all block types,
    weight replacement, monkey-patching, and inference. The pipeline
    automatically selects the compilation strategy based on N:

      - N <= 256: Monolithic PairFormerBlock tracing (fastest, lowest overhead)
      - N > 256: Decomposed sub-op tracing (enables N up to 2048+)
        - N <= 512: Merged TriAttn/APB + full/merged TriMul
        - N > 1024: Chunked TriAttn MHA (chunk_size=128)

    Non-PairFormer blocks (MSA, Template, DiffCond) are always compiled
    monolithically since they operate at small fixed sizes.

    Usage::

        from modeling_openfold3 import OpenFold3NeuronPipeline

        pipeline = OpenFold3NeuronPipeline(
            openfold3_src_path="/home/ubuntu/openfold-3",
            checkpoint_path="~/.openfold3/of3-p2-155k.pt",
            n_token=256,
        )
        pipeline.load_model()
        pipeline.compile_all()
        pipeline.patch_model()
        output = pipeline.run_inference(num_recycles=3, diff_steps=200)

    Args:
        openfold3_src_path: Path to OpenFold3 repository root
        checkpoint_path: Path to model checkpoint
        n_token: Sequence length (max ~2048 for decomposed, 256 for monolithic)
        n_atom: Number of atoms (default: same as n_token)
        n_msa: Number of MSA sequences (default: 4)
        n_templ: Number of templates (default: 1)
        compiler_args: Neuron compiler arguments
        pairformer_strategy: "auto", "monolithic", or "decomposed"
    """

    def __init__(
        self,
        openfold3_src_path: str = "/home/ubuntu/openfold-3",
        checkpoint_path: str = "~/.openfold3/of3-p2-155k.pt",
        n_token: int = 256,
        n_atom: Optional[int] = None,
        n_msa: int = 4,
        n_templ: int = 1,
        compiler_args: Optional[List[str]] = None,
        pairformer_strategy: str = "auto",
    ):
        self.openfold3_src_path = openfold3_src_path
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.n_token = n_token
        self.n_atom = n_atom or n_token
        self.n_msa = n_msa
        self.n_templ = n_templ
        self.compiler_args = compiler_args or ["--target", "trn2"]

        # Determine PairFormer strategy
        if pairformer_strategy == "auto":
            self.use_decomposed = n_token > 256
        elif pairformer_strategy == "decomposed":
            self.use_decomposed = True
        elif pairformer_strategy == "monolithic":
            self.use_decomposed = False
        else:
            raise ValueError(
                f"pairformer_strategy must be 'auto', 'monolithic', or 'decomposed', "
                f"got '{pairformer_strategy}'"
            )

        # Model (populated by load_model)
        self.model = None

        # Compiled blocks (populated by compile_all)
        self.traced_pf = None  # Monolithic PairFormer (N <= 256)
        self.decomposed_pf = None  # DecomposedPairFormerCompiler (N > 256)
        self.traced_msa_a = None
        self.traced_msa_b = None
        self.traced_tmpl = None
        self.traced_dc = None

        # Compilation times
        self.compile_times: Dict[str, float] = {}

    def load_model(self) -> None:
        """Load and configure OpenFold3 model.

        Applies source patches, imports the model, loads checkpoint
        weights, and configures for evaluation.
        """
        import sys
        import gc

        # Add OpenFold3 to path
        if self.openfold3_src_path not in sys.path:
            sys.path.insert(0, self.openfold3_src_path)

        # Apply source patches
        openfold3_pkg = os.path.join(self.openfold3_src_path, "openfold3")
        patches = patch_openfold3_source(openfold3_pkg)
        print(f"Applied {len(patches)} source patches")

        # Import and create model
        from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
        from openfold3.projects.of3_all_atom.model import OpenFold3
        from openfold3.core.utils.checkpoint_loading_utils import (
            load_checkpoint,
            get_state_dict_from_checkpoint,
        )

        project = OF3ProjectEntry()
        config = project.get_model_config_with_presets(
            presets=["predict", "pae_enabled"]
        )
        config.settings.memory.eval.use_deepspeed_evo_attention = False
        config.settings.memory.eval.use_cueq_triangle_kernels = False

        self.model = OpenFold3(config)
        self.model.eval()

        # Load weights
        ckpt = load_checkpoint(Path(self.checkpoint_path))
        state_dict, _ = get_state_dict_from_checkpoint(ckpt, init_from_ema_weights=True)
        bare_state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        self.model.load_state_dict(bare_state_dict, strict=False)
        del ckpt, state_dict, bare_state_dict
        gc.collect()

        strategy = "decomposed" if self.use_decomposed else "monolithic"
        chunked = (
            " (chunked MHA)"
            if self.n_token > CHUNKED_ATTN_THRESHOLD and self.use_decomposed
            else ""
        )
        print(
            f"Model loaded. PairFormer strategy: {strategy}{chunked}, N={self.n_token}"
        )

    def compile_all(self) -> Dict[str, float]:
        """Compile all Neuron blocks.

        Returns:
            dict mapping block names to compilation times in seconds
        """
        import torch_neuronx

        assert self.model is not None, "Call load_model() first"
        N = self.n_token
        N_MSA = self.n_msa
        N_TEMPL = self.n_templ

        # --- PairFormer ---
        if self.use_decomposed:
            print(f"  Compiling PairFormer (decomposed, N={N})...")
            self.decomposed_pf = DecomposedPairFormerCompiler(
                model=self.model,
                n_token=N,
                compiler_args=self.compiler_args,
            )
            pf_times = self.decomposed_pf.compile_all()
            self.compile_times["pairformer_decomposed"] = sum(pf_times.values())
            self.compile_times.update({f"pf_{k}": v for k, v in pf_times.items()})
        else:
            # Monolithic
            s_dummy = torch.randn(1, N, C_S)
            z_dummy = torch.randn(1, N, N, C_Z)
            single_mask = torch.ones(1, N)
            pair_mask = torch.ones(1, N, N)

            print("  Compiling PairFormerBlock (monolithic)...")
            pf_wrapper = PairFormerBlockWrapper(self.model.pairformer_stack.blocks[0])
            pf_wrapper.eval()
            t0 = time.time()
            self.traced_pf = torch_neuronx.trace(
                pf_wrapper,
                (s_dummy, z_dummy, single_mask, pair_mask),
                compiler_args=self.compiler_args,
                inline_weights_to_neff=False,
            )
            self.compile_times["pairformer"] = time.time() - t0
            print(f"    Done in {self.compile_times['pairformer']:.1f}s")

        # Common dummy tensors for non-PairFormer blocks
        s_dummy = torch.randn(1, N, C_S)
        z_dummy = torch.randn(1, N, N, C_Z)
        single_mask = torch.ones(1, N)
        pair_mask = torch.ones(1, N, N)
        m_dummy = torch.randn(1, N_MSA, N, C_M)
        msa_mask_dummy = torch.ones(1, N_MSA, N)
        t_dummy = torch.randn(1, N_TEMPL, N, N, 64)
        t_mask_dummy = torch.ones(1, N_TEMPL, N, N)

        # --- MSA type A (blocks 0-2) ---
        print("  Compiling MSA block (type A)...")
        msa_a_wrapper = MSABlockWrapper(self.model.msa_module.blocks[0])
        msa_a_wrapper.eval()
        t0 = time.time()
        self.traced_msa_a = torch_neuronx.trace(
            msa_a_wrapper,
            (m_dummy, z_dummy, msa_mask_dummy, pair_mask),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["msa_type_a"] = time.time() - t0
        print(f"    Done in {self.compile_times['msa_type_a']:.1f}s")

        # --- MSA type B (last block) ---
        num_msa_blocks = len(self.model.msa_module.blocks)
        print("  Compiling MSA block (type B - last)...")
        msa_b_wrapper = MSABlockWrapper(
            self.model.msa_module.blocks[num_msa_blocks - 1]
        )
        msa_b_wrapper.eval()
        t0 = time.time()
        self.traced_msa_b = torch_neuronx.trace(
            msa_b_wrapper,
            (m_dummy, z_dummy, msa_mask_dummy, pair_mask),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["msa_type_b"] = time.time() - t0
        print(f"    Done in {self.compile_times['msa_type_b']:.1f}s")

        # --- Template ---
        print("  Compiling Template block...")
        tmpl_wrapper = TemplatePairBlockWrapper(
            self.model.template_embedder.template_pair_stack.blocks[0]
        )
        tmpl_wrapper.eval()
        t0 = time.time()
        self.traced_tmpl = torch_neuronx.trace(
            tmpl_wrapper,
            (t_dummy, t_mask_dummy),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["template"] = time.time() - t0
        print(f"    Done in {self.compile_times['template']:.1f}s")

        # --- DiffusionConditioning._forward() ---
        print("  Compiling DiffusionConditioning._forward()...")
        dc_wrapper = DiffCondForwardWrapper(
            self.model.diffusion_module.diffusion_conditioning
        )
        dc_wrapper.eval()
        t0 = time.time()
        self.traced_dc = torch_neuronx.trace(
            dc_wrapper,
            (s_dummy, z_dummy, single_mask),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["diff_cond"] = time.time() - t0
        print(f"    Done in {self.compile_times['diff_cond']:.1f}s")

        # Warmup non-PairFormer blocks
        print("  Warming up traced models...")
        for _ in range(2):
            self.traced_msa_a(m_dummy, z_dummy, msa_mask_dummy, pair_mask)
            self.traced_msa_b(m_dummy, z_dummy, msa_mask_dummy, pair_mask)
            self.traced_tmpl(t_dummy, t_mask_dummy)
            self.traced_dc(s_dummy, z_dummy, single_mask)
            if not self.use_decomposed:
                self.traced_pf(s_dummy, z_dummy, single_mask, pair_mask)

        total = sum(self.compile_times.values())
        print(f"  All blocks compiled in {total:.1f}s total")
        return self.compile_times

    def patch_model(self) -> None:
        """Monkey-patch the OpenFold3 model to use compiled Neuron blocks.

        After calling this, the model's forward() will route PairFormer,
        MSA, Template, and DiffCond blocks through Neuron hardware while
        keeping the orchestration (recycling, diffusion, confidence) on CPU.
        """
        import torch_neuronx

        assert self.model is not None, "Call load_model() first"

        traced_msa_a = self.traced_msa_a
        traced_msa_b = self.traced_msa_b
        traced_tmpl = self.traced_tmpl
        traced_dc = self.traced_dc

        # --- PairFormer blocks ---
        num_pf_blocks = len(self.model.pairformer_stack.blocks)

        if self.use_decomposed:
            # Decomposed: replace the entire pairformer_stack.forward()
            decomposed_pf = self.decomposed_pf
            model_ref = self.model

            def neuron_pairformer_stack_forward(s, z, single_mask, pair_mask, **kwargs):
                """Run all 48 PairFormer layers using decomposed sub-ops."""
                with torch.no_grad():
                    for i in range(len(model_ref.pairformer_stack.blocks)):
                        z, s = decomposed_pf.run_layer(z, s, i)
                return s, z

            self.model.pairformer_stack.forward = neuron_pairformer_stack_forward
            print(f"  Patched PairFormer stack ({num_pf_blocks} layers, decomposed)")

        else:
            # Monolithic: patch individual block forward() methods
            traced_pf = self.traced_pf
            for i in range(num_pf_blocks):
                block = self.model.pairformer_stack.blocks[i]

                def make_pf_forward(block_idx, original_block):
                    def neuron_forward(s, z, single_mask, pair_mask, **kwargs):
                        w = PairFormerBlockWrapper(original_block)
                        torch_neuronx.replace_weights(traced_pf, w.state_dict())
                        return traced_pf(s, z, single_mask, pair_mask)

                    return neuron_forward

                block.forward = make_pf_forward(i, block)
            print(f"  Patched {num_pf_blocks} PairFormer blocks (monolithic)")

        # --- MSA blocks ---
        num_msa_blocks = len(self.model.msa_module.blocks)
        for i in range(num_msa_blocks):
            block = self.model.msa_module.blocks[i]

            if i < num_msa_blocks - 1:

                def make_msa_a_forward(block_idx, original_block):
                    def neuron_forward(m, z, msa_mask, pair_mask, **kwargs):
                        w = MSABlockWrapper(original_block)
                        torch_neuronx.replace_weights(traced_msa_a, w.state_dict())
                        return traced_msa_a(m, z, msa_mask, pair_mask)

                    return neuron_forward

                block.forward = make_msa_a_forward(i, block)
            else:

                def make_msa_b_forward(original_block):
                    def neuron_forward(m, z, msa_mask, pair_mask, **kwargs):
                        w = MSABlockWrapper(original_block)
                        torch_neuronx.replace_weights(traced_msa_b, w.state_dict())
                        return traced_msa_b(m, z, msa_mask, pair_mask)

                    return neuron_forward

                block.forward = make_msa_b_forward(block)
        print(f"  Patched {num_msa_blocks} MSA blocks")

        # --- Template blocks ---
        num_tmpl_blocks = len(self.model.template_embedder.template_pair_stack.blocks)
        for i in range(num_tmpl_blocks):
            block = self.model.template_embedder.template_pair_stack.blocks[i]

            def make_tmpl_forward(block_idx, original_block):
                def neuron_forward(t, mask, **kwargs):
                    w = TemplatePairBlockWrapper(original_block)
                    torch_neuronx.replace_weights(traced_tmpl, w.state_dict())
                    return traced_tmpl(t, mask)

                return neuron_forward

            block.forward = make_tmpl_forward(i, block)
        print(f"  Patched {num_tmpl_blocks} Template blocks")

        # --- DiffusionConditioning._forward() ---
        dc = self.model.diffusion_module.diffusion_conditioning

        def neuron_dc_forward(si, zij, token_mask, chunk_size=None):
            orig_shape_si = si.shape
            n_tok = orig_shape_si[-2]
            leading = orig_shape_si[:-2]
            si_flat = si.reshape(1, n_tok, -1)
            zij_flat = zij.reshape(1, n_tok, n_tok, -1)
            mask_flat = token_mask.reshape(1, n_tok)
            si_out, zij_out = traced_dc(si_flat, zij_flat, mask_flat)
            return (
                si_out.reshape(*leading, n_tok, -1),
                zij_out.reshape(*leading, n_tok, n_tok, -1),
            )

        dc._forward = neuron_dc_forward
        print("  Patched DiffusionConditioning._forward()")
        print("  All blocks monkey-patched.")

    def run_inference(
        self,
        batch: Optional[dict] = None,
        num_recycles: int = 3,
        diff_steps: int = 200,
        diff_samples: int = 1,
    ) -> Tuple[dict, dict]:
        """Run OpenFold3 inference.

        Args:
            batch: Input batch dict (if None, creates a dummy batch)
            num_recycles: Number of recycling iterations (default: 3)
            diff_steps: Number of diffusion steps (default: 200)
            diff_samples: Number of diffusion samples (default: 1)

        Returns:
            Tuple of (updated_batch, output_dict)
        """
        import copy

        assert self.model is not None, "Call load_model() first"

        if batch is None:
            batch = create_dummy_batch(
                n_token=self.n_token,
                n_atom=self.n_atom,
                n_msa=self.n_msa,
                n_templ=self.n_templ,
            )

        # Configure model
        self.model.shared.num_recycles = num_recycles
        self.model.shared.diffusion.no_full_rollout_steps = diff_steps
        self.model.shared.diffusion.no_full_rollout_samples = diff_samples

        batch_copy = copy.deepcopy(batch)

        with torch.no_grad():
            t0 = time.time()
            batch_out, output = self.model(batch_copy)
            elapsed = time.time() - t0

        print(f"  Inference completed in {elapsed:.1f}s")
        return batch_out, output
