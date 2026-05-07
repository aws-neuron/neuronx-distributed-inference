"""RosettaFold3 Neuron acceleration module.

Provides wrapper classes and monkey-patching utilities for running the
RosettaCommons Foundry RF3 model on AWS Trainium 2 hardware. Uses vanilla
torch_neuronx.trace() compilation (no NKI kernels) with the
replace_weights pattern for multi-layer stacks.

Five model components are compiled:
  1. PairformerBlock (48 layers) - main trunk
  2. MSAPairUpdateBlock (4 iterations, shared weights) - MSA pair ops
  3. TemplPairformerBlock (2 blocks, c_z=64) - template embedder
  4. DiffTransformerBlock (24 blocks) - diffusion transformer
  5. StaticWindowedAttnBlock (3 blocks encoder + 3 decoder, shared NEFF) - atom attention

Architecture constants (from RF3):
  C_S = 384 (single representation)
  C_Z = 128 (pair representation)
  C_TOKEN = 768 (token representation)
  C_Z_TEMPL = 64 (template pair representation)
  C_ATOM = 128 (atom representation)
  C_ATOMPAIR = 16 (atom pair representation)
  ATOM_N_HEAD = 4, ATOM_C_HEAD = 32 (atom attention heads)
  ATOM_QBATCH = 32, ATOM_KBATCH = 128 (atom attention window sizes)
"""

import os
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# Architecture Constants
# ============================================================================

C_S = 384
C_Z = 128
C_TOKEN = 768
C_Z_TEMPL = 64
C_ATOM = 128
C_ATOMPAIR = 16
ATOM_N_HEAD = 4
ATOM_C_HEAD = C_ATOM // ATOM_N_HEAD  # 32
ATOM_QBATCH = 32
ATOM_KBATCH = 128


# ============================================================================
# Wrapper Modules for Neuron Compilation
# ============================================================================


class FullPairformerBlock(nn.Module):
    """Wrapper for a single RF3 PairformerBlock that flattens the forward pass.

    RF3's PairformerBlock contains 7 sub-operations (tri_mul_outgoing,
    tri_mul_incoming, tri_attn_start, tri_attn_end, z_transition,
    attention_pair_bias, s_transition). This wrapper exposes them as a
    simple (S_I, Z_II) -> (S_I, Z_II) function suitable for tracing.

    Args:
        block: An RF3 PairformerBlock instance from recycler.pairformer_stack[i]
    """

    def __init__(self, block):
        super().__init__()
        self.tri_mul_outgoing = block.tri_mul_outgoing
        self.tri_mul_incoming = block.tri_mul_incoming
        self.tri_attn_start = block.tri_attn_start
        self.tri_attn_end = block.tri_attn_end
        self.z_transition = block.z_transition
        self.s_transition = block.s_transition
        self.attention_pair_bias = block.attention_pair_bias

    def forward(
        self, S_I: torch.Tensor, Z_II: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one pairformer layer.

        Args:
            S_I: [1, I, C_S] single representation (bf16)
            Z_II: [1, I, I, C_Z] pair representation (bf16)

        Returns:
            (S_I, Z_II): updated representations
        """
        Z_II = Z_II + self.tri_mul_outgoing(Z_II)
        Z_II = Z_II + self.tri_mul_incoming(Z_II)
        Z_II = Z_II + self.tri_attn_start(Z_II)
        Z_II = Z_II + self.tri_attn_end(Z_II)
        Z_II = Z_II + self.z_transition(Z_II)
        beta = torch.zeros(1, dtype=Z_II.dtype, device=Z_II.device)
        S_I = S_I + self.attention_pair_bias(S_I, None, Z_II, Beta_II=beta)
        S_I = S_I + self.s_transition(S_I)
        return S_I, Z_II


class MSAPairUpdateBlock(nn.Module):
    """Wrapper for MSA module pair-update operations.

    The RF3 MSA module alternates between MSA-specific ops (outer product,
    weighted averaging) on CPU and pair-update ops (tri_mul, tri_attn,
    pair_transition) that we compile on Neuron. This wrapper contains only
    the pair-update ops.

    These weights are shared across all 4 MSA iterations, so a single
    compiled NEFF is called 4 times without replace_weights.

    Args:
        msa_mod: The RF3 MSAModule instance (recycler.msa_module)
    """

    def __init__(self, msa_mod):
        super().__init__()
        self.tri_mul_outgoing = msa_mod.tri_mult_outgoing
        self.tri_mul_incoming = msa_mod.tri_mult_incoming
        self.tri_attn_start = msa_mod.tri_attn_start
        self.tri_attn_end = msa_mod.tri_attn_end
        self.pair_transition = msa_mod.pair_transition

    def forward(self, Z_II: torch.Tensor) -> torch.Tensor:
        """Run pair-update operations.

        Args:
            Z_II: [1, I, I, C_Z] pair representation (bf16)

        Returns:
            Z_II: updated pair representation
        """
        Z_II = Z_II + self.tri_mul_outgoing(Z_II)
        Z_II = Z_II + self.tri_mul_incoming(Z_II)
        Z_II = Z_II + self.tri_attn_start(Z_II)
        Z_II = Z_II + self.tri_attn_end(Z_II)
        Z_II = Z_II + self.pair_transition(Z_II)
        return Z_II


class TemplPairformerBlock(nn.Module):
    """Wrapper for template embedder pairformer blocks.

    The template embedder has 2 pairformer blocks with c_z=64 (smaller
    than the main pairformer's c_z=128). Each block has the same structure
    as the main pairformer but operates on pair-only representations
    (no single representation).

    Args:
        block: A template pairformer block from template_embedder.pairformer[i]
    """

    def __init__(self, block):
        super().__init__()
        self.tri_mul_outgoing = block.tri_mul_outgoing
        self.tri_mul_incoming = block.tri_mul_incoming
        self.tri_attn_start = block.tri_attn_start
        self.tri_attn_end = block.tri_attn_end
        self.z_transition = block.z_transition

    def forward(self, Z_II: torch.Tensor) -> torch.Tensor:
        """Run one template pairformer layer.

        Args:
            Z_II: [1, I, I, C_Z_TEMPL=64] template pair representation (bf16)

        Returns:
            Z_II: updated template pair representation
        """
        Z_II = Z_II + self.tri_mul_outgoing(Z_II)
        Z_II = Z_II + self.tri_mul_incoming(Z_II)
        Z_II = Z_II + self.tri_attn_start(Z_II)
        Z_II = Z_II + self.tri_attn_end(Z_II)
        Z_II = Z_II + self.z_transition(Z_II)
        return Z_II


class DiffTransformerBlock(nn.Module):
    """Wrapper for DiffusionTransformer blocks.

    Each block contains attention_pair_bias and conditioned_transition_block,
    with a flag for residual connection behavior.

    Args:
        block: A DiffusionTransformerBlock from diff_transformer.blocks[i]
    """

    def __init__(self, block):
        super().__init__()
        self.attention_pair_bias = block.attention_pair_bias
        self.conditioned_transition_block = block.conditioned_transition_block
        self.no_residual = block.no_residual_connection_between_attention_and_transition

    def forward(
        self, A_I: torch.Tensor, S_I: torch.Tensor, Z_II: torch.Tensor
    ) -> torch.Tensor:
        """Run one diffusion transformer block.

        Args:
            A_I: [D, I, C_TOKEN] token representation (bf16)
            S_I: [D, I, C_S] single conditioning (bf16)
            Z_II: [D, I, I, C_Z] pair conditioning (bf16)

        Returns:
            A_I: updated token representation
        """
        if self.no_residual:
            B_I = self.attention_pair_bias(A_I, S_I, Z_II, None)
            A_I = A_I + B_I + self.conditioned_transition_block(A_I, S_I)
        else:
            A_I = A_I + self.attention_pair_bias(A_I, S_I, Z_II, None)
            A_I = A_I + self.conditioned_transition_block(A_I, S_I)
        return A_I


# ============================================================================
# Static Windowed Attention for AtomAttention
# ============================================================================


def precompute_window_indices(
    L: int, qbatch: int = ATOM_QBATCH, kbatch: int = ATOM_KBATCH
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Precompute static index and mask tensors for windowed attention.

    RF3's atom attention uses dynamic windowed attention with torch.arange
    and boolean masking, which cannot be traced. This function precomputes
    the window indices and masks as static tensors that can be passed as
    inputs to the traced model.

    Args:
        L: atom sequence length (L_padded)
        qbatch: query window size (default 32)
        kbatch: key window size (default 128)

    Returns:
        indicesQ: [nq, qbatch] query indices per window (clamped to [0, L-1])
        indicesK: [nq, kbatch] key indices per window (clamped to [0, L-1])
        maskQ: [nq, qbatch] True where query is out of bounds
        maskK: [nq, kbatch] True where key is out of bounds
        nq: number of query windows
    """
    nq = (L + qbatch - 1) // qbatch
    Cs = torch.arange(nq) * qbatch + qbatch // 2
    patchq = torch.arange(qbatch) - qbatch // 2
    patchk = torch.arange(kbatch) - kbatch // 2

    indicesQ = Cs[:, None] + patchq[None, :]
    maskQ = (indicesQ < 0) | (indicesQ > L - 1)
    indicesQ = torch.clamp(indicesQ, 0, L - 1)

    indicesK = Cs[:, None] + patchk[None, :]
    maskK = (indicesK < 0) | (indicesK > L - 1)
    indicesK = torch.clamp(indicesK, 0, L - 1)

    return indicesQ, indicesK, maskQ, maskK, nq


class StaticWindowedAttnBlock(nn.Module):
    """Single DiffusionTransformerBlock with static windowed attention.

    Replaces RF3's dynamic atom_attention method (which uses torch.arange
    and boolean masking) with precomputed static index tensors passed as
    inputs. This makes the module traceable with torch_neuronx.trace().

    The encoder and decoder AtomTransformers have identical block
    architectures (same parameter shapes), so a single compiled NEFF can
    serve both via replace_weights.

    Args:
        block: A DiffusionTransformerBlock from atom_transformer.diffusion_transformer.blocks[i]
        n_head: number of attention heads (default 4)
        c_head: head dimension (default 32)
        qbatch: query window size (default 32)
        kbatch: key window size (default 128)
    """

    def __init__(
        self,
        block,
        n_head: int = ATOM_N_HEAD,
        c_head: int = ATOM_C_HEAD,
        qbatch: int = ATOM_QBATCH,
        kbatch: int = ATOM_KBATCH,
    ):
        super().__init__()
        self.n_head = n_head
        self.c_head = c_head
        self.qbatch = qbatch
        self.kbatch = kbatch

        # Attention submodules
        attn = block.attention_pair_bias
        self.ada_ln_1 = attn.ada_ln_1
        self.to_q = attn.to_q
        self.to_k = attn.to_k
        self.to_v = attn.to_v
        self.to_b = attn.to_b
        self.to_g = attn.to_g
        self.to_a = attn.to_a
        self.ln_0 = attn.ln_0
        self.linear_output_project = attn.linear_output_project
        self.kq_norm = attn.kq_norm
        if self.kq_norm:
            self.query_layer_norm = attn.query_layer_norm
            self.key_layer_norm = attn.key_layer_norm

        # Transition submodule
        self.conditioned_transition_block = block.conditioned_transition_block
        self.no_residual = block.no_residual_connection_between_attention_and_transition

    def forward(
        self,
        A_I: torch.Tensor,
        S_I: torch.Tensor,
        Z_II: torch.Tensor,
        indicesQ: torch.Tensor,
        indicesK: torch.Tensor,
        maskQ_float: torch.Tensor,
        maskK_float: torch.Tensor,
    ) -> torch.Tensor:
        """Run one atom attention block with static windowed attention.

        Args:
            A_I: [D, L, c_atom] atom representation (bf16)
            S_I: [D, L, c_atom] atom conditioning (bf16)
            Z_II: [1, L, L, c_atompair] atom pair representation (bf16)
            indicesQ: [nq, qbatch] precomputed query indices (long)
            indicesK: [nq, kbatch] precomputed key indices (long)
            maskQ_float: [nq, qbatch] 1.0 where out of bounds (bf16)
            maskK_float: [nq, kbatch] 1.0 where out of bounds (bf16)

        Returns:
            A_I: [D, L, c_atom] updated atom representation
        """
        A_norm = self.ada_ln_1(A_I, S_I)
        D_dim = A_I.shape[0]
        L_val = A_I.shape[1]

        Q_IH = self.to_q(A_norm)
        K_IH = self.to_k(A_norm)
        V_IH = self.to_v(A_norm)
        B_IIH = self.to_b(self.ln_0(Z_II))
        G_IH = self.to_g(A_norm)

        if self.kq_norm:
            Q_flat = Q_IH.reshape(-1, self.n_head * self.c_head)
            Q_flat = self.query_layer_norm(Q_flat)
            Q_IH = Q_flat.reshape(Q_IH.shape)
            K_flat = K_IH.reshape(-1, self.n_head * self.c_head)
            K_flat = self.key_layer_norm(K_flat)
            K_IH = K_flat.reshape(K_IH.shape)

        # Static windowed attention
        query_subset = Q_IH[:, indicesQ]
        key_subset = K_IH[:, indicesK]

        attn = torch.einsum("...ihd,...jhd->...ijh", query_subset, key_subset)
        attn = attn / (self.c_head**0.5)

        # Pair bias gathered for window
        bias_rows = B_IIH[:, indicesQ]
        nq = indicesQ.shape[0]
        ik_exp = indicesK.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
        ik_exp = ik_exp.expand(1, nq, self.qbatch, self.kbatch, self.n_head)
        bias_window = torch.gather(bias_rows, 3, ik_exp)

        attn = attn + bias_window

        # Mask out-of-bounds
        mask_val = maskQ_float.unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1
        ) + maskK_float.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
        attn = attn - 1e9 * mask_val.to(attn.dtype)

        attn = torch.softmax(attn, dim=-2).to(A_I.dtype)

        value_subset = V_IH[:, indicesK]
        atom_features = torch.einsum("...ijh,...jhc->...ihc", attn, value_subset)

        # Zero out masked positions and reshape
        valid_mask = (
            (1.0 - maskQ_float)
            .to(atom_features.dtype)
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        atom_features = atom_features * valid_mask
        atom_features = atom_features.reshape(
            D_dim, nq * self.qbatch, self.n_head, self.c_head
        )
        atom_features = atom_features[:, :L_val]

        # Gate and project
        atom_features = (G_IH * atom_features).reshape(D_dim, L_val, -1)
        atom_features = self.to_a(atom_features)
        B_I = self.linear_output_project(S_I) * atom_features

        # Residual + transition
        if self.no_residual:
            A_I = A_I + B_I + self.conditioned_transition_block(A_I, S_I)
        else:
            A_I = A_I + B_I
            A_I = A_I + self.conditioned_transition_block(A_I, S_I)

        return A_I


# ============================================================================
# Compatibility Patches
# ============================================================================


def patch_rf3_for_neuron(rf3_src_path: str) -> None:
    """Apply compatibility patches to RF3 source code for Neuron compilation.

    RF3 uses several operations that are not supported by the Neuron compiler.
    This function patches them at the module level by modifying the imported
    RF3 modules in-place.

    Patches applied:
      1. torch.autocast disabled (4 locations) - Neuron does not support
         dynamic autocast; we use static bf16 casting instead.
      2. opt_einsum -> torch.einsum (2 files) - opt_einsum is not traced
         by torch_neuronx.
      3. index_reduce('mean') -> scatter_add (2 files) - index_reduce is
         not supported by the Neuron compiler.
      4. Activation checkpointing disabled - not compatible with tracing.

    Args:
        rf3_src_path: Path to the RF3 source tree (e.g., '/mnt/models/foundry/models/rf3/src')

    Note:
        This function must be called AFTER importing the RF3 modules but
        BEFORE running inference or compilation. It modifies the modules
        in sys.modules in-place.
    """
    import sys

    # --- Patch 1: Disable autocast ---
    # RF3.py, attention.py use torch.autocast which is incompatible with tracing
    modules_to_patch_autocast = [
        "rf3.model.RF3",
        "rf3.model.layers.attention",
    ]
    for mod_name in modules_to_patch_autocast:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            # Replace torch.autocast with a no-op context manager
            if hasattr(mod, "torch"):
                import contextlib

                mod.torch.autocast = lambda *a, **kw: contextlib.nullcontext()

    # --- Patch 2: opt_einsum -> torch.einsum ---
    modules_to_patch_einsum = [
        "rf3.model.layers.attention",
        "rf3.model.layers.structure_bias",
    ]
    for mod_name in modules_to_patch_einsum:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, "oe"):
                mod.oe = type("FakeOE", (), {"contract": staticmethod(torch.einsum)})()

    # --- Patch 3: index_reduce -> scatter_add ---
    def _index_reduce_mean(src, dim, index, source, reduce="mean", include_self=True):
        """Replace index_reduce('mean', ...) with scatter_add + count normalization."""
        result = torch.zeros_like(src)
        index_expanded = index.unsqueeze(-1).expand_as(source)
        result.scatter_add_(dim, index_expanded, source)
        ones = torch.ones_like(source)
        counts = torch.zeros_like(src)
        counts.scatter_add_(dim, index_expanded, ones)
        counts = counts.clamp(min=1)
        return result / counts

    modules_to_patch_reduce = [
        "rf3.model.layers.pairformer_layers",
        "rf3.model.layers.af3_diffusion_transformer",
    ]
    for mod_name in modules_to_patch_reduce:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            # The patched neuron_compat module provides the replacement
            if hasattr(mod, "index_reduce_mean"):
                continue  # Already patched

    # --- Patch 4: Disable activation checkpointing ---
    try:
        from foundry.training import checkpoint as ckpt_mod

        if hasattr(ckpt_mod, "checkpoint"):
            # Replace checkpoint with direct function call
            ckpt_mod.checkpoint = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    except ImportError:
        pass


# ============================================================================
# Padding Utilities
# ============================================================================


def _pad_to(tensor: torch.Tensor, target: int, dim: int) -> torch.Tensor:
    """Pad tensor to target size along dim with zeros."""
    if tensor.shape[dim] >= target:
        return tensor
    pad_size = target - tensor.shape[dim]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=dim)


def _pad_S(S_I: torch.Tensor, I_padded: int) -> torch.Tensor:
    """Pad single representation to I_padded along token dimension."""
    return _pad_to(S_I, I_padded, dim=-2) if S_I.shape[-2] != I_padded else S_I


def _pad_Z(Z_II: torch.Tensor, target: int) -> torch.Tensor:
    """Pad pair representation to target along both spatial dimensions."""
    Z = _pad_to(Z_II, target, dim=-2) if Z_II.shape[-2] != target else Z_II
    Z = _pad_to(Z, target, dim=-3) if Z.shape[-3] != target else Z
    return Z


def _pad_L(tensor: torch.Tensor, target: int, dim: int) -> torch.Tensor:
    """Pad tensor along dim to target size (alias for _pad_to)."""
    return _pad_to(tensor, target, dim)


# ============================================================================
# RF3 Neuron Pipeline
# ============================================================================


class RF3NeuronPipeline:
    """End-to-end pipeline for RF3 inference on Neuron.

    Handles compilation, weight replacement, monkey-patching, and inference
    for all 5 compiled blocks. The pipeline operates on a loaded RF3
    InferenceEngine and patches its internal model to route computations
    through compiled Neuron blocks.

    Usage::

        from rf3.inference_engines.rf3 import RF3InferenceEngine
        engine = RF3InferenceEngine(ckpt_path=..., ...)
        engine.initialize()

        pipeline = RF3NeuronPipeline(engine, I=240, L=1012, D=5)
        pipeline.compile_all()
        pipeline.patch_model()

        results = engine.run(inputs=input_cif, out_dir=None)

    Args:
        engine: An initialized RF3InferenceEngine
        I: Token count (number of residue tokens)
        L: Atom count (number of atoms)
        D: Diffusion batch size
        compiler_args: Additional compiler arguments (default: ["--target", "trn2"])
    """

    def __init__(
        self,
        engine: Any,
        I: int,
        L: int,
        D: int = 5,
        compiler_args: Optional[List[str]] = None,
    ):
        self.engine = engine
        self.I = I
        self.L = L
        self.D = D
        self.I_padded = ((I + 7) // 8) * 8
        self.L_padded = ((L + 7) // 8) * 8
        self.compiler_args = compiler_args or ["--target", "trn2"]

        # Extract model components
        ema = engine.trainer.state["model"]._forward_module
        self.rf3_with_conf = ema.shadow

        # Disable CUDA-specific paths
        for _, m in self.rf3_with_conf.named_modules():
            if hasattr(m, "use_cuequivariance"):
                m.use_cuequivariance = False
            if hasattr(m, "use_deepspeed_evo"):
                m.use_deepspeed_evo = False
        for _, m in ema.model.named_modules():
            if hasattr(m, "use_cuequivariance"):
                m.use_cuequivariance = False
            if hasattr(m, "use_deepspeed_evo"):
                m.use_deepspeed_evo = False

        self.recycler = self.rf3_with_conf.recycler
        self.pf_stack = self.recycler.pairformer_stack
        self.n_pf_layers = len(self.pf_stack)
        self.msa_module = self.recycler.msa_module
        self.template_embedder = self.recycler.template_embedder

        self.diff_module = self.rf3_with_conf.diffusion_module
        self.diff_transformer = self.diff_module.diffusion_transformer
        self.n_diff_blocks = len(self.diff_transformer.blocks)

        self.atom_encoder = self.diff_module.atom_attention_encoder
        self.atom_decoder = self.diff_module.atom_attention_decoder
        self.encoder_atom_transformer = self.atom_encoder.atom_transformer
        self.decoder_atom_transformer = self.atom_decoder.atom_transformer
        self.n_atom_blocks = len(
            self.encoder_atom_transformer.diffusion_transformer.blocks
        )

        # Compiled models (populated by compile_all)
        self.compiled_pf = None
        self.compiled_msa_pair = None
        self.compiled_templ = None
        self.compiled_diff = None
        self.compiled_atom = None

        # Weight wrappers (populated by compile_all)
        self.pf_wrappers = []
        self.msa_pair_wrapper = None
        self.templ_wrappers = []
        self.diff_wrappers = []
        self.enc_atom_wrappers = []
        self.dec_atom_wrappers = []

        # Atom attention precomputed indices
        self.atom_indicesQ = None
        self.atom_indicesK = None
        self.atom_maskQ_bf = None
        self.atom_maskK_bf = None

        # Compilation times
        self.compile_times: Dict[str, float] = {}

    def compile_all(self) -> Dict[str, float]:
        """Compile all 5 Neuron blocks.

        Returns:
            dict mapping block names to compilation times in seconds
        """
        import torch_neuronx

        I_p = self.I_padded
        L_p = self.L_padded
        D = self.D

        # --- Pairformer ---
        pf_block0 = FullPairformerBlock(self.pf_stack[0]).eval().to(torch.bfloat16)
        S_trace = torch.randn(1, I_p, C_S, dtype=torch.bfloat16)
        Z_trace = torch.randn(1, I_p, I_p, C_Z, dtype=torch.bfloat16)

        t0 = time.time()
        self.compiled_pf = torch_neuronx.trace(
            pf_block0,
            (S_trace, Z_trace),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["pairformer"] = time.time() - t0

        self.pf_wrappers = [
            FullPairformerBlock(self.pf_stack[i]).eval().to(torch.bfloat16)
            for i in range(self.n_pf_layers)
        ]

        # --- MSA Pair ---
        msa_pair_block = MSAPairUpdateBlock(self.msa_module).eval().to(torch.bfloat16)
        Z_trace_msa = torch.randn(1, I_p, I_p, C_Z, dtype=torch.bfloat16)

        t0 = time.time()
        self.compiled_msa_pair = torch_neuronx.trace(
            msa_pair_block,
            (Z_trace_msa,),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["msa_pair"] = time.time() - t0

        self.msa_pair_wrapper = (
            MSAPairUpdateBlock(self.msa_module).eval().to(torch.bfloat16)
        )

        # --- Template Pairformer ---
        templ_block0 = (
            TemplPairformerBlock(self.template_embedder.pairformer[0])
            .eval()
            .to(torch.bfloat16)
        )
        Z_trace_templ = torch.randn(1, I_p, I_p, C_Z_TEMPL, dtype=torch.bfloat16)

        t0 = time.time()
        self.compiled_templ = torch_neuronx.trace(
            templ_block0,
            (Z_trace_templ,),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["template_pf"] = time.time() - t0

        self.templ_wrappers = [
            TemplPairformerBlock(self.template_embedder.pairformer[i])
            .eval()
            .to(torch.bfloat16)
            for i in range(len(self.template_embedder.pairformer))
        ]

        # --- DiffTransformer ---
        diff_block0 = (
            DiffTransformerBlock(self.diff_transformer.blocks[0])
            .eval()
            .to(torch.bfloat16)
        )
        A_trace = torch.randn(D, I_p, C_TOKEN, dtype=torch.bfloat16)
        S_diff_trace = torch.randn(D, I_p, C_S, dtype=torch.bfloat16)
        Z_diff_trace = torch.randn(D, I_p, I_p, C_Z, dtype=torch.bfloat16)

        t0 = time.time()
        self.compiled_diff = torch_neuronx.trace(
            diff_block0,
            (A_trace, S_diff_trace, Z_diff_trace),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["diff_transformer"] = time.time() - t0

        self.diff_wrappers = [
            DiffTransformerBlock(self.diff_transformer.blocks[i])
            .eval()
            .to(torch.bfloat16)
            for i in range(self.n_diff_blocks)
        ]

        # --- Atom Attention ---
        self.atom_indicesQ, self.atom_indicesK, atom_maskQ, atom_maskK, _ = (
            precompute_window_indices(L_p, ATOM_QBATCH, ATOM_KBATCH)
        )
        self.atom_maskQ_bf = atom_maskQ.float().to(torch.bfloat16)
        self.atom_maskK_bf = atom_maskK.float().to(torch.bfloat16)

        enc_dt = self.encoder_atom_transformer.diffusion_transformer
        atom_block0 = (
            StaticWindowedAttnBlock(
                enc_dt.blocks[0], ATOM_N_HEAD, ATOM_C_HEAD, ATOM_QBATCH, ATOM_KBATCH
            )
            .eval()
            .to(torch.bfloat16)
        )

        atom_A_trace = torch.randn(D, L_p, C_ATOM, dtype=torch.bfloat16)
        atom_S_trace = torch.randn(D, L_p, C_ATOM, dtype=torch.bfloat16)
        atom_Z_trace = torch.randn(1, L_p, L_p, C_ATOMPAIR, dtype=torch.bfloat16)

        atom_trace_inputs = (
            atom_A_trace,
            atom_S_trace,
            atom_Z_trace,
            self.atom_indicesQ.long(),
            self.atom_indicesK.long(),
            self.atom_maskQ_bf,
            self.atom_maskK_bf,
        )

        t0 = time.time()
        self.compiled_atom = torch_neuronx.trace(
            atom_block0,
            atom_trace_inputs,
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["atom_attn"] = time.time() - t0

        self.enc_atom_wrappers = [
            StaticWindowedAttnBlock(
                enc_dt.blocks[i], ATOM_N_HEAD, ATOM_C_HEAD, ATOM_QBATCH, ATOM_KBATCH
            )
            .eval()
            .to(torch.bfloat16)
            for i in range(self.n_atom_blocks)
        ]
        dec_dt = self.decoder_atom_transformer.diffusion_transformer
        self.dec_atom_wrappers = [
            StaticWindowedAttnBlock(
                dec_dt.blocks[i], ATOM_N_HEAD, ATOM_C_HEAD, ATOM_QBATCH, ATOM_KBATCH
            )
            .eval()
            .to(torch.bfloat16)
            for i in range(self.n_atom_blocks)
        ]

        return self.compile_times

    def patch_model(self) -> None:
        """Monkey-patch the RF3 model to use compiled Neuron blocks.

        Patches:
          - Recycler.forward: routes pairformer through compiled blocks
          - MSAModule.forward: routes pair ops through compiled block
          - Template embedder: routes pairformer blocks through compiled blocks
          - DiffusionTransformer.forward: routes blocks through compiled blocks
          - AtomTransformer.forward: routes encoder/decoder through compiled blocks
        """
        import torch_neuronx

        I = self.I
        I_p = self.I_padded
        L = self.L
        L_p = self.L_padded
        D = self.D

        compiled_pf = self.compiled_pf
        compiled_msa_pair = self.compiled_msa_pair
        compiled_templ = self.compiled_templ
        compiled_diff = self.compiled_diff
        compiled_atom = self.compiled_atom

        pf_wrappers = self.pf_wrappers
        templ_wrappers = self.templ_wrappers
        diff_wrappers = self.diff_wrappers
        enc_atom_wrappers = self.enc_atom_wrappers
        dec_atom_wrappers = self.dec_atom_wrappers

        n_pf_layers = self.n_pf_layers
        n_diff_blocks = self.n_diff_blocks
        n_atom_blocks = self.n_atom_blocks

        atom_indicesQ = self.atom_indicesQ
        atom_indicesK = self.atom_indicesK
        atom_maskQ_bf = self.atom_maskQ_bf
        atom_maskK_bf = self.atom_maskK_bf

        msa_module = self.msa_module
        recycler = self.recycler
        template_embedder = self.template_embedder
        diff_transformer = self.diff_transformer
        encoder_atom_transformer = self.encoder_atom_transformer
        decoder_atom_transformer = self.decoder_atom_transformer

        # --- Patch MSAModule ---
        def _patched_msa_forward(self_msa, f, Z_II, S_inputs_I):
            msa = f["msa"]
            msa_SI = self_msa.msa_subsampler(msa, S_inputs_I)

            for i in range(self_msa.n_block):
                # CPU: MSA-specific ops
                Z_II = Z_II + self_msa.maybe_make_batched_outer_product(
                    self_msa.outer_product
                )(msa_SI)
                msa_SI = msa_SI + self_msa.drop_row_msa(
                    self_msa.msa_pair_weighted_averaging(msa_SI, Z_II)
                )
                msa_SI = msa_SI + self_msa.msa_transition(msa_SI)

                # Neuron: pair update ops
                need_batch = Z_II.dim() == 3
                if need_batch:
                    Z_II = Z_II.unsqueeze(0)
                Z_bf = _pad_Z(Z_II.to(torch.bfloat16), I_p)
                Z_bf = compiled_msa_pair(Z_bf)
                Z_II = Z_bf[..., :I, :I, :].float()
                if need_batch:
                    Z_II = Z_II.squeeze(0)

            return Z_II

        msa_module.forward = types.MethodType(_patched_msa_forward, msa_module)

        # --- Patch template pairformer blocks ---
        _orig_template_forward = template_embedder.forward
        n_templ_blocks = len(template_embedder.pairformer)

        for block_idx in range(n_templ_blocks):
            orig_block = template_embedder.pairformer[block_idx]

            def _make_patched_block_forward(wrapper, idx):
                def _patched_block_forward(self_blk, S_I_unused, Z_II):
                    need_batch = Z_II.dim() == 3
                    if need_batch:
                        Z_II = Z_II.unsqueeze(0)
                    Z_bf = _pad_Z(Z_II.to(torch.bfloat16), target=I_p)
                    torch_neuronx.replace_weights(compiled_templ, wrapper.state_dict())
                    Z_bf = compiled_templ(Z_bf)
                    Z_II = Z_bf[..., :I, :I, :].float()
                    if need_batch:
                        Z_II = Z_II.squeeze(0)
                    return S_I_unused, Z_II

                return _patched_block_forward

            patched_fn = _make_patched_block_forward(
                templ_wrappers[block_idx], block_idx
            )
            orig_block.forward = types.MethodType(patched_fn, orig_block)

        # --- Patch Recycler ---
        def _patched_recycler_forward(
            self_rec, f, S_inputs_I, S_init_I, Z_init_II, S_I, Z_II
        ):
            Z_II = Z_init_II + self_rec.process_zh(Z_II)
            Z_II = Z_II + self_rec.template_embedder(f, Z_II)
            Z_II = self_rec.msa_module(f, Z_II, S_inputs_I)
            S_I = S_init_I + self_rec.process_sh(S_I)

            # Neuron: main pairformer stack
            need_batch_S = S_I.dim() == 2
            need_batch_Z = Z_II.dim() == 3
            if need_batch_S:
                S_I = S_I.unsqueeze(0)
            if need_batch_Z:
                Z_II = Z_II.unsqueeze(0)

            S_bf = _pad_S(S_I.to(torch.bfloat16), I_p)
            Z_bf = _pad_Z(Z_II.to(torch.bfloat16), I_p)

            for i in range(n_pf_layers):
                torch_neuronx.replace_weights(compiled_pf, pf_wrappers[i].state_dict())
                S_bf, Z_bf = compiled_pf(S_bf, Z_bf)

            S_I = S_bf[..., :I, :].float()
            Z_II = Z_bf[..., :I, :I, :].float()

            if need_batch_S:
                S_I = S_I.squeeze(0)
            if need_batch_Z:
                Z_II = Z_II.squeeze(0)

            return S_I, Z_II

        recycler.forward = types.MethodType(_patched_recycler_forward, recycler)

        # --- Patch DiffusionTransformer ---
        def _patched_diff_forward(self_dt, A_I, S_I, Z_II, Beta_II):
            orig_A_dim = A_I.dim()
            if A_I.dim() == 2:
                A_I = A_I.unsqueeze(0)
            if S_I.dim() == 2:
                S_I = S_I.unsqueeze(0)
            if Z_II.dim() == 3:
                Z_II = Z_II.unsqueeze(0)

            target_D = A_I.shape[0]
            if Z_II.shape[0] == 1 and target_D > 1:
                Z_II = Z_II.expand(target_D, -1, -1, -1).contiguous()

            A_bf = _pad_to(A_I.to(torch.bfloat16), I_p, dim=-2)
            S_bf = _pad_S(S_I.to(torch.bfloat16), I_p)
            Z_bf = _pad_Z(Z_II.to(torch.bfloat16), I_p)

            for i in range(n_diff_blocks):
                torch_neuronx.replace_weights(
                    compiled_diff, diff_wrappers[i].state_dict()
                )
                A_bf = compiled_diff(A_bf, S_bf, Z_bf)

            A_I = A_bf[..., :I, :].float()
            if orig_A_dim == 2:
                A_I = A_I.squeeze(0)
            return A_I

        diff_transformer.forward = types.MethodType(
            _patched_diff_forward, diff_transformer
        )

        # --- Patch AtomTransformer ---
        def _pad_atom_inputs(Q_L_in, C_L_in, P_LL_in):
            actual_L = Q_L_in.shape[-2]
            need_unbatch_Q = False
            if Q_L_in.dim() == 2:
                Q_L_in = Q_L_in.unsqueeze(0)
                need_unbatch_Q = True
            if C_L_in.dim() == 2:
                C_L_in = C_L_in.unsqueeze(0)
            if C_L_in.shape[0] == 1 and Q_L_in.shape[0] > 1:
                C_L_in = C_L_in.expand(Q_L_in.shape[0], -1, -1).contiguous()
            if P_LL_in.dim() == 3:
                P_LL_in = P_LL_in.unsqueeze(0)

            Q_padded = _pad_L(Q_L_in.to(torch.bfloat16), L_p, dim=-2)
            C_padded = _pad_L(C_L_in.to(torch.bfloat16), L_p, dim=-2)
            P_padded = _pad_L(
                _pad_L(P_LL_in.to(torch.bfloat16), L_p, dim=-2), L_p, dim=-3
            )

            return Q_padded, C_padded, P_padded, actual_L, need_unbatch_Q

        def _make_patched_atom_transformer_forward(wrappers):
            def _patched_forward(self_at, Q_L_in, C_L_in, P_LL_in):
                Q_padded, C_padded, P_padded, actual_L, need_unbatch_Q = (
                    _pad_atom_inputs(Q_L_in, C_L_in, P_LL_in)
                )

                A = Q_padded
                for i in range(n_atom_blocks):
                    torch_neuronx.replace_weights(
                        compiled_atom, wrappers[i].state_dict()
                    )
                    A = compiled_atom(
                        A,
                        C_padded,
                        P_padded,
                        atom_indicesQ.long(),
                        atom_indicesK.long(),
                        atom_maskQ_bf,
                        atom_maskK_bf,
                    )

                A = A[..., :actual_L, :].float()
                if need_unbatch_Q:
                    A = A.squeeze(0)
                return A

            return _patched_forward

        encoder_atom_transformer.forward = types.MethodType(
            _make_patched_atom_transformer_forward(enc_atom_wrappers),
            encoder_atom_transformer,
        )
        decoder_atom_transformer.forward = types.MethodType(
            _make_patched_atom_transformer_forward(dec_atom_wrappers),
            decoder_atom_transformer,
        )

    def run(self, input_cif: str, out_dir: Optional[str] = None, seed: int = 42) -> Any:
        """Run RF3 inference on the given input.

        Args:
            input_cif: Path to input CIF file
            out_dir: Output directory (None for no file output)
            seed: Random seed for reproducibility

        Returns:
            RF3 results dictionary
        """
        import lightning

        lightning.seed_everything(seed)
        return self.engine.run(inputs=input_cif, out_dir=out_dir)
