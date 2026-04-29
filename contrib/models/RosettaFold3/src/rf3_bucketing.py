#!/usr/bin/env python3
"""Shape bucketing and padding utilities for RF3 on Neuron.

Provides:
  - Bucket definitions for token (I) and atom (L) dimensions
  - pad_to_bucket / unpad functions for all tensor types
  - Memory requirement estimation per bucket
  - Bucket selection given arbitrary input sizes

Bucket sizes are chosen based on:
  - Neuron compiler limits: pairformer max I=448 (SB limit), DiffTransformer max I=512
  - Compilation time: larger buckets take longer to compile
  - Memory: pair representation is O(I^2), atom pairs are O(L^2)
  - Alignment: multiples of 8 (Neuron hardware alignment)

Usage:
    from rf3_bucketing import BucketConfig, pad_tensor, unpad_tensor, select_bucket

    config = BucketConfig()
    I_bucket = config.select_I_bucket(I_actual=240)  # -> 240
    L_bucket = config.select_L_bucket(L_actual=1012)  # -> 1016

    S_padded = pad_tensor(S_I, target=I_bucket, dims=[-2])
    S_out = unpad_tensor(S_padded, original=I_actual, dims=[-2])
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# ============================================================================
# Bucket Definitions
# ============================================================================

# Token-level (I) buckets — controls pairformer, DiffTransformer, MSA, template
# Max pairformer I=448 (vanilla compilation limit, Task 014)
# Max DiffTransformer I=512 with D=5 (Task 014)
I_BUCKETS = [128, 192, 256, 320, 384, 448]

# Atom-level (L) buckets — controls AtomAttention encoder/decoder
# L is typically ~4x I (each residue has ~4-5 atoms in the token representation)
# Windowed attention uses qbatch=32, kbatch=128, so L should be multiple of 128
# for optimal alignment, but the static windowed attention wrapper handles
# arbitrary L_padded as long as it's a multiple of 8.
L_BUCKETS = [256, 512, 768, 1024, 1280, 1536, 2048]

# Diffusion batch (D) — typically 1 or 5, fixed per run
D_VALUES = [1, 5]


@dataclass
class BucketConfig:
    """Configuration for RF3 shape bucketing on Neuron."""

    # Available bucket sizes
    I_buckets: List[int] = field(default_factory=lambda: list(I_BUCKETS))
    L_buckets: List[int] = field(default_factory=lambda: list(L_BUCKETS))

    # Compilation limits (from Task 014 experiments)
    max_I_pairformer: int = 448  # Compiler SB limit
    max_I_difftransformer: int = 512  # D=5, higher for D=1
    max_L_atom_attention: int = 2048  # Estimated limit

    # Architecture constants
    C_S: int = 384
    C_Z: int = 128
    C_Z_TEMPL: int = 64
    C_TOKEN: int = 768
    C_ATOM: int = 128
    C_ATOMPAIR: int = 16

    def select_I_bucket(self, I_actual: int) -> int:
        """Select the smallest I bucket that fits the input.

        Args:
            I_actual: actual token count

        Returns:
            I_padded: bucket size to pad to

        Raises:
            ValueError: if I_actual exceeds the largest bucket
        """
        for bucket in sorted(self.I_buckets):
            if bucket >= I_actual:
                return bucket
        raise ValueError(
            f"Token count I={I_actual} exceeds largest bucket "
            f"{max(self.I_buckets)}. Max compilable pairformer I={self.max_I_pairformer}."
        )

    def select_L_bucket(self, L_actual: int) -> int:
        """Select the smallest L bucket that fits the input.

        Args:
            L_actual: actual atom count

        Returns:
            L_padded: bucket size to pad to

        Raises:
            ValueError: if L_actual exceeds the largest bucket
        """
        for bucket in sorted(self.L_buckets):
            if bucket >= L_actual:
                return bucket
        raise ValueError(
            f"Atom count L={L_actual} exceeds largest bucket {max(self.L_buckets)}."
        )

    def estimate_memory_mb(self, I: int, L: int, D: int = 5) -> Dict[str, float]:
        """Estimate HBM memory requirements for a given bucket configuration.

        Args:
            I: token bucket size
            L: atom bucket size
            D: diffusion batch size

        Returns:
            dict with memory estimates in MB for each tensor category
        """
        bytes_per_elem = 2  # bf16
        MB = 1024 * 1024

        mem = {}

        # Pair representation Z_II: [I, I, C_Z] — biggest single tensor
        mem["Z_II"] = I * I * self.C_Z * bytes_per_elem / MB

        # Single representation S_I: [I, C_S]
        mem["S_I"] = I * self.C_S * bytes_per_elem / MB

        # Token representation A_I: [D, I, C_TOKEN]
        mem["A_I_D"] = D * I * self.C_TOKEN * bytes_per_elem / MB

        # Diffusion Z (batched): [D, I, I, C_Z]
        mem["Z_II_D"] = D * I * I * self.C_Z * bytes_per_elem / MB

        # Diffusion S: [D, I, C_S]
        mem["S_I_D"] = D * I * self.C_S * bytes_per_elem / MB

        # Atom pair representation P_LL: [L, L, C_ATOMPAIR]
        mem["P_LL"] = L * L * self.C_ATOMPAIR * bytes_per_elem / MB

        # Atom features Q_L: [D, L, C_ATOM]
        mem["Q_L_D"] = D * L * self.C_ATOM * bytes_per_elem / MB

        # Template Z: [I, I, C_Z_TEMPL]
        mem["Z_templ"] = I * I * self.C_Z_TEMPL * bytes_per_elem / MB

        # Total (approximate, not counting intermediates)
        mem["total_approximate"] = sum(mem.values())

        # Per-core HBM budget (trn2 LNC=2: 24 GB/core)
        mem["hbm_per_core_gb"] = 24.0
        mem["utilization_pct"] = mem["total_approximate"] / (24 * 1024) * 100

        return mem

    def print_bucket_table(self):
        """Print a formatted table of all buckets with memory estimates."""
        print(f"\n{'=' * 80}")
        print(f"RF3 Shape Bucket Configuration")
        print(f"{'=' * 80}")

        print(f"\nToken (I) Buckets:")
        print(
            f"  {'I':>6}  {'Z_II (MB)':>10}  {'Z_II_D5 (MB)':>13}  {'Compile?':>10}  {'Est. Compile Time':>18}"
        )
        print(f"  {'-' * 6}  {'-' * 10}  {'-' * 13}  {'-' * 10}  {'-' * 18}")
        for I in sorted(self.I_buckets):
            z_mb = I * I * self.C_Z * 2 / (1024 * 1024)
            z_d5_mb = 5 * z_mb
            can_compile = "YES" if I <= self.max_I_pairformer else "NO (SB)"
            # Rough compile time estimate based on Task 014 data
            if I <= 240:
                est_time = "~25s"
            elif I <= 256:
                est_time = "~25s"
            elif I <= 384:
                est_time = "~2 min"
            elif I <= 448:
                est_time = "~2.5 min"
            else:
                est_time = "N/A"
            print(
                f"  {I:>6}  {z_mb:>10.1f}  {z_d5_mb:>13.1f}  {can_compile:>10}  {est_time:>18}"
            )

        print(f"\nAtom (L) Buckets:")
        print(f"  {'L':>6}  {'P_LL (MB)':>10}  {'Q_L_D5 (MB)':>12}")
        print(f"  {'-' * 6}  {'-' * 10}  {'-' * 12}")
        for L in sorted(self.L_buckets):
            p_mb = L * L * self.C_ATOMPAIR * 2 / (1024 * 1024)
            q_mb = 5 * L * self.C_ATOM * 2 / (1024 * 1024)
            print(f"  {L:>6}  {p_mb:>10.1f}  {q_mb:>12.1f}")

        print(f"\nCompilation Limits:")
        print(f"  Pairformer max I:        {self.max_I_pairformer}")
        print(f"  DiffTransformer max I:   {self.max_I_difftransformer} (D=5)")
        print(f"  AtomAttention max L:     {self.max_L_atom_attention} (estimated)")
        print(f"{'=' * 80}")


# ============================================================================
# Padding / Unpadding Utilities
# ============================================================================


def pad_tensor(tensor: torch.Tensor, target: int, dim: int) -> torch.Tensor:
    """Pad a tensor to a target size along a given dimension with zeros.

    Args:
        tensor: input tensor
        target: target size for the dimension
        dim: which dimension to pad (supports negative indexing)

    Returns:
        padded tensor
    """
    actual = tensor.shape[dim]
    if actual >= target:
        return tensor
    if actual == target:
        return tensor

    pad_size = target - actual
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=dim)


def unpad_tensor(tensor: torch.Tensor, original: int, dim: int) -> torch.Tensor:
    """Remove padding from a tensor along a given dimension.

    Args:
        tensor: padded tensor
        original: original (unpadded) size
        dim: which dimension to unpad

    Returns:
        unpadded tensor via slicing
    """
    if tensor.shape[dim] == original:
        return tensor
    idx = [slice(None)] * tensor.ndim
    idx[dim] = slice(0, original)
    return tensor[tuple(idx)]


def pad_pair(Z: torch.Tensor, I_target: int) -> torch.Tensor:
    """Pad a pair representation [*, I, I, C] along both spatial dimensions.

    Handles 3D [I, I, C] and 4D [B, I, I, C] inputs.

    Args:
        Z: pair representation tensor
        I_target: target I dimension

    Returns:
        padded pair representation
    """
    if Z.dim() == 3:
        # [I, I, C]
        Z = pad_tensor(Z, I_target, dim=0)
        Z = pad_tensor(Z, I_target, dim=1)
    elif Z.dim() == 4:
        # [B, I, I, C]
        Z = pad_tensor(Z, I_target, dim=1)
        Z = pad_tensor(Z, I_target, dim=2)
    else:
        raise ValueError(f"Expected 3D or 4D pair tensor, got {Z.dim()}D")
    return Z


def unpad_pair(Z: torch.Tensor, I_original: int) -> torch.Tensor:
    """Remove padding from a pair representation.

    Args:
        Z: padded pair representation
        I_original: original I dimension

    Returns:
        unpadded pair representation
    """
    if Z.dim() == 3:
        return Z[:I_original, :I_original, :]
    elif Z.dim() == 4:
        return Z[:, :I_original, :I_original, :]
    else:
        raise ValueError(f"Expected 3D or 4D pair tensor, got {Z.dim()}D")


def pad_single(S: torch.Tensor, I_target: int) -> torch.Tensor:
    """Pad a single representation [*, I, C] along the I dimension.

    Handles 2D [I, C] and 3D [B, I, C] inputs.

    Args:
        S: single representation tensor
        I_target: target I dimension

    Returns:
        padded single representation
    """
    if S.dim() == 2:
        return pad_tensor(S, I_target, dim=0)
    elif S.dim() == 3:
        return pad_tensor(S, I_target, dim=1)
    else:
        raise ValueError(f"Expected 2D or 3D single tensor, got {S.dim()}D")


def unpad_single(S: torch.Tensor, I_original: int) -> torch.Tensor:
    """Remove padding from a single representation.

    Args:
        S: padded single representation
        I_original: original I dimension

    Returns:
        unpadded single representation
    """
    if S.dim() == 2:
        return S[:I_original, :]
    elif S.dim() == 3:
        return S[:, :I_original, :]
    else:
        raise ValueError(f"Expected 2D or 3D single tensor, got {S.dim()}D")


def pad_atom_pair(P: torch.Tensor, L_target: int) -> torch.Tensor:
    """Pad atom pair representation [*, L, L, C] along both L dimensions."""
    if P.dim() == 3:
        P = pad_tensor(P, L_target, dim=0)
        P = pad_tensor(P, L_target, dim=1)
    elif P.dim() == 4:
        P = pad_tensor(P, L_target, dim=1)
        P = pad_tensor(P, L_target, dim=2)
    else:
        raise ValueError(f"Expected 3D or 4D atom pair tensor, got {P.dim()}D")
    return P


def unpad_atom_pair(P: torch.Tensor, L_original: int) -> torch.Tensor:
    """Remove padding from atom pair representation."""
    if P.dim() == 3:
        return P[:L_original, :L_original, :]
    elif P.dim() == 4:
        return P[:, :L_original, :L_original, :]
    else:
        raise ValueError(f"Expected 3D or 4D atom pair tensor, got {P.dim()}D")


def pad_atom_single(Q: torch.Tensor, L_target: int) -> torch.Tensor:
    """Pad atom single representation [*, L, C] along the L dimension."""
    if Q.dim() == 2:
        return pad_tensor(Q, L_target, dim=0)
    elif Q.dim() == 3:
        return pad_tensor(Q, L_target, dim=1)
    else:
        raise ValueError(f"Expected 2D or 3D atom single tensor, got {Q.dim()}D")


def unpad_atom_single(Q: torch.Tensor, L_original: int) -> torch.Tensor:
    """Remove padding from atom single representation."""
    if Q.dim() == 2:
        return Q[:L_original, :]
    elif Q.dim() == 3:
        return Q[:, :L_original, :]
    else:
        raise ValueError(f"Expected 2D or 3D atom single tensor, got {Q.dim()}D")


# ============================================================================
# Mask Generation
# ============================================================================


def make_pair_mask(I_actual: int, I_padded: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Create a pair mask [I_padded, I_padded] where padded positions are 0.

    Used to zero out attention contributions from padded positions.

    Args:
        I_actual: actual (unpadded) token count
        I_padded: padded token count (bucket size)
        dtype: mask dtype

    Returns:
        mask tensor [I_padded, I_padded]
    """
    mask = torch.zeros(I_padded, I_padded, dtype=dtype)
    mask[:I_actual, :I_actual] = 1.0
    return mask


def make_single_mask(
    I_actual: int, I_padded: int, dtype=torch.bfloat16
) -> torch.Tensor:
    """Create a single mask [I_padded] where padded positions are 0.

    Args:
        I_actual: actual token count
        I_padded: padded token count

    Returns:
        mask tensor [I_padded]
    """
    mask = torch.zeros(I_padded, dtype=dtype)
    mask[:I_actual] = 1.0
    return mask


def make_atom_mask(L_actual: int, L_padded: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Create an atom mask [L_padded] where padded positions are 0."""
    mask = torch.zeros(L_padded, dtype=dtype)
    mask[:L_actual] = 1.0
    return mask


# ============================================================================
# Compiled Model Cache (multi-bucket support)
# ============================================================================


class CompiledModelCache:
    """Cache of compiled Neuron models for multiple bucket sizes.

    Supports lazy compilation: models are compiled on first use at each
    bucket size and cached for subsequent calls.

    Usage:
        cache = CompiledModelCache()

        # Register a compilation function for the pairformer
        def compile_pf(I_padded):
            block = FullPairformerBlock(pf_stack[0])
            S_trace = torch.randn(1, I_padded, C_S, dtype=torch.bfloat16)
            Z_trace = torch.randn(1, I_padded, I_padded, C_Z, dtype=torch.bfloat16)
            return torch_neuronx.trace(block, (S_trace, Z_trace))
        cache.register("pairformer", compile_pf)

        # Get compiled model for a specific bucket
        compiled_pf = cache.get("pairformer", I_padded=256)
    """

    def __init__(self):
        self._compile_fns: Dict[str, callable] = {}
        self._compiled: Dict[str, Dict[int, object]] = {}

    def register(self, name: str, compile_fn: callable):
        """Register a compilation function for a model component.

        Args:
            name: component name (e.g., "pairformer", "difftransformer")
            compile_fn: function(bucket_size) -> compiled_model
        """
        self._compile_fns[name] = compile_fn
        self._compiled[name] = {}

    def get(self, name: str, bucket_size: int) -> object:
        """Get a compiled model for a specific bucket size, compiling if needed.

        Args:
            name: component name
            bucket_size: the bucket size to compile for

        Returns:
            compiled model
        """
        if name not in self._compile_fns:
            raise KeyError(f"No compilation function registered for '{name}'")

        if bucket_size not in self._compiled[name]:
            print(f"  Compiling {name} at bucket_size={bucket_size}...")
            import time

            t0 = time.time()
            self._compiled[name][bucket_size] = self._compile_fns[name](bucket_size)
            t1 = time.time()
            print(f"  Compiled {name} at {bucket_size} in {t1 - t0:.1f}s")

        return self._compiled[name][bucket_size]

    def precompile(self, name: str, bucket_sizes: List[int]):
        """Pre-compile a model at multiple bucket sizes.

        Args:
            name: component name
            bucket_sizes: list of bucket sizes to pre-compile
        """
        for size in bucket_sizes:
            self.get(name, size)

    def is_compiled(self, name: str, bucket_size: int) -> bool:
        """Check if a model is already compiled at a specific bucket size."""
        return bucket_size in self._compiled.get(name, {})

    def compiled_sizes(self, name: str) -> List[int]:
        """List all compiled bucket sizes for a component."""
        return sorted(self._compiled.get(name, {}).keys())


# ============================================================================
# Main entry point for testing
# ============================================================================

if __name__ == "__main__":
    config = BucketConfig()
    config.print_bucket_table()

    # Test bucket selection
    print("\nBucket Selection Tests:")
    test_cases = [
        (58, 240),  # 5hkn: 58 residues -> I=58, L~240
        (240, 1012),  # 5hkn full: I=240 tokens, L=1012 atoms
        (150, 600),  # medium protein
        (380, 1500),  # large protein near limit
        (448, 1800),  # at pairformer limit
    ]

    for I, L in test_cases:
        try:
            I_b = config.select_I_bucket(I)
            L_b = config.select_L_bucket(L)
            mem = config.estimate_memory_mb(I_b, L_b, D=5)
            print(
                f"  I={I:>4} -> I_bucket={I_b:>4}  |  L={L:>5} -> L_bucket={L_b:>5}  |  "
                f"Z_II={mem['Z_II']:.1f}MB  P_LL={mem['P_LL']:.1f}MB  "
                f"total~{mem['total_approximate']:.0f}MB ({mem['utilization_pct']:.1f}% of 24GB)"
            )
        except ValueError as e:
            print(f"  I={I:>4}, L={L:>5}: {e}")

    # Test padding/unpadding round-trip
    print("\nPad/Unpad Round-Trip Tests:")
    I_actual, I_bucket = 240, 256
    L_actual, L_bucket = 1012, 1024

    # Single representation
    S = torch.randn(I_actual, 384)
    S_padded = pad_single(S, I_bucket)
    S_back = unpad_single(S_padded, I_actual)
    assert S_back.shape == S.shape
    assert torch.equal(S_back, S)
    print(f"  S:  {S.shape} -> {S_padded.shape} -> {S_back.shape}  OK")

    # Pair representation
    Z = torch.randn(I_actual, I_actual, 128)
    Z_padded = pad_pair(Z, I_bucket)
    Z_back = unpad_pair(Z_padded, I_actual)
    assert Z_back.shape == Z.shape
    assert torch.equal(Z_back, Z)
    print(f"  Z:  {Z.shape} -> {Z_padded.shape} -> {Z_back.shape}  OK")

    # Batched pair
    Z_b = torch.randn(5, I_actual, I_actual, 128)
    Z_b_padded = pad_pair(Z_b, I_bucket)
    Z_b_back = unpad_pair(Z_b_padded, I_actual)
    assert Z_b_back.shape == Z_b.shape
    assert torch.equal(Z_b_back, Z_b)
    print(f"  Z_b: {Z_b.shape} -> {Z_b_padded.shape} -> {Z_b_back.shape}  OK")

    # Atom single
    Q = torch.randn(5, L_actual, 128)
    Q_padded = pad_atom_single(Q, L_bucket)
    Q_back = unpad_atom_single(Q_padded, L_actual)
    assert Q_back.shape == Q.shape
    assert torch.equal(Q_back, Q)
    print(f"  Q:  {Q.shape} -> {Q_padded.shape} -> {Q_back.shape}  OK")

    # Atom pair
    P = torch.randn(L_actual, L_actual, 16)
    P_padded = pad_atom_pair(P, L_bucket)
    P_back = unpad_atom_pair(P_padded, L_actual)
    assert P_back.shape == P.shape
    assert torch.equal(P_back, P)
    print(f"  P:  {P.shape} -> {P_padded.shape} -> {P_back.shape}  OK")

    # Masks
    pair_mask = make_pair_mask(I_actual, I_bucket)
    single_mask = make_single_mask(I_actual, I_bucket)
    atom_mask = make_atom_mask(L_actual, L_bucket)
    print(
        f"  pair_mask: {pair_mask.shape}, nonzero={pair_mask.sum():.0f}/{pair_mask.numel()}"
    )
    print(
        f"  single_mask: {single_mask.shape}, nonzero={single_mask.sum():.0f}/{single_mask.numel()}"
    )
    print(
        f"  atom_mask: {atom_mask.shape}, nonzero={atom_mask.sum():.0f}/{atom_mask.numel()}"
    )

    print("\nAll tests passed!")
