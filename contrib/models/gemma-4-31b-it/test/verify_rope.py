# coding=utf-8
"""On-device verification of ProportionalRotaryEmbedding vs HuggingFace reference.

Purpose: verify the Task 024 fix on trn2 hardware. Compares the traced
`ProportionalRotaryEmbedding` output (running on Neuron device) against an
fp64 CPU reference computed from scratch (see `_hf_reference_cos_sin`).

Run on a trn2.3xlarge instance with SDK 2.28/2.29/2.31 DLAMI:

    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate  # SDK 2.28
    # OR
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate    # SDK 2.29+
    python3 verify_rope.py

Expected output:
    - fp32 max_abs_diff < 5e-3 (fp32 arg-reduction noise floor at pos=32k)
    - bf16 max_abs_diff < 5e-2 (bf16 noise floor, HF spec says 'bf16 precision')
    - PASS on all tested positions
"""

import os
import sys

import torch

# Enable torch_neuronx trace if available. If not, fall back to CPU-only comparison
# for local testing.
try:
    import torch_neuronx  # noqa: F401
    from neuronx_distributed_inference.modules.attention.utils import (
        RotaryEmbedding,
    )

    _HAS_NEURON = True
except ImportError:
    print("WARNING: torch_neuronx or NxDI not importable. Running CPU-only comparison.")
    _HAS_NEURON = False

    from torch import nn

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, max_position_embeddings=2048, base=10000, factor=None):
            super().__init__()
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            self.register_buffer("inv_freq", None, persistent=False)
            self.factor = factor

        def get_inv_freqs(self, device=None):
            freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
            inv_freq = 1.0 / (self.base ** (freq_indices / self.dim))
            if self.factor is not None:
                inv_freq = inv_freq / self.factor
            return inv_freq

        @torch.no_grad()
        def forward(self, x, position_ids):
            if self.inv_freq is None:
                self.inv_freq = self.get_inv_freqs(x.device)
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            )
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---- Import the fix. Assumes verify_rope.py is placed alongside modeling_gemma4.py. ----
# Try both layouts: (1) test/ next to src/, (2) test/verify_rope.py in same dir.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

_IMPORT_SOURCE = "unknown"
try:
    from modeling_gemma4 import ProportionalRotaryEmbedding
    _IMPORT_SOURCE = f"modeling_gemma4 ({ProportionalRotaryEmbedding.__module__})"
except ImportError as e:
    # Fallback: define ProportionalRotaryEmbedding locally so the script is also
    # usable standalone (e.g. quick math check without full NxDI env).
    print(f"WARNING: could not import from modeling_gemma4.py ({e})")
    print("         falling back to inline ProportionalRotaryEmbedding definition")

    class ProportionalRotaryEmbedding(RotaryEmbedding):
        def __init__(self, dim, rope_angles, max_position_embeddings=2048, base=10000):
            super().__init__(
                dim=dim, max_position_embeddings=max_position_embeddings, base=base
            )
            self.rope_angles = rope_angles

        def get_inv_freqs(self, device=None):
            idx = torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float, device=device)
            rot = 1.0 / (self.base ** (idx / self.dim))
            nope = self.dim // 2 - self.rope_angles
            if nope > 0:
                return torch.cat([rot, torch.zeros(nope, dtype=torch.float, device=device)])
            return rot

    _IMPORT_SOURCE = "inline fallback"

print(f"ProportionalRotaryEmbedding source: {_IMPORT_SOURCE}")


# ---- fp64 CPU reference (direct port of transformers._compute_proportional_rope_parameters) ----


def hf_reference_cos_sin(head_dim, partial_rotary_factor, base, positions, factor=1.0):
    """From-scratch fp64 cos/sin for a batch of positions."""
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    idx = torch.arange(0, 2 * rope_angles, 2, dtype=torch.float64)
    inv_freq_rotated = 1.0 / (base ** (idx / head_dim))
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat(
            [inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float64)]
        )
    else:
        inv_freq = inv_freq_rotated
    if factor != 1.0:
        inv_freq = inv_freq / factor

    pos = torch.tensor(positions, dtype=torch.float64).unsqueeze(0)  # [1, S]
    freqs = torch.einsum("bs,d->bsd", pos, inv_freq)  # [1, S, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [1, S, head_dim]
    return emb.cos(), emb.sin()


# ---- Neuron trace helper ----


class RopeTraceable(torch.nn.Module):
    """Wraps ProportionalRotaryEmbedding for torch_neuronx.trace()."""

    def __init__(self, head_dim, rope_angles, max_pos, base):
        super().__init__()
        self.rope = ProportionalRotaryEmbedding(
            dim=head_dim,
            rope_angles=rope_angles,
            max_position_embeddings=max_pos,
            base=base,
        )

    def forward(self, x, position_ids):
        cos, sin = self.rope(x, position_ids)
        return cos, sin


# ---- Test cases ----


def run_check(name, head_dim, partial, base, positions, dtype=torch.float32, use_neuron=False):
    print(f"\n=== {name} ===")
    print(
        f"head_dim={head_dim}, partial_rotary_factor={partial}, base={base}, dtype={dtype}, "
        f"positions={positions}, backend={'Neuron' if use_neuron else 'CPU-fp32'}"
    )

    rope_angles = int(partial * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles
    print(f"  rope_angles={rope_angles}, nope_angles={nope_angles}")

    x = torch.zeros(1, 1, len(positions), head_dim, dtype=dtype)
    pos_ids = torch.tensor([positions], dtype=torch.int64)

    if use_neuron and _HAS_NEURON:
        model = RopeTraceable(head_dim, rope_angles, 32768, base).eval()
        neuron_model = torch_neuronx.trace(model, (x, pos_ids))
        ours_cos, ours_sin = neuron_model(x, pos_ids)
    else:
        rope = ProportionalRotaryEmbedding(
            dim=head_dim, rope_angles=rope_angles, max_position_embeddings=32768, base=base
        )
        with torch.no_grad():
            ours_cos, ours_sin = rope(x, pos_ids)

    ref_cos, ref_sin = hf_reference_cos_sin(head_dim, partial, base, positions)  # fp64

    ours_cos_fp64 = ours_cos.double()
    ours_sin_fp64 = ours_sin.double()
    cos_diff = (ours_cos_fp64 - ref_cos).abs()
    sin_diff = (ours_sin_fp64 - ref_sin).abs()

    print(f"  cos: max_abs = {cos_diff.max().item():.3e}, mean_abs = {cos_diff.mean().item():.3e}")
    print(f"  sin: max_abs = {sin_diff.max().item():.3e}, mean_abs = {sin_diff.mean().item():.3e}")

    if dtype == torch.float32:
        threshold = 5e-3  # fp32 arg-reduction noise floor at large positions
    elif dtype == torch.bfloat16:
        threshold = 5e-2  # bf16 noise floor (HF spec calls this "bf16 precision")
    elif dtype == torch.float16:
        threshold = 1e-2
    else:
        threshold = 1e-6

    passed = cos_diff.max().item() < threshold and sin_diff.max().item() < threshold
    status = "PASS" if passed else "FAIL"
    print(f"  threshold={threshold:.0e}, status={status}")

    if nope_angles > 0:
        # Spot-check: nope dims should be pass-through (cos=1, sin=0)
        nope_cos = ours_cos_fp64[0, :, rope_angles : head_dim // 2]
        nope_sin = ours_sin_fp64[0, :, rope_angles : head_dim // 2]
        print(f"  nope-dim cos: mean = {nope_cos.mean().item():.6f} (should be 1.0)")
        print(f"  nope-dim sin: mean_abs = {nope_sin.abs().mean().item():.2e} (should be 0.0)")

    return passed


def main():
    print("=" * 70)
    print("Task 024: on-device RoPE verification (PR #106 Bug 1 fix)")
    print("=" * 70)

    if _HAS_NEURON:
        print(f"Neuron backend: available (torch_neuronx imported successfully)")
    else:
        print("Neuron backend: NOT available -- running CPU-only comparison")

    positions = [0, 1, 128, 500, 1024, 2048, 2400, 2837, 8192, 16384, 32000]

    results = []

    # SWA layer (should degenerate to standard RotaryEmbedding)
    results.append(
        run_check(
            "SWA layer (head_dim=256, partial=1.0, base=10000)",
            head_dim=256,
            partial=1.0,
            base=10000.0,
            positions=positions,
            dtype=torch.float32,
            use_neuron=_HAS_NEURON,
        )
    )

    # Global layer -- fp32
    results.append(
        run_check(
            "Global layer (head_dim=512, partial=0.25, base=1e6) [fp32]",
            head_dim=512,
            partial=0.25,
            base=1e6,
            positions=positions,
            dtype=torch.float32,
            use_neuron=_HAS_NEURON,
        )
    )

    # Global layer -- bf16 (Gemma4 native dtype)
    results.append(
        run_check(
            "Global layer (head_dim=512, partial=0.25, base=1e6) [bf16]",
            head_dim=512,
            partial=0.25,
            base=1e6,
            positions=positions,
            dtype=torch.bfloat16,
            use_neuron=_HAS_NEURON,
        )
    )

    print("\n" + "=" * 70)
    if all(results):
        print(f"ALL {len(results)}/{len(results)} CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"FAILED {results.count(False)}/{len(results)} checks")
        sys.exit(1)


if __name__ == "__main__":
    main()
