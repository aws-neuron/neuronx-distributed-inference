# Contrib Model: Boltz-2

Biomolecular structure prediction on AWS Trainium 2 using NKI custom kernels for the pairformer trunk.

## Model Information

- **HuggingFace ID:** N/A (PyPI: `boltz==2.2.1`, GitHub: [jwohlwend/boltz](https://github.com/jwohlwend/boltz))
- **Model Type:** Biomolecular structure prediction (pairformer + diffusion)
- **Parameters:** 507M (BF16)
- **Architecture:** 64-layer pairformer trunk with triangular attention, triangular multiplicative updates, and pair bias attention; diffusion score model for coordinate generation
- **License:** MIT

## Validation Results

**Validated:** 2026-02-27
**Instance:** trn2.3xlarge (LNC=2, 4 logical NeuronCores, 96 GB HBM)
**SDK:** Neuron SDK 2.28, PyTorch 2.9

### Pairformer Accuracy (Weight Replacement + NKI Kernels)

| N | Layers | s_cos | z_cos | Status |
|---|--------|-------|-------|--------|
| 128 | 1 | 0.999796 | 0.998359 | PASS |
| 128 | 2 | 0.999667 | 0.997491 | PASS |
| 128 | 4 | 0.999757 | 0.996898 | PASS |
| 128 | 8 | 0.999713 | 0.995417 | PASS |
| 256 | 1 | 0.999847 | 0.999971 | PASS |
| 256 | 64 | 0.999220 | 0.943929 | PASS |
| 512 | 64 | 0.999460 | 0.979214 | PASS |

### Standalone NKI Kernel Accuracy

| Kernel | N | Cosine Similarity | Status |
|--------|---|-------------------|--------|
| Triangular Attention | 128 | 0.999713 | PASS |
| Triangular Attention | 256 | 1.000029 | PASS |
| Triangular Mul (Outgoing) | 128 | 0.999967 | PASS |
| Triangular Mul (Outgoing) | 256 | 0.999903 | PASS |
| Triangular Mul (Incoming) | 128 | 0.999967 | PASS |

### Benchmark Results

**Fused NKI Mega-Kernel (SPMD grid=[2], single NKI kernel per layer):**

The fused mega-kernel combines ALL 7 sub-operations of a PairformerLayer into a single NKI kernel call, eliminating host-device round trips and sync overhead between operations.

| Approach | N=256 Per Layer | 64 Layers (est.) | Speedup vs Traced |
|---|---|---|---|
| Traced + weight replacement (original) | 173 ms | 11.08s | 1.0x |
| **Fused mega-kernel (SPMD)** | **65.9 ms** | **~4.2s** | **2.63x** |

Mega-kernel correctness at N=256: s_cos=0.999995, z_cos=0.999245 — PASS.

**Pairformer compilation (traced approach, trn2.3xlarge):**

| N | Compile Layer 0 | Weight Swaps (63 layers) | Total Setup |
|---|-----------------|-------------------------|-------------|
| 256 | 423s (7.1 min) | 21s (0.3s each) | 443s (7.4 min) |

**Pairformer inference (trn2.3xlarge, warm, 64 layers):**

| N | Total Latency | Per Layer |
|---|---------------|-----------|
| 256 | 11.082s | 173ms |
| 512 | 105.920s | 1655ms |

**Full pipeline (N=256, insulin B chain, 30 tokens, 20 diffusion steps):**

| Phase | Time |
|-------|------|
| Pairformer compilation | 7.6 min |
| Diffusion compilation | 2.4 min |
| **Total compilation** | **10.1 min** |
| Trunk inference (embed + MSA + PF) | 22.5s |
| Diffusion (20 steps) | 1.2s |
| Confidence | 0.8s |
| **Total inference** | **24.5s** |

## Usage

### Prerequisites

```bash
# Activate Neuron venv on trn2.3xlarge DLAMI (Ubuntu 24.04, SDK 2.28)
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# Install Boltz-2
pip install boltz==2.2.1

# Download checkpoint (auto-downloads on first boltz predict run)
boltz predict --help
```

### Pairformer-Only Inference

```python
import os
import sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

# Add src/ to path
sys.path.insert(0, "contrib/models/Boltz-2/src")

from modeling_boltz2 import (
    patch_boltz2_with_nki_kernels,
    compile_pairformer_weight_replaced,
    run_pairformer_layers,
)

# Step 1: Patch BEFORE importing Boltz-2 model
patch_boltz2_with_nki_kernels()

# Step 2: Load model
from dataclasses import asdict
from boltz.main import (
    Boltz2, Boltz2DiffusionParams, BoltzSteeringParams,
    MSAModuleArgs, PairformerArgsV2,
)

model = Boltz2.load_from_checkpoint(
    "~/.boltz/boltz2_conf.ckpt",
    strict=True,
    predict_args={"recycling_steps": 1, "sampling_steps": 20,
                  "diffusion_samples": 1, "max_parallel_samples": 1,
                  "write_confidence_summary": False,
                  "write_full_pae": False, "write_full_pde": False},
    map_location="cpu",
    diffusion_process_args=asdict(Boltz2DiffusionParams()),
    ema=False, use_kernels=False,
    pairformer_args=asdict(PairformerArgsV2()),
    msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
    steering_args=asdict(BoltzSteeringParams()),
)
model.eval()

# Step 3: Compile pairformer (7.4 min at N=256)
traced_layers, compile_time, swap_time = compile_pairformer_weight_replaced(
    model, N=256, target="trn2"
)

# Step 4: Run inference
import torch
s = torch.randn(1, 256, 384, dtype=torch.bfloat16) * 0.1
z = torch.randn(1, 256, 256, 128, dtype=torch.bfloat16) * 0.1
mask = torch.ones(1, 256, dtype=torch.float32)
pair_mask = torch.ones(1, 256, 256, dtype=torch.float32)

s_out, z_out, latency = run_pairformer_layers(traced_layers, s, z, mask, pair_mask)
print(f"Inference: {latency:.1f}s ({latency/64*1000:.0f}ms/layer)")
```

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.27 |
|----------|----------|----------|
| trn2.3xlarge (N=256) | VALIDATED | VALIDATED |
| trn2.48xlarge (N=512) | VALIDATED | Not tested |
| inf2.8xlarge (N=128, <=8 layers) | Not tested | VALIDATED |

## Example Checkpoints

* [boltz==2.2.1](https://pypi.org/project/boltz/2.2.1/) (PyPI) - auto-downloads to `~/.boltz/boltz2_conf.ckpt`

## Testing Instructions

```bash
# On trn2.3xlarge with Neuron SDK 2.28
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install boltz==2.2.1 pytest

export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_RT_VISIBLE_CORES=0

# Run tests (compiles 2 layers at N=128, ~2 min)
cd contrib/models/Boltz-2
PYTHONPATH=src pytest test/integration/test_model.py -v -s
```

## Architecture Details

### Approach

This contribution includes two approaches for running the Boltz-2 pairformer on Trainium 2:

**Approach 1: Traced + NKI Kernels (original)**

1. **torch_neuronx.trace()** (not NxDI model classes) for compilation
2. **NKI custom kernels** for the four O(N^3) triangular operations
3. **Weight replacement** pattern: compile one layer, clone 63 times with `replace_weights()`
4. **Monkey-patching** to inject NKI kernels into the upstream Boltz-2 codebase

**Approach 2: Fused Mega-Kernel (2.63x faster)**

A single NKI kernel (`full_pairformer_layer_spmd.py`) covers ALL 7 sub-operations of a PairformerLayer, including PairBiasAttention, both TriMul ops, both TriAttn ops, and both Transitions. SPMD grid=[2] uses both physical NeuronCores. This eliminates all host-device round trips within a layer, reducing latency from 173ms to 65.9ms at N=256.

- Requires N >= 256 (SPMD split needs at least 2 s-tiles)
- Compile time: ~5 min at N=256
- NEFF size: 24.2 MB at N=256

### NKI Kernels

Two NKI kernels replace the computationally expensive operations:

| Kernel | Operation | Complexity | Description |
|--------|-----------|-----------|-------------|
| `nki_triangular_attention.py` | Triangular Attention | O(N^3) | Multi-head attention per row with full 2D triangle bias, online softmax |
| `nki_triangular_mul.py` | Triangular Multiplicative Update | O(N^3) | Einsum contraction: `result[i,j,d] = sum_k a[i,k,d] * b[j,k,d]` |

Each kernel is called twice per pairformer layer (starting node + ending node variants), totaling 4 NKI kernel calls per layer.

### Compilation

| Parameter | Value |
|-----------|-------|
| `inline_weights_to_neff` | `False` (enables weight replacement) |
| `compiler_args` | `["--target", "trn2"]` |
| Compile time (N=256) | 423s for layer 0 |
| Weight swap time | 0.3s per layer (63 layers) |
| Total setup | 443s (7.4 min) |

### Model Constants

| Parameter | Value |
|-----------|-------|
| C_s (single repr) | 384 |
| C_z (pair repr) | 128 |
| H (attention heads) | 4 |
| d (head dim) | 32 |
| Pairformer layers | 64 |
| Total parameters | 507M |

### Important Notes

- **Do NOT use `--auto-cast matmult`** for this model. It destroys accuracy for Boltz-2's triangular operations.
- `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` must be set before any Neuron imports.
- `NEURON_RT_VISIBLE_CORES=0` restricts to a single NeuronCore (sufficient for single-structure inference).
- N must be a multiple of 128 (P_MAX tiling constraint of NKI kernels).

## Known Issues

1. **N=512 requires trn2.48xlarge**: The 64-layer pairformer at N=512 exceeds trn2.3xlarge memory during compilation. Use trn2.48xlarge instead.
2. **Per-layer host-device round trips**: Each of the 64 pairformer layers is a separate traced model, incurring host-device transfer overhead per layer (~173ms/layer at N=256, of which a significant fraction is sync overhead rather than compute).
3. **DMA descriptor limit on inf2**: N=128 with more than 8 layers hits the inf2 DMA descriptor limit. Use trn2 for full-model inference.
4. **Cold start**: First inference after compilation is slower due to device initialization.

## Source Files

| File | Description |
|------|-------------|
| `src/modeling_boltz2.py` | Main module: monkey-patching, compilation, inference (traced approach) |
| `src/nki_triangular_attention.py` | NKI kernel: triangular attention with online softmax |
| `src/nki_triangular_mul.py` | NKI kernel: triangular multiplicative update (einsum) |
| `src/fused_z_ops_spmd.py` | Fused mega-kernel: z-only operations (TriMul, TriAttn, Transition_z) |
| `src/full_pairformer_layer_spmd.py` | Fused mega-kernel: full PairformerLayer (all 7 ops, SPMD grid=[2]) |
| `src/__init__.py` | Package exports |
| `test/integration/test_model.py` | Accuracy + latency tests (traced approach) |
| `test/integration/compile_full_layer_spmd.py` | Compile script for fused mega-kernel |
| `test/integration/test_full_layer_spmd.py` | Correctness test for fused mega-kernel vs CPU reference |
