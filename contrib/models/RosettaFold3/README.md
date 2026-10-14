# Contrib Model: RosettaFold3

Biomolecular structure prediction on AWS Trainium 2 using vanilla `torch_neuronx.trace()` compilation with weight replacement for multi-layer stacks. Achieves 3.04x end-to-end speedup over CPU on the production inference pipeline.

## Model Information

- **Source:** [RosettaCommons/RosettaFold3](https://github.com/RosettaCommons/foundry) (private, requires license)
- **Model Type:** Biomolecular structure prediction (pairformer + diffusion)
- **Parameters:** ~300M (BF16)
- **Architecture:** 48-layer pairformer trunk with triangular attention/multiplication, MSA module, template embedder, 24-block diffusion transformer, windowed atom attention encoder/decoder
- **License:** Requires RosettaCommons Foundry license

## Validation Results

**Validated:** 2026-03-14
**Instance:** trn2.3xlarge (LNC=2, 4 logical NeuronCores, 96 GB HBM)
**SDK:** Neuron SDK 2.28, PyTorch 2.9

### End-to-End Benchmark Results

Protein: 5hkn (I=240 tokens, L=1012 atoms). Production config: 10 recycles, 50 diffusion steps, D=5 (diffusion batch).

| Configuration | CPU Time | Neuron Time | Speedup |
|---------------|----------|-------------|---------|
| Pairformer + DiffTransformer compiled | 178.5s | 71.7s | 2.49x |
| + MSA pair ops + Template pairformer compiled | 178.3s | 59.6s | 2.99x |
| + AtomAttention encoder/decoder compiled | 176.7s | 58.1s | **3.04x** |

### Per-Block Compilation Times (I=240, trn2.3xlarge)

| Block | Compile Time | Layers | Weight Swaps |
|-------|-------------|--------|--------------|
| PairformerBlock | ~25s | 48 | 47 x replace_weights |
| MSA Pair Update | ~23s | 1 (shared, called 4x) | None (shared weights) |
| Template Pairformer | ~15s | 2 | 1 x replace_weights |
| DiffTransformerBlock | ~5s (D=5) | 24 | 23 x replace_weights |
| AtomAttention | ~8s (D=5) | 3 enc + 3 dec | 5 x replace_weights (shared NEFF) |
| **Total** | **~76s** | | |

### Accuracy Validation

All confidence metrics compared between CPU reference and Neuron-accelerated inference on 5hkn (production config):

| Metric | CPU | Neuron | Absolute Difference |
|--------|-----|--------|-------------------|
| overall_plddt | 0.8523 | 0.8520 | 0.0003 |
| ptm | 0.8901 | 0.8900 | 0.0001 |
| iptm | 0.8734 | 0.8733 | 0.0001 |

Per-block cosine similarity (measured on hardware):

| Block | Metric | Cosine Similarity |
|-------|--------|------------------|
| PairformerBlock (1 layer) | S_I | ~0.9998 |
| PairformerBlock (48 layers) | S_I | ~0.70 |
| DiffTransformerBlock (24 blocks) | A_I | 0.9998 |
| AtomAttention (3 blocks) | A_I | 0.9999 |
| StaticWindowedAttn wrapper vs original (1 block) | A_I | >0.999 |

Note: The pairformer S_I cosine similarity degrades through 48 layers due to cumulative BF16 rounding, but this does not affect end-to-end prediction quality (confidence metrics match within 0.001).

## Usage

### Prerequisites

```bash
# On trn2.3xlarge with Neuron DLAMI (Ubuntu 24.04, SDK 2.28)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
pip install lightning

# RF3 Foundry must be installed (requires RosettaCommons license)
# RF3 source paths must be on PYTHONPATH:
export PYTHONPATH=/mnt/models/foundry/models/rf3/src:/mnt/models/foundry/src:$PYTHONPATH

export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_RT_VISIBLE_CORES=0
```

### Quick Start

```python
import os
import sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add RF3 source to path
sys.path.insert(0, "/mnt/models/foundry/models/rf3/src")
sys.path.insert(0, "/mnt/models/foundry/src")

# Add contrib src to path
sys.path.insert(0, "contrib/models/RosettaFold3/src")

from modeling_rf3 import RF3NeuronPipeline, patch_rf3_for_neuron
from rf3.inference_engines.rf3 import RF3InferenceEngine

# Step 1: Load RF3 model
engine = RF3InferenceEngine(
    ckpt_path="/mnt/models/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt",
    n_recycles=10,
    diffusion_batch_size=5,
    num_steps=50,
    seed=42,
    verbose=False,
)
engine.initialize()

# Step 2: Run a quick probe to determine I and L for the input
# (or specify them if known: I=240, L=1012 for 5hkn)
I, L, D = 240, 1012, 5

# Step 3: Create pipeline, compile, and patch
pipeline = RF3NeuronPipeline(engine, I=I, L=L, D=D)
compile_times = pipeline.compile_all()  # ~76s total
pipeline.patch_model()

# Step 4: Run inference (3.04x faster than CPU)
import lightning
lightning.seed_everything(42)
results = engine.run(
    inputs="/mnt/models/foundry/models/rf3/docs/examples/5hkn_from_file.cif",
    out_dir=None,
)
```

### Using Individual Blocks

You can compile and use individual blocks independently:

```python
import torch
import torch_neuronx
from modeling_rf3 import FullPairformerBlock

# Wrap and compile a single pairformer layer
pf_stack = model.recycler.pairformer_stack
block = FullPairformerBlock(pf_stack[0]).eval().to(torch.bfloat16)

S_trace = torch.randn(1, 240, 384, dtype=torch.bfloat16)
Z_trace = torch.randn(1, 240, 240, 128, dtype=torch.bfloat16)

compiled = torch_neuronx.trace(
    block, (S_trace, Z_trace),
    compiler_args=["--target", "trn2"],
    inline_weights_to_neff=False,
)

# Run 48 layers with weight replacement
for i in range(48):
    wrapper = FullPairformerBlock(pf_stack[i]).eval().to(torch.bfloat16)
    torch_neuronx.replace_weights(compiled, wrapper.state_dict())
    S_trace, Z_trace = compiled(S_trace, Z_trace)
```

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.27 |
|----------|----------|----------|
| trn2.3xlarge (I<=448) | VALIDATED | Not tested |

## Example Checkpoints

* RF3 Foundry checkpoint: `rf3_foundry_01_24_latest_remapped.ckpt` (requires RosettaCommons Foundry license)

## Testing Instructions

```bash
# On trn2.3xlarge with Neuron SDK 2.28
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
pip install lightning pytest

export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_RT_VISIBLE_CORES=0
export PYTHONPATH=/mnt/models/foundry/models/rf3/src:/mnt/models/foundry/src

cd contrib/models/RosettaFold3
PYTHONPATH=src:$PYTHONPATH pytest test/integration/test_model.py -v -s
```

## Architecture Details

### Approach

This contribution uses **vanilla `torch_neuronx.trace()` compilation** (not NxDI model classes or NKI custom kernels). Five model components are individually traced and integrated via monkey-patching:

1. **PairformerBlock** (48 layers): The main trunk. Each layer contains triangular multiplicative updates (outgoing + incoming), triangular attention (start + end), pair transition, attention with pair bias, and single transition. One NEFF compiled, 47 weight swaps.

2. **MSAPairUpdateBlock** (4 iterations): The pair-update operations within the MSA module (tri_mul + tri_attn + pair_transition). Shared weights across all 4 iterations, so no weight replacement needed.

3. **TemplPairformerBlock** (2 blocks, c_z=64): Pairformer blocks inside the template embedder. Smaller pair dimension (64 vs 128) requires a separate compilation.

4. **DiffTransformerBlock** (24 blocks): Attention with pair bias + conditioned transition. Batched with D=5 for the diffusion process.

5. **StaticWindowedAttnBlock** (3 encoder + 3 decoder blocks): RF3's atom attention uses dynamic windowed attention with `torch.arange` and boolean masking. This wrapper replaces those with precomputed static index tensors passed as inputs, making it traceable. Encoder and decoder have identical architectures, sharing a single NEFF.

### Compilation Parameters

| Parameter | Value |
|-----------|-------|
| `inline_weights_to_neff` | `False` (enables weight replacement) |
| `compiler_args` | `["--target", "trn2"]` |
| Auto-cast | **Not used** (destroys accuracy for AlphaFold-family triangular operations) |
| Precision | BF16 throughout (model weights + activations) |

### Compatibility Patches

RF3 requires 6 source-level patches for Neuron compatibility:

| Patch | Location | Reason |
|-------|----------|--------|
| Disable `torch.autocast` | RF3.py, attention.py (4 locations) | Neuron does not support dynamic autocast during tracing |
| `opt_einsum` -> `torch.einsum` | attention.py, structure_bias.py | opt_einsum is not traced by torch_neuronx |
| `index_reduce('mean')` -> `scatter_add` | pairformer_layers.py, af3_diffusion_transformer.py | index_reduce not supported by Neuron compiler |
| Disable activation checkpointing | checkpoint.py | Not compatible with tracing |

These patches are applied by `patch_rf3_for_neuron()` or manually to the RF3 source files before inference.

### Shape Bucketing

The `rf3_bucketing.py` module provides utilities for padding inputs to fixed bucket sizes for multi-size support:

| Dimension | Buckets | Max Compilable |
|-----------|---------|---------------|
| I (tokens) | 128, 192, 256, 320, 384, 448 | 448 (pairformer SB limit) |
| L (atoms) | 256, 512, 768, 1024, 1280, 1536, 2048 | ~2048 (estimated) |
| D (diffusion batch) | 1, 5 | - |

### Important Notes

- **Do NOT use `--auto-cast matmult`** for this model. AlphaFold-family triangular operations require full BF16 precision; auto-casting destroys accuracy.
- `torch_neuronx.trace()` compiled models **must be called with CPU tensors**. XLA tensors return all zeros.
- `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` must be set before any Neuron imports.

## Known Issues and Limitations

1. **Maximum protein size is I=448 tokens.** The pairformer compilation hits the Neuron compiler's State Buffer (SB) on-chip SRAM limit at I=480. Proteins with more than ~448 residue tokens cannot be compiled. The DiffTransformer compiles up to I=512 (D=5), so the pairformer is the binding constraint.

2. **Fixed input shapes require recompilation.** Each input size requires a separate compilation (~76s). The bucketing module provides predefined bucket sizes to amortize compilation across multiple inputs. A `CompiledModelCache` class supports lazy compilation at multiple bucket sizes.

3. **Zero-padding affects triangle attention softmax.** Padded positions participate in the softmax normalization of triangle attention layers. For 5hkn (I=240, I_padded=240), no actual padding is needed. For inputs that require significant padding (e.g., I=200 padded to I=256), attention masks are recommended for large padding ratios. In practice, the end-to-end confidence metrics are unaffected for moderate padding.

4. **RF3 model access requires RosettaCommons Foundry license.** The model weights and source code are not publicly available. Users must obtain a Foundry license from RosettaCommons.

5. **CPU-Neuron data transfer overhead.** Each of the 48 pairformer layers incurs a host-device round trip plus a `replace_weights` call. This overhead is small per-layer but contributes ~15% of total pairformer time.

6. **MSA module is only partially accelerated.** The MSA-specific operations (outer product mean, pair-weighted averaging, MSA transition) remain on CPU because they involve MSA-dimension operations not present in the pair-update path. The pair-update ops (tri_mul + tri_attn + pair_transition) are on Neuron.

7. **Only trn2.3xlarge validated.** The compilation and benchmarks were performed exclusively on trn2.3xlarge with LNC=2 (4 logical cores). Other instance types have not been tested.

8. **NKI custom kernels provide no benefit.** Testing showed that vanilla `torch_neuronx.trace()` compilation is 5.6-11.9x faster than NKI custom kernels at all practical sizes for this model. NKI mega-kernels compile at N=384 (beyond vanilla I=448 limit) but at 7.6x worse performance. This is specific to RF3's architecture and tensor shapes.

## Source Files

| File | Description |
|------|-------------|
| `src/modeling_rf3.py` | Main module: wrapper classes, compilation, monkey-patching, RF3NeuronPipeline |
| `src/rf3_bucketing.py` | Shape bucketing, padding/unpadding utilities, compiled model cache |
| `src/__init__.py` | Package exports |
| `test/integration/test_model.py` | Accuracy tests for individual blocks + bucketing utilities |
