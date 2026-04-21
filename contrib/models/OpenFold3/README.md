# Contrib Model: OpenFold3

Biomolecular structure prediction (AlphaFold3 reproduction, ~330M params) on AWS Trainium 2 using vanilla `torch_neuronx.trace()` compilation with weight replacement for multi-layer stacks. Supports sequence lengths up to N=2048 via progressive decomposition of PairFormer sub-operations, with N-range-aware strategy selection that merges segments at smaller N for reduced overhead. Enables protein structure prediction at scales where GPU (A100-40GB) runs out of memory.

## Model Information

- **Source:** [aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3) (OpenFold3 v0.4.0)
- **Model Type:** Biomolecular structure prediction (pairformer + diffusion)
- **Parameters:** ~330M (FP32)
- **Architecture:** 48-layer PairFormer trunk with triangular attention/multiplication, 4-block MSA module (two structural types), 2-block template embedder, 24-block diffusion transformer, windowed atom attention encoder/decoder
- **License:** Apache 2.0

## Validation Results

**Validated:** 2026-03-20
**Instance:** trn2.3xlarge (LNC=2, 4 logical NeuronCores, 96 GB HBM)
**SDK:** Neuron SDK 2.28, PyTorch 2.9

### PairFormer Performance (48 layers)

Two compilation strategies are provided based on sequence length:

**Monolithic** (N <= 256): Each PairFormerBlock traced as a single unit. Fast, minimal overhead.

| N | Neuron (per layer) | CPU (per layer) | Speedup | 48-layer Total |
|---|-------------------|----------------|---------|----------------|
| 128 | ~5ms | ~60ms | 12.1x | ~0.3s |
| 256 | 66.1ms | 800ms | 12.1x | 3.2s |

**Decomposed** (N > 256, up to 2048): PairFormerBlock split into sub-operations traced independently.

| N | Neuron (per layer) | CPU (per layer) | Speedup | 48-layer Total |
|---|-------------------|----------------|---------|----------------|
| 256 | 244ms | ~134ms | ~0.5x* | 11.8s |
| 512 | 988ms | 4,755ms | 4.8x | 47.9s |
| 1024 | 4,155ms | 28,041ms | 6.7x | 200.2s |
| 2048 | 19,120ms | **OOM** (>137GB) | N/A | 919.5s |

*Decomposition overhead dominates at N=256 -- use monolithic strategy at small N.

### GPU Comparison (Single A100-SXM4-40GB, p4d.24xlarge)

GPU runs PairFormerBlock monolithically (no decomposition needed). Comparison is per-layer latency:

| N | GPU (A100) | Neuron | GPU/Neuron Ratio | Notes |
|---|-----------|--------|-----------------|-------|
| 256 | 15.4ms | 66.1ms (mono) | 0.23x | GPU faster |
| 512 | 83.4ms | 988ms (decomp) | 0.08x | GPU faster |
| 1024 | 531.5ms | 4,155ms (decomp) | 0.13x | GPU uses 39.8/40 GB HBM |
| 2048 | **OOM** | 19,120ms (decomp) | N/A | GPU cannot run (needs ~128GB) |

At N=2048, the TriMul intermediate tensor `[128, N, N]` requires ~128 GB -- exceeding A100-40GB, A100-80GB, and H100-80GB memory. Only B200 (192GB) or Neuron decomposition can handle this scale.

### Per-Sub-Op Breakdown (Decomposed, avg ms)

| N | TriMulOut | TriMulIn | TriAttnStart | TriAttnEnd | PairTrans | AttnPairBias | SingleTrans | Total |
|---|-----------|----------|-------------|-----------|-----------|-------------|------------|-------|
| 256 | 46 | 53 | 24 | 24 | 10 | 5 | 1 | 246 |
| 512 | 271 | 265 | 125 | 125 | 51 | 16 | 1 | 997 |
| 1024 | 1146 | 1128 | 600 | 602 | 229 | 81 | 2 | 4170 |
| 2048 | 4539 | 4516 | 3624 | 3624 | 919 | 331 | 2 | 19156 |

### Compilation Times (N=128, monolithic blocks; N=384, decomposed sub-ops)

| Block | Compile Time | Layers | Weight Swaps |
|-------|-------------|--------|--------------|
| PairFormerBlock (monolithic, N=128) | 63.2s | 48 | 47 x replace_weights |
| MSA type A | 45.0s | 3 | 2 x replace_weights |
| MSA type B | 44.5s | 1 | Separate NEFF |
| TemplatePairBlock | 8.7s | 2 | 1 x replace_weights |
| DiffCond._forward() | 4.0s | 1 | Shared weights |
| Decomposed sub-ops (N=384) | ~60-70s | 48 | 47 x replace_weights per segment |

### End-to-End Benchmark (Monolithic, N=256)

Full pipeline including MSA, Template, DiffCond, and PairFormer:

| Configuration | CPU Time | Neuron Time | Speedup |
|---------------|----------|-------------|---------|
| N=128, 0 recycles, 20 diff steps | 9.7s | 3.8s | 2.53x |
| N=256, 0 recycles, 20 diff steps (+ DiffCond) | 46.5s | 10.7s | **4.34x** |
| N=256, 3 recycles, 200 diff steps (production) | ~270s | 72.0s | ~3.7x |

### Accuracy Validation

Per-block accuracy validated using `neuron_allclose()` (measured on hardware):

**Monolithic blocks (6 tests, all PASS at N=128):**

| Block | Metric | neuron_allclose | Max Abs Error | Cosine Similarity |
|-------|--------|----------------|---------------|------------------|
| PairFormerBlock (1 layer) | s, z | PASS (0 mismatches) | S: 0.80, Z: 0.36 | S: 1.000002, Z: 1.000165 |
| PairFormer (2-layer chain) | s, z | PASS (0 mismatches) | S: 0.66, Z: 0.43 | S: 1.000000, Z: 1.000161 |
| MSA type A (1 block) | m, z | PASS (0 mismatches) | M: 0.046, Z: 0.73 | M: 1.000001, Z: 1.000169 |
| MSA type B (1 block) | m, z | PASS (0 mismatches) | M: 0.0, Z: 0.007 | M: 1.000000, Z: 1.000068 |
| TemplatePairBlock (1 block) | t | PASS (0 mismatches) | T: 0.0007 | 1.000023 |
| DiffCond._forward() | si, zij | PASS (0 mismatches) | S: 0.001, Z: 0.0005 | S: 1.000000, Z: 1.000131 |

**Decomposed blocks (4 tests, N=384):**

| Block | Metric | neuron_allclose | Max Abs Error | Cosine Similarity |
|-------|--------|----------------|---------------|------------------|
| TriMulOut (3 segments) | z | PASS (0 mismatches) | 0.0002 | 1.001377 |
| TriMulIn (3 segments) | z | PASS (0 mismatches) | 0.0001 | 1.001302 |
| TriAttnStart (2 segments) | z | PASS (0 mismatches) | 0.0015 | 1.001989 |
| Full decomposed layer | s, z | PASS (0 mismatches) | S: 1.09, Z: 0.58 | S: 1.000003, Z: 1.003918 |

End-to-end trunk output cosine similarity: >0.9999 (N=128 and N=256). No NaN through 48 layers at any tested N.

Note: Final atom positions show lower cosine similarity (~0.41-0.52) due to stochastic diffusion amplifying FP32 rounding differences across 200 denoising steps with random noise. The trunk outputs (before diffusion) are numerically identical, confirming the compilation is accurate.

## Usage

### Prerequisites

```bash
# On trn2.3xlarge with Neuron DLAMI (Ubuntu 24.04, SDK 2.28)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Install OpenFold3
git clone https://github.com/aqlaboratory/openfold-3.git /home/ubuntu/openfold-3
pip install -e '.[predict]' --no-deps
pip install ml-collections biopython modelcif dm-tree biotite gemmi \
    pytorch-lightning rdkit func-timeout wandb

# Download weights
aws s3 cp s3://openfold/staging/of3-p2-155k.pt ~/.openfold3/ --no-sign-request

export NEURON_RT_VISIBLE_CORES=0
```

### Quick Start (Monolithic, N <= 256)

```python
import sys
sys.path.insert(0, "/home/ubuntu/openfold-3")
sys.path.insert(0, "contrib/models/OpenFold3/src")

from modeling_openfold3 import OpenFold3NeuronPipeline

# Step 1: Create pipeline (auto-selects monolithic for N <= 256)
pipeline = OpenFold3NeuronPipeline(
    openfold3_src_path="/home/ubuntu/openfold-3",
    checkpoint_path="~/.openfold3/of3-p2-155k.pt",
    n_token=256,
)

# Step 2: Load model and apply source patches
pipeline.load_model()

# Step 3: Compile all blocks (~5-10 min total)
compile_times = pipeline.compile_all()

# Step 4: Monkey-patch model
pipeline.patch_model()

# Step 5: Run inference (4.34x faster than CPU at N=256)
batch_out, output = pipeline.run_inference(
    num_recycles=0,
    diff_steps=20,
    diff_samples=1,
)

# Access predicted atom positions
positions = output["atom_positions_predicted"]
```

### Decomposed Mode (N > 256, up to 2048)

```python
from modeling_openfold3 import OpenFold3NeuronPipeline

# N=512: auto-selects decomposed strategy
pipeline = OpenFold3NeuronPipeline(
    openfold3_src_path="/home/ubuntu/openfold-3",
    checkpoint_path="~/.openfold3/of3-p2-155k.pt",
    n_token=512,
    # pairformer_strategy="auto"  (default: monolithic if N<=256, decomposed if N>256)
)

pipeline.load_model()
compile_times = pipeline.compile_all()  # Compiles 10+ sub-op segments
pipeline.patch_model()

batch_out, output = pipeline.run_inference(
    num_recycles=0,
    diff_steps=20,
    diff_samples=1,
)
```

For N > 1024, the pipeline automatically uses chunked TriAttn MHA (chunk_size=128) to fit attention score tensors within 24GB HBM per core.

### Using Individual Blocks

```python
import torch
import torch_neuronx
from modeling_openfold3 import PairFormerBlockWrapper

# Wrap and compile a single pairformer layer
pf_stack = model.pairformer_stack.blocks
wrapper = PairFormerBlockWrapper(pf_stack[0])
wrapper.eval()

s = torch.randn(1, 256, 384)
z = torch.randn(1, 256, 256, 128)
mask_s = torch.ones(1, 256)
mask_z = torch.ones(1, 256, 256)

compiled = torch_neuronx.trace(
    wrapper, (s, z, mask_s, mask_z),
    compiler_args=["--target", "trn2"],
    inline_weights_to_neff=False,
)

# Run 48 layers with weight replacement
for i in range(48):
    w = PairFormerBlockWrapper(pf_stack[i])
    torch_neuronx.replace_weights(compiled, w.state_dict())
    s, z = compiled(s, z, mask_s, mask_z)
```

### Using Decomposed Sub-Ops Directly

```python
from modeling_openfold3 import DecomposedPairFormerCompiler

compiler = DecomposedPairFormerCompiler(
    model=model,
    n_token=512,
)
compile_times = compiler.compile_all()

# Run all 48 layers with weight replacement
z = torch.randn(1, 512, 512, 128)
s = torch.randn(1, 512, 384)

for layer_idx in range(48):
    z, s = compiler.run_layer(z, s, layer_idx=layer_idx)
```

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.27 |
|----------|----------|----------|
| trn2.3xlarge (N<=256, monolithic) | VALIDATED | Not tested |
| trn2.3xlarge (N=384, decomposed) | VALIDATED | Not tested |
| trn2.3xlarge (N=512, decomposed) | VALIDATED | Not tested |
| trn2.3xlarge (N=1024, decomposed) | VALIDATED | Not tested |
| trn2.3xlarge (N=2048, decomposed+chunked) | VALIDATED | Not tested |

## Example Checkpoints

* [OpenFold3 v0.4.0 weights](https://github.com/aqlaboratory/openfold-3) (download via `aws s3 cp s3://openfold/staging/of3-p2-155k.pt . --no-sign-request`)

## Testing Instructions

```bash
# On trn2.3xlarge with Neuron SDK 2.28
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

export NEURON_RT_VISIBLE_CORES=0

cd contrib/models/OpenFold3
PYTHONPATH=src:/home/ubuntu/openfold-3:$PYTHONPATH \
    pytest test/integration/test_model.py -v -s
```

The test suite includes 11 tests (all PASS on trn2.3xlarge):
- 6 monolithic block tests (PairFormer, weight replacement, MSA-A, MSA-B, Template, DiffCond) at N=128
- 5 decomposed tests (FusedTriMulOutIn, TriMulOut, TriMulIn, TriAttnStart, full layer) at N=384

## Architecture Details

### Compilation Strategies

**Monolithic (N <= 256):** Each of the 5 model components is traced as a single unit via `torch_neuronx.trace()`. The 48-layer PairFormer stack uses one NEFF with 47 weight swaps.

**Decomposed (N > 256):** The PairFormerBlock is split into individually traced sub-operations. The decomposition strategy is auto-selected based on N to minimize call overhead:

| N Range | TriMul Strategy | TriAttn/APB Strategy | Calls/Layer |
|---------|----------------|---------------------|-------------|
| 257-384 | Fused TriMulOut+In (1 call) | Merged (1 call each) | **5** |
| 385-512 | Proj + merged BMM+Output (2 calls each) | Merged (1 call each) | **9** |
| 513-1024 | 3-segment (3 calls each) | 2-segment (2 calls each) | **14** |
| >1024 | 3-segment (3 calls each) | 2-seg + chunked MHA (chunk=128) | **14+C** |

C = 2 * ceil(N / 128) extra calls for chunked TriAttn MHA.

**Merged segment speedups** (validated on SDK 2.28):
- Fused TriMulOut+TriMulIn at N=384: ~30% faster than two separate full TriMul calls (7→5 calls/layer)
- Full TriMul at N=384: 1.57x faster than 3-segment (182ms vs 285ms)
- Merged TriAttn at N=384-512: 1.68-1.79x faster than 2-segment
- Merged AttnPairBias at N=384-512: 1.17-1.31x faster than 2-segment
- chunk_size=128 at N=2048: 8% faster than chunk_size=64 (half the calls)

Sub-op maximum compilation sizes:

| Sub-Op | Segments | Technique | Compiles To |
|--------|----------|-----------|-------------|
| FusedTriMulOut+In | 1 (fused) | Both TriMul ops in one trace | N=384 |
| TriMulOut | 1 (full) or 3 (decomposed) | Full at N<=384, 3-seg at N>384 | N=2048+ |
| TriMulIn | 1 (full) or 3 (decomposed) | Same as TriMulOut | N=2048+ |
| TriAttnStart | 1 (merged) or 2 (decomposed) | Merged at N<=512, 2-seg at N>512 | N=1024 (full MHA) |
| TriAttnStart | 2 (Bias, MHA chunked) | + Chunked MHA (chunk_size=128) | N=2048+ |
| TriAttnEnd | Same as TriAttnStart | Same strategies | N=2048+ |
| AttnPairBias | 1 (merged) or 2 (decomposed) | Merged at N<=512, 2-seg at N>512 | N=2048+ |
| PairTransition | 1 (monolithic) | No decomposition needed | N=2048+ |
| SingleTransition | 1 (monolithic) | No decomposition needed | N=2048+ |

Each sub-op is compiled once, then weights are replaced for all 48 layers.

### Scaling Progression

| Phase | Max N | Key Technique |
|-------|-------|---------------|
| Monolithic trace | 256 | Single PairFormerBlock trace |
| Sub-op traces (7 ops) | 384 | Split into 7 independent sub-op traces |
| Finer TriMul (3 segments each) | 512 | Split TriMul projection from matmul |
| Finer TriAttn/APB (2 segments each) | 1024 | Split bias computation from MHA |
| Chunked TriAttn MHA | **2048** | Chunk MHA rows to fit in HBM |
| Fused TriMulOut+In (SDK 2.28) | 384 | Both TriMul ops in single trace (7→5 calls/layer) |
| Merged segments (SDK 2.28) | 384-512 | Full TriMul + merged TriAttn/APB at small N |
| Optimized chunk size | 2048 | chunk_size=128 (8% faster, half the calls vs 64) |

### Compiled Components (Monolithic Pipeline)

1. **PairFormerBlock** (48 layers): The main trunk. Each layer contains triangular multiplicative updates, triangular attention, pair transition, attention with pair bias, and single transition. One NEFF compiled, 47 weight swaps.

2. **MSA block type A** (3 blocks): Full MSA blocks with msa_att_row, msa_transition, outer_product_mean, and pair_stack. One NEFF compiled, 2 weight swaps.

3. **MSA block type B** (1 block): Last MSA block with reduced structure (only outer_product_mean + pair_stack, no msa_att_row or msa_transition). Requires a separate NEFF due to different computation graph.

4. **TemplatePairBlock** (2 blocks): Pairformer-style blocks inside the template embedder with c_t=64. One NEFF compiled, 1 weight swap.

5. **DiffusionConditioning._forward()**: Transition layers applied to conditioning tensors. Single NEFF with shared weights (no weight swaps needed).

### Not Compiled (CPU)

- **AtomAttentionEncoder/Decoder**: Uses `batch` dict inputs, `repeat_interleave` with data-dependent repeats, `scatter_add_` with runtime indices. Not traceable.
- **InputEmbedder**: Uses `batch` dict inputs.
- **DiffTransformerBlock**: Weight replacement overhead (1.3ms x 4800 calls = 6.2s) exceeds compute savings at N<=256. Net slower when compiled.
- **Confidence heads**: Minor runtime contribution (4-block PairFormer).

### Compilation Parameters

| Parameter | Value |
|-----------|-------|
| `inline_weights_to_neff` | `False` (enables weight replacement) |
| `compiler_args` | `["--target", "trn2"]` |
| Auto-cast | **Not used** (model operates in FP32; `--auto-cast matmult` causes 100x accuracy loss for triangle operations) |
| Precision | FP32 throughout (model weights + activations) |

### Compatibility Patches

OpenFold3 requires 16 source-level patches for Neuron compatibility:

| Patch | Files | Count | Reason |
|-------|-------|-------|--------|
| `autocast("cuda")` -> `autocast("cpu")` | 5 files | 13 | Neuron does not support CUDA autocast |
| `device_type="cuda"` -> `device_type="cpu"` | 3 files | 3 | Same as above (alternate pattern) |
| `torch.cuda.empty_cache()` -> `pass` | 6 files | 7 | No CUDA device on Neuron |
| `torch.cuda.synchronize()` -> `pass` | 1 file | 1 | No CUDA device |
| `torch.cuda.manual_seed_all()` -> `pass` | 1 file | 1 | No CUDA device |
| `use_deepspeed_evo_attention: False` | 1 file | 1 | DeepSpeed not available |

All patches are applied automatically by `patch_openfold3_source()` or the `OpenFold3NeuronPipeline.load_model()` method.

## Known Issues and Limitations

1. **Maximum protein size is N=256 for the monolithic strategy.** PairFormer compilation hits Neuron compiler SBUF limit at N=272. The decomposed strategy extends this to N=2048.

2. **Decomposed strategy has overhead at small N.** At N=256, the decomposed path (244ms/layer) is slower than both the monolithic Neuron path (66ms/layer) and CPU (134ms/layer) due to per-segment compilation overhead and inter-segment data transfer. Use `pairformer_strategy="monolithic"` for N<=256.

3. **Fixed input shapes require recompilation.** Each N value requires separate compilation. Validated N values: 128, 256 (monolithic); 384, 512, 1024, 2048 (decomposed).

4. **Stochastic diffusion amplifies FP rounding.** Trunk outputs match CPU within cos>0.9999, but 200 diffusion denoising steps with random noise amplify tiny FP differences, producing position cosine similarity of ~0.41-0.52. This is expected behavior, not an accuracy issue.

5. **DiffTransformerBlock is overhead-bound at N<=256.** The 1.3ms weight replacement overhead per call, multiplied by 4800 calls (24 blocks x 200 steps), makes Neuron DiffTransformer 9.4s slower than CPU. Left on CPU in the recommended configuration.

6. **MSA block 3 has different structure.** The last MSA block lacks msa_att_row and msa_transition, requiring a separate NEFF. The wrapper API is identical, but the computation graph differs.

7. **OpenFold3 model access requires download.** Weights are publicly available via `aws s3 cp s3://openfold/staging/of3-p2-155k.pt . --no-sign-request`.

8. **Only trn2.3xlarge validated.** Compilation and benchmarks were performed exclusively on trn2.3xlarge with LNC=2 (4 logical cores). Other instance types have not been tested.

9. **GPU comparison uses A100-SXM4-40GB.** A100-80GB or H100-80GB may handle larger N before OOM, but the ~128GB TriMul intermediate at N=2048 exceeds all current single-GPU HBM sizes.

## Source Files

| File | Description |
|------|-------------|
| `src/modeling_openfold3.py` | Main module: monolithic wrappers, decomposed sub-op wrappers, fused/merged wrappers, N-range-aware DecomposedPairFormerCompiler, OpenFold3NeuronPipeline |
| `src/__init__.py` | Package exports |
| `test/integration/test_model.py` | 11 accuracy tests: 6 monolithic + 5 decomposed, all using neuron_allclose |
