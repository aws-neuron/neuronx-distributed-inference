# Contrib Model: Evo2-7B

Arc Institute's Evo 2 (7B, StripedHyena 2 architecture) DNA language model on AWS Trainium 2 using vanilla `torch_neuronx.trace()` compilation. Evo2 is a hybrid SSM+Attention model that processes genomic sequences at the single-nucleotide level (A, C, G, T tokens).

## Model Information

- **Source:** [arcinstitute/evo2](https://github.com/arcinstitute/evo2) (requires [Zymrael/vortex](https://github.com/Zymrael/vortex) inference engine)
- **Model Type:** DNA language model (single-nucleotide tokenizer, vocab=512)
- **Parameters:** ~7B (bf16)
- **Architecture:** 32-layer StripedHyena 2 (hybrid SSM+Attention)
  - 9 HCS blocks (FIR convolution, k=7)
  - 9 HCM blocks (FIR convolution, k=128)
  - 9 HCL blocks (IIR/SSM with FFT-based convolution)
  - 5 ATT blocks (multi-head attention with RoPE)
- **License:** Apache 2.0 (Evo2); check Vortex for its license terms

## Compatibility Matrix

| Component | Version |
|-----------|---------|
| **Neuron SDK** | 2.28 |
| **PyTorch** | 2.9 (torch-neuronx 2.9.0.2.11) |
| **Instance** | trn2.3xlarge (LNC=2, 4 logical NeuronCores, 96 GB HBM) |
| **DLAMI** | Deep Learning AMI Neuron (Ubuntu 24.04) 20260227 |
| **Python** | 3.12 |
| **Venv** | `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/` |

## Checkpoints

- **HuggingFace:** [arcinstitute/evo2_7b_base](https://huggingface.co/arcinstitute/evo2_7b_base)
- **Weight file:** `evo2_7b_base.pt` (~14 GB bf16)

## Usage

### Setup

```bash
# On trn2.3xlarge with Neuron DLAMI
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Clone repos
git clone https://github.com/arcinstitute/evo2.git
git clone https://github.com/Zymrael/vortex.git

# Install evo2 (without CUDA deps)
pip install -e evo2/ --no-deps
pip install pyyaml einops

# Download weights
python -c "from huggingface_hub import snapshot_download; snapshot_download('arcinstitute/evo2_7b_base')"
```

### Quick Start

```python
import sys
sys.path.insert(0, "vortex")
sys.path.insert(0, "evo2")

from modeling_evo2 import Evo2NeuronPipeline

pipeline = Evo2NeuronPipeline(
    evo2_src_path="evo2",
    vortex_src_path="vortex",
    weights_path="~/.cache/huggingface/hub/models--arcinstitute--evo2_7b_base/snapshots/.../evo2_7b_base.pt",
    prefill_seq_len=2048,
)

# Load and compile
pipeline.load_model()
pipeline.compile_prefill()        # ~168 min (32 blocks, cached: ~50s)
pipeline.compile_decode(batch_size=32)  # ~25 min

# Generate
import torch
input_ids = torch.randint(0, 512, (1, 128), dtype=torch.long)
tokens = pipeline.generate(input_ids, n_tokens=100)
```

### DP=4 Parallelism (Process-Per-Core)

For maximum throughput, run 4 independent processes, each pinned to one NeuronCore:

```bash
# Worker script (simplified)
NEURON_RT_VISIBLE_CORES=0 python worker.py --core-id 0 --batch-size 32 &
NEURON_RT_VISIBLE_CORES=1 python worker.py --core-id 1 --batch-size 32 &
NEURON_RT_VISIBLE_CORES=2 python worker.py --core-id 2 --batch-size 32 &
NEURON_RT_VISIBLE_CORES=3 python worker.py --core-id 3 --batch-size 32 &
```

IMPORTANT: Stagger launches by ~5s to avoid OOM from simultaneous compilations.

## How to Run Tests

```bash
# On trn2.3xlarge
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

cd contrib/models/Evo2-7B
PYTHONPATH=src:$PYTHONPATH \
    pytest test/integration/test_model.py -v -s
```

## Validation Results

**Validated:** 2026-03-25
**Instance:** trn2.3xlarge (LNC=2, 4 NeuronCores, SDK 2.28)

### Prefill Accuracy (seq_len=2048, 1 NeuronCore)

| Metric | Value |
|--------|-------|
| **CPU vs Neuron cosine sim (global)** | **0.99968** |
| **CPU vs Neuron cosine sim (per-position)** | **0.999999** |
| **CPU vs Neuron token match** | **99.01%** |
| Avg Neuron loss | 0.1527 |
| Avg Neuron accuracy | 95.52% |
| Compile time | 167.9 min (32 blocks) |
| Compiler args | `--auto-cast matmult --model-type=transformer` |

### Prefill Performance

| seq_len | Latency P50 | Throughput |
|---------|-------------|------------|
| 128 | 1763 ms | 73 tok/s |
| 2048 | 4956 ms | 413 tok/s |

### Decode Performance (E2E Generation, 100 tokens)

| Config | Throughput | Latency/Token | Token Match |
|--------|-----------|---------------|-------------|
| BS=1, 1 core | 0.72 tok/s | 1347 ms | 100/100 |
| BS=32, 1 core | 15.6 tok/s | 2056 ms | 50/50 (BS=4) |
| BS=64, 1 core | 19.9 tok/s | 3212 ms | -- |
| BS=32, DP=4 | **63.0 tok/s** | -- | -- |

### Batch Decode Throughput Scaling (Single NeuronCore)

| BS | Latency | Throughput | Scaling | Efficiency |
|----|---------|------------|---------|------------|
| 1 | 1347 ms | 0.7 tok/s | 1.0x | 100% |
| 4 | 1394 ms | 2.9 tok/s | 3.9x | 97% |
| 8 | 1457 ms | 5.5 tok/s | 7.4x | 92% |
| 16 | 1724 ms | 9.3 tok/s | 12.5x | 78% |
| 32 | 2056 ms | 15.6 tok/s | 21.0x | 66% |
| 64 | 3212 ms | 19.9 tok/s | 27.7x | 43% |
| 128 | OOM | -- | -- | -- |

### DP=4 Process-Per-Core (BS=32/core)

| Core | Throughput | Compile Time |
|------|-----------|-------------|
| 0 | 15.5 tok/s | 1487s |
| 1 | 15.7 tok/s | 1502s |
| 2 | 15.9 tok/s | 1491s |
| 3 | 16.0 tok/s | 1479s |
| **Total** | **63.0 tok/s** | -- |

Scaling efficiency: 4.04x (near-perfect).

### GPU Comparison (L40S, 48 GB VRAM)

| BS | GPU L40S (tok/s) | Neuron DP=4 (tok/s) | Ratio |
|----|-----------------|--------------------|----|
| 1 | 42.5 | 2.9 | 15x GPU |
| 8 | 327.9 | 21.8 | 15x GPU |
| 32 | 1047.8 | 63.0 | 17x GPU |

The GPU advantage comes from Flash Attention + fused CUDA kernels processing all 32 blocks within a single process. Neuron's 32 separate NEFFs each load ~386 MB weights from HBM for tiny (BS, 1, 4096) activations.

## Architecture Details

### NeuronFFT (Stockham FFT)

Evo2's HCL (IIR/SSM) layers use `torch.fft.rfft/irfft` which is unsupported on Neuron XLA (`aten::view_as_real` missing). NeuronFFT implements a Stockham auto-sort radix-2 FFT using only reshape/slice/cat/element-wise ops:

- Split (real, imag) tensor pairs (no complex tensors)
- Twiddle factors stored as plain Python attributes (not buffers) to avoid NeuronModule dtype mismatch
- O(n log n) scaling, compiles up to seq_len=2048 (fft_size=4096)

### Decode State Management

Each decode block maintains HBM-resident state via `input_output_aliases`:
- **ATT blocks:** KV-cache `(B, max_seqlen, 32, 128)` x 2
- **HCL blocks:** FIR state `(B, 12288, 2)` + IIR state `(B, 4096, 16)`
- **HCM blocks:** FIR state `(B, 12288, 2)` + inner FIR state `(B, 4096, 127)`
- **HCS blocks:** FIR state `(B, 12288, 2)` + inner FIR state `(B, 4096, 6)`

State is manually copied back after each decode step and pushed to HBM via `torch_neuronx.replace_weights()`.

### Known Limitations

1. **Max prefill seq_len = 2048** -- NeuronFFT compiler OOMs at fft_size=8192+
2. **Decode bottleneck is HBM bandwidth** -- 43ms/block x 32 blocks = ~1.4s per token (single core)
3. **IIR state in bf16** -- stable for 100+ tokens, not extensively tested at 1000+
4. **Block 24 NaN** -- requires 3 warmup inference passes to fully initialize HBM
5. **Simultaneous compilation OOMs** -- stagger DP=4 launches by ~5s

## File Structure

```
Evo2-7B/
  README.md                          # This file
  src/
    __init__.py                      # Package exports
    modeling_evo2.py                 # All wrapper classes and pipeline (2090 lines)
  test/
    integration/
      test_model.py                  # Prefill accuracy + decode compilation test
    unit/
      (empty)
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `NeuronFFT` | Stockham FFT for Neuron (replaces torch.fft.*) |
| `Evo2PrefillPipeline` | Block-by-block prefill compilation |
| `ATTDecodeBlock` | Attention decode with KV-cache |
| `HCLDecodeBlock` | IIR/SSM decode with FIR + IIR state |
| `FIRDecodeBlock` | HCM/HCS decode with FIR state |
| `BatchMegaDecode` | All 32 blocks as single batched NEFF |
| `Evo2NeuronPipeline` | High-level pipeline (load, compile, generate) |

### Key Functions

| Function | Purpose |
|----------|---------|
| `patch_vortex_for_neuron()` | Monkey-patch Vortex for Neuron |
| `install_neuron_fft_patch()` | Register NeuronFFT on HCL blocks |
| `load_config()` | Load evo2-7b-8k config |
| `build_model()` | Build StripedHyena with weights |
