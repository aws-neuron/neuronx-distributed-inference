# PLAN_trn2.md — V-JEPA 2.1 Trainium Execution Plan

## Instance

- **Type**: trn2.3xlarge (spot) in sa-east-1b
- **Instance ID**: i-0cae7b2ac61807cf9
- **SSH**: `ssh -i ~/.ssh/trn2-sa-east-1.pem ubuntu@52.67.239.128`
- **Hardware**: 1 Neuron device, 4 NeuronCores, 96 GB HBM, 124 GB system RAM, 418 GB disk free
- **OS**: Ubuntu 24.04, Python 3.12.3
- **Neuron driver**: aws-neuronx-dkms 2.27.4, runtime 2.31.24 (apt-installed)
- **Python venv**: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`
- **Neuron SDK**: torch-neuronx 2.9.0, neuronx-cc 2.24.5133, neuronx-distributed-inference 0.9.17334
- **PyTorch**: 2.9.1, torch-xla 2.9.0, torchvision 0.24.1
- **Tools**: pytest 9.0.3

## Workflow

Edit code locally at `~/dev/neuron-docs/neuronx-distributed-inference/contrib/models/jepa-2-1/`, rsync to trn2, run remotely. Same pattern as the autoresearch port.

```bash
# Sync
rsync -avz --exclude='__pycache__' --exclude='.DS_Store' --exclude='._*' \
  ~/dev/neuron-docs/neuronx-distributed-inference/contrib/models/jepa-2-1/ \
  -e "ssh -i ~/.ssh/trn2-sa-east-1.pem" ubuntu@52.67.239.128:jepa-2-1/

# Run remotely
ssh -i ~/.ssh/trn2-sa-east-1.pem ubuntu@52.67.239.128 \
  "cd jepa-2-1 && source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && python ..."
```

---

## Step 1 — Sync Code & Run CPU Smoke Test on trn2

Rsync the project and verify the encoder runs on CPU. The Neuron SDK venv is pre-installed.

```bash
# Sync
rsync -avz --exclude='__pycache__' --exclude='._*' \
  ~/dev/neuron-docs/neuronx-distributed-inference/contrib/models/jepa-2-1/ \
  -e "ssh -i ~/.ssh/trn2-sa-east-1.pem" ubuntu@52.67.239.128:jepa-2-1/

# CPU smoke test
ssh -i ~/.ssh/trn2-sa-east-1.pem ubuntu@52.67.239.128 << 'EOF'
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd jepa-2-1
python -c "
from src.modeling_jepa21 import build_vjepa21_encoder
import torch
encoder = build_vjepa21_encoder(arch='vit_base', img_size=384, num_frames=16, pretrained=False)
encoder.eval()
x = torch.randn(1, 3, 16, 384, 384)
with torch.no_grad():
    out = encoder(x)
print(f'Output shape: {out.shape}')  # expect (1, 4608, 768)
"
EOF
```

**Success criteria**: Output shape is `(1, 4608, 768)` for ViT-B with 16 frames.

## Step 2 — Trace ViT-B Encoder with torch_neuronx

First compilation attempt. Start with the smallest model (ViT-B, 86M params) and 16 frames. Cast to `bfloat16` — trn2 NeuronCores are heavily optimized for BF16/FP8, and explicit casting avoids unpredictable compiler auto-cast behavior.

```python
import torch
import torch_neuronx
from src.modeling_jepa21 import build_vjepa21_encoder

encoder = build_vjepa21_encoder(arch='vit_base', img_size=384, num_frames=16, pretrained=False)
encoder.eval().bfloat16()

example = torch.randn(1, 3, 16, 384, 384, dtype=torch.bfloat16)
traced = torch_neuronx.trace(encoder, example, compiler_args=['--auto-cast', 'none'])
traced.save("vjepa21_vitb_16f_384.pt")
print("Compilation succeeded")
```

**Note on masking**: The encoder's inference path uses `masks=None` by default — no tokens are dropped, so all tensor shapes are fully static. The masking codepath is training-only and won't be triggered during tracing.

**Expected issues** (debug in order of likelihood):

1. **SDPA not supported** → Set `use_sdpa=False` in the encoder config or add a manual attention fallback path in `modeling_jepa21.py`.
2. **Conv3d not supported** → Two options: (a) decompose `PatchEmbed3D` into reshape + Conv2d, or (b) replace with reshape + `nn.Linear` since stride == kernel_size makes the convolution equivalent to a linear projection over flattened tubelet patches — this maps directly to MatMul on NeuronCore and may be faster.
3. **`torch.arange` in RoPE** → Precompute RoPE frequency tensors before tracing (move out of forward pass).
4. **`repeat_interleave` not supported** → Replace with equivalent `reshape`/`expand`/`reshape` sequence.

**Success criteria**: `.pt` file saved, no compilation errors.

## Step 3 — Validate Traced Model Output

Compare Neuron-traced output against CPU reference.

```python
import torch
import torch_neuronx

# CPU reference (BF16)
encoder_cpu = build_vjepa21_encoder(arch='vit_base', img_size=384, num_frames=16, pretrained=False)
encoder_cpu.eval().bfloat16()
x = torch.randn(1, 3, 16, 384, 384, dtype=torch.bfloat16)
with torch.no_grad():
    ref = encoder_cpu(x)

# Neuron
traced = torch.jit.load("vjepa21_vitb_16f_384.pt")
neuron_out = traced(x)

cos_sim = torch.nn.functional.cosine_similarity(ref.flatten().float(), neuron_out.flatten().float(), dim=0)
print(f"Cosine similarity: {cos_sim.item():.6f}")  # target > 0.99
```

**Success criteria**: Cosine similarity > 0.99 between CPU and Neuron outputs.

## Step 4 — Trace ViT-L Encoder

Scale up to ViT-L (300M params, 16 frames, 4608 tokens).

```python
encoder = build_vjepa21_encoder(arch='vit_large', img_size=384, num_frames=16, pretrained=False)
encoder.eval().bfloat16()
example = torch.randn(1, 3, 16, 384, 384, dtype=torch.bfloat16)
traced = torch_neuronx.trace(encoder, example, compiler_args=['--auto-cast', 'none'])
traced.save("vjepa21_vitl_16f_384.pt")
```

**Potential issue**: ViT-L has 24 layers × 16 heads. Attention matrices are 4608×4608 per head. Should fit in 96 GB HBM on a single NeuronCore, but watch for OOM during compilation (neuronx-cc can be memory-hungry on the host side — 124 GB system RAM may be tight for large graphs).

**Success criteria**: Compilation succeeds, cosine similarity > 0.99 vs CPU.

## Step 5 — Benchmark Latency

Measure inference latency for both models.

```python
import time
import torch

traced = torch.jit.load("vjepa21_vitb_16f_384.pt")
x = torch.randn(1, 3, 16, 384, 384, dtype=torch.bfloat16)

# Warmup
for _ in range(5):
    traced(x)

# Benchmark
times = []
for _ in range(50):
    t0 = time.perf_counter()
    traced(x)
    t1 = time.perf_counter()
    times.append(t1 - t0)

import statistics
print(f"ViT-B 16f: {statistics.median(times)*1000:.1f} ms median, {statistics.mean(times)*1000:.1f} ms mean")
```

Repeat for ViT-L. Record results in README.md compatibility matrix.

## Step 6 — Test with Pretrained Weights (Optional)

If Meta's checkpoints are accessible via `torch.hub`:

```python
encoder = build_vjepa21_encoder(arch='vit_large', img_size=384, num_frames=16, pretrained=True)
```

This validates that the weight loading path works end-to-end on Neuron.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SDPA unsupported on Neuron | `use_sdpa=False` flag already in model; manual `q @ k.T / sqrt(d) → softmax → @ v` fallback |
| Conv3d unsupported | Decompose to `reshape` + `Conv2d`, or replace with `reshape` + `nn.Linear` (maps to NeuronCore MatMul engine) |
| Dynamic `torch.arange` in RoPE | Precompute freq tensors as buffers; register in `__init__` |
| `repeat_interleave` unsupported | Replace with `reshape`→`expand`→`reshape` |
| Host OOM during compilation (124 GB RAM) | Compile ViT-B first (smaller graph); use `NEURON_CC_FLAGS="--retry_failed_compilation"` |
| Spot instance termination | Save compiled `.pt` files to S3 after each successful compilation |

## Out of Scope (for now)

- ViT-g / ViT-G (need TP, Phase 3 in PLAN.md)
- 64-frame inference (18K tokens, likely needs NKI flash attention)
- Predictor / AC predictor compilation
- Attentive pooler
- Downstream task benchmarks

---

## Execution Results (2026-04-11)

All steps executed on trn2.3xlarge `i-0cae7b2ac61807cf9` in sa-east-1.
SDK: torch-neuronx 2.9.0, neuronx-cc 2.24.5133, Python 3.12.3.

### Step 1 — Sync & CPU Smoke Test ✅

- Rsync: 14 files transferred
- CPU output shape: `(1, 4608, 768)` — matches expected for ViT-B/16 with 16 frames

### Step 2 — Trace ViT-B ✅

- Compiled on **first attempt** with `use_sdpa=False` and `--auto-cast none`
- None of the anticipated workarounds were needed:
  - Conv3d: compiled natively
  - `torch.arange` in RoPE: compiled natively
  - `repeat_interleave`: compiled natively
- Only required change: `use_sdpa=False` to bypass `F.scaled_dot_product_attention`

### Step 2 (fix) — BF16 dtype fix

- `softmax()` promotes BF16→FP32 on CPU, causing dtype mismatch in manual attention path
- Fix: added `.to(v.dtype)` after `softmax` in both `RoPEAttention` and `Attention` classes

### Step 3 — Validate ViT-B ✅

| Metric | Value |
|--------|-------|
| CPU output shape | `(1, 4608, 768)` |
| Neuron output shape | `(1, 4608, 768)` |
| Cosine similarity | **0.999846** |
| Max abs diff | 0.078125 |
| Mean abs diff | 0.004509 |

### Step 4 — Trace ViT-L ✅

- Compilation time: **1073s (~18 min)**
- No host OOM — 124 GB system RAM was sufficient

### Step 4b — Validate ViT-L ✅

| Metric | Value |
|--------|-------|
| CPU output shape | `(1, 4608, 1024)` |
| Neuron output shape | `(1, 4608, 1024)` |
| Cosine similarity | **0.999873** |
| Max abs diff | 0.132812 |
| Mean abs diff | 0.007144 |

### Step 5 — Benchmark Latency ✅

Batch=1, BF16, 16 frames, 384×384, 50 iterations after 5 warmup:

| Model | Params | Median | Mean | p5 | p95 |
|-------|--------|--------|------|-----|-----|
| ViT-B | 86M | **164.5 ms** | 164.5 ms | 164.4 ms | 164.6 ms |
| ViT-L | 300M | **437.4 ms** | 437.5 ms | 437.4 ms | 437.6 ms |

Sub-millisecond variance — typical of Neuron hardware deterministic execution.

### Files modified

- `src/modeling_jepa21.py` — `.to(v.dtype)` after softmax in manual attention paths
- `README.md` — updated compatibility matrix, compilation example, known issues
