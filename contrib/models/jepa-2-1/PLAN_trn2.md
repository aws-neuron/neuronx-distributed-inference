# PLAN_trn2.md — V-JEPA 2.1 Trainium Execution Plan & Results

## Instance

- **Type**: trn2.3xlarge (persistent spot) in sa-east-1b
- **Instance ID**: i-0cae7b2ac61807cf9
- **SSH**: `ssh -i ~/.ssh/trn2-sa-east-1.pem ubuntu@52.67.239.128`
- **Hardware**: 1 Neuron device, 2 logical NeuronCores, 96 GB HBM, 124 GB system RAM
- **Neuron SDK**: torch-neuronx 2.9.0, neuronx-cc 2.24.5133, NxDI 0.9.17334
- **Venv**: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`

## Workflow

```bash
# Sync local → trn2
rsync -avz --exclude='__pycache__' --exclude='._*' \
  ~/dev/neuron-docs/neuronx-distributed-inference/contrib/models/jepa-2-1/ \
  -e "ssh -i ~/.ssh/trn2-sa-east-1.pem" ubuntu@52.67.239.128:jepa-2-1/

# Run on trn2
ssh -i ~/.ssh/trn2-sa-east-1.pem ubuntu@52.67.239.128 \
  "cd jepa-2-1 && source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && python ..."
```

---

## Results Summary

### Compilation & Validation (BF16, 16 frames, 384×384)

| Model | Params | Compile Time | Cosine Sim | Status |
|-------|--------|-------------|------------|--------|
| ViT-B | 86M | ~8 min | 0.999846 | ✅ |
| ViT-L | 300M | 18 min | 0.999873 | ✅ |
| ViT-g | 1.01B | OOM at ~30 min | — | ❌ Host OOM (>124GB RAM) |
| ViT-G | 1.8B | Not attempted | — | ❌ Blocked |

### Latency (batch=1, single NeuronCore)

| Model | Median | Mean | p5 | p95 |
|-------|--------|------|-----|-----|
| ViT-B | 164.5 ms | 164.5 ms | 164.4 ms | 164.6 ms |
| ViT-L | 437.4 ms | 437.5 ms | 437.4 ms | 437.6 ms |

Sub-millisecond variance — deterministic Neuron execution.

### DataParallel Throughput (2 logical NeuronCores)

| Model | Per-clip Latency | Throughput | Speedup |
|-------|-----------------|------------|---------|
| ViT-B | 83.2 ms | 12.0 clips/sec | 1.98x |
| ViT-L | 219.8 ms | 4.5 clips/sec | 1.99x |

Linear scaling with batch size. Any batch size works (dynamic batching).

### NKI Flash Attention (experimental)

| Model | Baseline | NKI Flash | Cosine Sim |
|-------|----------|-----------|------------|
| ViT-B | 164.5 ms | 307.4 ms (+87%) | 0.999972 |
| ViT-L | 437.4 ms | 787.2 ms (+80%) | 1.000006 |

Higher accuracy but slower at 4608 tokens. Reserved for 64-frame (18K token) inference.

---

## Compilation Commands

### ViT-B

```python
import torch, torch_neuronx
from src.modeling_jepa21 import build_vjepa21_encoder

encoder = build_vjepa21_encoder(arch='vit_base', img_size=384, num_frames=16, use_sdpa=False)
encoder.eval().bfloat16()
x = torch.randn(1, 3, 16, 384, 384, dtype=torch.bfloat16)
traced = torch_neuronx.trace(encoder, x, compiler_args=['--auto-cast', 'none'])
traced.save('vjepa21_vitb_16f_384.pt')
```

### ViT-L

Same as above with `arch='vit_large'`.

### Validation Pattern

```python
# Build CPU ref (no NKI) and NKI model with same seed for matching weights
torch.manual_seed(0)
encoder_cpu = build_vjepa21_encoder(..., use_nki_flash=False)
# ... get ref output ...

torch.manual_seed(0)
encoder_nki = build_vjepa21_encoder(..., use_nki_flash=True)
# ... trace and compare ...
```

---

## Compiled Files on Instance

```
~/jepa-2-1/vjepa21_vitb_16f_384_v2.pt   (335M) — ViT-B baseline (best)
~/jepa-2-1/vjepa21_vitl_16f_384.pt      (1.1G) — ViT-L baseline (best)
~/jepa-2-1/vjepa21_vitb_nki_16f_384.pt  (405M) — ViT-B + NKI flash (slower)
~/jepa-2-1/vjepa21_vitl_nki_16f_384.pt  (1.4G) — ViT-L + NKI flash (slower)
```

---

## ViT-g / ViT-G: Compilation Failure Analysis

**Root cause**: neuronx-cc compiler memory scales with graph size. Peak host RAM usage:
- ViT-L (24 layers): ~60GB → fits in 124GB ✅
- ViT-g (40 layers): >124GB → OOM ❌

The failure is in the compiler, not the model. The CPU forward pass succeeds. The compiled NEFF would likely fit in 96GB HBM at runtime.

**Attempted mitigation**: Added `ModuleMarkerStartWrapper`/`EndWrapper` from NxDI to split the graph into groups of 8 layers. Result: markers are inserted but `torch_neuronx.trace()` does NOT respect them — it still compiles the full graph as one unit. The markers are only respected by `parallel_model_trace` from NxD.

**Next steps** (see PLAN.md Phase 3):
1. Use `parallel_model_trace` from NxD (recommended — markers already in code)
2. Compile on trn2.48xlarge (2TB RAM)
3. Manual graph splitting (hacky)
