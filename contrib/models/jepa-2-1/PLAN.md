# PLAN.md — V-JEPA 2.1 Neuron Port Roadmap

## Current Status: Phase 1 — Initial Port (CPU-only)

### Completed
- [x] Read and analyzed V-JEPA 2.1 source code (encoder, predictor, AC predictor, modules)
- [x] Read the paper (arxiv 2506.09985)
- [x] Created project structure following NxDI contrib conventions
- [x] Created self-contained encoder module (`modeling_jepa21.py`) with no upstream imports
- [x] Created CPU-only unit tests for encoder forward pass
- [x] Created README.md, AGENT.md, PLAN.md

### In Progress
- [ ] Verify CPU forward pass matches upstream vjepa2 repo output (numerical equivalence)
- [ ] Test all 4 encoder variants (ViT-B, ViT-L, ViT-g, ViT-G)

## Phase 2 — Neuron Compilation (on Trainium)

### Tasks
- [ ] Set up trn2 instance with Neuron SDK 2.28+
- [ ] Install dependencies (torch-neuronx, neuronx-distributed-inference)
- [ ] Trace ViT-B encoder with `torch_neuronx.trace()` at 384×384, 16 frames
- [ ] Verify SDPA compatibility — if Neuron doesn't support `F.scaled_dot_product_attention`, add manual attention fallback
- [ ] Verify Conv3d support — if unsupported, decompose to reshape + Conv2d
- [ ] Handle `torch.arange` in RoPE forward pass (may need to precompute)
- [ ] Trace ViT-L encoder
- [ ] Compare traced output vs CPU reference (cosine similarity > 0.99)
- [ ] Benchmark latency and throughput

## Phase 3 — Scaling & Optimization

### Tasks
- [ ] Test ViT-g (1B params) — may need TP>1 or NKI flash attention for 64-frame clips
- [ ] Test ViT-G (1.8B params) — likely needs TP≥2
- [ ] If TP needed: port to NxDI pattern with NKI flash attention
- [ ] Profile memory usage at different frame counts (16, 32, 64)
- [ ] Optimize: batch compilation for multiple input shapes (frame count buckets)

## Phase 4 — Downstream Tasks

### Tasks
- [ ] Add attentive pooler for classification inference
- [ ] Add predictor for action anticipation inference
- [ ] Test with pretrained checkpoints on downstream benchmarks
- [ ] Add AC predictor for robotics planning inference (if applicable)

## Phase 5 — Contrib Submission

### Tasks
- [ ] Run full test suite on Trainium hardware
- [ ] Measure accuracy with `neuron_allclose()` against CPU reference
- [ ] Fill in compatibility matrix with actual test results
- [ ] Fill in benchmark results (throughput, latency)
- [ ] Ensure all tests pass with `pytest`
- [ ] Submit PR following NxDI contrib guidelines

## Key Decisions

### Why start with ViT-B/ViT-L?
- Smaller models compile faster and fit on single NeuronCore
- Validates the porting approach before scaling up
- ViT-B (86M params) and ViT-L (300M params) are practical for many downstream tasks

### Why `torch_neuronx.trace()` first?
- Simpler than full NxDI port
- Encoder is feedforward (no KV cache, no autoregressive)
- Can always upgrade to NxDI later if TP is needed for larger models

### Why not port the predictor first?
- Encoder is the primary inference component
- Predictor is only needed for pretraining and specific tasks (anticipation)
- Encoder features are sufficient for classification, VQA, and feature extraction

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| SDPA not supported on Neuron | Medium | Manual attention fallback already in codebase (`use_sdpa=False`) |
| Conv3d not supported | Low | Decompose to reshape + Conv2d |
| 64-frame ViT-G exceeds single-core HBM | High | Start with shorter clips; upgrade to NxDI with TP if needed |
| RoPE dynamic tensor creation | Medium | Precompute position tensors at trace time |
| `timm` dependency | Low | Replaced with inline `drop_path` (identity at eval) |
