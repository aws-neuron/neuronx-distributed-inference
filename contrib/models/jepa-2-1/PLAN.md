# PLAN.md — V-JEPA 2.1 Neuron Port Roadmap

## Phase 1 — Initial Port (CPU-only) ✅ COMPLETE

- [x] Analyzed V-JEPA 2.1 source code and paper
- [x] Created self-contained `modeling_jepa21.py` with no upstream imports
- [x] Created CPU-only unit tests
- [x] Created project structure following NxDI contrib conventions
- [ ] Verify CPU forward pass matches upstream vjepa2 repo (numerical equivalence)

## Phase 2 — Neuron Compilation (on Trainium) ✅ COMPLETE

- [x] Set up trn2.3xlarge instance (sa-east-1, persistent spot)
- [x] Traced ViT-B (86M) — compiled on first attempt with `use_sdpa=False`
- [x] Traced ViT-L (300M) — compiled in 18 min
- [x] Validated both: cosine similarity > 0.999 vs CPU reference
- [x] Benchmarked: ViT-B 164.5ms, ViT-L 437.4ms (batch=1, BF16, 16 frames)
- [x] DataParallel: 2x throughput with `torch_neuronx.DataParallel` (zero code changes)
- [x] Integrated NKI flash attention (`attention_isa_kernel`) — works but slower at 4608 tokens
- [x] Added modular compilation markers (`ModuleMarkerStartWrapper`/`EndWrapper`)

### Key findings
- Conv3d, `torch.arange`, `repeat_interleave` all compile natively — no workarounds needed
- Only required change: `use_sdpa=False` to bypass unsupported SDPA
- BF16 softmax dtype fix: `.to(v.dtype)` after softmax
- NKI flash attention: higher accuracy but 1.8x slower at 4608 tokens (designed for 16K+)
- DataParallel: linear throughput scaling, 83ms/clip for ViT-B (2 NeuronCores)

## Phase 3 — Scaling to ViT-g / ViT-G 🔴 BLOCKED

**Blocker**: neuronx-cc compiler OOMs on host (>124GB RAM) when compiling ViT-g (40 layers) as a monolithic graph. The `ModuleMarkerStartWrapper`/`EndWrapper` markers do NOT cause `torch_neuronx.trace()` to split the graph — they are only respected by `parallel_model_trace` from NxD.

### Options (in order of recommendation)

1. **`parallel_model_trace` from NxD** — Use `neuronx_distributed.trace.parallel_model_trace` instead of `torch_neuronx.trace()`. This is how Flux and other NxDI models compile with modular markers. Requires wrapping the model in a `ModelWrapper`-like class with `input_generator()` and `get_model_instance()`. The markers are already in the model code.

2. **Larger instance for compilation** — Compile on trn2.48xlarge (2TB RAM), then load the `.pt` on trn2.3xlarge for inference. Simplest approach, just costs more during compilation.

3. **Manual graph splitting** — Trace layers 0-19 and 20-39 as separate models, chain at runtime. Hacky but avoids NxD dependency.

### Tasks remaining
- [ ] Get ViT-g (1B) compiling via one of the above approaches
- [ ] Validate and benchmark ViT-g
- [ ] Compile, validate, and benchmark ViT-G (1.8B)

## Phase 4 — Downstream Tasks (NOT STARTED)

- [ ] Attentive pooler for classification
- [ ] Predictor for action anticipation
- [ ] Test with pretrained checkpoints
- [ ] 64-frame inference (NKI flash attention becomes relevant here)

## Phase 5 — Contrib Submission (NOT STARTED)

- [ ] Full test suite on Trainium
- [ ] `neuron_allclose()` validation
- [ ] Complete compatibility matrix
- [ ] Submit PR
