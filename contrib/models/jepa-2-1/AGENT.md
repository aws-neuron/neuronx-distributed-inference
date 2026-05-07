# AGENT.md — V-JEPA 2.1 Neuron Port Technical Reference

This file is for coding agents working on this model. It documents architecture decisions, Neuron compilation findings, file layout, and open work items.

## File Layout

```
contrib/models/jepa-2-1/
├── AGENT.md              ← You are here
├── README.md             ← User-facing documentation
├── pyproject.toml        ← Project config (pytest settings, dependencies)
├── examples/
│   └── demo_classify.py  ← CPU video classification demo (HF V-JEPA 2 + SSv2, no Neuron needed)
├── src/
│   ├── __init__.py       ← Exports: build_vjepa21_encoder, VisionTransformer, VisionTransformerPredictor
│   └── modeling_jepa21.py ← Self-contained encoder (~700 lines, no upstream imports)
└── test/
    ├── unit/
    │   └── test_encoder.py           ← 14 CPU-only tests (construction, forward, components)
    └── integration/
        ├── test_model.py             ← 4 Neuron tests (trace, accuracy, ViT-B/L, random weights)
        └── test_pretrained_smoke.py  ← 5 tests: 3 CPU + 2 Neuron (pretrained weight validation)
```

## Source Code

- **Upstream**: [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- **Neuron port**: `src/modeling_jepa21.py` — self-contained, no upstream imports
- **Examples**: `examples/demo_classify.py` — CPU-only video classification demo using HuggingFace V-JEPA 2 finetuned on SSv2 (174 action classes). Requires `transformers`, `accelerate`, `torchvision`, `decord`.
- **Key difference from most NxDI contrib models**: This is a vision encoder, NOT a causal LM. It does not use `NeuronBaseForCausalLM`, KV cache, token generation, or vLLM. Compilation uses `torch_neuronx.trace()` directly.

## Classes and Functions in `modeling_jepa21.py`

| Class/Function | Purpose |
|----------------|---------|
| `PatchEmbed` | 2D image → patch embedding (Conv2d) |
| `PatchEmbed3D` | 3D video → tubelet embedding (Conv3d, kernel=(tubelet_size, patch_size, patch_size)) |
| `rotate_queries_or_keys_v21` | 3D RoPE rotation using `repeat_interleave` (not `repeat`) |
| `RoPEAttention` | Multi-head attention with 3D RoPE (depth/height/width). Has `use_nki_flash` flag |
| `Attention` | Standard multi-head attention without RoPE (used by predictor) |
| `MLP` | Standard GELU MLP |
| `SwiGLUFFN` | SiLU-gated FFN (used when `use_silu=True`) |
| `Block` | Transformer block: LayerNorm → Attention → residual → LayerNorm → MLP → residual |
| `VisionTransformer` | Main encoder. Patch embed → blocks → hierarchical norm output |
| `VisionTransformerPredictor` | Predictor head for masked prediction (included for completeness, not needed for basic inference) |
| `build_vjepa21_encoder()` | Builder function. Main entry point. Accepts `arch`, `img_size`, `num_frames`, `pretrained`, etc. |
| `_load_pretrained_weights()` | Downloads checkpoints from `dl.fbaipublicfiles.com/vjepa2/` |
| `_ARCH_CONFIGS` | Dict mapping arch names to (embed_dim, depth, num_heads, mlp_ratio) |

## Architecture

### Encoder (VisionTransformer)

Standard ViT with 3D-RoPE, bidirectional attention, hierarchical output.

| Arch | Params | embed_dim | depth | num_heads | head_dim | mlp_ratio |
|------|--------|-----------|-------|-----------|----------|-----------|
| vit_base | 86M | 768 | 12 | 12 | 64 | 4.0 |
| vit_large | 300M | 1024 | 24 | 16 | 64 | 4.0 |
| vit_giant | 1.01B | 1408 | 40 | 22 | 64 | 48/11 |
| vit_gigantic | 1.8B | 1664 | 48 | 26 | 64 | 64/13 |

### Token Counts (384×384, patch_size=16, tubelet_size=2)

- 16 frames: 8 × 24 × 24 = **4,608 tokens**
- 64 frames: 32 × 24 × 24 = 18,432 tokens
- 1 frame (image): 1 × 24 × 24 = 576 tokens (uses `patch_embed_img` with tubelet_size=1)

### Key Features

- **PatchEmbed3D**: Conv3d with kernel=stride=(tubelet_size, patch_size, patch_size)
- **3D-RoPE**: Separate rotations for depth/height/width on head_dim slices (d_dim=h_dim=w_dim=20 for head_dim=64, 4 dims unrotated)
- **Hierarchical output**: Normed features from intermediate layers (e.g., [5,11,17,23] for depth=24). Inference returns only the last layer's normed output unless `return_hierarchical=True` or `training=True`.
- **Modality embeddings**: Separate learned `img_mod_embed` / `video_mod_embed` added after patch embedding
- **interpolate_rope**: Scales RoPE positions for resolution flexibility beyond pretrained grid size
- **Image vs video path**: `img_temporal_dim_size=1` triggers `patch_embed_img` (tubelet_size=1) for single-frame inputs

### Attention — NOT MLA

This model uses standard multi-head attention (MHA), NOT Multi-head Latent Attention (MLA). There is no latent compression, no `kv_lora_rank`, no weight absorption. The DeepSeek-V3 MLA kernel is not applicable here. The existing `attention_isa_kernel` NKI kernel is the correct one for this use case.

## Neuron Compilation — Verified Findings

### What works with `torch_neuronx.trace()` (neuronx-cc 2.24.5133)

- **Conv3d**: Compiles natively. No decomposition needed.
- **`torch.arange` in RoPE**: Compiles natively for fixed input shapes. No precomputation needed.
- **`repeat_interleave`**: Compiles natively. No reshape/expand workaround needed.
- **Manual attention** (`q @ k.T * scale → softmax → @ v`): Works correctly with `use_sdpa=False`.
- **BF16 inference**: Works with `--auto-cast none`. Cast model to `.bfloat16()` and use BF16 input tensors.

### What does NOT work

- **`F.scaled_dot_product_attention`**: Not supported by `torch_neuronx.trace()`. Must use `use_sdpa=False`.
- **BF16 softmax on CPU**: `softmax()` promotes BF16→FP32, causing dtype mismatch with V tensor. Fixed with `.to(v.dtype)` after softmax.
- **ViT-g/ViT-G monolithic compilation**: neuronx-cc OOMs on host (>124GB RAM needed for 40+ layer graph). See "Modular Compilation" below.

### NKI Flash Attention (`attention_isa_kernel`)

Integrated the NxDI production NKI flash attention kernel for bidirectional attention.

**Interface** (from `neuronxcc.nki._private_kernels.attention`):
```python
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit
_flash = nki_jit()(attention_isa_kernel)

# q: (B*H, d_head, seqlen), k: (B*H, d_head, seqlen), v: (B*H, seqlen, d_head)
# out: pre-allocated zeros (B*H, seqlen, d_head)
_flash(q, k, v, scale, out, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
```

**Result**: Higher numerical accuracy (cos_sim 0.9999 vs 0.9998) but **slower** at 4608 tokens — 307ms vs 165ms for ViT-B. The kernel overhead (reshape, launch) outweighs the flash attention benefit at this sequence length. The kernel is designed for 16K+ tokens. Use `use_nki_flash=False` for 16-frame inference.

**Important**: The NKI kernel cannot run on CPU. When using `use_nki_flash=True`, build a separate CPU reference model with `use_nki_flash=False` for validation. The kernel only executes during XLA tracing.

### Modular Compilation (Layer Boundary Markers)

Added `ModuleMarkerStartWrapper`/`ModuleMarkerEndWrapper` from NxDI to split the compiler graph into groups of N layers. Controlled by `modular_compilation_group_size` parameter.

**Status**: Markers are inserted correctly and validated on ViT-B (identical output and latency to baseline). However, **`torch_neuronx.trace()` does not respect the markers for graph splitting** — ViT-g still OOMs with group_size=8. The markers are only respected by `neuronx_distributed.trace.parallel_model_trace`.

**Next step**: Use `parallel_model_trace` from NxD instead of `torch_neuronx.trace()`, or compile on a larger instance (trn2.48xlarge with 2TB RAM).

### DataParallel Throughput

`torch_neuronx.DataParallel` distributes inference across NeuronCores with zero model changes:
```python
model_dp = torch_neuronx.DataParallel(traced_model)
output = model_dp(batched_input)  # splits batch across cores
```
trn2.3xlarge has 2 logical NeuronCores → 2x throughput. Scales linearly with batch size.

## Benchmark Results (trn2.3xlarge, BF16, 16 frames, 384×384)

### Compilation & Validation

| Model | Params | Compile Time | Cosine Sim vs CPU | Status |
|-------|--------|-------------|-------------------|--------|
| ViT-B | 86M | ~8 min | 0.9998 | ✅ |
| ViT-L | 300M | ~18 min | 0.9999 | ✅ |
| ViT-g | 1.01B | OOM at ~30 min | — | ❌ Host OOM |
| ViT-G | 1.8B | Not attempted | — | ❌ Blocked |

### Latency (batch=1, single NeuronCore)

| Model | Median | p5 | p95 |
|-------|--------|----|-----|
| ViT-B | 164.5 ms | 164.4 ms | 164.6 ms |
| ViT-L | 437.4 ms | 437.4 ms | 437.6 ms |

### DataParallel (2 NeuronCores)

| Model | Per-clip Latency | Throughput | Speedup |
|-------|-----------------|------------|---------|
| ViT-B | 83.2 ms | 12.0 clips/sec | 1.98x |
| ViT-L | 219.8 ms | 4.5 clips/sec | 1.99x |

### NKI Flash Attention (experimental)

| Model | Baseline | NKI Flash | Cosine Sim |
|-------|----------|-----------|------------|
| ViT-B | 164.5 ms | 307.4 ms (+87%) | 0.9999 |
| ViT-L | 437.4 ms | 787.2 ms (+80%) | 1.0000 |

## Testing

### Running tests

```bash
# Unit tests (CPU only, 14 tests)
cd contrib/models/jepa-2-1/
pytest test/unit/ -v

# Integration tests (needs Neuron hardware, 4 + 5 tests)
pytest test/integration/ -v

# Just the pretrained smoke tests (3 CPU + 2 Neuron)
pytest test/integration/test_pretrained_smoke.py -v
```

### What the tests cover

**Unit tests (`test/unit/test_encoder.py`):**
- `TestEncoderConstruction`: ViT-B/L/g construction, invalid arch raises ValueError (4 tests)
- `TestEncoderForward`: video/image/batch shapes, hierarchical output, determinism, 256×256 resolution (6 tests)
- `TestEncoderComponents`: PatchEmbed3D (video + image), RoPEAttention, Block (4 tests)

**Integration tests (`test/integration/test_model.py`):**
- `TestNeuronTrace`: trace ViT-B image, trace ViT-B video, Neuron vs CPU accuracy via `neuron_allclose` (3 tests)
- `TestNeuronTraceVitLarge`: trace ViT-L image (1 test)
- Uses random weights (fast compilation, no download)

**Pretrained smoke tests (`test/integration/test_pretrained_smoke.py`):**
- `TestPretrainedCPU`: loads pretrained ViT-B, checks output shape, checks no NaN/Inf (3 tests)
- `TestPretrainedNeuron`: compiles pretrained ViT-B, validates cosine similarity > 0.999 vs CPU, checks no NaN/Inf (2 tests)
- Downloads ~350MB pretrained weights on first run
- Neuron tests take ~14 min (two compilations of ViT-B 16-frame)

### Test gaps (future work)

- No ViT-g/ViT-G tests (blocked by compilation on trn2.3xlarge)
- No 64-frame tests
- No predictor tests

## Instance Requirements

- **Minimum for ViT-B/L**: trn2.3xlarge (1 Neuron device, 2 logical NeuronCores, 96 GB HBM, 124 GB system RAM)
- **Required for ViT-g/G compilation**: trn2.48xlarge (>130 GB host RAM needed for compilation; compiled models run on any trn2)
- **Neuron SDK**: torch-neuronx 2.9.0, neuronx-cc 2.24.5133, NxDI 0.9.17334
- **Venv**: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`

## Workflow

```bash
# Activate the Neuron venv on the instance
. /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Run unit tests (CPU only, no Neuron device needed)
cd contrib/models/jepa-2-1/
pytest test/unit/ -v

# Run integration tests (needs Neuron hardware)
pytest test/integration/ -v

# Run only the pretrained smoke tests
pytest test/integration/test_pretrained_smoke.py -v
```

## Weight Loading

Checkpoints loaded via `torch.hub.load_state_dict_from_url`. State dict keys prefixed with `module.` and `backbone.` are stripped. Distilled models (ViT-B, ViT-L) use key `ema_encoder`; self-supervised (ViT-g, ViT-G) use `target_encoder`.

| Arch | Checkpoint file | State dict key |
|------|----------------|----------------|
| vit_base | `vjepa2_1_vitb_dist_vitG_384.pt` | `ema_encoder` |
| vit_large | `vjepa2_1_vitl_dist_vitG_384.pt` | `ema_encoder` |
| vit_giant | `vjepa2_1_vitg_384.pt` | `target_encoder` |
| vit_gigantic | `vjepa2_1_vitG_384.pt` | `target_encoder` |

## Open Work Items

### P0 — Needed for production readiness
1. **Compile ViT-g (1B) and ViT-G (1.8B)**: Use `parallel_model_trace` from NxD (markers already in code) or compile on trn2.48xlarge (2TB RAM). The modular compilation markers are already inserted.

### P1 — Valuable additions
2. **64-frame inference**: 18,432 tokens — NKI flash attention should become beneficial here. Need to benchmark.
3. **Downstream tasks**: Attentive pooler for classification, predictor for action anticipation.

### P2 — Nice to have
4. **Tensor parallelism**: For ViT-G on multi-device instances. Would require wrapping with NxD parallel layers.
5. **Dynamic resolution**: Test with non-384 resolutions using `interpolate_rope=True`.

## Reference Code in the NxDI Repo

These files are useful references for future modifications:

- `src/neuronx_distributed_inference/modules/attention/attention_base.py` — NKI flash attention integration patterns, `attention_isa_kernel` usage
- `src/neuronx_distributed_inference/models/diffusers/flux/modeling_flux.py` — Non-autoregressive model using NKI attention + modular markers (closest analog to this model)
- `src/neuronx_distributed_inference/models/mllama/modeling_mllama_vision.py` — Vision encoder using NKI attention
- `src/neuronx_distributed_inference/models/layer_boundary_marker.py` — `ModuleMarkerStartWrapper`/`EndWrapper` for modular compilation
- `contrib/models/DeepSeek-V3/src/modeling_deepseek.py` — MLA attention (NOT applicable here, but useful reference for NxDI patterns like TP sharding, KV cache, config classes)
