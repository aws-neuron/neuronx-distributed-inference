# AGENT.md — V-JEPA 2.1 Neuron Port Technical Reference

## Source Code

- **Upstream**: `~/dev/vjepa2/` (Meta's vjepa2 repo)
- **Neuron port**: `src/modeling_jepa21.py` — self-contained, no upstream imports

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

### Key Features

- **PatchEmbed3D**: Conv3d with kernel=stride=(tubelet_size, patch_size, patch_size)
- **3D-RoPE**: Separate rotations for depth/height/width on head_dim slices (d_dim=h_dim=w_dim=20 for head_dim=64, 4 dims unrotated)
- **Hierarchical output**: Normed features from intermediate layers (e.g., [5,11,17,23] for depth=24). Inference returns only the last layer's normed output.
- **Modality embeddings**: Separate learned embeddings for image vs video
- **interpolate_rope**: Scales RoPE positions for resolution flexibility

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

**Status**: Markers are inserted correctly and validated on ViT-B (identical output and latency to baseline). However, **`torch_neuronx.trace()` does not respect the markers for graph splitting** — ViT-g still OOMs with group_size=8. The markers are likely only respected by `neuronx_distributed.trace.parallel_model_trace`.

**Next step**: Use `parallel_model_trace` from NxD instead of `torch_neuronx.trace()`, or compile on a larger instance (trn2.48xlarge with 2TB RAM).

### DataParallel Throughput

`torch_neuronx.DataParallel` distributes inference across NeuronCores with zero model changes:
```python
model_dp = torch_neuronx.DataParallel(traced_model)
output = model_dp(batched_input)  # splits batch across cores
```
trn2.3xlarge has 2 logical NeuronCores → 2x throughput. Scales linearly with batch size.

## Instance Details

- **Type**: trn2.3xlarge (persistent spot) in sa-east-1
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

## Weight Loading

Checkpoints loaded via `torch.hub.load_state_dict_from_url`. State dict keys prefixed with `module.` and `backbone.` are stripped. Distilled models (ViT-B, ViT-L) use key `ema_encoder`; self-supervised (ViT-g, ViT-G) use `target_encoder`.

## Reference Code in ~/dev/neuron-docs/

- `nki-library/src/.../core/attention/` — NKI flash attention kernels (production)
- `nki-library/src/.../core/embeddings/rope.py` — NKI RoPE kernel
- `neuronx-distributed-inference/src/.../models/diffusers/flux/` — Flux model (non-autoregressive, uses NKI attention + modular markers)
- `neuronx-distributed-inference/src/.../models/mllama/modeling_mllama_vision.py` — MLLama vision encoder (uses NKI attention)
- `neuronx-distributed-inference/src/.../models/layer_boundary_marker.py` — ModuleMarkerStart/End for modular compilation
