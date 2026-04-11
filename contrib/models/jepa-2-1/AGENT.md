# AGENT.md — V-JEPA 2.1 Neuron Port Technical Reference

## Source Code Location

- **Original model code**: `~/dev/vjepa2/`
  - V-JEPA 2 encoder: `src/models/vision_transformer.py` → `VisionTransformer`
  - V-JEPA 2 predictor: `src/models/predictor.py` → `VisionTransformerPredictor`
  - V-JEPA 2 AC predictor: `src/models/ac_predictor.py` → `VisionTransformerPredictorAC`
  - V-JEPA 2.1 encoder: `app/vjepa_2_1/models/vision_transformer.py` → `VisionTransformer` (extended)
  - V-JEPA 2.1 predictor: `app/vjepa_2_1/models/predictor.py` → `VisionTransformerPredictor` (extended)
  - V-JEPA 2.1 modules: `app/vjepa_2_1/models/utils/modules.py` (Block, RoPEAttention, MLP, SwiGLUFFN)
  - V-JEPA 2.1 patch embed: `app/vjepa_2_1/models/utils/patch_embed.py` (PatchEmbed, PatchEmbed3D)
  - Hub/loading: `src/hub/backbones.py` (checkpoint loading, arch configs)
  - Attentive pooler: `src/models/attentive_pooler.py`

## Architecture Details

### Encoder (VisionTransformer)

The V-JEPA 2.1 encoder is a standard ViT with these key features:

1. **Patch Embedding**: `PatchEmbed3D` with Conv3d kernel (tubelet_size, patch_size, patch_size). For video: stride matches kernel. For images: uses separate `PatchEmbed3D` with `tubelet_size=1` when `img_temporal_dim_size` is set.

2. **3D-RoPE**: Rotary position embeddings applied separately to depth/height/width dimensions. Head dim is split into 3 roughly equal segments:
   - `d_dim = 2 * ((head_dim // 3) // 2)` — temporal
   - `h_dim = 2 * ((head_dim // 3) // 2)` — height
   - `w_dim = 2 * ((head_dim // 3) // 2)` — width
   - Remaining dims get no rotation

3. **RoPE Bug**: The `rotate_queries_or_keys` function in V-JEPA 2 uses `.repeat(1,1,1,2)` instead of `.repeat_interleave(2, dim=-1)`. V-JEPA 2.1 fixes this with `repeat_interleave`. Both are preserved for checkpoint compatibility.

4. **Hierarchical Output**: The encoder outputs features from multiple intermediate layers. For ViT-L (depth=24): layers [5, 11, 17, 23]. During training, these are concatenated along the feature dimension. During inference, only the last layer's normed output is returned (unless `return_hierarchical=True`).

5. **Modality Embeddings**: Separate learned embeddings for image vs video input, added after patch embedding.

6. **interpolate_rope**: V-JEPA 2.1 adds RoPE interpolation for resolution flexibility. Height/width positions are scaled by `(pretrained_grid_size - 1) / (actual_grid_size - 1)`.

### Model Configurations

| Arch | embed_dim | depth | num_heads | mlp_ratio | head_dim | d/h/w_dim |
|------|-----------|-------|-----------|-----------|----------|-----------|
| vit_base | 768 | 12 | 12 | 4.0 | 64 | 20, 20, 20 |
| vit_large | 1024 | 24 | 16 | 4.0 | 64 | 20, 20, 20 |
| vit_giant_xformers | 1408 | 40 | 22 | 48/11 | 64 | 20, 20, 20 |
| vit_gigantic_xformers | 1664 | 48 | 26 | 64/13 | 64 | 20, 20, 20 |

### Token Counts

For 384×384 resolution, patch_size=16, tubelet_size=2:
- Image (1 frame, tubelet_size=1): 1 × 24 × 24 = 576 tokens
- 16 frames: 8 × 24 × 24 = 4,608 tokens
- 64 frames: 32 × 24 × 24 = 18,432 tokens

### Attention Pattern

Standard bidirectional self-attention (no causal mask) for the encoder. The AC predictor uses block-causal attention for autoregressive frame prediction.

## Neuron Porting Considerations

### Compilation Approach: `torch_neuronx.trace()`

The encoder is a standard feedforward ViT — no KV cache, no autoregressive generation. `torch_neuronx.trace()` is the right tool.

### Potential Issues

1. **`F.scaled_dot_product_attention`**: Neuron compiler may not support all SDPA backends. May need to fall back to manual attention (`q @ k.T * scale → softmax → @ v`). Set `use_sdpa=False` in the model config.

2. **`timm.models.layers.drop_path`**: External dependency. For inference (eval mode), drop_path is identity. Can be replaced with `nn.Identity()`.

3. **Dynamic shapes**: The encoder supports variable frame counts and resolutions via RoPE interpolation. For Neuron compilation, fix the input shape at trace time. Compile separate models for different input shapes if needed.

4. **`torch.arange` in RoPE**: Dynamic tensor creation inside forward pass. Neuron compiler should handle this for fixed input shapes, but verify.

5. **Large attention matrices**: At 64 frames × 384px, the attention matrix is 18432×18432. This may exceed single-core HBM. Options:
   - Use shorter clips (16 frames → 4608 tokens, manageable)
   - Use TP>1 via NxDI if needed for long clips
   - Use NKI flash attention kernels

6. **Conv3d patch embedding**: Verify Neuron compiler support for 3D convolutions. If unsupported, can be decomposed into reshape + Conv2d.

### Weight Loading

Checkpoints are loaded via `torch.hub.load_state_dict_from_url`. The state dict has keys prefixed with `module.` and `backbone.` which are stripped by `_clean_backbone_key()`. For V-JEPA 2.1 distilled models, the encoder key is `ema_encoder` (not `target_encoder`).

### Inference-Only Simplifications

For inference, these training-only features can be removed:
- Mask application (`apply_masks`) — not used during inference
- Drop path — identity at eval
- Predictor — only needed for pretraining/anticipation
- Activation checkpointing — only for training memory savings

## Reference Patterns

### NxDI Contrib Structure
See `~/dev/Neuron-steering-docs/steering/nxdi-contrib.md` for submission requirements.

### Neuron SDK Docs
See `~/dev/neuron-docs/` for:
- `neuronx-distributed/` — distributed inference patterns
- `nki-library/` — NKI kernel examples (flash attention, etc.)

### Similar Ports
- Vision-language models in NxDI (Qwen-VL, MLLama) have image encoder components
- The Flux diffusion model in NxDI uses TP + NKI attention for large sequence lengths
