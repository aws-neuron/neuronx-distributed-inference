#!/usr/bin/env python3
"""
Compile and test the Qwen3.5 MoE Vision Encoder on Neuron.

This script:
1. Builds a standalone ViT vision encoder from HF weights
2. Compiles it with torch_neuronx.trace for Neuron
3. Runs inference and compares against CPU reference
4. Measures compilation time and inference latency

The vision encoder is a 27-layer ViT:
  - hidden_size=1152, num_heads=16, intermediate=4304
  - patch_size=16, spatial_merge_size=2
  - out_hidden_size=2048 (matches text decoder hidden dim)

Environment:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Vision Encoder (pure PyTorch, no NxDI deps -- for CPU reference + tracing)
# ============================================================================


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen):
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class VisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states, attention_mask=None, cos=None, sin=None):
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)

        # Rotary embeddings (rotate_half style, matching HF reference)
        if cos is not None and sin is not None:
            cos_u = cos.unsqueeze(-2)  # (seq_len, 1, head_dim)
            sin_u = sin.unsqueeze(-2)

            # rotate_half: (-x2, x1) where x1/x2 are first/second half
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            q = (q * cos_u) + (rotate_half(q) * sin_u)
            k = (k * cos_u) + (rotate_half(k) * sin_u)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        return self.proj(out)


class VisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn="gelu_pytorch_tanh"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        if act_fn == "gelu_pytorch_tanh":
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, attention_mask=None, cos=None, sin=None):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), attention_mask, cos, sin
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionEncoder(nn.Module):
    """Pure PyTorch vision encoder for Qwen3.5 MoE.

    This module takes already-preprocessed inputs:
    - hidden_states: (seq_len, hidden_size) -- after patch_embed + pos_embed
    - attention_mask: (1, 1, seq_len, seq_len) -- block-diagonal mask
    - cos, sin: (seq_len, head_dim) -- rotary embeddings

    And returns merged vision embeddings: (merged_seq_len, out_hidden_size)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                VisionBlock(
                    config.hidden_size, config.num_heads, config.intermediate_size
                )
                for _ in range(config.depth)
            ]
        )
        # Merger
        self.merger_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        merger_hidden = config.hidden_size * (config.spatial_merge_size**2)
        self.merger_fc1 = nn.Linear(merger_hidden, merger_hidden)
        self.merger_act = nn.GELU()
        self.merger_fc2 = nn.Linear(merger_hidden, config.out_hidden_size)

    def forward(self, hidden_states, attention_mask, cos, sin):
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, cos, sin)

        # Merger: norm -> reshape -> fc1 -> gelu -> fc2
        hidden_states = self.merger_norm(hidden_states)
        merge_size = self.config.spatial_merge_size
        merged_hidden = self.config.hidden_size * (merge_size**2)
        hidden_states = hidden_states.view(-1, merged_hidden)
        hidden_states = self.merger_fc2(self.merger_act(self.merger_fc1(hidden_states)))
        return hidden_states


# ============================================================================
# Weight loading
# ============================================================================


def load_vision_weights(model, model_path):
    """Load HF vision encoder weights into our model."""
    from safetensors import safe_open

    # Find safetensors files (exclude .index.json)
    st_files = sorted(
        p for p in Path(model_path).glob("*.safetensors") if p.suffix == ".safetensors"
    )

    print(f"  Loading from {len(st_files)} safetensors files...")

    vision_keys_loaded = 0
    for sf_path in st_files:
        with safe_open(str(sf_path), framework="pt") as f:
            for key in f.keys():
                if not key.startswith("model.visual."):
                    continue
                # Strip "model.visual." prefix
                param_key = key[len("model.visual.") :]

                # Map HF names to our names
                # HF: visual.blocks.0.attn.qkv.weight -> blocks.0.attn.qkv.weight
                # HF: visual.merger.norm.weight -> merger_norm.weight
                # HF: visual.merger.linear_fc1.weight -> merger_fc1.weight
                param_key = param_key.replace("merger.norm.", "merger_norm.")
                param_key = param_key.replace("merger.linear_fc1.", "merger_fc1.")
                param_key = param_key.replace("merger.linear_fc2.", "merger_fc2.")
                param_key = param_key.replace("mlp.linear_fc1.", "mlp.fc1.")
                param_key = param_key.replace("mlp.linear_fc2.", "mlp.fc2.")

                # Skip patch_embed (runs on CPU) and pos_embed and rotary
                if (
                    "patch_embed" in param_key
                    or "pos_embed" in param_key
                    or "rotary" in param_key
                ):
                    continue

                try:
                    parts = param_key.split(".")
                    target = model
                    for part in parts[:-1]:
                        if part.isdigit():
                            target = target[int(part)]
                        else:
                            target = getattr(target, part)

                    param_name = parts[-1]
                    param = getattr(target, param_name)
                    tensor = f.get_tensor(key)
                    if param.shape != tensor.shape:
                        print(
                            f"  WARN: Shape mismatch {key}: {param.shape} vs {tensor.shape}"
                        )
                        continue
                    param.data.copy_(tensor)
                    vision_keys_loaded += 1
                except Exception as e:
                    # Skip unmapped keys silently
                    pass

    print(f"  Loaded {vision_keys_loaded} vision weight tensors")
    return vision_keys_loaded


# ============================================================================
# CPU preprocessing (same as VisionModelWrapper)
# ============================================================================


def preprocess_vision_inputs(pixel_values, image_grid_thw, config, model_path):
    """Run CPU-side preprocessing: patch_embed, pos_embed, rotary."""
    # Load patch_embed and pos_embed from HF weights
    from safetensors import safe_open

    hidden_size = config.hidden_size
    patch_size = config.patch_size
    temporal_patch_size = config.temporal_patch_size
    spatial_merge_size = config.spatial_merge_size
    num_position_embeddings = config.num_position_embeddings
    num_heads = config.num_heads
    head_dim = hidden_size // num_heads

    # Build patch embed
    patch_embed = nn.Conv3d(
        3,
        hidden_size,
        kernel_size=[temporal_patch_size, patch_size, patch_size],
        stride=[temporal_patch_size, patch_size, patch_size],
        bias=True,
    )
    pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
    rotary = VisionRotaryEmbedding(head_dim // 2)

    # Load weights (exclude .index.json)
    st_files = sorted(
        p for p in Path(model_path).glob("*.safetensors") if p.suffix == ".safetensors"
    )

    for sf_path in st_files:
        with safe_open(str(sf_path), framework="pt") as f:
            for key in f.keys():
                if key == "model.visual.patch_embed.proj.weight":
                    patch_embed.weight.data.copy_(f.get_tensor(key))
                elif key == "model.visual.patch_embed.proj.bias":
                    patch_embed.bias.data.copy_(f.get_tensor(key))
                elif key == "model.visual.pos_embed.weight":
                    pos_embed.weight.data.copy_(f.get_tensor(key))

    # Run patch embedding
    px = pixel_values.view(-1, 3, temporal_patch_size, patch_size, patch_size)
    with torch.no_grad():
        hidden_states = patch_embed(px.float()).view(-1, hidden_size)

    # Positional embedding (bilinear interpolation)
    num_grid = int(num_position_embeddings**0.5)
    grid_thw_list = image_grid_thw.tolist()

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, num_grid - 1, h)
        w_idxs = torch.linspace(0, num_grid - 1, w)
        h_floor, w_floor = h_idxs.int(), w_idxs.int()
        h_ceil = (h_floor + 1).clip(max=num_grid - 1)
        w_ceil = (w_floor + 1).clip(max=num_grid - 1)
        dh, dw = h_idxs - h_floor, w_idxs - w_floor

        base_h = h_floor * num_grid
        base_h_ceil = h_ceil * num_grid

        indices = [
            (base_h[None].T + w_floor[None]).flatten(),
            (base_h[None].T + w_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_floor[None]).flatten(),
            (base_h_ceil[None].T + w_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_t = torch.tensor(idx_list, dtype=torch.long)
    wt_t = torch.tensor(weight_list, dtype=torch.float32)
    with torch.no_grad():
        pe = pos_embed(idx_t) * wt_t[:, :, None]
    patch_pos = pe[0] + pe[1] + pe[2] + pe[3]

    # Permute for spatial merge ordering
    grid_ts = [r[0] for r in grid_thw_list]
    grid_hs = [r[1] for r in grid_thw_list]
    grid_ws = [r[2] for r in grid_thw_list]
    chunks = patch_pos.split([h * w for h, w in zip(grid_hs, grid_ws)])
    permuted = []
    for pos, t_val, h_val, w_val in zip(chunks, grid_ts, grid_hs, grid_ws):
        pos = pos.repeat(t_val, 1)
        mh = h_val // spatial_merge_size
        mw = w_val // spatial_merge_size
        pos = pos.view(t_val, mh, spatial_merge_size, mw, spatial_merge_size, -1)
        pos = pos.permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
        permuted.append(pos)
    pos_embeds = torch.cat(permuted)

    hidden_states = hidden_states + pos_embeds

    # Rotary position embeddings
    max_hw = max(max(h, w) for _, h, w in grid_thw_list)
    freq_table = rotary(max_hw)

    total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long)

    offset = 0
    for nf, h_val, w_val in grid_thw_list:
        mh = h_val // spatial_merge_size
        mw = w_val // spatial_merge_size
        block_rows = torch.arange(mh)
        block_cols = torch.arange(mw)
        intra_row = torch.arange(spatial_merge_size)
        intra_col = torch.arange(spatial_merge_size)
        row_idx = (
            block_rows[:, None, None, None] * spatial_merge_size
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * spatial_merge_size
            + intra_col[None, None, None, :]
        )
        row_idx = row_idx.expand(
            mh, mw, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        col_idx = col_idx.expand(
            mh, mw, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if nf > 1:
            coords = coords.repeat(nf, 1)
        n = coords.shape[0]
        pos_ids[offset : offset + n] = coords
        offset += n

    rot_emb = freq_table[pos_ids].flatten(1)
    emb = torch.cat((rot_emb, rot_emb), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    # Block-diagonal attention mask
    cu = torch.repeat_interleave(
        image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
    ).cumsum(0, dtype=torch.int32)
    cu = F.pad(cu, (1, 0), value=0)
    seq_len = hidden_states.shape[0]
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32)
    for i in range(len(cu) - 1):
        s, e = cu[i].item(), cu[i + 1].item()
        mask[s:e, s:e] = 0.0
    attention_mask = mask.unsqueeze(0).unsqueeze(0)

    return hidden_states, attention_mask, cos, sin


# ============================================================================
# Main
# ============================================================================


def main():
    model_path = os.environ.get("QWEN35_MODEL_PATH", "/mnt/models/Qwen3.5-35B-A3B")
    compiled_path = os.environ.get(
        "QWEN35_VISION_COMPILED_PATH", "/mnt/models/compiled_vision/"
    )

    with open(os.path.join(model_path, "config.json")) as f:
        full_config = json.load(f)
    vc = full_config["vision_config"]
    config = SimpleNamespace(**vc)

    print("=" * 70)
    print("Qwen3.5 MoE Vision Encoder -- Neuron Compilation & Test")
    print("=" * 70)
    print(
        f"  depth={config.depth}, hidden={config.hidden_size}, heads={config.num_heads}"
    )
    print(
        f"  intermediate={config.intermediate_size}, out_hidden={config.out_hidden_size}"
    )
    print(f"  patch={config.patch_size}, spatial_merge={config.spatial_merge_size}")

    # Step 1: Build model and load weights
    print("\n[1/5] Building vision encoder and loading weights...")
    model = VisionEncoder(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  Vision encoder parameters: {n_params:,} ({n_params * 2 / 1e9:.2f} GB in BF16)"
    )
    n_loaded = load_vision_weights(model, model_path)
    model.eval()
    model_bf16 = model.to(torch.bfloat16)

    # Step 2: Create test inputs (224x224 image -> 16x16 patches = 256 patches)
    print("\n[2/5] Preparing test inputs...")
    # Simulate a 224x224 image processed by the HF processor
    # grid_thw: 1 frame, 16 height patches, 16 width patches (after processor resizing)
    image_grid_thw = torch.tensor([[1, 16, 16]])
    total_patches = 1 * 16 * 16  # = 256 patches
    # After spatial merge (2x2): 8*8 = 64 merged tokens -> output shape (64, 2048)

    # Create dummy pixel values (normally from processor)
    pixel_values = torch.randn(
        total_patches,
        3 * config.temporal_patch_size * config.patch_size * config.patch_size,
    )

    # Preprocess on CPU
    hidden_states, attention_mask, cos, sin = preprocess_vision_inputs(
        pixel_values, image_grid_thw, config, model_path
    )
    print(f"  hidden_states: {hidden_states.shape} (seq_len={hidden_states.shape[0]})")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  cos/sin: {cos.shape}")

    # Step 3: CPU reference
    print("\n[3/5] Running CPU reference...")
    with torch.no_grad():
        cpu_output = model_bf16(
            hidden_states.to(torch.bfloat16),
            attention_mask.to(torch.bfloat16),
            cos.to(torch.bfloat16),
            sin.to(torch.bfloat16),
        )
    print(f"  CPU output shape: {cpu_output.shape}")
    print(f"  CPU output range: [{cpu_output.min():.4f}, {cpu_output.max():.4f}]")
    expected_merged = total_patches // (config.spatial_merge_size**2)
    assert cpu_output.shape == (expected_merged, config.out_hidden_size), (
        f"Expected ({expected_merged}, {config.out_hidden_size}), got {cpu_output.shape}"
    )
    print(f"  Shape verified: ({expected_merged}, {config.out_hidden_size})")

    # Step 4: Compile for Neuron
    print("\n[4/5] Compiling vision encoder for Neuron...")
    import torch_neuronx

    example_inputs = (
        hidden_states.to(torch.bfloat16),
        attention_mask.to(torch.bfloat16),
        cos.to(torch.bfloat16),
        sin.to(torch.bfloat16),
    )

    os.makedirs(compiled_path, exist_ok=True)
    compile_start = time.time()
    try:
        traced = torch_neuronx.trace(
            model_bf16,
            example_inputs,
            compiler_args=[
                "--auto-cast",
                "matmult",
                "--model-type",
                "transformer",
            ],
        )
        compile_time = time.time() - compile_start
        print(f"  Compilation succeeded in {compile_time:.1f}s")

        # Save
        save_path = os.path.join(compiled_path, "vision_encoder.pt")
        torch.jit.save(traced, save_path)
        print(f"  Saved to {save_path}")

    except Exception as e:
        compile_time = time.time() - compile_start
        print(f"  Compilation FAILED after {compile_time:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 5: Neuron inference and comparison
    print("\n[5/5] Running Neuron inference and comparing...")
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            neuron_output = traced(*example_inputs)

        # Timed run
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            neuron_output = traced(*example_inputs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    avg_latency = sum(times) / len(times)
    print(f"  Neuron output shape: {neuron_output.shape}")
    print(f"  Neuron latency: {avg_latency:.2f}ms (avg of 10 runs)")

    # Compare outputs
    cpu_fp32 = cpu_output.float()
    neuron_fp32 = neuron_output.float()

    cos_sim = F.cosine_similarity(
        cpu_fp32.flatten().unsqueeze(0), neuron_fp32.flatten().unsqueeze(0)
    ).item()
    max_diff = (cpu_fp32 - neuron_fp32).abs().max().item()
    mean_diff = (cpu_fp32 - neuron_fp32).abs().mean().item()

    print(f"\n  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")

    if cos_sim > 0.99:
        print(f"\n  PASS: Cosine similarity {cos_sim:.6f} > 0.99")
    else:
        print(
            f"\n  WARN: Cosine similarity {cos_sim:.6f} < 0.99 -- may need investigation"
        )

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Vision encoder params: {n_params:,}")
    print(f"  Compilation time: {compile_time:.1f}s")
    print(f"  Neuron latency: {avg_latency:.2f}ms")
    print(f"  Cosine sim (vs CPU): {cos_sim:.6f}")
    print(f"  Output shape: {neuron_output.shape}")
    print("=" * 70)


if __name__ == "__main__":
    main()
