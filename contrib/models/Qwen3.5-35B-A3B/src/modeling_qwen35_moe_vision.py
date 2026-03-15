"""
Qwen3.5-35B-A3B Vision Encoder for NeuronX Distributed Inference.

Ports the Qwen3.5 MoE ViT encoder to run on Neuron. The vision encoder
architecture is identical to Qwen3-VL (same patch embed, same rotary,
same merger) but without deepstack support.

The vision encoder runs as a separate compiled model from the text decoder,
compiled and loaded via NeuronBaseForImageToText.
"""

import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# CRITICAL: Use finite negative value instead of -inf for Neuron attention masks.
# The Neuron compiler's bfloat16 handling of -inf produces NaN that bleeds from
# padding positions into ALL positions through the transformer layers.
# -65504.0 is large enough for softmax masking but avoids NaN overflow.
_MASK_NEG_INF = -65504.0

logger = logging.getLogger(__name__)

# -- NxDI imports (available on Neuron instances) --
try:
    from neuronx_distributed_inference.models.application_base import (
        NeuronApplicationBase,
    )
    from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
    from neuronx_distributed_inference.modules.attention.attention_base import (
        NeuronAttentionBase,
    )
    from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
    from neuronx_distributed.parallel_layers import layers as nxd_layers
except ImportError:
    logger.warning(
        "NxDI imports unavailable -- vision module can only be used on Neuron instances"
    )

# -- HuggingFace imports for patch embed (runs on CPU) --
try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeVisionPatchEmbed,
        Qwen3_5MoeVisionPatchMerger,
        Qwen3_5MoeVisionRotaryEmbedding,
    )
except ImportError:
    try:
        # transformers 4.57+ uses Qwen3VL* class names
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchEmbed as Qwen3_5MoeVisionPatchEmbed,
            Qwen3VLVisionPatchMerger as Qwen3_5MoeVisionPatchMerger,
            Qwen3VLVisionRotaryEmbedding as Qwen3_5MoeVisionRotaryEmbedding,
        )
    except ImportError:
        try:
            # Older transformers uses Qwen2VL* class names
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLVisionPatchEmbed as Qwen3_5MoeVisionPatchEmbed,
                Qwen2VLVisionPatchMerger as Qwen3_5MoeVisionPatchMerger,
                Qwen2VLVisionRotaryEmbedding as Qwen3_5MoeVisionRotaryEmbedding,
            )
        except ImportError:
            Qwen3_5MoeVisionPatchEmbed = None
            Qwen3_5MoeVisionPatchMerger = None
            Qwen3_5MoeVisionRotaryEmbedding = None


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Apply rotary position embeddings to vision Q and K tensors.

    Uses rotate_half style (matching HF reference):
      q_embed = (q * cos) + (rotate_half(q) * sin)

    Args:
        q: (seq_len, num_heads, head_dim)
        k: (seq_len, num_heads, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    cos = cos.unsqueeze(-2)  # (seq_len, 1, head_dim)
    sin = sin.unsqueeze(-2)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class NeuronQwen35VisionAttention(nn.Module):
    """Vision attention for Qwen3.5 MoE.

    Uses fused QKV linear (no bias in Neuron port for efficiency).
    Non-causal attention with block-diagonal mask for variable-length images.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5

        # Fused QKV: (hidden_size -> 3 * hidden_size) with bias
        self.qkv = nxd_layers.ColumnParallelLinear(
            self.hidden_size,
            3 * self.hidden_size,
            bias=True,
            gather_output=True,
        )
        self.proj = nxd_layers.RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            input_is_parallel=False,
        )

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        """
        Args:
            hidden_states: (seq_len, hidden_size)
            attention_mask: (1, 1, seq_len, seq_len) block-diagonal mask
            position_embeddings: (cos, sin) tuple
        """
        seq_len = hidden_states.shape[0]

        # QKV projection
        qkv = self.qkv(hidden_states)  # (seq_len, 3 * hidden_size)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # (3, seq_len, num_heads, head_dim)
        q, k, v = qkv.unbind(0)  # each (seq_len, num_heads, head_dim)

        # Apply rotary embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Reshape for batched attention: (1, num_heads, seq_len, head_dim)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (seq_len, hidden_size)
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_len, -1)

        # Output projection
        attn_output = self.proj(attn_output)
        return attn_output


class NeuronQwen35VisionMLP(nn.Module):
    """Vision MLP with GELU activation."""

    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nxd_layers.ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=True,
        )
        self.linear_fc2 = nxd_layers.RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=False,
        )
        self.act_fn = nn.GELU()

    def forward(self, hidden_states):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class NeuronQwen35VisionBlock(nn.Module):
    """Single vision transformer block: LayerNorm + Attention + LayerNorm + MLP."""

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = NeuronQwen35VisionAttention(config)
        self.mlp = NeuronQwen35VisionMLP(config)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class NeuronQwen35VisionModel(nn.Module):
    """Qwen3.5 MoE Vision Encoder for Neuron.

    This is the nn.Module that gets compiled and traced onto Neuron.
    Patch embedding, positional embedding, and rotary embedding are computed
    on CPU in the ModelWrapper and passed as inputs.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [NeuronQwen35VisionBlock(config) for _ in range(config.depth)]
        )
        # Merger: spatial_merge_size^2 * hidden_size -> out_hidden_size
        self.merger_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        merger_hidden = config.hidden_size * (config.spatial_merge_size**2)
        self.merger_fc1 = nn.Linear(merger_hidden, merger_hidden)
        self.merger_act = nn.GELU()
        self.merger_fc2 = nn.Linear(merger_hidden, config.out_hidden_size)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        """
        Args:
            hidden_states: (seq_len, hidden_size) -- after patch_embed + pos_embed
            attention_mask: (1, 1, seq_len, seq_len) block-diagonal mask
            position_embeddings: (cos, sin) tuple for rotary

        Returns:
            vision_embeddings: (merged_seq_len, out_hidden_size)
        """
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        # Apply merger: norm -> spatial merge -> fc1 -> gelu -> fc2
        hidden_states = self.merger_norm(hidden_states)
        merge_size = self.config.spatial_merge_size
        merged_hidden = self.config.hidden_size * (merge_size**2)
        hidden_states = hidden_states.view(-1, merged_hidden)
        hidden_states = self.merger_fc2(self.merger_act(self.merger_fc1(hidden_states)))

        return hidden_states


class NeuronQwen35VisionModelWrapper(ModelWrapper):
    """Wraps the vision encoder for NxDI tracing.

    Handles CPU-side operations that cannot be traced:
    - Patch embedding (Conv3d)
    - Positional embedding (Embedding + bilinear interpolation)
    - Rotary position embedding computation
    - Vision attention mask construction (block-diagonal)
    - Sequence length bucketing and padding/unpadding

    Supports two modes:
    1. NxDI traced model (parallel layers) -- standard NxDI compilation
    2. Pre-compiled standalone model -- loaded from torch_neuronx.trace() output
    """

    def __init__(self, config, model_cls=None, **kwargs):
        if model_cls is not None:
            super().__init__(config, model_cls, **kwargs)
        else:
            # Standalone mode: no NxDI model_cls
            nn.Module.__init__(self)
        self.vision_config = config
        self._compiled_model = None  # Set by load_compiled() -- single bucket
        self._compiled_buckets = None  # Set by load_compiled() -- multi-bucket dict

        # These HF modules run on CPU, outside the traced graph
        if Qwen3_5MoeVisionPatchEmbed is not None:
            self.patch_embed = Qwen3_5MoeVisionPatchEmbed(config)
            self.pos_embed = nn.Embedding(
                config.num_position_embeddings, config.hidden_size
            )
            self.num_grid_per_side = int(config.num_position_embeddings**0.5)
            head_dim = config.hidden_size // config.num_heads
            self.rotary_pos_emb = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2)
        else:
            logger.warning("HF Qwen3.5 MoE vision classes not available")

        self.vision_seq_len_buckets = kwargs.get(
            "vision_seq_len_buckets", [1024, 4096, 16384]
        )

    def load_compiled(self, compiled_model_path):
        """Load pre-compiled standalone vision encoder(s).

        Supports two modes:
        1. Single .pt file: Legacy mode, loads one compiled model for one bucket size.
        2. Directory with multiple .pt files: Multi-bucket mode. Files must be named
           'vision_encoder_{bucket_size}.pt' (e.g., 'vision_encoder_256.pt').
           Falls back to single 'vision_encoder.pt' in the directory.

        Args:
            compiled_model_path: Path to a .pt file or directory containing bucket .pt files.
        """
        import glob as glob_module

        logger.info(f"Loading pre-compiled vision encoder from {compiled_model_path}")

        if os.path.isfile(compiled_model_path):
            # Single file mode (legacy)
            self._compiled_model = torch.jit.load(compiled_model_path)
            self._compiled_buckets = None
            logger.info("Vision encoder loaded successfully (single bucket)")
        elif os.path.isdir(compiled_model_path):
            # Directory mode: look for bucket-specific files
            bucket_files = sorted(
                glob_module.glob(
                    os.path.join(compiled_model_path, "vision_encoder_*.pt")
                )
            )
            if bucket_files:
                self._compiled_buckets = {}
                for bf in bucket_files:
                    # Extract bucket size from filename: vision_encoder_256.pt -> 256
                    basename = os.path.basename(bf)
                    try:
                        bucket_size = int(
                            basename.replace("vision_encoder_", "").replace(".pt", "")
                        )
                        self._compiled_buckets[bucket_size] = torch.jit.load(bf)
                        logger.info(f"  Loaded vision bucket {bucket_size} from {bf}")
                    except ValueError:
                        logger.warning(f"  Skipping unrecognized file: {bf}")
                self._compiled_model = None
                # Update vision_seq_len_buckets to match compiled buckets
                self.vision_seq_len_buckets = sorted(self._compiled_buckets.keys())
                logger.info(
                    f"Vision encoder loaded with {len(self._compiled_buckets)} buckets: "
                    f"{self.vision_seq_len_buckets}"
                )
            else:
                # Fall back to single vision_encoder.pt in directory
                single_path = os.path.join(compiled_model_path, "vision_encoder.pt")
                if os.path.exists(single_path):
                    self._compiled_model = torch.jit.load(single_path)
                    self._compiled_buckets = None
                    logger.info(
                        "Vision encoder loaded successfully (single file in dir)"
                    )
                else:
                    raise FileNotFoundError(
                        f"No vision encoder files found in {compiled_model_path}"
                    )
        else:
            raise FileNotFoundError(
                f"Vision encoder path not found: {compiled_model_path}"
            )

    def load_vision_weights_from_hf(self, model_path):
        """Load patch_embed and pos_embed weights from HF safetensors.

        Args:
            model_path: Path to HF model directory
        """
        from pathlib import Path
        from safetensors import safe_open

        st_files = sorted(
            p
            for p in Path(model_path).glob("*.safetensors")
            if p.suffix == ".safetensors"
        )
        loaded = 0
        for sf_path in st_files:
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    if key == "model.visual.patch_embed.proj.weight":
                        self.patch_embed.proj.weight.data.copy_(f.get_tensor(key))
                        loaded += 1
                    elif key == "model.visual.patch_embed.proj.bias":
                        self.patch_embed.proj.bias.data.copy_(f.get_tensor(key))
                        loaded += 1
                    elif key == "model.visual.pos_embed.weight":
                        self.pos_embed.weight.data.copy_(f.get_tensor(key))
                        loaded += 1
        logger.info(f"Loaded {loaded} CPU-side vision weight tensors from HF")

    def _get_vision_bucket(self, seq_len):
        """Find the smallest bucket that fits the sequence length."""
        for bucket in sorted(self.vision_seq_len_buckets):
            if seq_len <= bucket:
                return bucket
        return self.vision_seq_len_buckets[-1]

    def rot_pos_emb(self, grid_thw):
        """Compute rotary positional embeddings for vision tokens.

        Returns: (total_tokens, head_dim) tensor of rotary frequencies.
        """
        merge_size = self.vision_config.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )

            row_idx = row_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        """Bilinear interpolation of positional embeddings for variable resolution."""
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
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

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        merge_size = self.vision_config.spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return torch.cat(patch_pos_embeds_permute)

    def _build_vision_attention_mask(self, grid_thw, seq_len, dtype):
        """Build block-diagonal attention mask for variable-length images.

        Each image gets its own attention block (no cross-image attention).
        """
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Build block-diagonal mask
        mask = torch.full((seq_len, seq_len), _MASK_NEG_INF, dtype=dtype)
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            mask[start:end, start:end] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self, pixel_values, image_grid_thw):
        """Run vision encoding (CPU preprocessing + Neuron traced model).

        Args:
            pixel_values: Raw pixel values from HF processor
            image_grid_thw: (num_images, 3) -- temporal, height, width in patches

        Returns:
            vision_embeddings: (total_merged_tokens, out_hidden_size)
        """
        # 1. Patch embedding (CPU, Conv3d)
        hidden_states = self.patch_embed(pixel_values)

        # 2. Positional embedding (CPU, bilinear interpolation)
        pos_embeds = self.fast_pos_embed_interpolate(image_grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 3. Rotary position embeddings (CPU)
        rotary_pos_emb = self.rot_pos_emb(image_grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # 4. Vision attention mask (block-diagonal)
        seq_len = hidden_states.shape[0]
        attention_mask = self._build_vision_attention_mask(
            image_grid_thw, seq_len, hidden_states.dtype
        )

        # 5. Bucket and pad for Neuron compilation
        bucket_len = self._get_vision_bucket(seq_len)
        cos, sin = position_embeddings
        if seq_len < bucket_len:
            pad_len = bucket_len - seq_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            cos = F.pad(cos, (0, 0, 0, pad_len))
            sin = F.pad(sin, (0, 0, 0, pad_len))
            # Extend mask with _MASK_NEG_INF for padded positions (NOT -inf, which causes NaN on Neuron)
            mask = torch.full(
                (1, 1, bucket_len, bucket_len), _MASK_NEG_INF, dtype=hidden_states.dtype
            )
            mask[:, :, :seq_len, :seq_len] = attention_mask
            attention_mask = mask

        # 6. Run vision model on Neuron
        if self._compiled_buckets is not None:
            # Multi-bucket mode: select the compiled model for this bucket
            if bucket_len not in self._compiled_buckets:
                raise RuntimeError(
                    f"No compiled vision encoder for bucket size {bucket_len}. "
                    f"Available buckets: {sorted(self._compiled_buckets.keys())}. "
                    f"Input seq_len={seq_len} requires bucket {bucket_len}."
                )
            compiled_model = self._compiled_buckets[bucket_len]
            vision_output = compiled_model(
                hidden_states.to(torch.bfloat16),
                attention_mask.to(torch.bfloat16),
                cos.to(torch.bfloat16),
                sin.to(torch.bfloat16),
            )
        elif self._compiled_model is not None:
            # Single compiled model (legacy)
            vision_output = self._compiled_model(
                hidden_states.to(torch.bfloat16),
                attention_mask.to(torch.bfloat16),
                cos.to(torch.bfloat16),
                sin.to(torch.bfloat16),
            )
        else:
            # NxDI traced model: takes (hidden_states, attention_mask, position_embeddings)
            vision_output = self.model(hidden_states, attention_mask, (cos, sin))

        # 7. Unpad: only keep valid merged tokens
        merge_area = self.vision_config.spatial_merge_size**2
        total_merged_tokens = sum(
            t
            * (h // self.vision_config.spatial_merge_size)
            * (w // self.vision_config.spatial_merge_size)
            for t, h, w in image_grid_thw.tolist()
        )
        vision_output = vision_output[:total_merged_tokens]

        return vision_output


class NeuronQwen35VisionForImageEncoding(NeuronApplicationBase):
    """Standalone application class for vision encoding (for testing)."""

    model_cls = NeuronQwen35VisionModel
    model_wrapper_cls = NeuronQwen35VisionModelWrapper

    @staticmethod
    def prepare_input_args(image_path, processor):
        """Prepare vision inputs from an image path.

        Args:
            image_path: Path to image file
            processor: HF AutoProcessor

        Returns:
            pixel_values, image_grid_thw
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        return inputs["pixel_values"], inputs["image_grid_thw"]
