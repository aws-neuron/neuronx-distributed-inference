# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5 (Dense) Vision Encoder for NeuronX Distributed Inference.

Ports the Qwen3.5 ViT encoder to run on Neuron. The vision encoder
architecture is identical across model sizes -- dimensions are read from config:
- Qwen3.5-2B: depth=24, hidden=1024, out_hidden=2048

Dimensions are read from config so this module works for any Qwen3.5 dense model size.

The vision encoder runs as a separate compiled model from the text decoder,
compiled and loaded via NeuronBaseForImageToText.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# CRITICAL: Use finite negative value instead of -inf for Neuron attention masks.
# The Neuron compiler's bfloat16 handling of -inf produces NaN that bleeds from
# padding positions into ALL positions through the transformer layers.
# -65504.0 is large enough for softmax masking but avoids NaN overflow.
_MASK_NEG_INF = -65504.0

# Threshold above which we compile the vision encoder WITHOUT an attention mask.
# At seq_len > 4096, the full (1, 1, N, N) mask tensor causes the Neuron compiler
# to exceed its 5M instruction limit (e.g., 8192x8192 mask adds ~1.7M instructions).
# Without a mask, F.scaled_dot_product_attention is decomposed into tiled flash
# attention by the compiler, which handles longer sequences.
# For buckets <= this threshold, we pass a block-diagonal mask for multi-image isolation.
_MASKLESS_THRESHOLD = 4096

# Maximum vision sequence length bucket supported by the Neuron compiler.
# The trn2.3xlarge compiler (32GB host RAM) can compile the full 24-layer ViT
# at seq=8192 in maskless mode using --tiled-inst-limit to bypass the default 5M
# instruction verifier (actual instructions: ~9.2M). This supports images up to
# ~1150x1150 pixels. seq=16384 requires >26.7M instructions which OOMs the compiler
# on a monolithic model (>247GB RAM needed). Layer-by-layer compilation supports
# seq=16384 (2048x2048) by compiling each block independently (~7 min, <3GB RAM each).
_MAX_VISION_SEQ_LEN = 16384

logger = logging.getLogger(__name__)


def get_vision_compiler_args(bucket_size):
    """Return compiler args appropriate for the given vision bucket size.

    For buckets > _MASKLESS_THRESHOLD (4096), the SDPA maskless mode generates
    more than the default 5M instruction limit. We bypass the instruction count
    checks using internal compiler flags:
      - --internal-hlo2tensorizer-options=--hlo-sanity-check=false (frontend)
      - --tensorizer-options=--inst-count-limit=N (middleend)
      - --internal-max-instruction-limit=N (backend)

    For buckets <= 4096, standard compiler args suffice (under 5M instructions).
    """
    base_args = "--auto-cast=matmult --model-type=transformer"

    if bucket_size > _MASKLESS_THRESHOLD:
        # Maskless SDPA at 8192 generates ~9.2M instructions (backend).
        # Raise all limits to 15M for safety margin.
        inst_limit = 15000000
        return (
            f"{base_args} "
            f"--internal-hlo2tensorizer-options=--hlo-sanity-check=false "
            f"--tensorizer-options=--inst-count-limit={inst_limit} "
            f"--internal-max-instruction-limit={inst_limit}"
        )
    else:
        return base_args


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

        # Scaled dot-product attention (compiler decomposes into tiled flash attention
        # when mask is None, avoiding full NxN materialization that exceeds instruction
        # limits at N=16384)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

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


class CPUVisionModel(nn.Module):
    """CPU-only vision encoder (pure PyTorch, no Neuron dependencies).

    Used when HBM is insufficient to load the vision encoder on Neuron
    alongside the text decoder (e.g., when HBM is limited).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [self._make_block(config) for _ in range(config.depth)]
        )
        self.merger_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        merger_hidden = config.hidden_size * (config.spatial_merge_size**2)
        self.merger_fc1 = nn.Linear(merger_hidden, merger_hidden)
        self.merger_act = nn.GELU()
        self.merger_fc2 = nn.Linear(merger_hidden, config.out_hidden_size)

    @staticmethod
    def _make_block(config):
        """Build a single vision block with standard nn.Linear (no TP)."""
        block = nn.Module()
        block.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        block.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # Attention
        attn = nn.Module()
        attn.hidden_size = config.hidden_size
        attn.num_heads = config.num_heads
        attn.head_dim = config.hidden_size // config.num_heads
        attn.scaling = attn.head_dim**-0.5
        attn.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        attn.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        block.attn = attn

        # MLP
        mlp = nn.Module()
        mlp.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        mlp.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )
        mlp.act_fn = nn.GELU()
        block.mlp = mlp

        return block

    def _forward_attention(self, attn, hidden_states, attention_mask, cos, sin):
        seq_len = hidden_states.shape[0]
        qkv = attn.qkv(hidden_states).reshape(seq_len, 3, attn.num_heads, attn.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)

        if cos is not None and sin is not None:
            cos_u = cos.unsqueeze(-2)
            sin_u = sin.unsqueeze(-2)

            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            q = (q * cos_u) + (rotate_half(q) * sin_u)
            k = (k * cos_u) + (rotate_half(k) * sin_u)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = attn_output.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        return attn.proj(out)

    def forward(self, hidden_states, cos, sin):
        """Forward pass without attention mask (maskless mode).

        Used for buckets > _MASKLESS_THRESHOLD (4096) where passing a full NxN
        mask causes the Neuron compiler to exceed the 5M instruction limit.
        Without a mask, F.scaled_dot_product_attention is decomposed into tiled
        flash attention that handles long sequences efficiently.

        For single-image inputs, maskless and masked (with zeros) produce
        identical results (verified: max_diff=0, cosine_sim=0.99999).
        """
        for block in self.blocks:
            hidden_states = hidden_states + self._forward_attention(
                block.attn, block.norm1(hidden_states), None, cos, sin
            )
            hidden_states = hidden_states + block.mlp.linear_fc2(
                block.mlp.act_fn(block.mlp.linear_fc1(block.norm2(hidden_states)))
            )

        hidden_states = self.merger_norm(hidden_states)
        merge_size = self.config.spatial_merge_size
        merged_hidden = self.config.hidden_size * (merge_size**2)
        hidden_states = hidden_states.view(-1, merged_hidden)
        hidden_states = self.merger_fc2(self.merger_act(self.merger_fc1(hidden_states)))
        return hidden_states

    def forward_with_mask(self, hidden_states, attention_mask, cos, sin):
        """Forward pass with attention mask (for smaller buckets with multiple images)."""
        for block in self.blocks:
            hidden_states = hidden_states + self._forward_attention(
                block.attn, block.norm1(hidden_states), attention_mask, cos, sin
            )
            hidden_states = hidden_states + block.mlp.linear_fc2(
                block.mlp.act_fn(block.mlp.linear_fc1(block.norm2(hidden_states)))
            )

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

    Supports three modes:
    1. NxDI traced model (parallel layers) -- standard NxDI compilation
    2. Pre-compiled standalone model -- loaded from torch_neuronx.trace() output
    3. CPU-only model -- for when HBM is full for the vision encoder
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
        self._cpu_model = None  # Set by load_cpu_model()
        self._layerwise_blocks = (
            None  # Set by load_layerwise() -- list of 24 block NEFFs
        )
        self._layerwise_merger = None  # Set by load_layerwise() -- merger NEFF

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
            "vision_seq_len_buckets", [256, 1024, 4096, 8192, 16384]
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

    def load_layerwise(self, layerwise_dir):
        """Load layer-by-layer compiled vision encoder.

        This mode loads 24 individually compiled transformer blocks + a merger.
        Used for seq_len=16384 (2048x2048 images) where the full 24-layer model
        cannot be compiled monolithically due to compiler memory limits (>247GB RAM
        needed). Each block compiles independently in ~7 min with <3GB RAM.

        Directory layout:
            layerwise_dir/
                vision_block_0.pt   -- compiled transformer block 0
                vision_block_1.pt   -- compiled transformer block 1
                ...
                vision_block_23.pt  -- compiled transformer block 23
                vision_merger.pt    -- compiled merger (norm + spatial merge + MLP)

        At inference time, blocks are executed sequentially on the Neuron device:
            hidden_states -> block_0 -> block_1 -> ... -> block_23 -> merger -> output

        Args:
            layerwise_dir: Path to directory containing block_*.pt and merger.pt files.
        """
        logger.info(f"Loading layer-by-layer vision encoder from {layerwise_dir}")

        if not os.path.isdir(layerwise_dir):
            raise FileNotFoundError(f"Layerwise directory not found: {layerwise_dir}")

        # Load blocks
        depth = self.vision_config.depth
        self._layerwise_blocks = []
        for i in range(depth):
            block_path = os.path.join(layerwise_dir, f"vision_block_{i}.pt")
            if not os.path.exists(block_path):
                raise FileNotFoundError(
                    f"Missing block {i}: {block_path}. "
                    f"Expected {depth} blocks (vision_block_0.pt to vision_block_{depth - 1}.pt)"
                )
            self._layerwise_blocks.append(torch.jit.load(block_path))
            logger.info(f"  Loaded block {i}/{depth - 1}")

        # Load merger
        merger_path = os.path.join(layerwise_dir, "vision_merger.pt")
        if not os.path.exists(merger_path):
            raise FileNotFoundError(f"Missing merger: {merger_path}")
        self._layerwise_merger = torch.jit.load(merger_path)
        logger.info(f"  Loaded merger")

        logger.info(
            f"Layer-by-layer vision encoder loaded: {depth} blocks + merger "
            f"from {layerwise_dir}"
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

    def load_cpu_model(self, model_path):
        """Load a CPU-only vision encoder from HF safetensors.

        Use this when HBM is insufficient for the Neuron-compiled vision encoder
        (e.g., when the text decoder fills available HBM).

        Args:
            model_path: Path to HF model directory with safetensors
        """
        from pathlib import Path
        from safetensors import safe_open

        config = self.vision_config
        cpu_model = CPUVisionModel(config)

        # Build key mapping from HF safetensors to CPU model
        key_map = {}
        for i in range(config.depth):
            hf_pre = f"model.visual.blocks.{i}"
            loc_pre = f"blocks.{i}"
            for suffix in [
                "attn.qkv.weight",
                "attn.qkv.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias",
                "mlp.linear_fc2.weight",
                "mlp.linear_fc2.bias",
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
            ]:
                key_map[f"{hf_pre}.{suffix}"] = f"{loc_pre}.{suffix}"

        key_map["model.visual.merger.norm.weight"] = "merger_norm.weight"
        key_map["model.visual.merger.norm.bias"] = "merger_norm.bias"
        key_map["model.visual.merger.linear_fc1.weight"] = "merger_fc1.weight"
        key_map["model.visual.merger.linear_fc1.bias"] = "merger_fc1.bias"
        key_map["model.visual.merger.linear_fc2.weight"] = "merger_fc2.weight"
        key_map["model.visual.merger.linear_fc2.bias"] = "merger_fc2.bias"

        st_files = sorted(Path(model_path).glob("model*.safetensors"))
        loaded = 0
        state_dict = cpu_model.state_dict()

        for sf_path in st_files:
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    if key in key_map:
                        local_key = key_map[key]
                        if local_key in state_dict:
                            state_dict[local_key].copy_(f.get_tensor(key))
                            loaded += 1

        cpu_model.load_state_dict(state_dict)
        cpu_model = cpu_model.to(torch.bfloat16).eval()
        self._cpu_model = cpu_model
        logger.info(
            f"Loaded CPU vision encoder: {loaded} weights, "
            f"{sum(p.numel() for p in cpu_model.parameters()) / 1e6:.1f}M params"
        )

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

    def _run_single_bucket(
        self, hidden_states, cos, sin, use_maskless, attention_mask=None
    ):
        """Run vision encoder on a single bucket-sized input.

        Dispatches to the appropriate backend (compiled buckets, single compiled
        model, CPU model, or NxDI traced model).

        Args:
            hidden_states: (bucket_len, hidden_size) padded to bucket size
            cos, sin: (bucket_len, head_dim) rotary embeddings
            use_maskless: Whether this bucket uses maskless mode (no NxN mask)
            attention_mask: Optional (1, 1, bucket_len, bucket_len) mask

        Returns:
            vision_output: merged token embeddings from this bucket
        """
        bucket_len = hidden_states.shape[0]

        if self._layerwise_blocks is not None and use_maskless:
            # Layer-by-layer mode for very large sequences
            h = hidden_states.to(torch.bfloat16)
            c = cos.to(torch.bfloat16)
            s = sin.to(torch.bfloat16)
            for block_model in self._layerwise_blocks:
                h = block_model(h, c, s)
            vision_output = self._layerwise_merger(h)
        elif self._compiled_buckets is not None:
            if bucket_len not in self._compiled_buckets:
                raise RuntimeError(
                    f"No compiled vision encoder for bucket size {bucket_len}. "
                    f"Available buckets: {sorted(self._compiled_buckets.keys())}. "
                    f"Input bucket_len={bucket_len}."
                )
            compiled_model = self._compiled_buckets[bucket_len]
            if use_maskless:
                vision_output = compiled_model(
                    hidden_states.to(torch.bfloat16),
                    cos.to(torch.bfloat16),
                    sin.to(torch.bfloat16),
                )
            else:
                vision_output = compiled_model(
                    hidden_states.to(torch.bfloat16),
                    attention_mask.to(torch.bfloat16),
                    cos.to(torch.bfloat16),
                    sin.to(torch.bfloat16),
                )
        elif self._compiled_model is not None:
            vision_output = self._compiled_model(
                hidden_states.to(torch.bfloat16),
                attention_mask.to(torch.bfloat16),
                cos.to(torch.bfloat16),
                sin.to(torch.bfloat16),
            )
        elif self._cpu_model is not None:
            with torch.no_grad():
                if use_maskless:
                    vision_output = self._cpu_model(
                        hidden_states.to(torch.bfloat16),
                        cos.to(torch.bfloat16),
                        sin.to(torch.bfloat16),
                    )
                else:
                    vision_output = self._cpu_model.forward_with_mask(
                        hidden_states.to(torch.bfloat16),
                        attention_mask.to(torch.bfloat16),
                        cos.to(torch.bfloat16),
                        sin.to(torch.bfloat16),
                    )
        else:
            if use_maskless:
                attention_mask = None
            vision_output = self.model(hidden_states, attention_mask, (cos, sin))

        return vision_output

    def forward(self, pixel_values, image_grid_thw):
        """Run vision encoding (CPU preprocessing + Neuron traced model).

        Supports chunked processing for images larger than the maximum compiled
        bucket. For example, a 2048x2048 image produces 16384 patches which
        exceeds the 8192 max bucket; it is processed as 2 chunks of 8192.
        A 4096x4096 image produces 65536 patches -> 8 chunks of 8192.

        Chunking is valid because:
        - Self-attention within each chunk processes patches independently
        - Rotary position embeddings are per-token (no cross-token dependency)
        - The spatial merger operates on consecutive 2x2 blocks which remain
          intact within each chunk (chunk boundaries align with merge grid)
        - Only cross-chunk attention context is lost (acceptable quality tradeoff
          since the text decoder sees ALL merged tokens with full attention)

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

        # 4. Determine if chunking is needed.
        seq_len = hidden_states.shape[0]
        max_bucket = (
            max(self.vision_seq_len_buckets) if self.vision_seq_len_buckets else seq_len
        )
        cos, sin = position_embeddings

        if seq_len > max_bucket:
            # CHUNKED PROCESSING: seq_len exceeds largest compiled bucket.
            # Split into chunks of max_bucket and process independently.
            # Each chunk must be aligned to spatial_merge_size^2 to ensure
            # the merger can correctly group 2x2 patch blocks.
            merge_area = self.vision_config.spatial_merge_size**2
            chunk_size = max_bucket
            assert chunk_size % merge_area == 0, (
                f"Chunk size {chunk_size} must be divisible by merge_area {merge_area}"
            )

            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            logger.info(
                f"Chunked vision encoding: seq_len={seq_len} > max_bucket={max_bucket}, "
                f"processing as {num_chunks} chunks of {chunk_size}"
            )

            chunk_outputs = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start

                # Extract chunk
                h_chunk = hidden_states[start:end]
                c_chunk = cos[start:end]
                s_chunk = sin[start:end]

                # Pad to bucket size if last chunk is smaller
                if chunk_len < chunk_size:
                    pad_len = chunk_size - chunk_len
                    h_chunk = F.pad(h_chunk, (0, 0, 0, pad_len))
                    c_chunk = F.pad(c_chunk, (0, 0, 0, pad_len))
                    s_chunk = F.pad(s_chunk, (0, 0, 0, pad_len))

                # Chunked mode always uses maskless (chunk_size > _MASKLESS_THRESHOLD)
                chunk_out = self._run_single_bucket(
                    h_chunk, c_chunk, s_chunk, use_maskless=True
                )

                # Unpad: keep only valid merged tokens for this chunk
                merged_per_chunk = chunk_len // merge_area
                chunk_outputs.append(chunk_out[:merged_per_chunk])

            vision_output = torch.cat(chunk_outputs, dim=0)

        else:
            # STANDARD (non-chunked) processing
            bucket_len = self._get_vision_bucket(seq_len)
            use_maskless = bucket_len > _MASKLESS_THRESHOLD

            # Build mask only for smaller buckets
            attention_mask = None
            if not use_maskless:
                attention_mask = self._build_vision_attention_mask(
                    image_grid_thw, seq_len, hidden_states.dtype
                )

            # Bucket and pad
            if seq_len < bucket_len:
                pad_len = bucket_len - seq_len
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
                cos = F.pad(cos, (0, 0, 0, pad_len))
                sin = F.pad(sin, (0, 0, 0, pad_len))
                if not use_maskless:
                    mask = torch.full(
                        (1, 1, bucket_len, bucket_len),
                        _MASK_NEG_INF,
                        dtype=hidden_states.dtype,
                    )
                    mask[:, :, :seq_len, :seq_len] = attention_mask
                    attention_mask = mask

            # Run single-pass vision encoding
            vision_output = self._run_single_bucket(
                hidden_states, cos, sin, use_maskless, attention_mask
            )

            # Unpad: only keep valid merged tokens
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
