# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InternVL3-8B-Instruct: Vision encoder (InternViT-300M) for NxDI.

Architecture:
  - InternViT-300M-448px-V2.5
  - 24 layers, hidden_size=1024, 16 heads, head_dim=64
  - Patch size 14x14, image 448x448 -> 1024 patches + CLS = 1025 tokens
  - LayerNorm, GELU, LayerScale (ls1, ls2)
  - Fused QKV: attn.qkv [3072, 1024]
  - Position: absolute learned embeddings [1, 1025, 1024]

Full vision pipeline:
  1. Patch embedding + CLS + position -> 24 transformer layers -> [B, 1025, 1024]
  2. Strip CLS token: [B, 1024, 1024]
  3. Pixel shuffle (0.5x): [B, 256, 4096]
  4. Projector MLP: LayerNorm(4096) -> Linear(4096,3584) -> GELU -> Linear(3584,3584)
  5. Pad to text seq_len: [1, seq_len, 3584]

Weight key mapping (HF -> NxDI vision):
  vision_model.embeddings.class_embedding -> encoder.class_embedding
  vision_model.embeddings.patch_embedding.{weight,bias} -> encoder.patch_embedding.{weight,bias}
  vision_model.embeddings.position_embedding -> encoder.position_embedding
  vision_model.encoder.layers.{i}.* -> encoder.layers.{i}.*
  mlp1.0.{weight,bias} -> proj_norm.{weight,bias}
  mlp1.1.{weight,bias} -> proj_linear1.{weight,bias}
  mlp1.3.{weight,bias} -> proj_linear2.{weight,bias}
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import (
    EncoderModelInstance,
    ModelWrapper,
)

# InternVL3 vision constants
VISION_HIDDEN_SIZE = 1024
VISION_NUM_HEADS = 16
VISION_NUM_LAYERS = 24
VISION_INTERMEDIATE_SIZE = 4096
VISION_PATCH_SIZE = 14
VISION_IMAGE_SIZE = 448
VISION_NUM_PATCHES = (VISION_IMAGE_SIZE // VISION_PATCH_SIZE) ** 2  # 1024
VISION_NUM_OUTPUT_TOKENS = 256  # after pixel shuffle (0.5x): 1024 / 4
TEXT_HIDDEN_SIZE = 3584
DOWNSAMPLE_RATIO = 0.5


# ---------------------------------------------------------------------------
# Vision encoder components (pure PyTorch, for tracing)
# ---------------------------------------------------------------------------


class InternVisionAttention(nn.Module):
    """InternViT attention with fused QKV projection."""

    def __init__(self, hidden_size=VISION_HIDDEN_SIZE, num_heads=VISION_NUM_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, seq, head_dim]
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention (no causal mask for vision)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, heads, seq, head_dim]

        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.proj(out)


class InternVisionMLP(nn.Module):
    """InternViT MLP: fc1 -> GELU -> fc2."""

    def __init__(
        self, hidden_size=VISION_HIDDEN_SIZE, intermediate_size=VISION_INTERMEDIATE_SIZE
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class InternVisionLayer(nn.Module):
    """InternViT transformer layer with pre-norm and LayerScale."""

    def __init__(
        self,
        hidden_size=VISION_HIDDEN_SIZE,
        num_heads=VISION_NUM_HEADS,
        intermediate_size=VISION_INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = InternVisionAttention(hidden_size, num_heads)
        self.ls1 = nn.Parameter(torch.ones(hidden_size) * 0.1)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = InternVisionMLP(hidden_size, intermediate_size)
        self.ls2 = nn.Parameter(torch.ones(hidden_size) * 0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.ls1 * self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.ls2 * self.mlp(self.norm2(hidden_states))
        return hidden_states


class InternVisionEncoder(nn.Module):
    """
    InternViT-300M vision encoder.

    Takes pixel_values [B, 3, 448, 448], returns [B, 1025, 1024] (with CLS).
    """

    def __init__(
        self,
        hidden_size=VISION_HIDDEN_SIZE,
        num_heads=VISION_NUM_HEADS,
        num_layers=VISION_NUM_LAYERS,
        intermediate_size=VISION_INTERMEDIATE_SIZE,
        patch_size=VISION_PATCH_SIZE,
        image_size=VISION_IMAGE_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_size)
        )

        self.layers = nn.ModuleList(
            [
                InternVisionLayer(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch = pixel_values.shape[0]
        # Cast pixel_values to match Conv2d weight dtype to avoid
        # "Input type (BFloat16) and bias type (float) should be the same" error
        # when HF weights are loaded in float32 but input arrives in bf16.
        # The compiler's --auto-cast=matmult handles the actual mixed-precision math.
        pixel_values = pixel_values.to(dtype=self.patch_embedding.weight.dtype)
        patches = self.patch_embedding(pixel_values)
        patches = patches.flatten(2).transpose(1, 2)
        cls_tokens = self.class_embedding.expand(batch, -1, -1)
        hidden_states = torch.cat([cls_tokens, patches], dim=1)
        hidden_states = hidden_states + self.position_embedding
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Full vision pipeline (encoder + pixel shuffle + projector + padding)
# ---------------------------------------------------------------------------


class NeuronInternVL3VisionModel(nn.Module):
    """
    Full InternVL3 vision pipeline for NxDI tracing.

    Input: pixel_values [B, 3, 448, 448]
    Output: [1, text_seq_len, text_hidden_size] (padded for scatter_by_index_put)

    Pipeline:
      1. InternViT encoder -> [B, 1025, 1024]
      2. Strip CLS token -> [B, 1024, 1024]
      3. Pixel shuffle (0.5x) -> [B, 256, 4096]
      4. Projector MLP: LayerNorm -> Linear -> GELU -> Linear -> [B, 256, 3584]
      5. Pad to text seq_len -> [1, seq_len, 3584]
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config

        # Extract text seq_len for output padding
        self.text_seq_len = config.text_config.neuron_config.seq_len
        self.text_hidden_size = config.text_config.hidden_size
        self.text_dtype = config.text_config.neuron_config.torch_dtype

        # Vision encoder
        self.encoder = InternVisionEncoder(
            hidden_size=VISION_HIDDEN_SIZE,
            num_heads=VISION_NUM_HEADS,
            num_layers=VISION_NUM_LAYERS,
            intermediate_size=VISION_INTERMEDIATE_SIZE,
            patch_size=VISION_PATCH_SIZE,
            image_size=VISION_IMAGE_SIZE,
        )

        # Projector MLP (pixel_shuffle output -> text hidden size)
        proj_input_dim = int(VISION_HIDDEN_SIZE * (1.0 / DOWNSAMPLE_RATIO) ** 2)  # 4096
        self.proj_norm = nn.LayerNorm(proj_input_dim)  # mlp1.0
        self.proj_linear1 = nn.Linear(
            proj_input_dim, TEXT_HIDDEN_SIZE, bias=True
        )  # mlp1.1
        # mlp1.2 is GELU (no weights)
        self.proj_linear2 = nn.Linear(
            TEXT_HIDDEN_SIZE, TEXT_HIDDEN_SIZE, bias=True
        )  # mlp1.3

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pixel shuffle downsampling (downsample_ratio=0.5).

        Input: [B, H*W, C] where H=W=32, C=1024
        Output: [B, (H/2)*(W/2), C*4] = [B, 256, 4096]
        """
        batch, n_patches, channels = x.shape
        h = w = int(math.sqrt(n_patches))

        x = x.reshape(batch, h, w, channels)

        ratio = DOWNSAMPLE_RATIO
        x = x.reshape(batch, h, int(w * ratio), int(channels / ratio))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(
            batch, int(h * ratio), int(w * ratio), int(channels / (ratio * ratio))
        )
        x = x.permute(0, 2, 1, 3).contiguous()

        return x.reshape(batch, -1, x.shape[-1])

    def pad_to_text_seq_len(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pad vision embeddings to text model's sequence length.

        Input: [B, 256, 3584]
        Output: [1, text_seq_len, 3584] (zero-padded, batch=1)
        """
        hidden_states = hidden_states.to(self.text_dtype)
        batch, n_tokens, hidden_size = hidden_states.shape

        # Pad sequence dimension to text seq_len
        if n_tokens < self.text_seq_len:
            pad = torch.zeros(
                batch,
                self.text_seq_len - n_tokens,
                hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            hidden_states = torch.cat([hidden_states, pad], dim=1)

        # Reshape to [1, seq_len, hidden] (batch=1 for scatter)
        hidden_states = hidden_states.view(-1, hidden_size).unsqueeze(0)
        return hidden_states

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, 3, 448, 448]

        Returns:
            vision_embeddings: [1, text_seq_len, text_hidden_size] padded
        """
        # Encoder: [B, 1025, 1024]
        hidden_states = self.encoder(pixel_values)

        # Strip CLS: [B, 1024, 1024]
        hidden_states = hidden_states[:, 1:, :]

        # Pixel shuffle: [B, 256, 4096]
        hidden_states = self.pixel_shuffle(hidden_states)

        # Projector MLP: [B, 256, 3584]
        hidden_states = self.proj_norm(hidden_states)
        hidden_states = F.gelu(self.proj_linear1(hidden_states))
        hidden_states = self.proj_linear2(hidden_states)

        # Pad to text seq_len: [1, seq_len, 3584]
        hidden_states = self.pad_to_text_seq_len(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Vision model wrapper (for NxDI tracing)
# ---------------------------------------------------------------------------


class InternVL3VisionModelWrapper(ModelWrapper):
    """
    Wrapper for tracing the InternVL3 vision encoder on Neuron.

    Uses EncoderModelInstance (no KV cache).
    Vision buckets represent number of images (always 1 for InternVL3).
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        pipeline_execution=True,
        return_ranked_to_cpu=False,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """
        Generate example inputs for vision encoder tracing.

        InternVL3 processes one 448x448 image at a time (no dynamic patching
        at this stage). Single bucket with batch=1.
        """
        inputs = []
        # Vision buckets = [1] (single image)
        for bucket in self.config.vision_config.neuron_config.buckets:
            pixel_values = torch.ones(
                [bucket, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE],
                dtype=self.config.vision_config.neuron_config.torch_dtype,
            )
            inputs.append((pixel_values,))
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run vision encoder on Neuron.

        Args:
            pixel_values: [batch, 3, 448, 448]

        Returns:
            vision_embeddings: [1, text_seq_len, text_hidden_size]
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() first."
            )
        output = self._forward(pixel_values)
        return output


# ---------------------------------------------------------------------------
# Vision weight conversion
# ---------------------------------------------------------------------------


def convert_vision_hf_to_neuron_state_dict(state_dict: dict) -> dict:
    """
    Convert InternVL3 vision + projector weights from HF to NxDI format.

    HF keys:
      vision_model.embeddings.class_embedding
      vision_model.embeddings.patch_embedding.{weight,bias}
      vision_model.embeddings.position_embedding
      vision_model.encoder.layers.{i}.attn.qkv.{weight,bias}
      vision_model.encoder.layers.{i}.attn.proj.{weight,bias}
      vision_model.encoder.layers.{i}.ls1
      vision_model.encoder.layers.{i}.ls2
      vision_model.encoder.layers.{i}.norm1.{weight,bias}
      vision_model.encoder.layers.{i}.norm2.{weight,bias}
      vision_model.encoder.layers.{i}.mlp.fc1.{weight,bias}
      vision_model.encoder.layers.{i}.mlp.fc2.{weight,bias}
      mlp1.0.{weight,bias}  -> proj_norm
      mlp1.1.{weight,bias}  -> proj_linear1
      mlp1.3.{weight,bias}  -> proj_linear2

    NxDI keys:
      encoder.class_embedding
      encoder.patch_embedding.{weight,bias}
      encoder.position_embedding
      encoder.layers.{i}.*
      proj_norm.{weight,bias}
      proj_linear1.{weight,bias}
      proj_linear2.{weight,bias}
    """
    neuron_state_dict = {}

    # Projector key mapping
    PROJECTOR_MAP = {
        "mlp1.0.weight": "proj_norm.weight",
        "mlp1.0.bias": "proj_norm.bias",
        "mlp1.1.weight": "proj_linear1.weight",
        "mlp1.1.bias": "proj_linear1.bias",
        "mlp1.3.weight": "proj_linear2.weight",
        "mlp1.3.bias": "proj_linear2.bias",
    }

    for key, tensor in state_dict.items():
        # Projector weights
        if key in PROJECTOR_MAP:
            neuron_state_dict[PROJECTOR_MAP[key]] = tensor.detach().clone()
            continue

        # Vision encoder embeddings
        if key.startswith("vision_model.embeddings."):
            suffix = key[len("vision_model.embeddings.") :]
            neuron_state_dict[f"encoder.{suffix}"] = tensor.detach().clone()
            continue

        # Vision encoder layers
        if key.startswith("vision_model.encoder.layers."):
            suffix = key[len("vision_model.encoder.") :]
            neuron_state_dict[f"encoder.{suffix}"] = tensor.detach().clone()
            continue

        # Skip other vision_model keys and all non-vision keys
        # (text weights handled by text conversion)

    return neuron_state_dict
