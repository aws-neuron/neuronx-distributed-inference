# coding=utf-8
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Full-Neuron GitForCausalLM port for NeuronX Distributed Inference (NXDI).

All components run on Neuron:
- Vision encoder: CLIP ViT-B/16 (12 layers, 768 hidden, 12 heads)
- Visual projection: Linear + LayerNorm
- Text decoder: BERT-style causal LM with 6 layers

Architecture follows the LLaVA/Pixtral pattern using NeuronBaseForImageToText.

HF model: microsoft/git-base
"""

import copy
import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

import neuronx_distributed_inference.modules.autobucketing as autobucketing
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import ImageToTextModelWrapper
from neuronx_distributed_inference.models.model_wrapper import (
    VISION_ENCODER_MODEL_TAG,
    EncoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.padding import pad_tensor, unpad_tensor
from neuronx_distributed_inference.utils.distributed import get_tp_group

logger = logging.getLogger("Neuron")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class GitVisionInferenceConfig(ImageToTextInferenceConfig):
    """
    Configuration for full-Neuron Git inference (vision + text).

    Expects a HF config.json with:
    - Top-level text decoder params (hidden_size, num_hidden_layers, etc.)
    - vision_config dict (CLIP ViT parameters)
    """

    def __init__(
        self,
        text_neuron_config,
        vision_neuron_config,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.vocab_size",
            "text_config.max_position_embeddings",
            "vision_config.hidden_size",
            "vision_config.num_hidden_layers",
            "vision_config.num_attention_heads",
            "vision_config.image_size",
            "vision_config.patch_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        text_neuron_config: NeuronConfig = None,
        vision_neuron_config: NeuronConfig = None,
        **kwargs,
    ):
        """Load Git config from pretrained model directory."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            raw = json.load(f)

        # Vision config (CLIP ViT-B/16 defaults)
        raw_vision = raw.get("vision_config", {})
        vision_config = {
            "hidden_size": raw_vision.get("hidden_size", 768),
            "intermediate_size": raw_vision.get("intermediate_size", 3072),
            "num_hidden_layers": raw_vision.get("num_hidden_layers", 12),
            "num_attention_heads": raw_vision.get("num_attention_heads", 12),
            "image_size": raw_vision.get("image_size", 224),
            "patch_size": raw_vision.get("patch_size", 16),
            "num_channels": raw_vision.get("num_channels", 3),
            "hidden_act": raw_vision.get("hidden_act", "quick_gelu"),
            "layer_norm_eps": raw_vision.get("layer_norm_eps", 1e-5),
        }

        # Derived: number of patches including CLS token
        num_patches = (vision_config["image_size"] // vision_config["patch_size"]) ** 2
        vision_config["num_positions"] = num_patches + 1  # +1 for CLS token

        # Text config (BERT-style decoder)
        text_config = {
            "hidden_size": raw.get("hidden_size", 768),
            "num_attention_heads": raw.get("num_attention_heads", 12),
            "num_hidden_layers": raw.get("num_hidden_layers", 6),
            "num_key_value_heads": raw.get("num_attention_heads", 12),  # MHA
            "vocab_size": raw.get("vocab_size", 30522),
            "max_position_embeddings": raw.get("max_position_embeddings", 1024),
            "intermediate_size": raw.get("intermediate_size", 3072),
            "layer_norm_eps": raw.get("layer_norm_eps", 1e-12),
            "hidden_act": raw.get("hidden_act", "gelu"),
            "pad_token_id": raw.get("pad_token_id", 0),
            "bos_token_id": raw.get("bos_token_id", 101),
            "eos_token_id": raw.get("eos_token_id", 102),
            "tie_word_embeddings": raw.get("tie_word_embeddings", False),
            "output_attentions": False,
            "output_hidden_states": False,
            "use_cache": True,
        }

        # Number of vision tokens (for input construction)
        num_image_tokens = num_patches + 1  # patches + CLS

        merged = {
            "_name_or_path": model_path,
            "text_config": text_config,
            "vision_config": vision_config,
            "num_image_tokens": num_image_tokens,
        }
        merged.update(kwargs)

        return cls(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            **merged,
        )


# ---------------------------------------------------------------------------
# Vision encoder components (CLIP ViT on Neuron)
# ---------------------------------------------------------------------------

class NeuronGitVisionAttention(NeuronAttentionBase):
    """CLIP self-attention for Git's vision encoder.

    Bidirectional attention with bias on Q/K/V/O, no RoPE.
    """

    def __init__(self, config):
        head_dim = config.hidden_size // config.num_attention_heads
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            head_dim=head_dim,
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            sequence_parallel_enabled=False,
            qkv_bias=True,
            o_bias=True,
        )


class NeuronGitVisionMLP(nn.Module):
    """CLIP MLP: Linear -> activation -> Linear."""

    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class NeuronGitVisionEncoderLayer(nn.Module):
    """CLIP encoder layer: LN -> Attn -> residual -> LN -> MLP -> residual."""

    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = NeuronGitVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = NeuronGitVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=None,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NeuronGitVisionEncoder(nn.Module):
    """Stack of CLIP encoder layers."""

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [NeuronGitVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class NeuronGitVisualProjection(nn.Module):
    """Git's visual projection: Linear + LayerNorm.

    Maps vision hidden_size -> text hidden_size.
    Unlike LLaVA's 2-layer MLP, Git uses a single linear + LN.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        vision_hidden_size = config.vision_config.hidden_size
        text_hidden_size = config.text_config.hidden_size
        vision_ln_eps = config.vision_config.layer_norm_eps

        self.dense = ColumnParallelLinear(
            vision_hidden_size,
            text_hidden_size,
            bias=True,
            gather_output=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.layer_norm = nn.LayerNorm(text_hidden_size, eps=vision_ln_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class NeuronGitCLIPVisionModel(nn.Module):
    """
    Full CLIP ViT vision model + projection for Neuron.

    Takes flattened patch embeddings (unfolded from pixels on CPU) and runs
    through the CLIP transformer + Git projection, outputting embeddings
    in text hidden_size space.

    CLIP ViT uses learned absolute position embeddings (not RoPE), so we
    store position_embedding and class_embedding as parameters and apply
    them here rather than passing position_ids through to attention.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config

        # Patch embedding linear (replaces Conv2d which runs on CPU)
        # Input: [BS, seq_len, C * patch_size * patch_size]
        # Output: [BS, seq_len, vision_hidden_size]
        self.patch_linear = ColumnParallelLinear(
            self.vision_config.num_channels
            * self.vision_config.patch_size
            * self.vision_config.patch_size,
            self.vision_config.hidden_size,
            bias=False,
            gather_output=True,
            dtype=self.vision_config.neuron_config.torch_dtype,
        )

        # Learned CLS token embedding [hidden_size]
        self.class_embedding = nn.Parameter(
            torch.randn(self.vision_config.hidden_size,
                        dtype=self.vision_config.neuron_config.torch_dtype)
        )

        # Learned absolute position embeddings [num_positions, hidden_size]
        num_positions = self.vision_config.num_positions
        self.position_embedding = nn.Embedding(
            num_positions,
            self.vision_config.hidden_size,
        )

        # Register fixed position IDs as buffer so they are part of the traced graph
        self.register_buffer(
            "position_ids",
            torch.arange(num_positions, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

        # Pre-LayerNorm (CLIP applies LN before the transformer)
        self.pre_layernorm = nn.LayerNorm(
            self.vision_config.hidden_size,
            eps=self.vision_config.layer_norm_eps,
        )

        # Transformer encoder
        self.encoder = NeuronGitVisionEncoder(self.vision_config)

        # Post-LayerNorm (CLIP applies LN after the transformer)
        self.post_layernorm = nn.LayerNorm(
            self.vision_config.hidden_size,
            eps=self.vision_config.layer_norm_eps,
        )

        # Git's visual projection: Linear + LayerNorm
        self.visual_projection = NeuronGitVisualProjection(self.config)

    def forward(
        self,
        patch_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patch_embeds: [BS, seq_len, C*P*P] flattened patch pixels
            attention_mask: [BS, 1, seq_len, seq_len] attention mask

        Returns:
            projected_embeds: [BS, seq_len, text_hidden_size]
        """
        # Linear projection of patches (replaces Conv2d)
        hidden_states = self.patch_linear(patch_embeds)

        # Add CLS embedding to the first position (slot 0 was zeros from CPU).
        # Use functional approach to avoid in-place modification on a view.
        cls_emb = self.class_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        cls_emb_padded = torch.nn.functional.pad(
            cls_emb, (0, 0, 0, hidden_states.shape[1] - 1), value=0.0
        )  # [1, seq_len, hidden_size] with CLS at position 0
        hidden_states = hidden_states + cls_emb_padded

        # Add learned position embeddings using registered buffer.
        # position_ids is [1, num_positions]; position_embedding output
        # broadcasts over the batch dimension.
        hidden_states = hidden_states + self.position_embedding(self.position_ids)

        # Pre-LayerNorm
        hidden_states = self.pre_layernorm(hidden_states)

        # Run through CLIP transformer (no position_ids -- CLIP uses
        # absolute position embeddings applied above, not RoPE in attention)
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
        )

        # Post-LayerNorm
        hidden_states = self.post_layernorm(hidden_states)

        # Project to text hidden size
        projected = self.visual_projection(hidden_states)

        return projected


# ---------------------------------------------------------------------------
# Vision model wrapper (handles CPU preprocessing + Neuron forward)
# ---------------------------------------------------------------------------

class GitVisionModelWrapper(ModelWrapper):
    """
    Wrapper for Git's CLIP vision encoder that handles:
    1. CPU-side preprocessing (unfold pixels into patches, CLS token, positions)
    2. Padding to bucket size
    3. Forward through Neuron-compiled vision model
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = True,
        model_init_kwargs={},
    ):
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx,
            pipeline_execution, return_ranked_to_cpu, model_init_kwargs,
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """Generate sample inputs for tracing (one per bucket).

        Only patch_embeds and attention_mask are traced inputs. Position
        embeddings are applied inside the model using a learned nn.Embedding
        (not passed as an external tensor).
        """
        inputs = []
        batch_size = self.config.vision_config.neuron_config.batch_size

        for bucket in self.config.vision_config.neuron_config.buckets:
            patch_dim = (
                self.config.vision_config.num_channels
                * self.config.vision_config.patch_size
                * self.config.vision_config.patch_size
            )
            patch_embeds = torch.ones(
                [batch_size, bucket, patch_dim],
                dtype=self.config.vision_config.neuron_config.torch_dtype,
            )
            attention_mask = torch.ones(
                [batch_size, 1, bucket, bucket],
                dtype=torch.int32,
            )
            inputs.append((patch_embeds, attention_mask))

        return inputs

    def patchify(self, pixel_values: torch.Tensor):
        """
        Convert pixel_values into flattened patches + CLS token for CLIP.

        CLIP ViT: Conv2d patch embedding + prepend CLS token + position embeddings.
        We unfold on CPU, prepend a zero CLS slot, and send to Neuron where:
        - patch_linear replaces Conv2d
        - class_embedding is added to slot 0 in the model forward
        - position_embedding (learned) is applied in the model forward

        Args:
            pixel_values: [BS, C, H, W] preprocessed image tensor

        Returns:
            patch_embeds: [BS, num_patches+1, C*P*P] (CLS prepended as zeros)
            attention_mask: [BS, 1, num_patches+1, num_patches+1]
        """
        if len(pixel_values.shape) == 5:
            pixel_values = pixel_values.squeeze(0)

        bs, nc, h, w = pixel_values.shape
        ps = self.config.vision_config.patch_size
        grid_h = h // ps
        grid_w = w // ps
        num_patches = grid_h * grid_w
        patch_dim = nc * ps * ps

        # Unfold into patches: [BS, num_patches, C*P*P]
        x = pixel_values[:, :, :grid_h * ps, :grid_w * ps]
        x = x.reshape(bs, nc, grid_h, ps, grid_w, ps)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [BS, grid_h, grid_w, C, ps, ps]
        x = x.reshape(bs, num_patches, patch_dim)

        # Prepend CLS token slot (zeros -- the actual CLS embedding is in the model)
        cls_slot = torch.zeros(bs, 1, patch_dim, dtype=x.dtype, device=x.device)
        patch_embeds = torch.cat([cls_slot, x], dim=1)  # [BS, num_patches+1, C*P*P]

        patch_embeds = patch_embeds.to(self.config.vision_config.neuron_config.torch_dtype)

        seq_len = num_patches + 1  # patches + CLS

        # Full bidirectional attention (all tokens attend to all)
        attention_mask = torch.ones(
            [bs, 1, seq_len, seq_len], dtype=torch.int32
        )

        return patch_embeds, attention_mask

    def pad_inputs(self, patch_embeds, attention_mask):
        """Pad inputs to the nearest bucket size."""
        target_len = self.get_target_bucket(patch_embeds)

        target_patch_shape = [patch_embeds.shape[0], target_len, patch_embeds.shape[2]]
        padded_patches, self.original_slices = pad_tensor(patch_embeds, target_patch_shape)

        # Update slices for projector output (text_hidden_size, not vision_hidden_size)
        self.original_slices[-1][-1] = self.config.text_config.hidden_size

        target_mask_shape = [
            attention_mask.shape[0],
            attention_mask.shape[1],
            target_len,
            target_len,
        ]
        padded_mask, _ = pad_tensor(attention_mask, target_mask_shape, pad_value=0)

        return padded_patches, padded_mask

    def get_target_bucket(self, patch_embeds) -> int:
        """Find the smallest bucket that fits the input."""
        seq_len = patch_embeds.shape[1]
        for bucket in self.config.vision_config.neuron_config.buckets:
            if seq_len <= bucket:
                logger.info(f"Routing vision seq_len {seq_len} to bucket {bucket}")
                return bucket
        raise ValueError(
            f"Vision sequence length {seq_len} exceeds largest bucket "
            f"{self.config.vision_config.neuron_config.buckets[-1]}"
        )

    def forward(self, pixel_values: torch.Tensor):
        """
        Full forward: patchify -> pad -> Neuron forward -> unpad.

        Args:
            pixel_values: [BS, C, H, W]

        Returns:
            vision_embeddings: [BS, num_patches+1, text_hidden_size]
        """
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")

        patch_embeds, attention_mask = self.patchify(pixel_values)
        padded_patches, padded_mask = self.pad_inputs(patch_embeds, attention_mask)

        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(padded_patches, padded_mask)
        else:
            args = (padded_patches, padded_mask)

        vision_emb = self._forward(*args)
        vision_emb = unpad_tensor(vision_emb, self.original_slices)

        return vision_emb

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)


# ---------------------------------------------------------------------------
# Text decoder components (BERT-style, from the original text-only port)
# ---------------------------------------------------------------------------

class NeuronGitTextAttention(NeuronAttentionBase):
    """Git text decoder attention: BERT-style MHA with bias, no RoPE."""

    def __init__(self, config):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=None,
            qkv_bias=True,
            o_bias=True,
            tensor_model_parallel_group=get_tp_group(config),
        )


class NeuronGitTextMLP(nn.Module):
    """Git text decoder MLP: intermediate -> act -> output, with bias."""

    def __init__(self, config):
        super().__init__()
        self.fc_in = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.fc_out = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states, None


class NeuronGitTextBlock(nn.Module):
    """Git text decoder block: post-LN (BERT-style) residual connections."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.attn = NeuronGitTextAttention(config)
        self.attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = NeuronGitTextMLP(config)
        self.output_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple:
        residual = hidden_states
        attn_output = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = self.attn_ln(attn_output.hidden_states + residual)

        residual = hidden_states
        mlp_output, _ = self.mlp(hidden_states)
        hidden_states = self.output_ln(mlp_output + residual)

        return (
            hidden_states,
            attn_output.present_key_value,
            attn_output.cos_cache,
            attn_output.sin_cache,
            None,
        )


# ---------------------------------------------------------------------------
# Text model with vision hook
# ---------------------------------------------------------------------------

from neuronx_distributed_inference.models.model_base import NeuronBaseModel

try:
    from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
        scatter_by_index_put,
        generate_positions_from_mask,
        pad_positions,
        pad_vision_embeddings,
    )
except ImportError:
    # Fallback if llama4 utils not available
    scatter_by_index_put = None
    generate_positions_from_mask = None
    pad_positions = None
    pad_vision_embeddings = None


class NeuronGitTextModel(NeuronBaseModel):
    """
    Git text decoder model with encode_vision_to_input hook.

    During context encoding, vision embeddings are injected into text
    embeddings at positions indicated by the vision mask.
    """

    def setup_attr_for_model(self, config):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config):
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

        self.position_embeddings = ParallelEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        self.embed_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.layers = nn.ModuleList(
            [NeuronGitTextBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        self.norm = nn.Identity()

    def get_model_output(self, input_ids, position_ids=None, **kwargs):
        """Add absolute position embeddings + embedding LayerNorm.

        In Git vision+text mode, vision tokens occupy the first num_image_tokens
        positions but text position embeddings are 0-indexed (matching HF).
        We subtract the vision offset from position_ids for the position embedding
        lookup while keeping the original position_ids for KV cache addressing.
        """
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        # Offset position IDs for text position embedding lookup.
        # Vision tokens at positions 0..num_image_tokens-1 get clamped to 0
        # (irrelevant since encode_vision_to_input replaces them).
        # Text BOS at position num_image_tokens gets pos_embed(0), matching HF.
        num_image_tokens = getattr(self.config, "num_image_tokens", 197)
        text_pos_ids = torch.clamp(position_ids - num_image_tokens, min=0)

        position_embeds = self.position_embeddings(text_pos_ids)
        inputs_embeds = inputs_embeds + position_embeds
        inputs_embeds = self.embed_ln(inputs_embeds)

        kwargs["inputs_embeds"] = inputs_embeds
        return super().get_model_output(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs,
        )

    def encode_vision_to_input(
        self, inputs_embeds, vision_embeddings, vision_mask
    ) -> torch.Tensor:
        """
        Merge vision embeddings into text input embeddings.

        Uses scatter_by_index_put for efficient placement of vision features
        at positions indicated by vision_mask.
        """
        if scatter_by_index_put is not None:
            return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

        # Fallback: manual scatter
        batch_size = inputs_embeds.shape[0]
        for b in range(batch_size):
            mask_positions = vision_mask[b].nonzero(as_tuple=True)[0]
            num_vision = min(len(mask_positions), vision_embeddings.shape[1])
            if num_vision > 0:
                inputs_embeds[b, mask_positions[:num_vision]] = vision_embeddings[b, :num_vision]
        return inputs_embeds


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class NeuronGitForCausalLMVision(NeuronBaseForImageToText):
    """
    Full-Neuron Git: CLIP vision encoder + projection + BERT text decoder.

    Follows the LLaVA/Pixtral pattern:
    - text_model_cls -> NeuronGitTextModel (BERT decoder with vision hook)
    - vision_model_cls -> NeuronGitCLIPVisionModel (CLIP ViT + projection)
    - Separate compilation for text and vision models
    """

    text_model_cls = NeuronGitTextModel
    vision_model_cls = NeuronGitCLIPVisionModel
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = GitVisionModelWrapper

    # The HF state dict has "git." prefix on text decoder keys
    _STATE_DICT_MODEL_PREFIX = "git."

    # Vision encoder does not support these NeuronConfig features
    _VISION_UNSUPPORTED_NEURON_CONFIG = [
        "sequence_parallel_enabled",
        "flash_decoding_enabled",
        "attn_kernel_enabled",
        "fused_qkv",
        "qkv_kernel_enabled",
        "mlp_kernel_enabled",
        "attn_block_tkg_nki_kernel_cache_update",
        "attn_block_tkg_nki_kernel_enabled",
        "attn_block_cte_nki_kernel_enabled",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

        # Disable unsupported features for vision encoder
        for attr in self._VISION_UNSUPPORTED_NEURON_CONFIG:
            if getattr(self.vision_config.neuron_config, attr, False):
                setattr(self.vision_config.neuron_config, attr, False)
                logger.warning(
                    f"Git vision model does not support '{attr}'. Disabled."
                )

    @classmethod
    def get_config_cls(cls):
        return GitVisionInferenceConfig

    def get_vision_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = getattr(
            self.vision_config.neuron_config, "cc_pipeline_tiling_factor", 2
        )
        return (
            f"--enable-saturate-infinity --auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc_pipeline_tiling_factor}' -O1 "
            f"--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def get_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = getattr(
            self.text_config.neuron_config, "cc_pipeline_tiling_factor", 2
        )
        return (
            f"--enable-saturate-infinity --auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc_pipeline_tiling_factor}' -O1 "
            f"--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        """Initialize the vision encoder model wrapper."""
        new_config = copy.deepcopy(self.config)

        # Set up vision buckets
        v_nc = new_config.vision_config.neuron_config
        if v_nc.enable_bucketing:
            if v_nc.buckets == [v_nc.seq_len] or v_nc.buckets is None:
                v_nc.buckets = [v_nc.seq_len]

        new_config.neuron_config = copy.deepcopy(v_nc)

        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=True,
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        """
        Convert HF GitForCausalLM state dict to Neuron format.

        NOTE: By the time this is called, the base class has already stripped
        the "git." prefix via _STATE_DICT_MODEL_PREFIX. So keys arrive as:
        - image_encoder.vision_model.* (vision encoder)
        - visual_projection.* (projection)
        - embeddings.*, encoder.layer.* (text decoder)
        - output.* (LM head, never had "git." prefix)

        Handles all three weight categories:
        1. Text decoder weights (encoder.*, embeddings.*, output.*)
        2. Vision encoder weights (image_encoder.vision_model.*)
        3. Visual projection weights (visual_projection.*)
        """
        logger.info(f"Converting Git state dict: {len(state_dict)} keys")

        text_state_dict = {}
        vision_state_dict = {}
        projection_state_dict = {}
        skipped = []

        for key, value in state_dict.items():
            if key.startswith("image_encoder."):
                vision_state_dict[key] = value
            elif key.startswith("visual_projection."):
                projection_state_dict[key] = value
            elif key.startswith("output."):
                text_state_dict[key] = value
            elif (key.startswith("embeddings.") or key.startswith("encoder.layer.")):
                text_state_dict[key] = value
            else:
                skipped.append(key)

        if skipped:
            logger.warning(f"Skipped {len(skipped)} keys: {skipped[:10]}")

        logger.info(
            f"Categorized: {len(text_state_dict)} text, "
            f"{len(vision_state_dict)} vision, "
            f"{len(projection_state_dict)} projection"
        )

        neuron_state_dict = {}
        text_config = inference_config.text_config
        tp_degree = text_config.neuron_config.tp_degree

        # --- 1. Convert text decoder weights ---
        for key, value in text_state_dict.items():
            if key == "embeddings.word_embeddings.weight":
                neuron_state_dict["embed_tokens.weight"] = value.clone()
            elif key == "embeddings.position_embeddings.weight":
                neuron_state_dict["position_embeddings.weight"] = value.clone()
            elif key == "embeddings.LayerNorm.weight":
                neuron_state_dict["embed_ln.weight"] = value.clone()
            elif key == "embeddings.LayerNorm.bias":
                neuron_state_dict["embed_ln.bias"] = value.clone()
            elif key == "output.weight":
                neuron_state_dict["lm_head.weight"] = value.clone()
            elif key == "output.bias":
                neuron_state_dict["lm_head.bias"] = value.clone()
            elif key.startswith("encoder.layer."):
                parts = key.split(".")
                layer_idx = parts[2]
                rest = ".".join(parts[3:])

                if rest.startswith("attention.self.query."):
                    suffix = rest.split("attention.self.query.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn.qkv_proj.q_proj.{suffix}"] = value.clone()
                elif rest.startswith("attention.self.key."):
                    suffix = rest.split("attention.self.key.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn.qkv_proj.k_proj.{suffix}"] = value.clone()
                elif rest.startswith("attention.self.value."):
                    suffix = rest.split("attention.self.value.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn.qkv_proj.v_proj.{suffix}"] = value.clone()
                elif rest.startswith("attention.output.dense."):
                    suffix = rest.split("attention.output.dense.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn.o_proj.o_proj.{suffix}"] = value.clone()
                elif rest.startswith("attention.output.LayerNorm."):
                    suffix = rest.split("attention.output.LayerNorm.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn_ln.{suffix}"] = value.clone()
                elif rest.startswith("intermediate.dense."):
                    suffix = rest.split("intermediate.dense.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.mlp.fc_in.{suffix}"] = value.clone()
                elif rest.startswith("output.dense."):
                    suffix = rest.split("output.dense.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.mlp.fc_out.{suffix}"] = value.clone()
                elif rest.startswith("output.LayerNorm."):
                    suffix = rest.split("output.LayerNorm.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.output_ln.{suffix}"] = value.clone()
                else:
                    logger.warning(f"Unmapped text encoder key: {key}")
            elif key == "embeddings.position_ids":
                pass  # Skip position_ids buffer
            else:
                logger.warning(f"Unmapped text key: {key}")

        # Add rank_util tensors for text attention layers
        for i in range(text_config.num_hidden_layers):
            neuron_state_dict[f"layers.{i}.attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # --- 2. Convert vision encoder weights ---
        v_dtype = inference_config.vision_config.neuron_config.torch_dtype
        vis_prefix = "image_encoder.vision_model."

        for src_key, value in vision_state_dict.items():
            inner_key = src_key
            if inner_key.startswith(vis_prefix):
                inner_key = inner_key[len(vis_prefix):]

            if inner_key == "embeddings.class_embedding":
                # CLS token embedding [hidden_size] - stored as nn.Parameter
                neuron_state_dict["class_embedding"] = (
                    value.clone().detach().contiguous().to(v_dtype)
                )
                continue
            elif inner_key == "embeddings.patch_embedding.weight":
                # Conv2d weight [out, in, kH, kW] -> Linear weight [out, in*kH*kW]
                vc = inference_config.vision_config
                new_weight = value.reshape(
                    vc.hidden_size,
                    vc.num_channels * vc.patch_size * vc.patch_size,
                )
                neuron_state_dict["patch_linear.weight"] = (
                    new_weight.clone().detach().contiguous().to(v_dtype)
                )
                continue
            elif inner_key == "embeddings.position_embedding.weight":
                # Learned absolute position embeddings [num_positions, hidden_size]
                neuron_state_dict["position_embedding.weight"] = (
                    value.clone().detach().contiguous().to(v_dtype)
                )
                continue
            elif inner_key == "embeddings.position_ids":
                # HF buffer -- skip, we register our own
                continue
            elif "pre_layrnorm" in inner_key or "pre_layernorm" in inner_key:
                suffix = inner_key.split(".")[-1]
                dst = f"pre_layernorm.{suffix}"
            elif "post_layernorm" in inner_key:
                suffix = inner_key.split(".")[-1]
                dst = f"post_layernorm.{suffix}"
            elif "encoder.layers." in inner_key:
                dst = inner_key
                # Map Q/K/V/O proj keys to NXDI format
                attn_key_map = {
                    ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
                    ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
                    ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
                    ".self_attn.out_proj.": ".self_attn.o_proj.o_proj.",
                }
                for pattern, replacement in attn_key_map.items():
                    if pattern in dst:
                        dst = dst.replace(pattern, replacement)
                        break
            else:
                logger.warning(f"Unmapped vision key: {src_key} (inner: {inner_key})")
                dst = inner_key

            neuron_state_dict[dst] = (
                value.clone().detach().contiguous().to(v_dtype)
            )

        # Add rank_util for vision attention layers
        v_tp = inference_config.vision_config.neuron_config.tp_degree
        for i in range(inference_config.vision_config.num_hidden_layers):
            neuron_state_dict[f"encoder.layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, v_tp, dtype=torch.int32
            )

        # --- 3. Convert visual projection weights ---
        proj_prefix = "visual_projection.visual_projection."
        for src_key, value in projection_state_dict.items():
            inner_key = src_key
            if inner_key.startswith(proj_prefix):
                inner_key = inner_key[len(proj_prefix):]

            # Map: 0.weight -> dense.weight, 0.bias -> dense.bias
            #       1.weight -> layer_norm.weight, 1.bias -> layer_norm.bias
            if inner_key.startswith("0."):
                suffix = inner_key[2:]  # weight or bias
                dst = f"visual_projection.dense.{suffix}"
            elif inner_key.startswith("1."):
                suffix = inner_key[2:]
                dst = f"visual_projection.layer_norm.{suffix}"
            else:
                logger.warning(f"Unmapped projection key: {src_key}")
                continue

            neuron_state_dict[dst] = (
                value.clone().detach().contiguous().to(v_dtype)
            )

        logger.info(f"Final Neuron state dict: {len(neuron_state_dict)} keys")
        return neuron_state_dict

    def get_padding_length(self, input_ids):
        """Find the context encoding bucket for the given input length."""
        buckets = self.context_encoding_model.config.neuron_config.buckets
        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise ValueError(f"No bucket found for input length {input_ids.shape[1]}")

    def get_required_kwargs(self) -> List[str]:
        return ["pixel_values", "vision_mask"]

    def forward_atomic_prefill(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
    ):
        """Prefill with vision: encode image, merge with text, run context encoding."""
        # Text-only mode: skip vision encoding entirely
        if pixel_values is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
            )

        if vision_mask is None:
            # Create vision mask from a sentinel token (Git doesn't use a standard image token)
            # In practice, the first num_image_tokens positions are vision tokens
            num_img = getattr(self.config, "num_image_tokens", 197)
            vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            vision_mask[:, :num_img] = True
            vision_mask = vision_mask.unsqueeze(-1)

        # Convert bool mask to position indices
        vision_mask_positions = generate_positions_from_mask(vision_mask.squeeze())

        # Run vision encoder on Neuron
        vision_embeddings = self.vision_encoder_model(
            pixel_values.to(self.vision_config.neuron_config.torch_dtype)
        ).to(self.text_config.neuron_config.torch_dtype)

        # Pad to text bucket size
        pad_limit = self.get_padding_length(input_ids)
        vision_mask_positions = pad_positions(vision_mask_positions, pad_limit, (pad_limit - 1))
        vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask_positions,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        medusa_args=None,
        return_dict: Optional[bool] = None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        slot_mapping: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        full_context_lens: Optional[torch.LongTensor] = None,
        computed_context_lens: Optional[torch.LongTensor] = None,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass: routes to prefill-with-vision or standard forward."""
        is_prefill = position_ids is not None and (
            position_ids.dim() == 2 and position_ids.shape[1] > 1
        )

        if is_prefill and pixel_values is not None and pixel_values.numel() > 0:
            return self.forward_atomic_prefill(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                pixel_values=pixel_values,
                vision_mask=vision_mask,
            )

        return super().forward(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            sampling_params=sampling_params,
            prev_hidden=prev_hidden,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            adapter_ids=adapter_ids,
            medusa_args=medusa_args,
            return_dict=return_dict,
            input_capture_hook=input_capture_hook,
            tensor_capture_hook=tensor_capture_hook,
            slot_mapping=slot_mapping,
            block_table=block_table,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )


__all__ = [
    "GitVisionInferenceConfig",
    "NeuronGitCLIPVisionModel",
    "NeuronGitTextModel",
    "NeuronGitForCausalLMVision",
]
