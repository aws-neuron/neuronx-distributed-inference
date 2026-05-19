# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi-K2.5 (moonshotai/Kimi-K2.5) multimodal on Neuron via NxDI.
#
# Architecture: Kimi-K2 text decoder + MoonViT-400M vision encoder
#   - Text: 1T MoE (384 experts, 8 active) + MLA attention, 61 layers
#   - Vision: MoonViT (27-layer ViT, hidden=1152) + PatchMergerMLP → 7168
#   - Fusion: scatter_by_index_put (Llama4/Pixtral pattern)
#   - K2.5 weights: INT4 compressed-tensors (experts) + BF16 (non-experts)
#     → dequantized to BF16 → FP8 per-channel quantized for Neuron
#
# Supported configuration:
#   - trn2.48xlarge: TP=64, EP=1, LNC=2, seq_len=512, batch_size=1
#   - FP8 per-channel quantization for routed expert weights
#   - CPU greedy sampling (no on-device sampling)
#   - Pre-computed MoonViT embeddings (all cores used by text decoder)
#
# The text decoder reuses the K2 model code (modeling_kimi_k2.py) unchanged.
# This file adds:
#   1. K2.5 checkpoint loader (INT4 dequant, prefix stripping)
#   2. Vision embedding fusion (encode_vision_to_input)
#   3. ImageToTextModelWrapper with non-trivial tracing inputs
#   4. Forward/output overrides for 24-arg vision pipeline
#   5. MoonViT image preprocessing utilities
#
# References:
#   - Kimi-K2 NxDI contrib (PR #131)
#   - NxDI Llama4/Pixtral scatter_by_index_put pattern
#   - NxDI ImageToTextModelWrapper / NeuronBaseForImageToText

import gc
import json
import logging
import math
import os
import shutil
import types
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    RouterConfig,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
    ImageToTextModelWrapper,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

PATCH_SIZE = 14
MERGE_KERNEL = 2
DEFAULT_IMAGE_SIZE = 448

# K2.5 special token IDs
BOS_TOKEN_ID = 163584
IM_USER_TOKEN_ID = 163587
IM_END_TOKEN_ID = 163586
IM_ASSISTANT_TOKEN_ID = 163588
MEDIA_PLACEHOLDER_TOKEN_ID = 163605

# Image normalization
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD_INV = np.array([2.0, 2.0, 2.0], dtype=np.float32)

# FP8 E4M3 max representable value
_FP8_E4M3_MAX = 240.0

# K2.5 weight key prefixes
K25_PREFIX = "language_model.model."
K2_PREFIX = "language_model."
VISION_PREFIXES = ("vision_tower.", "mm_projector.", "multi_modal_projector.")


# ============================================================================
# Image Preprocessing
# ============================================================================


def preprocess_image(image, target_size=DEFAULT_IMAGE_SIZE):
    """Preprocess a PIL image for MoonViT.

    Args:
        image: PIL Image
        target_size: Target size (square)

    Returns:
        pixel_values: [N_patches, 3, 14, 14] bfloat16
        grid_thw: (1, h_patches, w_patches)
        n_merged_tokens: Number of vision tokens after 2x2 merge
    """
    image = image.convert("RGB")
    image = image.resize((target_size, target_size))
    img_np = np.array(image, dtype=np.float32)
    img_np = (img_np / 255.0 - 0.5) * 2.0
    img_np = img_np[np.newaxis, ...]

    T, H, W, C = img_np.shape
    h_patches = H // PATCH_SIZE
    w_patches = W // PATCH_SIZE
    patches = img_np.reshape(T, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, C)
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, C, PATCH_SIZE, PATCH_SIZE)

    pixel_values = torch.from_numpy(patches).to(torch.bfloat16)
    grid_thw = (1, h_patches, w_patches)
    n_merged_tokens = (h_patches // MERGE_KERNEL) * (w_patches // MERGE_KERNEL)

    return pixel_values, grid_thw, n_merged_tokens


def precompute_rope_tables(h_patches, w_patches, head_dim=72, theta=10000.0):
    """Precompute 2D RoPE cos/sin tables for MoonViT.

    Returns:
        cos_table: [h_patches * w_patches, head_dim // 2] bfloat16
        sin_table: [h_patches * w_patches, head_dim // 2] bfloat16
    """
    N = h_patches * w_patches
    flat_pos = torch.arange(N, dtype=torch.float32)
    x_pos = flat_pos % w_patches
    y_pos = flat_pos // w_patches

    dim = head_dim
    dim_range = torch.arange(0, dim, 4)[: dim // 4].float()
    freqs = 1.0 / (theta ** (dim_range / dim))

    x_freqs = torch.outer(x_pos, freqs)
    y_freqs = torch.outer(y_pos, freqs)

    angles = torch.cat([y_freqs, x_freqs], dim=-1)
    cos_table = torch.cos(angles)
    sin_table = torch.sin(angles)

    return cos_table.to(torch.bfloat16), sin_table.to(torch.bfloat16)


# ============================================================================
# K2.5 Checkpoint Loader
# ============================================================================


def _dequant_int4_packed_symmetric(weight_packed, weight_scale, group_size=32):
    """Dequantize INT4 symmetric pack-quantized weights to BF16.

    compressed-tensors 'pack-quantized' format:
    - weight_packed: INT32 [out_features, in_features // 8] (8 INT4 per int32)
    - weight_scale: FP16/BF16 [out_features, in_features // group_size]
    - Interleaved column ordering, offset-binary sign convention

    Returns: BF16 [out_features, in_features]
    """
    out_features = weight_packed.shape[0]
    packed_cols = weight_packed.shape[1]
    in_features = packed_cols * 8
    pack_factor = 8
    num_bits = 4
    mask = (1 << num_bits) - 1

    unpacked = torch.zeros(
        (out_features, in_features), device=weight_packed.device, dtype=torch.int32
    )
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (weight_packed >> (num_bits * i)) & mask

    unpacked = (unpacked - 8).to(torch.int8)

    unpacked_f32 = unpacked.to(torch.float32)
    scale = weight_scale.to(torch.float32)
    scale = (
        scale.unsqueeze(-1)
        .expand(-1, -1, group_size)
        .reshape(out_features, in_features)
    )

    result = unpacked_f32 * scale
    return result.to(torch.bfloat16)


def _strip_k25_prefix(key):
    """Strip K2.5 weight key prefix to K2-compatible format."""
    if key.startswith(K25_PREFIX):
        return key[len(K25_PREFIX) :]
    if key.startswith(K2_PREFIX):
        return key[len(K2_PREFIX) :]
    return key


def _is_vision_key(key):
    """Check if a weight key belongs to the vision encoder or projector."""
    for prefix in VISION_PREFIXES:
        if key.startswith(prefix):
            return True
    return False


def _requantize_per_channel_fp8(bf16_weight):
    """Re-quantize BF16 to per-expert per-channel FP8 E4M3.

    Args:
        bf16_weight: [E, H, W] bfloat16
    Returns:
        (fp8_weight, scale): fp8_weight [E, H, W], scale [E, 1, W] float32
    """
    fp32_weight = bf16_weight.float()
    amax = fp32_weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / _FP8_E4M3_MAX
    scaled = (fp32_weight / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    fp8_weight = scaled.to(torch.float8_e4m3fn)

    # Clamp exponent-15 bytes
    raw = fp8_weight.view(torch.uint8)
    pos_exp15 = (raw >= 0x78) & (raw <= 0x7E)
    neg_exp15 = (raw >= 0xF8) & (raw <= 0xFE)
    raw = torch.where(pos_exp15, torch.tensor(0x77, dtype=torch.uint8), raw)
    raw = torch.where(neg_exp15, torch.tensor(0xF7, dtype=torch.uint8), raw)
    fp8_weight = raw.view(torch.float8_e4m3fn)

    return fp8_weight, scale.to(torch.float32)


def k25_checkpoint_loader_fn(model_self, mmap=False):
    """K2.5-adapted checkpoint loader.

    Handles:
    1. 'language_model.model.' prefix stripping
    2. Vision key filtering
    3. INT4 compressed-tensors dequantization → BF16
    4. Expert packing (gate+up concat, down transpose)
    5. Optional FP8 per-channel re-quantization
    """
    from safetensors.torch import load_file

    model_path = getattr(model_self.config, "_name_or_path", None)
    if model_path is None or not os.path.exists(str(model_path)):
        model_path = model_self.model_path

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return model_self.__class__.__bases__[0].checkpoint_loader_fn(
            model_self, mmap=mmap
        )

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    quant_config = getattr(model_self.config, "quantization_config", None)
    group_size = 32
    if isinstance(quant_config, dict):
        config_groups = quant_config.get("config_groups", {})
        for group_cfg in config_groups.values():
            weights_cfg = group_cfg.get("weights", {})
            gs = weights_cfg.get("group_size", 32)
            if gs:
                group_size = gs
                break

    n_routed_experts = getattr(model_self.config, "n_routed_experts", 384)
    first_k_dense_replace = getattr(model_self.config, "first_k_dense_replace", 1)
    num_layers = model_self.config.num_hidden_layers
    keep_experts_fp8 = getattr(model_self.config.neuron_config, "quantized", False)

    # Determine needed shards
    needed_shards = set()
    for key, shard_file in weight_map.items():
        clean_key = _strip_k25_prefix(key)
        if _is_vision_key(clean_key) or _is_vision_key(key):
            continue
        if "layers." in clean_key:
            parts = clean_key.split(".")
            idx = parts.index("layers") + 1
            layer_idx = int(parts[idx])
            if layer_idx < num_layers:
                needed_shards.add(shard_file)
        else:
            needed_shards.add(shard_file)

    logger.info(
        f"K2.5 loader: {len(shard_files)} shards, {len(needed_shards)} needed, "
        f"num_layers={num_layers}, group_size={group_size}, "
        f"experts={n_routed_experts}, fp8={keep_experts_fp8}"
    )

    result_dict = {}
    for i, shard_file in enumerate(shard_files):
        if shard_file not in needed_shards:
            continue

        shard_path = os.path.join(model_path, shard_file)
        logger.info(f"Loading shard [{i + 1}/{len(shard_files)}]: {shard_file}")
        shard_data = load_file(shard_path)

        # Strip K2.5 prefix and filter vision keys
        for key in list(shard_data.keys()):
            clean_key = _strip_k25_prefix(key)
            if _is_vision_key(clean_key) or _is_vision_key(key):
                del shard_data[key]
                continue
            if clean_key != key:
                shard_data[clean_key] = shard_data.pop(key)

        # Filter layers beyond num_hidden_layers
        for key in list(shard_data.keys()):
            if "layers." in key:
                parts = key.split(".")
                idx = parts.index("layers") + 1
                layer_idx = int(parts[idx])
                if layer_idx >= num_layers:
                    del shard_data[key]

        # Dequantize INT4 packed expert weights
        packed_keys = [k for k in shard_data if k.endswith(".weight_packed")]
        shape_keys = [k for k in shard_data if k.endswith(".weight_shape")]
        zp_keys = [k for k in shard_data if k.endswith(".weight_zero_point")]

        for packed_key in packed_keys:
            scale_key = packed_key.replace(".weight_packed", ".weight_scale")
            weight_key = packed_key.replace(".weight_packed", ".weight")

            packed = shard_data[packed_key]
            scale = shard_data.get(scale_key)

            if scale is None:
                logger.warning(f"No scale for {packed_key}, skipping")
                continue

            if packed.dtype in (torch.int32, torch.int16, torch.int8, torch.uint8):
                dequant = _dequant_int4_packed_symmetric(packed, scale, group_size)
                shard_data[weight_key] = dequant
                del packed, dequant
            else:
                shard_data[weight_key] = shard_data[packed_key]

            del shard_data[packed_key]
            if scale_key in shard_data:
                del shard_data[scale_key]

        for k in shape_keys:
            shard_data.pop(k, None)
        for k in zp_keys:
            shard_data.pop(k, None)

        # Cast non-BF16 float tensors
        for key in list(shard_data.keys()):
            t = shard_data[key]
            if torch.is_floating_point(t) and t.dtype not in (
                torch.bfloat16,
                torch.float32,
            ):
                if t.dtype != torch.int64 and t.dtype != torch.int32:
                    shard_data[key] = t.to(torch.bfloat16)

        # Determine layers in this shard
        layer_ids = set()
        for key in shard_data:
            if "layers." in key:
                parts = key.split(".")
                idx = parts.index("layers") + 1
                layer_ids.add(int(parts[idx]))

        # Pack experts and rename for MoE layers
        for layer_idx in sorted(layer_ids):
            prefix = f"layers.{layer_idx}"
            if layer_idx >= first_k_dense_replace:
                # Router rename
                gate_key = f"{prefix}.mlp.gate.weight"
                if gate_key in shard_data:
                    shard_data[f"{prefix}.mlp.router.linear_router.weight"] = (
                        shard_data.pop(gate_key)
                    )
                bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
                if bias_key in shard_data:
                    shard_data[f"{prefix}.mlp.router.linear_router.bias"] = (
                        shard_data.pop(bias_key)
                    )

                # Pack experts
                e0_gate = f"{prefix}.mlp.experts.0.gate_proj.weight"
                if e0_gate in shard_data:
                    isize, hsize = shard_data[e0_gate].shape
                    dtype = shard_data[e0_gate].dtype

                    gate_up = torch.zeros(
                        n_routed_experts, hsize, 2 * isize, dtype=dtype, device="cpu"
                    )
                    for e in range(n_routed_experts):
                        gk = f"{prefix}.mlp.experts.{e}.gate_proj.weight"
                        uk = f"{prefix}.mlp.experts.{e}.up_proj.weight"
                        if gk in shard_data:
                            gate_up[e, :, :isize] = shard_data.pop(gk).T
                        if uk in shard_data:
                            gate_up[e, :, isize:] = shard_data.pop(uk).T

                    down = torch.zeros(
                        n_routed_experts, isize, hsize, dtype=dtype, device="cpu"
                    )
                    for e in range(n_routed_experts):
                        dk = f"{prefix}.mlp.experts.{e}.down_proj.weight"
                        if dk in shard_data:
                            down[e] = shard_data.pop(dk).T

                    if keep_experts_fp8:
                        gu_fp8, gu_scale = _requantize_per_channel_fp8(gate_up)
                        shard_data[
                            f"{prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                        ] = gu_fp8
                        shard_data[
                            f"{prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.scale"
                        ] = gu_scale
                        dn_fp8, dn_scale = _requantize_per_channel_fp8(down)
                        shard_data[
                            f"{prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"
                        ] = dn_fp8
                        shard_data[
                            f"{prefix}.mlp.expert_mlps.mlp_op.down_proj.scale"
                        ] = dn_scale
                        del gate_up, down
                    else:
                        shard_data[
                            f"{prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                        ] = gate_up
                        shard_data[
                            f"{prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"
                        ] = down

                # Clean up per-expert keys
                for e in range(n_routed_experts):
                    for proj in ["gate_proj", "up_proj", "down_proj"]:
                        for suffix in [
                            ".weight",
                            ".weight_scale",
                            ".weight_shape",
                            ".weight_zero_point",
                        ]:
                            shard_data.pop(
                                f"{prefix}.mlp.experts.{e}.{proj}{suffix}", None
                            )

                # Shared expert rename
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    hf_key = f"{prefix}.mlp.shared_experts.{proj}.weight"
                    nxdi_key = f"{prefix}.shared_experts.{proj}.weight"
                    if hf_key in shard_data:
                        shard_data[nxdi_key] = shard_data.pop(hf_key)

        # Cast float32 to BF16 (except scales and router bias)
        for key in list(shard_data.keys()):
            t = shard_data[key]
            if (
                t.dtype == torch.float32
                and not key.endswith(".scale")
                and not key.endswith("linear_router.bias")
            ):
                shard_data[key] = t.to(torch.bfloat16)

        result_dict.update(shard_data)
        del shard_data
        gc.collect()

    # Add rank_util tensors
    tp = model_self.config.neuron_config.tp_degree
    result_dict["rank_util.rank"] = torch.arange(0, tp, dtype=torch.int32)
    for layer_idx in range(num_layers):
        result_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = torch.arange(
            0, tp, dtype=torch.int32
        )

    # Add fused prefix if needed
    if model_self._FUSED_PREFIX != "":
        for key in list(result_dict.keys()):
            result_dict[f"{model_self._FUSED_PREFIX}.{key}"] = result_dict.pop(key)

    logger.info(f"K2.5 loader done. Total keys: {len(result_dict)}")
    return result_dict


# ============================================================================
# Vision Embedding Fusion
# ============================================================================


def patch_encode_vision_to_input(NeuronKimiK2Model):
    """Add encode_vision_to_input() to NeuronKimiK2Model.

    Called by NeuronBaseModel.get_model_output() during context encoding
    when vision_embeddings and vision_mask are non-None.

    Uses scatter_by_index_put (Llama4/Pixtral pattern): replaces text
    embeddings at vision token positions with projected vision embeddings.

    Args:
        NeuronKimiK2Model: The text decoder model class to patch.
    """

    def _encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        """Merge vision into text embeddings via index_put_.

        Args:
            inputs_embeds: [BS, n_active, hidden_size]
            vision_embeddings: [BS, n_active, hidden_size] — packed at front
            vision_mask: [BS, n_active, 1] — integer position indices,
                fill_value=(n_active - 1) for padding
        Returns:
            merged_embeds: [BS, n_active, hidden_size]
        """
        _, max_positions, embedding_dim = inputs_embeds.shape
        h = inputs_embeds.clone()
        flat_ve = vision_embeddings.view(-1, embedding_dim)
        positions = vision_mask.view(-1)
        num_positions = len(positions)
        flat_ve = flat_ve[:num_positions]
        h.view(-1, embedding_dim).index_put_((positions,), flat_ve, accumulate=False)
        return h

    NeuronKimiK2Model.encode_vision_to_input = _encode_vision_to_input


# ============================================================================
# ImageToTextModelWrapper with non-trivial tracing inputs
# ============================================================================


class K25ImageToTextModelWrapper(ImageToTextModelWrapper):
    """Custom wrapper with non-trivial (ones-like) vision tracing inputs.

    The standard ImageToTextModelWrapper provides zero-filled vision inputs
    for tracing, which the Neuron XLA compiler may optimize away (writing
    zeros at position 0 is a no-op). We use ones-like inputs matching
    NxDI's proven test_scatter.py pattern.
    """

    def input_generator(self):
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket
                if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )

            input_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            attention_mask = torch.zeros(
                (self.neuron_config.batch_size, bucket), dtype=torch.int32
            )
            position_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)

            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros(
                (self.neuron_config.batch_size, sampling_params_len),
                dtype=torch.float32,
            )

            if n_active_tokens > 1:
                # CTE: ones-like vision inputs to prevent compiler optimization
                vision_embeddings = torch.ones(
                    self.neuron_config.batch_size,
                    n_active_tokens,
                    self.config.hidden_size,
                    dtype=self.config.neuron_config.torch_dtype,
                )
                vision_mask = torch.ones(
                    self.neuron_config.batch_size,
                    n_active_tokens,
                    1,
                    dtype=torch.int32,
                )
            else:
                # TKG: empty vision inputs
                vision_embeddings = torch.zeros(
                    (0), dtype=self.config.neuron_config.torch_dtype
                )
                vision_mask = torch.zeros((0), dtype=torch.bool)

            inputs.append(
                (
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    torch.empty(0),  # prev_hidden
                    torch.empty(0),  # adapter_ids
                    torch.empty(0),  # accepted_indices
                    torch.empty(0),  # current_length
                    torch.empty(0),  # medusa_mask
                    torch.empty(0),  # scatter_index
                    torch.empty(0),  # slot_mapping
                    torch.empty(0),  # active_block_table
                    torch.empty(0),  # num_queries
                    torch.empty(0),  # computed_context_lens
                    torch.empty(0),  # tile_q_indices
                    torch.empty(0),  # tile_block_tables
                    torch.empty(0),  # tile_masks
                    torch.empty(0),  # inputs_embeds
                    torch.empty(0),  # kv_cache
                    torch.empty(0),  # active_mask
                    torch.empty(0),  # rotary_position_id
                    vision_embeddings,
                    vision_mask,
                )
            )

        return inputs


# ============================================================================
# Model Patching — Apply all K2.5-specific patches
# ============================================================================


def apply_k25_patches(NeuronKimiK2ForCausalLM, NeuronKimiK2Model, ep_degree=1):
    """Apply all patches to transform K2 text-only model into K2.5 multimodal.

    Must be called BEFORE model initialization (__init__ calls
    enable_context_encoding/enable_token_generation which use wrapper class).

    Args:
        NeuronKimiK2ForCausalLM: The top-level K2 model class
        NeuronKimiK2Model: The K2 model graph class
        ep_degree: Expert parallelism degree
    """
    # 1. Vision embedding fusion
    patch_encode_vision_to_input(NeuronKimiK2Model)

    # 2. ImageToTextModelWrapper
    def _get_model_wrapper_cls(self):
        return K25ImageToTextModelWrapper

    NeuronKimiK2ForCausalLM.get_model_wrapper_cls = _get_model_wrapper_cls

    # 3. Default compiler args (ModelWrapper handles flags)
    NeuronKimiK2ForCausalLM.get_compiler_args = lambda self: None

    # 4. Forward with vision support
    _orig_forward = NeuronKimiK2ForCausalLM.forward

    def _vl_forward(
        self,
        input_ids=None,
        seq_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        sampling_params=None,
        prev_hidden=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        adapter_ids=None,
        medusa_args=None,
        return_dict=None,
        llava_args=None,
        input_capture_hook=None,
        slot_mapping=None,
        block_table=None,
        full_context_lens=None,
        computed_context_lens=None,
        vision_embeddings=None,
        vision_mask=None,
        **kwargs,
    ):
        input_ids, attention_mask, position_ids, seq_ids, sampling_params = (
            self.preprocess_inputs(
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
                llava_args=llava_args if llava_args else [],
                input_capture_hook=input_capture_hook,
                slot_mapping=slot_mapping,
                block_table=block_table,
                full_context_lens=full_context_lens,
                computed_context_lens=computed_context_lens,
            )
        )

        outputs, is_run_on_neuron = self._get_model_outputs(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            prev_hidden,
            adapter_ids,
            medusa_args,
            llava_args if llava_args else [],
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

        generation_model = self.get_generation_model()
        if not generation_model.is_neuron():
            self._copy_past_key_values(outputs)

        if (
            self.on_device_sampling
            and self.neuron_config.output_logits
            and not (
                self.neuron_config.enable_fused_speculation
                or self.neuron_config.is_medusa
            )
        ):
            logits_or_next_tokens = outputs[:2]
            constructed_outputs = self._construct_output_with_tokens_and_logits(
                next_tokens=logits_or_next_tokens[0],
                logits=logits_or_next_tokens[1],
            )
        else:
            if is_run_on_neuron:
                logits_or_next_tokens = outputs
            else:
                logits_or_next_tokens, *_ = outputs
            constructed_outputs = self._construct_output(logits_or_next_tokens)

        return constructed_outputs

    NeuronKimiK2ForCausalLM.forward = _vl_forward

    # 5. _get_model_outputs with 24-arg ImageToText format
    def _vl_get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden,
        adapter_ids,
        medusa_args,
        llava_args,
        vision_embeddings=None,
        vision_mask=None,
        **kwargs,
    ):
        if vision_embeddings is None:
            vision_embeddings = torch.empty(0)
        if vision_mask is None:
            vision_mask = torch.empty(0)

        args_24 = (
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            vision_embeddings,
            vision_mask,
        )

        if self._is_prefill(position_ids):
            outputs = self.context_encoding_model(*args_24)
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        else:
            outputs = self.token_generation_model(*args_24)
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    NeuronKimiK2ForCausalLM._get_model_outputs = _vl_get_model_outputs

    # 6. EP-safe MoE forward (SDK 2.29 blockwise CTE regression workaround)
    if ep_degree > 1:
        from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2

        _original_forward = ExpertMLPsV2.forward

        def _ep_safe_forward(
            self_emv,
            hidden_states,
            expert_affinities,
            expert_index,
            seq_len,
            padding_mask=None,
            expert_affinities_masked_full=None,
        ):
            if (
                self_emv.moe_expert_model_parallel_group.size() > 1
                and not self_emv.training
            ):
                return self_emv.forward_all_experts_EP(
                    hidden_states, expert_affinities, expert_index
                )
            return _original_forward(
                self_emv,
                hidden_states,
                expert_affinities,
                expert_index,
                seq_len,
                padding_mask=padding_mask,
                expert_affinities_masked_full=expert_affinities_masked_full,
            )

        ExpertMLPsV2.forward = _ep_safe_forward


def apply_k25_checkpoint_patch(model):
    """Patch model to use K2.5 checkpoint loader and mmap weight loading.

    Must be called AFTER model initialization.

    Args:
        model: NeuronKimiK2ForCausalLM instance
    """
    # K2.5 checkpoint loader
    model.checkpoint_loader_fn = types.MethodType(k25_checkpoint_loader_fn, model)

    # No-op convert (K2.5 loader handles everything)
    def _noop_convert(state_dict, config):
        return state_dict

    model.convert_hf_to_neuron_state_dict = staticmethod(_noop_convert)

    # mmap-based weight loading
    def _mmap_load_weights(
        self_model, compiled_model_path, start_rank_id=None, local_ranks_size=None
    ):
        import resource
        from safetensors import safe_open

        if self_model.traced_model is None:
            raise ValueError("Model is not loaded")
        if start_rank_id is None:
            start_rank_id = self_model.neuron_config.start_rank_id
        if local_ranks_size is None:
            local_ranks_size = self_model.neuron_config.local_ranks_size

        weights_dir = os.path.join(compiled_model_path, "weights")
        first_shard = os.path.join(
            weights_dir, f"tp{start_rank_id}_sharded_checkpoint.safetensors"
        )

        if os.path.exists(first_shard):
            logger.info(f"Loading pre-sharded weights from {weights_dir}")
            weights = []
            for rank in range(start_rank_id, start_rank_id + local_ranks_size):
                fpath = os.path.join(
                    weights_dir, f"tp{rank}_sharded_checkpoint.safetensors"
                )
                sf = safe_open(fpath, framework="pt")
                shard = {key: sf.get_tensor(key) for key in sf.keys()}
                weights.append(shard)
        else:
            logger.info("No pre-sharded weights. Sharding at load time...")
            weights = self_model.get_builder().shard_checkpoint()

        start_rank_tensor = torch.tensor(
            [start_rank_id], dtype=torch.int32, device="cpu"
        )
        self_model.traced_model.nxd_model.initialize(weights, start_rank_tensor)
        del weights

    model.load_weights = types.MethodType(_mmap_load_weights, model)


# ============================================================================
# Text-Only Model Directory Setup
# ============================================================================


def create_text_only_model_dir(k25_model_path, output_dir):
    """Create text-only model directory with flat K2-compatible config.

    K2.5 config.json has text_config nested; the K2 model expects flat config.
    Creates symlinks for safetensors files and copies tokenizer files.

    Args:
        k25_model_path: Path to K2.5 HF model directory
        output_dir: Output directory for text-only model

    Returns:
        output_dir path
    """
    os.makedirs(output_dir, exist_ok=True)

    k25_config_path = os.path.join(k25_model_path, "config.json")
    with open(k25_config_path, "r") as f:
        k25_config = json.load(f)

    text_config = k25_config.get("text_config", k25_config)

    config_out = os.path.join(output_dir, "config.json")
    with open(config_out, "w") as f:
        json.dump(text_config, f, indent=2)

    # Symlink safetensors and index
    for fname in os.listdir(k25_model_path):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            src = os.path.join(k25_model_path, fname)
            dst = os.path.join(output_dir, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)

    # Copy tokenizer files
    for fname in os.listdir(k25_model_path):
        if "tokenizer" in fname or fname == "special_tokens_map.json":
            src = os.path.join(k25_model_path, fname)
            dst = os.path.join(output_dir, fname)
            if not os.path.exists(dst) and os.path.isfile(src):
                shutil.copy2(src, dst)

    return output_dir


# ============================================================================
# Config Builder
# ============================================================================


def build_k25_config(
    model_dir,
    tp_degree=64,
    ep_degree=1,
    lnc=2,
    batch_size=1,
    seq_len=512,
    n_active_tokens=128,
    quantized=True,
    num_layers=None,
):
    """Build KimiK2InferenceConfig for K2.5 multimodal inference.

    Args:
        model_dir: Path to text-only model directory (with flat config.json)
        tp_degree: Tensor parallel degree
        ep_degree: Expert parallel degree
        lnc: Logical neuron core config
        batch_size: Maximum batch size
        seq_len: Maximum sequence length
        n_active_tokens: Active tokens for TKG
        quantized: Use FP8 quantized experts
        num_layers: Override number of layers (for testing)

    Returns:
        KimiK2InferenceConfig
    """
    from modeling_kimi_k2 import KimiK2InferenceConfig

    neuron_config_kwargs = dict(
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        logical_nc_config=lnc,
        max_batch_size=batch_size,
        seq_len=seq_len,
        n_active_tokens=n_active_tokens,
        torch_dtype="bfloat16",
        capacity_factor=1.0,
        glu_mlp=True,
        moe_ep_degree=ep_degree,
        moe_tp_degree=tp_degree,
        router_config=RouterConfig(act_fn="sigmoid", dtype="float32"),
        save_sharded_checkpoint=False,
    )

    if quantized:
        neuron_config_kwargs["quantized"] = True
        neuron_config_kwargs["quantized_checkpoints_path"] = model_dir
        neuron_config_kwargs["quantization_dtype"] = "f8e4m3"
        neuron_config_kwargs["quantization_type"] = "expert_wise_per_channel_symmetric"
        neuron_config_kwargs["modules_to_not_convert"] = [
            "self_attn",
            "shared_experts",
            "embed_tokens",
            "lm_head",
            "norm",
            "router",
            "layers.0",
        ]

    neuron_config = MoENeuronConfig(**neuron_config_kwargs)

    with open(os.path.join(model_dir, "config.json"), "r") as f:
        hf_config = json.load(f)

    hf_kwargs = {
        k: v
        for k, v in hf_config.items()
        if k not in ("auto_map", "torch_dtype", "transformers_version", "architectures")
    }

    if num_layers is not None:
        hf_kwargs["num_hidden_layers"] = num_layers

    config = KimiK2InferenceConfig(neuron_config=neuron_config, **hf_kwargs)
    config.neuron_config.normalize_top_k_affinities = False
    config.neuron_config.blockwise_matmul_config.block_size = 2**30
    config.neuron_config.weights_to_skip_layout_optimization = [".*"]

    return config
