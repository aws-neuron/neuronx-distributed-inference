import gc
import math
from typing import Any, Dict

import torch

from neuronx_distributed_inference.models.config import InferenceConfig

from .encoder_utils import (
    initialize_global_position_embedding_from_local,
    resize_local_position_embedding,
)


def convert_hf_state_dict_to_neuron_state_dict(
    state_dict: dict, inference_config: InferenceConfig
) -> dict:
    vision_config = inference_config.vision_config
    config = inference_config.text_config

    # cross-attention layers: 20 for 90B, 8 for 11B
    cross_attention_num_layers = len(config.cross_attention_layers)
    self_attention_num_layers = config.num_hidden_layers - cross_attention_num_layers
    cross_attention_frequency = math.ceil(self_attention_num_layers / cross_attention_num_layers)
    text_num_total_layers = self_attention_num_layers + cross_attention_num_layers
    cross_attention_layers = list(
        range(cross_attention_frequency - 1, text_num_total_layers, cross_attention_frequency + 1)
    )

    state_dict["text_model.norm.weight"] = state_dict.pop("language_model.model.norm.weight")
    state_dict["text_model.lm_head.weight"] = state_dict.pop("language_model.lm_head.weight")
    state_dict["text_model.embed_tokens.weight"] = state_dict.pop("language_model.model.embed_tokens.weight")
    state_dict["text_model.rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )
    neuron_decoder_layer = 0
    for l in range(config.num_hidden_layers):  # noqa: E741
        if l in cross_attention_layers:
            # cross attention decoder layers
            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.gate_attn"] = (
                state_dict.pop(f"language_model.model.layers.{l}.cross_attn_attn_gate")[0].view(1)
            )
            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.gate_ffwd"] = (
                state_dict.pop(f"language_model.model.layers.{l}.cross_attn_mlp_gate")[0].view(1)
            )
            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.attention_norm.weight"] = (
                state_dict.pop(f"language_model.model.layers.{l}.input_layernorm.weight")
            )
            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.ffn_norm.weight"] = (
                state_dict.pop(f"language_model.model.layers.{l}.post_attention_layernorm.weight")
            )

            state_dict[
                f"text_model.layers.{neuron_decoder_layer}.xatten.feed_forward.gate_proj.weight"
            ] = state_dict.pop(f"language_model.model.layers.{l}.mlp.gate_proj.weight")
            state_dict[
                f"text_model.layers.{neuron_decoder_layer}.xatten.feed_forward.up_proj.weight"
            ] = state_dict.pop(f"language_model.model.layers.{l}.mlp.up_proj.weight")
            state_dict[
                f"text_model.layers.{neuron_decoder_layer}.xatten.feed_forward.down_proj.weight"
            ] = state_dict.pop(f"language_model.model.layers.{l}.mlp.down_proj.weight")
            # inner cross attention layers
            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.xatten.q_norm.weight"] = (
                state_dict.pop(f"language_model.model.layers.{l}.cross_attn.q_norm.weight")
            )
            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.xatten.k_norm.weight"] = (
                state_dict.pop(f"language_model.model.layers.{l}.cross_attn.k_norm.weight")
            )

            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.xatten.wq.weight"] = (
                state_dict.pop(f"language_model.model.layers.{l}.cross_attn.q_proj.weight")
            )
            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.xatten.wk.weight"] = (
                state_dict.pop(f"language_model.model.layers.{l}.cross_attn.k_proj.weight")
            )
            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.xatten.wv.weight"] = (
                state_dict.pop(f"language_model.model.layers.{l}.cross_attn.v_proj.weight")
            )

            state_dict[f"text_model.layers.{neuron_decoder_layer}.xatten.xatten.wo.weight"] = (
                state_dict.pop(f"language_model.model.layers.{l}.cross_attn.o_proj.weight")
            )

        else:
            # self attention decoder layers
            # attention mapping
            state_dict[
                f"text_model.layers.{neuron_decoder_layer}.self_attn.input_layernorm.weight"
            ] = state_dict.pop(f"language_model.model.layers.{l}.input_layernorm.weight")

            if config.neuron_config.fused_qkv:
                state_dict[
                    f"text_model.layers.{neuron_decoder_layer}.self_attn.self_attn.qkv_proj.Wqkv.weight"
                ] = torch.cat(
                    [
                        state_dict.pop(f"language_model.model.layers.{l}.self_attn.q_proj.weight"),
                        state_dict.pop(f"language_model.model.layers.{l}.self_attn.k_proj.weight"),
                        state_dict.pop(f"language_model.model.layers.{l}.self_attn.v_proj.weight"),
                    ]
                )
            else:
                state_dict[
                    f"text_model.layers.{neuron_decoder_layer}.self_attn.self_attn.qkv_proj.q_proj.weight"
                ] = state_dict.pop(f"language_model.model.layers.{l}.self_attn.q_proj.weight")
                state_dict[
                    f"text_model.layers.{neuron_decoder_layer}.self_attn.self_attn.qkv_proj.k_proj.weight"
                ] = state_dict.pop(f"language_model.model.layers.{l}.self_attn.k_proj.weight")
                state_dict[
                    f"text_model.layers.{neuron_decoder_layer}.self_attn.self_attn.qkv_proj.v_proj.weight"
                ] = state_dict.pop(f"language_model.model.layers.{l}.self_attn.v_proj.weight")

            state_dict[
                f"text_model.layers.{neuron_decoder_layer}.self_attn.self_attn.o_proj.weight"
            ] = state_dict.pop(f"language_model.model.layers.{l}.self_attn.o_proj.weight")

            # feed forward mapping
            state_dict[
                f"text_model.layers.{neuron_decoder_layer}.self_attn.post_attention_layernorm.weight"
            ] = state_dict.pop(f"language_model.model.layers.{l}.post_attention_layernorm.weight")
            state_dict[
                f"text_model.layers.{neuron_decoder_layer}.self_attn.feed_forward.gate_proj.weight"
            ] = state_dict.pop(f"language_model.model.layers.{l}.mlp.gate_proj.weight")
            state_dict[
                f"text_model.layers.{neuron_decoder_layer}.self_attn.feed_forward.up_proj.weight"
            ] = state_dict.pop(f"language_model.model.layers.{l}.mlp.up_proj.weight")
            state_dict[
                f"text_model.layers.{neuron_decoder_layer}.self_attn.feed_forward.down_proj.weight"
            ] = state_dict.pop(f"language_model.model.layers.{l}.mlp.down_proj.weight")

            neuron_decoder_layer += 1

    _convert_vision_encoder_state_dict(state_dict, config, vision_config, prefix="vision_model")

    _convert_vision_model_projection_state_dict(state_dict)

    for k in state_dict.keys():
        if "vision_model" in k:
            state_dict[k] = state_dict[k]

    to_remove = []
    for k in state_dict.keys():
        if "text_model.rope.freqs" in k:
            to_remove.append(k)

    for tm in to_remove:
        state_dict.pop(tm)

    gc.collect()
    return state_dict


def _convert_vision_encoder_state_dict(
    state_dict: Dict[str, Any],
    config: InferenceConfig,
    vision_config: InferenceConfig,
    prefix: str,
    return_state_dict: bool = False,
) -> None:
    _convert_gated_pos_emb_state_dict(
        state_dict,
        config,
        vision_config,
        prefix,
    )

    _convert_image_transformer_state_dict(state_dict, prefix, vision_config)

    _convert_tile_pos_emb_state_dict(state_dict, key_id="pre")
    _convert_tile_pos_emb_state_dict(state_dict, key_id="post")

    _convert_vision_encoder_layer_norm_state_dict(state_dict, key_id="pre")
    _convert_vision_encoder_layer_norm_state_dict(state_dict, key_id="post")

    _convert_vision_encoder_conv1_state_dict(state_dict, vision_config)

    if return_state_dict:
        return state_dict


def _convert_gated_pos_emb_state_dict(state_dict, config, vision_config, prefix):
    patch_size = vision_config.patch_size
    image_size = vision_config.image_size
    grid_size = (
        image_size // patch_size,
        image_size // patch_size,
    )
    max_num_tiles = vision_config.max_num_tiles

    neuron_prefix = f"{prefix}.vision_encoder"

    state_dict[f"{neuron_prefix}.class_embedding"] = state_dict.pop(f"{prefix}.class_embedding")

    orig_pos_embed = state_dict.pop(f"{prefix}.gated_positional_embedding.embedding")
    state_dict[f"{neuron_prefix}.gated_positional_embedding.embedding"] = orig_pos_embed
    if orig_pos_embed is not None:
        new_pos_embed = resize_local_position_embedding(orig_pos_embed, grid_size)
        state_dict[f"{neuron_prefix}.positional_embedding"] = new_pos_embed

    ckpt_tile_embedding_key = f"{prefix}.gated_positional_embedding.tile_embedding"
    if f"{ckpt_tile_embedding_key}.weight" not in state_dict:
        global_pos_embed = initialize_global_position_embedding_from_local(
            new_pos_embed,
            grid_size,
            max_num_tiles,
            max_num_tiles,
        )
        state_dict[f"{neuron_prefix}.gated_positional_embedding"] = global_pos_embed
        state_dict[f"{neuron_prefix}.gated_positional_embedding_gate"] = torch.zeros(
            1, dtype=global_pos_embed.dtype
        )
    else:
        state_dict[f"{neuron_prefix}.gated_positional_embedding.tile_embedding.weight"] = (
            state_dict.pop(f"{ckpt_tile_embedding_key}.weight")
        )
        state_dict[f"{neuron_prefix}.gated_positional_embedding.gate"] = state_dict.pop(
            "vision_model.gated_positional_embedding.gate"
        )


def _convert_image_transformer_state_dict(state_dict, prefix, config):
    neuron_prefix = f"{prefix}.vision_encoder"
    # NeuronImageAttention
    for transformer_key, num_layers in [
        ("transformer", config.num_hidden_layers),
        ("global_transformer", config.num_global_layers),
    ]:
        for layer_idx in range(num_layers):
            neuron_transformer_key_prefix = (
                f"{neuron_prefix}.{transformer_key}.resblocks.{layer_idx}"
            )
            ckpt_transformer_key_prefix = f"{prefix}.{transformer_key}.layers.{layer_idx}"
            # attention
            neuron_attn_key_prefix = f"{neuron_transformer_key_prefix}.attn"
            ckpt_attn_key_prefix = f"{ckpt_transformer_key_prefix}.self_attn"
            if config.neuron_config.fused_qkv:
                state_dict[f"{neuron_attn_key_prefix}.qkv_proj.Wqkv.weight"] = torch.cat(
                    [
                        state_dict.pop(f"{ckpt_attn_key_prefix}.q_proj.weight"),
                        state_dict.pop(f"{ckpt_attn_key_prefix}.k_proj.weight"),
                        state_dict.pop(f"{ckpt_attn_key_prefix}.v_proj.weight"),
                    ]
                )
            else:
                state_dict[f"{neuron_attn_key_prefix}.q_proj.weight"] = state_dict.pop(
                    f"{ckpt_attn_key_prefix}.q_proj.weight"
                )
                state_dict[f"{neuron_attn_key_prefix}.k_proj.weight"] = state_dict.pop(
                    f"{ckpt_attn_key_prefix}.k_proj.weight"
                )
                state_dict[f"{neuron_attn_key_prefix}.v_proj.weight"] = state_dict.pop(
                    f"{ckpt_attn_key_prefix}.v_proj.weight"
                )
            state_dict[f"{neuron_attn_key_prefix}.o_proj.weight"] = state_dict.pop(
                f"{ckpt_attn_key_prefix}.o_proj.weight"
            )

            # layernorm
            state_dict[f"{neuron_transformer_key_prefix}.ln_1.bias"] = state_dict.pop(
                f"{ckpt_transformer_key_prefix}.input_layernorm.bias"
            )
            state_dict[f"{neuron_transformer_key_prefix}.ln_1.weight"] = state_dict.pop(
                f"{ckpt_transformer_key_prefix}.input_layernorm.weight"
            )
            state_dict[f"{neuron_transformer_key_prefix}.ln_2.bias"] = state_dict.pop(
                f"{ckpt_transformer_key_prefix}.post_attention_layernorm.bias"
            )
            state_dict[f"{neuron_transformer_key_prefix}.ln_2.weight"] = state_dict.pop(
                f"{ckpt_transformer_key_prefix}.post_attention_layernorm.weight"
            )

            # mlp
            neuron_mlp_key_prefix = f"{neuron_transformer_key_prefix}.mlp"
            ckpt_mlp_key_prefix = f"{ckpt_transformer_key_prefix}.mlp"
            state_dict[f"{neuron_mlp_key_prefix}.c_fc.bias"] = state_dict.pop(
                f"{ckpt_mlp_key_prefix}.fc1.bias"
            )
            state_dict[f"{neuron_mlp_key_prefix}.c_fc.weight"] = state_dict.pop(
                f"{ckpt_mlp_key_prefix}.fc1.weight"
            )
            state_dict[f"{neuron_mlp_key_prefix}.c_proj.bias"] = state_dict.pop(
                f"{ckpt_mlp_key_prefix}.fc2.bias"
            )
            state_dict[f"{neuron_mlp_key_prefix}.c_proj.weight"] = state_dict.pop(
                f"{ckpt_mlp_key_prefix}.fc2.weight"
            )

            # gate
            if transformer_key == "global_transformer":
                state_dict[f"{neuron_transformer_key_prefix}.gate_attn"] = state_dict.pop(
                    f"{ckpt_transformer_key_prefix}.gate_attn"
                )
                state_dict[f"{neuron_transformer_key_prefix}.gate_ffn"] = state_dict.pop(
                    f"{ckpt_transformer_key_prefix}.gate_ffn"
                )


def _convert_tile_pos_emb_state_dict(state_dict, key_id):
    neuron_key_prefix = f"vision_model.vision_encoder.{key_id}_tile_pos_embed"
    ckpt_key_prefix = f"vision_model.{key_id}_tile_positional_embedding"

    state_dict[f"{neuron_key_prefix}.embedding.weight"] = state_dict.pop(
        f"{ckpt_key_prefix}.embedding.weight"
    )

    state_dict[f"{neuron_key_prefix}.gate"] = state_dict.pop(f"{ckpt_key_prefix}.gate")


def _convert_vision_encoder_layer_norm_state_dict(state_dict, key_id):
    state_dict[f"vision_model.vision_encoder.ln_{key_id}.bias"] = state_dict.pop(
        f"vision_model.layernorm_{key_id}.bias"
    )
    state_dict[f"vision_model.vision_encoder.ln_{key_id}.weight"] = state_dict.pop(
        f"vision_model.layernorm_{key_id}.weight"
    )


def _convert_vision_encoder_conv1_state_dict(state_dict, vision_config):
    weights = state_dict.pop("vision_model.patch_embedding.weight")
    state_dict["vision_model.vision_encoder.conv1._linear.weight"] = weights.reshape(
        -1, vision_config.num_channels * vision_config.patch_size * vision_config.patch_size
    )


def _convert_vision_model_projection_state_dict(state_dict):
    state_dict["vision_model.vision_projection.bias"] = state_dict.pop("multi_modal_projector.bias")
    state_dict["vision_model.vision_projection.weight"] = state_dict.pop(
        "multi_modal_projector.weight"
    )
