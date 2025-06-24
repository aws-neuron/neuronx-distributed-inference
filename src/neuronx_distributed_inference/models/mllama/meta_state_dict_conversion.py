import gc
import logging
import math
from typing import Any, Dict

import torch
import torch.nn.functional as F

from neuronx_distributed_inference.models.config import InferenceConfig

from .encoder_utils import (
    initialize_global_position_embedding_from_local,
    resize_global_position_embedding,
    resize_local_position_embedding,
)


def convert_meta_state_dict_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    text_config = config.text_config
    vision_config = config.vision_config
    num_layers = text_config.num_hidden_layers
    num_layers = list(range(num_layers))

    num_xatten_layers = text_config.vision_num_cross_attention_layers
    k = math.ceil(len(num_layers) / num_xatten_layers)

    fusion_schedule = num_layers[::-1][::k][:num_xatten_layers][::-1]

    state_dict["text_model.norm.weight"] = state_dict.pop("text_model.norm.weight")
    state_dict["text_model.lm_head.weight"] = state_dict.pop("text_model.output.weight")
    state_dict["text_model.embed_tokens.weight"] = torch.cat(
        [
            state_dict["text_model.tok_embeddings.weight"],
            state_dict.pop("text_model.learnable_embedding.weight"),
        ],
        dim=0,
    )
    state_dict["text_model.rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )
    xl = 0
    for l in range(text_config.num_hidden_layers):  # noqa: E741
        # attention mapping
        state_dict[f"text_model.layers.{l}.self_attn.input_layernorm.weight"] = state_dict.pop(
            f"text_model.layers.{l}.attention_norm.weight"
        )

        if config.neuron_config.fused_qkv:
            state_dict[f"text_model.layers.{l}.self_attn.self_attn.qkv_proj.Wqkv.weight"] = (
                torch.cat(
                    [
                        state_dict.pop(f"text_model.layers.{l}.attention.wq.weight"),
                        state_dict.pop(f"text_model.layers.{l}.attention.wk.weight"),
                        state_dict.pop(f"text_model.layers.{l}.attention.wv.weight"),
                    ]
                )
            )
        else:
            state_dict[f"text_model.layers.{l}.self_attn.self_attn.qkv_proj.q_proj.weight"] = (
                state_dict.pop(f"text_model.layers.{l}.attention.wq.weight")
            )
            state_dict[f"text_model.layers.{l}.self_attn.self_attn.qkv_proj.k_proj.weight"] = (
                state_dict.pop(f"text_model.layers.{l}.attention.wk.weight")
            )
            state_dict[f"text_model.layers.{l}.self_attn.self_attn.qkv_proj.v_proj.weight"] = (
                state_dict.pop(f"text_model.layers.{l}.attention.wv.weight")
            )
        state_dict[f"text_model.layers.{l}.self_attn.self_attn.o_proj.weight"] = state_dict.pop(
            f"text_model.layers.{l}.attention.wo.weight"
        )

        # feed forward mapping
        state_dict[f"text_model.layers.{l}.self_attn.post_attention_layernorm.weight"] = (
            state_dict.pop(f"text_model.layers.{l}.ffn_norm.weight")
        )
        state_dict[f"text_model.layers.{l}.self_attn.feed_forward.gate_proj.weight"] = (
            state_dict.pop(f"text_model.layers.{l}.feed_forward.w1.weight")
        )
        state_dict[f"text_model.layers.{l}.self_attn.feed_forward.up_proj.weight"] = state_dict.pop(
            f"text_model.layers.{l}.feed_forward.w3.weight"
        )
        state_dict[f"text_model.layers.{l}.self_attn.feed_forward.down_proj.weight"] = (
            state_dict.pop(f"text_model.layers.{l}.feed_forward.w2.weight")
        )

        if l in fusion_schedule:
            state_dict[f"text_model.layers.{l}.xatten.gate_attn"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.gate_attn"
            )[0].view(1)
            state_dict[f"text_model.layers.{l}.xatten.gate_ffwd"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.gate_ffwd"
            )[0].view(1)
            state_dict[f"text_model.layers.{l}.xatten.attention_norm.weight"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.attention_norm.weight"
            )
            state_dict[f"text_model.layers.{l}.xatten.ffn_norm.weight"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.ffn_norm.weight"
            )

            state_dict[f"text_model.layers.{l}.xatten.feed_forward.gate_proj.weight"] = (
                state_dict.pop(f"text_model.cross_attention_layers.{xl}.feed_forward.w1.weight")
            )
            state_dict[f"text_model.layers.{l}.xatten.feed_forward.up_proj.weight"] = (
                state_dict.pop(f"text_model.cross_attention_layers.{xl}.feed_forward.w3.weight")
            )
            state_dict[f"text_model.layers.{l}.xatten.feed_forward.down_proj.weight"] = (
                state_dict.pop(f"text_model.cross_attention_layers.{xl}.feed_forward.w2.weight")
            )
            # inner cross attention layers
            state_dict[f"text_model.layers.{l}.xatten.xatten.q_norm.weight"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.attention.q_norm.weight"
            )
            state_dict[f"text_model.layers.{l}.xatten.xatten.k_norm.weight"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.attention.k_norm.weight"
            )

            state_dict[f"text_model.layers.{l}.xatten.xatten.wq.weight"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.attention.wq.weight"
            )
            state_dict[f"text_model.layers.{l}.xatten.xatten.wk.weight"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.attention.wk.weight"
            )
            state_dict[f"text_model.layers.{l}.xatten.xatten.wv.weight"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.attention.wv.weight"
            )

            state_dict[f"text_model.layers.{l}.xatten.xatten.wo.weight"] = state_dict.pop(
                f"text_model.cross_attention_layers.{xl}.attention.wo.weight"
            )

            xl += 1

    _convert_vision_encoder_state_dict(
        state_dict, vision_config, prefix="vision_model.vision_encoder."
    )
    _convert_tile_pos_emb_state_dict(
        state_dict, vision_config, prefix="vision_model.vision_encoder.pre_tile_pos_embed."
    )
    _convert_tile_pos_emb_state_dict(
        state_dict, vision_config, prefix="vision_model.vision_encoder.post_tile_pos_embed."
    )

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
    prefix: str,
    return_state_dict: bool = False,
) -> None:
    patch_size = config.patch_size
    image_size = config.image_size
    grid_size = (
        image_size // patch_size,
        image_size // patch_size,
    )
    max_num_tiles = config.max_num_tiles

    orig_pos_embed = state_dict.pop(prefix + "positional_embedding")
    if orig_pos_embed is not None:
        new_pos_embed = resize_local_position_embedding(orig_pos_embed, grid_size)
        state_dict[prefix + "positional_embedding"] = new_pos_embed

    if prefix + "gated_positional_embedding" not in state_dict:
        global_pos_embed = initialize_global_position_embedding_from_local(
            new_pos_embed,
            grid_size,
            max_num_tiles,
            max_num_tiles,
        )
        state_dict[prefix + "gated_positional_embedding"] = global_pos_embed
        state_dict[prefix + "gated_positional_embedding_gate"] = torch.zeros(
            1, dtype=global_pos_embed.dtype
        )

    else:
        global_pos_embed = resize_global_position_embedding(
            state_dict[prefix + "gated_positional_embedding"],
            grid_size,
            max_num_tiles,
            max_num_tiles,
        )

        state_dict[prefix + "gated_positional_embedding"] = global_pos_embed

    # NeuronImageAttention
    for transformer_key, num_layers in [
        ("transformer", config.num_hidden_layers),
        ("global_transformer", config.num_global_layers),
    ]:
        for layer_idx in range(num_layers):
            attn_key_prefix = f"{prefix}{transformer_key}.resblocks.{layer_idx}.attn"
            if config.neuron_config.fused_qkv:
                state_dict[f"{attn_key_prefix}.Wqkv.weight"] = torch.cat(
                    [
                        state_dict.pop(f"{attn_key_prefix}.wq.weight"),
                        state_dict.pop(f"{attn_key_prefix}.wk.weight"),
                        state_dict.pop(f"{attn_key_prefix}.wv.weight"),
                    ]
                )
            else:
                state_dict[f"{attn_key_prefix}.q_proj.weight"] = state_dict.pop(
                    f"{attn_key_prefix}.wq.weight"
                )
                state_dict[f"{attn_key_prefix}.k_proj.weight"] = state_dict.pop(
                    f"{attn_key_prefix}.wk.weight"
                )
                state_dict[f"{attn_key_prefix}.v_proj.weight"] = state_dict.pop(
                    f"{attn_key_prefix}.wv.weight"
                )
            state_dict[f"{attn_key_prefix}.o_proj.weight"] = state_dict.pop(
                f"{attn_key_prefix}.wo.weight"
            )

    if return_state_dict:
        return state_dict


def _convert_tile_pos_emb_state_dict(
    state_dict,
    config,
    prefix: str,
):
    num_tiles = config.max_num_tiles
    # load the weights from the checkpoint
    embed = state_dict.pop(prefix + "embedding")
    if embed is not None:
        # reshape the weights to the correct shape
        nt_old, nt_old, _, w = embed.shape
        logging.info(f"Resizing tile embedding from {nt_old}x{nt_old} to {num_tiles}x{num_tiles}")
        embed_new = _dynamic_resize(embed, num_tiles)
        # assign the weights to the module
        state_dict[prefix + "embedding"] = embed_new.contiguous()


def _dynamic_resize(embed: torch.Tensor, num_tiles: int):
    embed = embed.permute(2, 3, 0, 1)

    embed_new = F.interpolate(
        embed,
        size=(num_tiles, num_tiles),
        mode="bilinear",
        align_corners=True,
    )
    # reshape the weights to the correct shape
    embed_new = embed_new.permute(2, 3, 0, 1)
    return embed_new
