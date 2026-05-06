"""
HF checkpoint -> Neuron subgraph weight mapping for SmolVLA.

The lerobot/smolvla_base checkpoint has this top-level layout (500 keys):

    model.vlm_with_expert.vlm.model.vision_model.*       (SigLIP + post_layernorm)
    model.vlm_with_expert.vlm.model.connector.modality_projection.proj.*
    model.vlm_with_expert.vlm.model.text_model.*         (16-layer SmolLM)
    model.vlm_with_expert.lm_expert.*                    (16-layer expert)
    model.action_in_proj / action_out_proj
    model.action_time_mlp_in / action_time_mlp_out
    model.state_proj
    model.vlm_with_expert.vlm.lm_head.weight             (unused at inference)

This file slices the flat HF state-dict into three per-subgraph state-dicts
matching the Neuron module trees.

Sharded outputs:
    vision_state_dict   -> SmolVLAVisionEncoder
    prefix_state_dict   -> SmolVLAPrefixModel
    denoise_state_dict  -> SmolVLADenoiseStep
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file

import config_constants as C


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_hf_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load the lerobot/smolvla_base safetensors into a flat dict."""
    sd_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.isfile(sd_path):
        raise FileNotFoundError(f"No model.safetensors at {sd_path}")
    return load_file(sd_path)


def split_hf_state_dict(
    hf_sd: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return (vision_sd, prefix_sd, denoise_sd) — three per-subgraph dicts."""
    return (
        _build_vision_sd(hf_sd),
        _build_prefix_sd(hf_sd),
        _build_denoise_sd(hf_sd),
    )


# ---------------------------------------------------------------------------
# Vision encoder mapping
# ---------------------------------------------------------------------------

def _build_vision_sd(hf: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    HF prefix:  model.vlm_with_expert.vlm.model.vision_model.<...>
                model.vlm_with_expert.vlm.model.connector.modality_projection.proj.<...>
    Neuron:     vision_model.<...>                  (SigLIP)
                connector.modality_projection_proj.<...>   (renamed: '.' to '_')
    """
    out: Dict[str, torch.Tensor] = {}
    vp = "model.vlm_with_expert.vlm.model.vision_model."
    cp = "model.vlm_with_expert.vlm.model.connector.modality_projection.proj."

    # --- vision tower ---
    for k, v in hf.items():
        if k.startswith(vp):
            tail = k[len(vp):]
            # encoder.layers.N.<...>  ->  layers.N.<...>
            tail = tail.replace("encoder.layers.", "layers.")
            # embeddings.patch_embedding.* / embeddings.position_embedding.*
            #     ->  patch_embedding.*    / position_embedding.*
            tail = tail.replace("embeddings.", "")
            out[f"vision_model.{tail}"] = v

    # --- connector ---
    for k, v in hf.items():
        if k.startswith(cp):
            tail = k[len(cp):]                     # 'weight'
            out[f"connector.modality_projection_proj.{tail}"] = v
    return out


# ---------------------------------------------------------------------------
# Prefix (VLM text decoder) mapping
# ---------------------------------------------------------------------------

def _build_prefix_sd(hf: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    HF prefix:  model.vlm_with_expert.vlm.model.text_model.<...>
                model.state_proj.<...>
    Neuron:     embed_tokens.weight
                layers.N.<llama-keys>
                norm.weight
                state_proj.<...>
    """
    out: Dict[str, torch.Tensor] = {}
    tp = "model.vlm_with_expert.vlm.model.text_model."
    for k, v in hf.items():
        if k.startswith(tp):
            tail = k[len(tp):]
            out[tail] = v
    # state_proj
    out["state_proj.weight"] = hf["model.state_proj.weight"]
    out["state_proj.bias"]   = hf["model.state_proj.bias"]
    return out


# ---------------------------------------------------------------------------
# Denoise (action expert) mapping
# ---------------------------------------------------------------------------

def _build_denoise_sd(hf: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    HF prefix:  model.vlm_with_expert.lm_expert.<...>
                model.action_in_proj / action_out_proj
                model.action_time_mlp_in / action_time_mlp_out
    Neuron:     layers.N.<...>     (renamed: self_attn.<x>_proj -> <x>_proj at layer level)
                norm.weight
                action_in_proj.<...>
                action_out_proj.<...>
                action_time_mlp_in.<...>
                action_time_mlp_out.<...>
    """
    out: Dict[str, torch.Tensor] = {}
    ep = "model.vlm_with_expert.lm_expert."

    for k, v in hf.items():
        if k.startswith(ep):
            tail = k[len(ep):]
            # Strip the "self_attn." segment so attention projections sit
            # directly on the layer module (matches the flatter layer module
            # _ExpertSelfAttnLayer / _ExpertCrossAttnLayer).
            tail = tail.replace("self_attn.", "")
            out[tail] = v

    # action / time MLP / out_proj
    for name in ("action_in_proj", "action_out_proj",
                 "action_time_mlp_in", "action_time_mlp_out"):
        out[f"{name}.weight"] = hf[f"model.{name}.weight"]
        out[f"{name}.bias"]   = hf[f"model.{name}.bias"]

    return out
