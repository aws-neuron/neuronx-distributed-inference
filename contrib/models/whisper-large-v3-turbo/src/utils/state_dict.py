import torch
import torch.nn.functional as F
from collections import OrderedDict


def convert_hf_state_dict_to_neuron(hf_state_dict, type):
    assert type in ["encoder", "decoder"], "Type must be either 'encoder' or 'decoder'."

    new_state_dict = OrderedDict()

    # First pass: rename keys (skip self-attn Q/K/V which will be fused)
    for name, param in hf_state_dict.items():
        # Self-attention Q/K/V: skip individual keys (fused below)
        if any(k in name for k in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]):
            continue

        # Self-attention output
        if "self_attn.out_proj" in name:
            name = name.replace("self_attn.out_proj", "attn.out")

        # Cross attention layers (not fused, separate Q/K/V)
        elif "encoder_attn.q_proj" in name:
            name = name.replace("encoder_attn.q_proj", "cross_attn.query")
        elif "encoder_attn.k_proj" in name:
            name = name.replace("encoder_attn.k_proj", "cross_attn.key")
        elif "encoder_attn.v_proj" in name:
            name = name.replace("encoder_attn.v_proj", "cross_attn.value")
        elif "encoder_attn.out_proj" in name:
            name = name.replace("encoder_attn.out_proj", "cross_attn.out")

        # LayerNorms
        elif "self_attn_layer_norm" in name:
            name = name.replace("self_attn_layer_norm", "attn_ln")
        elif "final_layer_norm" in name:
            name = name.replace("final_layer_norm", "mlp_ln")
        elif "encoder_attn_layer_norm" in name:
            name = name.replace("encoder_attn_layer_norm", "cross_attn_ln")

        # MLPs
        elif "fc1" in name:
            name = name.replace("fc1", "mlp.up_proj")
        elif "fc2" in name:
            name = name.replace("fc2", "mlp.down_proj")

        # Embedding
        elif "decoder.embed_tokens" in name:
            name = name.replace("decoder.embed_tokens", "decoder.token_embedding")
        elif "decoder.embed_positions" in name:
            name = name.replace("decoder.embed_positions.weight", "decoder.positional_embedding.weight")
        elif "encoder.embed_positions" in name:
            name = name.replace("encoder.embed_positions.weight", "encoder.positional_embedding")

        # Conv
        elif "encoder.conv1" in name:
            name = name.replace("encoder.conv1", "encoder.conv1")
        elif "encoder.conv2" in name:
            name = name.replace("encoder.conv2", "encoder.conv2")

        # Top-level layer norm
        elif name.startswith("encoder.layer_norm"):
            name = name.replace("encoder.layer_norm", "encoder.ln_post")
        elif name.startswith("decoder.layer_norm"):
            name = name.replace("decoder.layer_norm", "decoder.ln")

        # Layers
        name = name.replace("encoder.layers.", "encoder.blocks.")
        name = name.replace("decoder.layers.", "decoder.blocks.")

        prefix = type + "."
        if name.startswith(prefix):
            name = name[len(prefix) :]
            new_state_dict[name] = param

    # Second pass: fuse self-attention Q/K/V into qkv_proj
    _fuse_self_attn_qkv(hf_state_dict, new_state_dict, type)

    return new_state_dict


def _fuse_self_attn_qkv(hf_state_dict, new_state_dict, type):
    """Fuse separate Q/K/V weights and biases into a single qkv_proj."""
    import re

    # Find all layer indices that have self-attention Q/K/V
    layer_type = "encoder.layers" if type == "encoder" else "decoder.layers"
    block_type = "encoder.blocks" if type == "encoder" else "decoder.blocks"
    prefix = type + "."

    layer_indices = set()
    pattern = re.compile(rf"{layer_type}\.(\d+)\.self_attn\.[qkv]_proj\.")
    for name in hf_state_dict:
        m = pattern.search(name)
        if m:
            layer_indices.add(int(m.group(1)))

    for idx in sorted(layer_indices):
        # Fuse weights: cat([q_weight, k_weight, v_weight], dim=0)
        q_w = hf_state_dict[f"{layer_type}.{idx}.self_attn.q_proj.weight"]
        k_w = hf_state_dict[f"{layer_type}.{idx}.self_attn.k_proj.weight"]
        v_w = hf_state_dict[f"{layer_type}.{idx}.self_attn.v_proj.weight"]
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        key_w = f"{block_type}.{idx}.attn.qkv_proj.weight"
        if key_w.startswith(prefix):
            new_state_dict[key_w[len(prefix):]] = fused_w

        # Fuse biases: cat([q_bias, zeros_for_k, v_bias], dim=0)
        # Q has bias, K has no bias, V has bias
        q_b_key = f"{layer_type}.{idx}.self_attn.q_proj.bias"
        v_b_key = f"{layer_type}.{idx}.self_attn.v_proj.bias"
        if q_b_key in hf_state_dict:
            q_b = hf_state_dict[q_b_key]
            v_b = hf_state_dict[v_b_key]
            k_b = torch.zeros_like(q_b)  # K has no bias, use zeros
            fused_b = torch.cat([q_b, k_b, v_b], dim=0)
            key_b = f"{block_type}.{idx}.attn.qkv_proj.bias"
            if key_b.startswith(prefix):
                new_state_dict[key_b[len(prefix):]] = fused_b


def expand_state_dict(state_dict, dims, TP):
    """
    Pad attention heads so that the number of heads is a multiple of TP.
    This is necessary for the model to work correctly with tensor parallelism.
    """
    if dims.n_audio_head % TP == 0:
        # no need to pad
        return state_dict

    new_state_dict = OrderedDict()

    d = dims.n_audio_state  # embedding dim
    head_dim = d // dims.n_audio_head
    n_padded_heads = ((dims.n_audio_head + TP - 1) // TP) * TP
    padded_d = head_dim * n_padded_heads

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            new_state_dict[name] = param
            continue

        shape = param.shape

        # Case 1a: Fused "qkv_proj.weight" —> [3*d, d] → [3*padded_d, d]
        if "qkv_proj.weight" in name:
            if shape == (3 * d, d):
                q_w, k_w, v_w = torch.tensor_split(param, 3, dim=0)
                q_w = F.pad(q_w, (0, 0, 0, padded_d - d))
                k_w = F.pad(k_w, (0, 0, 0, padded_d - d))
                v_w = F.pad(v_w, (0, 0, 0, padded_d - d))
                padded = torch.cat([q_w, k_w, v_w], dim=0)
                new_state_dict[name] = padded
                print(f"Padded {name}: {shape} → {padded.shape}")
                continue

        # Case 1b: Fused "qkv_proj.bias" —> [3*d] → [3*padded_d]
        if "qkv_proj.bias" in name:
            if shape == (3 * d,):
                q_b, k_b, v_b = torch.tensor_split(param, 3, dim=0)
                q_b = F.pad(q_b, (0, padded_d - d))
                k_b = F.pad(k_b, (0, padded_d - d))
                v_b = F.pad(v_b, (0, padded_d - d))
                padded = torch.cat([q_b, k_b, v_b], dim=0)
                new_state_dict[name] = padded
                print(f"Padded {name}: {shape} → {padded.shape}")
                continue

        # Case 2: Cross-attn "query.weight", "key.weight", "value.weight" —> [d, d] → [padded_d, d]
        if any(k in name for k in ["query.weight", "key.weight", "value.weight"]):
            if shape == (d, d):
                padded = F.pad(param, (0, 0, 0, padded_d - d))  # pad rows
                new_state_dict[name] = padded
                print(f"Padded {name}: {shape} → {padded.shape}")
                continue

        # Case 3: Cross-attn "query.bias", "value.bias" —> [d] → [padded_d]
        if any(k in name for k in ["query.bias", "value.bias"]):
            if shape == (d,):
                padded = F.pad(param, (0, padded_d - d))  # pad 1D
                new_state_dict[name] = padded
                print(f"Padded {name}: {shape} → {padded.shape}")
                continue

        # Case 4: "out.weight" —> [d, d] → [d, padded_d]
        if "out.weight" in name:
            if shape == (d, d):
                padded = F.pad(param, (0, padded_d - d, 0, 0))  # pad columns
                new_state_dict[name] = padded
                print(f"Padded {name}: {shape} → {padded.shape}")
                continue

        # Default: unchanged
        new_state_dict[name] = param

    return new_state_dict
