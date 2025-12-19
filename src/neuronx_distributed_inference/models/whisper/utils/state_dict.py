import torch
import torch.nn.functional as F
from collections import OrderedDict


def convert_hf_state_dict_to_neuron(hf_state_dict, type):
    assert type in ["encoder", "decoder"], "Type must be either 'encoder' or 'decoder'."

    new_state_dict = OrderedDict()

    for name, param in hf_state_dict.items():
        # Attention layers
        if "self_attn.q_proj" in name:
            name = name.replace("self_attn.q_proj", "attn.query")
        elif "self_attn.k_proj" in name:
            name = name.replace("self_attn.k_proj", "attn.key")
        elif "self_attn.v_proj" in name:
            name = name.replace("self_attn.v_proj", "attn.value")
        elif "self_attn.out_proj" in name:
            name = name.replace("self_attn.out_proj", "attn.out")

        # Cross attention layers
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

    return new_state_dict


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

        # Case 1: "query.weight", "key.weight", "value.weight" —> [d, d] → [padded_d, d]
        if any(k in name for k in ["query.weight", "key.weight", "value.weight"]):
            if shape == (d, d):
                padded = F.pad(param, (0, 0, 0, padded_d - d))  # pad rows
                new_state_dict[name] = padded
                print(f"Padded {name}: {shape} → {padded.shape}")
                continue

        # Case 2: "query.bias", "value.bias" —> [d] → [padded_d]
        if any(k in name for k in ["query.bias", "value.bias"]):
            if shape == (d,):
                padded = F.pad(param, (0, padded_d - d))  # pad 1D
                new_state_dict[name] = padded
                print(f"Padded {name}: {shape} → {padded.shape}")
                continue

        # Case 3: "out.weight" —> [d, d] → [d, padded_d]
        if "out.weight" in name:
            if shape == (d, d):
                padded = F.pad(param, (0, padded_d - d, 0, 0))  # pad columns
                new_state_dict[name] = padded
                print(f"Padded {name}: {shape} → {padded.shape}")
                continue

        # Default: unchanged
        new_state_dict[name] = param

    return new_state_dict
