def is_before_nope_layer(config, layer_idx):
    num_layers = len(config.no_rope_layers)
    if layer_idx + 1 < num_layers:
        return config.no_rope_layers[layer_idx + 1] == 0
    return False


def is_after_nope_layer(config, layer_idx):
    if layer_idx == 0:
        return True
    else:
        return config.no_rope_layers[layer_idx - 1] == 0
