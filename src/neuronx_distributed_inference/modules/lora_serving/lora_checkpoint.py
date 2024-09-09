import torch


def is_lora_module(name):
    keywords = ["lora_A", "lora_B"]
    return any(keyword in name for keyword in keywords)


def update_weights_for_lora(model, model_sd):
    # add the base layer of LoraModule to checkpoint
    state_keys = model_sd.keys()
    lora_modules = []
    for name, _ in model.named_modules():
        if ".base_layer" in name:
            lora_modules.append(name.replace(".base_layer", ""))

    base_layer_weights = {}
    for name in lora_modules:
        weight_name = f"{name}.weight"
        lora_weight_name = f"{name}.base_layer.weight"
        if weight_name not in model_sd:
            name_list = weight_name.split(".")
            name_prefix = ".".join(name_list[:3])
            name_suffix = ".".join(name_list[-2:])
            for key in state_keys:
                if key.startswith(name_prefix) and key.endswith(name_suffix):
                    weight_name = key
                    break
        if lora_weight_name not in model_sd:
            base_layer_weights[lora_weight_name] = model_sd[weight_name]
    model_sd.update(base_layer_weights)

    lora_weights = {}
    for name, module in model.named_modules():
        if is_lora_module(name):
            weight_shape = module.get_checkpoint_shape()
            weight_dtype = module.get_weight_dtype()
            weight_name = f"{name}.weight"
            if weight_name not in model_sd:
                lora_weights[weight_name] = torch.zeros(
                    *weight_shape, dtype=weight_dtype, device="cpu"
                )
    model_sd.update(lora_weights)

    return model_sd
