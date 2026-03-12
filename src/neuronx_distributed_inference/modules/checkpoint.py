import json
import os
from typing import Callable, Dict, List, Any, Optional
from shutil import copytree, ignore_patterns

import torch
from huggingface_hub import save_torch_state_dict
from safetensors.torch import load_file

_SAFETENSORS_MODEL_INDEX_FILENAME_JSON = "model.safetensors.index.json"
_SAFETENSORS_MODEL_FILENAME = "model.safetensors"
_PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON = "pytorch_model.bin.index.json"
_PYTORCH_MODEL_BIN_FILENAME = "pytorch_model.bin"

_SAFETENSORS_DIFFUSERS_MODEL_INDEX_FILENAME_JSON = "diffusion_pytorch_model.safetensors.index.json"
_SAFETENSORS_DIFFUSERS_MODEL_FILENAME = "diffusion_pytorch_model.safetensors"


def _is_using_pt2() -> bool:
    pt_version = torch.__version__
    return pt_version.startswith("2.")


def load_state_dict(state_dict_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load state_dict from the given dir where its model weight files are in one of the
    following HF-compatbile formats:
        1. single file in safetensors format
        2. multiple sharded files in safetensors format
        3. single file in torch bin pt format
        4. multiple sharded files in torch bin pt format

    Loading is done in priority of fastest -> slowest (in case multiple variants exist).
    """
    # Standard checkpoint filenames
    state_dict_safetensor_path = os.path.join(state_dict_dir, _SAFETENSORS_MODEL_FILENAME)
    state_dict_safetensor_diffusers_path = os.path.join(
        state_dict_dir, _SAFETENSORS_DIFFUSERS_MODEL_FILENAME
    )
    safetensors_index_path = os.path.join(state_dict_dir, _SAFETENSORS_MODEL_INDEX_FILENAME_JSON)
    state_dict_path = os.path.join(state_dict_dir, _PYTORCH_MODEL_BIN_FILENAME)
    pytorch_model_bin_index_path = os.path.join(
        state_dict_dir, _PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON
    )
    safetensors_diffusers_index_path = os.path.join(
        state_dict_dir, _SAFETENSORS_DIFFUSERS_MODEL_INDEX_FILENAME_JSON
    )

    # Non-sharded safetensors checkpoint
    if os.path.isfile(state_dict_safetensor_path):
        state_dict = load_safetensors(state_dict_dir)
    elif os.path.isfile(state_dict_safetensor_diffusers_path):
        state_dict = load_diffusers_safetensors(state_dict_dir)
    # Sharded safetensors checkpoint
    elif os.path.isfile(safetensors_index_path):
        state_dict = load_safetensors_sharded(state_dict_dir)
    elif os.path.isfile(safetensors_diffusers_index_path):
        state_dict = load_safetensors_sharded_diffusers_model(state_dict_dir)
    # Non-sharded pytorch_model.bin checkpoint
    elif os.path.isfile(state_dict_path):
        state_dict = load_pytorch_model_bin(state_dict_dir)
    # Sharded pytorch model bin
    elif os.path.isfile(pytorch_model_bin_index_path):
        state_dict = load_pytorch_model_bin_sharded(state_dict_dir)
    else:
        raise FileNotFoundError(
            f"Can not find model.safetensors or pytorch_model.bin in {state_dict_dir}"
        )

    return state_dict


def load_safetensors(state_dict_dir: str) -> Dict[str, torch.Tensor]:
    filename = os.path.join(state_dict_dir, _SAFETENSORS_MODEL_FILENAME)
    return load_file(filename)


def load_diffusers_safetensors(state_dict_dir: str) -> Dict[str, torch.Tensor]:
    filename = os.path.join(state_dict_dir, _SAFETENSORS_DIFFUSERS_MODEL_FILENAME)
    return load_file(filename)


def _load_from_files(
    filenames: List[str], state_dict_dir: str, load_func: Callable
) -> Dict[str, torch.Tensor]:
    """
    Load from multiple files, using the provided load_func.

    Args:
        filenames: A list of filenames that contains the state dict.
        state_dict_dir: The dir that contains the files in `filenames`.
        load_func: A function to load file based on different file format.

    Returns:
        dict: The state dict provided by the files.
    """
    state_dict = {}
    for filename in set(filenames):
        part_state_dict_path = os.path.join(state_dict_dir, filename)
        part_state_dict = load_func(part_state_dict_path)

        for key in part_state_dict.keys():
            if key in state_dict:
                raise Exception(
                    f"Found value overriden for key {key} from file "
                    + f"{part_state_dict_path}, please ensure the provided files are correct."
                )

        state_dict.update(part_state_dict)
    return state_dict


def load_safetensors_sharded(state_dict_dir: str) -> Dict[str, torch.Tensor]:
    index_path = os.path.join(state_dict_dir, _SAFETENSORS_MODEL_INDEX_FILENAME_JSON)
    with open(index_path, "r") as f:
        key_to_filename = json.load(f)["weight_map"]

    state_dict = _load_from_files(
        key_to_filename.values(),
        state_dict_dir,
        load_file,
    )
    return state_dict


def load_safetensors_sharded_diffusers_model(state_dict_dir: str) -> Dict[str, torch.Tensor]:
    index_path = os.path.join(state_dict_dir, _SAFETENSORS_DIFFUSERS_MODEL_INDEX_FILENAME_JSON)
    with open(index_path, "r") as f:
        key_to_filename = json.load(f)["weight_map"]

    state_dict = _load_from_files(
        key_to_filename.values(),
        state_dict_dir,
        load_file,
    )
    return state_dict


def _torch_load(file_path: str) -> Dict[str, torch.Tensor]:
    """
    Load torch bin pt file.

    If pytorch2 is available, will load it using mmap mode,
    so it won't cause large memory overhead during loading.
    """
    if _is_using_pt2():
        pt_file = torch.load(file_path, mmap=True, map_location="cpu")
    else:
        pt_file = torch.load(file_path)
    return pt_file


def load_pytorch_model_bin(state_dict_dir: str) -> Dict[str, torch.Tensor]:
    state_dict_path = os.path.join(state_dict_dir, _PYTORCH_MODEL_BIN_FILENAME)
    return _torch_load(state_dict_path)


def load_pytorch_model_bin_sharded(state_dict_dir: str) -> Dict[str, torch.Tensor]:
    index = os.path.join(state_dict_dir, _PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON)
    with open(index, "r") as f:
        key_to_filename = json.load(f)["weight_map"]

    state_dict = _load_from_files(
        key_to_filename.values(),
        state_dict_dir,
        _torch_load,
    )
    return state_dict


def save_state_dict_safetensors(
    state_dict: dict, state_dict_dir: str, max_shard_size: str = "10GB"
):
    """
    Shard and save state dict in safetensors format following HF convention.
    """

    save_torch_state_dict(
        state_dict,
        save_directory=state_dict_dir,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )


def prune_state_dict(state_dict):
    """
    A helper function that deletes None values in the state_dict before saving
    as torch.save does not like None values in the state dict.
    """
    keys_to_delete = []
    for key in state_dict:
        if state_dict[key] is None:
            keys_to_delete.append(key)

    print(f"Will be deleting following keys as its Value is None: {keys_to_delete}")

    pruned_state_dict = {k: v for k, v in state_dict.items() if v is not None}
    return pruned_state_dict


def create_n_layer_checkpoint(
    n: int,
    src_dir: str,
    tgt_dir: str,
    layer_prefix: List[str] = ["layers"],
    layer_config_keys: List[str] = ["num_hidden_layers"],
):
    """
    Create an n-layer checkpoint given a full checkpoint.

    This function creates a smaller checkpoint by extracting only the first n layers
    from a full model checkpoint. It copies all non-weight files (config, tokenizer,
    processor files) and filters the model weights to include only layers with
    index < n. The config.json is automatically updated to reflect the new layer count.

    Args:
        n (int): Number of layers to keep in the new checkpoint. Layers with
            index 0 to n-1 will be retained. Use n=1 to create a minimal
            single-layer checkpoint for testing.
        src_dir (str): Path to the source checkpoint directory containing the
            full model weights, config.json, and tokenizer files.
        tgt_dir (str): Path to the target directory where the n-layer checkpoint
            will be saved. Directory will be created if it doesn't exist.
        layer_prefix (List[str], optional): List of prefix strings used to identify
            layer weights in the state dict keys. The function searches for these
            prefixes followed by a layer index (e.g., "layers.0", "blocks.5").
            Defaults to ["layers"].
        layer_config_keys (List[str], optional): List of config keys to update with
            the new layer count. Supports nested config structures.
            Defaults to ["num_hidden_layers"].

    Returns:
        Dict[str, Any]: The filtered state dictionary containing only weights for
            the first n layers and all non-layer weights (embeddings, output layers, etc.).

    Example:
        Basic usage with default parameters:

        >>> tiny_sd = create_n_layer_checkpoint(
        ...     n=1,
        ...     src_dir="path/to/Llama-3-8B",
        ...     tgt_dir="path/to/Llama-3-8B-tiny"
        ... )

        For vision-language models with multiple layer types:

        >>> tiny_sd = create_n_layer_checkpoint(
        ...     n=2,
        ...     src_dir="path/to/Qwen2-VL-7B",
        ...     tgt_dir="path/to/Qwen2-VL-7B-tiny",
        ...     layer_prefix=["layers", "blocks"],
        ...     layer_config_keys=["num_hidden_layers", "depth"]
        ... )
    """

    # copy all non safetensors and non pt files
    # this will include all model, precessor, and tokenizer config json and .model
    copytree(src_dir, tgt_dir, ignore=ignore_patterns('*.pt', '.safetensors'), dirs_exist_ok=True)

    # load the full state dict
    full_sd = load_state_dict(src_dir)
    print(f"Examining full state dict of keys: {full_sd.keys()}")

    # loop thru the full state dict, only save weight whose layer index < n or have no layer index
    n_layer_sd = {}
    for k, v in full_sd.items():
        layer_idx = find_layer_idx(k, layer_prefix=layer_prefix)
        if layer_idx < n:
            n_layer_sd[k] = v

    # save the n layer state dict
    print(f"Saving n layer state dict of keys: {n_layer_sd.keys()}")
    save_state_dict_safetensors(n_layer_sd, tgt_dir)
    print(f"Finished creating {n} layer checkpoint.")

    # update number of layers in config.json
    config_path = load_config_and_update_layer_num(os.path.join(tgt_dir, "config.json"), n, layer_config_keys)
    print(f"Finished updating config.json key {layer_config_keys} value to {n}.")

    print("----------------------")
    print(f"Please inspect {config_path} and manually update more configs incompatible with the number of layers change.")
    print("----------------------")

    return n_layer_sd


def find_layer_idx(
    key: str,
    layer_prefix: List[str] = ["layers"]
):
    """
    A helper function to find the layer index of a checkpoint key.
    Return -1 if layer_prefix is not in the key.
    """

    delim = key.split(".")

    for i, sub_str in enumerate(delim):
        if sub_str in layer_prefix:
            # next sub_str should be layer index
            try:
                layer_idx = delim[i + 1]
                return int(layer_idx)
            except Exception as e:
                raise ValueError(f"Unable to fetch layer index from key{key} of prefix {layer_prefix}") from e

    # no sub string from key matches layer_prfix
    return -1


def find_and_update_key(d: Dict[str, Any], target_key: str, new_value: Any) -> bool:
    """Recursively search and update a key in a nested dictionary."""
    updated = False

    for key, value in d.items():
        if key == target_key:
            d[key] = new_value
            updated = True
        elif isinstance(value, dict):
            if find_and_update_key(value, target_key, new_value):
                updated = True
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    if find_and_update_key(item, target_key, new_value):
                        updated = True

    return updated


def load_config_and_update_layer_num(
    config_path: str,
    num_layers: int,
    layer_config_keys: Optional[List[str]] = None,
) -> str:
    """
    Load a config.json and update layer-related keys, saving to the same path.

    Args:
        config_path: Path to the config.json file
        num_layers: New value for the number of layers
        layer_config_keys: List of keys to update. Defaults to ["num_hidden_layers"]

    Returns:
        Updated configuration dictionary
    """

    if layer_config_keys is None:
        layer_config_keys = ["num_hidden_layers"]

    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Update all specified keys
    for key in layer_config_keys:
        find_and_update_key(config, key, num_layers)

    # Save to the same path
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    return str(config_path)
