import json
import os
from typing import Callable, Dict, List

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
