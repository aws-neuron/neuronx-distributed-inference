import json
import os
from typing import Callable, Dict, List

import torch
from safetensors.torch import load_file

_HF_SAFETENSORS_MODEL_INDEX_FILENAME_JSON = "model.safetensors.index.json"


def load_hf_safetensors_sharded(state_dict_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load a sharded HuggingFace model from safetensors format.

    Reads the model.safetensors.index.json file to determine which safetensors
    files contain each weight, then loads and combines all weights into a single state dict.

    Args:
        state_dict_dir: Directory containing the safetensors files and index.json

    Returns:
        Combined state dictionary with all model weights
    """
    index_path = os.path.join(state_dict_dir, _HF_SAFETENSORS_MODEL_INDEX_FILENAME_JSON)
    with open(index_path, "r") as f:
        key_to_filename = json.load(f)["weight_map"]

    state_dict = _load_from_files(
        key_to_filename.values(),
        state_dict_dir,
        load_file,
    )
    return state_dict


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
