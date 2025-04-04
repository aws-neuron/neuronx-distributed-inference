import json
import logging
import os
from typing import Dict, List, Union

from neuronx_distributed_inference.modules.checkpoint import _torch_load, load_file


class LoraServingConfig:
    def __init__(
        self,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        max_loras_on_cpu: int = 2,
        target_modules: List[str] = None,
        lora_bias: str = "none",
        lora_ckpt_paths: Union[List[str], Dict[str, str]] = None,
        lora_memory_transpose: bool = True,
        lora_shard_linear_layer: bool = False,
    ):
        # The maximum number of concurrent LoRA adapters in device memory
        self.max_loras = max_loras
        # The highest LoRA rank that needs to be supported
        self.max_lora_rank = max_lora_rank
        # The maximum number of LoRA adapters stored in CPU memory
        self.max_loras_on_cpu = max_loras_on_cpu
        # List of module names or regex expression of the module names to replace with LoRA.
        self.target_modules = target_modules
        # Bias type for LoRA. Can be 'none', 'all'
        self.lora_bias = lora_bias
        # Checkpoint paths for LoRA adapters
        self.lora_ckpt_paths = self.convert_ckpt_paths_to_dict(lora_ckpt_paths)
        # Transpose memory layout to optimize inference performance
        self.lora_memory_transpose = lora_memory_transpose
        # Shard the linear layer across TP group to reduce memory consumption
        self.lora_shard_linear_layer = lora_shard_linear_layer

        lora_config_from_ckpt = self.get_lora_config_from_ckpt_paths()
        target_modules = lora_config_from_ckpt["target_modules"]
        lora_rank = lora_config_from_ckpt["max_lora_rank"]
        if self.target_modules is None or not set(target_modules).issubset(
            set(self.target_modules)
        ):
            logging.warning(
                f"Setting target modules to {target_modules} based on the LoRA configurations in checkpoint paths."
            )
            self.target_modules = target_modules
        if self.max_lora_rank < lora_rank:
            logging.warning(
                f"Setting max_lora_rank to {lora_rank} based on the LoRA configurations in checkpoint paths. "
                f"This is greater than the specified max_lora_rank: {self.max_lora_rank}."
            )
            self.max_lora_rank = lora_rank

    def convert_ckpt_paths_to_dict(self, lora_ckpt_paths):
        def _check_valid_ckpt_path(adapter_id, path, ckpt_path_dict):
            # the adapter_id must be unique
            if adapter_id in ckpt_path_dict:
                raise ValueError(
                    f"The adapter ID {adapter_id} appears more than once in lora_ckpt_paths. "
                    f"Please check lora_ckpt_path and try again."
                )
            path = os.path.expanduser(path)
            # we assume the LoRA adapter checkpoints are stored at local
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"LoRA adapter path {path} for adapter ID {adapter_id} is not found. "
                    f"Please check lora_ckpt_path and try again."
                )
            return path

        ckpt_path_dict = {}
        if lora_ckpt_paths is None:
            logging.warning("No LoRA adapter IDs and checkpoint paths are initialized.")
            return ckpt_path_dict

        if isinstance(lora_ckpt_paths, dict):
            for adapter_id, path in lora_ckpt_paths.items():
                path = _check_valid_ckpt_path(adapter_id, path, ckpt_path_dict)
                ckpt_path_dict[adapter_id] = path
            return ckpt_path_dict

        for ckpt_path in lora_ckpt_paths:
            keyvalue = ckpt_path.split(":")
            adapter_id = keyvalue[0].strip()
            path = keyvalue[1].strip()
            path = _check_valid_ckpt_path(adapter_id, path, ckpt_path_dict)
            ckpt_path_dict[adapter_id] = path
        return ckpt_path_dict

    def _extract_lora_config(self, lora_adapter_config):
        if lora_adapter_config is None:
            return [], self.max_lora_rank
        target_modules = lora_adapter_config["target_modules"]
        lora_rank = (
            lora_adapter_config["r"]
            if "r" in lora_adapter_config
            else lora_adapter_config["lora_rank"]
        )
        return target_modules, lora_rank

    def _extract_lora_config_from_folder(self, folder):
        if "adapter_config.json" in os.listdir(folder):
            with open(os.path.join(folder, "adapter_config.json")) as f:
                lora_adapter_config = json.load(f)
                target_modules, lora_rank = self._extract_lora_config(lora_adapter_config)
        else:
            raise FileNotFoundError(f"No LoRA configuration json file is found in {folder}")
        return target_modules, lora_rank

    def _extract_lora_config_from_file(self, filename):
        if filename.endswith(".safetensors"):
            state_dict = load_file(filename)
        elif filename.endswith(".bin") or filename.endswith(".pt"):
            state_dict = _torch_load(filename)
        else:
            raise FileNotFoundError(f"Invalid checkpoint filename {filename} for LoRA adapter.")

        lora_adapter_config = state_dict.get("lora_config")
        return self._extract_lora_config(lora_adapter_config)

    def get_lora_config_from_ckpt_paths(self):
        if self.lora_ckpt_paths is None:
            raise ValueError("No LoRA checkpoint paths are set.")

        adapters_target_modules = []
        lora_ranks = [self.max_lora_rank]
        for path in self.lora_ckpt_paths.values():
            if os.path.isdir(path):
                target_modules, lora_rank = self._extract_lora_config_from_folder(path)
            else:
                target_modules, lora_rank = self._extract_lora_config_from_file(path)
            adapters_target_modules.append(target_modules)
            lora_ranks.append(lora_rank)
        target_modules_union = set()
        for target_modules in adapters_target_modules:
            target_modules_union.update(target_modules)
        target_modules = list(target_modules_union)
        return {
            "target_modules": target_modules,
            "max_lora_rank": max(lora_ranks),
        }
