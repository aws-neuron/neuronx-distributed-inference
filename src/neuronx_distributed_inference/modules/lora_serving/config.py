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
        max_cpu_loras: int = 100,
        batch_size: int = 1,
        target_modules: List[str] = None,
        lora_bias: str = "none",
        lora_ckpt_paths: Union[List[str], Dict[str, str]] = None,
        lora_ckpt_paths_cpu: Union[List[str], Dict[str, str]] = None,
        lora_ckpt_json: str = None,
        lora_memory_transpose: bool = True,
        lora_shard_linear_layer: bool = True,
        is_context_encoding: bool = False,
        dynamic_multi_lora: bool = False,
        eviction_policy: str = "lru",
        lfu_decay_period: int = 128,
        base_model_quantized: bool = False,
    ):
        # The maximum number of concurrent LoRA adapters in device memory
        self.max_loras = max_loras
        # The highest LoRA rank that needs to be supported
        self.max_lora_rank = max_lora_rank
        # The maximum number of LoRA adapters stored in CPU memory
        self.max_cpu_loras = max_cpu_loras
        self.batch_size = batch_size
        # List of module names or regex expression of the module names to replace with LoRA.
        self.target_modules = target_modules
        # Bias type for LoRA. Can be 'none', 'all'
        self.lora_bias = lora_bias
        # Enable dynamic loading/unloading of LoRA adapters
        self.dynamic_multi_lora = dynamic_multi_lora
        self.eviction_policy = eviction_policy
        self.lfu_decay_period = lfu_decay_period
        # Checkpoint paths for LoRA adapters
        self.lora_ckpt_json = lora_ckpt_json
        self.lora_ckpt_paths = self.convert_ckpt_paths_to_dict(lora_ckpt_paths)
        self.lora_ckpt_paths_cpu = (self.convert_ckpt_paths_to_dict(lora_ckpt_paths_cpu, is_cpu=True) | self.lora_ckpt_paths) if self.dynamic_multi_lora else {}
        self._check_ckpt_config()
        # Transpose memory layout to optimize inference performance
        self.lora_memory_transpose = lora_memory_transpose
        # Shard the linear layer across TP group to reduce memory consumption
        self.lora_shard_linear_layer = lora_shard_linear_layer
        # Tag for context_encoding_model
        self.is_context_encoding = is_context_encoding
        # Whether the base model is quantized
        self.base_model_quantized = base_model_quantized

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

    def _expand_user_path(self, path):
        return os.path.expanduser(path)

    def _check_valid_path(self, path, adapter_id=None):
        path = self._expand_user_path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LoRA adapter path {path} for adapter ID {adapter_id} is not found. "
                f"Please check lora_ckpt_path and try again."
            )
        return path

    def _check_valid_dir(self, dir):
        if not os.path.isdir(dir):
            raise FileNotFoundError(
                f"The LoRA checkpoint directory {dir} specified in {self.lora_ckpt_json} doesn't exist. "
                f"Please check lora-ckpt-dir and try again."
            )

    def _check_valid_adapter_id(self, adapter_id, ckpt_path_dict):
        # the adapter_id must be unique
        if adapter_id in ckpt_path_dict:
            raise ValueError(
                f"The adapter ID {adapter_id} appears more than once in lora_ckpt_paths. "
                f"Please check lora_ckpt_path and try again."
            )

    def _check_valid_adapter_id_path(self, adapter_id, path, ckpt_path_dict):
        path = self._check_valid_path(path, adapter_id)
        self._check_valid_adapter_id(adapter_id, ckpt_path_dict)
        return path

    def parse_lora_ckpts_from_json(self, is_cpu):
        lora_ckpt_json = self.lora_ckpt_json
        if lora_ckpt_json is None:
            return {}
        lora_ckpt_json = self._check_valid_path(lora_ckpt_json)
        with open(lora_ckpt_json) as f:
            content = json.load(f)
            lora_ckpt_dir = content.get("lora-ckpt-dir", os.getcwd())
            self._check_valid_dir(lora_ckpt_dir)
            lora_ckpt_paths = content["lora-ckpt-paths-cpu"] if is_cpu else content["lora-ckpt-paths"]

        ckpt_path_dict = {}
        for adapter_id, path in lora_ckpt_paths.items():
            lora_ckpt_path = os.path.join(lora_ckpt_dir, path)
            ckpt_path_dict[adapter_id] = self._check_valid_adapter_id_path(adapter_id, lora_ckpt_path, ckpt_path_dict)
        return ckpt_path_dict

    def convert_ckpt_paths_to_dict(self, lora_ckpt_paths, is_cpu=False):
        ckpt_path_dict = self.parse_lora_ckpts_from_json(is_cpu)
        if lora_ckpt_paths is None:
            if len(ckpt_path_dict) == 0:
                if is_cpu:
                    logging.warning("No LoRA CPU adapter IDs and checkpoint paths are initialized.")
                else:
                    logging.warning("No LoRA adapter IDs and checkpoint paths are initialized.")
            return ckpt_path_dict

        if isinstance(lora_ckpt_paths, dict):
            for adapter_id, path in lora_ckpt_paths.items():
                ckpt_path_dict[adapter_id] = self._check_valid_adapter_id_path(adapter_id, path, ckpt_path_dict)
            return ckpt_path_dict

        for ckpt_path in lora_ckpt_paths:
            keyvalue = ckpt_path.split(":")
            adapter_id = keyvalue[0].strip()
            path = keyvalue[1].strip()
            ckpt_path_dict[adapter_id] = self._check_valid_adapter_id_path(adapter_id, path, ckpt_path_dict)
        return ckpt_path_dict

    def _check_ckpt_config(self):
        num_ckpts = len(self.lora_ckpt_paths)
        num_ckpts_cpu = len(self.lora_ckpt_paths_cpu)

        # adjust number of loras based on number of checkpoints
        if self.max_loras < num_ckpts:
            logging.warning(f"Setting the number of LoRA adapters in HBM to {num_ckpts}.")
            self.max_loras = num_ckpts

        if self.dynamic_multi_lora and num_ckpts_cpu > self.max_cpu_loras:
            raise ValueError(
                f"The total number of LoRA checkpoints specified in {self.lora_ckpt_json} is {num_ckpts_cpu}, "
                f"but the maximum number of adapters that can be hosted on CPU is {self.max_cpu_loras}."
            )

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
        if self.lora_ckpt_paths is None and self.lora_ckpt_paths_cpu is None:
            raise ValueError("No LoRA checkpoint paths are set.")

        if not self.dynamic_multi_lora and self.lora_ckpt_paths is None:
            logging.warning("No LoRA adapters will be loaded into device memory.")

        adapters_target_modules = []
        lora_ranks = [self.max_lora_rank]
        lora_ckpt_paths = list(self.lora_ckpt_paths.values()) + list(
            self.lora_ckpt_paths_cpu.values()
        )
        for path in lora_ckpt_paths:
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
