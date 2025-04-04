from __future__ import annotations

import re

import torch
from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.modules.attention.gqa import GQA, GroupQueryAttention_QKV

from .config import LoraServingConfig
from .lora_module import (
    MultiLoraModuleColumnParallelLinear,
    MultiLoraModuleConv2d,
    MultiLoraModuleEmbedding,
    MultiLoraModuleLinear,
    MultiLoraModuleRowParallelLinear,
)


def wrap_model_with_lora(model, config: LoraServingConfig):
    if config is not None:
        LoraModel(model, config)
        setattr(model, "lora_wrapped_model", True)


class LoraModel(torch.nn.Module):
    def __init__(self, module, config: LoraServingConfig = None) -> None:
        if config is not None:
            super().__init__()
            self.module = module
            self.lora_config = config
            self.inject_adapter()
            setattr(module, "lora_wrapped_model", True)

    def inject_adapter(self):
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers.
        It involves the following steps:
            Step 1: set the list of target modules rules in wildcard for LoRA injection
            Step 2: For each module in the base model, check if it matches any target module rules. If so
            Step 3: Create a LoraLayer for this module and replace it with the LoraLayer
        """
        lora_config = self.lora_config
        if lora_config.target_modules is None:
            raise ValueError("Target modules are not set for the base model.")

        is_target_modules_in_base_model = False
        key_list = self.get_leaf_module_names()

        for key in key_list:
            if not self._check_target_module_exists(key):
                continue
            is_target_modules_in_base_model = True
            parent, target, target_name = self._get_submodules(key)
            self._create_and_replace(target, target_name, parent, current_key=key)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def get_leaf_module_names(self):
        r"""
        Return the leaf module names.
        The keys of module.named_modules() may include non-leaf module names.
        For example, both "self_attn.o_proj" and "self_attn.o_proj.o_proj" are included for Llama2, but only the leaf module "self_attn.o_proj.o_proj" needs LoRA.
        """
        key_list = [key for key, _ in self.module.named_modules()]
        key_list = sorted(key_list, key=len, reverse=True)
        result = []
        for s in key_list:
            if not any(other_s.startswith(s) for other_s in result):
                result.append(s)
        return result

    def _get_submodules(self, key):
        module = self.module
        target_name = key.split(".")[-1]
        parent = module.get_submodule(".".join(key.split(".")[:-1]))
        target = module.get_submodule(key)
        return parent, target, target_name

    def _check_target_module_exists(self, key):
        r"""A helper method to check if the passed module's key name matches any of the target modules.

        Args:
            key (`str`): A key to search any matches in config

        Returns:
            `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
            None if no match found
        """
        config = self.lora_config
        if isinstance(config.target_modules, str):
            target_module_found = re.fullmatch(config.target_modules, key)
        elif key in config.target_modules:
            # this module is specified directly in target_modules
            target_module_found = True
        else:
            target_module_found = any(
                key.endswith(f".{target_key}") for target_key in config.target_modules
            )

        return target_module_found

    def _create_and_replace(
        self,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        new_module = self._create_new_module(parent, target, current_key)
        self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state

    def _create_new_module(self, parent, target, current_key):
        r"""
        Create the corresponding LoraLayer according to its module type, such as torch.nn.Linear and torch.nn.Embedding.
        """
        lora_config = self.lora_config
        lora_adapters = None
        # check basic module types
        if isinstance(target, (torch.nn.Embedding, ParallelEmbedding)):
            lora_adapters = MultiLoraModuleEmbedding(target, lora_config)
        elif isinstance(target, torch.nn.Linear):
            lora_adapters = MultiLoraModuleLinear(target, lora_config)
        elif isinstance(target, torch.nn.Conv2d):
            lora_adapters = MultiLoraModuleConv2d(target, lora_config)
        elif isinstance(target, ColumnParallelLinear):
            keywords = [".k_proj", ".v_proj"]
            # pass the kv replication information to LoRA module and LoRA layer
            if isinstance(parent, GroupQueryAttention_QKV) and any(
                key in current_key for key in keywords
            ):
                # the calculation of repeats is based on gqa.py
                source_heads = parent._src_num_key_value_heads
                if parent.sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
                    repeats = parent.tp_degree // source_heads
                elif parent.sharding_strategy == GQA.CONVERT_TO_MHA:
                    repeats = parent._src_num_attention_heads // source_heads
                lora_adapters = MultiLoraModuleColumnParallelLinear(
                    target, lora_config, (source_heads, repeats)
                )
            else:
                lora_adapters = MultiLoraModuleColumnParallelLinear(target, lora_config)
        elif isinstance(target, RowParallelLinear):
            lora_adapters = MultiLoraModuleRowParallelLinear(target, lora_config)

        if lora_adapters is None:
            # no module could be matched
            raise ValueError(
                f"""Target module {target} is not supported. Currently, only the following modules are supported: "
                    torch.nn.Linear,
                    torch.nn.Embedding,
                    torch.nn.Conv2d,
                    nxd.parallel_layers.ColumnParallelLinear,
                    nxd.parallel_layers.RowParallelLinear,
                    nxd.parallel_layers.ParallelEmbedding,
                """
            )
        return lora_adapters
