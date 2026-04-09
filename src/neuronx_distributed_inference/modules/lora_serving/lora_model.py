from __future__ import annotations

import re
import time

import copy
import torch
import logging
from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.quantization.quantization_layers import BaseQuantizeParallelLinear

from .config import LoraServingConfig
from .lora_checkpoint import LoraCheckpoint
from .lora_module import (
    MultiLoraModuleColumnParallelLinear,
    MultiLoraModuleConv2d,
    MultiLoraModuleEmbedding,
    MultiLoraModuleLinear,
    MultiLoraModuleRowParallelLinear,
)

logger = logging.getLogger("Neuron")


def wrap_model_with_lora(model, config: LoraServingConfig):
    if config is not None:
        LoraModel(model, config)
        setattr(model, "lora_wrapped_model", True)
        return LoraWeightManager(config, model)


class LoraModel(torch.nn.Module):
    def __init__(self, module, config: LoraServingConfig = None) -> None:
        if config is not None:
            super().__init__()
            self.module = module
            self.lora_config = config
            self.inject_adapter()
            setattr(module, "lora_wrapped_model", True)
            if config.base_model_quantized:
                # Virtually register LoRA module as quantized module
                BaseQuantizeParallelLinear.register(MultiLoraModuleColumnParallelLinear)
                BaseQuantizeParallelLinear.register(MultiLoraModuleRowParallelLinear)

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
        # Lazy import to avoid circular dependency
        from neuronx_distributed_inference.modules.attention.gqa import GQA, GroupQueryAttention_QKV

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
                repeats = 1
                source_heads = parent._src_num_key_value_heads
                if parent.num_key_value_heads != parent._src_num_key_value_heads:
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


class LoraWeightManager:
    def __init__(self, config: LoraServingConfig, base_model=None):
        self.base_model = base_model
        self.lora_config = config
        self.lora_checkpoint = LoraCheckpoint(self.lora_config)

        if self.base_model is not None:
            self.print_lora_memory_footprint()

    def _is_lora_module(self, name):
        keywords = ["lora_A", "lora_B"]
        return any(keyword in name for keyword in keywords)

    def update_lora_adapter_ids(self, adapter_ids, seq_ids):
        return seq_ids

    def get_lora_tensors(self):
        if self.base_model is None:
            raise ValueError("Base model is not set for LoraWeightManager.")
        lora_tensors = []
        for name, module in self.base_model.named_modules():
            if self._is_lora_module(name):
                lora_tensors.append(module.weight_active)
        return lora_tensors

    def update_lora_tensors(self, adapter_ids, seq_ids, is_context_encoding, is_continuous_batching=False):
        if self.base_model is None:
            raise ValueError("Base model is not set for LoraWeightManager.")

        if not is_context_encoding:
            return self.get_lora_tensors(), self.update_lora_adapter_ids(adapter_ids, seq_ids)
        lora_tensors = []
        for name, module in self.base_model.named_modules():
            if self._is_lora_module(name):
                gather_dim = -1 if not hasattr(module, "weight_active_partition_dim") else module.weight_active_partition_dim
                if is_continuous_batching:
                    lora_weight = module.update_lora_tensor_for_continuous_batching(adapter_ids, seq_ids, gather_dim)
                else:
                    lora_weight = module.update_lora_tensor(adapter_ids, gather_dim)
                lora_tensors.append(lora_weight)
        return lora_tensors, self.update_lora_adapter_ids(adapter_ids, seq_ids)

    def print_lora_memory_footprint(self):
        def _get_tensor_size(tensor):
            return tensor.numel() * tensor.element_size()

        lora_weight_size = 0
        lora_active_weight_size = 0
        for name, module in self.base_model.named_modules():
            if self._is_lora_module(name):
                lora_weight_size += _get_tensor_size(module.weight)
                lora_active_weight_size += _get_tensor_size(module.weight_active)

        total_memory_footprint = lora_weight_size + lora_active_weight_size
        logging.warning(
            f"The memory footprint for LoRA adapters on each Neuron core is {total_memory_footprint / 1024 / 1024} MB"
        )


class AdapterCacheEntry:
    """
    A class for maintaining LoRA adapter data in each adapter cache entry.

    This class initializes and updates weights for an adapter (for the CPU adapter cache)
    and tracks metrics, including recency and frequency of an adapter access to inform
    the cache eviction policy.
    """

    def __init__(self, position=0):
        self.timestamp = 0
        self.access_count = 0
        self.weights = None
        self.weight_idx = position
        self.in_use = False

    def init_weights_on_cpu(self, num_ranks):
        self.weights = [{} for _ in range(num_ranks)]

    def update_weights(self, rank, module, weight_tensor):
        self.weights[rank][module] = weight_tensor

    def access(self):
        self.timestamp = time.monotonic()
        self.in_use = True
        self.access_count += 1

    def decay(self):
        self.access_count //= 2


class AdapterCache:
    """
    A class for modeling a LoRA adapter cache for dynamic adapter management.

    This class provides functions for adding, removing, accessing, and updating
    adapters tracked in a cache. Both CPU adapters and device adapters are tracked
    and indexed by adapter ID, which maps to an adapter cache entry. Device adapters
    are a subset of CPU adapters and when the device adapter cache becomes full,
    swaps must be performed in order for the device to use requested adapters that
    only reside in the CPU adapter cache. The CPU cache tracks the weights of all
    adapters, whereas the device cache only tracks the adapter IDs and the metrics
    needed for the configurable eviction policy (can be LRU or LFU).
    """

    def __init__(self, capacity, device, adapter_ids=[], eviction_policy="lru", enable_base_model_only=False):
        self.size = 0
        self.capacity = capacity
        self.device = device  # CPU or device
        self.enable_base_model_only = enable_base_model_only  # add a dummy adapter for base model only serving or not

        # map adapter ID to AdapterCacheEntry
        self.map = {}
        self.eviction_policy = eviction_policy
        self.adapter_id_position_mapping = [-1] * self.capacity

        init_adapter_ids = copy.deepcopy(adapter_ids)
        if self.enable_base_model_only:
            # add a dummy adapter to support base model only serving
            init_adapter_ids.insert(0, 0)

        for idx, adapter_id in enumerate(init_adapter_ids):
            self.add_adapter(adapter_id, idx)
            self.update_adapter(adapter_id)

    def init_weights_on_cpu(self, adapter_ids, cpu_weights, modules):
        num_ranks = len(cpu_weights)
        for adapter_id in adapter_ids:
            self.map[adapter_id].init_weights_on_cpu(num_ranks)
            for rank in range(num_ranks):
                for module in modules:
                    # LoRA weights for streaming adapter
                    if cpu_weights[rank][module].shape[0] == 1:
                        weights = cpu_weights[rank][module][i]
                    else:
                        weights = cpu_weights[rank][module][i + self.enable_base_model_only]
                    self.map[adapter_id].update_weights(rank, module, weights)

    def update_adapter(self, adapter_id):
        assert adapter_id in self.map
        assert self.size == len(self.map)
        self.map[adapter_id].access()

        logger.debug(f"Updating entry for adapter ID {adapter_id}. Its access count is now {self.map[adapter_id].access_count}.")

    def access_adapter(self, adapter_id):
        logger.debug(f"Accessing entry for adapter ID {adapter_id}.")
        return adapter_id in self.map

    def add_adapter(self, adapter_id, position):
        self.map[adapter_id] = AdapterCacheEntry(position)
        self.adapter_id_position_mapping[position] = adapter_id
        self.size += 1

        logger.debug(f"Adding adapter ID {adapter_id}. There are now {self.size} entries in the cache.")

    def remove_adapter(self, adapter_id):
        del self.map[adapter_id]
        position = self.adapter_id_position_mapping.index(adapter_id)
        self.adapter_id_position_mapping[position] = -1
        self.size -= 1

        logger.debug(f"Removing adapter ID {adapter_id}. There are now {self.size} entries in the cache.")

    def is_full(self):
        return self.size == self.capacity

    def _evictable_adapters_map(self):
        if self.enable_base_model_only:
            return {k: v for k, v in self.map.items() if k != 0}
        return self.map

    def _evictable_adapters_items(self):
        return self._evictable_adapters_map().items()

    def evict_adapter(self):
        if self.eviction_policy == "lru":
            adapter_id, data = min(self._evictable_adapters_items(), key=lambda item: item[1].timestamp)
        elif self.eviction_policy == "lfu":
            adapter_id, data = min(self._evictable_adapters_items(), key=lambda item: (item[1].access_count, item[1].timestamp))
        else:
            raise ValueError(f"Invalid eviction policy {self.eviction_policy}.")
        self.remove_adapter(adapter_id)

        logger.debug(f"Evicting adapter ID {adapter_id} based on {self.eviction_policy} policy.")

        return adapter_id, data.weight_idx

    def decay_adapters(self):
        logger.debug("Decaying adapter access counts.")

        for adapter_id in self.map:
            adapter_model = self.map[adapter_id]
            adapter_model.decay()

    # Performed for CPU cache only since all adapter weights are stored in CPU map
    def swap_adapters(self, weights, modules, adapter_id, weight_idx):
        logger.debug(f"Swapping the adapter at position {weight_idx} with adapter ID {adapter_id}.")

        start = time.monotonic()
        adapter_cpu_weights = self.map[adapter_id].weights

        for rank in range(len(weights)):
            for module in modules:
                neuron_tensor = weights[rank][module]
                cpu_tensor = adapter_cpu_weights[rank][module]
                neuron_tensor[weight_idx] = cpu_tensor.clone()

        end = time.monotonic()
        logger.debug(f"Swap time: {end-start} sec")

    def get_adapter_ids(self):
        return list(self._evictable_adapters_map().keys())

    def get_adapter_id_position_mapping(self):
        return self.adapter_id_position_mapping

    def get_adapter_id_position(self, adapter_id):
        return self.adapter_id_position_mapping.index(adapter_id)

    def get_adapter_id_positions(self, adapter_ids):
        return [self.adapter_id_position_mapping.index(adapter_id) for adapter_id in adapter_ids]

    def get_size(self):
        return self.size

    def get_swap_position(self):
        # Check if there is space for more adapters on device
        if not self.is_full():
            # Add an entry to device adapter map if there is space
            swap_position = self.adapter_id_position_mapping.index(-1)
        else:
            # Evict an adapter if there is no space
            evicted_adapter_id, swap_position = self.evict_adapter()
            logger.debug(f"Adapter ID {evicted_adapter_id} is evicted from {self.device} because the cache is full.")
        return swap_position


class LoraModelManager:
    def __init__(self, config: LoraServingConfig) -> None:
        self.lora_config = config
        self.lora_checkpoint = LoraCheckpoint(self.lora_config)
        # store the adapter_ids for requests with continuous batching
        self.req_ids_to_adapter_ids_mapping = dict()
        self.lora_modules = []
        self.model = None
        self.init_adapter_id_mapping()

        cpu_adapter_ids = [self.lora_adapter_id_mapping[ckpt] for ckpt in self.lora_checkpoint.ckpt_paths_cpu.keys()]
        device_adapter_ids = [self.lora_adapter_id_mapping[ckpt] for ckpt in self.lora_checkpoint.ckpt_paths.keys()]
        self.cpu_adapter_cache = AdapterCache(self.lora_config.max_cpu_loras, "CPU", cpu_adapter_ids, self.lora_config.eviction_policy, self.lora_config.enable_base_model_only)
        self.device_adapter_cache = AdapterCache(self.lora_config.max_loras, "device", device_adapter_ids, self.lora_config.eviction_policy, self.lora_config.enable_base_model_only)

        # counters for LFU eviction
        self.decay_count = 0
        self.decay_max = self.lora_config.lfu_decay_period

        self.list_adapters()

    def _get_unique_adapter_id(self):
        current_id = self.global_adapter_id
        self.global_adapter_id += 1
        return current_id

    def _add_adapter_mapping(self, adapter_name, adapter_id):
        self.lora_adapter_id_mapping[adapter_name] = adapter_id
        self.lora_adapter_name_mapping[adapter_id] = adapter_name

    def add_adapter_mapping(self, adapter_name):
        if adapter_name not in self.lora_adapter_id_mapping:
            adapter_id = self._get_unique_adapter_id()
            self._add_adapter_mapping(adapter_name, adapter_id)

    # methods to convert adapter_ids in string to indices
    def init_adapter_id_mapping(self):
        # maps adapter name string, e.g. 'lora_id_1', to numerical adapter ID, e.g. 1
        self.lora_adapter_id_mapping = {}
        self.lora_adapter_name_mapping = {}
        self.global_adapter_id = 1 if self.lora_config.enable_base_model_only else 0
        all_ckpt_paths = self.lora_checkpoint.ckpt_paths | self.lora_checkpoint.ckpt_paths_cpu
        if all_ckpt_paths is not None:
            adapter_names = all_ckpt_paths.keys()
            for adapter_name in adapter_names:
                self.add_adapter_mapping(adapter_name)

    # LoRA adapter id conversion from string to index
    def convert_adapter_ids_to_indices(self, adapter_ids, batch_size):
        if adapter_ids is None:
            return torch.zeros((batch_size), dtype=torch.int32)

        if len(adapter_ids) > self.lora_config.max_loras:
            raise ValueError(
                f"The number of LoRA adapter IDs is {len(adapter_ids)}, "
                f"but the maximum number of adapters is {self.lora_config.max_loras}."
            )

        if len(adapter_ids) != batch_size:
            raise ValueError(
                f"The number of LoRA adapter IDs is {len(adapter_ids)}, "
                f"but it should equal the prompts number {batch_size}."
            )
        ret = [self.convert_adapter_id_to_index(id) for id in adapter_ids]
        return torch.tensor(ret, dtype=torch.int32)

    def convert_adapter_id_to_index(self, adapter_ids):
        return self.lora_adapter_id_mapping[adapter_ids]

    # the following methods are for continuous batching
    def add_req_id_to_adapter_id_mapping(self, req_id, adapter_id):
        assert req_id not in self.req_ids_to_adapter_ids_mapping
        self.req_ids_to_adapter_ids_mapping[req_id] = adapter_id

    def add_req_ids_to_adapter_ids_mapping(self, req_ids, adapter_ids):
        for req_id, adapter_id in zip(req_ids, adapter_ids):
            self.add_req_id_to_adapter_id_mapping(req_id, adapter_id)

    # update the mapping from cpu range to device positions
    def _update_req_ids_to_adapter_ids_mapping(self, cpu_adapter_id, device_position):
        for req_id, adapter_id in self.req_ids_to_adapter_ids_mapping.items():
            if adapter_id == cpu_adapter_id:
                self.req_ids_to_adapter_ids_mapping[req_id] = device_position

    def update_req_ids_to_adapter_ids_mapping(self, cpu_adapter_ids, device_positions):
        for cpu_adapter_id, device_position in zip(cpu_adapter_ids, device_positions):
            self._update_req_ids_to_adapter_ids_mapping(cpu_adapter_id, device_position)

    def get_adapter_id_with_req_id(self, req_id):
        return self.req_ids_to_adapter_ids_mapping[req_id]

    def get_adapter_ids_with_req_ids(self, req_ids):
        return [self.req_ids_to_adapter_ids_mapping[req_id] for req_id in req_ids]

    def remove_req_id(self, req_id):
        logger.debug(f"Removing request ID {req_id} from the mapping.")
        del self.req_ids_to_adapter_ids_mapping[req_id]

    def remove_req_ids(self, req_ids):
        for req_id in req_ids:
            self.remove_req_id(req_id)

    # Set up initialization of all adapter weights tracked in CPU memory
    def init_dynamic_multi_lora(self, cpu_weights):
        if not cpu_weights:
            return

        for name in cpu_weights[0].keys():
            if self.lora_checkpoint.is_lora_module(name) and "weight_active" not in name:
                self.lora_modules.append(name)

        self.cpu_adapter_cache.init_weights_on_cpu(
            self.cpu_adapter_cache.get_adapter_ids(),
            cpu_weights,
            self.lora_modules,
        )

        self.model = model  # used for streaming adapters

    def rank_swap(self, rank, weight_idx, weights, adapter_cpu_weights):
        start = time.monotonic()

        for module in self.lora_modules:
            neuron_tensor = weights[rank][module]
            cpu_tensor = adapter_cpu_weights[rank][module]
            neuron_tensor[weight_idx] = cpu_tensor.clone()

        end = time.monotonic()
        logger.debug(f"Rank {rank} swap time: {end-start} sec")

    # Performed for CPU cache only since all adapter weights are stored in CPU map
    def swap_adapters(self, weights, adapter_id, weight_idx):
        logger.debug(f"Swapping the adapter at position {weight_idx} with adapter ID {adapter_id}.")

        start = time.monotonic()
        adapter_cpu_weights = self.cpu_adapter_cache.map[adapter_id].weights

        for rank in range(len(weights)):
            self.rank_swap(rank, weight_idx, weights, adapter_cpu_weights)

        end = time.monotonic()
        logger.debug(f"Swap time: {end-start} sec")

    def insert_device_adapter(self, adapter_id, device_weights):
        # Check if adapter is already on device
        if self.device_adapter_cache.access_adapter(adapter_id):  # hit
            self.device_adapter_cache.update_adapter(adapter_id)
        else:  # miss
            device_swap_position = self.device_adapter_cache.get_swap_position()  # evicts adapter from cache
            logger.info(f"Swap Adapter ID {adapter_id} to position {device_swap_position} on device.")

            # Perform adapter data swap on device
            self.swap_adapters(
                device_weights,
                adapter_id,
                device_swap_position,
            )
            # Update HBM adapter management
            self.device_adapter_cache.add_adapter(adapter_id, device_swap_position)
            self.device_adapter_cache.update_adapter(adapter_id)

    def insert_cpu_adapter(self, adapter_id, cpu_weights=None):
        if self.cpu_adapter_cache.access_adapter(adapter_id):  # hit
            self.cpu_adapter_cache.update_adapter(adapter_id)
        else:  # miss

            # A CPU cache miss can happen due to one of two reasons:
            # 1. Streaming adapter evicted the adapter from the cache (cpu_weights is None)
            # 2. Inserting streaming adapter causes a cold miss

            if not cpu_weights:
                adapter_name = self.lora_adapter_name_mapping[adapter_id]
                cpu_weights = self.lora_checkpoint.load_streaming_ckpt(self.model, adapter_name, self.lora_checkpoint.ckpt_paths_cpu[adapter_name])

            cpu_swap_position = self.cpu_adapter_cache.get_swap_position()  # evicts adapter from cache
            logger.debug(f"Swap Adapter ID {adapter_id} to position {cpu_swap_position} on CPU.")

            # Update CPU adapter management (use sharded checkpoints)
            self.cpu_adapter_cache.add_adapter(adapter_id, cpu_swap_position)
            self.cpu_adapter_cache.init_weights_on_cpu([adapter_id], cpu_weights, self.lora_modules)
            self.cpu_adapter_cache.update_adapter(adapter_id)

    # 1. Add adapter ID to adapter mapping
    # 2. Extract checkpoint information from adapter path
    # 3. Shard checkpoint on CPU
    # 4. Load/swap sharded checkpoint into CPU
    # Note: streaming adapter does not have access to device weights, so cannot
    # load/swap new adapter into device until it is requested
    def add_new_cpu_adapter(self, adapter_name, adapter_path):
        self.add_adapter_mapping(adapter_name)

        # Check if adapter is already loaded into CPU memory
        adapter_id = self.lora_adapter_id_mapping[adapter_name]
        streaming_weights = None
        if not self.cpu_adapter_cache.access_adapter(adapter_id):
            streaming_weights = self.lora_checkpoint.load_streaming_ckpt(self.model, adapter_name, adapter_path)
            if not streaming_weights:
                return False

        self.insert_cpu_adapter(adapter_id, streaming_weights)
        return True

    # Main workflow function for dynamic LoRA adapter swapping
    def dynamic_update_weights_for_lora(self, device_weights, adapter_ids):
        for adapter_id in adapter_ids.tolist():
            # NxDI does not support streaming adapters (loading checkpoints on the fly), so all adapters must be in CPU memory already
            assert self.cpu_adapter_cache.access_adapter(adapter_id)

            # Check if adapter is already on device
            if self.device_adapter_cache.access_adapter(adapter_id):  # hit
                self.device_adapter_cache.update_adapter(adapter_id)
            else:  # miss
                swap_position = self.device_adapter_cache.get_swap_position()
                logger.info(f"Swap Adapter ID {adapter_id} to position {swap_position}.")

                # Perform adapter data swap
                self.cpu_adapter_cache.swap_adapters(
                    device_weights,
                    self.lora_modules,
                    adapter_id,
                    swap_position,
                )
                # Update HBM adapter management
                self.device_adapter_cache.add_adapter(adapter_id, swap_position)

        # Increment decay count if decay period has not been reached
        # otherwise, decay access counts
        if self.decay_count < self.decay_max:
            self.decay_count += 1
        else:
            self.cpu_adapter_cache.decay_adapters()
            self.device_adapter_cache.decay_adapters()
            self.decay_count = 0

        # convert the adapter_ids on CPU range to their positions on device
        adapter_positions = self.device_adapter_cache.get_adapter_id_positions(adapter_ids)
        # update the mapping for token generation phase with continuous batching
        self.update_req_ids_to_adapter_ids_mapping(adapter_ids.tolist(), adapter_positions)
        return torch.tensor(adapter_positions, dtype=adapter_ids.dtype)

    # LoRA APIs in vLLM
    # NxDI does not support streaming adapters (loading checkpoints on the fly), so all adapters must be in CPU memory already
    def add_adapter(self, lora_request) -> bool:
        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        self.cpu_adapter_cache.remove_adapter(adapter_id)
        return True

    def pin_adapter(self, lora_id: int) -> bool:
        return True

    def list_adapters(self):
        def _list_adapters(adapter_ids, mapping, ckpt_paths, device="CPU"):
            for adapter_id in adapter_ids:
                adapter_id_string = mapping[adapter_id]
                logger.info(f"Adapter ID: {adapter_id_string} on {device}, checkpoint path: {ckpt_paths[adapter_id_string]}.")

        adapter_id_string_mapping = {}
        for adapter_id_string, adapter_id_int in self.lora_adapter_id_mapping.items():
            adapter_id_string_mapping[adapter_id_int] = adapter_id_string

        _list_adapters(
            self.cpu_adapter_cache.get_adapter_ids(),
            adapter_id_string_mapping,
            self.lora_checkpoint.ckpt_paths_cpu,
            device="CPU",
        )
        _list_adapters(
            self.device_adapter_cache.get_adapter_ids(),
            adapter_id_string_mapping,
            self.lora_checkpoint.ckpt_paths,
            device="HBM",
        )
        return set(self.lora_adapter_id_mapping.keys())
