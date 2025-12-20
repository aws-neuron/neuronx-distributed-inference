# Standard Library
import unittest
from unittest.mock import Mock, patch

import neuronx_distributed as nxd
import torch

from neuronx_distributed_inference.modules.lora_serving.config import LoraServingConfig
from neuronx_distributed_inference.modules.lora_serving.lora_model import AdapterCacheEntry, AdapterCache, LoraModelManager

class DynamicLoraConfig:
    def __init__(self):
        self.ranks = 2
        self.modules = ["lora_A", "lora_B"]
        self.num_cpu_adapters = 5
        self.num_device_adapters = 2
        self.rows = 4096
        self.cols = 64

        self.cpu_adapter_ids = list(range(self.num_cpu_adapters))
        self.device_adapter_ids = list(range(self.num_device_adapters))

        self.lora_config = LoraServingConfig(
            max_loras=3,
            max_cpu_loras=5,
            max_lora_rank=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            dynamic_multi_lora=True,
            eviction_policy = "lru",
            lfu_decay_period = 100
        )

        self.device_weights = [{} for _ in range(self.ranks)]
        self.cpu_weights = [{} for _ in range(self.ranks)]
        self.init_weights()

    def init_weights(self):
        for i in range(self.ranks):
            for module in self.modules:
                self.device_weights[i][module] = torch.rand(
                    (self.num_device_adapters, self.rows, self.cols), dtype=torch.bfloat16
                )
                self.cpu_weights[i][module] = torch.rand(
                    (self.num_cpu_adapters, self.rows, self.cols), dtype=torch.bfloat16
                )


config = DynamicLoraConfig()

class TestAdapterCacheEntry(unittest.TestCase):
    adapter_cache_entry = AdapterCacheEntry(position=0)

    def test_init(self):
        self.adapter_cache_entry = AdapterCacheEntry(position=0)
        assert self.adapter_cache_entry.timestamp == 0
        assert self.adapter_cache_entry.access_count == 0
        assert not self.adapter_cache_entry.weights
        assert self.adapter_cache_entry.weight_idx == 0
        assert not self.adapter_cache_entry.in_use

    def test_init_weights_on_cpu(self):
        self.adapter_cache_entry.init_weights_on_cpu(config.ranks)
        assert len(self.adapter_cache_entry.weights) == config.ranks
        assert all(isinstance(w, dict) for w in self.adapter_cache_entry.weights)

    def test_update_weights(self):
        self.adapter_cache_entry.init_weights_on_cpu(config.ranks)
        test_tensor = torch.tensor([1, 2, 3])
        self.adapter_cache_entry.update_weights(0, "module1", test_tensor)
        assert torch.equal(self.adapter_cache_entry.weights[0]["module1"], test_tensor)

    def test_access(self):
        with patch('time.monotonic', return_value=123.45):
            self.adapter_cache_entry.access()
            assert self.adapter_cache_entry.timestamp == 123.45
            assert self.adapter_cache_entry.in_use
            assert self.adapter_cache_entry.access_count == 1

    def test_decay(self):
        self.adapter_cache_entry.access_count = 10
        self.adapter_cache_entry.decay()
        assert self.adapter_cache_entry.access_count == 5


class TestAdapterCache(unittest.TestCase):
    cache = AdapterCache(capacity=config.lora_config.max_cpu_loras, adapter_ids=config.cpu_adapter_ids)

    def test_init(self):
        assert self.cache.capacity == config.lora_config.max_cpu_loras
        assert self.cache.size == config.num_cpu_adapters
        assert len(self.cache.map) == config.num_cpu_adapters
        assert self.cache.adapter_id_position_mapping == config.cpu_adapter_ids

    def test_init_weights_on_cpu(self):
        self.cache.init_weights_on_cpu(config.cpu_adapter_ids, config.cpu_weights, config.modules)
        for adapter_id in config.cpu_adapter_ids:
            assert(self.cache.map[adapter_id].weights)
            assert all(isinstance(w, dict) for w in self.cache.map[adapter_id].weights)

    def test_update_adapter(self):
        new_cache = AdapterCache(capacity=4, adapter_ids=[0, 1])
        new_cache.update_adapter(0)
        assert new_cache.map[0].in_use
        assert new_cache.map[0].access_count == 1

    def test_access_adapter(self):
        for adapter_id in config.cpu_adapter_ids:
            assert(self.cache.access_adapter(adapter_id))

    def test_add_adapter(self):
        new_cache = AdapterCache(capacity=4, adapter_ids=[0, 1, 2])
        new_cache.add_adapter(4, 3)
        assert new_cache.size == new_cache.capacity
        assert 4 in new_cache.map
        assert new_cache.adapter_id_position_mapping[3] == 4

        new_cache.add_adapter(5, 2)
        assert 5 in new_cache.map
        assert new_cache.adapter_id_position_mapping[2] == 5

    def test_remove_adapter(self):
        new_cache = AdapterCache(capacity=4, adapter_ids=[0, 1, 2])
        new_cache.remove_adapter(2)
        assert new_cache.size == 2
        assert 2 not in new_cache.map
        assert new_cache.adapter_id_position_mapping[2] == -1

    def test_is_full(self):
        assert self.cache.is_full()

    def test_evict_adapter_lru(self):
        new_cache = AdapterCache(capacity=4, adapter_ids=[0, 1])
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 100
            new_cache.map[0].access()
            mock_time.return_value = 200
            new_cache.map[1].access()
            
            evicted_id, weight_idx = new_cache.evict_adapter()
            assert evicted_id == 0
            assert weight_idx == 0

    def test_evict_adapter_lfu(self):
        new_cache = AdapterCache(capacity=4, adapter_ids=[0, 1], eviction_policy="lfu")
        new_cache.map[0].access()
        new_cache.map[1].access()
        new_cache.map[1].access()
        
        evicted_id, weight_idx = new_cache.evict_adapter()
        assert evicted_id == 0
        assert weight_idx == 0

    def test_decay_adapters(self):
        self.cache.map[0].access_count = 10
        self.cache.map[1].access_count = 6
        self.cache.decay_adapters()
        assert self.cache.map[0].access_count == 5
        assert self.cache.map[1].access_count == 3

    def test_swap_adapters(self):
        new_cache = AdapterCache(capacity=config.lora_config.max_cpu_loras, adapter_ids=config.cpu_adapter_ids)
        new_cache.init_weights_on_cpu(config.cpu_adapter_ids, config.cpu_weights, config.modules)
        new_cache.swap_adapters(config.device_weights, config.modules, 2, 0)

        for rank in range(config.ranks):
            for module in config.modules:
                device_tensor = config.device_weights[rank][module]
                cpu_tensor = config.cpu_weights[rank][module]
                assert torch.equal(device_tensor[0], cpu_tensor[2])

    def test_get_adapter_ids(self):
        adapter_ids = self.cache.get_adapter_ids()
        assert adapter_ids == config.cpu_adapter_ids

    def test_get_adapter_id_position_mapping(self):
        assert self.cache.get_adapter_id_position_mapping() == config.cpu_adapter_ids

    def test_get_adapter_id_position(self):
        for adapter_id in config.cpu_adapter_ids:
            assert(self.cache.get_adapter_id_position(adapter_id) == adapter_id)

    def test_get_adapter_id_positions(self):
        assert self.cache.get_adapter_id_positions(config.cpu_adapter_ids) == config.cpu_adapter_ids

    def test_get_size(self):
        assert self.cache.get_size() == config.num_cpu_adapters

    def test_get_swap_position(self):
        new_cache = AdapterCache(capacity=2, adapter_ids=[0, 1])
        assert new_cache.get_swap_position() == 0


class TestLoraModelManager(unittest.TestCase):
    manager = LoraModelManager(config.lora_config)

    test_ckpt_paths = {
        "adapter1": "path/to/adapter1",
        "adapter2": "path/to/adapter2",
        
    }
    test_ckpt_paths_cpu = {
        "adapter3": "path/to/adapter3",
        "adapter4": "path/to/adapter4",
        "adapter5": "path/to/adapter5"
    }

    with patch('neuronx_distributed_inference.modules.lora_serving.lora_checkpoint.LoraCheckpoint') as mock_checkpoint:
        mock_checkpoint_instance = Mock()
        mock_checkpoint_instance.ckpt_paths = test_ckpt_paths
        mock_checkpoint_instance.ckpt_paths_cpu = test_ckpt_paths_cpu
        mock_checkpoint_instance.is_lora_module.return_value = True
        mock_checkpoint.return_value = mock_checkpoint_instance
        manager.lora_checkpoint = mock_checkpoint

    def test_init(self):
        assert self.manager.decay_count == 0
        assert self.manager.decay_max == 100
        self.assertIsInstance(self.manager.req_ids_to_adapter_ids_mapping, dict)

    def test_adapter_id_mapping(self):
        self.manager.lora_checkpoint.ckpt_paths = self.test_ckpt_paths
        self.manager.lora_checkpoint.ckpt_paths_cpu = self.test_ckpt_paths_cpu
        self.manager.get_adapter_id_mapping()

        expected_mapping = {
            "adapter1": 0,
            "adapter2": 1,
            "adapter3": 2,
            "adapter4": 3,
            "adapter5": 4
        }
        assert self.manager.lora_adapter_id_mapping == expected_mapping

        adapter_ids = self.test_ckpt_paths.keys()
        batch_size = len(adapter_ids)
        result = self.manager.convert_adapter_ids_to_indices(adapter_ids, batch_size)
        expected = torch.tensor(config.device_adapter_ids, dtype=torch.int32)
        assert torch.equal(result, expected)

        assert self.manager.list_adapters() == {"adapter1", "adapter2", "adapter3", "adapter4", "adapter5"}

    def test_convert_adapter_ids_to_indices_validation(self):
        new_manager = LoraModelManager(config.lora_config)
        with self.assertRaises(ValueError):
            new_manager.convert_adapter_ids_to_indices(
                ["adapter1", "adapter2", "adapter3"], 
                2
            )

        with self.assertRaises(ValueError):
            new_manager.convert_adapter_ids_to_indices(
                ["adapter1"], 
                2
            )

    def test_req_id_mapping_operations(self):
        new_manager = LoraModelManager(config.lora_config)
        new_manager.add_req_id_to_adapter_id_mapping("req1", "adapter1")
        assert new_manager.req_ids_to_adapter_ids_mapping["req1"] == "adapter1"

        req_ids = ["req2", "req3"]
        adapter_ids = ["adapter2", "adapter3"]
        new_manager.add_req_ids_to_adapter_ids_mapping(req_ids, adapter_ids)
        assert new_manager.get_adapter_id_with_req_id("req2") == "adapter2"
        assert new_manager.get_adapter_id_with_req_id("req3") == "adapter3"
        assert new_manager.get_adapter_ids_with_req_ids(["req1", "req2"]) == ["adapter1", "adapter2"]

        new_manager.remove_req_id("req1")
        assert "req1" not in new_manager.req_ids_to_adapter_ids_mapping

        new_manager.remove_req_ids(["req2", "req3"])
        assert "req2" not in new_manager.req_ids_to_adapter_ids_mapping
        assert "req3" not in new_manager.req_ids_to_adapter_ids_mapping
        assert len(new_manager.req_ids_to_adapter_ids_mapping) == 0

    def test_init_dynamic_multi_lora(self):
        cpu_weights = [{
            "lora_A": torch.randn(2, 2),
            "lora_B": torch.randn(2, 2)
        }]
        self.manager.init_dynamic_multi_lora(cpu_weights)
        assert self.manager.lora_modules == config.modules

    def test_dynamic_update_weights_for_lora(self):
        new_manager = LoraModelManager(config.lora_config)
        new_manager.cpu_adapter_cache = AdapterCache(config.lora_config.max_cpu_loras, config.cpu_adapter_ids)
        new_manager.device_adapter_cache = AdapterCache(config.lora_config.max_loras, config.device_adapter_ids)
        adapter_ids = torch.tensor([0, 1], dtype=torch.int32)
        result = new_manager.dynamic_update_weights_for_lora(config.device_weights, adapter_ids)
        assert torch.equal(result, adapter_ids)

        # Test adapter swap when cache is full
        adapter_ids = torch.tensor([0, 2], dtype=torch.int32)
        result = new_manager.dynamic_update_weights_for_lora(config.device_weights, adapter_ids)
        assert torch.equal(result, adapter_ids)


if __name__ == "__main__":
    unittest.main()
