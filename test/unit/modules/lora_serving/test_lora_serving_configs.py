# Standard Library
import unittest
import json
import os
from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig


class LoRAConfigGenerator:
    def __init__(self, lora_dir, lora_json_filename, lora_ckpt_num=4, lora_ckpt_num_cpu=0):
        self.lora_dir = lora_dir
        self.lora_json_filename = lora_json_filename
        self.lora_ckpt_num = lora_ckpt_num
        self.lora_ckpt_num_cpu = lora_ckpt_num_cpu
        self.lora_config_filename = "adapter_config.json"
        self.init_lora_config()
        
        self.generate_dummy_lora_ckpts(self.lora_ckpt_num_cpu > 0)

    def init_lora_config(self):
        self.lora_ids = [f"lora_id_{idx}" for idx in range(self.lora_ckpt_num)]
        self.lora_paths = [f"lora_folder_{idx}" for idx in range(self.lora_ckpt_num)]
        self.lora_paths_cpu = [f"cpu_lora_folder_{idx}" for idx in range(self.lora_ckpt_num_cpu)]
        self.target_modules = [f"module_{idx}" for idx in range(self.lora_ckpt_num)]
        os.makedirs(self.lora_dir, exist_ok=True)
        self.lora_json_file = os.path.join(self.lora_dir, self.lora_json_filename)
        
    def write_to_json(self, filename, data):
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def generate_lora_ckpt_folder(self, folder_name, config):
        folder_name = os.path.join(self.lora_dir, folder_name)
        os.makedirs(folder_name, exist_ok=True)
        filename = os.path.join(folder_name, self.lora_config_filename)
        self.write_to_json(filename, config)
        
    def generate_lora_config(self, rank, target_modules):
        return {
            "r": rank,
            "target_modules": target_modules,
        }
        
    def generate_lora_ckpt_json(self, has_cpu):
        data = {
            "lora-ckpt-dir": self.lora_dir,
            "lora-ckpt-paths": {
                lora_id : lora_path for lora_id, lora_path in zip(self.lora_ids, self.lora_paths)
            }
        }

        if has_cpu:
            data["lora-ckpt-paths-cpu"] = {lora_id : lora_path for lora_id, lora_path in zip(self.lora_ids, self.lora_paths_cpu)}
        
        self.write_to_json(self.lora_json_file, data)
    
    def get_lora_ckpt_num(self):
        return self.lora_ckpt_num
    
    def get_lora_ckpt_num_cpu(self):
        return self.lora_ckpt_num_cpu
    
    def get_lora_rank(self, lora_idx):
        return 2 ** (lora_idx + 3)
    
    def get_max_lora_rank(self):
        return 2 ** (self.lora_ckpt_num + 2)
    
    def get_target_modules(self):
        return self.target_modules
    
    def get_lora_ids(self):
        return self.lora_ids
    
    def get_lora_paths(self):
        return self.lora_paths
    
    def get_lora_paths_cpu(self):
        return self.lora_paths_cpu
    
    def get_lora_json_file(self):
        return self.lora_json_file
        
    def generate_dummy_lora_ckpts(self, has_cpu=False):
        self.generate_lora_ckpt_json(has_cpu)

        for i in range(self.lora_ckpt_num):
            rank = self.get_lora_rank(i)
            sub_target_modules = self.target_modules[:i+1]
            lora_config = self.generate_lora_config(rank, sub_target_modules)
            self.generate_lora_ckpt_folder(self.lora_paths[i], lora_config)

        for i in range(self.lora_ckpt_num_cpu):
            rank = self.get_lora_rank(i)
            sub_target_modules = self.target_modules[:i+1]
            lora_config = self.generate_lora_config(rank, sub_target_modules)
            self.generate_lora_ckpt_folder(self.lora_paths_cpu[i], lora_config)

    def generate_lora_ckpt_paths_as_list(self):
        lora_ckpt_paths = []
        for lora_id, lora_path in zip(self.lora_ids, self.lora_paths):
            lora_ckpt_path = f"{lora_id} : {os.path.join(self.lora_dir, lora_path)}"
            lora_ckpt_paths.append(lora_ckpt_path)
        return lora_ckpt_paths
    
    def generate_lora_ckpt_paths_as_dict(self):
        lora_ckpt_paths = {}
        for lora_id, lora_path in zip(self.lora_ids, self.lora_paths):
            lora_ckpt_paths[lora_id] = os.path.join(self.lora_dir, lora_path)
        return lora_ckpt_paths


lora_serving_unit_test_path = "test/unit/modules/lora_serving"
lora_ckpt_dir = os.path.join(lora_serving_unit_test_path, "lora_ckpts")
lora_json_filename = "adapters.json"
lora_ckpt_num = 4
lora_config_generator = LoRAConfigGenerator(lora_ckpt_dir, lora_json_filename, lora_ckpt_num)

lora_ckpt_num_cpu = 2
lora_config_generator_cpu = LoRAConfigGenerator(lora_ckpt_dir, lora_json_filename, lora_ckpt_num, lora_ckpt_num_cpu)

    
class TestLoraConfigs(unittest.TestCase):
    def test_lora_config_ckpt_paths_as_list(self):
        batch_size = lora_config_generator.get_lora_ckpt_num()
        lora_ckpt_paths_as_list = lora_config_generator.generate_lora_ckpt_paths_as_list()

        lora_config = LoraServingConfig(
            batch_size = batch_size,
            lora_ckpt_paths = lora_ckpt_paths_as_list
        )

        assert lora_config.max_loras == batch_size
        assert lora_config.batch_size == batch_size
        assert lora_config.max_lora_rank == lora_config_generator.get_max_lora_rank()
        assert set(lora_config.target_modules) == set(lora_config_generator.get_target_modules())

    def test_lora_config_ckpt_paths_as_dict(self):
        batch_size = lora_config_generator.get_lora_ckpt_num()
        lora_ckpt_paths_as_dict = lora_config_generator.generate_lora_ckpt_paths_as_dict()

        lora_config = LoraServingConfig(
            batch_size = batch_size,
            lora_ckpt_paths = lora_ckpt_paths_as_dict
        )

        assert lora_config.max_loras == batch_size
        assert lora_config.batch_size == batch_size
        assert lora_config.max_lora_rank == lora_config_generator.get_max_lora_rank()
        assert set(lora_config.target_modules) == set(lora_config_generator.get_target_modules())

    def test_lora_config_ckpt_paths_as_json(self):
        batch_size = lora_config_generator.get_lora_ckpt_num()
        lora_json_file = lora_config_generator.get_lora_json_file()

        lora_config = LoraServingConfig(
            batch_size = batch_size,
            lora_ckpt_json = lora_json_file
        )
        assert lora_config.max_loras == batch_size
        assert lora_config.batch_size == batch_size
        assert lora_config.max_lora_rank == lora_config_generator.get_max_lora_rank()
        assert set(lora_config.target_modules) == set(lora_config_generator.get_target_modules())

    def test_lora_config_ckpt_paths_as_json_cpu(self):
        batch_size = lora_config_generator_cpu.get_lora_ckpt_num()
        lora_json_file = lora_config_generator_cpu.get_lora_json_file()

        lora_config = LoraServingConfig(
            batch_size = batch_size,
            lora_ckpt_json = lora_json_file
        )
        assert lora_config.max_loras == batch_size
        assert lora_config.batch_size == batch_size
        assert lora_config.max_lora_rank == lora_config_generator_cpu.get_max_lora_rank()
        assert set(lora_config.target_modules) == set(lora_config_generator_cpu.get_target_modules())

if __name__ == "__main__":
    unittest.main()
