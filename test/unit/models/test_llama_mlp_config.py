import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
from neuronx_distributed.parallel_layers import parallel_state
from transformers import AutoConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    get_updated_configs,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


class TestUpdateConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_tensor_model_parallel_group = parallel_state._TENSOR_MODEL_PARALLEL_GROUP
        self.initial_world_size = parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
        self.initial_rank = parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK
        self.initial_data_parallel_group = parallel_state._DATA_PARALLEL_GROUP
        self.initial_world_group = parallel_state._WORLD_GROUP

        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = MagicMock(spec=torch.distributed.ProcessGroup)
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP.size.return_value = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0
        parallel_state._DATA_PARALLEL_GROUP = MagicMock(spec=torch.distributed.ProcessGroup)
        parallel_state._DATA_PARALLEL_GROUP.size.return_value = 1
        parallel_state._WORLD_GROUP = MagicMock()
        parallel_state._WORLD_GROUP.size.return_value = 1

    def tearDown(self) -> None:
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = self.initial_tensor_model_parallel_group
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = self.initial_world_size
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = self.initial_rank
        parallel_state._DATA_PARALLEL_GROUP = self.initial_data_parallel_group
        parallel_state._WORLD_GROUP = self.initial_world_group

    def _help_function_test_updated_mlp_configs(self):
        config_data = {
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
            "hidden_act": "silu",
            "hidden_size": 16384,
            "initializer_range": 0.02,
            "intermediate_size": 53248,
            "max_position_embeddings": 131072,
            "model_type": "llama",
            "num_attention_heads": 128,
            "num_hidden_layers": 4,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 128256,
            "do_sample": False,
            "top_k": 1,
        }

        modules_to_not_convert = [
            "lm_head",
            "layers.0.mlp",
            "layers.3.mlp",
            "layers.0.self_attn",
            "layers.1.self_attn",
            "layers.2.self_attn",
            "layers.3.self_attn",
        ]

        neuron_config = NeuronConfig()
        neuron_config.quantized_mlp_kernel_enabled = True
        neuron_config.modules_to_not_convert = modules_to_not_convert

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
            json.dump(config_data, temp_file)

        return temp_file, neuron_config

    def test_updated_mlp_configs(self):
        temp_file, neuron_config = self._help_function_test_updated_mlp_configs()

        config = LlamaInferenceConfig(
            neuron_config=neuron_config,
            load_config=load_pretrained_config(
                hf_config=AutoConfig.from_pretrained(temp_file.name)
            ),
        )

        updated_configs = get_updated_configs(config)

        for i in range(config.num_hidden_layers):
            module_pattern = f"layers.{i}.mlp"
            if module_pattern in config.neuron_config.modules_to_not_convert:
                assert (
                    updated_configs[i].neuron_config.quantized_mlp_kernel_enabled is False
                ), f"Layer {i} should have quantized_mlp_kernel_enabled=False"
            else:
                assert (
                    updated_configs[i].neuron_config.quantized_mlp_kernel_enabled is True
                ), f"Layer {i} should have quantized_mlp_kernel_enabled=True"

        os.remove(temp_file.name)


if __name__ == "__main__":
    unittest.main()
