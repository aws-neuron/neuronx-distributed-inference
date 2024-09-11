# Standard Library
import unittest
from unittest.mock import MagicMock, patch

import torch
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear

from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig
from neuronx_distributed_inference.modules.lora_serving.lora_module import (
    MultiLoraModuleColumnParallelLinear,
    MultiLoraModuleConv2d,
    MultiLoraModuleEmbedding,
    MultiLoraModuleLinear,
    MultiLoraModuleRowParallelLinear,
)

lora_config = LoraServingConfig(
    max_loras=2,
    max_lora_rank=16,
)


class TestLoraServingModules(unittest.TestCase):
    def test_torch_linear_layer(self):
        base_layer = torch.nn.Linear(32, 32)
        lora_module = MultiLoraModuleLinear(base_layer, lora_config)

        assert lora_module.lora_A.get_checkpoint_shape() == (2, 16, 32)
        assert lora_module.lora_B.get_checkpoint_shape() == (2, 32, 16)

    def test_torch_conv2d_layer(self):
        base_layer = torch.nn.Conv2d(32, 32, 2)
        lora_module = MultiLoraModuleConv2d(base_layer, lora_config)

        assert tuple(lora_module.lora_A.get_checkpoint_shape()) == (2, 32, 16, 2, 2)
        assert tuple(lora_module.lora_B.get_checkpoint_shape()) == (2, 16, 32, 1, 1)

    def test_torch_embedding_layer(self):
        base_layer = torch.nn.Embedding(32, 32)
        lora_module = MultiLoraModuleEmbedding(base_layer, lora_config)

        assert lora_module.lora_A.get_checkpoint_shape() == (2, 16, 32)
        assert lora_module.lora_B.get_checkpoint_shape() == (2, 32, 16)

    @patch(
        "neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size",
        MagicMock(return_value=8),
    )
    @patch(
        "neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank",
        MagicMock(return_value=1),
    )
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel",
        MagicMock(return_value=True),
    )
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized",
        MagicMock(return_value=True),
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=8))
    def test_column_parallel_linear_layer(self):
        base_layer = ColumnParallelLinear(32, 32)
        lora_module = MultiLoraModuleColumnParallelLinear(base_layer, lora_config)

        assert lora_module.lora_A.get_checkpoint_shape() == (2, 16, 32)
        assert lora_module.lora_B.get_checkpoint_shape() == (2, 32, 16)

    @patch(
        "neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size",
        MagicMock(return_value=8),
    )
    @patch(
        "neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank",
        MagicMock(return_value=1),
    )
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel",
        MagicMock(return_value=True),
    )
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized",
        MagicMock(return_value=True),
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=8))
    def test_row_parallel_linear_layer(self):
        base_layer = RowParallelLinear(32, 32)
        lora_module = MultiLoraModuleRowParallelLinear(base_layer, lora_config)

        assert lora_module.lora_A.get_checkpoint_shape() == (2, 16, 32)
        assert lora_module.lora_B.get_checkpoint_shape() == (2, 32, 16)


if __name__ == "__main__":
    unittest.main()
