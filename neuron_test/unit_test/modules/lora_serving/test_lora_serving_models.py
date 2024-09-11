# Standard Library
import unittest
from unittest.mock import MagicMock, patch

import torch
from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.modules.lora_serving import LoraModel, LoraServingConfig
from neuronx_distributed_inference.modules.lora_serving.lora_module import MultiLoraModule


class NxDModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # we will wrap the following layers with LoRA
        self.rpl = ColumnParallelLinear(32, 32)
        self.cpl = RowParallelLinear(32, 32)
        self.linear = torch.nn.Linear(32, 32)
        self.embedding = torch.nn.Embedding(32, 32)
        self.conv2d = torch.nn.Conv2d(32, 32, 2)
        self.pembedding = ParallelEmbedding(32, 32)

        # we will keep the following layers as they are
        self.as_rpl = ColumnParallelLinear(32, 32)
        self.as_cpl = RowParallelLinear(32, 32)
        self.as_linear = torch.nn.Linear(32, 32)
        self.as_embedding = torch.nn.Embedding(32, 32)
        self.as_conv2d = torch.nn.Conv2d(32, 32, 2)
        self.as_pembedding = ParallelEmbedding(32, 32)


lora_config = LoraServingConfig(
    max_loras=2,
    max_lora_rank=16,
    target_modules=["rpl", "cpl", "linear", "embedding", "conv2d", "pembedding"],
)


class TestLoraModels(unittest.TestCase):
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
    def test_nxd_model(self):
        model = NxDModule()
        LoraModel(model, lora_config)

        assert isinstance(model.rpl, MultiLoraModule)
        assert isinstance(model.cpl, MultiLoraModule)
        assert isinstance(model.linear, MultiLoraModule)
        assert isinstance(model.embedding, MultiLoraModule)
        assert isinstance(model.conv2d, MultiLoraModule)
        assert isinstance(model.pembedding, MultiLoraModule)

        assert isinstance(model.as_rpl, ColumnParallelLinear)
        assert isinstance(model.as_cpl, RowParallelLinear)
        assert isinstance(model.as_linear, torch.nn.Linear)
        assert isinstance(model.as_embedding, torch.nn.Embedding)
        assert isinstance(model.as_conv2d, torch.nn.Conv2d)
        assert isinstance(model.as_pembedding, ParallelEmbedding)


if __name__ == "__main__":
    unittest.main()
