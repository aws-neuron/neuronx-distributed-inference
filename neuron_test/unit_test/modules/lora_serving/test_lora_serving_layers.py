# Standard Library
import unittest
from unittest.mock import MagicMock, patch

import torch

from neuronx_distributed_inference.modules.lora_serving.lora_layer import (
    MultiLoraColumnParallelLinear,
    MultiLoraConv2d,
    MultiLoraEmbedding,
    MultiLoraLinear,
    MultiLoraRowParallelLinear,
)


class TestLoraServingLayers(unittest.TestCase):
    def test_torch_linear_layer(self):
        max_loras = 2
        input_size = 32
        output_size = 16
        dtype = torch.float32

        lora_layer = MultiLoraLinear(max_loras, input_size, output_size, dtype)
        assert lora_layer.get_checkpoint_shape() == (max_loras, output_size, input_size)

    def test_torch_conv2d_layer(self):
        base_layer = torch.nn.Conv2d(32, 32, 2)
        max_loras = 2
        input_size = 32
        output_size = 32
        dtype = torch.float32

        lora_layer = MultiLoraConv2d(
            max_loras,
            input_size,
            output_size,
            base_layer.kernel_size,
            base_layer.stride,
            base_layer.padding,
            dtype,
        )
        assert lora_layer.get_checkpoint_shape() == lora_layer.weight.size()

    def test_torch_embedding_layer(self):
        base_layer = torch.nn.Embedding(32, 32)
        max_loras = 2
        input_size = 32
        output_size = 32
        dtype = torch.float32
        lora_layer = MultiLoraEmbedding(
            max_loras,
            input_size,
            output_size,
            base_layer.padding_idx,
            base_layer.max_norm,
            base_layer.norm_type,
            base_layer.scale_grad_by_freq,
            base_layer.sparse,
            dtype,
        )
        assert lora_layer.get_checkpoint_shape() == (max_loras, output_size, input_size)

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
        max_loras = 2
        input_size = 32
        output_size = 16
        dtype = torch.float32
        lora_layer = MultiLoraColumnParallelLinear(max_loras, input_size, output_size, dtype)
        assert lora_layer.get_checkpoint_shape() == (max_loras, output_size, input_size)

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
        max_loras = 2
        input_size = 32
        output_size = 16
        dtype = torch.float32
        lora_layer = MultiLoraRowParallelLinear(max_loras, input_size, output_size, dtype)
        assert lora_layer.get_checkpoint_shape() == (max_loras, output_size, input_size)


if __name__ == "__main__":
    unittest.main()
