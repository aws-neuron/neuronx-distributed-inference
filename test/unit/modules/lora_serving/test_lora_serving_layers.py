import unittest

import neuronx_distributed as nxd
import torch
import torch.distributed
from neuronx_distributed.trace.mock_torchdist import mock_distributed

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

        for lora_memory_transpose in (False, True):
            lora_layer = MultiLoraLinear(
                max_loras, input_size, output_size, dtype, lora_memory_transpose
            )
            expected_shape = (
                (max_loras, input_size, output_size)
                if lora_memory_transpose
                else (max_loras, output_size, input_size)
            )
            assert lora_layer.get_checkpoint_shape() == expected_shape

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

        for lora_memory_transpose in (False, True):
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
                lora_memory_transpose,
            )
            expected_shape = (
                (max_loras, input_size, output_size)
                if lora_memory_transpose
                else (max_loras, output_size, input_size)
            )
            assert lora_layer.get_checkpoint_shape() == expected_shape

    def test_column_parallel_linear_layer(self):
        world_size = 8
        with mock_distributed(world_size=world_size):
            torch.distributed.init_process_group("xla", rank=0, world_size=world_size)
            nxd.parallel_layers.parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=world_size,
                skip_collective_init=True,
            )
            max_loras = 2
            input_size = 32
            output_size = 16
            dtype = torch.float32

            for lora_memory_transpose in (False, True):
                lora_layer = MultiLoraColumnParallelLinear(
                    max_loras,
                    input_size,
                    output_size,
                    dtype,
                    memory_transpose=lora_memory_transpose,
                )
                expected_shape = (
                    (max_loras, input_size, output_size)
                    if lora_memory_transpose
                    else (max_loras, output_size, input_size)
                )
                assert lora_layer.get_checkpoint_shape() == expected_shape

            nxd.parallel_layers.parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()

    def test_row_parallel_linear_layer(self):
        world_size = 8
        with mock_distributed(world_size=world_size):
            torch.distributed.init_process_group("xla", rank=0, world_size=world_size)
            nxd.parallel_layers.parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=world_size,
                skip_collective_init=True,
            )

            max_loras = 2
            input_size = 32
            output_size = 16
            dtype = torch.float32

            for lora_memory_transpose in (False, True):
                lora_layer = MultiLoraRowParallelLinear(
                    max_loras,
                    input_size,
                    output_size,
                    dtype,
                    memory_transpose=lora_memory_transpose,
                )
                expected_shape = (
                    (max_loras, input_size, output_size)
                    if lora_memory_transpose
                    else (max_loras, output_size, input_size)
                )
                assert lora_layer.get_checkpoint_shape() == expected_shape

            nxd.parallel_layers.parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
