import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.quantization.quantization_layers import (
    QuantizedColumnParallel,
    QuantizedRowParallel,
)
from neuronx_distributed.quantization.quantize import convert
from transformers import AutoConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    NeuronLlamaModel,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


class TestConvert(unittest.TestCase):
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

    def _help_function_test_convert_modules_to_not_convert(self):
        config_data = {
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "num_attention_heads": 16,
            "num_hidden_layers": 8,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 32000,
            "do_sample": False,
            "top_k": 1,
        }

        modules_to_not_convert = [
            "lm_head",
            "layers.0.mlp.down_proj",
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
            "layers.7.mlp.down_proj",
            "layers.7.mlp.gate_proj",
            "layers.7.mlp.up_proj",
        ]

        for i in range(8):
            modules_to_not_convert.append(f"layers.{i}.self_attn.qkv_proj.k_proj")
            modules_to_not_convert.append(f"layers.{i}.self_attn.o_proj.o_proj")
            modules_to_not_convert.append(f"layers.{i}.self_attn.qkv_proj.q_proj")
            modules_to_not_convert.append(f"layers.{i}.self_attn.qkv_proj.v_proj")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
            json.dump(config_data, temp_file)

        return modules_to_not_convert, temp_file

    def test_convert_modules_to_not_convert(self):
        (
            modules_to_not_convert,
            temp_file,
        ) = self._help_function_test_convert_modules_to_not_convert()
        config = LlamaInferenceConfig(
            neuron_config=NeuronConfig(),
            load_config=load_pretrained_config(
                hf_config=AutoConfig.from_pretrained(temp_file.name)
            ),
        )
        with torch.device("meta"):
            llama_model = NeuronLlamaModel(config=config)

        quantized_model = convert(
            module=llama_model,
            q_config=None,
            inplace=False,
            mapping=None,
            modules_to_not_convert=modules_to_not_convert,
        )

        for i in range(8):
            assert isinstance(
                quantized_model.layers[i].self_attn.qkv_proj.q_proj, ColumnParallelLinear
            )
            assert isinstance(
                quantized_model.layers[i].self_attn.qkv_proj.q_proj, ColumnParallelLinear
            )
            assert isinstance(
                quantized_model.layers[i].self_attn.qkv_proj.q_proj, ColumnParallelLinear
            )
            assert isinstance(quantized_model.layers[i].self_attn.o_proj.o_proj, RowParallelLinear)

            if i in [0, 7]:
                assert isinstance(quantized_model.layers[i].mlp.up_proj, ColumnParallelLinear)
                assert isinstance(quantized_model.layers[i].mlp.gate_proj, ColumnParallelLinear)
                assert isinstance(quantized_model.layers[i].mlp.down_proj, RowParallelLinear)
            else:
                assert isinstance(quantized_model.layers[i].mlp.up_proj, QuantizedColumnParallel)
                assert isinstance(quantized_model.layers[i].mlp.gate_proj, QuantizedColumnParallel)
                assert isinstance(quantized_model.layers[i].mlp.down_proj, QuantizedRowParallel)

        os.remove(temp_file.name)
