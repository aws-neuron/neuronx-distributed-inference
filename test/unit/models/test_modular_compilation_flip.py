from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig


class MockNeuronConfig(NeuronConfig):
    def __init__(self):
        self.tp_degree = 1
        self.pp_degree = 1
        self.ep_degree = 1
        self.world_size = 1
        self.start_rank_id = 0
        self.local_ranks_size = 1
        self.torch_dtype = None
        self.cast_type = None
        self.enable_cte_modular_flow = False
        self.buckets = [1024]
        self.skip_sharding = False
        self.save_sharded_checkpoint = False
        self.weights_to_skip_layout_optimization = []
        self.on_device_sampling_config = None
        self.logical_nc_config = None
        self.enable_fused_speculation = False
        self.speculation_length = 0
        self.medusa_speculation_length = 0
        self.lora_config = None


class MockInferenceConfig(InferenceConfig):
    def __init__(self):
        self.neuron_config = MockNeuronConfig()
        self.fused_spec_config = None
        self.num_cores_per_group = 1


# Create a test version of NeuronApplicationBase with implemented get_config_cls
class TestNeuronApplicationBase(NeuronApplicationBase):
    @classmethod
    def get_config_cls(cls) -> InferenceConfig:
        return MockInferenceConfig


@dataclass
class Config:
    num_hidden_layers: int
    model_type: str


@dataclass
class TestNeuronConfig:
    enable_cte_modular_flow: bool
    buckets: list


@dataclass
class ModelInstance:
    config: Config
    neuron_config: TestNeuronConfig


@dataclass
class ModelArtifacts:
    model_instance: ModelInstance


def test_check_and_apply_modular_flow_optimization():
    # Create an instance of TestNeuronApplicationBase instead of NeuronApplicationBase
    mock_config = MockInferenceConfig()
    app = TestNeuronApplicationBase(model_path="dummy_path", config=mock_config)

    # Test cases
    test_cases = [
        # Test case 1: Non-context encoding model
        {
            "key": "other_model",
            "bucket_length": 2048,
            "num_layers": 32,
            "model_type": "llama",
            "enable_cte_modular_flow": False,
            "compiler_args": "-O1",
            "expected": "-O1",
        },
        # Test case 2: Valid configuration for optimization
        {
            "key": "context_encoding_model",
            "bucket_length": 1024,
            "num_layers": 32,
            "model_type": "llama",
            "enable_cte_modular_flow": False,
            "compiler_args": "-O1",
            "expected": "-O3",
        },
        # Test case 3: Invalid layer count
        {
            "key": "context_encoding_model",
            "bucket_length": 1024,
            "num_layers": 24,
            "model_type": "llama",
            "enable_cte_modular_flow": False,
            "compiler_args": "-O1",
            "expected": "-O1",
        },
        # Test case 4: Invalid model type
        {
            "key": "context_encoding_model",
            "bucket_length": 1024,
            "num_layers": 32,
            "model_type": "gpt2",
            "enable_cte_modular_flow": False,
            "compiler_args": "-O1",
            "expected": "-O1",
        },
        # Test case 5: CTE modular flow enabled
        # Reenable this test after turning on the functionality for the flag.
        # The functionality is currently disabled due to logits regression from ticket: V1849736968
        # {
        #     "key": "context_encoding_model",
        #     "bucket_length": 1024,
        #     "num_layers": 32,
        #     "model_type": "llama",
        #     "enable_cte_modular_flow": True,
        #     "compiler_args": "-O1",
        #     "expected": "-O1",
        # },
        # Test case 6: No O1 flag in compiler args
        {
            "key": "context_encoding_model",
            "bucket_length": 1024,
            "num_layers": 32,
            "model_type": "llama",
            "enable_cte_modular_flow": False,
            "compiler_args": "-O2",
            "expected": "-O2",
        },
    ]

    for tc in test_cases:
        # Create mock objects
        config = Config(num_hidden_layers=tc["num_layers"], model_type=tc["model_type"])
        neuron_config = TestNeuronConfig(
            enable_cte_modular_flow=tc["enable_cte_modular_flow"],
            buckets=[tc["bucket_length"]],
        )
        model_instance = ModelInstance(config=config, neuron_config=neuron_config)
        model_artifacts = ModelArtifacts(model_instance=model_instance)

        # Call the method
        result = app.check_and_apply_modular_flow_optimization(
            tc["key"], model_artifacts, 0, tc["compiler_args"]  # bucket_rank
        )

        # Assert the result
        assert result == tc["expected"], f"Failed for test case: {tc}"
