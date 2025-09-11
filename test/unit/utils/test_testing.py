import os
import uuid
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from neuronx_distributed.operators.topk import topk as nxd_topk
from neuronx_distributed.parallel_layers import ColumnParallelLinear

from neuronx_distributed_inference.utils.testing import (
    build_function,
    build_module,
    validate_accuracy,
    build_cpu_model,
    _rand_interval,
    _get_rand_weights,
    _get_shared_checkpoint_path,
)

torch.manual_seed(0)

SAMPLE_SIZE = 4


def example_sum(tensor):
    return torch.sum(tensor)


def example_topk(tensor, k, dim, on_cpu):
    if on_cpu:
        return torch.topk(tensor, k, dim)
    else:
        return nxd_topk(tensor, k, dim, gather_dim=dim)


class ExampleModule(torch.nn.Module):
    def __init__(self, distributed):
        super().__init__()
        if distributed:
            self.linear = ColumnParallelLinear(
                input_size=SAMPLE_SIZE,
                output_size=SAMPLE_SIZE,
                bias=False,
                dtype=torch.float32,
            )
        else:
            self.linear = torch.nn.Linear(
                in_features=SAMPLE_SIZE,
                out_features=SAMPLE_SIZE,
                bias=False,
                dtype=torch.float32,
            )

    def forward(self, x):
        return self.linear(x)


def test_validate_accuracy_basic_function():
    inputs = [(torch.tensor([1, 2, 3], dtype=torch.float32),)]
    example_inputs = [(torch.zeros((3), dtype=torch.float32),)]

    neuron_model = build_function(example_sum, example_inputs)
    validate_accuracy(neuron_model, inputs, cpu_callable=example_sum)


def test_validate_accuracy_function_with_expected_outputs():
    inputs = [
        (torch.tensor([1, 2, 3], dtype=torch.float32),),
        (torch.tensor([3, 4, 5], dtype=torch.float32),),
    ]
    expected_outputs = [
        torch.tensor(6, dtype=torch.float32),
        torch.tensor(12, dtype=torch.float32),
    ]
    example_inputs = [(torch.zeros((3), dtype=torch.float32),)]

    neuron_model = build_function(example_sum, example_inputs)
    validate_accuracy(
        neuron_model, inputs, expected_outputs=expected_outputs, cpu_callable=example_sum
    )


def test_validate_accuracy_function_with_custom_cpu_func():
    inputs = [
        (torch.tensor((0, 1, 2, 3), dtype=torch.float32),),
    ]
    example_inputs = [
        (torch.zeros((4), dtype=torch.float32),),
    ]

    func = partial(example_topk, k=1, dim=0, on_cpu=False)
    func_cpu = partial(example_topk, k=1, dim=0, on_cpu=True)
    neuron_model = build_function(func, example_inputs)
    validate_accuracy(
        neuron_model, inputs, cpu_callable=func_cpu, assert_close_kwargs={"check_dtype": False}
    )


def test_validate_accuracy_function_with_distributed_func_tp2():
    inputs = [
        (torch.tensor((0, 1, 2, 3), dtype=torch.float32),),
    ]
    expected_outputs = [
        (torch.tensor((3,), dtype=torch.float32), torch.tensor((3,), dtype=torch.int64)),
    ]
    example_inputs = [(torch.zeros((4), dtype=torch.float32),)]

    func = partial(example_topk, k=1, dim=0, on_cpu=False)
    func_cpu = partial(example_topk, k=1, dim=0, on_cpu=True)
    neuron_model = build_function(func, example_inputs, tp_degree=2)
    validate_accuracy(
        neuron_model,
        inputs,
        expected_outputs=expected_outputs,
        cpu_callable=func_cpu,
        assert_close_kwargs={"check_dtype": False},
    )


def test_validate_accuracy_basic_module():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    module_cls = partial(ExampleModule, distributed=False)
    neuron_model = build_module(module_cls, example_inputs)

    validate_accuracy(neuron_model, inputs, cpu_callable=module_cpu)


def test_validate_accuracy_module_with_expected_outputs():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    expected_outputs = [torch.tensor([-1.6587, 1.3036, -0.4648, -0.6878], dtype=torch.float32)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    module_cls = partial(ExampleModule, distributed=False)
    neuron_model = build_module(module_cls, example_inputs)

    validate_accuracy(
        neuron_model,
        inputs,
        expected_outputs=expected_outputs,
        cpu_callable=module_cpu,
        assert_close_kwargs={"rtol": 1e-3, "atol": 1e-3},
    )


def test_validate_accuracy_module_with_custom_cpu_module():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    neuron_model = build_module(
        ExampleModule, example_inputs, module_init_kwargs={"distributed": True}
    )

    validate_accuracy(neuron_model, inputs, cpu_callable=module_cpu)


def test_validate_accuracy_module_with_custom_cpu_module_tp2():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    neuron_model = build_module(
        ExampleModule, example_inputs, tp_degree=2, module_init_kwargs={"distributed": True}
    )

    validate_accuracy(neuron_model, inputs, cpu_callable=module_cpu)


def test_validate_accuracy_module_with_multiple_inputs():
    inputs = [
        (torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),),
        (torch.arange(SAMPLE_SIZE, SAMPLE_SIZE * 2, dtype=torch.float32),),
    ]
    expected_outputs = [
        torch.tensor([-1.6587, 1.3036, -0.4648, -0.6878], dtype=torch.float32),
        torch.tensor([-3.7188, 2.6158, -1.1106, -4.6734], dtype=torch.float32),
    ]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    neuron_model = build_module(
        ExampleModule, example_inputs, module_init_kwargs={"distributed": True}
    )

    validate_accuracy(
        neuron_model,
        inputs,
        expected_outputs,
        cpu_callable=module_cpu,
        assert_close_kwargs={"rtol": 1e-3, "atol": 1e-3},
    )


def test_validate_accuracy_module_with_custom_compiler_workdir_and_checkpoint_path():
    checkpoint_path = "/tmp/nxdi_checkpoint.pt"
    compiler_workdir = "/tmp/nxdi_compiler_workdir"

    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    torch.save(module_cpu.state_dict(), checkpoint_path)

    neuron_model = build_module(
        ExampleModule,
        example_inputs,
        compiler_workdir=compiler_workdir,
        checkpoint_path=checkpoint_path,
        module_init_kwargs={"distributed": True},
    )
    validate_accuracy(
        neuron_model,
        inputs,
        cpu_callable=module_cpu,
        assert_close_kwargs={"rtol": 1e-3, "atol": 1e-3},
    )

    # Verify the custom compiler workdir is used.
    assert len(os.listdir(compiler_workdir)) >= 1


def test_validate_accuracy_no_expected_outputs():
    neuron_model = Mock()
    inputs = [(torch.rand((1), dtype=torch.float32),)]
    with pytest.raises(
        ValueError, match="Provide expected_outputs or a cpu_callable to produce expected outputs"
    ):
        validate_accuracy(neuron_model, inputs)


def test_validate_accuracy_inputs_not_a_list():
    neuron_model = Mock()
    cpu_callable = Mock()
    inputs = {}
    with pytest.raises(ValueError, match="inputs must be a list of tensor tuples"):
        validate_accuracy(neuron_model, inputs, cpu_callable=cpu_callable)


def test_validate_accuracy_inputs_is_empty_list():
    neuron_model = Mock()
    cpu_callable = Mock()
    inputs = []
    with pytest.raises(ValueError, match="inputs must not be empty"):
        validate_accuracy(neuron_model, inputs, cpu_callable=cpu_callable)


def test_validate_accuracy_inputs_contains_non_tuple():
    neuron_model = Mock()
    cpu_callable = Mock()
    inputs = [1]
    with pytest.raises(ValueError, match="inputs must be a list of tensor tuples"):
        validate_accuracy(neuron_model, inputs, cpu_callable=cpu_callable)


def test_validate_accuracy_inputs_contains_tuple_with_non_tensor():
    neuron_model = Mock()
    cpu_callable = Mock()
    inputs = [(1,)]
    with pytest.raises(ValueError, match="inputs must be a list of tensor tuples"):
        validate_accuracy(neuron_model, inputs, cpu_callable=cpu_callable)


def test_validate_accuracy_expected_outputs_not_a_list():
    neuron_model = Mock()
    inputs = [(torch.rand((1), dtype=torch.float32),)]
    expected_outputs = {}
    with pytest.raises(ValueError, match="expected_outputs must be a list"):
        validate_accuracy(neuron_model, inputs, expected_outputs)


def test_validate_accuracy_expected_outputs_len_mismatch():
    neuron_model = Mock()
    inputs = [(torch.rand((1), dtype=torch.float32),)]
    expected_outputs = [
        (torch.rand((1), dtype=torch.float32),),
        (torch.rand((1), dtype=torch.float32),),
    ]
    with pytest.raises(ValueError, match=r"len\(expected_outputs\) must match len\(inputs\)"):
        validate_accuracy(neuron_model, inputs, expected_outputs)


def test_build_module_example_inputs_not_a_list():
    module_cls = Mock()
    example_inputs = {}
    with pytest.raises(ValueError, match="example_inputs must be a list of tensor tuples"):
        build_module(module_cls, example_inputs)


def test_build_module_example_inputs_empty_list():
    module_cls = Mock()
    example_inputs = []
    with pytest.raises(ValueError, match="example_inputs must contain exactly one input"):
        build_module(module_cls, example_inputs)


def test_build_module_example_inputs_contains_non_tuple():
    module_cls = Mock()
    example_inputs = [1]
    with pytest.raises(ValueError, match="example_inputs must be a list of tensor tuples"):
        build_module(module_cls, example_inputs)


def test_build_module_example_inputs_contains_tuple_with_non_tensor():
    module_cls = Mock()
    example_inputs = [(1,)]
    with pytest.raises(ValueError, match="example_inputs must be a list of tensor tuples"):
        build_module(module_cls, example_inputs)


def test_build_module_with_multiple_example_inputs():
    module_cls = Mock()
    example_inputs = [
        (torch.zeros((SAMPLE_SIZE), dtype=torch.float32),),
        (torch.zeros((SAMPLE_SIZE * 2), dtype=torch.float32),),
    ]
    with pytest.raises(ValueError, match="example_inputs must contain exactly one input"):
        build_module(module_cls, example_inputs)


def test_build_cpu_model(monkeypatch):
    """Test build_cpu_model creates a model and returns it with a checkpoint path."""
    # Setup mocks
    config = {"test_config": "value"}
    mock_model = Mock()
    model_cls = Mock(return_value=mock_model)
    
    # Mock uuid.uuid4 to return a consistent UUID
    mock_uuid = Mock(return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"))
    monkeypatch.setattr(uuid, "uuid4", mock_uuid)
    
    # Mock _get_rand_weights to return the original model
    mock_get_rand_weights = Mock(return_value=mock_model)
    monkeypatch.setattr("neuronx_distributed_inference.utils.testing._get_rand_weights", mock_get_rand_weights)
    
    cpu_model, ckpt_path = build_cpu_model(model_cls, config)
    
    # Verify function behavior
    model_cls.assert_called_once_with(config)
    mock_get_rand_weights.assert_called_once()
    assert cpu_model == mock_model
    assert ckpt_path == "/tmp/nxd_inference/ckpt_12345678.pt"


@pytest.mark.parametrize(
    "low,high,dtype,shape", 
    [
        # Test case 1: 1D tensor with float32
        (-5.0, 10.0, torch.float32, (100,)),
        # Test case 2: Multi-dimensional output with float32
        (-5.0, 10.0, torch.float32, (3, 4, 5)),
        # Test case 3-5: Different dtypes with same shape
        (-5.0, 10.0, torch.float16, (3, 4, 5)),
        (-5.0, 10.0, torch.float32, (3, 4, 5)),
        (-5.0, 10.0, torch.float64, (3, 4, 5)),
    ]
)
def test_rand_interval(low, high, dtype, shape):
    """Test _rand_interval generates values in the expected range."""
    result = _rand_interval(low, high, dtype, *shape)
    
    # Check shape
    expected_shape = torch.Size(shape) if isinstance(shape, tuple) else shape
    assert result.shape == expected_shape, "Should return a tensor of specified shape"
    
    # Check range
    assert torch.all(result >= low) and torch.all(result < high), "All values should be in range"
    
    # Check dtype
    assert result.dtype == dtype, f"Should return a tensor with dtype {dtype}"

@pytest.mark.parametrize(
    "low1,high1,low2,high2", 
    [
        # Different distributions for different ranges
        (0.0, 1.0, -10.0, 10.0),
    ]
)
def test_rand_interval_distribution(low1, high1, low2, high2):
    """Test that _rand_interval distributions have expected statistical properties."""
    shape = (3, 4, 5)
    
    # Generate two result sets with different ranges
    result1 = _rand_interval(low1, high1, torch.float32, *shape)
    result2 = _rand_interval(low2, high2, torch.float32, *shape)
    
    # Rescale result2 to the range of result1 for comparison
    result2_rescaled = (result2 - low2) / (high2 - low2) * (high1 - low1) + low1
    
    # Verify means are similar (uniform distributions should have similar statistical properties)
    assert abs(result1.mean() - 0.5) < 0.1, "Mean should be close to the center of the range"
    assert abs(result2_rescaled.mean() - 0.5) < 0.1, "Mean should be close to the center of the range"


def test_get_rand_weights(monkeypatch):
    """Test _get_rand_weights initializes model weights and saves them to a checkpoint."""
    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_layer = torch.nn.Linear(10, 20)
            self.layernorm = torch.nn.LayerNorm(20)
        
        def forward(self, x):
            return self.layernorm(self.weight_layer(x))
    
    # Mock torch.save
    mock_save = Mock()
    monkeypatch.setattr(torch, "save", mock_save)
    
    model = SimpleModel()
    ckpt_path = "/tmp/test_weights.pt"
    dtype = torch.float16
    weight_range = (-1, 1)
    bias_range = (-2, 2)
    
    # Call function
    result_model = _get_rand_weights(
        model, 
        ckpt_path, 
        dtype=dtype, 
        weight_range=weight_range, 
        bias_range=bias_range
    )
    
    # Verify mock was called
    mock_save.assert_called_once()
    
    # Check model properties
    assert id(result_model) == id(model), "Should return the same model object"
    
    # Check weight parameters
    for name, param in result_model.named_parameters():
        if name.endswith("weight"):
            # LayerNorm weights should stay in FP32 as per _get_rand_weights implementation
            if 'layernorm' in name:
                assert param.dtype == torch.float32, f"{name} should be kept in torch.float32"
            else:
                assert param.dtype == dtype, f"{name} should be converted to {dtype}"
            assert torch.all(param >= weight_range[0]) and torch.all(param < weight_range[1]), \
                f"{name} should be in weight range"
        elif name.endswith("bias"):
            # LayerNorm bias should stay in FP32 as well
            if 'layernorm' in name:
                assert param.dtype == torch.float32, f"{name} should be kept in torch.float32"
            else:
                assert param.dtype == dtype, f"{name} should be converted to {dtype}"
            assert torch.all(param >= bias_range[0]) and torch.all(param < bias_range[1]), \
                f"{name} should be in bias range"


def test_get_shared_checkpoint_path(monkeypatch):
    """Test _get_shared_checkpoint_path returns a path with expected format."""
    # Mock
    mock_uuid = Mock(return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"))
    monkeypatch.setattr(uuid, "uuid4", mock_uuid)
    
    mock_mkdir = Mock()
    monkeypatch.setattr(Path, "mkdir", mock_mkdir)
    
    # Call function
    path = _get_shared_checkpoint_path("/tmp/nxd_inference")
    
    # Verify results
    assert path == "/tmp/nxd_inference/ckpt_12345678.pt", "Path should have correct format with UUID prefix"
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_build_module_with_custom_checkpoint_loader(monkeypatch):
    """Test build_module with a custom checkpoint_loader_fn."""
    # Define a simple dummy model class directly in the test
    class DummyModel(torch.nn.Module):
        def __init__(self, hidden_size=32):
            super().__init__()
            self.linear1 = torch.nn.Linear(16, hidden_size)
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(hidden_size, 8)
        
        def forward(self, x):
            x = self.activation(self.linear1(x))
            return self.linear2(x)
    
    # Setup mocks
    mock_model_builder = Mock()
    mock_model_builder.trace.return_value = Mock()
    mock_model_builder.add = Mock()
    
    mock_model_builder_cls = Mock(return_value=mock_model_builder)
    monkeypatch.setattr("neuronx_distributed_inference.utils.testing.ModelBuilder", mock_model_builder_cls)
    
    mock_save_checkpoint = Mock()
    monkeypatch.setattr("neuronx_distributed_inference.utils.testing._save_checkpoint", mock_save_checkpoint)
    
    # Create model instance to get correct shapes
    dummy = DummyModel(hidden_size=32)
    actual_state_dict = dummy.state_dict()
    
    # Define a custom checkpoint loader with correct key names and tensor shapes
    custom_state_dict = {
        'linear1.weight': torch.randn_like(actual_state_dict['linear1.weight']),
        'linear1.bias': torch.randn_like(actual_state_dict['linear1.bias']),
        'linear2.weight': torch.randn_like(actual_state_dict['linear2.weight']),
        'linear2.bias': torch.randn_like(actual_state_dict['linear2.bias'])
    }
    
    mock_custom_loader = Mock(return_value=custom_state_dict)
    
    # Create test inputs matching the model's expected input shape
    example_inputs = [(torch.zeros((1, 16), dtype=torch.float32),)]
    
    # Call the function with mocked custom checkpoint loader
    build_module(
        DummyModel, 
        example_inputs,
        module_init_kwargs={"hidden_size": 32},
        checkpoint_loader_fn=mock_custom_loader
    )
    
    # Verify ModelBuilder was instantiated correctly
    mock_model_builder_cls.assert_called_once()
    
    # Verify the checkpoint loader was passed to ModelBuilder
    _, kwargs = mock_model_builder_cls.call_args
    assert "checkpoint_loader" in kwargs
    checkpoint_loader_fn = kwargs["checkpoint_loader"]
    
    # Call the checkpoint loader function that was passed to ModelBuilder
    result = checkpoint_loader_fn()
    
    # Verify our mock was called exactly once
    mock_custom_loader.assert_called_once()
    
    # Verify the checkpoint path argument format (exact path will contain a UUID)
    checkpoint_path_arg = mock_custom_loader.call_args[0][0]
    assert str(checkpoint_path_arg).startswith("/tmp/nxdi_test_"), "Checkpoint path should start with the expected prefix"
    assert str(checkpoint_path_arg).endswith("checkpoint.pt"), "Checkpoint path should have the expected suffix"
    
    # Verify the result is our custom state dict
    assert result == custom_state_dict, "Custom checkpoint loader should be used and return our state dict"


def test_build_function_with_custom_checkpoint_loader(monkeypatch):
    """Test build_function with a custom checkpoint_loader_fn."""
    # Setup mocks
    mock_model_builder = Mock()
    mock_model_builder.trace.return_value = Mock()
    mock_model_builder.add = Mock()
    
    mock_model_builder_cls = Mock(return_value=mock_model_builder)
    monkeypatch.setattr("neuronx_distributed_inference.utils.testing.ModelBuilder", mock_model_builder_cls)
    
    mock_save_checkpoint = Mock()
    monkeypatch.setattr("neuronx_distributed_inference.utils.testing._save_checkpoint", mock_save_checkpoint)
    
    # Define a custom state dict and create a mock checkpoint loader
    custom_state_dict = {"function_weight": torch.tensor([2.0])}
    mock_custom_loader = Mock(return_value=custom_state_dict)
    
    # Create a simple function and example inputs
    def example_func(x):
        return x * 2
    
    example_inputs = [(torch.zeros((3), dtype=torch.float32),)]
    
    # Call the function with mocked custom checkpoint loader
    build_function(
        example_func,
        example_inputs,
        checkpoint_loader_fn=mock_custom_loader
    )

    # Verify ModelBuilder was instantiated correctly
    mock_model_builder_cls.assert_called_once()

    # Verify the checkpoint loader was passed to ModelBuilder
    _, kwargs = mock_model_builder_cls.call_args
    assert "checkpoint_loader" in kwargs
    checkpoint_loader_fn = kwargs["checkpoint_loader"]

    # Call the checkpoint loader function that was passed to ModelBuilder
    result = checkpoint_loader_fn()

    # Verify our mock was called exactly once
    mock_custom_loader.assert_called_once()

    # Verify the checkpoint path argument format (exact path will contain a UUID)
    checkpoint_path_arg = mock_custom_loader.call_args[0][0]
    assert str(checkpoint_path_arg).startswith(
        "/tmp/nxdi_test_"), "Checkpoint path should start with the expected prefix"
    assert str(checkpoint_path_arg).endswith("checkpoint.pt"), "Checkpoint path should have the expected suffix"

    # Verify the result is our custom state dict
    assert result == custom_state_dict, "Custom checkpoint loader should be used and return our state dict"
