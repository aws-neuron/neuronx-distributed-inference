import pytest
import os
import tempfile
import shutil
import pickle
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from neuronx_distributed_inference.utils.snapshot import (
    ScriptModuleWrapper,
    SnapshotOutputFormat,
    SnapshotCaptureConfig,
    get_snapshot_hook,
    _get_all_input_tensors,
    _get_weights_tensors,
    _save_tensors,
    _to_numpy,
    _dump_pickle,
    register_nxd_model_hook,
    unregister_nxd_model_hooks,
    _original_func_map,
    discover_bucket_request_mapping,
)
from torch_neuronx.proto import metaneff_pb2

class FlattenerMapMock:
    pass

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestScriptModuleWrapper:
    """Test cases for ScriptModuleWrapper class."""
    
    def test_init(self):
        """Test ScriptModuleWrapper initialization."""
        mock_module = Mock(spec=torch.jit.ScriptModule)
        wrapper = ScriptModuleWrapper(mock_module)
        assert wrapper.wrapped_module is mock_module
    
    def test_forward(self):
        """Test forward method delegates to wrapped module."""
        mock_module = Mock(spec=torch.jit.ScriptModule)
        mock_module.return_value = "test_output"
        wrapper = ScriptModuleWrapper(mock_module)
        
        result = wrapper("arg1", "arg2", kwarg1="value1")

        assert mock_module.called, "Mock was not called"
        expected_call = (("arg1", "arg2"), {"kwarg1": "value1"})
        actual_call = mock_module.call_args
        assert actual_call == expected_call, f"Expected {expected_call}, got {actual_call}"
        assert result == mock_module.return_value
    
    def test_class_property(self):
        """Test __class__ property returns ScriptModule."""
        mock_module = Mock(spec=torch.jit.ScriptModule)
        wrapper = ScriptModuleWrapper(mock_module)
        assert wrapper.__class__ == torch.jit.ScriptModule
    
    def test_getattr_delegation(self):
        """Test __getattr__ delegates to wrapped module."""
        mock_module = Mock(spec=torch.jit.ScriptModule)
        mock_module.some_attribute = "test_value"
        wrapper = ScriptModuleWrapper(mock_module)
        
        assert wrapper.some_attribute == "test_value"
    
    def test_getattr_super_first(self):
        """Test __getattr__ tries super() first."""
        mock_module = Mock(spec=torch.jit.ScriptModule)
        wrapper = ScriptModuleWrapper(mock_module)
        
        # This should access the wrapped_module attribute from super()
        assert wrapper.wrapped_module is mock_module
    
    def test_setattr_delegation(self):
        """Test __setattr__ delegates to wrapped module when appropriate."""
        mock_module = Mock(spec=torch.jit.ScriptModule)
        wrapper = ScriptModuleWrapper(mock_module)
        
        # Setting a new attribute should work on wrapper
        wrapper.new_attr = "new_value"
        assert wrapper.new_attr == "new_value"
    
    def test_delattr_delegation(self):
        """Test __delattr__ delegates to wrapped module."""
        mock_module = Mock(spec=torch.jit.ScriptModule)
        mock_module.some_attr = "value"
        wrapper = ScriptModuleWrapper(mock_module)

        del wrapper.some_attr
        assert not hasattr(mock_module, "some_attr")
    
    def test_repr(self):
        """Test __repr__ method."""
        mock_module = Mock(spec=torch.jit.ScriptModule)
        mock_module.__repr__ = Mock(return_value="MockModule()")
        wrapper = ScriptModuleWrapper(mock_module)
        
        result = repr(wrapper)
        assert result == "ScriptModuleWrapper(MockModule())"

class TestSnapshotOutputFormat:
    """Test cases for SnapshotOutputFormat enum."""
    
    def test_enum_values(self):
        """Test enum has expected values."""
        assert SnapshotOutputFormat.NUMPY_IMAGES.value == (0,)
        assert SnapshotOutputFormat.NUMPY_PICKLE.value == (1,)

class TestGetSnapshotHook:
    """Test cases for get_snapshot_hook function."""

    @pytest.fixture
    def mock_model_builder(self):
        """Create a mock model builder."""
        model_builder = Mock()
        model_builder.model_collection = []
        return model_builder
    
    def test_get_snapshot_hook_creation(self, mock_model_builder, temp_dir):
        """Test snapshot hook creation."""
        snapshot_config = SnapshotCaptureConfig().capture_at_request([0])
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            snapshot_config=snapshot_config,
            model_builder=mock_model_builder,
            ranks=[0]
        )
        assert callable(hook)
    
    @patch('neuronx_distributed_inference.utils.snapshot._get_all_input_tensors')
    @patch('neuronx_distributed_inference.utils.snapshot._save_tensors')
    def test_snapshot_hook_execution(
        self,
        mock_save_tensors,
        mock_get_tensors,
        mock_model_builder,
        temp_dir
    ):
        """Test snapshot hook execution."""
        # Setup mocks
        mock_traced_model = Mock()
        mock_traced_model.nxd_model.router.return_value = ("test_model", 0)
        mock_get_tensors.return_value = [[torch.tensor([1, 2, 3])]]
        mock_save_tensors.return_value = "test_path"
        
        snapshot_config = SnapshotCaptureConfig().capture_at_request([0])

        # Create hook
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            snapshot_config=snapshot_config,
            model_builder=mock_model_builder,
            ranks=[0]
        )
        
        # Execute hook
        args = (torch.tensor([1, 2, 3]),)
        hook(mock_traced_model, args, None)
        
        # Verify calls
        mock_traced_model.nxd_model.router.assert_called_once_with(args)
        mock_get_tensors.assert_called_once()
        mock_save_tensors.assert_called_once()
    
    @patch('neuronx_distributed_inference.utils.snapshot._get_all_input_tensors')
    @patch('neuronx_distributed_inference.utils.snapshot._save_tensors')
    def test_snapshot_hook_skip_non_capture_requests(
        self, 
        mock_save_tensors,
        mock_get_tensors,
        mock_model_builder,
        temp_dir
    ):
        """Test snapshot hook skips non-capture requests."""
        mock_traced_model = Mock()
        mock_traced_model.nxd_model.router.return_value = ("test_model", 0)

        snapshot_config = SnapshotCaptureConfig().capture_at_request([1]) # Only capture request 1
        
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            snapshot_config=snapshot_config,  
            model_builder=mock_model_builder,
            ranks=[0]
        )
        
        # Execute hook for request 0 (should be skipped)
        args = (torch.tensor([1, 2, 3]),)
        hook(mock_traced_model, args, None)
        
        # Verify no tensors were saved
        mock_get_tensors.assert_not_called()
        mock_save_tensors.assert_not_called()
    
    @patch('neuronx_distributed_inference.utils.snapshot._get_all_input_tensors')
    @patch('neuronx_distributed_inference.utils.snapshot._save_tensors')
    def test_snapshot_hook_execution_token_capture(
        self,
        mock_save_tensors,
        mock_get_tensors,
        mock_model_builder,
        temp_dir
    ):
        """Test snapshot hook execution."""
        # Setup mocks
        mock_traced_model = Mock()
        mock_traced_model.nxd_model.router.return_value = ("test_model", 0)
        mock_get_tensors.return_value = [[torch.tensor([1, 2, 3]), torch.tensor([1]), torch.tensor([[0, 1, 2]])]]
        mock_save_tensors.return_value = "test_path"
        
        snapshot_config = SnapshotCaptureConfig().capture_for_token([3], batch_line=0)

        # Create hook
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            snapshot_config=snapshot_config,
            model_builder=mock_model_builder,
            ranks=[0]
        )
        
        # Execute hook
        args = (torch.tensor([1, 2, 3]), torch.tensor([1]), torch.tensor([[0, 1, 2]]))
        hook(mock_traced_model, args, None)
        
        # Verify calls
        mock_traced_model.nxd_model.router.assert_called_once_with(args)
        mock_get_tensors.assert_called_once()
        mock_save_tensors.assert_called_once()

    @patch('neuronx_distributed_inference.utils.snapshot._get_all_input_tensors')
    @patch('neuronx_distributed_inference.utils.snapshot._save_tensors')
    def test_snapshot_hook_execution_non_token_capture(
        self,
        mock_save_tensors,
        mock_get_tensors,
        mock_model_builder,
        temp_dir
    ):
        """Test snapshot hook execution."""
        # Setup mocks
        mock_traced_model = Mock()
        mock_traced_model.nxd_model.router.return_value = ("test_model", 0)
        mock_get_tensors.return_value = [[torch.tensor([1, 2, 3]), torch.tensor([1]), torch.tensor([[2]])]]
        mock_save_tensors.return_value = "test_path"
        
        snapshot_config = SnapshotCaptureConfig().capture_for_token([1], batch_line=0)

        # Create hook
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            snapshot_config=snapshot_config,
            model_builder=mock_model_builder,
            ranks=[0]
        )
        
        # Execute hook
        args = (torch.tensor([1, 2, 3]), torch.tensor([1]), torch.tensor([[2]]))
        hook(mock_traced_model, args, None)
        
        # Verify no tensors were saved
        mock_get_tensors.assert_not_called()
        mock_save_tensors.assert_not_called()


class TestGetAllInputTensors:
    """Test cases for _get_all_input_tensors function."""
    
    def test_get_all_input_tensors_old(self):
        """Test _get_all_input_tensors function."""
        input_args = [torch.tensor([1]),]
        flattened_args = [torch.tensor([1, 2]),]
        state = torch.tensor([3, 4])
        weights = torch.tensor([5, 6])
        transformed_weights = torch.tensor([7, 8])

        # Setup mocks
        mock_app_model = Mock()
        mock_traced_model = Mock()
        mock_flattener = Mock(return_value=flattened_args)
        mock_traced_model.nxd_model.flattener_map = FlattenerMapMock()
        setattr(mock_traced_model.nxd_model.flattener_map, 'test_model', mock_flattener) # old torchscript models just use the tag
        mock_traced_model.nxd_model.state = {0: {"state0": state}}
        mock_traced_model.nxd_model.weights = {0: {"weight0": weights}}
        
        with patch('neuronx_distributed_inference.utils.snapshot._get_weights_tensors') as mock_get_weights:
            mock_get_weights.return_value = [transformed_weights]
            
            result = _get_all_input_tensors(
                mock_app_model,
                mock_traced_model,
                "test_model",
                bucket_idx=0, 
                input_args=input_args,
                ranks=[0],
            )
            
            assert len(result) == 1  # One rank
            assert len(result[0]) == 3  # input + state + weights tensors
            expected_result_rank0 = flattened_args + [state, transformed_weights]
            assert result[0] == expected_result_rank0

            mock_flattener.assert_called_once()
            mock_get_weights.assert_called_once()
    
    def test_get_all_input_tensors_new(self):
        """Test _get_all_input_tensors function."""
        input_args = [torch.tensor([1]),]
        flattened_args = [torch.tensor([1, 2]),]
        state = torch.tensor([3, 4])
        weights = torch.tensor([5, 6])
        transformed_weights = torch.tensor([7, 8])

        # Setup mocks
        mock_app_model = Mock()
        mock_traced_model = Mock()
        mock_flattener = Mock(return_value=flattened_args)
        mock_traced_model.nxd_model.flattener_map = FlattenerMapMock()
        setattr(mock_traced_model.nxd_model.flattener_map, 'test_model_0', mock_flattener) # new torchscript models use f"{key}_{bucket_idx}"
        mock_traced_model.nxd_model.state = {0: {"state0": state}}
        mock_traced_model.nxd_model.weights = {0: {"weight0": weights}}
        
        with patch('neuronx_distributed_inference.utils.snapshot._get_weights_tensors') as mock_get_weights:
            mock_get_weights.return_value = [transformed_weights]
            
            result = _get_all_input_tensors(
                mock_app_model,
                mock_traced_model,
                "test_model",
                bucket_idx=0, 
                input_args=input_args,
                ranks=[0],
            )
            
            assert len(result) == 1  # One rank
            assert len(result[0]) == 3  # input + state + weights tensors
            expected_result_rank0 = flattened_args + [state, transformed_weights]
            assert result[0] == expected_result_rank0

            mock_flattener.assert_called_once()
            mock_get_weights.assert_called_once()

class TestGetWeightsTensors:
    """Test cases for _get_weights_tensors function."""
    
    @pytest.fixture
    def mock_metaneff(self):
        """Create a mock metaneff object."""
        mock_metaneff = Mock()
        mock_input = Mock()
        mock_input.checkpoint_key.decode.return_value = "weight1"
        mock_input.type = 1  # Assuming INPUT_WEIGHT type
        mock_metaneff.input_tensors = [mock_input]
        return mock_metaneff
    
    @patch('neuronx_distributed_inference.utils.snapshot.read_metaneff')
    @patch('neuronx_distributed_inference.utils.snapshot.os.path.exists')
    def test_get_weights_tensors(self, mock_exists, mock_read_metaneff):
        """Test _get_weights_tensors without weight layout transformation."""
        # Setup mocks
        mock_exists.return_value = True
        mock_builder = Mock()
        mock_builder.compiler_workdir = "/test/workdir"
        
        mock_metaneff = Mock()
        mock_input = Mock()
        mock_input.checkpoint_key.decode.return_value = "weight1"
        mock_input.type = metaneff_pb2.MetaTensor.Type.INPUT_WEIGHT
        mock_metaneff.input_tensors = [mock_input]
        mock_read_metaneff.return_value = mock_metaneff
        
        rank_weights = {"weight1": torch.tensor([1, 2, 3])}
        
        result = _get_weights_tensors(
            mock_builder, rank_weights, "test_model", 0
        )
        
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([1, 2, 3]))

class TestSaveTensors:
    """Test cases for _save_tensors function."""
    
    def test_save_tensors_npy_images(self, temp_dir):
        """Test _save_tensors with NPY_IMAGES format."""
        tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        
        _save_tensors(tensors, temp_dir, SnapshotOutputFormat.NUMPY_IMAGES, rank=0)
        
        # Check files were created
        rank_dir = os.path.join(temp_dir, "rank0")
        assert os.path.exists(rank_dir)
        assert os.path.exists(os.path.join(rank_dir, "input0.npy"))
        assert os.path.exists(os.path.join(rank_dir, "input1.npy"))
        
        # Verify content
        loaded_tensor0 = np.load(os.path.join(rank_dir, "input0.npy"))
        np.testing.assert_array_equal(loaded_tensor0, [1, 2, 3])
    
    def test_save_tensors_npy_pickle(self, temp_dir):
        """Test _save_tensors with NPY_PICKLE format."""
        tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        
        _save_tensors(tensors, temp_dir, SnapshotOutputFormat.NUMPY_PICKLE, rank=0)
        
        # Check pickle file was created
        pickle_file = os.path.join(temp_dir, "inp-000.p")
        assert os.path.exists(pickle_file)
        
        # Verify content
        with open(pickle_file, "rb") as f:
            loaded_data = pickle.load(f)
        
        assert "input0" in loaded_data
        assert "input1" in loaded_data
        np.testing.assert_array_equal(loaded_data["input0"], [1, 2, 3])
    
    def test_save_tensors_invalid_format(self, temp_dir):
        """Test _save_tensors with invalid format raises ValueError."""
        tensors = [torch.tensor([1, 2, 3])]
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            _save_tensors(tensors, temp_dir, "INVALID_FORMAT", 0)

class TestToNumpy:
    """Test cases for _to_numpy function."""
    
    def test_to_numpy_regular_tensor(self):
        """Test _to_numpy with regular tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = _to_numpy(tensor)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    
    def test_to_numpy_bfloat16(self):
        """Test _to_numpy with bfloat16 tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        result = _to_numpy(tensor)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == "|V2"

class TestDumpPickle:
    """Test cases for _dump_pickle function."""
    
    def test_dump_pickle(self, temp_dir):
        """Test _dump_pickle function."""
        test_obj = {"key1": "value1", "key2": [1, 2, 3]}
        file_path = os.path.join(temp_dir, "test.pickle")
        
        _dump_pickle(file_path, test_obj)
        
        # Verify file was created and content is correct
        assert os.path.exists(file_path)
        with open(file_path, "rb") as f:
            loaded_obj = pickle.load(f)
        
        assert loaded_obj == test_obj

class TestIntegration:
    """Integration tests combining multiple components."""
    
    @patch("neuronx_distributed_inference.utils.snapshot._get_weights_tensors")
    def test_end_to_end_snapshot_creation(self, mock_get_weights, temp_dir):
        """Test end-to-end snapshot creation."""
        # Setup comprehensive mocks
        mock_get_weights.side_effect = [
            [torch.tensor([6, 7, 8])],
            [torch.tensor([7, 8, 9])],
        ]
        
        mock_model = Mock()
        mock_model.priority_model_idx = 1
        
        mock_builder = Mock()
        mock_builder.compiler_workdir = "/test/workdir"
        mock_builder.model_collection = {"test_model": mock_model}
        
        mock_traced_model = Mock()
        mock_traced_model.nxd_model.router.return_value = ("test_model", 0)
        mock_flattener = Mock(return_value=[torch.tensor([1, 2, 3])])
        mock_traced_model.nxd_model.flattener_map = FlattenerMapMock()
        setattr(mock_traced_model.nxd_model.flattener_map, 'test_model', mock_flattener) # old torchscript model use only the key
        mock_traced_model.nxd_model.state = {
            0: {"state1": torch.tensor([0, 1, 2])},
            1: {"state1": torch.tensor([4, 5, 6])}
        }
        mock_traced_model.nxd_model.weights = {0: {}, 1: {}}
        
        snapshot_config = SnapshotCaptureConfig().capture_at_request([0])
        # Create and execute hook
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            snapshot_config=snapshot_config,
            model_builder=mock_builder,
            ranks=[0, 1]
        )
        
        args = (torch.tensor([1, 2, 3]),)
        hook(mock_traced_model, args, None)
        
        # Verify files were created with correct contents
        expected_files = [
            ("rank0/input0.npy", [1, 2, 3]),
            ("rank0/input1.npy", [0, 1, 2]),
            ("rank0/input2.npy", [6, 7, 8]),
            ("rank1/input0.npy", [1, 2, 3]),
            ("rank1/input1.npy", [4, 5, 6]),
            ("rank1/input2.npy", [7, 8, 9]),
        ]
        for expected_file, expected_data in expected_files:
            expected_path = os.path.join(temp_dir, "test_model", "_tp0_bk0", "request0", expected_file)
            assert os.path.exists(expected_path)

            data = np.load(expected_path)
            np.testing.assert_equal(data, expected_data)

class TestNxDModelHooks:
    
    @pytest.fixture
    def mock_traced_model(self):
        """Create a mock traced model with nxd_model attribute."""
        traced_model = Mock()
        nxd_model = Mock()
        traced_model.nxd_model = nxd_model
        return traced_model
    
    @pytest.fixture
    def mock_function(self):
        """Create a mock function that returns a predictable value."""
        def test_func(*args, **kwargs):
            return "original_output"
        return test_func
    
    @pytest.fixture
    def mock_hook(self):
        """Create a mock hook function."""
        return Mock()
    
    @pytest.fixture(autouse=True)
    def clear_original_func_map(self):
        """Clear the global function map before each test."""
        _original_func_map.clear()
        yield
        _original_func_map.clear()
    
    def test_register_hook_success(self, mock_traced_model, mock_function, mock_hook):
        """Test successful hook registration."""
        func_name = "test_function"
        
        # Setup the nxd_model to have the function
        setattr(mock_traced_model.nxd_model, func_name, mock_function)
        
        # Register the hook
        register_nxd_model_hook(mock_traced_model, func_name, mock_hook)
        
        # Verify the original function is stored
        assert mock_traced_model.nxd_model in _original_func_map
        assert func_name in _original_func_map[mock_traced_model.nxd_model]
        assert _original_func_map[mock_traced_model.nxd_model][func_name] == mock_function
        
        # Verify the function was replaced
        wrapped_func = getattr(mock_traced_model.nxd_model, func_name)
        assert wrapped_func != mock_function
    
    def test_register_hook_function_not_exists(self, mock_traced_model, mock_hook):
        """Test hook registration fails when function doesn't exist."""
        func_name = "nonexistent_function"

        # Configure the mock to return False for hasattr check
        mock_traced_model.nxd_model = Mock(spec=[])
        
        with pytest.raises(AssertionError, match=f"nxd_model has no function named {func_name}"):
            register_nxd_model_hook(mock_traced_model, func_name, mock_hook)
    
    def test_wrapped_function_calls_original_and_hook(self, mock_traced_model, mock_function, mock_hook):
        """Test that the wrapped function calls both original function and hook."""
        func_name = "test_function"
        setattr(mock_traced_model.nxd_model, func_name, mock_function)
        
        register_nxd_model_hook(mock_traced_model, func_name, mock_hook)
        
        # Call the wrapped function
        wrapped_func = getattr(mock_traced_model.nxd_model, func_name)
        args = ("arg1", "arg2")
        kwargs = {"key": "value"}
        result = wrapped_func(*args, **kwargs)
        
        # Verify original function output is returned
        assert result == "original_output"
        
        # Verify hook was called with correct arguments
        mock_hook.assert_called_once_with(mock_traced_model, args, "original_output")
    
    def test_wrapped_function_preserves_original_behavior(self, mock_traced_model, mock_hook):
        """Test that wrapped function preserves original function's behavior."""
        func_name = "test_function"
        
        # Create a function that modifies its arguments
        original_calls = []
        def original_func(x, y=None):
            original_calls.append((x, y))
            return x * 2
        
        setattr(mock_traced_model.nxd_model, func_name, original_func)
        register_nxd_model_hook(mock_traced_model, func_name, mock_hook)
        
        # Call wrapped function
        wrapped_func = getattr(mock_traced_model.nxd_model, func_name)
        result = wrapped_func(5, y=10)
        
        # Verify original function was called and result is correct
        assert result == 10
        assert original_calls == [(5, 10)]
        
        # Verify hook was called with correct parameters
        mock_hook.assert_called_once_with(mock_traced_model, (5,), 10)
    
    def test_unregister_hook_success(self, mock_traced_model, mock_function, mock_hook):
        """Test successful hook unregistration."""
        func_name = "test_function"
        setattr(mock_traced_model.nxd_model, func_name, mock_function)
        
        # Register then unregister
        register_nxd_model_hook(mock_traced_model, func_name, mock_hook)
        unregister_nxd_model_hooks(mock_traced_model, func_name)
        
        # Verify original function is restored
        restored_func = getattr(mock_traced_model.nxd_model, func_name)
        assert restored_func == mock_function
        
        # Verify function is removed from original_func_map
        assert func_name not in _original_func_map[mock_traced_model.nxd_model]
    
    def test_unregister_hook_function_not_exists(self, mock_traced_model):
        """Test unregister fails when function doesn't exist."""
        func_name = "nonexistent_function"

        # Configure the mock to return False for hasattr check
        mock_traced_model.nxd_model = Mock(spec=[])
        
        with pytest.raises(AssertionError, match=f"nxd_model has no function named {func_name}"):
            unregister_nxd_model_hooks(mock_traced_model, func_name)
    
    def test_unregister_hook_not_registered(self, mock_traced_model, mock_function):
        """Test unregister when hook was never registered (should not raise error)."""
        func_name = "test_function"
        setattr(mock_traced_model.nxd_model, func_name, mock_function)
        
        # This should not raise an error
        unregister_nxd_model_hooks(mock_traced_model, func_name)
        
        # Function should remain unchanged
        assert getattr(mock_traced_model.nxd_model, func_name) == mock_function
    
    def test_multiple_hooks_on_same_model(self, mock_traced_model, mock_hook):
        """Test registering multiple hooks on the same model."""
        func1_name = "function1"
        func2_name = "function2"
        
        def func1():
            return "func1_output"
        
        def func2():
            return "func2_output"
        
        setattr(mock_traced_model.nxd_model, func1_name, func1)
        setattr(mock_traced_model.nxd_model, func2_name, func2)
        
        # Register hooks for both functions
        register_nxd_model_hook(mock_traced_model, func1_name, mock_hook)
        register_nxd_model_hook(mock_traced_model, func2_name, mock_hook)
        
        # Verify both functions are stored
        assert func1_name in _original_func_map[mock_traced_model.nxd_model]
        assert func2_name in _original_func_map[mock_traced_model.nxd_model]
        assert _original_func_map[mock_traced_model.nxd_model][func1_name] == func1
        assert _original_func_map[mock_traced_model.nxd_model][func2_name] == func2
    
    def test_multiple_models_with_hooks(self, mock_hook):
        """Test registering hooks on multiple different models."""
        # Create two different traced models
        traced_model1 = Mock()
        traced_model1.nxd_model = Mock()
        traced_model2 = Mock()
        traced_model2.nxd_model = Mock()
        
        func_name = "test_function"
        
        def func1():
            return "model1_output"
        
        def func2():
            return "model2_output"
        
        setattr(traced_model1.nxd_model, func_name, func1)
        setattr(traced_model2.nxd_model, func_name, func2)
        
        # Register hooks on both models
        register_nxd_model_hook(traced_model1, func_name, mock_hook)
        register_nxd_model_hook(traced_model2, func_name, mock_hook)
        
        # Verify both models are tracked separately
        assert traced_model1.nxd_model in _original_func_map
        assert traced_model2.nxd_model in _original_func_map
        assert _original_func_map[traced_model1.nxd_model][func_name] == func1
        assert _original_func_map[traced_model2.nxd_model][func_name] == func2
    
    def test_hook_exception_handling(self, mock_traced_model, mock_function):
        """Test behavior when hook function raises an exception."""
        func_name = "test_function"
        setattr(mock_traced_model.nxd_model, func_name, mock_function)
        
        # Create a hook that raises an exception
        def failing_hook(traced_model, args, output):
            raise ValueError("Hook failed")
        
        register_nxd_model_hook(mock_traced_model, func_name, failing_hook)
        
        # Call wrapped function - should propagate the exception
        wrapped_func = getattr(mock_traced_model.nxd_model, func_name)
        with pytest.raises(ValueError, match="Hook failed"):
            wrapped_func()
    
    def test_original_function_exception_handling(self, mock_traced_model, mock_hook):
        """Test behavior when original function raises an exception."""
        func_name = "test_function"
        
        def failing_function():
            raise RuntimeError("Original function failed")
        
        setattr(mock_traced_model.nxd_model, func_name, failing_function)
        register_nxd_model_hook(mock_traced_model, func_name, mock_hook)
        
        # Call wrapped function - should propagate the exception
        wrapped_func = getattr(mock_traced_model.nxd_model, func_name)
        with pytest.raises(RuntimeError, match="Original function failed"):
            wrapped_func()
        
        # Hook should not be called when original function fails
        mock_hook.assert_not_called()

# Additional integration tests
class TestHookIntegration:
    
    @pytest.fixture(autouse=True)
    def clear_original_func_map(self):
        """Clear the global function map before each test."""
        _original_func_map.clear()
        yield
        _original_func_map.clear()
    
    def test_register_unregister_cycle(self):
        """Test complete register/unregister cycle."""
        traced_model = Mock()
        nxd_model = Mock()
        traced_model.nxd_model = nxd_model
        
        func_name = "test_function"
        original_func = Mock(return_value="original")
        hook_func = Mock()
        
        setattr(nxd_model, func_name, original_func)
        
        # Register hook
        register_nxd_model_hook(traced_model, func_name, hook_func)
        
        # Call wrapped function
        wrapped_func = getattr(nxd_model, func_name)
        result = wrapped_func("arg1", key="value")
        
        assert result == "original"
        hook_func.assert_called_once_with(traced_model, ("arg1",), "original")
        
        # Unregister hook
        unregister_nxd_model_hooks(traced_model, func_name)
        
        # Verify original function is restored
        restored_func = getattr(nxd_model, func_name)
        assert restored_func == original_func
        
        # Call restored function (hook should not be called again)
        hook_func.reset_mock()
        result2 = restored_func("arg2")
        assert result2 == "original"
        hook_func.assert_not_called()

class TestDiscoverBucketRequestMapping:
    """Test cases for discover_bucket_request_mapping function."""
    
    def test_discover_bucket_request_mapping_without_gaps(self, temp_dir):
        """Test discovery with multiple buckets and requests."""
        model_name = "test_model"
        
        os.makedirs(os.path.join(temp_dir, model_name, "_tp0_bk0", "request0"))
        os.makedirs(os.path.join(temp_dir, model_name, "_tp0_bk0", "request1"))
        os.makedirs(os.path.join(temp_dir, model_name, "_tp0_bk1", "request0"))
        
        result = discover_bucket_request_mapping(Path(temp_dir), model_name)
        
        expected = [(0, 0), (0, 1), (1, 0)]
        assert result == expected
    
    def test_discover_bucket_request_mapping_with_gaps(self, temp_dir):
        """Test discovery with missing buckets (bk0 and bk2 exist, bk1 missing)."""
        model_name = "test_model"
        
        os.makedirs(os.path.join(temp_dir, model_name, "_tp0_bk0", "request0"))
        os.makedirs(os.path.join(temp_dir, model_name, "_tp0_bk2", "request0"))
        os.makedirs(os.path.join(temp_dir, model_name, "_tp0_bk2", "request1"))
        
        result = discover_bucket_request_mapping(Path(temp_dir), model_name)
        
        expected = [(0, 0), (2, 0), (2, 1)]
        assert result == expected

class TestSnapshotCapturerConfig:
    def test_initialization(self):
        """Test that the config initializes with default values."""
        config = SnapshotCaptureConfig()
        assert config.request_indices == set()
        assert config.token_indices == set()
        assert config.capture_all_requests is False
        assert config.capture_all_tokens is False
        assert config._capture_types == set()
        assert config.max_tokens_generated_per_request == 1

    def test_initialization_with_max_tokens(self):
        """Test initialization with custom max tokens."""
        config = SnapshotCaptureConfig(max_tokens_generated_per_request=4)
        assert config.max_tokens_generated_per_request == 4

    def test_capture_at_request_single(self):
        """Test capturing at a single request index."""
        config = SnapshotCaptureConfig().capture_at_request(0)
        assert config.request_indices == {0}
        assert 'request' in config._capture_types
        assert config.is_capturing_requests() is True

    def test_capture_at_request_multiple(self):
        """Test capturing at multiple request indices."""
        config = SnapshotCaptureConfig().capture_at_request([0, 1, 5])
        assert config.request_indices == {0, 1, 5}
        assert 'request' in config._capture_types

    def test_capture_at_all_requests(self):
        """Test capturing at all requests."""
        config = SnapshotCaptureConfig().capture_at_request(-1)
        assert config.capture_all_requests is True
        assert 'request' in config._capture_types

    def test_capture_for_token_single(self):
        """Test capturing at a single token index."""
        config = SnapshotCaptureConfig().capture_for_token(token_indices=244)
        assert (0, 244) in config.token_indices  # Default batch_line is 0
        assert 'token' in config._capture_types
        assert config.is_capturing_tokens() is True

    def test_capture_for_token_multiple(self):
        """Test capturing at multiple token indices."""
        config = SnapshotCaptureConfig().capture_for_token(token_indices=[244, 245, 246])
        assert (0, 244) in config.token_indices
        assert (0, 245) in config.token_indices
        assert (0, 246) in config.token_indices
        assert 'token' in config._capture_types

    def test_capture_for_token_specific_batch(self):
        """Test capturing at a specific batch line."""
        config = SnapshotCaptureConfig().capture_for_token(batch_line=2, token_indices=244)
        assert (2, 244) in config.token_indices
        assert (0, 244) not in config.token_indices

    def test_capture_at_all_tokens(self):
        """Test capturing all tokens."""
        config = SnapshotCaptureConfig().capture_for_token(token_indices=-1)
        assert config.capture_all_tokens is True
        assert 'token' in config._capture_types

    def test_which_token_found(self):
        """Test which_token finds the correct token to capture."""
        config = SnapshotCaptureConfig().capture_for_token(token_indices=244)
        result = config.which_token([243])
        assert result == (0, 244)  # Should find token 244 at batch line 0

    def test_which_token_not_found(self):
        """Test which_token returns None when no token matches."""
        config = SnapshotCaptureConfig().capture_for_token(token_indices=244)
        result = config.which_token([242])
        assert result is None

    def test_which_token_multiple_batch_lines(self):
        """Test which_token with multiple batch lines."""
        config = SnapshotCaptureConfig().capture_for_token(batch_line=1, token_indices=244)
        result = config.which_token([242, 243, 245])
        assert result == (1, 244)

    def test_which_token_speculative_decoding(self):
        """Test which_token with max_tokens_generated_per_request > 1."""
        config = SnapshotCaptureConfig(max_tokens_generated_per_request=3)
        config.capture_for_token(token_indices=246)
        result = config.which_token([243])
        assert result == (0, 246)  # Should find token 246 when we're at 243 and generating 3 tokens

    def test_should_capture_request(self):
        """Test should_capture with request-based capturing."""
        config = SnapshotCaptureConfig().capture_at_request(2)
        assert config.should_capture([10, 11], 2) is True
        assert config.should_capture([10, 11], 3) is False

    def test_should_capture_token(self):
        """Test should_capture with token-based capturing."""
        config = SnapshotCaptureConfig().capture_for_token(token_indices=244)
        assert config.should_capture([243], 0) is True  # Current token is 243, will generate 244
        assert config.should_capture([242], 0) is False  # Current token is 242, will generate 243

    def test_should_capture_all_tokens(self):
        """Test should_capture with all tokens option."""
        config = SnapshotCaptureConfig().capture_for_token(token_indices=-1)
        assert config.should_capture([100], 5) is True
        assert config.should_capture([200], 10) is True

    def test_should_capture_all_requests(self):
        """Test should_capture with all requests option."""
        config = SnapshotCaptureConfig().capture_at_request(-1)
        assert config.should_capture([100], 5) is True
        assert config.should_capture([200], 10) is True

    def test_should_capture_combined(self):
        """Test should_capture with both request and token conditions."""
        config = (SnapshotCaptureConfig()
                  .capture_at_request(2)
                  .capture_for_token(token_indices=244))
        
        # Should capture at request 2 regardless of token
        assert config.should_capture([100], 2) is True
        
        # Should capture when generating token 244 regardless of request
        assert config.should_capture([243], 5) is True
        
        # Should not capture when neither condition is met
        assert config.should_capture([100], 3) is False

    def test_empty_token_list(self):
        """Test capture_for_token with an empty list."""
        config = SnapshotCaptureConfig().capture_for_token(token_indices=[])
        assert config.token_indices == set()
        assert 'token' not in config._capture_types

    def test_method_chaining(self):
        """Test method chaining functionality."""
        config = (SnapshotCaptureConfig()
                  .capture_at_request(0)
                  .capture_for_token(token_indices=244)
                  .capture_for_token(batch_line=1, token_indices=255))
        
        assert 0 in config.request_indices
        assert (0, 244) in config.token_indices
        assert (1, 255) in config.token_indices
        assert 'request' in config._capture_types
        assert 'token' in config._capture_types


if __name__ == "__main__":
    pytest.main([__file__])