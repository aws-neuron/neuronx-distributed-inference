import pytest
import os
import tempfile
import shutil
import pickle
import numpy as np
import torch
from unittest.mock import Mock, patch

from neuronx_distributed_inference.utils.snapshot import (
    ScriptModuleWrapper,
    SnapshotOutputFormat,
    get_snapshot_hook,
    _get_all_input_tensors,
    _get_weights_tensors,
    _is_priority_model,
    _apply_weight_layout_transformation,
    _save_tensors,
    _to_numpy,
    _dump_pickle
)
from torch_neuronx.proto import metaneff_pb2


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
    def mock_app_model(self):
        """Create a mock app model."""
        app_model = Mock()
        app_model.models = []
        return app_model
    
    def test_get_snapshot_hook_creation(self, mock_app_model, temp_dir):
        """Test snapshot hook creation."""
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            capture_at_requests=[0],
            app_model=mock_app_model,
            ranks=[0]
        )
        assert callable(hook)
    
    @patch('neuronx_distributed_inference.utils.snapshot._get_all_input_tensors')
    @patch('neuronx_distributed_inference.utils.snapshot._save_tensors')
    @patch('neuronx_distributed_inference.utils.snapshot._is_priority_model')
    def test_snapshot_hook_execution(
        self,
        mock_is_priority,
        mock_save_tensors,
        mock_get_tensors,
        mock_app_model,
        temp_dir
    ):
        """Test snapshot hook execution."""
        # Setup mocks
        mock_traced_model = Mock()
        mock_traced_model.nxd_model.router.return_value = ("test_model", 0)
        mock_is_priority.return_value = False
        mock_get_tensors.return_value = [[torch.tensor([1, 2, 3])]]
        mock_save_tensors.return_value = "test_path"
        
        # Create hook
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            capture_at_requests=[0],
            app_model=mock_app_model,
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
    @patch('neuronx_distributed_inference.utils.snapshot._is_priority_model')
    def test_snapshot_hook_skip_non_capture_requests(
        self, 
        mock_is_priority,
        mock_save_tensors,
        mock_get_tensors,
        mock_app_model,
        temp_dir
    ):
        """Test snapshot hook skips non-capture requests."""
        mock_traced_model = Mock()
        mock_traced_model.nxd_model.router.return_value = ("test_model", 0)
        
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            capture_at_requests=[1],  # Only capture request 1
            app_model=mock_app_model,
            ranks=[0]
        )
        
        # Execute hook for request 0 (should be skipped)
        args = (torch.tensor([1, 2, 3]),)
        hook(mock_traced_model, args, None)
        
        # Verify no tensors were saved
        mock_get_tensors.assert_not_called()
        mock_save_tensors.assert_not_called()

class TestGetAllInputTensors:
    """Test cases for _get_all_input_tensors function."""
    
    def test_get_all_input_tensors(self):
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
        mock_traced_model.nxd_model.flattener_map.test_model = mock_flattener
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
                apply_wlt=True,
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
    def test_get_weights_tensors_no_wlt(self, mock_exists, mock_read_metaneff):
        """Test _get_weights_tensors without weight layout transformation."""
        # Setup mocks
        mock_exists.return_value = True
        mock_app_model = Mock()
        mock_builder = Mock()
        mock_builder.compiler_workdir = "/test/workdir"
        mock_app_model.get_builder.return_value = mock_builder
        
        mock_metaneff = Mock()
        mock_input = Mock()
        mock_input.checkpoint_key.decode.return_value = "weight1"
        mock_input.type = metaneff_pb2.MetaTensor.Type.INPUT_WEIGHT
        mock_metaneff.input_tensors = [mock_input]
        mock_read_metaneff.return_value = mock_metaneff
        
        rank_weights = {"weight1": torch.tensor([1, 2, 3])}
        
        result = _get_weights_tensors(
            mock_app_model, rank_weights, False, "test_model", 0
        )
        
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([1, 2, 3]))
    
    @patch('neuronx_distributed_inference.utils.snapshot._apply_weight_layout_transformation')
    @patch('neuronx_distributed_inference.utils.snapshot.get_input_order')
    @patch('neuronx_distributed_inference.utils.snapshot.read_metaneff')
    @patch('neuronx_distributed_inference.utils.snapshot.os.path.exists')
    def test_get_weights_tensors_with_wlt(
        self,
        mock_exists,
        mock_read_metaneff,
        mock_get_input_order, 
        mock_apply_wlt
    ):
        """Test _get_weights_tensors with weight layout transformation."""
        # Setup mocks
        mock_exists.return_value = True
        mock_app_model = Mock()
        mock_builder = Mock()
        mock_builder.compiler_workdir = "/test/workdir"
        mock_app_model.get_builder.return_value = mock_builder
        
        mock_metaneff = Mock()
        mock_input = Mock()
        mock_input.checkpoint_key.decode.return_value = "weight1"
        mock_input.type = metaneff_pb2.MetaTensor.Type.INPUT_WEIGHT
        mock_metaneff.input_tensors = [mock_input]
        mock_read_metaneff.return_value = mock_metaneff
        mock_get_input_order.return_value = (["weight1"], None)
        
        rank_weights = {"weight1": torch.tensor([1, 2, 3])}
        
        result = _get_weights_tensors(
            mock_app_model, rank_weights, True, "test_model", 0
        )
        
        mock_apply_wlt.assert_called_once()
        assert len(result) == 1

class TestIsPriorityModel:
    """Test cases for _is_priority_model function."""
    
    def test_is_priority_model_true(self):
        """Test _is_priority_model returns True for priority model."""
        mock_model = Mock()
        mock_model.tag = "test_model"
        mock_model.priority_model_idx = 0
        
        mock_app_model = Mock()
        mock_app_model.models = [mock_model]
        
        result = _is_priority_model(mock_app_model, "test_model", 0)
        assert result is True
    
    def test_is_priority_model_false(self):
        """Test _is_priority_model returns False for non-priority model."""
        mock_model = Mock()
        mock_model.tag = "test_model"
        mock_model.priority_model_idx = 1
        
        mock_app_model = Mock()
        mock_app_model.models = [mock_model]
        
        result = _is_priority_model(mock_app_model, "test_model", 0)
        assert result is False

class TestApplyWeightLayoutTransformation:
    """Test cases for _apply_weight_layout_transformation function."""
    
    @patch('neuronx_distributed_inference.utils.snapshot.get_wlt_map')
    @patch('neuronx_distributed_inference.utils.snapshot.read_hlo')
    def test_apply_weight_layout_transformation(self, mock_read_hlo, mock_get_wlt_map):
        """Test _apply_weight_layout_transformation function."""
        # Setup mocks
        mock_transform = Mock(return_value=torch.tensor([4, 5, 6]))
        mock_get_wlt_map.return_value = {0: mock_transform}
        
        checkpoint = {"weight1": torch.tensor([1, 2, 3])}
        checkpoint_keys = ["weight1"]
        
        _apply_weight_layout_transformation(
            checkpoint, "/test/hlo/path", checkpoint_keys
        )
        
        # Check that mock_transform was called once
        assert mock_transform.call_count == 1
        
        # Check the arguments manually using torch.equal
        args, kwargs = mock_transform.call_args
        assert len(args) == 1
        assert torch.equal(args[0], torch.tensor([1, 2, 3]))
        
        # Check the result
        assert torch.equal(checkpoint["weight1"], torch.tensor([4, 5, 6]))

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
        
        mock_app_model = Mock()
        mock_app_model.models = []
        
        mock_traced_model = Mock()
        mock_traced_model.nxd_model.router.return_value = ("test_model", 0)
        mock_flattener = Mock(return_value=[torch.tensor([1, 2, 3])])
        mock_traced_model.nxd_model.flattener_map.test_model = mock_flattener
        mock_traced_model.nxd_model.state = {
            0: {"state1": torch.tensor([0, 1, 2])},
            1: {"state1": torch.tensor([4, 5, 6])}
        }
        mock_traced_model.nxd_model.weights = {0: {}, 1: {}}
        
        # Create and execute hook
        hook = get_snapshot_hook(
            output_path=temp_dir,
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            capture_at_requests=[0],
            app_model=mock_app_model,
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

if __name__ == "__main__":
    pytest.main([__file__])