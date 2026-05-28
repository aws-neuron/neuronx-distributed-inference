"""
Unit tests for tensor_capture_utils module.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call
import torch
import logging

from neuronx_distributed_inference.utils.tensor_capture_utils import (
    capture_model_tensors,
    get_tensor_capture_hook,
    list_capturable_modules_in_application,
    analyze_captured_tensors,
    TensorCaptureMetadata,
    _get_tensor_capture_config,
    _save_tensor,
)
from neuronx_distributed_inference.models.config import TensorCaptureConfig


class TestTensorCaptureUtils(unittest.TestCase):
    """Test cases for tensor_capture_utils module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

        # Set up mock application model
        self.mock_application = MagicMock()
        self.mock_application._tensor_capture_step = 0

        # Set up mock tensor capture config
        self.mock_tensor_capture_config = TensorCaptureConfig(
            modules_to_capture=["layer1", "layer2"],
            max_intermediate_tensors=3,
            capture_inputs=True
        )

        # Set up mock neuron config with tensor capture config
        self.mock_neuron_config = MagicMock()
        self.mock_neuron_config.tensor_capture_config = self.mock_tensor_capture_config
        
        # Set up mock application model with neuron config
        self.mock_application.neuron_config = self.mock_neuron_config

        # Create some test tensors
        self.test_tensors = [
            torch.tensor([1.0, 2.0]),  # layer1 output
            torch.tensor([3.0, 4.0]),  # layer1 input
            torch.tensor([5.0, 6.0]),  # layer2 output
            torch.tensor([7.0, 8.0]),  # layer2 input
            torch.tensor([9.0, 10.0]),  # manual tensor 1
            torch.tensor([11.0, 12.0]),  # manual tensor 2
            torch.full((1,), float('nan'))  # padding tensor
        ]

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('neuronx_distributed_inference.utils.tensor_capture_utils._save_tensor')
    @patch('logging.info')
    def test_capture_model_tensors(self, mock_log_info, mock_save_tensor):
        """Test capture_model_tensors function."""
        # Test capturing all steps
        capture_model_tensors(
            self.mock_application,
            self.test_tensors,
            tensor_capture_save_dir=self.temp_dir
        )

        # Check that step counter was incremented
        self.assertEqual(self.mock_application._tensor_capture_step, 1)

        # Check that _save_tensor was called for each non-padding tensor
        self.assertEqual(mock_save_tensor.call_count, 6)  # 4 module tensors + 2 manual tensors

        # Check that the padding tensor was detected and logged
        mock_log_info.assert_any_call("Skipped 1 padding tensors (NaN values) in step 1")

        # Test with specific capture indices - should capture
        self.mock_application._tensor_capture_step = 0
        mock_save_tensor.reset_mock()
        mock_log_info.reset_mock()

        capture_model_tensors(
            self.mock_application,
            self.test_tensors,
            capture_indices=[1],
            tensor_capture_save_dir=self.temp_dir
        )

        # Check that tensors were saved (step 1 is in capture_indices)
        self.assertEqual(mock_save_tensor.call_count, 6)

        # Test with specific capture indices - should not capture
        self.mock_application._tensor_capture_step = 1
        mock_save_tensor.reset_mock()
        mock_log_info.reset_mock()

        capture_model_tensors(
            self.mock_application,
            self.test_tensors,
            capture_indices=[3],
            tensor_capture_save_dir=self.temp_dir
        )

        # Check that no tensors were saved (step 2 is not in capture_indices)
        self.assertEqual(mock_save_tensor.call_count, 0)


    def test_get_tensor_capture_hook(self):
        """Test get_tensor_capture_hook function."""
        # Test with default parameters
        hook = get_tensor_capture_hook()
        self.assertIsNotNone(hook)

        # Check that the hook is a partial function
        import functools
        self.assertTrue(isinstance(hook, functools.partial))

        # Check that the hook's function is capture_model_tensors
        self.assertEqual(hook.func, capture_model_tensors)

        # Check default parameters
        self.assertIsNone(hook.keywords['capture_indices'])
        self.assertEqual(hook.keywords['tensor_capture_save_dir'], "captured_tensors")

        # Test with custom parameters
        custom_indices = [1, 3, 5]
        custom_dir = "/custom/path"
        hook = get_tensor_capture_hook(capture_indices=custom_indices, tensor_capture_save_dir=custom_dir)

        # Check custom parameters
        self.assertEqual(hook.keywords['capture_indices'], custom_indices)
        self.assertEqual(hook.keywords['tensor_capture_save_dir'], custom_dir)

    def test_get_tensor_capture_config(self):
        """Test _get_tensor_capture_config helper function."""
        # Test with regular model
        config = _get_tensor_capture_config(self.mock_application)
        self.assertEqual(config, self.mock_tensor_capture_config)
        
        # Test with multimodal model
        from neuronx_distributed_inference.models.image_to_text_model_base import NeuronBaseForImageToText
        mock_multimodal = MagicMock(spec=NeuronBaseForImageToText)
        mock_multimodal.text_config = MagicMock()
        mock_multimodal.text_config.neuron_config = MagicMock()
        mock_multimodal.text_config.neuron_config.tensor_capture_config = self.mock_tensor_capture_config
        
        config = _get_tensor_capture_config(mock_multimodal)
        self.assertEqual(config, self.mock_tensor_capture_config)

    @patch('torch.save')
    @patch('logging.info')
    def test_save_tensor(self, mock_log_info, mock_torch_save):
        """Test _save_tensor helper function."""
        metadata_manager = TensorCaptureMetadata(self.temp_dir)
        test_tensor = torch.tensor([1.0, 2.0])
        file_path = os.path.join(self.temp_dir, "test_tensor.pt")
        
        # Test saving module tensor
        _save_tensor(test_tensor, file_path, metadata_manager, 1, "cte", "output", module_name="layer1")
        
        # Check that tensor was saved
        mock_torch_save.assert_called_once()
        
        # Check that metadata was added
        self.assertIn("test_tensor.pt", metadata_manager.metadata["tensors"])
        tensor_metadata = metadata_manager.metadata["tensors"]["test_tensor.pt"]
        self.assertEqual(tensor_metadata["step"], 1)
        self.assertEqual(tensor_metadata["phase"], "cte")
        self.assertEqual(tensor_metadata["tensor_type"], "output")
        self.assertEqual(tensor_metadata["module_name"], "layer1")
        
        # Check logging
        mock_log_info.assert_called_with(f"Saved layer1 output tensor to {os.path.abspath(file_path)}")

    def test_tensor_capture_metadata(self):
        """Test TensorCaptureMetadata class."""
        metadata_manager = TensorCaptureMetadata(self.temp_dir)
        
        # Test initial state
        self.assertIn("capture_session", metadata_manager.metadata)
        self.assertIn("tensors", metadata_manager.metadata)
        self.assertEqual(len(metadata_manager.metadata["tensors"]), 0)
        
        # Test adding tensor metadata
        metadata_manager.add_tensor(
            filename="test.pt",
            step=1,
            phase="cte",
            tensor_type="output",
            tensor_shape=[2, 3],
            tensor_dtype="torch.float32",
            module_name="layer1"
        )
        
        # Check metadata was added
        self.assertIn("test.pt", metadata_manager.metadata["tensors"])
        tensor_metadata = metadata_manager.metadata["tensors"]["test.pt"]
        self.assertEqual(tensor_metadata["step"], 1)
        self.assertEqual(tensor_metadata["phase"], "cte")
        self.assertEqual(tensor_metadata["tensor_type"], "output")
        self.assertEqual(tensor_metadata["tensor_shape"], [2, 3])
        self.assertEqual(tensor_metadata["tensor_dtype"], "torch.float32")
        self.assertEqual(tensor_metadata["module_name"], "layer1")
        
        # Check that metadata file was created
        metadata_file = os.path.join(self.temp_dir, "capture_metadata.json")
        self.assertTrue(os.path.exists(metadata_file))

    @patch('torch.save')
    @patch('logging.debug')
    def test_capture_model_tensors_disabled_config(self, mock_log_debug, mock_torch_save):
        """Test capture_model_tensors when tensor capture is disabled in config."""
        # Create a mock application with disabled tensor capture
        mock_app = MagicMock()
        mock_app._tensor_capture_step = 0
        
        # Set up neuron config with disabled tensor capture config (None)
        mock_app.neuron_config = MagicMock()
        mock_app.neuron_config.tensor_capture_config = None
        
        # Create test tensors
        test_tensors = [torch.tensor([1.0, 2.0])]
        
        # Call the function
        capture_model_tensors(
            mock_app,
            test_tensors,
            tensor_capture_save_dir=self.temp_dir
        )
        
        # Check that the function logged that tensor capture is disabled
        mock_log_debug.assert_called_with("Tensor capture is not enabled in the model configuration")
        
        # Check that no tensors were saved
        mock_torch_save.assert_not_called()
        
        # Step counter should not be incremented when tensor capture is disabled
        self.assertEqual(mock_app._tensor_capture_step, 0)
        """Test that the hook returned by get_tensor_capture_hook works when called."""
        # Create a hook with specific capture indices
        capture_indices = [2, 4]
        hook = get_tensor_capture_hook(
            capture_indices=capture_indices,
            tensor_capture_save_dir=self.temp_dir
        )
        
        # Create test tensors
        test_tensors = [torch.tensor([1.0, 2.0])]
        
        # Set up mock application
        mock_app = MagicMock()
        mock_app._tensor_capture_step = 0
        mock_app.neuron_config = self.mock_neuron_config
        
        # Mock modules_to_capture to be empty to avoid index error
        self.mock_tensor_capture_config.modules_to_capture = []
        
        # Call the hook - should not save tensors since step 1 is not in capture_indices
        hook(mock_app, test_tensors)
        
        # Check that no tensors were saved
        self.assertEqual(mock_torch_save.call_count, 0)
        
        # Call the hook again - should save tensors now
        hook(mock_app, test_tensors)
        
        # Check that tensors were saved
        self.assertEqual(mock_torch_save.call_count, 1)


    @patch('neuronx_distributed.utils.tensor_capture.get_available_modules')
    @patch('neuronx_distributed.parallel_layers.parallel_state')
    @patch('torch.distributed')
    def test_list_capturable_modules_in_application(self, mock_dist, mock_parallel_state, mock_get_available_modules):
        """Test list_capturable_modules_in_application function."""
        # Set up mocks
        mock_parallel_state.model_parallel_is_initialized.return_value = False
        mock_dist.is_initialized.return_value = False

        # Set up mock model wrapper
        mock_model_wrapper = MagicMock()
        mock_model_wrapper.tag = "test_model"
        mock_model_wrapper.config.neuron_config.tp_degree = 1
        mock_model_wrapper.config.neuron_config.pp_degree = 1
        mock_model_wrapper.config.neuron_config.ep_degree = 1
        mock_model_wrapper.config.neuron_config.world_size = 1
        mock_model_wrapper.config.neuron_config.logical_nc_config = 1

        # Set up mock model instance
        mock_model_instance = MagicMock()
        mock_model = MagicMock()
        mock_model_instance.model_cls.return_value = mock_model
        mock_model_wrapper.get_model_instance.return_value = mock_model_instance

        # Set up mock application model
        mock_app = MagicMock()
        mock_app.models = [mock_model_wrapper]

        # Set up mock available modules
        mock_available_modules = {
            "layer1": "LinearLayer",
            "layer2.attention": "AttentionLayer"
        }
        mock_get_available_modules.return_value = mock_available_modules

        # Call the function
        result = list_capturable_modules_in_application(mock_app)

        # Check the result
        self.assertEqual(result, {"test_model": mock_available_modules})

        # Check that the necessary functions were called
        mock_dist.init_process_group.assert_called_once()
        mock_parallel_state.initialize_model_parallel.assert_called_once()
        # Check that set_aot_mode was called (without checking the specific value)
        self.assertTrue(mock_parallel_state.set_aot_mode.called)
        mock_get_available_modules.assert_called_once_with(mock_model)

        # Test with no models attribute
        mock_app = MagicMock()
        delattr(mock_app, 'models')

        with self.assertRaises(ValueError) as context:
            list_capturable_modules_in_application(mock_app)

        self.assertIn("does not have 'models' attribute", str(context.exception))

        # Test with empty models list
        mock_app = MagicMock()
        mock_app.models = []

        with self.assertRaises(ValueError) as context:
            list_capturable_modules_in_application(mock_app)
        
        self.assertIn("has no model wrappers", str(context.exception))

    def test_analyze_captured_tensors(self):
        """Test analyze_captured_tensors function."""
        # Create test tensor files
        tensor_dir = os.path.join(self.temp_dir, "tensors")
        reference_dir = os.path.join(self.temp_dir, "reference")
        os.makedirs(tensor_dir, exist_ok=True)
        os.makedirs(reference_dir, exist_ok=True)

        # Create test tensors
        test_tensor1 = torch.tensor([1.0, 2.0, 3.0])
        test_tensor2 = torch.tensor([[4.0, 5.0], [6.0, 7.0]])

        # Create metadata file
        metadata = {
            "capture_session": {
                "created_at": 1234567890.0,
                "version": "1.0"
            },
            "tensors": {
                "captured_tensors_cte_step_1_module_layer1_output.pt": {
                    "step": 1,
                    "phase": "cte",
                    "tensor_type": "output",
                    "tensor_shape": [3],
                    "tensor_dtype": "torch.float32",
                    "module_name": "layer1",
                    "timestamp": 1234567890.0
                },
                "captured_tensors_cte_step_1_module_layer1_input.pt": {
                    "step": 1,
                    "phase": "cte",
                    "tensor_type": "input",
                    "tensor_shape": [2, 2],
                    "tensor_dtype": "torch.float32",
                    "module_name": "layer1",
                    "timestamp": 1234567890.0
                },
                "captured_tensors_tkg_step_2_module_layer2_output.pt": {
                    "step": 2,
                    "phase": "tkg",
                    "tensor_type": "output",
                    "tensor_shape": [3],
                    "tensor_dtype": "torch.float32",
                    "module_name": "layer2",
                    "timestamp": 1234567890.0
                },
                "captured_tensors_tkg_step_2_manual_tensor_0.pt": {
                    "step": 2,
                    "phase": "tkg",
                    "tensor_type": "manual",
                    "tensor_shape": [2, 2],
                    "tensor_dtype": "torch.float32",
                    "tensor_index": 0,
                    "timestamp": 1234567890.0
                }
            }
        }

        # Save metadata file
        with open(os.path.join(tensor_dir, "capture_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save test tensors
        tensor_files = [
            "captured_tensors_cte_step_1_module_layer1_output.pt",
            "captured_tensors_cte_step_1_module_layer1_input.pt",
            "captured_tensors_tkg_step_2_module_layer2_output.pt",
            "captured_tensors_tkg_step_2_manual_tensor_0.pt"
        ]

        for i, filename in enumerate(tensor_files):
            tensor = test_tensor1 if i % 2 == 0 else test_tensor2
            torch.save(tensor, os.path.join(tensor_dir, filename))
            # Save reference tensor with slight difference
            ref_tensor = tensor * 1.01
            torch.save(ref_tensor, os.path.join(reference_dir, filename))

        # Test analyze_captured_tensors without reference
        result = analyze_captured_tensors(tensor_dir)

        # Check the result
        self.assertEqual(result["summary"]["total_tensors"], 4)
        self.assertEqual(len(result["tensors"]), 4)
        self.assertEqual(set(result["summary"]["modules_captured"]), {"layer1", "layer2"})
        self.assertEqual(set(result["summary"]["steps_captured"]), {1, 2})
        self.assertEqual(set(result["summary"]["phases"]), {"cte", "tkg"})

        # Check tensor info
        for tensor_info in result["tensors"]:
            self.assertIn("name", tensor_info)
            self.assertIn("shape", tensor_info)
            self.assertIn("dtype", tensor_info)
            self.assertIn("min", tensor_info)
            self.assertIn("max", tensor_info)
            self.assertIn("mean", tensor_info)
            self.assertIn("std", tensor_info)
            self.assertIn("phase", tensor_info)
            self.assertIn("step", tensor_info)
            self.assertNotIn("ref_comparison", tensor_info)

        # Test analyze_captured_tensors with reference
        result = analyze_captured_tensors(tensor_dir, reference_dir)

        # Check reference comparison
        for tensor_info in result["tensors"]:
            self.assertIn("ref_comparison", tensor_info)

        # Test with non-existent directory
        result = analyze_captured_tensors("/nonexistent/path")
        self.assertEqual(result["summary"]["total_tensors"], 0)

        # Test with missing metadata file
        os.makedirs(os.path.join(self.temp_dir, "no_metadata"), exist_ok=True)
        torch.save(test_tensor1, os.path.join(self.temp_dir, "no_metadata", "captured_tensors_cte_step_1_module_layer1_output.pt"))
        result = analyze_captured_tensors(os.path.join(self.temp_dir, "no_metadata"))
        self.assertEqual(result["summary"]["total_tensors"], 0)

    def test_tensor_capture_config_moe_max_intermediate_tensors(self):
        tensor_capture_config = TensorCaptureConfig(
            modules_to_capture=[],
            capture_inputs=False,
            max_intermediate_tensors=128,
            auto_capture_moe_tensors=True,
        )
        self.assertEqual(tensor_capture_config.max_intermediate_tensors, 129)


if __name__ == "__main__":
    unittest.main()
