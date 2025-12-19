import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch
from typing import Dict

import torch

from neuronx_distributed_inference.experimental.core.checkpoint import (
    load_hf_safetensors_sharded,
    _load_from_files,
    _HF_SAFETENSORS_MODEL_INDEX_FILENAME_JSON
)


class TestCheckpoint(unittest.TestCase):
    """Test cases for checkpoint loading functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name
        self.sample_tensor_1 = torch.tensor([1.0, 2.0, 3.0])
        self.sample_tensor_2 = torch.tensor([4.0, 5.0, 6.0])
        self.sample_tensor_3 = torch.tensor([7.0, 8.0, 9.0])

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_load_hf_safetensors_sharded_success(self):
        """Test successful loading of sharded safetensors model."""
        # Mock index.json content
        mock_index_content = {
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
                "layer3.bias": "model-00001-of-00002.safetensors"
            }
        }
        
        # Mock file contents
        mock_file1_content = {
            "layer1.weight": self.sample_tensor_1,
            "layer3.bias": self.sample_tensor_3
        }
        mock_file2_content = {
            "layer2.weight": self.sample_tensor_2
        }
        
        def mock_load_file(path):
            if "model-00001-of-00002.safetensors" in path:
                return mock_file1_content
            elif "model-00002-of-00002.safetensors" in path:
                return mock_file2_content
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_index_content))):
            with patch('neuronx_distributed_inference.experimental.core.checkpoint.load_file', side_effect=mock_load_file):
                result = load_hf_safetensors_sharded(self.test_dir)
        
        # Verify results
        expected_result = {
            "layer1.weight": self.sample_tensor_1,
            "layer2.weight": self.sample_tensor_2,
            "layer3.bias": self.sample_tensor_3
        }
        
        self.assertEqual(len(result), 3)
        for key in expected_result:
            self.assertTrue(torch.equal(result[key], expected_result[key]))

    def test_load_hf_safetensors_sharded_missing_index_file(self):
        """Test behavior when index.json file is missing."""
        with patch('builtins.open', side_effect=FileNotFoundError("No such file")):
            with self.assertRaises(FileNotFoundError):
                load_hf_safetensors_sharded(self.test_dir)

    def test_load_hf_safetensors_sharded_missing_weight_map(self):
        """Test behavior when index.json is missing weight_map key."""
        mock_index_content = {"metadata": {"total_size": 1000}}
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_index_content))):
            with self.assertRaises(KeyError):
                load_hf_safetensors_sharded(self.test_dir)

    def test_load_hf_safetensors_sharded_missing_safetensors_file(self):
        """Test behavior when referenced safetensors file doesn't exist."""
        mock_index_content = {
            "weight_map": {
                "layer1.weight": "missing-file.safetensors"
            }
        }
        
        def mock_load_file(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_index_content))):
            with patch('neuronx_distributed_inference.experimental.core.checkpoint.load_file', side_effect=mock_load_file):
                with self.assertRaises(FileNotFoundError):
                    load_hf_safetensors_sharded(self.test_dir)

    def test_load_from_files_success(self):
        """Test successful loading from multiple files."""
        filenames = ["file1.safetensors", "file2.safetensors"]
        
        mock_file1_content = {"layer1.weight": self.sample_tensor_1}
        mock_file2_content = {"layer2.weight": self.sample_tensor_2}
        
        def mock_load_func(path):
            if "file1.safetensors" in path:
                return mock_file1_content
            elif "file2.safetensors" in path:
                return mock_file2_content
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        result = _load_from_files(filenames, self.test_dir, mock_load_func)
        
        expected_result = {
            "layer1.weight": self.sample_tensor_1,
            "layer2.weight": self.sample_tensor_2
        }
        
        self.assertEqual(len(result), 2)
        for key in expected_result:
            self.assertTrue(torch.equal(result[key], expected_result[key]))

    def test_load_from_files_duplicate_keys(self):
        """Test behavior when duplicate keys exist across files."""
        filenames = ["file1.safetensors", "file2.safetensors"]
        
        # Both files contain the same key
        mock_file1_content = {"layer1.weight": self.sample_tensor_1}
        mock_file2_content = {"layer1.weight": self.sample_tensor_2}  # Same key!
        
        def mock_load_func(path):
            if "file1.safetensors" in path:
                return mock_file1_content
            elif "file2.safetensors" in path:
                return mock_file2_content
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        with self.assertRaises(Exception) as context:
            _load_from_files(filenames, self.test_dir, mock_load_func)
        
        self.assertIn("Found value overriden for key layer1.weight", str(context.exception))

    def test_load_from_files_empty_file_list(self):
        """Test behavior with empty filename list."""
        result = _load_from_files([], self.test_dir, lambda x: {})
        self.assertEqual(result, {})

    def test_load_from_files_duplicate_filenames(self):
        """Test behavior when same filename appears multiple times in list."""
        filenames = ["file1.safetensors", "file1.safetensors"]  # Duplicate filename
        
        mock_file_content = {"layer1.weight": self.sample_tensor_1}
        call_count = 0
        
        def mock_load_func(path):
            nonlocal call_count
            call_count += 1
            return mock_file_content
        
        result = _load_from_files(filenames, self.test_dir, mock_load_func)
        
        # Should only load the file once due to set() usage
        self.assertEqual(call_count, 1)
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result["layer1.weight"], self.sample_tensor_1))

    def test_load_from_files_custom_load_function(self):
        """Test _load_from_files with custom load function."""
        filenames = ["file1.txt"]
        
        def custom_load_func(path):
            # Custom loader that returns a simple dict
            return {"custom_key": torch.tensor([10.0, 20.0])}
        
        result = _load_from_files(filenames, self.test_dir, custom_load_func)
        
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result["custom_key"], torch.tensor([10.0, 20.0])))


    def test_load_hf_safetensors_sharded_index_path_construction(self):
        """Test that index file path is constructed correctly."""
        with tempfile.TemporaryDirectory() as test_dir:
            with patch('builtins.open', mock_open(read_data='{"weight_map": {}}')):
                with patch('neuronx_distributed_inference.experimental.core.checkpoint.load_file', return_value={}):
                    with patch('os.path.join', wraps=os.path.join) as mock_join:
                        load_hf_safetensors_sharded(test_dir)
                        
                        # Verify os.path.join was called with correct arguments for index file
                        mock_join.assert_any_call(test_dir, _HF_SAFETENSORS_MODEL_INDEX_FILENAME_JSON)

    def test_load_from_files_empty_state_dict_from_file(self):
        """Test behavior when a file returns empty state dict."""
        filenames = ["empty_file.safetensors", "normal_file.safetensors"]
        
        def mock_load_func(path):
            if "empty_file.safetensors" in path:
                return {}  # Empty state dict
            elif "normal_file.safetensors" in path:
                return {"layer1.weight": self.sample_tensor_1}
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        result = _load_from_files(filenames, self.test_dir, mock_load_func)
        
        # Should only contain the tensor from the non-empty file
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result["layer1.weight"], self.sample_tensor_1))


if __name__ == "__main__":
    unittest.main()
