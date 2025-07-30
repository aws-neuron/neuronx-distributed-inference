"""
Unit tests for the test_utils.py module.
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
from argparse import Namespace

from test.integration.utils.test_utils import (
    save_checkpoint,
    validate_e2e_performance,
    load_text_model_inputs,
)


class TestSaveCheckpoint(unittest.TestCase):
    """Tests for the save_checkpoint function."""

    @patch("test.integration.utils.test_utils.AutoConfig")
    @patch("test.integration.utils.test_utils.AutoModel")
    def test_save_checkpoint_basic(self, mock_auto_model, mock_auto_config):
        """Test basic functionality of save_checkpoint."""
        # Setup mocks
        mock_config = MagicMock()
        mock_auto_config.from_pretrained.return_value = mock_config
        mock_model = MagicMock()
        mock_auto_model.from_config.return_value = mock_model

        # Call function
        with patch("test.integration.utils.test_utils.tempfile.TemporaryDirectory") as mock_tempdir:
            mock_tempdir_instance = MagicMock()
            mock_tempdir_instance.name = "/tmp/mock_dir"
            mock_tempdir.return_value = mock_tempdir_instance

            result = save_checkpoint("dummy/path")

            # Assertions
            mock_auto_config.from_pretrained.assert_called_once_with("dummy/path")
            mock_auto_model.from_config.assert_called_once_with(
                mock_config, torch_dtype=torch.bfloat16
            )
            mock_model.save_pretrained.assert_called_once_with("/tmp/mock_dir")
            self.assertEqual(result, mock_tempdir_instance)

    @patch("test.integration.utils.test_utils.AutoConfig")
    @patch("test.integration.utils.test_utils.AutoModel")
    def test_save_checkpoint_with_kwargs(self, mock_auto_model, mock_auto_config):
        """Test save_checkpoint with kwargs for config overrides and new attributes."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.get.return_value = "old_value"
        mock_auto_config.from_pretrained.return_value = mock_config

        # Call function with kwargs for both existing and new attributes
        with patch("test.integration.utils.test_utils.tempfile.TemporaryDirectory"):
            save_checkpoint(
                "dummy/path", dtype=torch.float16, existing_attr="new_value", new_attr="value"
            )

            # Assertions
            mock_auto_model.from_config.assert_called_once_with(
                mock_config, torch_dtype=torch.float16
            )

            # Instead of patching setattr, we can directly check if the attributes
            # were set on the mock_config by accessing them
            self.assertEqual(mock_config.existing_attr, "new_value")
            self.assertEqual(mock_config.new_attr, "value")


class TestValidateE2EPerformance(unittest.TestCase):
    """Tests for the validate_e2e_performance function."""

    def test_validate_e2e_performance_no_results(self):
        """Test validate_e2e_performance with no e2e_model results."""
        benchmark_results = {"other_key": "value"}
        with self.assertRaises(ValueError):
            validate_e2e_performance(benchmark_results)

    def test_validate_e2e_performance_no_thresholds(self):
        """Test validate_e2e_performance with no thresholds."""
        benchmark_results = {"e2e_model": {"latency_ms_p50": 100, "throughput": 50}}
        result = validate_e2e_performance(benchmark_results)
        self.assertTrue(result)

    def test_validate_e2e_performance_pass_thresholds(self):
        """Test validate_e2e_performance with passing thresholds."""
        benchmark_results = {"e2e_model": {"latency_ms_p50": 100, "throughput": 50}}
        result = validate_e2e_performance(
            benchmark_results, latency_threshold=200, throughput_threshold=25
        )
        self.assertTrue(result)

    def test_validate_e2e_performance_fail_latency(self):
        """Test validate_e2e_performance with failing latency threshold."""
        benchmark_results = {"e2e_model": {"latency_ms_p50": 100, "throughput": 50}}
        with self.assertRaises(AssertionError):
            validate_e2e_performance(benchmark_results, latency_threshold=50)

    def test_validate_e2e_performance_fail_throughput(self):
        """Test validate_e2e_performance with failing throughput threshold."""
        benchmark_results = {"e2e_model": {"latency_ms_p50": 100, "throughput": 50}}
        with self.assertRaises(AssertionError):
            validate_e2e_performance(benchmark_results, throughput_threshold=100)


class TestLoadModelInputs(unittest.TestCase):
    @patch("test.integration.utils.test_utils.save_checkpoint")
    def test_load_text_model_inputs(self, mock_save_checkpoint):
        """Test load_text_model_inputs function."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.config.neuron_config.batch_size = 2
        mock_model.config.vocab_size = 32000

        # Call the function
        inputs = load_text_model_inputs(mock_model, input_len=24)

        # Assertions
        self.assertIsInstance(inputs, Namespace)
        self.assertEqual(inputs.input_ids.shape, (2, 24))
        self.assertEqual(inputs.attention_mask.shape, (2, 24))
        self.assertEqual(inputs.attention_mask.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
