import unittest
from unittest.mock import Mock, MagicMock, patch, call
import torch
import torch.nn as nn
import logging

from neuronx_distributed_inference.experimental.core.build_flow import build_for_bucketing_on_seq_len


class TestBuildForBucketingOnSeqLen(unittest.TestCase):
    """Unit tests for build_for_bucketing_on_seq_len function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple mock model
        self.mock_model = Mock(spec=nn.Module)
        self.batch_size = 4
        self.max_seq_len = 2048
        
        # Set up logger mock to suppress log output during tests
        logging.getLogger("Neuron").setLevel(logging.CRITICAL)

    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.ModelBuilder')
    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.generate_buckets')
    def test_successful_build_with_default_buckets(self, mock_generate_buckets, mock_model_builder):
        """Test successful model building with default bucket generation."""
        # Setup mocks
        mock_generate_buckets.return_value = [128, 256, 512, 1024, 2048]
        mock_builder_instance = MagicMock()
        mock_traced_model = Mock()
        mock_builder_instance.compile.return_value = mock_traced_model
        mock_model_builder.return_value = mock_builder_instance

        # Call the function
        result = build_for_bucketing_on_seq_len(
            model=self.mock_model,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len
        )

        # Assertions
        mock_model_builder.assert_called_once_with(model=self.mock_model)

        # Verify generate_buckets was called twice (for prefill and decode)
        self.assertEqual(mock_generate_buckets.call_count, 2)
        mock_generate_buckets.assert_any_call(min_length=128, max_length=self.max_seq_len)

        # Verify trace was called for each bucket (prefill and decode)
        expected_trace_calls = 10  # 5 prefill + 5 decode buckets
        self.assertEqual(mock_builder_instance.trace.call_count, expected_trace_calls)

        # Verify compile was called with correct arguments
        mock_builder_instance.compile.assert_called_once_with(
            priority_model_key="decode_128",
        )
        
        # Verify the result
        self.assertEqual(result, mock_traced_model)

    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.ModelBuilder')
    def test_custom_prefill_and_decode_buckets(self, mock_model_builder):
        """Test with custom prefill and decode buckets."""
        # Setup
        custom_prefill_buckets = [256, 512, 1024]
        custom_decode_buckets = [128, 256]
        mock_builder_instance = MagicMock()
        mock_traced_model = Mock()
        mock_builder_instance.compile.return_value = mock_traced_model
        mock_model_builder.return_value = mock_builder_instance

        # Call the function with custom buckets
        build_for_bucketing_on_seq_len(
            model=self.mock_model,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            prefill_buckets=custom_prefill_buckets,
            decode_buckets=custom_decode_buckets
        )

        # Verify trace was called correct number of times
        expected_trace_calls = len(custom_prefill_buckets) + len(custom_decode_buckets)
        self.assertEqual(mock_builder_instance.trace.call_count, expected_trace_calls)
        
        # Verify compile uses first decode bucket
        mock_builder_instance.compile.assert_called_once_with(
            priority_model_key="decode_128",
        )

    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.ModelBuilder')
    def test_trace_kwargs_for_prefill(self, mock_model_builder):
        """Test that prefill trace is called with correct kwargs."""
        # Setup
        prefill_buckets = [512]
        mock_builder_instance = MagicMock()
        mock_model_builder.return_value = mock_builder_instance

        # Call the function
        build_for_bucketing_on_seq_len(
            model=self.mock_model,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            prefill_buckets=prefill_buckets,
            decode_buckets=[128]  # Minimal decode bucket to focus on prefill
        )

        # Get the first trace call (prefill)
        first_trace_call = mock_builder_instance.trace.call_args_list[0]
        kwargs = first_trace_call[1]['kwargs']
        
        # Verify tensor shapes and types
        self.assertEqual(kwargs['tokens'].shape, (self.batch_size, 512))
        self.assertEqual(kwargs['tokens'].dtype, torch.int32)
        self.assertEqual(kwargs['last_pos'].shape, (self.batch_size,))
        self.assertEqual(kwargs['last_pos'].dtype, torch.int32)
        self.assertEqual(kwargs['attention_mask'].shape, (self.batch_size, 512))
        self.assertEqual(kwargs['attention_mask'].dtype, torch.int32)
        
        # Verify tag
        self.assertEqual(first_trace_call[1]['tag'], 'prefill_512')

    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.ModelBuilder')
    def test_trace_kwargs_for_decode(self, mock_model_builder):
        """Test that decode trace is called with correct kwargs."""
        # Setup
        decode_buckets = [256]
        mock_builder_instance = MagicMock()
        mock_model_builder.return_value = mock_builder_instance

        # Call the function
        build_for_bucketing_on_seq_len(
            model=self.mock_model,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            prefill_buckets=[128],  # Minimal prefill bucket
            decode_buckets=decode_buckets
        )

        # Get the second trace call (decode)
        second_trace_call = mock_builder_instance.trace.call_args_list[1]
        kwargs = second_trace_call[1]['kwargs']
        
        # Verify tensor shapes and types for decode
        self.assertEqual(kwargs['tokens'].shape, (self.batch_size, 1))  # Note: decode uses seq_len=1
        self.assertEqual(kwargs['tokens'].dtype, torch.int32)
        self.assertEqual(kwargs['last_pos'].shape, (self.batch_size,))
        self.assertEqual(kwargs['attention_mask'].shape, (self.batch_size, 256))
        
        # Verify tag
        self.assertEqual(second_trace_call[1]['tag'], 'decode_256')

    def test_non_integer_max_seq_len_raises_assertion(self):
        """Test that non-integer max_seq_len raises AssertionError."""
        with self.assertRaises(AssertionError) as context:
            build_for_bucketing_on_seq_len(
                model=self.mock_model,
                batch_size=self.batch_size,
                max_seq_len="2048"  # String instead of int
            )
        
        self.assertIn("Only integer sequence length is supported", str(context.exception))

    def test_list_max_seq_len_raises_assertion(self):
        """Test that list max_seq_len raises AssertionError."""
        with self.assertRaises(AssertionError) as context:
            build_for_bucketing_on_seq_len(
                model=self.mock_model,
                batch_size=self.batch_size,
                max_seq_len=[512, 1024, 2048]  # List instead of int
            )
        
        self.assertIn("Only integer sequence length is supported", str(context.exception))

    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.logger')
    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.ModelBuilder')
    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.generate_buckets')
    def test_logging_output(self, mock_generate_buckets, mock_model_builder, mock_logger):
        """Test that appropriate logging messages are generated."""
        # Setup
        mock_generate_buckets.return_value = [128, 256]
        mock_builder_instance = MagicMock()
        mock_model_builder.return_value = mock_builder_instance

        # Call the function
        build_for_bucketing_on_seq_len(
            model=self.mock_model,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len
        )

        # Verify logging calls
        self.assertEqual(mock_logger.info.call_count, 2)
        mock_logger.info.assert_any_call("There are 2 prefill buckets: [128, 256]")
        mock_logger.info.assert_any_call("There are 2 decode buckets: [128, 256]")

    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.ModelBuilder')
    def test_different_batch_sizes(self, mock_model_builder):
        """Test with different batch sizes."""
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                mock_builder_instance = MagicMock()
                mock_model_builder.return_value = mock_builder_instance
                
                build_for_bucketing_on_seq_len(
                    model=self.mock_model,
                    batch_size=batch_size,
                    max_seq_len=self.max_seq_len,
                    prefill_buckets=[128],
                    decode_buckets=[128]
                )
                
                # Verify trace was called with correct batch size
                trace_call = mock_builder_instance.trace.call_args_list[0]
                kwargs = trace_call[1]['kwargs']
                self.assertEqual(kwargs['tokens'].shape[0], batch_size)
                self.assertEqual(kwargs['last_pos'].shape[0], batch_size)
                self.assertEqual(kwargs['attention_mask'].shape[0], batch_size)

    @patch('neuronx_distributed_inference.experimental.core.build_flow.bucketing_on_seq_len.ModelBuilder')
    def test_model_builder_initialization(self, mock_model_builder):
        """Test that ModelBuilder is initialized with the correct model."""
        mock_builder_instance = MagicMock()
        mock_model_builder.return_value = mock_builder_instance
        
        custom_model = Mock(spec=nn.Module)
        
        build_for_bucketing_on_seq_len(
            model=custom_model,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            prefill_buckets=[128],
            decode_buckets=[128]
        )
        
        mock_model_builder.assert_called_once_with(model=custom_model)
