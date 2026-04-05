import unittest
from unittest.mock import Mock, patch
import torch

from neuronx_distributed_inference.experimental.core.generate import generate, GenerateResult

class TestGenerate(unittest.TestCase):
    
    def setUp(self):
        """Set up common test fixtures."""
        self.mock_model = Mock()
        self.pad_token = 0
        self.stop_tokens = [2, 3]  # EOS tokens
        
    def test_single_prompt_basic_generation(self):
        """Test generation with a single prompt."""
        # Setup
        prompt_tokens = [[1, 4, 5]]
        max_len = 10
        
        # Mock model to return predictable logits
        logits_sequence = [
            torch.tensor([[[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]]]),  # Next token: 2 (stop token)
        ]
        self.mock_model.forward.side_effect = logits_sequence
        
        # Execute
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=self.pad_token
        )
        
        # Assert
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(result.prompt_tokens[0], [1, 4, 5, 2])  # Original prompt + generated token
        self.assertIsNone(result.logits)  # Default behavior: no logits returned
        self.mock_model.forward.assert_called_once()
        
    def test_batch_generation_different_lengths(self):
        """Test generation with multiple prompts of different lengths."""
        # Setup
        prompt_tokens = [[1, 4], [1, 4, 5, 6], [1]]
        max_len = 10
        
        # Mock model to generate different tokens for each batch
        logits_sequence = [
            torch.tensor([
                [[0.1, 0.2, 0.3, 0.4, 0.9, 0.5]],  # Batch 0: token 4
                [[0.1, 0.2, 0.3, 0.4, 0.9, 0.5]],  # Batch 1: token 4
                [[0.1, 0.2, 0.3, 0.4, 0.9, 0.5]],  # Batch 2: token 4
            ]),
            torch.tensor([
                [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 0: token 2 (stop)
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.9]],  # Batch 1: token 5
                [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 2: token 2 (stop)
            ]),
            torch.tensor([
                [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 0: already stopped
                [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 1: token 2 (stop)
                [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 2: already stopped
            ]),
        ]
        self.mock_model.forward.side_effect = logits_sequence
        
        # Execute
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=self.pad_token
        )
        
        # Assert
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(len(result.prompt_tokens), 3)
        self.assertEqual(result.prompt_tokens[0], [1, 4, 4, 2])
        self.assertEqual(result.prompt_tokens[1], [1, 4, 5, 6, 4, 5, 2])
        self.assertEqual(result.prompt_tokens[2], [1, 4, 2])
        self.assertIsNone(result.logits)  # Default behavior: no logits returned
        self.assertEqual(self.mock_model.forward.call_count, 3)
        
    def test_max_length_termination(self):
        """Test that generation stops when max_len is reached."""
        # Setup
        prompt_tokens = [[1, 4, 5]]
        max_len = 5  # Short max length
        
        # Mock model to never generate stop token
        self.mock_model.forward.return_value = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.9, 0.5]]])
        
        # Execute
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=self.pad_token
        )
        
        # Assert - should stop at max_len
        self.assertIsInstance(result, GenerateResult)
        self.assertLessEqual(len(result.prompt_tokens[0]), max_len + 1)  # +1 for potential off-by-one
        
    def test_multiple_stop_tokens(self):
        """Test that generation stops on any of the stop tokens."""
        # Setup
        prompt_tokens = [[1, 4]]
        max_len = 10
        stop_tokens = [2, 3, 7]
        
        # Mock model to generate stop token 3
        logits_sequence = [
            torch.tensor([[[0.1, 0.2, 0.3, 0.9, 0.4, 0.5]]]),  # Token 3 (stop)
        ]
        self.mock_model.forward.side_effect = logits_sequence
        
        # Execute
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=stop_tokens,
            pad_token=self.pad_token
        )
        
        # Assert
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(result.prompt_tokens[0][-1], 3)  # Should end with stop token 3
        self.assertIsNone(result.logits)  # Default behavior: no logits returned
        
    def test_attention_mask_padding(self):
        """Test that attention mask is correctly padded during generation."""
        # Setup
        prompt_tokens = [[1, 4], [1, 4, 5]]
        max_len = 10
        
        # Track attention mask calls
        attention_masks = []
        
        def mock_forward(input_tokens, last_pos, attention_mask):
            attention_masks.append(attention_mask.clone())
            return torch.tensor([
                [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],
                [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],
            ])
        
        self.mock_model.forward.side_effect = mock_forward
        
        # Execute
        generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=self.pad_token
        )
        
        # Assert - check initial attention mask
        initial_mask = attention_masks[0]
        self.assertEqual(initial_mask[0].tolist(), [1, 1, 0])  # First prompt padded
        self.assertEqual(initial_mask[1].tolist(), [1, 1, 1])  # Second prompt full
        
    def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        # Setup
        prompt_tokens = [[]]
        max_len = 5
        
        # This should handle empty prompts gracefully
        with self.assertRaises(Exception):
            # Empty prompts should cause issues with last_pos calculation
            generate(
                model=self.mock_model,
                max_len=max_len,
                prompt_tokens=prompt_tokens,
                stop_tokens=self.stop_tokens,
                pad_token=self.pad_token
            )
        
    def test_greedy_sampling(self):
        """Test that argmax is used for greedy sampling."""
        # Setup
        prompt_tokens = [[1, 4]]
        max_len = 10
        
        # Create logits where different positions have max values
        logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.95, 0.5]]])  # Token 4 has highest
        self.mock_model.forward.return_value = logits
        
        # Execute one step
        with patch('torch.argmax') as mock_argmax:
            mock_argmax.return_value = torch.tensor([[2]])  # Simulate stop token
            
            generate(
                model=self.mock_model,
                max_len=max_len,
                prompt_tokens=prompt_tokens,
                stop_tokens=self.stop_tokens,
                pad_token=self.pad_token
            )
            
            # Assert argmax was called correctly
            mock_argmax.assert_called()
            call_args = mock_argmax.call_args
            self.assertEqual(call_args[1]['dim'], -1)  # Should use last dimension
            self.assertTrue(call_args[1]['keepdim'])
            
    def test_is_gen_complete_tracking(self):
        """Test that generation completion is tracked correctly per batch."""
        # Setup
        prompt_tokens = [[1], [1], [1]]
        max_len = 10
        
        # Mock different stop times for each batch element
        logits_sequence = [
            torch.tensor([
                [[0.1, 0.2, 0.9, 0.3, 0.4]],  # Batch 0: token 2 (stop)
                [[0.1, 0.9, 0.2, 0.3, 0.4]],  # Batch 1: token 1
                [[0.1, 0.9, 0.2, 0.3, 0.4]],  # Batch 2: token 1
            ]),
            torch.tensor([
                [[0.1, 0.2, 0.3, 0.4, 0.9]],  # Batch 0: already stopped
                [[0.1, 0.2, 0.9, 0.3, 0.4]],  # Batch 1: token 2 (stop)
                [[0.1, 0.9, 0.2, 0.3, 0.4]],  # Batch 2: token 1
            ]),
            torch.tensor([
                [[0.1, 0.2, 0.3, 0.4, 0.9]],  # Batch 0: already stopped
                [[0.1, 0.2, 0.3, 0.4, 0.9]],  # Batch 1: already stopped
                [[0.1, 0.2, 0.9, 0.3, 0.4]],  # Batch 2: token 2 (stop)
            ]),
        ]
        self.mock_model.forward.side_effect = logits_sequence
        
        # Execute
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=self.pad_token
        )
        
        # Assert all batches have stop token
        self.assertIsInstance(result, GenerateResult)
        for prompt_result in result.prompt_tokens:
            self.assertIn(prompt_result[-1], self.stop_tokens)
        self.assertIsNone(result.logits)  # Default behavior: no logits returned

    def test_return_logits_false_default(self):
        """Test that logits are not returned by default (return_logits=False)."""
        # Setup
        prompt_tokens = [[1, 4, 5]]
        max_len = 10
        
        # Mock model to return predictable logits
        logits_sequence = [
            torch.tensor([[[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]]]),  # Next token: 2 (stop token)
        ]
        self.mock_model.forward.side_effect = logits_sequence
        
        # Execute with explicit return_logits=False
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=self.pad_token,
            return_logits=False
        )
        
        # Assert
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(result.prompt_tokens[0], [1, 4, 5, 2])
        self.assertIsNone(result.logits)  # Should be None when return_logits=False
        
    def test_return_logits_true(self):
        """Test that logits are returned when return_logits=True."""
        # Setup
        prompt_tokens = [[1, 4, 5]]
        max_len = 10
        
        # Mock model to return predictable logits
        logits_tensor = torch.tensor([[[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]]])
        logits_sequence = [logits_tensor]
        self.mock_model.forward.side_effect = logits_sequence
        
        # Execute with return_logits=True
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=self.pad_token,
            return_logits=True
        )
        
        # Assert
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(result.prompt_tokens[0], [1, 4, 5, 2])
        self.assertIsNotNone(result.logits)  # Should contain logits when return_logits=True
        self.assertIsInstance(result.logits, list)
        self.assertEqual(len(result.logits), 1)  # One generation step
        self.assertTrue(torch.equal(result.logits[0], logits_tensor[:, -1]))

    def test_pad_token_removal(self):
        """Test that pad tokens are properly removed from input prompt sequences."""
        # Setup - Test various scenarios with pad tokens
        pad_token = 0
        
        # Test Case 1: Basic pad token removal - pad tokens mixed with regular tokens
        prompt_tokens_with_pads = [
            [1, 0, 4, 5, 0],  # Pad tokens at beginning and end
            [0, 0, 1, 4, 5],  # Pad tokens at beginning
            [1, 4, 0, 0, 5],  # Pad tokens in middle
            [1, 4, 5, 0, 0],  # Pad tokens at end
        ]
        max_len = 10
        
        # Mock model to return stop tokens immediately
        logits_tensor = torch.tensor([
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 0: token 2 (stop)
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 1: token 2 (stop)
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 2: token 2 (stop)
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 3: token 2 (stop)
        ])
        self.mock_model.forward.return_value = logits_tensor
        
        # Execute
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens_with_pads,
            stop_tokens=self.stop_tokens,
            pad_token=pad_token
        )
        
        # Assert - verify pad tokens are removed and generation works correctly
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(len(result.prompt_tokens), 4)
        
        # Check that pad tokens were removed from each prompt
        self.assertEqual(result.prompt_tokens[0], [1, 4, 5, 2])  # [1, 0, 4, 5, 0] -> [1, 4, 5] + generated [2]
        self.assertEqual(result.prompt_tokens[1], [1, 4, 5, 2])  # [0, 0, 1, 4, 5] -> [1, 4, 5] + generated [2]
        self.assertEqual(result.prompt_tokens[2], [1, 4, 5, 2])  # [1, 4, 0, 0, 5] -> [1, 4, 5] + generated [2]
        self.assertEqual(result.prompt_tokens[3], [1, 4, 5, 2])  # [1, 4, 5, 0, 0] -> [1, 4, 5] + generated [2]
        
        # Test Case 2: Edge case - prompts with no pad tokens (should remain unchanged)
        self.mock_model.reset_mock()  # Reset mock to avoid tensor dimension conflicts
        prompt_tokens_no_pads = [[1, 4, 5], [1, 2, 3, 4]]
        
        # Set up logits tensor for 2 prompts
        logits_tensor_no_pads = torch.tensor([
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 0: token 2 (stop)
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 1: token 2 (stop)
        ])
        self.mock_model.forward.return_value = logits_tensor_no_pads
        
        result_no_pads = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens_no_pads,
            stop_tokens=self.stop_tokens,
            pad_token=pad_token
        )
        
        # Assert - prompts without pad tokens should work normally
        self.assertIsInstance(result_no_pads, GenerateResult)
        self.assertEqual(result_no_pads.prompt_tokens[0], [1, 4, 5, 2])
        self.assertEqual(result_no_pads.prompt_tokens[1], [1, 2, 3, 4, 2])
        
        # Test Case 3: Mixed batch - some prompts with pads, others without
        self.mock_model.reset_mock()  # Reset mock to avoid tensor dimension conflicts
        mixed_prompt_tokens = [
            [1, 4, 5],        # No pad tokens
            [0, 1, 4, 0, 5],  # Pad tokens mixed in
            [1, 2, 3],        # No pad tokens
        ]
        
        mixed_logits_tensor = torch.tensor([
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 0: token 2 (stop)
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 1: token 2 (stop)
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 2: token 2 (stop)
        ])
        self.mock_model.forward.return_value = mixed_logits_tensor
        
        result_mixed = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=mixed_prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=pad_token
        )
        
        # Assert - mixed batch should work correctly
        self.assertIsInstance(result_mixed, GenerateResult)
        self.assertEqual(len(result_mixed.prompt_tokens), 3)
        self.assertEqual(result_mixed.prompt_tokens[0], [1, 4, 5, 2])  # No change needed
        self.assertEqual(result_mixed.prompt_tokens[1], [1, 4, 5, 2])  # [0, 1, 4, 0, 5] -> [1, 4, 5] + [2]
        self.assertEqual(result_mixed.prompt_tokens[2], [1, 2, 3, 2])  # No change needed
        
        # Test Case 4: Verify pad token removal works with return_logits=True
        self.mock_model.reset_mock()  # Reset mock to avoid tensor dimension conflicts
        
        # Set up logits tensor for 1 prompt
        logits_tensor_with_logits = torch.tensor([
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 0: token 2 (stop)
        ])
        self.mock_model.forward.return_value = logits_tensor_with_logits
        
        result_with_logits = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=[[0, 1, 4, 5, 0]],  # Pad tokens at start and end
            stop_tokens=self.stop_tokens,
            pad_token=pad_token,
            return_logits=True
        )
        
        # Assert - should work with logits collection too
        self.assertIsInstance(result_with_logits, GenerateResult)
        self.assertEqual(result_with_logits.prompt_tokens[0], [1, 4, 5, 2])  # Pad tokens removed
        self.assertIsNotNone(result_with_logits.logits)
        self.assertIsInstance(result_with_logits.logits, list)
        
    def test_return_logits_true_multiple_steps(self):
        """Test that logits are collected for multiple generation steps."""
        # Setup
        prompt_tokens = [[1, 4]]
        max_len = 10
        
        # Mock model to generate multiple tokens before stopping
        logits_tensor_1 = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.9, 0.5]]])  # Token 4
        logits_tensor_2 = torch.tensor([[[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]]])  # Token 2 (stop)
        logits_sequence = [logits_tensor_1, logits_tensor_2]
        self.mock_model.forward.side_effect = logits_sequence
        
        # Execute with return_logits=True
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=self.pad_token,
            return_logits=True
        )
        
        # Assert
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(result.prompt_tokens[0], [1, 4, 4, 2])  # Original + 4 + 2
        self.assertIsNotNone(result.logits)
        self.assertIsInstance(result.logits, list)
        self.assertEqual(len(result.logits), 2)  # Two generation steps
        self.assertTrue(torch.equal(result.logits[0], logits_tensor_1[:, -1]))
        self.assertTrue(torch.equal(result.logits[1], logits_tensor_2[:, -1]))
        
    def test_return_logits_true_batch_generation(self):
        """Test that logits are returned correctly for batch generation."""
        # Setup
        prompt_tokens = [[1, 4], [1, 4, 5]]
        max_len = 10
        
        # Mock model to generate tokens for batch
        logits_tensor = torch.tensor([
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 0: token 2 (stop)
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]],  # Batch 1: token 2 (stop)
        ])
        logits_sequence = [logits_tensor]
        self.mock_model.forward.side_effect = logits_sequence
        
        # Execute with return_logits=True
        result = generate(
            model=self.mock_model,
            max_len=max_len,
            prompt_tokens=prompt_tokens,
            stop_tokens=self.stop_tokens,
            pad_token=self.pad_token,
            return_logits=True
        )
        
        # Assert
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(len(result.prompt_tokens), 2)
        self.assertEqual(result.prompt_tokens[0], [1, 4, 2])
        self.assertEqual(result.prompt_tokens[1], [1, 4, 5, 2])
        self.assertIsNotNone(result.logits)
        self.assertIsInstance(result.logits, list)
        self.assertEqual(len(result.logits), 1)  # One generation step
        self.assertTrue(torch.equal(result.logits[0], logits_tensor[:, -1]))
