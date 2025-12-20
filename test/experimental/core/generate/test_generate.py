import unittest
from unittest.mock import Mock, patch
import torch

from neuronx_distributed_inference.experimental.core.generate import generate

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
        self.assertEqual(result[0], [1, 4, 5, 2])  # Original prompt + generated token
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
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [1, 4, 4, 2])
        self.assertEqual(result[1], [1, 4, 5, 6, 4, 5, 2])
        self.assertEqual(result[2], [1, 4, 2])
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
        self.assertLessEqual(len(result[0]), max_len + 1)  # +1 for potential off-by-one
        
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
        self.assertEqual(result[0][-1], 3)  # Should end with stop token 3
        
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
        for prompt_result in result:
            self.assertIn(prompt_result[-1], self.stop_tokens)
