import pytest
import torch
import numpy as np
from typing import Union, Tuple
from neuronx_distributed_inference.experimental.core.accuracy.logit_validation import (
    logit_validation,
    DEFAULT_TOLERANCE_MAP,
    DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
)


class TestLogitValidation:
    """Test suite for logit_validation function."""

    @pytest.fixture
    def basic_setup(self):
        """Basic setup with small vocab size and sequence length."""
        batch_size = 2
        seq_len = 5
        vocab_size = 1500
        
        # Create input_ids
        input_ids = [[1, 2, 3], [4, 5, 6]]
        
        # Create expected logits with known pattern
        torch.manual_seed(42)
        expected_logits = torch.randn(seq_len, batch_size, vocab_size)
        
        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'vocab_size': vocab_size,
            'input_ids': input_ids,
            'expected_logits': expected_logits,
        }

    def test_exact_match_passes(self, basic_setup):
        """Test that validation passes when logits match exactly."""
        input_ids = basic_setup['input_ids']
        expected_logits = basic_setup['expected_logits']
        
        def generate_fn(ids):
            # Return exact match
            return expected_logits.clone()
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is True

    def test_small_difference_within_tolerance_passes(self, basic_setup):
        """Test that small differences within tolerance pass."""
        input_ids = basic_setup['input_ids']
        expected_logits = basic_setup['expected_logits']
        
        def generate_fn(ids):
            # Add small noise within tolerance
            noise = torch.randn_like(expected_logits) * 1e-6
            return expected_logits + noise
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is True

    def test_large_difference_fails(self, basic_setup):
        """Test that large differences beyond tolerance fail."""
        input_ids = basic_setup['input_ids']
        expected_logits = basic_setup['expected_logits']
        
        def generate_fn(ids):
            # Add large noise beyond tolerance
            noise = torch.randn_like(expected_logits) * 10.0
            return expected_logits + noise
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is False

    def test_custom_tolerance_map(self, basic_setup):
        """Test with custom tolerance map."""
        input_ids = basic_setup['input_ids']
        expected_logits = basic_setup['expected_logits']
        
        # Create logits by adding proportional error
        def generate_fn(ids):
            return expected_logits * 1.02
        
        # Strict tolerance should fail
        strict_tol = {
            "all": (1e-5, 0.001),
            "5": (1e-5, 0.001),
        }
        
        result_strict = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            tol_map=strict_tol,
            suppress_passing=True
        )
        
        # Relaxed tolerance should pass
        relaxed_tol = {
            "all": (0.1, 0.1),
            "5": (0.1, 0.1),
        }
        
        result_relaxed = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            tol_map=relaxed_tol,
            suppress_passing=True
        )
        
        assert result_strict is False
        assert result_relaxed is True

    def test_generate_fn_with_sequences(self, basic_setup):
        """Test generate_fn returning (logits, sequences) tuple."""
        input_ids = basic_setup['input_ids']
        expected_logits = basic_setup['expected_logits']
        
        def generate_fn(ids):
            logits = expected_logits.clone()
            # Return both logits and sampled sequences
            sequences = logits.argmax(dim=2).T
            return logits, sequences
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is True

    def test_generate_fn_only_logits(self, basic_setup):
        """Test generate_fn returning only logits."""
        input_ids = basic_setup['input_ids']
        expected_logits = basic_setup['expected_logits']
        
        def generate_fn(ids):
            # Return only logits
            return expected_logits.clone()
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is True

    def test_single_batch(self):
        """Test with single batch."""
        input_ids = [[1, 2, 3]]
        seq_len = 3
        vocab_size = 1500
        
        expected_logits = torch.randn(seq_len, 1, vocab_size)
        
        def generate_fn(ids):
            return expected_logits.clone()
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is True

    def test_single_token(self):
        """Test with single token generation."""
        input_ids = [[1, 2, 3], [4, 5, 6]]
        seq_len = 1
        vocab_size = 1500
        batch_size = 2
        
        expected_logits = torch.randn(seq_len, batch_size, vocab_size)
        
        def generate_fn(ids):
            return expected_logits.clone()
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is True

    def test_with_inf_values(self):
        """Test handling of -inf values in logits."""
        input_ids = [[1, 2]]
        seq_len = 2
        vocab_size = 1500
        batch_size = 1
        
        expected_logits = torch.randn(seq_len, batch_size, vocab_size)
        # Add -inf values (representing prohibited tokens)
        expected_logits[0, 0, 5] = float('-inf')
        expected_logits[1, 0, 3] = float('-inf')
        
        def generate_fn(ids):
            actual_logits = expected_logits.clone()
            # Also have -inf in actual at same positions
            return actual_logits
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is True

    def test_different_batch_sizes(self):
        """Test with varying batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            input_ids = [[i + j for j in range(3)] for i in range(batch_size)]
            seq_len = 3
            vocab_size = 1500
            
            expected_logits = torch.randn(seq_len, batch_size, vocab_size)
            
            def generate_fn(ids):
                return expected_logits.clone()
            
            result = logit_validation(
                input_ids=input_ids,
                generate_fn=generate_fn,
                expected_logits=expected_logits,
                suppress_passing=True
            )
            
            assert result is True, f"Failed for batch_size={batch_size}"

    def test_different_sequence_lengths(self):
        """Test with varying sequence lengths."""
        for seq_len in [1, 5, 10, 20]:
            input_ids = [[1, 2, 3], [4, 5, 6]]
            batch_size = 2
            vocab_size = 1500
            
            expected_logits = torch.randn(seq_len, batch_size, vocab_size)
            
            def generate_fn(ids):
                return expected_logits.clone()
            
            result = logit_validation(
                input_ids=input_ids,
                generate_fn=generate_fn,
                expected_logits=expected_logits,
                suppress_passing=True
            )
            
            assert result is True, f"Failed for seq_len={seq_len}"


    def test_constant_shift_removal(self):
        """Test that constant shifts in logits are handled correctly."""
        input_ids = [[1, 2]]
        seq_len = 2
        vocab_size = 1500
        batch_size = 1
        
        expected_logits = torch.randn(seq_len, batch_size, vocab_size)
        
        # Add constant shift to all logits
        shift_value = 5.0
        
        def generate_fn(ids):
            actual_logits = expected_logits.clone()
            actual_logits += shift_value
            return actual_logits
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        # Should pass because shift is removed during preprocessing
        assert result is True

    def test_empty_initial_input(self):
        """Test with minimal initial input."""
        input_ids = [[1]]
        seq_len = 1
        vocab_size = 1500
        batch_size = 1
        
        expected_logits = torch.randn(seq_len, batch_size, vocab_size)
        
        def generate_fn(ids):
            return expected_logits.clone()
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is True

    def test_shape_mismatch_assertion(self, basic_setup):
        """Test that shape mismatch raises assertion error."""
        input_ids = basic_setup['input_ids']
        expected_logits = basic_setup['expected_logits']
        seq_len = basic_setup['seq_len']
        batch_size = basic_setup['batch_size']
        
        def generate_fn(ids):
            logits = expected_logits.clone()
            # Return sequences with wrong shape
            wrong_sequences = torch.zeros(batch_size, seq_len + 10)  # Wrong length
            return logits, wrong_sequences
        
        with pytest.raises(AssertionError):
            logit_validation(
                input_ids=input_ids,
                generate_fn=generate_fn,
                expected_logits=expected_logits,
                suppress_passing=True
            )

    def test_default_parameters(self, basic_setup):
        """Test that default parameters are applied correctly."""
        input_ids = basic_setup['input_ids']
        expected_logits = basic_setup['expected_logits']
        
        def generate_fn(ids):
            return expected_logits.clone()
        
        # Call without tol_map and divergence_difference_tol
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
        )
        
        assert result is True

    def test_large_vocabulary(self):
        """Test with large vocabulary size."""
        input_ids = [[1, 2, 3]]
        seq_len = 2
        vocab_size = 50000  # Large vocabulary
        batch_size = 1
        
        torch.manual_seed(42)
        expected_logits = torch.randn(seq_len, batch_size, vocab_size)
        
        def generate_fn(ids):
            return expected_logits.clone()
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            suppress_passing=True
        )
        
        assert result is True


class TestLogitValidationIntegration:
    """Integration tests simulating real-world usage patterns."""

    def test_realistic_generation_scenario(self):
        """Test a realistic text generation scenario."""
        # Simulate a small language model
        vocab_size = 1000
        batch_size = 2
        seq_len = 10
        
        input_ids = [
            [1, 15, 23, 45],
            [2, 18, 31],
        ]
        
        # Simulate CPU reference model logits
        torch.manual_seed(123)
        cpu_logits = torch.randn(seq_len, batch_size, vocab_size) * 10
        
        # Simulate Neuron model with slight numerical differences
        def neuron_generate_fn(ids):
            # Add realistic numerical noise from hardware differences
            noise = torch.randn_like(cpu_logits) * 0.001
            neuron_logits = cpu_logits + noise
            return neuron_logits
        
        result = logit_validation(
            input_ids=input_ids,
            generate_fn=neuron_generate_fn,
            expected_logits=cpu_logits,
            suppress_passing=True
        )
        
        assert result is True
