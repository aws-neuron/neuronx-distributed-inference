import unittest
import torch
from neuronx_distributed_inference.experimental.core.pad import pad_to_shape, pad_at_end


class TestPaddingUtilities(unittest.TestCase):
    """Unit tests for padding utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)  # For reproducibility
        
    # ==================== Tests for pad_at_end ====================
    
    def test_pad_at_end_no_padding_needed(self):
        """Test that tensor is returned unchanged when no padding is needed."""
        tensor = torch.randn(3, 4, 5)
        result = pad_at_end(tensor, dim=1, padded_len=4)
        self.assertTrue(torch.equal(tensor, result))
        self.assertIs(tensor, result)  # Should be the same object
        
    def test_pad_at_end_1d_tensor(self):
        """Test padding a 1D tensor."""
        tensor = torch.tensor([1, 2, 3])
        result = pad_at_end(tensor, dim=0, padded_len=5, mode="constant", value=0)
        expected = torch.tensor([1, 2, 3, 0, 0])
        self.assertTrue(torch.equal(result, expected))
        
    def test_pad_at_end_2d_tensor_dim0(self):
        """Test padding a 2D tensor along dimension 0."""
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = pad_at_end(tensor, dim=0, padded_len=4, mode="constant", value=0)
        expected = torch.tensor([[1, 2], [3, 4], [0, 0], [0, 0]])
        self.assertTrue(torch.equal(result, expected))
        
    def test_pad_at_end_2d_tensor_dim1(self):
        """Test padding a 2D tensor along dimension 1."""
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = pad_at_end(tensor, dim=1, padded_len=5, mode="constant", value=0)
        expected = torch.tensor([[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]])
        self.assertTrue(torch.equal(result, expected))
        
    def test_pad_at_end_3d_tensor(self):
        """Test padding a 3D tensor."""
        tensor = torch.ones(2, 3, 4)
        result = pad_at_end(tensor, dim=2, padded_len=6, mode="constant", value=0)
        self.assertEqual(result.shape, (2, 3, 6))
        self.assertTrue(torch.all(result[:, :, :4] == 1))
        self.assertTrue(torch.all(result[:, :, 4:] == 0))
        
    def test_pad_at_end_custom_value(self):
        """Test padding with a custom value."""
        tensor = torch.tensor([1, 2, 3])
        result = pad_at_end(tensor, dim=0, padded_len=5, mode="constant", value=-1)
        expected = torch.tensor([1, 2, 3, -1, -1])
        self.assertTrue(torch.equal(result, expected))
        
    def test_pad_at_end_invalid_dim(self):
        """Test that invalid dimension raises assertion error."""
        tensor = torch.randn(3, 4)
        with self.assertRaises(AssertionError):
            pad_at_end(tensor, dim=-1, padded_len=5)  # Negative dim
        with self.assertRaises(AssertionError):
            pad_at_end(tensor, dim=2, padded_len=5)  # Dim >= ndim
            
    def test_pad_at_end_invalid_padded_len(self):
        """Test that invalid padded_len raises assertion error."""
        tensor = torch.randn(5)
        with self.assertRaises(AssertionError):
            pad_at_end(tensor, dim=0, padded_len=3)  # padded_len < current size
            
    def test_pad_at_end_non_tensor_input(self):
        """Test that non-tensor input raises assertion error."""
        with self.assertRaises(AssertionError):
            pad_at_end([1, 2, 3], dim=0, padded_len=5)
            
    # ==================== Tests for pad_to_shape ====================
    
    def test_pad_to_shape_no_padding_needed(self):
        """Test that tensor is returned unchanged when shape matches."""
        tensor = torch.randn(3, 4, 5)
        result = pad_to_shape(tensor, torch.Size([3, 4, 5]))
        self.assertTrue(torch.equal(tensor, result))
        self.assertIs(tensor, result)
        
    def test_pad_to_shape_1d_tensor(self):
        """Test padding a 1D tensor to a larger shape."""
        tensor = torch.tensor([1, 2, 3])
        result = pad_to_shape(tensor, torch.Size([5]), mode="constant", value=0)
        expected = torch.tensor([1, 2, 3, 0, 0])
        self.assertTrue(torch.equal(result, expected))
        
    def test_pad_to_shape_2d_tensor(self):
        """Test padding a 2D tensor to a larger shape."""
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = pad_to_shape(tensor, torch.Size([3, 4]), mode="constant", value=0)
        expected = torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0]])
        self.assertTrue(torch.equal(result, expected))
        
    def test_pad_to_shape_3d_tensor(self):
        """Test padding a 3D tensor to a larger shape."""
        tensor = torch.ones(2, 3, 4)
        result = pad_to_shape(tensor, torch.Size([3, 5, 6]), mode="constant", value=0)
        self.assertEqual(result.shape, (3, 5, 6))
        # Check original values are preserved
        self.assertTrue(torch.all(result[:2, :3, :4] == 1))
        # Check padded areas are zero
        self.assertTrue(torch.all(result[2:, :, :] == 0))
        self.assertTrue(torch.all(result[:, 3:, :] == 0))
        self.assertTrue(torch.all(result[:, :, 4:] == 0))
        
    def test_pad_to_shape_custom_value(self):
        """Test padding with a custom value."""
        tensor = torch.tensor([[1, 2]])
        result = pad_to_shape(tensor, torch.Size([2, 3]), mode="constant", value=-1)
        expected = torch.tensor([[1, 2, -1], [-1, -1, -1]])
        self.assertTrue(torch.equal(result, expected))
        
    def test_pad_to_shape_partial_padding(self):
        """Test padding where only some dimensions need padding."""
        tensor = torch.ones(3, 4, 2)
        result = pad_to_shape(tensor, torch.Size([3, 5, 6]), mode="constant", value=0)
        self.assertEqual(result.shape, (3, 5, 6))
        # First dimension shouldn't change
        self.assertEqual(result.shape[0], 3)
        # Check original values
        self.assertTrue(torch.all(result[:, :4, :2] == 1))
        
    def test_pad_to_shape_float_tensor(self):
        """Test padding with float tensors."""
        tensor = torch.randn(2, 3)
        original_values = tensor.clone()
        result = pad_to_shape(tensor, torch.Size([4, 5]), mode="constant", value=0.0)
        self.assertEqual(result.shape, (4, 5))
        # Check original values are preserved
        self.assertTrue(torch.allclose(result[:2, :3], original_values))
        
    def test_pad_to_shape_different_dtypes(self):
        """Test padding with different tensor dtypes."""
        # Test with int tensor
        int_tensor = torch.tensor([[1, 2]], dtype=torch.int32)
        int_result = pad_to_shape(int_tensor, torch.Size([2, 3]), value=0)
        self.assertEqual(int_result.dtype, torch.int32)
        
        # Test with float tensor
        float_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        float_result = pad_to_shape(float_tensor, torch.Size([2, 3]), value=0.0)
        self.assertEqual(float_result.dtype, torch.float64)
        
        # Test with bool tensor
        bool_tensor = torch.tensor([[True, False]], dtype=torch.bool)
        bool_result = pad_to_shape(bool_tensor, torch.Size([2, 3]), value=False)
        self.assertEqual(bool_result.dtype, torch.bool)
        
    def test_pad_to_shape_empty_tensor(self):
        """Test padding an empty tensor."""
        tensor = torch.empty(0, 3)
        result = pad_to_shape(tensor, torch.Size([2, 4]), mode="constant", value=0)
        self.assertEqual(result.shape, (2, 4))
        self.assertTrue(torch.all(result == 0))
        
    def test_pad_to_shape_large_tensor(self):
        """Test padding with larger tensors to ensure performance."""
        tensor = torch.randn(100, 200, 50)
        result = pad_to_shape(tensor, torch.Size([150, 250, 100]), mode="constant", value=0)
        self.assertEqual(result.shape, (150, 250, 100))
        # Check that original values are preserved
        self.assertTrue(torch.allclose(result[:100, :200, :50], tensor))
        
    # ==================== Integration Tests ====================
    
    def test_integration_multiple_calls(self):
        """Test multiple padding operations in sequence."""
        tensor = torch.tensor([1, 2])
        
        # First padding
        result1 = pad_at_end(tensor, dim=0, padded_len=4, value=0)
        self.assertEqual(result1.shape[0], 4)
        
        # Second padding on the result
        result2 = pad_at_end(result1, dim=0, padded_len=6, value=-1)
        self.assertEqual(result2.shape[0], 6)
        expected = torch.tensor([1, 2, 0, 0, -1, -1])
        self.assertTrue(torch.equal(result2, expected))
        
    def test_integration_pad_to_shape_uses_pad_at_end(self):
        """Test that pad_to_shape correctly uses pad_at_end internally."""
        tensor = torch.ones(2, 3)
        result = pad_to_shape(tensor, torch.Size([4, 5]), mode="constant", value=0)
        
        # Manually pad using pad_at_end
        manual_result = pad_at_end(tensor, dim=0, padded_len=4, mode="constant", value=0)
        manual_result = pad_at_end(manual_result, dim=1, padded_len=5, mode="constant", value=0)
        
        self.assertTrue(torch.equal(result, manual_result))
