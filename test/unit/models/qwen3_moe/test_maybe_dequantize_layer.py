import pytest
import torch
from unittest.mock import Mock

from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import maybe_dequantize_layer

class TestMaybeDequantizeLayer:
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object with BF16 output."""
        config = Mock()
        config.quantization_config = {
            "weight_block_size": [2, 2]
        }
        config.neuron_config = Mock()
        config.neuron_config.torch_dtype = torch.bfloat16
        return config
    
    def test_single_scale_layer_dequantization(self, mock_config):
        """Test dequantization of a single FP8 layer with BF16 scale."""
        neuron_state_dict = {
            "layer1_scale_inv": torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.bfloat16),
            "layer1": torch.tensor([[1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 16]], dtype=torch.float8_e4m3fn)
        }
        #breakpoint()
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        # Check that scale layer is removed
        assert "layer1_scale_inv" not in neuron_state_dict
        
        # Check that fp8 layer is dequantized to BF16
        assert "layer1" in neuron_state_dict
        assert neuron_state_dict["layer1"].dtype == torch.bfloat16
        
        # Verify shape is preserved
        assert neuron_state_dict["layer1"].shape == (4, 4)
    
    def test_multiple_scale_layers(self, mock_config):
        """Test dequantization of multiple FP8 layers."""
        neuron_state_dict = {
            "layer1_scale_inv": torch.tensor([[1.0]], dtype=torch.bfloat16),
            "layer1": torch.ones((2, 2), dtype=torch.float8_e4m3fn),
            "layer2_scale_inv": torch.tensor([[2.0]], dtype=torch.bfloat16),
            "layer2": torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        }
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        # Check all scale layers are removed
        assert "layer1_scale_inv" not in neuron_state_dict
        assert "layer2_scale_inv" not in neuron_state_dict
        
        # Check both layers are present and in BF16
        assert "layer1" in neuron_state_dict
        assert "layer2" in neuron_state_dict
        assert neuron_state_dict["layer1"].dtype == torch.bfloat16
        assert neuron_state_dict["layer2"].dtype == torch.bfloat16
    
    def test_no_scale_layers(self, mock_config):
        """Test when there are no scale layers to process."""
        neuron_state_dict = {
            "layer1": torch.ones((4, 4), dtype=torch.bfloat16),
            "layer2": torch.ones((4, 4), dtype=torch.bfloat16),
        }
        original_dict = neuron_state_dict.copy()
        
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        # Dictionary should remain unchanged
        assert set(neuron_state_dict.keys()) == set(original_dict.keys())
    
    def test_scale_expansion_correctness(self, mock_config):
        """Test that BF16 scales are correctly expanded and applied to FP8 weights."""
        # Simple case: 1x1 scale expanded to 2x2
        neuron_state_dict = {
            "layer_scale_inv": torch.tensor([[5.0]], dtype=torch.bfloat16),
            "layer": torch.tensor([[1.0, 2.0],
                                   [3.0, 4.0]], dtype=torch.float8_e4m3fn)
        }
        
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        # Expected result after scaling
        expected = torch.tensor([[5.0, 10.0],
                                [15.0, 20.0]], dtype=torch.bfloat16)
        
        assert neuron_state_dict["layer"].dtype == torch.bfloat16
        torch.testing.assert_close(
            neuron_state_dict["layer"], 
            expected,
            rtol=1e-2,  # BF16 has lower precision
            atol=1e-2
        )
    
    def test_different_block_sizes(self):
        """Test with different block sizes."""
        config = Mock()
        config.quantization_config = {
            "weight_block_size": [1, 4]  # Different block size
        }
        config.neuron_config = Mock()
        config.neuron_config.torch_dtype = torch.bfloat16
        
        neuron_state_dict = {
            "layer_scale_inv": torch.tensor([[2.0]], dtype=torch.bfloat16),
            "layer": torch.ones((1, 4), dtype=torch.float8_e4m3fn)
        }
        
        maybe_dequantize_layer(neuron_state_dict, config)
        
        # All values should be multiplied by 2.0 and converted to BF16
        expected = torch.ones((1, 4), dtype=torch.bfloat16) * 2.0
        assert neuron_state_dict["layer"].dtype == torch.bfloat16
        torch.testing.assert_close(neuron_state_dict["layer"], expected, rtol=1e-2, atol=1e-2)
    
    def test_preserves_non_scale_layers(self, mock_config):
        """Test that layers without scales are not modified."""
        original_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
        neuron_state_dict = {
            "layer_scale_inv": torch.tensor([[1.0]], dtype=torch.bfloat16),
            "layer": torch.ones((2, 2), dtype=torch.float8_e4m3fn),
            "other_layer": original_tensor.clone(),
        }
        
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        # other_layer should remain unchanged
        torch.testing.assert_close(neuron_state_dict["other_layer"], original_tensor)
        assert neuron_state_dict["other_layer"].dtype == torch.bfloat16
    
    def test_large_scale_tensor(self, mock_config):
        """Test with multiple scale values in BF16."""
        # 2x2 scale tensor with block_size [2, 2] -> 4x4 output
        neuron_state_dict = {
            "layer_scale_inv": torch.tensor([[1.0, 2.0],
                                            [3.0, 4.0]], dtype=torch.bfloat16),
            "layer": torch.ones((4, 4), dtype=torch.float8_e4m3fn)
        }
        
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        # Verify the expansion pattern
        # [[1,1,2,2],
        #  [1,1,2,2],
        #  [3,3,4,4],
        #  [3,3,4,4]]
        expected = torch.tensor([[1.0, 1.0, 2.0, 2.0],
                                [1.0, 1.0, 2.0, 2.0],
                                [3.0, 3.0, 4.0, 4.0],
                                [3.0, 3.0, 4.0, 4.0]], dtype=torch.bfloat16)
        
        assert neuron_state_dict["layer"].dtype == torch.bfloat16
        torch.testing.assert_close(
            neuron_state_dict["layer"], 
            expected,
            rtol=1e-2,
            atol=1e-2
        )
    
    def test_fp8_to_bf16_conversion(self):
        """Test proper FP8 to BF16 dtype conversion through scaling."""
        config = Mock()
        config.quantization_config = {"weight_block_size": [1, 1]}
        config.neuron_config = Mock()
        config.neuron_config.torch_dtype = torch.bfloat16
        neuron_state_dict = {
            "layer_scale_inv": torch.tensor([[2.0]], dtype=torch.bfloat16),
            "layer": torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
        }
        
        # Verify input types
        assert neuron_state_dict["layer_scale_inv"].dtype == torch.bfloat16
        assert neuron_state_dict["layer"].dtype == torch.float8_e4m3fn
        
        maybe_dequantize_layer(neuron_state_dict, config)
        
        # Verify output is BF16
        assert neuron_state_dict["layer"].dtype == torch.bfloat16
        expected = torch.tensor([[2.0]], dtype=torch.bfloat16)
        torch.testing.assert_close(neuron_state_dict["layer"], expected, rtol=1e-2, atol=1e-2)
    
    def test_empty_state_dict(self, mock_config):
        """Test with empty state dict."""
        neuron_state_dict = {}
        
        # Should not raise an error
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        assert len(neuron_state_dict) == 0
    
    def test_scale_layer_name_with_underscores(self, mock_config):
        """Test layer names that contain underscores."""
        neuron_state_dict = {
            "my_layer_name_scale_inv": torch.tensor([[2.0]], dtype=torch.bfloat16),
            "my_layer_name": torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        }
        
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        assert "my_layer_name_scale_inv" not in neuron_state_dict
        assert "my_layer_name" in neuron_state_dict
        assert neuron_state_dict["my_layer_name"].dtype == torch.bfloat16
    
    def test_realistic_quantization_scenario(self, mock_config):
        """Test realistic quantization/dequantization scenario."""
        # Simulate a weight matrix quantized to FP8 with per-block scaling
        # After quantization: weight is in FP8, scales are in BF16
        # For this test, we'll use simple scale values
        scale_values = torch.tensor([[10.0, 20.0],
                                    [30.0, 40.0]], dtype=torch.bfloat16)
        
        # Simulated FP8 values (would be quantized in real scenario)
        fp8_weight = torch.tensor([[1.0, 1.0, 1.5, 2.0],
                                  [1.67, 2.0, 2.33, 2.0],
                                  [3.0, 3.33, 3.67, 3.0],
                                  [3.25, 3.5, 3.75, 4.0]], dtype=torch.float8_e4m3fn)
        
        neuron_state_dict = {
            "weight_scale_inv": scale_values,
            "weight": fp8_weight
        }
        
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        # Verify scale is removed
        assert "weight_scale_inv" not in neuron_state_dict
        
        # Verify weight is dequantized and in BF16
        assert "weight" in neuron_state_dict
        assert neuron_state_dict["weight"].dtype == torch.bfloat16
        assert neuron_state_dict["weight"].shape == (4, 4)
    
    def test_numerical_precision_fp8_bf16(self, mock_config):
        """Test numerical precision during FP8 to BF16 conversion."""
        # Test with values that are exactly representable in FP8
        neuron_state_dict = {
            "layer_scale_inv": torch.tensor([[0.5]], dtype=torch.bfloat16),
            "layer": torch.tensor([[2.0, 4.0],
                                   [8.0, 16.0]], dtype=torch.float8_e4m3fn)
        }
        
        maybe_dequantize_layer(neuron_state_dict, mock_config)
        
        # Result should be scaled values in BF16
        expected = torch.tensor([[1.0, 2.0],
                                [4.0, 8.0]], dtype=torch.bfloat16)
        
        assert neuron_state_dict["layer"].dtype == torch.bfloat16
        torch.testing.assert_close(
            neuron_state_dict["layer"],
            expected,
            rtol=1e-2,
            atol=1e-2
        )
 
 
class TestMaybeDequantizeLayerEdgeCases:
    
    def test_scale_inv_in_middle_of_name(self):
        """Test when _scale_inv appears in the middle of layer name."""
        config = Mock()
        config.quantization_config = {"weight_block_size": [1, 1]}
        config.neuron_config = Mock()
        config.neuron_config.torch_dtype = torch.bfloat16
        
        neuron_state_dict = {
            "layer_scale_inv_extra": torch.tensor([[2.0]], dtype=torch.bfloat16),
            "layer": torch.ones((1, 1), dtype=torch.float8_e4m3fn)
        }
        
        maybe_dequantize_layer(neuron_state_dict, config)
        
        # Should process the scale layer
        assert "layer_scale_inv_extra" in neuron_state_dict
    
    def test_mutation_of_original_dict(self):
        """Test that the function mutates the original dictionary."""
        config = Mock()
        config.quantization_config = {"weight_block_size": [1, 1]}
        config.neuron_config = Mock()
        config.neuron_config.torch_dtype = torch.bfloat16
        
        neuron_state_dict = {
            "layer_scale_inv": torch.tensor([[2.0]], dtype=torch.bfloat16),
            "layer": torch.ones((1, 1), dtype=torch.float8_e4m3fn)
        }
        
        original_id = id(neuron_state_dict)
        maybe_dequantize_layer(neuron_state_dict, config)
        
        # Should mutate in place
        assert id(neuron_state_dict) == original_id
    
    def test_mixed_precision_layers(self):
        """Test state dict with both quantized and non-quantized layers."""
        config = Mock()
        config.quantization_config = {"weight_block_size": [2, 2]}
        config.neuron_config = Mock()
        config.neuron_config.torch_dtype = torch.bfloat16
        
        neuron_state_dict = {
            # Quantized layer
            "q_layer_scale_inv": torch.tensor([[2.0]], dtype=torch.bfloat16),
            "q_layer": torch.ones((2, 2), dtype=torch.float8_e4m3fn),
            # Non-quantized layer (already in BF16)
            "normal_layer": torch.ones((2, 2), dtype=torch.bfloat16) * 3.0,
        }
        
        maybe_dequantize_layer(neuron_state_dict, config)
        
        # Quantized layer should be dequantized
        assert "q_layer_scale_inv" not in neuron_state_dict
        assert neuron_state_dict["q_layer"].dtype == torch.bfloat16
        
        # Normal layer should be unchanged
        assert neuron_state_dict["normal_layer"].dtype == torch.bfloat16
        torch.testing.assert_close(
            neuron_state_dict["normal_layer"],
            torch.ones((2, 2), dtype=torch.bfloat16) * 3.0,
            rtol=1e-2,
            atol=1e-2
        )
    
    def test_large_weight_matrix(self):
        """Test with larger, more realistic weight matrices."""
        config = Mock()
        config.quantization_config = {"weight_block_size": [4, 4]}
        config.neuron_config = Mock()
        config.neuron_config.torch_dtype = torch.bfloat16
        
        # 4x4 scale for 16x16 weight matrix
        scale = torch.randn((4, 4), dtype=torch.bfloat16)
        weight = torch.ones((16, 16), dtype=torch.float8_e4m3fn)
        
        neuron_state_dict = {
            "large_layer_scale_inv": scale,
            "large_layer": weight
        }
        
        maybe_dequantize_layer(neuron_state_dict, config)
        
        assert "large_layer_scale_inv" not in neuron_state_dict
        assert neuron_state_dict["large_layer"].dtype == torch.bfloat16
        assert neuron_state_dict["large_layer"].shape == (16, 16)
 
 
