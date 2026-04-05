import pytest
import torch
from unittest.mock import MagicMock, patch
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import (
    _helper_concat_and_delete_qkv,
    convert_qwen3_moe_hf_to_neuron_state_dict,
)


class TestHelperConcatAndDeleteQKV:
    """Test suite for _helper_concat_and_delete_qkv function"""

    @pytest.fixture
    def sample_state_dict(self):
        """Create a sample state dictionary with QKV tensors"""
        return {
            "layers.0.self_attn.q_proj.weight": torch.randn(128, 64),
            "layers.0.self_attn.k_proj.weight": torch.randn(128, 64),
            "layers.0.self_attn.v_proj.weight": torch.randn(128, 64),
            "layers.0.self_attn.q_proj.scale": torch.randn(128),
            "layers.0.self_attn.k_proj.scale": torch.randn(128),
            "layers.0.self_attn.v_proj.scale": torch.randn(128),
            "layers.1.self_attn.q_proj.weight": torch.randn(256, 128),
            "layers.1.self_attn.k_proj.weight": torch.randn(256, 128),
            "layers.1.self_attn.v_proj.weight": torch.randn(256, 128),
        }

    def test_concatenates_weight_correctly(self, sample_state_dict):
        """Test that weights are concatenated along the correct dimension"""
        layer_num = 0
        attr = "weight"
        
        # Store original tensors for comparison
        q_weight = sample_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"].clone()
        k_weight = sample_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"].clone()
        v_weight = sample_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"].clone()
        
        _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)
        
        # Check concatenated result
        expected = torch.cat([q_weight, k_weight, v_weight])
        result = sample_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"]
        
        assert torch.allclose(result, expected)
        assert result.shape == torch.Size([384, 64])  # 128*3 x 64

    def test_concatenates_scale_correctly(self, sample_state_dict):
        """Test that scales are concatenated correctly"""
        layer_num = 0
        attr = "scale"
        
        # Store original tensors
        q_scale = sample_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"].clone()
        k_scale = sample_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"].clone()
        v_scale = sample_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"].clone()
        
        _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)
        
        # Check concatenated result
        expected = torch.cat([q_scale, k_scale, v_scale])
        result = sample_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"]
        
        assert torch.allclose(result, expected)
        assert result.shape == torch.Size([384])  # 128*3

    def test_deletes_original_keys(self, sample_state_dict):
        """Test that original q_proj, k_proj, v_proj keys are deleted"""
        layer_num = 0
        attr = "weight"
        
        _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)
        
        # Verify deletions
        assert f"layers.{layer_num}.self_attn.q_proj.{attr}" not in sample_state_dict
        assert f"layers.{layer_num}.self_attn.k_proj.{attr}" not in sample_state_dict
        assert f"layers.{layer_num}.self_attn.v_proj.{attr}" not in sample_state_dict

    def test_creates_new_wqkv_key(self, sample_state_dict):
        """Test that new Wqkv key is created"""
        layer_num = 0
        attr = "weight"
        
        assert f"layers.{layer_num}.self_attn.Wqkv.{attr}" not in sample_state_dict
        
        _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)
        
        assert f"layers.{layer_num}.self_attn.Wqkv.{attr}" in sample_state_dict

    def test_modifies_dict_in_place(self, sample_state_dict):
        """Test that the function modifies the dictionary in place"""
        layer_num = 0
        attr = "weight"
        original_dict_id = id(sample_state_dict)
        
        _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)
        
        assert id(sample_state_dict) == original_dict_id

    def test_different_layer_numbers(self, sample_state_dict):
        """Test that function works with different layer numbers"""
        layer_num = 1
        attr = "weight"
        
        _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)
        
        assert f"layers.{layer_num}.self_attn.Wqkv.{attr}" in sample_state_dict
        assert f"layers.{layer_num}.self_attn.q_proj.{attr}" not in sample_state_dict
        assert sample_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"].shape[0] == 768  # 256*3

    def test_does_not_affect_other_layers(self, sample_state_dict):
        """Test that processing one layer doesn't affect other layers"""
        layer_num = 0
        attr = "weight"
        
        # Store layer 1 keys
        layer_1_keys = [k for k in sample_state_dict.keys() if k.startswith("layers.1")]
        
        _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)
        
        # Check layer 1 is unaffected
        for key in layer_1_keys:
            assert key in sample_state_dict

    def test_missing_q_proj_raises_keyerror(self, sample_state_dict):
        """Test that missing q_proj key raises KeyError"""
        layer_num = 0
        attr = "weight"
        
        del sample_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
        
        with pytest.raises(KeyError):
            _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)

    def test_missing_k_proj_raises_keyerror(self, sample_state_dict):
        """Test that missing k_proj key raises KeyError"""
        layer_num = 0
        attr = "weight"
        
        del sample_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
        
        with pytest.raises(KeyError):
            _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)

    def test_missing_v_proj_raises_keyerror(self, sample_state_dict):
        """Test that missing v_proj key raises KeyError"""
        layer_num = 0
        attr = "weight"
        
        del sample_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]
        
        with pytest.raises(KeyError):
            _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)

    def test_concatenation_order(self, sample_state_dict):
        """Test that concatenation maintains Q, K, V order"""
        layer_num = 0
        attr = "weight"
        
        # Create distinctive tensors to verify order
        sample_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"] = torch.ones(2, 2)
        sample_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"] = torch.ones(2, 2) * 2
        sample_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"] = torch.ones(2, 2) * 3
        
        _helper_concat_and_delete_qkv(sample_state_dict, layer_num, attr)
        
        result = sample_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"]
        
        # Check order: first 2 rows should be 1s, next 2 rows 2s, last 2 rows 3s
        assert torch.allclose(result[:2], torch.ones(2, 2))
        assert torch.allclose(result[2:4], torch.ones(2, 2) * 2)
        assert torch.allclose(result[4:6], torch.ones(2, 2) * 3)

    def test_with_empty_tensors(self):
        """Test behavior with empty tensors"""
        state_dict = {
            "layers.0.self_attn.q_proj.weight": torch.empty(0, 64),
            "layers.0.self_attn.k_proj.weight": torch.empty(0, 64),
            "layers.0.self_attn.v_proj.weight": torch.empty(0, 64),
        }
        
        _helper_concat_and_delete_qkv(state_dict, 0, "weight")
        
        assert state_dict["layers.0.self_attn.Wqkv.weight"].shape == torch.Size([0, 64])

    def test_with_1d_tensors(self):
        """Test with 1D tensors (like biases or scales)"""
        state_dict = {
            "layers.0.self_attn.q_proj.bias": torch.randn(64),
            "layers.0.self_attn.k_proj.bias": torch.randn(64),
            "layers.0.self_attn.v_proj.bias": torch.randn(64),
        }
        
        _helper_concat_and_delete_qkv(state_dict, 0, "bias")
        
        assert state_dict["layers.0.self_attn.Wqkv.bias"].shape == torch.Size([192])


class TestConvertQwen3MoeHfToNeuronStateDict:
    """Test suite for convert_qwen3_moe_hf_to_neuron_state_dict function"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object"""
        config = MagicMock()
        config.num_hidden_layers = 2
        config.num_experts = 4
        config.neuron_config.glu_mlp = True
        config.neuron_config.tp_degree = 8
        config.neuron_config.fused_qkv = False
        config.neuron_config.quantized = False
        config.neuron_config.quantized_mlp_kernel_enabled = False
        config.neuron_config.torch_dtype = torch.float32
        config.moe_intermediate_pad_size = 0
        return config

    @pytest.fixture
    def sample_hf_state_dict(self):
        """Create a sample HuggingFace state dictionary"""
        hidden_size = 128
        intermediate_size = 256
        num_experts = 4
        
        state_dict = {}
        
        for layer_idx in range(2):
            # Attention weights
            state_dict[f"layers.{layer_idx}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
            state_dict[f"layers.{layer_idx}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
            state_dict[f"layers.{layer_idx}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
            state_dict[f"layers.{layer_idx}.self_attn.q_norm.weight"] = torch.randn(hidden_size)
            state_dict[f"layers.{layer_idx}.self_attn.k_norm.weight"] = torch.randn(hidden_size)
            
            # Router weights
            state_dict[f"layers.{layer_idx}.mlp.gate.weight"] = torch.randn(num_experts, hidden_size)
            
            # Expert weights
            for expert_idx in range(num_experts):
                state_dict[f"layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = torch.randn(
                    intermediate_size, hidden_size
                )
                state_dict[f"layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = torch.randn(
                    intermediate_size, hidden_size
                )
                state_dict[f"layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = torch.randn(
                    hidden_size, intermediate_size
                )
        
        return state_dict

    def test_adds_rank_util_tensors(self, sample_hf_state_dict, mock_config):
        """Test that rank_util.rank tensors are added correctly"""
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        # Check base model rank tensor
        assert "rank_util.rank" in result
        assert torch.allclose(result["rank_util.rank"], torch.arange(0, mock_config.neuron_config.tp_degree, dtype=torch.int32))
        
        # Check attention rank tensors
        for layer_idx in range(mock_config.num_hidden_layers):
            key = f"layers.{layer_idx}.self_attn.rank_util.rank"
            assert key in result
            assert torch.allclose(result[key], torch.arange(0, mock_config.neuron_config.tp_degree, dtype=torch.int32))

    def test_renames_norm_weights(self, sample_hf_state_dict, mock_config):
        """Test that q_norm and k_norm are renamed to q_layernorm and k_layernorm"""
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        for layer_idx in range(mock_config.num_hidden_layers):
            # Check new keys exist
            assert f"layers.{layer_idx}.self_attn.q_layernorm.weight" in result
            assert f"layers.{layer_idx}.self_attn.k_layernorm.weight" in result
            
            # Check old keys are removed
            assert f"layers.{layer_idx}.self_attn.q_norm.weight" not in result
            assert f"layers.{layer_idx}.self_attn.k_norm.weight" not in result

    def test_copies_router_weights(self, sample_hf_state_dict, mock_config):
        """Test that router weights are copied correctly"""
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        for layer_idx in range(mock_config.num_hidden_layers):
            # Check new router key exists
            assert f"layers.{layer_idx}.mlp.router.linear_router.weight" in result
            
            # Check old gate key is removed
            assert f"layers.{layer_idx}.mlp.gate.weight" not in result

    def test_reorganizes_expert_weights(self, sample_hf_state_dict, mock_config):
        """Test that expert MLP weights are reorganized correctly"""
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        for layer_idx in range(mock_config.num_hidden_layers):
            # Check gate_up_proj exists
            gate_up_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
            assert gate_up_key in result
            
            # Check down_proj exists
            down_proj_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"
            assert down_proj_key in result
            
            # Check original expert keys are removed
            for expert_idx in range(mock_config.num_experts):
                assert f"layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight" not in result
                assert f"layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight" not in result
                assert f"layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight" not in result

    def test_gate_up_proj_shape(self, sample_hf_state_dict, mock_config):
        """Test that gate_up_proj has correct shape"""
        # Get original shapes BEFORE conversion
        hidden_size = sample_hf_state_dict["layers.0.mlp.experts.0.gate_proj.weight"].shape[1]
        intermediate_size = sample_hf_state_dict["layers.0.mlp.experts.0.gate_proj.weight"].shape[0]
        
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        gate_up_proj = result["layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"]
        
        # Shape should be [num_experts, hidden_size, 2 * intermediate_size]
        expected_shape = (mock_config.num_experts, hidden_size, 2 * intermediate_size)
        assert gate_up_proj.shape == expected_shape

    def test_down_proj_shape(self, sample_hf_state_dict, mock_config):
        """Test that down_proj has correct shape"""
        # Get original shapes BEFORE conversion
        hidden_size = sample_hf_state_dict["layers.0.mlp.experts.0.down_proj.weight"].shape[0]
        intermediate_size = sample_hf_state_dict["layers.0.mlp.experts.0.down_proj.weight"].shape[1]
        
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        down_proj = result["layers.0.mlp.expert_mlps.mlp_op.down_proj.weight"]
        
        # Shape should be [num_experts, intermediate_size, hidden_size]
        expected_shape = (mock_config.num_experts, intermediate_size, hidden_size)
        assert down_proj.shape == expected_shape

    def test_gate_up_proj_concatenation(self, sample_hf_state_dict, mock_config):
        """Test that gate_proj and up_proj are concatenated correctly"""
        # Store original weights
        original_gate = sample_hf_state_dict["layers.0.mlp.experts.0.gate_proj.weight"].T.clone()
        original_up = sample_hf_state_dict["layers.0.mlp.experts.0.up_proj.weight"].T.clone()
        
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        gate_up_proj = result["layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"]
        
        # Extract gate and up parts for expert 0
        intermediate_size = original_gate.shape[1]
        expert_0_gate = gate_up_proj[0, :, :intermediate_size]
        expert_0_up = gate_up_proj[0, :, intermediate_size:]
        
        # Verify they match original weights
        assert torch.allclose(expert_0_gate, original_gate)
        assert torch.allclose(expert_0_up, original_up)

    def test_with_padding(self, sample_hf_state_dict, mock_config):
        """Test behavior with intermediate size padding"""
        # Get original shapes BEFORE conversion
        intermediate_size = sample_hf_state_dict["layers.0.mlp.experts.0.gate_proj.weight"].shape[0]
        hidden_size = sample_hf_state_dict["layers.0.mlp.experts.0.gate_proj.weight"].shape[1]
        
        mock_config.moe_intermediate_pad_size = 16
        
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        # Check gate_up_proj has padded size
        gate_up_proj = result["layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"]
        expected_gate_up_shape = (mock_config.num_experts, hidden_size, 2 * (intermediate_size + mock_config.moe_intermediate_pad_size))
        assert gate_up_proj.shape == expected_gate_up_shape
        
        # Check down_proj has padded size
        down_proj = result["layers.0.mlp.expert_mlps.mlp_op.down_proj.weight"]
        expected_down_shape = (mock_config.num_experts, intermediate_size + mock_config.moe_intermediate_pad_size, hidden_size)
        assert down_proj.shape == expected_down_shape

    @patch('neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe.convert_state_dict_to_fused_qkv')
    def test_calls_fused_qkv_when_enabled(self, mock_fused_qkv, sample_hf_state_dict, mock_config):
        """Test that convert_state_dict_to_fused_qkv is called when fused_qkv is enabled"""
        mock_config.neuron_config.fused_qkv = True
        mock_fused_qkv.return_value = sample_hf_state_dict
        
        convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        # Verify the function was called
        mock_fused_qkv.assert_called_once()

    @patch('neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe.convert_state_dict_to_fused_qkv')
    def test_skips_fused_qkv_when_disabled(self, mock_fused_qkv, sample_hf_state_dict, mock_config):
        """Test that convert_state_dict_to_fused_qkv is not called when fused_qkv is disabled"""
        mock_config.neuron_config.fused_qkv = False
        
        convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        # Verify the function was not called
        mock_fused_qkv.assert_not_called()

    def test_modifies_dict_in_place(self, sample_hf_state_dict, mock_config):
        """Test that the function modifies the dictionary in place"""
        original_dict_id = id(sample_hf_state_dict)
        
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        # Result should be the same object
        assert id(result) == original_dict_id

    def test_preserves_tensor_dtypes(self, sample_hf_state_dict, mock_config):
        """Test that tensor dtypes are preserved during conversion"""
        # Set some tensors to specific dtypes
        sample_hf_state_dict["layers.0.mlp.experts.0.gate_proj.weight"] = sample_hf_state_dict[
            "layers.0.mlp.experts.0.gate_proj.weight"
        ].to(torch.float16)
        
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        # Check that the dtype is preserved in the reorganized weights
        gate_up_proj = result["layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"]
        assert gate_up_proj.dtype == torch.float16

    def test_handles_multiple_layers(self, sample_hf_state_dict, mock_config):
        """Test that the function handles all layers correctly"""
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        # Check that all layers are processed
        for layer_idx in range(mock_config.num_hidden_layers):
            assert f"layers.{layer_idx}.self_attn.q_layernorm.weight" in result
            assert f"layers.{layer_idx}.mlp.router.linear_router.weight" in result
            assert f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight" in result
            assert f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight" in result

    def test_glu_mlp_assertion(self, sample_hf_state_dict, mock_config):
        """Test that function raises assertion error when glu_mlp is False"""
        mock_config.neuron_config.glu_mlp = False
        
        with pytest.raises(AssertionError, match="Only GLU MLP is supported"):
            convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)

    def test_expert_weight_transposition(self, sample_hf_state_dict, mock_config):
        """Test that expert weights are transposed correctly for all experts"""
        # Use smaller tensors - ALL experts must have same shape
        intermediate_size = 3
        hidden_size = 4
        
        # Update ALL experts in ALL layers to have same small shape
        for layer_idx in range(mock_config.num_hidden_layers):
            for expert_idx in range(mock_config.num_experts):
                # Each expert gets unique values based on expert_idx for verification
                offset = expert_idx * 100
                sample_hf_state_dict[f"layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = (
                    torch.arange(12).reshape(intermediate_size, hidden_size).float() + offset
                )
                sample_hf_state_dict[f"layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = (
                    torch.arange(12, 24).reshape(intermediate_size, hidden_size).float() + offset
                )
                sample_hf_state_dict[f"layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = (
                    torch.arange(24, 36).reshape(hidden_size, intermediate_size).float() + offset
                )
        
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        # Verify transposition for expert 0 (same logic applies to all experts)
        gate_up_proj = result["layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"]
        expected_gate = torch.arange(12).reshape(intermediate_size, hidden_size).float().T
        actual_gate = gate_up_proj[0, :, :intermediate_size]
        assert torch.allclose(actual_gate, expected_gate)
        
        down_proj = result["layers.0.mlp.expert_mlps.mlp_op.down_proj.weight"]
        expected_down = torch.arange(24, 36).reshape(hidden_size, intermediate_size).float().T
        actual_down = down_proj[0, :, :]
        assert torch.allclose(actual_down, expected_down)
        
        # Also verify expert 1 to show all experts are processed
        expected_gate_expert1 = (torch.arange(12).reshape(intermediate_size, hidden_size).float() + 100).T
        actual_gate_expert1 = gate_up_proj[1, :, :intermediate_size]
        assert torch.allclose(actual_gate_expert1, expected_gate_expert1)

    def test_all_experts_processed(self, sample_hf_state_dict, mock_config):
        """Test that all experts are processed for each layer"""
        # Create unique weights for each expert
        for layer_idx in range(mock_config.num_hidden_layers):
            for expert_idx in range(mock_config.num_experts):
                sample_hf_state_dict[f"layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = (
                    torch.ones(256, 128) * (expert_idx + 1)
                )
        
        result = convert_qwen3_moe_hf_to_neuron_state_dict(sample_hf_state_dict, mock_config)
        
        # Verify each expert's weights are in the combined tensor
        gate_up_proj = result["layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"]
        
        for expert_idx in range(mock_config.num_experts):
            expert_slice = gate_up_proj[expert_idx, :, :256]
            expected_value = expert_idx + 1
            assert torch.allclose(expert_slice, torch.ones_like(expert_slice) * expected_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
