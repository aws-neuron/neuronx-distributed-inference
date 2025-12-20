import torch
from unittest.mock import Mock
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed_inference.models.model_base import NeuronBaseModel

def test_mask_padding_tokens():
    # Create mock instance
    mock_obj = Mock(spec=ExpertMLPsV2)
    mock_obj.sequence_parallel_enabled = False
    mock_obj.mask_padding_tokens = ExpertMLPsV2.mask_padding_tokens.__get__(mock_obj)
    
    # Create mock model base with neuron_config
    mock_model_base = Mock(spec=NeuronBaseModel)
    mock_model_base.neuron_config = Mock()
    mock_model_base.neuron_config.moe_mask_padded_tokens = True
    mock_model_base.create_padding_mask = NeuronBaseModel.create_padding_mask.__get__(mock_model_base)
    
    # Test case 1: padding disabled
    mock_model_base.neuron_config.moe_mask_padded_tokens = False
    expert_mask = torch.ones(4, 3)
    expert_affinities = torch.ones(4, 3) * 0.5
    
    padding_mask = mock_model_base.create_padding_mask(None)
    result_mask, result_affinities = mock_obj.mask_padding_tokens(expert_mask, expert_affinities, padding_mask)
    assert torch.equal(result_mask, expert_mask)
    assert torch.equal(result_affinities, expert_affinities)
    
    # Re-enable padding for remaining tests
    mock_model_base.neuron_config.moe_mask_padded_tokens = True
    
    # Test case 2: with position_ids
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 1, 1, 1, 1, 1]])
    padding_mask = mock_model_base.create_padding_mask(position_ids)
    seq_len = position_ids.size(1)
    expert_mask = torch.ones(seq_len, 5)
    expert_affinities = torch.ones(seq_len, 5) * 0.5
    
    result_mask, result_affinities = mock_obj.mask_padding_tokens(expert_mask, expert_affinities, padding_mask)
    
    # Expected mask: positions 0-6 are valid (max_pos=6), positions 7-11 are masked
    expected_padding_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]).unsqueeze(1)
    expected_affinities = expert_affinities * expected_padding_mask
    expected_expert_mask = expert_mask * expected_padding_mask
    
    assert torch.allclose(result_affinities, expected_affinities)
    assert torch.allclose(result_mask, expected_expert_mask)
    
    # Test case 3: expert_mask is None
    result_mask, result_affinities = mock_obj.mask_padding_tokens(None, expert_affinities, padding_mask)
    assert result_mask is None
    assert torch.allclose(result_affinities, expected_affinities)

    # Test case 4: batch_size = 2
    position_ids = torch.tensor([[0, 1, 2, 3, 1, 1], [0, 1, 2, 1, 1, 1]])  # B=2, S=6
    padding_mask = mock_model_base.create_padding_mask(position_ids)
    B, S = position_ids.shape
    expert_mask = torch.ones(B * S, 4)  # (12, 4)
    expert_affinities = torch.ones(B * S, 4) * 0.7  # (12, 4)
    
    result_mask, result_affinities = mock_obj.mask_padding_tokens(expert_mask, expert_affinities, padding_mask)
    
    # Batch 0: max_pos_idx=3, valid positions 0-3, mask positions 4-5
    # Batch 1: max_pos_idx=2, valid positions 0-2, mask positions 3-5
    expected_padding_mask = torch.tensor([1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0.]).unsqueeze(1)
    expected_affinities = expert_affinities * expected_padding_mask
    expected_expert_mask = expert_mask * expected_padding_mask
    
    assert torch.allclose(result_affinities, expected_affinities)
    assert torch.allclose(result_mask, expected_expert_mask)

def test_mask_padding_tokens_on_device():
    device = "xla"
    
    mock_obj = Mock(spec=ExpertMLPsV2)
    mock_obj.mask_padding_tokens = ExpertMLPsV2.mask_padding_tokens.__get__(mock_obj)
    
    # Create mock model base with neuron_config
    mock_model_base = Mock(spec=NeuronBaseModel)
    mock_model_base.neuron_config = Mock()
    mock_model_base.neuron_config.moe_mask_padded_tokens = True
    mock_model_base.create_padding_mask = NeuronBaseModel.create_padding_mask.__get__(mock_model_base)
    
    # Test case 1: Single batch
    position_ids = torch.tensor([[0, 1, 2, 3, 1, 1, 1, 1]]).to(device)  # B=1, S=8
    padding_mask = mock_model_base.create_padding_mask(position_ids)
    seq_len = position_ids.size(1)
    expert_mask = torch.ones(seq_len, 3).to(device)
    expert_affinities = expert_mask * 0.5
    
    result_mask, result_affinities = mock_obj.mask_padding_tokens(expert_mask, expert_affinities, padding_mask)
    
    # Positions 0-3 are valid (max_pos=3), positions 4-7 are masked
    expected_padding_mask = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.]).unsqueeze(1).to(device)
    expected_affinities = expert_affinities * expected_padding_mask
    expected_expert_mask = expert_mask * expected_padding_mask

    assert torch.allclose(result_affinities, expected_affinities)
    assert torch.allclose(result_mask, expected_expert_mask)

    # Test case 2: Multi-batch
    position_ids = torch.tensor([
        [0, 1, 2, 1],  # batch 0: max_pos=2
        [0, 1, 1, 1]   # batch 1: max_pos=1
    ]).to(device)
    
    padding_mask = mock_model_base.create_padding_mask(position_ids)
    B, S = position_ids.shape
    expert_mask = torch.ones(B * S, 3).to(device)
    expert_affinities = expert_mask * 0.6
    
    result_mask, result_affinities = mock_obj.mask_padding_tokens(expert_mask, expert_affinities, padding_mask)
    
    # Batch 0: positions 0-2 valid, position 3 masked
    # Batch 1: positions 0-1 valid, positions 2-3 masked
    expected_padding_mask = torch.tensor([1., 1., 1., 0., 1., 1., 0., 0.]).unsqueeze(1).to(device)
    expected_expert_mask = expert_mask * expected_padding_mask
    expected_affinities = expert_affinities * expected_padding_mask
    
    assert torch.allclose(result_mask, expected_expert_mask)
    assert torch.allclose(result_affinities, expected_affinities)

if __name__ == "__main__":
    test_mask_padding_tokens()
    test_mask_padding_tokens_on_device()
    print("All tests passed!")