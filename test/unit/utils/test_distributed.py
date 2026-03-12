import os
import torch
import torch_xla
import pytest
from unittest.mock import patch, MagicMock
from torch_neuronx.utils import get_platform_target

from neuronx_distributed_inference.utils.distributed import (
    get_init_world_size,
    get_init_rank,
    get_tp_group,
    get_dp_rank_spmd,
    get_cp_rank,
    get_dp_rank,
    split_along_dim,
    split_along_dim0_kernel_wrapper,
    get_rank_8_by_8,
    get_kv_head_group_number,
)
from neuronxcc.nki.language import nc

# Tests for environment variable functions
@pytest.mark.parametrize(
    "env_vars,expected_result", 
    [
        ({"WORLD_SIZE": "4", "OMPI_COMM_WORLD_SIZE": "8"}, 4),  # WORLD_SIZE takes precedence
        ({"OMPI_COMM_WORLD_SIZE": "8"}, 8),  # Only OMPI var
        ({"WORLD_SIZE": "4"}, 4),  # Only WORLD_SIZE var
        ({}, -1),  # No env vars
    ]
)
def test_get_init_world_size(env_vars, expected_result):
    with patch.dict(os.environ, env_vars, clear=True):
        assert get_init_world_size() == expected_result

@pytest.mark.parametrize(
    "env_vars,expected_result", 
    [
        ({"RANK": "2", "OMPI_COMM_WORLD_RANK": "3"}, 2),  # RANK takes precedence
        ({"OMPI_COMM_WORLD_RANK": "3"}, 3),  # Only OMPI var
        ({"RANK": "2"}, 2),  # Only RANK var
        ({}, -1),  # No env vars
    ]
)
def test_get_init_rank(env_vars, expected_result):
    with patch.dict(os.environ, env_vars, clear=True):
        assert get_init_rank() == expected_result

# Tests for get_tp_group
@patch("neuronx_distributed.parallel_layers.parallel_state.get_speculative_draft_group")
@patch("neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_group")
def test_get_tp_group_with_draft(mock_get_tensor_mp_group, mock_get_draft_group):
    # Setup mocks
    mock_config = MagicMock()
    mock_config.neuron_config.use_draft_group = True
    mock_draft_group = MagicMock()
    mock_get_draft_group.return_value = mock_draft_group
    
    # Call function
    result = get_tp_group(mock_config)
    
    # Assertions
    mock_get_draft_group.assert_called_once_with(as_list=False)
    mock_get_tensor_mp_group.assert_not_called()
    assert result == mock_draft_group

@patch("neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_group")
def test_get_tp_group_without_draft(mock_get_tensor_mp_group):
    # Setup mocks
    mock_config = MagicMock()
    mock_config.neuron_config.use_draft_group = False
    mock_tp_group = MagicMock()
    mock_get_tensor_mp_group.return_value = mock_tp_group
    
    # Call function
    result = get_tp_group(mock_config)
    
    # Assertions
    mock_get_tensor_mp_group.assert_called_once_with(as_list=False)
    assert result == mock_tp_group

def test_get_tp_group_no_attribute():
    # Setup mock without the attribute
    mock_config = MagicMock()
    mock_config.neuron_config = MagicMock(spec=[])  # No 'use_draft_group' attribute
    
    # Call function
    result = get_tp_group(mock_config)
    
    # Assertions
    assert result is None

# Tests for rank calculation functions
@pytest.mark.parametrize(
    "global_rank,tp_degree,expected_result", 
    [
        (torch.tensor(0), 2, torch.tensor(0)),
        (torch.tensor(1), 2, torch.tensor(0)),
        (torch.tensor(2), 2, torch.tensor(1)),
        (torch.tensor(5), 2, torch.tensor(2)),
        (torch.tensor(6), 4, torch.tensor(1)),
    ]
)
def test_get_dp_rank_spmd(global_rank, tp_degree, expected_result):
    result = get_dp_rank_spmd(global_rank, tp_degree)
    assert result.item() == expected_result.item()
    assert result.dtype == torch.int32

@pytest.mark.parametrize(
    "global_rank,tp_degree,expected_result", 
    [
        (torch.tensor(0), 2, torch.tensor(0)),
        (torch.tensor(1), 2, torch.tensor(0)),
        (torch.tensor(2), 2, torch.tensor(1)),
        (torch.tensor(5), 2, torch.tensor(2)),
        (torch.tensor(6), 4, torch.tensor(1)),
    ]
)
def test_get_cp_rank(global_rank, tp_degree, expected_result):
    result = get_cp_rank(global_rank, tp_degree)
    assert result.item() == expected_result.item()
    assert result.dtype == torch.int32


@pytest.mark.parametrize(
    "global_rank,tp_degree,expected_result", 
    [
        (torch.tensor(0), 3, torch.tensor(0)),
        (torch.tensor(1), 3, torch.tensor(1)),
        (torch.tensor(2), 3, torch.tensor(2)),
        (torch.tensor(3), 3, torch.tensor(0)),
        (torch.tensor(4), 3, torch.tensor(1)),
    ]
)
def test_get_kv_head_group_number(global_rank, tp_degree, expected_result):
    result = get_kv_head_group_number(global_rank, tp_degree)
    assert result.item() == expected_result.item()
    assert result.dtype == torch.int32

@pytest.mark.parametrize(
    "global_rank,tp_degree,dp_degree,expected_result", 
    [
        (torch.tensor(0), 2, None, torch.tensor(0)),
        (torch.tensor(1), 2, None, torch.tensor(0)),
        (torch.tensor(2), 2, None, torch.tensor(1)),
        (torch.tensor(5), 2, None, torch.tensor(2)),
        (torch.tensor(6), 4, None, torch.tensor(1)),
        (torch.tensor(0), 8, 8, torch.tensor(0)),
        (torch.tensor(13), 8, 8, torch.tensor(0)),
        (torch.tensor(32), 8, 8, torch.tensor(4)),
        (torch.tensor(43), 8, 8, torch.tensor(5)),
        (torch.tensor(63), 8, 8, torch.tensor(6)),
    ]
)
def test_get_dp_rank(global_rank, tp_degree, dp_degree, expected_result):
    result = get_dp_rank(global_rank, tp_degree, dp_degree)
    assert result.item() == expected_result.item()
    assert result.dtype == torch.int32

def test_split_along_dim_1d():
    tensor = torch.arange(10)
    
    # Split into 2 parts
    result0 = split_along_dim(tensor, 0, 0, 2)
    result1 = split_along_dim(tensor, 0, 1, 2)
    
    assert torch.equal(result0, torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(result1, torch.tensor([5, 6, 7, 8, 9]))

def test_split_along_dim_2d_along_dim0():
    tensor = torch.arange(24).reshape(6, 4)
    
    # Split along dimension 0 into 2 parts
    result0 = split_along_dim(tensor, 0, 0, 2)
    result1 = split_along_dim(tensor, 0, 1, 2)
    
    assert result0.shape == (3, 4)
    assert result1.shape == (3, 4)
    assert torch.equal(result0, tensor[:3])
    assert torch.equal(result1, tensor[3:])

def test_split_along_dim_2d_along_dim1():
    tensor = torch.arange(24).reshape(4, 6)
    
    # Split along dimension 1 into 2 parts
    result0 = split_along_dim(tensor, 1, 0, 2)  # First 3 columns
    result1 = split_along_dim(tensor, 1, 1, 2)  # Last 3 columns
    
    assert result0.shape == (4, 3)
    assert result1.shape == (4, 3)
    assert torch.equal(result0, tensor[:, :3])
    assert torch.equal(result1, tensor[:, 3:])

def test_split_along_dim_3d():
    tensor = torch.arange(48).reshape(2, 6, 4)
    
    # Split along dimension 1 into 3 parts
    result0 = split_along_dim(tensor, 1, 0, 3)  # First 2 in dim 1
    result1 = split_along_dim(tensor, 1, 1, 3)  # Middle 2 in dim 1
    result2 = split_along_dim(tensor, 1, 2, 3)  # Last 2 in dim 1
    
    assert result0.shape == (2, 2, 4)
    assert result1.shape == (2, 2, 4)
    assert result2.shape == (2, 2, 4)
    assert torch.equal(result0, tensor[:, :2])
    assert torch.equal(result1, tensor[:, 2:4])
    assert torch.equal(result2, tensor[:, 4:])

def test_split_along_dim_none_tensor():
    result = split_along_dim(None, 0, 0, 2)
    assert result is None

@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("on_cpu", [True, False])
def test_split_along_dim0_kernel(lnc, on_cpu):
    if get_platform_target() == "trn1" and lnc == 2:
        pytest.skip("LNC-2 is not supported in trn1")

    def _split_along_dim0_kernel_test_wrap(tensor, rank, num_partitions):
        """Kernel wrapper for boilerplate to(device) and rank int to tensor code"""
        device = torch.device('cpu') if on_cpu else torch_xla.device()
        if tensor.dtype == torch.int64:  # Neuron doesn't like int64
            tensor = tensor.to(torch.int32)

        result = split_along_dim0_kernel_wrapper(
            tensor=tensor.to(device),
            rank=torch.tensor([rank], dtype=torch.int32).to(device),
            num_partitions=num_partitions,
            lnc=lnc,
        )
        expected = split_along_dim(tensor, 0, rank, num_partitions)
        assert torch.equal(result, expected)

    # Test 1D tensor.  Split into 3 parts.
    tensor = torch.arange(6)
    _split_along_dim0_kernel_test_wrap(tensor, 0, 3)
    _split_along_dim0_kernel_test_wrap(tensor, 1, 3)
    _split_along_dim0_kernel_test_wrap(tensor, 2, 3)

    # Test 2D tensor.
    tensor = torch.arange(24).reshape(6, 4)
    _split_along_dim0_kernel_test_wrap(tensor, 0, 2)
    _split_along_dim0_kernel_test_wrap(tensor, 1, 2)

    # Test 3D tensor.
    tensor = torch.arange(84).reshape(4, 3, 7)
    _split_along_dim0_kernel_test_wrap(tensor, 0, 2)
    _split_along_dim0_kernel_test_wrap(tensor, 1, 2)

    # Test with more realistic hidden_state, and a LNC-indivisible H
    for H in [1024, 4567]:
        tensor = torch.rand(32, 5, H, dtype=torch.bfloat16)
        # Test rank 1 & 7, assuming 8 total ranks.
        _split_along_dim0_kernel_test_wrap(tensor, 1, 8)
        _split_along_dim0_kernel_test_wrap(tensor, 7, 8)


def test_get_rank_8_by_8():
    tp = 8
    cp_8_by_8_mesh = [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 28, 29, 30, 31, 20, 21, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 44, 45, 46, 47, 36, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50, 51, 60, 61, 62, 63, 52, 53, 54, 55, 56, 57, 58, 59]
    
    expected_result = sum([[i] * tp for i in range(tp)], [])
    actual_result = []

    for rank in cp_8_by_8_mesh:
        result = get_rank_8_by_8(torch.tensor(rank))
        actual_result.append(result.item())
    assert actual_result == expected_result

def test_get_rank_8_by_8_with_switch():
    tp = 8
    pds_mesh = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30, 31, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 48, 49, 54, 55, 52, 53, 58, 59, 56, 57, 62, 63, 60, 61]
    
    expected_result = sum([[i] * tp for i in range(tp)], [])
    actual_result = []

    for rank in pds_mesh:
        result = get_rank_8_by_8(torch.tensor(rank), switch_cc=True)
        actual_result.append(result.item())
    assert actual_result == expected_result
