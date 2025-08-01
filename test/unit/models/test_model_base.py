import contextlib
import pytest
import torch

from unittest.mock import Mock, patch

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel


@pytest.fixture
def model_setup():
    """Fixture to create a base model instance with given config."""
    def _setup(batch_size, n_positions, neuron_config_kwargs=None):
        if neuron_config_kwargs is None:
            neuron_config_kwargs = {}
        config = InferenceConfig(
            NeuronConfig(
                batch_size=batch_size,
                n_positions=n_positions,
                **neuron_config_kwargs,
            )
        )
        
        # Patch the __init__ method to avoid the need for full initialization
        with patch.object(NeuronBaseModel, '__init__', return_value=None) as _:
            model = NeuronBaseModel(config)
            
            # Manually set the required attributes
            model.config = config
            model.batch_size = config.neuron_config.batch_size
            model.n_positions = config.neuron_config.n_positions
            model.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
            
            # Create a dummy attention mask
            attention_mask = torch.ones((batch_size, n_positions), dtype=torch.int32)
            
            return model, attention_mask, config
    
    return _setup


def verify_mask_properties(mask, attention_mask, expected_shape):
    """Helper function to verify common mask properties."""
    assert mask.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {mask.shape}"
    assert mask.dtype == torch.bool, f"Expected dtype torch.bool, got {mask.dtype}"


@pytest.mark.parametrize(
    "test_params",
    [
        # Test case 1: Basic case with small dimensions
        {
            "batch_size": 1,
            "n_positions": 8,
            "chunk_size": 4,
            "expected_pattern": [
                # First chunk (positions 0-3)
                [True, False, False, False, False, False, False, False],  # Position 0
                [True, True, False, False, False, False, False, False],   # Position 1
                [True, True, True, False, False, False, False, False],    # Position 2
                [True, True, True, True, False, False, False, False],     # Position 3
                # Second chunk (positions 4-7)
                [False, False, False, False, True, False, False, False],  # Position 4
                [False, False, False, False, True, True, False, False],   # Position 5
                [False, False, False, False, True, True, True, False],    # Position 6
                [False, False, False, False, True, True, True, True],     # Position 7
            ]
        },
        # Test case 2: n_positions not divisible by chunk_size
        {
            "batch_size": 1,
            "n_positions": 5,
            "chunk_size": 3,
            "expected_pattern": [
                # First chunk (positions 0-2)
                [True, False, False, False, False],  # Position 0
                [True, True, False, False, False],   # Position 1
                [True, True, True, False, False],    # Position 2
                # Second chunk (positions 3-4)
                [False, False, False, True, False],  # Position 3
                [False, False, False, True, True],   # Position 4
            ]
        },
        # Test case 3: Chunk size equals sequence length
        {
            "batch_size": 1,
            "n_positions": 4,
            "chunk_size": 4,
            "expected_pattern": [
                [True, False, False, False],  # Position 0
                [True, True, False, False],   # Position 1
                [True, True, True, False],    # Position 2
                [True, True, True, True],     # Position 3
            ]
        },
    ]
)
def test_create_chunked_attn_mask_cte(model_setup, test_params):
    """Test the creation of chunked attention masks for compile-time execution."""
    batch_size = test_params["batch_size"]
    n_positions = test_params["n_positions"]
    chunk_size = test_params["chunk_size"]
    expected_pattern = test_params["expected_pattern"]
    
    model, attention_mask, _ = model_setup(batch_size, n_positions)
    
    # Call the function under test
    mask = model._create_chunked_attn_mask_cte(attention_mask, chunk_size)
    
    # Verify mask properties
    expected_shape = (batch_size, 1, n_positions, n_positions)
    verify_mask_properties(mask, attention_mask, expected_shape)
    
    # Convert expected pattern to tensor for comparison
    expected_mask = torch.tensor(expected_pattern, dtype=torch.bool)
    
    # Verify the mask pattern for each batch
    for batch_idx in range(batch_size):
        assert torch.all(mask[batch_idx, 0] == expected_mask), \
            f"Mask pattern mismatch in batch {batch_idx}.\nExpected:\n{expected_mask}\nGot:\n{mask[batch_idx, 0]}"


@pytest.mark.parametrize(
    "test_params",
    [
        # Test case 1: Basic case - middle of chunk
        {
            "batch_size": 2,
            "n_positions": 16,
            "chunk_size": 4,
            "position_ids": torch.tensor([[6], [6]], dtype=torch.int32),
            "expected_masks": [
                torch.tensor([1,1,0,0], dtype=torch.bool),  # positions 4,5 True
                torch.tensor([1,1,0,0], dtype=torch.bool),  # positions 4,5 True
            ]
        },
        # Test case 2: Chunk boundary case
        {
            "batch_size": 2,
            "n_positions": 16,
            "chunk_size": 4,
            "position_ids": torch.tensor([[7], [7]], dtype=torch.int32),
            "expected_masks": [
                torch.tensor([1,1,1,0], dtype=torch.bool),  # positions 4-7 True
                torch.tensor([1,1,1,0], dtype=torch.bool),  # positions 4-7 True
            ]
        },
        # Test case 3: Multi-batch with different positions
        {
            "batch_size": 2,
            "n_positions": 16,
            "chunk_size": 4,
            "position_ids": torch.tensor([[5], [9]], dtype=torch.int32),
            "expected_masks": [
                torch.tensor([1,0,0,0], dtype=torch.bool),  # positions 4,5 True
                torch.tensor([1,0,0,0], dtype=torch.bool),  # positions 8,9 True
            ]
        },
        # Test case 4: First position in chunk
        {
            "batch_size": 2,
            "n_positions": 16,
            "chunk_size": 4,
            "position_ids": torch.tensor([[4], [4]], dtype=torch.int32),
            "expected_masks": [
                torch.tensor([0,0,0,0], dtype=torch.bool),  # position 4 True
                torch.tensor([0,0,0,0], dtype=torch.bool),  # position 4 True
            ]
        },
    ]
)
def test_create_chunked_attn_mask_tkg(model_setup, test_params):
    """Test the creation of chunked attention masks for token generation."""
    batch_size = test_params["batch_size"]
    n_positions = test_params["n_positions"]
    chunk_size = test_params["chunk_size"]
    position_ids = test_params["position_ids"]
    expected_masks = test_params["expected_masks"]
    
    model, attention_mask, _ = model_setup(batch_size, n_positions)
    
    # Call the function under test
    mask = model._create_chunked_attn_mask_tkg(attention_mask, chunk_size, position_ids)
    
    # Verify mask properties
    expected_shape = (batch_size, 1, 1, chunk_size)
    verify_mask_properties(mask, attention_mask, expected_shape)
    
    # Verify mask values for each batch
    for batch_idx, expected_mask in enumerate(expected_masks):
        assert torch.all(mask[batch_idx, 0, 0] == expected_mask), \
            f"Mask mismatch in batch {batch_idx}. Expected {expected_mask}, got {mask[batch_idx, 0, 0]}"


@pytest.mark.parametrize(
    "tp_group_size, seq_len, expect_error",
    [
        (4, 32, False),
        (4, 30, True),
    ]
)
@patch("neuronx_distributed_inference.models.model_base.get_tp_group")
def test_validate_sequence_parallel(mock_get_tp_group, model_setup, tp_group_size, seq_len, expect_error):
    mock_tp_group = Mock()
    mock_tp_group.size = Mock()
    mock_tp_group.size.return_value = tp_group_size
    mock_get_tp_group.return_value = mock_tp_group

    model, _, _ = model_setup(
        batch_size=1,
        n_positions=seq_len,
        neuron_config_kwargs={
            "sequence_parallel_enabled": True,
        },
    )

    context = pytest.raises(AssertionError) if expect_error else contextlib.nullcontext()
    with context:
        model.validate_sequence_parallel(seq_len)
    
    mock_get_tp_group.assert_called_once_with(model.config)
    mock_tp_group.size.assert_called_once()


@pytest.mark.parametrize(
    "test_params",
    [
        {
            "batch_size": 1,
            "n_positions": 8,
            "window_size": 4,
            "expected_pattern": [
                # First chunk (positions 0-3)
                [True, False, False, False, False, False, False, False],  # Position 0
                [True, True, False, False, False, False, False, False],   # Position 1
                [True, True, True, False, False, False, False, False],    # Position 2
                [True, True, True, True, False, False, False, False],     # Position 3
                # Second chunk (positions 4-7)
                [False, True, True, True, True, False, False, False],  # Position 4
                [False, False, True, True, True, True, False, False],   # Position 5
                [False, False, False, True, True, True, True, False],    # Position 6
                [False, False, False, False, True, True, True, True],     # Position 7
            ]
        },
        {
            "batch_size": 1,
            "n_positions": 8,
            "window_size": 1,   # smallest window size
            "expected_pattern": [
                # First chunk (positions 0-3)
                [True, False, False, False, False, False, False, False],  # Position 0
                [False, True, False, False, False, False, False, False],   # Position 1
                [False, False, True, False, False, False, False, False],    # Position 2
                [False, False, False, True, False, False, False, False],     # Position 3
                # Second chunk (positions 4-7)
                [False, False, False, False, True, False, False, False],  # Position 4
                [False, False, False, False, False, True, False, False],   # Position 5
                [False, False, False, False, False, False, True, False],    # Position 6
                [False, False, False, False, False, False, False, True],     # Position 7
            ]
        },
        {
            "batch_size": 1,
            "n_positions": 8,
            "window_size": 8,   # largest window size
            "expected_pattern": [
                # First chunk (positions 0-3)
                [True, False, False, False, False, False, False, False],  # Position 0
                [True, True, False, False, False, False, False, False],   # Position 1
                [True, True, True, False, False, False, False, False],    # Position 2
                [True, True, True, True, False, False, False, False],     # Position 3
                # Second chunk (positions 4-7)
                [True, True, True, True, True, False, False, False],  # Position 4
                [True, True, True, True, True, True, False, False],   # Position 5
                [True, True, True, True, True, True, True, False],    # Position 6
                [True, True, True, True, True, True, True, True],     # Position 7
            ]
        },
    ]
)
def test_create_windowed_attn_mask_cte(model_setup, test_params):
    """Test the creation of chunked attention masks for compile-time execution."""
    batch_size = test_params["batch_size"]
    n_positions = test_params["n_positions"]
    window_size = test_params["window_size"]
    expected_pattern = test_params["expected_pattern"]

    model, attention_mask, _ = model_setup(batch_size, n_positions)

    # Call the function under test
    mask = model._create_windowed_attn_mask_cte(attention_mask, window_size)

    # Verify mask properties
    expected_shape = (batch_size, 1, n_positions, n_positions)
    verify_mask_properties(mask, attention_mask, expected_shape)

    # Convert expected pattern to tensor for comparison
    expected_mask = torch.tensor(expected_pattern, dtype=torch.bool)

    # Verify the mask pattern for each batch
    for batch_idx in range(batch_size):
        assert torch.all(mask[batch_idx, 0] == expected_mask), \
            f"Mask pattern mismatch in batch {batch_idx}.\nExpected:\n{expected_mask}\nGot:\n{mask[batch_idx, 0]}"


@pytest.mark.parametrize(
    "test_params",
    [
        {  # bs=2, input_len > window_size
            "batch_size": 2,
            "n_positions": 8,
            "window_size": 4,
            "position_ids": torch.tensor([[4],[6]], dtype=torch.int32),
            "expected_masks": [
                torch.tensor([0,1,1,1], dtype=torch.bool),
                torch.tensor([1,1,0,1], dtype=torch.bool),
            ]
        },
        {  # bs=2, input_len <= window_size
            "batch_size": 2,
            "n_positions": 8,
            "window_size": 4,
            "position_ids": torch.tensor([[2],[3]], dtype=torch.int32),
            "expected_masks": [
                torch.tensor([1,1,0,0], dtype=torch.bool),
                torch.tensor([1,1,1,0], dtype=torch.bool),
            ]
        },
    ]
)
def test_create_windowed_attn_mask_tkg(model_setup, test_params):
    """Test the creation of chunked attention masks for token generation."""
    batch_size = test_params["batch_size"]
    n_positions = test_params["n_positions"]
    window_size = test_params["window_size"]
    position_ids = test_params["position_ids"]
    expected_masks = test_params["expected_masks"]

    model, attention_mask, _ = model_setup(batch_size, n_positions)
   
    # Call the function under test
    mask = model._create_windowed_attn_mask_tkg(attention_mask, window_size, position_ids)

    # Verify mask properties
    expected_shape = (batch_size, 1, 1, window_size)
    verify_mask_properties(mask, attention_mask, expected_shape)
   
    # Verify mask values for each batch
    for batch_idx, expected_mask in enumerate(expected_masks):
        assert torch.all(mask[batch_idx, 0, 0] == expected_mask), \
            f"Mask mismatch in batch {batch_idx}. Expected {expected_mask}, got {mask[batch_idx, 0, 0]}"
