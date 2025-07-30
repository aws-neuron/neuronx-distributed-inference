import pytest
import torch
import logging
from neuronx_distributed_inference.modules.eagle.dynamic_token_tree import DynamicTokenTree

@pytest.fixture(autouse=True)
def setup_logging(caplog):
    caplog.set_level(logging.WARNING)

@pytest.fixture
def sample_inputs():
    bs = 2
    max_length = 20
    return {
        'step': 0,
        'input_position_ids': torch.tensor([[0],[0]]),
        'prev_attn_mask': torch.zeros(bs, 1, max_length, max_length),
        'branching_factor': 4,
        'num_inputs': 2,
        'position_ids_offset' : torch.tensor([[0],[0]])
    }

def test_update_attention_mask_step1(sample_inputs):
    result = DynamicTokenTree.update_attention_mask(**sample_inputs)
    logging.warning(f"Step 1 result:\n{result}")
    
    assert result.shape == sample_inputs['prev_attn_mask'].shape
    assert torch.all(result[:, 0, 0, 0] == 1)  # Check if the first element is 1 (eye matrix)
    assert torch.all(result[:, 0, 1:5, 0] == 1)  # Check if the mask is copied correctly for 4 children

def test_update_attention_mask_step2(sample_inputs):
    sample_inputs['step'] = 1
    sample_inputs['input_position_ids'] = torch.tensor([[1, 2], [1, 2]])
    logging.warning(sample_inputs["prev_attn_mask"])
    result = DynamicTokenTree.update_attention_mask(**sample_inputs)
    logging.warning(f"Step 2 result:\n{result}")

    assert result.shape == sample_inputs['prev_attn_mask'].shape
    assert torch.all(result[:, 0, 1:3, 1:3] == torch.eye(2))  # Check if eye matrix is added for parent positions
    assert torch.any(result[:, 0, 5:13, :5] != 0)  # Check if mask is copied from parents (8 new nodes)

def test_update_attention_mask_step3(sample_inputs):
    sample_inputs['input_position_ids'] = torch.tensor([[1], [1]])
    sample_inputs['position_ids_offset'] = torch.tensor([[1], [1]])
    result = DynamicTokenTree.update_attention_mask(**sample_inputs)
    logging.warning(f"Step 3 result:\n{result}")
    
    assert result.shape == sample_inputs['prev_attn_mask'].shape
    assert torch.all(result[:, 0, 0, 0] == 1)  # Check if the first element is 1 (eye matrix)
    assert torch.all(result[:, 0, 1:5, 0] == 1)  # Check if the mask is copied correctly for 4 children


def test_update_attention_mask_different_parents(sample_inputs):
    sample_inputs['step'] = 1
    sample_inputs['input_position_ids'] = torch.tensor([[1, 2], [2,3]])
    
    result = DynamicTokenTree.update_attention_mask(**sample_inputs)
    logging.warning(f"Different parents result:\n{result}")

    assert result.shape == sample_inputs['prev_attn_mask'].shape
    assert torch.all(result[0, 0, 1:3, 1:3] == torch.eye(2))  # Check first batch item
    assert torch.all(result[1, 0, 2:4, 2:4] == torch.eye(2))  # Check second batch item

def test_get_draft_attention_mask(sample_inputs):
    # First, create an updated attention mask
    updated_mask = DynamicTokenTree.update_attention_mask(**sample_inputs)
    
    # Now, test get_draft_attention_mask
    bs = 2
    num_selected = 3
    selected_position_ids = torch.tensor([[0, 1, 2], [0, 2, 4]])
    result = DynamicTokenTree.get_draft_attention_mask(updated_mask, selected_position_ids, torch.tensor([[0], [0]]))
    

    assert result.shape == (bs, 1, num_selected, sample_inputs['prev_attn_mask'].shape[-1])
    
    # Check if the mask is correctly gathered for each batch and position
    for b in range(bs):
        for i, pos in enumerate(selected_position_ids[b]):
            assert torch.all(result[b, 0, i] == updated_mask[b, 0, pos])

    selected_position_ids = torch.tensor([[1, 2, 3], [1, 3, 5]])
    result = DynamicTokenTree.get_draft_attention_mask(updated_mask, selected_position_ids, torch.tensor([[1], [1]]))
    

    assert result.shape == (bs, 1, num_selected, sample_inputs['prev_attn_mask'].shape[-1])
    
    # Check if the mask is correctly gathered for each batch and position
    for b in range(bs):
        for i, pos in enumerate(selected_position_ids[b] - torch.tensor([[1], [1]])[b]):
            assert torch.all(result[b, 0, i] == updated_mask[b, 0, pos])
def test_get_draft_rotary_position_ids(sample_inputs):
    # First, create an updated attention mask
    updated_mask = DynamicTokenTree.update_attention_mask(**sample_inputs)
    
    # Now, test get_draft_rotary_position_ids
    bs = 2
    num_selected = 3
    selected_position_ids = torch.tensor([[0, 1, 2], [0, 2, 5]])
    
    result = DynamicTokenTree.get_draft_rotary_position_ids(updated_mask, selected_position_ids,torch.tensor([[0],[0]]))
    logging.warning(f"Get draft attention mask result:\n {updated_mask}")
    logging.warning(f"Get draft rotary position ids result:\n{result}")

    assert result.shape == (bs, num_selected)
    
    # Check if the rotary position ids are correctly calculated
    for b in range(bs):
        for i, pos in enumerate(selected_position_ids[b]):
            expected_value = updated_mask[b, 0, pos].sum()
            assert result[b, i] == expected_value    

    selected_position_ids = torch.tensor([[0, 1, 2], [1, 3, 6]])
    result = DynamicTokenTree.get_draft_rotary_position_ids(updated_mask, selected_position_ids,torch.tensor([[0],[1]]))
    logging.warning(f"Get draft attention mask result:\n {updated_mask}")
    logging.warning(f"Get draft rotary position ids result:\n{result}")

    assert result.shape == (bs, num_selected)
    
    # Check if the rotary position ids are correctly calculated
    for b in range(bs):
        for i, pos in enumerate(selected_position_ids[b] - torch.tensor([[0],[1]])[b]):
            expected_value = updated_mask[b, 0, pos].sum()
            assert result[b, i] == expected_value  

def test_get_path():
    bs = 2
    num_verified = 4
    
    verified_position_ids = torch.tensor([[0, 1, 2, 4], [0, 1, 2, 3]])
    
    updated_mask = torch.tensor([
        [[[1,0,0,0,0,0,0,0,0,0],
          [1,1,0,0,0,0,0,0,0,0],
          [1,0,1,0,0,0,0,0,0,0],
          [1,1,0,1,0,0,0,0,0,0],
          [1,0,1,0,1,0,0,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0]]],
        [[[1,0,0,0,0,0,0,0,0,0],
          [1,1,0,0,0,0,0,0,0,0],
          [1,0,1,0,0,0,0,0,0,0],
          [1,1,0,1,0,0,0,0,0,0],
          [1,0,1,0,1,0,0,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0]]]
    ])

    result = DynamicTokenTree.get_path(updated_mask, verified_position_ids)
    
    logging.warning(f"Get path result:\n{result}")
    

    assert result.shape == (bs, num_verified, num_verified), f"Expected shape {(bs, num_verified, num_verified)}, but got {result.shape}"
    
    # Check if the path is correctly calculated
    for b in range(bs):
        for i in range(num_verified):
            path = result[b, i]
            valid_path = path[path < num_verified]
            
            # Check the pattern of the attention mask along the path
            mask_values = []
            for idx in valid_path:
                pos = verified_position_ids[b, int(idx)]
                mask_values.append(updated_mask[b, 0, verified_position_ids[b, i],pos].item())
            
            # Check for the pattern of consecutive 1s followed by consecutive 0s
            if mask_values:
                first_zero = next((i for i, v in enumerate(mask_values) if v == 0), len(mask_values))
                logging.warning(path)
                logging.warning(mask_values)
                assert all(v == 1 for v in mask_values[:first_zero]), f"Non-consecutive 1s in mask for batch {b}, position {i}"
                assert all(v == 0 for v in mask_values[first_zero:]), f"Non-consecutive 0s in mask for batch {b}, position {i}"

    # Check if invalid indices are set to num_verified
    assert torch.all((result >= num_verified) == (result == num_verified)), "Invalid indices not set to num_verified"

    print("All tests passed successfully!")

def test_update_adj_matrix():
    # Set up test parameters
    bs = 2
    spec_len = 10  # length of the sequence
    a = 1  # number of parent positions
    b = 2  # number of new positions

    # Create a sample previous adjacency matrix
    prev_adj_matrix = torch.zeros((bs, spec_len, spec_len), dtype=torch.long)

    # Create sample parent and new position IDs
    parent_position_id = torch.tensor([[0], [0]], dtype=torch.long)  # shape: [bs, a]
    new_position_id = torch.tensor([[1,2], [2,3]], dtype=torch.long)  # shape: [bs, b]


    updated_adj_matrix = DynamicTokenTree.update_adj_matrix(prev_adj_matrix, parent_position_id, new_position_id, torch.tensor([[0],[0]]))
    logging.warning(f"updated_adj_matrix: {updated_adj_matrix}")

    # Perform assertions
    assert updated_adj_matrix.shape == (bs, spec_len, spec_len), f"Expected shape {(bs, spec_len, spec_len)}, but got {updated_adj_matrix.shape}"

    # Check if the correct elements are updated
    for batch in range(bs):
        for parent in parent_position_id[batch]:
            for new in new_position_id[batch]:
                assert updated_adj_matrix[batch, parent, new] == 1, f"Expected 1 at [{batch}, {parent}, {new}], but got {updated_adj_matrix[batch, parent, new]}"

    # Check if other elements remain unchanged
    expected_ones = bs * a * b
    assert torch.sum(updated_adj_matrix) == expected_ones, f"Expected {expected_ones} ones, but got {torch.sum(updated_adj_matrix)}"

    # Check if the method doesn't modify the input tensor
    assert not torch.all(prev_adj_matrix == updated_adj_matrix), "Input tensor was modified in-place"

    # Create a sample previous adjacency matrix
    prev_adj_matrix = torch.zeros((bs, spec_len, spec_len), dtype=torch.long)

    # Create sample parent and new position IDs
    parent_position_id = torch.tensor([[0], [1]], dtype=torch.long)
    new_position_id = torch.tensor([[1,2], [2,3]], dtype=torch.long)


    updated_adj_matrix = DynamicTokenTree.update_adj_matrix(prev_adj_matrix, parent_position_id, new_position_id, torch.tensor([[0],[1]]))
    logging.warning(f"updated_adj_matrix: {updated_adj_matrix}")

    # Perform assertions
    assert updated_adj_matrix.shape == (bs, spec_len, spec_len), f"Expected shape {(bs, spec_len, spec_len)}, but got {updated_adj_matrix.shape}"

    # Check if the correct elements are updated
    for batch in range(bs):
        for parent in (parent_position_id[batch] - torch.tensor([[0],[1]])[batch]):
            for new in new_position_id[batch]:
                assert updated_adj_matrix[batch, parent, new] == 1, f"Expected 1 at [{batch}, {parent}, {new}], but got {updated_adj_matrix[batch, parent, new]}"

    # Check if other elements remain unchanged
    expected_ones = bs * a * b
    assert torch.sum(updated_adj_matrix) == expected_ones, f"Expected {expected_ones} ones, but got {torch.sum(updated_adj_matrix)}"

    # Check if the method doesn't modify the input tensor
    assert not torch.all(prev_adj_matrix == updated_adj_matrix), "Input tensor was modified in-place"

    print("All tests passed successfully!")

def test_compute_cumulative_draft_prob_and_convert_logits_to_prob():
    # Test case for two different trees with batch size 2
    adj_matrix = torch.tensor([
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    ], dtype=torch.float16)

    node_values = torch.tensor([
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.4, 0.5, 0.6, 0.7, 0.8]
    ], dtype=torch.float16)

    # Expected results (calculated manually)
    # Batch 1: Same as before
    # Batch 2:
    # Node 0: 0.4
    # Node 1: 0.4 * 0.5 = 0.2
    # Node 2: 0.4 * 0.6 = 0.24
    # Node 3: 0.4 * 0.5 * 0.7 = 0.14
    # Node 4: 0.4 * 0.6 * 0.8 = 0.192
    expected_cumulative = torch.tensor([
        [0.5, 0.3, 0.21, 0.24, 0.189],
        [0.4, 0.2, 0.24, 0.14, 0.192]
    ], dtype=torch.float32)

    cumulative_prob = DynamicTokenTree.compute_cumulative_draft_prob(adj_matrix, node_values)

    assert cumulative_prob.shape == node_values.shape, f"Expected shape {node_values.shape}, but got {cumulative_prob.shape}"

    assert cumulative_prob.dtype == torch.float32, f"Expected dtype torch.float32, but got {cumulative_prob.dtype}"

    assert torch.allclose(cumulative_prob, expected_cumulative, rtol=1e-3, atol=1e-3), \
        f"Expected {expected_cumulative}, but got {cumulative_prob}"

    # Test convert_logits_to_prob
    logits = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0]
    ], dtype=torch.float32)
    expected_prob = torch.tensor([
        [0.0667, 0.1333, 0.2000, 0.2667, 0.3333],
        [0.3333, 0.2667, 0.2000, 0.1333, 0.0667]
    ], dtype=torch.float32)

    prob = DynamicTokenTree.convert_logits_to_prob(logits)


    assert prob.shape == logits.shape, f"Expected shape {logits.shape}, but got {prob.shape}"
    assert prob.dtype == torch.float32, f"Expected dtype torch.float32, but got {prob.dtype}"


    assert torch.allclose(prob, expected_prob, rtol=1e-3, atol=1e-3), \
        f"Expected {expected_prob}, but got {prob}"

    # Check that probabilities sum to 1 for each batch
    assert torch.allclose(torch.sum(prob, dim=1), torch.tensor([1.0, 1.0]), rtol=1e-5, atol=1e-5), \
        f"Probabilities should sum to 1 for each batch, but got {torch.sum(prob, dim=1)}"

    print("All tests passed successfully!")

def test_compute_cumulative_draft_prob():
    # Test case for two different trees with batch size 2
    adj_matrix = torch.tensor([
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    ], dtype=torch.float16)

    node_values = torch.tensor([
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.4, 0.5, 0.6, 0.7, 0.8]
    ], dtype=torch.float16)

    expected_cumulative = torch.tensor([
        [0.5, 0.3, 0.21, 0.24, 0.189],
        [0.4, 0.2, 0.24, 0.14, 0.192]
    ], dtype=torch.float32)


    cumulative_prob = DynamicTokenTree.compute_cumulative_draft_prob(adj_matrix, node_values)


    assert cumulative_prob.shape == node_values.shape, f"Expected shape {node_values.shape}, but got {cumulative_prob.shape}"

    # Check dtype
    assert cumulative_prob.dtype == torch.float32, f"Expected dtype torch.float32, but got {cumulative_prob.dtype}"

    # Check values (with larger tolerance due to float16 precision in calculations)
    assert torch.allclose(cumulative_prob, expected_cumulative, rtol=1e-3, atol=1e-3), \
        f"Expected {expected_cumulative}, but got {cumulative_prob}"

    print("compute_cumulative_draft_prob test passed successfully!")

def test_convert_logits_to_prob():
    # Test convert_logits_to_prob with multiple batches
    logits = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ], dtype=torch.float32)
    
    expected_prob = torch.tensor([
        [0.0667, 0.1333, 0.2000, 0.2667, 0.3333],
        [0.3333, 0.2667, 0.2000, 0.1333, 0.0667],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]
    ], dtype=torch.float32)

    prob = DynamicTokenTree.convert_logits_to_prob(logits)

    assert prob.shape == logits.shape, f"Expected shape {logits.shape}, but got {prob.shape}"
    assert prob.dtype == torch.float32, f"Expected dtype torch.float32, but got {prob.dtype}"


    assert torch.allclose(prob, expected_prob, rtol=1e-3, atol=1e-3), \
        f"Expected {expected_prob}, but got {prob}"

    # Check that probabilities sum to 1 for each batch
    assert torch.allclose(torch.sum(prob, dim=1), torch.ones(logits.shape[0]), rtol=1e-5, atol=1e-5), \
        f"Probabilities should sum to 1 for each batch, but got {torch.sum(prob, dim=1)}"

    print("convert_logits_to_prob test passed successfully!")

def test_get_target_cache_scatter_index():
    # Test case with the provided path
    path = torch.tensor([
        [[0, 1, 3, 5, 2, 4], [0, 2, 4, 1, 3, 5]],
        [[0, 1, 2, 4, 5, 3], [0, 1, 2, 3, 4, 5]]
    ])

    result = DynamicTokenTree.get_target_cache_scatter_index(path)

    expected_output = torch.tensor([
        [[0, 1, 4, 2, 5, 3],
         [0, 3, 1, 4, 2, 5]],
        [[0, 1, 2, 5, 3, 4],
         [0, 1, 2, 3, 4, 5]]
    ])

    assert result.shape == (2, 2, 6), f"Expected shape (2, 2, 6), but got {result.shape}"
    assert torch.all(result == expected_output), f"Expected {expected_output}, but got {result}"

    print("get_target_cache_scatter_index test passed successfully!")
def test_get_draft_hidden():
    # Create sample inputs
    bs, seq_len, hidden_dim = 2, 10, 4
    num_positions = 3

    all_hidden = torch.randn(bs, seq_len, hidden_dim)
    parent_position_ids = torch.tensor([[1, 3, 5], [2, 4, 6]])


    draft_hidden = DynamicTokenTree.get_draft_hidden(all_hidden, parent_position_ids, torch.tensor([[0],[0]]))


    expected_shape = (bs, num_positions, hidden_dim)
    assert draft_hidden.shape == expected_shape, f"Expected shape {expected_shape}, but got {draft_hidden.shape}"


    for b in range(bs):
        for i, pos in enumerate(parent_position_ids[b]):
            assert torch.all(draft_hidden[b, i] == all_hidden[b, pos]), \
                f"Mismatch at batch {b}, position {i}. Expected {all_hidden[b, pos]}, but got {draft_hidden[b, i]}"

    print("get_draft_hidden test passed successfully!")

    all_hidden = torch.randn(bs, seq_len, hidden_dim)
    parent_position_ids = torch.tensor([[1, 3, 5], [3, 5, 7]])


    draft_hidden = DynamicTokenTree.get_draft_hidden(all_hidden, parent_position_ids, torch.tensor([[0],[1]]))


    expected_shape = (bs, num_positions, hidden_dim)
    assert draft_hidden.shape == expected_shape, f"Expected shape {expected_shape}, but got {draft_hidden.shape}"


    for b in range(bs):
        for i, pos in enumerate(parent_position_ids[b] - torch.tensor([[0],[1]])[b]):
            assert torch.all(draft_hidden[b, i] == all_hidden[b, pos]), \
                f"Mismatch at batch {b}, position {i}. Expected {all_hidden[b, pos]}, but got {draft_hidden[b, i]}"

    print("get_draft_hidden test passed successfully!")

def test_get_parent_position_ids():
    bs = 2
    num_verified = 4
    
    position_ids = torch.tensor([[0, 1, 2, 4], [0, 1, 2, 3]])
    position_ids_offset = torch.tensor([[0], [0]])
    
    attention_mask = torch.tensor([
        [[[1,0,0,0,0,0,0,0,0,0],
          [1,1,0,0,0,0,0,0,0,0],
          [1,0,1,0,0,0,0,0,0,0],
          [1,1,0,1,0,0,0,0,0,0],
          [1,0,1,0,1,0,0,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0]]],
        [[[1,0,0,0,0,0,0,0,0,0],
          [1,1,0,0,0,0,0,0,0,0],
          [1,0,1,0,0,0,0,0,0,0],
          [1,1,0,1,0,0,0,0,0,0],
          [1,0,1,0,1,0,0,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0],
          [1,0,1,0,0,1,1,0,0,0]]]
    ])

    result = DynamicTokenTree.get_parent_position_ids(attention_mask, position_ids, position_ids_offset)
    
    logging.warning(f"Get parent position ids result:\n{result}")
    

    assert result.shape == (bs, num_verified), f"Expected shape {(bs, num_verified)}, but got {result.shape}"
    
    # Check if the parent position ids are correctly calculated
    expected_result = torch.tensor([[0, 1, 2, 4], [0, 1, 2, 3]])
    assert torch.all(result == expected_result), f"Expected {expected_result}, but got {result}"

    # Check individual cases
    assert result[0, 0] == 0, "Root node should have itself as parent"
    assert result[0, 1] == 1, "Second node should have root as parent"
    assert result[0, 2] == 2, "Third node should have second node as parent"
    assert result[0, 3] == 4, "Fourth node should have third node as parent"
    
    assert result[1, 0] == 0, "Root node should have itself as parent"
    assert result[1, 1] == 1, "Second node should have root as parent"
    assert result[1, 2] == 2, "Third node should have second node as parent"
    assert result[1, 3] == 3, "Fourth node should have second node as parent"

    print("All tests passed successfully!")

def test_get_selected_path():
    # Set up test parameters
    bs = 2
    path_len = 4

    # Create a sample all_path tensor
    all_path = torch.tensor([[[0,1,3,2],[0,1,2,3]], [[0,2,1,3],[0,3,1,2]]])
    
    # Create a sample selected_idx tensor
    selected_idx = torch.tensor([[0], [1]], dtype=torch.long)  # shape: [bs, 1]


    result = DynamicTokenTree.get_selected_path(all_path, selected_idx)
    logging.warning(f"result: {result}")

    # Perform assertions
    assert result.shape == (bs, path_len), f"Expected shape {(bs, path_len)}, but got {result.shape}"

    # Check if the correct paths are selected
    for batch in range(bs):
        expected_path = all_path[batch, selected_idx[batch, 0]]
        assert torch.all(result[batch] == expected_path), f"Mismatch in batch {batch}. Expected {expected_path}, but got {result[batch]}"

    print("All tests passed successfully!")

def test_get_generated_token_position_ids():
    # Set up test parameters
    step = 2
    branching_factor = 3
    num_inputs = 2
    bs = 4


    result = DynamicTokenTree.get_generated_token_position_ids(step, branching_factor, num_inputs, bs)
    
    # Perform assertions
    expected_shape = (bs, branching_factor * num_inputs)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    # Check if the values are correct
    expected_start = 1 + branching_factor + (step - 1) * branching_factor * num_inputs
    expected_end = 1 + branching_factor + step * branching_factor * num_inputs
    expected_values = torch.arange(expected_start, expected_end)
    
    assert torch.all(result[0] == expected_values), f"Mismatch in values. Expected {expected_values}, but got {result[0]}"

    # Check if all batches are the same
    for batch in range(1, bs):
        assert torch.all(result[batch] == result[0]), f"Mismatch in batch {batch}. Expected {result[0]}, but got {result[batch]}"

    # Test for step 0
    result_step0 = DynamicTokenTree.get_generated_token_position_ids(0, branching_factor, num_inputs, bs)
    result_step1 = DynamicTokenTree.get_generated_token_position_ids(1, branching_factor, num_inputs, bs)
    result_step2 = DynamicTokenTree.get_generated_token_position_ids(2, branching_factor, num_inputs, bs)
    result_step3 = DynamicTokenTree.get_generated_token_position_ids(3, branching_factor, num_inputs, bs)
    expected_step0 = torch.arange(1, 1 + branching_factor).unsqueeze(0).expand(bs, -1)
    print(f"result_step0: {result_step0}")
    print(f"result_step1: {result_step1}")
    print(f"result_step2: {result_step2}")
    print(f"result_step3: {result_step3}")
    assert torch.all(result_step0 == expected_step0), f"Mismatch for step 0. Expected {expected_step0}, but got {result_step0}"

def test_get_target_active_mask():
    # Set up test parameters
    bs = 2
    spec_len = 10
    num_verified = 3
    indices_offset = 2

    # Create a sample matrix
    matrix = torch.rand(bs, 1, spec_len, spec_len)

    # Create sample input indices
    input_indices = torch.tensor([[2, 4, 6], [3, 5, 7]])


    result = DynamicTokenTree.get_target_active_mask(matrix, input_indices, indices_offset)

    # Perform assertions
    expected_shape = (bs, 1, num_verified, num_verified)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    # Check if the values are correct
    for batch in range(bs):
        for i in range(num_verified):
            for j in range(num_verified):
                expected_value = matrix[batch, 0, input_indices[batch, i] - indices_offset, input_indices[batch, j] - indices_offset]
                assert torch.isclose(result[batch, 0, i, j], expected_value), f"Mismatch in batch {batch} at position ({i}, {j}). Expected {expected_value}, but got {result[batch, 0, i, j]}"

    # Test with different indices_offset
    indices_offset_0 = 0
    result_0 = DynamicTokenTree.get_target_active_mask(matrix, input_indices, indices_offset_0)
    for batch in range(bs):
        for i in range(num_verified):
            for j in range(num_verified):
                expected_value = matrix[batch, 0, input_indices[batch, i], input_indices[batch, j]]
                assert torch.isclose(result_0[batch, 0, i, j], expected_value), f"Mismatch with offset 0 in batch {batch} at position ({i}, {j}). Expected {expected_value}, but got {result_0[batch, 0, i, j]}"

    # Print some debug information
    print(f"Input matrix shape: {matrix.shape}")
    print(f"Input indices: {input_indices}")
    print(f"Result shape: {result.shape}")
    print(f"Result with offset {indices_offset}:\n{result}")
    print(f"Result with offset 0:\n{result_0}")

def test_get_comp():
    # Set up test parameters
    bs = 2
    spec_len = 4
    num_path = 3
    path_len = 3

    # Create sample inputs
    candidate_input_ids = torch.randint(0, 100, (bs, spec_len))
    target_tokens = torch.randint(0, 100, (bs, spec_len))
    paths = torch.randint(0, spec_len, (bs, num_path, path_len))


    result_candidate, result_target = DynamicTokenTree.get_comp(candidate_input_ids, target_tokens, paths)

    # Perform assertions
    expected_shape = (bs, num_path, path_len)
    assert result_candidate.shape == expected_shape, f"Expected shape {expected_shape}, but got {result_candidate.shape}"
    assert result_target.shape == expected_shape, f"Expected shape {expected_shape}, but got {result_target.shape}"

    # Check if the values are correct
    for batch in range(bs):
        for path in range(num_path):
            for pos in range(path_len):
                index = paths[batch, path, pos]
                expected_candidate = candidate_input_ids[batch, index]
                expected_target = target_tokens[batch, index]
                assert torch.isclose(result_candidate[batch, path, pos], expected_candidate), f"Mismatch in candidate_input_ids at batch {batch}, path {path}, position {pos}. Expected {expected_candidate}, but got {result_candidate[batch, path, pos]}"
                assert torch.isclose(result_target[batch, path, pos], expected_target), f"Mismatch in target_tokens at batch {batch}, path {path}, position {pos}. Expected {expected_target}, but got {result_target[batch, path, pos]}"

    # Test with different spec_len
    spec_len_2 = 15
    candidate_input_ids_2 = torch.randint(0, 100, (bs, spec_len_2))
    target_tokens_2 = torch.randint(0, 100, (bs, spec_len_2))
    paths_2 = torch.randint(0, spec_len_2, (bs, num_path, path_len))
    
    result_candidate_2, result_target_2 = DynamicTokenTree.get_comp(candidate_input_ids_2, target_tokens_2, paths_2)
    
    assert result_candidate_2.shape == expected_shape, f"Expected shape {expected_shape}, but got {result_candidate_2.shape}"
    assert result_target_2.shape == expected_shape, f"Expected shape {expected_shape}, but got {result_target_2.shape}"

    print(f"Input candidate_input_ids shape: {candidate_input_ids.shape}")
    print(f"Input target_tokens shape: {target_tokens.shape}")
    print(f"Input paths shape: {paths.shape}")
    print(f"Result candidate shape: {result_candidate.shape}")
    print(f"Result target shape: {result_target.shape}")
    print(f"Sample input candidate:\n{candidate_input_ids}")
    print(f"Sample input target:\n{target_tokens}")
    print(f"Sample paths:\n{paths}")
    print(f"Sample result candidate:\n{result_candidate}")
    print(f"Sample result target:\n{result_target}")

def test_new_get_parent_position_ids():
    # Set up test parameters
    bs = 2
    spec_len = 4
    max_len = 10

    # Create sample inputs
    adj = torch.zeros(bs, max_len, max_len)
    # Set up a reasonable adj matrix with only direct connections
    for b in range(bs):
        adj[b, 0, 1] = 1  # Root node (0) connected to node 1
        adj[b, 1, 2] = 1  # Node 1 connected to node 2
        adj[b, 2, 3] = 1  # Node 2 connected to node 3
        adj[b, 3, 4] = 1  # Node 3 connected to node 4
    
    position_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    position_ids_offset = torch.tensor([[0],
                                        [1]])


    result = DynamicTokenTree.new_get_parent_position_ids(adj, position_ids, position_ids_offset)

    # Perform assertions
    expected_shape = (bs, spec_len)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    # Check if the values are correct
    expected_result = torch.tensor([[0, 1, 2, 3],
                                    [1, 2, 3, 4]])
    assert torch.all(result == expected_result), f"Expected result {expected_result}, but got {result}"

    # Print some debug information
    print(f"Input adj shape: {adj.shape}")
    print(f"Input position_ids shape: {position_ids.shape}")
    print(f"Input position_ids_offset shape: {position_ids_offset.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Sample adj matrix:\n{adj[0]}")
    print(f"Sample position_ids:\n{position_ids}")
    print(f"Sample position_ids_offset:\n{position_ids_offset}")
    print(f"Result:\n{result}")

def test_update_attention_mask_for_leaf():
    # Set up test parameters
    bs = 2
    spec_len = 4

    # Create sample inputs
    attn_mask = torch.zeros(bs, spec_len, spec_len)
    # Set up a reasonable attention mask
    attn_mask[:, :, 0] = 1  # All tokens can attend to the first token
    attn_mask[:, 1, 1] = 1  # Second token can attend to itself
    attn_mask[:, 2, 2] = 1  # Third token can attend to itself

    result = DynamicTokenTree.update_attention_mask_for_leaf(attn_mask)

    # Perform assertions
    expected_shape = (bs, spec_len, spec_len)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    # Check if the values are correct
    expected_result = torch.tensor([
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 0, 1, 0],
         [1, 0, 0, 1]],
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 0, 1, 0],
         [1, 0, 0, 1]]
    ])
    assert torch.all(result == expected_result), f"Expected result {expected_result}, but got {result}"

    # Print some debug information
    print(f"Input attn_mask shape: {attn_mask.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Sample input attn_mask:\n{attn_mask[0]}")
    print(f"Result:\n{result[0]}")
if __name__ == "__main__":
    pytest.main([__file__])