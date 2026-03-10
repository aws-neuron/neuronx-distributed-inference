import pytest
import torch
import torch_xla

from neuronx_distributed_inference.modules.eagle.hidden_state import HiddenStateRollingBuffer


def check_token_generation_rollback(batch_size: int, padding: int, hidden_size: int):

    assert batch_size <= 8 # Hardcoded num_accepted tokens used in the test

    device = torch_xla.device()

    k = 5
    seq_ids = torch.randperm(batch_size, dtype=torch.int32).reshape(batch_size, 1).to(device)
    position_ids = torch.randint(128256, size=(batch_size, 1), dtype=torch.int32).to(device)

    # Tests minimum accepted, maximum accepted, and varying in-between values.
    num_accepted0 = torch.tensor([[1],[5],[3],[5],[2],[4],[1],[2]], dtype=torch.int32)[:batch_size].to(device)
    num_accepted1 = torch.tensor([[1],[5],[4],[1],[2],[3],[5],[4]], dtype=torch.int32)[:batch_size].to(device)

    hidden_state0 = torch.rand((batch_size, 1, hidden_size)).to(device)
    next_position_ids = position_ids + num_accepted0
    hidden_state1 = torch.rand((batch_size, 1, hidden_size)).to(device)

    state = HiddenStateRollingBuffer(batch_size + padding, (k * 2), hidden_size, inplace=True, use_kernel=True).to(device).to(device)

    # Set State Iteration 0
    state.set_state(seq_ids, next_position_ids, hidden_state0)

    # Get State Iteration 1
    position_ids = next_position_ids  # simulates next step
    actual = state.get_state(seq_ids, position_ids)
    position_ids_from_iter1 = position_ids
    torch.testing.assert_close(actual=actual.cpu(), expected=hidden_state0.cpu())

    # Set State Iteration 1
    next_position_ids = position_ids + num_accepted1
    state.set_state(seq_ids, next_position_ids, hidden_state1)

    # ----------- Simulated Rollback ---------------

    # Get State Iteration 1 - This should always correctly fetch the state that
    # was set in iteration 0 even after setting state for iteration 1. We must
    # guarantee that we do not overwrite.
    position_ids = next_position_ids
    actual_from_iter1 = state.get_state(seq_ids, position_ids)
    actual = state.get_state(seq_ids, position_ids_from_iter1)
    torch.testing.assert_close(actual=actual_from_iter1.cpu(), expected=hidden_state1.cpu())
    torch.testing.assert_close(actual=actual.cpu(), expected=hidden_state0.cpu())  # tests rollback


def test_rollback_llama3_70b_batch8():
    check_token_generation_rollback(batch_size=8, padding=1, hidden_size=8192)


def test_rollback_gpt_oss_120b_batch1():
    check_token_generation_rollback(batch_size=1, padding=1, hidden_size=3072)


def test_rollback_gpt_oss_120b_batch8_unpadded():
    check_token_generation_rollback(batch_size=8, padding=0, hidden_size=3072)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
