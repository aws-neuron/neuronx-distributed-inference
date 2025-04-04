import logging

import pytest
import torch

from neuronx_distributed_inference.modules.eagle.hidden_state import HiddenStateRollingBuffer

logger = logging.getLogger(__name__)


@pytest.fixture
def error_logger():
    def _error_logger(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__} with args: {args}, kwargs: {kwargs}")
                raise e

        return wrapper

    return _error_logger


def check_context_encode(batch_size, max_batch_size, hidden_size, position_ids, seq_ids):
    next_position_ids = position_ids + 1
    state = HiddenStateRollingBuffer(max_batch_size, 2, hidden_size, inplace=True)

    # Set state for next iteration (imagine this occurs when position_ids is in use)
    hidden_state = torch.rand((batch_size, 1, hidden_size))
    state.set_state(seq_ids, next_position_ids, hidden_state)

    # Get state for next iteration
    position_ids = next_position_ids  # simulates next step
    actual = state.get_state(seq_ids, position_ids)
    torch.testing.assert_close(actual=actual, expected=hidden_state)


def check_token_generation_b2(
    batch_size, max_batch_size, k, hidden_size, position_ids, seq_ids, num_accepted
):
    next_position_ids = position_ids + num_accepted
    state = HiddenStateRollingBuffer(max_batch_size, k * 2, hidden_size, inplace=True)

    # Set state for prior iteration
    hidden_state = torch.rand((batch_size, 1, hidden_size))
    state.set_state(seq_ids, next_position_ids, hidden_state)

    # Get state for next iteration
    position_ids = next_position_ids  # simulates next step
    actual = state.get_state(seq_ids, position_ids)
    torch.testing.assert_close(actual=actual, expected=hidden_state)


def check_token_generation_b4_rollback(
    batch_size, max_batch_size, k, hidden_size, position_ids, seq_ids, num_accepted0, num_accepted1
):
    hidden_state0 = torch.rand((batch_size, 1, hidden_size))
    next_position_ids = position_ids + num_accepted0
    hidden_state1 = torch.rand((batch_size, 1, hidden_size))

    state = HiddenStateRollingBuffer(max_batch_size, k * 2, hidden_size, inplace=True)

    # Set State Iteration 0
    state.set_state(seq_ids, next_position_ids, hidden_state0)

    # Get State Iteration 1
    position_ids = next_position_ids  # simulates next step
    actual = state.get_state(seq_ids, position_ids)
    position_ids_from_iter1 = position_ids
    torch.testing.assert_close(actual=actual, expected=hidden_state0)

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
    torch.testing.assert_close(actual=actual_from_iter1, expected=hidden_state1)
    torch.testing.assert_close(actual=actual, expected=hidden_state0)  # tests rollback


def test_context_encoding(error_logger):
    @error_logger
    def inner_test():
        for _ in range(1000):
            batch_size = 1
            max_batch_size = 7
            hidden_size = 128
            position_ids = torch.randint(128256, size=(batch_size, 1))
            seq_ids = torch.randint(0, max_batch_size, size=(batch_size, 1), dtype=torch.int32)
            check_context_encode(batch_size, max_batch_size, hidden_size, position_ids, seq_ids)

    inner_test()


def test_token_generation(error_logger):
    @error_logger
    def inner_test():
        for _ in range(1000):
            max_batch_size = batch_size = 2
            k = 4
            hidden_size = 128
            position_ids = torch.randint(128256, size=(batch_size, 1))
            seq_ids = torch.randperm(batch_size).reshape(batch_size, 1)
            num_accepted = torch.randint(1, k, size=(batch_size, 1))
            check_token_generation_b2(
                batch_size, max_batch_size, k, hidden_size, position_ids, seq_ids, num_accepted
            )

    inner_test()


def test_token_generation_b4_rollback(error_logger):
    @error_logger
    def inner_test():
        for _ in range(1000):
            max_batch_size = batch_size = 4
            k = 5
            hidden_size = 128
            seq_ids = torch.randperm(batch_size).reshape(batch_size, 1)
            position_ids = torch.randint(128256, size=(batch_size, 1))
            num_accepted0 = torch.randint(1, k, size=(batch_size, 1))
            num_accepted1 = torch.randint(1, k, size=(batch_size, 1))
            check_token_generation_b4_rollback(
                batch_size,
                max_batch_size,
                k,
                hidden_size,
                position_ids,
                seq_ids,
                num_accepted0,
                num_accepted1,
            )

    inner_test()
