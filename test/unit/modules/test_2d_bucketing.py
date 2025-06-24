import math
import unittest
from typing import List
from unittest.mock import Mock

import torch

from neuronx_distributed_inference.models.model_wrapper import ModelWrapper, FUSED_SPECULATION_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG


def generate_mock_args_tokengen(position_ids: List[int],
                                vocab_size: int = 131072,
                                max_sequence_len: int = 256,
                                speculation_enabled: bool = False,
                                speculation_len: int = 0):
    assert min(position_ids) >= 0

    batch_size = len(position_ids)
    prior_sequence_length = max(position_ids)

    # Input IDs - Generate random tokens from the vocabulary
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, 1), dtype=torch.int32)

    # Position IDs - Apply input position IDs
    position_ids = torch.tensor(position_ids, dtype=torch.int32).unsqueeze(dim=1)

    # Attention mask - Derive from position IDs
    mask = torch.arange(prior_sequence_length, dtype=torch.int32).view(1, -1).expand(batch_size, prior_sequence_length)
    attention_mask = (position_ids > mask).to(dtype=position_ids.dtype)

    # Sequence IDs - Batch size arange
    seq_ids = torch.arange(batch_size, dtype=torch.int32)

    # Sampling params - Mock with torch.ones
    sampling_params = torch.ones((batch_size, 3), dtype=torch.float32)

    # Slot mapping - Mock with torch.zeros
    if not speculation_enabled:
        slot_mapping = torch.zeros((batch_size, 1), dtype=torch.int32)
    else:
        slot_mapping = torch.zeros((batch_size, speculation_len), dtype=torch.int32)

    # Block table - Mock with torch.zeros
    mock_block_size = 32
    num_blocks = int(math.ceil(max_sequence_len / float(mock_block_size)))
    block_table = torch.zeros((batch_size, num_blocks), dtype=torch.int32)

    # Num queries - Set to ones for token generation
    num_queries = torch.ones((batch_size, 1), dtype=torch.int32)

    # Computed context - Copy position IDs
    computed_context_lens = position_ids.clone()

    # Assemble args
    if not speculation_enabled:
        args = [input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                slot_mapping,
                block_table,
                num_queries,
                computed_context_lens]
    else:
        args = [input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),
                torch.empty(0),
                slot_mapping,
                block_table,
                num_queries,
                computed_context_lens,
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0)]

    return args


class MockModelWrapper:
    def __init__(self, neuron_config, async_mode, tag=TOKEN_GENERATION_MODEL_TAG):
        self.neuron_config = neuron_config
        self.tag = tag
        self.async_mode = async_mode

MockModelWrapper.get_target_2d_bucket_for_prefix_caching = ModelWrapper.get_target_2d_bucket_for_prefix_caching


def tokengen_unit_test(buckets: List[List[int]],
                                position_ids: List[int],
                                expected: List[int],
                                allow_input_truncation: bool = False,
                                strategy: str ="first_fit",
                                async_mode=False):
    # Set up
    neuron_config = Mock()
    neuron_config.enable_fused_speculation = False
    neuron_config.allow_input_truncation = allow_input_truncation
    neuron_config.buckets = buckets

    model_wrapper = MockModelWrapper(neuron_config, async_mode=async_mode)

    args = generate_mock_args_tokengen(position_ids=position_ids)

    # Run test
    pad_lengths = model_wrapper.get_target_2d_bucket_for_prefix_caching(*args, strategy=strategy)

    # Evaluate result
    assert isinstance(pad_lengths, torch.Tensor)
    assert int(pad_lengths.ndim) == 1
    assert int(pad_lengths.shape[0]) == 2
    assert pad_lengths.dtype == torch.int32

    pad_lengths = pad_lengths.tolist()
    assert pad_lengths == expected, f"Expected: {expected}, Actual: {pad_lengths}"


def fusedspec_unit_test(buckets: List[List[int]],
                                 position_ids: List[int],
                                 expected: List[int],
                                 allow_input_truncation: bool = False,
                                 strategy: str = "first_fit",
                                 speculation_len: int = 5,
                                 async_mode=False):
    # Set up
    neuron_config = Mock()
    neuron_config.enable_eagle_speculation = True
    neuron_config.enable_fused_speculation = True
    neuron_config.allow_input_truncation = allow_input_truncation
    neuron_config.buckets = buckets
    neuron_config.speculation_length = speculation_len

    model_wrapper = MockModelWrapper(neuron_config, async_mode=async_mode, tag=FUSED_SPECULATION_MODEL_TAG)

    args = generate_mock_args_tokengen(position_ids=position_ids,
                                       speculation_enabled=True,
                                       speculation_len=speculation_len)

    # Run test
    pad_lengths = model_wrapper.get_target_2d_bucket_for_prefix_caching(*args, strategy=strategy)

    # Evaluate result
    assert isinstance(pad_lengths, torch.Tensor)
    assert int(pad_lengths.ndim) == 1
    assert int(pad_lengths.shape[0]) == 2
    assert pad_lengths.dtype == torch.int32

    pad_lengths = pad_lengths.tolist()
    assert pad_lengths == expected, f"Expected: {expected}, Actual: {pad_lengths}"


def async_2d_all_unit_tests(buckets: List[List[int]],
                            position_ids: List[int],
                            expected: List[int],
                            allow_input_truncation: bool = False,
                            strategy: str = "first_fit",
                            speculation_len: int = 5):
    # In async mode, non-speculative tokengen and fused speculation should behave identically.
    # The bucketing strategy is determined within async processing.
    tokengen_unit_test(buckets,
                       position_ids,
                       expected,
                       allow_input_truncation=allow_input_truncation,
                       strategy=strategy,
                       async_mode=True)
    fusedspec_unit_test(buckets,
                        position_ids,
                        expected,
                        allow_input_truncation=allow_input_truncation,
                        strategy=strategy,
                        speculation_len=speculation_len,
                        async_mode=True)


# Swap the first index in the list with all other positions in sequence.
def first_index_position_runner(position_ids):
    cur_positions = position_ids.copy()
    N = len(position_ids)
    for i in range(N):
        cur_positions = position_ids.copy()
        cur_positions[0], cur_positions[i] = cur_positions[i], cur_positions[0]
        yield cur_positions


class TestAsync2DTokenGenBucketing(unittest.TestCase):
    def test_basic_first_bucket_first_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [1] * 4
        expected = [1, 128]
        async_2d_all_unit_tests(buckets, position_ids, expected, strategy="first_fit")

    def test_one_seq_first_bucket_first_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [9, 1, 1, 1]
        expected = [1, 128]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, strategy="first_fit")

    def test_basic_second_bucket_first_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [135] * 4
        expected = [1, 256]
        async_2d_all_unit_tests(buckets, position_ids, expected, strategy="first_fit")

    def test_one_seq_second_bucket_first_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [135, 1, 1, 1]
        expected = [1, 256]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, strategy="first_fit")

    def test_edge_case_first_bucket_first_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [127, 1, 1, 1]
        expected = [1, 128]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, strategy="first_fit")

        buckets = [[1, 128], [1, 256]]
        position_ids = [128, 1, 1, 1]
        expected = [1, 256]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, strategy="first_fit")

    def test_two_bucket_truncation_first_fit(self):
        # Exhibits same behavior as without allow_input_truncation when position IDs in-bounds.
        buckets = [[1, 128], [1, 256]]
        position_ids = [135, 1, 1, 1]
        expected = [1, 256]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="first_fit")

        # Returns maximum position when position ID exceeds maximum bucket and allow_input_truncation set.
        position_ids = [259, 1, 1, 1]
        expected = [1, 256]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="first_fit")

        # Raises value error when position ID exceeds maximum bucket and allow_input_truncation unset.
        position_ids = [259, 1, 1, 1]
        for cur_positions in first_index_position_runner(position_ids):
            with self.assertRaises(ValueError):
                async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=False, strategy="first_fit")

    def test_four_bucket_first_fit(self):
        buckets = [[1, 128], [1, 256], [1, 512], [1, 1024]]

        position_ids = [10, 1, 1, 1]
        expected = [1, 128]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="first_fit")

        position_ids = [129, 1, 1, 1]
        expected = [1, 256]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="first_fit")

        position_ids = [257, 1, 1, 1]
        expected = [1, 512]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="first_fit")

        position_ids = [513, 1, 1, 1]
        expected = [1, 1024]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="first_fit")

    def test_invalid_fit_strategy(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [1] * 4
        with self.assertRaises(ValueError):
            async_2d_all_unit_tests(buckets, position_ids, None, strategy="third_fit")

    def test_basic_first_bucket_second_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [1] * 4
        expected = [1, 256]
        async_2d_all_unit_tests(buckets, position_ids, expected, strategy="second_fit")

    def test_one_seq_first_bucket_second_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [9, 1, 1, 1]
        expected = [1, 256]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, strategy="second_fit")

    def test_one_bucket_second_fit(self):
        buckets = [[1, 128]]
        position_ids = [1] * 4
        expected = [1, 128]
        async_2d_all_unit_tests(buckets, position_ids, expected, strategy="second_fit")

    def test_basic_second_bucket_second_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [135] * 4
        expected = [1, 256]
        async_2d_all_unit_tests(buckets, position_ids, expected, strategy="second_fit")

    def test_edge_cases_second_fit(self):
        buckets = [[1, 128], [1, 256], [1, 512], [1, 1024]]
        position_ids = [127, 1, 1, 1]
        expected = [1, 256]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, strategy="second_fit")

        position_ids = [128, 1, 1, 1]
        expected = [1, 512]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, strategy="second_fit")

        position_ids = [255, 1, 1, 1]
        expected = [1, 512]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, strategy="second_fit")

        position_ids = [256, 1, 1, 1]
        expected = [1, 1024]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, strategy="second_fit")

    def test_four_bucket_second_fit(self):
        buckets = [[1, 128], [1, 256], [1, 512], [1, 1024]]

        position_ids = [10, 1, 1, 1]
        expected = [1, 256]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="second_fit")

        position_ids = [129, 1, 1, 1]
        expected = [1, 512]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="second_fit")

        position_ids = [257, 1, 1, 1]
        expected = [1, 1024]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="second_fit")

        position_ids = [513, 1, 1, 1]
        expected = [1, 1024]
        for cur_positions in first_index_position_runner(position_ids):
            async_2d_all_unit_tests(buckets, cur_positions, expected, allow_input_truncation=True, strategy="second_fit")


class TestSync2DTokenGenBucketing(unittest.TestCase):
    def test_basic_first_bucket_first_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [1] * 4
        expected = [1, 128]

        fusedspec_unit_test(buckets, position_ids, expected, strategy="first_fit",  speculation_len=5, async_mode=False)

    def test_basic_second_bucket_first_fit(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [129, 1, 1, 1]
        expected = [1, 256]

        for cur_positions in first_index_position_runner(position_ids):
            fusedspec_unit_test(buckets, cur_positions, expected, strategy="first_fit", speculation_len=5, async_mode=False)

    def test_basic_first_bucket_edge(self):
        buckets = [[1, 128], [1, 256]]
        position_ids = [123] * 4
        expected_low = [1, 128]
        expected_high = [1, 256]
        fusedspec_unit_test(buckets, position_ids, expected_low, strategy="first_fit", speculation_len=4, async_mode=False)
        fusedspec_unit_test(buckets, position_ids, expected_high, strategy="first_fit", speculation_len=5, async_mode=False)
