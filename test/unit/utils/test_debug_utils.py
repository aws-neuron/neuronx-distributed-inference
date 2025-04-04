import os
import shutil
from unittest.mock import Mock

import pytest
import torch

from neuronx_distributed_inference.models.model_wrapper import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    FUSED_SPECULATION_MODEL_TAG,
    MEDUSA_MODEL_TAG,
    SPECULATION_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.utils.debug_utils import capture_model_inputs

torch.manual_seed(0)

# the values of the generated inputs (other than position ids) are not important for this test, only the shapes are needed
# values that would otherwise be padded are replaced with randomly generated numbers


def generate_cte_inputs(batch_size, input_size, largest_sequence_index=0):
    input_ids = torch.randint(0, 100_000, (batch_size, input_size), dtype=torch.int32)
    attention_mask = torch.ones((batch_size, input_size), dtype=torch.int32)

    largest_sequence_position_ids = torch.arange(0, input_size, 1, dtype=torch.int32)
    position_ids = torch.zeros((batch_size, input_size), dtype=torch.int32)
    position_ids[largest_sequence_index] = largest_sequence_position_ids

    sequence_ids = torch.arange(0, batch_size, 1, dtype=torch.int32)
    sampling_params = torch.ones((batch_size, 3), dtype=torch.int32)

    return [input_ids, attention_mask, position_ids, sequence_ids, sampling_params]


def generate_tkg_inputs(batch_size, largest_sequence_position_id, largest_sequence_index=0):
    input_ids = torch.randint(0, 100_000, (batch_size, 1), dtype=torch.int32)
    attention_mask = torch.ones((batch_size, largest_sequence_position_id), dtype=torch.int32)

    position_ids = torch.zeros((batch_size, 1), dtype=torch.int32)
    position_ids[largest_sequence_index] = largest_sequence_position_id

    sequence_ids = torch.arange(0, batch_size, 1, dtype=torch.int32)
    sampling_params = torch.ones((batch_size, 3), dtype=torch.int32)

    return [input_ids, attention_mask, position_ids, sequence_ids, sampling_params]


KV_CACHE = [
    {
        "kv_mgr.past_key_values.0": torch.randn((1, 1, 64, 16)),
        "kv_mgr.past_key_values.1": torch.randn((1, 1, 64, 16)),
    },
    {
        "kv_mgr.past_key_values.0": torch.randn((1, 1, 64, 16)),
        "kv_mgr.past_key_values.1": torch.randn((1, 1, 64, 16)),
    },
]
FLAT_KV_CACHE = [value for d in KV_CACHE for value in d.values()]

INPUT_CAPTURE_DIRECTORY = "saved_inputs"


def create_mock_neuron_causal_lm(
    initial_input_size,
    on_device_sampling,
    kv_cache_populated,
    generation_model_tag=TOKEN_GENERATION_MODEL_TAG,
):
    mock_neuron_casual_lm = Mock()

    mock_neuron_casual_lm.initial_input_size = initial_input_size
    mock_neuron_casual_lm.on_device_sampling = on_device_sampling
    mock_neuron_casual_lm.kv_cache_populated = kv_cache_populated

    if not kv_cache_populated:
        mock_neuron_casual_lm.context_encoding_model.convert_int64_to_int32 = Mock(
            side_effect=lambda *x, **kwargs: x
        )
        mock_neuron_casual_lm.context_encoding_model.pad_inputs = Mock(
            side_effect=lambda *x, **kwargs: x
        )
    else:
        mock_generation_model = Mock()
        mock_generation_model.tag = generation_model_tag
        mock_generation_model.convert_int64_to_int32 = Mock(side_effect=lambda *x, **kwargs: x)
        mock_generation_model.pad_inputs = Mock(side_effect=lambda *x, **kwargs: x)

        def mock_get_generation_model():
            return mock_generation_model

        mock_neuron_casual_lm.get_generation_model = mock_get_generation_model
        mock_neuron_casual_lm.context_encoding_model.model.nxd_model.state = KV_CACHE

    return mock_neuron_casual_lm


def cleanup():
    if os.path.exists(INPUT_CAPTURE_DIRECTORY):
        shutil.rmtree(INPUT_CAPTURE_DIRECTORY)


"""
Capture Condition:
position_ids are in the form [0...input_size-1] for CTE and [position_id] for TKG.
we offset capture_indices to be inline with the max position_id by doing input_size + capture_indices - 1.
if the max(position_ids) is in the capture indicies, then we capture. 
In CTE, capture will always be index 0 since max(position_ids) is input_size - 1 implying capture_indices must be 0.
In TKG, the capture index will always be max(position_ids) - input_size + 1.
"""


@pytest.mark.parametrize(
    "inputs, capture_indices, initial_input_size, on_device_sampling, expected_captured_index",
    # fmt: off
    [
        (generate_cte_inputs(1, 10, 0), [0, 3, 4], 10, True, 0), # CTE bs=1, capture happens
        (generate_cte_inputs(1, 10, 0), [3, 4], 10, True, None), # CTE bs=1, capture doesn't happen
        (generate_cte_inputs(3, 10, 1), [0, 3, 4], 10, False, 0), # CTE bs=3, capture happens, ODS disabled
        (generate_cte_inputs(3, 10, 1), [3, 4], 10, True, None), # CTE bs=3, capture doesn't happens
    ],
    # fmt: on
)
def test_capture_model_inputs_cte(
    inputs,
    capture_indices,
    initial_input_size,
    on_device_sampling,
    expected_captured_index,
):
    mock_neuron_casual_lm = create_mock_neuron_causal_lm(
        initial_input_size, on_device_sampling, False
    )

    capture_model_inputs(mock_neuron_casual_lm, inputs, capture_indices, INPUT_CAPTURE_DIRECTORY)

    cte_expected_output_path = (
        f"{INPUT_CAPTURE_DIRECTORY}/saved_inputs_cte_output_{expected_captured_index}.pt"
    )

    if expected_captured_index is not None:
        assert os.path.exists(cte_expected_output_path)

        captured_inputs = torch.load(cte_expected_output_path)

        assert isinstance(captured_inputs, tuple)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(inputs, captured_inputs))

        if not on_device_sampling:
            assert len(captured_inputs) == len(inputs) - 1

    else:
        assert not os.path.exists(cte_expected_output_path)

    cleanup()


@pytest.mark.parametrize(
    "inputs, capture_indices, initial_input_size, on_device_sampling, expected_captured_index, generation_model_tag",
    # fmt: off
    [
        (generate_tkg_inputs(1, 10, 0), [6, 7, 8], 4, True, 7, TOKEN_GENERATION_MODEL_TAG), # TKG bs=1, capture happens, 7 + 4 - 1 == 10
        (generate_tkg_inputs(1, 10, 0), [6, 7, 8], 10, True, None, TOKEN_GENERATION_MODEL_TAG), # TKG bs=1, capture doesn't happen
        (generate_tkg_inputs(3, 30, 1), [10, 11, 60], 20, False, 11, TOKEN_GENERATION_MODEL_TAG), # TKG bs=3, capture happens, ODS disabled, 11 + 20 - 1  == 30
        (generate_tkg_inputs(3, 30, 1), [0, 10], 20, False, None, TOKEN_GENERATION_MODEL_TAG), # TKG bs=3, capture doesn't happen
        (generate_tkg_inputs(1, 10, 0), [6, 7, 8], 4, True, 7, FUSED_SPECULATION_MODEL_TAG),
        (generate_tkg_inputs(1, 10, 0), [6, 7, 8], 4, True, 7, MEDUSA_MODEL_TAG),
        (generate_tkg_inputs(1, 10, 0), [6, 7, 8], 4, True, 7, SPECULATION_MODEL_TAG),
    ],
    # fmt: on
)
def test_capture_model_inputs_tkg(
    inputs,
    capture_indices,
    initial_input_size,
    on_device_sampling,
    expected_captured_index,
    generation_model_tag,
):
    mock_neuron_casual_lm = create_mock_neuron_causal_lm(
        initial_input_size, on_device_sampling, True, generation_model_tag=generation_model_tag
    )
    assert mock_neuron_casual_lm.get_generation_model().tag == generation_model_tag

    capture_model_inputs(mock_neuron_casual_lm, inputs, capture_indices, INPUT_CAPTURE_DIRECTORY)

    tkg_expected_output_path_kv_cache = f"{INPUT_CAPTURE_DIRECTORY}/saved_inputs_tkg_with_kv_cache_output_{expected_captured_index}.pt"
    tkg_expected_output_path_without_kv_cache = f"{INPUT_CAPTURE_DIRECTORY}/saved_inputs_tkg_without_kv_cache_output_{expected_captured_index}.pt"

    if expected_captured_index is not None:
        assert os.path.exists(tkg_expected_output_path_kv_cache)
        assert os.path.exists(tkg_expected_output_path_without_kv_cache)

        captured_inputs_kv = torch.load(tkg_expected_output_path_kv_cache)
        captured_inputs_without_kv = torch.load(tkg_expected_output_path_without_kv_cache)

        assert isinstance(captured_inputs_kv, tuple)
        assert isinstance(captured_inputs_without_kv, tuple)

        non_kv_cache_comparison_length = len(inputs) if on_device_sampling else len(inputs) - 1

        assert all(torch.equal(t1, t2) for t1, t2 in zip(inputs, captured_inputs_without_kv))
        assert all(
            torch.equal(t1, t2)
            for t1, t2 in zip(inputs, captured_inputs_kv[0:non_kv_cache_comparison_length])
        )
        assert all(
            torch.equal(t1, t2)
            for t1, t2 in zip(FLAT_KV_CACHE, captured_inputs_kv[non_kv_cache_comparison_length:])
        )

        if not on_device_sampling:
            assert len(captured_inputs_without_kv) == len(inputs) - 1
            assert len(captured_inputs_kv) == len(inputs) - 1 + len(FLAT_KV_CACHE)

    else:
        assert not os.path.exists(tkg_expected_output_path_kv_cache)
        assert not os.path.exists(tkg_expected_output_path_without_kv_cache)

    cleanup()
