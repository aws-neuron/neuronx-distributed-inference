import pytest
import math
import torch
from unittest.mock import Mock, patch

from neuronx_distributed_inference.utils.accuracy import generate_with_chunked_prefill


BATCH_SIZE = 1
MAX_NUM_SEQS = 8
CHUNK_SIZE = 256
SEQ_LEN = 1024
BLOCK_SIZE = 32
VOCAB_SIZE = 1234


@pytest.fixture
def mock_neuron_model():
    model = Mock()
    
    # Configure the model's config
    config = Mock()
    neuron_config = Mock()
    neuron_config.max_context_length = CHUNK_SIZE
    neuron_config.chunked_prefill_config = Mock()
    neuron_config.chunked_prefill_config.max_num_seqs = MAX_NUM_SEQS
    neuron_config.seq_len = SEQ_LEN
    neuron_config.pa_block_size = BLOCK_SIZE
    
    config.neuron_config = neuron_config
    model.config = config

    # Mock the model's forward pass
    torch.manual_seed(123)
    output = Mock()
    output.logits = torch.rand(BATCH_SIZE, MAX_NUM_SEQS, VOCAB_SIZE)
    model.return_value = output

    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.batch_decode.return_value = ["Generated text 1", "Generated text 2"]
    return tokenizer


@pytest.mark.parametrize("prompt_len", [5, 25])
def test_generate_with_chunked_prefill(mock_neuron_model, mock_tokenizer, prompt_len):
    # Prepare input
    input_ids = torch.randint(0, 32000, (BATCH_SIZE, prompt_len))

    # Call the function
    output_logits = generate_with_chunked_prefill(
        neuron_model=mock_neuron_model,
        tokenizer=mock_tokenizer,
        input_ids=input_ids,
    )

    # Verify the model was called correctly
    max_prefill_len_per_seq = CHUNK_SIZE // MAX_NUM_SEQS
    num_prefill_calls = math.ceil(prompt_len / max_prefill_len_per_seq)
    num_decode_calls = SEQ_LEN - prompt_len - 1
    expected_num_calls = num_prefill_calls + num_decode_calls
    assert mock_neuron_model.call_count == expected_num_calls

    # Check the first call arguments
    first_call_args = mock_neuron_model.call_args_list[0][1]
    assert 'input_ids' in first_call_args
    assert 'position_ids' in first_call_args
    assert 'slot_mapping' in first_call_args
    assert 'block_table' in first_call_args
    assert 'full_context_lens' in first_call_args
    assert 'computed_context_lens' in first_call_args
    
    # Verify shapes of the arguments in the first call
    assert first_call_args['input_ids'].dim() == 2
    assert first_call_args['position_ids'].dim() == 2
    assert first_call_args['slot_mapping'].dim() == 1
    assert first_call_args['block_table'].dim() == 2
    assert first_call_args['full_context_lens'].dim() == 1
    assert first_call_args['computed_context_lens'].dim() == 1

    # Verify the tokenizer was called
    assert mock_tokenizer.batch_decode.call_count == 1

    # Verify output shape
    expected_seq_len = mock_neuron_model.config.neuron_config.seq_len - prompt_len
    assert output_logits.shape == (expected_seq_len, MAX_NUM_SEQS, VOCAB_SIZE)
