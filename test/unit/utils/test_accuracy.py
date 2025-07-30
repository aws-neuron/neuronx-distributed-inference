import pytest
import math
import torch
from unittest.mock import Mock, patch, MagicMock

from neuronx_distributed_inference.utils.accuracy import (
    generate_with_chunked_prefill, 
    shift_inputs_by_offset,
    prepare_inputs_from_prompt,
    generate_expected_logits,
    check_accuracy_logits_v2,
    LogitMatchingValidationError
)
from neuronx_distributed_inference.utils.constants import TEST_PROMPT

from transformers import GenerationConfig

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


class TestShiftInputsByOffset:
    @pytest.fixture
    def mock_inputs(self):
        """Create mock inputs with input_ids and attention_mask attributes"""
        inputs = Mock()
        inputs.input_ids = torch.tensor([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10]
        ])
        inputs.attention_mask = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])
        return inputs
    
    def test_shift_with_single_offset(self, mock_inputs):
        """Test shifting with a single offset value for all sequences"""
        # Single offset for all sequences
        input_start_offsets = [3]
        pad_token_id = 0
        
        shifted_input_ids, shifted_attention_mask = shift_inputs_by_offset(
            mock_inputs, input_start_offsets, pad_token_id
        )
        
        # Expected shapes: batch_size=2, seq_len=8 (original 5 + offset 3)
        assert shifted_input_ids.shape == (2, 8)
        assert shifted_attention_mask.shape == (2, 8)
        
        # Expected content for input_ids
        expected_input_ids = torch.tensor([
            [0, 0, 0, 1, 2, 3, 4, 5],
            [0, 0, 0, 6, 7, 8, 9, 10]
        ])
        assert torch.equal(shifted_input_ids, expected_input_ids)
        
        # Expected content for attention_mask
        expected_attention_mask = torch.tensor([
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1]
        ])
        assert torch.equal(shifted_attention_mask, expected_attention_mask)
    
    def test_shift_with_multiple_offsets(self, mock_inputs):
        """Test shifting with different offset values for each sequence"""
        # Different offsets for each sequence
        input_start_offsets = [2, 4]
        pad_token_id = 0
        
        shifted_input_ids, shifted_attention_mask = shift_inputs_by_offset(
            mock_inputs, input_start_offsets, pad_token_id
        )
        
        # Expected shapes: batch_size=2, seq_len=9 (original 5 + max_offset 4)
        assert shifted_input_ids.shape == (2, 9)
        assert shifted_attention_mask.shape == (2, 9)
        
        # Expected content for input_ids
        expected_input_ids = torch.tensor([
            [0, 0, 1, 2, 3, 4, 5, 0, 0],  # offset 2
            [0, 0, 0, 0, 6, 7, 8, 9, 10]  # offset 4
        ])
        assert torch.equal(shifted_input_ids, expected_input_ids)
        
        # Expected content for attention_mask
        expected_attention_mask = torch.tensor([
            [0, 0, 1, 1, 1, 1, 1, 0, 0],  # offset 2
            [0, 0, 0, 0, 1, 1, 1, 1, 1]   # offset 4
        ])
        assert torch.equal(shifted_attention_mask, expected_attention_mask)
    
    def test_shift_with_zero_offset(self, mock_inputs):
        """Test with zero offset (no shifting)"""
        input_start_offsets = [0]
        pad_token_id = 0
        
        shifted_input_ids, shifted_attention_mask = shift_inputs_by_offset(
            mock_inputs, input_start_offsets, pad_token_id
        )
        
        # Should be no change from original
        assert torch.equal(shifted_input_ids, mock_inputs.input_ids)
        assert torch.equal(shifted_attention_mask, mock_inputs.attention_mask)
    
    def test_shift_with_custom_pad_token(self, mock_inputs):
        """Test using a custom pad token ID"""
        input_start_offsets = [2]
        pad_token_id = 99  # Custom pad token
        
        shifted_input_ids, shifted_attention_mask = shift_inputs_by_offset(
            mock_inputs, input_start_offsets, pad_token_id
        )
        
        # Expected content for input_ids with custom padding
        expected_input_ids = torch.tensor([
            [99, 99, 1, 2, 3, 4, 5],
            [99, 99, 6, 7, 8, 9, 10]
        ])
        assert torch.equal(shifted_input_ids, expected_input_ids)
        
        # Attention mask should still be 0 (not affected by pad_token_id)
        expected_attention_mask = torch.tensor([
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1]
        ])
        assert torch.equal(shifted_attention_mask, expected_attention_mask)
    
    def test_shift_with_no_offsets(self, mock_inputs):
        """Test with no offsets provided (should return original)"""
        input_start_offsets = None
        
        shifted_input_ids, shifted_attention_mask = shift_inputs_by_offset(
            mock_inputs, input_start_offsets
        )
        
        # Should be no change from original
        assert torch.equal(shifted_input_ids, mock_inputs.input_ids)
        assert torch.equal(shifted_attention_mask, mock_inputs.attention_mask)


class TestPrepareInputsFromPrompt:
    @pytest.fixture
    def mock_neuron_model(self):
        """Create a mock neuron model with configurable batch size and chunked prefill settings"""
        model = Mock()
        config = Mock()
        neuron_config = Mock()
        
        # Default settings
        neuron_config.batch_size = 4
        neuron_config.is_chunked_prefill = False
        
        # Used when is_chunked_prefill is True
        neuron_config.chunked_prefill_config = Mock()
        neuron_config.chunked_prefill_config.max_num_seqs = 8
        
        config.neuron_config = neuron_config
        model.config = config
        
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer that returns predictable input_ids and attention_mask"""
        tokenizer = Mock()
        
        # Return mock tokenized inputs with the right structure
        def mock_tokenize(prompts, padding=None, return_tensors=None):
            batch_size = len(prompts)
            inputs = Mock()
            # Create input_ids tensor with shape [batch_size, 10]
            inputs.input_ids = torch.arange(1, 11).repeat(batch_size, 1)
            # Create attention_mask tensor with shape [batch_size, 10] (all 1s)
            inputs.attention_mask = torch.ones(batch_size, 10)
            return inputs
            
        tokenizer.side_effect = mock_tokenize
        tokenizer.__call__ = mock_tokenize
        
        return tokenizer
    
    def test_prepare_inputs_with_default_prompt(self, mock_neuron_model, mock_tokenizer, monkeypatch):
        """Test preparing inputs with the default TEST_PROMPT"""

        # Call the function without providing a prompt
        inputs = prepare_inputs_from_prompt(
            neuron_model=mock_neuron_model,
            tokenizer=mock_tokenizer
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        
        # Verify the correct batch size was used (default non-chunked batch_size)
        assert mock_tokenizer.call_count == 1
        args, kwargs = mock_tokenizer.call_args
        prompts = args[0]
        assert len(prompts) == mock_neuron_model.config.neuron_config.batch_size
        assert all(p == TEST_PROMPT for p in prompts)
        
        # Verify the returned tensors have the expected shape
        assert input_ids.shape[0] == mock_neuron_model.config.neuron_config.batch_size
        assert attention_mask.shape[0] == mock_neuron_model.config.neuron_config.batch_size
    
    def test_prepare_inputs_with_custom_prompt(self, mock_neuron_model, mock_tokenizer):
        """Test preparing inputs with a custom prompt"""
        custom_prompt = "This is a custom prompt."
        
        # Call the function with the custom prompt
        inputs = prepare_inputs_from_prompt(
            neuron_model=mock_neuron_model,
            tokenizer=mock_tokenizer,
            prompt=custom_prompt
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        
        # Verify the correct prompt was used
        args, kwargs = mock_tokenizer.call_args
        prompts = args[0]
        assert all(p == custom_prompt for p in prompts)
        
        # Verify the returned tensors have the expected shape
        assert input_ids.shape[0] == mock_neuron_model.config.neuron_config.batch_size
        assert attention_mask.shape[0] == mock_neuron_model.config.neuron_config.batch_size
    
    def test_prepare_inputs_with_chunked_prefill(self, mock_neuron_model, mock_tokenizer):
        """Test preparing inputs when chunked prefill is enabled"""
        # Enable chunked prefill
        mock_neuron_model.config.neuron_config.is_chunked_prefill = True
        custom_prompt = "This is a test prompt for chunked prefill."
        
        # Call the function
        inputs = prepare_inputs_from_prompt(
            neuron_model=mock_neuron_model,
            tokenizer=mock_tokenizer,
            prompt=custom_prompt
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        
        # Verify the correct batch size was used (max_num_seqs for chunked prefill)
        args, kwargs = mock_tokenizer.call_args
        prompts = args[0]
        assert len(prompts) == mock_neuron_model.config.neuron_config.chunked_prefill_config.max_num_seqs
        assert all(p == custom_prompt for p in prompts)
        
        # Verify the returned tensors have the expected shape
        assert input_ids.shape[0] == mock_neuron_model.config.neuron_config.chunked_prefill_config.max_num_seqs
        assert attention_mask.shape[0] == mock_neuron_model.config.neuron_config.chunked_prefill_config.max_num_seqs
    
    def test_prepare_inputs_without_tokenizer(self, mock_neuron_model):
        """Test that ValueError is raised when tokenizer is not provided"""
        with pytest.raises(ValueError, match="A tokenizer is required to prepare inputs"):
            prepare_inputs_from_prompt(
                neuron_model=mock_neuron_model,
                tokenizer=None
            )


class TestGenerateExpectedLogits:
    @pytest.fixture
    def mock_neuron_model(self):
        """Create a mock neuron model with needed configuration and methods"""
        model = Mock()
        
        # Configure model properties and config
        config = Mock()
        neuron_config = Mock()
        neuron_config.seq_len = 512
        neuron_config.speculation_length = 5
        neuron_config.enable_fused_speculation = False
        
        config.neuron_config = neuron_config
        model.config = config
        model.neuron_config = neuron_config
        model.model_path = "test/model/path"
        
        # Create a true Mock for the HF model
        mock_hf_model = Mock()
        
        # Create a Mock for the generate method that can track calls
        mock_generate = Mock()
        
        # Configure the generate method's return value
        def side_effect(**kwargs):
            batch_size = kwargs.get('input_ids').shape[0]
            vocab_size = 1000
            num_tokens = kwargs.get('max_new_tokens', 10)
            
            # Create scores as a list of tensors
            scores = [torch.rand(batch_size, vocab_size) for _ in range(num_tokens)]
            
            # Create sequences tensor
            sequences = torch.randint(0, vocab_size, (batch_size, 5 + num_tokens))
            
            # Create mock output object
            output = Mock()
            output.scores = scores
            output.sequences = sequences
            return output
            
        mock_generate.side_effect = side_effect
        mock_hf_model.generate = mock_generate
        
        # Set load_hf_model as a Mock that returns our mock_hf_model
        model.load_hf_model = Mock(return_value=mock_hf_model)
        
        # Make sure the model is not recognized as NeuronMllamaForCausalLM
        model.__class__.__name__ = "NeuronLlama4ForCausalLM"
        
        return model

    @pytest.fixture
    def mock_generation_config(self):
        """Create a mock generation config"""
        return GenerationConfig()
    
    @pytest.fixture
    def mock_mllama_model(self, mock_neuron_model):
        """Create a mock model that will be detected as NeuronMllamaForCausalLM"""
        # Register NeuronMllamaForCausalLM as a real class so isinstance checks work
        mock_neuron_model.__class__.__name__ = "NeuronMllamaForCausalLM"
        # Make isinstance return True for NeuronMllamaForCausalLM
        with patch('neuronx_distributed_inference.utils.accuracy.isinstance', return_value=True):
            yield mock_neuron_model
    
    def test_generate_logits_with_speculation(self, mock_neuron_model, mock_generation_config):
        """Test that speculation settings are properly used"""
        # Enable fused speculation
        mock_neuron_model.neuron_config.enable_fused_speculation = True
        
        # Prepare test inputs
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        num_tokens = 15
        
        # Call the function
        generate_expected_logits(
            neuron_model=mock_neuron_model,
            input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=mock_generation_config,
            num_tokens=num_tokens
        )
        
        # Verify that prompt_lookup_num_tokens was set in generation_config
        assert mock_generation_config.prompt_lookup_num_tokens == mock_neuron_model.neuron_config.speculation_length
    
    def test_generate_with_additional_input_args(self, mock_neuron_model, mock_generation_config):
        """Test that additional input arguments are passed to the generate method"""
        # Prepare test inputs
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        num_tokens = 10
        
        # Additional arguments to be passed
        additional_input_args = {
            "custom_arg1": "test_value",
            "custom_arg2": 42,
            # Override input_ids as a test
            "input_ids": torch.zeros(batch_size, seq_len)
        }
        
        # Get the mock HF model and reset side_effect to track arguments
        mock_hf_instance = mock_neuron_model.load_hf_model.return_value
        
        # Define a side effect that can capture arguments
        captured_args = {}
        def capture_args_side_effect(**kwargs):
            # Save all kwargs for later inspection
            nonlocal captured_args
            captured_args = kwargs.copy()
            
            # Return expected shape
            scores = [torch.rand(batch_size, 1000) for _ in range(num_tokens)]
            output = Mock()
            output.scores = scores
            return output
        
        # Replace side_effect to capture args
        mock_hf_instance.generate.side_effect = capture_args_side_effect
        
        # Call the function
        generate_expected_logits(
            neuron_model=mock_neuron_model,
            input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=mock_generation_config,
            num_tokens=num_tokens,
            additional_input_args=additional_input_args
        )
        
        # Verify that additional args were passed to generate
        assert "custom_arg1" in captured_args
        assert captured_args["custom_arg1"] == "test_value"
        assert "custom_arg2" in captured_args
        assert captured_args["custom_arg2"] == 42
        # Check that input_ids was overridden (need to check if it's all zeros)
        assert torch.all(captured_args["input_ids"] == 0)
    
    def test_generate_with_tokenizer(self, mock_neuron_model, mock_generation_config):
        """Test generation with tokenizer for logging purposes"""
        # Prepare test inputs
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        num_tokens = 10
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.batch_decode.return_value = ["Generated text 1", "Generated text 2"]
        
        # Call the function
        generate_expected_logits(
            neuron_model=mock_neuron_model,
            input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=mock_generation_config,
            num_tokens=num_tokens,
            tokenizer=mock_tokenizer
        )
        
        # Verify tokenizer was used to decode and log output
        assert mock_tokenizer.batch_decode.call_count == 1
    
    def test_mllama_raises_error(self, mock_mllama_model, mock_generation_config):
        """Test that ValueError is raised when using NeuronMllamaForCausalLM model"""

        # Prepare test inputs
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Call the function and expect ValueError
        with pytest.raises(ValueError, match="Mllama does not support generating goldens with HF on CPU"):
            generate_expected_logits(
                neuron_model=mock_mllama_model,
                input_ids=input_ids,
                inputs_attention_mask=attention_mask,
                generation_config=mock_generation_config
            )


class TestCheckAccuracyLogitsV2:
    """
    Tests for the check_accuracy_logits_v2 function which validates logits output from Neuron models.
    The tests are based on the example in the function's docstring, particularly focusing on
    multimodal use cases (image + text).
    """
    
    @pytest.fixture
    def mock_neuron_model(self):
        """Create a mock neuron model for testing"""
        model = Mock()
        
        # Configure model properties and config
        config = Mock()
        neuron_config = Mock()
        neuron_config.seq_len = 100
        neuron_config.speculation_length = 4
        neuron_config.enable_fused_speculation = False
        neuron_config.is_chunked_prefill = False
        neuron_config.on_device_sampling_config = None
        neuron_config.max_context_length = 80
        neuron_config.output_logits = True
        
        config.neuron_config = neuron_config
        model.config = config
        model.neuron_config = neuron_config
        
        # Mock the HuggingFaceGenerationAdapter with a proper return value
        def mock_adapter_setup(*args, **kwargs):
            adapter = Mock()
            
            # Configure generate method to return proper outputs
            def mock_generate(**kwargs):
                batch_size = kwargs.get('input_ids', torch.zeros(2, 10)).shape[0]
                vocab_size = 30522  # Standard BERT vocab size for testing
                num_tokens = kwargs.get('max_new_tokens', 5)
                
                # Create scores as a list of tensors
                scores = [torch.rand(batch_size, vocab_size) for _ in range(num_tokens)]
                
                # Create sequences tensor
                sequences = torch.randint(0, vocab_size, (batch_size, 10 + num_tokens))
                
                # Create mock output object
                output = Mock()
                output.scores = scores
                output.sequences = sequences
                return output
                
            adapter.generate = mock_generate
            return adapter
            
        # Patch the HuggingFaceGenerationAdapter constructor
        with patch('neuronx_distributed_inference.utils.accuracy.HuggingFaceGenerationAdapter', 
                  side_effect=mock_adapter_setup):
            yield model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing"""
        tokenizer = Mock()
        tokenizer.batch_decode.return_value = ["Generated text 1", "Generated text 2"]
        tokenizer.pad_token_id = 0
        return tokenizer
    
    @pytest.fixture
    def expected_logits(self):
        """Create expected logits tensor for comparison"""
        batch_size = 2
        num_tokens = 5
        vocab_size = 30522
        return torch.rand(num_tokens, batch_size, vocab_size)
    
    @pytest.fixture
    def input_tensors(self):
        """Create input tensors for testing"""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        return input_ids, attention_mask
    
    @pytest.fixture
    def generation_config(self):
        """Create a generation config for testing"""
        return GenerationConfig()
    
    @patch('neuronx_distributed_inference.utils.accuracy.logit_validation')
    def test_check_accuracy_logits_v2_basic(self, mock_logit_validation, mock_neuron_model, 
                                        mock_tokenizer, expected_logits, input_tensors, generation_config):
        """Test the basic functionality of check_accuracy_logits_v2"""
        # Configure mock logit_validation to return success
        mock_logit_validation.return_value = (True, {"max_error": 0.001}, "Validation passed")
        
        # Get input tensors
        input_ids, attention_mask = input_tensors
        
        # Call the function
        results = check_accuracy_logits_v2(
            neuron_model=mock_neuron_model,
            expected_logits=expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            tokenizer=mock_tokenizer
        )
        
        # Verify the logit_validation was called with the right parameters
        assert mock_logit_validation.called
        call_args = mock_logit_validation.call_args[1]
        assert torch.equal(call_args['input_ids'], input_ids)
        assert torch.is_tensor(call_args['expected_logits'])
        assert call_args['divergence_difference_tol'] == 0.001  # Default value
        
        # Check that the function returns the validation results
        assert results == {"max_error": 0.001}
    
    @patch('neuronx_distributed_inference.utils.accuracy.logit_validation')
    def test_with_multimodal_inputs(self, mock_logit_validation, mock_neuron_model, expected_logits, 
                                input_tensors, generation_config):
        """Test with additional input arguments for multimodal models as shown in the docstring example"""
        # Configure mock logit_validation to return success
        mock_logit_validation.return_value = (True, {"max_error": 0.001}, "Validation passed")
        
        # Get input tensors
        input_ids, attention_mask = input_tensors
        batch_size = input_ids.shape[0]
        
        # Create multimodal additional inputs (similar to example in docstring)
        pixel_values = torch.rand(batch_size, 3, 224, 224)  # Typical image shape
        vision_mask = torch.ones(batch_size, 1).to(torch.bool)  # Vision mask
        
        additional_input_args = {
            "pixel_values": pixel_values,
            "vision_mask": vision_mask,
        }
        
        # Call the function
        check_accuracy_logits_v2(
            neuron_model=mock_neuron_model,
            expected_logits=expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            additional_input_args=additional_input_args
        )
        
        # Verify that generate was called with the additional_input_args
        # This is checked via the generate_fn passed to logit_validation
        assert mock_logit_validation.called
        
        # Extract the generate_fn passed to logit_validation
        generate_fn = mock_logit_validation.call_args[1]['generate_fn']
        
        # Create a small input to test if generate_fn uses the additional args
        test_input_ids = torch.zeros(1, 5, dtype=torch.long)
        # This would raise an error if additional_input_args weren't correctly passed
        generate_fn(test_input_ids)
    
    @patch('neuronx_distributed_inference.utils.accuracy.logit_validation')
    def test_with_custom_divergence_tol(self, mock_logit_validation, mock_neuron_model, 
                                    expected_logits, input_tensors, generation_config):
        """Test with custom divergence tolerance value"""
        # Configure mock logit_validation to return success
        mock_logit_validation.return_value = (True, {"max_error": 0.005}, "Validation passed")
        
        # Get input tensors
        input_ids, attention_mask = input_tensors
        
        # Custom tolerance value
        custom_tol = 0.005
        
        # Call the function with custom tolerance
        results = check_accuracy_logits_v2(
            neuron_model=mock_neuron_model,
            expected_logits=expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            divergence_difference_tol=custom_tol
        )
        
        # Verify the logit_validation was called with the custom tolerance
        assert mock_logit_validation.called
        assert mock_logit_validation.call_args[1]['divergence_difference_tol'] == custom_tol
        
        # Check that the function returns the validation results
        assert results == {"max_error": 0.005}
    
    @patch('neuronx_distributed_inference.utils.accuracy.logit_validation')
    def test_with_token_count_limit(self, mock_logit_validation, mock_neuron_model, 
                                input_tensors, generation_config):
        """Test with limited token count"""
        # Configure mock logit_validation to return success
        mock_logit_validation.return_value = (True, {"max_error": 0.001}, "Validation passed")
        
        # Get input tensors
        input_ids, attention_mask = input_tensors
        batch_size = input_ids.shape[0]
        
        # Create expected logits with specific token count
        num_tokens = 3  # Limit to just 3 tokens
        vocab_size = 30522
        limited_expected_logits = torch.rand(num_tokens, batch_size, vocab_size)
        
        # Call the function with the limited token count
        check_accuracy_logits_v2(
            neuron_model=mock_neuron_model,
            expected_logits=limited_expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            num_tokens_to_check=num_tokens
        )
        
        # Verify the logit_validation was called with the expected logits
        assert mock_logit_validation.called
        call_args = mock_logit_validation.call_args[1]
        assert call_args['expected_logits'].shape[0] == num_tokens
    
    @patch('neuronx_distributed_inference.utils.accuracy.logit_validation')
    def test_with_chunked_prefill(self, mock_logit_validation, mock_neuron_model, 
                              expected_logits, input_tensors, generation_config, mock_tokenizer):
        """Test when chunked prefill is enabled"""
        # Configure the model for chunked prefill
        mock_neuron_model.config.neuron_config.is_chunked_prefill = True
        
        # Get input tensors
        input_ids, attention_mask = input_tensors
        
        # Configure generate_with_chunked_prefill to return appropriate logits
        with patch('neuronx_distributed_inference.utils.accuracy.generate_with_chunked_prefill') as mock_chunked_prefill:
            # Set up the mock to return a tensor with the right shape
            mock_chunked_prefill.return_value = torch.rand_like(expected_logits)
            
            # Configure mock logit_validation to capture and call generate_fn
            def logit_validation_side_effect(input_ids, generate_fn, expected_logits, **kwargs):
                # Call the generate_fn with test inputs to trigger generate_with_chunked_prefill
                generate_fn(input_ids)
                return True, {"max_error": 0.001}, "Validation passed"
                
            mock_logit_validation.side_effect = logit_validation_side_effect
            
            # Call the function
            check_accuracy_logits_v2(
                neuron_model=mock_neuron_model,
                expected_logits=expected_logits,
                inputs_input_ids=input_ids,
                inputs_attention_mask=attention_mask,
                generation_config=generation_config,
                tokenizer=mock_tokenizer
            )
            
            # Verify generate_with_chunked_prefill was called
            assert mock_chunked_prefill.called
            # Verify it was called with the right parameters
            assert mock_chunked_prefill.call_args[0][0] == mock_neuron_model
            assert mock_chunked_prefill.call_args[0][1] == mock_tokenizer
            # The first argument passed to the generate_fn should be input_ids
            assert torch.equal(mock_chunked_prefill.call_args[0][2], input_ids)
    
    @patch('neuronx_distributed_inference.utils.accuracy.logit_validation')
    def test_with_validation_failure(self, mock_logit_validation, mock_neuron_model, 
                                 expected_logits, input_tensors, generation_config):
        """Test handling of validation failures"""
        # Configure mock logit_validation to indicate failure
        error_message = "Validation failed: maximum error exceeds tolerance"
        results = {"max_error": 0.1, "position": 2}
        mock_logit_validation.return_value = (False, results, error_message)
        
        # Get input tensors
        input_ids, attention_mask = input_tensors
        
        # Call the function and expect LogitMatchingValidationError
        with pytest.raises(LogitMatchingValidationError) as exc_info:
            check_accuracy_logits_v2(
                neuron_model=mock_neuron_model,
                expected_logits=expected_logits,
                inputs_input_ids=input_ids,
                inputs_attention_mask=attention_mask,
                generation_config=generation_config
            )
        
        # Verify the exception contains the right message and results
        assert error_message in str(exc_info.value)
        assert exc_info.value.results == results
    
    @patch('neuronx_distributed_inference.utils.accuracy.logit_validation')
    def test_with_on_device_sampling_no_output_logits(self, mock_logit_validation, mock_neuron_model,
                                                  expected_logits, input_tensors, generation_config):
        """Test that assertion error is raised if on_device_sampling is enabled without output_logits"""
        # Configure the model for on-device sampling but disable output_logits
        mock_neuron_model.neuron_config.on_device_sampling_config = Mock()
        mock_neuron_model.neuron_config.output_logits = False
        
        # Get input tensors
        input_ids, attention_mask = input_tensors
        
        # Call the function and expect assertion error
        with pytest.raises(AssertionError, match="output_logits is required to enable logit validation with on-device sampling"):
            check_accuracy_logits_v2(
                neuron_model=mock_neuron_model,
                expected_logits=expected_logits,
                inputs_input_ids=input_ids,
                inputs_attention_mask=attention_mask,
                generation_config=generation_config
            )
        
        # Verify logit_validation was not called
        assert not mock_logit_validation.called
        
    @patch('neuronx_distributed_inference.utils.accuracy.logit_validation')
    def test_with_fused_speculation(self, mock_logit_validation, mock_neuron_model,
                                expected_logits, input_tensors, generation_config):
        """Test that when enable_fused_speculation is True, prompt_lookup_num_tokens is set to speculation_length"""
        # Configure mock logit_validation to return success
        mock_logit_validation.return_value = (True, {"max_error": 0.001}, "Validation passed")
        
        # Enable fused speculation and set a specific speculation length
        mock_neuron_model.neuron_config.enable_fused_speculation = True
        mock_neuron_model.neuron_config.speculation_length = 8  # Use a distinct value for testing
        
        # Get input tensors
        input_ids, attention_mask = input_tensors
        
        # Call the function
        check_accuracy_logits_v2(
            neuron_model=mock_neuron_model,
            expected_logits=expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config
        )
        
        # Verify that prompt_lookup_num_tokens was set to speculation_length
        assert generation_config.prompt_lookup_num_tokens == mock_neuron_model.neuron_config.speculation_length
        assert generation_config.prompt_lookup_num_tokens == 8
