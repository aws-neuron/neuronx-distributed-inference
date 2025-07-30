"""
This is a temporary file to get the testing running for new package.

Some of the utitlies functions need to be redo or removed.
"""

# flake8: noqa

import warnings
from contextlib import nullcontext
from typing import List, Optional, Union
import math

import torch
from torch_neuronx.testing.validation import custom_allclose, logit_validation
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.mllama.utils import create_vision_mask, get_image_tensors
from neuronx_distributed_inference.models.mllama.modeling_mllama import NeuronMllamaForCausalLM
from neuronx_distributed_inference.models.llama4.modeling_llama4_text import NeuronLlama4TextForCausalLM
from neuronx_distributed_inference.models.llama4.modeling_llama4 import NeuronLlama4ForCausalLM
from neuronx_distributed_inference.models.llama4.utils.input_processor import prepare_generation_inputs_hf as llama4_prepare_generation_inputs_hf
from neuronx_distributed_inference.utils.constants import *
from neuronx_distributed_inference.utils.exceptions import LogitMatchingValidationError
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    warnings.warn(
        "Intel extension for pytorch not found. For faster CPU references install `intel-extension-for-pytorch`.",
        category=UserWarning,
    )
    ipex = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn(
        "matplotlib not found. Install via `pip install matplotlib`.",
        category=UserWarning,
    )
    matplotlib = None
    plt = None

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_accuracy_embeddings(
    actual_output: torch.Tensor,
    expected_output: torch.Tensor,
    plot_outputs: bool = False,
    rtol: float = 0.0,
    atol: float = 0.0,
):
    assert (
        expected_output.dtype == actual_output.dtype
    ), f"dtypes {expected_output.dtype} and {actual_output.dtype} does not match!"
    dtype = expected_output.dtype

    # Set default rtol, atol based on dtype if not provided
    if not rtol:
        if dtype == torch.bfloat16:
            rtol = 0.05
        elif dtype == torch.float32:
            rtol = 0.01
        else:
            NotImplementedError(f"Specify rtol for dtype {dtype}")
    logger.info(f"Using rtol = {rtol} for dtype {dtype}")
    if not atol:
        atol = 1e-5
    logger.info(f"Using atol = {atol}")

    if plot_outputs and matplotlib and plt:
        # Save plot, expecting a y=x straight line
        matplotlib.rcParams["agg.path.chunksize"] = 10000
        matplotlib.rcParams["path.simplify_threshold"] = 1.0
        plt.scatter(
            actual_output.float().detach().numpy().reshape(-1),
            expected_output.float().detach().numpy().reshape(-1),
            s=1,
        )
        plt.xlabel("Actual Output")
        plt.ylabel("Expected Output")
        plot_path = "plot.png"
        plt.savefig(plot_path, format="png")
        logger.info(f"Saved outputs plot to {plot_path}.")

    # NxD logit validation tests uses this method
    # equivalent to torch.allclose except rtol is multiplied by absolute max, not abs
    # this matches the behavior of the compiler's birsim-to-xla_infergoldens verification
    passed, max_err = custom_allclose(expected_output, actual_output, atol=atol, rtol=rtol)
    logger.info(f"Embeddings passed accuracy validation: {passed}, max_err: {max_err}")
    return passed, max_err


def get_generate_outputs_from_token_ids(
    model,
    token_ids,
    tokenizer,
    attention_mask=None,
    is_hf=False,
    draft_model=None,
    input_capture_hook=None,
    input_start_offsets=None,
    **generate_kwargs,
):
    if not is_hf:
        # Update generation kwargs to run Neuron model.
        if draft_model is not None:
            draft_generation_model = HuggingFaceGenerationAdapter(draft_model)
            draft_generation_model.generation_config.update(
                num_assistant_tokens=model.neuron_config.speculation_length
            )

            generate_kwargs.update(
                {
                    "assistant_model": draft_generation_model,
                    "do_sample": False,
                }
            )
        elif model.neuron_config.enable_fused_speculation:
            generate_kwargs.update(
                {
                    "prompt_lookup_num_tokens": model.neuron_config.speculation_length,
                }
            )
            if not model.neuron_config.enable_eagle_speculation:
                generate_kwargs.update(
                    {
                        "do_sample": False,
                    }
                )

    # If an attention mask is provided, the inputs are also expected to be padded to the correct shape.
    if attention_mask is None:
        logger.info("attention mask not provided, padding inputs and generating a mask")

        tokenizer.pad_token_id = tokenizer.eos_token_id

        padding_side = "left" if is_hf else "right"
        inputs = tokenizer.pad(
            {"input_ids": token_ids},
            padding_side=padding_side,
            return_attention_mask=True,
            return_tensors="pt",
        )

        token_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        attention_mask[token_ids == tokenizer.pad_token_id] = 0
    generation_model = model if is_hf else HuggingFaceGenerationAdapter(model, input_start_offsets)
    token_ids = _shift_tensors_by_offset(input_start_offsets, token_ids, tokenizer.pad_token_id)
    attention_mask = _shift_tensors_by_offset(input_start_offsets, attention_mask, 0)

    outputs = generation_model.generate(
        token_ids,
        attention_mask=attention_mask,
        input_capture_hook=input_capture_hook,
        **generate_kwargs,
    )
    if not is_hf:
        model.reset()
        if draft_model is not None:
            draft_model.reset()

    if isinstance(outputs, SampleOutput.__args__):
        # Get token ids from output when return_dict_in_generate=True
        output_ids = outputs.sequences
    else:
        output_ids = outputs
    output_tokens = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return outputs, output_tokens


def get_generate_outputs(
    model,
    prompts,
    tokenizer,
    is_hf=False,
    draft_model=None,
    device="neuron",
    input_capture_hook=None,
    input_start_offsets=None,
    **generate_kwargs,
):
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_hf:
        tokenizer.padding_side = "left"
    else:
        # FIXME: add cpu generation
        if device == "cpu":
            assert "get_generate_outputs from CPU yet avaialble"
        tokenizer.padding_side = "right"

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    is_bfloat16 = (
        model.dtype == torch.bfloat16
        if is_hf
        else model.config.neuron_config.torch_dtype == torch.bfloat16
    )
    use_ipex = ipex and is_bfloat16
    if use_ipex:
        model = ipex.optimize(model, dtype=model.config.torch_dtype)
        model = torch.compile(model, backend="ipex")

    with torch.cpu.amp.autocast() if use_ipex else nullcontext():
        return get_generate_outputs_from_token_ids(
            model,
            inputs.input_ids,
            tokenizer,
            attention_mask=inputs.attention_mask,
            is_hf=is_hf,
            draft_model=draft_model,
            input_capture_hook=input_capture_hook,
            input_start_offsets=input_start_offsets,
            **generate_kwargs,
        )


# FIXME: add on cpu check support
def check_accuracy(
    neuron_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: Optional[GenerationConfig] = None,
    expected_token_ids: Optional[List] = None,
    num_tokens_to_check: int = None,
    do_sample: bool = False,
    draft_model: PreTrainedModel = None,
    prompt: Optional[str] = None,
    image=None,
    input_start_offsets: List[int] = None,
    execution_mode: str = "config",
):
    """
    Function to compare outputs from huggingface model and neuronx NxD model
    """
    neuron_config = neuron_model.neuron_config
    generation_kwargs = {
        "do_sample": do_sample,
        "max_length": neuron_config.max_length,
    }

    logger.info(
        f"run accuracy check with generation_config as: {generation_kwargs} and execution_mode={execution_mode}"
    )
    if prompt is None:
        prompts = [TEST_PROMPT] * neuron_config.batch_size
    else:
        prompts = [prompt] * neuron_config.batch_size

    # FIXME: add image support
    if hasattr(expected_token_ids, "sequences"):
        expected_token_ids = expected_token_ids.sequences
    if expected_token_ids is not None:
        outputs_expected = tokenizer.batch_decode(
            expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    else:
        # Generate goldens with HF on CPU
        hf_model = neuron_model.load_hf_model(neuron_model.model_path)
        expected_token_ids, outputs_expected = get_generate_outputs(
            hf_model,
            prompts,
            tokenizer,
            is_hf=True,
            generation_config=generation_config,
            input_start_offsets=input_start_offsets,
            **generation_kwargs,
        )

    logger.info(f"Expected output: {outputs_expected}")
    mode_being_tested = "async mode" if neuron_model.neuron_config.async_mode else "sync mode"
    output_token_ids, outputs_actual = get_generate_outputs(
        neuron_model,
        prompts,
        tokenizer,
        is_hf=False,
        draft_model=draft_model,
        generation_config=generation_config,
        input_start_offsets=input_start_offsets,
        **generation_kwargs,
    )

    logger.info(f"Actual output  : {outputs_actual}")
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) if tokenizer else 0

    # Process each batch element separately to maintain the 2D structure
    expected_id_list = []
    actual_id_list = []
    bs, _ = output_token_ids.shape
    for i in range(bs):
        expected_seq = expected_token_ids[i]
        expected_seq = expected_seq[expected_seq != pad_token_id]
        actual_seq = output_token_ids[i]
        actual_seq = actual_seq[actual_seq != pad_token_id]
        if num_tokens_to_check:
            expected_seq = expected_seq[:num_tokens_to_check]
            actual_seq = actual_seq[:num_tokens_to_check]
        expected_id_list.append(expected_seq)
        actual_id_list.append(actual_seq)

    expected_token_ids = torch.stack(expected_id_list)
    output_token_ids = torch.stack(actual_id_list)


    if draft_model is not None or neuron_config.enable_fused_speculation:
        # Handle corner scenario where last few tokens are not generated as part of speculation.
        assert (
            abs(expected_token_ids.shape[-1] - output_token_ids.shape[-1])
            <= neuron_config.speculation_length
        ), "Unexpected number of tokens generated by target model"
        tokens_to_compare = min(expected_token_ids.shape[-1], output_token_ids.shape[-1])
        expected_token_ids = expected_token_ids[:tokens_to_compare]
        output_token_ids = output_token_ids[:tokens_to_compare]

    device = "neuron"
    assert torch.equal(
        output_token_ids, expected_token_ids
    ), f"\nActual: ({device}) {output_token_ids} \nExpected (hf-cpu): {expected_token_ids}"
    logger.info(f"The output from Neuronx NxD on {device} using {mode_being_tested} is accurate!")


def prepare_inputs_from_prompt(
        neuron_model: NeuronApplicationBase,
        tokenizer: PreTrainedTokenizer,
        prompt: str = None,
):
    """
    Prepares test inputs by tokenizing the provided prompt string for model inference.

    Args:
        neuron_model (NeuronApplicationBase): The neuron model application instance
        tokenizer (PreTrainedTokenizer): Tokenizer to convert text to tokens.
            Must be provided or will raise ValueError.
        prompt (str, optional): Input prompt string to be tokenized.
            Defaults to None, in which case TEST_PROMPT will be used.

    Returns:
        tokenizer outputs

    Raises:
        ValueError: If tokenizer is not provided
    """
    if tokenizer is None:
        raise ValueError("A tokenizer is required to prepare inputs")

    if prompt is None:
        prompt = TEST_PROMPT

    if neuron_model.config.neuron_config.is_chunked_prefill:
        # The actual batch size is stored as max_num_seqs
        prompts = [prompt] * neuron_model.config.neuron_config.chunked_prefill_config.max_num_seqs
    else:
        prompts = [prompt] * neuron_model.config.neuron_config.batch_size

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    return inputs


def generate_expected_logits(
    neuron_model: NeuronApplicationBase,
    input_ids,
    inputs_attention_mask,
    generation_config: GenerationConfig,
    num_tokens: int = None,
    additional_input_args=None,
    tokenizer=None,
):
    """
    Generates expected logits using the HuggingFace model on CPU for validation purposes.

    This function generates logits using the original HuggingFace model to create golden
    references for comparison with Neuron model outputs. It supports both standard and
    multimodal generation cases through additional_input_args.

    Args:
        neuron_model (NeuronApplicationBase): The neuron model application instance
        input_ids: Input token IDs for the model
        inputs_attention_mask: Attention mask corresponding to the input tokens
        generation_config (GenerationConfig): Configuration for HuggingFace LLM generation.
        num_tokens (int, optional): Number of tokens to generate. If None, generates
            maximum possible tokens based on model's sequence length. Defaults to None.
        additional_input_args (dict, optional): Additional generation parameters that can
            extend or override the basic parameters. Can be used to:
            - Override existing parameters (e.g., input_ids)
            - Add vision-related parameters for multimodal cases
            Defaults to None.
        tokenizer (PreTrainedTokenizer, optional): Tokenizer for decoding and logging
            generated tokens. Only used for logging purposes and is not required for
            core functionality. Defaults to None.

    Returns:
        torch.Tensor: Expected logits tensor with shape [num_tokens, batch_size, vocab_size]

    Raises:
        ValueError: If trying to generate goldens with Mllama model on CPU

    Notes:
        - The function adjusts generation length based on speculation settings if enabled
        - Uses greedy sampling for logit validation
    """
    if additional_input_args is None:
        additional_input_args = {}

    if isinstance(neuron_model, NeuronMllamaForCausalLM):
        raise ValueError("Mllama does not support generating goldens with HF on CPU. Please generate logits separately.")

    if neuron_model.neuron_config.enable_fused_speculation:
        generation_config.prompt_lookup_num_tokens = neuron_model.neuron_config.speculation_length

    max_new_tokens = neuron_model.config.neuron_config.seq_len - input_ids.shape[1]
    if num_tokens is None:
        num_tokens = max_new_tokens
    else:
        num_tokens = min(max_new_tokens, num_tokens)
    spec_len = neuron_model.config.neuron_config.speculation_length
    if spec_len > 0:
        # With speculation, generation stops (spec_len - 1) tokens early.
        num_tokens -= spec_len - 1

    # Generate goldens with HF on CPU
    # Create base parameters dictionary
    input_args = {
        "input_ids": input_ids,
        "attention_mask": inputs_attention_mask,
        "max_new_tokens": num_tokens,
        "min_new_tokens": num_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "generation_config": generation_config,
    }
    # Update input_args with additional_neuron_input_args
    # This will override any existing keys and add new ones
    input_args.update(additional_input_args)
    # logit_validation assumes greedy sampling
    hf_model = neuron_model.load_hf_model(neuron_model.model_path)

    outputs = hf_model.generate(**input_args)

    # Stack the scores and trim to the required number of tokens
    expected_logits = torch.stack(outputs.scores)[:num_tokens, :, :]
    expected_token_ids = expected_logits.argmax(dim=2).T
    if tokenizer is not None:
        expected_tokens = tokenizer.batch_decode(
            expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.info(f"Expected Output(tokens with the max logits): {expected_tokens}")
    logger.info(f"Expected Output (token ids with the max logits): {expected_token_ids}")
    logger.info(f"Expected Logits Shape: {expected_logits.shape}")

    return expected_logits


def check_accuracy_logits(
    neuron_model: NeuronApplicationBase,
    tokenizer: PreTrainedTokenizer = None,
    generation_config: GenerationConfig = None,
    prompt: str = None,
    expected_logits: torch.Tensor = None,
    divergence_difference_tol: float = 0.001,
    tol_map: dict = None,
    num_tokens_to_check: int = None,
    execution_mode="config",
    draft_model: NeuronApplicationBase = None,
    image=None,
    num_image_per_prompt=1,
    inputs=None,
    input_start_offsets=None,
    pad_token_id=0,
    image_processor=None,
):
    """
    DEPRECATED: Please use the function check_accuracy_logits_v2 instead.
    """
    warning_message = (
        "'check_accuracy_logits' is deprecated and replaced by 'check_accuracy_logits_v2'. "
        "In a future release, this function will be removed."
    )
    warnings.warn(warning_message, category=DeprecationWarning)

    if neuron_model.neuron_config.on_device_sampling_config is not None:
        # should output both tokens and logits for logit matching check
        assert (
            neuron_model.neuron_config.output_logits
        ), "output_logits is required to enable logit validation with on-device sampling"

    if neuron_model.neuron_config.enable_fused_speculation:
        generation_config.prompt_lookup_num_tokens = neuron_model.neuron_config.speculation_length

    if inputs is None and tokenizer is None:
        raise ValueError(
            "Must provide either a tokenizer or inputs that include input_ids and attention_mask"
        )

    is_chunked_prefill = neuron_model.config.neuron_config.is_chunked_prefill

    if inputs is None:
        if prompt is None:
            prompt = MM_TEST_PROMPT if image else TEST_PROMPT

        if is_chunked_prefill:
            # The actual batch size is stored as max_num_seqs
            prompts = [prompt] * neuron_model.config.neuron_config.chunked_prefill_config.max_num_seqs
        else:
            prompts = [prompt] * neuron_model.config.neuron_config.batch_size

        inputs = tokenizer(prompts, padding=True, return_tensors="pt")


    initial_input_ids = inputs.input_ids
    initial_attention_mask = inputs.attention_mask
    pad_token_id = tokenizer.pad_token_id if tokenizer else pad_token_id
    initial_input_ids = _shift_tensors_by_offset(input_start_offsets, initial_input_ids, pad_token_id)
    initial_attention_mask = _shift_tensors_by_offset(input_start_offsets, initial_attention_mask, 0)
    initial_input_len = initial_input_ids.shape[1]
    seq_len = neuron_model.config.neuron_config.seq_len
    max_new_tokens = seq_len - initial_input_len
    if num_tokens_to_check is None:
        num_tokens_to_check = max_new_tokens
    else:
        num_tokens_to_check = min(max_new_tokens, num_tokens_to_check)
    spec_len = neuron_model.config.neuron_config.speculation_length
    if spec_len > 0:
        # With speculation, generation stops (spec_len - 1) tokens early.
        num_tokens_to_check -= spec_len - 1
    if (
        initial_input_len + num_tokens_to_check
        > neuron_model.config.neuron_config.max_context_length
    ):
        warnings.warn(
            (
                "input_len + num_tokens_to_check exceeds max_context_length. "
                "If output divergences at an index greater than max_context_length, "
                "a ValueError will occur because the next input len exceeds max_context_length. "
                "To avoid this, set num_tokens_to_check to a value of max_context_length - input_len or less."
            ),
            category=UserWarning,
        )
    # Prepare vision input args
    # TODO: use image processor to process vision inputs and avoid conditions based on model type
    if isinstance(neuron_model, NeuronMllamaForCausalLM):
        neuron_vision_input_args, hf_vision_input_args = _prepare_mllama_vision_args(neuron_model, tokenizer, prompt, image,\
                                                                                     num_image_per_prompt, inputs, image_processor, initial_input_ids)

    elif isinstance(neuron_model, NeuronLlama4ForCausalLM) and image is not None:
        neuron_vision_input_args, hf_vision_input_args = _prepare_llama4_vision_args(neuron_model, prompt,
                                                                                     [image] * num_image_per_prompt, \
                                                                                     inputs, image_processor)

    else:
        neuron_vision_input_args, hf_vision_input_args = {}, {}

    if expected_logits is None:
        # Generate goldens with HF on CPU
        # logit_validation assumes greedy sampling
        hf_model = neuron_model.load_hf_model(neuron_model.model_path)
        outputs = hf_model.generate(
            inputs.input_ids,
            max_new_tokens=num_tokens_to_check,
            min_new_tokens=num_tokens_to_check,
            do_sample=False,
            attention_mask=inputs.attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=generation_config,
            **hf_vision_input_args,
        )
        expected_logits = torch.stack(outputs.scores)
    expected_logits = expected_logits[:num_tokens_to_check, :, :]
    expected_token_ids = expected_logits.argmax(dim=2).T
    if tokenizer is not None:
        expected_tokens = tokenizer.batch_decode(
            expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.info(f"Expected Output: {expected_tokens} {expected_token_ids}")
    else:
        logger.info(f"Expected Output: {expected_token_ids}")
    logger.info(f"Expected Logits Shape: {expected_logits.shape}")

    model = HuggingFaceGenerationAdapter(neuron_model, input_start_offsets=input_start_offsets)
    expected_attention_mask = torch.ones(
        (
            initial_attention_mask.shape[0],
            expected_token_ids.shape[1],
        ),
        dtype=torch.int32,
    )
    extrapolated_attention_mask = torch.cat(
        (initial_attention_mask, expected_attention_mask), dim=1
    )

    def generate_fn_base(input_ids):
        input_length = input_ids.shape[1]
        attention_mask = extrapolated_attention_mask[:, :input_length]
        new_tokens = num_tokens_to_check + initial_input_len - input_length
        if spec_len > 0:
            # With speculation, generation stops (spec_len - 1) tokens early.
            new_tokens += spec_len - 1

        # use the input_ids and attention_mask generated from the prompt in _prepare_XXX_vision_args that include vision tokens
        if "input_ids" in neuron_vision_input_args and "attention_mask" in neuron_vision_input_args:
            input_ids = neuron_vision_input_args.pop("input_ids")
            attention_mask = neuron_vision_input_args.pop("attention_mask")

        model_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=generation_config,
            **neuron_vision_input_args,
        )

        actual_logits = torch.stack(model_outputs.scores)
        # TODO: Remove this workaround once lm_head de-padding has been implemented properly.
        if isinstance(neuron_model, NeuronLlama4ForCausalLM) or isinstance(neuron_model, NeuronLlama4TextForCausalLM):
            vocab_size = expected_logits.shape[-1]
            actual_logits = actual_logits[:, :, :vocab_size]
            
        actual_token_ids = actual_logits.argmax(dim=2).T
        if tokenizer is not None:
            actual_tokens = tokenizer.batch_decode(
                actual_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            logger.info(f"Actual Output: {actual_tokens} {actual_token_ids}")
        else:
            logger.info(f"Actual Output: {actual_token_ids}")
        logger.info(f"Actual Logits Shape: {actual_logits.shape}")
        return actual_logits

    def generate_fn_with_chunked_prefill(input_ids):
        return generate_with_chunked_prefill(neuron_model, tokenizer, input_ids)

    if is_chunked_prefill:
        generate_fn = generate_fn_with_chunked_prefill
    else:
        generate_fn = generate_fn_base
    passed, results, status_msg = logit_validation(
        input_ids=initial_input_ids,
        generate_fn=generate_fn,
        expected_logits=expected_logits,
        tol_map=tol_map,
        divergence_difference_tol=divergence_difference_tol,
    )
    if not passed:
        raise LogitMatchingValidationError(status_msg, results)

    logger.info(status_msg)

    return results


def check_accuracy_logits_v2(
        neuron_model: NeuronApplicationBase,
        expected_logits: torch.Tensor,
        inputs_input_ids: torch.Tensor,
        inputs_attention_mask: torch.Tensor,
        generation_config: GenerationConfig,
        divergence_difference_tol: float = 0.001,
        tol_map: dict = None,
        num_tokens_to_check: int = None,
        input_start_offsets=None,
        additional_input_args=None,
        tokenizer=None,
):
    """
        Validates the logits output from a Neuron model against expected logits. This is a refactored version
        of check_accuracy_logits that separates concerns into individual utility functions:
        - prepare_inputs_from_prompt: Handles input preparation
        - generate_expected_logits: Generates golden logits
        - shift_inputs_by_offset: Manages input offset adjustments

        This modular approach allows users to:
        1. Use each component independently as needed
        2. Handle different use cases more cleanly (e.g., image+text inputs)
        3. Maintain and test components separately

        The original function's functionality is preserved but reorganized for better maintainability
        and flexibility.

        Example:
        # Prepare inputs for multimodal model (image + text)
        input_ids, attention_mask, pixel_values, vision_mask = prepare_generation_inputs_hf(
            prompt, image, image_processor, role="user", config=neuron_model.config)
        additional_input_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "vision_mask": vision_mask.to(torch.bool),
        }
        accuracy.check_accuracy_logits_v2(
            neuron_model=neuron_model,
            expected_logits=expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            divergence_difference_tol=divergence_difference_tol,
            tol_map=tol_map,
            num_tokens_to_check=num_tokens_to_check,
            additional_input_args=additional_input_args,
        )

        Args:
            neuron_model (NeuronApplicationBase): The Neuron model to validate.
            expected_logits (torch.Tensor): Reference logits to compare against.
            inputs_input_ids (torch.Tensor): Input token IDs.
            inputs_attention_mask (torch.Tensor): Attention mask for the input tokens.
            generation_config (GenerationConfig): Configuration for generation.
            divergence_difference_tol (float, optional): Tolerance for logit divergence. Defaults to 0.001.
            tol_map (dict, optional): Map of position-specific tolerances.
            num_tokens_to_check (int, optional): Number of tokens to validate. If None, checks all possible tokens.
            input_start_offsets (optional): Starting offsets for input processing.
            additional_input_args (dict, optional): Additional arguments for model generation.
            tokenizer (optional): Tokenizer for decoding output tokens for debugging.

        Returns:
            dict: Validation results containing comparison metrics and statistics.

        Raises:
            LogitMatchingValidationError: If the logits don't match within the specified tolerance.
            AssertionError: If output_logits is not enabled when using on-device sampling.
        """
    if additional_input_args is None:
        additional_input_args = {}
    if neuron_model.neuron_config.on_device_sampling_config is not None:
        # should output both tokens and logits for logit matching check
        assert (
            neuron_model.neuron_config.output_logits
        ), "output_logits is required to enable logit validation with on-device sampling"

    if neuron_model.neuron_config.enable_fused_speculation:
        generation_config.prompt_lookup_num_tokens = neuron_model.neuron_config.speculation_length

    max_new_tokens = neuron_model.config.neuron_config.seq_len - inputs_input_ids.shape[1]
    if num_tokens_to_check is None:
        num_tokens_to_check = max_new_tokens
    else:
        num_tokens_to_check = min(max_new_tokens, num_tokens_to_check)
    spec_len = neuron_model.config.neuron_config.speculation_length
    if spec_len > 0:
        # With speculation, generation stops (spec_len - 1) tokens early.
        num_tokens_to_check -= spec_len - 1
    if (
            inputs_input_ids.shape[1] + num_tokens_to_check
            > neuron_model.config.neuron_config.max_context_length
    ):
        warnings.warn(
            (
                "input_len + num_tokens_to_check exceeds max_context_length. "
                "If output divergences at an index greater than max_context_length, "
                "a ValueError will occur because the next input len exceeds max_context_length. "
                "To avoid this, set num_tokens_to_check to a value of max_context_length - input_len or less."
            ),
            category=UserWarning,
        )

    model = HuggingFaceGenerationAdapter(neuron_model, input_start_offsets=input_start_offsets)

    expected_attention_mask = torch.ones(
        (
            inputs_attention_mask.shape[0],
            expected_logits.shape[0],
        ),
        dtype=torch.int32,
    )
    extrapolated_attention_mask = torch.cat(
        (inputs_attention_mask, expected_attention_mask), dim=1
    )

    def generate_fn_base(input_ids):
        input_length = input_ids.shape[1]
        attention_mask = extrapolated_attention_mask[:, :input_length]
        new_tokens = num_tokens_to_check + inputs_input_ids.shape[1] - input_length
        if spec_len > 0:
            # With speculation, generation stops (spec_len - 1) tokens early.
            new_tokens += spec_len - 1

        # Create base parameters dictionary
        input_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": new_tokens,
            "min_new_tokens": new_tokens,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
            "generation_config": generation_config,
        }
        # Update input_args with additional_neuron_input_args
        # This will override any existing keys and add new ones
        input_args.update(additional_input_args)
        # Call generate with the merged parameters
        model_outputs = model.generate(**input_args)

        actual_logits = torch.stack(model_outputs.scores)
        # Temporary fix for Llama4 vocab size error
        if isinstance(neuron_model, NeuronLlama4ForCausalLM) and actual_logits.shape[-1] > expected_logits.shape[-1]:
            actual_logits = actual_logits[..., :expected_logits.shape[-1]]

        actual_token_ids = actual_logits.argmax(dim=2).T
        if tokenizer is not None:
            actual_tokens = tokenizer.batch_decode(
                actual_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            logger.info(f"Actual Output(tokens with the max logits): {actual_tokens}")
        logger.info(f"Actual Output (token ids with the max logits): {actual_token_ids}")
        logger.info(f"Actual Logits Shape: {actual_logits.shape}")
        return actual_logits

    def generate_fn_with_chunked_prefill(input_ids):
        return generate_with_chunked_prefill(neuron_model, tokenizer, input_ids)

    if neuron_model.config.neuron_config.is_chunked_prefill:
        generate_fn = generate_fn_with_chunked_prefill
    else:
        generate_fn = generate_fn_base

    passed, results, status_msg = logit_validation(
        input_ids=inputs_input_ids,
        generate_fn=generate_fn,
        expected_logits=expected_logits,
        tol_map=tol_map,
        divergence_difference_tol=divergence_difference_tol,
    )
    if not passed:
        raise LogitMatchingValidationError(status_msg, results)

    logger.info(status_msg)

    return results



def shift_inputs_by_offset(inputs, input_start_offsets, pad_token_id=0):
    """
    Shifts input tensors by specified offsets, typically used for preparing model inputs
    by adjusting their positions and padding appropriately.

    Args:
        inputs: Input object containing input_ids and attention_mask tensors
        input_start_offsets: Tensor or list of integers specifying the shift amount
            for each sequence in the batch
        pad_token_id (int, optional): Token ID used for padding. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - shifted_input_ids (torch.Tensor): Input IDs shifted according to the
              specified offsets and padded with pad_token_id
            - shifted_attention_mask (torch.Tensor): Attention mask shifted according
              to the offsets and padded with 0s

    Note:
        The function shifts both input_ids and attention_mask by the same offsets,
        maintaining their alignment while using appropriate padding values for each.
    """
    shifted_input_ids = _shift_tensors_by_offset(input_start_offsets, inputs.input_ids, pad_token_id)
    shifted_attention_mask = _shift_tensors_by_offset(input_start_offsets, inputs.attention_mask, 0)

    return shifted_input_ids, shifted_attention_mask


def _shift_tensors_by_offset(input_start_offsets, input_tensors, pad_token_id):
    """Shift input tensor right by offsets

    Args:
      input_start_offsets: Tensor of (bs, 1) or (1, 1) shape
      input_tensors: Tensor of (bs, input_len) shape
      pad_token_id: after shifting, pad the rest of the output tensors with this token id

    Returns:
      input tensor shifted by offsets, padded by token_id
    """
    if input_start_offsets:
        bs, seq_len = input_tensors.shape
        max_offset = max(input_start_offsets)
        new_token_ids = torch.full((bs, max_offset + seq_len), pad_token_id, dtype= input_tensors.dtype, device= input_tensors.device)
        if len(input_start_offsets) > 1:
            for idx, offset in enumerate(input_start_offsets):
                new_token_ids[idx, offset:offset + seq_len] = input_tensors[idx, :]
        else:
            offset = input_start_offsets[0]
            new_token_ids[:, offset:offset + seq_len] = input_tensors # if there is only one offset value, shift all sequences the same amount
        return new_token_ids
    return input_tensors


def generate_with_chunked_prefill(
    neuron_model: NeuronApplicationBase,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
):
    """
    Generate sequences with chunked prefill.

    This func will generate the block table and slot mapping by default, 
    because chunked prefill uses block KV cache.

    To simplify the process, for now it will 
    1. First run prefilling for all of the seq, where the len to prefill is
       the same for all seq for each iteration.
    2. And then decode all seq together.

    In future, we can extend this func to run both prefill and decode in 
    one iteration.

    Args:
        neuron_model: NeuronApplicationBase
        input_ids: [max_num_seqs, input_len]

    Return:
        output_logits: [output_len, max_num_seqs, vocab_size], and output_len
            equals to seq_len - input_len
    """
    neuron_config = neuron_model.config.neuron_config

    chunk_size = neuron_config.max_context_length
    max_num_seqs = neuron_config.chunked_prefill_config.max_num_seqs

    seq_len = neuron_config.seq_len
    block_size = neuron_config.pa_block_size
    num_blocks_per_seq = math.ceil(seq_len / block_size)

    _, input_len = input_ids.shape

    # Prepare block table and slot mapping
    slot_mapping = torch.arange(max_num_seqs*seq_len).reshape(max_num_seqs, -1)
    block_table = torch.arange(max_num_seqs*num_blocks_per_seq).reshape(max_num_seqs, -1)

    # previous context only
    computed_context_lens = torch.zeros(max_num_seqs, dtype=torch.int)

    output_logits = []
    output_token_ids = []

    # Step 1: Prefill for all the seq
    assert chunk_size % max_num_seqs == 0
    max_prefill_len_per_seq = chunk_size // max_num_seqs
    assert chunk_size >= max_num_seqs
    num_iter_for_prefill = math.ceil(input_len / max_prefill_len_per_seq)
    for i in range(num_iter_for_prefill):
        start = i * max_prefill_len_per_seq
        end = min(input_len, (i + 1) * max_prefill_len_per_seq)
        actual_prefill_len = end - start

        # input_ids: (1, seq_len)
        cur_input_ids = input_ids[:, start: end].reshape(1, -1)
        # slot_mapping: (seq_len,)
        cur_slot_mapping = slot_mapping[:, start: end].reshape(-1)
        # block_table: (cp_max_num_seqs, num_active_blocks)
        last_block_id = math.ceil(end / block_size)
        cur_block_table = block_table[:, :last_block_id]

        full_context_lens = computed_context_lens + actual_prefill_len

        position_ids = torch.arange(start, end).repeat(max_num_seqs).reshape(1, -1)

        prefill_outputs = neuron_model(
            input_ids=cur_input_ids,
            position_ids=position_ids,
            slot_mapping=cur_slot_mapping,
            block_table=cur_block_table,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
        )

        computed_context_lens += actual_prefill_len
    # Only take the last logits because it is the first generated output, and
    # it is of shape (1, cp_max_num_seqs, vocab_size)
    prefill_logits = prefill_outputs.logits.squeeze()
    output_logits.append(prefill_logits)

    decode_input_ids = prefill_logits.argmax(dim=1)
    output_token_ids.append(decode_input_ids)

    # Step 2: Decode for all seq
    num_iter_for_decode = seq_len - input_len - 1
    assert num_iter_for_decode >= 0
    for i in range(num_iter_for_decode):
        start = input_len + i
        end = start + 1
        actual_prefill_len = end - start

        # input_ids: (1, seq_len)
        cur_input_ids = decode_input_ids.reshape(1, -1)
        # slot_mapping: (seq_len,)
        cur_slot_mapping = slot_mapping[:, start: end].reshape(-1)
        # block_table: (cp_max_num_seqs, num_active_blocks)
        last_block_id = math.ceil(end / block_size)
        cur_block_table = block_table[:, :last_block_id]

        full_context_lens = computed_context_lens + actual_prefill_len

        position_ids = torch.arange(start, end).repeat(max_num_seqs).reshape(1, -1)

        decode_outputs = neuron_model(
            input_ids=cur_input_ids,
            position_ids=position_ids,
            slot_mapping=cur_slot_mapping,
            block_table=cur_block_table,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
        )
        decode_logits = decode_outputs.logits.squeeze()
        output_logits.append(decode_logits)

        decode_input_ids = decode_logits.argmax(dim=1)
        output_token_ids.append(decode_input_ids)

        computed_context_lens += actual_prefill_len

    output_logits = torch.stack(output_logits).squeeze()
    output_token_ids = torch.stack(output_token_ids).squeeze().T
    if tokenizer is not None:
        output_tokens = tokenizer.batch_decode(
                output_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
        )
        logger.info(f"Actual Output: {output_tokens}")
    logger.info(f"Actual Output Tokens: {output_token_ids}")
    logger.info(f"Actual Logits Shape: {output_logits.shape}")
    return output_logits


def _prepare_mllama_vision_args(neuron_model, tokenizer, prompt, image, num_image_per_prompt, inputs, image_processor, initial_input_ids):
    if not hasattr(inputs, 'pixel_values'):
        if image is None:
            raise ValueError("Image input is required to check logit accuracy for a Mllama model.")
        if tokenizer is None:
            raise ValueError("A tokenizer is required to check logit accuracy for a Mllama model.")

        # Llama3.2 vision inputs
        batch_image = [
            [image] * 1
        ] * neuron_model.config.neuron_config.batch_size
        vision_token_id = neuron_model.config.image_token_index
        vision_mask = create_vision_mask(initial_input_ids, vision_token_id)
        pixel_values, aspect_ratios, num_chunks, has_image = get_image_tensors(
            neuron_model.config, batch_image
        )
        neuron_vision_input_args = {
            "pixel_values": pixel_values,
            "vision_mask": vision_mask,
            "aspect_ratios": aspect_ratios,
            "num_chunks": num_chunks,
            "has_image": has_image,
        }
        if image_processor is not None:
            hf_vision_input_args = image_processor(text=prompt, images=image, return_tensors="pt")
        else:
            hf_vision_input_args = {}
    else:
        neuron_vision_input_args = {
            "pixel_values": inputs.pixel_values,
            "vision_mask": inputs.vision_mask,
            "aspect_ratios": inputs.aspect_ratios,
            "num_chunks": inputs.num_chunks,
            "has_image": inputs.has_image,
        }
        hf_vision_input_args = {
            "pixel_values": inputs.pixel_values,
            "aspect_ratio_ids": inputs.aspect_ratio_ids,
            "aspect_ratio_mask": inputs.aspect_ratio_mask,
            "cross_attention_mask": inputs.cross_attention_mask,            
        }
    return neuron_vision_input_args, hf_vision_input_args


def _prepare_llama4_vision_args(neuron_model, prompt, image, inputs, image_processor):
    # Llama4 vision inputs
    if prompt is not None:
        assert image_processor is not None, \
            f"Image processor input is required to check logit accuracy for a Llama4 model.\
                    Alternatively you can pass in processed `inputs` with `pixel_values` and `vision_mask` directly."
        input_ids, attention_mask, pixel_values, vision_mask = llama4_prepare_generation_inputs_hf(prompt, image,
                                                                                            image_processor,
                                                                                            role="user",
                                                                                            config=neuron_model.config)
        neuron_vision_input_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "vision_mask": vision_mask.to(torch.bool),
        }
    else:
        if inputs is None or getattr(inputs, 'pixel_values', None) is None or getattr(inputs, 'vision_mask',
                                                                                      None) is None:
            assert (image is not None) and (image_processor is not None), \
                f"Image input and image processor input are required to check logit accuracy for a Llama4 model.\
            Alternatively you can pass in processed `inputs` with `pixel_values` and `vision_mask` directly."
            _, _, pixel_values, vision_mask = llama4_prepare_generation_inputs_hf(prompt, image, image_processor,
                                                                           role="user", config=neuron_model.config)
        else:
            pixel_values = inputs.pixel_values
            vision_mask = inputs.vision_mask

        neuron_vision_input_args = {
            "pixel_values": pixel_values,
            "vision_mask": vision_mask,
        }

    # HF model calculates vision_mask in Llama4ForConditionalGeneration.forward():
    # special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
    # so HF model does not take vision_mask as an input
    hf_vision_input_args = {
        "pixel_values": pixel_values.to(torch.float32),
    }

    return neuron_vision_input_args, hf_vision_input_args