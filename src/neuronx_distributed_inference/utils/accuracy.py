"""
This is a temporary file to get the testing running for new package.

Some of the utitlies functions need to be redo or removed.
"""

# flake8: noqa

import warnings
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Union

import torch
from torch_neuronx.testing.validation import custom_allclose, logit_validation
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.mllama.utils import create_vision_mask, get_image_tensors
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
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
        plt.plot(
            actual_output.float().detach().numpy().reshape(-1),
            expected_output.float().detach().numpy().reshape(-1),
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
        print("attention mask not provided, padding inputs and generating a mask")

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

    generation_model = model if is_hf else HuggingFaceGenerationAdapter(model)
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
            **generate_kwargs,
        )

# FIXME: add on cpu check support
def check_accuracy(
    neuron_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: Optional[GenerationConfig] = None,
    expected_token_ids: Optional[List] = None,
    num_tokens_to_check: int = None,
    do_sample: bool = True,
    draft_model: PreTrainedModel = None,
    prompt: Optional[str] = None,
    image=None,
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

    print(
        f"run accuracy check with generation_config as: {generation_kwargs} and {execution_mode=}"
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
            **generation_kwargs,
        )

    print(f"Expected output: {outputs_expected}")

    mode_being_tested = "async mode" if neuron_model.neuron_config.async_mode else "sync mode"
    output_token_ids, outputs_actual = get_generate_outputs(
        neuron_model,
        prompts,
        tokenizer,
        is_hf=False,
        draft_model=draft_model,
        generation_config=generation_config,
        **generation_kwargs,
    )
    print(f"Actual output  : {outputs_actual}")

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) if tokenizer else 0
    output_token_ids = output_token_ids[output_token_ids != pad_token_id]
    expected_token_ids = expected_token_ids[expected_token_ids != pad_token_id]
    if num_tokens_to_check is not None:
        print(f"Validating the first {num_tokens_to_check} tokens")
        expected_token_ids = expected_token_ids[:num_tokens_to_check]
        output_token_ids = output_token_ids[:num_tokens_to_check]

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
    print(f"The output from Neuronx NxD on {device} using {mode_being_tested} is accurate!")

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
):
    if neuron_model.neuron_config.enable_fused_speculation:
        generation_config.prompt_lookup_num_tokens = neuron_model.neuron_config.speculation_length
        assert (
            neuron_model.neuron_config.output_logits
        ), "output_logits is required to enable logits tests for fused speculation"
    elif neuron_model.neuron_config.on_device_sampling_config is not None:
        raise ValueError("Logits validation is not supported with on-device sampling.")

    if image is not None and tokenizer is None:
        raise ValueError("A tokenizer is required to check logit accuracy for a multimodal model")

    if inputs is None and tokenizer is None:
        raise ValueError("Must provide either a tokenizer or inputs that include input_ids and attention_mask")

    if inputs is None:
        if prompt is None:
            prompt = MM_TEST_PROMPT if image else TEST_PROMPT
        prompts = [prompt] * neuron_model.config.neuron_config.batch_size

        inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    initial_input_ids = inputs.input_ids
    initial_attention_mask = inputs.attention_mask
    initial_input_len = inputs.input_ids.shape[1]
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
    if initial_input_len + num_tokens_to_check > neuron_model.config.neuron_config.max_context_length:
        warnings.warn(
            (
                "input_len + num_tokens_to_check exceeds max_context_length. "
                "If output divergences at an index greater than max_context_length, "
                "a ValueError will occur because the next input len exceeds max_context_length. "
                "To avoid this, set num_tokens_to_check to a value of max_context_length - input_len or less."
            ),
            category=UserWarning,
        )

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
        )
        expected_logits = torch.stack(outputs.scores)

    expected_logits = expected_logits[:num_tokens_to_check, :, :]
    expected_token_ids = expected_logits.argmax(dim=2).T
    if tokenizer is not None:
        expected_tokens = tokenizer.batch_decode(
            expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Expected Output: ", expected_tokens, expected_token_ids)
    else:
        print("Expected Output: ", expected_token_ids)
    print("Expected Logits Shape: ", expected_logits.shape)

    model = HuggingFaceGenerationAdapter(neuron_model)
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

    def generate_fn(input_ids):
        input_length = input_ids.shape[1]
        attention_mask = extrapolated_attention_mask[:, :input_length]
        new_tokens = num_tokens_to_check + initial_input_len - input_length
        if spec_len > 0:
            # With speculation, generation stops (spec_len - 1) tokens early.
            new_tokens += spec_len - 1
        model_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=generation_config,
        )

        actual_logits = torch.stack(model_outputs.scores)
        actual_token_ids = actual_logits.argmax(dim=2).T
        if tokenizer is not None:
            actual_tokens = tokenizer.batch_decode(
                actual_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print("Actual Output: ", actual_tokens, actual_token_ids)
        else:
            print("Actual Output: ", actual_token_ids)
        print("Actual Logits Shape: ", actual_logits.shape)
        return torch.stack(model_outputs.scores)

    batch_image = []
    # generate the batch image
    if image is not None:
        batch_image = [
            [image] * num_image_per_prompt
        ] * neuron_model.config.neuron_config.batch_size

    def generate_mm_fn(input_ids):
        input_length = input_ids.shape[1]
        new_tokens = num_tokens_to_check + initial_input_len - input_length
        if spec_len > 0:
            # With speculation, generation stops (spec_len - 1) tokens early.
            new_tokens += spec_len - 1
        vision_token_id = tokenizer("<|image|>", add_special_tokens=False).input_ids[0]
        vision_mask = create_vision_mask(initial_input_ids, vision_token_id)
        attention_mask = extrapolated_attention_mask[:, :input_length]
        pixel_values, aspect_ratios, num_chunks, has_image = get_image_tensors(
            neuron_model.config, batch_image
        )
        sampling_params = prepare_sampling_params(
            batch_size=neuron_model.config.neuron_config.batch_size,
            top_k=[1],
            top_p=[1.0],
            temperature=[1.0],
        )
        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                sampling_params=sampling_params,
                pixel_values=pixel_values,
                aspect_ratios=aspect_ratios,
                vision_mask=vision_mask,
                num_chunks=num_chunks,
                has_image=has_image,
                return_dict_in_generate=True,
                output_scores=True,
                generation_config=generation_config,
            )
        actual_logits = torch.stack(model_outputs.scores)
        actual_token_ids = actual_logits.argmax(dim=2).T
        actual_tokens = tokenizer.batch_decode(
            actual_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Actual Output: ", actual_tokens, actual_token_ids)
        print("Actual Logits Shape: ", actual_logits.shape)
        return torch.stack(model_outputs.scores)

    passed, results, status_msg = logit_validation(
        input_ids=initial_input_ids,
        generate_fn=(
            generate_fn if image is None else generate_mm_fn
        ),  # use different generation functions
        expected_logits=expected_logits,
        tol_map=tol_map,
        divergence_difference_tol=divergence_difference_tol,
    )

    if not passed:
        raise LogitMatchingValidationError(status_msg, results)

    print("Passed logits validation!")

    return results
