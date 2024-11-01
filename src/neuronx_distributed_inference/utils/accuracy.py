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
from torch_neuronx.testing.validation import logit_validation
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.utils.constants import *
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    warnings.warn(
        "Intel extension for pytorch not found. For faster CPU references install `intel-extension-for-pytorch`.",
        category=UserWarning,
    )
    ipex = None

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


def get_generate_outputs_from_token_ids(
    model,
    token_ids,
    tokenizer,
    attention_mask=None,
    is_hf=False,
    draft_model=None,
    **generate_kwargs,
):
    if draft_model is not None and not is_hf:
        # TODO: Fix draft model support on HF. HF supports speculative decoding, but output is garbage currently.
        #       The current check_accuracy behavior (compare Neuron w/ speculation against HF w/o speculation)
        #       is consistent with the old runner.py implementation.
        assert not is_hf, "Draft model not supported for generating on HF"
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
    outputs = generation_model.generate(token_ids, attention_mask=attention_mask, **generate_kwargs)

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
    model, prompts, tokenizer, is_hf=False, draft_model=None, device="neuron", **generate_kwargs
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
            **generate_kwargs,
        )


def get_async_modes_to_test(execution_mode: str, neuron_model: PreTrainedModel):
    if execution_mode == "config":
        async_modes_to_test = [neuron_model.neuron_config.async_mode]
    elif execution_mode == "sync":
        async_modes_to_test = [False]
    elif execution_mode == "async":
        async_modes_to_test = [True]
    elif execution_mode == "both":
        async_modes_to_test = [
            neuron_model.neuron_config.async_mode,
            not neuron_model.neuron_config.async_mode,
        ]
    else:
        raise ValueError(
            f"`{execution_mode=}` is not a supported mode. Please use one of the following: config, sync, async, both."
        )

    return async_modes_to_test


# FIXME: add on cpu check support
def check_accuracy(
    neuron_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: Optional[GenerationConfig] = None,
    expected_token_ids: Optional[List] = None,
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
    if neuron_config.enable_fused_speculation:
        generation_kwargs.update({"do_sample": False})

    async_modes_to_test = get_async_modes_to_test(execution_mode, neuron_model)
    for async_mode in async_modes_to_test:
        neuron_model.neuron_config.async_mode = async_mode
        mode_being_tested = "async mode" if async_mode else "sync mode"

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
        if draft_model is not None:
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
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    prompt: str = None,
    expected_logits: torch.Tensor = None,
    divergence_difference_tol: float = 0.001,
    tol_map: dict = None,
    num_tokens_to_check: int = None,
    execution_mode="config",
):
    if prompt is None:
        prompt = TEST_PROMPT
    prompts = [prompt] * neuron_model.config.neuron_config.batch_size

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    initial_input_ids = inputs.input_ids
    initial_attention_mask = inputs.attention_mask
    seq_len = neuron_model.config.neuron_config.seq_len

    if expected_logits is None:
        # Generate goldens with HF on CPU
        # logit_validation assumes greedy sampling
        hf_model = neuron_model.load_hf_model(neuron_model.model_path)
        new_tokens = seq_len - inputs.input_ids.shape[1]
        outputs = hf_model.generate(
            inputs.input_ids,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            attention_mask=inputs.attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=generation_config,
        )
        expected_logits = torch.stack(outputs.scores)

    if num_tokens_to_check is not None:
        print(f"Validating logits for first {num_tokens_to_check} tokens")
        expected_logits = expected_logits[:num_tokens_to_check, :, :]

    expected_token_ids = expected_logits.argmax(dim=2).T
    expected_tokens = tokenizer.batch_decode(
        expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Expected Output: ", expected_tokens, expected_token_ids)
    print("Expected Logits Shape: ", expected_logits.shape)

    model = HuggingFaceGenerationAdapter(neuron_model)
    expected_attention_mask = torch.ones(
        (
            initial_attention_mask.shape[0],
            expected_token_ids.shape[1] - initial_attention_mask.shape[1],
        ),
        dtype=torch.int32,
    )
    extrapolated_attention_mask = torch.cat(
        (initial_attention_mask, expected_attention_mask), dim=1
    )

    def generate_fn(input_ids):
        input_length = input_ids.shape[1]
        attention_mask = extrapolated_attention_mask[:, :input_length]
        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=seq_len - input_length,
                min_new_tokens=seq_len - input_length,
                do_sample=False,
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

    original_async_mode = neuron_model.neuron_config.async_mode
    try:
        async_modes_to_test = get_async_modes_to_test(execution_mode, neuron_model)
        for async_mode in async_modes_to_test:
            neuron_model.neuron_config.async_mode = async_mode
            mode_being_tested = "async mode" if async_mode else "sync mode"

            passed, results, status_msg = logit_validation(
                initial_input_ids,
                generate_fn,
                expected_logits,
                tol_map=tol_map,
                divergence_difference_tol=divergence_difference_tol,
            )
            assert passed, status_msg
            print(f"Passed logits validation for {mode_being_tested}")
    finally:
        neuron_model.neuron_config.async_mode = original_async_mode
