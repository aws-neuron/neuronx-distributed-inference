import logging
import argparse
from typing import Dict, List, Tuple, Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronx_distributed import NxDParallelState, shard_checkpoint

from neuronx_distributed_inference.experimental.core.generate import generate
from neuronx_distributed_inference.experimental.core.build_flow import build_for_bucketing_on_seq_len
from neuronx_distributed_inference.experimental.core.processor import BucketingProcessor
from neuronx_distributed_inference.experimental.core.config.neuron_config_handler import load_yaml_config
from neuronx_distributed_inference.experimental.models.llama3.model import (
    load_llama_checkpoint,
    Llama3Config,
    Llama3Transformer,
)
from neuronx_distributed_inference.experimental.models.llama3.tokenizer import Tokenizer
from neuronx_distributed_inference.experimental.core.accuracy.logit_validation import logit_validation


def parse_args():
    parser = argparse.ArgumentParser(description="Run Llama3 inference with custom model and tokenizer paths")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file (e.g., /path/to/consolidated.00.pth)"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to the tokenizer model file (e.g., /path/to/tokenizer.model)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="src/neuronx_distributed_inference/experimental/models/llama3/examples/llama3_1b.yml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["How tall is the Space Needle?"],
        help="Input prompts for generation (can specify multiple) (e.g., \"How tall is the Space Needle?\" \"What is the capital of France?\")"
    )
    parser.add_argument(
        "--validate-accuracy",
        action="store_true",
        help="Enable accuracy validation against reference results using logit matching"
    )

    return parser.parse_args()


def get_config(args):
    config: Llama3Config = load_yaml_config(args.config_path)
    return config


def compile_model(args, config):
    batch_size = config.build.batch_size
    seq_len = config.build.sequence_length
    world_size = config.build.world_size

    # Compile the model
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=world_size):
        model = Llama3Transformer(config, batch_size, seq_len)
        nxd_model = build_for_bucketing_on_seq_len(
            model, batch_size=batch_size, max_seq_len=seq_len
        )

    # Shard the checkpoint
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=world_size):
        checkpoint = load_llama_checkpoint(config, args.model_path)
        model = Llama3Transformer(config, batch_size, seq_len)
        weights: List[Dict[str, torch.Tensor]] = shard_checkpoint(
            checkpoint=checkpoint,
            model=model,
            start_rank=0,
            end_rank=world_size - 1,
            load_on_device=True,
        )

    # Load the sharded weights to the NxDModel
    nxd_model.set_weights(weights)
    nxd_model.to_neuron()

    # Wrap nxd_model with an input and output processor
    nxd_model = BucketingProcessor(nxd_model, config.model.pad_token)
    return nxd_model


def generate_hf_reference_logits(
    prompts: List[str],
    config: Llama3Config,
    model_name: str
) -> Tuple[torch.Tensor, List[str]]:
    """
    Generate reference logits and text using HuggingFace Llama-3* model

    Args:
        prompts: List of input prompt strings
        config: Model configuration
        model_name: HuggingFace model name or local path to model directory

    Returns:
        Tuple containing:
        - Expected logits tensor with shape [output_length, batch_size, vocab_size]
        - List of generated text outputs
    """
    print(f"Loading HuggingFace '{model_name}' for reference...")

    # Load HF model and tokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not present
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    # Set left padding
    hf_tokenizer.padding_side = "left"

    # Tokenize prompts
    inputs = hf_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    # Calculate max new tokens
    max_new_tokens = config.build.sequence_length - inputs["input_ids"].shape[1]

    # Generate with logits collection
    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy generation to match Neuron model
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=hf_tokenizer.pad_token_id
        )

    # Batch decode all sequences
    full_texts = hf_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    # Extract generated parts
    hf_generated_text = []
    for i, full_text in enumerate(full_texts):
        prompt_text = prompts[i]
        if full_text.startswith(prompt_text):
            generated_part = full_text[len(prompt_text):]
        else:
            generated_part = full_text
        hf_generated_text.append(prompt_text + generated_part)

    # outputs.scores contains logits for each generation step
    # Stack them: [output_length, batch_size, vocab_size]
    if outputs.scores:
        expected_logits = torch.stack(outputs.scores, dim=0)
    else:
        # Handle case where no tokens were generated
        print("Warning: No tokens were generated by reference model")
        expected_logits = torch.empty(0, inputs["input_ids"].shape[0], hf_model.config.vocab_size)

    print("Generated reference logits")
    return expected_logits, hf_generated_text


def generate_tokens(args, config, nxdi_model) -> Tuple[List[str], List[List[int]], List[torch.Tensor]]:
    # Generation example
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in args.prompts]

    # Generate
    output_tokens = generate(
        model=nxdi_model,
        max_len=config.build.sequence_length,
        prompt_tokens=prompt_tokens,
        stop_tokens=tokenizer.stop_tokens,
        pad_token=config.model.pad_token,
    ).prompt_tokens

    # Decode the output tokens to strings
    output = [tokenizer.decode(tokens) for tokens in output_tokens]

    return output, prompt_tokens


def compare_logits(
    input_ids: List[List[int]],
    nxdi_model: Any,
    output: List[str],
    config: Llama3Config,
) -> None:
    print("\nComparing logits...")

    # Generate reference logits and text using HF models
    # Note: Reference logits can be obtained from any source, not necessarily HuggingFace models
    expected_logits, expected_output = generate_hf_reference_logits(args.prompts, config, "meta-llama/Llama-3.2-1B-Instruct")

    print("\n" + "=" * 80)
    print("OUTPUT COMPARISON")
    print("=" * 80)

    for i, (neuron_out, expected_out) in enumerate(zip(output, expected_output)):
        print(f"\nPrompt {i+1}: {args.prompts[i]}")
        print("-" * 60)
        print(f"Actual Output:\n{neuron_out}")
        print("-" * 60)
        print(f"Reference Output:\n{expected_out}")
        print("-" * 60)

    generate_logits_fn = _create_logit_generator(
        model=nxdi_model,
        max_len=config.build.sequence_length,
        pad_token=config.model.pad_token
    )

    passed = logit_validation(
        input_ids=input_ids,
        generate_fn=generate_logits_fn,
        expected_logits=expected_logits,
    )

    if passed:
        print("Accurate")
    else:
        print("Inaccurate")


def main(args):
    config = get_config(args)

    nxdi_model = compile_model(args, config)
    output, input_ids = generate_tokens(args, config, nxdi_model)
    print("Generated output:")
    print(output)

    # Perform accuracy check if requested
    if args.validate_accuracy:
        compare_logits(input_ids, nxdi_model, output, config)


def _create_logit_generator(
    model: Any,
    max_len: int,
    pad_token: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a callable that wraps the model's generation process for logit validation.

    This function returns a closure that can be used by logit_validation() to generate
    logits from input token sequences. The returned function handles the model's
    generation process and formats the output logits appropriately.

    Args:
        model: The NeuronX Distributed model to use for generation
        max_len: Maximum sequence length for generation
        pad_token: Token ID used for padding sequences

    Returns:
        A callable that takes input_ids (List[List[int]]) and returns logits tensor
        with shape [output_length, batch_size, vocab_size]
    """

    def generate_logits_fn(input_ids):

        # Run generation
        generation_result = generate(
            model=model,
            max_len=max_len,
            prompt_tokens=input_ids,
            stop_tokens=[],
            pad_token=pad_token,
            return_logits=True,
        )

        # Stack logits into expected shape [output_length, batch_size, vocab_size]
        # logit_validation() expects this specific tensor format for comparison
        return torch.stack(generation_result.logits)

    return generate_logits_fn


if __name__ == "__main__":
    logging.getLogger("Neuron").setLevel(logging.INFO)
    args = parse_args()
    main(args)
