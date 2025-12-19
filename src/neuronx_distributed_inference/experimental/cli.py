import copy
import json
import os
import sys
from functools import partial
from typing import Dict, List

import fire
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed import NxDParallelState
from neuronx_distributed.trace.model_builder import ModelBuilderV2, shard_checkpoint
from neuronx_distributed.trace.nxd_model import NxDModel

# TODO: Remove hardcoded Llama3_2_1B config in this file
from neuronx_distributed_inference.experimental.models.config import Llama3_2_1B
from neuronx_distributed_inference.experimental.models.llama3.model import (
    Llama3Transformer,
    load_llama_checkpoint,
)
from neuronx_distributed_inference.experimental.models.llama3.tokenizer import Tokenizer

"""
EXPERIMENTAL

This is a POC to find a better NXDI APIs. This is a base implementation which
will be iterated on top of to find the right target UX for NXDI.

Do not take a dependency on this.

TODO: migrate away from the generate and compile in this file

"""

# TODO: Update generate_cpu, generate_nxd, compile to be generic


def generate(
    model: torch.nn.Module,
    max_len: int,
    prompt_tokens: List[List[int]],
    stop_tokens: List[int],
    pad_token: int,
    jit_flow=False,
):
    prompt_tokens = copy.deepcopy(prompt_tokens)

    # Track max pos per batch
    last_pos = torch.tensor([len(prompt) - 1 for prompt in prompt_tokens], dtype=torch.int32)

    # Pad all batch lines to the same sequence length
    padded_tokens = [prompt + [pad_token] * (max_len - len(prompt)) for prompt in prompt_tokens]
    tokens = torch.tensor(padded_tokens, dtype=torch.int32)

    input_tokens = tokens
    input_bs, input_len = input_tokens.shape

    attention_mask = torch.where(tokens != pad_token, 1, 0).to(torch.int32)

    # A tensor to keep track of generation completion per batch line
    is_gen_complete = torch.full((input_bs, 1), False)

    while True:
        if jit_flow:
            # Move everything to device
            # Printing only the logs for Rank 0 for easier debugging, this can be changed later.
            (
                print(f"[RANK: {os.environ['RANK']}] Moving the model and inputs to xla")
                if str(os.environ["RANK"]) == "0"
                else None
            )
            model = model.to("xla")
            last_pos = last_pos.to("xla")
            input_tokens = input_tokens.to("xla")
            attention_mask = attention_mask.to("xla")
            is_gen_complete = is_gen_complete.to("xla")
            xm.mark_step()

        logits = model.forward(input_tokens, last_pos, attention_mask)

        if jit_flow:
            (
                print(f"[RANK: {os.environ['RANK']}] Logits: {logits.shape}")
                if str(os.environ["RANK"]) == "0"
                else None
            )
            xm.mark_step()

            logits.to("cpu")
            (
                print(f"[RANK: {os.environ['RANK']}] Moved logits to CPU: {logits.shape}")
                if str(os.environ["RANK"]) == "0"
                else None
            )

            last_pos = last_pos.to("cpu")
            input_tokens = input_tokens.to("cpu")
            attention_mask = attention_mask.to("cpu")
            is_gen_complete = is_gen_complete.to("cpu")
            (
                print(f"[RANK: {os.environ['RANK']}] All objects moved to CPU")
                if str(os.environ["RANK"]) == "0"
                else None
            )

        last_pos = last_pos + 1

        # assuming we are doing greedy sampling
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        input_tokens = next_token.to(torch.int32)
        next_token = next_token.to("cpu") if jit_flow else next_token

        # Add the new token to prompt
        for idx, prompt in enumerate(prompt_tokens):
            if not is_gen_complete[idx][0].item():
                prompt.append(next_token[idx].item())

        for stop_token in stop_tokens:
            is_gen_complete = is_gen_complete.logical_or(next_token == stop_token)

        # Stop generation when all batch lines are complete
        if is_gen_complete.all():
            break

        if torch.max(last_pos).item() >= max_len:
            break

        # Update mask
        attention_mask[torch.arange(last_pos.shape[0]), last_pos] = 1

    return prompt_tokens


def compile(
    tp_degree,
    batch_size,
    seq_len,
    model_path,
    output_path,
):
    """
    Compile action for the CLI.

    Args:
        model_path (Optional[str]): Path to the model to compile
    """

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        model = Llama3Transformer(Llama3_2_1B, batch_size, seq_len)

        # Initialize ModelBuilderV2
        builder = ModelBuilderV2(
            model=model,
        )

        # Add prefill trace
        builder.trace(
            args=(
                torch.ones((batch_size, seq_len), dtype=torch.int32),  # input tokens
                torch.tensor([0] * batch_size, dtype=torch.int32),
                torch.ones((batch_size, seq_len), dtype=torch.int32),  # attention mask
            ),
            tag="prefill",
        )

        # Add decode trace
        builder.trace(
            args=(
                # input tokens
                torch.ones((batch_size, 1), dtype=torch.int32),
                torch.tensor([0] * batch_size, dtype=torch.int32),
                torch.ones((batch_size, seq_len), dtype=torch.int32),  # attention mask
            ),
            tag="decode",
        )

        # Compile the model
        traced_model = builder.compile(
            compiler_args="--auto-cast=none",
        )

    if not output_path.endswith("/"):
        output_path += "/"

    os.makedirs(output_path, exist_ok=True)

    # Save the traced model
    traced_model.save(output_path + "nxd_model.pt")

    # Save the config
    data = {"batch_size": batch_size, "seq_len": seq_len, "tp_degree": tp_degree}
    with open(output_path + "config.json", "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved compiled model to {output_path}")


@torch.inference_mode()
def generate_cpu(
    batch_size,
    seq_len,
    model_path,
    tokenizer_path,
    prompts,
):

    checkpoint = load_llama_checkpoint(Llama3_2_1B, model_path)

    model: Llama3Transformer = Llama3Transformer(Llama3_2_1B, batch_size, seq_len)
    model.load_state_dict(checkpoint, strict=False)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    output_tokens = generate(model, seq_len, prompt_tokens, stop_tokens=tokenizer.stop_tokens)

    return [tokenizer.decode(tokens) for tokens in output_tokens]


@torch.inference_mode()
def generate_nxd(
    model_path,
    compiled_model_path,
    tokenizer_path,
    prompts,
):

    if not compiled_model_path.endswith("/"):
        compiled_model_path += "/"
    with open(compiled_model_path + "config.json", "r") as file:
        cfg = json.load(file)

    batch_size, seq_len, tp_degree = cfg["batch_size"], cfg["seq_len"], cfg["tp_degree"]

    if len(prompts) != batch_size:
        raise ValueError(f"Prompts size does not match batch size {cfg['batch_size']}")

    checkpoint = load_llama_checkpoint(Llama3_2_1B, model_path, tp_degree)

    print("Sharding the model")
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        model = Llama3Transformer(Llama3_2_1B, batch_size, seq_len)
        weights: List[Dict[str, torch.Tensor]] = shard_checkpoint(
            checkpoint=checkpoint,
            model=model,
            start_rank=0,
            end_rank=tp_degree - 1,
            load_on_device=True,
        )

    nxd_model = NxDModel.load(compiled_model_path + "nxd_model.pt")
    nxd_model.set_weights(weights)
    nxd_model.to_neuron()

    tokenizer = Tokenizer(model_path=tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    output_tokens = generate(nxd_model, seq_len, prompt_tokens, stop_tokens=tokenizer.stop_tokens)

    return [tokenizer.decode(tokens) for tokens in output_tokens]


def _print_string_list_output(func, *args, **kwargs):
    # Print output with newlines. Fire removes newlines when printing a list of strings.
    string_list = func(*args, **kwargs)
    for string in string_list:
        print(string)


def main():
    fire.Fire(
        {
            "generate_cpu": partial(_print_string_list_output, generate_cpu),
            "generate_nxd": partial(_print_string_list_output, generate_nxd),
            "compile": compile,
        }
    )


if __name__ == "__main__":
    sys.exit(main())
