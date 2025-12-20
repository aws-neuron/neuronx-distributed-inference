import logging
import argparse
from typing import Dict, List

import torch

# TODO Move this to not to mocking if torch distributed is already initialized
from neuronx_distributed import NxDParallelState
from neuronx_distributed.trace.model_builder import shard_checkpoint

from neuronx_distributed_inference.experimental.core.generate import generate
from neuronx_distributed_inference.experimental.core.build_flow import build_for_bucketing_on_seq_len
from neuronx_distributed_inference.experimental.core.processor import BucketingProcessor

# TODO simplify the imports
from neuronx_distributed_inference.experimental.models.llama3.model import (
    load_llama_checkpoint,
    Llama3Config,
    Llama3Transformer,
)
from neuronx_distributed_inference.experimental.models.llama3.tokenizer import Tokenizer
from neuronx_distributed_inference.experimental.core.config.neuron_config_handler import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run Llama3 inference with custom model and tokenizer paths")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file (e.g., /path/to/consolidated.00.pth)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer model file (e.g., /path/to/tokenizer.model)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/neuronx_distributed_inference/experimental/models/llama3/llama3_1b.yml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["How tall is the Space Needle?"],
        help="Input prompts for generation (can specify multiple)"
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


def generate_tokens(args, config, nxd_model_obj):
    # Generation example
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in args.prompts]

    # Move away from this generate to HF Adapter
    output_tokens = generate(
        nxd_model_obj, config.build.sequence_length,
        prompt_tokens, stop_tokens=tokenizer.stop_tokens, pad_token=config.model.pad_token
    )
    output = [tokenizer.decode(tokens) for tokens in output_tokens]
    return output


def main(args):
    config = get_config(args)
    nxd_model_obj = compile_model(args, config)
    output = generate_tokens(args, config, nxd_model_obj)
    print(output)


if __name__ == "__main__":
    logging.getLogger("Neuron").setLevel(logging.INFO)
    args = parse_args()
    main(args)
