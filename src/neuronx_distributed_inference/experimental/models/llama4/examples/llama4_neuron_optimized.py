import argparse
from pathlib import Path

from neuronx_distributed import NxDParallelState, shard_checkpoint

from neuronx_distributed_inference.experimental.core.functions import build
from neuronx_distributed_inference.experimental.cli import generate
from neuronx_distributed_inference.experimental.models.llama4.model_optimized import (
    Transformer,
    load_llama_checkpoint,
)
from neuronx_distributed_inference.experimental.models.llama4.tokenizer import Tokenizer
from neuronx_distributed_inference.experimental.core.config.neuron_config_handler import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run Llama4 inference with custom model and tokenizer paths")
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
        default="src/neuronx_distributed_inference/experimental/models/llama4/examples/llama_4_scout_config.yml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["How tall is the Space Needle?", "What is the capital of France?"],
        help="Input prompts for generation (can specify multiple)"
    )
    return parser.parse_args()


def get_config(args):
    config = load_yaml_config(args.config_path)
    return config


def compile_model(args, config):
    batch_size = config.generation.batch_size
    seq_len = config.generation.sequence_length
    tp_degree = config.generation.tp_degree
    model_config = config.model

    # Compile the model
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        model = Transformer(model_config, batch_size, seq_len)
        nxd_model = build(
            model, world_size=tp_degree, batch_size=batch_size, sequence_length=seq_len
        )

    # Shard the checkpoint
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        checkpoint = load_llama_checkpoint(model_config, args.model_path, tp_degree)
        model = Transformer(model_config, batch_size, seq_len)
        weights = shard_checkpoint(
            checkpoint=checkpoint,
            model=model,
            start_rank=0,
            end_rank=tp_degree - 1,
            load_on_device=True,
        )

    # Load the sharded weights to the NxDModel
    nxd_model.set_weights(weights)
    nxd_model.to_neuron()
    return nxd_model


def generate_tokens(args, config, nxd_model_obj):
    # Generation example
    tokenizer = Tokenizer(model_path=Path(args.tokenizer_path))
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in args.prompts]

    output_tokens = generate(
        nxd_model_obj, config.generation.sequence_length, prompt_tokens, stop_tokens=tokenizer.stop_tokens,
        pad_token=config.model.pad_token
    )
    output = [tokenizer.decode(tokens) for tokens in output_tokens]
    return output


def main():
    args = parse_args()
    config = get_config(args)
    nxd_model_obj = compile_model(args, config)
    output = generate_tokens(args, config, nxd_model_obj)
    print(output)


if __name__ == "__main__":
    main()
