import argparse
import torch
from pathlib import Path

from neuronx_distributed.parallel_layers.utils import initialize_fallback_parallel_state

from neuronx_distributed_inference.experimental.cli import generate
from neuronx_distributed_inference.experimental.core.config.neuron_config_handler import load_yaml_config
from neuronx_distributed_inference.experimental.models.llama4.model import (
    Transformer,
    load_llama_checkpoint,
)
from neuronx_distributed_inference.experimental.models.llama4.tokenizer import Tokenizer


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


@torch.inference_mode()
def main(args):
    config = load_yaml_config(args.config_path)

    batch_size = config.generation.batch_size
    seq_len = config.generation.sequence_length
    model_config = config.model
    checkpoint = load_llama_checkpoint(model_config, args.model_path)

    initialize_fallback_parallel_state()

    model: Transformer = Transformer(model_config, batch_size, seq_len)
    model.load_state_dict(checkpoint, strict=False)

    tokenizer = Tokenizer(model_path=Path(args.tokenizer_path))
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in args.prompts]

    output_tokens = generate(model, seq_len, prompt_tokens,
                             stop_tokens=tokenizer.stop_tokens, pad_token=config.model.pad_token)

    outputs = [tokenizer.decode(tokens) for tokens in output_tokens]

    print(outputs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
