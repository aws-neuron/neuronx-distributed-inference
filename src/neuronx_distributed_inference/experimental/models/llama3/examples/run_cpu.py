# TODO simplify the imports
import argparse
from neuronx_distributed.parallel_layers.utils import initialize_fallback_parallel_state
from neuronx_distributed_inference.experimental.cli import generate
from neuronx_distributed_inference.experimental.models.llama3.model import (
    Llama3Transformer,
    load_llama_checkpoint,
    Llama3Config,
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


def main(args):
    config: Llama3Config = load_yaml_config(args.config_path)

    # TODO: Fix parallel run on CPU so we can run with world_size > 1
    config.build.world_size = 1
    config.attention.cp_degree = 1

    batch_size = config.build.batch_size
    seq_len = config.build.sequence_length
    checkpoint = load_llama_checkpoint(config, args.model_path)

    initialize_fallback_parallel_state()

    model = Llama3Transformer(config, batch_size, seq_len)
    model.load_state_dict(checkpoint, strict=False)

    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in args.prompts]

    output_tokens = generate(
        model, seq_len, prompt_tokens, stop_tokens=tokenizer.stop_tokens, pad_token=config.model.pad_token
    )

    outputs = [tokenizer.decode(tokens) for tokens in output_tokens]

    print(outputs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
