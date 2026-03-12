# TODO simplify the imports
import argparse
import logging
import os

import neuronx_distributed
import torch
import torch.distributed as dist
from neuronx_distributed.parallel_layers import parallel_state

# TODO move to HF adapter
from neuronx_distributed_inference.experimental.cli import generate

from neuronx_distributed_inference.experimental.models.llama3.model import (
    Llama3ModelConfig,
    Llama3Transformer,
    load_llama_checkpoint,
)
from neuronx_distributed_inference.experimental.models.llama3.tokenizer import Tokenizer
from neuronx_distributed_inference.experimental.core.config.neuron_config_handler import load_yaml_config

os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"
logger = logging.getLogger("NeuronJIT")
logger.setLevel(logging.INFO)


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


# Run using: torchrun --nproc_per_node=32 llama3_1b_jit.py
def main(args):
    config: Llama3ModelConfig = load_yaml_config(args.config_path)

    batch_size = config.build.batch_size
    seq_len = config.build.sequence_length
    tp_degree = config.build.world_size

    checkpoint = load_llama_checkpoint(config, args.model_path)
    checkpoint_path = "chkpt.pt"
    torch.save(checkpoint, checkpoint_path)
    dist.init_process_group("xla", world_size=config.build.world_size)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)
    model = Llama3Transformer(config, batch_size, seq_len)
    neuronx_distributed.parallel_layers.load(
        checkpoint_path, model=model, sharded=False, model_key=None, strict=False
    )

    # Subsquent processing logic
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in args.prompts]
    output_tokens = generate(
        model, seq_len, prompt_tokens, stop_tokens=tokenizer.stop_tokens,
        pad_token=config.model.pad_token, jit_flow=True
    )

    logger.info("Decoding generated output tokens")
    outputs = [tokenizer.decode(tokens) for tokens in output_tokens]
    logger.info(outputs)

    # Remove saved checkpoint if not already removed by one of the processes
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
