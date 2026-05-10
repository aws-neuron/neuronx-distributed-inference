#!/usr/bin/env python3
"""
vLLM Serving Script for ZAYA1 models on Neuron.

This script verifies NxDI MODEL_TYPES registration for the ZAYA contrib model,
then launches vLLM's OpenAI-compatible server with reasoning parser support.

Usage:
    # ZAYA1-8B with reasoning (TP=2, pre-compiled):
    python serve_vllm.py --model /mnt/models/ZAYA1-8B --tp 2 \
        --compiled-model-path /mnt/models/ZAYA1-8B-compiled-tp2-seq2048 \
        --max-model-len 2048 --max-num-seqs 1

    # ZAYA1-base (TP=1):
    python serve_vllm.py --model /mnt/models/ZAYA1-base --tp 1 \
        --compiled-model-path /mnt/models/ZAYA1-base-compiled-tp1 \
        --reasoning-parser none

    # Without pre-compiled model (will compile on startup):
    python serve_vllm.py --model /mnt/models/ZAYA1-8B --tp 2 \
        --max-model-len 2048 --max-num-seqs 1

Note on sequence length:
    The Neuron compiler (NCC_ITEN404) crashes on CTE buckets >= 1024 for this
    model. The --context-buckets flag caps CTE bucket sizes to work around this.
    Default: 128,256,512 (max prefill chunk = 512 tokens; longer prompts are
    padded to 512). The KV cache (--max-model-len) can be much larger since
    TKG buckets are always seq_len=1 and are unaffected.

Note on reasoning parser:
    ZAYA1-8B uses <think>...</think> tags for chain-of-thought reasoning.
    The 'deepseek_r1' parser (default) extracts thinking content into the
    'reasoning_content' field of the OpenAI response, leaving only the
    final answer in 'content'.

Prerequisites:
    - Zyphra's custom transformers fork installed:
        pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya"
    - ZAYA NxDI contrib model registered in constants.py (both venvs)
    - ZayaForCausalLM registered in vLLM's model registry (registry.py)
    - vLLM venv: /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/
"""

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def setup_environment():
    """Configure sys.path and environment variables."""
    # Note: The ZAYA model is registered in NxDI's constants.py directly
    # (patched on the remote instance). This ensures it's available in all
    # vLLM worker processes, including those spawned via multiprocessing.

    # Ensure Neuron platform target is set for trn2
    if "NEURON_PLATFORM_TARGET_OVERRIDE" not in os.environ:
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"


def register_zaya_model():
    """Verify ZAYA model is registered in NxDI's MODEL_TYPES registry.

    The model should already be registered via the constants.py patch.
    This function just verifies the registration is present.
    """
    from neuronx_distributed_inference.utils import constants

    if "zaya" not in constants.MODEL_TYPES:
        # Fallback: register manually (only works in this process, not workers)
        logger.warning(
            "ZAYA not found in MODEL_TYPES. "
            "Attempting manual registration (may not work with vLLM spawn)."
        )
        src_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
        )
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from modeling_zaya import NeuronZayaForCausalLM

        constants.MODEL_TYPES["zaya"] = {"causal-lm": NeuronZayaForCausalLM}
        logger.info("Registered 'zaya' in MODEL_TYPES (manual fallback)")
    else:
        logger.info(
            "'zaya' already registered in MODEL_TYPES: %s",
            constants.MODEL_TYPES["zaya"],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM server for ZAYA1-base on Neuron"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ZAYA1-base HuggingFace checkpoint directory",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel degree (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length / KV cache size (default: 4096)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of concurrent sequences / batch size (default: 1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default=None,
        help=(
            "Path to pre-compiled Neuron model (skips compilation). "
            "Sets NEURON_COMPILED_ARTIFACTS env var for the vLLM Neuron loader."
        ),
    )
    parser.add_argument(
        "--context-buckets",
        type=str,
        default="128,256,512",
        help=(
            "Comma-separated list of CTE bucket sizes (default: '128,256,512'). "
            "Caps prefill at the largest bucket (512 tokens by default). "
            "Required workaround for NCC_ITEN404 compiler crash at seq_len>=1024. "
            "Set to 'auto' to let NxDI auto-generate buckets up to max-model-len "
            "(will fail if any bucket >= 1024)."
        ),
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default="deepseek_r1",
        help=(
            "Reasoning parser for extracting think/answer from model output. "
            "Use 'deepseek_r1' for <think>...</think> format (default). "
            "Set to 'none' to disable reasoning parsing."
        ),
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Step 1: Setup environment and register model
    setup_environment()
    register_zaya_model()

    # Step 2: Set compiled model path via env var (vLLM Neuron loader reads this)
    if args.compiled_model_path:
        os.environ["NEURON_COMPILED_ARTIFACTS"] = args.compiled_model_path
        logger.info("Set NEURON_COMPILED_ARTIFACTS=%s", args.compiled_model_path)

    # Step 3: Build override_neuron_config for ZAYA-specific settings
    override_neuron_config = {
        # ZAYA REQUIRES is_continuous_batching=True even for batch=1
        # because the non-CB KV cache path uses torch.scatter which
        # Neuron doesn't support.
        "is_continuous_batching": True,
    }

    # Custom CTE bucket sizes (workaround for NCC_ITEN404 at seq_len>=1024)
    if args.context_buckets and args.context_buckets.lower() != "auto":
        buckets = [int(b.strip()) for b in args.context_buckets.split(",")]
        override_neuron_config["context_encoding_buckets"] = sorted(buckets)
        logger.info(
            "Custom CTE buckets: %s", override_neuron_config["context_encoding_buckets"]
        )

    # Step 4: Build vLLM CLI arguments
    vllm_args = [
        "serve",
        args.model,
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--port",
        str(args.port),
        "--host",
        args.host,
        "--trust-remote-code",
        "--block-size",
        "128",
        "--no-enable-prefix-caching",
        "--additional-config",
        json.dumps({"override_neuron_config": override_neuron_config}),
    ]

    # Add reasoning parser if specified
    if args.reasoning_parser and args.reasoning_parser.lower() != "none":
        vllm_args.extend(["--reasoning-parser", args.reasoning_parser])
        logger.info("Reasoning parser: %s", args.reasoning_parser)

    logger.info("Starting vLLM server with args: %s", " ".join(vllm_args))
    logger.info("Override neuron config: %s", override_neuron_config)

    # Step 4: Launch vLLM via its CLI
    sys.argv = ["vllm"] + vllm_args
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
