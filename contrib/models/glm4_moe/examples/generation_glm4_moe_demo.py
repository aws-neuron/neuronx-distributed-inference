"""
GLM-4.5 MoE Generation Demo for NXD Inference.

This script demonstrates how to compile and run inference with the GLM-4.5 MoE model
using neuronx-distributed-inference.

Usage:
    # From contrib/models/glm4_moe/ directory:

    # Compile and generate (default tiny random model):
    python examples/generation_glm4_moe_demo.py

    # Skip compile (load from existing traced model):
    python examples/generation_glm4_moe_demo.py --skip-compile

    # Use real checkpoint:
    python examples/generation_glm4_moe_demo.py \\
        --model-path /path/to/glm4_moe \\
        --traced-model-path /path/to/traced \\
        --tp-degree 8 --seq-len 2048
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig

# Add src to path so glm4_moe package can be found
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glm4_moe.modeling_glm4_moe import (
    Glm4MoeInferenceConfig,
    NeuronGlm4MoeForCausalLM,
)
from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

# Default paths — override via CLI
MODEL_PATH = str(
    Path(__file__).parent.parent.parent.parent.parent / "glm4_moe_tiny_random"
)
TRACED_MODEL_PATH = str(
    Path(__file__).parent.parent.parent.parent.parent / "glm4_moe_tiny_random_traced"
)

torch.manual_seed(0)

DTYPE = torch.bfloat16


def get_neuron_config(tp_degree: int = 2, seq_len: int = 64) -> MoENeuronConfig:
    """Create MoENeuronConfig for GLM-4.5 MoE model.

    Args:
        tp_degree: Tensor parallelism degree (number of NeuronCores).
        seq_len: Maximum sequence length.

    Returns:
        Configured MoENeuronConfig instance.
    """
    moe_tp_degree = tp_degree  # align MoE TP degree with overall TP degree
    return MoENeuronConfig(
        tp_degree=tp_degree,
        moe_tp_degree=moe_tp_degree,
        moe_ep_degree=1,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=seq_len,
        max_context_length=seq_len - 16,
        torch_dtype=DTYPE,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=False,
            top_k=1,
        ),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        fused_qkv=True,
        sequence_parallel_enabled=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
    )


def generate(
    model_path: str,
    traced_model_path: str,
    skip_compile: bool = False,
    tp_degree: int = 2,
    seq_len: int = 64,
) -> None:
    """Compile (if needed) and run GLM-4.5 MoE inference.

    Args:
        model_path: Path to the HuggingFace model checkpoint.
        traced_model_path: Path to save/load the compiled Neuron model.
        skip_compile: If True, skip compilation and load existing traced model.
        tp_degree: Tensor parallelism degree.
        seq_len: Maximum sequence length.
    """
    if not skip_compile:
        print("=" * 60)
        print("Compiling GLM-4.5 MoE model...")
        print("=" * 60)

        neuron_config = get_neuron_config(tp_degree=tp_degree, seq_len=seq_len)
        config = Glm4MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )

        print(
            f"  Model config: hidden_size={config.hidden_size}, "
            f"n_routed_experts={config.n_routed_experts}, "
            f"n_shared_experts={config.n_shared_experts}, "
            f"first_k_dense_replace={config.first_k_dense_replace}"
        )

        model = NeuronGlm4MoeForCausalLM(model_path, config)
        model.compile(traced_model_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.save_pretrained(traced_model_path)
        except Exception as e:
            print(f"Warning: could not save tokenizer: {e}")

        print(f"Model compiled and saved to {traced_model_path}")

    # Load compiled model
    print("\n" + "=" * 60)
    print("Loading compiled GLM-4.5 MoE model...")
    print("=" * 60)
    model = NeuronGlm4MoeForCausalLM(traced_model_path)
    model.load(traced_model_path)

    # Try to load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(traced_model_path)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            tokenizer = None

    # Generate
    print("\n" + "=" * 60)
    print("Generating outputs...")
    print("=" * 60)

    prompt = "What is the capital of France?"

    if tokenizer is not None:
        inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        print(f"Prompt: {prompt!r}")
        print(f"Input token ids: {input_ids}")
    else:
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        print(f"Using dummy input_ids: {input_ids}")

    try:
        generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception:
        generation_config = GenerationConfig(
            max_new_tokens=10,
            do_sample=False,
            top_k=1,
        )

    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        input_ids,
        generation_config=generation_config,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
    )

    print(f"Output token ids: {outputs}")

    if tokenizer is not None:
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("Generated text:")
        for i, text in enumerate(decoded):
            print(f"  [{i}]: {text}")


def main() -> None:
    """Parse CLI arguments and run generation demo."""
    parser = argparse.ArgumentParser(description="GLM-4.5 MoE generation demo")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to HF model")
    parser.add_argument(
        "--traced-model-path",
        default=TRACED_MODEL_PATH,
        help="Path to save/load traced model",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip compilation, load existing traced model",
    )
    parser.add_argument(
        "--tp-degree", type=int, default=2, help="Tensor parallelism degree"
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    args = parser.parse_args()

    generate(
        model_path=args.model_path,
        traced_model_path=args.traced_model_path,
        skip_compile=args.skip_compile,
        tp_degree=args.tp_degree,  # Fixed: was ignored before
        seq_len=args.seq_len,  # Fixed: was ignored before
    )


if __name__ == "__main__":
    main()
