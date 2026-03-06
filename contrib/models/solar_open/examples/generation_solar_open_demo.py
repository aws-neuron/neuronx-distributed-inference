"""
Solar Open MoE Generation Demo (contrib version).

This script demonstrates how to compile and run inference with the Solar Open MoE model
using neuronx-distributed-inference. It uses the contrib src path directly.

Based on examples/generation_glm4_moe_demo.py.

Usage:
    # Compile and generate:
    python generation_solar_open_demo.py

    # Skip compile (load from existing traced model):
    python generation_solar_open_demo.py --skip-compile

    # Custom paths:
    python generation_solar_open_demo.py \\
        --model-path /path/to/solar_open_model \\
        --traced-model-path /path/to/traced_model
"""

import argparse
import sys
from pathlib import Path

# Add contrib src to path so we can import solar_open directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from solar_open.modeling_solar_open import (
    SolarOpenInferenceConfig,
    NeuronSolarOpenForCausalLM,
    load_solar_open_config,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
)

# Paths - update these to your model paths
MODEL_PATH = "solar_open_tiny_random"
TRACED_MODEL_PATH = "solar_open_tiny_random_traced"

torch.manual_seed(0)

DTYPE = torch.bfloat16


def get_neuron_config(tp_degree: int = 2, seq_len: int = 64) -> MoENeuronConfig:
    """Create MoENeuronConfig for Solar Open tiny model."""
    return MoENeuronConfig(
        tp_degree=tp_degree,
        moe_tp_degree=1,
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


def generate(model_path: str, traced_model_path: str, skip_compile: bool = False):
    """Compile (if needed) and run Solar Open MoE inference."""
    if not skip_compile:
        print("=" * 60)
        print("Compiling Solar Open MoE model...")
        print("=" * 60)

        neuron_config = get_neuron_config()
        config = SolarOpenInferenceConfig(
            neuron_config,
            load_config=load_solar_open_config(model_path),
        )

        print(
            f"  Model config: hidden_size={config.hidden_size}, "
            f"n_routed_experts={config.n_routed_experts}, "
            f"n_shared_experts={config.n_shared_experts}, "
            f"num_experts_per_tok={config.num_experts_per_tok}"
        )

        model = NeuronSolarOpenForCausalLM(model_path, config)
        model.compile(traced_model_path)

        # Copy model weights to traced path so load() can find them
        # (solar_open is not in transformers; checkpoint_loader_fn looks in _name_or_path first)
        import shutil
        import os

        src_weights = os.path.join(model_path, "model.safetensors")
        dst_weights = os.path.join(traced_model_path, "model.safetensors")
        if os.path.exists(src_weights) and not os.path.exists(dst_weights):
            shutil.copy2(src_weights, dst_weights)
            print(f"Copied model weights to {traced_model_path}")

        # Save tokenizer if available
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.save_pretrained(traced_model_path)
        except Exception as e:
            print(f"Warning: could not save tokenizer: {e}")

        print(f"Model compiled and saved to {traced_model_path}")

    # Load compiled model
    print("\n" + "=" * 60)
    print("Loading compiled Solar Open MoE model...")
    print("=" * 60)
    model = NeuronSolarOpenForCausalLM(traced_model_path)
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
        # Use dummy tokens if no tokenizer
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

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Solar Open MoE generation demo")
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
    )


if __name__ == "__main__":
    main()
