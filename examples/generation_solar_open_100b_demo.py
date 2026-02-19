"""
Solar Open 100B MoE Generation Demo for NXD Inference.

Compiles and runs a 2-layer random Solar Open model configured to match the
upstage/Solar-Open-100B architecture. Uses tp_degree=4, moe_tp_degree=4,
moe_ep_degree=2 for sharding on trn2.3xlarge (4 NeuronCores).

Based on examples/generation_solar_open_demo.py.

Usage:
    # Compile and generate:
    python examples/generation_solar_open_100b_demo.py

    # Skip compile (load from existing traced model):
    python examples/generation_solar_open_100b_demo.py --skip-compile

    # Custom paths:
    python examples/generation_solar_open_100b_demo.py \\
        --model-path /path/to/solar_open_100b_random \\
        --traced-model-path /path/to/solar_open_100b_random_traced
"""

import argparse
import os
import shutil

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.models.solar_open.modeling_solar_open import (
    SolarOpenInferenceConfig,
    NeuronSolarOpenForCausalLM,
    load_solar_open_config,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
)

# Default paths - update MODEL_PATH to where you downloaded upstage/Solar-Open-100B
MODEL_PATH = "/home/ubuntu/model_hf/Solar-Open-100B"
TRACED_MODEL_PATH = "solar_open_100b_traced"

torch.manual_seed(0)

DTYPE = torch.bfloat16

# Sequence lengths: keep small to avoid OOM on trn2.3xlarge (4 NeuronCores, 96 GB HBM)
# NOTE: max_context_length must satisfy:
#   1. max_context_length * num_experts_per_tok > block_size (512) → forward_blockwise (not forward_all_experts)
#      With top_k=8: 68 * 8 = 544 > 512 ✓
#   2. max_context_length % tp_degree == 0 → required for scatter_to_process_group_spmd
#      68 % 4 = 0 ✓
SEQ_LEN = 128
MAX_CONTEXT_LENGTH = 68


def get_neuron_config() -> MoENeuronConfig:
    """
    Create MoENeuronConfig for Solar Open 100B architecture.
    - tp_degree=4: full tensor parallelism across 4 NeuronCores
    - moe_tp_degree=4: MoE expert tensor parallelism (EP=1 for stability)
    - moe_ep_degree=1: no expert parallelism (EP+token-gen not supported by library)

    Note: moe_ep_degree=2 was attempted but neuronx_distributed ExpertMLPsV2
    raises NotImplementedError for EP + token generation (selective loading).
    Using moe_ep_degree=1, moe_tp_degree=4 instead (fully TP-sharded experts).
    """
    return MoENeuronConfig(
        tp_degree=4,
        moe_tp_degree=4,
        moe_ep_degree=1,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=SEQ_LEN,
        max_context_length=MAX_CONTEXT_LENGTH,
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
    """Compile (if needed) and run Solar Open 100B MoE inference."""
    if not skip_compile:
        print("=" * 60)
        print("Compiling Solar Open 100B MoE model...")
        print(
            f"  Architecture: hidden_size=4096, n_routed_experts=128, n_shared_experts=1"
        )
        print(f"  Sharding: tp_degree=4, moe_tp_degree=4, moe_ep_degree=2")
        print(f"  Layers: 2 (reduced from 48 for fast testing)")
        print(f"  YaRN RoPE: factor=2.0, original_max_position_embeddings=65536")
        print("=" * 60)

        neuron_config = get_neuron_config()
        config = SolarOpenInferenceConfig(
            neuron_config,
            load_config=load_solar_open_config(model_path),
        )

        print(
            f"  Config loaded: hidden_size={config.hidden_size}, "
            f"n_routed_experts={config.n_routed_experts}, "
            f"n_shared_experts={config.n_shared_experts}, "
            f"num_experts_per_tok={config.num_experts_per_tok}, "
            f"rope_scaling={getattr(config, 'rope_scaling', None)}"
        )

        model = NeuronSolarOpenForCausalLM(model_path, config)
        model.compile(traced_model_path)

        # Copy model weights to traced path for loading
        src_weights = os.path.join(model_path, "model.safetensors")
        dst_weights = os.path.join(traced_model_path, "model.safetensors")
        if os.path.exists(src_weights) and not os.path.exists(dst_weights):
            shutil.copy2(src_weights, dst_weights)
            print(f"Copied model weights to {traced_model_path}")

        # Copy config.json
        src_config = os.path.join(model_path, "config.json")
        dst_config = os.path.join(traced_model_path, "config.json")
        if os.path.exists(src_config) and not os.path.exists(dst_config):
            shutil.copy2(src_config, dst_config)
            print(f"Copied config.json to {traced_model_path}")

        # Save tokenizer if available (Solar-Open-100B uses upstage tokenizer)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.save_pretrained(traced_model_path)
            print(f"Saved tokenizer to {traced_model_path}")
        except Exception as e:
            print(f"Warning: could not save tokenizer: {e}")

        print(f"\nModel compiled and saved to {traced_model_path}")

    # Load compiled model
    print("\n" + "=" * 60)
    print("Loading compiled Solar Open 100B MoE model...")
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
        # Use dummy tokens if no tokenizer (random model has no tokenizer)
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
    parser = argparse.ArgumentParser(
        description="Solar Open 100B MoE generation demo (tp_degree=4, moe_tp_degree=4, moe_ep_degree=2)"
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to HF model (or random model created by create_solar_open_100b_random.py)",
    )
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
    args = parser.parse_args()

    generate(
        model_path=args.model_path,
        traced_model_path=args.traced_model_path,
        skip_compile=args.skip_compile,
    )


if __name__ == "__main__":
    main()
