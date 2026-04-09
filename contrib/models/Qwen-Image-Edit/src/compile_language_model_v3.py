"""
Language Model Compilation using ModelBuilder API (V3) for V3 CP Compatibility.

This script compiles the Qwen2.5-VL Language Model using ModelBuilder API with
tp_degree=4 and world_size=8 to be compatible with the V3 CP transformer.

Key features:
- Uses ModelBuilder API (NxDModel) for compilation
- Configuration: tp_degree=4, world_size=8 (matching V3 CP transformer)
- TP=4 is perfect for Qwen2.5-VL GQA: 28Q/4=7 heads/rank, 4KV/4=1 head/rank
- No Context Parallel needed (language model processes full sequence)

Usage:
    python compile_language_model_v3.py --max_sequence_length 1024
"""

import os
import json
import gc

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Compiler flags
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import argparse

from diffusers import QwenImageEditPlusPipeline

# ModelBuilder imports
from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers import parallel_state

from neuron_parallel_utils import (
    shard_qwen2_attention,
    shard_qwen2_mlp,
    get_sharded_data,
)

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


def load_pipeline(dtype=torch.bfloat16):
    """Load pipeline with appropriate kwargs."""
    load_kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    return QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, **load_kwargs)


class f32Wrapper(nn.Module):
    """Wrapper to run normalization layers in float32 for numerical stability."""
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, *args, **kwargs):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y, *args, **kwargs)
        return output.type(t)


def upcast_norms_to_f32(module):
    """Upcast normalization layers to float32 for numerical stability."""
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.LayerNorm,)):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        elif 'RMSNorm' in child.__class__.__name__:
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


class NeuronLanguageModelV3(nn.Module):
    """
    Neuron-optimized Qwen2.5-VL Language Model for V3 CP compatibility.

    Uses ModelBuilder API with tp_degree=4, world_size=8.

    Key differences from compile_text_encoder.py:
    - Uses ModelBuilder API instead of parallel_model_trace
    - world_size=8 to match transformer (even though CP is not used for language model)
    - TP=4 for perfect GQA alignment (28Q/4=7, 4KV/4=1 - no padding needed!)

    Note: Unlike V3 CP transformer which splits sequence, language model processes
    full sequence on all ranks. The world_size=8 is for compatibility only.

    IMPORTANT: We keep the full language_model structure and just shard the layers,
    rather than recreating the forward pass. This ensures position_embeddings are
    properly computed from position_ids by the original model's rotary_emb.
    """

    def __init__(self, original_language_model, tp_degree):
        super().__init__()

        self.tp_degree = tp_degree

        # Keep the full language model (we'll modify its layers in-place)
        self.language_model = original_language_model

        # Copy config for reference
        self.config = original_language_model.config

        # Get model structure info
        self.hidden_size = self.config.hidden_size  # 3584
        self.num_hidden_layers = self.config.num_hidden_layers  # 28

        print(f"  Language model config:")
        print(f"    hidden_size: {self.hidden_size}")
        print(f"    num_hidden_layers: {self.num_hidden_layers}")
        print(f"    num_attention_heads: {self.config.num_attention_heads}")  # 28
        print(f"    num_key_value_heads: {self.config.num_key_value_heads}")  # 4

        # Shard the layers in-place
        for i, layer in enumerate(self.language_model.layers):
            # Shard attention
            layer.self_attn = shard_qwen2_attention(tp_degree, layer.self_attn)
            # Shard MLP
            layer.mlp = shard_qwen2_mlp(layer.mlp)
            if i == 0:
                print(f"  Sharded layer 0 attention and MLP")

        print(f"  Sharded all {len(self.language_model.layers)} layers")

        # Upcast norms to float32 for numerical stability
        upcast_norms_to_f32(self.language_model)

    def forward(self, inputs_embeds, attention_mask, position_ids):
        """
        Forward pass for language model.

        Args:
            inputs_embeds: (batch, seq_len, hidden_size) - combined text+vision embeddings
            attention_mask: (batch, seq_len) - 1 for valid tokens, 0 for padding
            position_ids: (3, batch, seq_len) - 3D position IDs for M-RoPE
                         Dims: [t (temporal), h (height), w (width)] x batch x seq_len

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        # Call the full language model, which handles:
        # 1. Computing position_embeddings from position_ids via rotary_emb
        # 2. Creating the attention mask
        # 3. Running through all layers
        # 4. Final layer norm
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.last_hidden_state


class TracingWrapper(nn.Module):
    """Wrapper for ModelBuilder tracing."""

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        return self.language_model(inputs_embeds, attention_mask, position_ids)


def compile_language_model_v3(args):
    """
    Compile Language Model using ModelBuilder API.

    Configuration:
    - tp_degree=4: Perfect for GQA (28Q/4=7, 4KV/4=1)
    - world_size=8: Matches V3 CP transformer (even though CP is not used)
    """
    tp_degree = 4  # Fixed: perfect GQA alignment
    world_size = 8  # Fixed: match V3 CP transformer

    batch_size = args.batch_size
    sequence_length = args.max_sequence_length
    hidden_size = 3584  # Qwen2.5-VL hidden size

    print("=" * 60)
    print("Compiling Language Model V3 (ModelBuilder API)")
    print("=" * 60)
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  TP degree: {tp_degree}")
    print(f"  World size: {world_size}")
    print(f"  GQA: 28 Q heads / 4 = 7 per rank, 4 KV heads / 4 = 1 per rank")
    print("")

    # Sample inputs
    sample_inputs_embeds = torch.randn(
        batch_size, sequence_length, hidden_size, dtype=torch.bfloat16
    )
    sample_attention_mask = torch.ones(
        batch_size, sequence_length, dtype=torch.int64
    )
    # 3D position_ids for M-RoPE: (3, batch, seq_len)
    # For tracing, use simple sequential positions (text-only pattern)
    sample_position_ids = torch.arange(sequence_length).view(1, 1, -1).expand(3, batch_size, -1).clone()

    print(f"Sample input shapes:")
    print(f"  inputs_embeds: {sample_inputs_embeds.shape}")
    print(f"  attention_mask: {sample_attention_mask.shape}")
    print(f"  position_ids: {sample_position_ids.shape}")
    print("")

    # Use NxDParallelState context for compilation
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        print("Loading model...")
        pipe = load_pipeline(torch.bfloat16)

        # Extract language model
        original_language_model = pipe.text_encoder.model.language_model

        # Save unsharded state dict before modifications
        print("Saving unsharded state dict...")
        unsharded_state = original_language_model.state_dict()

        # Create Neuron language model with sharding
        print(f"\nCreating Neuron language model (sharding layers with TP={tp_degree})...")
        neuron_language_model = NeuronLanguageModelV3(
            original_language_model, tp_degree
        )
        neuron_language_model = neuron_language_model.to(torch.bfloat16)
        neuron_language_model.eval()

        # Clear pipeline to save memory (language model is now owned by neuron_language_model)
        del pipe
        gc.collect()

        # Wrap for tracing
        model = TracingWrapper(neuron_language_model)

        print("\nInitializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model...")
        builder.trace(
            kwargs={
                "inputs_embeds": sample_inputs_embeds,
                "attention_mask": sample_attention_mask,
                "position_ids": sample_position_ids,
            },
            tag="inference",
        )

        print("Compiling model...")
        # NOTE: Using -O1 instead of -O2 because -O2 can cause numerical issues in some cases
        compile_args = "--model-type=transformer -O1 --auto-cast=none"
        traced_model = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save
        output_path = f"{args.compiled_models_dir}/language_model_v3"
        os.makedirs(output_path, exist_ok=True)

        print(f"\nSaving to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        # Prepare checkpoint for sharding
        print("Preparing checkpoint...")
        checkpoint = {}
        for key, value in model.state_dict().items():
            # Use unsharded weights where available
            # Key format: language_model.language_model.layers.X... -> layers.X...
            # (TracingWrapper.language_model -> NeuronLanguageModelV3.language_model -> Qwen2_5_VLTextModel)
            orig_key = key.replace("language_model.language_model.", "", 1)
            if orig_key in unsharded_state:
                checkpoint[key] = unsharded_state[orig_key].clone()
            else:
                checkpoint[key] = value.clone()

        # Shard checkpoint
        print("Sharding weights...")
        shard_checkpoint(
            checkpoint=checkpoint,
            model=model,
            serialize_path=weights_path,
        )

        # Post-process checkpoints: remove master_weight and add inv_freq
        print("\nPost-processing checkpoints...")
        from safetensors.torch import load_file, save_file

        # Collect inv_freq buffers from original model (they are not in state_dict)
        inv_freq_buffers = {}
        for name, buf in neuron_language_model.language_model.named_buffers():
            if 'inv_freq' in name:
                full_key = f"language_model.language_model.{name}"
                inv_freq_buffers[full_key] = buf.to(torch.bfloat16).clone()
        print(f"  Collected {len(inv_freq_buffers)} inv_freq buffers")

        for rank in range(tp_degree):
            shard_file = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
            if not os.path.exists(shard_file):
                print(f"  WARNING: {shard_file} not found!")
                continue

            # Load checkpoint
            data = dict(load_file(shard_file))
            original_count = len(data)
            original_size = sum(v.numel() * v.element_size() for v in data.values())

            # Remove master_weight tensors (they duplicate the sharded weights)
            cleaned = {k: v for k, v in data.items() if 'master_weight' not in k}

            # Add inv_freq buffers
            cleaned.update(inv_freq_buffers)

            cleaned_size = sum(v.numel() * v.element_size() for v in cleaned.values())

            # Save optimized checkpoint
            save_file(cleaned, shard_file)
            print(f"  tp{rank}: {original_count} -> {len(cleaned)} tensors, "
                  f"{original_size/1e9:.2f}GB -> {cleaned_size/1e9:.2f}GB")

        # Save config
        config = {
            "max_sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
        }
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to {config_path}")

        print("\n" + "=" * 60)
        print("Compilation complete!")
        print("=" * 60)
        print(f"Model saved to: {output_path}")
        print(f"  - nxd_model.pt")
        print(f"  - weights/tp{{0,1,2,3}}_sharded_checkpoint.safetensors")
        print(f"  - config.json")
        print("")
        print("To use with V3 CP transformer:")
        print("  python run_qwen_image_edit.py --use_v3_cp --use_v3_language_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Language Model V3 using ModelBuilder API")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model (local dir or HuggingFace ID). If not set, uses MODEL_ID with CACHE_DIR")
    parser.add_argument("--max_sequence_length", type=int, default=1024,
                        help="Maximum sequence length for compilation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for language model (default: 1)")
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models",
                        help="Directory to save compiled models")
    parser.add_argument("--compiler_workdir", type=str, default="/opt/dlami/nvme/compiler_workdir",
                        help="Directory for compiler artifacts")
    args = parser.parse_args()

    # Override MODEL_ID and CACHE_DIR if model_path is provided
    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    compile_language_model_v3(args)
