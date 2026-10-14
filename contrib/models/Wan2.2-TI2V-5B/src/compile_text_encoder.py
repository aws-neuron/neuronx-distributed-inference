"""
Wan2.2 Text Encoder (UMT5) Compilation using Model Builder V2 API.

This script uses the new ModelBuilder API instead of the deprecated parallel_model_trace.
"""
import os
import json

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # Comment this line out if using trn1/inf2
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import argparse
from torch import nn

from neuronx_distributed import ModelBuilder, NxDModel, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers import parallel_state

from transformers.models.umt5 import UMT5EncoderModel
from transformers.models.umt5.modeling_umt5 import UMT5Block, UMT5LayerSelfAttention, UMT5LayerFF

from neuron_commons import attention_wrapper, f32Wrapper
from neuron_parallel_utils import get_sharded_data, shard_umt5_self_attention, shard_umt5_ff

torch.nn.functional.scaled_dot_product_attention = attention_wrapper


class TracingUMT5WrapperV2(nn.Module):
    """Wrapper for UMT5 encoder tracing with Model Builder V2."""
    def __init__(self, t: UMT5EncoderModel, seqlen: int):
        super().__init__()
        self.t = t
        self.device = t.device

        # Precompute position bias for each block
        for block_idx in range(len(self.t.encoder.block)):
            precomputed_bias = self.t.encoder.block[block_idx].layer[0].SelfAttention.compute_bias(seqlen, seqlen)
            precomputed_bias_tp = get_sharded_data(precomputed_bias, 1)
            self.t.encoder.block[block_idx].layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias_tp

    def forward(self, text_input_ids, attention_mask):
        return self.t(
            text_input_ids,
            attention_mask=attention_mask
        )


def shard_text_encoder(text_encoder: UMT5EncoderModel, tp_degree: int):
    """Shard UMT5 encoder blocks for tensor parallelism."""
    for idx, block in enumerate(text_encoder.encoder.block):
        block: UMT5Block = block
        selfAttention: UMT5LayerSelfAttention = block.layer[0].SelfAttention
        ff: UMT5LayerFF = block.layer[1]

        # Upcast layer norms to float32 for numerical stability
        layer_norm_0 = block.layer[0].layer_norm.to(torch.float32)
        layer_norm_1 = block.layer[1].layer_norm.to(torch.float32)

        # Shard attention and feedforward layers
        block.layer[1] = shard_umt5_ff(ff)
        block.layer[0].SelfAttention = shard_umt5_self_attention(tp_degree, selfAttention)

        # Wrap layer norms
        block.layer[0].layer_norm = f32Wrapper(layer_norm_0)
        block.layer[1].layer_norm = f32Wrapper(layer_norm_1)

    # Wrap final layer norm
    final_layer_norm = text_encoder.encoder.final_layer_norm.to(torch.float32)
    text_encoder.encoder.final_layer_norm = f32Wrapper(final_layer_norm)

    return text_encoder


def save_model_config(output_path, config):
    """Save model configuration for loading."""
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_text_encoder_v2(args):
    """Compile text encoder using Model Builder V2 API."""
    batch_size = 1
    sequence_length = args.max_sequence_length
    tp_degree = args.tp_degree
    world_size = args.world_size  # Match transformer's world_size for compatibility
    compiled_models_dir = args.compiled_models_dir

    print(f"Compiling text encoder with TP={tp_degree}, world_size={world_size}, seq_len={sequence_length}")

    # Prepare sample inputs
    sample_input_ids = torch.ones((batch_size, sequence_length), dtype=torch.int64)
    sample_attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.int64)

    # Use NxDParallelState context manager - MUST match transformer's world_size!
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        print("Loading UMT5 text encoder...")
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        DTYPE = torch.bfloat16
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=DTYPE,
            cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
        )
        text_encoder.eval()

        # Save UNSHARDED state dict BEFORE sharding (for shard_checkpoint later)
        unsharded_text_encoder_state = text_encoder.state_dict()

        print("Sharding text encoder blocks...")
        text_encoder = shard_text_encoder(text_encoder, tp_degree)

        # Wrap for tracing
        model = TracingUMT5WrapperV2(text_encoder, sequence_length)

        print("Initializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model...")
        builder.trace(
            kwargs={
                "text_input_ids": sample_input_ids,
                "attention_mask": sample_attention_mask,
            },
            tag="encode",
        )

        print("Compiling model...")
        traced_model = builder.compile()

        # Save model
        output_path = f"{compiled_models_dir}/text_encoder"
        os.makedirs(output_path, exist_ok=True)

        print(f"Saving compiled model to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save sharded weights
        print("Saving sharded weights...")
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        # Get the full state dict from the sharded model (includes all buffers)
        sharded_model_state = model.state_dict()

        # Transform unsharded keys to match sharded model structure
        def get_transformed_key(key):
            new_key = "t." + key
            if "layer_norm.weight" in new_key:
                new_key = new_key.replace("layer_norm.weight", "layer_norm.original.weight")
            elif "final_layer_norm.weight" in new_key:
                new_key = new_key.replace("final_layer_norm.weight", "final_layer_norm.original.weight")
            return new_key

        # Build mapping of transformed keys to original values
        unsharded_key_map = {}
        for orig_key, value in unsharded_text_encoder_state.items():
            transformed_key = get_transformed_key(orig_key)
            # Convert layer_norm weights to float32
            if "layer_norm" in transformed_key:
                unsharded_key_map[transformed_key] = value.clone().to(torch.float32)
            else:
                unsharded_key_map[transformed_key] = value.clone()

        # Build checkpoint: use unsharded values for weights (to be properly sharded),
        # but use sharded model values for buffers
        unsharded_checkpoint = {}
        for key, sharded_value in sharded_model_state.items():
            if key in unsharded_key_map:
                # Use unsharded value (will be sharded by shard_checkpoint)
                unsharded_checkpoint[key] = unsharded_key_map[key]
            else:
                # Use value from sharded model (buffers or computed values)
                unsharded_checkpoint[key] = sharded_value.clone()

        # Use shard_checkpoint with checkpoint - it will shard parallel layer weights per rank
        shard_checkpoint(
            checkpoint=unsharded_checkpoint,
            model=model,
            start_rank=0,
            end_rank=tp_degree - 1,
            serialize_path=weights_path,
        )

        # Save config for loading
        config = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "tp_degree": tp_degree,
            "world_size": world_size,
        }
        save_model_config(output_path, config)

        print(f"Done! Text encoder saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Wan2.2 Text Encoder using Model Builder V2")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--tp_degree", type=int, default=4, help="Tensor parallelism degree")
    parser.add_argument("--world_size", type=int, default=8, help="World size (must match transformer)")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models", help="Output directory")
    args = parser.parse_args()

    compile_text_encoder_v2(args)
