import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # For trn2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # For trn2

# Compiler flags optimized for transformer models (based on Flux reference)
# Key optimizations:
# - --model-type=transformer: Enables transformer-specific optimizations
# - --enable-ccop-compute-overlap: Overlaps communication with computation
# - --auto-cast=none: Preserves bfloat16 precision
# - -O1: Basic optimization level (O2 can cause issues with some models)
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer -O1 --auto-cast=none --enable-fast-loading-neuron-binaries --tensorizer-options='--enable-ccop-compute-overlap' --internal-hlo2tensorizer-options='--fuse-dot-logistic=false' """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import argparse
import neuronx_distributed
from functools import partial
from torch import nn

from diffusers import QwenImageEditPlusPipeline
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from neuron_commons import neuron_scaled_dot_product_attention
from neuron_parallel_utils import shard_qwen_attention, shard_feedforward, shard_modulation
from neuron_rope import patch_qwenimage_rope

# Override SDPA globally for Neuron compatibility during compilation
# NOTE: NKI Flash Attention kernel doesn't work with parallel_model_trace (XLA tracing limitation)
# Using basic attention implementation instead
print("Using Neuron-compatible SDPA for compilation")
torch.nn.functional.scaled_dot_product_attention = neuron_scaled_dot_product_attention

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


class TracingTransformerWrapper(nn.Module):
    """Wrapper for tracing the transformer model."""
    def __init__(self, transformer: QwenImageTransformer2DModel, img_shapes):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device
        # Store img_shapes as a fixed attribute for tracing
        self.img_shapes = img_shapes

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        """
        Forward pass matching QwenImageTransformer2DModel signature.

        Args:
            hidden_states: (batch, num_patches, in_channels) - patchified latents
            encoder_hidden_states: (batch, text_seq_len, text_hidden_dim) - text embeddings
            timestep: (batch,) - diffusion timestep
        """
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=self.img_shapes,
            return_dict=False)


def get_transformer_model(tp_degree: int, img_shapes: list):
    """Load and shard the transformer model for tensor parallelism."""
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir=CACHE_DIR)

    # Patch RoPE to use Neuron-compatible implementation (no complex numbers)
    print("Patching RoPE for Neuron compatibility...")
    pipe.transformer = patch_qwenimage_rope(pipe.transformer)

    num_blocks = len(pipe.transformer.transformer_blocks)
    print(f"Sharding {num_blocks} transformer blocks with TP={tp_degree}")

    # Shard transformer blocks
    for block_idx, block in enumerate(pipe.transformer.transformer_blocks):
        if block_idx == 0:
            print(f"Block 0 attention heads: {block.attn.heads}")
            print(f"Block 0 to_q shape: {block.attn.to_q.weight.shape}")
            print(f"Block 0 img_mod shape: {block.img_mod[1].weight.shape}")

        # Shard attention
        block.attn = shard_qwen_attention(tp_degree, block.attn)

        if block_idx == 0:
            print(f"After sharding - Block 0 attention heads: {block.attn.heads}")

        # Shard feedforward (img_mlp and txt_mlp)
        block.img_mlp = shard_feedforward(block.img_mlp)
        block.txt_mlp = shard_feedforward(block.txt_mlp)

        # Shard modulation layers (img_mod and txt_mod) - THIS WAS MISSING!
        # These account for 6.8B params that were duplicated on every rank!
        block.img_mod = shard_modulation(block.img_mod)
        block.txt_mod = shard_modulation(block.txt_mod)

        if block_idx == 0:
            print(f"After sharding - Block 0 img_mod shape: {block.img_mod[1].weight.shape}")

        if (block_idx + 1) % 10 == 0:
            print(f"  Processed {block_idx + 1}/{num_blocks} blocks")

    print(f"All {num_blocks} blocks sharded successfully")

    transformer_wrapper = TracingTransformerWrapper(pipe.transformer, img_shapes)
    return transformer_wrapper, {}


def compile_transformer(args):
    tp_degree = args.tp_degree  # Tensor parallel degree
    os.environ["LOCAL_WORLD_SIZE"] = str(tp_degree)

    latent_height = args.height // 8
    latent_width = args.width // 8
    max_sequence_length = args.max_sequence_length
    text_hidden_size = 3584  # Text encoder hidden size
    in_channels = 64  # QwenImage transformer in_channels
    patch_size = 2  # QwenImage patch size

    # For IMAGE EDITING, the pipeline concatenates source image latents with noise latents.
    # This is handled by increasing temporal_frames to match patch_multiplier.
    # - patch_multiplier=1 (generation): temporal_frames=1, patches = 1 * 32 * 32 = 1024
    # - patch_multiplier=2 (editing): temporal_frames=2, patches = 2 * 32 * 32 = 2048
    temporal_frames = args.patch_multiplier

    # Calculate number of patches
    # QwenImage uses patch_size=2, so num_patches = T * (H/8/2) * (W/8/2)
    patch_h = latent_height // patch_size
    patch_w = latent_width // patch_size
    num_patches = temporal_frames * patch_h * patch_w

    if args.patch_multiplier > 1:
        print(f"  NOTE: Image editing mode with patch_multiplier={args.patch_multiplier}")
        print(f"  Using temporal_frames={temporal_frames} to generate RoPE for {num_patches} patches")

    # img_shapes: List of (frame, height, width) for each batch item
    # Note: height/width here are in patch space (latent_h // patch_size)
    # temporal_frames is set to patch_multiplier to match the concatenated patches
    img_shapes = [(temporal_frames, patch_h, patch_w)] * args.batch_size

    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    batch_size = args.batch_size  # Always 1, CFG runs transformer twice sequentially

    print(f"Compiling transformer with:")
    print(f"  Image size: {args.height}x{args.width}")
    print(f"  Latent size: {latent_height}x{latent_width}")
    print(f"  Patch size: {patch_size}")
    print(f"  Num patches: {num_patches}")
    print(f"  Text sequence length: {max_sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  img_shapes: {img_shapes}")

    # Sample inputs matching transformer wrapper forward signature
    # hidden_states: (batch, num_patches, in_channels)
    sample_hidden_states = torch.ones(
        (batch_size, num_patches, in_channels), dtype=torch.bfloat16)
    # encoder_hidden_states: (batch, text_seq_len, text_hidden_size)
    sample_encoder_hidden_states = torch.ones(
        (batch_size, max_sequence_length, text_hidden_size), dtype=torch.bfloat16)
    # timestep: (batch,)
    sample_timestep = torch.ones((batch_size,), dtype=torch.float32)

    get_transformer_f = partial(get_transformer_model, tp_degree, img_shapes)

    with torch.no_grad():
        sample_inputs = (
            sample_hidden_states,
            sample_encoder_hidden_states,
            sample_timestep,
        )

        compiled_transformer = neuronx_distributed.trace.parallel_model_trace(
            get_transformer_f,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/transformer",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )

        compiled_model_dir = f"{compiled_models_dir}/transformer"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)

        neuronx_distributed.trace.parallel_model_save(
            compiled_transformer, compiled_model_dir)
        print(f"Transformer compiled and saved to {compiled_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=512,
                        help="Height of generated image")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of generated image")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max sequence length for text encoder")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (always 1, CFG runs transformer twice sequentially)")
    parser.add_argument("--tp_degree", type=int, default=8,
                        help="Tensor parallel degree (8 to match language model)")
    parser.add_argument("--patch_multiplier", type=int, default=2,
                        help="Patch multiplier for image editing (2 for src+noise concat, 1 for generation)")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir",
                        help="Directory for compiler artifacts")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models",
                        help="Directory for compiled models")
    args = parser.parse_args()
    compile_transformer(args)
