"""
Text Encoder Compilation for Qwen-Image-Edit-2509

The text encoder (Qwen2.5-VL) is a multimodal vision-language model with:
1. Vision Encoder (Qwen2_5_VisionTransformerPretrainedModel) - 32 blocks
2. Language Model (Qwen2_5_VLTextModel) - 28 layers

This script compiles both components for Trainium2 using tensor parallelism.
"""

import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # For trn2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # For trn2

compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries """  #  --verbose=INFO
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import argparse
import torch_neuronx
import neuronx_distributed
from functools import partial
from torch import nn

from diffusers import QwenImageEditPlusPipeline
from neuron_commons import attention_wrapper, f32Wrapper
from neuron_parallel_utils import (
    shard_qwen2_attention, shard_qwen2_mlp,
    shard_vision_attention, shard_vision_mlp
)

# Override SDPA
torch.nn.functional.scaled_dot_product_attention = attention_wrapper

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


def load_pipeline(dtype=torch.bfloat16):
    """Load pipeline with appropriate kwargs based on MODEL_ID and CACHE_DIR."""
    load_kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    return QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, **load_kwargs)


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper for the Qwen2.5-VL Vision Encoder.
    Compiles the vision transformer that processes image patches.
    """
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values, grid_thw):
        """
        Args:
            pixel_values: (num_patches, 3*temporal*patch_h*patch_w) - flattened patches
            grid_thw: (num_images, 3) - temporal, height, width in grid space
        Returns:
            image_embeds: (total_patches, hidden_size)
        """
        return self.visual(pixel_values, grid_thw)


class LanguageModelWrapper(nn.Module):
    """
    Wrapper for the Qwen2.5-VL Language Model.
    Processes the combined text and vision embeddings.

    IMPORTANT: Must accept position_ids for M-RoPE (Multimodal RoPE) to work correctly.
    Qwen2.5-VL uses 3D position_ids with shape [3, batch, seq_len] for:
    - t (temporal): frame index for video, 0 for images
    - h (height): spatial row position for image tokens
    - w (width): spatial column position for image tokens
    """
    def __init__(self, language_model, embed_tokens):
        super().__init__()
        self.language_model = language_model
        self.embed_tokens = embed_tokens

    def forward(self, inputs_embeds, attention_mask, position_ids):
        """
        Args:
            inputs_embeds: (batch, seq_len, hidden_size) - combined text+vision embeddings
            attention_mask: (batch, seq_len)
            position_ids: (3, batch, seq_len) - 3D position IDs for M-RoPE
        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.last_hidden_state


class FullTextEncoderWrapper(nn.Module):
    """
    Full wrapper for the Qwen2.5-VL text encoder with fixed shapes.
    This is used when compiling the complete text encoder for image editing.

    For simplicity in compilation, we use a fixed sequence length and image size.
    """
    def __init__(self, text_encoder, max_seq_len, num_image_tokens):
        super().__init__()
        self.text_encoder = text_encoder
        self.config = text_encoder.config
        self.max_seq_len = max_seq_len
        self.num_image_tokens = num_image_tokens

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        """
        Fixed-shape forward pass for tracing.

        Args:
            input_ids: (batch, text_seq_len)
            attention_mask: (batch, total_seq_len)
            pixel_values: (num_patches, channels) - preprocessed image patches
            image_grid_thw: (num_images, 3) - grid dimensions
        Returns:
            hidden_states: (batch, total_seq_len, hidden_size)
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states[-1]


def upcast_norms_to_f32(module):
    """Upcast normalization layers to float32 for numerical stability."""
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.LayerNorm,)):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        # Handle RMSNorm (Qwen uses this)
        elif 'RMSNorm' in child.__class__.__name__:
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


def get_vision_encoder(tp_degree: int):
    """Load and prepare vision encoder for tracing."""
    pipe = load_pipeline(torch.bfloat16)

    visual = pipe.text_encoder.model.visual
    visual.eval()
    upcast_norms_to_f32(visual)

    return VisionEncoderWrapper(visual), {}


def get_language_model(tp_degree: int):
    """Load and shard language model for tensor parallelism."""
    pipe = load_pipeline(torch.bfloat16)

    text_encoder = pipe.text_encoder
    lang_model = text_encoder.model.language_model
    embed_tokens = lang_model.embed_tokens
    lang_model.eval()

    # Shard the language model layers
    for layer in lang_model.layers:
        if hasattr(layer, 'self_attn'):
            layer.self_attn = shard_qwen2_attention(tp_degree, layer.self_attn)
        if hasattr(layer, 'mlp'):
            layer.mlp = shard_qwen2_mlp(layer.mlp)

    upcast_norms_to_f32(lang_model)

    return LanguageModelWrapper(lang_model, embed_tokens), {}


def compile_vision_encoder(args):
    """
    Compile the Vision Encoder component (single device mode).

    The vision encoder processes image patches and outputs vision embeddings.
    Input shape depends on image size and patch configuration.

    Note: For better memory distribution, use compile_vision_encoder_tp() with --vision_tp flag.
    """
    batch_size = 1
    image_size = args.image_size
    patch_size = 14
    temporal_patch_size = 2
    spatial_merge_size = 2

    # Validate image_size
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size}). "
            f"Valid sizes: 224, 336, 448, 560, etc.")

    num_patches_per_side = image_size // patch_size
    if num_patches_per_side % spatial_merge_size != 0:
        raise ValueError(
            f"image_size / patch_size ({num_patches_per_side}) must be divisible by "
            f"spatial_merge_size ({spatial_merge_size}). "
            f"Valid image sizes: 224, 336, 448, 560, etc.")

    # Calculate number of patches for a single image
    # Qwen2.5-VL uses Conv3d with kernel (temporal_patch_size, patch_size, patch_size)
    # For a single frame: num_patches = (H/patch_size) * (W/patch_size)
    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size
    num_patches = num_patches_h * num_patches_w

    # pixel_values shape for the vision encoder
    # After preprocessing, it's (num_patches, 3 * temporal_patch_size * patch_size * patch_size)
    channels_per_patch = 3 * temporal_patch_size * patch_size * patch_size  # 3*2*14*14 = 1176

    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir

    # Always use float32 for vision encoder (required for accuracy)
    dtype = torch.float32

    print("=" * 50)
    print("Compiling Vision Encoder (Single Device, float32)")
    print("=" * 50)
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Num patches: {num_patches}")
    print(f"  Channels per patch: {channels_per_patch}")
    print(f"  Dtype: float32 (required for accuracy)")

    pipe = load_pipeline(dtype)

    visual = pipe.text_encoder.model.visual
    visual.eval()

    # Keep everything in float32 for maximum precision

    # Sample inputs
    # pixel_values: (total_patches, patch_dim)
    sample_pixel_values = torch.ones((num_patches, channels_per_patch), dtype=dtype)
    # grid_thw: (num_images, 3) - temporal, height, width in grid units
    sample_grid_thw = torch.tensor([[1, num_patches_h, num_patches_w]], dtype=torch.int64)

    vision_wrapper = VisionEncoderWrapper(visual)

    # Use --auto-cast=none to prevent precision loss
    vision_compiler_flags = compiler_flags + " --auto-cast=none"

    with torch.no_grad():
        try:
            compiled_vision = torch_neuronx.trace(
                vision_wrapper,
                (sample_pixel_values, sample_grid_thw),
                compiler_workdir=f"{compiler_workdir}/vision_encoder",
                compiler_args=vision_compiler_flags,
                inline_weights_to_neff=False
            )

            # Save to vision_encoder/ directory
            vision_dir = f"{compiled_models_dir}/vision_encoder"
            if not os.path.exists(vision_dir):
                os.makedirs(vision_dir)
            torch.jit.save(compiled_vision, f"{vision_dir}/model.pt")
            print(f"Vision encoder (float32) compiled and saved to {vision_dir}")
            return True

        except Exception as e:
            print(f"Vision encoder compilation failed: {e}")
            return False


def get_vision_encoder_tp(tp_degree: int, image_size: int):
    """Load and shard vision encoder for tensor parallelism."""
    pipe = load_pipeline(torch.bfloat16)

    visual = pipe.text_encoder.model.visual
    visual.eval()

    # Shard the vision encoder blocks
    for block in visual.blocks:
        if hasattr(block, 'attn'):
            block.attn = shard_vision_attention(tp_degree, block.attn)
        if hasattr(block, 'mlp'):
            block.mlp = shard_vision_mlp(block.mlp)

    upcast_norms_to_f32(visual)

    return VisionEncoderWrapper(visual), {}


def compile_vision_encoder_tp(args):
    """
    Compile the Vision Encoder with tensor parallelism.

    NOTE: The Qwen2.5-VL vision encoder has dimensions that are NOT divisible by 8.
    Specifically, the fused QKV projection has dimension 3420 (1140 * 3).
    - 3420 / 8 = 427.5 (NOT divisible)
    - 3420 / 4 = 855 (divisible)
    - 3420 / 2 = 1710 (divisible)

    Since transformer and language model require TP=8, and mixing different TP degrees
    causes world_size conflicts, vision encoder TP is NOT recommended.

    This function will attempt TP compilation but is expected to fail with TP=8.
    Use single-device compilation (--vision_only without --vision_tp) instead.
    """
    batch_size = 1
    image_size = args.image_size
    patch_size = 14
    temporal_patch_size = 2
    spatial_merge_size = 2
    tp_degree = args.tp_degree

    # Check if vision encoder dimensions are compatible with TP degree
    vision_embed_dim = 1140  # Qwen2.5-VL vision encoder embed_dim
    qkv_dim = vision_embed_dim * 3  # 3420

    if qkv_dim % tp_degree != 0:
        print("=" * 60)
        print("WARNING: Vision Encoder TP Compilation Not Supported")
        print("=" * 60)
        print(f"  Vision encoder QKV dimension: {qkv_dim}")
        print(f"  Requested TP degree: {tp_degree}")
        print(f"  {qkv_dim} is NOT divisible by {tp_degree}")
        print("")
        print("  The Qwen2.5-VL vision encoder has dimensions incompatible with TP=8.")
        print("  Falling back to single-device compilation...")
        print("")

        # Fall back to single device compilation
        return compile_vision_encoder(args)

    os.environ["LOCAL_WORLD_SIZE"] = str(tp_degree)

    # Validate image_size
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size}). "
            f"Valid sizes: 224, 336, 448, 560, etc.")

    num_patches_per_side = image_size // patch_size
    if num_patches_per_side % spatial_merge_size != 0:
        raise ValueError(
            f"image_size / patch_size ({num_patches_per_side}) must be divisible by "
            f"spatial_merge_size ({spatial_merge_size}). "
            f"Valid image sizes: 224, 336, 448, 560, etc.")

    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size
    num_patches = num_patches_h * num_patches_w

    channels_per_patch = 3 * temporal_patch_size * patch_size * patch_size  # 1176

    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    dtype = torch.bfloat16

    print("=" * 50)
    print("Compiling Vision Encoder with Tensor Parallelism")
    print("=" * 50)
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Num patches: {num_patches}")
    print(f"  Channels per patch: {channels_per_patch}")
    print(f"  TP degree: {tp_degree}")

    get_vision_f = partial(get_vision_encoder_tp, tp_degree, image_size)

    # Sample inputs
    sample_pixel_values = torch.ones((num_patches, channels_per_patch), dtype=dtype)
    sample_grid_thw = torch.tensor([[1, num_patches_h, num_patches_w]], dtype=torch.int64)

    sample_inputs = (sample_pixel_values, sample_grid_thw)

    with torch.no_grad():
        try:
            compiled_vision = neuronx_distributed.trace.parallel_model_trace(
                get_vision_f,
                sample_inputs,
                compiler_workdir=f"{compiler_workdir}/vision_encoder_tp",
                compiler_args=compiler_flags,
                tp_degree=tp_degree,
                inline_weights_to_neff=False
            )

            vision_dir = f"{compiled_models_dir}/vision_encoder_tp"
            if not os.path.exists(vision_dir):
                os.makedirs(vision_dir)

            neuronx_distributed.trace.parallel_model_save(
                compiled_vision, vision_dir)
            print(f"Vision encoder (TP={tp_degree}) compiled and saved to {vision_dir}")
            return True

        except Exception as e:
            print(f"Vision encoder TP compilation failed: {e}")
            print("Falling back to single-device compilation...")
            return compile_vision_encoder(args)


def compile_language_model(args):
    """
    Compile the Language Model component with tensor parallelism.

    The language model processes text tokens combined with vision embeddings.

    Qwen2.5-VL-7B GQA configuration:
    - 28 Q heads, 4 KV heads -> each KV head shared by 7 Q heads

    Supported TP degrees:
    - TP=4: Standard sharding (7 Q heads, 1 KV head per rank)
    - TP=8: KV replication mode (Q padded to 32 -> 4 per rank, KV replicated -> 1 per rank)

    The KV replication logic in shard_qwen2_attention handles TP=8 correctly by:
    1. Padding Q heads from 28 to 32 (divisible by 8)
    2. Replicating each KV head to pairs of ranks
    3. Updating num_key_value_groups to 4 (4 Q heads / 1 KV head per rank)
    """
    batch_size = 1
    sequence_length = args.max_sequence_length
    hidden_size = 3584  # Qwen2.5-VL hidden size

    # Use language-specific TP degree
    tp_degree = getattr(args, 'language_tp_degree', 8)

    # Validate TP degree
    num_kv_heads = 4
    if tp_degree > num_kv_heads and tp_degree % num_kv_heads != 0:
        raise ValueError(
            f"For TP={tp_degree} > num_kv_heads={num_kv_heads}, "
            f"tp_degree must be divisible by num_kv_heads. "
            f"Valid TP degrees: 1, 2, 4, 8"
        )

    if tp_degree == 8:
        print("=" * 60)
        print("INFO: Using KV Head Replication Mode (TP=8)")
        print("=" * 60)
        print(f"  Q heads: 28 -> padded to 32 -> 4 per rank")
        print(f"  KV heads: 4 -> replicated -> 1 per rank")
        print(f"  num_key_value_groups: 4 (Q_per_rank / KV_per_rank)")
        print("=" * 60)

    os.environ["LOCAL_WORLD_SIZE"] = str(tp_degree)

    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir

    print("=" * 50)
    print("Compiling Language Model")
    print("=" * 50)
    print(f"  Sequence length: {sequence_length}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  TP degree: {tp_degree}")

    get_lang_model_f = partial(get_language_model, tp_degree)

    with torch.no_grad():
        # inputs_embeds: (batch, seq_len, hidden_size)
        sample_inputs_embeds = torch.ones(
            (batch_size, sequence_length, hidden_size), dtype=torch.bfloat16)
        # attention_mask: (batch, seq_len)
        sample_attention_mask = torch.ones(
            (batch_size, sequence_length), dtype=torch.int64)
        # position_ids: (3, batch, seq_len) - 3D for M-RoPE
        # For tracing, use simple sequential positions (text-only pattern)
        sample_position_ids = torch.arange(sequence_length).view(1, 1, -1).expand(3, batch_size, -1).clone()

        sample_inputs = (sample_inputs_embeds, sample_attention_mask, sample_position_ids)

        try:
            compiled_lang_model = neuronx_distributed.trace.parallel_model_trace(
                get_lang_model_f,
                sample_inputs,
                compiler_workdir=f"{compiler_workdir}/language_model",
                compiler_args=compiler_flags,
                tp_degree=tp_degree,
                inline_weights_to_neff=False
            )

            lang_model_dir = f"{compiled_models_dir}/language_model"
            if not os.path.exists(lang_model_dir):
                os.makedirs(lang_model_dir)

            neuronx_distributed.trace.parallel_model_save(
                compiled_lang_model, lang_model_dir)
            print(f"Language model compiled and saved to {lang_model_dir}")
            return True

        except Exception as e:
            print(f"Language model compilation failed: {e}")
            return False


def compile_text_encoder_full(args):
    """
    Compile the full text encoder (vision + language) with fixed shapes.
    This is more complex but allows end-to-end compilation.
    """
    batch_size = 1
    text_seq_len = args.max_sequence_length
    image_size = args.image_size
    patch_size = 14
    spatial_merge_size = 2  # Qwen2.5-VL spatial merge

    # Calculate image token count after spatial merge
    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size
    merged_h = num_patches_h // spatial_merge_size
    merged_w = num_patches_w // spatial_merge_size
    num_image_tokens = merged_h * merged_w

    total_seq_len = text_seq_len + num_image_tokens
    tp_degree = args.tp_degree  # Use configurable TP degree (default=8)

    os.environ["LOCAL_WORLD_SIZE"] = str(tp_degree)

    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir

    print("=" * 50)
    print("Compiling Full Text Encoder")
    print("=" * 50)
    print(f"  Image size: {image_size}")
    print(f"  Text sequence length: {text_seq_len}")
    print(f"  Image tokens: {num_image_tokens}")
    print(f"  Total sequence length: {total_seq_len}")
    print(f"  TP degree: {tp_degree}")

    def get_full_text_encoder(tp_degree):
        pipe = load_pipeline(torch.bfloat16)

        text_encoder = pipe.text_encoder
        text_encoder.eval()

        # Shard language model
        lang_model = text_encoder.model.language_model
        for layer in lang_model.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn = shard_qwen2_attention(tp_degree, layer.self_attn)
            if hasattr(layer, 'mlp'):
                layer.mlp = shard_qwen2_mlp(layer.mlp)

        upcast_norms_to_f32(text_encoder)

        return FullTextEncoderWrapper(text_encoder, total_seq_len, num_image_tokens), {}

    get_encoder_f = partial(get_full_text_encoder, tp_degree)

    # Calculate pixel_values shape
    num_patches = num_patches_h * num_patches_w
    channels_per_patch = 3 * 2 * patch_size * patch_size  # 1176

    with torch.no_grad():
        sample_inputs = (
            torch.ones((batch_size, text_seq_len), dtype=torch.int64),
            torch.ones((batch_size, total_seq_len), dtype=torch.int64),
            torch.ones((num_patches, channels_per_patch), dtype=torch.bfloat16),
            torch.tensor([[1, num_patches_h, num_patches_w]], dtype=torch.int64),
        )

        try:
            compiled_encoder = neuronx_distributed.trace.parallel_model_trace(
                get_encoder_f,
                sample_inputs,
                compiler_workdir=f"{compiler_workdir}/text_encoder",
                compiler_args=compiler_flags,
                tp_degree=tp_degree,
                inline_weights_to_neff=False
            )

            encoder_dir = f"{compiled_models_dir}/text_encoder"
            if not os.path.exists(encoder_dir):
                os.makedirs(encoder_dir)

            neuronx_distributed.trace.parallel_model_save(
                compiled_encoder, encoder_dir)
            print(f"Full text encoder compiled and saved to {encoder_dir}")
            return True

        except Exception as e:
            print(f"Full text encoder compilation failed: {e}")
            print("Try compiling vision encoder and language model separately.")
            return False


def run_in_subprocess(func_name, args, vision_tp=False):
    """Run a compilation function in a separate subprocess to avoid XLA conflicts."""
    import subprocess
    import sys

    cmd = [
        sys.executable, __file__,
        "--mode", "separate",
        "--image_size", str(args.image_size),
        "--max_sequence_length", str(args.max_sequence_length),
        "--compiler_workdir", args.compiler_workdir,
        "--compiled_models_dir", args.compiled_models_dir,
        "--tp_degree", str(args.tp_degree),
        "--language_tp_degree", str(getattr(args, 'language_tp_degree', 4)),
    ]

    # Pass model_path if set
    if getattr(args, 'model_path', None):
        cmd.extend(["--model_path", args.model_path])

    if func_name == "vision":
        cmd.append("--vision_only")
        if vision_tp:
            cmd.append("--vision_tp")
    elif func_name == "language":
        cmd.append("--language_only")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="separate",
                        choices=["separate", "full"],
                        help="Compilation mode: 'separate' compiles vision and language separately, "
                             "'full' compiles the entire text encoder together")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max text sequence length")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size for vision encoder. Must be divisible by 14 (patch_size) "
                             "and result in even grid for spatial merge. Valid: 224, 336, 448, 560")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir",
                        help="Directory for compiler artifacts")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models",
                        help="Directory for compiled models")
    parser.add_argument("--vision_only", action="store_true",
                        help="Only compile vision encoder")
    parser.add_argument("--vision_tp", action="store_true",
                        help="Compile vision encoder with tensor parallelism (TP=8) instead of single device. "
                             "Helps reduce per-device memory usage.")
    parser.add_argument("--language_only", action="store_true",
                        help="Only compile language model")
    parser.add_argument("--use_subprocess", action="store_true",
                        help="Run each compilation in separate subprocess (avoids XLA conflicts)")
    parser.add_argument("--tp_degree", type=int, default=8,
                        help="Tensor parallel degree for vision encoder TP mode (default=8)")
    parser.add_argument("--language_tp_degree", type=int, default=8,
                        help="Tensor parallel degree for language model. "
                             "TP=4: Standard sharding. TP=8: KV head replication mode. "
                             "Default=8 to match transformer TP degree.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model (local dir or HuggingFace ID). If not set, uses MODEL_ID with CACHE_DIR")
    # Note: Vision encoder is always compiled in float32 for accuracy (required)
    args = parser.parse_args()

    # Override MODEL_ID and CACHE_DIR if model_path is provided
    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    if args.mode == "separate":
        # If specific component requested, run directly
        if args.vision_only:
            if args.vision_tp:
                print("\n[Vision Only] Compiling Vision Encoder with TP...")
                compile_vision_encoder_tp(args)
            else:
                print("\n[Vision Only] Compiling Vision Encoder (single device)...")
                compile_vision_encoder(args)
        elif args.language_only:
            print("\n[Language Only] Compiling Language Model...")
            compile_language_model(args)
        elif args.use_subprocess:
            # Run in separate subprocesses to avoid XLA initialization conflicts
            if args.vision_tp:
                print("\n[Step 1] Compiling Vision Encoder with TP (subprocess)...")
            else:
                print("\n[Step 1] Compiling Vision Encoder (subprocess)...")
            vision_success = run_in_subprocess("vision", args, vision_tp=args.vision_tp)

            print("\n[Step 2] Compiling Language Model (subprocess)...")
            lang_success = run_in_subprocess("language", args)

            if vision_success and lang_success:
                print("\n" + "=" * 50)
                print("Text Encoder Compilation Complete!")
                print("=" * 50)
                if args.vision_tp:
                    print("  Vision Encoder: TP={} (saved to vision_encoder_tp/)".format(args.tp_degree))
                else:
                    print("  Vision Encoder: Single device (saved to vision_encoder/)")
                print("  Language Model: TP={} (saved to language_model/)".format(args.language_tp_degree))
        else:
            # Default: try sequential but warn about XLA issue
            print("\nNOTE: If language model compilation fails with 'Runtime is already initialized',")
            print("      run with --use_subprocess flag or compile separately:")
            print("      python compile_text_encoder.py --vision_only [--vision_tp]")
            print("      python compile_text_encoder.py --language_only")
            print("")

            if args.vision_tp:
                print("\n[Step 1] Compiling Vision Encoder with TP...")
                vision_success = compile_vision_encoder_tp(args)
            else:
                print("\n[Step 1] Compiling Vision Encoder...")
                vision_success = compile_vision_encoder(args)

            print("\n[Step 2] Compiling Language Model...")
            lang_success = compile_language_model(args)

            if vision_success and lang_success:
                print("\n" + "=" * 50)
                print("Text Encoder Compilation Complete!")
                print("=" * 50)
                if args.vision_tp:
                    print("  Vision Encoder: TP={} (saved to vision_encoder_tp/)".format(args.tp_degree))
                else:
                    print("  Vision Encoder: Single device (saved to vision_encoder/)")
                print("  Language Model: TP={} (saved to language_model/)".format(args.language_tp_degree))
    else:
        compile_text_encoder_full(args)
