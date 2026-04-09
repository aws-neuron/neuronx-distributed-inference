"""
Wan2.2 TI2V Inference with Context Parallel (V3 CP).

This script uses:
- NxDModel.load() for text_encoder (V2 API)
- NxDModel.load() for transformer with CP (V3 CP API)
- NxDModel.load() for decoder and post_quant_conv (V2 API) if available
- Falls back to torch.jit.load() for decoder and post_quant_conv (V1 API)

Key differences from v2:
- Transformer uses TP=4, CP=2 (world_size=8)
- Checkpoints are duplicated for CP ranks with unique global_rank
- Pre-computed RoPE is loaded and passed to transformer
- V2 decoder accepts 34 individual feat_cache tensor arguments

Usage:
    NEURON_RT_NUM_CORES=8 python run_wan2.2_ti2v.py --compiled_models_dir compiled_models
"""
# IMPORTANT: Set environment variables BEFORE any imports
import os
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video, load_image

import argparse
import json
import numpy as np
from PIL import Image
import random
import time
import torch
import torch_neuronx

# Patch xm.mark_step() to prevent unwanted per-step synchronization.
# The diffusers pipeline calls it inside the denoising loop, which
# triggers a global XLA sync across all NeuronCores. NxDModel handles
# its own synchronization internally, so this is unnecessary overhead.
try:
    import torch_xla.core.xla_model as xm
    xm.mark_step = lambda *args, **kwargs: None
except ImportError:
    pass

from neuronx_distributed import NxDModel
from safetensors.torch import load_file

from neuron_commons import (
    InferenceTextEncoderWrapperV2,
    DecoderWrapperV3NoCache, DecoderWrapperV3Rolling,
    DecoderWrapperV3Tiled,
    PostQuantConvWrapperV2, EncoderWrapperV3, QuantConvWrapperV3,
)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def load_model_config(model_path):
    """Load model configuration from config.json."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_sharded_weights(model_path, tp_degree):
    """Load TP sharded weights from safetensors files.

    Filters out master_weight tensors which are artifacts from shard_checkpoint()
    and not actual model parameters. Including them causes _parallel_load to fail
    with replica group assertion errors.
    """
    weights_path = os.path.join(model_path, "weights")
    sharded_weights = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        raw_ckpt = load_file(ckpt_path)
        # Remove master_weight tensors (duplicates created by shard_checkpoint)
        ckpt = {k: v for k, v in raw_ckpt.items() if 'master_weight' not in k}
        if rank == 0:
            removed = len(raw_ckpt) - len(ckpt)
            if removed > 0:
                print(f"  Filtered {removed} master_weight tensors from checkpoints ({len(ckpt)} keys remaining)")
        sharded_weights.append(ckpt)
    return sharded_weights


def load_duplicated_weights(model_path, world_size):
    """
    Load single checkpoint and duplicate for all ranks.

    For models like decoder that don't use actual TP sharding,
    we load tp0 checkpoint and duplicate for all world_size ranks.

    Args:
        model_path: Path to the compiled model directory
        world_size: Number of ranks to duplicate to

    Returns:
        List of world_size checkpoint dicts (all identical)
    """
    weights_path = os.path.join(model_path, "weights")
    base_ckpt_path = os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors")
    base_ckpt = load_file(base_ckpt_path)

    # Duplicate for all ranks
    sharded_weights = []
    for rank in range(world_size):
        ckpt = {k: v.clone() for k, v in base_ckpt.items()}
        sharded_weights.append(ckpt)

    return sharded_weights


def prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree):
    """
    Duplicate TP checkpoints for CP ranks with unique global_rank.

    With TP=4, CP=2, world_size=8:
    - Ranks 0-3 (CP rank 0): use tp_checkpoints[0-3]
    - Ranks 4-7 (CP rank 1): use tp_checkpoints[0-3] with different global_rank

    Args:
        tp_checkpoints: List of TP checkpoint dicts (length = tp_degree)
        tp_degree: Tensor parallel degree (4)
        cp_degree: Context parallel degree (2)

    Returns:
        List of world_size checkpoints with unique global_rank per rank
    """
    world_size = tp_degree * cp_degree
    sharded_checkpoints = []

    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            world_rank = cp_rank * tp_degree + tp_rank

            # Clone checkpoint
            ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}

            # Set unique global_rank for SPMD scatter/gather
            global_rank_key = "transformer.global_rank.rank"
            if global_rank_key in ckpt:
                ckpt[global_rank_key] = torch.tensor([world_rank], dtype=torch.int32)

            sharded_checkpoints.append(ckpt)

    print(f"Prepared {len(sharded_checkpoints)} checkpoints for world_size={world_size} (TP={tp_degree}, CP={cp_degree})")
    return sharded_checkpoints


class InferenceTransformerWrapperV3CP(torch.nn.Module):
    """
    Wrapper for transformer with Context Parallel (V3 CP) or CFG Parallel.

    Key differences from V2:
    - Passes pre-computed RoPE (cos, sin) to transformer
    - Handles CP-specific input shapes
    - Supports I2V by replacing frame 0 in model input (simulates WanImageToVideoPipeline)

    CFG Parallel mode:
    - The model is compiled with batch_size=2 (uncond + cond stacked along dim=0)
    - The pipeline still makes 2 forward calls per step (cond then uncond)
    - On the first call (cond): batch with stored negative embeddings, run single
      forward pass, cache uncond result, return cond result
    - On the second call (uncond): return cached result (no forward pass)
    - This halves the number of actual device forward passes per step
    """

    def __init__(self, transformer, nxd_model, rotary_emb_cos, rotary_emb_sin,
                 cfg_parallel=False):
        super().__init__()
        self.transformer = transformer  # Original transformer for config access
        self.nxd_model = nxd_model
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device
        self.cache_context = transformer.cache_context

        # Pre-computed RoPE
        self.rotary_emb_cos = rotary_emb_cos
        self.rotary_emb_sin = rotary_emb_sin

        # I2V: image condition for model-input replacement
        self.image_condition = None

        # CFG Parallel state
        self.cfg_parallel = cfg_parallel
        self._negative_embeds = None  # Set before inference with stored negative prompt embeddings
        self._cached_uncond_result = None
        self._is_cond_call = True  # Toggles between cond/uncond calls

    def _run_nxd_model(self, hidden_states, timestep, encoder_hidden_states):
        """Run NxDModel forward pass."""
        if hasattr(self.nxd_model, 'inference'):
            output = self.nxd_model.inference(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                rotary_emb_cos=self.rotary_emb_cos,
                rotary_emb_sin=self.rotary_emb_sin,
            )
        else:
            output = self.nxd_model(
                hidden_states,
                timestep,
                encoder_hidden_states,
                self.rotary_emb_cos,
                self.rotary_emb_sin,
            )
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def _prepare_timestep(self, timestep):
        """Normalize timestep to correct shape."""
        if timestep is not None:
            if timestep.dim() > 1:
                timestep = timestep.flatten()[0:1]
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)
            timestep = timestep.to(torch.float32)
        return timestep

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, return_dict=False, **kwargs):
        """Forward with pre-computed RoPE. Supports CP and CFG Parallel modes."""
        # I2V: replace frame 0 in model input so the model always sees the clean image
        if self.image_condition is not None:
            hidden_states = hidden_states.clone()
            hidden_states[:, :, 0:1, :, :] = self.image_condition.to(hidden_states.dtype)

        timestep = self._prepare_timestep(timestep)

        # CFG Parallel: batch cond + uncond into single forward pass
        if self.cfg_parallel and self._negative_embeds is not None:
            return self._forward_cfg_parallel(hidden_states, timestep, encoder_hidden_states)

        # Standard CP mode: single forward pass
        output = self._run_nxd_model(hidden_states, timestep, encoder_hidden_states)
        return (output,)

    def _forward_cfg_parallel(self, hidden_states, timestep, encoder_hidden_states):
        """CFG Parallel forward: batch cond+uncond, single forward, split results.

        The pipeline calls forward twice per step:
        1. First call with prompt_embeds (cond) -> we run batched forward, return cond result
        2. Second call with negative_prompt_embeds (uncond) -> return cached uncond result
        """
        if self._is_cond_call:
            # First call (cond): batch with negative embeddings and run once
            hs_batched = torch.cat([hidden_states, hidden_states], dim=0)  # [2, C, F, H, W]
            enc_batched = torch.cat(
                [self._negative_embeds.to(encoder_hidden_states.dtype), encoder_hidden_states],
                dim=0,
            )  # [2, text_len, D]
            ts_batched = torch.cat([timestep, timestep], dim=0) if timestep is not None else None

            output = self._run_nxd_model(hs_batched, ts_batched, enc_batched)

            # Split: batch[0] = uncond (from negative embeds), batch[1] = cond
            noise_uncond = output[0:1]
            noise_cond = output[1:2]

            self._cached_uncond_result = noise_uncond
            self._is_cond_call = False
            return (noise_cond,)
        else:
            # Second call (uncond): return cached result without running model
            result = self._cached_uncond_result
            self._cached_uncond_result = None
            self._is_cond_call = True
            return (result,)


def load_transformer(compiled_models_dir, pipe):
    """
    Load compiled transformer.

    Steps:
    1. Check for CFG parallel (transformer_cfg/) or CP (transformer/) directory
    2. Load config to get TP/CP degrees and cfg_parallel flag
    3. Load TP checkpoints
    4. Duplicate for CP ranks with unique global_rank
    5. Load NxDModel and set weights
    6. Load pre-computed RoPE
    7. Create wrapper

    Args:
        compiled_models_dir: Directory containing compiled models
        pipe: Original pipeline for config access

    Returns:
        InferenceTransformerWrapperV3CP instance
    """
    # Check for CFG parallel first, fall back to CP
    transformer_cfg_path = f"{compiled_models_dir}/transformer_cfg"
    transformer_cp_path = f"{compiled_models_dir}/transformer"
    if os.path.exists(transformer_cfg_path):
        transformer_path = transformer_cfg_path
    else:
        transformer_path = transformer_cp_path

    # Load config
    config = load_model_config(transformer_path)
    tp_degree = config["tp_degree"]
    cp_degree = config["cp_degree"]
    world_size = config["world_size"]
    cfg_parallel = config.get("cfg_parallel", False)

    mode = "CFG Parallel" if cfg_parallel else "Context Parallel"
    print(f"Loading V3 transformer ({mode}, TP={tp_degree}, CP={cp_degree}, world_size={world_size})...")

    # Load TP checkpoints
    tp_checkpoints = load_sharded_weights(transformer_path, tp_degree)

    # Duplicate for CP ranks
    cp_checkpoints = prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree)

    # Load NxDModel
    nxd_model_path = os.path.join(transformer_path, "nxd_model.pt")
    nxd_model = NxDModel.load(nxd_model_path)
    nxd_model.set_weights(cp_checkpoints)
    nxd_model.to_neuron()

    # Load pre-computed RoPE
    rope_cache_path = os.path.join(transformer_path, "rope_cache.pt")
    rope_cache = torch.load(rope_cache_path)
    rotary_emb_cos = rope_cache["rotary_emb_cos"].to(torch.bfloat16)
    rotary_emb_sin = rope_cache["rotary_emb_sin"].to(torch.bfloat16)
    print(f"  Loaded RoPE: cos={rotary_emb_cos.shape}, sin={rotary_emb_sin.shape}")

    # Create wrapper
    wrapper = InferenceTransformerWrapperV3CP(
        transformer=pipe.transformer,
        nxd_model=nxd_model,
        rotary_emb_cos=rotary_emb_cos,
        rotary_emb_sin=rotary_emb_sin,
        cfg_parallel=cfg_parallel,
    )

    print(f"Transformer loaded ({mode}).")
    return wrapper


def prepare_image_latents(pipe, image, num_frames, height, width, device, dtype, generator=None):
    """
    Encode input image and prepare latents for I2V generation.

    Uses (raw - mean) / std normalization for stronger signal on V3 (bfloat16).
    Returns (latents, image_condition) for model-input replacement.
    """
    if isinstance(image, str):
        image = load_image(image)

    if isinstance(image, Image.Image):
        image = image.resize((width, height), Image.LANCZOS)
        image = np.array(image)

    image = torch.from_numpy(image).float() / 127.5 - 1.0
    image = image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    image = image.unsqueeze(2)  # [1, C, 1, H, W]
    image = image.to(device=device, dtype=dtype)

    with torch.no_grad():
        image_latents = pipe.vae.encode(image).latent_dist.sample(generator)

        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype)
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype)
        # Use / latents_std for amplified signal — V3 bfloat16 needs stronger signal
        # (reference uses * latents_std which works for V1 float32 but is too weak for V3)
        image_latents = (image_latents - latents_mean) / latents_std

    num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    latent_height = height // pipe.vae_scale_factor_spatial
    latent_width = width // pipe.vae_scale_factor_spatial

    shape = (1, image_latents.shape[1], num_latent_frames, latent_height, latent_width)
    latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

    image_condition = image_latents.to(torch.float32)
    latents[:, :, 0:1, :, :] = image_condition

    return latents, image_condition


# Defaults
DEFAULT_COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b"
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
SEED = 42


def main(args):
    set_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    DTYPE = torch.bfloat16
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    # Load base pipeline
    print("Loading base pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=DTYPE, cache_dir=HUGGINGFACE_CACHE_DIR
    )

    compiled_models_dir = args.compiled_models_dir
    seqlen = args.max_sequence_length

    # IMPORTANT: Load Transformer FIRST to set up correct process groups
    # The transformer uses DP groups for Context Parallel communication
    # Loading it first ensures the process groups are properly initialized
    print("\nLoading transformer...")
    transformer_wrapper = load_transformer(compiled_models_dir, pipe)

    # Load Text Encoder - after transformer to share process groups
    print("\nLoading text encoder...")
    text_encoder_dir = f"{compiled_models_dir}/text_encoder"
    text_encoder_wrapper = InferenceTextEncoderWrapperV2(
        torch.bfloat16, pipe.text_encoder, seqlen
    )
    text_encoder_config = load_model_config(text_encoder_dir)
    text_encoder_tp = text_encoder_config["tp_degree"]
    text_encoder_world_size = text_encoder_config.get("world_size", text_encoder_tp)
    text_encoder_nxd = NxDModel.load(os.path.join(text_encoder_dir, "nxd_model.pt"))
    text_encoder_weights = load_sharded_weights(text_encoder_dir, text_encoder_tp)
    # Duplicate weights for CP ranks if world_size > tp_degree
    if text_encoder_world_size > text_encoder_tp:
        cp_degree = text_encoder_world_size // text_encoder_tp
        text_encoder_weights = prepare_cp_checkpoints(text_encoder_weights, text_encoder_tp, cp_degree)
    text_encoder_nxd.set_weights(text_encoder_weights)
    text_encoder_nxd.to_neuron()
    text_encoder_wrapper.t = text_encoder_nxd
    print("Text encoder loaded.")

    # Load Decoder - check for Tiled, Rolling, NoCache
    decoder_tiled_path = f"{compiled_models_dir}/decoder_tiled"
    decoder_rolling_path = f"{compiled_models_dir}/decoder_rolling"
    decoder_nocache_path = f"{compiled_models_dir}/decoder_nocache"

    if os.path.exists(decoder_tiled_path):
        print("\nLoading decoder (Tiled - spatial tiling for large resolutions)...")
        decoder_config = load_model_config(decoder_tiled_path)
        decoder_frames = decoder_config.get("decoder_frames", 2)
        tile_h = decoder_config["height"] // 16   # tile latent height
        tile_w = decoder_config["width"] // 16     # tile latent width
        overlap = decoder_config.get("overlap_latent", 4)
        vae_decoder_wrapper = DecoderWrapperV3Tiled(
            pipe.vae.decoder, decoder_frames=decoder_frames,
            tile_h_latent=tile_h, tile_w_latent=tile_w, overlap_latent=overlap)
        decoder_nxd = NxDModel.load(os.path.join(decoder_tiled_path, "nxd_model.pt"))
        decoder_world_size = decoder_config.get("world_size", 8)

        decoder_weights = load_duplicated_weights(decoder_tiled_path, decoder_world_size)
        decoder_nxd.set_weights(decoder_weights)
        decoder_nxd.to_neuron()

        vae_decoder_wrapper.nxd_model = decoder_nxd
        print(f"Decoder (Tiled) loaded. tile={tile_h}x{tile_w} latent, "
              f"overlap={overlap}, decoder_frames={decoder_frames}")
    elif os.path.exists(decoder_rolling_path):
        decoder_config = load_model_config(decoder_rolling_path)
        decoder_frames = decoder_config.get("decoder_frames", 2)
        is_stateful = decoder_config.get("stateful", False)
        mode = "Stateful" if is_stateful else "Legacy I/O"
        print(f"\nLoading decoder (Rolling Cache - {mode}, flicker-free)...")
        vae_decoder_wrapper = DecoderWrapperV3Rolling(
            pipe.vae.decoder, decoder_frames=decoder_frames, stateful=is_stateful)
        decoder_nxd = NxDModel.load(os.path.join(decoder_rolling_path, "nxd_model.pt"))
        decoder_world_size = decoder_config.get("world_size", 8)

        decoder_weights = load_duplicated_weights(decoder_rolling_path, decoder_world_size)
        decoder_nxd.set_weights(decoder_weights)
        decoder_nxd.to_neuron()

        vae_decoder_wrapper.nxd_model = decoder_nxd
        print(f"Decoder (Rolling, {mode}) loaded. decoder_frames={decoder_frames}")
    elif os.path.exists(decoder_nocache_path):
        print("\nLoading decoder (NoCache)...")
        decoder_config = load_model_config(decoder_nocache_path)
        decoder_frames = decoder_config.get("decoder_frames", 2)
        vae_decoder_wrapper = DecoderWrapperV3NoCache(pipe.vae.decoder, decoder_frames=decoder_frames)
        decoder_nxd = NxDModel.load(os.path.join(decoder_nocache_path, "nxd_model.pt"))
        decoder_world_size = decoder_config.get("world_size", 8)

        decoder_weights = load_duplicated_weights(decoder_nocache_path, decoder_world_size)
        decoder_nxd.set_weights(decoder_weights)
        decoder_nxd.to_neuron()

        vae_decoder_wrapper.nxd_model = decoder_nxd
        print(f"Decoder (NoCache) loaded. decoder_frames={decoder_frames}")
    else:
        raise RuntimeError(
            f"No compiled decoder found in {compiled_models_dir}. "
            f"Expected one of: decoder_tiled/, decoder_rolling/, decoder_nocache/. "
            f"Run compile.sh first."
        )

    # Load post_quant_conv
    pqc_path = f"{compiled_models_dir}/post_quant_conv"
    if not os.path.exists(pqc_path):
        raise RuntimeError(
            f"No compiled post_quant_conv found in {compiled_models_dir}. "
            f"Run compile.sh first."
        )
    print("\nLoading post_quant_conv...")
    vae_post_quant_conv_wrapper = PostQuantConvWrapperV2(pipe.vae.post_quant_conv)
    pqc_nxd = NxDModel.load(os.path.join(pqc_path, "nxd_model.pt"))
    pqc_config = load_model_config(pqc_path)
    pqc_world_size = pqc_config.get("world_size", 8)

    pqc_weights = load_duplicated_weights(pqc_path, pqc_world_size)
    pqc_nxd.set_weights(pqc_weights)
    pqc_nxd.to_neuron()

    vae_post_quant_conv_wrapper.nxd_model = pqc_nxd
    print("post_quant_conv loaded.")

    # Load Encoder and quant_conv for I2V (optional, only if --image is provided)
    if args.image:
        encoder_path = f"{compiled_models_dir}/encoder"
        qc_path = f"{compiled_models_dir}/quant_conv"

        if os.path.exists(encoder_path):
            print("\nLoading encoder...")
            vae_encoder_wrapper = EncoderWrapperV3(pipe.vae.encoder)
            vae_encoder_wrapper.model = torch.jit.load(
                os.path.join(encoder_path, "model.pt")
            )
            pipe.vae.encoder = vae_encoder_wrapper
            print("Encoder loaded.")
        else:
            print("\nCompiled encoder not found, using CPU encoder for I2V.")

        if os.path.exists(qc_path):
            print("\nLoading quant_conv...")
            vae_quant_conv_wrapper = QuantConvWrapperV3(pipe.vae.quant_conv)
            vae_quant_conv_wrapper.model = torch.jit.load(
                os.path.join(qc_path, "model.pt")
            )
            pipe.vae.quant_conv = vae_quant_conv_wrapper
            print("quant_conv loaded.")
        else:
            print("\nCompiled quant_conv not found, using CPU quant_conv for I2V.")

    # Replace pipeline components
    pipe.text_encoder = text_encoder_wrapper
    pipe.transformer = transformer_wrapper
    pipe.vae.decoder = vae_decoder_wrapper
    pipe.vae.post_quant_conv = vae_post_quant_conv_wrapper

    # Override _decode to use rolling-cache decode_latents directly,
    # bypassing diffusers' per-frame loop which causes cache pollution.
    if hasattr(vae_decoder_wrapper, 'decode_latents'):
        original_post_quant_conv = pipe.vae.post_quant_conv
        vae_config = pipe.vae.config
        def _decode_override(z, return_dict=True):
            from diffusers.models.autoencoders.vae import DecoderOutput
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
            vae_decoder_wrapper.reset_cache()
            x = original_post_quant_conv(z)
            out = vae_decoder_wrapper.decode_latents(x)
            if vae_config.patch_size is not None:
                out = unpatchify(out, patch_size=vae_config.patch_size)
            out = torch.clamp(out, min=-1.0, max=1.0)
            if not return_dict:
                return (out,)
            return DecoderOutput(sample=out)
        pipe.vae._decode = _decode_override
        print("VAE _decode overridden to use rolling-cache decode_latents directly.")

    prompt = args.prompt
    negative_prompt = args.negative_prompt

    # CFG Parallel: pre-encode prompts and store negative embeddings in wrapper
    prompt_embeds = None
    negative_prompt_embeds = None
    if transformer_wrapper.cfg_parallel:
        print("\nCFG Parallel: pre-encoding prompts...")
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=seqlen,
            device=torch.device('cpu'),
        )
        prompt_embeds = prompt_embeds.to(torch.bfloat16)
        negative_prompt_embeds = negative_prompt_embeds.to(torch.bfloat16)
        transformer_wrapper._negative_embeds = negative_prompt_embeds
        print(f"  prompt_embeds: {prompt_embeds.shape}")
        print(f"  negative_prompt_embeds: {negative_prompt_embeds.shape}")

    # Prepare I2V latents BEFORE warmup
    i2v_latents = None
    image_condition = None
    generator = torch.Generator().manual_seed(SEED)
    if args.image:
        print(f"\nEncoding input image: {args.image}")
        i2v_latents, image_condition = prepare_image_latents(
            pipe, args.image, args.num_frames, args.height, args.width,
            torch.device('cpu'), dtype=torch.float32,
            generator=generator
        )
        print(f"I2V latents: {i2v_latents.shape}")

    # Build common pipeline kwargs
    pipe_kwargs = dict(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=seqlen,
    )
    if transformer_wrapper.cfg_parallel:
        # Pass pre-encoded embeddings (pipeline won't re-encode)
        pipe_kwargs["prompt_embeds"] = prompt_embeds
        pipe_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
    else:
        pipe_kwargs["prompt"] = prompt
        pipe_kwargs["negative_prompt"] = negative_prompt

    # Warmup (without I2V latents, no generator)
    print("\nStarting warmup inference...")
    start = time.time()
    output_warmup = pipe(**pipe_kwargs).frames[0]
    end = time.time()
    print(f"Warmup time: {end - start:.2f}s")

    # Main inference (multiple runs for accurate benchmarking)
    num_runs = args.num_runs
    mode = "I2V" if args.image else "T2V"
    run_times = []

    for run_idx in range(num_runs):
        # Reset state before each run
        if hasattr(vae_decoder_wrapper, 'reset_cache'):
            vae_decoder_wrapper.reset_cache()
        if transformer_wrapper.cfg_parallel:
            transformer_wrapper._is_cond_call = True
            transformer_wrapper._cached_uncond_result = None

        run_label = f"Run {run_idx + 1}/{num_runs}" if num_runs > 1 else "Main inference"
        print(f"\nStarting {run_label}...")
        start = time.time()

        # Enable model-input replacement for I2V
        if image_condition is not None:
            transformer_wrapper.image_condition = image_condition

        # Reset generator for reproducibility
        generator = torch.Generator().manual_seed(SEED)

        main_kwargs = dict(pipe_kwargs)  # Copy common kwargs
        main_kwargs["generator"] = generator
        if i2v_latents is not None:
            main_kwargs["latents"] = i2v_latents.clone()

            # Restore frame 0 only on the last step (for correct decode)
            num_steps = args.num_inference_steps
            def i2v_callback(pipe_ref, step_index, timestep, callback_kwargs):
                if step_index == num_steps - 1:
                    callback_kwargs["latents"][:, :, 0:1, :, :] = image_condition.to(
                        callback_kwargs["latents"].dtype
                    )
                return callback_kwargs

            main_kwargs["callback_on_step_end"] = i2v_callback
            main_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        output = pipe(**main_kwargs).frames[0]
        end = time.time()

        # Reset
        transformer_wrapper.image_condition = None

        inference_time = end - start
        per_step_time = inference_time / args.num_inference_steps
        run_times.append(inference_time)
        print(f"{run_label}: {inference_time:.2f}s ({per_step_time:.3f}s/step)")

    # Report results
    print(f"\nOutput frames: {len(output)}")
    if num_runs > 1:
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)
        avg_per_step = avg_time / args.num_inference_steps
        print(f"\n{mode} benchmark ({num_runs} runs):")
        print(f"  Avg: {avg_time:.2f}s ({avg_per_step:.3f}s/step)")
        print(f"  Min: {min_time:.2f}s  Max: {max_time:.2f}s")
        # Print in the standard format (avg) for test_resolutions.sh parsing
        print(f"\n{mode} inference time: {avg_time:.2f}s")
        print(f"Per step (denoise only): {avg_per_step:.3f}s")
    else:
        print(f"\n{mode} inference time: {run_times[0]:.2f}s")
        print(f"Per step (denoise only): {run_times[0] / args.num_inference_steps:.3f}s")

    # Save video (from last run)
    output_path = args.output
    export_to_video(output, output_path, fps=args.fps)
    print(f"\nVideo saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V Inference with Context Parallel")
    parser.add_argument("--compiled_models_dir", type=str, default=DEFAULT_COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max text sequence length")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic",
                        help="Text prompt")
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        help="Negative prompt")
    parser.add_argument("--image", type=str, default=None, help="Input image for I2V (omit for T2V)")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS (default: 16)")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of inference runs for benchmarking")
    args = parser.parse_args()

    main(args)
