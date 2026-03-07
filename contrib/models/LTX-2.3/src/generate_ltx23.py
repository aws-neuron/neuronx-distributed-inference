#!/usr/bin/env python3
"""
LTX-2.3 E2E Generation on Neuron
==================================
Full end-to-end video+audio generation pipeline:
  1. Text encoding (Gemma 3 12B on Neuron TP=4, CPU fallback, or random embeddings)
  2. Denoising loop (48-block DiT on Neuron TP=4, 8 Euler steps)
  3. Optional latent upscaling (spatial x2 + temporal x2 on CPU)
  4. Video decode (VideoDecoder on CPU)
  5. Audio decode (AudioDecoder + VocoderWithBWE on CPU)

Outputs: video frames (PNG), MP4 video, WAV audio.

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

  # With random embeddings (no Gemma required):
  python3 generate_ltx23.py --no-text-encoder

  # With Neuron-compiled Gemma3 (fastest, recommended):
  python3 generate_ltx23.py --neuron-gemma \
    --gemma-compiled-dir /home/ubuntu/gemma3_encoder_compiled \
    --gemma-sharded-dir /home/ubuntu/gemma3_encoder_sharded \
    --gemma-path /home/ubuntu/models/gemma-3-12b \
    --prompt "A dog plays in a meadow"

  # With CPU Gemma3 (slow, no compilation needed):
  python3 generate_ltx23.py --gemma-path /path/to/gemma-3-12b --prompt "A dog plays in a meadow"

  # With upscaling (384x512 @ 25 frames -> 768x1024 @ 49 frames):
  python3 generate_ltx23.py --gemma-path /path/to/gemma-3-12b --prompt "A dog plays in a meadow" --upscale
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import torch
import numpy as np

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Defaults
MODEL_PATH = "/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors"
COMPILE_DIR = "/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_v2"
OUTPUT_DIR = "/home/ubuntu/ltx23_output"
TP_DEGREE = 4
TEXT_SEQ = 256

# Gemma3 Neuron defaults
GEMMA3_COMPILED_DIR = "/home/ubuntu/gemma3_encoder_compiled"
GEMMA3_SHARDED_DIR = "/home/ubuntu/gemma3_encoder_sharded"
GEMMA3_SEQ_LEN = 1024

# Default upscaler paths
SPATIAL_UPSCALER_PATH = (
    "/home/ubuntu/models/LTX-2.3/upscalers/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
)
TEMPORAL_UPSCALER_PATH = (
    "/home/ubuntu/models/LTX-2.3/upscalers/ltx-2.3-temporal-upscaler-x2-1.0.safetensors"
)


def load_config(model_path):
    from safetensors import safe_open

    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
    return json.loads(metadata["config"])


def build_cpu_components(config, model_path, dtype=torch.bfloat16):
    """Build all CPU-side components from the safetensors file.

    Uses SingleGPUModelBuilder (with SDOps key remapping) for components that
    need it (LTXModel, VideoDecoder, AudioDecoder), and manual loading for
    components with complex key mappings (Vocoder, EmbeddingsProcessor).

    Returns:
        dict with keys: ltx_model, video_decoder, audio_decoder, vocoder,
        embeddings_processor
    """
    from safetensors.torch import load_file
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.loader.sd_ops import SDOps

    t0 = time.time()

    # Patch SDPA before building any models
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modeling_ltx23 import replace_sdpa_with_bmm

    replace_sdpa_with_bmm()

    # 1. LTXModel (for preprocessors) via SingleGPUModelBuilder
    logger.info("Building LTXModel (preprocessors)...")
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator

    ltx_ops = (
        SDOps("ltx")
        .with_matching(prefix="model.diffusion_model.")
        .with_replacement("model.diffusion_model.", "")
    )
    ltx_builder = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=model_path,
        model_sd_ops=ltx_ops,
    )
    ltx_model = ltx_builder.build(device=torch.device("cpu"), dtype=dtype)
    ltx_model.eval()
    logger.info("  LTXModel: %d params loaded", sum(1 for _ in ltx_model.parameters()))

    # 2. VideoDecoder via SingleGPUModelBuilder
    logger.info("Building VideoDecoder...")
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator

    vd_ops = (
        SDOps("v")
        .with_matching(prefix="vae.")
        .with_replacement("vae.decoder.", "")
        .with_replacement("vae.", "")
    )
    vd_builder = SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path=model_path,
        model_sd_ops=vd_ops,
    )
    video_decoder = vd_builder.build(device=torch.device("cpu"), dtype=dtype)
    video_decoder.eval()
    logger.info(
        "  VideoDecoder: %d params loaded", sum(1 for _ in video_decoder.parameters())
    )

    # 3. AudioDecoder via SingleGPUModelBuilder
    logger.info("Building AudioDecoder...")
    from ltx_core.model.audio_vae.model_configurator import AudioDecoderConfigurator

    ad_ops = (
        SDOps("a")
        .with_matching(prefix="audio_vae.")
        .with_replacement("audio_vae.decoder.", "")
        .with_replacement("audio_vae.", "")
    )
    ad_builder = SingleGPUModelBuilder(
        model_class_configurator=AudioDecoderConfigurator,
        model_path=model_path,
        model_sd_ops=ad_ops,
    )
    audio_decoder = ad_builder.build(device=torch.device("cpu"), dtype=dtype)
    audio_decoder.eval()
    logger.info(
        "  AudioDecoder: %d params loaded", sum(1 for _ in audio_decoder.parameters())
    )

    # 4. Vocoder (manual loading — SDOps can't handle nested "vocoder." prefix)
    logger.info("Building Vocoder...")
    from ltx_core.model.audio_vae.model_configurator import VocoderConfigurator

    vocoder = VocoderConfigurator.from_config(config)
    full_sd = load_file(model_path)
    voc_sd = {}
    for k, v in full_sd.items():
        if k.startswith("vocoder."):
            rest = k[len("vocoder.") :]
            voc_sd[rest] = v.to(torch.float32) if v.is_floating_point() else v
    m, u = vocoder.load_state_dict(voc_sd, strict=False)
    vocoder.eval()
    logger.info(
        "  Vocoder: %d loaded, %d missing, %d unexpected",
        len(voc_sd) - len(m),
        len(m),
        len(u),
    )
    del full_sd

    # 5. EmbeddingsProcessor (manual loading — multiple prefix remappings)
    logger.info("Building EmbeddingsProcessor...")
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
        EmbeddingsProcessorConfigurator,
    )

    embeddings_processor = EmbeddingsProcessorConfigurator.from_config(config)
    embeddings_processor = embeddings_processor.to(dtype=dtype)
    embeddings_processor.eval()

    full_sd = load_file(model_path)
    prefix = "model.diffusion_model."
    emb_keys = {}
    for k, v in full_sd.items():
        sk = k[len(prefix) :] if k.startswith(prefix) else k
        if sk.startswith("video_embeddings_connector."):
            new_key = "video_connector." + sk[len("video_embeddings_connector.") :]
            emb_keys[new_key] = v.to(dtype) if v.is_floating_point() else v
        elif sk.startswith("audio_embeddings_connector."):
            new_key = "audio_connector." + sk[len("audio_embeddings_connector.") :]
            emb_keys[new_key] = v.to(dtype) if v.is_floating_point() else v
        elif sk.startswith("text_embedding_projection."):
            new_key = "feature_extractor." + sk[len("text_embedding_projection.") :]
            emb_keys[new_key] = v.to(dtype) if v.is_floating_point() else v
    m, u = embeddings_processor.load_state_dict(emb_keys, strict=False)
    logger.info(
        "  EmbeddingsProcessor: %d loaded, %d missing, %d unexpected",
        len(emb_keys) - len(m),
        len(m),
        len(u),
    )
    del full_sd

    logger.info("All CPU components loaded in %.1fs", time.time() - t0)

    # Free transformer block weights from CPU model (they run on Neuron)
    if (
        hasattr(ltx_model, "transformer_blocks")
        and ltx_model.transformer_blocks is not None
    ):
        del ltx_model.transformer_blocks
        ltx_model.transformer_blocks = None
    for attr in ("norm_out", "proj_out", "audio_norm_out", "audio_proj_out"):
        if hasattr(ltx_model, attr):
            delattr(ltx_model, attr)
    gc.collect()

    return {
        "ltx_model": ltx_model,
        "video_decoder": video_decoder,
        "audio_decoder": audio_decoder,
        "vocoder": vocoder,
        "embeddings_processor": embeddings_processor,
    }


def load_neuron_backbone(compile_dir, model_path, tp_degree=4):
    """Load compiled Neuron backbone with real weights."""
    import torch_neuronx
    from neuronx_distributed.trace.trace import TensorParallelNeuronModel
    from load_with_weights import shard_weight
    from safetensors.torch import load_file

    tp_0_path = os.path.join(compile_dir, "tp_0.pt")

    # Load and shard weights
    logger.info("Loading safetensors for weight injection...")
    full_sd = load_file(model_path)
    prefix = "model.diffusion_model."
    backbone_prefixes = (
        "transformer_blocks.",
        "norm_out.",
        "proj_out.",
        "scale_shift_table",
        "audio_norm_out.",
        "audio_proj_out.",
        "audio_scale_shift_table",
    )
    backbone_sd = {}
    for k, v in full_sd.items():
        stripped = k[len(prefix) :] if k.startswith(prefix) else k
        if stripped.startswith(backbone_prefixes):
            backbone_sd[stripped] = v.to(torch.bfloat16).contiguous()
    backbone_sd["spmd_rank.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
    del full_sd
    gc.collect()

    def sf_key_to_jit_key(sf_key):
        return "weights." + sf_key.replace(".", "->")

    # Create per-rank state dicts
    rank_sds = [{} for _ in range(tp_degree)]
    for sf_key, full_weight in backbone_sd.items():
        jit_key = sf_key_to_jit_key(sf_key)
        for rank in range(tp_degree):
            rank_sds[rank][jit_key] = shard_weight(
                full_weight, jit_key, rank, tp_degree
            )
    del backbone_sd
    gc.collect()

    # Load compiled models and inject weights
    models = []
    t0 = time.time()
    for rank in range(tp_degree):
        logger.info("  Loading Neuron rank %d...", rank)
        with torch_neuronx.contexts.disable_nrt_load():
            model = torch.jit.load(tp_0_path)
        model_sd = dict(model.named_parameters())
        injected = 0
        for jit_key, sharded_weight in rank_sds[rank].items():
            if jit_key in model_sd and model_sd[jit_key].shape == sharded_weight.shape:
                model_sd[jit_key].data.copy_(sharded_weight)
                injected += 1
        if rank == 0:
            logger.info("    Injected %d/%d weights", injected, len(rank_sds[rank]))
        models.append(model)

    logger.info("  All Neuron models loaded in %.1fs", time.time() - t0)
    del rank_sds
    gc.collect()

    return TensorParallelNeuronModel(models)


def load_neuron_gemma3(compiled_dir, sharded_dir, tp_degree=4):
    """Load Neuron-compiled Gemma3 encoder with pre-sharded weights.

    Both models (DiT backbone and Gemma3 encoder) share the same 4 NeuronCores
    and execute sequentially.
    """
    import torch_neuronx
    from neuronx_distributed.trace.trace import (
        TensorParallelNeuronModel,
        replace_weights,
    )

    tp_0_path = os.path.join(compiled_dir, "tp_0.pt")
    if not os.path.exists(tp_0_path):
        raise FileNotFoundError(
            f"Compiled Gemma3 not found at {tp_0_path}. Run compile_gemma3.py first."
        )

    models = []
    t0 = time.time()
    for rank in range(tp_degree):
        logger.info("  Loading Gemma3 Neuron rank %d...", rank)
        rank_path = os.path.join(sharded_dir, "rank_%d.pt" % rank)
        if not os.path.exists(rank_path):
            raise FileNotFoundError(
                f"Sharded weights not found at {rank_path}. "
                "Run shard_gemma3_weights.py first."
            )
        ckpt = torch.load(rank_path, weights_only=True)
        tp_path = os.path.join(compiled_dir, "tp_%d.pt" % rank)
        with torch_neuronx.contexts.disable_nrt_load():
            traced_model = torch.jit.load(tp_path)
        replace_weights(traced_model, ckpt)
        models.append(traced_model)
        del ckpt
        gc.collect()

    logger.info("  Gemma3 encoder loaded in %.1fs", time.time() - t0)
    return TensorParallelNeuronModel(models)


def encode_text_neuron(
    neuron_gemma3, tokenizer_path, prompt, text_seq, embeddings_processor
):
    """Encode text using Neuron-compiled Gemma3 encoder.

    The Neuron encoder returns stacked hidden states (B, seq_len, 3840, 49).
    We convert to a tuple of per-layer tensors for process_hidden_states().
    """
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer

    tokenizer = LTXVGemmaTokenizer(
        tokenizer_path=tokenizer_path,
        max_length=text_seq,
    )

    token_pairs = tokenizer.tokenize_with_weights(prompt)["gemma"]
    input_ids = torch.tensor([[t[0] for t in token_pairs]], dtype=torch.int64)
    attention_mask = torch.tensor([[w[1] for w in token_pairs]], dtype=torch.int64)
    actual_len = input_ids.shape[1]

    compiled_seq_len = GEMMA3_SEQ_LEN
    if actual_len < compiled_seq_len:
        pad_len = compiled_seq_len - actual_len
        input_ids = torch.cat(
            [torch.zeros(1, pad_len, dtype=torch.int64), input_ids], dim=1
        )
        attention_mask = torch.cat(
            [torch.zeros(1, pad_len, dtype=torch.int64), attention_mask], dim=1
        )
    elif actual_len > compiled_seq_len:
        input_ids = input_ids[:, :compiled_seq_len]
        attention_mask = attention_mask[:, :compiled_seq_len]

    logger.info(
        "  Tokenized: %d tokens -> padded to %d", actual_len, input_ids.shape[1]
    )

    t0 = time.time()
    with torch.no_grad():
        stacked = neuron_gemma3(input_ids, attention_mask)
    logger.info("  Neuron Gemma3 forward: %.1fs", time.time() - t0)

    # Trim back to actual token length if we padded
    if actual_len < compiled_seq_len:
        pad_len = compiled_seq_len - actual_len
        stacked = stacked[:, pad_len:, :, :]
        attention_mask = attention_mask[:, pad_len:]

    # Convert stacked tensor to tuple of per-layer tensors
    hidden_states = tuple(stacked[:, :, :, i] for i in range(stacked.shape[-1]))

    result = embeddings_processor.process_hidden_states(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )
    return result.video_encoding, result.audio_encoding, result.attention_mask


def load_upscalers(spatial_path, temporal_path, dtype=torch.bfloat16):
    """Load spatial and temporal latent upscalers from separate safetensors files.

    Each upscaler safetensors file has its config embedded in metadata.
    Uses LatentUpsamplerConfigurator.from_config() + manual load_state_dict.

    Returns:
        dict with keys: spatial_upsampler, temporal_upsampler
    """
    import json as _json

    from safetensors import safe_open
    from safetensors.torch import load_file
    from ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator

    result = {}

    for name, path in [
        ("spatial_upsampler", spatial_path),
        ("temporal_upsampler", temporal_path),
    ]:
        logger.info("Loading %s from %s...", name, path)
        t0 = time.time()

        # Read config from safetensors metadata
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
        upsampler_config = _json.loads(metadata["config"])

        # Build model from config
        upsampler = LatentUpsamplerConfigurator.from_config(upsampler_config)
        upsampler = upsampler.to(dtype=dtype)
        upsampler.eval()

        # Load weights
        sd = load_file(path)
        sd = {k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items()}
        m, u = upsampler.load_state_dict(sd, strict=False)
        total_params = sum(p.numel() for p in upsampler.parameters())
        logger.info(
            "  %s: %.1fM params, %d missing, %d unexpected, loaded in %.1fs",
            name,
            total_params / 1e6,
            len(m),
            len(u),
            time.time() - t0,
        )
        if m:
            logger.warning("  Missing keys[:5]: %s", m[:5])

        result[name] = upsampler
        del sd

    gc.collect()
    return result


def upscale_video_latent(
    video_latent_5d, video_decoder, spatial_upsampler, temporal_upsampler
):
    """Upscale video latent using spatial and temporal upsamplers.

    Flow: un_normalize -> spatial upsample -> temporal upsample -> normalize
    Uses video_decoder.per_channel_statistics for normalization.

    Args:
        video_latent_5d: (B, C, F, H, W) normalized video latent
        video_decoder: VideoDecoder with per_channel_statistics
        spatial_upsampler: LatentUpsampler for spatial x2
        temporal_upsampler: LatentUpsampler for temporal x2

    Returns:
        Upscaled (B, C, F', H*2, W*2) normalized video latent
    """
    pcs = video_decoder.per_channel_statistics
    logger.info("  Input latent: %s", video_latent_5d.shape)

    # Un-normalize to raw latent space
    latent = pcs.un_normalize(video_latent_5d)
    logger.info(
        "  Un-normalized: %s (mean=%.3f, std=%.3f)",
        latent.shape,
        latent.float().mean().item(),
        latent.float().std().item(),
    )

    # Spatial upsample (H, W doubled)
    t0 = time.time()
    with torch.no_grad():
        latent = spatial_upsampler(latent)
    logger.info("  After spatial upsample: %s in %.1fs", latent.shape, time.time() - t0)

    # Temporal upsample (F roughly doubled, first frame removed)
    t0 = time.time()
    with torch.no_grad():
        latent = temporal_upsampler(latent)
    logger.info(
        "  After temporal upsample: %s in %.1fs", latent.shape, time.time() - t0
    )

    # Re-normalize
    latent = pcs.normalize(latent)
    logger.info(
        "  Re-normalized: %s (mean=%.3f, std=%.3f)",
        latent.shape,
        latent.float().mean().item(),
        latent.float().std().item(),
    )

    return latent


def generate(args):
    """Main generation pipeline."""
    config = load_config(args.model_path)
    tc = config["transformer"]
    logger.info(
        "Model: %d layers, %d heads", tc["num_layers"], tc["num_attention_heads"]
    )

    # Build CPU components
    logger.info("\n=== Building CPU components ===")
    cpu = build_cpu_components(config, args.model_path)

    # Load Neuron backbone
    logger.info("\n=== Loading Neuron backbone ===")
    neuron_backbone = load_neuron_backbone(
        args.compile_dir, args.model_path, args.tp_degree
    )

    # Build pipeline wrapper
    from pipeline import NeuronTransformerWrapper

    wrapper = NeuronTransformerWrapper(
        compiled_backbone=neuron_backbone,
        cpu_ltx_model=cpu["ltx_model"],
        text_seq=args.text_seq,
    )

    # Get context embeddings
    dtype = torch.bfloat16
    if args.no_text_encoder:
        logger.info("\n=== Using random embeddings (no text encoder) ===")
        torch.manual_seed(args.seed)
        video_context = torch.randn(1, args.text_seq, 4096, dtype=dtype)
        audio_context = torch.randn(1, args.text_seq, 2048, dtype=dtype)
        context_mask = torch.ones(1, args.text_seq, dtype=torch.int64)
        context_mask[:, 50:] = 0  # mask out most tokens
    elif args.neuron_gemma:
        logger.info("\n=== Running text encoder on Neuron ===")
        logger.info("Loading Neuron-compiled Gemma3 encoder...")
        t0 = time.time()
        neuron_gemma3 = load_neuron_gemma3(
            args.gemma_compiled_dir, args.gemma_sharded_dir, args.tp_degree
        )
        logger.info("Gemma3 encoder ready in %.1fs", time.time() - t0)

        # Warmup pass
        logger.info("  Warmup forward pass...")
        t0 = time.time()
        warmup_ids = torch.zeros(1, GEMMA3_SEQ_LEN, dtype=torch.int64)
        warmup_mask = torch.ones(1, GEMMA3_SEQ_LEN, dtype=torch.int64)
        with torch.no_grad():
            _ = neuron_gemma3(warmup_ids, warmup_mask)
        logger.info("  Warmup done in %.1fs", time.time() - t0)

        t0 = time.time()
        video_context, audio_context, context_mask = encode_text_neuron(
            neuron_gemma3,
            args.gemma_path,
            args.prompt,
            args.text_seq,
            cpu["embeddings_processor"],
        )
        logger.info("Text encoded on Neuron in %.1fs", time.time() - t0)
        logger.info(
            "  video_context: %s, audio_context: %s",
            video_context.shape,
            audio_context.shape,
        )

        # Free Gemma3 Neuron model
        del neuron_gemma3
        gc.collect()
    else:
        logger.info("\n=== Running text encoder ===")
        # Load Gemma 3 12B
        from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
        from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
        from transformers.models.gemma3 import Gemma3ForConditionalGeneration
        from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
        from transformers import Gemma3Config

        logger.info("Loading Gemma 3 12B from %s...", args.gemma_path)
        t0 = time.time()

        # Build config and load model
        gemma_config = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict())
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            args.gemma_path,
            config=gemma_config,
            dtype=dtype,
        )
        gemma_model = gemma_model.to(dtype=dtype)
        gemma_model.eval()
        logger.info("Gemma loaded in %.1fs", time.time() - t0)

        tokenizer = LTXVGemmaTokenizer(
            tokenizer_path=args.gemma_path,
            max_length=args.text_seq,
        )
        text_encoder = GemmaTextEncoder(
            model=gemma_model,
            tokenizer=tokenizer,
            dtype=dtype,
        )

        # Encode prompt
        logger.info("Encoding prompt: '%s'", args.prompt)
        t0 = time.time()
        with torch.no_grad():
            hidden_states, attention_mask = text_encoder.encode(args.prompt)

        # Run embeddings processor (handles additive mask conversion internally)
        result = cpu["embeddings_processor"].process_hidden_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        video_context = result.video_encoding
        audio_context = result.audio_encoding
        context_mask = (
            result.attention_mask
        )  # keep as int64 for proper additive mask conversion
        logger.info("Text encoded in %.1fs", time.time() - t0)
        logger.info(
            "  video_context: %s, audio_context: %s",
            video_context.shape,
            audio_context.shape,
        )

        # Free Gemma to save RAM
        del gemma_model, text_encoder
        gc.collect()

    # Setup latent tools
    from ltx_core.tools import (
        VideoLatentTools,
        VideoLatentPatchifier,
        VideoLatentShape,
        AudioLatentTools,
        AudioPatchifier,
        AudioLatentShape,
        SpatioTemporalScaleFactors,
    )
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.components.schedulers import LTX2Scheduler

    # Compute latent dimensions
    # LTX-2.3 VAE downsamples spatially by 32x (not 16x as in some other models)
    # For 384x512 -> height=12, width=16 latent grid
    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_f = (args.num_frames - 1) // 8 + 1

    video_shape = VideoLatentShape(
        batch=1, channels=128, frames=latent_f, height=latent_h, width=latent_w
    )
    v_patchifier = VideoLatentPatchifier(patch_size=1)
    v_scale = SpatioTemporalScaleFactors(time=1, width=8, height=8)
    video_tools = VideoLatentTools(
        target_shape=video_shape,
        patchifier=v_patchifier,
        scale_factors=v_scale,
        causal_fix=False,
        fps=args.fps,
    )

    audio_shape = AudioLatentShape(
        batch=1, channels=8, frames=args.audio_num_frames, mel_bins=16
    )
    a_patchifier = AudioPatchifier(patch_size=16)
    audio_tools = AudioLatentTools(patchifier=a_patchifier, target_shape=audio_shape)

    # Create initial noise
    gen = torch.Generator().manual_seed(args.seed)
    video_state = video_tools.create_initial_state(device="cpu", dtype=dtype)
    audio_state = audio_tools.create_initial_state(device="cpu", dtype=dtype)

    video_sample = torch.randn(video_state.latent.shape, dtype=dtype, generator=gen)
    audio_sample = torch.randn(audio_state.latent.shape, dtype=dtype, generator=gen)

    logger.info("\n=== Denoising (%d steps) ===", args.num_steps)
    logger.info(
        "  Video latent: %s, Audio latent: %s", video_sample.shape, audio_sample.shape
    )

    # Sigma schedule
    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=args.num_steps, latent=video_state.latent)
    logger.info("  Sigmas: %s", [f"{s:.4f}" for s in sigmas.tolist()])

    # Denoising loop
    total_time = 0.0
    for step_idx in range(args.num_steps):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        video_seq_len = video_state.latent.shape[1]
        audio_seq_len = audio_state.latent.shape[1]
        v_ts = sigma.unsqueeze(0).unsqueeze(0).expand(1, video_seq_len)
        a_ts = sigma.unsqueeze(0).unsqueeze(0).expand(1, audio_seq_len)

        video_mod = Modality(
            latent=video_sample,
            sigma=sigma.unsqueeze(0),
            timesteps=v_ts,
            positions=video_state.positions,
            context=video_context,
            enabled=True,
            context_mask=context_mask,
            attention_mask=None,
        )
        audio_mod = Modality(
            latent=audio_sample,
            sigma=sigma.unsqueeze(0),
            timesteps=a_ts,
            positions=audio_state.positions,
            context=audio_context,
            enabled=True,
            context_mask=context_mask.clone(),
            attention_mask=None,
        )

        t0 = time.time()
        with torch.no_grad():
            video_velocity, audio_velocity = wrapper(video_mod, audio_mod)
        step_time = time.time() - t0
        total_time += step_time

        # Euler step using velocity directly (backbone outputs velocity, NOT denoised)
        # next = sample + velocity * (sigma_next - sigma)
        dt = sigma_next - sigma
        video_sample = (video_sample.float() + video_velocity.float() * dt).to(dtype)
        audio_sample = (audio_sample.float() + audio_velocity.float() * dt).to(dtype)

        logger.info(
            "  Step %d/%d: sigma %.4f -> %.4f (%.1fs)",
            step_idx + 1,
            args.num_steps,
            sigma.item(),
            sigma_next.item(),
            step_time,
        )

    logger.info(
        "  Total denoising: %.1fs (%.1fs/step)", total_time, total_time / args.num_steps
    )

    # Unpatchify latents back to spatial format for VAE
    logger.info("\n=== Decoding ===")

    # Video: unpatchify (B, seq, C) -> (B, C, F, H, W) -> (C, F, H, W) for VAE
    video_latent_spatial = v_patchifier.unpatchify(video_sample, video_shape)
    logger.info("  Video latent after unpatchify: %s", video_latent_spatial.shape)

    # Audio: unpatchify (B, seq, C) -> spatial format for VAE
    audio_latent_spatial = a_patchifier.unpatchify(audio_sample, audio_shape)
    logger.info("  Audio latent for VAE: %s", audio_latent_spatial.shape)

    # Optional upscaling (spatial x2 + temporal x2)
    if args.upscale:
        logger.info("\n=== Upscaling latents ===")
        upscalers = load_upscalers(
            args.spatial_upscaler_path, args.temporal_upscaler_path, dtype=dtype
        )
        video_latent_spatial = upscale_video_latent(
            video_latent_spatial,
            cpu["video_decoder"],
            upscalers["spatial_upsampler"],
            upscalers["temporal_upsampler"],
        )
        # Free upscalers after use
        del upscalers
        gc.collect()

    video_latent_4d = video_latent_spatial[0]  # remove batch dim -> (C, F, H, W)
    logger.info("  Video latent for VAE: %s", video_latent_4d.shape)

    # Video decode
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("  Decoding video...")
    t0 = time.time()
    from ltx_core.model.video_vae.video_vae import decode_video

    video_chunks = []
    with torch.no_grad():
        for chunk in decode_video(video_latent_4d, cpu["video_decoder"]):
            video_chunks.append(chunk)
    video_frames = torch.cat(video_chunks, dim=0)  # (F, H, W, 3) uint8
    logger.info("  Video decoded: %s in %.1fs", video_frames.shape, time.time() - t0)

    # Save video frames
    from PIL import Image

    for i in range(video_frames.shape[0]):
        frame = video_frames[i].numpy()
        img = Image.fromarray(frame)
        img.save(os.path.join(args.output_dir, f"frame_{i:04d}.png"))
    logger.info("  Saved %d frames to %s", video_frames.shape[0], args.output_dir)

    # Save as MP4
    try:
        import subprocess

        frame_pattern = os.path.join(args.output_dir, "frame_%04d.png")
        mp4_path = os.path.join(args.output_dir, "output.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(int(args.fps)),
                "-i",
                frame_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                mp4_path,
            ],
            capture_output=True,
            check=True,
        )
        logger.info("  Saved MP4: %s", mp4_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("  ffmpeg not available, skipping MP4: %s", e)

    # Audio decode
    logger.info("  Decoding audio...")
    t0 = time.time()
    from ltx_core.model.audio_vae.audio_vae import decode_audio

    with torch.no_grad():
        audio_result = decode_audio(
            audio_latent_spatial.float(), cpu["audio_decoder"].float(), cpu["vocoder"]
        )
    logger.info(
        "  Audio decoded: waveform %s, sr=%d in %.1fs",
        audio_result.waveform.shape,
        audio_result.sampling_rate,
        time.time() - t0,
    )

    # Save audio
    try:
        import torchaudio

        wav_path = os.path.join(args.output_dir, "output.wav")
        torchaudio.save(
            wav_path, audio_result.waveform.cpu(), audio_result.sampling_rate
        )
        logger.info("  Saved WAV: %s", wav_path)
    except ImportError:
        # Fallback: save as raw tensor
        wav_path = os.path.join(args.output_dir, "audio_waveform.pt")
        torch.save(
            {"waveform": audio_result.waveform.cpu(), "sr": audio_result.sampling_rate},
            wav_path,
        )
        logger.info("  Saved audio tensor: %s", wav_path)

    # Save latents for analysis
    torch.save(
        {
            "video_latent": video_sample.cpu(),
            "audio_latent": audio_sample.cpu(),
            "video_latent_spatial": video_latent_spatial.cpu(),
            "audio_latent_spatial": audio_latent_spatial.cpu(),
        },
        os.path.join(args.output_dir, "latents.pt"),
    )

    logger.info("\n=== Done! Output saved to %s ===", args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 E2E Generation on Neuron")
    parser.add_argument(
        "--model-path", default=MODEL_PATH, help="Safetensors model path"
    )
    parser.add_argument(
        "--compile-dir", default=COMPILE_DIR, help="Compiled model directory"
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--prompt",
        default="A golden retriever puppy runs across a sunny green meadow",
        help="Text prompt",
    )
    parser.add_argument(
        "--no-text-encoder",
        action="store_true",
        help="Use random embeddings instead of Gemma 3",
    )
    parser.add_argument("--gemma-path", default=None, help="Path to Gemma 3 12B model")
    parser.add_argument(
        "--neuron-gemma",
        action="store_true",
        help="Use Neuron-compiled Gemma3 encoder (requires compile_gemma3.py + shard_gemma3_weights.py)",
    )
    parser.add_argument(
        "--gemma-compiled-dir",
        default=GEMMA3_COMPILED_DIR,
        help="Directory with compiled Gemma3 encoder (from compile_gemma3.py)",
    )
    parser.add_argument(
        "--gemma-sharded-dir",
        default=GEMMA3_SHARDED_DIR,
        help="Directory with pre-sharded Gemma3 weights (from shard_gemma3_weights.py)",
    )
    parser.add_argument("--height", type=int, default=384, help="Video height")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument(
        "--num-frames", type=int, default=25, help="Number of video frames"
    )
    parser.add_argument("--num-steps", type=int, default=8, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=float, default=24.0, help="Video frame rate")
    parser.add_argument(
        "--audio-num-frames", type=int, default=26, help="Audio latent frames"
    )
    parser.add_argument(
        "--text-seq", type=int, default=TEXT_SEQ, help="Text sequence length"
    )
    parser.add_argument("--tp-degree", type=int, default=TP_DEGREE, help="TP degree")
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Apply spatial x2 + temporal x2 upscaling before VAE decode",
    )
    parser.add_argument(
        "--spatial-upscaler-path",
        default=SPATIAL_UPSCALER_PATH,
        help="Path to spatial upscaler x2 safetensors",
    )
    parser.add_argument(
        "--temporal-upscaler-path",
        default=TEMPORAL_UPSCALER_PATH,
        help="Path to temporal upscaler x2 safetensors",
    )

    args = parser.parse_args()

    if not args.no_text_encoder and args.gemma_path is None:
        parser.error(
            "Either --no-text-encoder or --gemma-path must be specified. "
            "Use --neuron-gemma for Neuron-compiled Gemma3 (fastest)."
        )

    generate(args)


if __name__ == "__main__":
    main()
