#!/usr/bin/env python3
"""
LTX-2.3 E2E Generation on Neuron
==================================
Full end-to-end video+audio generation pipeline:
  1. Text encoding (Gemma 3 12B on Neuron TP=4, CPU fallback, or random embeddings)
  2. Optional image encoding for image-to-video (Diffusers VAE encoder on CPU)
  3. Denoising loop (48-block DiT on Neuron TP=4, 8 Euler steps)
  4. Optional latent upscaling (spatial x2 + temporal x2 on CPU)
  5. Video decode (VideoDecoder on CPU)
  6. Audio decode (AudioDecoder + VocoderWithBWE on CPU)

Outputs: video frames (PNG), MP4 video, WAV audio.

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

  # Recommended: Application path (fastest, ~21% E2E speedup):
  python3 generate_ltx23.py --use-app \
    --gemma-path /mnt/models/gemma-3-12b \
    --app-compiled-dir /mnt/models/compiled/e2e \
    --prompt "A dog plays in a meadow"

  # Text-to-Video with random embeddings (no Gemma required):
  python3 generate_ltx23.py --no-text-encoder

  # Text-to-Video with Neuron-compiled Gemma3 (fastest, recommended):
  python3 generate_ltx23.py --neuron-gemma \
    --gemma-compiled-dir /home/ubuntu/gemma3_encoder_compiled \
    --gemma-sharded-dir /home/ubuntu/gemma3_encoder_sharded \
    --gemma-path /home/ubuntu/models/gemma-3-12b \
    --prompt "A dog plays in a meadow"

  # Image-to-Video with Neuron Gemma3:
  python3 generate_ltx23.py --neuron-gemma \
    --gemma-compiled-dir /home/ubuntu/gemma3_encoder_compiled \
    --gemma-sharded-dir /home/ubuntu/gemma3_encoder_sharded \
    --gemma-path /home/ubuntu/models/gemma-3-12b \
    --prompt "The woman turns and smiles at the camera" \
    --image /path/to/photo.png

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

# Distilled sigma values from the reference LTX-2.3 pipeline
# See: ltx_pipelines/utils/constants.py DISTILLED_SIGMA_VALUES
DISTILLED_SIGMA_VALUES = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

# Default half-res compilation paths (for two-stage mode)
HALFRES_COMPILE_DIR = "/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_halfres"
HALFRES_SHARDED_DIR = "/home/ubuntu/backbone_sharded_halfres"

# Default backbone text_seq for Application-compiled NEFFs
APP_BACKBONE_TEXT_SEQ = 256  # DiT backbone text_seq for Application-compiled NEFF
APP_COMPILED_DIR = "/mnt/models/compiled/e2e_v2"
APP_ENCODER_SEQ_LEN = 512  # Gemma3 encoder seq_len for Application-compiled NEFF
APP_BACKBONE_AUDIO_SEQ = (
    26  # Audio latent frames (AudioLatentShape frames=26, mel_bins=16, patch=16)
)
APP_HALFRES_COMPILED_DIR = "/mnt/models/compiled/e2e_v2_halfres"
GEMMA3_MODEL_PATH = "/mnt/models/gemma-3-12b"


def encode_image(image_path, model_path, height, width, dtype=torch.bfloat16):
    """Encode an input image into normalized latent space for image-to-video.

    Uses ltx-core's native VideoEncoder loaded from the same safetensors
    checkpoint (which contains both encoder and decoder weights under
    vae.encoder.* and vae.decoder.* prefixes). The encoder includes
    per_channel_statistics and outputs PCS-normalized latents (mean~0, std~1)
    matching the scale of Gaussian noise in the denoising loop.

    Args:
        image_path: Path to input image file
        model_path: Path to LTX-2.3 safetensors checkpoint
        height: Target video height (image will be resized to this)
        width: Target video width (image will be resized to this)
        dtype: Tensor dtype (default bf16)

    Returns:
        Latent tensor of shape (1, 128, 1, H//32, W//32)
    """
    from PIL import Image
    from ltx_core.model.video_vae.model_configurator import VideoEncoderConfigurator
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.loader.sd_ops import SDOps

    logger.info("Encoding image: %s", image_path)
    t0 = time.time()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    logger.info("  Image resized to %dx%d", width, height)

    # Convert to tensor: (H, W, 3) uint8 -> (1, 3, 1, H, W) bf16 [-1, 1]
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
    img_tensor = img_tensor.to(dtype=dtype)
    logger.info("  Image tensor: %s", img_tensor.shape)

    # Build ltx-core VideoEncoder from the safetensors checkpoint
    # The encoder weights are stored under vae.encoder.* and vae.per_channel_statistics.*
    logger.info("  Building ltx-core VideoEncoder...")
    ve_ops = (
        SDOps("ve")
        .with_matching(prefix="vae.encoder.")
        .with_matching(prefix="vae.per_channel_statistics.")
        .with_replacement("vae.encoder.", "")
        .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
    )
    ve_builder = SingleGPUModelBuilder(
        model_class_configurator=VideoEncoderConfigurator,
        model_path=model_path,
        model_sd_ops=ve_ops,
    )
    video_encoder = ve_builder.build(device=torch.device("cpu"), dtype=dtype)
    video_encoder.eval()

    # Encode image — the ltx-core VideoEncoder includes per_channel_statistics
    # and outputs latents already PCS-normalized (mean~0, std~1), matching
    # the scale of Gaussian noise used in the T2V denoising loop.
    # No additional normalization is needed.
    with torch.no_grad():
        latent = video_encoder(img_tensor)
    logger.info(
        "  Encoded latent: %s (mean=%.3f, std=%.3f) in %.1fs",
        latent.shape,
        latent.float().mean().item(),
        latent.float().std().item(),
        time.time() - t0,
    )

    # Free the encoder
    del video_encoder
    gc.collect()

    return latent


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


def load_neuron_backbone(compile_dir, model_path, tp_degree=4, sharded_dir=None):
    """Load compiled Neuron backbone with real weights.

    If sharded_dir is provided, loads pre-sharded per-rank .pt files (~5GB each)
    instead of reading the full 41GB safetensors. This reduces peak CPU memory
    from ~80GB to ~15GB, eliminating swap thrashing on trn2.3xlarge.

    Pre-shard weights with: python3 shard_backbone_weights.py
    """
    import torch_neuronx
    from neuronx_distributed.trace.trace import TensorParallelNeuronModel

    tp_0_path = os.path.join(compile_dir, "tp_0.pt")

    if sharded_dir and os.path.isdir(sharded_dir):
        # Fast path: load pre-sharded weights (~5GB per rank)
        logger.info("Loading pre-sharded backbone weights from %s...", sharded_dir)
        rank_sds = []
        for rank in range(tp_degree):
            rank_path = os.path.join(sharded_dir, "rank_%d.pt" % rank)
            if not os.path.exists(rank_path):
                raise FileNotFoundError(
                    f"Sharded weights not found at {rank_path}. "
                    "Run shard_backbone_weights.py first."
                )
            ckpt = torch.load(rank_path, weights_only=True)
            rank_sds.append(ckpt)
            if rank == 0:
                logger.info("  rank_0: %d keys", len(ckpt))
    else:
        # Fallback: load from safetensors (slower, more memory)
        from safetensors import safe_open
        from load_with_weights import shard_weight

        logger.info("Loading backbone weights (memory-mapped)...")
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
        with safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                stripped = k[len(prefix) :] if k.startswith(prefix) else k
                if stripped.startswith(backbone_prefixes):
                    backbone_sd[stripped] = (
                        f.get_tensor(k).to(torch.bfloat16).contiguous()
                    )
        backbone_sd["spmd_rank.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        logger.info("  Loaded %d backbone tensors", len(backbone_sd))

        def sf_key_to_jit_key(sf_key):
            return "weights." + sf_key.replace(".", "->")

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
        # Free this rank's weights immediately after injection
        rank_sds[rank] = None

    logger.info("  All Neuron models loaded in %.1fs", time.time() - t0)
    del rank_sds
    gc.collect()

    return TensorParallelNeuronModel(models)


def unload_neuron_model(tp_model, name="model"):
    """Fully unload a TensorParallelNeuronModel from NeuronCores.

    The simple `del model; gc.collect()` pattern is insufficient because:
    1. torch.jit.ScriptModule holds NRT (Neuron Runtime) resources
    2. Python's GC may not immediately release the underlying NEFF allocations
    3. The NRT resources occupy HBM even after Python references are dropped

    This function explicitly destroys each rank's model and forces cleanup,
    ensuring the NeuronCores are fully free for the next model to load.
    """
    logger.info("Unloading %s from NeuronCores...", name)
    t0 = time.time()

    # Access the underlying per-rank models
    if hasattr(tp_model, "models"):
        models = tp_model.models
    elif hasattr(tp_model, "model_list"):
        models = tp_model.model_list
    else:
        # Fallback: just delete the top-level object
        logger.warning("  Cannot access per-rank models, using simple delete")
        del tp_model
        gc.collect()
        return

    # Delete each rank's model individually
    for i in range(len(models)):
        models[i] = None
    del models
    del tp_model
    gc.collect()

    # Force Python GC to run multiple generations
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    # Give NRT time to release resources
    import time as _time

    _time.sleep(2)

    logger.info("  %s unloaded in %.1fs", name, time.time() - t0)


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


def load_spatial_upscaler(spatial_path, dtype=torch.bfloat16):
    """Load only the spatial upscaler (for two-stage mode).

    Two-stage mode uses spatial-only upscaling between stages (no temporal).
    This follows the DistilledPipeline reference which calls upsample_video()
    with only the spatial upsampler.
    """
    import json as _json

    from safetensors import safe_open
    from safetensors.torch import load_file
    from ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator

    logger.info("Loading spatial upsampler from %s...", spatial_path)
    t0 = time.time()

    with safe_open(spatial_path, framework="pt") as f:
        metadata = f.metadata()
    upsampler_config = _json.loads(metadata["config"])

    upsampler = LatentUpsamplerConfigurator.from_config(upsampler_config)
    upsampler = upsampler.to(dtype=dtype)
    upsampler.eval()

    sd = load_file(spatial_path)
    sd = {k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items()}
    upsampler.load_state_dict(sd, strict=False)
    total_params = sum(p.numel() for p in upsampler.parameters())
    logger.info(
        "  Spatial upsampler: %.1fM params, loaded in %.1fs",
        total_params / 1e6,
        time.time() - t0,
    )
    del sd
    return upsampler


def spatial_upscale_latent(video_latent_5d, video_decoder, spatial_upsampler):
    """Spatial-only upscale for two-stage pipeline.

    Doubles H and W while preserving frame count F.
    Flow: un_normalize -> spatial upsample x2 -> re_normalize.

    This follows the DistilledPipeline reference (ltx-core upsample_video):
      latent = video_encoder.per_channel_statistics.un_normalize(latent)
      latent = upsampler(latent)
      latent = video_encoder.per_channel_statistics.normalize(latent)

    Args:
        video_latent_5d: (B, C, F, H, W) normalized video latent
        video_decoder: VideoDecoder with per_channel_statistics
        spatial_upsampler: LatentUpsampler for spatial x2

    Returns:
        (B, C, F, H*2, W*2) normalized video latent (F unchanged)
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

    # Spatial upsample only (H, W doubled, F unchanged)
    t0 = time.time()
    with torch.no_grad():
        latent = spatial_upsampler(latent)
    logger.info("  After spatial x2: %s in %.1fs", latent.shape, time.time() - t0)

    # Re-normalize
    latent = pcs.normalize(latent)
    logger.info(
        "  Re-normalized: %s (mean=%.3f, std=%.3f)",
        latent.shape,
        latent.float().mean().item(),
        latent.float().std().item(),
    )

    return latent


def create_app_compositor(
    model_path,
    encoder_path,
    tp_degree=4,
    text_seq=256,
    height=384,
    width=512,
    num_frames=25,
    audio_seq=APP_BACKBONE_AUDIO_SEQ,
):
    """Create NeuronLTX23Application compositor with proper configs.

    Returns:
        NeuronLTX23Application instance ready for compile/load.
    """
    from neuronx_distributed_inference.models.config import NeuronConfig
    from modeling_ltx23 import LTX23BackboneInferenceConfig
    from modeling_gemma3_encoder import (
        Gemma3EncoderInferenceConfig,
        GEMMA3_12B_CONFIG,
    )
    from application import NeuronLTX23Application

    config = load_config(model_path)
    tc = config["transformer"]

    num_heads = tc["num_attention_heads"]
    head_dim = tc["attention_head_dim"]
    inner_dim = num_heads * head_dim
    audio_num_heads = tc["audio_num_attention_heads"]
    audio_head_dim = tc["audio_attention_head_dim"]
    audio_inner_dim = audio_num_heads * audio_head_dim
    audio_ca_dim = tc.get("audio_cross_attention_dim", 2048)

    latent_h = height // 32
    latent_w = width // 32
    latent_f = (num_frames - 1) // 8 + 1
    video_seq = latent_f * latent_h * latent_w

    dtype = torch.bfloat16

    # Backbone InferenceConfig
    backbone_neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        world_size=tp_degree,
        batch_size=1,
        seq_len=video_seq,
        torch_dtype=dtype,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )
    backbone_config = LTX23BackboneInferenceConfig(
        neuron_config=backbone_neuron_config,
        num_layers=tc["num_layers"],
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        inner_dim=inner_dim,
        audio_num_attention_heads=audio_num_heads,
        audio_attention_head_dim=audio_head_dim,
        audio_inner_dim=audio_inner_dim,
        audio_cross_attention_dim=audio_ca_dim,
        video_seq=video_seq,
        audio_seq=audio_seq,
        text_seq=APP_BACKBONE_TEXT_SEQ,  # Must match Application-compiled backbone
        height=latent_h,
        width=latent_w,
        num_frames=latent_f,
        ltx_config_dict=config,
    )

    # Gemma3 Encoder InferenceConfig
    encoder_neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        world_size=tp_degree,
        batch_size=1,
        seq_len=APP_ENCODER_SEQ_LEN,  # Must match Application-compiled encoder
        torch_dtype=dtype,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )
    encoder_config = Gemma3EncoderInferenceConfig(
        neuron_config=encoder_neuron_config,
        vocab_size=GEMMA3_12B_CONFIG["vocab_size"],
        hidden_size=GEMMA3_12B_CONFIG["hidden_size"],
        num_hidden_layers=GEMMA3_12B_CONFIG["num_hidden_layers"],
        num_attention_heads=GEMMA3_12B_CONFIG["num_attention_heads"],
        num_key_value_heads=GEMMA3_12B_CONFIG["num_key_value_heads"],
        head_dim=GEMMA3_12B_CONFIG["head_dim"],
        intermediate_size=GEMMA3_12B_CONFIG["intermediate_size"],
        rms_norm_eps=GEMMA3_12B_CONFIG["rms_norm_eps"],
        rope_theta=GEMMA3_12B_CONFIG["rope_theta"],
        max_position_embeddings=GEMMA3_12B_CONFIG["max_position_embeddings"],
        query_pre_attn_scalar=GEMMA3_12B_CONFIG["query_pre_attn_scalar"],
        pad_token_id=GEMMA3_12B_CONFIG["pad_token_id"],
    )

    app = NeuronLTX23Application(
        backbone_config=backbone_config,
        encoder_config=encoder_config,
        model_path=model_path,
        encoder_path=encoder_path,
    )
    logger.info(
        "NeuronLTX23Application created (video_seq=%d, text_seq=%d)",
        video_seq,
        text_seq,
    )
    return app


def encode_text_with_app(
    app, compiled_dir, tokenizer_path, prompt, text_seq, embeddings_processor
):
    """Encode text using the Application-loaded Gemma3 encoder.

    Handles: load encoder → tokenize → forward → process → unload.

    Args:
        app: NeuronLTX23Application compositor
        compiled_dir: Base compiled directory with text_encoder/ subdir
        tokenizer_path: Path to Gemma3 tokenizer (HuggingFace dir)
        prompt: Text prompt to encode
        text_seq: Text sequence length for the backbone (256)
        embeddings_processor: CPU EmbeddingsProcessor for post-processing

    Returns:
        (video_context, audio_context, context_mask) tensors
    """
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer

    # Load encoder to NeuronCores
    logger.info("Loading Gemma3 encoder via Application...")
    t0 = time.time()
    app.load_text_encoder(compiled_dir)
    logger.info("Gemma3 encoder loaded in %.1fs", time.time() - t0)

    # Use the Application-compiled encoder seq_len (512), not the standalone one (1024)
    compiled_seq_len = APP_ENCODER_SEQ_LEN

    # Warmup
    logger.info("  Warmup forward pass...")
    t0 = time.time()
    warmup_ids = torch.zeros(1, compiled_seq_len, dtype=torch.int64)
    warmup_mask = torch.ones(1, compiled_seq_len, dtype=torch.int64)
    with torch.no_grad():
        _ = app.encode_text(warmup_ids, warmup_mask)
    logger.info("  Warmup done in %.1fs", time.time() - t0)

    # Tokenize
    tokenizer = LTXVGemmaTokenizer(
        tokenizer_path=tokenizer_path,
        max_length=text_seq,
    )
    token_pairs = tokenizer.tokenize_with_weights(prompt)["gemma"]
    input_ids = torch.tensor([[t[0] for t in token_pairs]], dtype=torch.int64)
    attention_mask = torch.tensor([[w[1] for w in token_pairs]], dtype=torch.int64)
    actual_len = input_ids.shape[1]

    # compiled_seq_len already set to APP_ENCODER_SEQ_LEN above
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

    # Forward
    t0 = time.time()
    with torch.no_grad():
        stacked = app.encode_text(input_ids, attention_mask)
    logger.info("  Application Gemma3 forward: %.3fs", time.time() - t0)

    # Trim padding
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

    # Unload encoder to free NeuronCores for backbone
    app.unload_text_encoder()
    logger.info("  Gemma3 encoder unloaded")

    return result.video_encoding, result.audio_encoding, result.attention_mask


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

    # When using Neuron Gemma3, we must load/run/unload Gemma3 BEFORE loading
    # the DiT backbone, since both share the same 4 NeuronCores. Loading both
    # simultaneously causes memory contention and extreme swap thrashing
    # (144s+ for the first denoising step instead of 0.3s).
    #
    # Sequence: CPU components -> Gemma3 on Neuron -> encode -> unload Gemma3
    #           -> DiT on Neuron -> denoise -> decode

    # Get context embeddings
    dtype = torch.bfloat16
    app = None  # NeuronLTX23Application compositor (only used with --use-app)
    if args.use_app:
        logger.info("\n=== Using Application path (recommended) ===")
        app = create_app_compositor(
            model_path=args.model_path,
            encoder_path=args.gemma_path,
            tp_degree=args.tp_degree,
            text_seq=args.text_seq,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
        t0 = time.time()
        video_context, audio_context, context_mask = encode_text_with_app(
            app=app,
            compiled_dir=args.app_compiled_dir,
            tokenizer_path=args.gemma_path,
            prompt=args.prompt,
            text_seq=args.text_seq,
            embeddings_processor=cpu["embeddings_processor"],
        )
        logger.info("Text encoded via Application in %.1fs", time.time() - t0)
        logger.info(
            "  video_context: %s, audio_context: %s",
            video_context.shape,
            audio_context.shape,
        )
    elif args.no_text_encoder:
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

        # Free Gemma3 Neuron model — must fully release NeuronCores
        # before loading the DiT backbone which shares the same 4 cores
        unload_neuron_model(neuron_gemma3, "Gemma3 encoder")
        del neuron_gemma3
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

    # =========================================================================
    # TWO-STAGE PIPELINE
    # =========================================================================
    if args.two_stage:
        logger.info("\n" + "=" * 60)
        logger.info("TWO-STAGE GENERATION")
        logger.info(
            "  Stage 1: %dx%d (%d steps)",
            args.height // 2,
            args.width // 2,
            args.num_steps,
        )
        logger.info("  Stage 2: %dx%d (3 steps, refinement)", args.height, args.width)
        logger.info("=" * 60)

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
        from pipeline import NeuronTransformerWrapper

        # --- Stage 1: Half-res generation ---
        logger.info("\n=== Stage 1: Half-resolution generation ===")

        # Half-res latent dimensions
        s1_height = args.height // 2
        s1_width = args.width // 2
        s1_latent_h = s1_height // 32
        s1_latent_w = s1_width // 32
        s1_latent_f = (args.num_frames - 1) // 8 + 1
        logger.info(
            "  Half-res: %dx%d, latent %dx%dx%d",
            s1_height,
            s1_width,
            s1_latent_f,
            s1_latent_h,
            s1_latent_w,
        )

        s1_video_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=s1_latent_f,
            height=s1_latent_h,
            width=s1_latent_w,
        )
        v_patchifier = VideoLatentPatchifier(patch_size=1)
        v_scale = SpatioTemporalScaleFactors.default()
        s1_video_tools = VideoLatentTools(
            target_shape=s1_video_shape,
            patchifier=v_patchifier,
            scale_factors=v_scale,
            causal_fix=False,
            fps=args.fps,
        )
        audio_shape = AudioLatentShape(
            batch=1, channels=8, frames=args.audio_num_frames, mel_bins=16
        )
        a_patchifier = AudioPatchifier(patch_size=16)
        audio_tools = AudioLatentTools(
            patchifier=a_patchifier, target_shape=audio_shape
        )

        # Create initial noise at half-res
        gen = torch.Generator().manual_seed(args.seed)
        s1_video_state = s1_video_tools.create_initial_state(device="cpu", dtype=dtype)
        audio_state = audio_tools.create_initial_state(device="cpu", dtype=dtype)
        video_sample = torch.randn(
            s1_video_state.latent.shape, dtype=dtype, generator=gen
        )
        audio_sample = torch.randn(audio_state.latent.shape, dtype=dtype, generator=gen)

        # Load half-res Neuron backbone
        if args.use_app:
            # Application path: create half-res compositor and load backbone
            s1_app = create_app_compositor(
                model_path=args.model_path,
                encoder_path=args.gemma_path,
                tp_degree=args.tp_degree,
                text_seq=args.text_seq,
                height=s1_height,
                width=s1_width,
                num_frames=args.num_frames,
            )
            logger.info(
                "Loading half-res backbone from %s...",
                args.app_halfres_compiled_dir,
            )
            t0 = time.time()
            s1_app.load_backbone(args.app_halfres_compiled_dir)
            logger.info(
                "Half-res backbone loaded via Application in %.1fs",
                time.time() - t0,
            )
            neuron_backbone = s1_app
            wrapper = NeuronTransformerWrapper(
                compiled_backbone=neuron_backbone,
                cpu_ltx_model=cpu["ltx_model"],
                text_seq=args.text_seq,
                mask_4d=True,
                compiled_text_seq=APP_BACKBONE_TEXT_SEQ,
            )
        else:
            logger.info(
                "Loading half-res backbone from %s...", args.halfres_compiled_dir
            )
            neuron_backbone = load_neuron_backbone(
                args.halfres_compiled_dir,
                args.model_path,
                args.tp_degree,
                sharded_dir=args.halfres_sharded_dir
                if os.path.isdir(args.halfres_sharded_dir)
                else args.backbone_sharded_dir,
            )
            wrapper = NeuronTransformerWrapper(
                compiled_backbone=neuron_backbone,
                cpu_ltx_model=cpu["ltx_model"],
                text_seq=args.text_seq,
            )

        # Warmup half-res backbone
        logger.info("Warming up half-res DiT backbone...")
        warmup_sigma = torch.tensor([1.0])
        warmup_v_ts = warmup_sigma.unsqueeze(0).expand(
            1, s1_video_state.latent.shape[1]
        )
        warmup_a_ts = warmup_sigma.unsqueeze(0).expand(1, audio_state.latent.shape[1])
        warmup_video_mod = Modality(
            latent=torch.randn_like(video_sample),
            sigma=warmup_sigma,
            timesteps=warmup_v_ts,
            positions=s1_video_state.positions,
            context=video_context,
            enabled=True,
            context_mask=context_mask,
            attention_mask=None,
        )
        warmup_audio_mod = Modality(
            latent=torch.randn_like(audio_sample),
            sigma=warmup_sigma,
            timesteps=warmup_a_ts,
            positions=audio_state.positions,
            context=audio_context,
            enabled=True,
            context_mask=context_mask.clone(),
            attention_mask=None,
        )
        t0 = time.time()
        with torch.no_grad():
            _ = wrapper(warmup_video_mod, warmup_audio_mod)
        logger.info("  Half-res warmup done in %.1fs", time.time() - t0)
        del warmup_video_mod, warmup_audio_mod

        # Stage 1 denoising (8 steps at half-res)
        logger.info(
            "\n=== Stage 1 Denoising (%d steps at %dx%d) ===",
            args.num_steps,
            s1_height,
            s1_width,
        )
        sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32)
        s1_total_time = 0.0
        for step_idx in range(args.num_steps):
            sigma = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]
            video_seq_len = s1_video_state.latent.shape[1]
            audio_seq_len = audio_state.latent.shape[1]
            v_ts = sigma.unsqueeze(0).unsqueeze(0).expand(1, video_seq_len)
            a_ts = sigma.unsqueeze(0).unsqueeze(0).expand(1, audio_seq_len)
            video_mod = Modality(
                latent=video_sample,
                sigma=sigma.unsqueeze(0),
                timesteps=v_ts,
                positions=s1_video_state.positions,
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
            s1_total_time += step_time
            dt = sigma_next - sigma
            video_sample = (video_sample.float() + video_velocity.float() * dt).to(
                dtype
            )
            audio_sample = (audio_sample.float() + audio_velocity.float() * dt).to(
                dtype
            )
            logger.info(
                "  S1 Step %d/%d: sigma %.4f -> %.4f (%.1fs)",
                step_idx + 1,
                args.num_steps,
                sigma.item(),
                sigma_next.item(),
                step_time,
            )

        logger.info(
            "  Stage 1 total: %.1fs (%.1fs/step)",
            s1_total_time,
            s1_total_time / args.num_steps,
        )

        # Unload half-res backbone
        if args.use_app:
            s1_app.unload_backbone()
            del s1_app
        else:
            unload_neuron_model(neuron_backbone, "half-res DiT backbone")
        del neuron_backbone, wrapper

        # Unpatchify Stage 1 output to spatial format
        s1_video_latent = v_patchifier.unpatchify(video_sample, s1_video_shape)
        logger.info("  Stage 1 video latent: %s", s1_video_latent.shape)

        # --- Spatial upsample x2 ---
        logger.info("\n=== Spatial Upsample x2 ===")
        spatial_up = load_spatial_upscaler(args.spatial_upscaler_path, dtype=dtype)
        s2_video_latent = spatial_upscale_latent(
            s1_video_latent, cpu["video_decoder"], spatial_up
        )
        logger.info(
            "  Upscaled: %s -> %s", s1_video_latent.shape, s2_video_latent.shape
        )
        del spatial_up, s1_video_latent
        gc.collect()

        # --- Stage 2: Full-res refinement ---
        logger.info("\n=== Stage 2: Full-resolution refinement ===")

        # Full-res latent dimensions
        s2_latent_h = args.height // 32
        s2_latent_w = args.width // 32
        s2_latent_f = s2_video_latent.shape[2]  # Same frame count as Stage 1
        s2_video_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=s2_latent_f,
            height=s2_latent_h,
            width=s2_latent_w,
        )
        s2_video_tools = VideoLatentTools(
            target_shape=s2_video_shape,
            patchifier=v_patchifier,
            scale_factors=v_scale,
            causal_fix=False,
            fps=args.fps,
        )
        s2_video_state = s2_video_tools.create_initial_state(device="cpu", dtype=dtype)

        # Patchify the upscaled latent for Stage 2 denoising
        s2_upscaled_tokens = v_patchifier.patchify(s2_video_latent)
        logger.info("  S2 upscaled tokens: %s", s2_upscaled_tokens.shape)

        # Noise injection: mix upscaled latent with noise at sigma=0.909375
        s2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32)
        noise_scale = s2_sigmas[0].item()
        gen_s2 = torch.Generator().manual_seed(args.seed + 42)
        s2_noise = torch.randn(s2_upscaled_tokens.shape, dtype=dtype, generator=gen_s2)
        video_sample = (
            noise_scale * s2_noise + (1.0 - noise_scale) * s2_upscaled_tokens
        ).to(dtype)
        logger.info(
            "  Noise injected at sigma=%.4f: %.1f%% noise + %.1f%% signal",
            noise_scale,
            noise_scale * 100,
            (1 - noise_scale) * 100,
        )
        del s2_noise, s2_upscaled_tokens, s2_video_latent

        # Load full-res Neuron backbone (same compiled model, same weights)
        if args.use_app:
            # Application path: reuse the full-res compositor created for encoder
            logger.info("Loading full-res backbone from %s...", args.app_compiled_dir)
            t0 = time.time()
            app.load_backbone(args.app_compiled_dir)
            logger.info(
                "Full-res backbone loaded via Application in %.1fs",
                time.time() - t0,
            )
            neuron_backbone = app
            wrapper = NeuronTransformerWrapper(
                compiled_backbone=neuron_backbone,
                cpu_ltx_model=cpu["ltx_model"],
                text_seq=args.text_seq,
                mask_4d=True,
                compiled_text_seq=APP_BACKBONE_TEXT_SEQ,
            )
        else:
            logger.info("Loading full-res backbone from %s...", args.compile_dir)
            neuron_backbone = load_neuron_backbone(
                args.compile_dir,
                args.model_path,
                args.tp_degree,
                sharded_dir=args.backbone_sharded_dir,
            )
            wrapper = NeuronTransformerWrapper(
                compiled_backbone=neuron_backbone,
                cpu_ltx_model=cpu["ltx_model"],
                text_seq=args.text_seq,
            )

        # Warmup full-res backbone
        logger.info("Warming up full-res DiT backbone...")
        warmup_sigma = torch.tensor([1.0])
        warmup_v_ts = warmup_sigma.unsqueeze(0).expand(
            1, s2_video_state.latent.shape[1]
        )
        warmup_a_ts = warmup_sigma.unsqueeze(0).expand(1, audio_state.latent.shape[1])
        warmup_video_mod = Modality(
            latent=torch.randn(1, s2_video_state.latent.shape[1], 128, dtype=dtype),
            sigma=warmup_sigma,
            timesteps=warmup_v_ts,
            positions=s2_video_state.positions,
            context=video_context,
            enabled=True,
            context_mask=context_mask,
            attention_mask=None,
        )
        warmup_audio_mod = Modality(
            latent=torch.randn_like(audio_sample),
            sigma=warmup_sigma,
            timesteps=warmup_a_ts,
            positions=audio_state.positions,
            context=audio_context,
            enabled=True,
            context_mask=context_mask.clone(),
            attention_mask=None,
        )
        t0 = time.time()
        with torch.no_grad():
            _ = wrapper(warmup_video_mod, warmup_audio_mod)
        logger.info("  Full-res warmup done in %.1fs", time.time() - t0)
        del warmup_video_mod, warmup_audio_mod

        # Stage 2 denoising (3 steps at full-res, no CFG)
        s2_num_steps = len(s2_sigmas) - 1
        logger.info(
            "\n=== Stage 2 Denoising (%d steps at %dx%d) ===",
            s2_num_steps,
            args.height,
            args.width,
        )
        s2_total_time = 0.0
        for step_idx in range(s2_num_steps):
            sigma = s2_sigmas[step_idx]
            sigma_next = s2_sigmas[step_idx + 1]
            video_seq_len = s2_video_state.latent.shape[1]
            audio_seq_len = audio_state.latent.shape[1]
            v_ts = sigma.unsqueeze(0).unsqueeze(0).expand(1, video_seq_len)
            a_ts = sigma.unsqueeze(0).unsqueeze(0).expand(1, audio_seq_len)
            video_mod = Modality(
                latent=video_sample,
                sigma=sigma.unsqueeze(0),
                timesteps=v_ts,
                positions=s2_video_state.positions,
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
            s2_total_time += step_time
            dt = sigma_next - sigma
            video_sample = (video_sample.float() + video_velocity.float() * dt).to(
                dtype
            )
            audio_sample = (audio_sample.float() + audio_velocity.float() * dt).to(
                dtype
            )
            logger.info(
                "  S2 Step %d/%d: sigma %.4f -> %.4f (%.1fs)",
                step_idx + 1,
                s2_num_steps,
                sigma.item(),
                sigma_next.item(),
                step_time,
            )

        logger.info(
            "  Stage 2 total: %.1fs (%.1fs/step)",
            s2_total_time,
            s2_total_time / s2_num_steps,
        )
        logger.info(
            "\n  Combined denoising: S1=%.1fs + S2=%.1fs = %.1fs",
            s1_total_time,
            s2_total_time,
            s1_total_time + s2_total_time,
        )

        # Unpatchify and decode (reuse the existing decode path)
        video_latent_spatial = v_patchifier.unpatchify(video_sample, s2_video_shape)
        audio_latent_spatial = a_patchifier.unpatchify(audio_sample, audio_shape)
        video_shape = s2_video_shape  # for the decode path below

        # Unload full-res backbone before decode
        if args.use_app:
            app.unload_backbone()
        else:
            unload_neuron_model(neuron_backbone, "full-res DiT backbone")
        del neuron_backbone, wrapper

        # Skip to decode section (jump past single-stage code)
        # Set these for the shared decode path
        logger.info("\n=== Decoding ===")
        video_latent_4d = video_latent_spatial[0]
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
        video_frames = torch.cat(video_chunks, dim=0)
        logger.info(
            "  Video decoded: %s in %.1fs", video_frames.shape, time.time() - t0
        )

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
                audio_latent_spatial.float(),
                cpu["audio_decoder"].float(),
                cpu["vocoder"],
            )
        logger.info(
            "  Audio decoded: waveform %s, sr=%d in %.1fs",
            audio_result.waveform.shape,
            audio_result.sampling_rate,
            time.time() - t0,
        )

        try:
            import torchaudio

            wav_path = os.path.join(args.output_dir, "output.wav")
            torchaudio.save(
                wav_path, audio_result.waveform.cpu(), audio_result.sampling_rate
            )
            logger.info("  Saved WAV: %s", wav_path)
        except ImportError:
            wav_path = os.path.join(args.output_dir, "audio_waveform.pt")
            torch.save(
                {
                    "waveform": audio_result.waveform.cpu(),
                    "sr": audio_result.sampling_rate,
                },
                wav_path,
            )
            logger.info("  Saved audio tensor: %s", wav_path)

        logger.info("\n=== Done! Two-stage output saved to %s ===", args.output_dir)
        return

    # =========================================================================
    # SINGLE-STAGE PIPELINE (original path)
    # =========================================================================

    # Load Neuron backbone — AFTER text encoding to avoid NeuronCore contention
    # When using --neuron-gemma or --use-app, Gemma3 was already unloaded above
    logger.info("\n=== Loading Neuron backbone ===")
    if args.use_app:
        # Application path: load backbone via compositor
        t0 = time.time()
        app.load_backbone(args.app_compiled_dir)
        logger.info("Backbone loaded via Application in %.1fs", time.time() - t0)
        neuron_backbone = app  # app is callable with the same 24-tensor interface
    else:
        neuron_backbone = load_neuron_backbone(
            args.compile_dir,
            args.model_path,
            args.tp_degree,
            sharded_dir=args.backbone_sharded_dir,
        )

    # Build pipeline wrapper
    from pipeline import NeuronTransformerWrapper

    # Application-compiled NEFFs use text_seq=256 (matching standalone); no padding needed
    compiled_text_seq = APP_BACKBONE_TEXT_SEQ if args.use_app else None
    wrapper = NeuronTransformerWrapper(
        compiled_backbone=neuron_backbone,
        cpu_ltx_model=cpu["ltx_model"],
        text_seq=args.text_seq,
        mask_4d=args.use_app,
        compiled_text_seq=compiled_text_seq,
    )

    # Setup latent tools (needed for warmup and denoising)
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
    v_scale = SpatioTemporalScaleFactors.default()  # time=8, height=32, width=32
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

    # Image-to-Video conditioning: encode input image, replace frame 0 tokens,
    # build denoise_mask for per-token sigma control
    denoise_mask = None  # None = text-to-video mode (all tokens denoised equally)
    clean_latent = None
    if args.image:
        logger.info("\n=== Image-to-Video conditioning ===")
        # Encode the input image into normalized latent space
        # Returns (1, 128, 1, latent_h, latent_w) normalized latent
        image_latent_5d = encode_image(
            args.image, args.model_path, args.height, args.width, dtype
        )

        # Patchify the image latent to get token representation
        # Same patchifier as video — patch_size=1 so it's rearrange(b c f h w -> b (f*h*w) c)
        image_tokens = v_patchifier.patchify(image_latent_5d)
        # image_tokens shape: (1, latent_h * latent_w, 128)
        frame_0_tokens = latent_h * latent_w
        logger.info(
            "  Image patchified: %s (frame 0 = %d tokens)",
            image_tokens.shape,
            frame_0_tokens,
        )

        # Replace frame 0 noise tokens with encoded image tokens
        video_sample[:, :frame_0_tokens] = image_tokens[:, :frame_0_tokens]

        # Build denoise_mask: 0.0 for conditioned frame 0, 1.0 for unconditioned rest
        # Shape: (1, video_seq_len, 1) — broadcastable with latent (1, seq, C)
        video_seq_len = video_sample.shape[1]
        denoise_mask = torch.ones(1, video_seq_len, 1, dtype=dtype)
        denoise_mask[:, :frame_0_tokens, :] = 0.0
        logger.info(
            "  Denoise mask: %d conditioned tokens, %d unconditioned tokens",
            frame_0_tokens,
            video_seq_len - frame_0_tokens,
        )

        # Store clean latent for post-step preservation of frame 0
        clean_latent = video_sample.clone()

        logger.info("  I2V conditioning applied")

    # Warmup the DiT backbone — first call loads NEFF onto NeuronCores
    # Without this, the first 1-2 denoising steps take 100-200s instead of 0.3s
    logger.info("\n=== Warming up DiT backbone ===")
    warmup_sigma = torch.tensor([1.0])
    warmup_v_ts = warmup_sigma.unsqueeze(0).expand(1, video_state.latent.shape[1])
    warmup_a_ts = warmup_sigma.unsqueeze(0).expand(1, audio_state.latent.shape[1])
    warmup_ctx = (
        video_context
        if video_context is not None
        else torch.randn(1, args.text_seq, 4096, dtype=dtype)
    )
    warmup_actx = (
        audio_context
        if audio_context is not None
        else torch.randn(1, args.text_seq, 2048, dtype=dtype)
    )
    warmup_mask = (
        context_mask
        if context_mask is not None
        else torch.ones(1, args.text_seq, dtype=torch.int64)
    )

    warmup_video_mod = Modality(
        latent=torch.randn_like(video_sample),
        sigma=warmup_sigma,
        timesteps=warmup_v_ts,
        positions=video_state.positions,
        context=warmup_ctx,
        enabled=True,
        context_mask=warmup_mask,
        attention_mask=None,
    )
    warmup_audio_mod = Modality(
        latent=torch.randn_like(audio_sample),
        sigma=warmup_sigma,
        timesteps=warmup_a_ts,
        positions=audio_state.positions,
        context=warmup_actx,
        enabled=True,
        context_mask=warmup_mask.clone(),
        attention_mask=None,
    )
    t0 = time.time()
    with torch.no_grad():
        _ = wrapper(warmup_video_mod, warmup_audio_mod)
    logger.info("  DiT backbone warmup done in %.1fs", time.time() - t0)
    del warmup_video_mod, warmup_audio_mod

    logger.info("\n=== Denoising (%d steps) ===", args.num_steps)
    logger.info(
        "  Video latent: %s, Audio latent: %s", video_sample.shape, audio_sample.shape
    )

    # Sigma schedule — use distilled values for the distilled model
    # The distilled model was trained with these exact sigma values
    sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32)
    assert len(sigmas) == args.num_steps + 1, (
        f"Distilled sigma values have {len(sigmas)} entries "
        f"but {args.num_steps} steps require {args.num_steps + 1}"
    )
    logger.info("  Sigmas: %s", [f"{s:.4f}" for s in sigmas.tolist()])

    # Denoising loop
    total_time = 0.0
    for step_idx in range(args.num_steps):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        video_seq_len = video_state.latent.shape[1]
        audio_seq_len = audio_state.latent.shape[1]

        # Per-token timesteps: in I2V mode, frame 0 tokens get timestep=0
        # (already clean), while all other tokens get timestep=sigma
        if denoise_mask is not None:
            # denoise_mask: (1, video_seq, 1), squeeze last dim for timesteps
            v_ts = denoise_mask.squeeze(-1) * sigma  # (1, video_seq)
        else:
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

        # I2V: preserve frame 0 tokens (conditioned) after each Euler step
        # This ensures the clean image latent is never corrupted by denoising
        if denoise_mask is not None and clean_latent is not None:
            # denoise_mask: (1, seq, 1), video_sample: (1, seq, C)
            # Where mask=0 (frame 0): use clean_latent; where mask=1: keep denoised
            video_sample = (
                video_sample * denoise_mask + clean_latent * (1.0 - denoise_mask)
            ).to(dtype)

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
    parser.add_argument(
        "--backbone-sharded-dir",
        default="/home/ubuntu/backbone_sharded",
        help="Directory with pre-sharded backbone weights (from shard_backbone_weights.py). "
        "Falls back to safetensors loading if not found.",
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
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input image for image-to-video generation. "
        "When specified, frame 0 is conditioned on the encoded image "
        "and only subsequent frames are denoised.",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Two-stage generation: Stage 1 at half-res (192x256), spatial upsample x2, "
        "then Stage 2 refinement at full-res (384x512). Produces sharper output than "
        "single-stage at the cost of additional compilation + denoising steps.",
    )
    parser.add_argument(
        "--halfres-compiled-dir",
        default=HALFRES_COMPILE_DIR,
        help="Directory with half-res compiled backbone (from compile_transformer_halfres.py)",
    )
    parser.add_argument(
        "--halfres-sharded-dir",
        default=HALFRES_SHARDED_DIR,
        help="Directory with pre-sharded backbone weights for half-res model "
        "(same weights, different compiled shape)",
    )
    parser.add_argument(
        "--use-app",
        action="store_true",
        help="Use NeuronApplicationBase path for both Gemma3 encoder and DiT backbone. "
        "Faster than --neuron-gemma due to ModelBuilder weight layout optimization. "
        "Requires --app-compiled-dir with backbone/ and text_encoder/ subdirs.",
    )
    parser.add_argument(
        "--app-compiled-dir",
        default=APP_COMPILED_DIR,
        help="Base directory with Application-compiled artifacts. "
        "Must contain backbone/ and text_encoder/ subdirs (from NeuronLTX23Application.compile()).",
    )
    parser.add_argument(
        "--app-halfres-compiled-dir",
        default=APP_HALFRES_COMPILED_DIR,
        help="Base directory with half-res Application-compiled backbone. "
        "Used for Stage 1 of --two-stage --use-app. Must contain backbone/ subdir.",
    )

    args = parser.parse_args()

    if args.use_app:
        if args.gemma_path is None:
            args.gemma_path = GEMMA3_MODEL_PATH
    elif not args.no_text_encoder and not args.neuron_gemma and args.gemma_path is None:
        parser.error(
            "Either --no-text-encoder, --neuron-gemma, --use-app, or --gemma-path must be specified. "
            "Use --use-app for Application-compiled models (fastest, recommended)."
        )

    generate(args)


if __name__ == "__main__":
    main()
