"""
Phase 2 of the two-stage LTX-2.3 benchmark on trn2.48xlarge.

Loads Stage 1 latent saved by Phase 1, then runs:
  - Spatial upsample x2 (CPU)
  - Stage 2 denoising at full resolution (Neuron, TP=16)
  - VAE decode (Neuron tiled or CPU fallback) + save frames + MP4

Usage:
    # Phase 1 (TP=4, separate process):
    python generate_ltx23.py --two-stage --use-app --save-s1-latent /mnt/models/s1_latent.pt ...

    # Phase 2 (TP=16, this script) with Neuron VAE:
    python run_phase2.py \
        --model-path /mnt/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
        --s1-latent /mnt/models/s1_latent.pt \
        --s2-compiled-dir /mnt/models/compiled/benchmark/s2_tp16 \
        --spatial-upscaler-path /mnt/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
        --vae-compiled-dir /mnt/models/compiled/vae_tp4_4x16 \
        --height 1024 --width 1536 --num-frames 121 \
        --tp-degree 16 \
        --output-dir /mnt/models/output/benchmark

    # Without --vae-compiled-dir, falls back to CPU decode.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("run_phase2")

# Must match Phase 1 compile constants
APP_BACKBONE_TEXT_SEQ = 256
APP_BACKBONE_AUDIO_SEQ = 26
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


def load_config(model_path):
    from safetensors import safe_open

    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
    return json.loads(metadata["config"])


def load_cpu_components(model_path, dtype=torch.bfloat16):
    """Load CPU components needed for decode."""
    from generate_ltx23 import build_cpu_components

    config = load_config(model_path)
    return build_cpu_components(config, model_path, dtype)


def create_app_compositor(
    model_path,
    encoder_path,
    tp_degree,
    text_seq,
    height,
    width,
    num_frames,
    audio_seq=APP_BACKBONE_AUDIO_SEQ,
):
    """Create NeuronLTX23Application compositor for S2 backbone."""
    from neuronx_distributed_inference.models.config import NeuronConfig
    from modeling_ltx23 import LTX23BackboneInferenceConfig
    from modeling_gemma3_encoder import Gemma3EncoderInferenceConfig, GEMMA3_12B_CONFIG
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

    backbone_nc = NeuronConfig(
        tp_degree=tp_degree,
        world_size=tp_degree,
        batch_size=1,
        seq_len=video_seq,
        torch_dtype=dtype,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )
    backbone_config = LTX23BackboneInferenceConfig(
        neuron_config=backbone_nc,
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
        text_seq=APP_BACKBONE_TEXT_SEQ,
        height=latent_h,
        width=latent_w,
        num_frames=latent_f,
        ltx_config_dict=config,
    )

    # Encoder config needed by Application even though we don't use it in Phase 2
    # Use TP=4 for encoder config since we won't actually load it
    enc_tp = min(tp_degree, 4)  # Gemma3 KV heads = 4, can't go beyond TP=4
    encoder_nc = NeuronConfig(
        tp_degree=enc_tp,
        world_size=enc_tp,
        batch_size=1,
        seq_len=512,
        torch_dtype=dtype,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )
    encoder_config = Gemma3EncoderInferenceConfig(
        neuron_config=encoder_nc,
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
        encoder_path=encoder_path or model_path,
    )
    logger.info("S2 compositor created (video_seq=%d, TP=%d)", video_seq, tp_degree)
    return app


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Phase 2: S2 refinement")
    parser.add_argument("--model-path", required=True, help="LTX-2.3 safetensors")
    parser.add_argument(
        "--s1-latent", required=True, help="Path to Phase 1 saved latent"
    )
    parser.add_argument(
        "--s2-compiled-dir", required=True, help="S2 compiled backbone dir"
    )
    parser.add_argument(
        "--spatial-upscaler-path", required=True, help="Spatial upscaler safetensors"
    )
    parser.add_argument(
        "--gemma-path", default=None, help="Gemma3 path (for Application init)"
    )
    parser.add_argument(
        "--vae-compiled-dir",
        default=None,
        help="Compiled Neuron VAE directory (if omitted, falls back to CPU decode)",
    )
    parser.add_argument("--height", type=int, required=True, help="Full-res height")
    parser.add_argument("--width", type=int, required=True, help="Full-res width")
    parser.add_argument(
        "--num-frames", type=int, required=True, help="Number of frames"
    )
    parser.add_argument("--tp-degree", type=int, default=16, help="TP degree for S2")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for S2 noise")
    parser.add_argument("--fps", type=float, default=25.0, help="Video frame rate")
    parser.add_argument(
        "--audio-num-frames", type=int, default=26, help="Audio latent frames"
    )
    parser.add_argument("--text-seq", type=int, default=256, help="Text seq length")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    dtype = torch.bfloat16
    total_t0 = time.time()

    # 1. Load Phase 1 latent
    logger.info("\n=== Loading Phase 1 latent from %s ===", args.s1_latent)
    saved = torch.load(args.s1_latent, map_location="cpu", weights_only=True)
    s1_video_latent = saved["s1_video_latent"]
    audio_sample = saved["audio_sample"]
    video_context = saved["video_context"]
    audio_context = saved["audio_context"]
    context_mask = saved["context_mask"]
    s1_total_time = saved.get("s1_total_time", 0.0)
    logger.info("  S1 video latent: %s", s1_video_latent.shape)
    logger.info("  S1 denoising time: %.1fs", s1_total_time)

    # 2. Load CPU components (for VAE decode and spatial upscaler)
    logger.info("\n=== Loading CPU components ===")
    t0 = time.time()
    cpu = load_cpu_components(args.model_path, dtype)
    logger.info("  CPU components loaded in %.1fs", time.time() - t0)

    # 3. Spatial upsample
    logger.info("\n=== Spatial Upsample x2 ===")
    from generate_ltx23 import load_spatial_upscaler, spatial_upscale_latent

    spatial_up = load_spatial_upscaler(args.spatial_upscaler_path, dtype=dtype)
    s2_video_latent = spatial_upscale_latent(
        s1_video_latent, cpu["video_decoder"], spatial_up
    )
    logger.info("  Upscaled: %s -> %s", s1_video_latent.shape, s2_video_latent.shape)
    del spatial_up, s1_video_latent
    gc.collect()

    # 4. Setup Stage 2 latent tools
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

    s2_latent_h = args.height // 32
    s2_latent_w = args.width // 32
    s2_latent_f = s2_video_latent.shape[2]
    s2_video_shape = VideoLatentShape(
        batch=1,
        channels=128,
        frames=s2_latent_f,
        height=s2_latent_h,
        width=s2_latent_w,
    )
    v_patchifier = VideoLatentPatchifier(patch_size=1)
    v_scale = SpatioTemporalScaleFactors.default()
    s2_video_tools = VideoLatentTools(
        target_shape=s2_video_shape,
        patchifier=v_patchifier,
        scale_factors=v_scale,
        causal_fix=False,
        fps=args.fps,
    )
    s2_video_state = s2_video_tools.create_initial_state(device="cpu", dtype=dtype)

    audio_shape = AudioLatentShape(
        batch=1, channels=8, frames=args.audio_num_frames, mel_bins=16
    )
    a_patchifier = AudioPatchifier(patch_size=16)
    audio_tools = AudioLatentTools(patchifier=a_patchifier, target_shape=audio_shape)
    audio_state = audio_tools.create_initial_state(device="cpu", dtype=dtype)

    # Patchify upscaled latent
    s2_upscaled_tokens = v_patchifier.patchify(s2_video_latent)
    logger.info("  S2 upscaled tokens: %s", s2_upscaled_tokens.shape)

    # Noise injection
    s2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32)
    noise_scale = s2_sigmas[0].item()
    gen_s2 = torch.Generator().manual_seed(args.seed + 42)
    s2_noise = torch.randn(s2_upscaled_tokens.shape, dtype=dtype, generator=gen_s2)
    video_sample = (
        noise_scale * s2_noise + (1.0 - noise_scale) * s2_upscaled_tokens
    ).to(dtype)
    logger.info("  Noise injected at sigma=%.4f", noise_scale)
    del s2_noise, s2_upscaled_tokens, s2_video_latent

    # 5. Load S2 backbone via Application
    logger.info("\n=== Loading S2 backbone (TP=%d) ===", args.tp_degree)
    s2_app = create_app_compositor(
        model_path=args.model_path,
        encoder_path=args.gemma_path,
        tp_degree=args.tp_degree,
        text_seq=args.text_seq,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    t0 = time.time()
    s2_app.load_backbone(args.s2_compiled_dir)
    logger.info("  S2 backbone loaded in %.1fs", time.time() - t0)

    wrapper = NeuronTransformerWrapper(
        compiled_backbone=s2_app,
        cpu_ltx_model=cpu["ltx_model"],
        text_seq=args.text_seq,
        mask_4d=True,
        compiled_text_seq=APP_BACKBONE_TEXT_SEQ,
    )

    # Warmup
    logger.info("  Warming up S2 backbone...")
    warmup_sigma = torch.tensor([1.0])
    warmup_v_ts = warmup_sigma.unsqueeze(0).expand(1, s2_video_state.latent.shape[1])
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
    logger.info("  S2 warmup done in %.1fs", time.time() - t0)
    del warmup_video_mod, warmup_audio_mod

    # 6. Stage 2 denoising
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
        video_sample = (video_sample.float() + video_velocity.float() * dt).to(dtype)
        audio_sample = (audio_sample.float() + audio_velocity.float() * dt).to(dtype)
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
        "\n  Combined: S1=%.1fs + S2=%.1fs = %.1fs",
        s1_total_time,
        s2_total_time,
        s1_total_time + s2_total_time,
    )

    # Unload S2 backbone
    s2_app.unload_backbone()
    del s2_app, wrapper
    gc.collect()

    # 7. Decode
    logger.info("\n=== Decoding ===")
    video_latent_spatial = v_patchifier.unpatchify(video_sample, s2_video_shape)
    audio_latent_spatial = a_patchifier.unpatchify(audio_sample, audio_shape)
    video_latent_4d = video_latent_spatial[0]
    logger.info("  Video latent for VAE: %s", video_latent_4d.shape)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.vae_compiled_dir:
        # --- Neuron tiled VAE decode ---
        logger.info("  Using Neuron VAE from %s", args.vae_compiled_dir)
        from tiled_vae_decode_23 import (
            preprocess_latent,
            get_scaled_timestep,
            load_compiled_vae,
            tiled_decode,
        )

        # Preprocess on CPU (noise injection + denormalization)
        t0 = time.time()
        preprocessed = preprocess_latent(
            video_latent_spatial, cpu["video_decoder"], seed=42
        )
        scaled_ts = get_scaled_timestep(cpu["video_decoder"], batch_size=1)
        # The compiled VAE NEF always expects 2 inputs (latent + scaled_timestep),
        # even when timestep_conditioning=False in config (the value was constant-
        # folded during tracing). Provide a default if get_scaled_timestep returns None.
        if scaled_ts is None:
            scaled_ts = torch.tensor([0.05 * 1000.0], dtype=torch.float32)
            logger.info(
                "  Using default scaled_timestep=50.0 (constant-folded in compiled model)"
            )
        logger.info(
            "  Preprocessing: %.1fs (scaled_ts=%s)",
            time.time() - t0,
            scaled_ts,
        )

        # Unload S2 backbone models from Neuron before loading VAE
        # (they were already unloaded above, but ensure NRT is clear)

        # Load compiled VAE
        t0 = time.time()
        compiled_vae = load_compiled_vae(args.vae_compiled_dir)
        logger.info("  Compiled VAE loaded in %.1fs", time.time() - t0)

        # Tiled decode
        t0 = time.time()
        video_output = tiled_decode(
            preprocessed,
            compiled_vae,
            scaled_timestep=scaled_ts,
            tile_latent_h=4,
            tile_latent_w=16,
            overlap_latent_h=1,
            overlap_latent_w=0,
            verbose=True,
        )
        decode_time = time.time() - t0
        logger.info("  Neuron VAE decode: %.1fs", decode_time)

        # Convert to uint8 frames: [1, 3, T_out, H, W] -> [T_out, H, W, 3]
        video_output = video_output.clamp(0, 1)
        video_frames = (video_output[0].permute(1, 2, 3, 0) * 255).to(torch.uint8)
        logger.info("  Video frames: %s", video_frames.shape)

        del compiled_vae, preprocessed
        gc.collect()
    else:
        # --- CPU fallback decode ---
        logger.info("  Using CPU VAE decode (no --vae-compiled-dir provided)")
        logger.info("  Decoding video...")
        t0 = time.time()
        from ltx_core.model.video_vae.video_vae import decode_video

        video_chunks = []
        with torch.no_grad():
            for chunk in decode_video(video_latent_4d, cpu["video_decoder"]):
                video_chunks.append(chunk)
        video_frames = torch.cat(video_chunks, dim=0)
        decode_time = time.time() - t0
        logger.info("  Video decoded: %s in %.1fs", video_frames.shape, decode_time)

    from PIL import Image

    for i in range(video_frames.shape[0]):
        frame = video_frames[i].numpy()
        img = Image.fromarray(frame)
        img.save(os.path.join(args.output_dir, f"frame_{i:04d}.png"))
    logger.info("  Saved %d frames to %s", video_frames.shape[0], args.output_dir)

    # MP4
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
        logger.warning("  ffmpeg not available: %s", e)

    # Audio decode
    logger.info("  Decoding audio...")
    t0 = time.time()
    from ltx_core.model.audio_vae.audio_vae import decode_audio

    with torch.no_grad():
        audio_result = decode_audio(
            audio_latent_spatial.float(), cpu["audio_decoder"].float(), cpu["vocoder"]
        )
    logger.info("  Audio decoded in %.1fs", time.time() - t0)

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
            {"waveform": audio_result.waveform.cpu(), "sr": audio_result.sampling_rate},
            wav_path,
        )

    total_time = time.time() - total_t0
    vae_mode = "Neuron" if args.vae_compiled_dir else "CPU"
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info("  S1 denoising (from Phase 1): %.1fs", s1_total_time)
    logger.info(
        "  S2 denoising:                %.1fs (%.1fs/step)",
        s2_total_time,
        s2_total_time / s2_num_steps,
    )
    logger.info("  VAE decode (%s):          %.1fs", vae_mode, decode_time)
    logger.info("  Total Phase 2 wall time:     %.1fs", total_time)
    logger.info("  Output: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
