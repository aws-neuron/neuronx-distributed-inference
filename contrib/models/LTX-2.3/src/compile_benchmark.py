"""
Compile all 3 Application components for the full LTX-2.3 benchmark on trn2.48xlarge.

Components:
  1. Gemma3 encoder: TP=4, seq=512
  2. Stage 1 backbone: 512x768, 121 frames, TP=4 (video_seq=16*24*16=6144)
  3. Stage 2 backbone: 1024x1536, 121 frames, TP=16 (video_seq=32*48*16=24576)

Usage:
    python compile_benchmark.py \
        --model-path /mnt/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
        --encoder-path /mnt/models/gemma-3-12b \
        --output-base /mnt/models/compiled/benchmark

This creates:
    /mnt/models/compiled/benchmark/encoder_tp4/     (Gemma3 encoder TP=4)
    /mnt/models/compiled/benchmark/s1_tp4/          (Stage 1 backbone TP=4)
    /mnt/models/compiled/benchmark/s2_tp16/         (Stage 2 backbone TP=16)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("compile_benchmark")

# Benchmark dimensions
S1_HEIGHT = 512
S1_WIDTH = 768
S2_HEIGHT = 1024
S2_WIDTH = 1536
NUM_FRAMES = 121
TEXT_SEQ = 256
AUDIO_SEQ = 26
ENCODER_SEQ = 512
S1_TP = 4
S2_TP = 16


def load_ltx_config(model_path):
    from safetensors import safe_open

    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
    return json.loads(metadata["config"])


def compile_encoder(model_path, encoder_path, output_dir, tp_degree=S1_TP):
    """Compile Gemma3 encoder via Application path."""
    from neuronx_distributed_inference.models.config import NeuronConfig
    from modeling_ltx23 import LTX23BackboneInferenceConfig
    from modeling_gemma3_encoder import Gemma3EncoderInferenceConfig, GEMMA3_12B_CONFIG
    from application import NeuronLTX23Application

    config = load_ltx_config(model_path)
    tc = config["transformer"]

    num_heads = tc["num_attention_heads"]
    head_dim = tc["attention_head_dim"]
    inner_dim = num_heads * head_dim
    audio_num_heads = tc["audio_num_attention_heads"]
    audio_head_dim = tc["audio_attention_head_dim"]
    audio_inner_dim = audio_num_heads * audio_head_dim
    audio_ca_dim = tc.get("audio_cross_attention_dim", 2048)

    # Use S1 dimensions for the backbone config (encoder doesn't care about backbone dims)
    latent_h = S1_HEIGHT // 32
    latent_w = S1_WIDTH // 32
    latent_f = (NUM_FRAMES - 1) // 8 + 1
    video_seq = latent_f * latent_h * latent_w

    dtype = torch.bfloat16

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
        audio_seq=AUDIO_SEQ,
        text_seq=TEXT_SEQ,
        height=latent_h,
        width=latent_w,
        num_frames=latent_f,
        ltx_config_dict=config,
    )

    encoder_neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        world_size=tp_degree,
        batch_size=1,
        seq_len=ENCODER_SEQ,
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

    logger.info("Compiling Gemma3 encoder (TP=%d, seq=%d)...", tp_degree, ENCODER_SEQ)
    t0 = time.time()
    app.compile_encoder(output_dir)
    logger.info("Encoder compiled in %.1fs -> %s", time.time() - t0, output_dir)

    del app
    gc.collect()


def compile_backbone(
    model_path, encoder_path, output_dir, height, width, tp_degree, label="backbone"
):
    """Compile DiT backbone via Application path."""
    from neuronx_distributed_inference.models.config import NeuronConfig
    from modeling_ltx23 import LTX23BackboneInferenceConfig
    from modeling_gemma3_encoder import Gemma3EncoderInferenceConfig, GEMMA3_12B_CONFIG
    from application import NeuronLTX23Application

    config = load_ltx_config(model_path)
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
    latent_f = (NUM_FRAMES - 1) // 8 + 1
    video_seq = latent_f * latent_h * latent_w

    logger.info(
        "  %s: %dx%d -> latent %dx%dx%d -> video_seq=%d",
        label,
        height,
        width,
        latent_f,
        latent_h,
        latent_w,
        video_seq,
    )

    dtype = torch.bfloat16

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
        audio_seq=AUDIO_SEQ,
        text_seq=TEXT_SEQ,
        height=latent_h,
        width=latent_w,
        num_frames=latent_f,
        ltx_config_dict=config,
    )

    # Encoder config (needed for Application even if we only compile backbone)
    encoder_neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        world_size=tp_degree,
        batch_size=1,
        seq_len=ENCODER_SEQ,
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

    logger.info("Compiling %s (TP=%d, video_seq=%d)...", label, tp_degree, video_seq)
    t0 = time.time()
    app.compile_backbone(output_dir)
    logger.info("%s compiled in %.1fs -> %s", label, time.time() - t0, output_dir)

    del app
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Compile all benchmark components")
    parser.add_argument("--model-path", required=True, help="LTX-2.3 safetensors path")
    parser.add_argument("--encoder-path", required=True, help="Gemma3 12B model dir")
    parser.add_argument("--output-base", required=True, help="Base output directory")
    parser.add_argument(
        "--skip-encoder", action="store_true", help="Skip encoder compilation"
    )
    parser.add_argument(
        "--skip-s1", action="store_true", help="Skip Stage 1 backbone compilation"
    )
    parser.add_argument(
        "--skip-s2", action="store_true", help="Skip Stage 2 backbone compilation"
    )
    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)

    total_t0 = time.time()

    # 1. Gemma3 encoder at TP=4
    if not args.skip_encoder:
        encoder_dir = os.path.join(args.output_base, "encoder_tp4")
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Compile Gemma3 encoder (TP=%d)", S1_TP)
        logger.info("=" * 60)
        compile_encoder(
            args.model_path, args.encoder_path, encoder_dir, tp_degree=S1_TP
        )
    else:
        logger.info("Skipping encoder compilation")

    # 2. Stage 1 backbone at 512x768, TP=4
    if not args.skip_s1:
        s1_dir = os.path.join(args.output_base, "s1_tp4")
        logger.info("\n" + "=" * 60)
        logger.info(
            "STEP 2: Compile Stage 1 backbone (%dx%d, TP=%d)",
            S1_HEIGHT,
            S1_WIDTH,
            S1_TP,
        )
        logger.info("=" * 60)
        compile_backbone(
            args.model_path,
            args.encoder_path,
            s1_dir,
            height=S1_HEIGHT,
            width=S1_WIDTH,
            tp_degree=S1_TP,
            label="Stage 1 backbone",
        )
    else:
        logger.info("Skipping Stage 1 backbone compilation")

    # 3. Stage 2 backbone at 1024x1536, TP=16
    if not args.skip_s2:
        s2_dir = os.path.join(args.output_base, "s2_tp16")
        logger.info("\n" + "=" * 60)
        logger.info(
            "STEP 3: Compile Stage 2 backbone (%dx%d, TP=%d)",
            S2_HEIGHT,
            S2_WIDTH,
            S2_TP,
        )
        logger.info("=" * 60)
        compile_backbone(
            args.model_path,
            args.encoder_path,
            s2_dir,
            height=S2_HEIGHT,
            width=S2_WIDTH,
            tp_degree=S2_TP,
            label="Stage 2 backbone",
        )
    else:
        logger.info("Skipping Stage 2 backbone compilation")

    logger.info("\n" + "=" * 60)
    logger.info("ALL COMPILATIONS COMPLETE in %.1fs", time.time() - total_t0)
    logger.info("=" * 60)
    logger.info("  Encoder:  %s/encoder_tp4/", args.output_base)
    logger.info("  Stage 1:  %s/s1_tp4/", args.output_base)
    logger.info("  Stage 2:  %s/s2_tp16/", args.output_base)


if __name__ == "__main__":
    main()
