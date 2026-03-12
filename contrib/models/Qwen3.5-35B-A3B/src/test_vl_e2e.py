#!/usr/bin/env python3
"""
End-to-end VL test: Text decoder + vision encoder on Neuron.

Tests the full pipeline:
1. Compile text decoder WITH vision args (24-arg trace)
2. Load pre-compiled vision encoder
3. Process a real image through both models
4. Verify vision embedding injection works
5. Generate tokens from image+text prompt

Usage:
  export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  cd /home/ubuntu/nxdi-qwen35
  python test_vl_e2e.py 2>&1 | tee test_vl_e2e.log

Stages (can skip earlier stages with --skip-compile, --skip-load):
  Stage 1: Compile text decoder (24-arg trace with vision inputs)
  Stage 2: Load compiled models (text + vision)
  Stage 3: Text-only generation (sanity check)
  Stage 4: Vision + text generation (full VL pipeline)
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = "/mnt/models/Qwen3.5-35B-A3B"
COMPILED_TEXT_PATH = "/mnt/models/compiled_qwen35_vl/"
COMPILED_VISION_PATH = "/mnt/models/compiled_vision/vision_encoder.pt"

# Add the contrib model src to path
SRC_PATH = os.path.join(os.path.dirname(__file__), "contrib/models/Qwen3.5-35B-A3B/src")
if os.path.exists(SRC_PATH):
    sys.path.insert(0, SRC_PATH)
else:
    sys.path.insert(0, "/home/ubuntu/nxdi-qwen35/contrib/models/Qwen3.5-35B-A3B/src")


def create_text_config():
    """Create the NxDI text model config."""
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    from modeling_qwen35_moe import Qwen35MoeInferenceConfig

    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)

    neuron_config = MoENeuronConfig(
        tp_degree=4,
        max_batch_size=1,
        max_context_length=256,  # Larger for vision tokens
        max_new_tokens=32,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
        moe_tp_degree=4,
        moe_ep_degree=1,
        blockwise_matmul_config={"block_size": 2048},
    )

    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )
    if (
        "tie_word_embeddings" not in config_dict
        or config_dict["tie_word_embeddings"] is None
    ):
        config_dict["tie_word_embeddings"] = False

    config = Qwen35MoeInferenceConfig(
        neuron_config=neuron_config,
        **config_dict,
    )
    return config, full_config


def create_vl_config(full_config):
    """Create the VL config wrapping text + vision configs."""
    from modeling_qwen35_moe_vl import Qwen35MoeVLInferenceConfig

    return Qwen35MoeVLInferenceConfig(
        text_config=None,  # We pass text config separately
        vision_config=full_config.get("vision_config", {}),
        image_token_id=full_config.get("image_token_id", 248056),
        video_token_id=full_config.get("video_token_id", 248057),
        vision_start_token_id=full_config.get("vision_start_token_id", 248053),
        vision_end_token_id=full_config.get("vision_end_token_id", 248054),
        spatial_merge_size=full_config.get("vision_config", {}).get(
            "spatial_merge_size", 2
        ),
        vision_seq_len_buckets=[256, 1024, 4096],
    )


# ============================================================
# Stage 1: Compile text decoder with 24-arg vision trace
# ============================================================


def stage_compile(config):
    """Compile text decoder (with vision args in trace signature)."""
    from modeling_qwen35_moe import NeuronQwen35MoeForCausalLM

    logger.info("=" * 60)
    logger.info("STAGE 1: Compile text decoder (24-arg trace)")
    logger.info("=" * 60)

    os.makedirs(COMPILED_TEXT_PATH, exist_ok=True)

    # Remove XLA_DISABLE_FUNCTIONALIZATION if set (causes compilation issues)
    if "XLA_DISABLE_FUNCTIONALIZATION" in os.environ:
        del os.environ["XLA_DISABLE_FUNCTIONALIZATION"]

    model = NeuronQwen35MoeForCausalLM(model_path=MODEL_PATH, config=config)

    t0 = time.time()
    model.compile(COMPILED_TEXT_PATH)
    compile_time = time.time() - t0
    logger.info(f"  Compilation time: {compile_time:.1f}s")

    return model


# ============================================================
# Stage 2: Load compiled models
# ============================================================


def stage_load(config, full_config):
    """Load compiled text + vision models."""
    from modeling_qwen35_moe import NeuronQwen35MoeForCausalLM
    from modeling_qwen35_moe_vl import NeuronQwen35MoeVLForCausalLM

    logger.info("=" * 60)
    logger.info("STAGE 2: Load compiled text + vision models")
    logger.info("=" * 60)

    vl_config = create_vl_config(full_config)

    # Create VL model (wraps text + vision)
    vl_model = NeuronQwen35MoeVLForCausalLM(
        model_path=MODEL_PATH,
        text_config=config,
        vision_config=vl_config,
    )

    # Load text model
    logger.info("  Loading text model...")
    t0 = time.time()
    vl_model.text_model.load(COMPILED_TEXT_PATH)
    logger.info(f"  Text model loaded in {time.time() - t0:.1f}s")

    # Load vision model
    logger.info("  Loading vision model...")
    t0 = time.time()
    if os.path.exists(COMPILED_VISION_PATH):
        vl_model.vision_model_wrapper.load_compiled(COMPILED_VISION_PATH)
        vl_model.vision_model_wrapper.load_vision_weights_from_hf(MODEL_PATH)
        logger.info(f"  Vision model loaded in {time.time() - t0:.1f}s")
    else:
        logger.warning(f"  Vision model not found at {COMPILED_VISION_PATH}")
        logger.warning("  Vision encoding will not be available")

    return vl_model


# ============================================================
# Stage 3: Text-only generation sanity check
# ============================================================


def stage_text_only(vl_model):
    """Test text-only generation to verify 24-arg trace doesn't break text."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Text-only generation (sanity check)")
    logger.info("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    logger.info(f"  Prompt: '{prompt}'")
    logger.info(f"  Input IDs shape: {input_ids.shape}")

    # Generate without vision (llava_args should be empty)
    t0 = time.time()
    generated = vl_model.generate(
        input_ids=input_ids,
        max_new_tokens=16,
    )
    gen_time = time.time() - t0

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    logger.info(f"  Generated: '{output_text}'")
    logger.info(f"  Time: {gen_time:.2f}s")

    # Basic sanity: should contain "Paris"
    if "paris" in output_text.lower():
        logger.info("  PASS: Text-only generation working (contains 'Paris')")
    else:
        logger.warning(f"  WARN: Expected 'Paris' in output, got: '{output_text}'")

    return output_text


# ============================================================
# Stage 4: Vision + text generation
# ============================================================


def stage_vision_text(vl_model):
    """Test full VL generation with a real image."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Vision + text generation (full VL)")
    logger.info("=" * 60)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Create a simple test image (solid color)
    try:
        from PIL import Image
        import numpy as np

        # Create a red square image (224x224)
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel
        test_image = Image.fromarray(img_array)
        logger.info("  Created test image: 224x224 red square")
    except ImportError:
        logger.error("  PIL not available, cannot create test image")
        return None

    # Prepare inputs using HF processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What color is this image?"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[test_image],
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    logger.info(f"  input_ids shape: {input_ids.shape}")
    logger.info(
        f"  pixel_values shape: {pixel_values.shape if pixel_values is not None else None}"
    )
    logger.info(f"  image_grid_thw: {image_grid_thw}")

    # Count image tokens
    image_token_id = 248056
    n_image_tokens = (input_ids == image_token_id).sum().item()
    logger.info(f"  Image tokens in input: {n_image_tokens}")

    # Run full VL generation
    t0 = time.time()
    generated = vl_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=32,
    )
    gen_time = time.time() - t0

    output_text = processor.decode(generated[0], skip_special_tokens=True)
    logger.info(f"  Generated: '{output_text}'")
    logger.info(f"  Time: {gen_time:.2f}s")

    # Basic check
    if len(output_text) > 0:
        logger.info("  PASS: VL generation produced output")
    else:
        logger.warning("  WARN: VL generation produced empty output")

    if "red" in output_text.lower():
        logger.info("  PASS: Model correctly identified red color")

    return output_text


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="E2E VL test for Qwen3.5-35B-A3B")
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip compilation (use existing compiled model)",
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Only compile, don't run inference"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only test text generation (skip vision)",
    )
    args = parser.parse_args()

    logger.info("Qwen3.5-35B-A3B VL E2E Test")
    logger.info(f"  Model: {MODEL_PATH}")
    logger.info(f"  Compiled text: {COMPILED_TEXT_PATH}")
    logger.info(f"  Compiled vision: {COMPILED_VISION_PATH}")

    config, full_config = create_text_config()

    if not args.skip_compile:
        stage_compile(config)
        if args.compile_only:
            logger.info("Compilation complete. Exiting.")
            return

    vl_model = stage_load(config, full_config)

    # Warmup
    logger.info("Running warmup...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        vl_model.generate(input_ids=warmup_ids, max_new_tokens=4)
    logger.info("Warmup done")

    stage_text_only(vl_model)

    if not args.text_only:
        stage_vision_text(vl_model)

    logger.info("\n" + "=" * 60)
    logger.info("ALL STAGES COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
