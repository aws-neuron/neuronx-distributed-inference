#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Vision-Language smoke test for Qwen3.5-2B on Neuron.

Flow:
  * Compile the text decoder with `use_text_only_cte_inputs=False` so it accepts
    vision_embeddings, vision_mask, and mRoPE position_ids as CTE inputs.
  * Load a CPU-side vision encoder (Qwen3-VL vision weights from HF safetensors).
  * Preprocess a test image with `AutoProcessor`.
  * Feed pixel_values → CPU vision → vision_embeddings.
  * Run the Neuron-compiled text decoder with the vision embeddings scattered in.
  * Print the generated caption.

This uses CPU for vision (the Qwen3.5 ViT weights don't require huge compute
and the encoder is small). Tracing the vision encoder to Neuron is a
separate follow-up.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python contrib/models/Qwen3.5-2B/test/integration/run_vl_smoke.py \\
        --model-path /mnt/nvme/models/Qwen3.5-2B \\
        --compiled-path /tmp/qwen35_2b_vl_traced \\
        --image /path/to/image.jpg \\
        --prompt "Describe this image."
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


def build_vl_text_config(model_path: str, tp: int, seq_len: int):
    """Build a Qwen35InferenceConfig with vision-aware CTE inputs enabled."""
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig

    neuron_config = NeuronConfig(
        tp_degree=tp,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )

    with open(os.path.join(model_path, "config.json")) as f:
        full = json.load(f)
    text_cfg = full.get("text_config", full)

    cfg = dict(text_cfg)
    cfg["pad_token_id"] = text_cfg.get("eos_token_id", 248044)
    if "rope_parameters" in text_cfg:
        rp = text_cfg["rope_parameters"]
        cfg["rope_theta"] = rp.get("rope_theta", 10000000)
        cfg["partial_rotary_factor"] = rp.get("partial_rotary_factor", 0.25)
        cfg["mrope_section"] = rp.get("mrope_section", [11, 11, 10])
    cfg.setdefault("tie_word_embeddings", text_cfg.get("tie_word_embeddings", True))

    return Qwen35InferenceConfig(
        neuron_config=neuron_config,
        use_hybrid_cache_manager=False,
        use_text_only_cte_inputs=False,  # ← accept vision + mrope inputs
        **cfg,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/nvme/models/Qwen3.5-2B")
    ap.add_argument("--compiled-path", default="/tmp/qwen35_2b_vl_traced")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--image", default=None,
                    help="Path or URL to an image; if None, uses a random dummy")
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    from src.modeling_qwen35 import NeuronQwen35ForCausalLM
    from src.modeling_qwen35_vl import (
        NeuronQwen35VLForCausalLM, Qwen35VLInferenceConfig,
    )
    from transformers import AutoProcessor, AutoTokenizer

    text_config = build_vl_text_config(args.model_path, args.tp, args.seq_len)

    # Extract vision_config dict from HF
    with open(os.path.join(args.model_path, "config.json")) as f:
        full = json.load(f)
    vision_config_dict = full["vision_config"]
    # Qwen3-VL VisionModel expects out_hidden_size / spatial_merge_size on the config
    vision_config_dict.setdefault("spatial_merge_size", 2)
    vision_config_dict.setdefault("temporal_patch_size", 2)

    vl_config = Qwen35VLInferenceConfig(
        text_config=text_config,
        vision_config=vision_config_dict,
        image_token_id=full.get("image_token_id", 248056),
        video_token_id=full.get("video_token_id", 248057),
        vision_start_token_id=full.get("vision_start_token_id", 248053),
        vision_end_token_id=full.get("vision_end_token_id", 248054),
        spatial_merge_size=vision_config_dict["spatial_merge_size"],
    )

    # ---- Compile / load text model ----
    text_path = os.path.join(args.compiled_path, "text_model")
    neff = os.path.join(text_path, "model.pt")
    if not args.skip_compile and not os.path.exists(neff):
        print(f"[compile-text] → {text_path}")
        os.makedirs(text_path, exist_ok=True)
        m = NeuronQwen35ForCausalLM(args.model_path, text_config)
        t0 = time.perf_counter()
        m.compile(text_path)
        print(f"[compile-text] done in {(time.perf_counter()-t0):.1f} s")
        del m
        gc.collect()

    # ---- Build VL orchestrator ----
    vl_model = NeuronQwen35VLForCausalLM(
        model_path=args.model_path,
        text_config=text_config,
        vision_config=vl_config,
    )
    vl_model.text_model.load(text_path)

    # Load CPU vision weights (patch_embed, pos_embed, transformer blocks)
    print("[load-vision] loading CPU vision encoder weights")
    vl_model.vision_model_wrapper.load_cpu_model(args.model_path)
    vl_model.vision_model_wrapper.load_vision_weights_from_hf(args.model_path)

    # ---- Prepare inputs ----
    processor = AutoProcessor.from_pretrained(args.model_path)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    # If no image, generate a dummy 224x224 RGB
    if args.image is None:
        print("[input] no --image provided; using dummy random image (224x224)")
        from PIL import Image
        import numpy as np
        dummy = (np.random.rand(224, 224, 3) * 255).astype("uint8")
        image = Image.fromarray(dummy)
    else:
        if args.image.startswith(("http://", "https://")):
            import io
            import urllib.request
            with urllib.request.urlopen(args.image) as f:
                data = f.read()
            from PIL import Image
            image = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            from PIL import Image
            image = Image.open(args.image).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.prompt},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    print(f"[input] input_ids shape={input_ids.shape}")
    print(f"[input] pixel_values shape={pixel_values.shape}")
    print(f"[input] image_grid_thw={image_grid_thw.tolist()}")

    # ---- Generate ----
    print("[generate] running VL generation")
    t0 = time.perf_counter()
    generated = vl_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    dt = time.perf_counter() - t0

    new_ids = generated[0].tolist()[input_ids.shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True)

    print("=" * 72)
    print(f"prompt      : {args.prompt!r}")
    print(f"image       : {args.image}")
    print(f"n_new       : {len(new_ids)}")
    print(f"elapsed     : {dt:.1f} s")
    print(f"generated   : {text!r}")
    print("=" * 72)


if __name__ == "__main__":
    main()
