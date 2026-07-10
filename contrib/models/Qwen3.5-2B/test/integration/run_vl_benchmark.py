#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Vision-Language benchmark: TTFT / TPOT / output vs image size.

Compiles ONCE with NxDI CTE bucketing enabled so the same model handles
512×512, 1024×1024, 2048×2048 images (input_ids ~280 / ~1048 / ~4120,
CTE buckets [512, 1024, 2048, 4096, 8192]).

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python contrib/models/Qwen3.5-2B/test/integration/run_vl_benchmark.py \\
        --model-path    /mnt/nvme/models/Qwen3.5-2B \\
        --compiled-path /tmp/qwen35_2b_vl_bench \\
        --images 512 1024 2048 \\
        --max-new-tokens 48 --repeats 3
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

os.environ.setdefault("QWEN36_DELTANET_CTE_IMPL", "legacy_direct")
os.environ.setdefault("QWEN36_DELTANET_MULTIHEAD_CTE", "0")

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


DEFAULT_PROMPT = "What is in this image? Describe it briefly."


def build_vl_config(model_path: str, tp: int, max_seq_len: int, buckets):
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig
    from src.modeling_qwen35_vl import Qwen35VLInferenceConfig

    neuron_config = NeuronConfig(
        tp_degree=tp,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=max_seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=True,
        context_encoding_buckets=buckets,
        token_generation_buckets=[max_seq_len],
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

    text_config = Qwen35InferenceConfig(
        neuron_config=neuron_config,
        use_hybrid_cache_manager=False,
        use_text_only_cte_inputs=False,
        **cfg,
    )

    vision_config_dict = full["vision_config"]
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
    return text_config, vl_config


def compile_and_load(model_path, compiled_path, text_config, vl_config, skip_compile):
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM
    from src.modeling_qwen35_vl import NeuronQwen35VLForCausalLM

    text_path = os.path.join(compiled_path, "text_model")
    neff = os.path.join(text_path, "model.pt")
    if not skip_compile and not os.path.exists(neff):
        print(f"[compile-text] → {text_path}")
        os.makedirs(text_path, exist_ok=True)
        m = NeuronQwen35ForCausalLM(model_path, text_config)
        t0 = time.perf_counter()
        m.compile(text_path)
        print(f"[compile-text] done in {(time.perf_counter()-t0):.1f} s")
        del m
        gc.collect()

    vl_model = NeuronQwen35VLForCausalLM(
        model_path=model_path,
        text_config=text_config,
        vision_config=vl_config,
    )
    vl_model.text_model.load(text_path)
    print("[load-vision] loading CPU vision encoder weights")
    vl_model.vision_model_wrapper.load_cpu_model(model_path)
    vl_model.vision_model_wrapper.load_vision_weights_from_hf(model_path)
    return vl_model


def one_image_run(vl_model, processor, tok, image, prompt, max_new_tokens):
    """Return (ttft_ms, tpot_ms, n_new, text)."""
    msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

    # TTFT: 1-token generate
    t0 = time.perf_counter()
    _ = vl_model.generate(
        input_ids=inp["input_ids"],
        attention_mask=inp.get("attention_mask", torch.ones_like(inp["input_ids"])),
        pixel_values=inp["pixel_values"],
        image_grid_thw=inp["image_grid_thw"],
        max_new_tokens=1,
        temperature=0.0,
    )
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    # Full generate
    t1 = time.perf_counter()
    out = vl_model.generate(
        input_ids=inp["input_ids"],
        attention_mask=inp.get("attention_mask", torch.ones_like(inp["input_ids"])),
        pixel_values=inp["pixel_values"],
        image_grid_thw=inp["image_grid_thw"],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    total_ms = (time.perf_counter() - t1) * 1000.0

    new_ids = out[0].tolist()[inp["input_ids"].shape[1]:]
    n_new = len(new_ids)
    tpot_ms = (total_ms - ttft_ms) / max(1, n_new - 1)
    text = tok.decode(new_ids, skip_special_tokens=True)
    return ttft_ms, tpot_ms, n_new, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/nvme/models/Qwen3.5-2B")
    ap.add_argument("--compiled-path", default="/tmp/qwen35_2b_vl_bench")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--images", nargs="+", type=int, default=[512, 1024, 2048],
                    help="Image side sizes in pixels; expects /tmp/test_image_<s>.jpg")
    ap.add_argument("--buckets", nargs="+", type=int,
                    default=[512, 1024, 2048, 4096, 8192],
                    help="CTE bucket sizes to compile")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    max_seq = max(args.buckets)
    print(f"[bench] buckets={args.buckets}, max_seq={max_seq}, tp={args.tp}")

    text_config, vl_config = build_vl_config(args.model_path, args.tp, max_seq, args.buckets)
    vl_model = compile_and_load(args.model_path, args.compiled_path, text_config, vl_config, args.skip_compile)

    from transformers import AutoProcessor, AutoTokenizer
    from PIL import Image
    tok = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)

    results = []
    for size in args.images:
        img_path = f"/tmp/test_image_{size}.jpg"
        image = Image.open(img_path).convert("RGB")
        print(f"\n=== image {size}x{size} ({img_path}) ===")

        ttfts, tpots, texts = [], [], []
        for r in range(args.repeats + 1):  # +1 warmup
            ttft, tpot, n_new, text = one_image_run(vl_model, processor, tok, image, args.prompt, args.max_new_tokens)
            tag = "warmup" if r == 0 else f"r{r}"
            print(f"  [{tag}] TTFT={ttft:7.1f} ms  TPOT={tpot:6.1f} ms  n_new={n_new}")
            if r == 0:
                continue  # discard warmup
            ttfts.append(ttft)
            tpots.append(tpot)
            texts.append(text)

        result = {
            "size": size,
            "prompt": args.prompt,
            "ttft_ms_median": statistics.median(ttfts),
            "ttft_ms_mean": statistics.mean(ttfts),
            "tpot_ms_median": statistics.median(tpots),
            "tpot_ms_mean": statistics.mean(tpots),
            "throughput_tok_per_s": 1000.0 / statistics.mean(tpots),
            "text_sample": texts[0],
        }
        results.append(result)
        print(f"  output: {texts[0]!r}")

    print("\n=== SUMMARY ===")
    print(f"{'size':>6}  {'TTFT (ms)':>10}  {'TPOT (ms)':>10}  {'tok/s':>7}")
    for r in results:
        print(f"{r['size']:>6}  {r['ttft_ms_median']:>10.1f}  {r['tpot_ms_median']:>10.1f}  {r['throughput_tok_per_s']:>7.1f}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "buckets": args.buckets,
                "max_new_tokens": args.max_new_tokens,
                "repeats": args.repeats,
                "prompt": args.prompt,
                "results": results,
            }, f, indent=2)
        print(f"\n[bench] wrote {args.out_json}")


if __name__ == "__main__":
    main()
