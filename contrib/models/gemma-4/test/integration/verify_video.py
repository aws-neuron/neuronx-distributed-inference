#!/usr/bin/env python3
"""
End-to-end video verification for Gemma-4 on Neuron.

Pipeline:
1. Download sample video (ForBiggerBlazes.mp4)
2. Extract frames at uniform intervals
3. Process each frame through vision encoder (resize, patchify, ViT, embedder)
4. Build input_ids with timestamp-based video token format
5. Build vision_embeddings / vision_mask for Neuron text decoder
6. Compile/load text decoder with multimodal support
7. Run inference on Neuron
8. Print result

Token format per frame:
    "MM:SS" <boi> <video_token>*N <eoi>

Usage:
    python verify_video.py [--model_path PATH] [--compiled_path PATH] [--num_frames N]
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuronx_distributed_inference.models.config import NeuronConfig
from modeling_gemma4 import (
    NeuronGemma4ForConditionalGeneration,
    Gemma4InferenceConfig,
)
from gemma4_vision_encoder import (
    Gemma4VisionConfig,
    preprocess_image,
    load_vision_encoder,
)

VIDEO_URL = "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4"
VIDEO_LOCAL = "/tmp/ForBiggerBlazes.mp4"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Gemma-4 video on Neuron")
    parser.add_argument("--model_path", type=str,
                        default=os.path.expanduser("~/models/gemma-4-E2B-it"))
    parser.add_argument("--compiled_path", type=str,
                        default=os.path.expanduser("~/neuron_models/gemma-4-E2B-it-multimodal"))
    parser.add_argument("--tp_degree", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Number of frames to extract (default 4 for seq_len=512)")
    parser.add_argument("--max_soft_tokens", type=int, default=70,
                        help="Max soft tokens per frame (default 70 for video)")
    return parser.parse_args()


def download_video(url: str, local_path: str) -> str:
    if os.path.exists(local_path):
        print(f"  Using cached video: {local_path}")
        return local_path
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, local_path)
    print(f"  Saved to: {local_path}")
    return local_path


def extract_frames(video_path: str, num_frames: int) -> list:
    """
    Extract uniformly-spaced frames from video.

    Returns:
        List of (pil_image, timestamp_seconds) tuples
    """
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    duration = total_frames / fps

    print(f"  Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

    # Sample uniformly (like HF do_sample_frames)
    if num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        timestamp = idx / fps
        frames.append((pil_img, timestamp))

    cap.release()
    print(f"  Extracted {len(frames)} frames at timestamps: "
          f"{[f'{t:.1f}s' for _, t in frames]}")
    return frames


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def build_video_input_ids(
    frame_infos: list,
    prompt_text: str,
    model_path: str,
) -> tuple:
    """
    Build input_ids with video token placeholders.

    Format:
        <bos><start_of_turn>user\n
        MM:SS <boi><video>*N<eoi>   (per frame)
        \nPrompt\n<end_of_turn>\n<start_of_turn>model\n

    Args:
        frame_infos: list of (num_soft_tokens, timestamp_seconds)

    Returns:
        (input_ids tensor, video_token_id)
    """
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))

    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)

    bos_id = 2
    video_token_id = cfg["video_token_id"]   # 258884
    boi_id = cfg["boi_token_id"]             # 255999
    eoi_id = cfg["eoi_token_id"]             # 258882
    start_turn_id = 105
    end_turn_id = 106

    user_text = tokenizer.encode("user\n").ids

    # Build frame token sequences
    frame_ids = []
    for num_soft_tokens, timestamp in frame_infos:
        ts_str = format_timestamp(timestamp)
        ts_ids = tokenizer.encode(ts_str + " ").ids
        frame_ids.extend(ts_ids)
        frame_ids.append(boi_id)
        frame_ids.extend([video_token_id] * num_soft_tokens)
        frame_ids.append(eoi_id)
        # newline between frames
        frame_ids.extend(tokenizer.encode("\n").ids)

    prompt_ids = tokenizer.encode(prompt_text).ids
    model_text = tokenizer.encode("model\n").ids

    ids = ([bos_id, start_turn_id] + user_text
           + frame_ids
           + prompt_ids + [end_turn_id, 107, start_turn_id] + model_text)

    return torch.tensor([ids], dtype=torch.int32), video_token_id


def build_vision_embeddings(
    input_ids: torch.Tensor,
    all_frame_embeddings: torch.Tensor,
    video_token_id: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> tuple:
    """
    Build vision_embeddings and vision_mask for the Neuron text decoder.

    Args:
        all_frame_embeddings: [total_soft_tokens, hidden_size] - concatenated frame embeddings
    """
    B, S = input_ids.shape

    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    num_video_tokens = len(video_positions)

    flat_embeds = all_frame_embeddings[:num_video_tokens]

    vision_embeddings = torch.zeros(B, S, hidden_size, dtype=dtype)
    vision_embeddings[0, :num_video_tokens] = flat_embeds.to(dtype)

    # vision_mask: maps sequential embedding positions to actual token positions
    vision_mask = torch.full((B, S, 1), S - 1, dtype=torch.int32)
    vision_mask[0, :num_video_tokens, 0] = video_positions.to(torch.int32)

    return vision_embeddings, vision_mask


def generate_with_video(
    model,
    input_ids: torch.Tensor,
    vision_embeddings: torch.Tensor,
    vision_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: list,
    video_token_id: int,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Generate tokens autoregressively using NxDI stateful KV cache."""
    from neuronx_distributed_inference.models.model_wrapper import prepare_sampling_params

    generated_ids = input_ids.clone()
    B = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    sampling_params = prepare_sampling_params(B)

    for step in range(max_new_tokens):
        with torch.no_grad():
            if step == 0:
                model_input_ids = input_ids.clone()
                model_input_ids[model_input_ids == video_token_id] = pad_token_id

                position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(B, -1)
                outputs = model(
                    model_input_ids,
                    position_ids=position_ids,
                    sampling_params=sampling_params,
                    vision_embeddings=vision_embeddings,
                    vision_mask=vision_mask,
                )
            else:
                cur_pos = seq_len + step - 1
                position_ids = torch.tensor([[cur_pos]], dtype=torch.int32)
                dummy_ve = torch.zeros(B, 1, vision_embeddings.shape[2],
                                       dtype=vision_embeddings.dtype)
                dummy_vm = torch.zeros(B, 1, 1, dtype=torch.int32)
                outputs = model(
                    next_token,
                    position_ids=position_ids,
                    sampling_params=sampling_params,
                    vision_embeddings=dummy_ve,
                    vision_mask=dummy_vm,
                )

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to(torch.int32)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        tok_id = next_token.item()
        if step < 30:
            top5 = torch.topk(next_token_logits[0], 5)
            print(f"  Step {step}: tok={tok_id} top5_ids={top5.indices.tolist()} "
                  f"top5_vals={[f'{v:.1f}' for v in top5.values.tolist()]}")

        if tok_id in eos_token_ids:
            break

    return generated_ids


def main():
    args = parse_args()

    print("=" * 70)
    print("Gemma-4 E2B Video Verification on Neuron")
    print("=" * 70)

    # ---- Step 1: Load configs ----
    print("\n[1/7] Loading configurations...")
    with open(os.path.join(args.model_path, "config.json")) as f:
        raw_config = json.load(f)

    vision_cfg = Gemma4VisionConfig.from_dict(raw_config["vision_config"])
    text_hidden_size = raw_config["text_config"]["hidden_size"]
    eos_token_ids = raw_config.get("eos_token_id", [1, 106])
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    print(f"  Vision: {vision_cfg.num_hidden_layers} ViT layers, hidden={vision_cfg.hidden_size}")
    print(f"  Text:   hidden={text_hidden_size}")
    print(f"  Frames: {args.num_frames}, max_soft_tokens={args.max_soft_tokens}/frame")

    # ---- Step 2: Download video and extract frames ----
    print("\n[2/7] Processing video...")
    video_path = download_video(VIDEO_URL, VIDEO_LOCAL)
    frames = extract_frames(video_path, args.num_frames)

    # ---- Step 3: Run vision encoder on each frame ----
    print("\n[3/7] Running vision encoder (CPU) on each frame...")
    vision_encoder, vision_embedder = load_vision_encoder(
        args.model_path, vision_cfg, text_hidden_size,
    )

    all_embeddings = []
    frame_infos = []  # (num_soft_tokens, timestamp)
    total_soft_tokens = 0

    t0 = time.time()
    for i, (pil_img, timestamp) in enumerate(frames):
        pixel_values, position_ids, num_soft_tokens = preprocess_image(
            pil_img,
            patch_size=vision_cfg.patch_size,
            max_soft_tokens=args.max_soft_tokens,
            pooling_kernel_size=vision_cfg.pooling_kernel_size,
        )

        with torch.no_grad():
            pixel_bf16 = pixel_values.to(torch.bfloat16)
            vision_hidden = vision_encoder(pixel_bf16, position_ids)
            frame_embeddings = vision_embedder(vision_hidden.unsqueeze(0))  # [1, N, text_hidden]

        n_tokens = frame_embeddings.shape[1]
        all_embeddings.append(frame_embeddings.squeeze(0))  # [N, text_hidden]
        frame_infos.append((n_tokens, timestamp))
        total_soft_tokens += n_tokens

        ts_str = format_timestamp(timestamp)
        print(f"  Frame {i}: {ts_str} → {pil_img.size[0]}x{pil_img.size[1]} → "
              f"{pixel_values.shape[1]} patches → {n_tokens} soft tokens")

    t1 = time.time()
    # Concatenate all frame embeddings
    all_frame_embeddings = torch.cat(all_embeddings, dim=0)  # [total_soft_tokens, hidden]
    print(f"  Total: {total_soft_tokens} soft tokens, encoder time: {t1-t0:.2f}s")
    print(f"  Combined embeddings: {all_frame_embeddings.shape}")

    # ---- Step 4: Build input sequence ----
    print("\n[4/7] Building input sequence...")
    prompt = "Describe this video."
    input_ids, video_token_id = build_video_input_ids(frame_infos, prompt, args.model_path)
    print(f"  Input sequence length: {input_ids.shape[1]}")
    print(f"  Video token count: {(input_ids == video_token_id).sum().item()}")

    if input_ids.shape[1] > args.seq_len:
        print(f"  ERROR: Input ({input_ids.shape[1]}) > seq_len ({args.seq_len})!")
        print(f"  Try reducing --num_frames or increasing --seq_len")
        sys.exit(1)

    padded_input_ids = torch.nn.functional.pad(
        input_ids, (0, args.seq_len - input_ids.shape[1]), value=0,
    )
    vision_embeddings, vision_mask = build_vision_embeddings(
        padded_input_ids, all_frame_embeddings,
        video_token_id, text_hidden_size, torch.bfloat16,
    )
    print(f"  vision_embeddings: {vision_embeddings.shape}")
    print(f"  vision_mask: {vision_mask.shape}")

    # ---- Step 5: Compile text decoder with multimodal support ----
    print(f"\n[5/7] Compiling text decoder (multimodal)...")
    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        seq_len=args.seq_len,
        max_context_length=args.seq_len,
        torch_dtype=torch.bfloat16,
        save_sharded_checkpoint=True,
        attn_kernel_enabled=False,
    )

    config = Gemma4InferenceConfig.from_pretrained(args.model_path, neuron_config=neuron_config)
    compiled_path = args.compiled_path
    model = NeuronGemma4ForConditionalGeneration(args.model_path, config)

    if not os.path.exists(os.path.join(compiled_path, "text_model", "model.pt")):
        print(f"  Compiling to {compiled_path}...")
        t0 = time.time()
        model.compile(compiled_path)
        print(f"  Compilation done in {time.time()-t0:.1f}s")
    else:
        print(f"  Using cached compilation: {compiled_path}")

    # ---- Step 6: Load and run inference ----
    print(f"\n[6/7] Loading and running inference on Neuron...")
    model.load(compiled_path)

    max_gen = min(args.max_new_tokens, args.seq_len - input_ids.shape[1])
    print(f"  Max generation tokens: {max_gen} (seq_len={args.seq_len}, input={input_ids.shape[1]})")

    t0 = time.time()
    generated_ids = generate_with_video(
        model, padded_input_ids, vision_embeddings, vision_mask,
        max_new_tokens=max_gen,
        eos_token_ids=eos_token_ids,
        video_token_id=video_token_id,
    )
    t1 = time.time()

    # ---- Step 7: Decode and display result ----
    print(f"\n[7/7] Results:")

    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(args.model_path, "tokenizer.json"))
        new_ids = generated_ids[0, args.seq_len:].tolist()
        response = tokenizer.decode(new_ids)
    except Exception as e:
        response = f"[decode error: {e}]"
        new_ids = generated_ids[0, args.seq_len:].tolist()

    num_new_tokens = len(new_ids)
    total_time = t1 - t0

    print(f"  Generated {num_new_tokens} tokens in {total_time:.2f}s")
    if num_new_tokens > 0:
        print(f"  Throughput: {num_new_tokens/total_time:.1f} tok/s")
    print(f"\n  Prompt: {prompt}")
    print(f"  Response:")
    print(f"  {response}")
    print("\n" + "=" * 70)
    print("Video verification complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
