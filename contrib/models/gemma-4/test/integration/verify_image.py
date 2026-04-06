#!/usr/bin/env python3
"""
End-to-end image verification for Gemma-4 on Neuron.

Pipeline:
1. Download sample image (GoldenGate.png from HF reference)
2. Preprocess image (resize, patchify, position IDs)
3. Run vision encoder on CPU (16-layer ViT with 2D RoPE)
4. Run vision embedder on CPU (RMSNorm + Linear → text_hidden_size)
5. Build vision_embeddings / vision_mask for Neuron text decoder
6. Compile/load text decoder with multimodal support
7. Run inference on Neuron
8. Print result

Usage:
    python verify_image.py [--model_path PATH] [--compiled_path PATH]
"""

import argparse
import json
import os
import sys
import time
import urllib.request

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

IMAGE_URL = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/GoldenGate.png"
IMAGE_LOCAL = "/tmp/GoldenGate.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Gemma-4 image on Neuron")
    parser.add_argument("--model_path", type=str,
                        default=os.path.expanduser("~/models/gemma-4-E2B-it"))
    parser.add_argument("--compiled_path", type=str,
                        default=os.path.expanduser("~/neuron_models/gemma-4-E2B-it-multimodal"))
    parser.add_argument("--tp_degree", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    return parser.parse_args()


def download_image(url: str, local_path: str) -> str:
    if os.path.exists(local_path):
        print(f"  Using cached image: {local_path}")
        return local_path
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, local_path)
    print(f"  Saved to: {local_path}")
    return local_path


def build_image_input_ids(
    num_image_tokens: int,
    prompt_text: str,
    model_path: str,
) -> tuple:
    """
    Build input_ids with image token placeholders.
    Format: <bos><start_of_turn>user\n<boi><image>*N<eoi>\nPrompt\n<end_of_turn>\n<start_of_turn>model\n
    """
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))

    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)

    bos_id = 2
    image_token_id = cfg["image_token_id"]   # 258880
    boi_id = cfg["boi_token_id"]             # 255999
    eoi_id = cfg["eoi_token_id"]             # 258882
    start_turn_id = 105
    end_turn_id = 106

    user_text = tokenizer.encode("user\n").ids
    prompt_ids = tokenizer.encode("\n" + prompt_text).ids
    model_text = tokenizer.encode("model\n").ids

    ids = ([bos_id, start_turn_id] + user_text
           + [boi_id] + [image_token_id] * num_image_tokens + [eoi_id]
           + prompt_ids + [end_turn_id, 107, start_turn_id] + model_text)

    return torch.tensor([ids], dtype=torch.int32), image_token_id


def build_vision_embeddings(
    input_ids: torch.Tensor,
    image_embeddings: torch.Tensor,
    image_token_id: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> tuple:
    """Build vision_embeddings and vision_mask for the Neuron text decoder."""
    B, S = input_ids.shape

    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    num_image_tokens = len(image_positions)

    flat_embeds = image_embeddings.reshape(-1, hidden_size)[:num_image_tokens]

    vision_embeddings = torch.zeros(B, S, hidden_size, dtype=dtype)
    vision_embeddings[0, :num_image_tokens] = flat_embeds.to(dtype)

    vision_mask = torch.full((B, S, 1), S - 1, dtype=torch.int32)
    vision_mask[0, :num_image_tokens, 0] = image_positions.to(torch.int32)

    return vision_embeddings, vision_mask


def generate_with_image(
    model,
    input_ids: torch.Tensor,
    vision_embeddings: torch.Tensor,
    vision_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: list,
    image_token_id: int,
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
                model_input_ids[model_input_ids == image_token_id] = pad_token_id

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
    print("Gemma-4 E2B Image Verification on Neuron")
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

    # ---- Step 2: Download and preprocess image ----
    print("\n[2/7] Processing image...")
    image_path = download_image(IMAGE_URL, IMAGE_LOCAL)
    pixel_values, position_ids, num_soft_tokens = preprocess_image(image_path)

    print(f"  Patches: {pixel_values.shape}")
    print(f"  Position IDs: {position_ids.shape}")
    print(f"  Soft tokens after pooling: {num_soft_tokens}")

    # ---- Step 3: Run vision encoder on CPU ----
    print("\n[3/7] Running vision encoder (CPU)...")
    vision_encoder, vision_embedder = load_vision_encoder(
        args.model_path, vision_cfg, text_hidden_size,
    )

    t0 = time.time()
    with torch.no_grad():
        pixel_bf16 = pixel_values.to(torch.bfloat16)
        vision_hidden = vision_encoder(pixel_bf16, position_ids)  # [N, 768]
        image_embeddings = vision_embedder(vision_hidden.unsqueeze(0))  # [1, N, 1536]
    t1 = time.time()

    print(f"  Vision hidden: {vision_hidden.shape}")
    print(f"  Image embeddings: {image_embeddings.shape}")
    print(f"  Encoder time: {t1-t0:.2f}s")

    # ---- Step 4: Build input sequence ----
    print("\n[4/7] Building input sequence...")
    prompt = "What is shown in this image?"
    actual_image_tokens = image_embeddings.shape[1]
    input_ids, image_token_id = build_image_input_ids(actual_image_tokens, prompt, args.model_path)
    print(f"  Input sequence length: {input_ids.shape[1]}")
    print(f"  Image token positions: {actual_image_tokens}")

    if input_ids.shape[1] > args.seq_len:
        print(f"  WARNING: Input ({input_ids.shape[1]}) > seq_len ({args.seq_len}), truncating")
        input_ids = input_ids[:, :args.seq_len]

    padded_input_ids = torch.nn.functional.pad(
        input_ids, (0, args.seq_len - input_ids.shape[1]), value=0,
    )
    vision_embeddings, vision_mask = build_vision_embeddings(
        padded_input_ids, image_embeddings,
        image_token_id, text_hidden_size, torch.bfloat16,
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
    generated_ids = generate_with_image(
        model, input_ids, vision_embeddings, vision_mask,
        max_new_tokens=max_gen,
        eos_token_ids=eos_token_ids,
        image_token_id=image_token_id,
    )
    t1 = time.time()

    # ---- Step 7: Decode and display result ----
    print(f"\n[7/7] Results:")

    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(args.model_path, "tokenizer.json"))
        new_ids = generated_ids[0, input_ids.shape[1]:].tolist()
        response = tokenizer.decode(new_ids)
    except Exception as e:
        response = f"[decode error: {e}]"
        new_ids = generated_ids[0, input_ids.shape[1]:].tolist()

    num_new_tokens = len(new_ids)
    total_time = t1 - t0

    print(f"  Generated {num_new_tokens} tokens in {total_time:.2f}s")
    if num_new_tokens > 0:
        print(f"  Throughput: {num_new_tokens/total_time:.1f} tok/s")
    print(f"\n  Response:")
    print(f"  {response}")
    print("\n" + "=" * 70)
    print("Image verification complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
