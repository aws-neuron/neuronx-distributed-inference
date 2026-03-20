"""
HunyuanVideo-1.5 Neuron Pipeline

End-to-end text-to-video generation — ALL components on Neuron hardware.

Compiled artifacts required (see compile_all.py):
  - compiled_transformer/  (NxDI, TP=2)
  - compiled_qwen2vl/      (NxDI, TP=2)
  - byt5_traced.pt          (torch_neuronx.trace)
  - refiner_traced.pt       (torch_neuronx.trace)
  - reorder_traced.pt       (torch_neuronx.trace)
  - cond_type_embed_weight.pt (extracted weights)
  - vae_shards/             (24 torch_neuronx.trace shards)

Usage:
    python run_inference.py --prompt "A cat on a beach" --output_dir output/
"""
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
BYT5_MODEL_ID = "google/byt5-small"

T_LAT, H_LAT, W_LAT = 9, 30, 40
N_ENCODER = 1985
CROP_START = 103
MLLM_SEQ_LEN = 1108

SYSTEM_MESSAGE = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    "1. The main content and theme of the video. "
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    "4. background environment, light, style and atmosphere. "
    "5. camera angles, movements, and transitions used in the video."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _masked_mean(embeds, mask):
    mf = mask.float().unsqueeze(-1)
    return (embeds.float() * mf).sum(dim=1) / mf.sum(dim=1)


def _sinusoidal_timestep_proj(timestep):
    half = 128
    exp = -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
    emb = timestep.float().unsqueeze(-1) * exp.unsqueeze(0).exp()
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


def _build_refiner_mask(mllm_mask):
    m = mllm_mask.float().squeeze(0)
    outer = m.unsqueeze(0) * m.unsqueeze(1)
    return ((1.0 - outer) * -1e9).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def generate(
    prompt: str,
    negative_prompt: str = "",
    num_steps: int = 50,
    guidance_scale: float = 9.0,
    seed: int = 42,
    output_dir: str = "output",
    compiled_dir: str = None,
    vae_on_cpu: bool = False,
):
    from huggingface_hub import snapshot_download
    import torch_neuronx  # noqa: F401
    import time

    hf_path = snapshot_download(HF_MODEL_ID)
    base = Path(compiled_dir) if compiled_dir else Path(__file__).parent
    d = torch.bfloat16

    from modeling_qwen2vl_encoder import Qwen2VLEncoderConfig, NeuronQwen2VLEncoder
    from modeling_hunyuan_video15_transformer import (
        HunyuanVideo15TransformerConfig, NeuronHunyuanVideo15Transformer,
    )
    from modeling_hunyuan_video15_text import load_traced_model
    from modeling_hunyuan_video15_vae import NeuronVAEDecoder

    # ── 1. Text Encoding ──
    print(f"[{time.strftime('%H:%M:%S')}] [1/6] MLLM encoding (Neuron)...")
    qwen_cfg = Qwen2VLEncoderConfig.from_pretrained(QWEN_MODEL_ID)
    qwen = NeuronQwen2VLEncoder(QWEN_MODEL_ID, config=qwen_cfg)
    qwen.load(str(base / "compiled_qwen2vl"))

    from transformers import AutoTokenizer, ByT5Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)

    def encode_mllm(text):
        fmt = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        inp = tokenizer.apply_chat_template(fmt, add_generation_prompt=True, tokenize=True,
            return_dict=True, padding="max_length", max_length=MLLM_SEQ_LEN,
            truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = qwen.forward(inp.input_ids, inp.attention_mask.long())
        return h[:, CROP_START:CROP_START + 1000], inp.attention_mask[:, CROP_START:CROP_START + 1000].float()

    pos_mllm_e, pos_mllm_m = encode_mllm(prompt)
    neg_mllm_e, neg_mllm_m = encode_mllm(negative_prompt)
    del qwen; gc.collect()

    # ByT5
    print(f"[{time.strftime('%H:%M:%S')}] [2/6] ByT5 encoding (Neuron)...")
    byt5 = load_traced_model(str(base / "byt5_traced.pt"))
    byt5_tok = ByT5Tokenizer.from_pretrained(BYT5_MODEL_ID)

    def encode_byt5(text):
        inp = byt5_tok(text, padding="max_length", max_length=256,
                       truncation=True, add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            return byt5(inp.input_ids, inp.attention_mask.float()), inp.attention_mask.float()

    pos_byt5_e, pos_byt5_m = encode_byt5(prompt)
    neg_byt5_e, neg_byt5_m = encode_byt5(negative_prompt)
    del byt5; gc.collect()

    img_e = torch.zeros(1, 729, 1152, dtype=d)
    print(f"  MLLM: {pos_mllm_e.shape}, ByT5: {pos_byt5_e.shape}")

    # ── 2. Token Refiner ──
    print(f"[{time.strftime('%H:%M:%S')}] [3/6] Token refiner (Neuron)...")
    refiner = load_traced_model(str(base / "refiner_traced.pt"))
    cte_w = torch.load(str(base / "cond_type_embed_weight.pt"), weights_only=True)

    def refine(mllm_e, mllm_m):
        pooled = _masked_mean(mllm_e, mllm_m)
        ts_proj = _sinusoidal_timestep_proj(torch.tensor([1000.0]))
        mask = _build_refiner_mask(mllm_m)
        with torch.no_grad():
            r = refiner(mllm_e.bfloat16(), ts_proj.bfloat16(), pooled.bfloat16(), mask.bfloat16())
        return r + cte_w[0].unsqueeze(0).unsqueeze(0)

    pos_refined = refine(pos_mllm_e, pos_mllm_m)
    neg_refined = refine(neg_mllm_e, neg_mllm_m)
    del refiner; gc.collect()

    # ── 3. Token Reorder ──
    print(f"[{time.strftime('%H:%M:%S')}] [4/6] Token reorder (Neuron)...")
    reorder = load_traced_model(str(base / "reorder_traced.pt"))

    def do_reorder(mllm_m, byt5_m):
        idx, zm = reorder(mllm_m.squeeze().float(), byt5_m.squeeze().float())
        return idx.long(), zm.bfloat16()

    pos_idx, pos_zero = do_reorder(pos_mllm_m, pos_byt5_m)
    neg_idx, neg_zero = do_reorder(neg_mllm_m, neg_byt5_m)
    del reorder; gc.collect()

    # ── 4. Denoising Loop ──
    print(f"[{time.strftime('%H:%M:%S')}] [5/6] Denoising ({num_steps} steps, CFG={guidance_scale})...")
    tx_cfg = HunyuanVideo15TransformerConfig.from_pretrained(HF_MODEL_ID)
    backbone = NeuronHunyuanVideo15Transformer(HF_MODEL_ID, config=tx_cfg)
    backbone.load(str(base / "compiled_transformer"))

    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import retrieve_timesteps
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(hf_path, subfolder="scheduler")
    timesteps, _ = retrieve_timesteps(sched, num_steps, "cpu", sigmas=np.linspace(1.0, 0.0, num_steps + 1)[:-1])

    gen = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 32, T_LAT, H_LAT, W_LAT, generator=gen, dtype=d)
    cond = torch.zeros(1, 32, T_LAT, H_LAT, W_LAT, dtype=d)
    mask = torch.zeros(1, 1, T_LAT, H_LAT, W_LAT, dtype=d)
    total = T_LAT * H_LAT * W_LAT + N_ENCODER
    attn_mask = torch.zeros(1, 1, total, total, dtype=d)

    t_start = time.time()
    for i, t_val in enumerate(timesteps):
        video_in = torch.cat([latents, cond, mask], dim=1)
        ts = t_val.to(d).unsqueeze(0)
        with torch.no_grad():
            pred_c = backbone.forward(video_in, pos_refined, pos_byt5_e.bfloat16(), img_e,
                                      ts, pos_idx, pos_zero, attn_mask)
            pred_u = backbone.forward(video_in, neg_refined, neg_byt5_e.bfloat16(), img_e,
                                      ts, neg_idx, neg_zero, attn_mask)
        noise_pred = pred_u + guidance_scale * (pred_c - pred_u)
        latents = sched.step(noise_pred.float(), t_val, latents.float(), return_dict=False)[0].to(d)
        if i % 10 == 0 or i == len(timesteps) - 1:
            elapsed = time.time() - t_start
            print(f"  Step {i+1}/{num_steps}: t={t_val.item():.1f}, norm={latents.float().norm():.1f} ({elapsed:.0f}s)")

    del backbone; gc.collect()

    # ── 5. VAE Decode ──
    with open(Path(hf_path) / "vae" / "config.json") as f:
        scaling_factor = json.load(f).get("scaling_factor", 0.476986)

    if vae_on_cpu:
        print(f"[{time.strftime('%H:%M:%S')}] [6/6] VAE decode (CPU)...")
        from diffusers import AutoencoderKLHunyuanVideo15
        vae = AutoencoderKLHunyuanVideo15.from_pretrained(hf_path, subfolder="vae", torch_dtype=torch.float32).eval()
        with torch.no_grad():
            video = vae.decoder(latents.float() / scaling_factor)
        del vae; gc.collect()
    else:
        print(f"[{time.strftime('%H:%M:%S')}] [6/6] VAE decode (Neuron, 24 shards)...")
        vae = NeuronVAEDecoder(str(base / "vae_shards"), scaling_factor=scaling_factor)
        with torch.no_grad():
            video = vae.decode(latents)
        del vae; gc.collect()
    video = video.float().clamp(-1, 1) * 0.5 + 0.5

    # Save frames
    from PIL import Image
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    for fi in range(video.shape[2]):
        frame = (video[0, :, fi] * 255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(frame).save(out / f"frame_{fi:04d}.png")
    for fi in [0, 8, 16, 24, 32]:
        if fi < video.shape[2]:
            fr = (video[0, :, fi] * 255).byte().permute(1, 2, 0).numpy()
            print(f"  Frame {fi:2d}: R={fr[:,:,0].mean():.0f} G={fr[:,:,1].mean():.0f} "
                  f"B={fr[:,:,2].mean():.0f} mean={fr.mean():.0f} std={fr.std():.0f}")
    print(f"  Saved {video.shape[2]} frames to {out}/")
    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default='cat walking on the beach at sunset, with the word "CAT" over it.')
    p.add_argument("--output_dir", default="pipeline_output")
    p.add_argument("--compiled_dir", default=None)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vae_on_cpu", action="store_true", help="Run VAE decoder on CPU instead of Neuron (no VAE NEFF compilation needed)")
    args = p.parse_args()
    generate(
        prompt=args.prompt, num_steps=args.steps, guidance_scale=args.cfg,
        seed=args.seed, output_dir=args.output_dir, compiled_dir=args.compiled_dir,
        vae_on_cpu=args.vae_on_cpu,
    )
