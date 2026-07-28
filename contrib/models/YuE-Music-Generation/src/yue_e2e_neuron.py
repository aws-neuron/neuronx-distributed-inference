#!/usr/bin/env python3
"""YuE E2E Music Generation on AWS Neuron (NxDI) -- Orchestrator.

Runs S1 and S2 in separate subprocesses (required because NxDI models with different
TP degrees cannot coexist in the same process -- the Neuron runtime segfaults on warmup).

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    cd /mnt/models
    python yue_e2e_neuron.py --genre_txt genre.txt --lyrics_txt lyrics.txt [--skip-compile]

Environment variables:
    MODEL_DIR: Root directory for models and compiled artifacts (default: /mnt/models)
"""

import os
import sys
import time
import re
import random
import uuid
import argparse
import subprocess
import json

import numpy as np
import torch
import soundfile as sf

MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/models")

# Add YuE inference code to path
sys.path.insert(0, os.path.join(MODEL_DIR, "YuE/inference"))
sys.path.insert(0, os.path.join(MODEL_DIR, "xcodec_mini_infer"))
sys.path.insert(0, os.path.join(MODEL_DIR, "xcodec_mini_infer/descriptaudiocodec"))

from omegaconf import OmegaConf
from models.soundstream_hubert_new import SoundStream

XCODEC_CONFIG = os.path.join(MODEL_DIR, "xcodec_mini_infer/final_ckpt/config.yaml")
XCODEC_CKPT = os.path.join(MODEL_DIR, "xcodec_mini_infer/final_ckpt/ckpt_00360000.pth")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_lyrics(lyrics):
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    return [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]


def save_audio(wav, path, sample_rate, rescale=False):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    limit = 0.99
    mx = wav.abs().max()
    wav = wav * min(limit / mx, 1) if rescale else wav.clamp(-limit, limit)
    wav_np = wav.numpy()
    if wav_np.ndim == 2:
        wav_np = wav_np.T  # (channels, samples) -> (samples, channels)
    sf.write(str(path), wav_np, sample_rate, subtype="PCM_16")


def load_xcodec():
    model_config = OmegaConf.load(XCODEC_CONFIG)
    codec_model = eval(model_config.generator.name)(**model_config.generator.config)
    param_dict = torch.load(XCODEC_CKPT, map_location="cpu", weights_only=False)
    codec_model.load_state_dict(param_dict["codec_model"])
    codec_model.eval()
    return codec_model


def decode_to_audio(codec_model, codec_result):
    with torch.no_grad():
        wav = codec_model.decode(
            torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
            .unsqueeze(0)
            .permute(1, 0, 2)
        )
    return wav.cpu().squeeze(0)


def run_subprocess(script_path, args_dict, label):
    """Run a stage script in a subprocess, passing args as JSON via env var."""
    print(f"\n{'=' * 60}")
    print(f"Running {label} in subprocess: {script_path}")
    print(f"{'=' * 60}")

    env = os.environ.copy()
    env["YUE_STAGE_ARGS"] = json.dumps(args_dict)
    env["MODEL_DIR"] = MODEL_DIR

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=MODEL_DIR,
        env=env,
    )
    wall_time = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with return code {result.returncode}")

    print(f"{label} completed in {wall_time:.1f}s")
    return wall_time


def main():
    parser = argparse.ArgumentParser(description="YuE E2E Music Generation on Neuron")
    parser.add_argument("--genre_txt", type=str, required=True)
    parser.add_argument("--lyrics_txt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output_neuron")
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--run_n_segments", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument(
        "--no-cfg", action="store_true", help="Disable CFG (default: CFG enabled)"
    )
    parser.add_argument(
        "--guidance-scale-first",
        type=float,
        default=1.5,
        help="CFG guidance scale for first segment",
    )
    parser.add_argument(
        "--guidance-scale-rest",
        type=float,
        default=1.2,
        help="CFG guidance scale for subsequent segments",
    )
    parser.add_argument(
        "--nki-kernels",
        action="store_true",
        help="Enable NKI MLP TKG fused kernels for both S1 and S2 (20%% speedup)",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV-cache-aware S2 teacher-forcing (use legacy generate() loop)",
    )
    parser.add_argument(
        "--s2-batch-size",
        type=int,
        default=1,
        help="S2 batch size: number of chunks processed simultaneously (requires recompile)",
    )
    parser.add_argument(
        "--s1-tp-degree",
        type=int,
        default=2,
        help="S1 tensor parallelism degree (default: 2, use 1 for LNC=1 single-core)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    # Create output dirs
    stage1_dir = os.path.join(args.output_dir, "stage1")
    stage2_dir = os.path.join(args.output_dir, "stage2")
    audio_dir = os.path.join(args.output_dir, "audio")
    for d in [stage1_dir, stage2_dir, audio_dir]:
        os.makedirs(d, exist_ok=True)

    # Read inputs
    with open(args.genre_txt) as f:
        genres = f.read().strip()
    with open(args.lyrics_txt) as f:
        lyrics = split_lyrics(f.read())

    print(f"Genre: {genres}")
    print(f"Lyrics segments: {len(lyrics)}")
    for i, seg in enumerate(lyrics):
        print(f"  [{i}] {seg[:60].strip()}...")

    pipeline_start = time.time()
    random_id = str(uuid.uuid4())

    # --------------------------------------------------------
    # Stage 1: S1 (7B) in subprocess
    # --------------------------------------------------------
    s1_args = {
        "genres": genres,
        "lyrics": lyrics,
        "stage1_dir": stage1_dir,
        "random_id": random_id,
        "max_new_tokens": args.max_new_tokens,
        "run_n_segments": args.run_n_segments,
        "skip_compile": args.skip_compile,
        "seed": args.seed,
        "use_cfg": not args.no_cfg,
        "guidance_scale_first": args.guidance_scale_first,
        "guidance_scale_rest": args.guidance_scale_rest,
        "s1_tp_degree": args.s1_tp_degree,
        "use_nki_kernels": args.nki_kernels,
    }
    s1_wall = run_subprocess(
        os.path.join(SCRIPT_DIR, "yue_stage1_worker.py"), s1_args, "Stage 1 (S1 7B)"
    )

    # --------------------------------------------------------
    # Stage 2: S2 (1B) in subprocess
    # --------------------------------------------------------
    s2_args = {
        "stage1_dir": stage1_dir,
        "stage2_dir": stage2_dir,
        "random_id": random_id,
        "skip_compile": args.skip_compile,
        "nki_mlp": args.nki_kernels,
        "use_kv_cache": not args.no_kv_cache,
        "s2_batch_size": args.s2_batch_size,
    }
    s2_wall = run_subprocess(
        os.path.join(SCRIPT_DIR, "yue_stage2_worker.py"), s2_args, "Stage 2 (S2 1B)"
    )

    # --------------------------------------------------------
    # Stage 3: Decode to audio (CPU, main process)
    # --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STAGE 3: Decoding to audio (xcodec_mini, CPU)")
    print(f"{'=' * 60}")

    codec_model = load_xcodec()
    s3_start = time.time()

    vocal_refined = np.load(os.path.join(stage2_dir, f"vocals_{random_id}.npy"))
    inst_refined = np.load(os.path.join(stage2_dir, f"instrumentals_{random_id}.npy"))

    print("Decoding vocals...")
    vocal_wav = decode_to_audio(codec_model, vocal_refined)
    vocal_path = os.path.join(audio_dir, f"vocals_{random_id}.wav")
    save_audio(vocal_wav, vocal_path, 16000, args.rescale)
    print(f"  Saved: {vocal_path}")

    print("Decoding instrumentals...")
    inst_wav = decode_to_audio(codec_model, inst_refined)
    inst_path = os.path.join(audio_dir, f"instrumentals_{random_id}.wav")
    save_audio(inst_wav, inst_path, 16000, args.rescale)
    print(f"  Saved: {inst_path}")

    print("Mixing tracks...")
    vocal_audio, sr = sf.read(vocal_path)
    inst_audio, _ = sf.read(inst_path)
    min_len = min(len(vocal_audio), len(inst_audio))
    mix = vocal_audio[:min_len] + inst_audio[:min_len]
    mix_path = os.path.join(audio_dir, f"mix_{random_id}.wav")
    sf.write(mix_path, mix, sr)
    print(f"  Saved mix: {mix_path}")

    s3_time = time.time() - s3_start

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    total_time = time.time() - pipeline_start
    vocals_s1 = np.load(os.path.join(stage1_dir, f"vocals_{random_id}.npy"))
    audio_duration = vocals_s1.shape[1] / 50

    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Genre: {genres}")
    print(f"  Segments: {min(args.run_n_segments, len(lyrics))}")
    print(f"  Audio duration: {audio_duration:.1f}s")
    print(f"  Stage 1 (S1 7B): {s1_wall:.1f}s (incl. load)")
    print(f"  Stage 2 (S2 1B): {s2_wall:.1f}s (incl. load)")
    print(f"  Stage 3 (decode): {s3_time:.1f}s")
    print(f"  Total pipeline: {total_time:.1f}s")
    print(f"  Real-time factor: {total_time / max(audio_duration, 0.1):.2f}x")
    print(f"")
    print(f"  Output files:")
    print(f"    Vocals:        {vocal_path}")
    print(f"    Instrumentals: {inst_path}")
    print(f"    Mix:           {mix_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
