#!/usr/bin/env python3
"""End-to-end test for Qwen2.5-Omni on Trn2 (CPU inference).

Tests multimodal input (text, image, audio) → text and audio output
using HF Qwen2_5OmniForConditionalGeneration directly.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python3 test_e2e_qwen25_omni.py [--model-path MODEL_PATH] [--output-dir OUTPUT_DIR]
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_OUTPUT = "/home/ubuntu/e2e_test_results"

# Qwen2.5-Omni requires this specific system prompt for audio output
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def download_test_assets(output_dir):
    """Create sample image and test audio."""
    from PIL import Image, ImageDraw
    import wave

    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    image_path = os.path.join(assets_dir, "test_image.jpg")
    if not os.path.exists(image_path):
        logger.info("Creating test image...")
        img = Image.new("RGB", (320, 240), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 100, 100], fill="red", outline="black")
        draw.ellipse([130, 30, 230, 130], fill="blue", outline="black")
        draw.polygon([(260, 120), (310, 20), (210, 20)], fill="green", outline="black")
        draw.rectangle([0, 160, 320, 240], fill="lightgreen")
        draw.ellipse([220, 60, 300, 140], fill="yellow")
        img.save(image_path)
        logger.info("Saved %s", image_path)

    audio_path = os.path.join(assets_dir, "test_audio.wav")
    if not os.path.exists(audio_path):
        logger.info("Creating test audio...")
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        with wave.open(audio_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        logger.info("Saved %s", audio_path)

    return image_path, audio_path


def load_model(model_path):
    """Load Qwen2.5-Omni model and processor."""
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    logger.info("Loading processor from %s ...", model_path)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    logger.info("Loading model from %s (this may take a few minutes)...", model_path)
    start = time.time()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    elapsed = time.time() - start
    logger.info("Model loaded in %.1fs", elapsed)

    return model, processor


# ============================================================================
# Test 1: Text → Text
# ============================================================================

def test_text_only(model, processor, output_dir):
    logger.info("=" * 60)
    logger.info("TEST 1: Text → Text")
    logger.info("=" * 60)

    prompt = "What is the capital of France? Answer in one sentence."

    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=100,
            )
        elapsed = time.time() - start

        response = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        path = os.path.join(output_dir, "test1_text_response.txt")
        with open(path, "w") as f:
            f.write(f"Prompt: {prompt}\n\nResponse: {response}\n\nTime: {elapsed:.1f}s\n")

        logger.info("Response: %s", response)
        logger.info("Time: %.1fs", elapsed)
        return True, response, elapsed

    except Exception as e:
        logger.error("Test 1 FAILED: %s", e)
        traceback.print_exc()
        return False, str(e), 0


# ============================================================================
# Test 2: Image + Text → Text
# ============================================================================

def test_image_text(model, processor, output_dir, image_path):
    logger.info("=" * 60)
    logger.info("TEST 2: Image + Text → Text")
    logger.info("=" * 60)

    prompt = "Describe this image in detail. What shapes and colors do you see?"

    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=200,
            )
        elapsed = time.time() - start

        response = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        path = os.path.join(output_dir, "test2_image_response.txt")
        with open(path, "w") as f:
            f.write(f"Prompt: {prompt}\nImage: {image_path}\n\nResponse: {response}\n\nTime: {elapsed:.1f}s\n")

        logger.info("Response: %s", response[:300])
        logger.info("Time: %.1fs", elapsed)
        return True, response, elapsed

    except Exception as e:
        logger.error("Test 2 FAILED: %s", e)
        traceback.print_exc()
        return False, str(e), 0


# ============================================================================
# Test 3: Audio + Text → Text
# ============================================================================

def test_audio_text(model, processor, output_dir, audio_path):
    logger.info("=" * 60)
    logger.info("TEST 3: Audio + Text → Text")
    logger.info("=" * 60)

    prompt = "What do you hear in this audio? Describe it."

    try:
        import wave

        with wave.open(audio_path, "r") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_data, "sampling_rate": sr},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text,
            audios=[audio_data],
            sampling_rate=sr,
            return_tensors="pt",
        )

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=200,
            )
        elapsed = time.time() - start

        response = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        path = os.path.join(output_dir, "test3_audio_response.txt")
        with open(path, "w") as f:
            f.write(f"Prompt: {prompt}\nAudio: {audio_path}\n\nResponse: {response}\n\nTime: {elapsed:.1f}s\n")

        logger.info("Response: %s", response[:300])
        logger.info("Time: %.1fs", elapsed)
        return True, response, elapsed

    except Exception as e:
        logger.error("Test 3 FAILED: %s", e)
        traceback.print_exc()
        return False, str(e), 0


# ============================================================================
# Test 4: Text → Speech
# ============================================================================

def test_speech_output(model, processor, output_dir):
    logger.info("=" * 60)
    logger.info("TEST 4: Text → Speech")
    logger.info("=" * 60)

    prompt = "Say hello and tell me what the weather is like today."

    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            result = model.generate(
                **inputs,
                return_audio=True,
                thinker_max_new_tokens=200,
                talker_max_new_tokens=2000,
                speaker="Chelsie",
            )
        elapsed = time.time() - start

        # result is (text_ids, audio_waveform) when return_audio=True
        text_response = ""
        audio_waveform = None

        if isinstance(result, tuple) and len(result) >= 2:
            text_ids, audio_waveform = result[0], result[1]
            text_response = processor.batch_decode(
                text_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0]
        else:
            text_ids = result if not isinstance(result, tuple) else result[0]
            text_response = processor.batch_decode(
                text_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0]

        # Save text
        text_path = os.path.join(output_dir, "test4_speech_text.txt")
        with open(text_path, "w") as f:
            f.write(f"Prompt: {prompt}\n\nResponse: {text_response}\n\nTime: {elapsed:.1f}s\n")
            if audio_waveform is not None:
                f.write(f"Audio: generated ({type(audio_waveform)})\n")
            else:
                f.write("Audio: none returned\n")

        # Save audio
        wav_path = os.path.join(output_dir, "test4_speech_response.wav")
        if audio_waveform is not None:
            import wave as wave_mod

            if isinstance(audio_waveform, torch.Tensor):
                audio_np = audio_waveform.cpu().float().numpy()
            else:
                audio_np = np.array(audio_waveform, dtype=np.float32)

            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()

            max_val = max(abs(audio_np.max()), abs(audio_np.min()), 1e-8)
            audio_int16 = (audio_np / max_val * 32767).astype(np.int16)

            with wave_mod.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_int16.tobytes())

            logger.info("Speech saved: %s (%d samples, %.1fs audio)",
                        wav_path, len(audio_int16), len(audio_int16) / 24000)
        else:
            logger.warning("No audio waveform returned.")
            with open(os.path.join(output_dir, "test4_no_audio.txt"), "w") as f:
                f.write("model.generate(return_audio=True) did not return audio waveform.\n"
                        "This may require spk_dict.pt or the Talker/Token2Wav to be initialized.\n")

        logger.info("Text: %s", text_response[:300])
        logger.info("Time: %.1fs", elapsed)
        return True, text_response, elapsed

    except Exception as e:
        logger.error("Test 4 FAILED: %s", e)
        traceback.print_exc()
        return False, str(e), 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Qwen2.5-Omni End-to-End Test")
    logger.info("Model: %s", args.model_path)
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 60)

    image_path, audio_path = download_test_assets(args.output_dir)
    model, processor = load_model(args.model_path)

    results = {}

    for name, fn, extra_args in [
        ("test1_text_to_text", test_text_only, []),
        ("test2_image_text", test_image_text, [image_path]),
        ("test3_audio_text", test_audio_text, [audio_path]),
        ("test4_text_to_speech", test_speech_output, []),
    ]:
        ok, resp, t = fn(model, processor, args.output_dir, *extra_args)
        results[name] = {"passed": ok, "response": resp[:500], "time": t}
        gc.collect()

    passed = sum(1 for r in results.values() if r["passed"])
    total = len(results)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Qwen2.5-Omni End-to-End Test Results\n")
        f.write(f"Model: {args.model_path}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Results: {passed}/{total} passed\n\n")
        for name, r in results.items():
            status = "PASS" if r["passed"] else "FAIL"
            f.write(f"[{status}] {name} ({r['time']:.1f}s)\n")
            f.write(f"  {r['response'][:300]}\n\n")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("FINAL: %d/%d tests passed", passed, total)
    logger.info("Results: %s", args.output_dir)
    logger.info("=" * 60)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
