#!/usr/bin/env python3
"""
End-to-end multimodal inference for Qwen2.5-Omni-7B on NeuronX (TP=4).

Supports:
  1. Text-only: Thinker generates text
  2. Image + text: Vision encoder + Thinker
  3. Audio + text: Audio encoder + Thinker
  4. Image + audio + text: All encoders + Thinker
  5. Speech output: Thinker → Talker → Token2Wav (optional)

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  pip install qwen-omni-utils[decord]

  # First run compiles (~30 min), subsequent runs load from cache:
  python3 examples/generate_qwen25_omni.py

  # Text-only (fastest, uses simpler text-only model):
  python3 examples/generate_qwen25_omni.py --mode text

  # Image understanding:
  python3 examples/generate_qwen25_omni.py --mode image

  # Audio understanding:
  python3 examples/generate_qwen25_omni.py --mode audio

  # Full multimodal (image + audio + text → text + speech):
  python3 examples/generate_qwen25_omni.py --mode full
"""

# --- Qwen2.5-Omni contrib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
import _upstream_compat  # noqa: F401  (applies hf_adapter shim)
# --- end bootstrap ---

import argparse
import gc
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "QWEN25_OMNI_MODEL_PATH", "/opt/dlami/nvme/models/Qwen2.5-Omni-7B"
)
COMPILED_PATH = os.environ.get(
    "QWEN25_OMNI_COMPILED_PATH", "/tmp/qwen25_omni_compiled"
)
TP_DEGREE = int(os.environ.get("QWEN25_OMNI_TP_DEGREE", "4"))

# Test media from Qwen official examples
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/cough.wav"

# Sequence lengths
TEXT_SEQ_LENGTH = 4096
TEXT_BUCKETS = [256, 512, 1024, 2048, 4096]
VISION_SEQ_LENGTH = 1012  # single image
VISION_BUCKETS = [1]  # 1 image


class Timer:
    def __init__(self, label):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"  [{self.label}] {self.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Mode 1: Text-only (uses the simpler NeuronQwen25OmniForCausalLM)
# ---------------------------------------------------------------------------
def run_text_only(model_path=MODEL_PATH, compiled_path=COMPILED_PATH):
    """Text-only inference using the Thinker model."""
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        load_pretrained_config,
        HuggingFaceGenerationAdapter,
    )
    from modeling_qwen25_omni import (
        NeuronQwen25OmniForCausalLM,
        Qwen25OmniInferenceConfig,
    )
    from transformers import AutoTokenizer

    print("\n" + "=" * 60)
    print("Mode: Text-only (Thinker on Neuron, TP=4)")
    print("=" * 60)

    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        seq_len=2048,
        max_context_length=2048,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False, top_k=1),
    )

    hf_config = load_pretrained_config(model_path)
    config = Qwen25OmniInferenceConfig(neuron_config, load_config=hf_config)

    compiled_dir = os.path.join(compiled_path, "thinker_tp4")

    with Timer("Create model"):
        model = NeuronQwen25OmniForCausalLM(model_path, config)

    if not os.path.exists(os.path.join(compiled_dir, "neuron_config.json")):
        with Timer("Compile"):
            model.compile(compiled_dir)
    else:
        print("  Compiled artifacts found")

    with Timer("Load"):
        model.load(compiled_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter = HuggingFaceGenerationAdapter(model)

    prompts = [
        "What is 2+3? Answer with just the number.",
        "Write a haiku about the ocean.",
        "Explain quantum computing in one sentence.",
    ]

    print("\n--- Generation Results ---")
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(text, return_tensors="pt")

        with Timer(f"Generate"):
            output_ids = adapter.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=128,
                eos_token_id=[tokenizer.eos_token_id, 151645],
            )

        new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"  Q: {prompt}")
        print(f"  A: {output_text.strip()[:300]}")
        print()

    del model, adapter
    gc.collect()


# ---------------------------------------------------------------------------
# Mode 2: Multimodal (image/audio + text → text, optional speech)
# ---------------------------------------------------------------------------
def run_multimodal(mode="full", model_path=MODEL_PATH, compiled_path=COMPILED_PATH):
    """Multimodal inference: image + audio + text → text (+ optional speech).

    Args:
        mode: "image" (image+text), "audio" (audio+text), "full" (image+audio+text)
        model_path: Path to HF model weights
        compiled_path: Path for compiled artifacts
    """
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        load_pretrained_config,
        HuggingFaceGenerationAdapter,
    )
    from modeling_qwen25_omni import (
        NeuronQwen25OmniMultimodalForCausalLM,
        Qwen25OmniMultimodalInferenceConfig,
    )
    from transformers import AutoProcessor, GenerationConfig

    print("\n" + "=" * 60)
    print(f"Mode: Multimodal ({mode}) on Neuron, TP={TP_DEGREE}")
    print("=" * 60)

    # --- Step 1: Create configs ---
    text_neuron_config = NeuronConfig(
        batch_size=1,
        seq_len=TEXT_SEQ_LENGTH,
        max_context_length=TEXT_SEQ_LENGTH,
        ctx_batch_size=1,
        tp_degree=TP_DEGREE,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
        sequence_parallel_enabled=False,
        flash_decoding_enabled=False,
        qkv_kernel_enabled=False,
        qkv_nki_kernel_enabled=False,
        attn_kernel_enabled=False,
        enable_bucketing=True,
        context_encoding_buckets=TEXT_BUCKETS,
        token_generation_buckets=TEXT_BUCKETS,
        on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False, top_k=1),
    )

    vision_neuron_config = NeuronConfig(
        batch_size=1,
        seq_len=VISION_SEQ_LENGTH,
        tp_degree=TP_DEGREE,
        torch_dtype=torch.bfloat16,
        enable_bucketing=True,
        buckets=VISION_BUCKETS,
        fused_qkv=False,  # Qwen2.5-Omni vision uses separate Q/K/V
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
    )

    config = Qwen25OmniMultimodalInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # --- Step 2: Create and compile/load the model ---
    compiled_dir = os.path.join(compiled_path, "multimodal_tp4")

    with Timer("Create multimodal model"):
        model = NeuronQwen25OmniMultimodalForCausalLM(
            model_path=model_path, config=config
        )

    if not os.path.exists(os.path.join(compiled_dir, "neuron_config.json")):
        with Timer("Compile text + vision (this takes 20-40 minutes)"):
            model.compile(compiled_dir)
            processor.save_pretrained(compiled_dir)
    else:
        print("  Compiled artifacts found")

    with Timer("Load compiled model"):
        model.load(compiled_dir)
        processor = AutoProcessor.from_pretrained(compiled_dir)

    # --- Step 3: Enable audio encoder (if needed) ---
    if mode in ("audio", "full"):
        print("\n  Enabling audio encoder...")
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniForConditionalGeneration,
        )
        from modeling_qwen25_omni_audio import (
            NeuronQwen25OmniAudioEncoder,
        )

        with Timer("Load HF model for audio weights"):
            hf_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            full_sd = hf_model.state_dict()

        audio_sd = NeuronQwen25OmniAudioEncoder.convert_hf_to_neuron_state_dict(
            full_sd, dtype=torch.bfloat16
        )
        del hf_model, full_sd
        gc.collect()

        model.enable_audio_encoder(audio_sd)

        compiled_audio_dir = os.path.join(compiled_path, "audio_encoder_tp4")
        if not os.path.exists(
            os.path.join(compiled_audio_dir, "neuron_config.json")
        ):
            with Timer("Compile audio encoder transformer"):
                model.compile_audio_encoder(compiled_audio_dir)
        else:
            print("  Audio encoder compiled artifacts found")

        with Timer("Load audio encoder"):
            model.load_audio_encoder(compiled_audio_dir)

        del audio_sd
        gc.collect()

    # --- Step 4: Run inference ---
    adapter = HuggingFaceGenerationAdapter(model)
    generation_config = GenerationConfig(
        do_sample=False,
        eos_token_id=[151645],
        pad_token_id=151645,
    )

    # Prepare inputs based on mode
    from qwen_omni_utils import process_mm_info

    if mode == "image":
        print("\n--- Image Understanding ---")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": IMAGE_URL},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            },
        ]
    elif mode == "audio":
        print("\n--- Audio Understanding ---")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": AUDIO_URL},
                    {"type": "text", "text": "What sound is this? Describe what you hear."},
                ],
            },
        ]
    else:  # full
        print("\n--- Full Multimodal (Image + Audio + Text) ---")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": IMAGE_URL},
                    {"type": "audio", "audio": AUDIO_URL},
                    {
                        "type": "text",
                        "text": "Describe the image and the audio you received.",
                    },
                ],
            },
        ]

    # Process multimodal inputs
    with Timer("Process multimodal inputs"):
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=images if images else None,
            audios=audios if audios else None,
            return_tensors="pt",
            padding=True,
        )

    print(f"  input_ids shape: {inputs.input_ids.shape}")
    if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
        print(f"  pixel_values shape: {inputs.pixel_values.shape}")
    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
        print(f"  image_grid_thw: {inputs.image_grid_thw}")
    if hasattr(inputs, "input_features") and inputs.input_features is not None:
        print(f"  input_features shape: {inputs.input_features.shape}")

    # Generate
    generate_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "generation_config": generation_config,
        "max_new_tokens": 256,
    }
    if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
        generate_kwargs["pixel_values"] = inputs.pixel_values
    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
        generate_kwargs["image_grid_thw"] = inputs.image_grid_thw
    if hasattr(inputs, "input_features") and inputs.input_features is not None:
        generate_kwargs["input_features"] = inputs.input_features
    if (
        hasattr(inputs, "feature_attention_mask")
        and inputs.feature_attention_mask is not None
    ):
        generate_kwargs["feature_attention_mask"] = inputs.feature_attention_mask

    with Timer("Generate"):
        output_ids = adapter.generate(**generate_kwargs)

    output_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"\n  Output: {output_text[0].strip()[:500]}")

    del model, adapter
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni-7B inference on Neuron")
    parser.add_argument(
        "--mode",
        choices=["text", "image", "audio", "full"],
        default="text",
        help="Inference mode (default: text)",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help=f"Model path (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--compiled-path",
        default=COMPILED_PATH,
        help=f"Compiled model path (default: {COMPILED_PATH})",
    )
    args = parser.parse_args()

    model_path = args.model_path
    compiled_path = args.compiled_path

    print(f"Model: {model_path}")
    print(f"Compiled: {compiled_path}")
    print(f"TP: {TP_DEGREE}")

    if args.mode == "text":
        run_text_only(model_path, compiled_path)
    else:
        run_multimodal(args.mode, model_path, compiled_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
