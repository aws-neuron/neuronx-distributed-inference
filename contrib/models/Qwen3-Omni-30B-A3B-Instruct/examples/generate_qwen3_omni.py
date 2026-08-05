#!/usr/bin/env python3
"""
Generate text from Qwen3-Omni-30B-A3B-Instruct on Neuron.

Supports three modes:
  --mode text     : Text-only generation (vision + MoE text on Neuron)
  --mode image    : Image + text generation (vision + MoE text on Neuron)
  --mode audio    : Audio + text generation (audio + vision + MoE text on Neuron)

All neural network components run on Neuron.

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd /home/ubuntu/whn-ndi

  # Set environment
  export NEURON_RT_VISIBLE_CORES=0-7   # or 0-15 for larger TP
  export QWEN3_OMNI_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct

  # Text mode
  python contrib/models/Qwen3-Omni-30B-A3B-Instruct/examples/generate_qwen3_omni.py \\
      --mode text --prompt "What is quantum computing?"

  # Image mode
  python contrib/models/Qwen3-Omni-30B-A3B-Instruct/examples/generate_qwen3_omni.py \\
      --mode image --image /path/to/image.jpg --prompt "Describe this image."

  # Audio mode
  python contrib/models/Qwen3-Omni-30B-A3B-Instruct/examples/generate_qwen3_omni.py \\
      --mode audio --audio /path/to/audio.wav --prompt "Transcribe the speech."
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src to path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import _upstream_compat  # noqa: F401

import gc
import torch
from _model_path import resolve_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate with Qwen3-Omni on Neuron")
    parser.add_argument("--mode", choices=["text", "image", "audio"], default="text")
    parser.add_argument("--prompt", type=str, default="What is quantum computing?")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--compiled-path", type=str, default="/tmp/qwen3_omni_compiled")
    parser.add_argument("--tp-degree", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=4096)
    return parser.parse_args()


def build_model(model_path, compiled_path, tp_degree, seq_len):
    from neuronx_distributed_inference.models.config import (
        MoENeuronConfig,
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        load_pretrained_config,
        HuggingFaceGenerationAdapter,
    )
    from transformers import AutoProcessor

    from modeling_qwen3_omni import (
        Qwen3OmniMoEInferenceConfig,
        NeuronQwen3OmniForCausalLM,
    )

    text_buckets = [256, 512, 1024, 2048, seq_len]
    vision_seq_len = 1012
    vision_buckets = [vision_seq_len]

    text_neuron_config = MoENeuronConfig(
        batch_size=1,
        seq_len=seq_len,
        max_context_length=seq_len,
        ctx_batch_size=1,
        tp_degree=tp_degree,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
        sequence_parallel_enabled=False,
        flash_decoding_enabled=False,
        qkv_kernel_enabled=False,
        qkv_nki_kernel_enabled=False,
        attn_kernel_enabled=False,
        enable_bucketing=True,
        context_encoding_buckets=text_buckets,
        token_generation_buckets=text_buckets,
        on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False, top_k=1),
        blockwise_matmul_config={"use_torch_block_wise": True},
    )
    vision_neuron_config = NeuronConfig(
        batch_size=1,
        seq_len=vision_seq_len,
        tp_degree=tp_degree,
        torch_dtype=torch.bfloat16,
        enable_bucketing=True,
        buckets=vision_buckets,
        fused_qkv=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
    )

    config = Qwen3OmniMoEInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    compiled_dir = os.path.join(compiled_path, f"multimodal_tp{tp_degree}")

    print("Creating model...")
    t0 = time.time()
    model = NeuronQwen3OmniForCausalLM(model_path=model_path, config=config)
    print(f"  Created in {time.time()-t0:.1f}s")

    if not os.path.exists(os.path.join(compiled_dir, "neuron_config.json")):
        print("Compiling text + vision...")
        t0 = time.time()
        model.compile(compiled_dir)
        processor.save_pretrained(compiled_dir)
        print(f"  Compiled in {time.time()-t0:.1f}s")
    else:
        print("  Compiled artifacts found")

    print("Loading compiled model...")
    t0 = time.time()
    model.load(compiled_dir)
    processor = AutoProcessor.from_pretrained(compiled_dir)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    adapter = HuggingFaceGenerationAdapter(model)
    return adapter, processor, config


def build_audio_encoder(model, model_path, compiled_path, tp_degree):
    from modeling_qwen3_omni_audio import NeuronQwen3OmniAudioEncoder

    print("\nBuilding audio encoder...")
    from transformers import AutoModelForCausalLM

    print("  Loading HF model for audio weights...")
    t0 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    full_sd = hf_model.state_dict()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    audio_sd = NeuronQwen3OmniAudioEncoder.convert_hf_to_neuron_state_dict(
        full_sd, dtype=torch.bfloat16
    )
    del hf_model, full_sd
    gc.collect()

    model.neuron_model.enable_audio_encoder(audio_sd)

    compiled_audio_dir = os.path.join(compiled_path, f"audio_encoder_tp{tp_degree}")
    if not os.path.exists(os.path.join(compiled_audio_dir, "neuron_config.json")):
        print("  Compiling audio encoder transformer...")
        t0 = time.time()
        model.neuron_model.compile_audio_encoder(compiled_audio_dir)
        print(f"  Compiled in {time.time()-t0:.1f}s")
    else:
        print("  Audio encoder compiled artifacts found")

    print("  Loading audio encoder...")
    t0 = time.time()
    model.neuron_model.load_audio_encoder(compiled_audio_dir)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    del audio_sd
    gc.collect()


def generate_text(adapter, processor, prompt, max_new_tokens):
    from transformers import GenerationConfig

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    gen_config = GenerationConfig(
        do_sample=False,
        eos_token_id=[151645],
        pad_token_id=151645,
    )

    t0 = time.time()
    output_ids = adapter.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        generation_config=gen_config,
        max_new_tokens=max_new_tokens,
    )
    gen_time = time.time() - t0

    prompt_len = inputs.input_ids.shape[1]
    new_tokens = output_ids[:, prompt_len:]
    response = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return response, gen_time, new_tokens.shape[1]


def generate_with_image(adapter, processor, prompt, image_path, max_new_tokens):
    from transformers import GenerationConfig

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_path], return_tensors="pt", padding=True)

    gen_config = GenerationConfig(
        do_sample=False,
        eos_token_id=[151645],
        pad_token_id=151645,
    )

    generate_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "generation_config": gen_config,
        "max_new_tokens": max_new_tokens,
    }
    if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
        generate_kwargs["pixel_values"] = inputs.pixel_values.to(torch.bfloat16)
    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
        generate_kwargs["image_grid_thw"] = inputs.image_grid_thw

    t0 = time.time()
    output_ids = adapter.generate(**generate_kwargs)
    gen_time = time.time() - t0

    prompt_len = inputs.input_ids.shape[1]
    new_tokens = output_ids[:, prompt_len:]
    response = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return response, gen_time, new_tokens.shape[1]


def generate_with_audio(adapter, processor, prompt, audio_path, max_new_tokens):
    from transformers import GenerationConfig

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], audio=[audio_path], return_tensors="pt", padding=True)

    gen_config = GenerationConfig(
        do_sample=False,
        eos_token_id=[151645],
        pad_token_id=151645,
    )

    generate_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "generation_config": gen_config,
        "max_new_tokens": max_new_tokens,
    }
    if hasattr(inputs, "input_features") and inputs.input_features is not None:
        generate_kwargs["input_features"] = inputs.input_features.to(torch.bfloat16)
    if hasattr(inputs, "feature_attention_mask") and inputs.feature_attention_mask is not None:
        generate_kwargs["feature_attention_mask"] = inputs.feature_attention_mask

    t0 = time.time()
    output_ids = adapter.generate(**generate_kwargs)
    gen_time = time.time() - t0

    prompt_len = inputs.input_ids.shape[1]
    new_tokens = output_ids[:, prompt_len:]
    response = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return response, gen_time, new_tokens.shape[1]


def main():
    args = parse_args()
    model_path = args.model_path or resolve_model_path()
    compiled_path = args.compiled_path

    print("=" * 60)
    print(f"Qwen3-Omni-30B-A3B-Instruct on Neuron (mode={args.mode})")
    print(f"  Model: {model_path}")
    print(f"  TP: {args.tp_degree}")
    print(f"  Seq len: {args.seq_len}")
    print("=" * 60)

    adapter, processor, config = build_model(
        model_path, compiled_path, args.tp_degree, args.seq_len
    )

    if args.mode == "audio":
        build_audio_encoder(
            adapter, model_path, compiled_path, args.tp_degree
        )

    print("\n" + "=" * 60)
    print("Generating...")
    print("=" * 60)

    if args.mode == "text":
        response, gen_time, n_tokens = generate_text(
            adapter, processor, args.prompt, args.max_new_tokens
        )
    elif args.mode == "image":
        if args.image is None:
            print("ERROR: --image required for image mode")
            sys.exit(1)
        response, gen_time, n_tokens = generate_with_image(
            adapter, processor, args.prompt, args.image, args.max_new_tokens
        )
    elif args.mode == "audio":
        if args.audio is None:
            print("ERROR: --audio required for audio mode")
            sys.exit(1)
        response, gen_time, n_tokens = generate_with_audio(
            adapter, processor, args.prompt, args.audio, args.max_new_tokens
        )

    print(f"\nPrompt: {args.prompt}")
    print(f"Response: {response}")
    print(f"\n  Tokens: {n_tokens}")
    print(f"  Time: {gen_time:.2f}s")
    print(f"  Speed: {n_tokens/gen_time:.1f} tok/s")


if __name__ == "__main__":
    main()
