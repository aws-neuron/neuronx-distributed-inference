#!/usr/bin/env python3
"""
ASR benchmark: evaluate Qwen3-Omni-30B-A3B-Instruct on LibriSpeech test-clean.

Runs the MoE text model on Neuron (TP=8, LNC=2) and audio encoder transformer
layers on a single Neuron core (no TP). Conv2d frontend stays on CPU.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    pip install jiwer "datasets<4" soundfile librosa

    NEURON_RT_VISIBLE_CORES=0-31 python test_asr.py \
        --model-path /home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct \
        --compiled-model-path /home/ubuntu/traced_model/Qwen3-Omni-asr \
        --audio-compiled-path /home/ubuntu/traced_model/Qwen3-Omni-audio \
        --num-samples 100
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))


def compile_and_load_model(model_path, compiled_model_path, tp_degree, seq_len,
                           max_context_length, vision_tp_degree, vision_seq_len,
                           audio_compiled_path=None):
    """Compile (if needed) and load the multimodal model on Neuron."""
    from modeling_qwen3_omni import (
        NeuronQwen3OmniForCausalLM,
        Qwen3OmniInferenceConfig,
        load_qwen3_omni_multimodal_config,
    )
    from neuronx_distributed_inference.models.config import MoENeuronConfig, NeuronConfig

    text_neuron_config = MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        seq_len=seq_len,
        max_context_length=max_context_length,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config={"top_k": 1, "do_sample": False},
        blockwise_matmul_config={"use_torch_block_wise": True},
    )

    vision_neuron_config = NeuronConfig(
        tp_degree=vision_tp_degree,
        batch_size=1,
        seq_len=vision_seq_len,
        torch_dtype=torch.bfloat16,
    )

    config = Qwen3OmniInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_qwen3_omni_multimodal_config(model_path),
    )

    model = NeuronQwen3OmniForCausalLM(model_path, config, skip_vision_encoder=True)

    compiled_path = Path(compiled_model_path)
    if not compiled_path.exists():
        print("Compiling multimodal model (this may take 20-40 minutes)...")
        t0 = time.perf_counter()
        model.compile(compiled_model_path)
        print(f"Compilation took {time.perf_counter() - t0:.1f}s")

    print("Loading model to Neuron...")
    t0 = time.perf_counter()
    model.load(compiled_model_path)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Initialize audio encoder
    if audio_compiled_path:
        print("Loading audio encoder (Neuron + CPU hybrid)...")
        t0 = time.perf_counter()
        model.init_audio_encoder_neuron(model_path, audio_compiled_path)
        print(f"Audio encoder loaded in {time.perf_counter() - t0:.1f}s")
    else:
        print("Loading audio encoder on CPU...")
        model.init_audio_encoder(model_path)
        print("Audio encoder loaded.")

    return model, config


def main():
    parser = argparse.ArgumentParser(description="ASR benchmark on LibriSpeech (Neuron)")
    parser.add_argument("--model-path", type=str,
                        default="/home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--compiled-model-path", type=str,
                        default="/home/ubuntu/traced_model/Qwen3-Omni-asr")
    parser.add_argument("--audio-compiled-path", type=str, default=None,
                        help="Path for compiled audio encoder (Neuron). If not set, uses CPU.")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tp-degree", type=int, default=8)
    parser.add_argument("--vision-tp-degree", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-context-length", type=int, default=2048)
    parser.add_argument("--vision-seq-len", type=int, default=4096)
    parser.add_argument("--split", type=str, default="test.clean",
                        help="LibriSpeech split: test.clean, test.other, etc.")
    parser.add_argument("--output-json", type=str, default="asr_results.json",
                        help="Path to save per-sample results")
    args = parser.parse_args()

    # ── 1. Load dataset ──────────────────────────────────────────────────
    from datasets import load_dataset

    print(f"Loading openslr/librispeech_asr split={args.split} ...")
    ds = load_dataset("openslr/librispeech_asr", split=args.split, streaming=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= args.num_samples:
            break
        samples.append(item)
    print(f"Loaded {len(samples)} samples")

    # ── 2. Compile and load model on Neuron ──────────────────────────────
    model, config = compile_and_load_model(
        args.model_path, args.compiled_model_path,
        args.tp_degree, args.seq_len, args.max_context_length,
        args.vision_tp_degree, args.vision_seq_len,
        args.audio_compiled_path,
    )

    # ── 3. Load processor ────────────────────────────────────────────────
    from transformers import Qwen3OmniMoeProcessor, AutoTokenizer
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter = HuggingFaceGenerationAdapter(model)

    # ── 4. Run ASR ───────────────────────────────────────────────────────
    ASR_PROMPT = "Please transcribe the following audio into English text."
    results = []
    total_audio_duration = 0.0

    print(f"\nRunning ASR on {len(samples)} samples ...")
    inference_start = time.perf_counter()

    for idx, sample in enumerate(samples):
        audio_array = np.array(sample["audio"]["array"])
        sampling_rate = sample["audio"]["sampling_rate"]
        reference = sample["text"].strip()
        audio_duration = len(audio_array) / sampling_rate
        total_audio_duration += audio_duration

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": ASR_PROMPT},
                ],
            }
        ]

        text_input = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=text_input,
            audio=[audio_array],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        generate_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        }
        if hasattr(inputs, "input_features") and inputs.input_features is not None:
            generate_kwargs["input_features"] = inputs.input_features
        if hasattr(inputs, "feature_attention_mask") and inputs.feature_attention_mask is not None:
            generate_kwargs["feature_attention_mask"] = inputs.feature_attention_mask

        t0 = time.perf_counter()
        output_ids = adapter.generate(**generate_kwargs)
        gen_time = time.perf_counter() - t0

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        eos_id = tokenizer.eos_token_id
        gen_ids = generated_ids[0].tolist()
        if eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos_id)]
        hypothesis = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        results.append({
            "id": sample.get("id", idx),
            "reference": reference,
            "hypothesis": hypothesis,
            "audio_duration_s": round(audio_duration, 2),
            "gen_time_s": round(gen_time, 3),
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(samples)}] {gen_time:.2f}s "
                  f"ref: {reference[:50]}... | hyp: {hypothesis[:50]}...")

    total_inference_time = time.perf_counter() - inference_start

    # ── 5. Compute metrics ───────────────────────────────────────────────
    from jiwer import wer, cer

    references = [r["reference"] for r in results]
    hypotheses = [r["hypothesis"] for r in results]

    refs_norm = [r.lower() for r in references]
    hyps_norm = [h.lower() for h in hypotheses]

    overall_wer = wer(refs_norm, hyps_norm)
    overall_cer = cer(refs_norm, hyps_norm)

    per_sample_wer = [wer(r, h) for r, h in zip(refs_norm, hyps_norm)]
    for i, r in enumerate(results):
        r["wer"] = round(per_sample_wer[i], 4)

    # ── 6. Report ────────────────────────────────────────────────────────
    rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else float("inf")
    avg_gen_time = sum(r["gen_time_s"] for r in results) / len(results)

    print("\n" + "=" * 70)
    print("ASR Benchmark Results (Neuron)")
    print("=" * 70)
    print(f"Model:              {args.model_path}")
    print(f"TP degree:          {args.tp_degree} (text MoE), {args.vision_tp_degree} (vision)")
    print(f"Audio encoder:      {'Neuron' if args.audio_compiled_path else 'CPU'}")
    print(f"Dataset:            openslr/librispeech_asr ({args.split})")
    print(f"Samples:            {len(results)}")
    print(f"Total audio:        {total_audio_duration:.1f}s")
    print(f"Total inference:    {total_inference_time:.1f}s")
    print(f"Avg per sample:     {avg_gen_time:.3f}s")
    print(f"RTF:                {rtf:.2f}x")
    print(f"WER:                {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"CER:                {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    print("=" * 70)

    print("\nSample results:")
    for r in results[:5]:
        print(f"  REF: {r['reference']}")
        print(f"  HYP: {r['hypothesis']}")
        print(f"  WER: {r['wer']:.4f}")
        print()

    output = {
        "model": args.model_path,
        "dataset": f"openslr/librispeech_asr:{args.split}",
        "tp_degree": args.tp_degree,
        "audio_on_neuron": args.audio_compiled_path is not None,
        "num_samples": len(results),
        "total_audio_duration_s": round(total_audio_duration, 2),
        "total_inference_time_s": round(total_inference_time, 2),
        "avg_gen_time_s": round(avg_gen_time, 3),
        "rtf": round(rtf, 4),
        "wer": round(overall_wer, 4),
        "cer": round(overall_cer, 4),
        "samples": results,
    }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
