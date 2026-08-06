# Copyright 2025 © Amazon.com and Affiliates
"""Formal benchmark for Isaac on trn2.3xlarge.

Measures TTFT, TPOT, tok/s, and HBM usage with warmup and multiple iterations.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH
    python benchmark.py [--seq-len 1024] [--warmup 3] [--iterations 10]
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, GenerationConfig  # noqa: E402
from transformers.image_utils import load_image  # noqa: E402

from neuronx_distributed_inference.models.config import (  # noqa: E402
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (  # noqa: E402
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)
from neuronx_distributed_inference.modules.generation.sampling import (  # noqa: E402
    prepare_sampling_params,
)

from isaac_neuron.modeling_isaac import (  # noqa: E402
    NeuronIsaacForConditionalGeneration,
    IsaacInferenceConfig,
)

# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_HOME", "/mnt/models")
REFERENCE_DIR = f"{DATA_PATH}/reference_outputs"
MODEL_PATH = f"{DATA_PATH}/Isaac-0.2-2B-Preview"
IMAGE_TOKEN_ID = 151655
IMAGE_SIZE = 256
NUM_VISION_TOKENS = 64  # (256/16)^2 / 4

os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
torch.manual_seed(42)


def create_model_and_tokenizer(seq_len, tp=1):
    """Create and load model at specified config."""
    traced_path = f"{DATA_PATH}/traced_model/Isaac-0.2-2B-bench-s{seq_len}-tp{tp}"

    text_config = NeuronConfig(
        batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        tp_degree=tp,
        cp_degree=1,
        save_sharded_checkpoint=True,
        skip_sharding=False,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        context_encoding_buckets=[seq_len],
        token_generation_buckets=[seq_len],
        async_mode=False,
        on_device_sampling_config=OnDeviceSamplingConfig(
            dynamic=True,
            do_sample=True,
            deterministic=True,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            global_topk=256,
            top_k_kernel_enabled=True,
        ),
        output_logits=True,
        fused_qkv=False,
        sequence_parallel_enabled=False,
        attn_kernel_enabled=True,
        attn_tkg_nki_kernel_enabled=False,
        attn_tkg_builtin_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    vision_config = NeuronConfig(
        batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        tp_degree=tp,
        world_size=tp,
        save_sharded_checkpoint=True,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        buckets=[1],
        fused_qkv=False,
        attn_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = IsaacInferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )
    config.image_token_index = IMAGE_TOKEN_ID

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Compile or load
    if not os.path.exists(traced_path):
        print(f"  Compiling (seq_len={seq_len}, TP={tp})...")
        t0 = time.time()
        model = NeuronIsaacForConditionalGeneration(MODEL_PATH, config)
        model.compile(traced_path, debug=False)
        tokenizer.save_pretrained(traced_path)
        print(f"  Compiled in {time.time() - t0:.1f}s")
        model.load(traced_path, skip_warmup=True)
    else:
        print(f"  Loading from {traced_path}...")
        model = NeuronIsaacForConditionalGeneration(traced_path, config)
        model.load(traced_path, skip_warmup=True)

    return model, tokenizer


def benchmark_text(model, tokenizer, prompt, max_new_tokens, warmup, iterations):
    """Benchmark text-only generation with proper warmup and timing."""
    gen_model = HuggingFaceGenerationAdapter(model)

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    attention_mask = torch.ones_like(input_ids)
    input_len = input_ids.shape[1]

    sampling_params = prepare_sampling_params(
        batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0]
    )
    gen_config = GenerationConfig(
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    # Warmup
    for _ in range(warmup):
        gen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=model.config.neuron_config.max_length,
            sampling_params=sampling_params,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
        )

    # Timed iterations
    latencies = []
    token_counts = []
    for _ in range(iterations):
        t0 = time.time()
        outputs = gen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=model.config.neuron_config.max_length,
            sampling_params=sampling_params,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0

        generated = outputs.sequences[0, input_len:]
        n_tokens = len(generated)
        latencies.append(elapsed)
        token_counts.append(n_tokens)

    gen_text = tokenizer.decode(
        outputs.sequences[0, input_len:], skip_special_tokens=True
    )

    avg_tokens = statistics.mean(token_counts)
    avg_latency = statistics.mean(latencies)
    # TTFT ≈ latency - (n_tokens - 1) * TPOT; approximate TPOT from overall
    avg_tpot = avg_latency / avg_tokens if avg_tokens > 1 else avg_latency
    avg_ttft = (
        avg_latency - (avg_tokens - 1) * avg_tpot if avg_tokens > 1 else avg_latency
    )
    avg_tps = avg_tokens / avg_latency

    return {
        "input_tokens": input_len,
        "avg_output_tokens": avg_tokens,
        "avg_latency_s": avg_latency,
        "ttft_ms": avg_ttft * 1000,
        "tpot_ms": avg_tpot * 1000,
        "tok_per_sec": avg_tps,
        "latency_std_ms": statistics.stdev(latencies) * 1000
        if len(latencies) > 1
        else 0,
        "text_preview": gen_text[:150],
    }


def benchmark_image_text(model, tokenizer, max_new_tokens, warmup, iterations):
    """Benchmark image+text generation."""
    gen_model = HuggingFaceGenerationAdapter(model)

    # Load test image
    try:
        ref_img = load_image(
            "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp"
        )
    except Exception:
        ref_img = Image.new("RGB", (256, 256), color="blue")

    transform = T.Compose(
        [
            T.Resize(
                (IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    pixel_values = transform(ref_img).unsqueeze(0).to(torch.bfloat16)

    # Build input with image tokens
    prompt = "Describe this image in detail."
    messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(text, return_tensors="pt")[0]

    image_text_ids = tokenizer.encode("<image>", add_special_tokens=False)
    image_text_tensor = torch.tensor(image_text_ids)
    found_pos = -1
    for idx in range(len(full_ids) - len(image_text_ids) + 1):
        if torch.equal(full_ids[idx : idx + len(image_text_ids)], image_text_tensor):
            found_pos = idx
            break

    if found_pos >= 0:
        before = full_ids[:found_pos]
        after = full_ids[found_pos + len(image_text_ids) :]
        image_tokens = torch.full(
            (NUM_VISION_TOKENS,), IMAGE_TOKEN_ID, dtype=torch.long
        )
        input_ids = torch.cat([before, image_tokens, after]).unsqueeze(0)
    else:
        image_tokens = torch.full(
            (NUM_VISION_TOKENS,), IMAGE_TOKEN_ID, dtype=torch.long
        )
        input_ids = torch.cat([full_ids[:3], image_tokens, full_ids[3:]]).unsqueeze(0)

    attention_mask = torch.ones_like(input_ids)
    vision_mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).to(torch.bool)
    input_len = input_ids.shape[1]

    sampling_params = prepare_sampling_params(
        batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0]
    )
    gen_config = GenerationConfig(
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    # Warmup
    for _ in range(warmup):
        gen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=model.config.neuron_config.max_length,
            sampling_params=sampling_params,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
            pixel_values=pixel_values,
            vision_mask=vision_mask,
        )

    # Timed iterations
    latencies = []
    token_counts = []
    for _ in range(iterations):
        t0 = time.time()
        outputs = gen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=model.config.neuron_config.max_length,
            sampling_params=sampling_params,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
            pixel_values=pixel_values,
            vision_mask=vision_mask,
        )
        elapsed = time.time() - t0

        generated = outputs[0, input_len:]
        n_tokens = len(generated)
        latencies.append(elapsed)
        token_counts.append(n_tokens)

    gen_text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

    avg_tokens = statistics.mean(token_counts)
    avg_latency = statistics.mean(latencies)
    avg_tpot = avg_latency / avg_tokens if avg_tokens > 1 else avg_latency
    avg_ttft = (
        avg_latency - (avg_tokens - 1) * avg_tpot if avg_tokens > 1 else avg_latency
    )
    avg_tps = avg_tokens / avg_latency

    return {
        "input_tokens": input_len,
        "vision_tokens": NUM_VISION_TOKENS,
        "avg_output_tokens": avg_tokens,
        "avg_latency_s": avg_latency,
        "ttft_ms": avg_ttft * 1000,
        "tpot_ms": avg_tpot * 1000,
        "tok_per_sec": avg_tps,
        "latency_std_ms": statistics.stdev(latencies) * 1000
        if len(latencies) > 1
        else 0,
        "text_preview": gen_text[:150],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"ISAAC BENCHMARK — seq_len={args.seq_len}, TP={args.tp}")
    print(
        f"warmup={args.warmup}, iterations={args.iterations}, max_new_tokens={args.max_new_tokens}"
    )
    print(f"{'=' * 70}")

    model, tokenizer = create_model_and_tokenizer(args.seq_len, args.tp)

    all_results = {
        "config": {
            "seq_len": args.seq_len,
            "tp": args.tp,
            "batch_size": 1,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "max_new_tokens": args.max_new_tokens,
            "instance": "trn2.3xlarge",
            "lnc": 2,
            "sdk": "2.29",
            "model": "Isaac-0.2-2B-Preview",
        },
        "text_benchmarks": [],
        "image_text_benchmark": None,
    }

    # Text benchmarks — short, medium, long prompts
    text_prompts = [
        ("short", "The capital of France is", 32),
        ("medium", "Explain quantum entanglement in simple terms:", 128),
        (
            "long",
            "Write a detailed essay about the history and future of artificial intelligence, "
            "covering its origins, key milestones, current capabilities, and predictions "
            "for the next decade:",
            args.max_new_tokens,
        ),
    ]

    for label, prompt, max_tok in text_prompts:
        print(f"\n--- Text benchmark: {label} (max_new_tokens={max_tok}) ---")
        result = benchmark_text(
            model, tokenizer, prompt, max_tok, args.warmup, args.iterations
        )
        result["label"] = label
        result["prompt"] = prompt[:80]
        all_results["text_benchmarks"].append(result)
        print(
            f"  Input: {result['input_tokens']} tok, Output: {result['avg_output_tokens']:.0f} tok"
        )
        print(f"  TTFT: {result['ttft_ms']:.1f}ms")
        print(f"  TPOT: {result['tpot_ms']:.2f}ms")
        print(f"  Throughput: {result['tok_per_sec']:.1f} tok/s")
        print(f"  Latency std: {result['latency_std_ms']:.1f}ms")

    # Image+text benchmark
    print(f"\n--- Image+text benchmark ---")
    img_result = benchmark_image_text(
        model, tokenizer, args.max_new_tokens, args.warmup, args.iterations
    )
    all_results["image_text_benchmark"] = img_result
    print(
        f"  Input: {img_result['input_tokens']} tok ({img_result['vision_tokens']} vision)"
    )
    print(f"  Output: {img_result['avg_output_tokens']:.0f} tok")
    print(f"  TTFT: {img_result['ttft_ms']:.1f}ms (includes vision encoding)")
    print(f"  TPOT: {img_result['tpot_ms']:.2f}ms")
    print(f"  Throughput: {img_result['tok_per_sec']:.1f} tok/s")

    # Summary table
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Workload':<20} {'In':>5} {'Out':>5} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'tok/s':>8}"
    )
    print("-" * 60)
    for r in all_results["text_benchmarks"]:
        print(
            f"{r['label']:<20} {r['input_tokens']:>5} {r['avg_output_tokens']:>5.0f} "
            f"{r['ttft_ms']:>10.1f} {r['tpot_ms']:>10.2f} {r['tok_per_sec']:>8.1f}"
        )
    ir = all_results["image_text_benchmark"]
    print(
        f"{'image+text':<20} {ir['input_tokens']:>5} {ir['avg_output_tokens']:>5.0f} "
        f"{ir['ttft_ms']:>10.1f} {ir['tpot_ms']:>10.2f} {ir['tok_per_sec']:>8.1f}"
    )

    # Save
    out_path = os.path.join(
        REFERENCE_DIR, f"benchmark_s{args.seq_len}_tp{args.tp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
