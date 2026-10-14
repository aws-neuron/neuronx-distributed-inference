# Copyright 2025 © Amazon.com and Affiliates
"""Test Isaac at TP=2 and TP=4 on trn2.3xlarge (LNC=2, 4 logical cores).

Compiles fresh models at each TP degree, runs text-only + image+text,
and compares first-token logits against CPU reference.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH
    # TP=2:
    python test_tp.py --tp 2
    # TP=4:
    python test_tp.py --tp 4
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
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
NUM_VISION_TOKENS = (IMAGE_SIZE // 16) ** 2 // 4  # 64

TEXT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain quantum entanglement in simple terms:",
]

os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
torch.manual_seed(42)


def create_configs(tp_degree):
    """Create neuron configs for a given TP degree."""
    traced_path = f"{DATA_PATH}/traced_model/Isaac-0.2-2B-tp{tp_degree}"

    text_config = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=tp_degree,
        cp_degree=1,
        save_sharded_checkpoint=True,
        skip_sharding=False,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        context_encoding_buckets=[1024],
        token_generation_buckets=[1024],
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
        attn_kernel_enabled=False,
        attn_tkg_nki_kernel_enabled=False,
        attn_tkg_builtin_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    vision_config = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=tp_degree,
        world_size=tp_degree,
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

    return config, tokenizer, traced_path


def compile_and_load(config, tokenizer, traced_path, force_recompile=False):
    """Compile (if needed) and load the model."""
    if force_recompile and os.path.exists(traced_path):
        print(f"  Removing old traced model at {traced_path}...")
        shutil.rmtree(traced_path)

    if not os.path.exists(traced_path):
        print(f"  Compiling at TP={config.neuron_config.tp_degree}...")
        t0 = time.time()
        model = NeuronIsaacForConditionalGeneration(MODEL_PATH, config)
        model.compile(traced_path, debug=False)
        tokenizer.save_pretrained(traced_path)
        compile_time = time.time() - t0
        print(f"  Compilation complete in {compile_time:.1f}s")
        model.load(traced_path, skip_warmup=True)
    else:
        print(f"  Loading existing model from {traced_path}...")
        model = NeuronIsaacForConditionalGeneration(traced_path, config)
        model.load(traced_path, skip_warmup=True)

    return model


def validate_text(model, tokenizer, tp_degree):
    """Run text-only validation and compare against CPU reference."""
    print(f"\n  --- Text-only validation (TP={tp_degree}) ---")
    generation_model = HuggingFaceGenerationAdapter(model)

    results = []
    for i, prompt in enumerate(TEXT_PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        attention_mask = torch.ones_like(input_ids)

        sampling_params = prepare_sampling_params(
            batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0]
        )
        gen_config = GenerationConfig(
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=20,
        )

        t0 = time.time()
        outputs = generation_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=model.config.neuron_config.max_length,
            sampling_params=sampling_params,
            generation_config=gen_config,
            max_new_tokens=20,
        )
        elapsed = time.time() - t0

        generated = outputs.sequences[0, input_ids.shape[1] :]
        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        n_tokens = len(generated)

        # First-token logits comparison
        neuron_logits = outputs.scores[0][0].float().cpu()
        ref_path = os.path.join(REFERENCE_DIR, f"text_logits_{i:03d}.pt")
        cosine = -1.0
        if os.path.exists(ref_path):
            ref_logits = torch.load(ref_path, map_location="cpu")
            cosine = F.cosine_similarity(
                neuron_logits.unsqueeze(0), ref_logits.unsqueeze(0)
            ).item()

        top1_match = neuron_logits.argmax().item() == 151667  # <think>

        passed = cosine >= 0.99 and top1_match
        print(
            f"    Prompt {i}: cosine={cosine:.6f}, top1={'match' if top1_match else 'MISS'}, "
            f"{n_tokens} tok in {elapsed:.2f}s | {gen_text[:80]!r}"
        )

        results.append(
            {
                "prompt": prompt,
                "cosine": cosine,
                "top1_match": top1_match,
                "passed": passed,
                "text": gen_text[:200],
                "n_tokens": n_tokens,
                "elapsed": elapsed,
            }
        )

    all_passed = all(r["passed"] for r in results)
    return results, all_passed


def validate_image_text(model, tokenizer, tp_degree):
    """Run image+text validation."""
    print(f"\n  --- Image+text validation (TP={tp_degree}) ---")
    generation_model = HuggingFaceGenerationAdapter(model)

    try:
        ref_img = load_image(
            "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp"
        )
    except Exception:
        ref_img = Image.new("RGB", (256, 256), color="blue")

    # Prepare image inputs
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

    prompt = "Describe this image in detail."
    messages_with_image = [{"role": "user", "content": f"<image>\n{prompt}"}]
    text_with_image = tokenizer.apply_chat_template(
        messages_with_image, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(text_with_image, return_tensors="pt")[0]

    # Find and replace <image> tokens
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

    sampling_params = prepare_sampling_params(
        batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0]
    )
    gen_config = GenerationConfig(
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=30,
    )

    t0 = time.time()
    outputs = generation_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
        generation_config=gen_config,
        max_new_tokens=30,
        pixel_values=pixel_values,
        vision_mask=vision_mask,
    )
    elapsed = time.time() - t0

    generated = outputs[0, input_ids.shape[1] :]
    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    n_tokens = len(generated)

    passed = len(gen_text.strip()) > 0 and n_tokens > 0
    print(f"    Image+text: {n_tokens} tok in {elapsed:.2f}s | {gen_text[:150]!r}")
    print(f"    {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "text": gen_text[:200],
        "n_tokens": n_tokens,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True, choices=[2, 4])
    parser.add_argument("--force-recompile", action="store_true")
    args = parser.parse_args()

    tp = args.tp
    print(f"{'=' * 70}")
    print(f"TENSOR PARALLELISM TEST: TP={tp}")
    print(f"{'=' * 70}")

    config, tokenizer, traced_path = create_configs(tp)
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Traced path: {traced_path}")
    print(f"  Text TP={config.neuron_config.tp_degree}")
    print(f"  Vision TP={config.vision_config.neuron_config.tp_degree}")

    model = compile_and_load(
        config, tokenizer, traced_path, force_recompile=args.force_recompile
    )

    text_results, text_passed = validate_text(model, tokenizer, tp)
    img_result = validate_image_text(model, tokenizer, tp)

    # Summary
    all_passed = text_passed and img_result["passed"]
    print(f"\n{'=' * 70}")
    print(f"TP={tp} SUMMARY")
    print(f"{'=' * 70}")
    for r in text_results:
        print(
            f'  {"PASS" if r["passed"] else "FAIL"}: "{r["prompt"][:40]}" cosine={r["cosine"]:.6f}'
        )
    print(
        f"  {'PASS' if img_result['passed'] else 'FAIL'}: Image+text ({img_result['n_tokens']} tokens)"
    )

    if all_passed:
        print(f"\n  ALL TP={tp} TESTS PASSED")
    else:
        print(f"\n  SOME TP={tp} TESTS FAILED")
        sys.exit(1)

    # Save
    out_path = os.path.join(REFERENCE_DIR, f"neuron_tp{tp}_validation.json")
    with open(out_path, "w") as f:
        json.dump(
            {"tp_degree": tp, "text_results": text_results, "image_result": img_result},
            f,
            indent=2,
            default=str,
        )
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
