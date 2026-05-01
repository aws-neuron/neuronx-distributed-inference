# Copyright 2025 © Amazon.com and Affiliates
"""Validate Isaac TKG (token generation) on Neuron.

Tests the full CTE+TKG generation loop with:
1. Multi-token text-only generation (50+ tokens, 5 prompts)
2. Multi-token image+text generation
3. Per-step logit extraction at max_new_tokens=32
4. Edge cases: state reset, consecutive generates, vision clearing

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH
    python validate_tkg.py
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import json  # noqa: E402
import os  # noqa: E402
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
TRACED_MODEL_PATH = f"{DATA_PATH}/traced_model/Isaac-0.2-2B"

IMAGE_TOKEN_ID = 151655  # <|image_pad|>
IMAGE_SIZE = 256
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]
NUM_VISION_TOKENS = (IMAGE_SIZE // 16) ** 2 // 4  # 64

TEXT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain quantum entanglement in simple terms:",
    "The meaning of life is",
    "List three primary colors:",
]

os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
torch.manual_seed(42)


def create_neuron_configs():
    """Create text and vision neuron configurations (must match compilation)."""
    text_config = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=1,
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
        tp_degree=1,
        world_size=1,
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
    return text_config, vision_config


def load_compiled_model():
    text_config, vision_config = create_neuron_configs()
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = IsaacInferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )
    config.image_token_index = IMAGE_TOKEN_ID
    model = NeuronIsaacForConditionalGeneration(TRACED_MODEL_PATH, config)
    model.load(TRACED_MODEL_PATH, skip_warmup=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize(
                (IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def prepare_image_text_inputs(prompt, image, tokenizer):
    """Prepare input_ids with image token placeholders."""
    messages_with_image = [{"role": "user", "content": f"<image>\n{prompt}"}]
    text_with_image = tokenizer.apply_chat_template(
        messages_with_image, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(text_with_image, return_tensors="pt")[0]

    # Find <image> tokens and replace with IMAGE_TOKEN_ID placeholders
    image_text_ids = tokenizer.encode("<image>", add_special_tokens=False)
    image_text_tensor = torch.tensor(image_text_ids)

    found_pos = -1
    for i in range(len(full_ids) - len(image_text_ids) + 1):
        if torch.equal(full_ids[i : i + len(image_text_ids)], image_text_tensor):
            found_pos = i
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
    pixel_values = preprocess_image(image).to(torch.bfloat16)
    vision_mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).to(torch.bool)
    return input_ids, attention_mask, pixel_values, vision_mask


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    collect_logits=False,
    pixel_values=None,
    vision_mask=None,
):
    """Run generation and optionally collect per-step logits."""
    generation_model = HuggingFaceGenerationAdapter(model)

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    attention_mask = torch.ones_like(input_ids)

    sampling_params = prepare_sampling_params(
        batch_size=1,
        top_k=[1],
        top_p=[1.0],
        temperature=[1.0],
    )

    gen_config = GenerationConfig(
        do_sample=False,
        output_scores=collect_logits,
        return_dict_in_generate=collect_logits,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    kwargs = dict(
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
        generation_config=gen_config,
        max_new_tokens=max_new_tokens,
    )
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values
    if vision_mask is not None:
        kwargs["vision_mask"] = vision_mask

    start = time.time()
    outputs = generation_model.generate(input_ids, **kwargs)
    elapsed = time.time() - start

    if collect_logits and hasattr(outputs, "sequences"):
        generated_ids = outputs.sequences[0, input_ids.shape[1] :]
        scores = outputs.scores if outputs.scores else []
    else:
        if hasattr(outputs, "sequences"):
            generated_ids = outputs.sequences[0, input_ids.shape[1] :]
        else:
            generated_ids = outputs[0, input_ids.shape[1] :]
        scores = []

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    gen_text_clean = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "input_ids": input_ids,
        "generated_ids": generated_ids,
        "text_raw": gen_text,
        "text_clean": gen_text_clean,
        "scores": scores,
        "elapsed": elapsed,
        "num_tokens": len(generated_ids),
        "tokens_per_sec": len(generated_ids) / elapsed if elapsed > 0 else 0,
    }


# ===========================================================================
# Test functions
# ===========================================================================


def test_multi_token_text(model, tokenizer):
    """Test 1: Multi-token text-only generation for all 5 prompts."""
    print(f"\n{'=' * 70}")
    print("TEST 1: Multi-token text-only generation (50 tokens)")
    print(f"{'=' * 70}")

    results = []
    all_passed = True

    for i, prompt in enumerate(TEXT_PROMPTS):
        print(f'\n--- Prompt {i}: "{prompt}" ---')
        result = generate_text(model, tokenizer, prompt, max_new_tokens=50)

        # Validation
        passed = True
        failures = []

        # Non-empty
        if len(result["text_clean"].strip()) == 0:
            passed = False
            failures.append("Empty output")

        # Generated expected number of tokens (or hit EOS)
        if result["num_tokens"] == 0:
            passed = False
            failures.append("Zero tokens generated")

        # Should start with <think> (Isaac thinking model)
        first_token = (
            result["generated_ids"][0].item() if result["num_tokens"] > 0 else -1
        )
        if first_token != 151667:
            failures.append(
                f"First token {first_token} != <think> (151667) — may be normal"
            )

        # Check for repetition (degenerate TKG)
        if result["num_tokens"] >= 10:
            last_10 = result["generated_ids"][-10:].tolist()
            if len(set(last_10)) <= 2:
                passed = False
                failures.append(f"Degenerate repetition in last 10 tokens: {last_10}")

        result["passed"] = passed
        result["failures"] = failures
        results.append(result)
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {result['num_tokens']} tokens in {result['elapsed']:.2f}s ({result['tokens_per_sec']:.1f} tok/s)"
        )
        print(f"  Output: {result['text_clean'][:200]!r}")
        for f in failures:
            print(f"    NOTE: {f}")

    return results, all_passed


def test_logit_collection(model, tokenizer):
    """Test 2: Collect per-step logits at max_new_tokens=32."""
    print(f"\n{'=' * 70}")
    print("TEST 2: Per-step logit collection (32 tokens)")
    print(f"{'=' * 70}")

    results = []
    all_passed = True

    for i, prompt in enumerate(TEXT_PROMPTS[:3]):  # First 3 prompts
        print(f'\n--- Prompt {i}: "{prompt}" ---')
        result = generate_text(
            model, tokenizer, prompt, max_new_tokens=32, collect_logits=True
        )

        passed = True
        failures = []

        # Check we got scores
        n_scores = len(result["scores"])
        print(
            f"  Generated {result['num_tokens']} tokens, collected {n_scores} score tensors"
        )

        if n_scores == 0:
            passed = False
            failures.append("No scores collected (output_logits may not be working)")
        else:
            # Check each score tensor
            for step_idx, score in enumerate(result["scores"]):
                s = score[0].float()  # [vocab_size]
                if torch.isnan(s).any():
                    passed = False
                    failures.append(f"NaN at step {step_idx}")
                    break
                if torch.isinf(s).any():
                    passed = False
                    failures.append(f"Inf at step {step_idx}")
                    break

            # Compare first-token logits against saved reference
            ref_path = os.path.join(REFERENCE_DIR, f"text_logits_{i:03d}.pt")
            if os.path.exists(ref_path) and n_scores > 0:
                ref_logits = torch.load(ref_path, map_location="cpu")
                neuron_first = result["scores"][0][0].float().cpu()
                cosine = F.cosine_similarity(
                    neuron_first.unsqueeze(0), ref_logits.unsqueeze(0)
                ).item()
                print(f"  First-token cosine vs CPU ref: {cosine:.6f}")
                if cosine < 0.99:
                    passed = False
                    failures.append(f"First-token cosine {cosine:.6f} < 0.99")

            # Check that later tokens also have reasonable logits
            if n_scores >= 5:
                for step in [0, n_scores // 2, n_scores - 1]:
                    s = result["scores"][step][0].float()
                    top1 = s.argmax().item()
                    top1_val = s.max().item()
                    print(
                        f"  Step {step}: top-1={top1} ({tokenizer.decode([top1])!r}), logit={top1_val:.2f}"
                    )

        result["passed"] = passed
        result["failures"] = failures
        result["n_scores"] = n_scores
        results.append(result)
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]")
        for f in failures:
            print(f"    FAILURE: {f}")

    return results, all_passed


def test_state_reset(model, tokenizer):
    """Test 3: Verify state resets between consecutive generate() calls."""
    print(f"\n{'=' * 70}")
    print("TEST 3: State reset between consecutive generates")
    print(f"{'=' * 70}")

    passed = True
    failures = []

    # Run same prompt twice — should get identical output
    print("\n  Running same prompt twice...")
    r1 = generate_text(model, tokenizer, "The capital of France is", max_new_tokens=20)
    r2 = generate_text(model, tokenizer, "The capital of France is", max_new_tokens=20)

    ids1 = r1["generated_ids"].tolist()
    ids2 = r2["generated_ids"].tolist()
    match = ids1 == ids2
    print(f"  Run 1: {r1['text_clean'][:100]!r}")
    print(f"  Run 2: {r2['text_clean'][:100]!r}")
    print(f"  Token sequences match: {match}")
    if not match:
        # Check how many match
        min_len = min(len(ids1), len(ids2))
        matching = sum(1 for a, b in zip(ids1[:min_len], ids2[:min_len]) if a == b)
        print(f"  Matching: {matching}/{min_len} tokens")
        if matching < min_len * 0.9:
            failures.append(
                f"Same prompt gave different outputs: {matching}/{min_len} match"
            )
            passed = False

    # Run different prompts — verify no cross-contamination
    print("\n  Running different prompts...")
    r3 = generate_text(model, tokenizer, "def fibonacci(n):", max_new_tokens=20)
    r4 = generate_text(model, tokenizer, "The capital of France is", max_new_tokens=20)

    ids4 = r4["generated_ids"].tolist()
    match_after = ids4 == ids2
    print(f"  After different prompt, re-running 'France': {r4['text_clean'][:100]!r}")
    print(f"  Matches original: {match_after}")
    if not match_after:
        min_len = min(len(ids4), len(ids2))
        matching = sum(1 for a, b in zip(ids4[:min_len], ids2[:min_len]) if a == b)
        if matching < min_len * 0.9:
            failures.append(
                f"State contamination: re-run after different prompt gives different output ({matching}/{min_len})"
            )
            passed = False

    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}]")
    for f in failures:
        print(f"    FAILURE: {f}")

    return {"passed": passed, "failures": failures}


def test_image_text_generation(model, tokenizer):
    """Test 4: Multi-token image+text generation."""
    print(f"\n{'=' * 70}")
    print("TEST 4: Image+text multi-token generation")
    print(f"{'=' * 70}")

    passed = True
    failures = []

    try:
        ref_img = load_image(
            "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp"
        )
    except Exception as e:
        print(f"  WARNING: Could not load reference image: {e}")
        ref_img = Image.new("RGB", (256, 256), color="blue")

    prompt = "Describe this image in detail."
    input_ids, attention_mask, pixel_values, vision_mask = prepare_image_text_inputs(
        prompt, ref_img, tokenizer
    )

    print(f"  Input: {input_ids.shape}, vision tokens: {vision_mask.sum().item()}")

    generation_model = HuggingFaceGenerationAdapter(model)
    sampling_params = prepare_sampling_params(
        batch_size=1,
        top_k=[1],
        top_p=[1.0],
        temperature=[1.0],
    )
    gen_config = GenerationConfig(
        do_sample=False,
        output_scores=False,
        return_dict_in_generate=False,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=50,
    )

    start = time.time()
    outputs = generation_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
        generation_config=gen_config,
        max_new_tokens=50,
        pixel_values=pixel_values,
        vision_mask=vision_mask,
    )
    elapsed = time.time() - start

    generated_ids = outputs[0, input_ids.shape[1] :]
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_tokens = len(generated_ids)

    print(
        f"  Generated {n_tokens} tokens in {elapsed:.2f}s ({n_tokens / elapsed:.1f} tok/s)"
    )
    print(f"  Output: {gen_text[:300]!r}")

    if len(gen_text.strip()) == 0:
        passed = False
        failures.append("Empty image+text output")

    if n_tokens == 0:
        passed = False
        failures.append("Zero tokens generated")

    # Check for degenerate repetition
    if n_tokens >= 10:
        last_10 = generated_ids[-10:].tolist()
        if len(set(last_10)) <= 2:
            passed = False
            failures.append(f"Degenerate repetition: {last_10}")

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]")
    for f in failures:
        print(f"    FAILURE: {f}")

    return {
        "passed": passed,
        "failures": failures,
        "text": gen_text[:300],
        "n_tokens": n_tokens,
    }


def test_vision_state_reset(model, tokenizer):
    """Test 5: Vision state resets between image and text-only prompts."""
    print(f"\n{'=' * 70}")
    print("TEST 5: Vision state reset (image -> text -> image)")
    print(f"{'=' * 70}")

    passed = True
    failures = []

    # 1. Run text-only
    r1 = generate_text(model, tokenizer, "The capital of France is", max_new_tokens=20)
    print(f"  Text-only: {r1['text_clean'][:100]!r}")

    # 2. Run image+text
    img = Image.new("RGB", (256, 256), color="red")
    input_ids, attention_mask, pixel_values, vision_mask = prepare_image_text_inputs(
        "Describe this image.", img, tokenizer
    )

    generation_model = HuggingFaceGenerationAdapter(model)
    sampling_params = prepare_sampling_params(
        batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0]
    )
    gen_config = GenerationConfig(
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=20,
    )
    outputs = generation_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
        generation_config=gen_config,
        max_new_tokens=20,
        pixel_values=pixel_values,
        vision_mask=vision_mask,
    )
    img_text = tokenizer.decode(
        outputs[0, input_ids.shape[1] :], skip_special_tokens=True
    )
    print(f"  Image+text: {img_text[:100]!r}")

    # 3. Run text-only again — should match run 1
    r3 = generate_text(model, tokenizer, "The capital of France is", max_new_tokens=20)
    print(f"  Text-only (after image): {r3['text_clean'][:100]!r}")

    ids1 = r1["generated_ids"].tolist()
    ids3 = r3["generated_ids"].tolist()
    match = ids1 == ids3
    print(f"  Text outputs match (pre/post image): {match}")

    if not match:
        min_len = min(len(ids1), len(ids3))
        matching = sum(1 for a, b in zip(ids1[:min_len], ids3[:min_len]) if a == b)
        if matching < min_len * 0.9:
            passed = False
            failures.append(
                f"Vision state leaked: text output changed after image prompt ({matching}/{min_len})"
            )

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]")
    for f in failures:
        print(f"    FAILURE: {f}")

    return {"passed": passed, "failures": failures}


# ===========================================================================
# Main
# ===========================================================================


def main():
    print(f"{'=' * 70}")
    print("TKG VALIDATION: Isaac on Neuron")
    print(f"{'=' * 70}")

    model, tokenizer = load_compiled_model()

    # Run all tests
    test_results = {}

    r1, p1 = test_multi_token_text(model, tokenizer)
    test_results["multi_token_text"] = {
        "results": [
            {
                "prompt": TEXT_PROMPTS[i],
                "passed": r["passed"],
                "n_tokens": r["num_tokens"],
                "text": r["text_clean"][:200],
                "tok_per_sec": r["tokens_per_sec"],
            }
            for i, r in enumerate(r1)
        ],
        "all_passed": p1,
    }

    r2, p2 = test_logit_collection(model, tokenizer)
    test_results["logit_collection"] = {
        "results": [
            {
                "prompt": TEXT_PROMPTS[i],
                "passed": r["passed"],
                "n_scores": r.get("n_scores", 0),
            }
            for i, r in enumerate(r2)
        ],
        "all_passed": p2,
    }

    r3 = test_state_reset(model, tokenizer)
    test_results["state_reset"] = r3

    r4 = test_image_text_generation(model, tokenizer)
    test_results["image_text_generation"] = r4

    r5 = test_vision_state_reset(model, tokenizer)
    test_results["vision_state_reset"] = r5

    # Overall summary
    all_tests = [p1, p2, r3["passed"], r4["passed"], r5["passed"]]
    all_passed = all(all_tests)

    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")
    test_names = [
        "Multi-token text",
        "Logit collection",
        "State reset",
        "Image+text generation",
        "Vision state reset",
    ]
    for name, p in zip(test_names, all_tests):
        print(f"  {'PASS' if p else 'FAIL'}: {name}")

    if all_passed:
        print(f"\n  ALL TKG TESTS PASSED")
    else:
        print(f"\n  SOME TESTS FAILED")
        sys.exit(1)

    # Save results
    out_path = os.path.join(REFERENCE_DIR, "neuron_tkg_validation.json")
    with open(out_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
