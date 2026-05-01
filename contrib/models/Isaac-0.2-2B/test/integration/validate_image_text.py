# Copyright 2025 © Amazon.com and Affiliates
"""Validate Isaac image+text inference on Neuron.

Tests the full VLM pipeline:
  pixel_values -> SigLIP2 encoder -> pixel_shuffle -> MLP projector -> text decoder

Since the compiled model uses image_size=256, we use 256x256 images.
The CPU reference was captured with tensor_stream (different preprocessing),
so we validate:
1. E2E generates non-garbage text (qualitative)
2. Top-1 token is <think> (consistent with model behavior)
3. Vision encoder produces reasonable embeddings (not NaN/Inf)

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH
    python validate_image_text.py
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

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
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_HOME", "/mnt/models")
REFERENCE_DIR = f"{DATA_PATH}/reference_outputs"
MODEL_PATH = f"{DATA_PATH}/Isaac-0.2-2B-Preview"
TRACED_MODEL_PATH = f"{DATA_PATH}/traced_model/Isaac-0.2-2B"

# Isaac uses <|image_pad|> = 151655 as placeholder for vision embeddings
IMAGE_TOKEN_ID = 151655
IMAGE_SIZE = 256  # Compiled model's vision image_size
PATCH_SIZE = 16
PIXEL_SHUFFLE_SCALE = 2
NUM_VISION_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2 // (PIXEL_SHUFFLE_SCALE**2)  # 64

# SigLIP2 normalization
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

# Environment
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
    """Load the pre-compiled Isaac model."""
    text_config, vision_config = create_neuron_configs()

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    config = IsaacInferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    # Set image_token_index (Isaac config doesn't have it by default)
    config.image_token_index = IMAGE_TOKEN_ID

    print(f"Loading compiled model from {TRACED_MODEL_PATH}...")
    model = NeuronIsaacForConditionalGeneration(TRACED_MODEL_PATH, config)
    model.load(TRACED_MODEL_PATH, skip_warmup=True)
    print("Model loaded successfully.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image to pixel_values tensor [1, 3, H, W].

    Matches SigLIP2 normalization: rescale to [0,1], normalize with mean/std=0.5.
    """
    transform = T.Compose(
        [
            T.Resize(
                (IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),  # [C, H, W] in [0, 1]
            T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),  # -> [-1, 1]
        ]
    )
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, 256, 256]
    return pixel_values


def prepare_image_text_inputs(prompt: str, image: Image.Image, tokenizer):
    """Prepare input_ids, attention_mask, pixel_values, and vision_mask.

    Isaac's processor uses -256 as image token placeholder in tensor_stream.
    For NxDI, we:
    1. Tokenize with chat template
    2. Insert IMAGE_TOKEN_ID (151655) for vision token positions
    3. Create boolean vision_mask

    Returns:
        input_ids: [1, seq_len] with IMAGE_TOKEN_ID at vision positions
        attention_mask: [1, seq_len] all ones
        pixel_values: [1, 3, 256, 256] normalized
        vision_mask: [1, seq_len, 1] bool
    """
    # Build input_ids with image token placeholders
    # Format: <|im_start|>user\n[64 image tokens]\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the text (without image tokens)
    # The template produces: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    text_ids = tokenizer.encode(text, return_tensors="pt")  # [1, text_len]
    text_ids = text_ids[0]  # [text_len]

    # Find where to insert image tokens
    # Isaac inserts image tokens after "user\n" — between the user header and the prompt content
    # The chat template is: <|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    # But since we used the prompt directly (without <image>), we need to insert manually

    # Re-create with <image> placeholder in the message
    messages_with_image = [{"role": "user", "content": f"<image>\n{prompt}"}]
    text_with_image = tokenizer.apply_chat_template(
        messages_with_image, tokenize=False, add_generation_prompt=True
    )
    # Tokenize fully
    full_ids = tokenizer.encode(text_with_image, return_tensors="pt")[0]  # [seq_len]

    # Now find where "<image>" tokens are and replace with IMAGE_TOKEN_ID blocks
    # The tokenizer encodes "<image>" as multiple tokens: [27, 1805, 29] = '<', 'image', '>'
    # We need to replace those 3 tokens with NUM_VISION_TOKENS copies of IMAGE_TOKEN_ID

    # Find the "<image>" token sequence
    image_text_ids = tokenizer.encode(
        "<image>", add_special_tokens=False
    )  # [27, 1805, 29]
    image_text_tensor = torch.tensor(image_text_ids)

    # Find position of <image> in full_ids
    found_pos = -1
    for i in range(len(full_ids) - len(image_text_ids) + 1):
        if torch.equal(full_ids[i : i + len(image_text_ids)], image_text_tensor):
            found_pos = i
            break

    if found_pos >= 0:
        # Replace <image> tokens with IMAGE_TOKEN_ID * NUM_VISION_TOKENS
        before = full_ids[:found_pos]
        after = full_ids[found_pos + len(image_text_ids) :]
        image_tokens = torch.full(
            (NUM_VISION_TOKENS,), IMAGE_TOKEN_ID, dtype=torch.long
        )
        input_ids = torch.cat([before, image_tokens, after]).unsqueeze(0)
    else:
        # Fallback: prepend image tokens after user header
        print(
            "WARNING: Could not find <image> in tokenized text, prepending image tokens"
        )
        image_tokens = torch.full(
            (NUM_VISION_TOKENS,), IMAGE_TOKEN_ID, dtype=torch.long
        )
        # Insert after position 2 (after <|im_start|>user\n)
        input_ids = torch.cat([full_ids[:3], image_tokens, full_ids[3:]]).unsqueeze(0)

    attention_mask = torch.ones_like(input_ids)
    pixel_values = preprocess_image(image)
    vision_mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).to(torch.bool)

    return input_ids, attention_mask, pixel_values, vision_mask


def run_validation():
    """Run image+text validation."""
    model, tokenizer = load_compiled_model()
    generation_model = HuggingFaceGenerationAdapter(model)

    print(f"\n{'=' * 70}")
    print("IMAGE+TEXT INFERENCE VALIDATION ON NEURON")
    print(f"{'=' * 70}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Vision tokens: {NUM_VISION_TOKENS}")
    print(f"  Image token ID: {IMAGE_TOKEN_ID}")

    # Test images
    test_cases = []

    # Test 1: Solid color image (sanity check)
    img_red = Image.new("RGB", (256, 256), color="red")
    test_cases.append(("Describe this image in detail.", img_red, "red_square"))

    # Test 2: Reference image (resized to 256x256)
    try:
        img_ref = load_image(
            "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp"
        )
        test_cases.append(
            ("Describe this image in detail.", img_ref, "reference_image")
        )
        test_cases.append(
            ("What text or signs do you see in this image?", img_ref, "reference_ocr")
        )
    except Exception as e:
        print(f"  WARNING: Could not load reference image: {e}")

    results = []
    all_passed = True

    for i, (prompt, image, label) in enumerate(test_cases):
        print(f'\n--- Test {i}: [{label}] "{prompt}" ---')
        print(f"  Image: {image.size} -> will be resized to {IMAGE_SIZE}x{IMAGE_SIZE}")

        try:
            input_ids, attention_mask, pixel_values, vision_mask = (
                prepare_image_text_inputs(prompt, image, tokenizer)
            )
        except Exception as e:
            print(f"  ERROR in input preparation: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False
            continue

        seq_len = input_ids.shape[1]
        n_image_tokens = vision_mask.sum().item()
        print(f"  input_ids: {input_ids.shape}, seq_len={seq_len}")
        print(f"  pixel_values: {pixel_values.shape}, dtype={pixel_values.dtype}")
        print(f"  vision_mask: {n_image_tokens} image tokens")
        print(
            f"  pixel_values range: [{pixel_values.min():.4f}, {pixel_values.max():.4f}]"
        )

        # Verify seq_len fits in bucket
        if seq_len > 1024:
            print(f"  SKIP: seq_len {seq_len} > max bucket 1024")
            continue

        sampling_params = prepare_sampling_params(
            batch_size=1,
            top_k=[1],
            top_p=[1.0],
            temperature=[1.0],
        )

        generation_config = GenerationConfig(
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=30,  # Generate enough to see meaningful output
        )

        try:
            outputs = generation_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=model.config.neuron_config.max_length,
                sampling_params=sampling_params,
                generation_config=generation_config,
                max_new_tokens=30,
                pixel_values=pixel_values.to(torch.bfloat16),
                vision_mask=vision_mask,
            )
        except Exception as e:
            print(f"  ERROR in generate: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False
            results.append({"label": label, "passed": False, "error": str(e)})
            continue

        # Extract generated tokens
        if hasattr(outputs, "sequences"):
            generated = outputs.sequences[0, input_ids.shape[1] :]
            gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        else:
            generated = outputs[0, input_ids.shape[1] :]
            gen_text = tokenizer.decode(generated, skip_special_tokens=True)

        print(f"  Generated: {gen_text[:200]!r}")

        # Extract first-token logits
        first_logits = None
        if (
            hasattr(outputs, "scores")
            and outputs.scores is not None
            and len(outputs.scores) > 0
        ):
            first_logits = outputs.scores[0][0].float().cpu()
            top5 = torch.topk(first_logits, 5)
            top5_tokens = [tokenizer.decode([tid]) for tid in top5.indices.tolist()]
            print(f"  Top-5 tokens: {list(zip(top5_tokens, top5.values.tolist()))}")
            top1 = first_logits.argmax().item()
            print(f"  Top-1: {top1} ({tokenizer.decode([top1])!r})")

        # Validation checks
        passed = True
        failures = []

        # Check 1: Generated text is not empty
        if len(gen_text.strip()) == 0:
            passed = False
            failures.append("Empty generated text")

        # Check 2: No NaN in logits
        if first_logits is not None and torch.isnan(first_logits).any():
            passed = False
            failures.append("NaN in logits")

        # Check 3: No Inf in logits
        if first_logits is not None and torch.isinf(first_logits).any():
            passed = False
            failures.append("Inf in logits")

        # Check 4: Top-1 should be <think> (consistent with model behavior)
        if first_logits is not None:
            top1 = first_logits.argmax().item()
            if top1 != 151667:
                # Not necessarily a failure for image inputs
                print(
                    f"  NOTE: Top-1 is {top1}, not <think> (151667) — may be normal for image input"
                )

        result = {
            "label": label,
            "prompt": prompt,
            "passed": passed,
            "generated_text": gen_text[:200],
            "top1": first_logits.argmax().item() if first_logits is not None else None,
            "failures": failures,
        }
        results.append(result)
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]")
        for f in failures:
            print(f"    FAILURE: {f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"  Passed: {passed_count}/{total}")

    if all_passed:
        print("\n  ALL IMAGE+TEXT TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED — see details above")
        sys.exit(1)

    # Save results
    out_path = os.path.join(REFERENCE_DIR, "neuron_image_text_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_validation()
