#!/usr/bin/env python3
"""
Integration tests for Qwen2.5-VL-7B-Instruct NeuronX implementation.

Tests cover:
  - Model loading from pre-compiled artifacts
  - Text-only generation with logit validation
  - Vision-language generation with synthetic image
  - Performance metrics (TTFT, throughput)

Prerequisites:
  - Pre-compiled model at COMPILED_MODEL_PATH
  - HuggingFace weights at MODEL_PATH
  - Neuron device available (trn2 or inf2)

Usage:
  # Run all tests
  pytest test/integration/test_model.py -v --capture=tee-sys

  # Run specific test
  pytest test/integration/test_model.py::test_text_generation -v
"""

import os
import sys
import time
import logging

import pytest
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ---- Configuration ----
# Override via environment variables if needed
MODEL_PATH = os.environ.get("QWEN25VL_MODEL_PATH", "/mnt/models/Qwen2.5-VL-7B-Instruct")
COMPILED_MODEL_PATH = os.environ.get(
    "QWEN25VL_COMPILED_PATH", "/mnt/models/qwen25vl_compiled"
)
TP_DEGREE = int(os.environ.get("QWEN25VL_TP_DEGREE", "4"))
SEQ_LEN = int(os.environ.get("QWEN25VL_SEQ_LEN", "4096"))

# Token IDs for Qwen2.5-VL
BOS_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151645


# ---- Fixtures ----


@pytest.fixture(scope="module")
def model_and_adapter():
    """Load pre-compiled model and create generation adapter."""
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import (
        load_pretrained_config,
        HuggingFaceGenerationAdapter,
    )
    from modeling_qwen2_5_vl import (
        NeuronQwen2_5_VLForCausalLM,
        Qwen2_5_VLInferenceConfig,
    )

    text_neuron_config = NeuronConfig(
        batch_size=1,
        ctx_batch_size=1,
        seq_len=SEQ_LEN,
        tp_degree=TP_DEGREE,
        world_size=TP_DEGREE,
        torch_dtype=torch.bfloat16,
        save_sharded_checkpoint=True,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        mlp_kernel_enabled=False,
        attn_kernel_enabled=True,
        attn_tkg_nki_kernel_enabled=True,
        logical_neuron_cores=2,
        cc_pipeline_tiling_factor=2,
        on_device_sampling_config=None,
        cast_type="as-declared",
        enable_bucketing=True,  # Multi-bucket CTE for TTFT optimization
        context_encoding_buckets=[
            512,
            1024,
            2048,
            4096,
        ],  # Min 512 for TKG NKI kernel compat
        token_generation_buckets=[4096],  # Single TKG bucket at full seq_len
    )

    vision_neuron_config = NeuronConfig(
        batch_size=1,
        seq_len=SEQ_LEN,
        tp_degree=TP_DEGREE,
        world_size=TP_DEGREE,
        save_sharded_checkpoint=True,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
        attn_kernel_enabled=True,  # Flash attention for bidirectional vision
        mlp_kernel_enabled=False,
        qkv_kernel_enabled=False,  # Fused RMSNorm+QKV fails with vision RMSNorm eps type
        cc_pipeline_tiling_factor=2,
        cast_type="as-declared",
        logical_neuron_cores=2,
        enable_bucketing=True,
        buckets=[2],
    )

    config = Qwen2_5_VLInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    logger.info("Loading compiled model from %s", COMPILED_MODEL_PATH)
    model = NeuronQwen2_5_VLForCausalLM(model_path=MODEL_PATH, config=config)
    model.load(COMPILED_MODEL_PATH)

    adapter = HuggingFaceGenerationAdapter(model)
    logger.info("Model loaded successfully.")
    return model, adapter


@pytest.fixture(scope="module")
def processor():
    """Load the Qwen2.5-VL processor (tokenizer + image processor)."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return proc


@pytest.fixture(scope="module")
def generation_config():
    """Standard generation config for tests."""
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=False,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=[EOS_TOKEN_ID],
        pad_token_id=PAD_TOKEN_ID,
    )


@pytest.fixture(scope="module")
def sampling_params():
    """Sampling params for greedy decoding."""
    from neuronx_distributed_inference.modules.generation.sampling import (
        prepare_sampling_params,
    )

    return prepare_sampling_params(
        batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0]
    )


# ---- Helper functions ----


def generate_text(
    adapter, processor, prompt, gen_config, samp_params, max_new_tokens=64
):
    """Run text-only generation and return output text + metadata."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt")

    start = time.time()
    with torch.no_grad():
        generated = adapter.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            sampling_params=samp_params,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
        )
    elapsed = time.time() - start

    output_ids = generated[0][inputs.input_ids.shape[1] :]
    output_text = processor.decode(output_ids, skip_special_tokens=True)
    return output_text, len(output_ids), elapsed


def generate_vl(
    adapter, processor, prompt, image, gen_config, samp_params, max_new_tokens=64
):
    """Run vision-language generation and return output text + metadata."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

    start = time.time()
    with torch.no_grad():
        generated = adapter.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            sampling_params=samp_params,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
        )
    elapsed = time.time() - start

    output_ids = generated[0][inputs.input_ids.shape[1] :]
    output_text = processor.decode(output_ids, skip_special_tokens=True)
    return output_text, len(output_ids), elapsed


def make_test_image(width=224, height=224, color="red", shape="circle"):
    """Create a synthetic test image."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    margin = min(width, height) // 8
    if shape == "circle":
        draw.ellipse(
            [margin, margin, width - margin, height - margin],
            fill=color,
            outline="black",
            width=2,
        )
    elif shape == "rectangle":
        draw.rectangle(
            [margin, margin, width - margin, height - margin],
            fill=color,
            outline="black",
            width=2,
        )
    return img


# ---- Tests ----


def test_model_loads(model_and_adapter):
    """Smoke test: model loads successfully from compiled artifacts."""
    model, adapter = model_and_adapter
    assert model is not None
    assert adapter is not None
    assert hasattr(model, "config")
    assert model.config.text_config.hidden_size == 3584
    assert model.config.text_config.num_hidden_layers == 28
    logger.info("PASS: Model loaded (hidden_size=3584, layers=28)")


def test_text_generation(
    model_and_adapter, processor, generation_config, sampling_params
):
    """Test text-only generation produces correct output."""
    _, adapter = model_and_adapter
    output, num_tokens, elapsed = generate_text(
        adapter,
        processor,
        "What is the capital of France? Answer in one sentence.",
        generation_config,
        sampling_params,
        max_new_tokens=32,
    )
    logger.info("Text output (%d tokens, %.2fs): %s", num_tokens, elapsed, output)

    assert num_tokens > 0, "Should generate at least one token"
    assert "Paris" in output, f"Expected 'Paris' in output, got: {output}"


def test_logit_validation(
    model_and_adapter, processor, generation_config, sampling_params
):
    """Validate Neuron logits against CPU HF reference using NxDI logit_validation.

    Uses teacher forcing to compare per-position logit distributions, not just
    top-1 token matching.
    """
    from neuronx_distributed_inference.experimental.core.accuracy.logit_validation import (
        logit_validation,
    )
    from transformers import GenerationConfig

    _, adapter = model_and_adapter
    prompt = "What is the capital of France? Answer in one sentence."
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt")
    input_ids = inputs.input_ids

    max_new_tokens = 16

    # 1. Generate CPU reference logits
    logger.info("Loading CPU reference model for logit validation...")
    from transformers import Qwen2_5_VLForConditionalGeneration

    cpu_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32
    )
    cpu_model.eval()

    cpu_gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=[EOS_TOKEN_ID],
        pad_token_id=PAD_TOKEN_ID,
    )

    with torch.no_grad():
        cpu_result = cpu_model.generate(input_ids, generation_config=cpu_gen_config)
    expected_logits = torch.stack(cpu_result.scores)  # (seq_len, batch, vocab)
    logger.info("CPU reference logits shape: %s", expected_logits.shape)

    # Clean up CPU model to free memory
    del cpu_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. Build Neuron generate_fn
    neuron_gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=[EOS_TOKEN_ID],
        pad_token_id=PAD_TOKEN_ID,
    )

    def generate_fn(input_ids_list):
        input_tensor = torch.tensor(input_ids_list)
        attention_mask = torch.ones_like(input_tensor)
        result = adapter.generate(
            input_tensor,
            attention_mask=attention_mask,
            sampling_params=sampling_params,
            generation_config=neuron_gen_config,
        )
        return torch.stack(result.scores)

    # 3. Validate
    passed = logit_validation(
        input_ids=input_ids.tolist(),
        generate_fn=generate_fn,
        expected_logits=expected_logits,
    )
    assert passed, "Logit validation failed: Neuron logits diverged from CPU reference"
    logger.info("PASS: logit_validation -- Neuron logits match CPU reference")


def test_vl_generation(
    model_and_adapter, processor, generation_config, sampling_params
):
    """Test vision-language generation with a synthetic image."""
    _, adapter = model_and_adapter
    image = make_test_image(448, 448, "green", "circle")
    output, num_tokens, elapsed = generate_vl(
        adapter,
        processor,
        "What shape and color do you see? Be brief.",
        image,
        generation_config,
        sampling_params,
        max_new_tokens=32,
    )
    logger.info("VL output (%d tokens, %.2fs): %s", num_tokens, elapsed, output)

    assert num_tokens > 0, "Should generate at least one token"
    output_lower = output.lower()
    assert "green" in output_lower or "circle" in output_lower, (
        f"Expected 'green' or 'circle' in output, got: {output}"
    )


def test_vl_different_resolutions(
    model_and_adapter, processor, generation_config, sampling_params
):
    """Test VL generation works across different image resolutions."""
    _, adapter = model_and_adapter

    resolutions = [(224, 224), (448, 448), (672, 672), (640, 480)]
    for w, h in resolutions:
        image = make_test_image(w, h, "blue", "rectangle")
        output, num_tokens, elapsed = generate_vl(
            adapter,
            processor,
            "What do you see?",
            image,
            generation_config,
            sampling_params,
            max_new_tokens=16,
        )
        assert num_tokens > 0, f"Failed for resolution {w}x{h}"
        logger.info(
            "  %dx%d: %d tokens, %.2fs -- %s", w, h, num_tokens, elapsed, output[:60]
        )

    logger.info("PASS: All %d resolutions produced output", len(resolutions))


def test_performance_ttft(
    model_and_adapter, processor, generation_config, sampling_params
):
    """Measure Time To First Token for text-only input."""
    _, adapter = model_and_adapter

    # Warmup
    generate_text(
        adapter,
        processor,
        "Hello",
        generation_config,
        sampling_params,
        max_new_tokens=2,
    )

    # Measure
    times = []
    for _ in range(5):
        _, _, elapsed = generate_text(
            adapter,
            processor,
            "Hello",
            generation_config,
            sampling_params,
            max_new_tokens=1,
        )
        times.append(elapsed * 1000)  # ms

    avg_ttft = sum(times) / len(times)
    logger.info("TTFT: %.1f ms (avg of %d runs)", avg_ttft, len(times))
    assert avg_ttft < 500, f"TTFT {avg_ttft:.1f}ms exceeds 500ms threshold"


def test_performance_throughput(
    model_and_adapter, processor, generation_config, sampling_params
):
    """Measure token generation throughput."""
    _, adapter = model_and_adapter
    num_tokens_target = 64

    # Warmup
    generate_text(
        adapter,
        processor,
        "Hello",
        generation_config,
        sampling_params,
        max_new_tokens=4,
    )

    _, num_tokens, elapsed = generate_text(
        adapter,
        processor,
        "Write a short paragraph about machine learning.",
        generation_config,
        sampling_params,
        max_new_tokens=num_tokens_target,
    )
    throughput = num_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        "Throughput: %.1f tok/s (%d tokens in %.2fs)", throughput, num_tokens, elapsed
    )
    assert throughput > 10, (
        f"Throughput {throughput:.1f} tok/s below 10 tok/s threshold"
    )


# ---- Main (for running outside pytest) ----

if __name__ == "__main__":
    print("=" * 80)
    print("Qwen2.5-VL-7B-Instruct Integration Tests")
    print("=" * 80)
    print(f"Model path:    {MODEL_PATH}")
    print(f"Compiled path: {COMPILED_MODEL_PATH}")
    print(f"TP degree:     {TP_DEGREE}")
    print()
    print("Run with: pytest test/integration/test_model.py -v --capture=tee-sys")
    print("=" * 80)
