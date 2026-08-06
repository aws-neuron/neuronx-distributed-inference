#!/usr/bin/env python3
# Copyright 2025 (c) Amazon.com and Affiliates
"""Offline inference for Isaac-0.2-2B via vLLM on Neuron.

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    export NEURON_COMPILED_ARTIFACTS="/mnt/models/traced_model/Isaac-0.2-2B"
    PYTHONPATH="/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:/mnt/models/neuronx-distributed-inference/src:$PYTHONPATH" \
        python run_offline_inference.py
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import os  # noqa: E402
from pathlib import Path  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

HOME_DIR = Path.home()
DATA_PATH = os.getenv("DATA_HOME", "/mnt/models")
MODEL_PATH = f"{DATA_PATH}/Isaac-0.2-2B-Preview"
COMPILED_PATH = f"{DATA_PATH}/traced_model/Isaac-0.2-2B"

os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"
os.environ["NEURON_COMPILED_ARTIFACTS"] = COMPILED_PATH


def main(max_seq_len: int = 1024) -> None:
    llm = LLM(
        model=MODEL_PATH,
        max_num_seqs=1,
        max_model_len=max_seq_len,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=HOME_DIR.as_posix(),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        trust_remote_code=True,
        additional_config={
            "override_neuron_config": {
                "text_neuron_config": {
                    "attn_kernel_enabled": True,
                    "enable_bucketing": True,
                    "context_encoding_buckets": [max_seq_len],
                    "token_generation_buckets": [max_seq_len],
                    "is_continuous_batching": True,
                    "async_mode": False,
                },
                "vision_neuron_config": {
                    "enable_bucketing": True,
                    "buckets": [1],
                    "is_continuous_batching": True,
                },
            },
        },
    )

    sampling_params = SamplingParams(top_k=1, max_tokens=100)

    # Test 1: Text-only
    print("=" * 60)
    print("Test 1: Text-only")
    print("=" * 60)
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the capital of France? Explain briefly.",
                },
            ],
        }
    ]
    for output in llm.chat(conversation, sampling_params):
        print(f"Generated: {output.outputs[0].text!r}")

    # Test 2: Text-only (longer)
    print("\n" + "=" * 60)
    print("Test 2: Text-only (longer)")
    print("=" * 60)
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Explain quantum entanglement in simple terms.",
                },
            ],
        }
    ]
    for output in llm.chat(conversation, sampling_params):
        print(f"Generated: {output.outputs[0].text!r}")

    # Test 3: Image+text (requires a test image)
    print("\n" + "=" * 60)
    print("Test 3: Image+text")
    print("=" * 60)
    test_image = Path(__file__).resolve().parent / "data" / "test_image.jpg"
    if test_image.exists():
        image_url = f"file://{test_image.as_posix()}"
    else:
        # Use a publicly accessible image URL
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    try:
        for output in llm.chat(conversation, sampling_params):
            print(f"Generated: {output.outputs[0].text!r}")
    except Exception as e:
        print(f"Image+text failed (may need local image): {e}")

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
