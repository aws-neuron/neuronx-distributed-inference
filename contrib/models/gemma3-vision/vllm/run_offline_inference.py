from gemma3_vision.ndxi_patch import apply_patch
apply_patch()

import os # noqa: E402
from pathlib import Path # noqa: E402

from vllm import LLM, SamplingParams

HOME_DIR = Path.home()

os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
os.environ['NEURON_COMPILED_ARTIFACTS'] = f"{HOME_DIR.as_posix()}/traced_model/gemma-3-27b-it"

input_image_path = Path(__file__).resolve().parent / "data" / "dog.jpg"
IMAGE_URL = f"file://{input_image_path.as_posix()}"


def main(max_seq_len: int = 1024, images_per_sample: int = 1) -> None:
    llm = LLM(
        model=f"{HOME_DIR.as_posix()}/models/gemma-3-27b-it",  # HuggingFace model ID or path to downloaded HF model artifacts
        max_num_seqs=1,
        max_model_len=max_seq_len,
        tensor_parallel_size=8,
        limit_mm_per_prompt={"image": images_per_sample}, # Accept up to 5 images per prompt
        allowed_local_media_path=HOME_DIR.as_posix(),     # Allow loading local images
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        additional_config={
            "override_neuron_config": {
                "text_neuron_config": {
                    "attn_kernel_enabled": True,
                    "enable_bucketing": True,
                    "context_encoding_buckets": [max_seq_len],
                    "token_generation_buckets": [max_seq_len],
                    "is_continuous_batching": True,
                    "async_mode": True,
                },
                "vision_neuron_config": {
                    "enable_bucketing": True,
                    "buckets": [images_per_sample],
                    "is_continuous_batching": True,
                }

            },
        },
    )

    sampling_params = SamplingParams(top_k=1, max_tokens=100)

    # Test 1: Text-only input
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is the recipe of mayonnaise in two sentences?"},
            ]
        }
    ]
    for output in llm.chat(conversation, sampling_params):
        print(f"Generated text: {output.outputs[0].text !r}")

    # Test 2: Single image with text
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                {"type": "text", "text": "Describe this image"},
            ]
        }
    ]
    for output in llm.chat(conversation, sampling_params):
        print(f"Generated text: {output.outputs[0].text !r}")

if __name__ == "__main__":
    main()
