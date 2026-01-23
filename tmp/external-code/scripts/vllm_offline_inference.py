# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

import os
from vllm import LLM, SamplingParams

# Hugging Face authentication (replace with your token)
# from huggingface_hub import login
# login(token="your_hf_token_here")

# Configure Neuron environment for inference
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
os.environ['NEURON_COMPILED_ARTIFACTS'] = "/home/ubuntu/traced_model/gemma-3-27b-it"
os.environ['NEURON_ON_DEVICE_SAMPLING_DISABLED'] = "1"

IMAGE_URL = "file:///home/ubuntu/daanggn-neuron-inference-migration/scripts/dog.jpg"

# Initialize LLM with Neuron device configuration
llm = LLM(
    model="/home/ubuntu/model_hf/gemma-3-27b-it",  # or the file path to the downloaded checkpoint
    max_num_seqs=1,
    max_model_len=2048,
    device="neuron",
    tensor_parallel_size=8,
    # use_v2_block_manager=True,
    limit_mm_per_prompt={"image": 1}, # Accepts up to 5 images per prompt
    allowed_local_media_path="/home/ubuntu",  # Allow loading local images
)
# Configure sampling for deterministic output
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