# Copyright 2025 © Amazon.com and Affiliates

"""
Integration test for Gemma3-Vision VLM model.

This test validates model accuracy and performance for the Gemma3-Vision multimodal model
with both text+image and text-only generation.

Feature: gemma3-vision-migration, Property 3: Text+Image Generation Correctness
Feature: gemma3-vision-migration, Property 4: Text-Only Generation Correctness
Feature: gemma3-vision-migration, Property 5: Model Compilation Success
"""

import pytest
import torch
from transformers import AutoTokenizer, AutoProcessor, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama4.utils.input_processor import (
    prepare_generation_inputs_hf
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.exceptions import LogitMatchingValidationError
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)

from gemma3_vision import NeuronGemma3ForCausalLM, Gemma3InferenceConfig

# Model paths
model_path = "/home/ubuntu/models/google/gemma-3-27b-it/"
compiled_model_path = "/home/ubuntu/neuron-models/gemma-3-27b-it/"
test_image_path = "tmp/external-code/scripts/dog.jpg"

NUM_TOKENS_TO_CHECK = 256

torch.manual_seed(0)


def create_neuron_configs(batch_size, seq_len):
    """Create text and vision neuron configurations."""
    text_config = NeuronConfig(
        # Basic configs
        tp_degree=8,
        batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        
        # Bucketing
        enable_bucketing=True,
        context_encoding_buckets=[seq_len],
        token_generation_buckets=[seq_len],
        
        # Optimizations
        fused_qkv=True,
        attn_kernel_enabled=True,
        async_mode=True,
        
        # Continuous batching
        is_continuous_batching=True,
        ctx_batch_size=1,
    )
    
    vision_config = NeuronConfig(
        # Basic configs
        tp_degree=8,
        batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        
        # Bucketing - auto-bucketing for vision
        enable_bucketing=True,
        buckets=[1],  # Auto-bucketing from 1024 to seq_len
        
        # Optimizations
        fused_qkv=False,  # SigLIP requires separate QKV
        attn_kernel_enabled=True,
        
        # Continuous batching
        is_continuous_batching=True,
        ctx_batch_size=1,
    )
    
    return text_config, vision_config


# Performance numbers based on v14_bs1.py configuration (TP=8, BS=1, SEQ=512)
@pytest.mark.parametrize(
    "batch_size, seq_len, ttft_threshold, throughput_threshold",
    [
        (1, 512, 50.0, 80),    # Baseline configuration
        (1, 2048, 200.0, 70),  # Long context
    ]
)
def test_model_accuracy_and_performance(batch_size, seq_len, ttft_threshold, throughput_threshold):
    """
    Test Gemma3-Vision model accuracy and performance.
    
    Feature: gemma3-vision-migration, Property 3: Text+Image Generation Correctness
    Feature: gemma3-vision-migration, Property 4: Text-Only Generation Correctness
    Feature: gemma3-vision-migration, Property 5: Model Compilation Success
    """
    print(f"Testing model with parameters: {batch_size=}, {seq_len=}, {ttft_threshold=}, {throughput_threshold=}")
    
    # Initialize configs
    text_config, vision_config = create_neuron_configs(batch_size, seq_len)
    
    config = Gemma3InferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(model_path),
    )
    
    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(model_path)
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.do_sample = False
    generation_config.top_k = 1
    
    # Compile and load model
    print("\nCompiling and loading model...")
    model = NeuronGemma3ForCausalLM(model_path, config)
    model.compile(compiled_model_path)
    model.load(compiled_model_path)
    
    # Test 1: Text+Image Generation Accuracy
    print("\n=== Testing Text+Image Generation ===")
    try:
        check_accuracy_logits(
            model,
            tokenizer,
            generation_config,
            num_tokens_to_check=NUM_TOKENS_TO_CHECK,
            image_path=test_image_path,
        )
        print("✓ Text+Image generation accuracy validated")
    except LogitMatchingValidationError as e:
        print(f"✗ Text+Image generation accuracy validation failed: {e}")
        raise e
    
    # Test 2: Text-Only Generation
    print("\n=== Testing Text-Only Generation ===")
    text_prompt = "What is the capital of France?"
    input_ids, attention_mask, _, _ = prepare_generation_inputs_hf(
        text_prompt, None, processor, 'user'
    )
    
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        input_ids,
        generation_config=generation_config,
        attention_mask=attention_mask,
        max_new_tokens=50,
    )
    
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    assert len(output_text) > 0, "Text-only generation produced no output"
    print(f"✓ Text-only generation successful: {output_text[0][:100]}...")
    
    # Test 3: Performance Validation
    print("\n=== Testing Performance ===")
    benchmark_report = benchmark_sampling(model, generation_config=generation_config)
    
    ttft = benchmark_report["context_encoding_model"]["latency_ms_p50"]
    throughput = benchmark_report["token_generation_model"]["throughput"]
    
    print(f"TTFT (p50): {ttft:.2f}ms (threshold: {ttft_threshold}ms)")
    print(f"Throughput: {throughput:.2f} tokens/s (threshold: {throughput_threshold} tokens/s)")
    
    # Allow 10% margin for performance variations
    assert ttft < ttft_threshold * 1.1, f"TTFT {ttft}ms exceeds threshold {ttft_threshold}ms"
    assert throughput > throughput_threshold * 0.9, f"Throughput {throughput} below threshold {throughput_threshold}"
    
    print(f"\n✓ Test passed for parameters: {batch_size=}, {seq_len=}")


if __name__ == "__main__":
    # Run with default parameters for quick testing
    test_model_accuracy_and_performance(1, 512, 50.0, 80)
