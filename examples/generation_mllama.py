import torch
import os

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MultimodalVisionNeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.mllama.modeling_mllama import MllamaInferenceConfig, NeuronMllamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.models.mllama.model_wrapper_mllama import NUM_IMAGE_PER_PROMPT
from neuronx_distributed_inference.models.mllama.utils import create_vision_mask, get_image, get_image_tensors, add_instruct
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

# TODO : Either read from os_environment var or from arg_parser.
model_variant = "11B"
model_path = f"/home/ubuntu/models/Llama-3.2-{model_variant}-Vision-Instruct/"
traced_model_path =  f"/home/ubuntu/workplace/traced_models/Llama-3.2-{model_variant}-Vision-Instruct/"

torch.manual_seed(0)


def run_llama_generate():
    # Initialize configs and tokenizer.
    batch_size = 1
    num_img_per_prompt = 1
    max_context_length = 1024
    seq_len = 2048

    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "top_k": 1,
    }
    generation_config.update(**generation_config_kwargs)

    on_device_sampling_config=OnDeviceSamplingConfig(
                                                     dynamic=True, 
                                                     )

    neuron_config = MultimodalVisionNeuronConfig(
        tp_degree=32,
        batch_size=batch_size,
        max_context_length=max_context_length,
        seq_len=seq_len,
        on_device_sampling_config=on_device_sampling_config,
        enable_bucketing=True,
        sequence_parallel_enabled=True,
        fused_qkv=True,
        async_mode=False,
    )
    config = MllamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )
    config.neuron_config.skip_vision = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

 
    # Generate outputs.
    image = get_image("dog.jpg")
    batch_image = [[image] * num_img_per_prompt] * batch_size
    pixel_values, aspect_ratios, num_chunks, has_image = get_image_tensors(config, batch_image)

    prompt = add_instruct("What is in this image? Tell me a story", has_image)
    batch_prompt = [prompt] * batch_size

    if not os.path.exists(traced_model_path):
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronMllamaForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronMllamaForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    print("\nGenerating outputs...")
    print(f"Prompts: {batch_prompt}")

    inputs = tokenizer(batch_prompt, padding=True, return_tensors="pt", add_special_tokens=False)

    vision_token_id = tokenizer("<|image|>", add_special_tokens=False).input_ids[0]
    vision_mask = create_vision_mask(inputs.input_ids, vision_token_id)

    generation_model = HuggingFaceGenerationAdapter(model)

    # Test Sampling Parameters
    sampling_params = prepare_sampling_params(batch_size=batch_size, top_k=[1], top_p=[1.0],  temperature=[1.0])
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params, 
        pixel_values=pixel_values, 
        aspect_ratios=aspect_ratios,
        vision_mask =vision_mask,
        num_chunks=num_chunks, 
        has_image=has_image,
        max_new_tokens=512,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


    # Test with text-only input
    pixel_values, aspect_ratios, num_chunks, has_image = get_image_tensors(config, [[]] * batch_size)

    prompt = add_instruct("what is the recipe of mayonnaise in two sentences?", has_image)
    batch_prompt = [prompt] * batch_size
    inputs = tokenizer(batch_prompt, padding=True, return_tensors="pt")

    sampling_params = prepare_sampling_params(batch_size=batch_size, top_k=[1], top_p=[1.0],  temperature=[1.0])
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params, 
        pixel_values=pixel_values, 
        aspect_ratios=aspect_ratios,
        vision_mask=vision_mask,
        num_chunks=num_chunks, 
        has_image=has_image,
        max_new_tokens=512,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")
        
    print("\nPerformance Benchmarking!")
    benchmark_sampling(model=model, draft_model=None, generation_config=generation_config, target="all", image=True)

if __name__ == "__main__":
    run_llama_generate()

