# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

from models.ndxi_patch import apply_patch
apply_patch()
import neuronx_distributed_inference.modules.sliding_window.attention as nxdi_swa
nxdi_swa.MIN_SLIDING_WINDOW_SEQ_TILE_SIZE = 1024

import os

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from models.gemma3.modeling_causal_lm_gemma3 import TextGemma3InferenceConfig, NeuronTextGemma3ForCausalLM


model_path = "/home/ubuntu/model_hf/gemma-3-1b-it"
traced_model_path = "/home/ubuntu/traced_model/gemma-3-1b-it"

torch.manual_seed(0)


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", revision="main")  # nosec B615
    generation_config = GenerationConfig.from_pretrained(model_path, revision="main")  # nosec B615
    generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
    }
    generation_config.update(**generation_config_kwargs)

    neuron_config = NeuronConfig(
        tp_degree=2,
        sequence_parallel_enabled=False,
        batch_size=2,
        torch_dtype=torch.bfloat16,
        save_sharded_checkpoint=True,
        seq_len=768,
        on_device_sampling_config=None,
        enable_bucketing=True,
        context_encoding_buckets=[768],
        token_generation_buckets=[768],
        target="inf2",
        logical_nc_config=1,
        fused_qkv=True,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
        attn_kernel_enabled=False
    )

    hf_config = Gemma3TextConfig.from_pretrained(model_path, revision="main")  # nosec B615
    hf_config = hf_config.to_dict()

    config = TextGemma3InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    # config.num_hidden_layers = 1
            
    if not os.path.exists(traced_model_path):
        print("\nCompiling and saving model...")
        model = NeuronTextGemma3ForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    print("\nLoading model from compiled checkpoint...")
    model = NeuronTextGemma3ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    generation_model = HuggingFaceGenerationAdapter(model)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompts = ["Tell me what you believe is the meaning of life.", "Tell me what is the color of the sky."]
    sampling_params = prepare_sampling_params(batch_size=neuron_config.batch_size, top_k=[10, 5], top_p=[0.5, 0.9],  temperature=[0.9, 0.5])
    print(f"Prompts: {prompts}")

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            },
        ]
        for prompt in prompts
    ]

    formatted_texts = [
        tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        for conversation in conversations
    ]

    inputs = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
    )

    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


if __name__ == "__main__":
    main()
