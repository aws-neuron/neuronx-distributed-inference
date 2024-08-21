import os
import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaConfigAdapter, NeuronLlamaForCausalLM

model_path = "/home/ubuntu/model_hf/Llama-2-7b/"
traced_model_path = "/home/ubuntu/traced_model/Llama-2-7b/"

torch.manual_seed(0)


def run_llama_generate():
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
        "pad_token_id": generation_config.eos_token_id
    }
    generation_config.update(**generation_config_kwargs)

    # TODO: Separate GenerationConfig from PretrainedConfig
    config = LlamaConfigAdapter.from_pretrained(model_path, **generation_config_kwargs)
    config.neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=2,
        max_context_length=32,
        seq_len=64,
        on_device_sampling=True,
        enable_bucketing=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(traced_model_path):
        os.makedirs(traced_model_path)
        
    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM.from_pretrained(model_path, config)
    model.compile(traced_model_path)
    config.save_pretrained(traced_model_path)
    tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    config = LlamaConfigAdapter.from_pretrained(traced_model_path)
    model = NeuronLlamaForCausalLM.from_pretrained("", config)
    model.load(traced_model_path)
    if config.torch_dtype == torch.bfloat16:
        model.bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompts = ["I believe the meaning of life is", "The color of the sky is"]
    print(f"Prompts: {prompts}")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=config.neuron_config.max_length,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


if __name__ == "__main__":
    run_llama_generate()
