"""
DeepSeek V3 (671B) inference demo on AWS Trainium.

Hardware requirements:
  - Full model (671B): trn2.48xlarge (tp_degree=32 or tp_degree=64)
  - Mini model (testing): trn2.3xlarge (tp_degree=2)

Usage:
  # Full model (requires trn2.48xlarge and DeepSeek-V3 weights):
  python generation_deepseek_v3.py --model-path /path/to/DeepSeek-V3

  # Load from compiled checkpoint:
  python generation_deepseek_v3.py --traced-model-path /path/to/traced_model --skip-compile
"""

import argparse

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    DeepseekV3InferenceConfig,
    NeuronDeepseekV3ForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config


torch.manual_seed(0)


def get_neuron_config(tp_degree=32, batch_size=1, seq_len=4096):
    return MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        ctx_batch_size=1,
        tkg_batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
    )


def generate(model_path, traced_model_path, skip_compile=False, tp_degree=32, seq_len=4096):
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.update(do_sample=True, top_k=1, pad_token_id=generation_config.eos_token_id)

    if not skip_compile:
        neuron_config = get_neuron_config(tp_degree=tp_degree, seq_len=seq_len)
        config = DeepseekV3InferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        print("\nCompiling and saving model...")
        model = NeuronDeepseekV3ForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    print("\nLoading model from compiled checkpoint...")
    model = NeuronDeepseekV3ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    print("\nGenerating outputs...")
    prompts = ["The capital of France is"]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for i, text in enumerate(output_tokens):
        print(f"Output {i}: {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 inference on Neuron")
    parser.add_argument("--model-path", type=str, required=True, help="Path to HF model weights")
    parser.add_argument("--traced-model-path", type=str, default="/tmp/deepseek_v3_traced", help="Path for compiled model")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation, load from traced path")
    parser.add_argument("--tp-degree", type=int, default=32, help="Tensor parallelism degree")
    parser.add_argument("--seq-len", type=int, default=4096, help="Max sequence length")
    args = parser.parse_args()

    generate(
        model_path=args.model_path,
        traced_model_path=args.traced_model_path,
        skip_compile=args.skip_compile,
        tp_degree=args.tp_degree,
        seq_len=args.seq_len,
    )
