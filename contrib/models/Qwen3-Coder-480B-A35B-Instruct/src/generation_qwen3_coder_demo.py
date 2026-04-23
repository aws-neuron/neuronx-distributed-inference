"""
NxD Inference direct usage example for Qwen3-Coder-480B-A35B-Instruct.

This demonstrates using NxDI directly (without vLLM) for compile + inference.
For production serving, use the vLLM launch script instead.

Requirements:
    - trn2.48xlarge instance
    - SDK 2.28+
    - Model weights at MODEL_PATH

Usage:
    # First run (compiles + infers):
    python generation_qwen3_coder_demo.py

    # Subsequent runs (loads from cache):
    python generation_qwen3_coder_demo.py --skip-compile
"""

import argparse
import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeInferenceConfig,
    NeuronQwen3MoeForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

MODEL_PATH = "/mnt/nvme/Qwen3-Coder-480B-A35B-Instruct/"
COMPILED_MODEL_PATH = "/mnt/nvme/Qwen3-Coder-480B-A35B-Instruct/traced_model/"

DTYPE = torch.bfloat16


def generate(skip_compile=False):
    generation_config = GenerationConfig.from_pretrained(MODEL_PATH)

    if not skip_compile:
        neuron_config = MoENeuronConfig(
            tp_degree=64,
            moe_tp_degree=64,
            moe_ep_degree=1,
            batch_size=16,
            ctx_batch_size=1,
            seq_len=8192,
            scratchpad_page_size=1024,
            torch_dtype=DTYPE,
            enable_bucketing=True,
            flash_decoding_enabled=False,
            cp_degree=1,
            fused_qkv=True,
            is_continuous_batching=True,
            logical_nc_config=2,
            sequence_parallel_enabled=True,
            qkv_kernel_enabled=True,
            qkv_nki_kernel_enabled=True,
            qkv_cte_nki_kernel_fuse_rope=True,
            attn_kernel_enabled=True,
            async_mode=True,
            cc_pipeline_tiling_factor=2,
            mode_mask_padded_tokens=True,
        )
        config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(MODEL_PATH),
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        print("\nCompiling model (first run takes ~22 min)...")
        model = NeuronQwen3MoeForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
        tokenizer.save_pretrained(COMPILED_MODEL_PATH)

    print("\nLoading model from compiled checkpoint (~10 min)...")
    model = NeuronQwen3MoeForCausalLM(COMPILED_MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(COMPILED_MODEL_PATH)

    print("\nGenerating...")
    prompt = "Write a Python function that implements quicksort with type hints."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
    )
    output_tokens = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Generated output:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-compile", action="store_true", help="Skip compilation, load from cache"
    )
    args = parser.parse_args()
    generate(skip_compile=args.skip_compile)
