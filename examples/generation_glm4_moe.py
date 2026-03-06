import os
import sys

import torch
from transformers import AutoTokenizer, GenerationConfig

# GLM-4.5 MoE is a contrib model; add its src to the Python path.
_CONTRIB_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "contrib",
    "models",
    "glm4_moe",
    "src",
)
if _CONTRIB_SRC not in sys.path:
    sys.path.insert(0, _CONTRIB_SRC)

from glm4_moe.modeling_glm4_moe import Glm4MoeInferenceConfig, NeuronGlm4MoeForCausalLM
from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

model_path = "/path/to/GLM-4.5-Air"  # HuggingFace checkpoint root
traced_model_path = "/path/to/GLM-4.5-Air-traced"  # Compiled Neuron artifacts

torch.manual_seed(0)

DTYPE = torch.bfloat16


def generate(skip_compile=False):
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        neuron_config = MoENeuronConfig(
            tp_degree=32,
            moe_tp_degree=4,
            moe_ep_degree=8,
            batch_size=4,
            ctx_batch_size=1,
            tkg_batch_size=4,
            seq_len=4096,
            scratchpad_page_size=512,
            torch_dtype=DTYPE,
            on_device_sampling_config=OnDeviceSamplingConfig(
                do_sample=True, temperature=0.6, top_k=20, top_p=0.95
            ),
            enable_bucketing=False,
            flash_decoding_enabled=True,
            fused_qkv=True,
            logical_nc_config=2,
        )
        config = Glm4MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronGlm4MoeForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronGlm4MoeForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    print("\nPerformance Benchmarking!")
    benchmark_sampling(
        model=model,
        draft_model=None,
        generation_config=generation_config,
        target="all",
        benchmark_report_path="benchmark_report.json",
        num_runs=5,
    )


if __name__ == "__main__":
    generate()
