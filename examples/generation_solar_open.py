import sys
from pathlib import Path

# Add contrib src to path so we can import solar_open directly
sys.path.insert(0, str(Path(__file__).parent.parent / "contrib/models/solar_open/src"))

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from solar_open.modeling_solar_open import (
    SolarOpenInferenceConfig,
    NeuronSolarOpenForCausalLM,
    load_solar_open_config,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

model_path = "/shared/cache/checkpoints/upstage/Solar-Open-100B"
traced_model_path = "/shared/cache/checkpoints/upstage/Solar-Open-100B/traced_model/"

torch.manual_seed(0)

DTYPE = torch.bfloat16


def generate(skip_compile=False):
    # Initialize configs and tokenizer.
    try:
        generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception:
        generation_config = GenerationConfig(
            max_new_tokens=128,
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
        )

    if not skip_compile:
        neuron_config = MoENeuronConfig(
            tp_degree=32,
            moe_tp_degree=4,
            moe_ep_degree=8,
            batch_size=4,
            ctx_batch_size=1,
            tkg_batch_size=4,
            seq_len=65536,
            scratchpad_page_size=1024,
            torch_dtype=DTYPE,
            on_device_sampling_config=OnDeviceSamplingConfig(
                do_sample=True,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
            ),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            fused_qkv=True,
            sequence_parallel_enabled=False,
            qkv_kernel_enabled=True,
            attn_kernel_enabled=True,
        )
        config = SolarOpenInferenceConfig(
            neuron_config,
            load_config=load_solar_open_config(model_path),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronSolarOpenForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronSolarOpenForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompt = "Give me a short introduction to large language models."
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
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
