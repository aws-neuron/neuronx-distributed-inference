import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig, NeuronQwen3MoeForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

model_path = "/shared/cache/checkpoints/qwen/qwen-3-235b"
traced_model_path = "/shared/cache/checkpoints/qwen/qwen-3-235b/traced_model/"

torch.manual_seed(0)

DTYPE = torch.float16

def generate(skip_compile=False):
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        neuron_config = MoENeuronConfig(
            tp_degree=64,
            moe_tp_degree=4,
            moe_ep_degree=16,
            batch_size=16,
            ctx_batch_size=1,
            tkg_batch_size=16,
            seq_len=10240,
            scratchpad_page_size=1024,
            torch_dtype=DTYPE,
            on_device_sampling_config=OnDeviceSamplingConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            attention_dp_degree=8,
            cp_degree=16,
            fused_qkv=True,
            is_continuous_batching=True,
            logical_nc_config=2,
            sequence_parallel_enabled=True,
            qkv_kernel_enabled=True,
            attn_kernel_enabled=True,
            strided_context_parallel_kernel_enabled=True,
            blockwise_matmul_config={"use_shard_on_intermediate_dynamic_while": True, "skip_dma_token": True},  
        )
        config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )        
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronQwen3MoeForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronQwen3MoeForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    print("\nPerformance Benchmarking!")
    benchmark_sampling(model=model, draft_model=None, generation_config=generation_config, target="all",benchmark_report_path="benchmark_report.json", num_runs=5)

if __name__ == "__main__":
    generate()
