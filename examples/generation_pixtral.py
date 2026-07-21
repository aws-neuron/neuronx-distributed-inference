import torch
import os

from transformers import AutoTokenizer, AutoProcessor, GenerationConfig
from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

from neuronx_distributed_inference.models.pixtral.modeling_pixtral import NeuronPixtralForCausalLM, PixtralInferenceConfig
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.models.pixtral.utils.input_processor import prepare_generation_inputs_hf

TEXT_TP_DEGREE = 64
VISION_TP_DEGERE = 16
WORLD_SIZE = 64
BATCH_SIZE  = 1
CP_DEGREE = 1
TEXT_SEQ_LENGTH = 10 * 1024
VISION_SEQ_LENGTH = 10 * 1024
DTYPE = torch.float16
BUCKETS = [2*1024, 4*1024, 10*1024]

model_path = "/shared/cache/checkpoints/Pixtral-Large-Instruct-2411/"
traced_model_path = "./traced_model_Pixtral-Large-Instruct-2411"

# We need to increase SCRATCHPAD_PAGE_SIZE to support 16K sequence lenghts
os.environ['NEURON_SCRATCHPAD_PAGE_SIZE'] = '1024'

torch.manual_seed(0)

def run_llama_generate_image_to_text():
    # Initialize configs and tokenizer.
    text_neuron_config = NeuronConfig(batch_size=BATCH_SIZE,
                                seq_len=TEXT_SEQ_LENGTH,
                                ctx_batch_size=1,
                                tp_degree=TEXT_TP_DEGREE,
                                world_size=WORLD_SIZE,
                                torch_dtype=DTYPE,
                                attention_dtype=DTYPE,
                                rpl_reduce_dtype=DTYPE,
                                cp_degree=CP_DEGREE,
                                save_sharded_checkpoint=True, # Set to False to save time on compilation and more time on loading
                                cast_type="as-declared",
                                sequence_parallel_enabled=True, 
                                fused_qkv=True,
                                qkv_kernel_enabled=True,
                                mlp_kernel_enabled=True,
                                enable_bucketing = True,
                                context_encoding_buckets=BUCKETS,
                                token_generation_buckets=BUCKETS,
                                attn_block_tkg_nki_kernel_enabled=True,
                                attn_block_tkg_nki_kernel_cache_update=True,
                                on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
                                cc_pipeline_tiling_factor=1,
                                )

    vision_neuron_config = NeuronConfig(batch_size=1,
                                seq_len=VISION_SEQ_LENGTH, 
                                tp_degree=VISION_TP_DEGERE,
                                world_size=WORLD_SIZE,
                                enable_bucketing=True,
                                save_sharded_checkpoint=True, # Set to False to save time on compilation and more time on loading
                                torch_dtype=DTYPE,
                                buckets=BUCKETS,
                                cc_pipeline_tiling_factor=1,
                                fused_qkv=False,
                                qkv_kernel_enabled=False,
                                attn_kernel_enabled=False,
                                mlp_kernel_enabled=False,
                                cast_type="as-declared",
                                logical_neuron_cores=2,
                                )

    config = PixtralInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    hf_pixtral_processor = AutoProcessor.from_pretrained(model_path)
    
    if not os.path.exists(traced_model_path):
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronPixtralForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronPixtralForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    generation_model = HuggingFaceGenerationAdapter(model)
    generation_config = GenerationConfig(do_sample=False,
                                         bos_token_id = 1,
                                         eos_token_id = [2],
                                         pad_token_id=2,
                                         output_logits=True)
    
    # Prepare generate outputs.
    text_prompt="What do you see in these images?"
    role='user'
    image_path = "./dog.jpg"
    input_ids, attention_mask, pixel_values,  vision_mask, image_sizes = prepare_generation_inputs_hf(text_prompt, image_path, hf_pixtral_processor, role, config)

    sampling_params = prepare_sampling_params(batch_size=BATCH_SIZE, top_k=[1], top_p=[1.0],  temperature=[1.0])
    outputs = generation_model.generate(
        input_ids,
        generation_config=generation_config,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params, 
        pixel_values=pixel_values,
        vision_mask=vision_mask.to(torch.bool),
        image_sizes=image_sizes,
        max_new_tokens=512,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"Generated outputs shape: {outputs.shape}")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


    # Test Text-Only inputs
    text_prompt="what is the recipe of mayonnaise in two sentences?"
    image_path=None
    role='user'

    input_ids, attention_mask, _,  _, _ = prepare_generation_inputs_hf(text_prompt, image_path, hf_pixtral_processor, role)
    sampling_params = prepare_sampling_params(batch_size=BATCH_SIZE, top_k=[1], top_p=[1.0],  temperature=[1.0])
    outputs = generation_model.generate(
        input_ids,
        generation_config=generation_config,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
        pixel_values=None,
        vision_mask=None,
        max_new_tokens=100,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"Generated outputs shape: {outputs.shape}")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    print("\nPerformance Benchmarking text+image!")
    benchmark_sampling(model=model, draft_model=None, generation_config=generation_config, target="all", image=True,benchmark_report_path="benchmark_report_text_and_image.json", num_runs=5)

    print("\nPerformance Benchmarking text-only!")
    benchmark_sampling(model=model, draft_model=None, generation_config=generation_config, target="all", image=False,benchmark_report_path="benchmark_report_text_only.json", num_runs=5)

if __name__ == "__main__":
    run_llama_generate_image_to_text()
    