import torch
import os
import logging
import base64

from transformers import AutoTokenizer, AutoProcessor, GenerationConfig
from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

from neuronx_distributed_inference.models.llama4.modeling_llama4 import NeuronLlama4ForCausalLM, Llama4InferenceConfig, Llama4NeuronConfig
from neuronx_distributed_inference.models.llama4.modeling_llama4_text import NeuronLlama4TextForCausalLM, LlamaInferenceConfig
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.models.llama4.utils.input_processor import prepare_generation_inputs_hf

# TODO : Either read from os_environment var or from arg_parser.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TEXT_TP_DEGREE = 64
VISION_TP_DEGERE = 16
WORLD_SIZE = 64
BATCH_SIZE  = 1
SEQ_LENGTH = 8192
# SEQ_LENGTH = 10240 for chunked attention
TEXT_TO_TEXT = False
# TEXT_TO_TEXT = True for text only generation
DTYPE = torch.bfloat16

os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ['NEURON_RT_NUM_CORES']=f'{TEXT_TP_DEGREE}'
os.environ['BASE_COMPILE_WORK_DIR'] = "./compiler_path/"

model_path = "/home/ubuntu/models/Llama-4-Scout-17B-16E-Instruct/"
traced_model_path = "/home/ubuntu/traced_model_Llama-4-Scout-17B-16E-Instruct"

torch.manual_seed(0)

def run_llama_generate_image_to_text():
    # Initialize configs and tokenizer.
    batch_size = 1
    text_neuron_config = Llama4NeuronConfig(batch_size=1,
                                seq_len=SEQ_LENGTH,
                                torch_dtype=torch.bfloat16,
                                skip_sharding=False,
                                save_sharded_checkpoint=False,
                                tp_degree=TEXT_TP_DEGREE,
                                cp_degree=1,
                                on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
                                world_size=WORLD_SIZE,
                                capacity_factor=None,
                                fused_qkv=False,
                                attention_dtype=torch.float16,
                                rpl_reduce_dtype=torch.float32,
                                cast_type="as-declared",
                                logical_neuron_cores=2)

    vision_neuron_config = Llama4NeuronConfig(batch_size=1,
                                seq_len=SEQ_LENGTH, 
                                torch_dtype=torch.float16,
                                skip_sharding=False,
                                save_sharded_checkpoint=False,
                                tp_degree=VISION_TP_DEGERE,
                                cp_degree=1,
                                on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
                                dp_degree=4, 
                                world_size=WORLD_SIZE,
                                fused_qkv=True,
                                qkv_kernel_enabled=True,
                                attn_kernel_enabled=True,
                                mlp_kernel_enabled=True,
                                enable_bucketing=False,                                
                                logical_neuron_cores=2)

    config = Llama4InferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    hf_llama4_processor = AutoProcessor.from_pretrained(model_path)
    # Prepare generate outputs.
    text_prompt="If I had to write a haiku for this one"
    image_path="./dog.jpg"
    role='user'
    
    with torch.profiler.record_function("prepare_generation_inputs"):
        input_ids, attention_mask, pixel_values,  vision_mask = prepare_generation_inputs_hf(text_prompt, image_path, hf_llama4_processor, role, config)
    
    if not os.path.exists(traced_model_path):
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronLlama4ForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    
    print("\nLoading model from compiled checkpoint...")
    model = NeuronLlama4ForCausalLM(traced_model_path)
    model.load(traced_model_path, skip_warmup=True)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    generation_model = HuggingFaceGenerationAdapter(model)

    generation_config = GenerationConfig.from_pretrained(model_path)

    # Test Sampling Parameters
    sampling_params = prepare_sampling_params(batch_size=batch_size, top_k=[1], top_p=[1.0],  temperature=[1.0])
    outputs = generation_model.generate(
        input_ids,
        generation_config=generation_config,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params, 
        pixel_values=pixel_values,
        vision_mask=vision_mask.to(torch.bool),
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

    input_ids, attention_mask, _,  _ = prepare_generation_inputs_hf(text_prompt, image_path, hf_llama4_processor, role)
    sampling_params = prepare_sampling_params(batch_size=batch_size, top_k=[1], top_p=[1.0],  temperature=[1.0])
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

    print("\nPerformance Benchmarking text-only!")
    benchmark_sampling(model=model, draft_model=None, generation_config=generation_config, target="all", image=None,benchmark_report_path="benchmark_report_text_only.json", num_runs=5)

    print("\nPerformance Benchmarking text+image!")
    benchmark_sampling(model=model, draft_model=None, generation_config=generation_config, target="all", image=True,benchmark_report_path="benchmark_report_text_and_image.json", num_runs=5)

def run_llama_generate_text_to_text():
    # Initialize configs and tokenizer.
    batch_size = 1
    neuron_config = Llama4NeuronConfig(batch_size=1,
                                seq_len=SEQ_LENGTH,
                                torch_dtype=torch.bfloat16,
                                skip_sharding=False,
                                save_sharded_checkpoint=True,
                                tp_degree=TEXT_TP_DEGREE,
                                cp_degree=16,
                                on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
                                world_size=WORLD_SIZE,
                                capacity_factor=None,
                                fused_qkv=False,
                                attention_dtype=torch.float16,
                                rpl_reduce_dtype=torch.float32,
                                cast_type="as-declared",
                                logical_neuron_cores=2)


    config = LlamaInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    hf_llama4_processor = AutoProcessor.from_pretrained(model_path)
 
    if not os.path.exists(traced_model_path):
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronLlama4TextForCausalLM(model_path, config.get_text_config())
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)
    # Load from compiled checkpoint.
    
    print("\nLoading model from compiled checkpoint...")
    model = NeuronLlama4TextForCausalLM(traced_model_path, config.get_text_config())
    model.load(traced_model_path, skip_warmup=True)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Test Text-Only inputs
    text_prompt="what is the recipe of mayonnaise in two sentences?"

    # Uncomment for a longer prompt
    # int_list = list(str(i) for i in range(2500))
    # int_str = ', '.join(int_list)
    # text_prompt = f"Keep counting until 3000. I will start {int_str}..."
    image_path=None
    role='user'

    generation_model = HuggingFaceGenerationAdapter(model)
    generation_config = GenerationConfig.from_pretrained(model_path)

    input_ids, attention_mask, _,  _ = prepare_generation_inputs_hf(text_prompt, image_path, hf_llama4_processor, role)
    print(f"input shape {input_ids.shape}")
    sampling_params = prepare_sampling_params(batch_size=batch_size, top_k=[1], top_p=[1.0],  temperature=[1.0])
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

    print("\nPerformance Benchmarking text-only!")
    benchmark_sampling(model=model, draft_model=None, generation_config=generation_config, target="all", image=False,benchmark_report_path="benchmark_report_text_only.json", num_runs=5)

if __name__ == "__main__":
    if TEXT_TO_TEXT:
        run_llama_generate_text_to_text()
    else:
        run_llama_generate_image_to_text()
    