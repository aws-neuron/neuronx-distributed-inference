import torch
import os

from transformers import AutoProcessor, GenerationConfig
from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLInferenceConfig, Qwen3VLNeuronConfig, NeuronQwen3VLForCausalLM

model_path = "/shared/cache/checkpoints/Qwen3-VL/Qwen3-VL-8B-Thinking/"
traced_model_path = "./traced_model/"

torch.manual_seed(0)

DTYPE = torch.bfloat16  # use bf16 to align with HF checkpoint 
VISION_SEQ_LENGTH = 16 * 1024
VISION_BUCKETS = [1024, VISION_SEQ_LENGTH]

TEXT_SEQ_LENGTH = 32 * 1024
TEXT_BUCKETS = [2048, 15360, TEXT_SEQ_LENGTH]

MAX_NEW_TOKENS = 1024
TP_DEGREE = 16

def generate_image_to_text():
    # Initialize configs and tokenizer.
    text_neuron_config = Qwen3VLNeuronConfig(batch_size=1,
                                seq_len=TEXT_SEQ_LENGTH,
                                ctx_batch_size=1,
                                tp_degree=TP_DEGREE,
                                world_size=TP_DEGREE,
                                torch_dtype=DTYPE,
                                attention_dtype=DTYPE,
                                rpl_reduce_dtype=DTYPE,
                                cp_degree=1,
                                save_sharded_checkpoint=True,
                                sequence_parallel_enabled=False, 
                                fused_qkv=True,
                                attn_kernel_enabled=True,
                                qkv_kernel_enabled=True,
                                mlp_kernel_enabled=True,
                                enable_bucketing = True,
                                buckets=TEXT_BUCKETS,
                                context_encoding_buckets=TEXT_BUCKETS,
                                token_generation_buckets=TEXT_BUCKETS,
                                attn_block_tkg_nki_kernel_enabled=False,
                                attn_block_tkg_nki_kernel_cache_update=False,
                                on_device_sampling_config=SmplConfig(dynamic=False, top_k=1), 
                                cc_pipeline_tiling_factor=2,
                                cast_type="as-declared",
                                logical_neuron_cores=2,
                                )
    vision_neuron_config = Qwen3VLNeuronConfig(batch_size=1,
                                seq_len=VISION_SEQ_LENGTH,
                                ctx_batch_size=1,
                                tp_degree=TP_DEGREE,
                                world_size=TP_DEGREE,
                                torch_dtype=DTYPE,
                                attention_dtype=DTYPE,
                                rpl_reduce_dtype=DTYPE,
                                cp_degree=1,
                                save_sharded_checkpoint=True,
                                sequence_parallel_enabled=False, 
                                fused_qkv=True,
                                qkv_kernel_enabled=False,
                                # TODO: Currently, to support dynamic image resolution, attention mask will be non-causal block mask which is not supported by kernel yet.
                                attn_kernel_enabled=False,
                                mlp_kernel_enabled=False,
                                enable_bucketing=True,
                                buckets=VISION_BUCKETS,
                                cc_pipeline_tiling_factor=2,
                                cast_type="as-declared",
                                logical_neuron_cores=2,
                                )

    config = Qwen3VLInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )
 
    hf_qwen3_vl_processor = AutoProcessor.from_pretrained(model_path)
    model = NeuronQwen3VLForCausalLM(model_path=model_path, config=config)

    if not os.path.exists(traced_model_path):
        # Compile and save model.
        print("\nCompiling and saving model...")
        model.compile(traced_model_path)
        hf_qwen3_vl_processor.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model.load(traced_model_path)
    hf_qwen3_vl_processor = AutoProcessor.from_pretrained(traced_model_path)

    generation_model = HuggingFaceGenerationAdapter(model)
    generation_config = GenerationConfig(do_sample=False,
                                         bos_token_id = 151643,
                                         eos_token_id = [151645],
                                         pad_token_id=151645, 
                                         output_logits=True)
    
    # Test Text-Only inputs
    text_prompt="what is the recipe of mayonnaise in two sentences?"
    image_path=None
    role='user'

    input_ids, attention_mask, _ = NeuronQwen3VLForCausalLM.prepare_input_args(text_prompt, image_path, hf_qwen3_vl_processor, role)
    sampling_params = prepare_sampling_params(batch_size=1, top_k=[1], top_p=[1.0],  temperature=[1.0])
    outputs = generation_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sampling_params=sampling_params,
        generation_config=generation_config,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    output_tokens = hf_qwen3_vl_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"Generated outputs shape: {outputs.shape}")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    # Test Vision+Text inputs
    text_prompt="How many images do you see? What do you see in these images?"
    for image_path in [["dog.jpg"], ["car.jpg"], ["dog.jpg", "car.jpg"], ["dog.jpg", "car.jpg", "cat.png", "cat.png", "car.jpg","dog.jpg"]]:
        role='user'

        input_ids, attention_mask, vision_inputs = NeuronQwen3VLForCausalLM.prepare_input_args(text_prompt, image_path, hf_qwen3_vl_processor, role)
        pixel_values = vision_inputs["pixel_values"]
        print(f"pixel_values shape: {pixel_values.shape}")
        sampling_params = prepare_sampling_params(batch_size=1, top_k=[1], top_p=[1.0],  temperature=[1.0])

        outputs = generation_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=vision_inputs["pixel_values"],
            image_grid_thw=vision_inputs["image_grid_thw"],
            sampling_params=sampling_params,
            generation_config=generation_config,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        output_tokens = hf_qwen3_vl_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"Generated outputs shape: {outputs.shape}")
        for i, output_token in enumerate(output_tokens):
            print(f"Output {i}: {output_token}")

  
if __name__ == "__main__":
    generate_image_to_text()
